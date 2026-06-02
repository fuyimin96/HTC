from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCore(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.num_layers = params["encoder_num_layers"]
        self.hidden_size = params["encoder_hidden_size"]
        self.emb_size = params["encoder_emb_size"]
        self.mlp_num_layers = params["mlp_num_layers"]
        self.mlp_hidden_size = params["mlp_hidden_size"]
        self.mlp_dropout = params["mlp_dropout"]
        self.encoder_length = params["encoder_length"]
        self.vocab_size = params["encoder_vocab_size"]
        self.encoder_dropout = params["encoder_dropout"]
        self.image_hidden_size = params["image_hidden_size"]

        # 序列 embedding
        self.emb = nn.Embedding(self.vocab_size, self.emb_size)

        # 架构编码 MLP: 输入维度 = encoder_length * emb_size
        enc_layers = []
        in_dim = self.encoder_length * self.emb_size
        for _ in range(self.num_layers):
            enc_layers.append(nn.Linear(in_dim, self.hidden_size))
            enc_layers.append(nn.ReLU(inplace=True))
            if self.encoder_dropout > 0.0:
                enc_layers.append(nn.Dropout(self.encoder_dropout))
            enc_layers.append(
                nn.BatchNorm1d(
                    self.hidden_size,
                    eps=1e-5,
                    momentum=0.9,
                )
            )
            in_dim = self.hidden_size
        self.encoder_mlp = nn.Sequential(*enc_layers)

        # 图像 FC（输入维度在第一次 forward 时根据 image_emb 自动确定）
        self.image_fc: nn.Linear = None  # type: ignore

        # predictor MLP: 输入维度 = arch_emb_dim + image_hidden_size
        pred_layers = []
        pred_in_dim = self.hidden_size + self.image_hidden_size
        for _ in range(self.mlp_num_layers):
            pred_layers.append(nn.Linear(pred_in_dim, self.mlp_hidden_size))
            pred_layers.append(nn.ReLU(inplace=True))
            if self.mlp_dropout > 0.0:
                pred_layers.append(nn.Dropout(self.mlp_dropout))
            pred_in_dim = self.mlp_hidden_size
        self.predictor_mlp = nn.Sequential(*pred_layers)
        self.regression = nn.Linear(pred_in_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 模拟 TF random_uniform_initializer(-0.1, 0.1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(
        self, image_emb: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image_emb: [B, D_img]
            x: [B, encoder_length] (LongTensor)

        Returns:
            arch_emb: [B, encoder_hidden_size]
            image_emb_out: [B, image_hidden_size]
            predict_value: [B, 1]
        """
        B = x.size(0)

        # --- process x (架构序列) ---
        # [B, L] -> [B, L, emb_size]
        x_emb = self.emb(x.long())
        # -> [B, L * emb_size]
        x_flat = x_emb.view(B, self.encoder_length * self.emb_size)

        if len(self.encoder_mlp) > 0:
            arch = self.encoder_mlp(x_flat)
        else:
            arch = x_flat
        arch = F.normalize(arch, p=2, dim=-1)
        arch_emb = arch

        # --- process image ---
        if self.image_fc is None:
            # lazy init，依据输入特征维度确定
            self.image_fc = nn.Linear(
                image_emb.size(1), self.image_hidden_size, bias=True
            ).to(image_emb.device)
            nn.init.uniform_(self.image_fc.weight, -0.1, 0.1)
            if self.image_fc.bias is not None:
                nn.init.constant_(self.image_fc.bias, 0.0)

        img = self.image_fc(image_emb)
        img = F.relu(img, inplace=True)
        img = F.normalize(img, p=2, dim=-1)
        image_emb_out = img

        # --- predictor ---
        h = torch.cat([arch_emb, image_emb_out], dim=1)
        if len(self.predictor_mlp) > 0:
            h = self.predictor_mlp(h)
        predict_value = torch.sigmoid(self.regression(h))

        return arch_emb, image_emb_out, predict_value


class EPDEncoderModel(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params
        self.weight_decay = params.get("weight_decay", 0.0)
        self.predict_lambda = params.get("predict_lambda", 0.0)
        self.core = EncoderCore(params)

    def forward(
        self, image_emb: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        arch_emb, _, predict_value = self.core(image_emb, x)
        return arch_emb, predict_value

    def compute_loss(
        self, image_emb: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        arch_emb, image_emb_out, predict_value = self.core(image_emb, x)

        # 权重: y == -1 的位置不参与损失
        weights = (y != -1.0).float()
        # 对齐形状
        if weights.dim() < predict_value.dim():
            weights = weights.view_as(predict_value)

        # 加权 MSE
        mse = ((predict_value - y) ** 2 * weights).sum() / (
            weights.sum() + 1e-8
        )

        if self.weight_decay > 0.0:
            l2 = torch.sum(
                torch.stack(
                    [
                        (p ** 2).sum()
                        for name, p in self.named_parameters()
                        if p.requires_grad and "bias" not in name
                    ]
                )
            )
            total_loss = mse + self.weight_decay * l2
        else:
            total_loss = mse

        return {
            "loss": total_loss,
            "mse_loss": mse,
            "arch_emb": arch_emb,
            "image_emb": image_emb_out,
            "predict_value": predict_value,
        }

    def infer(
        self, image_emb: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PyTorch 对应 TF 的 infer():
          new_arch_emb = arch_emb - predict_lambda * d(predict_value)/d(arch_emb)
          然后对 new_arch_emb 做 L2 归一化。
        """
        image_emb = image_emb.detach()
        x = x.detach()

        # 需要对 arch_emb 求梯度
        image_emb_req = image_emb.requires_grad_(False)
        x_req = x.long().requires_grad_(False)

        arch_emb, _, predict_value = self.core(image_emb_req, x_req)
        arch_emb = arch_emb.requires_grad_(True)

        # 重新计算 predict_value，显式依赖 arch_emb
        with torch.no_grad():
            if self.core.image_fc is None:
                self.core(image_emb_req, x_req)  # 触发 image_fc 初始化
        img = self.core.image_fc(image_emb_req)
        img = F.relu(img, inplace=False)
        img = F.normalize(img, p=2, dim=-1)
        h = torch.cat([arch_emb, img], dim=1)
        if len(self.core.predictor_mlp) > 0:
            h = self.core.predictor_mlp(h)
        predict_value2 = torch.sigmoid(self.core.regression(h))

        # 对 predict_value2 求和再对 arch_emb 求梯度
        grad_on_arch = torch.autograd.grad(
            outputs=predict_value2.sum(),
            inputs=arch_emb,
            create_graph=False,
            retain_graph=False,
        )[0]

        new_arch_emb = arch_emb - self.predict_lambda * grad_on_arch
        new_arch_emb = F.normalize(new_arch_emb, p=2, dim=-1)

        return arch_emb.detach(), predict_value2.detach(), new_arch_emb.detach()


__all__ = ["EncoderCore", "EPDEncoderModel"]


