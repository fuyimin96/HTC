import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from .utils.encoder import EPDEncoderModel
from .utils.decoder import EPDDecoderModel

from rsiattack import ATTACK


class AITL(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        parser = self.create_options(parser)
        self.args = self.add_options(parser)

        super(AITL, self).__init__(self.args)
        self._init_encoder_decoder()

    def add_options(self, parser):
        parser.add_argument("--alpha", type=float, default=1)
        parser.add_argument("--eps", type=float, default=16)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--mu", type=float, default=1.0)
        parser.add_argument("--num_noise", type=int, default=4, help="噪声分支个数")
        parser.add_argument(
            "--encoder_length",
            type=int,
            default=5,
            help="每个噪声分支中操作序列的长度（与 AITL encoder_length 概念类似）",
        )
        # Encoder-Decoder 相关参数
        parser.add_argument("--AITL_encoder_num_layers", type=int, default=1)
        parser.add_argument("--AITL_encoder_hidden_size", type=int, default=96)
        parser.add_argument("--AITL_encoder_emb_size", type=int, default=32)
        parser.add_argument("--AITL_mlp_num_layers", type=int, default=0)
        parser.add_argument("--AITL_mlp_hidden_size", type=int, default=200)
        parser.add_argument("--AITL_decoder_num_layers", type=int, default=1)
        parser.add_argument("--AITL_decoder_hidden_size", type=int, default=96)
        parser.add_argument("--AITL_encoder_dropout", type=float, default=0.1)
        parser.add_argument("--AITL_mlp_dropout", type=float, default=0.1)
        parser.add_argument("--AITL_decoder_dropout", type=float, default=0.0)
        parser.add_argument("--AITL_weight_decay", type=float, default=1e-4)
        parser.add_argument("--AITL_encoder_vocab_size", type=int, default=12)
        parser.add_argument("--AITL_decoder_vocab_size", type=int, default=12)
        parser.add_argument("--AITL_image_hidden_size", type=int, default=128)
        parser.add_argument("--AITL_predict_lambda", type=float, default=1.0)
        parser.add_argument("--AITL_predict_num_steps", type=int, default=1)
        parser.add_argument("--AITL_predict_num_seeds", type=int, default=1)
        parser.add_argument("--AITL_model_path", type=str, default="", help="Encoder-Decoder 模型路径")
        args = parser.parse_args()
        args.att = self.__class__.__name__
        return args

    def _init_encoder_decoder(self):
        args = self.args
        device = args.device
        
        # 构建 Encoder 参数
        encoder_params = {
            "encoder_num_layers": args.AITL_encoder_num_layers,
            "encoder_hidden_size": args.AITL_encoder_hidden_size,
            "encoder_emb_size": args.AITL_encoder_emb_size,
            "mlp_num_layers": args.AITL_mlp_num_layers,
            "mlp_hidden_size": args.AITL_mlp_hidden_size,
            "mlp_dropout": args.AITL_mlp_dropout,
            "encoder_length": args.encoder_length,
            "encoder_vocab_size": args.AITL_encoder_vocab_size,
            "encoder_dropout": args.AITL_encoder_dropout,
            "image_hidden_size": args.AITL_image_hidden_size,
            "weight_decay": args.AITL_weight_decay,
            "predict_lambda": args.AITL_predict_lambda,
        }
        
        # 构建 Decoder 参数
        decoder_params = {
            "decoder_num_layers": args.AITL_decoder_num_layers,
            "decoder_hidden_size": args.AITL_decoder_hidden_size,
            "decoder_length": args.encoder_length,
            "decoder_vocab_size": args.AITL_decoder_vocab_size,
            "decoder_dropout": args.AITL_decoder_dropout,
            "encoder_hidden_size": args.AITL_encoder_hidden_size,
            "weight_decay": args.AITL_weight_decay,
        }
        
        self.encoder_model = EPDEncoderModel(encoder_params).to(device)
        self.decoder_model = EPDDecoderModel(decoder_params).to(device)
        
        #加载预训练权重
        if args.AITL_model_path and os.path.exists(args.AITL_model_path):
            checkpoint = torch.load(args.AITL_model_path, map_location=device)
            if "encoder" in checkpoint:
                self.encoder_model.load_state_dict(checkpoint["encoder"])
            if "decoder" in checkpoint:
                self.decoder_model.load_state_dict(checkpoint["decoder"])
        
        self.encoder_model.eval()
        self.decoder_model.eval()

    def _extract_prelogits(self, net: nn.Module, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        mid_output = None
        
        def get_mid_output(module, input, output):
            nonlocal mid_output
            mid_output = output.detach().clone()
        
        model_name = net.__class__.__name__
        hook = None
        
        if model_name == 'ResNet':
            if hasattr(net, 'avgpool') and hasattr(net, 'fc'):
                hook = net.avgpool.register_forward_hook(get_mid_output)
        elif model_name == 'VGG':
            if hasattr(net, 'features'):
                hook = net.features[-1].register_forward_hook(get_mid_output)
        elif model_name == 'DenseNet':
            if hasattr(net, 'features') and hasattr(net.features, 'norm5'):
                hook = net.features.norm5.register_forward_hook(get_mid_output)
        elif model_name == 'Inception_ResNetv2':
            if hasattr(net, 'features'):
                hook = net.features[-1].register_forward_hook(get_mid_output)
        elif model_name == 'Inception3':
            if hasattr(net, 'Mixed_7c'):
                hook = net.Mixed_7c.branch3x3dbl_3b.bn.register_forward_hook(get_mid_output)
        else:
            if hasattr(net, 'classifier') and len(net.classifier) > 1:
                hook = net.classifier[-2].register_forward_hook(get_mid_output)
            elif hasattr(net, 'fc') and hasattr(net, 'avgpool'):
                hook = net.avgpool.register_forward_hook(get_mid_output)
        
        try:
            with torch.no_grad():
                logits = net(images)
            
            if mid_output is None:
                if hasattr(net, 'get_features'):
                    mid_output = net.get_features(images)
                else:
                    mid_output = torch.zeros(images.size(0), 2048, device=device)
            
            if mid_output.dim() == 4:
                mid_output = F.adaptive_avg_pool2d(mid_output, (1, 1))
                mid_output = mid_output.view(mid_output.size(0), -1)
            elif mid_output.dim() == 2:
                pass  
            else:
                mid_output = mid_output.view(mid_output.size(0), -1)
            
            return mid_output.to(device)
        
        finally:
            if hook is not None:
                hook.remove()

    #使用 Encoder-Decoder 预测操作序列。
    def _predict_op_sequences(
        self, 
        net: nn.Module, 
        images: torch.Tensor,
        num_seeds: int = 1,
        num_steps: int = 1
    ) -> torch.Tensor:
        
        device = images.device
        B = images.size(0)
        encoder_length = self.args.encoder_length
        vocab_size = self.args.AITL_encoder_vocab_size
        
        # 提取 PreLogits 特征
        with torch.no_grad():
            image_emb = self._extract_prelogits(net, images)  # [B, D_img]
        
        # 初始化操作序列
        op_chosen = torch.zeros(
            (B, num_seeds, encoder_length),
            dtype=torch.long,
            device=device
        )
        
        for seed_id in range(num_seeds):
            encoder_input = torch.randint(
                low=0,
                high=vocab_size,
                size=(B, encoder_length),
                device=device,
                dtype=torch.long
            )
            
            with torch.no_grad():
                arch_emb, _ = self.encoder_model(image_emb, encoder_input)
            
            for step in range(num_steps):
                _, _, new_arch_emb = self.encoder_model.infer(image_emb, encoder_input)
                
                with torch.no_grad():
                    _, new_sample_id = self.decoder_model(new_arch_emb)

                arch_emb = new_arch_emb
            
            with torch.no_grad():
                _, sample_id = self.decoder_model(arch_emb)
                op_chosen[:, seed_id] = sample_id
        
        return op_chosen

    # ================== 图像变换操作 ================== #

    @staticmethod
    def _blend(img1: torch.Tensor, img2: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        img = img1 * (1.0 - factor) + img2 * factor
        return torch.clamp(img, 0.0, 1.0)

    def _input_admix(self, x: torch.Tensor, noise: torch.Tensor, i: int, prob: float = 1.0) -> torch.Tensor:
        processed = x + 0.2 * noise[:, i]
        processed = torch.clamp(processed, 0.0, 1.0)
        if prob >= 1.0:
            return processed
        mask = (torch.rand(x.size(0), device=x.device) < prob).float().view(-1, 1, 1, 1)
        return processed * mask + x * (1.0 - mask)

    def _input_scale(self, x: torch.Tensor, prob: float = 1.0) -> torch.Tensor:
        if prob < 1.0:
            mask = (torch.rand(x.size(0), device=x.device) < prob).float().view(-1, 1, 1, 1)
        else:
            mask = None
        scale_factor = torch.randint(0, 5, (1,), device=x.device).float()
        processed = x / (2.0 ** scale_factor)
        processed = torch.clamp(processed, 0.0, 1.0)
        if mask is None:
            return processed
        return processed * mask + x * (1.0 - mask)

    def _input_admix_and_scale(self, x: torch.Tensor, noise: torch.Tensor, i: int, prob: float = 1.0) -> torch.Tensor:
        scale_factor = torch.randint(0, 5, (1,), device=x.device).float()
        processed = (x + 0.2 * noise[:, i]) / (2.0 ** scale_factor)
        return torch.clamp(processed, 0.0, 1.0)

    def _input_brightness(self, x: torch.Tensor, factor_delta: float = 0.5, prob: float = 1.0) -> torch.Tensor:
        factor = torch.empty(1, device=x.device).uniform_(1 - factor_delta, 1 + factor_delta)
        degenerate = torch.zeros_like(x)
        return self._blend(degenerate, x, factor)

    def _input_color(self, x: torch.Tensor, factor_delta: float = 0.5, prob: float = 1.0) -> torch.Tensor:
        factor = torch.empty(1, device=x.device).uniform_(1 - factor_delta, 1 + factor_delta)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = gray.expand_as(x)
        return self._blend(gray, x, factor)

    def _input_contrast(self, x: torch.Tensor, factor_delta: float = 0.5, prob: float = 1.0) -> torch.Tensor:
        factor = torch.empty(1, device=x.device).uniform_(1 - factor_delta, 1 + factor_delta)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        mean = gray.mean(dim=[2, 3], keepdim=True)
        degenerate = mean.expand_as(x)
        return self._blend(degenerate, x, factor)

    def _input_sharpness(self, x: torch.Tensor, factor_delta: float = 0.5, prob: float = 1.0) -> torch.Tensor:
        factor = torch.empty(1, device=x.device).uniform_(1 - factor_delta, 1 + factor_delta)
        sharp_kernel = torch.tensor(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=torch.float32, device=x.device
        )
        sharp_kernel = sharp_kernel / sharp_kernel.sum()
        sharp_kernel = sharp_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        degenerate = F.conv2d(x, sharp_kernel, padding=1, groups=3)
        return self._blend(degenerate, x, factor)

    def _input_invert(self, x: torch.Tensor, prob: float = 1.0) -> torch.Tensor:
        return 1.0 - x

    def _rgb_to_hsv(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        maxc, _ = x.max(dim=1)
        minc, _ = x.min(dim=1)
        v = maxc + 1e-10
        c = maxc - minc + 1e-10

        h = torch.zeros_like(maxc)
        mask = c != 0

        rc = (((g - b) / c) / 6.0) % 1.0
        gc = ((2.0 + (b - r) / c) / 6.0) % 1.0
        bc = ((4.0 + (r - g) / c) / 6.0) % 1.0

        h = torch.where((maxc == r) & mask, rc, h)
        h = torch.where((maxc == g) & mask, gc, h)
        h = torch.where((maxc == b) & mask, bc, h)

        s = torch.where(v == 0, torch.zeros_like(v), c / v)
        hsv = torch.stack([h, s, v], dim=1)
        return hsv

    def _hsv_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        h, s, v = x[:, 0], x[:, 1], x[:, 2]
        c = s * v
        m = v - c
        dh = h * 6.0
        hi = dh.long()
        f = dh - hi.float()
        p = v - c
        q = v - f * c
        t = v - (1.0 - f) * c

        zeros = torch.zeros_like(h)
        r = zeros.clone()
        g = zeros.clone()
        b = zeros.clone()

        for idx in range(6):
            mask = hi == idx
            if idx == 0:
                r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
            elif idx == 1:
                r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
            elif idx == 2:
                r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
            elif idx == 3:
                r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
            elif idx == 4:
                r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
            elif idx == 5:
                r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

        rgb = torch.stack([r, g, b], dim=1)
        return rgb

    def _input_hue(self, x: torch.Tensor, delta: float = 0.2, prob: float = 1.0) -> torch.Tensor:
        B, C, H, W = x.shape
        rand_delta = torch.empty(B, 1, 1, device=x.device).uniform_(-delta, delta)
        y = self._rgb_to_hsv(x)
        h, s, v = y[:, 0], y[:, 1], y[:, 2]
        h = h + rand_delta
        h = torch.clamp(h, 0.0, 1.0)
        y = torch.stack([h, s, v], dim=1)
        y = self._hsv_to_rgb(y)
        return torch.clamp(y, 0.0, 1.0)

    def _input_saturation(self, x: torch.Tensor, delta: float = 0.5, prob: float = 1.0) -> torch.Tensor:
        """饱和度调整"""
        B, C, H, W = x.shape
        rand_scale = torch.empty(B, 1, 1, device=x.device).uniform_(1 - delta, 1 + delta)
        y = self._rgb_to_hsv(x)
        h, s, v = y[:, 0], y[:, 1], y[:, 2]
        s = s * rand_scale
        s = torch.clamp(s, 0.0, 1.0)
        y = torch.stack([h, s, v], dim=1)
        y = self._hsv_to_rgb(y)
        return torch.clamp(y, 0.0, 1.0)

    def _input_gamma(self, x: torch.Tensor, delta: float = 0.4, prob: float = 1.0) -> torch.Tensor:
        """Gamma 调整"""
        B, C, H, W = x.shape
        rand_gamma = torch.empty(B, 1, 1, 1, device=x.device).uniform_(1 - delta, 1 + delta)
        y = x + 1e-10
        y = torch.pow(y, rand_gamma)
        return torch.clamp(y, 0.0, 1.0)

    def _input_identity(self, x: torch.Tensor, noise: torch.Tensor, idx: int) -> torch.Tensor:
        """恒等变换"""
        return x

    def _apply_ops(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        op_ids: torch.Tensor,
        noise_idx: int,
    ) -> torch.Tensor:
        
        if op_ids.dim() == 1:
            op_ids = op_ids.unsqueeze(0).expand(x.size(0), -1)

        op_table = {
            0: lambda img, n, idx: self._input_admix(img, n, idx),
            1: lambda img, n, idx: self._input_scale(img),
            2: lambda img, n, idx: self._input_admix_and_scale(img, n, idx),
            3: lambda img, n, idx: self._input_brightness(img, 0.5),
            4: lambda img, n, idx: self._input_color(img, 0.5),
            5: lambda img, n, idx: self._input_contrast(img, 0.5),
            6: lambda img, n, idx: self._input_sharpness(img, 0.5),
            7: lambda img, n, idx: self._input_invert(img),
            8: lambda img, n, idx: self._input_hue(img, 0.2),
            9: lambda img, n, idx: self._input_saturation(img, 0.5),
            10: lambda img, n, idx: self._input_gamma(img, 0.4),
        }

        for t in range(op_ids.size(1)):
            op_id = int(op_ids[0, t].item())
            fn = op_table.get(op_id, self._input_identity)
            x = fn(x, noise, noise_idx)
        return x

    #对同一图像重复执行 n_variants 次随机变换，返回变换后的图像列表。
    def _apply_ops_multi(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        op_ids: torch.Tensor,
        noise_idx: int,
        n_variants: int = 4,
    ) -> list[torch.Tensor]:
        
        variants = []
        for _ in range(n_variants):
            variants.append(self._apply_ops(x, noise, op_ids, noise_idx))
        return variants




    def attack(self):
        args = self.args
        device = args.device
        net = self.net.to(device)
        net.eval()

        data_loader = DataLoader(
            self.dataset, batch_size=args.batch_size, num_workers=10
        )
        epsilon = args.eps / 255.0
        alpha = args.alpha / 255.0

        for images, te, filename in tqdm(data_loader):
            images = images.to(device)
            te = te.to(device)
            images_adv = images.clone().detach()

            B, C, H, W = images.shape
            num_noise = args.num_noise
            encoder_length = args.encoder_length

            # 初始化噪声图
            noise_x = torch.rand(B, num_noise, C, H, W, device=device)
            noise_x = torch.clamp(noise_x, 0.0, 1.0)

            # 使用 Encoder-Decoder 预测操作序列
            op_chosen = self._predict_op_sequences(
                net,
                images,
                num_seeds=args.AITL_predict_num_seeds,
                num_steps=args.AITL_predict_num_steps
            )
            
            if op_chosen.size(1) < num_noise:
                num_missing = num_noise - op_chosen.size(1)
                if num_missing > 0:
                    additional_ops = torch.randint(
                        low=0,
                        high=args.AITL_decoder_vocab_size,
                        size=(B, num_missing, encoder_length),
                        device=device,
                        dtype=torch.long,
                    )
                    op_chosen = torch.cat([op_chosen, additional_ops], dim=1)
            
            op_chosen = op_chosen[:, :num_noise]

            last_g = torch.zeros_like(images_adv, dtype=torch.float32, device=device)

            for _ in range(args.epochs):
                x_adv = images_adv.clone().detach().requires_grad_(True)

                grad_list = []
                models = [net] 
                total_calls = len(models) * num_noise * 3
                call_idx = 0

                for m in models:
                    for ix in range(num_noise):
                        variants = self._apply_ops_multi(x_adv, noise_x, op_chosen[:, ix], ix, n_variants=3)
                        for v in variants:
                            outputs = m(v)
                            loss = self.loss(outputs, te)
                            retain = call_idx < total_calls - 1
                            grad_i = torch.autograd.grad(
                                loss, x_adv, retain_graph=retain, create_graph=False
                            )[0]
                            grad_list.append(grad_i)
                            call_idx += 1

                grad = torch.stack(grad_list, dim=0).mean(dim=0)
                grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-12
                g = last_g * args.mu + grad / grad_norm

                images_adv = images_adv + alpha * torch.sign(g)
                images_adv = torch.where(
                    images_adv > images + epsilon, images + epsilon, images_adv
                )
                images_adv = torch.where(
                    images_adv < images - epsilon, images - epsilon, images_adv
                )
                images_adv = images_adv.clamp(0.0, 1.0).detach()
                last_g = g.detach()

            self.save_images(images_adv, te, filename)

        eval_result = self.eval()

        return eval_result


__all__ = ["AITL_Attack"]
