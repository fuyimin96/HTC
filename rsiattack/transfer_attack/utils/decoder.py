import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.num_layers = params["decoder_num_layers"]
        self.hidden_size = params["decoder_hidden_size"]
        self.length = params["decoder_length"]
        self.vocab_size = params["decoder_vocab_size"]
        self.dropout_p = params.get("decoder_dropout", 0.0)

        layers = []
        in_dim = params["encoder_hidden_size"]
        for i in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout_p > 0.0:
                layers.append(nn.Dropout(self.dropout_p))
            layers.append(
                nn.BatchNorm1d(
                    self.hidden_size,
                    eps=1e-5,
                    momentum=0.9,
                )
            )
            in_dim = self.hidden_size
        self.mlp = nn.Sequential(*layers)

        self.proj = nn.Linear(in_dim, self.length * self.vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Match the original TF random_uniform_initializer(-0.1, 0.1) roughly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = encoder_outputs  # [B, H]
        x = self.mlp(x) if self.mlp is not None else x
        x = self.proj(x)  # [B, L * V]
        x = x.view(x.size(0), self.length, self.vocab_size)
        logits = x
        sample_id = torch.argmax(logits, dim=-1)
        return logits, sample_id


class EPDDecoderModel(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params
        self.vocab_size = params["decoder_vocab_size"]
        self.decoder_length = params["decoder_length"]
        self.weight_decay = params.get("weight_decay", 0.0)

        self.decoder = Decoder(params)

    def forward(
        self, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(encoder_outputs)

    def compute_loss(
        self,
        encoder_outputs: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        logits, sample_id = self.forward(encoder_outputs)
        # Flatten for CE: [B * L, V]
        B, L, V = logits.shape
        ce = F.cross_entropy(
            logits.view(B * L, V),
            target.view(B * L),
            reduction="mean",
        )

        # L2 weight decay
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
            total_loss = ce + self.weight_decay * l2
        else:
            total_loss = ce

        with torch.no_grad():
            correct = (sample_id == target).float().mean()

        return {
            "loss": total_loss,
            "ce_loss": ce,
            "correct_rate": correct,
            "logits": logits,
            "sample_id": sample_id,
        }


def build_optimizer(
    model: nn.Module, params: Dict
) -> torch.optim.Optimizer:

    lr = float(params.get("lr", 1e-3))
    opt_name = params.get("optimizer", "adam").lower()

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    else:  # default adam
        assert lr <= 1e-3, f"High Adam learning rate {lr}"
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return optimizer


