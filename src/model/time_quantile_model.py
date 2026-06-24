import torch, torch.nn as nn
import torch.nn.functional as F
import src.config as cfg
from src.model.mlp import MLP

class TimeQuantileModel(nn.Module):
    def __init__(        
            self,
            numerical_dim: int,
            hidden_dim: int,
            n_layers: int,
            dropout: bool,
            dropout_p: float,
            dc_ori_vocab_size: int | None = None,
            dc_des_vocab_size: int | None = None,
            dc_ori_embedding_dim: int = 8,
            dc_des_embedding_dim: int = 8,
            target_min: float = 1.0,          # lower bound of y
            target_max: float = 5.0,          # upper bound of y
            quantiles: list[float] = cfg.DELIVERY_TIME_QUANTILES
        ):
        super().__init__()
        
        # Store embedding parameters
        self.use_embeddings = dc_ori_vocab_size is not None and dc_des_vocab_size is not None
        self.target_min = target_min
        self.target_range = target_max - target_min
        self.nq = len(quantiles)
        self.mid_idx = quantiles.index(0.5)           # assumes 0.5 is present
        self.t_min   = target_min
        self.t_rng   = target_max - target_min

        # ── Embeddings (optional) ────────────────────────────────────────────────
        extra_dim = 0
        if self.use_embeddings:
            self.dc_ori_embedding = nn.Embedding(dc_ori_vocab_size, dc_ori_embedding_dim)
            self.dc_des_embedding = nn.Embedding(dc_des_vocab_size, dc_des_embedding_dim)
            extra_dim = dc_ori_embedding_dim + dc_des_embedding_dim

        # ── Core MLP ────────────────────────────────────────────────────────────
        input_dim = numerical_dim + extra_dim
        channels = [input_dim] + [hidden_dim] * n_layers + [self.nq]
        self.net = MLP(
            channels=channels,
            do_bn=True,
            linear=True,
            dropout=dropout,
            p=dropout_p,
        )

    def forward(self, x_num: torch.Tensor,
                      x_dc_ori: torch.Tensor | None = None,
                      x_dc_des: torch.Tensor | None = None
               ) -> torch.Tensor:
        """
        Returns tensor of shape (batch, len(QUANTILES))
        Guaranteed:
            • monotone increasing across columns
            • each element ∈ [target_min, target_max]
        """        
        if self.use_embeddings:
            x = torch.cat(
                [x_num,
                 self.dc_ori_embedding(x_dc_ori),
                 self.dc_des_embedding(x_dc_des)],
                dim=1
            )
        else:
            x = x_num

        logits = self.net(x)                         # (B, nq)
        logits = logits.clamp(-10, 10)      # prevent overflow
        
        # ── softplus trick ─────────────────────────────────────────────────────
        m  = logits[:, self.mid_idx:self.mid_idx+1]             # median anchor
        up = F.softplus(logits[:, self.mid_idx+1:])             # gaps ↑
        dn = F.softplus(logits[:, :self.mid_idx])               # gaps ↓

        q_up = m + torch.cumsum(up, dim=1)                      # higher τs
        q_dn = m - torch.cumsum(dn.flip(1), dim=1).flip(1)      # lower τs

        q_raw = torch.cat([q_dn, m, q_up], dim=1)               # ordered

        # ── bound to [1,5] with a monotone map ────────────────────────────────
        q_bounded = self.target_min + self.target_range * torch.sigmoid(q_raw)
        return q_bounded