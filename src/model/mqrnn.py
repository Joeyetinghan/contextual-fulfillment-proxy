import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.mlp import MLP   

# ── ordered‑quantile head ───────────────────────────────
class OrderedQuantileHead(nn.Module):
    def __init__(self, in_dim, num_q, dropout=False, dropout_p=0.5):
        super().__init__()
        self.mid  = num_q // 2
        self.mlp  = MLP(channels=[in_dim, num_q], linear=True, do_bn=True, dropout=dropout, p=dropout_p)

    def forward(self, z):
        logits = self.mlp(z)
        logits = logits.clamp(-10, 10)      # prevent overflow
        m  = logits[:, self.mid:self.mid+1]
        up = F.softplus(logits[:, self.mid+1:])
        dn = F.softplus(logits[:, :self.mid])
        q_up = m + torch.cumsum(up, 1)
        q_dn = m - torch.cumsum(dn.flip(1), 1).flip(1)
        q_raw = torch.cat([q_dn, m, q_up], 1)
        return torch.relu(q_raw)

# ── two‑level decoder  ───────────────────────────────────
class TwoLevelDecoder(nn.Module):
    def __init__(self, 
                 hidden: int, 
                 ctx: int, 
                 num_q: int, 
                 pred_length: int, 
                 dropout: bool = False, 
                 dropout_p: float = 0.5):
        super().__init__()
        self.pred_length = pred_length
        self.ctx = ctx
        # Input: the encoder output (hidden) + the future inputs (ctx * pred_length)
        self.global_mlp = MLP(channels=[hidden + ctx * pred_length, ctx * (pred_length + 1)], 
                              linear=True, do_bn=True, dropout=dropout, p=dropout_p)
        # Input: the horizon-specific context (ctx), the horizon-agnostic context (ctx), and the future input (ctx)
        self.local_head = OrderedQuantileHead(ctx*3, num_q, 
                                              dropout=dropout, dropout_p=dropout_p)

    def forward(self, h_T, x_pred):
        B = h_T.size(0)
        flat = x_pred.reshape(B, -1)
        g_ctx = torch.sigmoid(self.global_mlp(torch.cat([h_T, flat], 1)))
        c_a   = g_ctx[:, -self.ctx:]
        out = []
        for k in range(self.pred_length):
            c_k = g_ctx[:, k*self.ctx:(k+1)*self.ctx]
            fused = torch.cat([c_k, c_a, x_pred[:, k]], 1)
            out.append(self.local_head(fused).unsqueeze(1))
        return torch.cat(out, 1)        # (B, Lp, Q)

# ── full model with TWO encoders ────────────────────────────────────
class MQRNN(nn.Module):
    """
    • enc_hist : processes calendar + order‑stats + ID embeddings
    • enc_pred : processes calendar + ID embeddings
    """
    def __init__(self,
                 num_cal: int,             # calendar feature dim (4)
                 num_ord: int,             # order‑stat feature dim (18)
                 num_skus: int,
                 num_brands: int,
                 sku_emb: int = 16,
                 brand_emb: int = 12,
                 hidden: int = 64,  # hidden dimension
                 ctx: int = 32,    # context dimension
                 num_q: int = 11,  # number of quantiles
                 Lp: int = 24,     # prediction length
                 Lh: int = 36,     # history length
                 layers: int = 1,   # number of layers in LSTM
                 dropout: bool = False,
                 dropout_p: float = 0.5,
                 bidirectional: bool = False):
        super().__init__()
        self.Lp, self.Lh = Lp, Lh

        # Embeddings
        self.sku_emb   = nn.Embedding(num_skus,   sku_emb)
        self.brand_emb = nn.Embedding(num_brands, brand_emb)
        emb_dim = sku_emb + brand_emb

        # Dimensions
        self.x_hist_dim = num_cal + num_ord + emb_dim
        self.x_pred_dim = num_cal + emb_dim

        # Two encoders (history and prediction)
        self.enc_hist = MLP(channels=[self.x_hist_dim, ctx], linear=True, do_bn=True, dropout=dropout, p=dropout_p)
        self.enc_pred = MLP(channels=[self.x_pred_dim, ctx], linear=True, do_bn=True, dropout=dropout, p=dropout_p)

        # LSTM on history (ctx + target)
        self.rnn = nn.LSTM(ctx + 1, hidden, layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

        # Two‑level decoder
        self.decoder = TwoLevelDecoder(hidden, ctx,
                                       num_q, Lp, dropout=dropout, dropout_p=dropout_p)


    # ---------- helpers ----------
    def _embed_ids(self, sku_idx, brand_idx, L):
        sku_v   = self.sku_emb(sku_idx)        # (B, d1)
        brand_v = self.brand_emb(brand_idx)    # (B, d2)
        rep = torch.cat([sku_v, brand_v], 1)   # (B, emb_dim)
        return rep.unsqueeze(1).expand(-1, L, -1)

    # ---------- forward ----------
    def forward(self, x_hist_num, y_hist,
                      x_pred_num, sku_idx, brand_idx):
        """
        x_hist_num : (B, Lh, num_cal + num_ord)     – numeric part (history)
        x_pred_num : (B, Lp, num_cal)               – numeric part (prediction)
        """
        # build full feature tensors with embeddings
        id_hist = self._embed_ids(sku_idx, brand_idx, self.Lh)
        id_pred = self._embed_ids(sku_idx, brand_idx, self.Lp)
        xh_full = torch.cat([x_hist_num, id_hist], 2)   # (B,Lh,x_hist_dim)
        xp_full = torch.cat([x_pred_num, id_pred], 2)   # (B,Lp,x_pred_dim)

        # encode
        xh_ctx = torch.tanh(self.enc_hist(xh_full))
        xp_ctx = torch.tanh(self.enc_pred(xp_full))

        # RNN on history (+ target)
        y_unsq = y_hist.unsqueeze(2)                   # (B,Lh,1)
        rnn_in = torch.cat([xh_ctx, y_unsq], 2)
        self.rnn.flatten_parameters()
        _, (h_T, _) = self.rnn(rnn_in)
        if self.rnn.bidirectional:
            # h_T shape: (2*layers, B, hidden)
            # concat last layer's fwd (−2) & bwd (−1)
            h_T = torch.cat([h_T[-2], h_T[-1]], dim=1)   # (B, 2*hidden)
        else:
            h_T = h_T[-1]                                # (B, hidden)
        q_hat = self.decoder(h_T, xp_ctx)            # (B,Lp,Q)
        return q_hat
