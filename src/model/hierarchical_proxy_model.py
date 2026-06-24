"""
Hierarchical proxy model for DC selection and carrier assignment.

The model emits explicit hierarchical outputs:
  logits_dc [B, D]
  logits_carrier [B, D, C]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mlp import MLP


class _ScenarioBranch(nn.Module):
    """Embed either demand (E*1) or cost (E*D*C) scenarios and aggregate."""
    def __init__(self, in_dim, hidden, agg="mean", p=0.2):
        super().__init__()
        self.agg, self.p = agg, p
        self.l0 = nn.Linear(in_dim, hidden)
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)

    def forward(self, x):  # x:[B,E,in_dim]
        x = F.relu(self.l0(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.l1(x))
        if self.agg == "sum":
            x = x.sum(1)
        else:
            x = x.mean(1)
        x = F.relu(self.l2(x))
        return x


class ScenarioModule(nn.Module):
    """Separate demand / cost branches -> fuse."""
    def __init__(self, cost_dim, hidden, p=0.2, combine="concat", agg="mean"):
        super().__init__()
        self.dem = _ScenarioBranch(1, hidden, p=p, agg=agg)
        self.cost = _ScenarioBranch(cost_dim, hidden, p=p, agg=agg)
        self.combine = combine
        if combine == "concat":
            self.fuse = nn.Linear(2 * hidden, hidden)
        else:
            self.fuse = nn.Identity()

    def forward(self, dem, cost):  # dem:[B,E], cost:[B,E,D*C]
        z1 = self.dem(dem.unsqueeze(-1))
        z2 = self.cost(cost)
        if self.combine == "concat":
            return F.relu(self.fuse(torch.cat([z1, z2], -1)))
        return z1 + z2


class StandardModule(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout_p):
        super().__init__()
        self.mlp = MLP(
            [in_dim] + [hidden] * n_layers,
            do_bn=True,
            linear=True,
            dropout=True,
            p=dropout_p
        )

    def forward(self, x):
        return self.mlp(x)


class DCModule(nn.Module):
    """Shared MLP over per-DC features."""
    def __init__(self, feature_dim, hidden, n_layers, dropout_p):
        super().__init__()
        self.mlp = MLP(
            [feature_dim] + [hidden] * n_layers,
            do_bn=True,
            linear=True,
            dropout=True,
            p=dropout_p
        )

    def forward(self, dc_feats):
        B, D, K = dc_feats.shape
        dc_flat = dc_feats.view(B * D, K)
        out = self.mlp(dc_flat)
        return out.view(B, D, -1)


class OutputModule(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers, p):
        super().__init__()
        self.mlp = MLP(
            [in_dim] + [hidden] * (n_layers - 1) + [out_dim],
            do_bn=True,
            linear=True,
            dropout=True,
            p=p
        )

    def forward(self, x):
        return self.mlp(x)


class HierarchicalProxyModel(nn.Module):
    """
    Hierarchical proxy model used by the main paper workflow.

    Architecture:
    1) Standard features (global + SKU/brand) -> z_std
    2) Scenario module (demand + full cost scenarios) -> z_scen
    3) DC module over full per-DC features -> z_dc
    4) Per-DC cost summary (carrier stats + scenario penalty stats) -> z_cost
    5) DC head: per-DC context -> logits_dc [B, D]
    6) Carrier head: per-DC context + DC embedding -> logits_carrier [B, D, C]

    Optional ablation switches:
    - use_scenario_module
    - use_dc_module
    - use_cost_summary
    - use_option_features_in_carrier
    - use_dc_embedding
    - use_carrier_embedding
    - scenario_combine
    """
    def __init__(
        self,
        global_feature_dim,
        dc_feature_dim,
        option_feature_dim,  # kept for API compatibility
        num_dcs,
        num_carriers,
        sku_dim,
        brand_dim,
        sku_emb_dim=8,
        brand_emb_dim=6,
        hidden_dim=128,
        n_layers=3,
        dropout_p=0.2,
        agg_type='mean',
        use_num_proj=False,  # not used, for API compatibility
        dc_embedding_dim=32,
        carrier_embedding_dim=8,
        option_proj_dim=8,
        use_option_features_in_carrier: bool = True,
        use_cost_summary: bool = True,
        use_scenario_module: bool = True,
        use_dc_module: bool = True,
        use_dc_embedding: bool = True,
        use_carrier_embedding: bool = True,
        scenario_combine: str = 'add',
    ):
        super().__init__()
        if scenario_combine not in {'add', 'concat'}:
            raise ValueError(f"scenario_combine must be 'add' or 'concat', got {scenario_combine!r}")
        self.D = num_dcs
        self.C = num_carriers
        self.H = hidden_dim
        self.use_cost_summary = bool(use_cost_summary)
        self.use_option_features_in_carrier = bool(use_option_features_in_carrier)
        self.use_scenario_module = bool(use_scenario_module)
        self.use_dc_module = bool(use_dc_module)
        self.use_dc_embedding = bool(use_dc_embedding)
        self.use_carrier_embedding = bool(use_carrier_embedding)
        self.scenario_combine = scenario_combine

        # Embeddings
        self.sku_emb = nn.Embedding(sku_dim, sku_emb_dim)
        self.brand_emb = nn.Embedding(brand_dim, brand_emb_dim)
        self.dc_emb = nn.Embedding(num_dcs, dc_embedding_dim) if self.use_dc_embedding else None
        self.carrier_emb = nn.Embedding(num_carriers, carrier_embedding_dim) if self.use_carrier_embedding else None
        dc_emb_dim = dc_embedding_dim if self.use_dc_embedding else 0
        carrier_emb_dim = carrier_embedding_dim if self.use_carrier_embedding else 0

        # Feature processing modules
        self.std_mod = StandardModule(
            global_feature_dim + sku_emb_dim + brand_emb_dim,
            hidden_dim, n_layers, dropout_p
        )

        cost_dim = num_dcs * num_carriers
        self.scen_mod = (
            ScenarioModule(
                cost_dim=cost_dim,
                hidden=hidden_dim,
                p=dropout_p,
                combine=scenario_combine,
                agg=agg_type,
            )
            if self.use_scenario_module
            else None
        )

        self.dc_mod = (
            DCModule(dc_feature_dim, hidden_dim, n_layers, dropout_p)
            if self.use_dc_module
            else None
        )

        # Per-DC cost summary projection (base cost stats + scenario penalty stats)
        self.cost_proj = None
        if self.use_cost_summary:
            self.cost_proj = MLP(
                [8] + [hidden_dim] * (n_layers - 1) + [hidden_dim],
                do_bn=True,
                linear=True,
                dropout=True,
                p=dropout_p
            )

        self.option_proj = None
        if self.use_option_features_in_carrier and option_feature_dim > 0:
            self.option_proj = MLP(
                [option_feature_dim] + [option_proj_dim],
                do_bn=True,
                linear=True,
                dropout=True,
                p=dropout_p
            )

        # Heads
        # DC head operates per-DC on [z_std, z_scen, z_dc, z_cost]
        self.dc_head = OutputModule(4 * hidden_dim, hidden_dim, 1, n_layers, dropout_p)
        # Carrier head scores per (DC, carrier) with carrier embedding + dc logit conditioning
        carrier_in_dim = 4 * hidden_dim + 1 + dc_emb_dim + carrier_emb_dim
        if self.option_proj is not None:
            carrier_in_dim += option_proj_dim
        self.carrier_head = OutputModule(carrier_in_dim, hidden_dim, 1, n_layers, dropout_p)

    def forward(self, global_feats, dc_feats, option_feats, demand_scenarios,
                delivery_penalty, sku_idx, brand_idx, return_raw_logits=True):
        """
        Returns:
            (logits_dc [B, D], logits_carrier [B, D, C])
        """
        B = global_feats.size(0)

        global_feats = global_feats.float()
        dc_feats = dc_feats.float()
        option_feats = option_feats.float()
        demand_scenarios = demand_scenarios.float()
        delivery_penalty = delivery_penalty.float()

        # Standard/global features
        std_cat = torch.cat([
            global_feats,
            self.sku_emb(sku_idx).flatten(start_dim=1),
            self.brand_emb(brand_idx).flatten(start_dim=1)
        ], dim=-1)
        z_std = self.std_mod(std_cat)  # [B, H]

        # Scenario features (global entangled cost)
        if self.scen_mod is not None:
            base_cost = option_feats[:, :, :, 0]  # [B, D, C]
            scenario_cost = delivery_penalty + base_cost.unsqueeze(1)  # [B, S, D, C]
            scenario_cost_flat = scenario_cost.view(B, scenario_cost.size(1), -1)  # [B, S, D*C]
            z_scen = self.scen_mod(demand_scenarios, scenario_cost_flat)  # [B, H]
        else:
            z_scen = torch.zeros(B, self.H, device=global_feats.device)

        # DC features
        if self.dc_mod is not None:
            z_dc = self.dc_mod(dc_feats)  # [B, D, H]
        else:
            z_dc = torch.zeros(B, self.D, self.H, device=dc_feats.device)

        # Per-DC cost summary: base_cost stats + scenario penalty stats, then project (optional)
        if self.use_cost_summary and self.cost_proj is not None:
            base_cost = option_feats[:, :, :, 0]  # [B, D, C]
            mean_dc = base_cost.mean(dim=2)  # [B, D]
            min_dc = base_cost.min(dim=2).values  # [B, D]
            std_dc = base_cost.std(dim=2)  # [B, D]
            p90_dc = torch.quantile(base_cost, 0.9, dim=2)  # [B, D]
            # Gap between best and second best
            sorted_costs, _ = torch.sort(base_cost, dim=2)  # [B, D, C]
            best_cost = sorted_costs[:, :, 0]  # [B, D]
            second_best_cost = sorted_costs[:, :, 1]  # [B, D]
            gap_dc = second_best_cost - best_cost  # [B, D]
            # Scenario penalty stats per DC (aggregate across scenarios and carriers)
            pen_flat = delivery_penalty.permute(0, 2, 1, 3).contiguous().view(B, self.D, -1)
            pen_mean = pen_flat.mean(dim=2)  # [B, D]
            pen_std = pen_flat.std(dim=2, unbiased=False)  # [B, D]
            pen_p90 = torch.quantile(pen_flat, 0.9, dim=2)  # [B, D]
            cost_stats = torch.stack(
                [mean_dc, min_dc, std_dc, p90_dc, gap_dc, pen_mean, pen_std, pen_p90],
                dim=-1
            )  # [B, D, 8]
            cost_flat = cost_stats.view(B * self.D, -1)
            z_cost = self.cost_proj(cost_flat).view(B, self.D, -1)  # [B, D, H]
        else:
            z_cost = torch.zeros(B, self.D, self.H, device=dc_feats.device)

        # Build per-DC context
        z_global = torch.cat([z_std, z_scen], dim=-1)  # [B, 2H]
        z_global_exp = z_global.unsqueeze(1).expand(B, self.D, 2 * self.H)
        dc_context = torch.cat([z_global_exp, z_dc, z_cost], dim=-1)  # [B, D, 4H]

        # DC head (per-DC)
        dc_flat = dc_context.view(B * self.D, -1)
        logits_dc = self.dc_head(dc_flat).squeeze(-1).view(B, self.D)

        # Carrier head (per-DC), conditioned on DC logits
        dc_logit_feat = logits_dc.unsqueeze(-1)  # [B, D, 1]
        carrier_base_parts = [dc_context, dc_logit_feat]
        if self.dc_emb is not None:
            dc_embs = self.dc_emb.weight.unsqueeze(0).expand(B, -1, -1)
            carrier_base_parts.append(dc_embs)
        carrier_base = torch.cat(carrier_base_parts, dim=-1)  # [B, D, F]
        carrier_base = carrier_base.unsqueeze(2).expand(B, self.D, self.C, -1)

        carrier_input_parts = [carrier_base]
        if self.carrier_emb is not None:
            carrier_embs = self.carrier_emb.weight.unsqueeze(0).unsqueeze(0).expand(B, self.D, self.C, -1)
            carrier_input_parts.append(carrier_embs)

        if self.option_proj is not None:
            opt_flat = option_feats.view(B * self.D * self.C, -1)
            opt_proj = self.option_proj(opt_flat).view(B, self.D, self.C, -1)
            carrier_input_parts.append(opt_proj)

        carrier_input = torch.cat(carrier_input_parts, dim=-1)  # [B, D, C, F]
        carrier_flat = carrier_input.view(B * self.D * self.C, -1)
        logits_carrier = self.carrier_head(carrier_flat).view(B, self.D, self.C)

        if return_raw_logits:
            return logits_dc, logits_carrier
        return torch.softmax(logits_dc, dim=1), torch.softmax(logits_carrier, dim=2)
