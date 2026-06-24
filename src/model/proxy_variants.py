import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mlp import MLP
from src.model.hierarchical_proxy_model import HierarchicalProxyModel


SUPPORTED_PROXY_ARCHITECTURES = {'hierarchical_proxy_v2', 'single_tower'}


def canonicalize_architecture(arch: str) -> str:
    return arch


class SingleTowerProxyModel(nn.Module):
    """
    Single-tower option scoring model (Tier 1/3).

    Builds per-option features by concatenating:
    - Global features + SKU/brand embeddings + demand summary
    - DC features
    - Option features
    - Delivery penalty mean per option

    Outputs unified logits [B, D*C].
    """
    def __init__(
        self,
        global_feature_dim,
        dc_feature_dim,
        option_feature_dim,
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
        use_num_proj=False,
    ):
        super().__init__()
        self.D = num_dcs
        self.C = num_carriers
        self.H = hidden_dim
        self.use_num_proj = use_num_proj

        self.sku_emb = nn.Embedding(sku_dim, sku_emb_dim)
        self.brand_emb = nn.Embedding(brand_dim, brand_emb_dim)

        # Demand summary is a single scalar (mean)
        global_dim = global_feature_dim + sku_emb_dim + brand_emb_dim + 1
        per_option_dim = global_dim + dc_feature_dim + option_feature_dim + 1

        if use_num_proj:
            self.num_proj = MLP(
                [per_option_dim, hidden_dim, hidden_dim],
                do_bn=True, linear=True, dropout=True, p=dropout_p
            )
            head_in = hidden_dim
        else:
            self.num_proj = None
            head_in = per_option_dim

        self.scoring_head = MLP(
            [head_in] + [hidden_dim] * (n_layers - 1) + [1],
            do_bn=True, linear=True, dropout=True, p=dropout_p
        )

    def forward(self, global_feats, dc_feats, option_feats, demand_scenarios,
                delivery_penalty, sku_idx, brand_idx, return_raw_logits=True):
        B = global_feats.size(0)

        global_feats = global_feats.float()
        dc_feats = dc_feats.float()
        option_feats = option_feats.float()
        demand_scenarios = demand_scenarios.float()
        delivery_penalty = delivery_penalty.float()

        # Global feature block
        dem_mean = demand_scenarios.mean(dim=1, keepdim=True)  # [B, 1]
        global_cat = torch.cat([
            global_feats,
            self.sku_emb(sku_idx).flatten(start_dim=1),
            self.brand_emb(brand_idx).flatten(start_dim=1),
            dem_mean,
        ], dim=-1)  # [B, G]

        # Expand globals to options
        global_exp = global_cat.unsqueeze(1).unsqueeze(2).expand(B, self.D, self.C, -1)
        dc_exp = dc_feats.unsqueeze(2).expand(B, self.D, self.C, -1)

        # Delivery penalty summary per option
        delivery_mean = delivery_penalty.mean(dim=1, keepdim=False)  # [B, D, C]
        delivery_mean = delivery_mean.unsqueeze(-1)  # [B, D, C, 1]

        feats = torch.cat([global_exp, dc_exp, option_feats, delivery_mean], dim=-1)  # [B, D, C, F]
        feats_flat = feats.view(B * self.D * self.C, -1)

        if self.num_proj is not None:
            feats_flat = self.num_proj(feats_flat)

        scores_flat = self.scoring_head(feats_flat).squeeze(-1)  # [B*D*C]
        logits = scores_flat.view(B, -1)  # [B, D*C]

        if return_raw_logits:
            return logits
        return F.softmax(logits, dim=1)


def build_proxy_model(model_params: dict):
    """
    Build a proxy model from checkpoint/config parameters.

    Supported architectures:
    - hierarchical_proxy_v2
    - single_tower
    """
    arch = canonicalize_architecture(model_params.get('architecture', 'hierarchical_proxy_v2'))
    if arch == 'hierarchical_proxy_v2':
        return HierarchicalProxyModel(**_filter_params(model_params, arch))
    if arch == 'single_tower':
        return SingleTowerProxyModel(**_filter_params(model_params, arch))
    raise ValueError(f"Unsupported proxy architecture '{arch}'")


def _filter_params(model_params: dict, arch: str) -> dict:
    arch = canonicalize_architecture(arch)
    if arch == 'hierarchical_proxy_v2':
        valid = {
            'global_feature_dim', 'dc_feature_dim', 'option_feature_dim',
            'num_dcs', 'num_carriers',
            'sku_dim', 'brand_dim', 'sku_emb_dim', 'brand_emb_dim',
            'hidden_dim', 'n_layers', 'dropout_p', 'agg_type', 'use_num_proj',
            'dc_embedding_dim', 'carrier_embedding_dim', 'option_proj_dim',
            'use_option_features_in_carrier', 'use_cost_summary',
            'use_scenario_module', 'use_dc_module',
            'use_dc_embedding', 'use_carrier_embedding', 'scenario_combine',
        }
    elif arch == 'single_tower':
        valid = {
            'global_feature_dim', 'dc_feature_dim', 'option_feature_dim',
            'num_dcs', 'num_carriers',
            'sku_dim', 'brand_dim', 'sku_emb_dim', 'brand_emb_dim',
            'hidden_dim', 'n_layers', 'dropout_p', 'agg_type',
            'use_num_proj',
        }
    else:
        raise ValueError(f"Unsupported proxy architecture '{arch}'")
    return {k: v for k, v in model_params.items() if k in valid}
