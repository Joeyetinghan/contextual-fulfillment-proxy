#!/usr/bin/env python3
"""
Generate proxy hyperparameter grids.

Usage:
  # Dry-run to see config count
  python -m scripts.proxy.generate_and_tune_proxy --grid hierarchical_proxy_paper_full --dry-run
  
  # Generate parameter file
  python -m scripts.proxy.generate_and_tune_proxy --grid hierarchical_proxy_paper_full --output configs/proxy/paper/grids/proxy_full.txt
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import itertools
import math
import random

GRID_SAMPLE_SEED = 42


# ============================================================================
# GRID DEFINITIONS
# ============================================================================

def _weighted_choice(rng: random.Random, values: List[Any], weights: List[float]) -> Any:
    return rng.choices(values, weights=weights, k=1)[0]


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return 10.0 ** rng.uniform(math.log10(lo), math.log10(hi))


def _dedupe_preserve_order(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    uniq_configs: List[Dict[str, Any]] = []
    for cfg_item in configs:
        key = tuple(sorted(cfg_item.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq_configs.append(cfg_item)
    return uniq_configs

def generate_hierarchical_proxy_focus_grid(n_jobs=45, configs_per_job=3):
    """
    Focused grid for hierarchical_proxy_v2 tuning.

    Designed to be used with a base config (e.g. hierarchical_proxy_inventory_weighted.json)
    and only override a small, high-impact set of hyperparameters.
    """
    configs = []

    # Grid A: LR / WD / Cost loss sweep
    for lr in [1e-4, 2e-4, 3e-4, 5e-4]:
        for wd in [1e-5, 5e-5, 1e-4]:
            for cost_w in [0.0, 0.05, 0.1]:
                configs.append({
                    'learning_rate': lr,
                    'weight_decay': wd,
                    'cost_loss_weight': cost_w,
                })

    # Grid B: Constraint loss weight
    for c_w in [0.1, 0.2, 0.3, 0.4]:
        configs.append({'constraint_loss_weight': c_w})

    # Grid C: Hidden dim / dropout
    for hd in [128, 160, 192, 224]:
        for do in [0.1, 0.2, 0.3]:
            configs.append({
                'hidden_dim': hd,
                'dropout_p': do,
            })

    # Grid D: Embedding dims
    for dc_emb in [32, 64]:
        configs.append({'dc_embedding_dim': dc_emb})

    for carrier_emb in [8, 16, 24, 32]:
        for opt_proj in [8, 16, 24]:
            configs.append({
                'carrier_emb_dim': carrier_emb,
                'option_proj_dim': opt_proj,
            })

    # Grid E: Carrier class weights on/off (power sweep)
    for power in [0.5, 1.0]:
        configs.append({
            'use_carrier_class_weights': True,
            'carrier_class_weight_power': power,
            'carrier_class_weight_max': 0.0,
        })

    # Grid F: Label smoothing
    for ls in [0.0, 0.05, 0.1]:
        configs.append({'label_smoothing': ls})

    target_count = n_jobs * configs_per_job
    return _trim_or_pad(configs, target_count)


def generate_hierarchical_proxy_sim_cost_focus_grid(n_jobs=35, configs_per_job=4):
    """
    Simulation-oriented grid for hierarchical_proxy_v2.

    Focused around the regime that tends to perform best in simulation:
    - keep cost loss in a low-but-active band (0.01 to 0.08)
    - prefer higher LR among stable values
    - keep dropout moderate
    - increase carrier embedding capacity

    Recommended base config:
      configs/proxy/hierarchical_proxy_main.json
    """
    target_count = n_jobs * configs_per_job
    configs = []

    # Tier A (96): core region around low-but-active cost-loss settings.
    for cost_w in [0.03, 0.05, 0.08]:
        for c_w in [0.1, 0.2]:
            for lr in [5e-4, 3e-4]:
                for wd in [1e-4, 5e-5]:
                    for do in [0.1, 0.2]:
                        for carrier_emb in [16, 32]:
                            configs.append({
                                'learning_rate': lr,
                                'weight_decay': wd,
                                'dropout_p': do,
                                'hidden_dim': 160,
                                'cost_loss_weight': cost_w,
                                'constraint_loss_weight': c_w,
                                'carrier_emb_dim': carrier_emb,
                                'option_proj_dim': 8,
                            })

    # Tier B (32): very low cost-loss edge exploration + narrower hidden dim.
    for cost_w in [0.01, 0.02]:
        for c_w in [0.1, 0.2]:
            for lr in [5e-4, 3e-4]:
                for do in [0.1, 0.2]:
                    for carrier_emb in [16, 32]:
                        configs.append({
                            'learning_rate': lr,
                            'weight_decay': 1e-4,
                            'dropout_p': do,
                            'hidden_dim': 128,
                            'cost_loss_weight': cost_w,
                            'constraint_loss_weight': c_w,
                            'carrier_emb_dim': carrier_emb,
                            'option_proj_dim': 8,
                        })

    # Tier C (8): stronger constraint-loss sweep within the same cost-loss band.
    for c_w in [0.3, 0.4]:
        for do in [0.1, 0.2]:
            for carrier_emb in [16, 32]:
                configs.append({
                    'learning_rate': 5e-4,
                    'weight_decay': 1e-4,
                    'dropout_p': do,
                    'hidden_dim': 160,
                    'cost_loss_weight': 0.05,
                    'constraint_loss_weight': c_w,
                    'carrier_emb_dim': carrier_emb,
                    'option_proj_dim': 8,
                })

    # Tier D (4): high-constraint corner check (cost still in preferred band).
    for do in [0.1, 0.2]:
        for carrier_emb in [16, 32]:
            configs.append({
                'learning_rate': 5e-4,
                'weight_decay': 1e-4,
                'dropout_p': do,
                'hidden_dim': 160,
                'cost_loss_weight': 0.08,
                'constraint_loss_weight': 0.4,
                'carrier_emb_dim': carrier_emb,
                'option_proj_dim': 8,
            })

    return _trim_or_pad(_dedupe_preserve_order(configs), target_count)


def generate_hierarchical_proxy_paper_full_grid(n_jobs=35, configs_per_job=8):
    """
    Paper-facing full grid, cost-biased hybrid search for hierarchical_proxy_v2.

    Design:
      - deterministic anchors from the strongest cost-focused basin
      - focused sampling around low-but-active cost-loss settings
      - smaller exploration tail to preserve discovery

    The search budget is controlled by n_jobs * configs_per_job (280 by default).
    """
    target_count = n_jobs * configs_per_job
    rng = random.Random(GRID_SAMPLE_SEED)
    seen = set()
    configs: List[Dict[str, Any]] = []

    def _append_cfg(cfg: Dict[str, Any]) -> bool:
        key = tuple(sorted(cfg.items()))
        if key in seen:
            return False
        seen.add(key)
        configs.append(cfg)
        return True

    def _anchor_cfg(
        lr: float,
        wd: float,
        do: float,
        cost_w: float,
        c_w: float,
        carrier_emb: int,
        op_dim: int,
        *,
        hd: int = 160,
        nl: int = 3,
        dc_emb: int = 64,
        ls: float = 0.05,
        carrier_w: float = 0.002,
        bs: int = 64,
    ) -> Dict[str, Any]:
        return {
            'learning_rate': lr,
            'weight_decay': wd,
            'dropout_p': do,
            'cost_loss_weight': cost_w,
            'constraint_loss_weight': c_w,
            'carrier_emb_dim': carrier_emb,
            'option_proj_dim': op_dim,
            'hidden_dim': hd,
            'n_layers': nl,
            'dc_embedding_dim': dc_emb,
            'label_smoothing': ls,
            'carrier_loss_weight': carrier_w,
            'batch_size': bs,
        }

    # Anchors keep well-performing cost-focused settings explicitly represented.
    anchor_configs: List[Dict[str, Any]] = [
        _anchor_cfg(3e-4, 1e-4, 0.10, 0.03, 0.10, 32, 8),
        _anchor_cfg(3e-4, 5e-5, 0.10, 0.03, 0.10, 32, 8),
        _anchor_cfg(5e-4, 1e-4, 0.10, 0.03, 0.10, 32, 8),
        _anchor_cfg(3e-4, 1e-4, 0.20, 0.03, 0.10, 32, 8),
        _anchor_cfg(3e-4, 1e-4, 0.10, 0.05, 0.20, 32, 8),
        _anchor_cfg(3e-4, 5e-5, 0.15, 0.02, 0.15, 32, 8),
        _anchor_cfg(2e-4, 3e-5, 0.10, 0.015, 0.10, 16, 8),
        _anchor_cfg(2e-4, 3e-5, 0.15, 0.02, 0.20, 32, 8),
        _anchor_cfg(3e-4, 1e-4, 0.15, 0.03, 0.15, 32, 8, hd=192),
        _anchor_cfg(3e-4, 1e-4, 0.15, 0.02, 0.15, 32, 8, nl=2),
        _anchor_cfg(3e-4, 1e-5, 0.15, 0.015, 0.10, 32, 8, ls=0.03, carrier_w=0.001),
        _anchor_cfg(3e-4, 3e-5, 0.20, 0.02, 0.20, 32, 8, ls=0.08, carrier_w=0.005),
        _anchor_cfg(2e-4, 1e-4, 0.10, 0.02, 0.15, 32, 8, bs=96),
        _anchor_cfg(5e-4, 5e-5, 0.10, 0.02, 0.10, 32, 8),
        _anchor_cfg(3e-4, 1e-4, 0.10, 0.03, 0.10, 16, 8),
        _anchor_cfg(3e-4, 1e-4, 0.10, 0.03, 0.10, 32, 16),
    ]
    for anchor_cfg in anchor_configs:
        _append_cfg(anchor_cfg)
        if len(configs) >= target_count:
            return configs[:target_count]

    exploit_target = min(target_count, max(len(configs), int(round(target_count * 0.78))))
    max_attempts_focus = max(6000, target_count * 80)
    attempts = 0
    while len(configs) < exploit_target and attempts < max_attempts_focus:
        attempts += 1
        cfg = {
            'learning_rate': _weighted_choice(rng, [2e-4, 3e-4, 5e-4], [0.25, 0.50, 0.25]),
            'weight_decay': _weighted_choice(rng, [1e-4, 5e-5, 3e-5, 1e-5], [0.35, 0.30, 0.20, 0.15]),
            'dropout_p': _weighted_choice(rng, [0.10, 0.15, 0.20], [0.45, 0.25, 0.30]),
            'cost_loss_weight': _weighted_choice(
                rng,
                [0.01, 0.015, 0.02, 0.03, 0.05, 0.08],
                [0.10, 0.15, 0.25, 0.25, 0.20, 0.05],
            ),
            'constraint_loss_weight': _weighted_choice(rng, [0.1, 0.15, 0.2, 0.25], [0.35, 0.30, 0.25, 0.10]),
            'carrier_emb_dim': _weighted_choice(rng, [16, 32], [0.35, 0.65]),
            'option_proj_dim': _weighted_choice(rng, [8, 16], [0.80, 0.20]),
            'n_layers': _weighted_choice(rng, [3, 2], [0.80, 0.20]),
            'hidden_dim': _weighted_choice(rng, [160, 192, 128], [0.60, 0.20, 0.20]),
            'dc_embedding_dim': _weighted_choice(rng, [64, 32, 96], [0.70, 0.15, 0.15]),
            'label_smoothing': _weighted_choice(rng, [0.03, 0.05, 0.08, 0.10], [0.25, 0.40, 0.25, 0.10]),
            'carrier_loss_weight': _weighted_choice(rng, [0.001, 0.002, 0.005], [0.20, 0.60, 0.20]),
            'batch_size': _weighted_choice(rng, [64, 96], [0.75, 0.25]),
        }
        _append_cfg(cfg)

    max_attempts_explore = max(4000, target_count * 60)
    attempts = 0
    while len(configs) < target_count and attempts < max_attempts_explore:
        attempts += 1
        cfg = {
            'learning_rate': round(_log_uniform(rng, 1.5e-4, 6e-4), 7),
            'weight_decay': round(_log_uniform(rng, 1e-6, 3e-4), 7),
            'dropout_p': round(rng.uniform(0.08, 0.27), 2),
            'cost_loss_weight': round(_log_uniform(rng, 0.005, 0.08), 4),
            'constraint_loss_weight': _weighted_choice(rng, [0.1, 0.15, 0.2, 0.25, 0.3], [0.25, 0.25, 0.25, 0.15, 0.10]),
            'carrier_emb_dim': _weighted_choice(rng, [8, 16, 32], [0.15, 0.45, 0.40]),
            'option_proj_dim': _weighted_choice(rng, [8, 16, 24], [0.55, 0.30, 0.15]),
            'n_layers': _weighted_choice(rng, [2, 3, 4], [0.25, 0.60, 0.15]),
            'hidden_dim': _weighted_choice(rng, [128, 160, 192, 224], [0.20, 0.45, 0.25, 0.10]),
            'dc_embedding_dim': _weighted_choice(rng, [32, 64, 96], [0.20, 0.60, 0.20]),
            'label_smoothing': round(rng.uniform(0.0, 0.12), 3),
            'carrier_loss_weight': round(_log_uniform(rng, 5e-4, 1e-2), 6),
            'batch_size': _weighted_choice(rng, [32, 64, 96, 128], [0.10, 0.55, 0.25, 0.10]),
        }
        _append_cfg(cfg)

    if len(configs) < target_count:
        print(
            f"Warning: cost-biased full grid generated {len(configs)} unique configs "
            f"(target {target_count}); padding with repeated samples."
        )
        configs = _trim_or_pad(configs, target_count)
    return configs


def generate_hierarchical_proxy_incumbent_local_grid(n_jobs=60, configs_per_job=6):
    """
    Exploitative local search centered on hierarchical_proxy incumbent simulation winners.

    Compared to the older broader local-search grid, this version narrows the search
    to the high-performing region seen in simulation:
      - low-but-active cost loss (0.008 to 0.03)
      - moderate constraint weight (0.1 to 0.2, with light 0.15 support)
      - lr around 2e-4 to 3e-4, wd around 1e-5 to 1e-4
      - architecture mostly fixed (hierarchical_proxy, hd=160, nl=3, dc_emb=64, op=8)
      - targeted sweeps for carrier embedding, label smoothing, and carrier loss
    """
    target_count = n_jobs * configs_per_job
    configs: List[Dict[str, Any]] = []

    base_anchor = {
        'repair_strategy': 'inventory_weighted',
        'model_variant': 'hierarchical_proxy_v2',
        'hidden_dim': 160,
        'n_layers': 3,
        'dropout_p': 0.2,
        'dc_embedding_dim': 64,
        'sku_emb_dim': 8,
        'brand_emb_dim': 6,
        'option_proj_dim': 8,
        'carrier_emb_dim': 8,
        'label_smoothing': 0.05,
        'carrier_loss_weight': 0.002,
        'batch_size': 64,
        # Final-eval UB settings are intentionally NOT overridden here.
        # They should come from --base-config / --base-model so a single
        # config edit (e.g., final_ub_eval_orders=2000) propagates to all runs.
    }

    # Tier A (270): core local region around incumbent-like settings.
    for lr in [2e-4, 3e-4]:
        for wd in [1e-5, 3e-5, 1e-4]:
            for do in [0.1, 0.15, 0.2]:
                for cost_w in [0.008, 0.01, 0.015, 0.02, 0.03]:
                    for c_w in [0.1, 0.15, 0.2]:
                        cfg = dict(base_anchor)
                        cfg.update({
                            'learning_rate': lr,
                            'weight_decay': wd,
                            'dropout_p': do,
                            'cost_loss_weight': cost_w,
                            'constraint_loss_weight': c_w,
                        })
                        configs.append(cfg)

    # Tier B (72): targeted carrier embedding sweep in the strongest loss region.
    for carrier_emb in [8, 16, 32]:
        for lr in [2e-4, 3e-4]:
            for do in [0.1, 0.2]:
                for cost_w in [0.01, 0.015, 0.02]:
                    for c_w in [0.1, 0.2]:
                        cfg = dict(base_anchor)
                        cfg.update({
                            'learning_rate': lr,
                            'weight_decay': 3e-5,
                            'dropout_p': do,
                            'carrier_emb_dim': carrier_emb,
                            'cost_loss_weight': cost_w,
                            'constraint_loss_weight': c_w,
                        })
                        configs.append(cfg)

    # Tier C (54): calibration sweep for label smoothing and carrier loss.
    for ls in [0.03, 0.05, 0.08]:
        for carrier_w in [0.001, 0.002, 0.005]:
            for cost_w in [0.01, 0.015, 0.02]:
                for c_w in [0.1, 0.2]:
                    cfg = dict(base_anchor)
                    cfg.update({
                        'learning_rate': 3e-4,
                        'weight_decay': 1e-5,
                        'dropout_p': 0.15,
                        'label_smoothing': ls,
                        'carrier_loss_weight': carrier_w,
                        'cost_loss_weight': cost_w,
                        'constraint_loss_weight': c_w,
                    })
                    configs.append(cfg)

    # Tier D (32): light batch-size sensitivity around the same local basin.
    for bs in [64, 96]:
        for lr in [2e-4, 3e-4]:
            for do in [0.1, 0.2]:
                for cost_w in [0.01, 0.02]:
                    for c_w in [0.15, 0.2]:
                        cfg = dict(base_anchor)
                        cfg.update({
                            'batch_size': bs,
                            'learning_rate': lr,
                            'weight_decay': 3e-5,
                            'dropout_p': do,
                            'cost_loss_weight': cost_w,
                            'constraint_loss_weight': c_w,
                        })
                        configs.append(cfg)

    return _trim_or_pad(_dedupe_preserve_order(configs), target_count)


def _hierarchical_proxy_ablation_definitions(include_full: bool = True):
    ablations = [
        ("no_scenario_module", {
            'model_variant': 'hierarchical_proxy_v2',
            'use_cost_summary': True,
            'use_scenario_module': False,
            'use_dc_module': True,
        }),
        ("no_dc_module", {
            'model_variant': 'hierarchical_proxy_v2',
            'use_cost_summary': True,
            'use_scenario_module': True,
            'use_dc_module': False,
        }),
        ("single_tower", {
            'model_variant': 'single_tower',
            'use_num_proj': False,
        }),
    ]
    if include_full:
        ablations.insert(0, ("full", {
            'model_variant': 'hierarchical_proxy_v2',
            'use_cost_summary': True,
            'use_scenario_module': True,
            'use_dc_module': True,
        }))
    return ablations


def _hierarchical_proxy_loss_ablation_definitions(include_full: bool = True):
    """
    Loss-component ablations focused on components that are ON in hierarchical_proxy_base.

    Main config ON components:
      - selection_weight (DC + carrier CE terms via ProxyLoss selection block)
      - constraint_loss_weight
      - cost_loss_weight
    """
    ablations = [
        ("no_constraint_loss", {
            'constraint_loss_weight': 0.0,
        }),
        ("no_cost_loss", {
            'cost_loss_weight': 0.0,
        }),
        ("no_selection_loss", {
            # selection_weight multiplies (dc CE + carrier CE) in hierarchical mode.
            # carrier_loss_weight is set to 0.0 as an explicit companion toggle.
            'selection_weight': 0.0,
            'carrier_loss_weight': 0.0,
        }),
    ]
    if include_full:
        ablations.insert(0, ("full_loss", {}))
    return ablations


def _hierarchical_proxy_common_tune_candidates() -> List[Dict[str, Any]]:
    """Shared tuning candidates used across ablation grids for fair comparison."""
    tune_candidates: List[Dict[str, Any]] = []
    for lr in [3e-4, 5e-4]:
        for wd in [1e-5, 1e-4]:
            for hd in [128, 160]:
                for do in [0.1, 0.2]:
                    for cost_w in [0.01, 0.03, 0.05, 0.08]:
                        for c_w in [0.1, 0.2, 0.3, 0.4]:
                            tune_candidates.append({
                                'learning_rate': lr,
                                'weight_decay': wd,
                                'hidden_dim': hd,
                                'dropout_p': do,
                                'n_layers': 3,
                                'batch_size': 64,
                                'cost_loss_weight': cost_w,
                                'constraint_loss_weight': c_w,
                            })
    return tune_candidates


def _generate_hierarchical_proxy_ablation_grid(n_jobs=35, configs_per_job=4, include_full: bool = True):
    """
    Fair ablation grid centered on the hierarchical_proxy_v2 model.

    Includes:
      - single_tower architecture baseline
      - focused module-off ablations for hierarchical_proxy_v2:
        no_scenario_module / no_dc_module

    Fairness policy:
      - All ablations share the same hyperparameter candidates.
      - Config list is built in round-robin across ablations, so truncation
        at n_jobs * configs_per_job keeps budgets approximately balanced.
    """
    target_count = n_jobs * configs_per_job

    ablations = _hierarchical_proxy_ablation_definitions(include_full=include_full)
    if target_count % len(ablations) != 0:
        print(
            f"Warning: target_count={target_count} is not divisible by "
            f"{len(ablations)} ablation groups; resulting counts may be unbalanced."
        )

    tune_candidates = _hierarchical_proxy_common_tune_candidates()

    configs = []
    for tune_cfg in tune_candidates:
        for ablation_tag, ablation_override in ablations:
            cfg = dict(tune_cfg)
            cfg.update(ablation_override)
            cfg['ablation_tag'] = ablation_tag
            configs.append(cfg)

    return _trim_or_pad(configs, target_count)


def generate_hierarchical_proxy_ablation_fair_grid(n_jobs=35, configs_per_job=4):
    """Ablation grid including the full hierarchical_proxy model as baseline."""
    return _generate_hierarchical_proxy_ablation_grid(
        n_jobs=n_jobs,
        configs_per_job=configs_per_job,
        include_full=True,
    )


def generate_hierarchical_proxy_ablation_modules_only_grid(n_jobs=35, configs_per_job=4):
    """Ablation grid excluding the full model (modules-off + single_tower only)."""
    return _generate_hierarchical_proxy_ablation_grid(
        n_jobs=n_jobs,
        configs_per_job=configs_per_job,
        include_full=False,
    )


def _generate_hierarchical_proxy_loss_ablation_grid(n_jobs=35, configs_per_job=4, include_full: bool = True):
    """
    Fair loss-component ablation grid for hierarchical_proxy_v2.
    """
    target_count = n_jobs * configs_per_job
    ablations = _hierarchical_proxy_loss_ablation_definitions(include_full=include_full)
    if target_count % len(ablations) != 0:
        print(
            f"Warning: target_count={target_count} is not divisible by "
            f"{len(ablations)} loss-ablation groups; resulting counts may be unbalanced."
        )

    base_arch = {
        'model_variant': 'hierarchical_proxy_v2',
        'use_cost_summary': True,
        'use_option_features_in_carrier': True,
        'use_scenario_module': True,
        'use_dc_module': True,
        'use_dc_embedding': True,
        'use_carrier_embedding': True,
    }
    tune_candidates = _hierarchical_proxy_common_tune_candidates()

    configs = []
    for tune_cfg in tune_candidates:
        for ablation_tag, ablation_override in ablations:
            cfg = dict(base_arch)
            cfg.update(tune_cfg)
            cfg.update(ablation_override)
            cfg['ablation_tag'] = f"loss_{ablation_tag}"
            configs.append(cfg)
    return _trim_or_pad(configs, target_count)


def generate_hierarchical_proxy_loss_ablation_fair_grid(n_jobs=35, configs_per_job=4):
    """Loss ablation grid including full-loss baseline."""
    return _generate_hierarchical_proxy_loss_ablation_grid(
        n_jobs=n_jobs,
        configs_per_job=configs_per_job,
        include_full=True,
    )


def generate_hierarchical_proxy_loss_ablation_components_only_grid(n_jobs=35, configs_per_job=4):
    """Loss ablation grid excluding full-loss baseline."""
    return _generate_hierarchical_proxy_loss_ablation_grid(
        n_jobs=n_jobs,
        configs_per_job=configs_per_job,
        include_full=False,
    )


GRID_REGISTRY = {
    'hierarchical_proxy_focus': generate_hierarchical_proxy_focus_grid,
    'hierarchical_proxy_sim_cost_focus': generate_hierarchical_proxy_sim_cost_focus_grid,
    'hierarchical_proxy_paper_full': generate_hierarchical_proxy_paper_full_grid,
    'hierarchical_proxy_incumbent_local': generate_hierarchical_proxy_incumbent_local_grid,
    'hierarchical_proxy_ablation_fair': generate_hierarchical_proxy_ablation_fair_grid,
    'hierarchical_proxy_ablation_modules_only': generate_hierarchical_proxy_ablation_modules_only_grid,
    'hierarchical_proxy_loss_ablation_fair': generate_hierarchical_proxy_loss_ablation_fair_grid,
    'hierarchical_proxy_loss_ablation_components_only': generate_hierarchical_proxy_loss_ablation_components_only_grid,
}

DEFAULT_GRID_OUTPUTS = {
    'hierarchical_proxy_focus': 'proxy_focus.txt',
    'hierarchical_proxy_sim_cost_focus': 'proxy_sim_cost_focus.txt',
    'hierarchical_proxy_paper_full': 'proxy_full.txt',
    'hierarchical_proxy_incumbent_local': 'proxy_incumbent_local.txt',
    'hierarchical_proxy_ablation_fair': 'proxy_arch_ablation.txt',
    'hierarchical_proxy_ablation_modules_only': 'proxy_arch_ablation_modules_only.txt',
    'hierarchical_proxy_loss_ablation_fair': 'proxy_loss_ablation.txt',
    'hierarchical_proxy_loss_ablation_components_only': 'proxy_loss_ablation_components_only.txt',
}


# ============================================================================
# FORMATTING & OUTPUT
# ============================================================================

ALLOWED_KEYS = {
    'batch_size', 'learning_rate', 'weight_decay',
    'selection_weight', 'carrier_loss_weight', 'constraint_loss_weight',
    'cardinality_penalty_weight', 'entropy_weight', 'cost_loss_weight',
    'threshold_on_sel', 'repair_strategy',
    'class_weight_power', 'class_weight_max',
    'use_dc_class_weights', 'dc_class_weight_power', 'dc_class_weight_max',
    'use_carrier_class_weights', 'carrier_class_weight_power', 'carrier_class_weight_max',
    'gumbel_tau', 'label_smoothing',
    'hidden_dim', 'n_layers', 'dropout_p',
    'sku_emb_dim', 'brand_emb_dim', 'dc_embedding_dim', 'carrier_emb_dim', 'option_proj_dim',
    'use_option_features_in_carrier', 'use_cost_summary',
    'use_scenario_module', 'use_dc_module', 'use_dc_embedding', 'use_carrier_embedding',
    'ablation_tag',
    'epochs', 'agg_type', 'model_variant', 'use_num_proj',
    'use_eligibility_mask', 'aux_dc_weight', 'aux_carrier_weight',
    'split_ratio', 'normalize_global_features', 'normalize_dc_features', 'normalize_option_features',
    'refit_full_data',
    'early_stopping_patience',
    'ub_eval_orders',
    'final_eval_on_best', 'final_ub_eval_orders', 'final_ub_eval_scenarios', 'final_ub_eval_batch_size',
}

FLAG_ONLY_KEYS = {
    'use_dc_class_weights',
    'use_carrier_class_weights',
    'use_num_proj',
    'refit_full_data',
}

FLAG_NEGATION_KEYS = {
    'use_option_features_in_carrier': ('--use_option_features_in_carrier', '--no-use_option_features_in_carrier'),
    'use_cost_summary': ('--use_cost_summary', '--no-use_cost_summary'),
    'use_scenario_module': ('--use_scenario_module', '--no-use_scenario_module'),
    'use_dc_module': ('--use_dc_module', '--no-use_dc_module'),
    'use_dc_embedding': ('--use_dc_embedding', '--no-use_dc_embedding'),
    'use_carrier_embedding': ('--use_carrier_embedding', '--no-use_carrier_embedding'),
    'final_eval_on_best': ('--final_eval_on_best', '--no-final_eval_on_best'),
    'use_eligibility_mask': ('--use_eligibility_mask', '--no-use_eligibility_mask'),
    'normalize_global_features': ('--normalize_global_features', '--no-normalize_global_features'),
    'normalize_dc_features': ('--normalize_dc_features', '--no-normalize_dc_features'),
    'normalize_option_features': ('--normalize_option_features', '--no-normalize_option_features'),
}


def _sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k in ALLOWED_KEYS}


def _merge_base_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _trim_or_pad(configs: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    if target_count <= 0:
        return []
    if len(configs) >= target_count:
        return configs[:target_count]
    if not configs:
        return configs
    i = 0
    while len(configs) < target_count:
        configs.append(dict(configs[i % len(configs)]))
        i += 1
    return configs


def _ensure_base_config_first(configs: List[Dict[str, Any]], base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ensure the exact base config is present and pinned at index 0.

    Any duplicate of the exact base config already in the grid is removed so the
    base appears once, then re-inserted at the head.
    """
    if not configs:
        return configs
    base_sanitized = _sanitize_config(base_cfg)
    if not base_sanitized:
        return configs

    base_key = tuple(sorted(base_sanitized.items()))
    out: List[Dict[str, Any]] = [dict(base_sanitized)]
    for cfg in configs:
        cfg_key = tuple(sorted(cfg.items()))
        if cfg_key == base_key:
            continue
        out.append(cfg)
    return out


def _load_base_from_model_checkpoint(model_path: Path) -> Dict[str, Any]:
    """
    Recover a trainable base config from a trained proxy checkpoint.

    Priority:
      1) checkpoint['hyperparams'] sections (training/loss/inference/architecture)
      2) checkpoint['model_params'] fallback for architecture fields
    """
    import torch

    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    out: Dict[str, Any] = {}
    hp = blob.get("hyperparams", {}) if isinstance(blob, dict) else {}
    mp = blob.get("model_params", {}) if isinstance(blob, dict) else {}

    for section in ("training", "loss", "inference", "architecture"):
        section_payload = hp.get(section, {})
        if not isinstance(section_payload, dict):
            continue
        for k, v in section_payload.items():
            if k in ALLOWED_KEYS:
                out[k] = v

    fallback_key_map = {
        "architecture": "model_variant",
        "carrier_embedding_dim": "carrier_emb_dim",
    }
    if isinstance(mp, dict):
        for k, v in mp.items():
            kk = fallback_key_map.get(k, k)
            if kk in ALLOWED_KEYS and kk not in out:
                out[kk] = v

    # Ensure we keep the current inference strategy from checkpoint when present.
    if "repair_strategy" not in out:
        rs = None
        if isinstance(mp, dict):
            rs = mp.get("repair_strategy")
        if rs is None and isinstance(hp, dict):
            rs = hp.get("inference", {}).get("repair_strategy")
        if rs is not None:
            out["repair_strategy"] = rs

    return _sanitize_config(out)


def format_model_name(config: Dict[str, Any], job_id: int, config_idx: int) -> str:
    """Generate descriptive model name from config."""
    parts = []
    lr = config.get('learning_rate')
    hd = config.get('hidden_dim')
    do = config.get('dropout_p')
    nl = config.get('n_layers')
    parts.append(f"lr{lr:.0e}" if lr is not None else "lrNA")
    parts.append(f"hd{hd}" if hd is not None else "hdNA")
    parts.append(f"do{do:.2f}" if do is not None else "doNA")
    parts.append(f"nl{nl}" if nl is not None else "nlNA")
    
    if 'weight_decay' in config:
        parts.append(f"wd{config['weight_decay']:.0e}")

    if 'model_variant' in config:
        model_variant = str(config['model_variant'])
        mv_short = {
            'hierarchical_proxy_v2': 'hpv2',
            'single_tower': 'st',
        }.get(model_variant, model_variant)
        parts.append(f"mv{mv_short}")
    if 'ablation_tag' in config:
        parts.append(f"abl{config['ablation_tag']}")
    
    # Loss weights
    if 'constraint_loss_weight' in config:
        parts.append(f"cw{config['constraint_loss_weight']:.1f}")
    if 'cost_loss_weight' in config:
        parts.append(f"cost{config['cost_loss_weight']:.2f}")
    if 'gumbel_tau' in config and config['gumbel_tau'] != 1.0:
        parts.append(f"tau{config['gumbel_tau']:.1f}")
        
    if 'entropy_weight' in config:
        parts.append(f"ew{config['entropy_weight']:.2f}")
    if 'sku_emb_dim' in config:
        parts.append(f"sku{config['sku_emb_dim']}")
    if 'brand_emb_dim' in config:
        parts.append(f"br{config['brand_emb_dim']}")
    if 'carrier_emb_dim' in config:
        parts.append(f"cemb{config['carrier_emb_dim']}")
    if 'option_proj_dim' in config:
        parts.append(f"op{config['option_proj_dim']}")
    dc_emb = config.get('dc_embedding_dim')
    if dc_emb is not None:
        parts.append(f"dcemb{dc_emb}")
    
    th = config.get('threshold_on_sel')
    parts.append(f"th{th:.2f}" if th is not None else "thNA")
    parts.append(f"job{job_id}")
    parts.append(f"cfg{config_idx}")
    
    return "proxy_tune_" + "_".join(parts)


def format_config_line(config: Dict[str, Any], job_id: int, config_idx: int) -> str:
    """Format config as CLI argument string."""
    model_name = format_model_name(config, job_id, config_idx)
    
    parts = []
    for key, value in config.items():
        if key in {'model_name', 'ablation_tag'}:
            continue
        if key == 'threshold_on_sel':
            parts.append(f"--threshold_on_sel {value}")
        elif key in FLAG_NEGATION_KEYS:
            flag_true, flag_false = FLAG_NEGATION_KEYS[key]
            parts.append(flag_true if value else flag_false)
        elif key in FLAG_ONLY_KEYS:
            if value:
                parts.append(f"--{key}")
        elif key == 'dc_embedding_dim':
            parts.append(f"--dc_embedding_dim {value}")
        else:
            parts.append(f"--{key} {value}")
    
    parts.append(f"--model_name {model_name}")
    return " ".join(parts)


def write_grid_file(configs: List[Dict], output_path: Path, configs_per_job: int):
    """Write configs to parameter file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_jobs = len(configs) // configs_per_job
    lines = []
    
    for job_id in range(n_jobs):
        start_idx = job_id * configs_per_job
        end_idx = start_idx + configs_per_job
        job_configs = configs[start_idx:end_idx]
        
        for cfg_idx, config in enumerate(job_configs):
            line = format_config_line(config, job_id, cfg_idx)
            lines.append(line)
    
    with output_path.open('w') as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"Wrote {len(lines)} parameter lines to {output_path}")
    print(f"   ({n_jobs} jobs x {configs_per_job} configs/job)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global GRID_SAMPLE_SEED
    parser = argparse.ArgumentParser(
        description="Generate proxy tuning grids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available grids:
  hierarchical_proxy_focus - Focused grid for hierarchical_proxy_v2 (use with --base-config)
  hierarchical_proxy_sim_cost_focus - Simulation-focused hierarchical_proxy grid (cost-loss regime)
  hierarchical_proxy_paper_full - Paper-facing cost-biased hybrid search (anchors + focused + tail)
  hierarchical_proxy_incumbent_local - Local search around incumbent hierarchical_proxy costloss model
  hierarchical_proxy_ablation_fair - Fair hierarchical_proxy ablations + single_tower with shared tuning budget
  hierarchical_proxy_ablation_modules_only - Same as ablation_fair, but excludes full hierarchical_proxy model
  hierarchical_proxy_loss_ablation_fair - Fair loss-component ablations (includes full-loss baseline)
  hierarchical_proxy_loss_ablation_components_only - Loss ablations only (excludes full-loss baseline)

Examples:
  # Dry-run to see config count
  python -m scripts.proxy.generate_and_tune_proxy --grid hierarchical_proxy_paper_full --dry-run
  
  # Generate parameter file
  python -m scripts.proxy.generate_and_tune_proxy --grid hierarchical_proxy_paper_full --output configs/proxy/paper/grids/proxy_full.txt

  # Focused grid (base config + overrides)
  python -m scripts.proxy.generate_and_tune_proxy --grid hierarchical_proxy_focus \
    --base-config configs/proxy/paper/repair_strategies/hierarchical_proxy_inventory_weighted.json \
    --output configs/proxy/paper/grids/proxy_focus.txt

  # Incumbent-local search (larger budget)
  python -m scripts.proxy.generate_and_tune_proxy --grid hierarchical_proxy_incumbent_local \
    --base-config configs/proxy/hierarchical_proxy_main.json \
    --n_jobs 60 --configs_per_job 6 \
    --output configs/proxy/paper/grids/proxy_incumbent_local.txt

  # Paper-facing full grid (280 configs by default: 35 x 8)
  python -m scripts.proxy.generate_and_tune_proxy --grid hierarchical_proxy_paper_full \
    --base-config configs/proxy/hierarchical_proxy_main.json \
    --n_jobs 35 --configs_per_job 8 \
    --output configs/proxy/paper/grids/proxy_full.txt

  # Then run each line with src.training.proxy.train_proxy, or adapt scripts/reproduce/run_proxy_ablation_fixed.sh.
"""
    )
    
    parser.add_argument('--grid', type=str, required=True, 
                       choices=list(GRID_REGISTRY.keys()),
                       help='Grid configuration to use')
    parser.add_argument('--n_jobs', type=int, default=45,
                       help='Number of config-batches used to size the grid (default: 45)')
    parser.add_argument('--configs_per_job', type=int, default=3,
                       help='Configs per job (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible grid shuffling (default: 42)')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=True,
                       help='Shuffle configs before writing (default: enabled)')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                       help='Keep original generation order (deterministic)')
    
    # Output options
    parser.add_argument('--output', type=Path, default=None,
                       help='Output parameter file path (default: curated name under configs/proxy/paper/grids/)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print config count and exit without writing')
    parser.add_argument('--base-config', type=Path, default=None,
                       help='Optional JSON config to use as base for all grid configs')
    parser.add_argument('--base-model', type=Path, default=None,
                       help='Optional trained proxy checkpoint (best.pt) to recover base hyperparameters')
    
    args = parser.parse_args()
    GRID_SAMPLE_SEED = int(args.seed)
    
    # Generate grid
    print(f"Generating grid: {args.grid}")
    grid_fn = GRID_REGISTRY[args.grid]
    configs = grid_fn(args.n_jobs, args.configs_per_job)

    if args.base_config is not None and args.base_model is not None:
        raise ValueError("Use only one of --base-config or --base-model.")

    base_cfg = None
    if args.base_config is not None:
        with args.base_config.open('r') as f:
            base_cfg = json.load(f)
        base_cfg.pop('model_name', None)
        base_cfg = _sanitize_config(base_cfg)
        configs = [_merge_base_config(base_cfg, cfg) for cfg in configs]
    elif args.base_model is not None:
        base_cfg = _load_base_from_model_checkpoint(args.base_model)
        print(f"Loaded base hyperparameters from checkpoint: {args.base_model}")
        configs = [_merge_base_config(base_cfg, cfg) for cfg in configs]

    configs = [_sanitize_config(cfg) for cfg in configs]

    # Keep a true baseline run in the same tune job family.
    # For ablation grids we skip this to preserve balanced ablation budgets.
    has_ablation_grid = any("ablation_tag" in cfg for cfg in configs)
    pin_base_first = base_cfg is not None and not has_ablation_grid
    if pin_base_first:
        configs = _ensure_base_config_first(configs, base_cfg)

    target_count = args.n_jobs * args.configs_per_job
    if len(configs) != target_count:
        print(f"Warning: Generated {len(configs)} configs (target {target_count})")
        configs = _trim_or_pad(configs, target_count)

    if args.shuffle:
        rng = random.Random(args.seed)
        if pin_base_first and len(configs) > 1:
            head, tail = configs[0], configs[1:]
            rng.shuffle(tail)
            configs = [head] + tail
            print(f"Applied deterministic shuffle with seed={args.seed} (kept base config first)")
        else:
            rng.shuffle(configs)
            print(f"Applied deterministic shuffle with seed={args.seed}")
    else:
        print("Shuffle disabled; using original config order")
    
    print(f"Generated {len(configs)} configs")
    print(f"   ({args.n_jobs} jobs x {args.configs_per_job} configs/job)")
    if any('ablation_tag' in cfg for cfg in configs):
        counts: Dict[str, int] = {}
        for cfg_item in configs:
            tag = cfg_item.get('ablation_tag')
            if not tag:
                continue
            counts[tag] = counts.get(tag, 0) + 1
        if counts:
            print("Ablation counts:")
            for tag in sorted(counts):
                print(f"   {tag}: {counts[tag]}")
    
    if args.dry_run:
        print("\nSample configs:")
        for i in range(min(3, len(configs))):
            sample = format_config_line(configs[i], 0, i)
            print(f"   Config {i+1}: {sample[:100]}...")
        return
    
    # Determine output path
    if args.output is None:
        args.output = Path("configs/proxy/paper/grids") / DEFAULT_GRID_OUTPUTS[args.grid]
    
    # Write parameter file
    write_grid_file(configs, args.output, args.configs_per_job)
    print(f"\nRun grid lines with: python -m src.training.proxy.train_proxy <args from one grid line>")
    
    return 0


if __name__ == '__main__':
    exit(main())
