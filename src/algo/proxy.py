"""
Proxy fulfillment algorithm with on-the-fly feature engineering.
"""

import time
import json
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

from src.model.proxy_inference import proxy_inference
from src.model.hierarchical_proxy_inference import hierarchical_proxy_inference
from src.scenario_generator import generate_scenarios
from src.model.proxy_variants import build_proxy_model
from src.utils import calculate_lookahead_periods
import src.config as cfg
from src.saa_procedure import _sample_penalty_matrix, evaluate_candidate_solution
from src.data_utils import (
    preprocess_proxy_features,
    load_dc_carrier_metadata,
    compute_base_costs_for_order,
    compute_global_features_for_order,
)


def _normalize_repair_strategy(strategy: Optional[str]) -> Optional[str]:
    if strategy is None:
        return None
    s = str(strategy).strip().lower()
    aliases = {
        "default": "argmax_then_split",
        "argmax_split": "argmax_then_split",
        "argmax_then_split": "argmax_then_split",
    }
    return aliases.get(s, s)


def prepare_proxy_data(
    model: torch.nn.Module,
    inference_params: Dict[str, Any],
    device: torch.device,
    model_info: Dict[str, Any],
    feature_scalers: Optional[dict[str, Any]] = None,
    scaler: Optional[Any] = None,
    order_set: Optional[str] = None,
    simulation_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Prepare proxy inference data with optional static feature preloading."""
    pre = pd.read_csv(cfg.PREPROCESSED_PATH)
    pre['order_time'] = pd.to_datetime(pre['order_time'])
    pre = preprocess_proxy_features(pre)

    # Categorical factorisation
    for col in cfg.PROXY_CATEGORICAL_ORDER_FEATURES:
        pre[col], _ = pd.factorize(pre[col], sort=True)
    pre["sku_idx"], _ = pd.factorize(pre[cfg.SKU_COL], sort=True)
    pre["brand_idx"], _ = pd.factorize(pre[cfg.BRAND_COL], sort=True)

    dcs = model_info.get('dcs', [])
    carriers = model_info.get('carriers', [])
    # Handle missing or empty carriers list from old checkpoints
    if not carriers:
        carriers = [c for c in range(1, 19) if c not in cfg.EXCLUDED_CARRIER_SERVICES]
        print(f"[proxy] Warning: checkpoint had empty/missing carriers list, using default: {len(carriers)} carriers")
    if not dcs:
        print(f"[proxy] ERROR: checkpoint has no 'dcs' list! Cannot proceed.")
        raise ValueError("Model checkpoint missing 'dcs' metadata")
    supported_variants = {'hierarchical_proxy_v2', 'single_tower'}
    use_proxy_features = (
        'global_feature_dim' in model_info
        or model_info.get('architecture') in supported_variants
        or model_info.get('model_variant') in supported_variants
    )
    if not use_proxy_features:
        print("[proxy] Warning: model metadata missing proxy feature flag; assuming proxy feature inputs.")
        use_proxy_features = True
    
    # Try to load precomputed static features
    static_features = None
    if order_set and simulation_date:
        static_path = cfg.PROXY_DATA_DIR / "static_features" / order_set / f"static_{simulation_date}.pt"
        if static_path.exists():
            try:
                static_data = torch.load(static_path, map_location='cpu')
                static_features = static_data['features']
                print(f"[proxy] Loaded static features for {len(static_features)} (order, sku) pairs")
            except Exception as e:
                print(f"[proxy] Failed to load static features: {e}")
                static_features = None
    
    # Load DC metadata and network for dynamic feature engineering
    dc_metadata, cost_coef_map = load_dc_carrier_metadata()
    dc_meta_dict = {
        int(row['dc_id']): {'lat': float(row['lat']), 'lon': float(row['lon'])}
        for _, row in dc_metadata.iterrows()
    }
    network_df = pd.read_csv(cfg.NETWORK_PATH)
    dc_region_map = network_df.set_index('dc_ID')['region_ID'].to_dict()
    
    # Compute SKU daily demand (only if static features not available)
    if static_features is None:
        historical_data = pre[pre['order_time'] < pd.to_datetime(simulation_date)].copy() if simulation_date else pre
        if not historical_data.empty:
            sku_daily_demand = historical_data.groupby(cfg.SKU_COL)['quantity'].sum()
            date_range_days = (historical_data['order_time'].max() - historical_data['order_time'].min()).days
            if date_range_days > 0:
                sku_daily_demand = sku_daily_demand / date_range_days
            sku_daily_demand_dict = sku_daily_demand.to_dict()
            
            historical_data['order_hour'] = historical_data['order_time'].dt.hour
            hourly_volume = historical_data.groupby('order_hour').size().to_dict()
        else:
            sku_daily_demand_dict = {}
            hourly_volume = {}
    else:
        sku_daily_demand_dict = {}
        hourly_volume = {}
    
    if feature_scalers is None and scaler is not None:
        feature_scalers = {'global': scaler}

    option_space = len(dcs) * len(carriers)
    if option_space == 55 * 17:
        print(f"[proxy] option space: {len(dcs)} dcs x {len(carriers)} carriers = {option_space}")
    else:
        print(
            f"[proxy] option space: {len(dcs)} dcs x {len(carriers)} carriers = {option_space} "
            "(expected 55x17=935)"
        )

    return {
        'model': model,
        'inference_params': inference_params,
        'device': device,
        'features_df': pre,
        'feature_scalers': feature_scalers or {},
        'static_features': static_features,  # Precomputed static features (if available)
        'dc_metadata': dc_metadata,
        'dc_meta_dict': dc_meta_dict,
        'dc_region_map': dc_region_map,
        'cost_coef_map': cost_coef_map,
        'sku_daily_demand': sku_daily_demand_dict,
        'hourly_volume': hourly_volume,
        'info': {
            'dcs': dcs,
            'num_dcs': len(dcs),
            'carriers': carriers,
            'num_carriers': len(carriers),
            'sku_dim': pre["sku_idx"].nunique(),
            'brand_dim': pre["brand_idx"].nunique(),
            'scenario_len': model_info.get('scenario_len', cfg.SAA_N1 * cfg.SAA_Q + cfg.SAA_N2),
            'use_proxy_features': use_proxy_features,
            'global_feature_dim': model_info.get('global_feature_dim'),
            'dc_feature_dim': model_info.get('dc_feature_dim'),
        },
    }


def _apply_scaler(arr: np.ndarray, scaler: Optional[Any]) -> np.ndarray:
    if scaler is None:
        return arr
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        mean = scaler.mean_
        scale = scaler.scale_
        indices = None
    elif isinstance(scaler, dict) and 'mean' in scaler and 'scale' in scaler:
        mean = scaler['mean']
        scale = scaler['scale']
        indices = scaler.get('indices')
    else:
        return arr
    arr = arr.astype(np.float32, copy=True)
    if indices is None:
        if arr.shape[1] == len(mean):
            arr = (arr - mean) / (scale + 1e-8)
        return arr
    indices = list(indices)
    if len(indices) != len(mean):
        return arr
    arr[:, indices] = (arr[:, indices] - mean) / (scale + 1e-8)
    return arr


def _compute_proxy_upper_bound_stats(
    *,
    order_skus: list[str],
    plan: torch.Tensor,
    carrier_selection: torch.Tensor,
    dcs: list[int],
    carriers: list[int],
    order_options: pd.DataFrame,
    on_hand_inventory_pivot: pd.DataFrame,
    order_items: pd.DataFrame,
    demand_scenarios: Dict[str, pd.Series],
    shipping_costs: Dict[str, pd.DataFrame],
    order_set: str,
    promise_days: float,
    scenario_seed: int,
) -> Dict[str, Any]:
    """Estimate per-order proxy upper bound and CI from generated scenarios."""
    stats: Dict[str, Any] = {
        'policy_ub_mean': float('nan'),
        'policy_ub_std': float('nan'),
        'policy_ub_ci95': float('nan'),
        'policy_current_stage_mean': float('nan'),
        'policy_current_stage_std': float('nan'),
        'policy_current_stage_ci95': float('nan'),
        'policy_future_recourse_mean': float('nan'),
        'policy_future_recourse_std': float('nan'),
        'policy_future_recourse_ci95': float('nan'),
        'policy_ub_n_eval': 0,
        'policy_ub_scenarios_total': 0,
        'policy_ub_eval_start': 0,
        'policy_bound_source': 'proxy_mc',
    }

    if order_options is None or order_options.empty or not order_skus or not demand_scenarios:
        return stats

    option_df = order_options.copy()
    option_df['option_id'] = pd.to_numeric(option_df['option_id'], errors='coerce').astype('Int64')
    option_df['dc_id'] = pd.to_numeric(option_df['dc_id'], errors='coerce').astype('Int64')
    option_df['carrier_service_id'] = pd.to_numeric(option_df['carrier_service_id'], errors='coerce').astype('Int64')
    option_df['base_cost'] = pd.to_numeric(option_df.get('base_cost', 0.0), errors='coerce').fillna(0.0)
    option_df = option_df.dropna(subset=['option_id', 'dc_id', 'carrier_service_id'])
    if option_df.empty:
        return stats

    option_ids = [int(v) for v in option_df['option_id'].tolist()]
    pair_to_opt = {
        (int(row.dc_id), int(row.carrier_service_id)): int(row.option_id)
        for row in option_df.itertuples(index=False)
    }
    base_costs_by_option = pd.Series(
        option_df['base_cost'].to_numpy(dtype=float),
        index=option_ids,
        dtype=float,
    )
    option_to_dc = {int(row.option_id): int(row.dc_id) for row in option_df.itertuples(index=False)}
    option_to_carrier = {int(row.option_id): int(row.carrier_service_id) for row in option_df.itertuples(index=False)}
    dc_to_option_ids: Dict[int, list[int]] = {}
    for opt_id, dc_id in option_to_dc.items():
        dc_to_option_ids.setdefault(int(dc_id), []).append(int(opt_id))

    first_sku = next(iter(demand_scenarios.keys()))
    total_scenarios = int(len(demand_scenarios[first_sku]))
    stats['policy_ub_scenarios_total'] = total_scenarios
    if total_scenarios <= 0:
        return stats

    candidate_scenarios = int(cfg.SAA_Q * cfg.SAA_N1)
    eval_start = candidate_scenarios if total_scenarios > candidate_scenarios else 0
    n_eval = int(total_scenarios - eval_start)
    if n_eval <= 0:
        return stats
    stats['policy_ub_eval_start'] = int(eval_start)
    stats['policy_ub_n_eval'] = int(n_eval)

    # Build proxy plan in the evaluate_candidate_solution format z[(sku, option_id)] = qty.
    z_candidate: Dict[tuple[str, int], float] = {}
    for b, sku in enumerate(order_skus):
        alloc_row = plan[b]
        carrier_row = carrier_selection[b]
        for d_idx in (alloc_row > 0).nonzero(as_tuple=True)[0].tolist():
            qty = int(alloc_row[d_idx].item())
            if qty <= 0:
                continue
            c_idx = int(carrier_row[d_idx].item())
            if c_idx < 0 or c_idx >= len(carriers):
                continue
            dc_id = int(dcs[d_idx])
            carrier_id = int(carriers[c_idx])
            opt_id = pair_to_opt.get((dc_id, carrier_id))
            if opt_id is None:
                continue
            key = (str(sku), int(opt_id))
            z_candidate[key] = z_candidate.get(key, 0.0) + float(qty)

    scenario_index = [f"scenario_{i}" for i in range(n_eval)]
    demand_scenarios_eval: Dict[str, pd.Series] = {}
    shipping_costs_eval: Dict[str, pd.DataFrame] = {}
    for sku in order_skus:
        demand_series = demand_scenarios.get(sku)
        if demand_series is None:
            demand_arr = np.zeros(total_scenarios, dtype=float)
        else:
            demand_arr = np.asarray(demand_series.values, dtype=float)
        demand_arr = demand_arr[:total_scenarios]
        demand_scenarios_eval[sku] = pd.Series(demand_arr[eval_start:], index=scenario_index)

        cost_df = shipping_costs.get(sku)
        if cost_df is None or cost_df.empty:
            eval_df = pd.DataFrame(
                np.repeat(base_costs_by_option.to_numpy(dtype=float)[None, :], n_eval, axis=0),
                index=scenario_index,
                columns=option_ids,
            )
        else:
            eval_df = cost_df.iloc[eval_start:].copy()
            eval_df.index = scenario_index
            parsed_cols = []
            for col in eval_df.columns:
                try:
                    parsed_cols.append(int(col))
                except (TypeError, ValueError):
                    parsed_cols.append(col)
            eval_df.columns = parsed_cols
            for opt_id in option_ids:
                if opt_id not in eval_df.columns:
                    eval_df[opt_id] = float(base_costs_by_option.loc[opt_id])
            eval_df = eval_df[option_ids].apply(pd.to_numeric, errors='coerce')
            for opt_id in option_ids:
                eval_df[opt_id] = eval_df[opt_id].fillna(float(base_costs_by_option.loc[opt_id]))
        shipping_costs_eval[sku] = eval_df

    inv_df = on_hand_inventory_pivot.reindex(index=order_skus).copy()
    for dc_id in dc_to_option_ids.keys():
        if dc_id not in inv_df.columns and str(dc_id) not in inv_df.columns:
            inv_df[dc_id] = 0.0
    inv_df = inv_df.fillna(0.0)
    order_items_eval = order_items[['sku_ID', 'quantity']].copy()
    order_items_eval['sku_ID'] = order_items_eval['sku_ID'].astype(str)
    order_items_eval['quantity'] = pd.to_numeric(order_items_eval['quantity'], errors='coerce').fillna(0.0)

    future_penalties_eval = None
    if bool(getattr(cfg, "SAA_USE_FUTURE_PENALTIES", False)):
        pen_rng = np.random.default_rng(int(scenario_seed) + 1)
        penalties_all = _sample_penalty_matrix(
            option_ids=option_ids,
            option_to_carrier=option_to_carrier,
            order_set=order_set,
            promise_days=float(promise_days),
            num_scenarios=total_scenarios,
            rng=pen_rng,
        )
        future_penalties_eval = penalties_all[eval_start:]

    eval_stats = evaluate_candidate_solution(
        z_candidate=z_candidate,
        on_hand_inventory=inv_df,
        demand_scenarios_N2=demand_scenarios_eval,
        shipping_costs_N2=shipping_costs_eval,
        stockout_penalty=cfg.STOCKOUT_PENALTY_PER_UNIT,
        base_costs_by_option=base_costs_by_option,
        dc_to_option_ids=dc_to_option_ids,
        option_to_dc=option_to_dc,
        option_to_carrier=option_to_carrier,
        order_set=order_set,
        promise_days=float(promise_days),
        future_penalties=future_penalties_eval,
        order_items=order_items_eval,
        return_components=True,
    )
    obj_mean = float(eval_stats["objective_mean"])
    obj_std = float(eval_stats["objective_std"])
    current_stage_mean = float(eval_stats["current_stage_mean"])
    current_stage_std = float(eval_stats["current_stage_std"])
    future_recourse_mean = float(eval_stats["future_recourse_mean"])
    future_recourse_std = float(eval_stats["future_recourse_std"])
    ci95 = 1.96 * float(obj_std) / np.sqrt(float(n_eval))
    current_stage_ci95 = 1.96 * float(current_stage_std) / np.sqrt(float(n_eval))
    future_recourse_ci95 = 1.96 * float(future_recourse_std) / np.sqrt(float(n_eval))
    stats['policy_ub_mean'] = float(obj_mean)
    stats['policy_ub_std'] = float(obj_std)
    stats['policy_ub_ci95'] = float(ci95)
    stats['policy_current_stage_mean'] = float(current_stage_mean)
    stats['policy_current_stage_std'] = float(current_stage_std)
    stats['policy_current_stage_ci95'] = float(current_stage_ci95)
    stats['policy_future_recourse_mean'] = float(future_recourse_mean)
    stats['policy_future_recourse_std'] = float(future_recourse_std)
    stats['policy_future_recourse_ci95'] = float(future_recourse_ci95)
    return stats


def proxy_fulfillment(
    order_info: pd.Series,
    order_items: pd.DataFrame,
    on_hand_inventory_pivot: pd.DataFrame,
    proxy_data: Dict[str, Any],
    order_set: str,
    simulation_date: str,
    order_idx: int,
    order_options: Optional[pd.DataFrame] = None,
    verbose: bool = False,
    dynamic_features: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, dict]:
    """Generate fulfillment plan using a proxy model."""
    start = time.perf_counter()
    
    if order_options is None or order_options.empty:
        raise ValueError("proxy_fulfillment requires order_options")
    
    order_id = str(order_info['order_ID'])
    features_df = proxy_data['features_df']
    order_pre = features_df[features_df['order_ID'] == order_id]
    
    if order_pre.empty:
        return pd.DataFrame(), {'runtime_seconds': time.perf_counter() - start, 'no_data': True}
    
    # Setup
    device = proxy_data['device']
    feature_scalers = proxy_data.get('feature_scalers', {})
    supported_variants = {'hierarchical_proxy_v2', 'single_tower'}
    use_proxy_features = proxy_data['info'].get('use_proxy_features', False)
    if not use_proxy_features:
        arch = proxy_data['info'].get('architecture') or proxy_data['info'].get('model_variant')
        if arch in supported_variants:
            use_proxy_features = True
        else:
            print("[proxy] Warning: missing proxy feature metadata in proxy_data; proceeding anyway.")
            use_proxy_features = True
    dcs = proxy_data['info']['dcs']
    carriers = proxy_data['info']['carriers']
    
    # Validate that dcs and carriers are properly populated
    if not dcs or not carriers:
        print(f"[proxy] ERROR in proxy_fulfillment: dcs={len(dcs) if dcs else 0}, carriers={len(carriers) if carriers else 0}")
        raise ValueError(f"Invalid proxy_data: dcs={dcs}, carriers={carriers}")
    order_skus = [str(s) for s in order_items['sku_ID'].unique()]
    B = len(order_skus)
    D, C = len(dcs), len(carriers)
    expected_dc_dim = proxy_data['info'].get('dc_feature_dim')
    dyn_feat_dim = len(cfg.DYNAMIC_FEATURES)
    base_dc_dim = 5
    use_dynamic_dc = expected_dc_dim == (base_dc_dim + dyn_feat_dim)
    if use_dynamic_dc and dynamic_features is not None and not dynamic_features.empty and 'dc_id' in dynamic_features.columns:
        dyn_df = dynamic_features.copy()
        dyn_df['dc_id'] = pd.to_numeric(dyn_df['dc_id'], errors='coerce').astype('Int64')
        dyn_df = dyn_df.dropna(subset=['dc_id']).drop_duplicates('dc_id')
        dyn_df = dyn_df.set_index('dc_id')
        dyn_cols = []
        for feat in cfg.DYNAMIC_FEATURES:
            if feat in dyn_df.columns:
                vals = dyn_df[feat].reindex(dcs).fillna(0.0).astype(float).to_numpy()
            else:
                vals = np.zeros(D, dtype=np.float32)
            dyn_cols.append(vals)
        dynamic_dc_feats = np.stack(dyn_cols, axis=1).astype(np.float32)
    elif use_dynamic_dc:
        dynamic_dc_feats = np.zeros((D, dyn_feat_dim), dtype=np.float32)
    else:
        dynamic_dc_feats = None
    
    # Reindex data
    order_pre = order_pre.set_index('sku_ID').reindex(order_skus).reset_index()
    qty_vec = order_items.set_index('sku_ID')['quantity'].loc[order_skus].astype(float).values
    inv_mat = on_hand_inventory_pivot.reindex(index=order_skus, columns=dcs).fillna(0).to_numpy()
    
    # Generate scenarios
    E = proxy_data['info']['scenario_len']
    scenario_seed = int(cfg.RANDOM_SEED + order_idx)
    demand_scenarios, shipping_costs = generate_scenarios(
        order_items=order_items,
        options_df=order_options,
        customer_dc=order_info['dc_des'],
        promise_days=order_info['promise_delivery_days'],
        num_scenarios=E,
        seed=scenario_seed,
        period='proxy' if order_set == 'proxy_train' else 'test',
        lookahead_periods=calculate_lookahead_periods(order_info['order_time'], simulation_date),
        verbose=verbose,
        dynamic_features=dynamic_features,
    )
    
    # Stack demand scenarios
    scen_dem = np.stack([
        demand_scenarios.get(sku, pd.Series(np.zeros(E, dtype=np.float32))).values[:E]
        for sku in order_skus
    ], axis=0)  # [B, S]
    
    if use_proxy_features:
        # Check if we have precomputed static features
        static_features = proxy_data.get('static_features')
        use_static = static_features is not None
        
        if use_static:
            # ===== Fast Path: Use Precomputed Static Features =====
            global_feats_list, base_cost_list, distance_list, region_match_list, sku_daily_demand_list = [], [], [], [], []
            
            for sku in order_skus:
                static_key = (order_id, sku)
                if static_key in static_features:
                    sf = static_features[static_key]
                    if proxy_data['info'].get('global_feature_dim') is not None:
                        expected_dim = proxy_data['info']['global_feature_dim']
                        if len(sf['global_features']) != expected_dim:
                            use_static = False
                            break
                    global_feats_list.append(sf['global_features'])
                    base_cost_list.append(sf['base_cost_grid'])
                    distance_list.append(sf['distance_vec'])
                    region_match_list.append(sf['region_match_vec'])
                    sku_daily_demand_list.append(sf['sku_daily_demand'])
                else:
                        # Fallback to on-the-fly for this SKU
                        use_static = False
                        break
                
                if use_static:
                    # All SKUs found in static features
                    base_cost_grid = base_cost_list[0]  # Same for all SKUs in order
                    distance_vec = distance_list[0]
                    region_match_vec = region_match_list[0]
                else:
                    # Fallback: at least one SKU missing, compute on-the-fly for all
                    pass
        
        if not use_static:
            # ===== Slow Path: Compute Static Features On-the-Fly =====
            customer_lat = order_pre['customer_lat'].iloc[0] if 'customer_lat' in order_pre.columns else 0.0
            customer_lon = order_pre['customer_lon'].iloc[0] if 'customer_lon' in order_pre.columns else 0.0
            order_hour = order_pre['order_hour'].iloc[0] if 'order_hour' in order_pre.columns else 0
            global_volume = proxy_data['hourly_volume'].get(order_hour, 0)
            cust_region = proxy_data['dc_region_map'].get(order_info['dc_des'], None)
            
            # Compute base costs [D, C]
            base_cost_grid = compute_base_costs_for_order(
                customer_lat, customer_lon,
                dcs, carriers,
                proxy_data['dc_metadata'],
                proxy_data['cost_coef_map'],
            )
            
            # Compute region match vector [D]
            region_match_vec = np.zeros(D, dtype=np.float32)
            if cust_region is not None:
                for d_idx, dc_id in enumerate(dcs):
                    if proxy_data['dc_region_map'].get(dc_id, None) == cust_region:
                        region_match_vec[d_idx] = 1.0
            
            # Compute distances [D]
            distance_vec = np.zeros(D, dtype=np.float32)
            for d_idx, dc_id in enumerate(dcs):
                if dc_id not in proxy_data['dc_meta_dict']:
                    distance_vec[d_idx] = 9999.0
                    continue
                dc_info = proxy_data['dc_meta_dict'][dc_id]
                lat1, lon1 = np.radians(dc_info['lat']), np.radians(dc_info['lon'])
                lat2, lon2 = np.radians(customer_lat), np.radians(customer_lon)
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance_vec[d_idx] = 6371.0 * c
            
            # Global features per SKU
            global_feats_list = []
            sku_daily_demand_list = []
            for b, sku in enumerate(order_skus):
                order_feat_vec = order_pre[cfg.PROXY_ORDER_FEATURES].iloc[b].to_dict()
                global_feats = compute_global_features_for_order(
                    order_feat_vec, qty_vec[b], global_volume, base_cost_grid=base_cost_grid
                )
                global_feats_list.append(list(global_feats.values()))
                sku_daily_demand_list.append(proxy_data['sku_daily_demand'].get(sku, 0.0))
        
        # ===== Dynamic Feature Engineering (always computed) =====
        # Basket consolidation [D] - depends on current inventory
        inv_binary = (inv_mat > 0).astype(np.float32)
        total_items_at_dc = inv_binary.sum(axis=0)
        
        # Convert shipping costs to [B, S, D, C] delivery penalty grid
        delivery_penalty = np.zeros((B, E, D, C), dtype=np.float32)
        for b, sku in enumerate(order_skus):
            cost_df = shipping_costs.get(sku)
            if cost_df is None or cost_df.empty:
                continue
            for _, opt_row in order_options.iterrows():
                opt_id = int(opt_row['option_id'])
                dc_id, carrier_id = int(opt_row['dc_id']), int(opt_row['carrier_service_id'])
                if opt_id in cost_df.columns and dc_id in dcs and carrier_id in carriers:
                    d_idx, c_idx = dcs.index(dc_id), carriers.index(carrier_id)
                    # Delivery penalty = total_cost - base_cost
                    delivery_penalty[b, :, d_idx, c_idx] = (
                        cost_df[opt_id].values[:E] - base_cost_grid[d_idx, c_idx]
                    )
        
        # Build features for each SKU
        dc_feats_list, option_feats_list = [], []
        for b, sku in enumerate(order_skus):
            consolidation = total_items_at_dc - inv_binary[b]
            sku_daily_demand = sku_daily_demand_list[b]
            
            # DC features [D, K_dc=5] - uses current inventory + static distance/region
            dos_vec = inv_mat[b] / (sku_daily_demand + 1e-6)
            dc_feats = np.stack([
                inv_mat[b],           # inventory_level (dynamic)
                dos_vec,              # days_of_supply (dynamic)
                region_match_vec,     # region_match (static)
                consolidation,        # consolidation_potential (dynamic)
                distance_vec          # distance_km (static)
            ], axis=1)  # [D, 5]
            if dynamic_dc_feats is not None:
                dc_feats = np.concatenate([dc_feats, dynamic_dc_feats], axis=1)
            
            # Option features [D, C, K_opt] - base_cost + delivery penalty summaries
            penalty_mean = delivery_penalty[b].mean(axis=0)
            penalty_std = delivery_penalty[b].std(axis=0)
            penalty_p90 = np.quantile(delivery_penalty[b], 0.9, axis=0)
            option_feats = np.stack(
                [base_cost_grid, penalty_mean, penalty_std, penalty_p90],
                axis=-1
            ).astype(np.float32)
            
            dc_feats_list.append(dc_feats)
            option_feats_list.append(option_feats)
        
        # Convert to tensors
        global_features = torch.tensor(np.array(global_feats_list), dtype=torch.float32, device=device)
        dc_features = torch.tensor(np.array(dc_feats_list), dtype=torch.float32, device=device)
        option_features = torch.tensor(np.array(option_feats_list), dtype=torch.float32, device=device)
        scenario_demand = torch.tensor(scen_dem, dtype=torch.float32, device=device)
        delivery_penalty_tensor = torch.tensor(delivery_penalty, dtype=torch.float32, device=device)
        # sku_idx and brand_idx: [B] shape (model's embedding layer + flatten handles this correctly)
        # Note: Training data saves as [N,1] but model.flatten(start_dim=1) handles both [B] and [B,1]
        sku_idx = torch.tensor(order_pre['sku_idx'].values, dtype=torch.long, device=device)
        brand_idx = torch.tensor(order_pre['brand_idx'].values, dtype=torch.long, device=device)
        demand = torch.tensor(qty_vec, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Apply feature scaling (global, DC, option)
        global_np = global_features.cpu().numpy()
        global_np = _apply_scaler(global_np, feature_scalers.get('global'))
        global_features = torch.tensor(global_np, dtype=torch.float32, device=device)

        dc_np = dc_features.cpu().numpy().reshape(-1, dc_features.shape[-1])
        dc_np = _apply_scaler(dc_np, feature_scalers.get('dc'))
        dc_features = torch.tensor(dc_np.reshape(B, D, -1), dtype=torch.float32, device=device)

        option_np = option_features.cpu().numpy().reshape(-1, option_features.shape[-1])
        option_np = _apply_scaler(option_np, feature_scalers.get('option'))
        option_features = torch.tensor(option_np.reshape(B, D, C, -1), dtype=torch.float32, device=device)
        
        # Build eligibility mask [B, D, C] from order_options
        eligibility_mask = np.zeros((D, C), dtype=np.float32)
        for _, opt_row in order_options.iterrows():
            dc_id = int(opt_row['dc_id'])
            carrier_id = int(opt_row['carrier_service_id'])
            if dc_id in dcs and carrier_id in carriers:
                d_idx = dcs.index(dc_id)
                c_idx = carriers.index(carrier_id)
                eligibility_mask[d_idx, c_idx] = 1.0
        eligibility_mask = torch.tensor(
            np.broadcast_to(eligibility_mask, (B, D, C)),
            dtype=torch.float32,
            device=device
        )
        
        inventory_tensor = torch.tensor(inv_mat, dtype=torch.float32, device=device)
        
        # Model inference (forward + decode only)
        model = proxy_data['model']
        model.eval()
        policy_t0 = time.perf_counter()
        with torch.inference_mode():
            output = model(
                global_feats=global_features,
                dc_feats=dc_features,
                option_feats=option_features,
                demand_scenarios=scenario_demand,
                delivery_penalty=delivery_penalty_tensor,
                sku_idx=sku_idx,
                brand_idx=brand_idx,
            )
        
        # Decode plan
        if isinstance(output, tuple):
            # Hierarchical mode: (logits_dc, logits_carrier)
            _, plan, carrier_selection = hierarchical_proxy_inference(
                output,
                inventory=inventory_tensor,
                demand=demand,
                eligibility_mask=eligibility_mask,
                **proxy_data['inference_params'],
            )
        else:
            # Unified mode: logits [B, D*C]
            _, plan, carrier_selection = proxy_inference(
                output,
                inventory=inventory_tensor,
                demand=demand,
                eligibility_mask=eligibility_mask,
                num_dcs=D,
                num_carriers=C,
                **proxy_data['inference_params'],
            )
    
    # Note: use_proxy_features is always True due to fallbacks in prepare_proxy_data and above.
    
    policy_runtime = time.perf_counter() - policy_t0
    
    # Build fulfillment plan
    pos = (plan > 0).nonzero(as_tuple=False)
    plan_rows = []
    for b, d in pos.tolist():
        carrier_idx = int(carrier_selection[b, d].item())
        plan_rows.append({
            'sku_ID': order_skus[b],
            'dc_ori': dcs[d],
            'carrier_service_id': carriers[carrier_idx],
            'quantity': int(plan[b, d].item())
        })
    
    plan_df = pd.DataFrame(plan_rows)
    total_runtime = time.perf_counter() - start
    
    metadata = {
        'runtime_seconds': policy_runtime,
        'total_runtime_seconds': total_runtime,
        'algorithm': 'proxy',
        'num_skus_processed': B
    }
    try:
        ub_stats = _compute_proxy_upper_bound_stats(
            order_skus=order_skus,
            plan=plan,
            carrier_selection=carrier_selection,
            dcs=dcs,
            carriers=carriers,
            order_options=order_options,
            on_hand_inventory_pivot=on_hand_inventory_pivot,
            order_items=order_items,
            demand_scenarios=demand_scenarios,
            shipping_costs=shipping_costs,
            order_set=order_set,
            promise_days=float(order_info.get('promise_delivery_days', 0)),
            scenario_seed=scenario_seed,
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "[proxy] failed to compute per-order UB stats for order %s",
            order_id,
        )
        ub_stats = {
            'policy_ub_mean': float('nan'),
            'policy_ub_std': float('nan'),
            'policy_ub_ci95': float('nan'),
            'policy_current_stage_mean': float('nan'),
            'policy_current_stage_std': float('nan'),
            'policy_current_stage_ci95': float('nan'),
            'policy_future_recourse_mean': float('nan'),
            'policy_future_recourse_std': float('nan'),
            'policy_future_recourse_ci95': float('nan'),
            'policy_ub_n_eval': 0,
            'policy_ub_scenarios_total': 0,
            'policy_ub_eval_start': 0,
            'policy_bound_source': 'proxy_mc',
        }
    metadata.update(ub_stats)
    
    if verbose:
        print(f"Proxy: order {order_id} | {len(plan_rows)} decisions | {policy_runtime:.3f}s")
    
    return plan_df, metadata


def choose_fulfillment_option_level(
    order,
    items,
    option_ids,
    eligible_mask,
    features_df,
    costs_series,
    inventory_snapshot,
    proxy_data,
    order_set: str = 'test',
    simulation_date: str = None,
    order_idx: int = 0,
    verbose: bool = False,
    dynamic_features: Optional[pd.DataFrame] = None,
):
    """Proxy policy wrapper for simulation."""
    from src.simulator.entities import OrderDecision, ItemAllocation
    
    logger = logging.getLogger(__name__)
    
    # Convert simulator objects to DataFrames
    order_items_df = pd.DataFrame([{
        'order_ID': order.order_id,
        'order_time': order.order_time or pd.Timestamp(simulation_date),
        'order_date': pd.to_datetime(order.order_time).date() if order.order_time else pd.Timestamp(simulation_date).date(),
        'sku_ID': item.sku_id,
        'quantity': item.quantity,
    } for item in items])
    
    order_info = pd.Series({
        'order_ID': order.order_id,
        'order_time': order.order_time or pd.Timestamp.now(),
        'dc_des': order.customer_dc if order.customer_dc is not None else order.dest_state,
        'promise_delivery_days': order.promise_delivery_days,
    })
    
    # Build inventory pivot
    dc_ids = sorted(set(int(opt[0]) for opt in option_ids))
    unique_skus = [str(item.sku_id) for item in items]
    inv_matrix = np.zeros((len(unique_skus), len(dc_ids)), dtype=float)
    sku_index = {sku: idx for idx, sku in enumerate(unique_skus)}
    dc_index = {dc: idx for idx, dc in enumerate(dc_ids)}
    
    for dc_key, sku_map in inventory_snapshot.items():
        dc_id = int(dc_key) if isinstance(dc_key, (int, str)) else dc_key
        dc_idx = dc_index.get(dc_id)
        if dc_idx is None:
            continue
        for sku_id, qty in sku_map.items():
            sku_idx = sku_index.get(str(sku_id))
            if sku_idx is not None:
                inv_matrix[sku_idx, dc_idx] = float(qty)
    
    inventory_pivot = pd.DataFrame(inv_matrix, index=unique_skus, columns=dc_ids)
    
    # Build order_options from option_ids
    order_options = pd.DataFrame([{
        'option_id': idx,
        'dc_id': int(opt_id[0]),
        'carrier_service_id': int(opt_id[1]),
        'base_cost': costs_series.iloc[idx] if idx < len(costs_series) else 0.0
    } for idx, opt_id in enumerate(option_ids)])
    
    # Call proxy fulfillment
    plan_df, metadata = proxy_fulfillment(
        order_info, order_items_df, inventory_pivot, proxy_data,
        order_set, simulation_date, order_idx, order_options, verbose, dynamic_features
    )
    
    # Convert to OrderDecision
    allocations = []
    original_qty = {str(item.sku_id): item.quantity for item in items}
    allocated_qty = {sku: 0 for sku in original_qty.keys()}
    
    for _, row in plan_df.iterrows():
        sku_id = str(row['sku_ID'])
        dc_id = int(row['dc_ori'])
        carrier_id = int(row['carrier_service_id'])
        qty = int(row['quantity'])
        
        allocations.append(ItemAllocation(
            sku_id=sku_id,
            option_id=(dc_id, carrier_id),
            quantity=qty,
        ))
        allocated_qty[sku_id] = allocated_qty.get(sku_id, 0) + qty
    
    unfilled = {sku: orig - allocated_qty.get(sku, 0) 
                for sku, orig in original_qty.items() 
                if orig - allocated_qty.get(sku, 0) > 0}
    
    decision = OrderDecision(allocations=allocations, unfilled=unfilled or None)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[proxy] order {order.order_id}: {len(allocations)} allocations")
    
    policy_stats = {
        'policy_ub_mean': metadata.get('policy_ub_mean', float('nan')),
        'policy_ub_ci95': metadata.get('policy_ub_ci95', float('nan')),
        'policy_current_stage_mean': metadata.get('policy_current_stage_mean', float('nan')),
        'policy_current_stage_ci95': metadata.get('policy_current_stage_ci95', float('nan')),
        'policy_future_recourse_mean': metadata.get('policy_future_recourse_mean', float('nan')),
        'policy_future_recourse_ci95': metadata.get('policy_future_recourse_ci95', float('nan')),
        'policy_ub_n_eval': int(metadata.get('policy_ub_n_eval', 0) or 0),
        'policy_ub_scenarios_total': int(metadata.get('policy_ub_scenarios_total', 0) or 0),
        'policy_ub_eval_start': int(metadata.get('policy_ub_eval_start', 0) or 0),
        'policy_bound_source': metadata.get('policy_bound_source', 'proxy_mc'),
    }

    return decision, metadata.get('runtime_seconds', 0.0), policy_stats


def create_policy_for_simulation(
    catalog,
    precompute,
    state,
    proxy_model=None,
    proxy_stochastic=False,
    proxy_top_k=5,
    proxy_scenario_len: int | None = None,
    order_set='test',
    simulation_date=None,
    eligibility_audit_path: str | None = None,
    **kwargs
):
    """
    Create proxy policy closure for simulation.
    
    Args:
        catalog: OptionsCatalog instance
        precompute: PrecomputeStore instance
        state: SimulationState instance
        proxy_model: str - Path to trained model checkpoint
        proxy_stochastic: bool - Enable stochastic sampling
        proxy_top_k: int - Top-k for stochastic sampling
        order_set: str - 'test' or 'proxy_train'
        simulation_date: str - Simulation date
        **kwargs: Additional arguments
    
    Returns:
        Policy function returning (OrderDecision, runtime_seconds, policy_stats_dict)
    """
    from src.simulator.features import build_costs
    from src.simulator.entities import OrderDecision
    import torch
    
    if proxy_model is None:
        raise ValueError("proxy_model path is required")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[proxy] Loading model from {proxy_model} on {device}")
    
    checkpoint = torch.load(proxy_model, map_location=device, weights_only=False)
    model_params = checkpoint['model_params']
    info = checkpoint.get('info', {})
    if 'info' not in checkpoint:
         info = model_params
    
    # Older checkpoints may store the architecture under hyperparams.
    if 'architecture' not in model_params and 'model_variant' in checkpoint.get('hyperparams', {}):
        model_params['architecture'] = checkpoint['hyperparams']['model_variant']

    # Merge missing info fields from model_params for robust inference
    if info is None:
        info = {}
    info = dict(info)
    info.setdefault('architecture', model_params.get('architecture'))
    info.setdefault('model_variant', model_params.get('architecture'))
    for key in (
        'global_feature_dim', 'dc_feature_dim', 'option_feature_dim',
        'num_dcs', 'num_carriers', 'sku_dim', 'brand_dim', 'scenario_len',
        'dcs', 'carriers'
    ):
        if key not in info and key in model_params:
            info[key] = model_params[key]
    if proxy_scenario_len is not None:
        info['scenario_len'] = int(proxy_scenario_len)
        print(f"[proxy] Overriding scenario_len to {info['scenario_len']}")
        
    model = build_proxy_model(model_params).to(device)

    state_dict = checkpoint['model']
    if state_dict:
        first_key = next(iter(state_dict))
        if first_key.startswith('module.'):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    
    feature_scalers = checkpoint.get('feature_scalers')
    scaler = None
    if feature_scalers is None:
        scaler = checkpoint.get('scaler')
        
    repair_strategy = (
        _normalize_repair_strategy(kwargs.get('repair_strategy'))
        or _normalize_repair_strategy(model_params.get('repair_strategy'))
        or _normalize_repair_strategy(checkpoint.get('hyperparams', {}).get('inference', {}).get('repair_strategy'))
        or _normalize_repair_strategy(getattr(cfg, "PROXY_MODEL_REPAIR_STRATEGY", "argmax_then_split"))
    )
    inference_params = {
        'stochastic': proxy_stochastic,
        'top_k': proxy_top_k,
        'repair': True,
        'threshold': model_params.get('threshold', 0.5),
        'repair_strategy': repair_strategy,
        'inventory_weight_power': float(
            kwargs.get('inventory_weight_power')
            if kwargs.get('inventory_weight_power') is not None
            else model_params.get('inventory_weight_power', getattr(cfg, "PROXY_MODEL_INVENTORY_WEIGHT_POWER", 1.0))
        ),
    }
    
    proxy_data = prepare_proxy_data(
        model=model,
        inference_params=inference_params,
        device=device,
        model_info=info,
        feature_scalers=feature_scalers,
        scaler=scaler,
        order_set=order_set,
        simulation_date=simulation_date,
    )
    
    audit_handle = None
    if eligibility_audit_path:
        audit_path = Path(eligibility_audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_handle = audit_path.open("a", encoding="utf-8")

    def _audit(order_id: str, option_ids, info):
        if not audit_handle:
            return
        dcs = info.get("dcs", [])
        carriers = info.get("carriers", [])
        dc_set = {int(d) for d in dcs}
        carrier_set = {int(c) for c in carriers}
        missing = [(int(dc), int(car)) for dc, car in option_ids if int(dc) not in dc_set or int(car) not in carrier_set]
        payload = {
            "order_id": order_id,
            "eligible_count": len(option_ids),
            "missing_in_model": len(missing),
            "sample_missing": missing[:10],
        }
        audit_handle.write(json.dumps(payload) + "\n")
        audit_handle.flush()

    def policy(order):
         option_ids = catalog.eligible_for_order(order)
         if not option_ids:
             return OrderDecision(
                allocations=[],
                unfilled={item.sku_id: item.quantity for item in order.items}
            ), 0.0
         _audit(order.order_id, option_ids, proxy_data["info"])
         
         costs_series = build_costs(order, option_ids, catalog, precompute)
         
         dynamic_features = None
         if hasattr(state, 'build_dc_event_snapshot'):
             try:
                 dynamic_features = state.build_dc_event_snapshot()
             except Exception:
                 dynamic_features = None

         decision, runtime, policy_stats = choose_fulfillment_option_level(
             order=order,
             items=order.items,
             option_ids=option_ids,
             eligible_mask=None,
             features_df=None,
             costs_series=costs_series,
             inventory_snapshot=state.inventory,
             proxy_data=proxy_data,
             order_set=order_set,
             simulation_date=simulation_date,
             order_idx=0, 
             dynamic_features=dynamic_features,
         )
         return decision, runtime, policy_stats

    return policy
