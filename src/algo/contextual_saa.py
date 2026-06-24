"""
Contextual SAA (Stochastic Average Approximation) baseline algorithm.

This module implements the standard SAA approach that uses ML model predictions
to generate scenarios for demand and delivery times, then applies the SAA procedure.
"""

from typing import Tuple
import time
import logging
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import src.config as cfg
from src.scenario_generator import generate_scenarios
from src.saa_procedure import run_saa_procedure
from src.utils import calculate_lookahead_periods

logger = logging.getLogger(__name__)
from src.data_structures.options import OptionMatrix, ensure_option_matrix


def contextual_saa_fulfillment(
    order_info: pd.Series,
    order_items: pd.DataFrame,
    on_hand_inventory_pivot: pd.DataFrame,
    order_options,
    order_set: str,
    simulation_date: str,
    order_idx: int,
    verbose: bool = False,
    dynamic_features: pd.DataFrame | None = None,
    order_id: str | None = None,
    debug_dump_dir: Path | None = None,
    debug_dump_level: str = "full",
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate fulfillment plan using contextual SAA.
    
    Uses ML model predictions to generate scenarios for demand and delivery times,
    then applies the SAA procedure to find the optimal fulfillment plan.
    
    Args:
        order_info: Order information Series
        order_items: Order items DataFrame
        on_hand_inventory_pivot: Pivoted inventory DataFrame
        base_costs: Base shipping costs Series
        unit_costs_df: Unit costs DataFrame
        compatible_dcs: List of compatible DCs
        dc_to_region: DC to region mapping
        order_set: Order set ('test' or 'proxy_train')
        simulation_date: Simulation date string
        order_idx: Order index for seeding
        verbose: Whether to enable verbose logging
        
    Returns:
        Tuple of (plan_df, metadata_dict)
    """
    current_order_time = order_info['order_time']
    customer_dc = order_info['dc_des']
    promise_days = order_info['promise_delivery_days']
    lookahead_periods = calculate_lookahead_periods(current_order_time, simulation_date)
    total_scenarios_needed = cfg.SAA_Q * cfg.SAA_N1 + cfg.SAA_N2
    period = 'proxy' if order_set == 'proxy_train' else 'test'

    options_matrix = ensure_option_matrix(order_options)
    order_options_df = options_matrix.to_dataframe()

    timings: dict[str, float] = {}
    scenario_start = time.perf_counter()
    # Generate scenarios using ML models
    demand_scenarios, shipping_costs = generate_scenarios(
        order_items=order_items,
        options_df=order_options_df,
        customer_dc=customer_dc,
        promise_days=promise_days,
        num_scenarios=total_scenarios_needed,
        seed=cfg.RANDOM_SEED + order_idx,
        period=period,
        lookahead_periods=lookahead_periods,
        verbose=verbose,
        dynamic_features=dynamic_features,
    )
    timings['scenario_generation'] = time.perf_counter() - scenario_start

    debug_metadata_payload: dict | None = None
    saa_payload: dict | None = None
    dump_level = (debug_dump_level or "full").lower()
    if dump_level not in {"full", "stats"}:
        dump_level = "full"
    if debug_dump_dir is not None:
        debug_dump_dir.mkdir(parents=True, exist_ok=True)

        num_scenarios_val = 0
        if demand_scenarios:
            any_sku = next(iter(demand_scenarios.keys()))
            num_scenarios_val = len(demand_scenarios[any_sku])

        if dump_level == "stats":
            # Compact per-order stats for later reporting (bounds/CI, timings, seeds).
            debug_metadata_payload = {
                'order_id': order_id,
                'policy': 'csaa',
                'customer_dc': customer_dc,
                'promise_delivery_days': promise_days,
                'lookahead_periods': lookahead_periods,
                'num_scenarios': num_scenarios_val,
                'scenario_seed': int(cfg.RANDOM_SEED + order_idx),
                'saa_master_seed': int(cfg.RANDOM_SEED + order_idx),
                'created_epoch': time.time(),
                'dump_level': dump_level,
            }
        else:
            scenario_file = debug_dump_dir / 'candidate_scenarios.npz'
        
            # Convert dicts to arrays for saving (if needed)
            save_dict = {}
            if demand_scenarios:
                # Convert demand dict to arrays per SKU
                for sku, series in demand_scenarios.items():
                    vals = series.values
                    if np.issubdtype(vals.dtype, np.floating):
                        vals = np.rint(vals)
                    max_val = np.nanmax(vals) if vals.size else 0
                    if max_val <= np.iinfo(np.int16).max:
                        vals = vals.astype(np.int16, copy=False)
                    else:
                        vals = vals.astype(np.int32, copy=False)
                    save_dict[f'demand_{sku}'] = vals
            # NOTE: shipping costs are intentionally not stored in candidate_scenarios.npz
            # to reduce storage. Use base_costs.npy + delivery_penalties.npz instead.
            np.savez_compressed(scenario_file, **save_dict)

            # Save dynamic feature snapshot (space-efficient)
            if dynamic_features is not None and isinstance(dynamic_features, pd.DataFrame):
                dyn_df = dynamic_features.copy()
                if 'dc_id' in dyn_df.columns:
                    dyn_df['dc_id'] = dyn_df['dc_id'].astype('int32')
                dyn_path = debug_dump_dir / 'dynamic_features.parquet'
                dyn_df.to_parquet(dyn_path, index=False, compression='snappy')

            # Save per-order option snapshot with deterministic base costs
            order_options_df.to_parquet(
                debug_dump_dir / 'order_options.parquet',
                index=False,
                compression='snappy'
            )

            # Store deterministic base costs and delivery-time penalties separately
            base_costs_arr = order_options_df['base_cost'].astype(float).values
            np.save(debug_dump_dir / 'base_costs.npy', base_costs_arr.astype(np.float16, copy=False))

            if shipping_costs:
                any_sku = next(iter(shipping_costs.keys()))
                shipping_df = shipping_costs[any_sku].reindex(columns=order_options_df['option_id'].values)
                shipping_mat = shipping_df.values.astype(np.float32, copy=False)
                penalties = shipping_mat - base_costs_arr.reshape(1, -1)
                np.savez_compressed(
                    debug_dump_dir / 'delivery_penalties.npz',
                    penalty=penalties.astype(np.float16, copy=False),
                    option_id=order_options_df['option_id'].values.astype(np.int32, copy=False)
                )

            # Compute statistics from dictionaries
            demand_mean = []
            demand_std = []
            if demand_scenarios:
                for sku in sorted(demand_scenarios.keys()):
                    series = demand_scenarios[sku]
                    demand_mean.append(float(series.mean()) if len(series) > 0 else 0.0)
                    demand_std.append(float(series.std()) if len(series) > 0 else 0.0)

            shipping_mean = []
            shipping_std = []
            if shipping_costs:
                # Aggregate across all SKUs (they should have same structure)
                any_sku = next(iter(shipping_costs.keys()))
                df = shipping_costs[any_sku]
                shipping_mean = df.mean(axis=0).tolist() if len(df) > 0 else []
                shipping_std = df.std(axis=0).tolist() if len(df) > 0 else []

            debug_metadata_payload = {
                'order_id': order_id,
                'policy': 'csaa',
                'customer_dc': customer_dc,
                'promise_delivery_days': promise_days,
                'lookahead_periods': lookahead_periods,
                'num_scenarios': num_scenarios_val,
                'scenario_seed': int(cfg.RANDOM_SEED + order_idx),
                'saa_master_seed': int(cfg.RANDOM_SEED + order_idx),
                'created_epoch': time.time(),
                'dump_level': dump_level,
                'sku_ids': order_items['sku_ID'].astype(str).tolist(),
                'option_snapshot': order_options_df[['option_id', 'dc_id', 'carrier_service_id']].to_dict('records'),
                'scenario_statistics': {
                    'demand_mean_per_sku': demand_mean,
                    'demand_std_per_sku': demand_std,
                    'shipping_costs_mean': shipping_mean,
                    'shipping_costs_std': shipping_std,
                },
            }
    
    solve_start = time.perf_counter()
    # Run SAA procedure with ML-generated scenarios
    if debug_dump_dir is not None:
        plan_df, runtime, saa_timings, saa_payload = run_saa_procedure(
            order_items=order_items,
            options_df=options_matrix,
            on_hand_inventory_pivot=on_hand_inventory_pivot.copy(),
            customer_dc=customer_dc,
            promise_days=promise_days,
            stockout_penalty=cfg.STOCKOUT_PENALTY_PER_UNIT,
            order_set=order_set,
            master_seed=cfg.RANDOM_SEED + order_idx,
            demand_scenarios=demand_scenarios,
            shipping_costs=shipping_costs,
            verbose=verbose,
            return_candidates=True,
            use_future_penalties=bool(getattr(cfg, "SAA_USE_FUTURE_PENALTIES", False)),
        )
    else:
        plan_df, runtime, saa_timings = run_saa_procedure(
            order_items=order_items,
            options_df=options_matrix,
            on_hand_inventory_pivot=on_hand_inventory_pivot.copy(),
            customer_dc=customer_dc,
            promise_days=promise_days,
            stockout_penalty=cfg.STOCKOUT_PENALTY_PER_UNIT,
            order_set=order_set,
            master_seed=cfg.RANDOM_SEED + order_idx,
            demand_scenarios=demand_scenarios,
            shipping_costs=shipping_costs,
            verbose=verbose,
            use_future_penalties=bool(getattr(cfg, "SAA_USE_FUTURE_PENALTIES", False)),
        )
    if plan_df is None:
        plan_df = pd.DataFrame(columns=['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity'])
    timings['saa_solver'] = time.perf_counter() - solve_start
    timings['total_contextual'] = timings['scenario_generation'] + timings['saa_solver']
    
    if debug_dump_dir is not None:
        if dump_level == "full":
            if not plan_df.empty:
                plan_df.to_parquet(debug_dump_dir / 'fulfillment_plan.parquet', index=False, compression='snappy')
            order_items.to_parquet(debug_dump_dir / 'order_items.parquet', index=False, compression='snappy')
            on_hand_inventory_pivot.to_parquet(debug_dump_dir / 'on_hand_inventory_pivot.parquet', compression='snappy')

            # Save all SAA candidate solutions (compact row format).
            try:
                cand_stats = (saa_payload or {}).get('stats', {})
                cand_indices = cand_stats.get('candidate_indices', [])
                cand_solutions = (saa_payload or {}).get('candidates', [])
                if cand_solutions:
                    opt_map = order_options_df.set_index('option_id')[['dc_id', 'carrier_service_id']]
                    rows = []
                    for pos, z in enumerate(cand_solutions):
                        cand_idx = int(cand_indices[pos]) if pos < len(cand_indices) else pos
                        for (sku, opt_id), qty in z.items():
                            if qty <= 1e-6:
                                continue
                            dc_val = int(opt_map.loc[opt_id, 'dc_id'])
                            carrier_val = int(opt_map.loc[opt_id, 'carrier_service_id'])
                            rows.append((pos, cand_idx, str(sku), dc_val, carrier_val, int(round(qty))))
                    if rows:
                        cand_df = pd.DataFrame(
                            rows,
                            columns=['candidate_pos', 'candidate_idx', 'sku_ID', 'dc_ori', 'carrier_service_id', 'quantity'],
                        )
                        cand_df['candidate_pos'] = cand_df['candidate_pos'].astype('int16')
                        cand_df['candidate_idx'] = cand_df['candidate_idx'].astype('int16')
                        cand_df['dc_ori'] = cand_df['dc_ori'].astype('int16')
                        cand_df['carrier_service_id'] = cand_df['carrier_service_id'].astype('int16')
                        cand_df['quantity'] = cand_df['quantity'].astype('int16')
                        cand_df.to_parquet(debug_dump_dir / 'saa_candidates.parquet', index=False, compression='snappy')
            except Exception as e:
                logger.warning("Failed to dump SAA candidates for order_id=%s: %s", order_id, e)

        # Persist per-order metadata (single pickle write for speed).
        if debug_metadata_payload is not None:
            if saa_payload is not None:
                cand_stats = (saa_payload or {}).get('stats', {})
                debug_metadata_payload['saa_bounds'] = {
                    'saa_q': int(cfg.SAA_Q),
                    'saa_n1': int(cfg.SAA_N1),
                    'saa_n2': int(cfg.SAA_N2),
                    'ubar_f': float(saa_timings.get('ubar_f', float('nan'))),
                    'ubar_f_std': float(saa_timings.get('ubar_f_std', float('nan'))),
                    'ubar_f_ci95': float(saa_timings.get('ubar_f_ci95', float('nan'))),
                    'ubar_f_n': int(saa_timings.get('ubar_f_n', 0) or 0),
                    'ubar_f_current_stage': float(saa_timings.get('ubar_f_current_stage', float('nan'))),
                    'ubar_f_current_stage_std': float(saa_timings.get('ubar_f_current_stage_std', float('nan'))),
                    'ubar_f_current_stage_ci95': float(saa_timings.get('ubar_f_current_stage_ci95', float('nan'))),
                    'ubar_f_future_recourse': float(saa_timings.get('ubar_f_future_recourse', float('nan'))),
                    'ubar_f_future_recourse_std': float(saa_timings.get('ubar_f_future_recourse_std', float('nan'))),
                    'ubar_f_future_recourse_ci95': float(saa_timings.get('ubar_f_future_recourse_ci95', float('nan'))),
                    'bar_f': float(saa_timings.get('bar_f', float('nan'))),
                    'bar_f_std': float(saa_timings.get('bar_f_std', float('nan'))),
                    'bar_f_ci95': float(saa_timings.get('bar_f_ci95', float('nan'))),
                    'bar_f_n': int(saa_timings.get('bar_f_n', 0) or 0),
                    'bar_f_current_stage': float(saa_timings.get('bar_f_current_stage', float('nan'))),
                    'bar_f_current_stage_std': float(saa_timings.get('bar_f_current_stage_std', float('nan'))),
                    'bar_f_current_stage_ci95': float(saa_timings.get('bar_f_current_stage_ci95', float('nan'))),
                    'bar_f_future_recourse': float(saa_timings.get('bar_f_future_recourse', float('nan'))),
                    'bar_f_future_recourse_std': float(saa_timings.get('bar_f_future_recourse_std', float('nan'))),
                    'bar_f_future_recourse_ci95': float(saa_timings.get('bar_f_future_recourse_ci95', float('nan'))),
                    'gap': float(saa_timings.get('gap', float('nan'))),
                    'master_seed': int(cand_stats.get('master_seed', cfg.RANDOM_SEED + order_idx)),
                    'future_penalty_seed': cand_stats.get('future_penalty_seed', None),
                    'use_future_penalties': bool(cand_stats.get('use_future_penalties', getattr(cfg, "SAA_USE_FUTURE_PENALTIES", False))),
                    'candidate_indices': cand_stats.get('candidate_indices', []),
                    'candidate_obj_n1': cand_stats.get('candidate_obj_n1', []),
                    'candidate_obj_n1_current_stage': cand_stats.get('candidate_obj_n1_current_stage', []),
                    'candidate_obj_n1_future_recourse': cand_stats.get('candidate_obj_n1_future_recourse', []),
                    'candidate_eval_mean': cand_stats.get('candidate_eval_mean', []),
                    'candidate_eval_std': cand_stats.get('candidate_eval_std', []),
                    'candidate_eval_current_stage_mean': cand_stats.get('candidate_eval_current_stage_mean', []),
                    'candidate_eval_current_stage_std': cand_stats.get('candidate_eval_current_stage_std', []),
                    'candidate_eval_future_recourse_mean': cand_stats.get('candidate_eval_future_recourse_mean', []),
                    'candidate_eval_future_recourse_std': cand_stats.get('candidate_eval_future_recourse_std', []),
                    'best_pos': cand_stats.get('best_pos', None),
                    'best_candidate_idx': cand_stats.get('best_candidate_idx', None),
                }
            debug_metadata_payload['timings'] = {**timings, 'saa_details': saa_timings}

            metadata_file = debug_dump_dir / 'metadata.pkl'
            with metadata_file.open('wb') as f:
                pickle.dump(debug_metadata_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    metadata = {
        'runtime_seconds': timings['saa_solver'],  # Only SAA optimization time (not scenario generation)
        'total_runtime': timings['total_contextual'],  # Includes scenario generation
        'saa_n1': cfg.SAA_N1,
        'saa_n2': cfg.SAA_N2,
        'saa_q': cfg.SAA_Q,
        'source': 'ml_models',
        'lookahead_periods': lookahead_periods,
        'timings': {**timings, 'saa_details': saa_timings},
        'debug_dump_dir': str(debug_dump_dir) if debug_dump_dir else None,
        'order_id': order_id,
    }
    
    return plan_df, metadata


# Imported shared utilities - see src/algo/algo_utils.py for implementations
from src.algo.algo_utils import (
    prepare_order_items as _prepare_order_items,
    build_inventory_pivot as _build_inventory_pivot,
    build_option_matrix as _build_option_matrix,
)


def choose_fulfillment_option_level(
    order,
    items,
    option_ids,
    eligible_mask,
    features_df,
    costs_series,
    inventory_snapshot,
    order_set: str = 'test',
    simulation_date: str = None,
    order_idx: int = 0,
    verbose: bool = False,
    dynamic_features: pd.DataFrame | None = None,
    scenario_debug_dir: Path | None = None,
    csaa_dump_level: str = "full",
):
    """
    Option-level wrapper for contextual SAA fulfillment.
    
    Args:
        order: Order object
        items: List of OrderItem objects
        option_ids: List of OptionId tuples
        eligible_mask: Boolean mask (not used, all provided options are eligible)
        features_df: Option-indexed features DataFrame
        costs_series: Option-indexed cost Series
        inventory_snapshot: dc_id -> {sku_id: onhand} dict
        order_set: Order set ('test' or 'proxy_train')
        simulation_date: Simulation date string (required)
        order_idx: Order index for seeding
        verbose: Whether to enable verbose logging
        
    Returns:
        Tuple of (OrderDecision, runtime_seconds, policy_stats_dict)
    """
    from src.simulator.entities import OrderDecision, ItemAllocation
    import pandas as pd
    
    if simulation_date is None:
        raise ValueError("simulation_date is required for contextual SAA")
    
    # Convert items to lightweight arrays/DataFrames
    fallback_order_time = order.order_time or (pd.Timestamp(simulation_date) if simulation_date else None)
    order_items_df, unique_skus, original_qty = _prepare_order_items(order, items, fallback_order_time)
    
    # Build inventory pivot using NumPy
    inventory_pivot = _build_inventory_pivot(unique_skus, option_ids, inventory_snapshot)
    
    # Build order_options DataFrame (required format for SAA)
    order_options = _build_option_matrix(option_ids, costs_series)
    
    # Build order_info Series
    dest_dc_value = order.customer_dc
    order_info = pd.Series({
        'order_time': order.order_time or pd.Timestamp.now(),
        'dc_des': dest_dc_value if dest_dc_value is not None else order.dest_state,
        'promise_delivery_days': order.promise_delivery_days,
    })
    
    # Call contextual SAA
    plan_df, metadata = contextual_saa_fulfillment(
        order_info=order_info,
        order_items=order_items_df,
        on_hand_inventory_pivot=inventory_pivot,
        order_options=order_options,
        order_set=order_set,
        simulation_date=simulation_date,
        order_idx=order_idx,
        verbose=verbose,
        dynamic_features=dynamic_features,
        order_id=str(getattr(order, 'order_id', '')),
        debug_dump_dir=scenario_debug_dir,
        debug_dump_level=csaa_dump_level,
    )
    
    # Extract runtime and bound stats from metadata
    runtime = metadata.get('runtime_seconds', 0.0)
    timing_info = metadata.get('timings', {}) or {}
    saa_details = timing_info.get('saa_details') or {}

    def _as_float(value) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float('nan')
        return out if np.isfinite(out) else float('nan')

    ub_mean = _as_float(saa_details.get('bar_f'))
    ub_ci95 = _as_float(saa_details.get('bar_f_ci95'))
    current_stage_mean = _as_float(saa_details.get('bar_f_current_stage'))
    current_stage_ci95 = _as_float(saa_details.get('bar_f_current_stage_ci95'))
    future_recourse_mean = _as_float(saa_details.get('bar_f_future_recourse'))
    future_recourse_ci95 = _as_float(saa_details.get('bar_f_future_recourse_ci95'))
    lb_mean = _as_float(saa_details.get('ubar_f'))
    lb_ci95 = _as_float(saa_details.get('ubar_f_ci95'))
    lb_current_stage_mean = _as_float(saa_details.get('ubar_f_current_stage'))
    lb_current_stage_ci95 = _as_float(saa_details.get('ubar_f_current_stage_ci95'))
    lb_future_recourse_mean = _as_float(saa_details.get('ubar_f_future_recourse'))
    lb_future_recourse_ci95 = _as_float(saa_details.get('ubar_f_future_recourse_ci95'))
    ub_n = int(saa_details.get('bar_f_n', 0) or 0)
    lb_n = int(saa_details.get('ubar_f_n', 0) or 0)
    policy_stats = {
        'policy_ub_mean': ub_mean,
        'policy_ub_ci95': ub_ci95,
        'policy_ub_n_eval': ub_n,
        'policy_current_stage_mean': current_stage_mean,
        'policy_current_stage_ci95': current_stage_ci95,
        'policy_future_recourse_mean': future_recourse_mean,
        'policy_future_recourse_ci95': future_recourse_ci95,
        'policy_lb_mean': lb_mean,
        'policy_lb_ci95': lb_ci95,
        'policy_lb_n_eval': lb_n,
        'policy_lb_current_stage_mean': lb_current_stage_mean,
        'policy_lb_current_stage_ci95': lb_current_stage_ci95,
        'policy_lb_future_recourse_mean': lb_future_recourse_mean,
        'policy_lb_future_recourse_ci95': lb_future_recourse_ci95,
        'policy_bound_source': 'csaa',
    }
    if np.isfinite(ub_mean) and np.isfinite(lb_mean):
        policy_stats['policy_ub_minus_lb'] = float(ub_mean - lb_mean)
    if np.isfinite(ub_ci95) and np.isfinite(lb_ci95):
        policy_stats['policy_ub_minus_lb_ci95'] = float(np.sqrt(ub_ci95 * ub_ci95 + lb_ci95 * lb_ci95))
    
    # Convert to OrderDecision
    allocations = []
    allocated_qty: dict[str, int] = {}
    
    if plan_df is not None and not plan_df.empty:
        plan_values = plan_df[['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity']].to_numpy()
    else:
        plan_values = np.empty((0, 4))
    
    for sku_id, dc_id, carrier_id, qty in plan_values:
        sku_id = str(sku_id)
        qty_int = int(float(qty))
        if qty_int <= 0 or pd.isna(dc_id) or pd.isna(carrier_id):
            continue
        allocations.append(ItemAllocation(
            sku_id=sku_id,
            option_id=(int(dc_id), int(carrier_id)),
            quantity=qty_int,
        ))
        allocated_qty[sku_id] = allocated_qty.get(sku_id, 0) + qty_int

    unfilled = {}
    for sku_id, orig in original_qty.items():
        rem = int(orig) - int(allocated_qty.get(sku_id, 0))
        if rem > 0:
            unfilled[sku_id] = rem

    decision = OrderDecision(allocations=allocations, unfilled=unfilled or None)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[csaa] order %s timings: scenarios=%.4fs saa=%.4fs total=%.4fs allocations=%d unfilled=%d",
            getattr(order, 'order_id', '<unknown>'),
            timing_info.get('scenario_generation', 0.0),
            timing_info.get('saa_solver', 0.0),
            timing_info.get('total_contextual', 0.0),
            len(allocations),
            sum(unfilled.values()) if unfilled else 0,
        )
        if saa_details:
            logger.debug(
                "[csaa] order %s SAA detail timings: candidates=%d total=%.4fs avg=%.4fs | evaluations=%d total=%.4fs avg=%.4fs | split=%.4fs overall=%.4fs | ubar_f=%.4f bar_f=%.4f gap=%.4f",
                getattr(order, 'order_id', '<unknown>'),
                int(saa_details.get('num_candidates', 0) or 0),
                float(saa_details.get('candidate_total', 0.0) or 0.0),
                float(saa_details.get('candidate_avg', 0.0) or 0.0),
                int(saa_details.get('num_evaluations', 0) or 0),
                float(saa_details.get('evaluation_total', 0.0) or 0.0),
                float(saa_details.get('evaluation_avg', 0.0) or 0.0),
                float(saa_details.get('scenario_split', 0.0) or 0.0),
                float(saa_details.get('total', 0.0) or 0.0),
                float(saa_details.get('ubar_f', float('nan'))),
                float(saa_details.get('bar_f', float('nan'))),
                float(saa_details.get('gap', float('nan'))),
            )

    return decision, runtime, policy_stats


def create_policy_for_simulation(catalog, precompute, state, order_set, 
                                simulation_date, csaa_debug_dir=None, 
                                csaa_log_limit=None, csaa_dump_level: str = "full", **kwargs):
    """
    Create CSAA policy closure for simulation.
    
    Args:
        catalog: OptionsCatalog instance
        precompute: PrecomputeStore instance
        state: SimulationState instance
        order_set: Order set ('test' or 'proxy_train')
        simulation_date: Simulation date string (YYYY-MM-DD)
        csaa_debug_dir: Optional directory to save CSAA debug information
        csaa_log_limit: Maximum number of orders to log debug info for
        csaa_dump_level: 'full' (scenarios/options/etc) or 'stats' (bounds/timings only)
        **kwargs: Additional unused arguments for API consistency
    
    Returns:
        Policy function that takes an Order and returns
        (OrderDecision, runtime_seconds, policy_stats_dict).
    """
    from src.simulator.features import build_costs
    from src.simulator.entities import OrderDecision
    
    csaa_logged_orders = 0
    
    def policy(order):
        """CSAA policy that uses contextual scenario generation."""
        nonlocal csaa_logged_orders
        
        option_ids = catalog.eligible_for_order(order)
        if not option_ids:
            return OrderDecision(
                allocations=[],
                unfilled={item.sku_id: item.quantity for item in order.items}
            ), 0.0
        
        costs_series = build_costs(order, option_ids, catalog, precompute)
        candidate_dc_ids = sorted(set(int(opt[0]) for opt in option_ids))
        dynamic_snapshot = state.build_dc_event_snapshot(dc_ids=candidate_dc_ids)
        
        # Setup debug directory if enabled
        scenario_debug_dir = None
        if csaa_debug_dir and (csaa_log_limit is None or csaa_logged_orders < csaa_log_limit):
            scenario_debug_dir = csaa_debug_dir / str(order.order_id)
            scenario_debug_dir.mkdir(parents=True, exist_ok=True)
            csaa_logged_orders += 1
        
        order_idx = hash(order.order_id) % 10000
        return choose_fulfillment_option_level(
            order=order,
            items=order.items,
            option_ids=option_ids,
            eligible_mask=None,
            features_df=None,
            costs_series=costs_series,
            inventory_snapshot=state.inventory,
            order_set=order_set,
            simulation_date=simulation_date,
            order_idx=order_idx,
            dynamic_features=dynamic_snapshot,
            scenario_debug_dir=scenario_debug_dir,
            csaa_dump_level=csaa_dump_level,
        )
    
    return policy
