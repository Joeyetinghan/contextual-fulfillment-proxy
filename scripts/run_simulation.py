"""Unified simulation script for online policy evaluation."""

import sys
import argparse
import hashlib
import json
import logging
import os
import pickle
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path so `import src.*` works when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.config as cfg
from src.data_utils import get_initial_inventory
from src.simulator import (
    OptionsCatalog, PrecomputeStore, SimulationState, SimulationEngine, OutcomeSampler,
    Order, OrderItem
)
# Algorithm imports removed - now using dynamic module loading
def summarize_delivery_outcomes(results, policy_name):
    if not results:
        return pd.DataFrame()
    rows = []
    for row in results:
        allocations = row.get('allocations') or []
        option_signature_parts = []
        delivered_days = []
        for alloc in allocations:
            dc_id = alloc.get('dc_id')
            carrier_id = alloc.get('carrier_service_id')
            if dc_id is not None and carrier_id is not None:
                option_signature_parts.append(f"dc{dc_id}-cs{carrier_id}")
            delivered_day = alloc.get('delivered_days')
            if delivered_day is not None:
                delivered_days.append(delivered_day)
        option_signature = " | ".join(sorted(option_signature_parts)) if option_signature_parts else 'UNFILLED'
        avg_delivered_days = float(np.mean(delivered_days)) if delivered_days else np.nan
        rows.append({
            'policy': policy_name,
            'order_id': row.get('order_id'),
            'replication': row.get('replication'),
            'realized_cost': row.get('realized_cost'),
            'late_delivery_pct': row.get('late_delivery_pct'),
            'cumulative_lateness': row.get('cumulative_lateness'),
            'lost_sales_quantity': row.get('lost_sales_quantity'),
            'option_signature': option_signature,
            'avg_delivered_days': avg_delivered_days,
            'allocations_json': json.dumps(allocations),
        })
    return pd.DataFrame(rows)


logger = logging.getLogger(__name__)

def _float_to_tag(value: float) -> str:
    """Convert float to filename-safe token."""
    return str(value).replace('-', 'm').replace('.', 'p')


def _sanitize_tag(value: str) -> str:
    """Convert arbitrary text to a filename-safe tag token."""
    s = str(value).strip().lower()
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _normalize_proxy_repair_strategy(strategy: str | None) -> str | None:
    """Normalize repair-strategy aliases to canonical names."""
    if strategy is None:
        return None
    s = str(strategy).strip().lower()
    aliases = {
        "default": "argmax_then_split",
        "argmax_split": "argmax_then_split",
        "argmax_then_split": "argmax_then_split",
    }
    return aliases.get(s, s)

def log_top_options(order_id, costs_series, top_k=10):
    """Log the cheapest and most expensive options for an order when debugging."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if costs_series is None or costs_series.empty:
        logger.debug("Order %s has no costed options to display.", order_id)
        return
    series = costs_series.dropna()
    if series.empty:
        logger.debug("Order %s has costed options but all are NaN.", order_id)
        return
    top_k = min(top_k, len(series))
    top_options = series.nsmallest(top_k)
    bottom_options = series.nlargest(top_k)
    cheap_formatted = []
    for option_id, cost in top_options.items():
        try:
            dc_id, carrier_id = option_id
        except (TypeError, ValueError):
            dc_id, carrier_id = option_id, ''
        cheap_formatted.append(f"{dc_id}-{carrier_id}: ${cost:,.2f}")
    pricey_formatted = []
    for option_id, cost in bottom_options.items():
        try:
            dc_id, carrier_id = option_id
        except (TypeError, ValueError):
            dc_id, carrier_id = option_id, ''
        pricey_formatted.append(f"{dc_id}-{carrier_id}: ${cost:,.2f}")
    logger.debug(
        "Order %s top %d cheapest (dc-carrier : cost): %s",
        order_id,
        top_k,
        "; ".join(cheap_formatted),
    )
    logger.debug(
        "Order %s top %d most expensive (dc-carrier : cost): %s",
        order_id,
        top_k,
        "; ".join(pricey_formatted),
    )


def _compute_orders_hash(order_ids):
    joined = "\n".join(str(oid) for oid in order_ids)
    return hashlib.sha1(joined.encode('utf-8')).hexdigest()

def _resolve_proxy_repair_strategy(model_path: str | None) -> str | None:
    if not model_path:
        return None
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    model_params = checkpoint.get('model_params', {})
    if 'repair_strategy' in model_params:
        return model_params['repair_strategy']
    return checkpoint.get('hyperparams', {}).get('inference', {}).get('repair_strategy')


def _load_checkpoint(checkpoint_path: Path):
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open('rb') as f:
        return pickle.load(f)


def _save_checkpoint(checkpoint_path: Path, payload: dict):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
    with tmp_path.open('wb') as f:
        pickle.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, checkpoint_path)


def _validate_checkpoint_metadata(meta: dict, expected: dict) -> None:
    """
    Validate resume checkpoint metadata with backward compatibility.

    Older checkpoints may not include newly added fields (e.g. gamma settings).
    Those optional keys are skipped when missing.
    """
    optional_metadata_keys = {
        # Cost parameter keys were added after early checkpoints.
        'gamma_plus',
        'gamma_minus',
        'stockout_penalty',
        # Collection/debug and proxy-specific keys can be absent in old checkpoints.
        'collect_only',
        'csaa_debug_dir',
        'csaa_dump_level',
        'proxy_model',
        'proxy_stochastic',
        'proxy_top_k',
        'proxy_repair_strategy',
        'proxy_inventory_weight_power',
        'proxy_run_tag',
        'proxy_scenario_len',
        'saa_n1',
        'saa_q',
        'saa_n2',
    }
    for key, expected_value in expected.items():
        actual_value = meta.get(key, None)
        if key in optional_metadata_keys and (key not in meta or actual_value is None):
            logger.warning(
                "Checkpoint metadata is missing optional key '%s'. "
                "Skipping strict validation for this key.",
                key,
            )
            continue
        if key == 'proxy_repair_strategy':
            if _normalize_proxy_repair_strategy(actual_value) == _normalize_proxy_repair_strategy(expected_value):
                continue
        if actual_value != expected_value:
            raise ValueError(
                f"Checkpoint metadata mismatch for '{key}': expected {expected_value}, found {actual_value}"
            )

def load_simulation_orders(order_set: str) -> pd.DataFrame:
    """Load rich customer/order data for the selected split."""
    preprocessed_path = cfg.PROCESSED_DATA_DIR / 'preprocessed_data_cs.csv'
    print(f"Loading simulation orders from {preprocessed_path}...")
    header_df = pd.read_csv(preprocessed_path, nrows=0)
    csv_columns = header_df.columns
    required_cols = {
        'order_ID', 'order_time', 'order_date', 'sku_ID', 'quantity',
        'promise_delivery_days', 'customer_zip5', 'customer_state',
        'customer_lat', 'customer_lon', 'dc_des'
    }
    columns_to_read = [col for col in csv_columns if col in required_cols]
    date_cols = ['order_time', 'order_date']
    parse_dates = [col for col in date_cols if col in columns_to_read]
    orders_df = pd.read_csv(
        preprocessed_path,
        usecols=columns_to_read if columns_to_read else None,
        parse_dates=parse_dates
    )
    if 'order_time' not in orders_df.columns:
        raise ValueError("preprocessed_data_cs.csv must include 'order_time'")
    core_cols = {'order_ID', 'sku_ID', 'quantity'}
    missing_core = [col for col in core_cols if col not in orders_df.columns]
    if missing_core:
        raise ValueError(f"preprocessed_data_cs.csv must include columns: {missing_core}")
    orders_df['order_time'] = pd.to_datetime(orders_df['order_time'])
    for col in required_cols:
        if col not in orders_df.columns:
            orders_df[col] = np.nan
    if 'order_date' in orders_df.columns:
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    else:
        orders_df['order_date'] = orders_df['order_time'].dt.normalize()
    
    forecast_train_end = pd.to_datetime(cfg.FORECAST_TRAIN_END_DATE)
    proxy_train_end = pd.to_datetime(cfg.PROXY_TRAIN_END_DATE)
    
    if order_set == 'test':
        mask = orders_df['order_date'] > proxy_train_end
    else:
        mask = (orders_df['order_date'] > forecast_train_end) & (orders_df['order_date'] <= proxy_train_end)
    
    filtered = orders_df.loc[mask].copy()
    print(f"  Loaded {len(filtered):,} rows for order_set='{order_set}'")
    return filtered


def load_orders_as_objects(
    order_set: str,
    simulation_date: str,
    peak_only: bool = False,
    hour_range: str = None,
    max_orders: int | None = None,
):
    """Load orders and convert to Order objects."""
    orders_df = load_simulation_orders(order_set)
    sim_date_obj = pd.to_datetime(simulation_date).date()
    date_orders = orders_df[orders_df['order_time'].dt.date == sim_date_obj].sort_values('order_time')
    
    # Filter by hour range
    if hour_range:
        parts = hour_range.split('-')
        h1, h2 = int(parts[0]), int(parts[1]) if len(parts) > 1 else (int(parts[0]) + 1)
        hours = date_orders['order_time'].dt.hour
        date_orders = date_orders[(hours >= h1) & (hours < h2)]
    elif peak_only:
        h1, h2 = cfg.PEAK_START_HOUR, cfg.PEAK_END_HOUR
        hours = date_orders['order_time'].dt.hour
        date_orders = date_orders[(hours >= h1) & (hours <= h2)]

    total_unique_orders = int(date_orders['order_ID'].nunique())

    # Build Order objects in a single pass (avoid per-order dataframe filtering).
    order_objects = []
    for order_id_raw, order_items_df in date_orders.groupby('order_ID', sort=False):
        order_row = order_items_df.iloc[0]
        order_id = str(order_id_raw)
        
        zip_val = order_row.get('customer_zip5')
        if pd.notna(zip_val):
            zip_str = str(zip_val).strip()
            if zip_str.endswith('.0'):
                zip_str = zip_str[:-2]
            dest_zip5 = zip_str.zfill(5) if zip_str.isdigit() else zip_str
        else:
            dest_zip5 = ''
        
        state_val = order_row.get('customer_state')
        dest_state = '' if pd.isna(state_val) else str(state_val).strip().upper()
        
        lat_val = order_row.get('customer_lat')
        dest_lat = float(lat_val) if pd.notna(lat_val) else 0.0
        lng_val = order_row.get('customer_lon')
        dest_lng = float(lng_val) if pd.notna(lng_val) else 0.0
        
        promise_days_val = order_row.get('promise_delivery_days', 0)
        promise_days = int(promise_days_val) if pd.notna(promise_days_val) else 0
        
        customer_dc_val = order_row.get('dc_des')
        if pd.notna(customer_dc_val):
            try:
                customer_dc = int(customer_dc_val)
            except (TypeError, ValueError):
                customer_dc = None
        else:
            customer_dc = None

        items = [
            OrderItem(sku_id=str(row.sku_ID), quantity=int(row.quantity))
            for row in order_items_df[['sku_ID', 'quantity']].itertuples(index=False)
        ]
        
        # Get static features (will be loaded from precompute later)
        static_features = {}
        
        order_obj = Order(
            order_id=order_id,
            dest_zip5=dest_zip5,
            dest_state=dest_state,
            dest_lat=dest_lat,
            dest_lng=dest_lng,
            static_features=static_features,
            items=items,
            promise_delivery_days=promise_days,
            order_time=order_row['order_time'],
            customer_dc=customer_dc,
        )
        order_objects.append(order_obj)

        if max_orders is not None and len(order_objects) >= max_orders:
            break
    
    return order_objects, total_unique_orders


def _add_replication_uncertainty(summary: dict, df: pd.DataFrame) -> None:
    """Attach replication-level std/ste/ci95 for key metrics to summary."""
    if df.empty or 'replication' not in df.columns:
        return

    rep_df = (
        df.groupby('replication', as_index=False)
        .agg(
            realized_cost=('realized_cost', 'mean'),
            lost_sales_qty=('lost_sales_quantity', 'mean'),
            late_delivery_pct=('late_delivery_pct', 'mean'),
            cumulative_lateness=('cumulative_lateness', 'mean'),
        )
    )
    n_rep = int(len(rep_df))
    if n_rep == 0:
        return

    summary['replications_observed'] = n_rep
    metric_cols = {
        'realized_cost': 'realized_cost',
        'lost_sales_qty': 'lost_sales_qty',
        'late_delivery_pct': 'late_delivery_pct',
        'cumulative_lateness': 'cumulative_lateness',
    }

    quantile_levels = (
        (0.10, "p10"),
        (0.25, "p25"),
        (0.50, "p50"),
        (0.75, "p75"),
        (0.90, "p90"),
        (0.95, "p95"),
        (0.99, "p99"),
    )
    for metric_name, col in metric_cols.items():
        values = rep_df[col].to_numpy(dtype=float)
        mean_val = float(np.mean(values))
        if n_rep > 1:
            std_val = float(np.std(values, ddof=1))
            ste_val = float(std_val / np.sqrt(n_rep))
        else:
            std_val = 0.0
            ste_val = 0.0
        ci95_val = float(1.96 * ste_val)
        summary[f'{metric_name}_rep_mean'] = round(mean_val, 6)
        summary[f'{metric_name}_rep_std'] = round(std_val, 6)
        summary[f'{metric_name}_rep_ste'] = round(ste_val, 6)
        summary[f'{metric_name}_rep_ci95'] = round(ci95_val, 6)
        for q, q_label in quantile_levels:
            summary[f'{metric_name}_rep_{q_label}'] = round(float(np.quantile(values, q)), 6)


def _add_evaluation_quantiles(summary: dict, df: pd.DataFrame) -> None:
    """Attach evaluation-level quantiles (across order-replication rows)."""
    if df.empty:
        return

    metric_cols = {
        'realized_cost': 'realized_cost',
        'lost_sales_qty': 'lost_sales_quantity',
        'late_delivery_pct': 'late_delivery_pct',
        'cumulative_lateness': 'cumulative_lateness',
    }
    quantile_levels = (
        (0.10, "p10"),
        (0.25, "p25"),
        (0.50, "p50"),
        (0.75, "p75"),
        (0.90, "p90"),
        (0.95, "p95"),
        (0.99, "p99"),
    )
    for metric_name, col in metric_cols.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        for q, q_label in quantile_levels:
            summary[f'{metric_name}_eval_{q_label}'] = round(float(np.quantile(values, q)), 6)


def _add_policy_bound_summary(summary: dict, order_runtimes: list[dict]) -> None:
    """Attach per-order policy bound aggregates (total +/- CI) when available."""
    if not order_runtimes:
        return
    rt_df = pd.DataFrame(order_runtimes)
    if rt_df.empty:
        return

    def _series(name: str) -> pd.Series:
        if name not in rt_df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(rt_df[name], errors='coerce')

    def _add_total_metrics(metric_prefix: str, summary_prefix=None) -> dict[str, pd.Series]:
        mean = _series(f'{metric_prefix}_mean')
        ci95 = _series(f'{metric_prefix}_ci95')
        valid = mean.notna() & ci95.notna()
        if valid.any():
            vals = mean[valid].to_numpy(dtype=float)
            cis = ci95[valid].to_numpy(dtype=float)
            prefix = summary_prefix or metric_prefix
            summary[f'{prefix}_orders'] = int(valid.sum())
            summary[f'{prefix}_total'] = round(float(np.sum(vals)), 6)
            summary[f'{prefix}_total_ci95'] = round(float(np.sqrt(np.sum(np.square(cis)))), 6)
            summary[f'{prefix}_mean_per_order'] = round(float(np.mean(vals)), 6)
        return {
            'mean': mean,
            'ci95': ci95,
            'valid': valid,
        }

    ub_stats = _add_total_metrics('policy_ub')
    _add_total_metrics('policy_current_stage')
    _add_total_metrics('policy_future_recourse')

    lb_stats = _add_total_metrics('policy_lb')
    _add_total_metrics('policy_lb_current_stage')
    _add_total_metrics('policy_lb_future_recourse')

    ub_mean = ub_stats['mean']
    ub_ci95 = ub_stats['ci95']
    ub_valid = ub_stats['valid']
    lb_mean = lb_stats['mean']
    lb_ci95 = lb_stats['ci95']
    lb_valid = lb_stats['valid']
    paired = ub_valid & lb_valid
    if paired.any():
        delta_vals = (ub_mean[paired] - lb_mean[paired]).to_numpy(dtype=float)
        delta_cis = np.sqrt(
            np.square(ub_ci95[paired].to_numpy(dtype=float))
            + np.square(lb_ci95[paired].to_numpy(dtype=float))
        )
        summary['policy_bound_paired_orders'] = int(paired.sum())
        summary['policy_ub_minus_lb_total'] = round(float(np.sum(delta_vals)), 6)
        summary['policy_ub_minus_lb_total_ci95'] = round(float(np.sqrt(np.sum(np.square(delta_cis)))), 6)
        summary['policy_ub_minus_lb_mean_per_order'] = round(float(np.mean(delta_vals)), 6)

    for metric in ('policy_ub_n_eval', 'policy_lb_n_eval'):
        vals = _series(metric).dropna()
        if vals.empty:
            continue
        summary[f'{metric}_min'] = int(vals.min())
        summary[f'{metric}_max'] = int(vals.max())
        summary[f'{metric}_mean'] = round(float(vals.mean()), 3)


def summarize_results(results, order_runtimes, num_replications):
    if not results:
        return {}
    df = pd.DataFrame(results)
    orders = int(df['order_id'].nunique())
    evaluations = len(df)
    avg_rep = evaluations / orders if orders else 0.0
    summary = {
        'orders_evaluated': orders,
        'evaluations': evaluations,
        'avg_replications_per_order': f"{avg_rep:.1f}",
        'avg_realized_cost': f"${df['realized_cost'].mean():,.2f}",
        'avg_lost_sales_qty': f"{df['lost_sales_quantity'].mean():.2f}",
        'avg_late_delivery_pct': f"{df['late_delivery_pct'].mean():.2f}%",
        'avg_cumulative_lateness': f"{df['cumulative_lateness'].mean():.2f}",
    }
    expected = orders * num_replications if num_replications else None
    if expected:
        coverage = evaluations / expected * 100.0
        summary['replication_coverage'] = f"{coverage:.1f}%"
    _add_replication_uncertainty(summary, df)
    _add_evaluation_quantiles(summary, df)
    _add_policy_bound_summary(summary, order_runtimes)
    if order_runtimes:
        n_orders_rt = len(order_runtimes)
        total_policy = sum(rt['runtime_seconds'] for rt in order_runtimes)
        avg_policy = total_policy / n_orders_rt
        summary['total_policy_runtime_s'] = f"{total_policy:.2f}"
        summary['avg_policy_runtime_ms'] = f"{avg_policy * 1000:.1f}"
        # Optional decision/solve split (policies that re-solve on a cadence, e.g.
        # DTLP, report 'solve_seconds'/'decision_seconds' per order). Emit both
        # totals and order-amortized averages, plus the re-solve rate, so
        # postprocessing scripts can tabulate them directly. Absent for policies
        # that do not provide the split, leaving their summaries unchanged.
        if any('solve_seconds' in rt for rt in order_runtimes):
            total_solve = sum(float(rt.get('solve_seconds', 0.0) or 0.0) for rt in order_runtimes)
            total_decision = sum(float(rt.get('decision_seconds', 0.0) or 0.0) for rt in order_runtimes)
            lp_solves = sum(int(rt.get('lp_solves_this_order', 0) or 0) for rt in order_runtimes)
            summary['total_decision_runtime_s'] = f"{total_decision:.2f}"
            summary['avg_decision_runtime_ms'] = f"{total_decision / n_orders_rt * 1000:.3f}"
            summary['total_solve_runtime_s'] = f"{total_solve:.2f}"
            summary['avg_solve_runtime_ms'] = f"{total_solve / n_orders_rt * 1000:.1f}"
            summary['lp_solves_total'] = lp_solves
            summary['lp_resolve_rate_pct'] = f"{lp_solves / n_orders_rt * 100:.1f}"
    return summary


# Algorithm dispatch table
ALGORITHM_MODULES = {
    'greedy': 'src.algo.greedy',
    'csaa': 'src.algo.contextual_saa',
    'proxy': 'src.algo.proxy',
    'pto': 'src.algo.pto',
    'empirical_saa': 'src.algo.empirical_saa',
    'dtlp_bidprice': 'src.algo.dtlp_bidprice',
    'primal_dual': 'src.algo.primal_dual',
}


def create_policy_wrapper(
    algo: str,
    order_set: str,
    simulation_date: str,
    catalog: OptionsCatalog,
    precompute: PrecomputeStore,
    state: SimulationState,
    csaa_debug_dir: Path | None = None,
    csaa_log_limit: int | None = None,
    csaa_dump_level: str | None = None,
    proxy_model: str | None = None,
    proxy_stochastic: bool | None = None,
    proxy_top_k: int | None = None,
    proxy_eligibility_audit: str | None = None,
    proxy_repair_strategy: str | None = None,
    proxy_inventory_weight_power: float | None = None,
    proxy_scenario_len: int | None = None,
    **algo_kwargs,
):
    """Create policy wrapper using algorithm plugin system."""
    
    if algo not in ALGORITHM_MODULES:
        raise ValueError(f"Unknown algorithm: {algo}. Available: {list(ALGORITHM_MODULES.keys())}")
    
    # Import algorithm module dynamically
    import importlib
    module = importlib.import_module(ALGORITHM_MODULES[algo])
    
    # Build kwargs for policy creation
    kwargs = {
        'catalog': catalog,
        'precompute': precompute,
        'state': state,
        'order_set': order_set,
        'simulation_date': simulation_date,
    }
    
    # Add algorithm-specific kwargs
    if algo == 'csaa':
        kwargs.update({
            'csaa_debug_dir': csaa_debug_dir,
            'csaa_log_limit': csaa_log_limit,
            'csaa_dump_level': csaa_dump_level,
        })
    elif algo == 'proxy':
        if proxy_model is None:
            raise ValueError("--proxy-model required for proxy algorithm")
        kwargs['proxy_model'] = proxy_model
        if proxy_stochastic is not None:
            kwargs['proxy_stochastic'] = proxy_stochastic
        if proxy_top_k is not None:
            kwargs['proxy_top_k'] = proxy_top_k
        if proxy_eligibility_audit is not None:
            kwargs['eligibility_audit_path'] = proxy_eligibility_audit
        if proxy_repair_strategy is not None:
            kwargs['repair_strategy'] = proxy_repair_strategy
        if proxy_inventory_weight_power is not None:
            kwargs['inventory_weight_power'] = proxy_inventory_weight_power
        if proxy_scenario_len is not None:
            kwargs['proxy_scenario_len'] = int(proxy_scenario_len)
    else:
        kwargs.update(algo_kwargs)
    
    # Create and return policy using the algorithm's create_policy_for_simulation function
    return module.create_policy_for_simulation(**kwargs)


def main():
    parser = argparse.ArgumentParser(description='Run unified simulation')
    parser.add_argument(
        '--algo',
        type=str,
        required=True,
        choices=[
            'greedy',
            'csaa',
            'proxy',
            'pto',
            'empirical_saa',
            'dtlp_bidprice',
            'primal_dual',
        ],
    )
    parser.add_argument('--order_set', type=str, default='test', choices=['test', 'proxy_train'])
    parser.add_argument('--simulation_date', type=str, required=True)
    parser.add_argument('--num_replications', type=int, default=50)
    parser.add_argument('--saa-n1', type=int, default=None, help='Override cfg.SAA_N1 for this run.')
    parser.add_argument('--saa-q', type=int, default=None, help='Override cfg.SAA_Q for this run.')
    parser.add_argument('--saa-n2', type=int, default=None, help='Override cfg.SAA_N2 for this run.')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--simulator_type', type=str, default='dl', choices=['dl', 'catboost'])
    parser.add_argument('--peak-only', action='store_true')
    parser.add_argument('--hour-range', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=cfg.RANDOM_SEED)
    parser.add_argument('--gamma-plus', type=float, default=None, help='Override late-delivery penalty per day.')
    parser.add_argument('--gamma-minus', type=float, default=None, help='Override early-delivery penalty per day.')
    parser.add_argument('--stockout-penalty', type=float, default=None, help='Override stockout penalty per unit.')
    parser.add_argument('--max-orders', type=int, default=None, help='Limit to the first N orders for quick inspection')
    parser.add_argument('--no-progress', action='store_true', help='Disable per-order progress updates')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--csaa-debug-dir', type=str, default=None, help='Optional directory to store CSAA scenario dumps and delivery outcomes (organized by order_set/simulation_date).')
    parser.add_argument('--csaa-log-limit', type=int, default=None, help='Maximum number of orders to dump CSAA scenarios for (default: all orders).')
    parser.add_argument(
        '--csaa-dump-level',
        type=str,
        choices=['full', 'stats'],
        default='full',
        help="When dumping CSAA debug artifacts, 'full' writes scenarios/options/etc (used for proxy data collection), "
             "while 'stats' writes only compact per-order bounds/timings (recommended for test reporting).",
    )
    parser.add_argument('--collect-only', action='store_true', help='Data collection mode: skip outcome simulation and result aggregation (only update state and save policy decisions).')
    parser.add_argument('--proxy-model', type=str, default=None, help='Path to trained proxy model checkpoint (required for --algo proxy)')
    parser.add_argument('--proxy-stochastic', action='store_true', help='Enable stochastic top-k sampling for proxy (default: deterministic argmax)')
    parser.add_argument('--proxy-top-k', type=int, default=5, help='Number of top candidates for proxy stochastic sampling (default: 5)')
    parser.add_argument(
        '--proxy-scenario-len',
        type=int,
        default=None,
        help=(
            'Override number of generated scenarios used by proxy inference. '
            'If omitted and --algo proxy with --saa-n1 provided, defaults to saa_n1 * saa_q.'
        ),
    )
    parser.add_argument('--proxy-repair-strategy', type=str, default=None,
                        choices=['argmax_then_split', 'default', 'inventory_first', 'feasible_topk', 'inventory_weighted', 'feasible_joint_topk'],
                        help='Override proxy repair strategy (defaults to checkpoint/config)')
    parser.add_argument(
        '--proxy-inventory-weight-power',
        type=float,
        default=None,
        help='Power parameter for inventory_weighted repair strategy (1.0 = linear weighting).',
    )
    parser.add_argument(
        '--proxy-run-tag',
        type=str,
        default=None,
        help='Optional extra tag appended to proxy output filenames (useful for inference-strategy ablations).',
    )
    parser.add_argument('--proxy-eligibility-audit', type=str, default=None,
                        help='Write JSONL audit of proxy eligibility vs model dcs/carriers')
    parser.add_argument('--precompute-dir', type=str, default=None, help='Override simulation precompute directory (useful when shared data/derived/sim is read-only).')
    # ---- DTLP / Primal-dual knobs (optional) ----
    parser.add_argument('--dtlp-cadence', type=str, default='per_order', choices=['per_order', 'inventory_buckets', 'every_n_orders'], help='DTLP dual refresh policy')
    parser.add_argument('--dtlp-q', type=int, default=100, help='DTLP inventory bucket size q (inventory_buckets cadence)')
    parser.add_argument('--dtlp-every-n-orders', type=int, default=1, help='DTLP re-solve every N orders (every_n_orders cadence)')
    parser.add_argument('--dtlp-tau-days', type=float, default=1.0, help='DTLP lookahead horizon τ (days)')

    parser.add_argument('--pd-auto-params', dest='pd_auto_params', action='store_true', default=True, help='Primal-dual: derive params from kappa (default)')
    parser.add_argument('--no-pd-auto-params', dest='pd_auto_params', action='store_false', help='Primal-dual: use manual alpha/beta params')
    parser.add_argument('--pd-kappa', type=float, default=2.0, help='Primal-dual: kappa (>1) for auto params')
    parser.add_argument('--pd-alpha1', type=float, default=None)
    parser.add_argument('--pd-alpha2', type=float, default=None)
    parser.add_argument('--pd-beta', type=float, default=None)
    parser.add_argument('--pd-allow-null', dest='pd_allow_null', action='store_true', default=True, help='Primal-dual: enable null option')
    parser.add_argument('--no-pd-allow-null', dest='pd_allow_null', action='store_false', help='Primal-dual: disable null option')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available.')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to checkpoint file (default: output_dir/algo_checkpoint.pkl).')
    parser.add_argument('--checkpoint-every', type=int, default=1, help='Save checkpoint every N processed orders (0 disables checkpoints).')
    args = parser.parse_args()

    auto_resume = args.max_orders is None
    resume_enabled = args.resume or auto_resume
    if auto_resume and not args.resume:
        print("Auto-enabling resume/checkpoint because --max-orders was not specified.")
    
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')

    effective_gamma_plus = float(cfg.GAMMA_PLUS_LATE_PENALTY)
    effective_gamma_minus = float(cfg.GAMMA_MINUS_EARLY_PENALTY)
    effective_stockout_penalty = float(cfg.STOCKOUT_PENALTY_PER_UNIT)
    effective_saa_n1 = int(cfg.SAA_N1)
    effective_saa_q = int(cfg.SAA_Q)
    effective_saa_n2 = int(cfg.SAA_N2)
    if args.gamma_plus is not None:
        effective_gamma_plus = float(args.gamma_plus)
        cfg.GAMMA_PLUS_LATE_PENALTY = effective_gamma_plus
    if args.gamma_minus is not None:
        effective_gamma_minus = float(args.gamma_minus)
        cfg.GAMMA_MINUS_EARLY_PENALTY = effective_gamma_minus
    if args.stockout_penalty is not None:
        effective_stockout_penalty = float(args.stockout_penalty)
        cfg.STOCKOUT_PENALTY_PER_UNIT = effective_stockout_penalty
    if args.saa_n1 is not None:
        if int(args.saa_n1) <= 0:
            raise ValueError("--saa-n1 must be positive")
        effective_saa_n1 = int(args.saa_n1)
        cfg.SAA_N1 = effective_saa_n1
    if args.saa_q is not None:
        if int(args.saa_q) <= 0:
            raise ValueError("--saa-q must be positive")
        effective_saa_q = int(args.saa_q)
        cfg.SAA_Q = effective_saa_q
    if args.saa_n2 is not None:
        if int(args.saa_n2) < 0:
            raise ValueError("--saa-n2 must be non-negative")
        effective_saa_n2 = int(args.saa_n2)
        cfg.SAA_N2 = effective_saa_n2
    effective_proxy_scenario_len = args.proxy_scenario_len
    if (
        effective_proxy_scenario_len is None
        and args.algo == 'proxy'
        and args.saa_n1 is not None
    ):
        # Scenario-sensitivity convenience: proxy uses candidate pool size S = N1 * Q.
        effective_proxy_scenario_len = int(effective_saa_n1 * effective_saa_q)
    if effective_proxy_scenario_len is not None:
        if int(effective_proxy_scenario_len) <= 0:
            raise ValueError("--proxy-scenario-len must be positive")
        effective_proxy_scenario_len = int(effective_proxy_scenario_len)
    print(
        f"Using penalties: gamma_plus={effective_gamma_plus:.4f}, "
        f"gamma_minus={effective_gamma_minus:.4f}, "
        f"stockout_penalty={effective_stockout_penalty:.4f}"
    )
    print(
        f"Using SAA settings: N1={effective_saa_n1}, Q={effective_saa_q}, N2={effective_saa_n2}"
    )
    if effective_proxy_scenario_len is not None:
        print(f"Using proxy scenario length override: {effective_proxy_scenario_len}")

    csaa_debug_dir = None
    if args.csaa_debug_dir:
        csaa_debug_dir = Path(args.csaa_debug_dir) / args.order_set / args.simulation_date
        csaa_debug_dir.mkdir(parents=True, exist_ok=True)
    elif args.algo == 'csaa' and args.collect_only:
        # Auto-infer csaa debug dir for data collection
        base_csaa_dir = Path('data/peak/csaa_solutions') if args.peak_only else Path('data/csaa_solutions')
        csaa_debug_dir = base_csaa_dir / args.order_set / args.simulation_date
        csaa_debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Auto-inferred CSAA debug dir: {csaa_debug_dir}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_dir = Path('data/peak/simulation_results') if args.peak_only else Path('data/simulation_results')
        output_dir = base_dir / args.order_set / args.simulation_date / 'solutions_eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct filename base for consistent naming across outputs (checkpoints, JSON, parquet)
    run_tags = []
    if args.peak_only:
        run_tags.append('peak')
    if args.hour_range:
        sanitized_range = args.hour_range.replace(':', '')
        run_tags.append(f"hrs{sanitized_range.replace('-', 'to')}")
    if args.max_orders is not None:
        run_tags.append(f"max{args.max_orders}")
    if args.gamma_plus is not None:
        run_tags.append(f"gp{_float_to_tag(effective_gamma_plus)}")
    if args.gamma_minus is not None:
        run_tags.append(f"gm{_float_to_tag(effective_gamma_minus)}")
    if args.stockout_penalty is not None:
        run_tags.append(f"sp{_float_to_tag(effective_stockout_penalty)}")
    if args.saa_n1 is not None:
        run_tags.append(f"n1{effective_saa_n1}")
    if args.saa_q is not None:
        run_tags.append(f"q{effective_saa_q}")
    if args.saa_n2 is not None:
        run_tags.append(f"n2{effective_saa_n2}")
    if effective_proxy_scenario_len is not None:
        run_tags.append(f"ps{effective_proxy_scenario_len}")
    
    tag_suffix = f"_{'_'.join(run_tags)}" if run_tags else ''
    proxy_repair_strategy = None
    proxy_inventory_weight_power = None
    proxy_run_tag = _sanitize_tag(args.proxy_run_tag) if args.proxy_run_tag else None
    if args.algo == 'proxy' and args.proxy_model:
        # Use model dir name (e.g. "my_model") not checkpoint stem ("best")
        proxy_model_name = Path(args.proxy_model).parent.name or Path(args.proxy_model).stem
        proxy_repair_strategy = (
            _normalize_proxy_repair_strategy(args.proxy_repair_strategy)
            or _normalize_proxy_repair_strategy(_resolve_proxy_repair_strategy(args.proxy_model))
            or _normalize_proxy_repair_strategy(cfg.PROXY_MODEL_REPAIR_STRATEGY)
        )
        proxy_inventory_weight_power = (
            float(args.proxy_inventory_weight_power)
            if args.proxy_inventory_weight_power is not None
            else None
        )
        repair_tag = f"rs_{proxy_repair_strategy}" if proxy_repair_strategy else "rs_unknown"
        tags = []
        if args.proxy_stochastic:
            tags.append("stochastic")
            if args.proxy_top_k:
                tags.append(f"k{args.proxy_top_k}")
        tags.append(repair_tag)
        if proxy_run_tag:
            tags.append(f"tag_{proxy_run_tag}")
        tags.append(proxy_model_name)
        filename_base = f"{args.algo}_{'_'.join(tags)}{tag_suffix}"
    else:
        filename_base = f"{args.algo}{tag_suffix}"
    
    checkpoint_every = max(0, int(args.checkpoint_every))
    if resume_enabled and checkpoint_every == 0:
        checkpoint_every = 1
    checkpoint_path = None
    if resume_enabled:
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else (output_dir / f"{filename_base}_checkpoint.pkl")
    
    print(f"Loading orders for {args.simulation_date}...")
    order_objects, total_orders_available = load_orders_as_objects(
        args.order_set,
        args.simulation_date,
        args.peak_only,
        args.hour_range,
        max_orders=args.max_orders,
    )
    # For full runs, avoid expensive per-order checkpoint writes unless explicitly tuned.
    # Keep user-specified values untouched; only adapt the implicit default (=1).
    if resume_enabled and args.max_orders is None and int(args.checkpoint_every) == 1:
        # Target ~50 checkpoints per run, bounded for practicality.
        checkpoint_every = max(25, min(200, max(1, len(order_objects) // 50)))
        print(f"Auto checkpoint frequency: every {checkpoint_every} orders (full-run mode).")
    if args.max_orders is not None:
        if args.max_orders <= 0:
            raise ValueError("--max-orders must be positive")
        print(f"  Loaded {total_orders_available} orders (limiting to first {len(order_objects)})")
    else:
        print(f"  Loaded {len(order_objects)} orders")

    order_ids = [str(getattr(order, 'order_id', '')) for order in order_objects]
    orders_hash = _compute_orders_hash(order_ids)
    checkpoint_metadata = {
        'algo': args.algo,
        'order_set': args.order_set,
        'simulation_date': args.simulation_date,
        'order_count': len(order_objects),
        'orders_hash': orders_hash,
        'num_replications': args.num_replications,
        'simulator_type': args.simulator_type,
        'peak_only': bool(args.peak_only),
        'hour_range': args.hour_range,
        'max_orders': args.max_orders,
        'gamma_plus': effective_gamma_plus,
        'gamma_minus': effective_gamma_minus,
        'stockout_penalty': effective_stockout_penalty,
        'saa_n1': effective_saa_n1,
        'saa_q': effective_saa_q,
        'saa_n2': effective_saa_n2,
        'collect_only': bool(args.collect_only),
        'csaa_debug_dir': str(csaa_debug_dir.resolve()) if csaa_debug_dir is not None else None,
        'csaa_dump_level': args.csaa_dump_level if args.algo == 'csaa' else None,
        'proxy_run_tag': proxy_run_tag if args.algo == 'proxy' else None,
        'proxy_inventory_weight_power': proxy_inventory_weight_power if args.algo == 'proxy' else None,
        'proxy_scenario_len': effective_proxy_scenario_len if args.algo == 'proxy' else None,
    }
    resume_data = None
    start_index = 0
    resume_results = None
    resume_order_runtimes = None
    rng_state = None
    state_override = None
    if resume_enabled and checkpoint_path is not None:
        if checkpoint_path.exists():
            resume_data = _load_checkpoint(checkpoint_path)
        if resume_data is None:
            print(f"No checkpoint found at {checkpoint_path}; starting fresh.")
        else:
            meta = resume_data.get('metadata', {})
            _validate_checkpoint_metadata(meta, checkpoint_metadata)
            state_override = resume_data.get('state')
            if state_override is None:
                raise ValueError("Checkpoint is missing simulation state data.")
            start_index = int(resume_data.get('next_order_idx', 0))
            resume_results = resume_data.get('results') or []
            resume_order_runtimes = resume_data.get('order_runtimes') or []
            rng_state = resume_data.get('rng_state')
            if args.algo == 'csaa' and args.collect_only and csaa_debug_dir is not None and args.csaa_log_limit is None:
                dumped_dirs = sum(1 for p in csaa_debug_dir.iterdir() if p.is_dir()) if csaa_debug_dir.exists() else 0
                if dumped_dirs < start_index:
                    raise ValueError(
                        "Resume checkpoint and CSAA dump directory are inconsistent: "
                        f"checkpoint start_index={start_index}, but dumped order folders={dumped_dirs}. "
                        "This usually means a prior run used a different dump mode/path. "
                        "Delete checkpoint and rerun this date from scratch for complete collection."
                    )
                if dumped_dirs > start_index:
                    logger.warning(
                        "CSAA dump folder count (%d) exceeds checkpoint start_index (%d). "
                        "Proceeding with resume, but verify dump consistency for this date.",
                        dumped_dirs,
                        start_index,
                    )
            print(
                f"Resuming from checkpoint {checkpoint_path} (next order index {start_index})."
            )
    else:
        if checkpoint_path is not None and checkpoint_path.exists():
            print(
                f"Warning: checkpoint file {checkpoint_path} exists but resume is disabled; it will be overwritten."
            )
    
    print("Initializing simulation components...")
    state: SimulationState
    if state_override is None:
        # Load initial inventory
        initial_inventory_df, _ = get_initial_inventory(args.simulation_date)

        # Initialize dynamic features from historical data
        print("  Initializing dynamic features from previous date...")
        from src.simulator.state import initialize_dynamic_features_from_history

        snapshot_hour = None
        if args.peak_only:
            snapshot_hour = cfg.PEAK_START_HOUR
        elif args.hour_range:
            try:
                snapshot_hour = int(args.hour_range.split('-')[0])
            except (ValueError, AttributeError):
                snapshot_hour = None

        historical_state = initialize_dynamic_features_from_history(
            args.simulation_date,
            snapshot_hour=snapshot_hour,
        )

        state = SimulationState(initial_inventory_df, initial_dynamic_features=historical_state)
    else:
        state = state_override
        print("  Loaded simulation state from checkpoint.")

    # Initialize components
    precompute_dir = Path(args.precompute_dir) if args.precompute_dir else None
    if precompute_dir is not None:
        logger.info("Using precompute_dir=%s", precompute_dir)
    precompute = PrecomputeStore(precompute_dir=precompute_dir)

    # Populate static features for orders
    for order_obj in order_objects:
        static_feat = precompute.get_static_features(order_obj.order_id)
        if static_feat:
            order_obj.static_features = static_feat

    catalog = OptionsCatalog(precompute_store=precompute)
    if logger.isEnabledFor(logging.DEBUG):
        baseline_dc = 0
        if getattr(state, '_baseline_dynamic_features', None) is not None:
            baseline_dc = len(state._baseline_dynamic_features)
        logger.debug(
            "Simulation state initialized with baseline for %d DCs.",
            baseline_dc,
        )
    sampler = OutcomeSampler(
        simulator_type=args.simulator_type,
        scenario_source='simulator',
    )
    
    # Set proxy model path if using proxy algorithm
    # Prepare proxy model path if needed
    proxy_model_path = None
    if args.algo == 'proxy':
        if args.proxy_model is None:
            raise ValueError("--proxy-model is required when using --algo proxy")
        proxy_model_path = args.proxy_model
        proxy_stochastic = args.proxy_stochastic
        proxy_top_k = args.proxy_top_k
        proxy_repair_strategy = (
            _normalize_proxy_repair_strategy(args.proxy_repair_strategy)
            or _normalize_proxy_repair_strategy(proxy_repair_strategy)
            or _normalize_proxy_repair_strategy(_resolve_proxy_repair_strategy(proxy_model_path))
            or _normalize_proxy_repair_strategy(cfg.PROXY_MODEL_REPAIR_STRATEGY)
        )
        logger.info(f"Using proxy model: {proxy_model_path} (stochastic={proxy_stochastic}, top_k={proxy_top_k})")
        checkpoint_metadata['proxy_model'] = proxy_model_path
        checkpoint_metadata['proxy_stochastic'] = proxy_stochastic
        checkpoint_metadata['proxy_top_k'] = proxy_top_k
        checkpoint_metadata['proxy_repair_strategy'] = proxy_repair_strategy
        checkpoint_metadata['proxy_inventory_weight_power'] = proxy_inventory_weight_power
        checkpoint_metadata['proxy_run_tag'] = proxy_run_tag
        checkpoint_metadata['proxy_scenario_len'] = effective_proxy_scenario_len
    
    policy = create_policy_wrapper(
        args.algo,
        args.order_set,
        args.simulation_date,
        catalog,
        precompute,
        state,
        csaa_debug_dir=csaa_debug_dir,
        csaa_log_limit=args.csaa_log_limit,
        csaa_dump_level=args.csaa_dump_level,
        proxy_model=proxy_model_path,
        proxy_stochastic=args.proxy_stochastic if args.algo == 'proxy' else None,
        proxy_top_k=args.proxy_top_k if args.algo == 'proxy' else None,
        proxy_repair_strategy=proxy_repair_strategy if args.algo == 'proxy' else None,
        proxy_inventory_weight_power=proxy_inventory_weight_power if args.algo == 'proxy' else None,
        proxy_scenario_len=effective_proxy_scenario_len if args.algo == 'proxy' else None,
        proxy_eligibility_audit=args.proxy_eligibility_audit if args.algo == 'proxy' else None,
        **(
            {
                'dtlp_cadence': args.dtlp_cadence,
                'dtlp_q': int(args.dtlp_q),
                'dtlp_every_n_orders': int(args.dtlp_every_n_orders),
                'tau_days': float(args.dtlp_tau_days),
            }
            if args.algo == 'dtlp_bidprice'
            else {}
        ),
        **(
            {
                'pd_auto_params': bool(args.pd_auto_params),
                'pd_kappa': float(args.pd_kappa),
                'pd_alpha1': args.pd_alpha1,
                'pd_alpha2': args.pd_alpha2,
                'pd_beta': args.pd_beta,
                'pd_allow_null': bool(args.pd_allow_null),
            }
            if args.algo == 'primal_dual'
            else {}
        ),
    )
    
    engine = SimulationEngine(catalog, precompute, state, sampler, policy)
    checkpoint_callback = None
    if resume_enabled and checkpoint_every > 0 and checkpoint_path is not None:
        total_orders = len(order_objects)

        def checkpoint_callback(processed_idx: int, engine_instance: SimulationEngine):
            if processed_idx % checkpoint_every != 0 and processed_idx != total_orders:
                return
            payload = {
                'metadata': checkpoint_metadata,
                'state': engine_instance.state,
                'results': engine_instance.results,
                'order_runtimes': engine_instance.order_runtimes,
                'next_order_idx': processed_idx,
                'rng_state': engine_instance.rng.bit_generator.state,
            }
            _save_checkpoint(checkpoint_path, payload)
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "Checkpoint saved at order %d/%d -> %s",
                    processed_idx,
                    total_orders,
                    checkpoint_path,
                )
    
    if args.collect_only:
        print("Running in data collection mode (no outcome simulation)...")
    else:
        print(f"Running simulation with {args.num_replications} replications...")
    
    results = engine.run(
        order_objects,
        num_replications=args.num_replications,
        rng_seed=args.random_seed,
        show_progress=not args.no_progress,
        start_index=start_index,
        resume_results=resume_results,
        resume_order_runtimes=resume_order_runtimes,
        rng_state=rng_state,
        checkpoint_callback=checkpoint_callback,
        collect_only=args.collect_only,
    )
    processed_this_run = int(getattr(engine, 'orders_processed_this_run', max(0, len(order_objects) - start_index)))
    no_eligible_this_run = int(getattr(engine, 'orders_no_eligible_this_run', 0))
    
    # Get runtime information from engine
    order_runtimes = engine.order_runtimes
    total_runtime_seconds = sum(rt['runtime_seconds'] for rt in order_runtimes)
    run_runtime_seconds = sum(rt['runtime_seconds'] for rt in order_runtimes[-processed_this_run:]) if processed_this_run > 0 else 0.0

    if (
        resume_enabled
        and checkpoint_path is not None
        and len(order_runtimes) >= len(order_objects)
        and checkpoint_path.exists()
    ):
        try:
            checkpoint_path.unlink()
            if logger.isEnabledFor(logging.INFO):
                logger.info("Completed all orders; removed checkpoint %s", checkpoint_path)
        except OSError:
            logger.warning("Unable to delete checkpoint file %s", checkpoint_path)
    
    if args.collect_only:
        print(f"\nData collection complete!")
        print(
            f"Processed this run: {processed_this_run} orders "
            f"(start_index={start_index}, loaded_total={len(order_objects)})"
        )
        print(f"Orders with no eligible options this run: {no_eligible_this_run}")
        print(f"Policy runtime this run: {run_runtime_seconds:.2f} seconds")
        print(f"Total policy runtime: {total_runtime_seconds:.2f} seconds")
        if csaa_debug_dir is not None:
            print(f"CSAA solutions saved to {csaa_debug_dir}")
            try:
                dumped_dirs = sum(1 for p in csaa_debug_dir.iterdir() if p.is_dir())
                print(f"CSAA dumped order folders (current total): {dumped_dirs}")
                if args.algo == 'csaa' and dumped_dirs < len(order_objects):
                    print(
                        "NOTE: dumped folders < loaded orders. This can happen with resume runs "
                        "(partial processing in current job) and/or orders with no eligible options."
                    )
            except Exception as exc:
                print(f"WARNING: failed to count CSAA dump folders under {csaa_debug_dir}: {exc}")
    else:
        print(f"Simulation complete. Saving results...")
        
        # Write results and runtimes to parquet
        try:
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                # Convert dict/struct columns to JSON strings for parquet compatibility
                import json
                if 'unfilled' in results_df.columns:
                    results_df['unfilled'] = results_df['unfilled'].apply(
                        lambda x: json.dumps(x) if isinstance(x, dict) else x
                    )
                if 'allocations' in results_df.columns:
                    results_df['allocations'] = results_df['allocations'].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )
                
                results_df['policy'] = args.algo
                results_df['proxy_model'] = proxy_model_path if proxy_model_path else None
                if args.algo == 'proxy':
                    results_df['proxy_stochastic'] = args.proxy_stochastic
                    results_df['proxy_top_k'] = args.proxy_top_k if args.proxy_stochastic else None
                    results_df['proxy_repair_strategy'] = proxy_repair_strategy
                    if proxy_inventory_weight_power is not None:
                        results_df['proxy_inventory_weight_power'] = proxy_inventory_weight_power
                    results_df['proxy_run_tag'] = proxy_run_tag
                results_df['order_set'] = args.order_set
                results_df['simulation_date'] = args.simulation_date
                results_df['gamma_plus'] = effective_gamma_plus
                results_df['gamma_minus'] = effective_gamma_minus
                results_df['stockout_penalty'] = effective_stockout_penalty
                results_df['saa_n1'] = effective_saa_n1
                results_df['saa_q'] = effective_saa_q
                results_df['saa_n2'] = effective_saa_n2
                if args.algo == 'proxy':
                    results_df['proxy_scenario_len'] = effective_proxy_scenario_len
            
            runtimes_df = pd.DataFrame(order_runtimes)
            if not runtimes_df.empty:
                runtimes_df['policy'] = args.algo
                runtimes_df['proxy_model'] = proxy_model_path if proxy_model_path else None
                if args.algo == 'proxy':
                    runtimes_df['proxy_stochastic'] = args.proxy_stochastic
                    runtimes_df['proxy_top_k'] = args.proxy_top_k if args.proxy_stochastic else None
                    runtimes_df['proxy_repair_strategy'] = proxy_repair_strategy
                    if proxy_inventory_weight_power is not None:
                        runtimes_df['proxy_inventory_weight_power'] = proxy_inventory_weight_power
                    runtimes_df['proxy_run_tag'] = proxy_run_tag
                runtimes_df['order_set'] = args.order_set
                runtimes_df['simulation_date'] = args.simulation_date
                runtimes_df['total_runtime_seconds'] = total_runtime_seconds
                runtimes_df['gamma_plus'] = effective_gamma_plus
                runtimes_df['gamma_minus'] = effective_gamma_minus
                runtimes_df['stockout_penalty'] = effective_stockout_penalty
                runtimes_df['saa_n1'] = effective_saa_n1
                runtimes_df['saa_q'] = effective_saa_q
                runtimes_df['saa_n2'] = effective_saa_n2
                if args.algo == 'proxy':
                    runtimes_df['proxy_scenario_len'] = effective_proxy_scenario_len

            parquet_path = output_dir / f"{filename_base}.parquet"
            runtimes_path = output_dir / f"{filename_base}_runtimes.parquet"
            
            results_df.to_parquet(parquet_path, index=False, compression='snappy')
            runtimes_df.to_parquet(runtimes_path, index=False, compression='snappy')
            
            print(f"Saved results to {parquet_path} ({len(results)} rows)")
            print(f"Saved runtimes to {runtimes_path} ({len(order_runtimes)} rows)")
        except Exception as exc:
            print("Failed to write parquet outputs. Ensure pyarrow or fastparquet is installed.")
            raise

        if csaa_debug_dir is not None:
            delivery_outcomes_df = summarize_delivery_outcomes(results, args.algo)
            if not delivery_outcomes_df.empty:
                delivery_path = csaa_debug_dir / f"{args.algo}_delivery_outcomes.csv"
                delivery_outcomes_df.to_csv(delivery_path, index=False)
                print(f"Delivery outcomes saved to {delivery_path}")

        summary = summarize_results(results, order_runtimes, args.num_replications)
        if summary:
            summary['gamma_plus'] = effective_gamma_plus
            summary['gamma_minus'] = effective_gamma_minus
            summary['stockout_penalty'] = effective_stockout_penalty
            summary['saa_n1'] = effective_saa_n1
            summary['saa_q'] = effective_saa_q
            summary['saa_n2'] = effective_saa_n2
            if args.algo == 'proxy' and proxy_model_path:
                summary['proxy_model'] = str(proxy_model_path)
                summary['proxy_model_name'] = Path(proxy_model_path).parent.name or Path(proxy_model_path).stem
                summary['proxy_repair_strategy'] = proxy_repair_strategy
                if proxy_inventory_weight_power is not None:
                    summary['proxy_inventory_weight_power'] = proxy_inventory_weight_power
                if effective_proxy_scenario_len is not None:
                    summary['proxy_scenario_len'] = effective_proxy_scenario_len
                summary['proxy_run_tag'] = proxy_run_tag
            summary_path = output_dir / f"{filename_base}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                f.write('\n')
            print(f"Summary metrics saved to {summary_path}")
            print("\nSummary metrics:")
            for label, value in summary.items():
                print(f"  {label}: {value}")


if __name__ == '__main__':
    main()
