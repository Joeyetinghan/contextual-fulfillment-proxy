from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from functools import lru_cache
import json

import src.config as cfg


def load_carrier_delivery_samples(order_set: str) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Load delivery-time samples grouped by carrier_service_id_anon from preprocessed_data_cs.
    
    Returns:
        - dict mapping carrier_service_id_anon -> np.ndarray of delivery_time_days
        - np.ndarray of all delivery_time_days across carriers (fallback)
    """
    preprocessed_path = cfg.PROCESSED_DATA_DIR / 'preprocessed_data_cs.csv'
    header = pd.read_csv(preprocessed_path, nrows=0)
    required_cols = {'delivery_time_days', 'order_time', 'carrier_service_id_anon'}
    optional_cols = {'order_date'}
    missing = required_cols - set(header.columns)
    if missing:
        raise ValueError(
            f"preprocessed_data_cs.csv must include columns: {sorted(missing)}"
        )

    cols_to_read = sorted(required_cols | (optional_cols & set(header.columns)))
    date_cols = [col for col in ['order_time', 'order_date'] if col in cols_to_read]
    df = pd.read_csv(
        preprocessed_path,
        usecols=cols_to_read,
        parse_dates=date_cols,
    )

    df['order_time'] = pd.to_datetime(df['order_time'])
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
    else:
        df['order_date'] = df['order_time'].dt.normalize()

    forecast_train_end = pd.to_datetime(cfg.FORECAST_TRAIN_END_DATE)
    proxy_train_end = pd.to_datetime(cfg.PROXY_TRAIN_END_DATE)
    if order_set == 'test':
        mask = df['order_date'] > proxy_train_end
    else:
        mask = (df['order_date'] > forecast_train_end) & (df['order_date'] <= proxy_train_end)

    filtered = df.loc[mask].copy()
    filtered['carrier_service_id_anon'] = pd.to_numeric(
        filtered['carrier_service_id_anon'], errors='coerce'
    )

    per_carrier: dict[int, np.ndarray] = {}
    for carrier_id, grp in filtered.groupby('carrier_service_id_anon'):
        if pd.isna(carrier_id):
            continue
        arr = grp['delivery_time_days'].dropna().to_numpy(dtype=float)
        if arr.size > 0:
            per_carrier[int(carrier_id)] = arr

    global_arr = filtered['delivery_time_days'].dropna().to_numpy(dtype=float)
    if global_arr.size == 0:
        global_arr = np.zeros(1, dtype=float)

    return per_carrier, global_arr


def _load_npz(path: Path) -> dict:
    """Load NPZ file and return as dict."""
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


# Global cache for training data
_training_data_cache = {}


def _get_cached_training_data(order_set: str, verbose: bool = False):
    """
    Load and cache training data for empirical scenario generation.

    This function caches the expensive data loading operations so they only
    happen once per simulation run, not once per order.
    """
    cache_key = f"training_data_{order_set}"

    if cache_key in _training_data_cache:
        return _training_data_cache[cache_key]

    if verbose:
        print(f"Loading training data for empirical scenarios (order_set: {order_set})...")

    # Load delivery time training data (carrier agnostic, global draws)
    if order_set == 'proxy_train':
        df_forecast = pd.read_csv(cfg.DELIVERY_FORECAST_TRAIN_PATH)
        delivery_df = df_forecast
    else:
        df_forecast = pd.read_csv(cfg.DELIVERY_FORECAST_TRAIN_PATH)
        df_proxy = pd.read_csv(cfg.DELIVERY_PROXY_TRAIN_PATH)
        delivery_df = pd.concat([df_forecast, df_proxy])

    delivery_df.columns = [c.strip() for c in delivery_df.columns]
    delivery_times = delivery_df[cfg.DELIVERY_TIME_TARGET].dropna().values
    if delivery_times.size == 0:
        delivery_times = np.zeros(1, dtype=float)

    # Load demand training data
    if order_set == 'proxy_train':
        # For proxy-train simulations, only use the *forecast-train* history to avoid
        # leaking information from the proxy-train period into its own scenarios.
        forecast_data = _load_npz(cfg.PROCESSED_DATA_DIR / "mqrnn_forecast_train.npz")
        demand_samples = forecast_data['yp']
        sku_indices = forecast_data['sku_idx']
    else:
        # For test simulations, use both forecast-train and proxy-train histories.
        forecast_data = _load_npz(cfg.PROCESSED_DATA_DIR / "mqrnn_forecast_train.npz")
        proxy_data = _load_npz(cfg.PROCESSED_DATA_DIR / "mqrnn_proxy_train.npz")
        demand_samples = np.concatenate([forecast_data['yp'], proxy_data['yp']], axis=0)
        sku_indices = np.concatenate([forecast_data['sku_idx'], proxy_data['sku_idx']], axis=0)

    cached_data = {
        'delivery_times': delivery_times,
        'demand_samples': demand_samples,
        'sku_indices': sku_indices,
        'num_delivery_samples': len(delivery_times),
        'num_demand_samples': demand_samples.shape[0],
        'demand_horizon': demand_samples.shape[1]
    }

    _training_data_cache[cache_key] = cached_data

    if verbose:
        print(f"Cached training data: {cached_data['num_delivery_samples']} delivery samples, "
              f"{cached_data['num_demand_samples']} demand samples")

    return cached_data


@lru_cache(maxsize=1)
def _sku_to_idx_mapping() -> dict[str, int]:
    """Load sku_ID -> sku_idx mapping used in the demand NPZs."""
    mapping_path = cfg.PROCESSED_DATA_DIR / "mappings" / "sku_to_idx.json"
    try:
        raw = json.loads(Path(mapping_path).read_text())
    except Exception:
        return {}
    # Normalize to int indices
    out: dict[str, int] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def clear_training_data_cache():
    """
    Clear the cached training data.

    This can be useful for testing or memory management.
    """
    global _training_data_cache
    _training_data_cache.clear()


def generate_empirical_scenarios(
    order_items: pd.DataFrame,
    order_options: pd.DataFrame,
    promise_days: float,
    num_scenarios: int,
    order_set: str,
    lookahead_periods: int,
    seed: int | None = None,
    verbose: bool = False,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame]]:
    """
    Generate scenarios by sampling from empirical distributions of training data labels.

    Shipping cost penalties are drawn from the *global* empirical delivery-time distribution
    (carrier-agnostic), matching the intended empirical baseline.
    """
    rng = np.random.default_rng(seed or cfg.RANDOM_SEED)
    unique_skus = [str(sku) for sku in order_items['sku_ID'].unique()]
    option_ids = order_options['option_id'].astype(int).tolist()
    base_costs_by_option = order_options.set_index('option_id')['base_cost'].astype(float)

    cached_data = _get_cached_training_data(order_set, verbose)
    delivery_times = cached_data['delivery_times']
    demand_samples = cached_data['demand_samples']
    sku_indices = cached_data['sku_indices']
    sku_to_idx = _sku_to_idx_mapping()

    # --- Demand scenarios per SKU ---
    demand_scenarios: Dict[str, pd.Series] = {}
    horizon = max(1, int(lookahead_periods) if lookahead_periods is not None else cached_data['demand_horizon'])

    for sku in unique_skus:
        sku_idx = sku_to_idx.get(str(sku))
        if sku_idx is None:
            sku_demand_samples = np.empty((0, 0), dtype=float)
        else:
            sku_mask = sku_indices == int(sku_idx)
            sku_demand_samples = demand_samples[sku_mask]

        if sku_demand_samples.size == 0:
            draws = np.zeros(num_scenarios, dtype=int)
            if verbose:
                print(f"  No training data for SKU {sku} (idx={sku_idx}), using zeros")
        else:
            horizon_to_use = min(horizon, sku_demand_samples.shape[1])
            cumulative_demands = sku_demand_samples[:, :horizon_to_use].sum(axis=1)
            sample_indices = rng.integers(0, len(cumulative_demands), size=num_scenarios)
            draws = np.rint(cumulative_demands[sample_indices]).astype(int)
            if verbose:
                print(
                    f"  SKU {sku}: {len(cumulative_demands)} samples, lookahead={horizon_to_use}, "
                    f"demand range=[{cumulative_demands.min():.1f}, {cumulative_demands.max():.1f}]"
                )

        demand_scenarios[sku] = pd.Series(draws, index=[f'scenario_{s}' for s in range(num_scenarios)])

    # --- Shipping cost scenarios (global, carrier-agnostic penalties) ---
    # Sample delivery times independently for each option (even though from same distribution)
    # This ensures each option has independent uncertainty in each scenario
    scenario_index = [f'scenario_{s}' for s in range(num_scenarios)]
    num_options = len(option_ids)
    base_cost_matrix = np.tile(
        base_costs_by_option.loc[option_ids].to_numpy(dtype=float),
        (num_scenarios, 1)
    )

    # Sample delivery times independently for each (scenario, option) pair
    # Shape: (num_scenarios, num_options)
    sampled_times = rng.choice(delivery_times, size=(num_scenarios, num_options), replace=True)
    deviation = sampled_times - promise_days
    penalty_matrix = (
        cfg.GAMMA_PLUS_LATE_PENALTY * np.maximum(0, deviation)
        + cfg.GAMMA_MINUS_EARLY_PENALTY * np.maximum(0, -deviation)
    )

    costs_matrix = base_cost_matrix + penalty_matrix
    costs_df = pd.DataFrame(costs_matrix, index=scenario_index, columns=option_ids)

    shipping_costs: Dict[str, pd.DataFrame] = {
        sku: costs_df.copy() for sku in unique_skus
    }

    if verbose:
        print(f"Generated {num_scenarios} empirical scenarios for {len(unique_skus)} SKUs")

    return demand_scenarios, shipping_costs


def compute_empirical_medians(
    order_items: pd.DataFrame,
    order_options: pd.DataFrame,
    promise_days: float,
    order_set: str,
    lookahead_periods: int,
    verbose: bool = False,
) -> Tuple[Dict[str, int], Dict[int, float]]:
    """
    Compute empirical median total demand per SKU over lookahead, and empirical median
    shipping cost per option (base cost + penalty), without sampling.
    """
    unique_skus = [str(sku) for sku in order_items['sku_ID'].unique()]
    option_ids = order_options['option_id'].astype(int).tolist()
    base_costs_by_option = order_options.set_index('option_id')['base_cost'].astype(float)

    cached_data = _get_cached_training_data(order_set, verbose)
    delivery_times = cached_data['delivery_times']
    demand_samples = cached_data['demand_samples']
    sku_indices = cached_data['sku_indices']
    sku_to_idx = _sku_to_idx_mapping()

    demand_medians: Dict[str, int] = {}
    horizon = max(1, int(lookahead_periods) if lookahead_periods is not None else cached_data['demand_horizon'])

    for sku in unique_skus:
        sku_idx = sku_to_idx.get(str(sku))
        if sku_idx is None:
            sku_demand_samples = np.empty((0, 0), dtype=float)
        else:
            sku_mask = sku_indices == int(sku_idx)
            sku_demand_samples = demand_samples[sku_mask]

        if sku_demand_samples.size == 0:
            median_val = 0.0
            if verbose:
                print(f"  [medians] No training data for SKU {sku} (idx={sku_idx}), median=0")
        else:
            horizon_to_use = min(horizon, sku_demand_samples.shape[1])
            cumulative_demands = sku_demand_samples[:, :horizon_to_use].sum(axis=1)
            median_val = float(np.median(cumulative_demands))
        demand_medians[sku] = int(max(0, round(median_val)))

    deviations = delivery_times - promise_days
    penalties = (
        cfg.GAMMA_PLUS_LATE_PENALTY * np.maximum(0.0, deviations)
        + cfg.GAMMA_MINUS_EARLY_PENALTY * np.maximum(0.0, -deviations)
    )
    penalty_median = float(np.median(penalties))

    cost_medians: Dict[int, float] = {
        int(opt): float(base_costs_by_option.loc[opt] + penalty_median)
        for opt in option_ids
    }

    if verbose:
        print(f"Computed empirical medians for {len(unique_skus)} SKUs and {len(option_ids)} options")

    return demand_medians, cost_medians
