import hashlib
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Set, Optional

import numpy as np
import pandas as pd

from src import config as cfg

def build_dc_event_snapshot(data: pd.DataFrame) -> pd.DataFrame:
    """
    Accurate backlog + 2‑hour shipment counts, keeping native timestamps.
    """
    # === DC Operations ===
    data.sort_values('order_time', inplace=True)
    data['ship_out_time'] = pd.to_datetime(data['ship_out_time'])

    # --- Waiting Orders (DC-level) ---
    placed_events = data[['order_time', 'dc_ori']].copy()
    placed_events.rename(columns={'order_time': 'timestamp'}, inplace=True)
    placed_events['change'] = 1

    shipped_events = data[['ship_out_time', 'dc_ori']].copy()
    shipped_events.rename(columns={'ship_out_time': 'timestamp'}, inplace=True)
    shipped_events['change'] = -1

    all_events = pd.concat([placed_events, shipped_events])
    all_events.sort_values('timestamp', inplace=True)
    all_events['waiting_orders'] = all_events.groupby('dc_ori')['change'].cumsum().shift(1).fillna(0)

    data = pd.merge_asof(
        data,
        all_events[['timestamp', 'dc_ori', 'waiting_orders']],
        left_on='order_time',
        right_on='timestamp',
        by='dc_ori',
        direction='backward'
    )
    # Clean up timestamp from first merge
    data.drop(columns=['timestamp'], inplace=True, errors='ignore')

    # --- Waiting SKUs (DC-SKU-level) ---
    sku_placed_events = data[['order_time', 'dc_ori', 'sku_ID']].copy()
    sku_placed_events.rename(columns={'order_time': 'timestamp'}, inplace=True)
    sku_placed_events['change_sku'] = 1

    sku_shipped_events = data[['ship_out_time', 'dc_ori', 'sku_ID']].copy()
    sku_shipped_events.rename(columns={'ship_out_time': 'timestamp'}, inplace=True)
    sku_shipped_events['change_sku'] = -1
    
    all_sku_events = pd.concat([sku_placed_events, sku_shipped_events])
    all_sku_events.sort_values('timestamp', inplace=True)
    all_sku_events['waiting_skus'] = all_sku_events.groupby(['dc_ori', 'sku_ID'])['change_sku'].cumsum().shift(1).fillna(0)

    data = pd.merge_asof(
        data,
        all_sku_events[['timestamp', 'dc_ori', 'sku_ID', 'waiting_skus']],
        left_on='order_time',
        right_on='timestamp',
        by=['dc_ori', 'sku_ID'],
        direction='backward'
    )

    # --- Shipped Orders/SKUs in last 2h ---
    # ----------------------------------------------------------
    # 1.  Build a shipment‑only table indexed by ship_out_time
    # ----------------------------------------------------------
    ship = data[["ship_out_time", "dc_ori", "sku_ID", "order_ID"]].copy()
    ship.rename(columns={"ship_out_time": "ts"}, inplace=True)
    ship.sort_values("ts", inplace=True)
    ship.set_index("ts", inplace=True)     # rolling window will slide on this

    # ----------------------------------------------------------
    # 2.  Rolling counts on that index
    #     Window = [t‑2 h, t)   (same as closed='left')
    # ----------------------------------------------------------
    dc_roll = (
        ship.groupby("dc_ori")["order_ID"]
            .rolling("2h", closed="left")
            .count()
            .rename("shipped_orders_last_2h")
            .reset_index()                # → columns: dc_ori, ts, shipped_orders_last_2h
    )

    sku_roll = (
        ship.groupby(["dc_ori", "sku_ID"])["order_ID"]
            .rolling("2h", closed="left")
            .count()
            .rename("shipped_skus_last_2h")
            .reset_index()                # → dc_ori, sku_ID, ts, shipped_skus_last_2h
    )

    # ----------------------------------------------------------
    # 3.  Point‑in‑time join back to the *order_time*
    # ----------------------------------------------------------
    data.sort_values("order_time", inplace=True)

    #   (a) DC‑level shipment count
    data = pd.merge_asof(
        data,
        dc_roll.sort_values("ts"),
        by="dc_ori",
        left_on="order_time",
        right_on="ts",
        direction="backward"
    )

    #   (b) DC×SKU shipment count
    data = pd.merge_asof(
        data,
        sku_roll.sort_values("ts"),
        by=["dc_ori", "sku_ID"],
        left_on="order_time",
        right_on="ts",
        direction="backward"
    )

    # For multi-item orders, select the max SKU-level feature value
    sku_level_features = ['shipped_skus_last_2h', 'waiting_skus']
    for feature in sku_level_features:
        max_val_per_order = data.groupby('order_ID')[feature].transform('max')
        data[feature] = max_val_per_order

    # Fill NaNs created by rolling windows and merges
    cols_to_fill = sku_level_features + ['shipped_orders_last_2h', 'waiting_orders']
    data[cols_to_fill] = data[cols_to_fill].fillna(0)
    
        
    # Clean up temporary columns
    data.drop(columns=['discount_rate', 'timestamp', 'ts', 'change', 'change_sku'], inplace=True, errors='ignore')
    
    # ------------------------------------------------------------------
    # 5. Return snapshot
    # ------------------------------------------------------------------
    return data


def load_data(order_set: str = "test"):
    """Loads orders and pre-computed setup files.

    Args:
        order_set (str): The set of orders to load ('test' or 'proxy_train').
    """
    print(f"Loading data for '{order_set}' order set...")
    try:
        if order_set == "test":
            orders_df = pd.read_csv(cfg.DELIVERY_TEST_PATH)
        elif order_set == "proxy_train":
            orders_df = pd.read_csv(cfg.DELIVERY_PROXY_TRAIN_PATH)
        else:
            raise ValueError(
                f"Invalid order_set: {order_set}. Must be 'test' or 'proxy_train'."
            )

        network_df = pd.read_csv(cfg.NETWORK_PATH)
        sku_df = pd.read_csv(cfg.SKU_DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please run the setup script first.")
        return None, None, None
    return orders_df, network_df, sku_df


@lru_cache(maxsize=1)
def get_observed_dcs_from_preprocessed(preprocessed_path: Optional[Path] = None) -> set[int]:
    """Return DC ids observed in the preprocessed data (dc_des or dc_ori)."""
    path = preprocessed_path or cfg.PREPROCESSED_PATH
    if not Path(path).exists():
        return set()
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in ('dc_des', 'dc_ori') if c in header.columns]
    if not cols:
        return set()
    df = pd.read_csv(path, usecols=cols)
    observed: set[int] = set()
    for col in cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        observed.update(series.astype(int).tolist())
    return observed


@lru_cache(maxsize=1)
def load_dc_carrier_metadata() -> tuple[pd.DataFrame, dict[int, float]]:
    """Load and cache DC-carrier eligibility metadata and carrier cost coefficients."""
    try:
        eligibility_df = pd.read_csv(cfg.DC_CARRIER_ELIGIBILITY_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"DC-carrier eligibility data not found at {cfg.DC_CARRIER_ELIGIBILITY_PATH}. "
            "This is a small tracked aggregate shipped under data/params/ "
            "(see DATA.md)."
        ) from exc

    try:
        cost_models_df = pd.read_csv(cfg.REAL_COST_MODELS_CS_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Carrier cost model coefficients not found at {cfg.REAL_COST_MODELS_CS_PATH}. "
            "Ensure data/params/real_cost_models_cs.csv is present."
        ) from exc

    try:
        dc_coords_df = pd.read_csv(cfg.HUB_BASED_COORDS_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"DC coordinate file not found at {cfg.HUB_BASED_COORDS_PATH}. "
            "Run the coordinate generation tool first."
        ) from exc

    # Filter out excluded carrier services (insufficient training data)
    if cfg.EXCLUDED_CARRIER_SERVICES:
        eligibility_df = eligibility_df[
            ~eligibility_df['carrier_service_id'].isin(cfg.EXCLUDED_CARRIER_SERVICES)
        ].copy()
        cost_models_df = cost_models_df[
            ~cost_models_df['carrier_service_id'].isin(cfg.EXCLUDED_CARRIER_SERVICES)
        ].copy()

    metadata = prepare_option_metadata(eligibility_df, dc_coords_df)
    cost_coef_map = (
        cost_models_df[['carrier_service_id', 'coef_distance_km']]
        .dropna(subset=['carrier_service_id', 'coef_distance_km'])
        .drop_duplicates(subset='carrier_service_id')
        .set_index('carrier_service_id')['coef_distance_km']
        .astype(float)
        .to_dict()
    )
    if not cost_coef_map:
        raise ValueError("No carrier_service_id coefficients found in real_cost_models_cs.csv")

    return metadata, cost_coef_map


def _normalize_zip3(value) -> Optional[str]:
    """Normalize a ZIP code-like value to a 3-digit ZIP prefix."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    digits = ''.join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(3)[:3]


def _aggregate_limited_coverage_to_zip3(raw_coverage: Dict[str, Dict[str, list[str]]]) -> Dict[int, Dict[str, Set[str]]]:
    """
    Aggregate limited coverage data to ZIP3 level per carrier-service.
    
    Returns:
        Dict[int, Dict[str, Set[str]]]: carrier_id -> zip3 -> set(states)
    """
    from collections import defaultdict
    aggregated: Dict[int, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for carrier_key, origin_map in raw_coverage.items():
        try:
            carrier_id = int(carrier_key)
        except (TypeError, ValueError):
            continue
        for origin_zip, states in origin_map.items():
            zip3 = _normalize_zip3(origin_zip)
            if zip3 is None:
                continue
            for state in states or []:
                state_clean = str(state).strip().upper()
                if state_clean:
                    aggregated[carrier_id][zip3].add(state_clean)
    return aggregated


def _parse_allowed_states(states_str: str) -> set[str]:
    """Parse a pipe-delimited list of states into a Python set."""
    if not isinstance(states_str, str) or not states_str:
        return set()
    return {state.strip().upper() for state in states_str.split('|') if state.strip()}


def compute_haversine_km(latitudes: np.ndarray, longitudes: np.ndarray, customer_lat: float, customer_lon: float) -> np.ndarray:
    """Vectorized haversine distance (in km) from arrays of lat/lon to a single customer coordinate."""
    lat1 = np.radians(latitudes.astype(float))
    lon1 = np.radians(longitudes.astype(float))
    lat2 = np.radians(float(customer_lat))
    lon2 = np.radians(float(customer_lon))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c  # Earth's radius in kilometers


@lru_cache(maxsize=1)
def get_central_warehouse_ids() -> Set[int]:
    """Return the set of DC IDs that operate as central warehouses (region IDs)."""
    network_df = pd.read_csv(cfg.NETWORK_PATH)
    region_col = 'region_ID' if 'region_ID' in network_df.columns else 'region_id'
    dc_col = 'dc_ID' if 'dc_ID' in network_df.columns else 'dc_id'
    if region_col not in network_df.columns or dc_col not in network_df.columns:
        raise ValueError("NETWORK_PATH must contain region and dc identifier columns.")
    region_ids = pd.to_numeric(network_df[region_col], errors='coerce').dropna().astype(int)
    dc_ids = pd.to_numeric(network_df[dc_col], errors='coerce').dropna().astype(int)
    return set(dc_ids).intersection(set(region_ids))


def get_per_unit_base_cost(dc_id: int) -> float:
    """Fixed per-unit shipping cost component based on DC type."""
    try:
        dc_value = int(dc_id)
    except (TypeError, ValueError):
        dc_value = None
    if dc_value is not None and dc_value in get_central_warehouse_ids():
        return cfg.CENTRAL_WAREHOUSE_BASE_COST_PER_UNIT
    return cfg.LOCAL_DC_BASE_COST_PER_UNIT


def prepare_option_metadata(eligibility_df: pd.DataFrame, dc_coords_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DC-carrier eligibility table with coordinate information."""
    metadata = eligibility_df.copy()
    metadata['dc_id'] = metadata['dc_id'].astype(int)
    metadata['carrier_service_id'] = metadata['carrier_service_id'].astype(int)
    metadata['allowed_states'] = metadata['allowed_states'].fillna('')
    metadata['allowed_states_set'] = metadata['allowed_states'].apply(_parse_allowed_states)

    coords_cols = ['dc_id', 'lat', 'lon']
    coords = dc_coords_df[coords_cols].drop_duplicates(subset='dc_id', keep='first')
    metadata = metadata.merge(coords, on='dc_id', how='left', validate='m:1')
    return metadata


def compute_order_options(order_info: pd.Series, option_metadata: pd.DataFrame, cost_coef_map: dict[int, float]) -> pd.DataFrame:
    """Build the feasible (dc, carrier) options for a single order."""
    customer_state = str(order_info.get('customer_state', '')).strip().upper()
    customer_lat = order_info.get('customer_lat')
    customer_lon = order_info.get('customer_lon')

    required_cols = ['option_id', 'dc_id', 'carrier_service_id', 'zip3', 'distance_km', 'base_cost', 'region_id']

    if not customer_state or pd.isna(customer_lat) or pd.isna(customer_lon):
        return pd.DataFrame(columns=required_cols)

    # Check eligibility: carriers with "ALL" in allowed_states_set have full coverage
    eligible_mask = option_metadata['allowed_states_set'].map(
        lambda states: 'ALL' in states or customer_state in states
    )
    eligible = option_metadata.loc[eligible_mask].copy()
    if eligible.empty:
        return pd.DataFrame(columns=required_cols)

    eligible.dropna(subset=['lat', 'lon'], inplace=True)
    eligible['carrier_service_id'] = eligible['carrier_service_id'].astype(int)
    eligible['dc_id'] = eligible['dc_id'].astype(int)
    eligible['coef_distance_km'] = eligible['carrier_service_id'].map(cost_coef_map)
    eligible.dropna(subset=['coef_distance_km'], inplace=True)
    if eligible.empty:
        return pd.DataFrame(columns=required_cols)

    distances = compute_haversine_km(
        eligible['lat'].to_numpy(dtype=float),
        eligible['lon'].to_numpy(dtype=float),
        float(customer_lat),
        float(customer_lon),
    )
    eligible['distance_km'] = distances
    central_ids = get_central_warehouse_ids()
    fixed_local = cfg.LOCAL_DC_BASE_COST_PER_UNIT
    fixed_central = cfg.CENTRAL_WAREHOUSE_BASE_COST_PER_UNIT
    eligible['base_cost'] = (
        eligible['distance_km'] * eligible['coef_distance_km']
        + np.where(eligible['dc_id'].isin(central_ids), fixed_central, fixed_local)
    )
    eligible = eligible[eligible['base_cost'].notna()].reset_index(drop=True)

    if eligible.empty:
        return pd.DataFrame(columns=required_cols)

    eligible['option_id'] = np.arange(len(eligible), dtype=int)
    return eligible[['option_id', 'dc_id', 'carrier_service_id', 'zip3', 'distance_km', 'base_cost', 'region_id']]


def derive_base_costs_from_options(order_options: pd.DataFrame, compatible_dcs: list | None = None) -> pd.Series:
    """
    Derive base costs per DC from order_options by taking the minimum cost per DC.
    
    This replaces the need for unit_costs_df when we have order_options with distance-based costs.
    
    Args:
        order_options: DataFrame with columns ['dc_id', 'base_cost']
        compatible_dcs: Optional list of DCs to filter to
        
    Returns:
        Series indexed by DC ID with minimum base_cost per DC
    """
    if order_options is None or order_options.empty:
        return pd.Series(dtype=float)
    
    # Group by DC and take minimum cost (cheapest carrier option per DC)
    dc_costs = order_options.groupby('dc_id')['base_cost'].min()
    
    if compatible_dcs is not None:
        dc_costs = dc_costs.reindex(compatible_dcs)
    
    return dc_costs


def load_precomputed_order_dc_features(
    order_dc_features_dir: str,
    simulation_date: str = None,
    order_set: str | None = None,
) -> pd.DataFrame:
    """
    Load and combine pre-computed order-DC features from a directory of
    partitioned Parquet files.

    Args:
        order_dc_features_dir (str): Path to the directory containing DC-partitioned files.
        simulation_date (str, optional): If provided, filters the features to this date.
        order_set (str, optional): Optional subdirectory (e.g., 'test', 'proxy_train') to use.

    Returns:
        pd.DataFrame: A combined DataFrame of all features.
    """
    feature_dir = Path(order_dc_features_dir)
    if order_set:
        candidate_dir = feature_dir / order_set
        if candidate_dir.exists():
            feature_dir = candidate_dir
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found at: {feature_dir}")

    all_features = []
    for dc_file in feature_dir.glob("dc=*.parquet"):
        df = pd.read_parquet(dc_file)
        all_features.append(df)

    if not all_features:
        print(f"Warning: No feature files found in {feature_dir}")
        return pd.DataFrame()

    combined_df = pd.concat(all_features, ignore_index=True)

    if simulation_date:
        combined_df['order_time'] = pd.to_datetime(combined_df['order_time'])
        sim_date = pd.to_datetime(simulation_date).date()
        combined_df = combined_df[combined_df['order_time'].dt.date == sim_date]

    if 'order_id' in combined_df.columns and 'order_ID' not in combined_df.columns:
        combined_df = combined_df.rename(columns={'order_id': 'order_ID'})

    return combined_df

def get_initial_inventory(simulation_date):
    """Estimate the initial SKU × DC inventory state from historical demand."""

    def _hash_to_unit(value: str) -> float:
        digest = hashlib.sha256(value.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], 'big') / float(2**64)

    def _compute_target_level(demand_sum: float, demand_sq_sum: float, days: int, z_value: float) -> float:
        if days <= 0:
            return 0.0
        mean_daily = demand_sum / days
        variance_daily = max((demand_sq_sum / days) - (mean_daily ** 2), 0.0)
        std_daily = np.sqrt(variance_daily)
        return mean_daily + z_value * std_daily

    preprocessed_df = pd.read_csv(cfg.PREPROCESSED_PATH)
    preprocessed_df['order_time'] = pd.to_datetime(preprocessed_df['order_time'])
    if 'dc_des' not in preprocessed_df.columns:
        preprocessed_df['dc_des'] = preprocessed_df['dc_ori']
    preprocessed_df['dc_ori'] = pd.to_numeric(preprocessed_df['dc_ori'], errors='coerce')
    preprocessed_df['dc_des'] = pd.to_numeric(preprocessed_df['dc_des'], errors='coerce')

    sku_df = pd.read_csv(cfg.SKU_DATA_PATH)
    all_skus = sku_df['sku_ID'].unique()
    all_dcs = np.sort(preprocessed_df['dc_ori'].dropna().unique())
    inventory_idx = pd.MultiIndex.from_product([all_skus, all_dcs], names=['sku_ID', 'dc_ID'])
    initial_inventory_df = pd.DataFrame(0, index=inventory_idx, columns=['onhand_inventory'])

    sim_timestamp = pd.to_datetime(simulation_date)
    train_df = preprocessed_df.loc[preprocessed_df['order_time'] < sim_timestamp]

    network_df = pd.read_csv(cfg.NETWORK_PATH)
    dc_to_region = dict(zip(network_df['dc_ID'], network_df['region_ID']))
    region_ids = network_df['region_ID'].unique()
    central_warehouses: Set[int] = set(all_dcs).intersection(set(region_ids))

    if train_df.empty:
        history_start = (sim_timestamp - pd.Timedelta(days=1)).normalize()
    else:
        history_start = train_df['order_time'].min().normalize()
    history_end = (sim_timestamp - pd.Timedelta(days=1)).normalize()
    history_days = max(int((history_end - history_start).days) + 1, 1)

    requested_orders = train_df[train_df['dc_des'].notna()].copy()
    requested_orders['order_date'] = requested_orders['order_time'].dt.floor('D')

    local_stats = pd.DataFrame(columns=['sku_ID', 'dc_ID', 'demand_sum', 'demand_sq_sum'])
    central_stats = pd.DataFrame(columns=['sku_ID', 'dc_ID', 'demand_sum', 'demand_sq_sum'])

    if not requested_orders.empty:
        local_daily = (
            requested_orders
            .groupby(['order_date', 'sku_ID', 'dc_des'])['quantity']
            .sum()
            .reset_index()
            .rename(columns={'dc_des': 'dc_ID'})
        )
        local_daily['dc_ID'] = pd.to_numeric(local_daily['dc_ID'], errors='coerce')
        local_daily.dropna(subset=['dc_ID'], inplace=True)
        local_daily['dc_ID'] = local_daily['dc_ID'].astype(int)
        local_daily['quantity_sq'] = local_daily['quantity'] ** 2
        local_stats = (
            local_daily
            .groupby(['sku_ID', 'dc_ID'])
            .agg(demand_sum=('quantity', 'sum'), demand_sq_sum=('quantity_sq', 'sum'))
            .reset_index()
        )
        local_daily['region_ID'] = local_daily['dc_ID'].map(dc_to_region)
        central_daily = local_daily.dropna(subset=['region_ID']).copy()
        if not central_daily.empty:
            central_daily = (
                central_daily
                .groupby(['order_date', 'sku_ID', 'region_ID'])['quantity']
                .sum()
                .reset_index()
            )
            central_daily['quantity_sq'] = central_daily['quantity'] ** 2
            central_stats = (
                central_daily
                .groupby(['sku_ID', 'region_ID'])
                .agg(demand_sum=('quantity', 'sum'), demand_sq_sum=('quantity_sq', 'sum'))
                .reset_index()
                .rename(columns={'region_ID': 'dc_ID'})
            )
            central_stats = central_stats[central_stats['dc_ID'].isin(central_warehouses)]

    norm_dist = NormalDist()
    local_z = norm_dist.inv_cdf(cfg.LOCAL_DC_CSL)
    central_z = norm_dist.inv_cdf(cfg.CENTRAL_WAREHOUSE_CSL)

    local_stocked = 0
    for row in local_stats.itertuples(index=False):
        dc_id = int(row.dc_ID)
        if dc_id in central_warehouses:
            # Central warehouses are handled separately using regional demand.
            continue
        if cfg.LOCAL_DC_STOCK_PROB < 1.0:
            if _hash_to_unit(f"{row.sku_ID}|{dc_id}") > cfg.LOCAL_DC_STOCK_PROB:
                continue
        target = _compute_target_level(row.demand_sum, row.demand_sq_sum, history_days, local_z)
        if target <= 0:
            continue
        local_stocked += 1
        initial_inventory_df.at[(row.sku_ID, dc_id), 'onhand_inventory'] = int(np.ceil(target))

    central_stocked = 0
    for row in central_stats.itertuples(index=False):
        dc_id = int(row.dc_ID)
        target = _compute_target_level(row.demand_sum, row.demand_sq_sum, history_days, central_z)
        if target <= 0:
            continue
        central_stocked += 1
        initial_inventory_df.at[(row.sku_ID, dc_id), 'onhand_inventory'] = int(np.ceil(target))

    initial_inventory_df['onhand_inventory'] = initial_inventory_df['onhand_inventory'].astype(int)

    print(
        "Initial inventory derived from requested demand (dc_des) "
        f"over {history_days} day(s) prior to {simulation_date}. "
        f"Local DC CSL={cfg.LOCAL_DC_CSL} (z={local_z:.2f}), central CSL={cfg.CENTRAL_WAREHOUSE_CSL} "
        f"(z={central_z:.2f}); stocked combos — local: {local_stocked}, central: {central_stocked}."
    )

    return initial_inventory_df, all_dcs



def preprocess_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    df['order_hour'] = df['order_time'].dt.hour
    df['weekday']    = df['order_time'].dt.dayofweek
    df['num_skus_in_order']   = df.groupby('order_ID')['sku_ID'].transform('nunique')
    df['has_bundle_discount'] = (df['bundle_discount_per_unit'] > 0).astype(int)
    df['has_coupon_discount'] = (df['coupon_discount_per_unit'] > 0).astype(int)
    df['has_gift_item']       = df.groupby('order_ID')['gift_item'].transform('max').astype(int)
    df['total_quantity_in_order'] = df.groupby('order_ID')['quantity'].transform('sum')
    disc = -1 * (df['final_unit_price'] - df['original_unit_price']) / df['original_unit_price']
    df['discount_rate'] = disc.replace([np.inf, -np.inf], 0).fillna(0)
    df['avg_discount_rate_in_order'] = df.groupby('order_ID')['discount_rate'].transform('mean').round(4)
    df['gender']         = df['gender'].map({'M': 1, 'F': 0, 'U': -1})
    df['age']            = df['age'].map({'0-25':0,'26-35':1,'36-45':2,'46-55':3,'56+':4,'U':-1})
    df['marital_status'] = df['marital_status'].map({'S':0,'M':1,'U':-1})
    for col in ['purchase_power','attribute1','attribute2','education']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
    
    return df.fillna(-1)


def compute_base_costs_for_order(
    customer_lat: float,
    customer_lon: float,
    all_dcs: list,
    all_carriers: list,
    dc_metadata: pd.DataFrame,
    cost_coef_map: dict,
) -> np.ndarray:
    """
    Compute base shipping costs for one order.
    
    Returns:
        [D, C] array of base costs
    """
    num_dcs = len(all_dcs)
    num_carriers = len(all_carriers)
    base_cost_grid = np.zeros((num_dcs, num_carriers), dtype=np.float32)
    
    central_ids = get_central_warehouse_ids()
    fixed_local = cfg.LOCAL_DC_BASE_COST_PER_UNIT
    fixed_central = cfg.CENTRAL_WAREHOUSE_BASE_COST_PER_UNIT
    
    dc_meta_dict = {}
    for _, row in dc_metadata.iterrows():
        dc_id = int(row['dc_id'])
        dc_meta_dict[dc_id] = {
            'lat': float(row['lat']),
            'lon': float(row['lon']),
        }
    
    for d_idx, dc_id in enumerate(all_dcs):
        if dc_id not in dc_meta_dict:
            continue
        
        dc_info = dc_meta_dict[dc_id]
        lat1, lon1 = np.radians(dc_info['lat']), np.radians(dc_info['lon'])
        lat2, lon2 = np.radians(customer_lat), np.radians(customer_lon)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance_km = 6371.0 * c
        
        fixed_cost = fixed_central if dc_id in central_ids else fixed_local
        
        for c_idx, carrier_id in enumerate(all_carriers):
            coef = cost_coef_map.get(carrier_id, 0.0)
            base_cost_grid[d_idx, c_idx] = (distance_km * coef) + fixed_cost
    
    return base_cost_grid


def compute_global_features_for_order(
    order_feat_vec: dict,
    sku_quantity: float,
    global_order_volume: float,
    base_cost_grid: Optional[np.ndarray] = None,
) -> dict:
    """Build global features for one SKU."""
    global_feats = dict(order_feat_vec)
    global_feats.update({
        'sku_quantity': float(sku_quantity),
        'global_order_volume': float(global_order_volume),
    })
    if base_cost_grid is not None:
        base_cost_arr = np.asarray(base_cost_grid, dtype=float)
        if base_cost_arr.size == 0 or np.all(np.isnan(base_cost_arr)):
            base_cost_mean = 0.0
            base_cost_min = 0.0
            base_cost_max = 0.0
        else:
            base_cost_mean = float(np.nanmean(base_cost_arr))
            base_cost_min = float(np.nanmin(base_cost_arr))
            base_cost_max = float(np.nanmax(base_cost_arr))
        global_feats.update({
            'base_cost_mean': base_cost_mean,
            'base_cost_min': base_cost_min,
            'base_cost_max': base_cost_max,
        })
    return global_feats


def compute_dc_features_for_order(
    inv_vec: np.ndarray,
    sku_daily_demand: float,
    region_match_vec: np.ndarray,
    consolidation_potential: np.ndarray,
    customer_lat: float,
    customer_lon: float,
    dc_meta_dict: dict,
    all_dcs: list,
) -> np.ndarray:
    """
    Build DC features for one SKU.
    
    Returns:
        [D, 5] array: [inventory, days_of_supply, region_match, consolidation, distance_km]
    """
    num_dcs = len(all_dcs)
    dos_vec = inv_vec / (sku_daily_demand + 1e-6)
    
    distance_vec = np.zeros(num_dcs, dtype=np.float32)
    for d_idx, dc_id in enumerate(all_dcs):
        if dc_id not in dc_meta_dict:
            distance_vec[d_idx] = 9999.0
            continue
        dc_info = dc_meta_dict[dc_id]
        lat1, lon1 = np.radians(dc_info['lat']), np.radians(dc_info['lon'])
        lat2, lon2 = np.radians(customer_lat), np.radians(customer_lon)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance_vec[d_idx] = 6371.0 * c
    
    return np.stack([
        inv_vec,
        dos_vec,
        region_match_vec,
        consolidation_potential,
        distance_vec
    ], axis=1)
