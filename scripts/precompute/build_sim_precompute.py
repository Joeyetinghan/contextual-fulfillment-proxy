"""Build precomputed artifacts for simulation."""

import argparse
from pathlib import Path
import pandas as pd
import pyarrow.feather as feather
import json

import src.config as cfg
from src.data_utils import (
    load_dc_carrier_metadata,
    _aggregate_limited_coverage_to_zip3,
    build_dc_event_snapshot,
    get_observed_dcs_from_preprocessed,
)
import numpy as np
from typing import List


def load_and_split_data(order_set: str = 'test') -> pd.DataFrame:
    """Load preprocessed_cs data and split by time based on order_set.
    
    Args:
        order_set: 'test' or 'proxy_train' to select the appropriate time split
        
    Returns:
        DataFrame with the selected time split
    """
    # Load preprocessed_cs data
    preprocessed_cs_path = cfg.PROCESSED_DATA_DIR / 'preprocessed_data_cs.csv'
    print(f"Loading data from {preprocessed_cs_path}...")
    data = pd.read_csv(preprocessed_cs_path, parse_dates=['order_time', 'order_date'])
    
    # Split by time (same logic as delivery_time_feature_engineering.py)
    forecast_train_start = pd.to_datetime(cfg.FORECAST_TRAIN_START_DATE)
    forecast_train_end = pd.to_datetime(cfg.FORECAST_TRAIN_END_DATE)
    proxy_train_end = pd.to_datetime(cfg.PROXY_TRAIN_END_DATE)
    
    if order_set == 'test':
        # Test set: orders after proxy_train_end
        split_df = data[data['order_date'] > proxy_train_end].copy()
        print(f"  Selected test set: {len(split_df):,} orders (order_date > {proxy_train_end.date()})")
    else:  # proxy_train
        # Proxy train set: orders between forecast_train_end and proxy_train_end
        split_df = data[(data['order_date'] > forecast_train_end) & 
                       (data['order_date'] <= proxy_train_end)].copy()
        print(f"  Selected proxy_train set: {len(split_df):,} orders ({forecast_train_end.date()} < order_date <= {proxy_train_end.date()})")
    
    return split_df


def build_options_catalog(output_dir: Path):
    """Build options catalog from source data."""
    print("Building options catalog...")
    from src.data_utils import _normalize_zip3

    metadata_df, cost_coef_map = load_dc_carrier_metadata()
    observed_dcs = get_observed_dcs_from_preprocessed()
    if observed_dcs:
        before_dcs = set(metadata_df['dc_id'].unique())
        metadata_df = metadata_df[metadata_df['dc_id'].isin(observed_dcs)].copy()
        after_dcs = set(metadata_df['dc_id'].unique())
        dropped = sorted(before_dcs - after_dcs)
        if dropped:
            print(
                f"  Filtered DCs to observed preprocessed set: "
                f"{len(after_dcs)}/{len(before_dcs)} (dropped {dropped})"
            )
    
    catalog_df = metadata_df[[
        'dc_id', 'carrier_service_id', 'zip3', 'lat', 'lon'
    ]].rename(columns={
        'zip3': 'dc_zip3',
        'lat': 'dc_lat',
        'lon': 'dc_lng',
    }).drop_duplicates()
    catalog_df['dc_zip3'] = catalog_df['dc_zip3'].apply(_normalize_zip3)
    
    output_path = output_dir / 'options_catalog.parquet'
    catalog_df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")


def build_coverage_allowed_states(output_dir: Path):
    """Build option-level coverage eligibility table.
    
    The runtime reader expects one row per option. Restricted options list the
    allowed destination states; unrestricted options are marked as ALL.
    """
    print("Building coverage allowed states...")
    
    try:
        with open(cfg.LIMITED_COVERAGE_PATH, 'r') as f:
            raw_coverage = json.load(f)
    except FileNotFoundError:
        print(f"  Warning: {cfg.LIMITED_COVERAGE_PATH} not found, skipping coverage table")
        return
    
    # Debug: Show coverage data format
    print(f"  Coverage data format: JSON file with structure:")
    print(f"    {{carrier_id: {{origin_zip: [states], ...}}, ...}}")
    if raw_coverage:
        sample_carrier = list(raw_coverage.keys())[0]
        sample_zip = list(raw_coverage[sample_carrier].keys())[0] if raw_coverage[sample_carrier] else None
        print(f"    Example: carrier '{sample_carrier}' has {len(raw_coverage[sample_carrier])} origin zip entries")
        if sample_zip:
            sample_states = raw_coverage[sample_carrier][sample_zip]
            print(f"      origin_zip '{sample_zip}' -> states: {sample_states[:3]}{'...' if len(sample_states) > 3 else ''}")
    
    # Load options catalog to get dc_id mapping
    catalog_path = output_dir / 'options_catalog.parquet'
    if not catalog_path.exists():
        print(f"  Warning: {catalog_path} not found, building options catalog first...")
        build_options_catalog(output_dir)
    
    catalog_df = pd.read_parquet(catalog_path)
    
    # Aggregate coverage restrictions
    aggregated = _aggregate_limited_coverage_to_zip3(raw_coverage)
    
    if not aggregated:
        print("  Warning: No coverage restrictions found in LIMITED_COVERAGE_PATH; writing all options as unrestricted")
    
    # Debug: Print summary of aggregated data
    total_coverage_entries = sum(len(zip3_map) for zip3_map in aggregated.values())
    print(f"  Found {len(aggregated)} carrier(s) with {total_coverage_entries} zip3 entries in coverage data")
    
    # Collect all zip3s from coverage data for comparison
    coverage_zip3s = set()
    for zip3_map in aggregated.values():
        coverage_zip3s.update(zip3_map.keys())
    print(f"  Coverage data has {len(coverage_zip3s)} unique zip3s")
    
    # Normalize catalog dc_zip3 values to match coverage data format (zero-padded 3-digit strings)
    from src.data_utils import _normalize_zip3
    catalog_df['dc_zip3'] = catalog_df['dc_zip3'].apply(_normalize_zip3)
    catalog_df = catalog_df[catalog_df['dc_zip3'].notna()].copy()  # Remove rows with invalid zip3s
    
    # Ensure data types match for comparison
    catalog_df['carrier_service_id'] = catalog_df['carrier_service_id'].astype(int)
    catalog_df['dc_zip3'] = catalog_df['dc_zip3'].astype(str)
    
    # Debug: Check catalog structure after normalization
    catalog_zip3s = set(catalog_df['dc_zip3'].unique())
    print(f"  Catalog has {len(catalog_df)} rows with carrier_service_ids: {sorted(catalog_df['carrier_service_id'].unique())[:10]}")
    print(f"  Catalog has {len(catalog_zip3s)} unique dc_zip3s (normalized): {sorted(catalog_zip3s)[:10]}")
    
    # Debug: Check overlap between coverage and catalog zip3s
    matching_zip3s = coverage_zip3s & catalog_zip3s
    missing_in_catalog = coverage_zip3s - catalog_zip3s
    missing_in_coverage = catalog_zip3s - coverage_zip3s
    print(f"  Zip3 overlap: {len(matching_zip3s)} in both, {len(missing_in_catalog)} only in coverage, {len(missing_in_coverage)} only in catalog")
    if missing_in_catalog:
        print(f"  Sample zip3s in coverage but not catalog: {sorted(list(missing_in_catalog))[:10]}")
    
    # Track source coverage entries that do not correspond to this option catalog.
    matches_found = 0
    matches_missed = 0
    
    for carrier_id, zip3_map in aggregated.items():
        for dc_zip3, states in zip3_map.items():
            # Find all dc_ids with this zip3
            matching_dcs = catalog_df[
                (catalog_df['dc_zip3'] == dc_zip3) & 
                (catalog_df['carrier_service_id'] == carrier_id)
            ]['dc_id'].unique()
            
            if len(matching_dcs) > 0:
                matches_found += 1
            else:
                matches_missed += 1
                if matches_missed <= 10:  # Print first few misses for debugging
                    # Check if carrier exists but zip3 doesn't match
                    carrier_exists = (catalog_df['carrier_service_id'] == carrier_id).any()
                    zip3_exists = (catalog_df['dc_zip3'] == dc_zip3).any()
                    # Check if zip3 exists for this carrier
                    zip3_for_carrier = ((catalog_df['dc_zip3'] == dc_zip3) & 
                                       (catalog_df['carrier_service_id'] == carrier_id)).any()
                    # Check if carrier has this zip3 with different carrier_id
                    carrier_has_zip3 = (catalog_df['dc_zip3'] == dc_zip3).any()
                    carrier_zip3s = catalog_df[catalog_df['carrier_service_id'] == carrier_id]['dc_zip3'].unique() if carrier_exists else []
                    print(f"    No match: carrier_id={carrier_id}, dc_zip3={dc_zip3} "
                          f"(carrier_exists={carrier_exists}, zip3_exists={zip3_exists}, "
                          f"zip3_for_carrier={zip3_for_carrier}, carrier_has_zip3={carrier_has_zip3})")
                    if carrier_exists and not zip3_for_carrier and len(carrier_zip3s) > 0:
                        print(f"      Carrier {carrier_id} has zip3s: {sorted(carrier_zip3s)[:5]}...")
    
    print(f"  Matches: {matches_found} found, {matches_missed} missed")
    records = []
    restricted_count = 0
    unrestricted_count = 0
    for _, opt_row in catalog_df.iterrows():
        dc_id = int(opt_row['dc_id'])
        carrier_id = int(opt_row['carrier_service_id'])
        dc_zip3 = str(opt_row['dc_zip3'])
        states = aggregated.get(carrier_id, {}).get(dc_zip3, set())
        if states:
            allowed_states = '|'.join(sorted(states))
            restricted_count += 1
        else:
            allowed_states = 'ALL'
            unrestricted_count += 1
        records.append({
            'dc_id': dc_id,
            'carrier_service_id': carrier_id,
            'dc_zip3': dc_zip3,
            'allowed_states': allowed_states,
            'coverage_scope': 'all_options',
        })

    print(f"  Total option records created: {len(records)}")
    print(f"  Restricted option records: {restricted_count}")
    print(f"  Unrestricted option records: {unrestricted_count}")
    
    # Summary explanation
    if matches_missed > 0:
        print(f"\n  Note: {matches_missed} coverage entries didn't match because:")
        print(f"    - Coverage data has restrictions for {len(missing_in_catalog)} zip3s that don't have DCs in the catalog")
        print(f"    - This is expected: coverage data is more comprehensive than the actual DC network")
        print(f"    - Only coverage restrictions for zip3s with actual DCs are saved to the precomputed file")
    
    if records:
        coverage_df = pd.DataFrame(records)
        # Set multi-index for efficient lookups by (dc_id, carrier_service_id)
        coverage_df.set_index(['dc_id', 'carrier_service_id'], inplace=True)
        output_path = output_dir / 'coverage_allowed_states.parquet'
        coverage_df.to_parquet(output_path)
        print(f"  Saved {len(coverage_df)} coverage records to {output_path}")
        
        # Show sample of output structure
        print(f"\n  Output DataFrame structure:")
        print(f"    Index: MultiIndex (dc_id, carrier_service_id)")
        print(f"    Columns: {list(coverage_df.columns)}")
        print(f"    Shape: {coverage_df.shape}")
        print(f"\n  Sample records (first 5):")
        sample_df = coverage_df.head(5).reset_index()
        for idx, row in sample_df.iterrows():
            print(f"    dc_id={row['dc_id']}, carrier_service_id={row['carrier_service_id']}, "
                  f"dc_zip3={row['dc_zip3']}, allowed_states={row['allowed_states'][:50]}...")
    else:
        print("  Warning: No coverage records generated (records list is empty)")


def build_distances(output_dir: Path, order_set: str = 'test'):
    """Build distance matrix (dc_id, customer_zip5) -> distance_km based on dc_ori and customer zip5 coordinates."""
    print("Building distance matrix...")
    
    # Load DC coordinates
    hub_coords_df = pd.read_csv(cfg.HUB_BASED_COORDS_PATH)
    hub_coords_df['dc_id'] = hub_coords_df['dc_id'].astype(int)
    dc_map_lat = hub_coords_df.set_index('dc_id')['lat'].to_dict()
    dc_map_lon = hub_coords_df.set_index('dc_id')['lon'].to_dict()
    
    # Load and split orders from preprocessed_cs
    orders_df = load_and_split_data(order_set)
    
    # Convert dc_ori to int for consistent mapping
    orders_df['dc_ori'] = pd.to_numeric(orders_df['dc_ori'], errors='coerce')
    orders_df = orders_df[orders_df['dc_ori'].notna()].copy()
    orders_df['dc_ori'] = orders_df['dc_ori'].astype(int)
    
    # Get dc_ori coordinates
    lon1 = orders_df['dc_ori'].map(dc_map_lon).astype(float).values
    lat1 = orders_df['dc_ori'].map(dc_map_lat).astype(float).values
    
    # Get customer coordinates
    lon2 = orders_df['customer_lon'].astype(float).values
    lat2 = orders_df['customer_lat'].astype(float).values
    
    # Vectorized Haversine (only for valid coordinates)
    valid_mask = ~(np.isnan(lon1) | np.isnan(lat1) | np.isnan(lon2) | np.isnan(lat2))
    
    distances_km = np.full(len(orders_df), np.nan)
    if valid_mask.any():
        lon1_rad = np.radians(lon1[valid_mask])
        lat1_rad = np.radians(lat1[valid_mask])
        lon2_rad = np.radians(lon2[valid_mask])
        lat2_rad = np.radians(lat2[valid_mask])
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distances_km[valid_mask] = 6371.0 * c
    
    # Create distance lookup table: (dc_id, customer_zip5) -> distance_km
    records = []
    for pos_idx, (_, row) in enumerate(orders_df.iterrows()):
        if valid_mask[pos_idx]:
            dc_id = int(row['dc_ori'])
            customer_zip5 = str(row['customer_zip5']) if pd.notna(row['customer_zip5']) else None
            if customer_zip5:
                records.append({
                    'dc_id': dc_id,
                    'customer_zip5': customer_zip5,
                    'distance_km': float(distances_km[pos_idx]),
                })
    
    if records:
        dist_df = pd.DataFrame(records).drop_duplicates(subset=['dc_id', 'customer_zip5'])
        output_path = output_dir / 'dc_zip5_distance.feather'
        feather.write_feather(dist_df, output_path)
        print(f"  Saved to {output_path}")
    else:
        print("  Warning: No valid distances computed")


def build_static_features(output_dir: Path, order_set: str = 'test'):
    """Build static features for orders."""
    print("Building static features...")
    
    # Load and split orders from preprocessed_cs
    orders_df = load_and_split_data(order_set)
    
    orders_df['order_time'] = pd.to_datetime(orders_df['order_time'])
    orders_df['order_hour'] = orders_df['order_time'].dt.hour
    orders_df['weekday'] = orders_df['order_time'].dt.dayofweek
    orders_df['num_skus_in_order'] = orders_df.groupby('order_ID')['sku_ID'].transform('nunique')
    
    # Handle optional discount columns
    if 'bundle_discount_per_unit' in orders_df.columns:
        orders_df['has_bundle_discount'] = (orders_df['bundle_discount_per_unit'] > 0).astype(int)
    else:
        orders_df['has_bundle_discount'] = 0
    
    if 'coupon_discount_per_unit' in orders_df.columns:
        orders_df['has_coupon_discount'] = (orders_df['coupon_discount_per_unit'] > 0).astype(int)
    else:
        orders_df['has_coupon_discount'] = 0
    
    if 'gift_item' in orders_df.columns:
        orders_df['has_gift_item'] = orders_df.groupby('order_ID')['gift_item'].transform('max').astype(int)
    else:
        orders_df['has_gift_item'] = 0
    
    orders_df['total_quantity_in_order'] = orders_df.groupby('order_ID')['quantity'].transform('sum')
    
    # Aggregate to order level
    order_features = orders_df.groupby('order_ID').agg({
        feat: 'first' for feat in cfg.STATIC_FEATURES if feat in orders_df.columns
    }).reset_index()
    
    # Rename order_ID to order_id for consistency with PrecomputeStore
    order_features.rename(columns={'order_ID': 'order_id'}, inplace=True)
    
    # Fill missing features
    for feat in cfg.STATIC_FEATURES:
        if feat not in order_features.columns:
            order_features[feat] = 0.0
    
    output_path = output_dir / 'orders_static_features.parquet'
    order_features.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")


def build_order_dc_features(output_dir: Path, order_set: str = 'test'):
    """Build per-order candidate DC features for delivery-time inference."""
    print("Building order-DC feature grid...")
    orders_df = load_and_split_data(order_set)
    if orders_df.empty:
        print("  No orders available; skipping order-DC features.")
        return

    # Ensure required columns are present
    required_defaults = {
        'bundle_discount_per_unit': 0.0,
        'coupon_discount_per_unit': 0.0,
        'gift_item': 0,
        'type': 'standard',
        'user_level': 'unknown',
        'city_level': 'unknown',
        'plus': 0,
        'dc_des': np.nan,
        'promise_delivery_days': 0,
        'ship_out_time': pd.NaT,
        'original_unit_price': 1.0,
        'final_unit_price': 1.0,
    }
    for col, default in required_defaults.items():
        if col not in orders_df.columns:
            orders_df[col] = default

    orders_df['order_time'] = pd.to_datetime(orders_df['order_time'])
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    orders_df['ship_out_time'] = pd.to_datetime(orders_df['ship_out_time'])
    orders_df['ship_out_time'] = orders_df['ship_out_time'].fillna(orders_df['order_time'])
    orders_df['type'] = orders_df['type'].astype(str)
    orders_df['user_level'] = orders_df['user_level'].astype(str)
    orders_df['city_level'] = orders_df['city_level'].astype(str)
    orders_df['plus'] = pd.to_numeric(orders_df['plus'], errors='coerce').fillna(0).astype(int)

    orders_df.sort_values('order_time', inplace=True)
    orders_df['order_hour'] = orders_df['order_time'].dt.hour.astype(int)
    orders_df['weekday'] = orders_df['order_time'].dt.dayofweek.astype(int)
    orders_df['num_skus_in_order'] = orders_df.groupby('order_ID')['sku_ID'].transform('nunique')
    orders_df['has_bundle_discount'] = (orders_df['bundle_discount_per_unit'] > 0).astype(int)
    orders_df['has_coupon_discount'] = (orders_df['coupon_discount_per_unit'] > 0).astype(int)
    orders_df['has_gift_item'] = orders_df.groupby('order_ID')['gift_item'].transform('max').fillna(0).astype(int)
    orders_df['total_quantity_in_order'] = orders_df.groupby('order_ID')['quantity'].transform('sum')
    orders_df['discount_rate'] = (
        -1.0 * (orders_df['final_unit_price'] - orders_df['original_unit_price'])
        / orders_df['original_unit_price']
    )
    discount_rate = orders_df['discount_rate'].replace([np.inf, -np.inf], 0)
    discount_rate = discount_rate.fillna(0)
    orders_df['discount_rate'] = discount_rate
    orders_df['avg_discount_rate_in_order'] = (
        orders_df.groupby('order_ID')['discount_rate'].transform('mean').round(4)
    )
    orders_df['dc_des'] = pd.to_numeric(orders_df['dc_des'], errors='coerce').fillna(-1).astype(int)

    snapshot = build_dc_event_snapshot(orders_df.copy())
    snapshot = snapshot[[
        'order_time',
        'dc_ori',
        'waiting_orders',
        'waiting_skus',
        'shipped_orders_last_2h',
        'shipped_skus_last_2h',
    ]].sort_values(['dc_ori', 'order_time'])

    base_cols = [
        'order_ID', 'order_time', 'order_date', 'order_hour', 'weekday',
        'num_skus_in_order', 'has_bundle_discount', 'has_coupon_discount',
        'type', 'has_gift_item', 'total_quantity_in_order',
        'avg_discount_rate_in_order', 'user_level', 'city_level', 'plus',
        'promise_delivery_days', 'dc_des'
    ]
    order_base = (
        orders_df[base_cols]
        .drop_duplicates(subset=['order_ID'])
        .rename(columns={'order_ID': 'order_id'})
    )
    order_base_sorted = order_base.sort_values('order_time').reset_index(drop=True)

    metadata_df, _ = load_dc_carrier_metadata()
    catalog_dcs = metadata_df['dc_id'].dropna().astype(int).unique().tolist()
    observed_dcs = orders_df['dc_ori'].dropna().astype(int).unique().tolist()
    candidate_dcs = sorted(set(catalog_dcs) | set(observed_dcs))
    if not candidate_dcs:
        print("  No candidate DCs found; skipping.")
        return

    feature_dir = Path(cfg.PRECOMPUTED_ORDER_DC_FEATURES_DIR) / order_set
    feature_dir.mkdir(parents=True, exist_ok=True)
    for existing in feature_dir.glob("dc=*.parquet"):
        existing.unlink()

    dynamic_cols = [
        'waiting_orders',
        'waiting_skus',
        'shipped_orders_last_2h',
        'shipped_skus_last_2h',
    ]

    print(f"  Generating features for {len(candidate_dcs)} DCs...")
    for dc in candidate_dcs:
        snap_dc = snapshot[snapshot['dc_ori'] == dc][['order_time'] + dynamic_cols].copy()
        if snap_dc.empty:
            df_dc = order_base_sorted.copy()
            for col in dynamic_cols:
                df_dc[col] = 0.0
        else:
            merged = pd.merge_asof(
                order_base_sorted,
                snap_dc.sort_values('order_time'),
                on='order_time',
                direction='backward',
                allow_exact_matches=True,
            )
            merged[dynamic_cols] = merged[dynamic_cols].fillna(0.0)
            df_dc = merged
        df_dc['candidate_dc'] = dc
        df_dc['dc_ori'] = dc
        output_path = feature_dir / f"dc={dc}.parquet"
        df_dc.to_parquet(output_path, index=False)
    print(f"  Saved order-DC features to {feature_dir}")


def build_processing_rates(output_dir: Path, order_set: str = 'proxy_train'):
    """Build DC-level processing rate buckets for queueing approximations."""
    print("Building processing rate table...")
    
    orders_df = load_and_split_data(order_set)
    required_cols = {'order_ID', 'dc_ori', 'order_time', 'ship_out_time', 'quantity'}
    missing_cols = required_cols - set(orders_df.columns)
    if missing_cols:
        print(f"  Skipping processing rate build; missing columns: {missing_cols}")
        return
    
    df = orders_df[list(required_cols)].copy()
    df['order_time'] = pd.to_datetime(df['order_time'])
    df['ship_out_time'] = pd.to_datetime(df['ship_out_time'])
    df = df.dropna(subset=['dc_ori', 'order_time', 'ship_out_time', 'quantity'])
    if df.empty:
        print("  No usable historical rows for processing rates; skipping")
        return
    
    df['dc_id'] = pd.to_numeric(df['dc_ori'], errors='coerce').astype('Int64')
    df = df[df['dc_id'].notna()].copy()
    df['dc_id'] = df['dc_id'].astype(int)
    
    order_level = (
        df.groupby(['order_ID', 'dc_id'])
        .agg({
            'order_time': 'first',
            'ship_out_time': 'last',
            'quantity': 'sum'
        })
        .reset_index()
        .rename(columns={'order_ID': 'order_id', 'quantity': 'total_quantity'})
    )
    order_level.dropna(subset=['order_time', 'ship_out_time', 'total_quantity'], inplace=True)
    order_level = order_level[order_level['total_quantity'] > 0].copy()
    if order_level.empty:
        print("  No orders with positive quantity for processing rates; skipping")
        return
    
    order_level['handle_minutes'] = (
        (order_level['ship_out_time'] - order_level['order_time']).dt.total_seconds() / 60.0
    )
    order_level = order_level[order_level['handle_minutes'] >= 0].copy()
    if order_level.empty:
        print("  No orders with non-negative handle times; skipping")
        return
    
    order_level['handle_minutes'] = order_level['handle_minutes'].clip(
        lower=1.0,
        upper=cfg.PROCESSING_MAX_SERVICE_MINUTES
    )
    order_level['order_hour'] = order_level['order_time'].dt.hour.astype(int)
    order_level['is_weekend'] = (order_level['order_time'].dt.dayofweek >= 5).astype(int)
    
    sentinel = cfg.PROCESSING_SENTINEL_VALUE
    group_specs = [
        ('dc_hour_weekend', ['dc_id', 'order_hour', 'is_weekend']),
        ('dc_hour', ['dc_id', 'order_hour']),
        ('dc_weekend', ['dc_id', 'is_weekend']),
        ('dc', ['dc_id']),
        ('hour_weekend', ['order_hour', 'is_weekend']),
        ('hour', ['order_hour']),
        ('weekend', ['is_weekend']),
        ('global', []),
    ]
    
    aggregates = []
    
    def aggregate(level_name: str, columns: List[str]):
        if columns:
            grouped = (
                order_level
                .groupby(columns, dropna=False)
                .agg(
                    total_quantity=('total_quantity', 'sum'),
                    handle_minutes=('handle_minutes', 'sum'),
                    samples=('order_id', 'nunique'),
                )
                .reset_index()
            )
        else:
            grouped = pd.DataFrame([{
                'total_quantity': order_level['total_quantity'].sum(),
                'handle_minutes': order_level['handle_minutes'].sum(),
                'samples': order_level['order_id'].nunique(),
            }])
        agg = grouped[(grouped['total_quantity'] > 0) & (grouped['handle_minutes'] > 0)].copy()
        if agg.empty:
            return
        agg['avg_minutes_per_unit'] = agg['handle_minutes'] / agg['total_quantity']
        agg['units_per_minute'] = agg['total_quantity'] / agg['handle_minutes']
        for col in ['dc_id', 'order_hour', 'is_weekend']:
            if col not in columns:
                agg[col] = sentinel
        agg['level'] = level_name
        aggregates.append(agg[['dc_id', 'order_hour', 'is_weekend', 'level',
                               'avg_minutes_per_unit', 'units_per_minute', 'samples']])
    
    for level_name, cols in group_specs:
        aggregate(level_name, cols)
    
    if not aggregates:
        print("  No aggregates produced for processing rates; skipping")
        return
    
    rates_df = pd.concat(aggregates, ignore_index=True)
    output_path = output_dir / 'dc_processing_rates.parquet'
    rates_df.to_parquet(output_path, index=False)
    print(f"  Saved processing rates to {output_path} ({len(rates_df)} rows)")


def main():
    parser = argparse.ArgumentParser(description='Build precomputed artifacts for simulation')
    parser.add_argument('--order_set', type=str, default='test', choices=['test', 'proxy_train'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--skip-order-dc-features', action='store_true', help='Skip generating order-DC feature grid (writes to shared data/ by default).')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else cfg.SIM_PRECOMPUTE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building precomputed artifacts in {output_dir}...")
    
    build_options_catalog(output_dir)
    build_coverage_allowed_states(output_dir)
    build_distances(output_dir, args.order_set)
    build_static_features(output_dir, args.order_set)
    if args.skip_order_dc_features:
        print("Skipping order-DC feature grid generation (--skip-order-dc-features).")
    else:
        build_order_dc_features(output_dir, args.order_set)
    build_processing_rates(output_dir, args.order_set)
    
    print("Done!")


if __name__ == '__main__':
    main()
