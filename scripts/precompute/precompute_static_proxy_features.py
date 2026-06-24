"""
Precompute static proxy features for fast inference.

Static features don't depend on current simulation state:
- Global/order features (demographics, order hour, etc.)
- Base costs [D, C] (distance-based + fixed costs)
- DC distances from customer
- Region match vectors
- Historical SKU demand (for DoS calculation)
- SKU/brand embeddings indices

Dynamic features (computed at inference time):
- Current inventory vectors
- Days of supply (depends on current inventory)
- Consolidation potential (depends on current basket inventory)
- Scenario demand/delivery penalty (stochastic)
- Dynamic ops features (waiting orders, etc.)

Usage:
    python -m scripts.precompute.precompute_static_proxy_features --order_set test --date 2016-02-15
"""

import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import src.config as cfg
from src.data_utils import (
    preprocess_proxy_features,
    load_dc_carrier_metadata,
    compute_base_costs_for_order,
    compute_global_features_for_order,
    get_observed_dcs_from_preprocessed,
)
from src.simulator.precompute import PrecomputeStore
from src.simulator.catalog import OptionsCatalog


def precompute_static_features(order_set: str, date: str):
    """Precompute static features for all orders on a given date."""
    
    print(f"Precomputing static features for {order_set}/{date}")
    
    # Load data
    if order_set == "test":
        orders_df = pd.read_csv(cfg.DELIVERY_TEST_PATH)
    elif order_set == "proxy_train":
        orders_df = pd.read_csv(cfg.DELIVERY_PROXY_TRAIN_PATH)
    else:
        raise ValueError(f"Invalid order_set: {order_set}")
    
    # Filter to date
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    orders_df = orders_df[orders_df['order_date'] == date].copy()
    
    if orders_df.empty:
        print(f"No orders found for {date}")
        return
    
    print(f"Found {len(orders_df)} order items ({orders_df['order_ID'].nunique()} orders)")
    
    # Load metadata
    network_df = pd.read_csv(cfg.NETWORK_PATH)
    dc_region_map = network_df.set_index('dc_ID')['region_ID'].to_dict()
    dc_metadata, cost_coef_map = load_dc_carrier_metadata()
    precompute_store = PrecomputeStore()
    catalog = OptionsCatalog(precompute_store=precompute_store)
    catalog_dcs = sorted({int(opt.dc_id) for opt in catalog.all_options})
    carriers = sorted({int(opt.carrier_service_id) for opt in catalog.all_options})
    observed_dcs = get_observed_dcs_from_preprocessed()
    if observed_dcs:
        filtered_dcs = [dc for dc in catalog_dcs if dc in observed_dcs]
        dropped = [dc for dc in catalog_dcs if dc not in observed_dcs]
        if dropped:
            print(
                f"Filtered DC universe to observed preprocessed DCs: "
                f"{len(filtered_dcs)}/{len(catalog_dcs)} (dropped {dropped})"
            )
        dcs = filtered_dcs
    else:
        dcs = catalog_dcs
    option_space = len(dcs) * len(carriers)
    expected_space = len(observed_dcs) * len(carriers) if observed_dcs else None
    if expected_space and option_space == expected_space:
        print(f"Option space: {len(dcs)} dcs x {len(carriers)} carriers = {option_space}")
    else:
        expected_note = f"{len(observed_dcs)}x{len(carriers)}={expected_space}" if expected_space else "55x17=935"
        print(
            f"Option space: {len(dcs)} dcs x {len(carriers)} carriers = {option_space} "
            f"(expected {expected_note})"
        )
    
    dc_meta_dict = {
        int(row['dc_id']): {'lat': float(row['lat']), 'lon': float(row['lon'])}
        for _, row in dc_metadata.iterrows()
    }
    
    # Preprocess features
    pre = pd.read_csv(cfg.PREPROCESSED_PATH)
    pre['order_time'] = pd.to_datetime(pre['order_time'])
    pre = preprocess_proxy_features(pre)
    
    # Factorize
    for col in cfg.PROXY_CATEGORICAL_ORDER_FEATURES:
        pre[col], _ = pd.factorize(pre[col], sort=True)
    pre["sku_idx"], _ = pd.factorize(pre[cfg.SKU_COL], sort=True)
    pre["brand_idx"], _ = pd.factorize(pre[cfg.BRAND_COL], sort=True)
    
    # Compute historical stats (for DoS and peak detection)
    current_date_dt = pd.to_datetime(date)
    historical_data = pre[pre['order_time'] < current_date_dt].copy()
    
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
    
    # Precompute static features per (order, sku)
    static_features = {}
    
    for order_id in tqdm(orders_df['order_ID'].unique(), desc="Processing orders"):
        order_items = orders_df[orders_df['order_ID'] == order_id]
        order_pre = pre[pre['order_ID'] == order_id]
        
        if order_pre.empty:
            continue
        
        # Extract customer location
        customer_lat = order_pre['customer_lat'].iloc[0] if 'customer_lat' in order_pre.columns else 0.0
        customer_lon = order_pre['customer_lon'].iloc[0] if 'customer_lon' in order_pre.columns else 0.0
        customer_dc = order_pre['dc_des'].iloc[0] if 'dc_des' in order_pre.columns else None
        order_hour = order_pre['order_hour'].iloc[0] if 'order_hour' in order_pre.columns else 0
        
        # Compute base costs [D, C] (same for all SKUs in order)
        base_cost_grid = compute_base_costs_for_order(
            customer_lat, customer_lon,
            dcs, carriers,
            dc_metadata, cost_coef_map
        )
        
        # Compute DC distances [D] (same for all SKUs in order)
        distance_vec = np.zeros(len(dcs), dtype=np.float32)
        for d_idx, dc_id in enumerate(dcs):
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
        
        # Compute region match [D] (same for all SKUs in order)
        cust_region = dc_region_map.get(customer_dc, None)
        region_match_vec = np.zeros(len(dcs), dtype=np.float32)
        if cust_region is not None:
            for d_idx, dc_id in enumerate(dcs):
                if dc_region_map.get(dc_id, None) == cust_region:
                    region_match_vec[d_idx] = 1.0
        
        # Global order volume
        global_volume = hourly_volume.get(order_hour, 0)
        
        # Per-SKU features
        order_skus = order_items['sku_ID'].unique()
        order_pre_indexed = order_pre.set_index('sku_ID').reindex(order_skus).reset_index()
        
        for idx, sku in enumerate(order_skus):
            sku_row = order_pre_indexed.iloc[idx]
            qty = order_items[order_items['sku_ID'] == sku]['quantity'].sum()
            
            # Global features
            order_feat_vec = sku_row[cfg.PROXY_ORDER_FEATURES].to_dict()
            global_feats = compute_global_features_for_order(
                order_feat_vec, qty, global_volume, base_cost_grid=base_cost_grid
            )
            
            # Historical daily demand for this SKU
            sku_daily_demand = sku_daily_demand_dict.get(sku, 0.0)
            
            # Store
            static_features[(order_id, sku)] = {
                'global_features': np.array(list(global_feats.values()), dtype=np.float32),
                'base_cost_grid': base_cost_grid.astype(np.float32),
                'distance_vec': distance_vec,
                'region_match_vec': region_match_vec,
                'sku_daily_demand': float(sku_daily_demand),
                'sku_idx': int(sku_row['sku_idx']),
                'brand_idx': int(sku_row['brand_idx']),
                'quantity': float(qty),
            }
    
    # Save
    output_dir = cfg.PROXY_DATA_DIR / "static_features" / order_set
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"static_{date}.pt"
    
    torch.save({
        'features': static_features,
        'dcs': dcs,
        'carriers': carriers,
        'dc_meta_dict': dc_meta_dict,
        'dc_region_map': dc_region_map,
        'date': date,
        'order_set': order_set,
        'num_orders': len(static_features),
    }, output_path)
    
    print(f"\nSaved static features for {len(static_features)} (order, sku) pairs")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Precompute static proxy features")
    parser.add_argument("--order_set", choices=["test", "proxy_train"], required=True)
    parser.add_argument("--date", help="Date (YYYY-MM-DD)")
    parser.add_argument("--start_date", help="Start date for batch processing")
    parser.add_argument("--end_date", help="End date for batch processing")
    args = parser.parse_args()
    
    if args.order_set == "test":
        orders_df = pd.read_csv(cfg.DELIVERY_TEST_PATH)
    else:
        orders_df = pd.read_csv(cfg.DELIVERY_PROXY_TRAIN_PATH)

    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    dates = sorted(orders_df['order_date'].dt.strftime('%Y-%m-%d').unique())

    if args.start_date and args.end_date:
        dates = [d for d in dates if args.start_date <= d <= args.end_date]
        print(f"Processing {len(dates)} dates from {args.start_date} to {args.end_date}")
    elif args.date:
        dates = [args.date]
        print(f"Processing single date {args.date}")
    else:
        print(f"Processing all dates ({len(dates)}) for order_set={args.order_set}")

    for date in dates:
        precompute_static_features(args.order_set, date)


if __name__ == "__main__":
    main()
