"""
Precompute base shipping cost tensor for all (customer_node, DC, Carrier) combinations.

This script creates a lookup table to replace on-the-fly haversine distance computation
during proxy feature engineering, significantly speeding up the pipeline.

Strategy:
1. Load customer location data (either hub-based coords or lat/lon grid)
2. Compute haversine distance from each customer node to all DCs
3. Apply carrier-specific cost coefficients and DC fixed costs
4. Save as a tensor: [N_nodes, D, C]

Usage:
  python -m scripts.precompute.precompute_base_cost_tensor

Output:
  data/proxy_data/precomputed_base_cost_tensor.pt
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

import src.config as cfg
from src.data_utils import load_dc_carrier_metadata, get_central_warehouse_ids
from src.simulator.precompute import PrecomputeStore
from src.simulator.catalog import OptionsCatalog


def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    """
    Compute haversine distance from one point to multiple points (vectorized).
    
    Args:
        lat1, lon1: Single point coordinates (scalars)
        lat2_array, lon2_array: Array of coordinates [N]
    
    Returns:
        distances: [N] array of distances in km
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_array)
    lon2_rad = np.radians(lon2_array)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = 6371.0 * c  # Earth radius in km
    
    return distance_km


def load_customer_nodes(use_hubs=True):
    """
    Load customer node definitions (either hubs or grid cells).
    
    Returns:
        nodes_df: DataFrame with columns [node_id, lat, lon]
        node_type: 'hub' or 'grid'
    """
    if use_hubs:
        # Use existing hub-based coordinates (already used in sim)
        hub_path = cfg.HUB_BASED_COORDS_PATH
        if not hub_path.exists():
            raise FileNotFoundError(
                f"Hub coordinates not found: {hub_path}\n"
                f"Run: python -m src.data_augmentation.generate_hub_based_coords"
            )
        
        hubs = pd.read_csv(hub_path)
        
        # Standardize column names
        if 'dc_id' in hubs.columns:
            hubs = hubs.rename(columns={'dc_id': 'node_id'})
        elif 'dc_ID' in hubs.columns:
            hubs = hubs.rename(columns={'dc_ID': 'node_id'})
        
        nodes_df = hubs[['node_id', 'lat', 'lon']].copy()
        nodes_df['node_id'] = nodes_df['node_id'].astype(int)
        node_type = 'hub'
        
        print(f"Loaded {len(nodes_df)} hub nodes from {hub_path}")
        return nodes_df, node_type
    else:
        # TODO: Implement lat/lon grid-based nodes
        raise NotImplementedError("Grid-based customer nodes not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Precompute base shipping cost tensor")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/proxy_data/precomputed_base_cost_tensor.pt)"
    )
    parser.add_argument(
        "--precompute-dir",
        type=str,
        default=None,
        help="Precompute dir containing options_catalog.parquet (default: SIM_PRECOMPUTE_DIR)"
    )
    parser.add_argument(
        "--use-hubs",
        action="store_true",
        default=True,
        help="Use hub-based coordinates for customer nodes (default: True)"
    )
    args = parser.parse_args()
    
    # Set output path
    output_path = Path(args.output) if args.output else cfg.PROXY_DATA_DIR / "precomputed_base_cost_tensor.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PRECOMPUTING BASE COST TENSOR")
    print("="*70)
    
    # Load customer nodes
    print("\n[1/4] Loading customer nodes...")
    nodes_df, node_type = load_customer_nodes(use_hubs=args.use_hubs)
    all_nodes = sorted(nodes_df['node_id'].unique())
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    
    # Load DC metadata
    print("\n[2/4] Loading DC metadata...")
    dc_metadata, cost_models = load_dc_carrier_metadata()
    precompute_dir = Path(args.precompute_dir) if args.precompute_dir else None
    precompute_store = PrecomputeStore(precompute_dir=precompute_dir)
    catalog = OptionsCatalog(precompute_store=precompute_store)
    all_dcs = sorted({int(opt.dc_id) for opt in catalog.all_options})
    num_dcs = len(all_dcs)
    
    # Get carriers from catalog universe (aligns with simulation/CSAA eligibility)
    all_carriers = sorted({int(opt.carrier_service_id) for opt in catalog.all_options})
    num_carriers = len(all_carriers)
    
    # Get central warehouse IDs
    central_ids = get_central_warehouse_ids()
    fixed_local = cfg.LOCAL_DC_BASE_COST_PER_UNIT
    fixed_central = cfg.CENTRAL_WAREHOUSE_BASE_COST_PER_UNIT
    
    print(f"  Nodes: {num_nodes} ({node_type})")
    print(f"  DCs: {num_dcs}")
    print(f"  Carriers: {num_carriers}")
    print(f"  Total combinations: {num_nodes * num_dcs * num_carriers:,}")
    
    # Prepare DC arrays for vectorization
    print("\n[3/4] Preparing DC coordinate arrays...")
    dc_lat_array = np.zeros(num_dcs, dtype=np.float32)
    dc_lon_array = np.zeros(num_dcs, dtype=np.float32)
    is_central_array = np.zeros(num_dcs, dtype=bool)
    
    for d_idx, dc_id in enumerate(all_dcs):
        dc_row = dc_metadata[dc_metadata['dc_id'] == dc_id].iloc[0]
        dc_lat_array[d_idx] = float(dc_row['lat'])
        dc_lon_array[d_idx] = float(dc_row['lon'])
        is_central_array[d_idx] = dc_id in central_ids
    
    # Fixed cost vector per DC
    fixed_cost_array = np.where(is_central_array, fixed_central, fixed_local)
    
    # Carrier coefficient array
    carrier_coef_array = np.array([cost_models.get(c, 0.0) for c in all_carriers], dtype=np.float32)
    
    # Precompute base cost tensor
    print("\n[4/4] Computing base costs...")
    base_cost_tensor = np.zeros((num_nodes, num_dcs, num_carriers), dtype=np.float32)
    
    for n_idx, node_id in enumerate(tqdm(all_nodes, desc="Processing nodes")):
        # Get node coordinates
        node_row = nodes_df[nodes_df['node_id'] == node_id].iloc[0]
        node_lat = float(node_row['lat'])
        node_lon = float(node_row['lon'])
        
        # Vectorized distance computation to all DCs
        distance_km = haversine_vectorized(node_lat, node_lon, dc_lat_array, dc_lon_array)  # [D]
        
        # Broadcast to compute base costs for all carriers
        # base_cost[d, c] = distance_km[d] * coef[c] + fixed_cost[d]
        for c_idx in range(num_carriers):
            base_cost_tensor[n_idx, :, c_idx] = (
                distance_km * carrier_coef_array[c_idx] + fixed_cost_array
            )
    
    # Save tensor
    print(f"\n[5/5] Saving to {output_path}...")
    torch.save(
        {
            "base_cost_tensor": torch.from_numpy(base_cost_tensor),  # [N_nodes, D, C]
            "nodes": all_nodes,
            "dcs": all_dcs,
            "carriers": all_carriers,
            "node_type": node_type,
            "num_nodes": num_nodes,
            "num_dcs": num_dcs,
            "num_carriers": num_carriers,
        },
        output_path,
    )
    
    # Print statistics
    print("\n" + "="*70)
    print("PRECOMPUTATION COMPLETE")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"Tensor shape: {base_cost_tensor.shape}")
    print(f"Memory size: {base_cost_tensor.nbytes / 1024 / 1024:.2f} MB")
    print(f"\nBase cost statistics:")
    print(f"  Min: {base_cost_tensor.min():.2f}")
    print(f"  Max: {base_cost_tensor.max():.2f}")
    print(f"  Mean: {base_cost_tensor.mean():.2f}")
    print(f"  Median: {np.median(base_cost_tensor):.2f}")
    print("\nUsage:")
    print("  1. Run preprocessing to assign customer_node IDs (see docs)")
    print("  2. Run feature engineering with --use-precomputed-base-cost flag")
    print("="*70)


if __name__ == "__main__":
    main()
