#!/usr/bin/env python3
"""
Sample carrier-services for each order and scale delivery times accordingly.
Also generates random customer zip5 codes based on destination DC zip3.

Usage:
    python -m src.data_augmentation.sample_carrier_services [--input PATH] [--output PATH] [--seed N]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pgeocode

# Import distance calculation utilities
from src.data_augmentation.pseudo_distance import load_dc_coords


def assign_distance_bin(distance_km: float, bin_edges: list[float]) -> int:
    """Assign distance to bin based on edges.
    
    Bins: 0: <edges[0], 1: edges[0]-edges[1], ..., len(edges): >=edges[-1]
    """
    for i, edge in enumerate(bin_edges):
        if distance_km < edge:
            return i
    return len(bin_edges)


def generate_zip5(
    zip3: str, 
    rng: np.random.Generator,
    valid_zip5_by_zip3: dict[str, list[str]] | None = None
) -> str:
    """Generate a random zip5 code from a zip3 code.
    
    Prioritizes valid zip5 codes from zip5_geocoding.json if available.
    
    Args:
        zip3: 3-digit zip code (may have leading zeros)
        rng: Random number generator
        valid_zip5_by_zip3: Optional dict mapping zip3 -> list of valid zip5 codes
        
    Returns:
        5-digit zip code (zip3 + random 2-digit suffix 00-99, or from valid list)
    """
    if zip3 is None or pd.isna(zip3):
        return None
    
    # Ensure zip3 is formatted as 3-digit string with leading zeros
    zip3_str = str(zip3).strip()
    if zip3_str.isdigit():
        zip3_str = f"{int(zip3_str):03d}"
    else:
        return None
    
    # First try: use valid zip5 codes from zip5_geocoding.json if available
    if valid_zip5_by_zip3 and zip3_str in valid_zip5_by_zip3:
        valid_zip5s = valid_zip5_by_zip3[zip3_str]
        if valid_zip5s:
            return rng.choice(valid_zip5s)
    
    # Fallback: generate random 2-digit suffix (00-99)
    suffix = rng.integers(0, 100)
    return f"{zip3_str}{suffix:02d}"


def load_or_create_zip5_mapping(
    zip5_codes: list[str], 
    mapping_path: Path, 
    rng: np.random.Generator,
    zip5_geocoding_path: Path | None = None,
    valid_zip5_by_zip3: dict[str, list[str]] | None = None,
    max_retries: int = 10
) -> dict:
    """Load or create zip5 to lat/lon/state mapping.
    
    Uses cached mapping file and zip5_geocoding.json if available, only queries missing zip5 codes.
    
    Args:
        zip5_codes: List of zip5 codes to query (may contain duplicates/None)
        mapping_path: Path to JSON file to save/load mapping cache
        rng: Random number generator for retrying failed queries
        zip5_geocoding_path: Optional path to zip5_geocoding.json with additional zip5 mappings
        valid_zip5_by_zip3: Optional dict mapping zip3 -> list of valid zip5 codes for retry logic
        max_retries: Maximum retry attempts per zip3
        
    Returns:
        Dictionary mapping zip5 -> {'latitude': float, 'longitude': float, 'state': str}
    """
    # Load existing mapping from cache
    zip5_map = {}
    if mapping_path.exists():
        try:
            with open(mapping_path, 'r') as f:
                cached_map = json.load(f)
            # Normalize keys to strings
            zip5_map = {str(k): v for k, v in cached_map.items()}
            print(f"  Loaded {len(zip5_map):,} zip5 mappings from cache")
        except Exception as e:
            print(f"  Warning: Could not load cache ({e}), starting fresh")
    
    # Load zip5_geocoding.json as additional source
    if zip5_geocoding_path and zip5_geocoding_path.exists():
        try:
            with open(zip5_geocoding_path, 'r') as f:
                geocoding_map = json.load(f)
            # Normalize keys to strings and merge into zip5_map (don't overwrite existing)
            geocoding_map_normalized = {str(k): v for k, v in geocoding_map.items()}
            n_added = sum(1 for z in geocoding_map_normalized if z not in zip5_map)
            zip5_map.update(geocoding_map_normalized)
            if n_added > 0:
                print(f"  Added {n_added:,} zip5 mappings from {zip5_geocoding_path.name}")
        except Exception as e:
            print(f"  Warning: Could not load {zip5_geocoding_path.name} ({e})")
    
    # Find missing zip5 codes or those with invalid coordinates
    unique_zip5 = sorted(set(str(z) for z in zip5_codes if z is not None))
    missing_zip5 = []
    invalid_in_map = []
    for z in unique_zip5:
        entry = zip5_map.get(z)
        if entry is None:
            missing_zip5.append(z)
        elif isinstance(entry, dict):
            lat = entry.get('latitude')
            lon = entry.get('longitude')
            if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
                invalid_in_map.append(z)
    
    if invalid_in_map:
        print(f"  Found {len(invalid_in_map):,} existing zip5 codes with invalid coords; will re-query")
        missing_zip5.extend(invalid_in_map)
    
    if not missing_zip5:
        print(f"  All {len(unique_zip5):,} unique zip5 codes already mapped - skipping geocoding")
        return zip5_map
    
    print(f"  Querying {len(missing_zip5):,} zip5 codes with pgeocode (missing or invalid)...")
    nomi = pgeocode.Nominatim('us')
    geo_query = nomi.query_postal_code(missing_zip5)
    
    # Process results
    failed_by_zip3 = {}  # zip3 -> list of failed zip5 codes
    for zip5, lat, lon, state_code in zip(
        geo_query['postal_code'], geo_query['latitude'], 
        geo_query['longitude'], geo_query['state_code']
    ):
        zip5_str = str(zip5) if zip5 is not None else None
        if pd.notna(lat) and pd.notna(lon) and zip5_str:
            zip5_map[zip5_str] = {
                'latitude': float(lat),
                'longitude': float(lon),
                'state': str(state_code) if pd.notna(state_code) else None
            }
        elif zip5_str and len(zip5_str) >= 3:
            zip3 = zip5_str[:3]
            if zip3 not in failed_by_zip3:
                failed_by_zip3[zip3] = []
            failed_by_zip3[zip3].append(zip5_str)
    
    # Retry failed zip3s with multiple attempts
    if failed_by_zip3:
        print(f"  Retrying {sum(len(v) for v in failed_by_zip3.values()):,} failed queries across {len(failed_by_zip3)} zip3s...")
        zip3_attempt_counts = {zip3: 0 for zip3 in failed_by_zip3.keys()}
        
        for attempt in range(max_retries):
            retry_zip5 = []
            retry_zip3_map = {}
            for zip3 in list(failed_by_zip3.keys()):
                # Prioritize valid zip5 codes from zip5_geocoding.json
                if zip3 in valid_zip5_by_zip3 and valid_zip5_by_zip3[zip3]:
                    valid_options = [str(z) for z in valid_zip5_by_zip3[zip3] if str(z) not in zip5_map]
                    new_zip5 = rng.choice(valid_options) if valid_options else generate_zip5(zip3, rng, valid_zip5_by_zip3)
                else:
                    new_zip5 = generate_zip5(zip3, rng, valid_zip5_by_zip3)
                retry_zip5.append(new_zip5)
                retry_zip3_map[str(new_zip5)] = zip3
                zip3_attempt_counts[zip3] += 1
            
            if not retry_zip5:
                break
            
            retry_query = nomi.query_postal_code(retry_zip5)
            resolved_zip3s = set()
            for zip5, lat, lon, state_code in zip(
                retry_query['postal_code'], retry_query['latitude'],
                retry_query['longitude'], retry_query['state_code']
            ):
                zip5_str = str(zip5) if zip5 is not None else None
                if pd.notna(lat) and pd.notna(lon) and zip5_str:
                    zip5_map[zip5_str] = {
                        'latitude': float(lat),
                        'longitude': float(lon),
                        'state': str(state_code) if pd.notna(state_code) else None
                    }
                    resolved_zip3s.add(retry_zip3_map.get(zip5_str))
            
            # Remove resolved zip3s
            for zip3 in resolved_zip3s:
                del failed_by_zip3[zip3]
            
            if not failed_by_zip3:
                break
        
        # Report diagnostic info for still-failed zip3s
        if failed_by_zip3:
            n_failed = sum(len(v) for v in failed_by_zip3.values())
            print(f"  Warning: {n_failed} zip5 codes still failed after {max_retries} retries")
            print(f"  Failed zip3s ({len(failed_by_zip3)} total):")
            # Show top 10 most problematic zip3s
            zip3_fail_counts = {zip3: len(failed_zip5s) for zip3, failed_zip5s in failed_by_zip3.items()}
            sorted_failed = sorted(zip3_fail_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for zip3, count in sorted_failed:
                attempts = zip3_attempt_counts.get(zip3, 0)
                print(f"    zip3 {zip3}: {count} failed zip5s, {attempts} retry attempts")
            if len(failed_by_zip3) > 10:
                print(f"    ... and {len(failed_by_zip3) - 10} more zip3s")
    
    # Save updated mapping
    if missing_zip5:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, 'w') as f:
            json.dump(zip5_map, f, indent=2)
        print(f"  Updated cache: {len(zip5_map):,} total zip5 mappings saved")
    
    return zip5_map


def sample_carrier_service(dist_bin: int, cs_data: pd.DataFrame, rng: np.random.Generator) -> tuple[int, float]:
    """Sample a carrier-service for given distance bin.
    
    Returns:
        (carrier_service_id, ratio_multiplier)
    """
    # Filter to carrier-services in this bin
    cs_in_bin = cs_data[cs_data['dist_bin'] == dist_bin].copy()
    
    if cs_in_bin.empty:
        # Fallback: use all bins if no data for this specific bin
        cs_in_bin = cs_data.copy()
    
    # Sample uniformly from available carrier-services
    idx = rng.choice(len(cs_in_bin))
    row = cs_in_bin.iloc[idx]
    
    # Use median ratio (r_p050) for delivery time scaling
    carrier_id = row['carrier_service_id_anon']
    ratio = row['r_p050']
    
    return carrier_id, ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        default='data/processed/preprocessed_data.csv',
        help='Input preprocessed data CSV'
    )
    parser.add_argument(
        '--output',
        default='data/processed/preprocessed_data_cs.csv',
        help='Output CSV with carrier-services and scaled delivery times'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--ratios',
        default='data/params/real_ratios_cs.csv',
        help='Carrier-service ratios CSV'
    )
    parser.add_argument(
        '--bins',
        default='data/params/calibration_dist_bins_km.json',
        help='Distance bin edges JSON'
    )
    parser.add_argument(
        '--dc_coords',
        default='data/derived/pseudo_coords/hub_based_coords.csv',
        help='DC coordinates CSV with zip3 column'
    )
    parser.add_argument(
        '--zip5_mapping',
        default='data/params/zip5_geocoding_cache.json',
        help='JSON file to cache zip5 to lat/lon/state mapping'
    )
    parser.add_argument(
        '--zip5_geocoding',
        default='data/params/zip5_geocoding.json',
        help='JSON file with additional zip5 to lat/lon/state mappings'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    rng = np.random.default_rng(args.seed)
    
    print("Loading data...")
    # Load preprocessed data
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df):,} orders")
    
    # Load carrier-service ratios
    cs_ratios = pd.read_csv(args.ratios)
    print(f"  Loaded {len(cs_ratios):,} carrier-service configs")
    
    # Load distance bins
    with open(args.bins, 'r') as f:
        bin_config = json.load(f)
    bin_edges = bin_config['edges_km']
    print(f"  Distance bins: {bin_edges}")
    
    # Load DC coordinates
    dc_coords = load_dc_coords(path=args.dc_coords)
    dc_map_lon = dc_coords.set_index('dc_ID')['x'].to_dict()
    dc_map_lat = dc_coords.set_index('dc_ID')['y'].to_dict()
    
    # Load DC zip3 mapping for customer zip5 generation
    print(f"\nLoading DC zip3 mapping from {args.dc_coords}...")
    dc_coords_with_zip3 = pd.read_csv(args.dc_coords)
    dc_zip3_map = {}
    if 'zip3' in dc_coords_with_zip3.columns:
        # Create lookup: dc_id -> zip3
        for _, row in dc_coords_with_zip3.iterrows():
            dc_id = str(row['dc_id'])
            zip3_val = row.get('zip3')
            if pd.notna(zip3_val):
                # Format zip3 as 3-digit string with leading zeros
                zip3_str = str(zip3_val).strip()
                if zip3_str.isdigit():
                    dc_zip3_map[dc_id] = f"{int(zip3_str):03d}"
        print(f"  Loaded zip3 for {len(dc_zip3_map)} DCs")
    else:
        print(f"  Warning: zip3 column not found in {args.dc_coords}")
    
    # Build zip3 -> valid zip5 codes lookup from zip5_geocoding.json
    valid_zip5_by_zip3 = {}
    zip5_geocoding_path = Path(args.zip5_geocoding) if args.zip5_geocoding else None
    if zip5_geocoding_path and zip5_geocoding_path.exists():
        try:
            with open(zip5_geocoding_path, 'r') as f:
                geocoding_map = json.load(f)
            # Group zip5 codes by zip3 (normalize keys to strings)
            for zip5, _ in geocoding_map.items():
                zip5_str = str(zip5)
                if zip5_str and len(zip5_str) >= 3:
                    zip3 = zip5_str[:3]
                    if zip3 not in valid_zip5_by_zip3:
                        valid_zip5_by_zip3[zip3] = []
                    valid_zip5_by_zip3[zip3].append(zip5_str)
            print(f"  Built valid zip5 lookup: {len(valid_zip5_by_zip3)} zip3s with {sum(len(v) for v in valid_zip5_by_zip3.values()):,} total zip5 codes")
        except Exception as e:
            print(f"  Warning: Could not load {zip5_geocoding_path.name} for zip5 lookup ({e})")
    
    # Generate customer zip5 codes based on destination DC zip3
    print("\nGenerating customer zip5 codes...")
    customer_zip5 = []
    for dc_des in df['dc_des'].astype(str).values:
        zip3 = dc_zip3_map.get(dc_des)
        zip5 = generate_zip5(zip3, rng, valid_zip5_by_zip3)
        customer_zip5.append(zip5)
    
    df['customer_zip5'] = customer_zip5
    n_with_zip5 = sum(1 for z in customer_zip5 if z is not None)
    print(f"  Generated zip5 for {n_with_zip5:,} orders ({n_with_zip5/len(df)*100:.1f}%)")
    
    # Load or create zip5 to lat/lon/state mapping
    print("\nGeocoding customer zip5 codes...")
    mapping_path = Path(args.zip5_mapping)
    zip5_geocoding_path = Path(args.zip5_geocoding) if args.zip5_geocoding else None
    zip5_map = load_or_create_zip5_mapping(
        customer_zip5, mapping_path, rng, zip5_geocoding_path, valid_zip5_by_zip3
    )
    
    # Map zip5 to lat/lon/state for each order (with fallback to any zip5 from same zip3)
    customer_lat = []
    customer_lon = []
    customer_state = []
    
    # Build zip3 -> valid zip5 lookup for fallback (only entries with valid coordinates)
    zip3_to_valid_zip5 = {}
    for zip5, data in zip5_map.items():
        if not isinstance(data, dict):
            continue
        zip5_str = str(zip5)
        if len(zip5_str) < 3:
            continue
        lat = data.get('latitude')
        lon = data.get('longitude')
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            continue
        zip3 = zip5_str[:3]
        if zip3 not in zip3_to_valid_zip5:
            zip3_to_valid_zip5[zip3] = zip5_str
    
    # Augment with valid zip5 codes from zip5_geocoding.json
    if valid_zip5_by_zip3:
        for zip3, valid_zip5s in valid_zip5_by_zip3.items():
            if zip3 not in zip3_to_valid_zip5:
                for zip5 in valid_zip5s:
                    zip5_str = str(zip5)
                    data = zip5_map.get(zip5_str)
                    if isinstance(data, dict):
                        lat = data.get('latitude')
                        lon = data.get('longitude')
                        if lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon):
                            zip3_to_valid_zip5[zip3] = zip5_str
                            break
    
    for zip5 in customer_zip5:
        zip5_str = str(zip5) if zip5 is not None else None
        if not zip5_str:
            customer_lat.append(np.nan)
            customer_lon.append(np.nan)
            customer_state.append(None)
            continue
        
        # Try primary zip5
        entry = zip5_map.get(zip5_str)
        if isinstance(entry, dict):
            lat = entry.get('latitude')
            lon = entry.get('longitude')
            if lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon):
                customer_lat.append(lat)
                customer_lon.append(lon)
                customer_state.append(entry.get('state'))
                continue
        
        # Fallback: try any valid zip5 from same zip3
        if len(zip5_str) >= 3:
            zip3 = zip5_str[:3]
            fallback_zip5 = zip3_to_valid_zip5.get(zip3)
            if fallback_zip5:
                entry = zip5_map.get(str(fallback_zip5))
                if isinstance(entry, dict):
                    lat = entry.get('latitude')
                    lon = entry.get('longitude')
                    if lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon):
                        customer_lat.append(lat)
                        customer_lon.append(lon)
                        customer_state.append(entry.get('state'))
                        continue
        
        # No valid coordinates found
        customer_lat.append(np.nan)
        customer_lon.append(np.nan)
        customer_state.append(None)
    
    df['customer_lat'] = customer_lat
    df['customer_lon'] = customer_lon
    df['customer_state'] = customer_state
    
    n_with_coords = sum(1 for x in customer_lat if pd.notna(x))
    n_missing = len(df) - n_with_coords
    print(f"  Geocoded {n_with_coords:,} orders ({n_with_coords/len(df)*100:.1f}%)")
    if n_missing > 0:
        print(f"  Warning: {n_missing:,} orders not geocoded")
        # Diagnose missing orders
        missing_zip5s = [z for z, lat in zip(customer_zip5, customer_lat) if pd.isna(lat)]
        missing_zip5s_unique = sorted(set(m for m in missing_zip5s if m is not None))
        missing_zip3s = set()
        for zip5 in missing_zip5s_unique:
            if zip5 and len(zip5) >= 3:
                missing_zip3s.add(zip5[:3])
        if missing_zip3s:
            print(f"  Missing zip3s ({len(missing_zip3s)}): {sorted(missing_zip3s)[:10]}{'...' if len(missing_zip3s) > 10 else ''}")
            print(f"  Missing zip5s (sample): {missing_zip5s_unique[:10]}{'...' if len(missing_zip5s_unique) > 10 else ''}")
        
        # Check how many have None zip5
        n_none_zip5 = sum(1 for z in customer_zip5 if z is None)
        if n_none_zip5 > 0:
            print(f"  Orders with None zip5 (no zip3 for dc_des): {n_none_zip5:,}")
    
    # Calculate distances using dc_ori and customer coordinates
    print("\nCalculating distances (dc_ori to customer zip5)...")
    dc_ori_str = df['dc_ori'].astype(str)
    
    lon1 = dc_ori_str.map(dc_map_lon).astype(float).values
    lat1 = dc_ori_str.map(dc_map_lat).astype(float).values
    lon2 = np.array(customer_lon, dtype=float)
    lat2 = np.array(customer_lat, dtype=float)
    
    # Vectorized Haversine (only for valid coordinates)
    valid_mask = ~(np.isnan(lon1) | np.isnan(lat1) | np.isnan(lon2) | np.isnan(lat2))
    
    distances_km = np.full(len(df), np.nan)
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
    
    df['distance_km'] = distances_km
    
    print("\nAssigning carrier-services...")
    
    # Assign distance bins
    df['dist_bin'] = df['distance_km'].apply(lambda d: assign_distance_bin(d, bin_edges) if pd.notna(d) else -1)
    
    print(f"  Distance bin distribution:")
    print(df['dist_bin'].value_counts().sort_index())
    
    # Sample carrier-service for each order
    carrier_ids = []
    ratios = []
    
    for dist_bin in df['dist_bin'].values:
        if dist_bin == -1:
            # Fallback for missing distance: use middle bin
            cs_id, ratio = sample_carrier_service(len(bin_edges) // 2, cs_ratios, rng)
        else:
            cs_id, ratio = sample_carrier_service(dist_bin, cs_ratios, rng)
        carrier_ids.append(cs_id)
        ratios.append(ratio)
    
    df['carrier_service_id_anon'] = carrier_ids
    df['delivery_ratio'] = ratios
    
    print(f"  Carrier-service distribution:")
    print(df['carrier_service_id_anon'].value_counts().sort_index())
    
    # Scale delivery times
    print("\nScaling delivery times...")
    df['delivery_time_hours_original'] = df['delivery_time_hours']
    df['delivery_time_days_original'] = df['delivery_time_days']
    
    df['delivery_time_hours'] = df['delivery_time_hours'] * df['delivery_ratio']
    df['delivery_time_days'] = df['delivery_time_hours'] / 24.0
    
    print(f"  Original mean delivery time: {df['delivery_time_hours_original'].mean():.2f} hours")
    print(f"  Scaled mean delivery time: {df['delivery_time_hours'].mean():.2f} hours")
    
    # Save output
    print(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False)
    print(f"  Saved {len(df):,} orders")
    
    # Summary statistics
    print("\nSummary by carrier-service and distance bin:")
    summary = df.groupby(['carrier_service_id_anon', 'dist_bin']).agg({
        'order_ID': 'count',
        'delivery_ratio': 'first',
        'delivery_time_hours': 'mean'
    }).rename(columns={'order_ID': 'n_orders', 'delivery_time_hours': 'mean_delivery_hours'})
    print(summary)
    
    # Save summary statistics
    summary_path = args.output.replace('.csv', '_summary.csv')
    summary.to_csv(summary_path)
    print(f"\nSummary statistics saved to {summary_path}")


if __name__ == '__main__':
    main()
