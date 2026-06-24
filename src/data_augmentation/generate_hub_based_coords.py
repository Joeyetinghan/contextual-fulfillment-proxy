"""
Generate pseudo US coordinates for DCs using hub-and-spoke model based on shipping times.

This script positions distribution centers around regional central warehouses (hubs).
Each hub has a fixed real-world lat/lon, and local DCs are positioned radially
based on their shipping time from the hub.

Input files:
  - hubs.csv: Central warehouse coordinates (dc_id, lat, lon) or (dc_id, hub_lat, hub_lon),
              optionally includes city_name and zip3
  - region mapping CSV: (region_id, dc_id) or (region_ID, dc_ID), optional role column
  - shipping times CSV: (dc_ori, dc_des, travel_time_hours)
  - cities.csv (optional): Major US cities (city_name/city, lat/latitude, lon/longitude, zip3)

Output:
  - CSV with columns: dc_id, region_id, role, lat, lon, radius_mi_from_hub, 
    bearing_deg_from_hub, ship_time_from_cw, city_name, zip3

Algorithm:
  For each region:
    1. Assign CW the hub's coordinates
    2. Extract shipping times from CW to local DCs
    3. Convert times to quantiles (0=closest, 1=farthest)
    4. Map quantiles to radii in miles (configurable range)
    5. For local DCs:
       If --cities provided (city-based assignment):
         - Find cities within radius tolerance (±tolerance or ±20% of radius)
         - Select closest city within range, or closest overall if none in range
         - Uses haversine distance calculations
       Else (acceptance-rejection sampling):
         - Sample random bearing (0-360°)
         - Compute destination at (radius, bearing) from hub
         - Accept if on US land (not in water); else retry
         - Max 100 attempts per DC (reduces radius if needed)

Usage:
  python -m src.data_augmentation.generate_hub_based_coords \
    --hubs data/params/hubs.csv \
    --regions data/fulfillment/JD_network_data.csv \
    --ship_times data/processed/aggregated_travel_times.csv \
    --output data/derived/pseudo_coords/hub_based_coords.csv
  
  # Use city-based assignment (RECOMMENDED)
  # Note: cities.csv should include zip3 column for zip3 assignment to local DCs
  python -m src.data_augmentation.generate_hub_based_coords \
    --cities data/params/major_us_cities.csv \
    --city_radius_tolerance 50.0
  
  # Snap hubs to US land (if in water/outside boundary)
  python -m src.data_augmentation.generate_hub_based_coords --snap_to_us
  
  # Generate with acceptance-rejection (avoids water, all on land)
  python -m src.data_augmentation.generate_hub_based_coords --snap_to_us --validate_all
"""

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Generate hub-based pseudo coordinates for DCs")
    ap.add_argument(
        "--hubs",
        type=str,
        default="data/params/hubs.csv",
        help="CSV with hub coordinates (dc_id, lat, lon)",
    )
    ap.add_argument(
        "--regions",
        type=str,
        default="data/fulfillment/JD_network_data.csv",
        help="CSV with region mapping (region_id, dc_id, role)",
    )
    ap.add_argument(
        "--ship_times",
        type=str,
        default="data/processed/aggregated_travel_times.csv",
        help="CSV with shipping times (dc_ori, dc_des, travel_time_hours)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="data/derived/pseudo_coords/hub_based_coords.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--radius_min",
        type=float,
        default=50.0,
        help="Minimum radius from hub in miles",
    )
    ap.add_argument(
        "--radius_max",
        type=float,
        default=400.0,
        help="Maximum radius from hub in miles",
    )
    ap.add_argument(
        "--bearing_jitter",
        type=float,
        default=15.0,
        help="Random jitter for bearing in degrees (±)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    ap.add_argument(
        "--radius_overrides",
        type=str,
        default=None,
        help="Optional JSON file with per-region radius overrides: {region_id: [r_min, r_max]}",
    )
    ap.add_argument(
        "--snap_to_us",
        action="store_true",
        help="Snap hub coordinates outside US boundary to nearest point inside US (requires cartopy/shapely)",
    )
    ap.add_argument(
        "--validate_all",
        action="store_true",
        help="Validate and snap all generated coordinates (hubs and local DCs) to be within US",
    )
    ap.add_argument(
        "--cities",
        type=str,
        default="data/params/major_us_cities.csv",
        help="CSV with major US cities (city_name/city, lat/latitude, lon/longitude). If provided, local DCs will be mapped to closest cities instead of random coordinates.",
    )
    ap.add_argument(
        "--city_radius_tolerance",
        type=float,
        default=50.0,
        help="Radius tolerance in miles when matching cities to target radius (±value or ±20%% of radius, whichever is larger)",
    )
    ap.add_argument(
        "--min_distance_from_hub",
        type=float,
        default=25.0,
        help="Minimum distance in miles from hub for local DCs (prevents overlap with central warehouse)",
    )
    ap.add_argument(
        "--min_distance_between_dcs",
        type=float,
        default=10.0,
        help="Minimum distance in miles between local DCs (prevents overlap)",
    )
    return ap.parse_args()


def load_us_land_polygon():
    """Load USA land polygon excluding major water bodies (EPSG:4326) via Cartopy Natural Earth."""
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union
        from shapely.geometry import MultiPolygon

        # Get US country boundary
        shp_path = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
        reader = shpreader.Reader(shp_path)
        us_geoms = [rec.geometry for rec in reader.records() if rec.attributes.get("NAME") == "United States of America"]
        if not us_geoms:
            warnings.warn("US polygon not found; skipping boundary validation.")
            return None
        us_boundary = unary_union(us_geoms)
        
        # Get lakes and subtract them from US boundary
        try:
            lakes_path = shpreader.natural_earth(resolution="110m", category="physical", name="lakes")
            lakes_reader = shpreader.Reader(lakes_path)
            # Get lakes that intersect with US
            lakes = [rec.geometry for rec in lakes_reader.records() if us_boundary.intersects(rec.geometry)]
            if lakes:
                lakes_union = unary_union(lakes)
                us_land = us_boundary.difference(lakes_union)
                return us_land
        except Exception as e:
            warnings.warn(f"Could not load lakes data ({e}); using boundary without water exclusion.")
        
        return us_boundary
    except Exception as e:
        warnings.warn(f"US polygon unavailable ({e}); skipping boundary validation.")
        return None


def snap_to_us_boundary(lon: float, lat: float, us_polygon) -> tuple[float, float]:
    """
    Snap a single point to US boundary if it's outside.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        us_polygon: Shapely polygon of US boundary
        
    Returns:
        (lon, lat) adjusted to be inside or on US boundary
    """
    if us_polygon is None:
        return lon, lat
    
    try:
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        
        p = Point(lon, lat)
        
        # If inside, return as-is
        if us_polygon.contains(p):
            return lon, lat
        
        # Find nearest point on boundary
        boundary_point = nearest_points(p, us_polygon.boundary)[1]
        
        # Move slightly inside by 0.01 degrees (~1km)
        # Calculate direction from boundary to polygon center
        center = us_polygon.centroid
        dx = center.x - boundary_point.x
        dy = center.y - boundary_point.y
        norm = np.sqrt(dx**2 + dy**2)
        
        if norm > 0:
            # Move 0.01 degrees toward center
            offset = 0.01
            new_lon = boundary_point.x + (dx / norm) * offset
            new_lat = boundary_point.y + (dy / norm) * offset
            
            # Verify it's inside
            new_p = Point(new_lon, new_lat)
            if us_polygon.contains(new_p):
                return new_lon, new_lat
        
        # Fallback: just use boundary point
        return boundary_point.x, boundary_point.y
        
    except Exception as e:
        warnings.warn(f"Failed to snap point ({lon}, {lat}) to US boundary: {e}")
        return lon, lat


def haversine_distance_miles(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate great-circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
        
    Returns:
        Distance in miles
    """
    # Earth radius in miles (mean radius)
    R = 3958.8
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def destination_from_bearing_distance(
    lat: float, lon: float, bearing_deg: float, distance_mi: float
) -> tuple[float, float]:
    """
    Calculate destination coordinates given origin, bearing, and distance.
    
    Uses haversine-based spherical Earth model (sufficient accuracy for pseudo-coordinates).
    
    Args:
        lat: Origin latitude in degrees
        lon: Origin longitude in degrees
        bearing_deg: Bearing in degrees (0=North, 90=East)
        distance_mi: Distance in miles
        
    Returns:
        (dest_lat, dest_lon) in degrees
    """
    # Earth radius in miles (mean radius)
    R = 3958.8
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing_deg)
    
    # Angular distance
    d_rad = distance_mi / R
    
    # Destination latitude
    lat2_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(d_rad)
        + np.cos(lat_rad) * np.sin(d_rad) * np.cos(bearing_rad)
    )
    
    # Destination longitude
    lon2_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(d_rad) * np.cos(lat_rad),
        np.cos(d_rad) - np.sin(lat_rad) * np.sin(lat2_rad),
    )
    
    return np.degrees(lat2_rad), np.degrees(lon2_rad)


def process_region(
    region_id: str,
    region_dcs: pd.DataFrame,
    cw_dc_id: str,
    hub_coords: tuple[float, float],
    ship_times_df: pd.DataFrame,
    radius_min: float,
    radius_max: float,
    bearing_jitter: float,
    rng: np.random.Generator,
    us_land_polygon=None,
    max_attempts: int = 100,
    cities_df: Optional[pd.DataFrame] = None,
    city_radius_tolerance: float = 50.0,
    min_distance_from_hub: float = 25.0,
    min_distance_between_dcs: float = 10.0,
    global_assigned_coords: Optional[list] = None,
    hub_city_name: Optional[str] = None,
    hub_zip3: Optional[str] = None,
) -> list[dict]:
    """
    Process a single region to generate coordinates for all DCs.
    Uses city-based assignment if cities_df is provided, otherwise falls back to
    acceptance-rejection sampling.
    
    Args:
        region_id: Region identifier
        region_dcs: DataFrame with dc_id for all DCs in this region
        cw_dc_id: DC ID of the central warehouse
        hub_coords: (lat, lon) of the hub
        ship_times_df: DataFrame with shipping times
        radius_min: Minimum radius in miles
        radius_max: Maximum radius in miles
        bearing_jitter: Bearing jitter in degrees (unused with city-based assignment)
        rng: Random number generator
        us_land_polygon: Shapely polygon of US land (optional, for validation)
        max_attempts: Maximum sampling attempts per DC (for acceptance-rejection fallback)
        cities_df: Optional DataFrame with cities (city_name, lat, lon, zip3). If provided,
                   local DCs will be mapped to closest cities instead of random coordinates.
        city_radius_tolerance: Radius tolerance in miles when matching cities to target radius
        global_assigned_coords: Mutable list to track all assigned DC coordinates across regions
                               [(lat, lon), ...]. If None, only tracks within this region.
        hub_city_name: City name for the central warehouse (from hubs.csv)
        hub_zip3: Zip3 code for the central warehouse (from hubs.csv)
        
    Returns:
        List of dicts with DC coordinates and metadata (including city_name and zip3)
    """
    hub_lat, hub_lon = hub_coords
    results = []
    
    # Process central warehouse first
    results.append({
        "dc_id": cw_dc_id,
        "region_id": region_id,
        "role": "cw",
        "lat": hub_lat,
        "lon": hub_lon,
        "radius_mi_from_hub": 0.0,
        "bearing_deg_from_hub": np.nan,
        "ship_time_from_cw": 0.0,
        "city_name": hub_city_name,
        "zip3": hub_zip3,
    })
    
    # Get local DCs (exclude CW)
    local_dcs = region_dcs[region_dcs["dc_id"] != cw_dc_id]["dc_id"].tolist()
    
    if not local_dcs:
        return results
    
    # Extract shipping times from CW to local DCs
    cw_to_local = ship_times_df[
        (ship_times_df["dc_ori"] == cw_dc_id) & (ship_times_df["dc_des"].isin(local_dcs))
    ].copy()
    
    # Also check reverse direction if needed
    local_to_cw = ship_times_df[
        (ship_times_df["dc_ori"].isin(local_dcs)) & (ship_times_df["dc_des"] == cw_dc_id)
    ].copy()
    local_to_cw = local_to_cw.rename(columns={"dc_ori": "dc_des", "dc_des": "dc_ori"})
    
    # Combine and take average if both directions exist
    all_times = pd.concat([cw_to_local, local_to_cw], ignore_index=True)
    ship_times = all_times.groupby("dc_des")["travel_time_hours"].mean().to_dict()
    
    # Handle DCs with no shipping time data
    dcs_with_times = []
    dcs_without_times = []
    for dc in local_dcs:
        if dc in ship_times:
            dcs_with_times.append(dc)
        else:
            dcs_without_times.append(dc)
    
    if dcs_without_times:
        warnings.warn(
            f"Region {region_id}: {len(dcs_without_times)} DCs have no shipping time from CW {cw_dc_id}. "
            f"Assigning median radius."
        )
    
    # Convert shipping times to quantiles for DCs with times
    if dcs_with_times:
        times_array = np.array([ship_times[dc] for dc in dcs_with_times])
        
        if len(dcs_with_times) > 1:
            # Rank-based quantiles (0 to 1)
            ranks = np.argsort(np.argsort(times_array))
            quantiles = ranks / (len(dcs_with_times) - 1)
        else:
            # Single DC - place at middle
            quantiles = np.array([0.5])
        
        # Map quantiles to radii
        radii = radius_min + quantiles * (radius_max - radius_min)
        
        dc_to_radius = dict(zip(dcs_with_times, radii))
    else:
        dc_to_radius = {}
    
    # Assign median radius to DCs without times
    median_radius = (radius_min + radius_max) / 2.0
    for dc in dcs_without_times:
        dc_to_radius[dc] = median_radius
    
    # Helper to check if point is on US land
    def is_on_land(lon, lat):
        if us_land_polygon is None:
            return True
        try:
            from shapely.geometry import Point
            return us_land_polygon.contains(Point(lon, lat))
        except:
            return True
    
    # Generate coordinates for each local DC
    # Use city-based assignment if cities_df is provided, otherwise use acceptance-rejection
    assigned_city_indices = set()  # Track assigned cities to prevent duplicates within region
    if global_assigned_coords is None:
        global_assigned_coords = []  # Track within this region only
    assigned_coords = global_assigned_coords  # Use global list to check across all regions
    
    for dc_id in local_dcs:
        radius = dc_to_radius[dc_id]
        ship_time = ship_times.get(dc_id, np.nan)
        
        city_name = None
        zip3 = None
        dest_lat = None
        dest_lon = None
        current_radius = radius
        bearing = np.nan
        
        if cities_df is not None and len(cities_df) > 0:
            # City-based assignment
            city_distances = cities_df.apply(
                lambda row: haversine_distance_miles(hub_lat, hub_lon, row["lat"], row["lon"]), axis=1
            )
            
            tolerance = max(city_radius_tolerance, radius * 0.2)
            radius_min_target = max(min_distance_from_hub, radius - tolerance)
            
            # Filter: not assigned, far enough from hub, and not too close to other DCs
            def filter_available(city_idx_mask):
                mask = city_idx_mask & (city_distances >= min_distance_from_hub)
                if assigned_coords:
                    for assigned_lat, assigned_lon in assigned_coords:
                        too_close = cities_df.apply(
                            lambda row: haversine_distance_miles(assigned_lat, assigned_lon, row["lat"], row["lon"]) < min_distance_between_dcs,
                            axis=1
                        )
                        mask = mask & ~too_close
                return mask
            
            # Try cities in radius range first
            in_range = (city_distances >= radius_min_target) & (city_distances <= radius + tolerance)
            available = filter_available(in_range & ~cities_df.index.isin(assigned_city_indices))
            
            if not available.any():
                # Fallback: try all unassigned cities
                available = filter_available(~cities_df.index.isin(assigned_city_indices))
            
            if available.any():
                closest_idx = city_distances[available].idxmin()
                closest_city = cities_df.loc[closest_idx]
                city_name = closest_city.get("city_name", closest_city.get("city", "Unknown"))
                dest_lat = closest_city["lat"]
                dest_lon = closest_city["lon"]
                # Extract zip3 and convert to string if available, preserving leading zeros
                zip3_val = closest_city.get("zip3", None)
                if pd.notna(zip3_val):
                    # If numeric, format as 3-digit string with leading zeros
                    if isinstance(zip3_val, (int, float)):
                        zip3 = f"{int(zip3_val):03d}"
                    else:
                        # Already a string, preserve as-is but ensure it's 3 digits
                        zip3 = str(zip3_val).strip()
                        if zip3.isdigit():
                            zip3 = f"{int(zip3):03d}"
                else:
                    zip3 = None
                current_radius = city_distances[closest_idx]
                assigned_city_indices.add(closest_idx)
                assigned_coords.append((dest_lat, dest_lon))
            else:
                # No available cities - will fall back to random generation
                warnings.warn(
                    f"Region {region_id}, DC {dc_id}: No available cities (all assigned or too close to hub/other DCs). "
                    f"Falling back to random coordinate generation."
                )
        
        if dest_lat is None or dest_lon is None:
            # Fallback to acceptance-rejection sampling
            accepted = False
            current_radius = max(radius, min_distance_from_hub)
            for attempt in range(max_attempts):
                bearing = rng.uniform(0, 360)
                dest_lat, dest_lon = destination_from_bearing_distance(hub_lat, hub_lon, bearing, current_radius)
                
                # Check distance constraints
                if haversine_distance_miles(hub_lat, hub_lon, dest_lat, dest_lon) < min_distance_from_hub:
                    continue
                if assigned_coords:
                    if any(haversine_distance_miles(assigned_lat, assigned_lon, dest_lat, dest_lon) < min_distance_between_dcs
                           for assigned_lat, assigned_lon in assigned_coords):
                        continue
                
                if is_on_land(dest_lon, dest_lat):
                    accepted = True
                    assigned_coords.append((dest_lat, dest_lon))
                    break
                
                if attempt > max_attempts // 2:
                    current_radius *= 0.98
                    current_radius = max(current_radius, min_distance_from_hub)
            
            if not accepted:
                dest_lon, dest_lat = snap_to_us_boundary(dest_lon, dest_lat, us_land_polygon)
                warnings.warn(f"Region {region_id}, DC {dc_id}: No valid position after {max_attempts} attempts. Snapped to boundary.")
                assigned_coords.append((dest_lat, dest_lon))
        
        # Calculate bearing from hub if not already set
        if np.isnan(bearing) and dest_lat is not None and dest_lon is not None:
            # Calculate bearing using atan2
            lat1_rad = np.radians(hub_lat)
            lon1_rad = np.radians(hub_lon)
            lat2_rad = np.radians(dest_lat)
            lon2_rad = np.radians(dest_lon)
            
            dlon = lon2_rad - lon1_rad
            y = np.sin(dlon) * np.cos(lat2_rad)
            x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
            bearing = np.degrees(np.arctan2(y, x))
            bearing = (bearing + 360) % 360  # Normalize to 0-360
        
        results.append({
            "dc_id": dc_id,
            "region_id": region_id,
            "role": "local",
            "lat": dest_lat,
            "lon": dest_lon,
            "radius_mi_from_hub": current_radius,
            "bearing_deg_from_hub": bearing,
            "ship_time_from_cw": ship_time,
            "city_name": city_name,
            "zip3": zip3,
        })
    
    return results


def main():
    args = parse_args()
    
    # Set random seed
    rng = np.random.default_rng(args.seed)
    
    # Load US land polygon (excluding water bodies) for validation
    us_land_polygon = None
    if args.snap_to_us or args.validate_all:
        print("[INFO] Loading US land polygon (excluding lakes/rivers)...")
        us_land_polygon = load_us_land_polygon()
        if us_land_polygon is None:
            warnings.warn("Could not load US land polygon. Install cartopy/shapely for validation.")
    
    # Read input files
    print(f"[INFO] Reading hubs from: {args.hubs}")
    hubs_df = pd.read_csv(args.hubs)
    
    print(f"[INFO] Reading regions from: {args.regions}")
    regions_df = pd.read_csv(args.regions)
    
    print(f"[INFO] Reading shipping times from: {args.ship_times}")
    ship_times_df = pd.read_csv(args.ship_times)
    
    # Load cities data if provided
    cities_df = None
    if args.cities:
        print(f"[INFO] Reading cities from: {args.cities}")
        cities_df = pd.read_csv(args.cities)
        cities_df.columns = cities_df.columns.str.lower()
        
        # Handle flexible column names
        if "city_name" not in cities_df.columns and "city" in cities_df.columns:
            cities_df = cities_df.rename(columns={"city": "city_name"})
        
        if "latitude" in cities_df.columns:
            cities_df = cities_df.rename(columns={"latitude": "lat"})
        if "longitude" in cities_df.columns:
            cities_df = cities_df.rename(columns={"longitude": "lon"})
        
        # Validate required columns
        if not {"lat", "lon"}.issubset(cities_df.columns):
            raise ValueError(
                f"cities CSV must have columns: (city_name or city), (lat or latitude), (lon or longitude). "
                f"Found: {list(cities_df.columns)}"
            )
        
        # Convert to float and filter valid coordinates
        cities_df["lat"] = pd.to_numeric(cities_df["lat"], errors="coerce")
        cities_df["lon"] = pd.to_numeric(cities_df["lon"], errors="coerce")
        
        # Filter to valid US coordinate ranges
        initial_count = len(cities_df)
        cities_df = cities_df[
            (cities_df["lat"].notna()) & (cities_df["lon"].notna())
            & (cities_df["lat"] >= 24.0) & (cities_df["lat"] <= 50.0)  # Rough US lat bounds
            & (cities_df["lon"] >= -125.0) & (cities_df["lon"] <= -66.0)  # Rough US lon bounds
        ].copy()
        
        if len(cities_df) < initial_count:
            warnings.warn(
                f"Filtered {initial_count - len(cities_df)} cities with invalid coordinates. "
                f"Using {len(cities_df)} valid cities."
            )
        
        if len(cities_df) == 0:
            warnings.warn("No valid cities found. Falling back to random coordinate generation.")
            cities_df = None
        else:
            print(f"[INFO] Loaded {len(cities_df)} valid cities for assignment")
    
    # Normalize column names to lowercase for consistency
    hubs_df.columns = hubs_df.columns.str.lower()
    regions_df.columns = regions_df.columns.str.lower()
    
    # Handle alternate column names
    if "hub_lat" in hubs_df.columns:
        hubs_df = hubs_df.rename(columns={"hub_lat": "lat"})
    if "hub_lon" in hubs_df.columns:
        hubs_df = hubs_df.rename(columns={"hub_lon": "lon"})
    
    # Validate required columns
    if not {"dc_id", "lat", "lon"}.issubset(hubs_df.columns):
        raise ValueError(f"hubs CSV must have columns: dc_id and (lat, lon) or (hub_lat, hub_lon). Found: {list(hubs_df.columns)}")
    
    if not {"region_id", "dc_id"}.issubset(regions_df.columns):
        raise ValueError(f"regions CSV must have columns: region_id, dc_id. Found: {list(regions_df.columns)}")
    
    # Normalize DC IDs to string for consistent matching
    hubs_df["dc_id"] = hubs_df["dc_id"].astype(str)
    regions_df["dc_id"] = regions_df["dc_id"].astype(str)
    regions_df["region_id"] = regions_df["region_id"].astype(str)
    ship_times_df["dc_ori"] = ship_times_df["dc_ori"].astype(str)
    ship_times_df["dc_des"] = ship_times_df["dc_des"].astype(str)
    
    # Validate and snap hub coordinates to US land if requested
    if args.snap_to_us and us_land_polygon is not None:
        print("[INFO] Validating hub coordinates against US land...")
        n_snapped = 0
        for idx, row in hubs_df.iterrows():
            orig_lon, orig_lat = row["lon"], row["lat"]
            new_lon, new_lat = snap_to_us_boundary(orig_lon, orig_lat, us_land_polygon)
            if (new_lon, new_lat) != (orig_lon, orig_lat):
                hubs_df.at[idx, "lon"] = new_lon
                hubs_df.at[idx, "lat"] = new_lat
                n_snapped += 1
                print(f"  [SNAP] Hub {row['dc_id']}: ({orig_lat:.4f}, {orig_lon:.4f}) -> ({new_lat:.4f}, {new_lon:.4f})")
        if n_snapped > 0:
            print(f"[INFO] Snapped {n_snapped} hub(s) to US land")
        else:
            print("[INFO] All hubs are on US land")
    
    # Load radius overrides if provided
    radius_overrides = {}
    if args.radius_overrides:
        with open(args.radius_overrides, "r") as f:
            radius_overrides = json.load(f)
        print(f"[INFO] Loaded radius overrides for {len(radius_overrides)} regions")
    
    # Detect role column or infer from hubs
    if "role" not in regions_df.columns:
        # Infer: if DC is in hubs, it's a CW; otherwise local
        warnings.warn("No 'role' column in regions CSV. Inferring: DCs in hubs.csv are CWs, others are local.")
        hub_dc_ids = set(hubs_df["dc_id"])
        regions_df["role"] = regions_df["dc_id"].apply(lambda x: "cw" if x in hub_dc_ids else "local")
    
    # Create hub lookup: dc_id -> (lat, lon)
    hub_coords_dict = dict(zip(hubs_df["dc_id"], zip(hubs_df["lat"], hubs_df["lon"])))
    
    # Create hub city_name and zip3 lookups (if available)
    hub_city_name_dict = {}
    hub_zip3_dict = {}
    if "city_name" in hubs_df.columns:
        hub_city_name_dict = hubs_df.set_index("dc_id")["city_name"].to_dict()
    if "zip3" in hubs_df.columns:
        # Convert zip3 to string, handling NaN values and preserving leading zeros
        def format_zip3(x):
            if pd.notna(x):
                # If numeric, format as 3-digit string with leading zeros
                if isinstance(x, (int, float)):
                    return f"{int(x):03d}"
                else:
                    # Already a string, preserve as-is but ensure it's 3 digits
                    s = str(x).strip()
                    if s.isdigit():
                        return f"{int(s):03d}"
                    return s
            return None
        hub_zip3_dict = hubs_df.set_index("dc_id")["zip3"].apply(format_zip3).to_dict()
    
    # Process each region
    all_results = []
    regions = regions_df["region_id"].unique()
    global_assigned_coords = []  # Track all assigned DC coordinates across all regions
    
    print(f"[INFO] Processing {len(regions)} regions...")
    
    for region_id in regions:
        region_dcs = regions_df[regions_df["region_id"] == region_id]
        
        # Find central warehouse for this region
        cw_dcs = region_dcs[region_dcs["role"] == "cw"]
        
        if len(cw_dcs) == 0:
            warnings.warn(f"Region {region_id} has no central warehouse. Skipping.")
            continue
        
        if len(cw_dcs) > 1:
            warnings.warn(f"Region {region_id} has multiple CWs. Using first one: {cw_dcs.iloc[0]['dc_id']}")
        
        cw_dc_id = str(cw_dcs.iloc[0]["dc_id"])
        
        # Get hub coordinates
        if cw_dc_id not in hub_coords_dict:
            warnings.warn(f"Region {region_id}: CW {cw_dc_id} not found in hubs CSV. Skipping.")
            continue
        
        hub_coords = hub_coords_dict[cw_dc_id]
        
        # Get hub city_name and zip3
        hub_city_name = hub_city_name_dict.get(cw_dc_id, None)
        hub_zip3 = hub_zip3_dict.get(cw_dc_id, None)
        
        # Get radius bounds (use overrides if available)
        if str(region_id) in radius_overrides:
            r_min, r_max = radius_overrides[str(region_id)]
        else:
            r_min, r_max = args.radius_min, args.radius_max
        
        # Process region (uses city-based assignment if cities_df provided)
        region_results = process_region(
            region_id=str(region_id),
            region_dcs=region_dcs,
            cw_dc_id=cw_dc_id,
            hub_coords=hub_coords,
            ship_times_df=ship_times_df,
            radius_min=r_min,
            radius_max=r_max,
            bearing_jitter=args.bearing_jitter,
            rng=rng,
            us_land_polygon=us_land_polygon if args.validate_all else None,
            max_attempts=100,
            cities_df=cities_df,
            city_radius_tolerance=args.city_radius_tolerance,
            min_distance_from_hub=args.min_distance_from_hub,
            min_distance_between_dcs=args.min_distance_between_dcs,
            global_assigned_coords=global_assigned_coords,
            hub_city_name=hub_city_name,
            hub_zip3=hub_zip3,
        )
        
        all_results.extend(region_results)
    
    # Create output DataFrame
    output_df = pd.DataFrame(all_results)
    
    # Report validation results
    if cities_df is not None:
        n_with_cities = int(output_df["city_name"].notna().sum())
        print(f"[INFO] Assigned {n_with_cities} DCs to cities")
    
    if args.validate_all and us_land_polygon is not None:
        if cities_df is None:
            print("[INFO] All coordinates generated using acceptance-rejection sampling on US land")
        # Count any fallback snaps (should be rare)
        n_local = int((output_df["role"] == "local").sum())
        print(f"[INFO] Generated {n_local} local DCs on US land (avoiding water bodies)")
    
    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save output
    output_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(output_df)} DC coordinates to {out_path}")
    
    # Generate statistics
    stats = {
        "n_regions": len(regions),
        "n_dcs_total": len(output_df),
        "n_central_warehouses": int((output_df["role"] == "cw").sum()),
        "n_local_dcs": int((output_df["role"] == "local").sum()),
        "radius_range_mi": [float(args.radius_min), float(args.radius_max)],
        "bearing_jitter_deg": float(args.bearing_jitter),
        "seed": int(args.seed),
        "snap_to_us": bool(args.snap_to_us),
        "validate_all": bool(args.validate_all),
        "city_based_assignment": cities_df is not None,
        "n_dcs_assigned_to_cities": int(output_df["city_name"].notna().sum()) if cities_df is not None else 0,
        "lat_range": [float(output_df["lat"].min()), float(output_df["lat"].max())],
        "lon_range": [float(output_df["lon"].min()), float(output_df["lon"].max())],
        "radius_stats": {
            "min": float(output_df["radius_mi_from_hub"].min()),
            "max": float(output_df["radius_mi_from_hub"].max()),
            "mean": float(output_df["radius_mi_from_hub"].mean()),
            "median": float(output_df["radius_mi_from_hub"].median()),
        },
        "ship_time_stats": {
            "min": float(output_df["ship_time_from_cw"].min()),
            "max": float(output_df["ship_time_from_cw"].max()),
            "mean": float(output_df[output_df["role"] == "local"]["ship_time_from_cw"].mean()),
            "median": float(output_df[output_df["role"] == "local"]["ship_time_from_cw"].median()),
        },
    }
    
    stats_path = out_path.parent / "hub_based_coords_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Wrote statistics to {stats_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Regions processed: {stats['n_regions']}")
    print(f"Central warehouses: {stats['n_central_warehouses']}")
    print(f"Local DCs: {stats['n_local_dcs']}")
    print(f"Radius range: {stats['radius_stats']['min']:.1f} - {stats['radius_stats']['max']:.1f} miles")
    print(f"Lat range: {stats['lat_range'][0]:.2f} - {stats['lat_range'][1]:.2f}")
    print(f"Lon range: {stats['lon_range'][0]:.2f} - {stats['lon_range'][1]:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
