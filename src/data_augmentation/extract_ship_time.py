"""
Extract and aggregate travel times between DC pairs from raw event data.

This script processes DC-to-DC shipping events to compute median travel times
for each origin-destination pair. It serves as a preprocessing step before
building pseudo-coordinates or other downstream analyses.

Input CSV (auto-detected by columns):
  Preprocessed orders: data/processed/preprocessed_data.csv with columns
    dc_ori, dc_des, travel_time_hours  (the standard input for this repo).
  A raw DC-to-DC event table with columns
    dc_i, dc_j, ship_out_time, arr_station_time  is also accepted.

Output:
  data/processed/aggregated_travel_times.csv with columns:
    - dc_ori, dc_des, travel_time_hours (median), n_events (count)
  data/processed/travel_time_stats.json (basic diagnostics)

Usage:
  python -m src.data_augmentation.extract_ship_time --input data/processed/preprocessed_data.csv
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Extract and aggregate DC-to-DC travel times")
    ap.add_argument(
        "--input",
        type=str,
        default="data/processed/preprocessed_data.csv",
        help="Input CSV with raw events or preprocessed data",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="data/processed/aggregated_travel_times.csv",
        help="Output CSV path for aggregated travel times",
    )
    ap.add_argument(
        "--min_pairs",
        type=int,
        default=5,
        help="Minimum number of events per (dc_ori, dc_des) pair to keep",
    )
    ap.add_argument(
        "--winsor_p",
        type=float,
        default=0.05,
        help="Winsorize travel times at this quantile (e.g., 0.01 clips at 1st and 99th percentiles)",
    )
    ap.add_argument(
        "--aggregation",
        type=str,
        default="median",
        choices=["median", "mean", "min", "max"],
        help="Aggregation function for travel times per OD pair",
    )
    return ap.parse_args()


def compute_travel_hours(df: pd.DataFrame, col_ship="ship_out_time", col_arr="arr_station_time") -> pd.Series:
    """
    Compute travel time in hours from ship_out_time to arr_station_time.
    
    Args:
        df: DataFrame with timestamp columns
        col_ship: Column name for shipment departure time
        col_arr: Column name for arrival time
        
    Returns:
        Series of travel times in hours
    """
    df[col_ship] = pd.to_datetime(df[col_ship], errors="coerce")
    df[col_arr] = pd.to_datetime(df[col_arr], errors="coerce")
    hrs = (df[col_arr] - df[col_ship]).dt.total_seconds() / 3600.0
    return hrs


def winsorize(x: pd.Series, p: float) -> pd.Series:
    """
    Winsorize a series by clipping extreme values at the p and (1-p) quantiles.
    
    Args:
        x: Series to winsorize
        p: Quantile threshold (e.g., 0.01 for 1st and 99th percentiles)
        
    Returns:
        Winsorized series
    """
    if p <= 0:
        return x
    lo, hi = x.quantile(p), x.quantile(1 - p)
    return x.clip(lower=lo, upper=hi)


def main():
    args = parse_args()
    inp = Path(args.input)
    
    # Check if input file exists, try fallbacks
    if not inp.exists():
        # Try to find a reasonable fallback under data/
        raw_candidates = sorted(Path("data").rglob("dc_dc_events*.csv"))
        preprocessed_candidates = sorted(Path("data").rglob("preprocessed_data*.csv"))
        
        if raw_candidates:
            print(f"[WARN] Input {inp} not found. Using fallback raw events: {raw_candidates[0]}")
            inp = raw_candidates[0]
        elif preprocessed_candidates:
            print(f"[WARN] Input {inp} not found. Using fallback preprocessed: {preprocessed_candidates[0]}")
            inp = preprocessed_candidates[0]
        else:
            raise FileNotFoundError(
                f"Input CSV not found at {args.input}. Pass --input pointing to "
                "data/processed/preprocessed_data.csv (the standard input)."
            )
    
    print(f"[INFO] Reading input from: {inp}")
    df = pd.read_csv(inp)
    
    # Detect input schema: preprocessed vs raw
    if {"dc_ori", "dc_des", "travel_time_hours"}.issubset(df.columns):
        # Already preprocessed with travel times
        print("[INFO] Detected preprocessed data with travel_time_hours")
        # Keep as dc_ori, dc_des
    elif {"dc_i", "dc_j", "ship_out_time", "arr_station_time"}.issubset(df.columns):
        # Raw event data - compute travel times and rename to standard names
        print("[INFO] Detected raw event data - computing travel times from timestamps")
        df["travel_time_hours"] = compute_travel_hours(df)
        df = df.rename(columns={"dc_i": "dc_ori", "dc_j": "dc_des"})
    elif {"dc_i", "dc_j", "travel_time_hours"}.issubset(df.columns):
        # Alternate format - rename to standard names
        print("[INFO] Detected dc_i, dc_j format - renaming to dc_ori, dc_des")
        df = df.rename(columns={"dc_i": "dc_ori", "dc_j": "dc_des"})
    else:
        raise ValueError(
            "Input CSV must have either:\n"
            "  - Preprocessed columns: {dc_ori, dc_des, travel_time_hours}\n"
            "  - Raw event columns: {dc_i, dc_j, ship_out_time, arr_station_time}\n"
            "  - Alternate format: {dc_i, dc_j, travel_time_hours}"
        )
    
    # Drop rows with invalid/negative times
    initial_count = len(df)
    df = df.dropna(subset=["dc_ori", "dc_des", "travel_time_hours"]).copy()
    df = df[df["travel_time_hours"] >= 0].copy()
    print(f"[INFO] Removed {initial_count - len(df)} events with invalid/negative travel times")
    print(f"[INFO] Remaining events: {len(df)}")
    
    # Optional winsorization to handle outliers/mis-scans
    if args.winsor_p > 0:
        pre_winsor = df["travel_time_hours"].copy()
        df["travel_time_hours"] = winsorize(df["travel_time_hours"], args.winsor_p)
        n_changed = (pre_winsor != df["travel_time_hours"]).sum()
        print(f"[INFO] Winsorized {n_changed} travel times at p={args.winsor_p}")
    
    # Count events per OD pair and filter by minimum support
    grp = df.groupby(["dc_ori", "dc_des"])
    pair_counts = grp["travel_time_hours"].count().rename("n_events").reset_index()
    
    pairs_before = len(pair_counts)
    keep_pairs = pair_counts[pair_counts["n_events"] >= args.min_pairs]
    pairs_after = len(keep_pairs)
    print(f"[INFO] OD pairs before min_pairs filter: {pairs_before}")
    print(f"[INFO] OD pairs after min_pairs>={args.min_pairs} filter: {pairs_after}")
    
    # Filter original data to keep only pairs with sufficient support
    df = df.merge(keep_pairs[["dc_ori", "dc_des"]], on=["dc_ori", "dc_des"], how="inner")
    
    # Aggregate using specified function (default: median)
    agg_func = args.aggregation
    print(f"[INFO] Aggregating travel times using {agg_func}")
    
    aggregated = (
        df.groupby(["dc_ori", "dc_des"])
        .agg(
            travel_time_hours=(("travel_time_hours", agg_func)),
            n_events=(("travel_time_hours", "count")),
        )
        .reset_index()
    )
    
    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated travel times
    aggregated.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(aggregated)} OD pairs to {out_path}")
    
    # Compute and save statistics
    unique_dcs = set(aggregated["dc_ori"]).union(set(aggregated["dc_des"]))
    stats = {
        "input_file": str(inp),
        "output_file": str(out_path),
        "n_events_total": int(initial_count),
        "n_events_valid": int(len(df)),
        "n_unique_dcs": len(unique_dcs),
        "n_od_pairs": int(len(aggregated)),
        "min_pairs_threshold": int(args.min_pairs),
        "winsor_p": float(args.winsor_p),
        "aggregation": agg_func,
        "travel_time_stats": {
            "min_hours": float(aggregated["travel_time_hours"].min()),
            "max_hours": float(aggregated["travel_time_hours"].max()),
            "mean_hours": float(aggregated["travel_time_hours"].mean()),
            "median_hours": float(aggregated["travel_time_hours"].median()),
        },
        "events_per_pair_stats": {
            "min": int(aggregated["n_events"].min()),
            "max": int(aggregated["n_events"].max()),
            "mean": float(aggregated["n_events"].mean()),
            "median": float(aggregated["n_events"].median()),
        },
    }
    
    stats_path = out_path.parent / "travel_time_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Wrote statistics to {stats_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Unique DCs: {len(unique_dcs)}")
    print(f"OD pairs: {len(aggregated)}")
    print(f"Travel time range: {stats['travel_time_stats']['min_hours']:.2f} - {stats['travel_time_stats']['max_hours']:.2f} hours")
    print(f"Events per pair: {stats['events_per_pair_stats']['min']} - {stats['events_per_pair_stats']['max']}")
    print("=" * 60)


if __name__ == "__main__":
    main()


