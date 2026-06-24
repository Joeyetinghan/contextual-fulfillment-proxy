"""
Utilities for loading DC coordinates from hub-based coordinate files.

Outputs and files this module expects to exist:
- data/derived/pseudo_coords/hub_based_coords.csv  (dc_id, lat, lon) or (dc_ID, lon, lat)
- data/derived/pseudo_coords/dc_coords.csv         (dc_ID, z1, z2) or (dc_ID, lon, lat) [fallback]

Exposed helpers:
- load_dc_coords(): resilient reader with column unification for hub-based or MDS-based coordinates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_DC_COORDS_DEFAULT = Path("data/derived/pseudo_coords/hub_based_coords.csv")


def _standardize_xy(df: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """Return a copy of df with unified coordinate columns as x,y.

    Accepts either (z1,z2) or (lon,lat) for DCs. For hub-based coords, handles (lat,lon) order.
    """
    cols = set(df.columns)
    df = df.copy()
    if kind == "dc":
        # Handle hub-based format: dc_id, lat, lon (or lon, lat)
        if {"lat", "lon"}.issubset(cols):
            # Standardize: x = lon, y = lat (regardless of input column order)
            df.rename(columns={"lon": "x", "lat": "y"}, inplace=True)
        # Handle MDS format: z1, z2
        elif {"z1", "z2"}.issubset(cols):
            df.rename(columns={"z1": "x", "z2": "y"}, inplace=True)
        else:
            raise ValueError("dc_coords must include either (z1,z2) or (lon,lat) or (lat,lon)")
    else:
        raise ValueError("kind must be 'dc'")
    return df


def load_dc_coords(path: str | Path = _DC_COORDS_DEFAULT) -> pd.DataFrame:
    """Load DC coordinates from CSV file.
    
    Supports multiple formats:
    - Hub-based: dc_id (or dc_ID), lat, lon
    - MDS-based: dc_ID, z1, z2
    - Generic: dc_ID, lon, lat
    
    Returns DataFrame with columns: dc_ID, x, y (where x=lon, y=lat)
    """
    p = Path(path)
    if not p.exists():
        # Try to find a fallback by name anywhere under data/
        # Prefer hub_based_coords.csv, then dc_coords.csv
        hub_cands = sorted(Path("data").rglob("hub_based_coords.csv"))
        dc_cands = sorted(Path("data").rglob("dc_coords.csv"))
        if hub_cands:
            p = hub_cands[0]
        elif dc_cands:
            p = dc_cands[0]
        else:
            raise FileNotFoundError(f"DC coords not found at {p} (no fallback found)")
    
    df = pd.read_csv(p)
    
    # Handle both dc_id and dc_ID column names
    if "dc_id" in df.columns:
        df = df.rename(columns={"dc_id": "dc_ID"})
    elif "dc_ID" not in df.columns:
        raise ValueError(f"DC coords CSV must have a dc_id or dc_ID column. Found: {list(df.columns)}")
    
    df = _standardize_xy(df, kind="dc")
    df = df[['dc_ID', 'x', 'y']].copy()
    # force string IDs for join stability
    df['dc_ID'] = df['dc_ID'].astype(str)
    return df


__all__ = [
    "load_dc_coords",
]



