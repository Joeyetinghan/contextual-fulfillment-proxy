"""
Utilities for tuning/exporting the DL delivery-time *simulator* (outcome sampler).

These helpers intentionally operate on the TEST period only (post proxy-train),
because the simulator is meant to mimic the true environment used in simulation.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

import src.config as cfg


def expected_carriers() -> list[int]:
    """Carrier IDs we expect to support (from cost-model metadata)."""
    cost_models_df = pd.read_csv(cfg.REAL_COST_MODELS_CS_PATH)
    carriers = sorted(cost_models_df["carrier_service_id"].dropna().astype(int).unique().tolist())

    excluded = set(getattr(cfg, "EXCLUDED_CARRIER_SERVICES", []) or [])
    if excluded:
        carriers = [c for c in carriers if c not in excluded]

    return carriers


def load_test_cs_data() -> pd.DataFrame:
    """
    Load `preprocessed_data_cs.csv` and engineer delivery-time features for the TEST period.

    Returns a DataFrame sorted by `order_time` (required for rolling features).
    """
    df = pd.read_csv(
        "data/processed/preprocessed_data_cs.csv",
        parse_dates=["order_time", "order_date"],
    )
    df = df[df["order_date"] > cfg.PROXY_TRAIN_END_DATE].copy()
    df.sort_values(by="order_time", inplace=True)
    from src.training.delivery_time.common import create_delivery_time_features

    df = create_delivery_time_features(df)
    return df


def tune_output_path(out_dir: Path, carrier_id: int, combination_idx: int) -> Path:
    from src.training.delivery_time.common import format_carrier_id_for_path

    carrier_id_str = format_carrier_id_for_path(carrier_id)
    return out_dir / f"dl_sim_grid_carrier_{carrier_id_str}_combo_{combination_idx}.json"
