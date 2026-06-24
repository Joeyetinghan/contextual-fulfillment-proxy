#!/usr/bin/env python3
"""
Train/evaluate a single (carrier, hyperparam-combo) for the DL *simulator*.

This is meant to be used with SLURM array jobs for massive parallel grid tuning.

Outputs a small JSON per task with:
  - best_val_loss (pinball loss)
  - best_epoch (argmin epoch on the val curve)

We then select the best combo per carrier and retrain on the full TEST set
to produce the actual simulator bundles used by `src/simulator/delivery_sampler.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import src.config as cfg
from src.training.delivery_time._train_dl import train_dl_model
from src.training.delivery_time.common import prepare_categorical_encoders
from src.training.delivery_time.by_carrier.simulator_grid_utils import (
    expected_carriers,
    load_test_cs_data,
    tune_output_path,
)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_grid_params(grid_file: Path, combination_idx: int) -> dict:
    grid = json.loads(grid_file.read_text(encoding="utf-8"))
    combos = grid.get("combinations") or []
    if not combos:
        raise ValueError(f"Grid file has no combinations: {grid_file}")
    if combination_idx < 0 or combination_idx >= len(combos):
        raise ValueError(f"combination_idx {combination_idx} out of range [0, {len(combos)-1}]")
    params = dict(combos[combination_idx])
    # Make sure required keys exist for `train_dl_model`.
    required = ["lr", "hidden_dim", "n_layers", "dropout_p", "weight_decay", "batch_size", "epochs"]
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"Grid params missing keys {missing}: got keys={sorted(params.keys())}")
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True, help="Global task id (maps to carrier+combo).")
    parser.add_argument(
        "--grid_file",
        type=Path,
        default=Path("data/delivery_dl_simulator_grid_search_combinations.json"),
        help="Path to hyperparameter grid JSON.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/models/delivery_time_cs/simulator_tune/dl_grid"),
        help="Directory to write per-task JSON results.",
    )
    parser.add_argument(
        "--min_rows",
        type=int,
        default=150,
        help=(
            "Minimum number of (non-null target) TEST rows required to tune a carrier. "
            "If fewer rows are available, the task is marked skipped_insufficient_data."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result JSON if present.",
    )
    args = parser.parse_args()

    carriers = expected_carriers()
    grid = json.loads(args.grid_file.read_text(encoding="utf-8"))
    num_combos = int(grid.get("num_combinations") or len(grid.get("combinations") or []))
    if num_combos <= 0:
        raise ValueError(f"Invalid num_combinations in grid file: {args.grid_file}")

    num_tasks = len(carriers) * num_combos
    if args.task_id < 0 or args.task_id >= num_tasks:
        print(f"[grid-task] task_id={args.task_id} out of range (0..{num_tasks-1}); exiting.")
        return

    carrier_idx = args.task_id // num_combos
    combo_idx = args.task_id % num_combos
    carrier_id = int(carriers[carrier_idx])

    out_path = tune_output_path(args.out_dir, carrier_id, combo_idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"[grid-task] exists, skipping: {out_path}")
        return

    params = _load_grid_params(args.grid_file, combo_idx)

    # Use the same seed across combos for a given carrier to reduce tuning noise.
    seed = int(cfg.RANDOM_SEED + carrier_id)
    _set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    status = "ok"
    err = None
    payload: dict = {
        "task_id": args.task_id,
        "carrier_id": carrier_id,
        "combo_idx": combo_idx,
        "seed": seed,
        "params": params,
        "status": status,
    }

    try:
        df = load_test_cs_data()
        df_carrier = df[df["carrier_service_id_anon"] == carrier_id].copy()
        payload["n_rows_total"] = int(len(df_carrier))
        df_carrier = df_carrier.dropna(subset=[cfg.DELIVERY_TIME_TARGET]).reset_index(drop=True)
        payload["n_rows_nonnull_target"] = int(len(df_carrier))
        min_rows = int(max(0, args.min_rows))
        payload["min_rows_required"] = min_rows

        if len(df_carrier) < min_rows:
            status = "skipped_insufficient_data"
            payload["status"] = status
            payload["skip_reason"] = f"n_rows_nonnull_target<{min_rows}"
        else:
            X = df_carrier[cfg.DELIVERY_TIME_FEATURES].fillna(0)
            y = df_carrier[cfg.DELIVERY_TIME_TARGET].astype(float).to_numpy()

            encoders, vocab_sizes = prepare_categorical_encoders(df_carrier)
            X_num = X[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
            x_scaler = StandardScaler().fit(X_num)
            X_num_scaled = x_scaler.transform(X_num).astype(np.float32, copy=False)

            Xn = torch.tensor(X_num_scaled, dtype=torch.float32)
            Xc = {
                "dc_ori": torch.tensor(encoders["dc_ori"].transform(X["dc_ori"].astype(str)), dtype=torch.long),
                "dc_des": torch.tensor(encoders["dc_des"].transform(X["dc_des"].astype(str)), dtype=torch.long),
            }
            yt = torch.tensor(y, dtype=torch.float32)

            # Random split (intentionally shuffle to overfit the test environment).
            split = int((1.0 - cfg.DELIVERY_DL_VALIDATION_SPLIT_RATIO) * len(Xn))
            if split <= 0 or split >= len(Xn):
                raise ValueError(f"Bad train/val split: n={len(Xn)} split={split}")

            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(Xn))
            Xn = Xn[perm]
            yt = yt[perm]
            for key in Xc:
                Xc[key] = Xc[key][perm]

            model, train_losses, val_losses = train_dl_model(
                params,
                Xn[:split],
                {"dc_ori": Xc["dc_ori"][:split], "dc_des": Xc["dc_des"][:split]},
                yt[:split],
                Xn[split:],
                {"dc_ori": Xc["dc_ori"][split:], "dc_des": Xc["dc_des"][split:]},
                yt[split:],
                vocab_sizes,
                device,
                return_model=True,
            )
            del model  # we only care about the metrics; final export retrains on full TEST

            if not val_losses:
                raise ValueError("No validation losses recorded (unexpected).")

            best_val = float(min(val_losses))
            best_epoch = int(np.argmin(np.asarray(val_losses)) + 1)
            best_train = float(min(train_losses)) if train_losses else float("nan")
            final_train = float(train_losses[-1]) if train_losses else float("nan")
            payload.update(
                {
                    "status": "ok",
                    "n_train": int(split),
                    "n_val": int(len(Xn) - split),
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "best_train_loss": best_train,
                    "final_train_loss": final_train,
                }
            )
    except Exception as e:
        status = "error"
        err = repr(e)
        payload["status"] = status
        payload["error"] = err

    payload["runtime_s"] = float(time.perf_counter() - t0)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[grid-task] wrote {out_path} status={payload['status']} runtime_s={payload['runtime_s']:.2f}")
    if err is not None:
        print(f"[grid-task] error: {err}")


if __name__ == "__main__":
    main()
