#!/usr/bin/env python3
"""
Select best DL simulator hyperparameters from grid results and export simulators.

Workflow:
  1) Run grid tasks (many) -> write JSONs to a tune dir.
  2) This script:
     - picks best combo per carrier (min best_val_loss),
     - retrains on *full TEST* data for that carrier using the selected params,
     - writes the simulator bundle + joblib pointer used by simulation.

The simulator artifacts must live under `cfg.DELIVERY_TIME_SIMULATOR_PATH` and be named:
  - simulator_dl_{carrier_id}.pt
  - simulator_dl_{carrier_id}.joblib
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from pathlib import Path
from statistics import mean

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import src.config as cfg
from src.training.delivery_time._train_dl import train_dl_model
from src.training.delivery_time.common import prepare_categorical_encoders, format_carrier_id_for_path
from src.training.delivery_time.by_carrier.simulator_grid_utils import expected_carriers, load_test_cs_data


def _load_best_results(tune_dir: Path, selection_metric: str = "best_val_loss") -> tuple[dict[int, dict], dict]:
    best: dict[int, dict] = {}
    report: dict = {
        "missing_no_files": [],
        "missing_no_ok": {},
        "json_parse_errors": {},
    }
    carriers = expected_carriers()
    for carrier_id in carriers:
        carrier_id_str = format_carrier_id_for_path(carrier_id)
        paths = sorted(tune_dir.glob(f"dl_sim_grid_carrier_{carrier_id_str}_combo_*.json"))
        if not paths:
            report["missing_no_files"].append(int(carrier_id))
            continue
        best_path = None
        best_score = float("inf")
        best_payload = None
        status_counts: Counter[str] = Counter()
        parse_errors = 0
        for p in paths:
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                parse_errors += 1
                continue
            status_counts[str(payload.get("status"))] += 1
            if payload.get("status") != "ok":
                continue
            val_loss = payload.get("best_val_loss")
            train_loss = payload.get("best_train_loss")
            if selection_metric == "best_train_loss":
                if train_loss is None:
                    continue
                score = float(train_loss)
                tie = float(val_loss) if val_loss is not None else float("inf")
            else:
                if val_loss is None:
                    continue
                score = float(val_loss)
                tie = float(train_loss) if train_loss is not None else float("inf")

            if score < best_score or (score == best_score and tie < float("inf")):
                best_score = score
                best_path = p
                best_payload = payload
        if best_payload is not None and best_path is not None:
            best[int(carrier_id)] = {"path": str(best_path), **best_payload}
        else:
            report["missing_no_ok"][int(carrier_id)] = {
                "n_files": int(len(paths)),
                "status_counts": dict(status_counts),
            }
        if parse_errors:
            report["json_parse_errors"][int(carrier_id)] = int(parse_errors)
    return best, report


def _load_all_results(tune_dir: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    parse_errors: dict[str, str] = {}
    for p in sorted(tune_dir.glob("dl_sim_grid_carrier_*_combo_*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            parse_errors[str(p)] = repr(e)
            continue
        rows.append(
            {
                "path": str(p),
                "carrier_id": payload.get("carrier_id"),
                "combo_idx": payload.get("combo_idx"),
                "status": payload.get("status"),
                "best_val_loss": payload.get("best_val_loss"),
                "best_epoch": payload.get("best_epoch"),
                "best_train_loss": payload.get("best_train_loss"),
                "final_train_loss": payload.get("final_train_loss"),
                "n_rows_total": payload.get("n_rows_total"),
                "n_rows_nonnull_target": payload.get("n_rows_nonnull_target"),
                "min_rows_required": payload.get("min_rows_required"),
                "runtime_s": payload.get("runtime_s"),
                "params": payload.get("params") or {},
            }
        )
    df = pd.DataFrame(rows)
    return df, parse_errors


def _summarize_results(df: pd.DataFrame, parse_errors: dict, grid_file: Path | None, top_k: int = 5) -> None:
    carriers = expected_carriers()
    carriers_expected = len(carriers)
    carriers_with_any = int(df["carrier_id"].nunique()) if not df.empty else 0

    print("\n=== Simulator DL Grid Summary ===")
    print(f"carriers_expected: {carriers_expected}")
    print(f"carriers_with_any_results: {carriers_with_any}")
    if parse_errors:
        print(f"json_parse_errors: {len(parse_errors)} (use --report-json to inspect)")

    if df.empty:
        print("No results found.")
        return

    status_counts = df["status"].value_counts(dropna=False).to_dict()
    print(f"status_counts: {status_counts}")

    # Per-carrier coverage
    per_carrier_counts = df.groupby(["carrier_id", "status"]).size().unstack(fill_value=0)
    combos_per_carrier = per_carrier_counts.sum(axis=1)
    ok_per_carrier = per_carrier_counts.get("ok", pd.Series(dtype=int))
    print(
        f"combos_per_carrier (min/mean/max): "
        f"{int(combos_per_carrier.min())}/{combos_per_carrier.mean():.1f}/{int(combos_per_carrier.max())}"
    )
    if not ok_per_carrier.empty:
        print(
            f"ok_per_carrier (min/mean/max): "
            f"{int(ok_per_carrier.min())}/{ok_per_carrier.mean():.1f}/{int(ok_per_carrier.max())}"
        )

    # Best per carrier
    ok_df = df[df["status"] == "ok"].copy()
    ok_df = ok_df.dropna(subset=["best_val_loss"])
    if ok_df.empty:
        print("No status=ok results to summarize.")
        return

    best_df = ok_df.sort_values("best_val_loss").groupby("carrier_id", as_index=False).head(1)
    best_losses = best_df["best_val_loss"].astype(float).to_numpy()
    print(
        "best_val_loss stats (min/mean/p50/p90/max): "
        f"{best_losses.min():.6f}/{best_losses.mean():.6f}/"
        f"{np.percentile(best_losses, 50):.6f}/{np.percentile(best_losses, 90):.6f}/"
        f"{best_losses.max():.6f}"
    )
    if "best_train_loss" in best_df.columns:
        train_losses = best_df["best_train_loss"].astype(float).to_numpy()
        print(
            "best_train_loss stats (min/mean/p50/p90/max): "
            f"{np.nanmin(train_losses):.6f}/{np.nanmean(train_losses):.6f}/"
            f"{np.nanpercentile(train_losses, 50):.6f}/{np.nanpercentile(train_losses, 90):.6f}/"
            f"{np.nanmax(train_losses):.6f}"
        )
        gap = best_df["best_val_loss"].astype(float) - best_df["best_train_loss"].astype(float)
        print(
            "val-train gap stats (min/mean/p50/p90/max): "
            f"{np.nanmin(gap):.6f}/{np.nanmean(gap):.6f}/"
            f"{np.nanpercentile(gap, 50):.6f}/{np.nanpercentile(gap, 90):.6f}/"
            f"{np.nanmax(gap):.6f}"
        )

    # Hyperparam frequency among winners
    param_counts: dict[str, Counter] = {}
    for params in best_df["params"]:
        for k, v in (params or {}).items():
            param_counts.setdefault(k, Counter()).update([v])
    if param_counts:
        print("\n=== Winner Hyperparameter Frequencies ===")
        for k in sorted(param_counts.keys()):
            counts = param_counts[k]
            items = ", ".join(f"{val}:{cnt}" for val, cnt in counts.most_common(8))
            print(f"{k}: {items}")

        # Boundary check against grid file (if available)
        if grid_file and grid_file.exists():
            try:
                grid = json.loads(grid_file.read_text(encoding="utf-8"))
                combos = grid.get("combinations") or []
                grid_vals: dict[str, set] = {}
                for combo in combos:
                    for k, v in combo.items():
                        if k == "epochs":
                            continue
                        grid_vals.setdefault(k, set()).add(v)
                print("\n=== Boundary Winner Check (grid min/max) ===")
                for k, vals in sorted(grid_vals.items()):
                    if not vals or k not in param_counts:
                        continue
                    vals_sorted = sorted(vals)
                    vmin, vmax = vals_sorted[0], vals_sorted[-1]
                    winners = param_counts[k]
                    min_frac = winners.get(vmin, 0) / max(1, sum(winners.values()))
                    max_frac = winners.get(vmax, 0) / max(1, sum(winners.values()))
                    print(f"{k}: min={vmin} ({min_frac:.2f}) max={vmax} ({max_frac:.2f})")
            except Exception as e:
                print(f"Warning: failed to parse grid file for boundary check: {e}")

    # Per-carrier worst cases
    print(f"\n=== Worst {top_k} Carriers (by best_val_loss) ===")
    worst_df = best_df.sort_values("best_val_loss", ascending=False).head(top_k)
    for _, row in worst_df.iterrows():
        cid = row.get("carrier_id")
        loss = row.get("best_val_loss")
        t_loss = row.get("best_train_loss")
        combo = row.get("combo_idx")
        epoch = row.get("best_epoch")
        if t_loss is not None:
            gap = float(loss) - float(t_loss)
            print(
                f"carrier={cid} best_val_loss={loss:.6f} best_train_loss={t_loss:.6f} gap={gap:.6f} "
                f"combo={combo} best_epoch={epoch}"
            )
        else:
            print(f"carrier={cid} best_val_loss={loss:.6f} combo={combo} best_epoch={epoch}")
        params = row.get("params") or {}
        if params:
            short = ", ".join(f"{k}={params[k]}" for k in sorted(params.keys()) if k != "epochs")
            print(f"  params: {short}")


def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak{ts}")
    path.rename(backup)


def _retrain_and_save_simulator(
    df_test: pd.DataFrame,
    carrier_id: int,
    params: dict,
    epochs: int,
    out_dir: Path,
) -> None:
    df_carrier = df_test[df_test["carrier_service_id_anon"] == carrier_id].copy()
    df_carrier = df_carrier.dropna(subset=[cfg.DELIVERY_TIME_TARGET]).reset_index(drop=True)
    if df_carrier.empty:
        raise ValueError(f"No test rows for carrier_id={carrier_id} after dropping NaNs.")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params_full = dict(params)
    params_full["epochs"] = int(max(1, epochs))

    model, _, _ = train_dl_model(
        params_full,
        Xn,
        Xc,
        yt,
        None,
        None,
        None,
        vocab_sizes,
        device,
        return_model=True,
    )

    carrier_id_str = format_carrier_id_for_path(carrier_id)
    bundle_path = out_dir / f"simulator_dl_{carrier_id_str}.pt"
    meta_path = out_dir / f"simulator_dl_{carrier_id_str}.joblib"

    _backup_if_exists(bundle_path)
    _backup_if_exists(meta_path)

    dc_ori_emb_dim = int(params_full.get("dc_ori_embedding_dim", cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM))
    dc_des_emb_dim = int(params_full.get("dc_des_embedding_dim", cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_scaler": x_scaler,
            "categorical_encoders": encoders,
            "vocab_sizes": vocab_sizes,
            "numerical_dim": int(X_num.shape[1]),
            "hidden_dim": int(params_full["hidden_dim"]),
            "n_layers": int(params_full["n_layers"]),
            "dropout_p": float(params_full["dropout_p"]),
            "dc_ori_embedding_dim": dc_ori_emb_dim,
            "dc_des_embedding_dim": dc_des_emb_dim,
            # Keep a record of the chosen hyperparameters for reproducibility.
            "grid_params": params_full,
            "carrier_service_id": int(carrier_id),
        },
        bundle_path,
    )
    joblib.dump(
        {"type": "dl", "dl_model_path": str(bundle_path), "carrier_service_id": int(carrier_id)},
        meta_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tune_dir",
        type=Path,
        default=Path("data/models/delivery_time_cs/simulator_tune/dl_grid"),
        help="Directory containing per-task JSON grid results.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["best_val_loss", "best_train_loss"],
        default="best_train_loss",
        help="Metric to pick best combo per carrier (default: best_train_loss).",
    )
    parser.add_argument(
        "--grid_file",
        type=Path,
        default=Path("data/delivery_dl_simulator_grid_search_combinations.json"),
        help="Grid file used to generate combinations (optional; for reporting only).",
    )
    parser.add_argument(
        "--carrier_id",
        type=int,
        default=None,
        help="If set, only export the simulator for this carrier_id (useful for SLURM arrays).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(cfg.DELIVERY_TIME_SIMULATOR_PATH),
        help="Output directory for exported simulator bundles (default: DELIVERY_TIME_SIMULATOR_PATH).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the best combo per carrier; do not retrain/export.",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Print tuning summary and exit (no retrain/export).",
    )
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary report.",
    )
    parser.add_argument(
        "--report_top_k",
        type=int,
        default=5,
        help="How many worst carriers to print in the report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail (non-zero exit) if any expected carrier has no valid tuning result.",
    )
    args = parser.parse_args()

    best, report = _load_best_results(args.tune_dir, args.selection_metric)
    df_all, parse_errors = _load_all_results(args.tune_dir)
    _summarize_results(df_all, parse_errors, args.grid_file, top_k=args.report_top_k)
    if args.report_json:
        summary_payload = {
            "status_counts": df_all["status"].value_counts(dropna=False).to_dict()
            if not df_all.empty else {},
            "carriers_expected": len(expected_carriers()),
            "carriers_with_any_results": int(df_all["carrier_id"].nunique()) if not df_all.empty else 0,
            "parse_errors": parse_errors,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"\nWrote summary JSON to {args.report_json}")
    if args.report_only:
        return
    if not best:
        raise SystemExit(f"No valid tuning results found under: {args.tune_dir}")

    if args.carrier_id is not None:
        best = {int(args.carrier_id): best.get(int(args.carrier_id))} if int(args.carrier_id) in best else {}
        if not best:
            raise SystemExit(f"No valid tuning result found for carrier_id={args.carrier_id} under: {args.tune_dir}")
        # When exporting a single carrier, don't warn about missing others.
        report = {"missing_no_files": [], "missing_no_ok": {}, "json_parse_errors": {}}

    print("=== Best DL simulator combos (by carrier) ===")
    for carrier_id in sorted(best.keys()):
        payload = best[carrier_id]
        print(
            f"carrier={carrier_id} combo={payload.get('combo_idx')} "
            f"best_val_loss={payload.get('best_val_loss'):.6f} best_epoch={payload.get('best_epoch')} "
            f"path={payload.get('path')}"
        )

    missing = []
    if args.carrier_id is None:
        missing = sorted(set(expected_carriers()) - set(best.keys()))
    if missing:
        print("\nWARNING: Missing valid tuning results for some carriers.")
        if report.get("missing_no_files"):
            print(f"  no JSONs found: {sorted(report['missing_no_files'])}")
        if report.get("missing_no_ok"):
            missing_no_ok = sorted(report["missing_no_ok"].keys())
            print(f"  JSONs found but none status=ok: {missing_no_ok}")
            for carrier_id in missing_no_ok:
                info = report["missing_no_ok"].get(carrier_id) or {}
                n_files = info.get("n_files")
                status_counts = info.get("status_counts")
                print(f"    carrier={carrier_id} n_files={n_files} status_counts={status_counts}")
        if report.get("json_parse_errors"):
            parse_err_carriers = sorted(report["json_parse_errors"].keys())
            print(f"  JSON parse errors seen for: {parse_err_carriers}")
        print(
            "  Tip: rerun grid tuning for the missing carriers (see --min_rows if you only see skipped_insufficient_data), "
            "or export a single carrier via --carrier_id."
        )
        if args.strict:
            raise SystemExit(f"Missing valid tuning results for carriers: {missing}")

    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_test = load_test_cs_data()

    print("\n=== Retraining on full TEST set + exporting simulator bundles ===")
    for carrier_id in sorted(best.keys()):
        payload = best[carrier_id]
        params = payload.get("params") or {}
        best_epoch = int(payload.get("best_epoch") or cfg.DELIVERY_DL_EPOCHS)
        print(f"carrier={carrier_id} epochs={best_epoch} ...", flush=True)
        _retrain_and_save_simulator(df_test, carrier_id, params, best_epoch, args.out_dir)
        print(f"  wrote simulator_dl_{format_carrier_id_for_path(carrier_id)}.(pt|joblib)")

    print("\nDone.")


if __name__ == "__main__":
    main()
