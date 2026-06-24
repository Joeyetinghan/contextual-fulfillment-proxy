#!/usr/bin/env python3
"""
Summarize proxy ablation validation metrics and compare against a full model.

This script is intentionally focused on ablation reporting and produces
metric-rich CSV/TXT artifacts (not only model-name lists).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.proxy.analyze_tuning_results import (  # noqa: E402
    _apply_final_best_eval_from_json,
    collect_all_results,
    resolve_selection_metric,
    select_best_per_group,
)


DEFAULT_LEADER_METRICS = [
    "min:val_proxy_ub_mean",
    "max:val_joint_hit1_repaired",
    "max:val_joint_hit5_raw",
    "max:val_proxy_le_csaa_frac",
]

SUMMARY_METRICS = [
    "val_proxy_ub_mean",
    "val_proxy_ub_ci95_mean",
    "val_proxy_minus_csaa_bar",
    "val_proxy_le_csaa_frac",
    "val_joint_hit1_repaired",
    "val_joint_hit5_raw",
    "last_val_loss",
    "val_ub_orders_evaluated",
]

ARCH_SWITCH_COLS = [
    "model_variant",
    "use_cost_summary",
    "use_option_features_in_carrier",
    "use_scenario_module",
    "use_dc_module",
    "use_dc_embedding",
    "use_carrier_embedding",
]


def _split_tokens(values: list[str] | None) -> list[str]:
    tokens: list[str] = []
    if not values:
        return tokens
    for raw in values:
        for tok in str(raw).split(","):
            tok = tok.strip()
            if tok:
                tokens.append(tok)
    return tokens


def _derive_ablation_tag(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    m = re.search(r"_abl([a-z0-9_]+)", name)
    return m.group(1) if m else None


def _ensure_group_col(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = df.copy()
    if group_col not in out.columns or not out[group_col].notna().any():
        if "model_name" in out.columns:
            out[group_col] = out["model_name"].apply(_derive_ablation_tag)
        else:
            out[group_col] = np.nan
    return out


def _resolve_full_model_dir(full_model_name: str, models_root: Path) -> Path | None:
    cand = Path(full_model_name)
    if cand.is_file():
        if cand.name == "best.pt":
            return cand.parent
        return None
    if cand.is_dir():
        return cand

    model_dir = models_root / full_model_name
    if model_dir.is_dir():
        return model_dir

    return None


def _load_hparams_architecture(model_dir: Path) -> dict[str, Any]:
    hp_path = model_dir / "hyperparams.json"
    if not hp_path.exists():
        return {}
    try:
        payload = json.loads(hp_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    arch = payload.get("architecture", {})
    if not isinstance(arch, dict):
        return {}
    return {k: arch[k] for k in ARCH_SWITCH_COLS if k in arch}


def _load_full_model_row(
    full_model_name: str | None,
    models_root: Path,
    group_col: str,
) -> dict[str, Any] | None:
    if not full_model_name:
        return None

    model_dir = _resolve_full_model_dir(full_model_name, models_root)
    if model_dir is None:
        print(f"Warning: could not resolve full model '{full_model_name}' under '{models_root}'.")
        return None

    row: dict[str, Any] = {
        group_col: "full_model",
        "model_name": model_dir.name,
        "model_output_name": model_dir.name,
        "model_output_dir": str(model_dir),
        "log_file": "full_model_final_best_eval.json",
    }
    row.update(_load_hparams_architecture(model_dir))
    _apply_final_best_eval_from_json(row)
    if "final_best_eval_json" not in row:
        print(f"Warning: final_best_eval.json not found for full model '{model_dir.name}'.")
        return None
    return row


def _pick_model_name(row: pd.Series) -> str:
    out_name = str(row.get("model_output_name") or "").strip()
    if out_name:
        return out_name
    return str(row.get("model_name") or "")


def _sort_by_metric(df: pd.DataFrame, metric: str, maximize: bool, use_abs: bool) -> pd.DataFrame:
    work = df.dropna(subset=[metric]).copy()
    if work.empty:
        return work
    if use_abs:
        work["_sort_metric"] = work[metric].abs()
        return work.sort_values("_sort_metric", ascending=True).drop(columns=["_sort_metric"])
    return work.sort_values(metric, ascending=not maximize)


def _write_txt_table(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if df.empty:
            f.write("No rows.\n")
            return
        f.write(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n")


def _build_summary_df(
    best_df: pd.DataFrame,
    full_row: dict[str, Any] | None,
    group_col: str,
    selection_metric: str,
    maximize: bool,
    use_abs: bool,
) -> pd.DataFrame:
    work = best_df.copy()
    work["_selected_model_name"] = work.apply(_pick_model_name, axis=1)
    work["_source"] = "ablation_run"

    if full_row:
        full_df = pd.DataFrame([full_row])
        full_df["_selected_model_name"] = full_df.apply(_pick_model_name, axis=1)
        full_df["_source"] = "full_model"
        work = pd.concat([full_df, work], ignore_index=True, sort=False)

    keep_cols = [c for c in [group_col, "_selected_model_name", "_source", "log_file"] + ARCH_SWITCH_COLS + SUMMARY_METRICS if c in work.columns]
    work = work[keep_cols].copy()

    # Rank only ablation rows by the primary selection metric.
    if selection_metric in work.columns:
        ablation_mask = work[group_col] != "full_model"
        ranked = work.loc[ablation_mask].dropna(subset=[selection_metric]).copy()
        if not ranked.empty:
            if use_abs:
                ranked["_rank_metric"] = ranked[selection_metric].abs()
                ranked = ranked.sort_values("_rank_metric", ascending=True)
            else:
                ranked = ranked.sort_values(selection_metric, ascending=not maximize)
            ranks = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
            work["rank_ablation"] = np.nan
            work.loc[ranks.index, "rank_ablation"] = ranks.astype(float)

    # Delta vs full model (where available).
    if full_row:
        full_values = pd.Series(full_row)
        for metric in SUMMARY_METRICS:
            if metric not in work.columns:
                continue
            full_val = full_values.get(metric)
            if full_val is None or pd.isna(full_val):
                continue
            delta_col = f"delta_vs_full_{metric}"
            work[delta_col] = work[metric] - float(full_val)

    rename_map = {
        group_col: "ablation_tag",
        "_selected_model_name": "model_name",
        "_source": "source",
    }
    return work.rename(columns=rename_map)


def _build_leaders_df(
    df: pd.DataFrame,
    metric_specs: list[str],
    group_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_spec in metric_specs:
        try:
            metric, maximize, use_abs, label = resolve_selection_metric(df, metric_spec)
        except ValueError as exc:
            print(f"Skipping leader metric '{metric_spec}': {exc}")
            continue
        ranked = _sort_by_metric(df, metric, maximize=maximize, use_abs=use_abs)
        if ranked.empty:
            continue
        row = ranked.iloc[0]
        rows.append(
            {
                "metric_spec": metric_spec,
                "selection_label": label,
                "ablation_tag": row.get(group_col),
                "model_name": _pick_model_name(row),
                "metric_value": row.get(metric),
                "val_proxy_ub_mean": row.get("val_proxy_ub_mean"),
                "val_joint_hit1_repaired": row.get("val_joint_hit1_repaired"),
                "val_joint_hit5_raw": row.get("val_joint_hit5_raw"),
                "val_proxy_le_csaa_frac": row.get("val_proxy_le_csaa_frac"),
                "last_val_loss": row.get("last_val_loss"),
                "model_variant": row.get("model_variant"),
                "use_cost_summary": row.get("use_cost_summary"),
                "use_option_features_in_carrier": row.get("use_option_features_in_carrier"),
                "use_scenario_module": row.get("use_scenario_module"),
                "use_dc_module": row.get("use_dc_module"),
                "use_dc_embedding": row.get("use_dc_embedding"),
                "use_carrier_embedding": row.get("use_carrier_embedding"),
                "log_file": row.get("log_file"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize ablation validation metrics (with optional full-model comparison)."
    )
    parser.add_argument("--log-dir", type=str, default="logs/tune")
    parser.add_argument("--run-name", action="append", default=None, help="Repeatable; also supports comma-separated values.")
    parser.add_argument("--group-col", type=str, default="ablation_tag")
    parser.add_argument("--selection-metric", type=str, default="min:val_proxy_ub_mean")
    parser.add_argument("--leader-metric", action="append", default=None, help="Repeatable; defaults to four main metrics.")
    parser.add_argument("--models-root", type=Path, default=Path("data/models/proxy"))
    parser.add_argument("--full-model-name", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None, help="Write parsed rows.")
    parser.add_argument("--best-per-group-csv", type=str, default=None, help="Write best model per ablation tag.")
    parser.add_argument("--best-models-txt", type=str, default=None, help="Write selected model names (one per line).")
    parser.add_argument("--summary-csv", type=str, default=None, help="Write summary table with metrics + deltas vs full model.")
    parser.add_argument("--summary-txt", type=str, default=None, help="Write text summary table.")
    parser.add_argument("--leaders-csv", type=str, default=None, help="Write metric leader rows.")
    parser.add_argument("--leaders-txt", type=str, default=None, help="Write metric leader rows as text.")
    args = parser.parse_args()

    run_names = _split_tokens(args.run_name)
    df = collect_all_results(args.log_dir, run_names if run_names else None)
    if df.empty:
        print("No parsed tuning rows; nothing to summarize.")
        return 2

    df = _ensure_group_col(df, args.group_col)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote parsed rows: {args.out_csv}")

    metric, maximize, use_abs, label = resolve_selection_metric(df, args.selection_metric)
    best_df = select_best_per_group(df, args.group_col, metric, maximize, use_abs=use_abs)
    if best_df.empty:
        print(f"No best-per-group rows for selection metric '{args.selection_metric}'.")
        return 2

    sorted_best = _sort_by_metric(best_df, metric, maximize=maximize, use_abs=use_abs)

    if args.best_per_group_csv:
        os.makedirs(os.path.dirname(args.best_per_group_csv) or ".", exist_ok=True)
        sorted_best.to_csv(args.best_per_group_csv, index=False)
        print(f"Wrote best-per-group rows: {args.best_per_group_csv}")

    if args.best_models_txt:
        os.makedirs(os.path.dirname(args.best_models_txt) or ".", exist_ok=True)
        with open(args.best_models_txt, "w", encoding="utf-8") as f:
            for _, row in sorted_best.iterrows():
                name = _pick_model_name(row)
                if name:
                    f.write(f"{name}\n")
        print(f"Wrote model list: {args.best_models_txt}")

    full_row = _load_full_model_row(args.full_model_name, args.models_root, args.group_col)
    summary_df = _build_summary_df(
        best_df=sorted_best,
        full_row=full_row,
        group_col=args.group_col,
        selection_metric=metric,
        maximize=maximize,
        use_abs=use_abs,
    )
    print(f"Primary selection metric: {label}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if args.summary_csv:
        os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
        summary_df.to_csv(args.summary_csv, index=False)
        print(f"Wrote summary CSV: {args.summary_csv}")
    if args.summary_txt:
        _write_txt_table(summary_df, args.summary_txt)
        print(f"Wrote summary TXT: {args.summary_txt}")

    leader_metrics = _split_tokens(args.leader_metric) or DEFAULT_LEADER_METRICS
    leaders_df = _build_leaders_df(df, leader_metrics, args.group_col)
    if not leaders_df.empty:
        print("\nMetric leaders:")
        print(leaders_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    else:
        print("\nNo leader rows produced from requested metrics.")

    if args.leaders_csv:
        os.makedirs(os.path.dirname(args.leaders_csv) or ".", exist_ok=True)
        leaders_df.to_csv(args.leaders_csv, index=False)
        print(f"Wrote leaders CSV: {args.leaders_csv}")
    if args.leaders_txt:
        _write_txt_table(leaders_df, args.leaders_txt)
        print(f"Wrote leaders TXT: {args.leaders_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
