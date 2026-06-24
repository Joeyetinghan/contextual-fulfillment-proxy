#!/usr/bin/env python3
"""
Aggregate per-day simulation summary JSONs into a single table.

Default layout expected (peak runs):
  data/peak/simulation_results/<order_set>/<YYYY-MM-DD>/solutions_eval/*_summary.json

Outputs:
  - a "long" CSV with one row per (date, run_id)
  - an optional "wide" pivot table (dates x run_id) for a chosen metric

Dedicated analyses:
  - UB/LB bounds summary: scripts.analysis.summarize_sim_bounds
  - Runtime/gap vs instance size: scripts.analysis.analyze_runtime_gap_by_size
  - Intraday trajectory plots: scripts.analysis.plot_intraday_trajectory
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.sim_summary_common import (
    _build_filtered_summary_df,
    _dedupe_latest_method_rows,
    _group_columns,
    _latex_escape,
    _method_label,
    _should_collapse_proxy_name,
)

_MAIN_ALGO_DEFAULTS = ["csaa", "greedy", "empirical", "empirical_saa", "pto", "proxy", "dtlp_bidprice", "primal_dual"]
_APPENDIX_ALGO_EXTRAS: list[str] = []


def _suffix_path(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _algo_hatch_map(algos: list[str]) -> dict[str, str]:
    """Assign deterministic hatch styles by algorithm family."""
    patterns = ["", "//", "\\\\", "xx", "..", "++", "oo"]
    norm_algos = sorted({str(algo).strip().lower() for algo in algos if str(algo).strip()})
    return {
        algo: patterns[i % len(patterns)]
        for i, algo in enumerate(norm_algos)
    }


def _write_delta_tables(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    baseline_label: str,
    delta_out: Path,
    delta_long_out: Path,
    delta_metrics: list[str] | None,
) -> None:
    # Identify numeric metrics for delta computation
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if delta_metrics:
        missing = [m for m in delta_metrics if m not in numeric_cols]
        if missing:
            raise ValueError(f"Delta metric(s) not found or non-numeric: {missing}")
        numeric_cols = [m for m in numeric_cols if m in set(delta_metrics)]
    id_cols = ["simulation_date", "run_id", "algo"]
    keep_cols = [c for c in id_cols if c in df.columns]

    merged = df.merge(
        baseline_df[["simulation_date"] + numeric_cols],
        on="simulation_date",
        how="inner",
        suffixes=("", "_baseline"),
    )
    if "run_id" in merged.columns and "run_id_baseline" in merged.columns:
        merged = merged[merged["run_id"] != merged["run_id_baseline"]]

    # Compute deltas
    for col in numeric_cols:
        merged[f"delta_{col}"] = merged[col] - merged[f"{col}_baseline"]

    # Save per-date deltas
    delta_long_out.parent.mkdir(parents=True, exist_ok=True)
    merged[keep_cols + [c for c in merged.columns if c.startswith("delta_")]].to_csv(
        delta_long_out,
        index=False,
    )
    print(f"Wrote per-date delta table to {delta_long_out}")

    # Aggregate deltas across dates
    delta_cols = [c for c in merged.columns if c.startswith("delta_")]
    group_cols = ["run_id"] if "run_id" in merged.columns else []
    if "proxy_model_name" in merged.columns:
        group_cols.append("proxy_model_name")
    if "proxy_repair_strategy" in merged.columns:
        group_cols.append("proxy_repair_strategy")
    if "algo" in merged.columns:
        group_cols.append("algo")

    delta_agg = merged.groupby(group_cols, dropna=False)[delta_cols].agg(["mean", "std", "count"]).reset_index()
    delta_out.parent.mkdir(parents=True, exist_ok=True)
    delta_agg.to_csv(delta_out, index=False)
    print(f"Wrote delta-vs-baseline table to {delta_out} (baseline={baseline_label})")


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flat_cols = []
    for col in out.columns:
        if isinstance(col, tuple):
            left, right = col
            if right == "":
                flat_cols.append(str(left))
            else:
                flat_cols.append(f"{left}_{right}")
        else:
            flat_cols.append(str(col))
    out.columns = flat_cols
    return out


def _fmt_mean_std(mean: Any, std: Any, digits: int = 3) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"


def _interval_half_width(std: Any, count: Any, mode: str) -> float | None:
    if pd.isna(std) or pd.isna(count):
        return None
    n = int(count)
    if n <= 0:
        return None
    se = float(std) / math.sqrt(n)
    if mode == "none":
        return None
    if mode == "se":
        return se
    if mode == "2se":
        return 2.0 * se
    if mode == "ci95":
        return 1.96 * se
    return None


def _format_interval_value(mean: Any, std: Any, count: Any, digits: int, mode: str) -> str:
    if pd.isna(mean):
        return "NA"
    value = f"{float(mean):,.{digits}f}"
    half = _interval_half_width(std, count, mode)
    if half is None:
        return value
    return f"{value} $\\pm$ {half:,.{digits}f}"


def _build_group_agg(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    group_cols = ["algo"]
    if "proxy_model_name" in df.columns:
        group_cols.append("proxy_model_name")
    if "proxy_repair_strategy" in df.columns:
        group_cols.append("proxy_repair_strategy")

    agg = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    agg = _flatten_multiindex_columns(agg)
    if "proxy_model_name" in agg.columns:
        agg["proxy_model_name"] = agg["proxy_model_name"].fillna("-")
    if "proxy_repair_strategy" in agg.columns:
        agg["proxy_repair_strategy"] = agg["proxy_repair_strategy"].fillna("-")
    return agg


def _pooled_replication_stats(group_df: pd.DataFrame, metric_prefix: str) -> dict[str, float] | None:
    mean_col = f"{metric_prefix}_rep_mean"
    std_col = f"{metric_prefix}_rep_std"
    n_col = "replications_observed"
    if any(col not in group_df.columns for col in (mean_col, std_col, n_col)):
        return None

    sub = group_df[[mean_col, std_col, n_col]].copy()
    sub[mean_col] = pd.to_numeric(sub[mean_col], errors="coerce")
    sub[std_col] = pd.to_numeric(sub[std_col], errors="coerce")
    sub[n_col] = pd.to_numeric(sub[n_col], errors="coerce")
    if metric_prefix == "realized_cost" and "orders_evaluated" in group_df.columns:
        orders = pd.to_numeric(group_df["orders_evaluated"], errors="coerce")
        sub[mean_col] = sub[mean_col] * orders
        sub[std_col] = sub[std_col] * orders
    sub = sub.dropna()
    sub = sub[sub[n_col] > 0]
    if sub.empty:
        return None

    n = sub[n_col].to_numpy(dtype=float)
    m = sub[mean_col].to_numpy(dtype=float)
    s = sub[std_col].to_numpy(dtype=float)
    total_n = float(np.sum(n))
    if total_n <= 0:
        return None

    pooled_mean = float(np.sum(n * m) / total_n)
    if total_n > 1:
        var_num = float(np.sum((n - 1.0) * (s ** 2) + n * ((m - pooled_mean) ** 2)))
        pooled_var = var_num / (total_n - 1.0)
    else:
        pooled_var = 0.0
    pooled_std = float(np.sqrt(max(0.0, pooled_var)))
    pooled_se = float(pooled_std / np.sqrt(total_n))
    return {
        "mean": pooled_mean,
        "std": pooled_std,
        "se": pooled_se,
        "count": total_n,
    }


def _compute_group_raw_stats(
    df_norm: pd.DataFrame,
    group_cols: list[str],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """
    Compute group-level replication and quantile stats directly from raw parquet files.

    Replication stats:
      - pool rows across dates by replication id.
      - realized_cost is aggregated as SUM over orders per replication.
      - late_delivery_pct and cumulative_lateness are aggregated as MEAN over orders per replication.
      - uncertainty is derived from these replication-level aggregates.
    Quantiles:
      - computed over replication-level means (replication quantiles).
    """
    if "parquet_path" not in df_norm.columns:
        return {}

    metric_cols = ["realized_cost", "late_delivery_pct", "cumulative_lateness"]
    quantile_levels = {"p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95}
    results: dict[tuple[Any, ...], dict[str, Any]] = {}

    for key, gdf in df_norm.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rep_accum: pd.DataFrame | None = None

        parquet_paths = [Path(p) for p in gdf.get("parquet_path", pd.Series(dtype=str)).dropna().unique()]
        for parquet_path in parquet_paths:
            if not parquet_path.exists():
                continue
            try:
                raw = pd.read_parquet(parquet_path, columns=["replication"] + metric_cols)
            except Exception:
                continue
            if raw.empty:
                continue

            raw["replication"] = pd.to_numeric(raw["replication"], errors="coerce")
            raw = raw.dropna(subset=["replication"])
            if raw.empty:
                continue
            raw["replication"] = raw["replication"].astype(int)

            rep_group = raw.groupby("replication", as_index=True)[metric_cols].agg(["sum", "count"])
            rep_accum = rep_group if rep_accum is None else rep_accum.add(rep_group, fill_value=0.0)

        group_payload: dict[str, Any] = {"rep": {}, "rep_quantiles": {}}

        if rep_accum is not None:
            for metric in metric_cols:
                sum_col = (metric, "sum")
                cnt_col = (metric, "count")
                if sum_col not in rep_accum.columns or cnt_col not in rep_accum.columns:
                    continue
                sums = rep_accum[sum_col].to_numpy(dtype=float)
                cnts = rep_accum[cnt_col].to_numpy(dtype=float)
                valid = cnts > 0
                if not np.any(valid):
                    continue
                if metric == "realized_cost":
                    rep_values = sums[valid]
                else:
                    rep_values = sums[valid] / cnts[valid]
                n_rep = int(rep_values.size)
                if n_rep == 0:
                    continue
                mean_val = float(np.mean(rep_values))
                if n_rep > 1:
                    std_val = float(np.std(rep_values, ddof=1))
                else:
                    std_val = 0.0
                se_val = float(std_val / np.sqrt(n_rep))
                group_payload["rep"][metric] = {
                    "mean": mean_val,
                    "std": std_val,
                    "se": se_val,
                    "count": float(n_rep),
                }
                group_payload["rep_quantiles"][metric] = {
                    q_name: float(np.quantile(rep_values, q_level))
                    for q_name, q_level in quantile_levels.items()
                }

        results[key] = group_payload

    return results


def _build_latex_summary_table(
    df: pd.DataFrame,
    order_set: str,
    ci_mode: str,
    collapse_proxy_name: bool = False,
) -> str | None:
    metrics_needed = [
        "avg_realized_cost",
        "avg_late_delivery_pct",
        "avg_cumulative_lateness",
        "avg_policy_runtime_ms",
        "realized_cost_eval_p50",
        "realized_cost_eval_p75",
        "realized_cost_eval_p90",
        "realized_cost_eval_p95",
        "late_delivery_pct_eval_p50",
        "late_delivery_pct_eval_p75",
        "late_delivery_pct_eval_p90",
        "late_delivery_pct_eval_p95",
        "cumulative_lateness_eval_p50",
        "cumulative_lateness_eval_p75",
        "cumulative_lateness_eval_p90",
        "cumulative_lateness_eval_p95",
    ]
    present = [m for m in metrics_needed if m in df.columns]
    if not present:
        return None

    group_cols = _group_columns(df)
    df_norm = df.copy()
    for col in group_cols:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col].fillna("-")

    agg = _build_group_agg(df_norm, present)
    raw_group_stats = _compute_group_raw_stats(df_norm, group_cols)

    methods = [_method_label(row, collapse_proxy_name=collapse_proxy_name) for _, row in agg.iterrows()]
    if not methods:
        return None

    row_specs = [
        ("avg_realized_cost", "Total Objective Value ($\\downarrow$)", 2),
        ("avg_late_delivery_pct", "Avg Late Delivery Rate (\\%) ($\\downarrow$)", 2),
        ("avg_cumulative_lateness", "Avg Cumulative Lateness ($\\downarrow$)", 3),
        ("avg_policy_runtime_ms", "Avg Run Time (s) ($\\downarrow$)", 3),
        ("realized_cost_eval_p50", "Objective q50 ($\\downarrow$)", 2),
        ("realized_cost_eval_p75", "Objective q75 ($\\downarrow$)", 2),
        ("realized_cost_eval_p90", "Objective q90 ($\\downarrow$)", 2),
        ("realized_cost_eval_p95", "Objective q95 ($\\downarrow$)", 2),
        ("late_delivery_pct_eval_p50", "Late Delivery q50 (\\%) ($\\downarrow$)", 2),
        ("late_delivery_pct_eval_p75", "Late Delivery q75 (\\%) ($\\downarrow$)", 2),
        ("late_delivery_pct_eval_p90", "Late Delivery q90 (\\%) ($\\downarrow$)", 2),
        ("late_delivery_pct_eval_p95", "Late Delivery q95 (\\%) ($\\downarrow$)", 2),
        ("cumulative_lateness_eval_p50", "Cumulative Lateness q50 ($\\downarrow$)", 3),
        ("cumulative_lateness_eval_p75", "Cumulative Lateness q75 ($\\downarrow$)", 3),
        ("cumulative_lateness_eval_p90", "Cumulative Lateness q90 ($\\downarrow$)", 3),
        ("cumulative_lateness_eval_p95", "Cumulative Lateness q95 ($\\downarrow$)", 3),
    ]
    row_specs = [spec for spec in row_specs if f"{spec[0]}_mean" in agg.columns or spec[0].endswith("_eval_p50") or spec[0].endswith("_eval_p75") or spec[0].endswith("_eval_p90") or spec[0].endswith("_eval_p95")]
    if not row_specs:
        return None

    ci_label = {
        "none": "mean only",
        "se": "mean $\\pm$ SE",
        "2se": "mean $\\pm$ 2SE",
        "ci95": "mean $\\pm$ 95\\% CI",
    }.get(ci_mode, "mean $\\pm$ 95\\% CI")

    lines: list[str] = []
    lines.append("\\begin{table}[!ht]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Cumulative realized metrics across simulation dates for order set={_latex_escape(order_set)}. "
        f"Total objective plus avg late/cumulative rows use replication uncertainty from raw parquet ({ci_label}); "
        "runtime and replication-quantile rows (q50/q75/q90/q95) are point estimates.}"
    )
    lines.append(f"\\label{{tab:sim_summary_{_latex_escape(order_set)}}}")
    lines.append("\\begin{adjustbox}{width=\\linewidth,center}")
    lines.append("\\begin{tabular}{" + "l" + ("r" * len(methods)) + "}")
    lines.append("\\toprule")
    header = ["\\textbf{Metric}"] + [f"\\textbf{{{_latex_escape(method)}}}" for method in methods]
    header_line_idx = len(lines)
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    rep_metric_map = {
        "avg_realized_cost": "realized_cost",
        "avg_late_delivery_pct": "late_delivery_pct",
        "avg_cumulative_lateness": "cumulative_lateness",
    }
    quantile_metric_map = {
        "realized_cost_eval_p50": ("realized_cost", "p50"),
        "realized_cost_eval_p75": ("realized_cost", "p75"),
        "realized_cost_eval_p90": ("realized_cost", "p90"),
        "realized_cost_eval_p95": ("realized_cost", "p95"),
        "late_delivery_pct_eval_p50": ("late_delivery_pct", "p50"),
        "late_delivery_pct_eval_p75": ("late_delivery_pct", "p75"),
        "late_delivery_pct_eval_p90": ("late_delivery_pct", "p90"),
        "late_delivery_pct_eval_p95": ("late_delivery_pct", "p95"),
        "cumulative_lateness_eval_p50": ("cumulative_lateness", "p50"),
        "cumulative_lateness_eval_p75": ("cumulative_lateness", "p75"),
        "cumulative_lateness_eval_p90": ("cumulative_lateness", "p90"),
        "cumulative_lateness_eval_p95": ("cumulative_lateness", "p95"),
    }

    sort_keys: list[tuple[float, int]] = []
    for idx, row in agg.iterrows():
        key = tuple(row[col] for col in group_cols if col in row.index)
        rep_stats = raw_group_stats.get(key, {}).get("rep", {}).get("realized_cost")
        if rep_stats is not None:
            sort_keys.append((float(rep_stats["mean"]), idx))
        else:
            sort_keys.append((float(row.get("avg_realized_cost_mean", np.inf)), idx))
    ordered_idx = [idx for _, idx in sorted(sort_keys, key=lambda x: x[0])]
    agg = agg.iloc[ordered_idx].reset_index(drop=True)
    methods = [_method_label(row, collapse_proxy_name=collapse_proxy_name) for _, row in agg.iterrows()]
    lines[header_line_idx] = " & ".join(["\\textbf{Metric}"] + [f"\\textbf{{{_latex_escape(method)}}}" for method in methods]) + " \\\\"

    for metric, label, digits in row_specs:
        vals: list[str] = []
        for _, row in agg.iterrows():
            key = tuple(row[col] for col in group_cols if col in row.index)
            group_df = None

            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            count = row.get(f"{metric}_count")
            mode = ci_mode

            rep_prefix = rep_metric_map.get(metric)
            if rep_prefix is not None:
                rep_stats = raw_group_stats.get(key, {}).get("rep", {}).get(rep_prefix)
                if rep_stats is not None:
                    mean = rep_stats["mean"]
                    std = rep_stats["std"]
                    count = rep_stats["count"]
                else:
                    # Backward-compatible fallback when raw parquet is unavailable.
                    if group_df is None:
                        mask = pd.Series(True, index=df_norm.index)
                        for col in group_cols:
                            if col in df_norm.columns and col in row.index:
                                mask &= (df_norm[col] == row[col])
                        group_df = df_norm[mask]
                    pooled = _pooled_replication_stats(group_df, rep_prefix)
                    if pooled is not None:
                        mean = pooled["mean"]
                        std = pooled["std"]
                        count = pooled["count"]

            if metric == "avg_policy_runtime_ms":
                mode = "none"
                if not pd.isna(mean):
                    mean = float(mean) / 1000.0
                    if not pd.isna(std):
                        std = float(std) / 1000.0
            elif metric in quantile_metric_map:
                mode = "none"
                q_metric, q_name = quantile_metric_map[metric]
                q_val = raw_group_stats.get(key, {}).get("rep_quantiles", {}).get(q_metric, {}).get(q_name)
                if q_val is not None:
                    mean = q_val
                    std = np.nan
                    count = np.nan

            vals.append(_format_interval_value(mean, std, count, digits=digits, mode=mode))
        lines.append(f"{label} & " + " & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _row_replication_stats_from_raw(row: pd.Series, metric_prefix: str) -> dict[str, float] | None:
    parquet_path_val = row.get("parquet_path")
    if pd.isna(parquet_path_val):
        return None
    parquet_path = Path(str(parquet_path_val))
    if not parquet_path.exists():
        return None
    try:
        raw = pd.read_parquet(parquet_path, columns=["replication", metric_prefix])
    except Exception:
        return None
    if raw.empty or "replication" not in raw.columns or metric_prefix not in raw.columns:
        return None

    raw["replication"] = pd.to_numeric(raw["replication"], errors="coerce")
    raw[metric_prefix] = pd.to_numeric(raw[metric_prefix], errors="coerce")
    raw = raw.dropna(subset=["replication", metric_prefix])
    if raw.empty:
        return None

    rep_series = raw.groupby(raw["replication"].astype(int))[metric_prefix]
    if metric_prefix == "realized_cost":
        rep_values = rep_series.sum().to_numpy(dtype=float)
    else:
        rep_values = rep_series.mean().to_numpy(dtype=float)
    n_rep = int(rep_values.size)
    if n_rep == 0:
        return None
    mean_val = float(np.mean(rep_values))
    if n_rep > 1:
        std_val = float(np.std(rep_values, ddof=1))
    else:
        std_val = 0.0
    return {"mean": mean_val, "std": std_val, "count": float(n_rep)}


def _row_replication_stats_from_summary(row: pd.Series, metric_prefix: str) -> dict[str, float] | None:
    mean_col = f"{metric_prefix}_rep_mean"
    std_col = f"{metric_prefix}_rep_std"
    n_col = "replications_observed"
    if mean_col not in row.index:
        return None
    mean_val = pd.to_numeric(pd.Series([row.get(mean_col)]), errors="coerce").iloc[0]
    std_val = pd.to_numeric(pd.Series([row.get(std_col)]), errors="coerce").iloc[0] if std_col in row.index else np.nan
    cnt_val = pd.to_numeric(pd.Series([row.get(n_col)]), errors="coerce").iloc[0] if n_col in row.index else np.nan
    if pd.isna(mean_val):
        return None
    if pd.isna(std_val):
        std_val = 0.0
    if metric_prefix == "realized_cost":
        orders_val = pd.to_numeric(pd.Series([row.get("orders_evaluated")]), errors="coerce").iloc[0]
        if not pd.isna(orders_val):
            mean_val = float(mean_val) * float(orders_val)
            std_val = float(std_val) * float(orders_val)
    if pd.isna(cnt_val) or cnt_val <= 0:
        cnt_val = np.nan
    return {"mean": float(mean_val), "std": float(std_val), "count": float(cnt_val) if not pd.isna(cnt_val) else np.nan}


def _plot_avg_ci_across_dates(
    df: pd.DataFrame,
    metric_prefix: str,
    ci_mode: str,
    out_path: Path,
    collapse_proxy_name: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Skipping plot generation: matplotlib is not available.")
        return

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        stats = _row_replication_stats_from_raw(row, metric_prefix)
        if stats is None:
            stats = _row_replication_stats_from_summary(row, metric_prefix)
        if stats is None:
            continue
        sim_date = pd.to_datetime(row.get("simulation_date"), errors="coerce")
        if pd.isna(sim_date):
            continue
        rows.append(
            {
                "simulation_date": sim_date,
                "method_label": _method_label(row, collapse_proxy_name=collapse_proxy_name),
                "algo": str(row.get("algo", "")).strip().lower(),
                "mean": stats["mean"],
                "std": stats["std"],
                "count": stats["count"],
            }
        )

    if not rows:
        print(f"Skipping plot generation for {metric_prefix}: no replication stats found.")
        return

    plot_df = pd.DataFrame(rows)
    # Defensive: enforce one row per (method, date). With proper proxy filtering
    # there are none, but guard against accidental collapse (e.g. several proxy
    # inference strategies sharing a label) so the per-method reindex below cannot
    # raise "cannot reindex on an axis with duplicate labels".
    _dup = plot_df.duplicated(subset=["method_label", "simulation_date"], keep=False)
    if _dup.any():
        dup_labels = sorted(plot_df.loc[_dup, "method_label"].astype(str).unique())
        print(
            f"[plot] WARNING: duplicate (method, date) rows for {dup_labels}; "
            "keeping the lowest-mean row each. Filter to one proxy to avoid this."
        )
        plot_df = (
            plot_df.sort_values("mean")
            .drop_duplicates(subset=["method_label", "simulation_date"], keep="first")
        )
    date_order = sorted(plot_df["simulation_date"].dropna().unique())
    method_order = (
        plot_df.groupby("method_label", dropna=False)["mean"].mean().sort_values(ascending=True).index.tolist()
    )
    # Colorblind-safe qualitative palette: Okabe-Ito (8) extended with Tol muted
    # colors (12 total) so each method gets a distinct, colorblind-friendly color.
    colorblind_palette = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#000000",
        "#332288", "#117733", "#88CCEE", "#882255",
    ]
    if len(method_order) > len(colorblind_palette):
        print(
            f"[plot] WARNING: {len(method_order)} methods exceed the "
            f"{len(colorblind_palette)}-color palette; colors will repeat."
        )
    hatch_map = _algo_hatch_map(plot_df["algo"].dropna().astype(str).tolist())
    n_methods = max(1, len(method_order))
    x = np.arange(len(date_order), dtype=float)
    width = min(0.8 / n_methods, 0.35)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(date_order)), 5.5))
    for i, method in enumerate(method_order):
        sub = (
            plot_df[plot_df["method_label"] == method]
            .set_index("simulation_date")
            .reindex(date_order)
        )
        means = sub["mean"].to_numpy(dtype=float)
        stds = sub["std"].to_numpy(dtype=float)
        counts = sub["count"].to_numpy(dtype=float)
        errs = np.array(
            [
                (_interval_half_width(stds[j], counts[j], ci_mode) or 0.0)
                if not np.isnan(means[j]) else 0.0
                for j in range(len(means))
            ],
            dtype=float,
        )
        valid = ~np.isnan(means)
        if not np.any(valid):
            continue
        offset = (i - (n_methods - 1) / 2.0) * width
        algo_name = str(sub["algo"].dropna().iloc[0]).strip().lower() if "algo" in sub.columns and not sub["algo"].dropna().empty else ""
        ax.bar(
            x[valid] + offset,
            means[valid],
            width=width,
            yerr=errs[valid],
            capsize=3,
            label=method,
            color=colorblind_palette[i % len(colorblind_palette)],
            alpha=0.9,
            hatch=hatch_map.get(algo_name, ""),
            edgecolor="#333333",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(d).strftime("%Y-%m-%d") for d in date_order], rotation=45, ha="right")
    y_label_map = {
        "realized_cost": "Total realized cost",
        "late_delivery_pct": "Avg late delivery rate (%)",
        "cumulative_lateness": "Avg cumulative lateness",
    }
    ax.set_ylabel(y_label_map.get(metric_prefix, metric_prefix))
    ci_label = {"none": "none", "se": "SE", "2se": "2SE", "ci95": "95% CI"}.get(ci_mode, ci_mode)
    ax.set_title(f"{y_label_map.get(metric_prefix, metric_prefix)} across dates ({ci_label})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote CI bar plot to {out_path}")


def _print_structured_summary(df: pd.DataFrame) -> None:
    metric_candidates = [
        "avg_realized_cost",
        "avg_lost_sales_qty",
        "avg_late_delivery_pct",
        "avg_cumulative_lateness",
        "avg_policy_runtime_ms",
        # Evaluation-level quantiles (if present in summary JSON).
        "realized_cost_eval_p50",
        "realized_cost_eval_p75",
        "realized_cost_eval_p90",
        "realized_cost_eval_p95",
        "realized_cost_eval_p99",
        "late_delivery_pct_eval_p50",
        "late_delivery_pct_eval_p75",
        "late_delivery_pct_eval_p90",
        "late_delivery_pct_eval_p95",
        "late_delivery_pct_eval_p99",
        "lost_sales_qty_eval_p50",
        "lost_sales_qty_eval_p75",
        "lost_sales_qty_eval_p90",
        "lost_sales_qty_eval_p95",
        "lost_sales_qty_eval_p99",
        "cumulative_lateness_eval_p50",
        "cumulative_lateness_eval_p75",
        "cumulative_lateness_eval_p90",
        "cumulative_lateness_eval_p95",
        "cumulative_lateness_eval_p99",
    ]
    present = [m for m in metric_candidates if m in df.columns]
    if not present:
        return

    agg = _build_group_agg(df, present)
    group_cols = _group_columns(df)
    df_norm = df.copy()
    for col in group_cols:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col].fillna("-")
    raw_group_stats = _compute_group_raw_stats(df_norm, group_cols)

    if "avg_realized_cost_mean" in agg.columns:
        cost = agg.copy()
        rank_vals: list[float] = []
        cost_display: list[str] = []
        cost_source: list[str] = []
        for _, row in cost.iterrows():
            key = tuple(row[col] for col in group_cols if col in row.index)
            rep_stats = raw_group_stats.get(key, {}).get("rep", {}).get("realized_cost")
            if rep_stats is not None:
                rank_val = float(rep_stats["mean"])
                rank_vals.append(rank_val)
                cost_display.append(
                    _format_interval_value(
                        rep_stats["mean"],
                        rep_stats["std"],
                        rep_stats["count"],
                        digits=3,
                        mode="ci95",
                    )
                )
                cost_source.append("raw_rep_total")
            else:
                rank_val = float(row.get("avg_realized_cost_mean", np.nan))
                rank_vals.append(rank_val)
                cost_display.append(
                    _fmt_mean_std(row.get("avg_realized_cost_mean"), row.get("avg_realized_cost_std"), digits=3)
                )
                cost_source.append("summary_avg")

        cost["cost_rank_value"] = rank_vals
        cost["objective_mean_ci95"] = cost_display
        cost["cost_source"] = cost_source
        cost = cost.sort_values("cost_rank_value", ascending=True).reset_index(drop=True)
        best = float(cost.loc[0, "cost_rank_value"])
        cost["rank"] = range(1, len(cost) + 1)
        cost["delta_vs_best_cost"] = cost["cost_rank_value"] - best
        cols = ["rank", "algo"]
        if "proxy_model_name" in cost.columns:
            cols.append("proxy_model_name")
        if "proxy_repair_strategy" in cost.columns:
            cols.append("proxy_repair_strategy")
        cols.extend(["objective_mean_ci95", "delta_vs_best_cost", "cost_source"])
        print("\n=== Cost-First Leaderboard (aligned with LaTeX objective) ===")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
            print(cost[cols].to_string(index=False))

    service_metrics = [
        ("avg_lost_sales_qty", "lost_sales"),
        ("avg_late_delivery_pct", "late_delivery_pct"),
        ("avg_cumulative_lateness", "cum_lateness"),
    ]
    service_available = [m for m, _ in service_metrics if f"{m}_mean" in agg.columns]
    if service_available:
        service = agg.copy()
        for metric, label in service_metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in service.columns:
                service[f"{label}_mean_std"] = service.apply(
                    lambda r: _fmt_mean_std(r.get(mean_col), r.get(std_col), digits=3),
                    axis=1,
                )
        cols = ["algo"]
        if "proxy_model_name" in service.columns:
            cols.append("proxy_model_name")
        if "proxy_repair_strategy" in service.columns:
            cols.append("proxy_repair_strategy")
        for _, label in service_metrics:
            col = f"{label}_mean_std"
            if col in service.columns:
                cols.append(col)
        print("\n=== Service Quality Summary ===")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
            print(service[cols].to_string(index=False))

    if "avg_policy_runtime_ms_mean" in agg.columns:
        runtime = agg.copy()
        runtime["runtime_ms_mean_std"] = runtime.apply(
            lambda r: _fmt_mean_std(r.get("avg_policy_runtime_ms_mean"), r.get("avg_policy_runtime_ms_std"), digits=2),
            axis=1,
        )
        cols = ["algo"]
        if "proxy_model_name" in runtime.columns:
            cols.append("proxy_model_name")
        if "proxy_repair_strategy" in runtime.columns:
            cols.append("proxy_repair_strategy")
        cols.append("runtime_ms_mean_std")
        print("\n=== Runtime Summary ===")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
            print(runtime[cols].to_string(index=False))

    tail_metric_specs = [
        ("realized_cost_eval_p50", "cost_p50"),
        ("realized_cost_eval_p75", "cost_p75"),
        ("realized_cost_eval_p90", "cost_p90"),
        ("realized_cost_eval_p95", "cost_p95"),
        ("realized_cost_eval_p99", "cost_p99"),
        ("late_delivery_pct_eval_p50", "late_p50"),
        ("late_delivery_pct_eval_p75", "late_p75"),
        ("late_delivery_pct_eval_p90", "late_p90"),
        ("late_delivery_pct_eval_p95", "late_p95"),
        ("late_delivery_pct_eval_p99", "late_p99"),
        ("cumulative_lateness_eval_p50", "cum_late_p50"),
        ("cumulative_lateness_eval_p75", "cum_late_p75"),
        ("cumulative_lateness_eval_p90", "cum_late_p90"),
        ("cumulative_lateness_eval_p95", "cum_late_p95"),
    ]
    present_tail = [m for m, _ in tail_metric_specs if f"{m}_mean" in agg.columns]
    if present_tail:
        tail = agg.copy()
        for metric, label in tail_metric_specs:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in tail.columns:
                tail[f"{label}_mean_std"] = tail.apply(
                    lambda r: _fmt_mean_std(r.get(mean_col), r.get(std_col), digits=3),
                    axis=1,
                )
        cols = ["algo"]
        if "proxy_model_name" in tail.columns:
            cols.append("proxy_model_name")
        if "proxy_repair_strategy" in tail.columns:
            cols.append("proxy_repair_strategy")
        for _, label in tail_metric_specs:
            col = f"{label}_mean_std"
            if col in tail.columns:
                cols.append(col)
        print("\n=== Distribution Summary (Eval Quantiles) ===")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
            print(tail[cols].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect simulation summary JSONs into a CSV.")
    parser.add_argument("--root", type=Path, default=Path("data/peak/simulation_results"))
    parser.add_argument("--order-set", type=str, default="test", choices=["test", "proxy_train"])
    parser.add_argument("--date", type=str, default=None, help="Optional single date (YYYY-MM-DD) to scan.")
    parser.add_argument("--date-from", type=str, default=None, help="Optional inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--date-to", type=str, default=None, help="Optional inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--require-full-date-coverage",
        dest="require_full_date_coverage",
        action="store_true",
        default=True,
        help=(
            "Exclude runs missing one or more dates in the selected date set "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--allow-partial-date-coverage",
        dest="require_full_date_coverage",
        action="store_false",
        help="Keep runs even when some dates are missing.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional algo prefix filter (repeatable). "
            "Default when omitted: csaa, greedy, empirical/empirical_saa, pto, proxy, "
            "dtlp_bidprice, primal_dual (all main algorithms)."
        ),
    )
    parser.add_argument(
        "--appendix",
        action="store_true",
        help="Deprecated/no-op: dtlp_bidprice and primal_dual are now main algorithms (included by default).",
    )
    parser.add_argument(
        "--proxy-model-name",
        type=str,
        action="append",
        default=None,
        help="Exact proxy_model_name to include (repeatable).",
    )
    parser.add_argument(
        "--proxy-model-contains",
        type=str,
        action="append",
        default=None,
        help="Substring match on proxy_model_name to include (repeatable, case-insensitive).",
    )
    parser.add_argument(
        "--include-max-orders",
        action="store_true",
        help="Include partial runs tagged with maxN (from --max-orders). Default: ignore.",
    )
    parser.add_argument("--out", type=Path, default=Path("logs/simulation_summaries.csv"))
    parser.add_argument("--pivot-metric", type=str, default=None, help="Optional metric key to pivot (wide table).")
    parser.add_argument("--pivot-out", type=Path, default=Path("logs/simulation_summaries_pivot.csv"))
    parser.add_argument(
        "--baseline-run-id",
        type=str,
        action="append",
        default=None,
        help="Run ID to use as baseline for delta table (repeatable).",
    )
    parser.add_argument(
        "--baseline-algo",
        type=str,
        action="append",
        default=None,
        help="Algo name to use as baseline (repeatable). Uses one run per date for that algo.",
    )
    parser.add_argument(
        "--delta-out",
        type=Path,
        default=Path("logs/simulation_summaries_delta_vs_baseline.csv"),
        help="Output CSV for delta-vs-baseline aggregate table.",
    )
    parser.add_argument(
        "--delta-long-out",
        type=Path,
        default=Path("logs/simulation_summaries_delta_vs_baseline_long.csv"),
        help="Output CSV for per-date delta table.",
    )
    parser.add_argument(
        "--delta-metric",
        type=str,
        action="append",
        default=None,
        help="Metric key to compute deltas for (repeatable). Default: all numeric metrics.",
    )
    parser.add_argument(
        "--latex-ci",
        type=str,
        default="ci95",
        choices=["none", "se", "2se", "ci95"],
        help="Uncertainty format for LaTeX summary table. Default: ci95 (mean +/- 1.96*SE).",
    )
    parser.add_argument(
        "--latex-out",
        type=Path,
        default=None,
        help="Optional path to also write the LaTeX summary table.",
    )
    parser.add_argument(
        "--plot-metric",
        type=str,
        action="append",
        default=None,
        choices=["realized_cost", "late_delivery_pct", "cumulative_lateness"],
        help="Generate date-wise bar plots with replication uncertainty for the selected metric (repeatable).",
    )
    parser.add_argument(
        "--plot-out-dir",
        type=Path,
        default=Path("logs/figures"),
        help="Output directory for generated CI bar plots.",
    )
    parser.add_argument(
        "--plot-ci",
        type=str,
        default="ci95",
        choices=["none", "se", "2se", "ci95"],
        help="Uncertainty band for CI bar plots.",
    )
    args = parser.parse_args()
    if args.date and (args.date_from or args.date_to):
        print("Use either --date or --date-from/--date-to, not both.")
        return 2

    algo_filters = args.algo
    if algo_filters is None:
        algo_filters = list(_MAIN_ALGO_DEFAULTS)
        if args.appendix:
            algo_filters.extend(_APPENDIX_ALGO_EXTRAS)

    try:
        df = _build_filtered_summary_df(
            root=args.root,
            order_set=args.order_set,
            date=args.date,
            date_from=args.date_from,
            date_to=args.date_to,
            algo=algo_filters,
            include_max_orders=args.include_max_orders,
            require_full_date_coverage=args.require_full_date_coverage,
            proxy_model_name=args.proxy_model_name,
            proxy_model_contains=args.proxy_model_contains,
        )
    except ValueError as exc:
        print(str(exc))
        return 2
    if df.empty:
        print("No rows available after filtering.")
        return 2

    df = _dedupe_latest_method_rows(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

    if args.pivot_metric:
        metric = args.pivot_metric
        if metric not in df.columns:
            print(f"Pivot metric '{metric}' not found in columns. Available keys: {sorted(df.columns)}")
            return 2
        wide = df.pivot_table(index="simulation_date", columns="run_id", values=metric, aggfunc="first")
        args.pivot_out.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(args.pivot_out)
        print(f"Wrote pivot table to {args.pivot_out} (metric={metric})")

    baseline_run_ids = args.baseline_run_id or []
    baseline_algos = args.baseline_algo or []
    if baseline_run_ids or baseline_algos:
        known_run_ids = set(df["run_id"].unique()) if "run_id" in df.columns else set()

        for baseline_id in baseline_run_ids:
            if baseline_id not in known_run_ids:
                print(f"Baseline run_id '{baseline_id}' not found in scanned summaries.")
                return 2
            baseline_df = df[df["run_id"] == baseline_id].copy()
            delta_out = _suffix_path(args.delta_out, f"baseline_{baseline_id}")
            delta_long_out = _suffix_path(args.delta_long_out, f"baseline_{baseline_id}")
            _write_delta_tables(df, baseline_df, baseline_id, delta_out, delta_long_out, args.delta_metric)

        for algo in baseline_algos:
            algo_df = df[df.get("algo") == algo].copy() if "algo" in df.columns else pd.DataFrame()
            if algo_df.empty:
                print(f"Baseline algo '{algo}' not found in scanned summaries.")
                return 2

            # One baseline run per date for the algo; if multiple, pick the first run_id per date.
            dup_counts = algo_df.groupby("simulation_date").size()
            if (dup_counts > 1).any():
                print(f"[delta] Warning: multiple '{algo}' runs per date; using first run_id per date.")
            algo_df = algo_df.sort_values("run_id").drop_duplicates("simulation_date")

            delta_out = _suffix_path(args.delta_out, f"baseline_algo_{algo}")
            delta_long_out = _suffix_path(args.delta_long_out, f"baseline_algo_{algo}")
            _write_delta_tables(df, algo_df, f"algo:{algo}", delta_out, delta_long_out, args.delta_metric)

    collapse_proxy_name = _should_collapse_proxy_name(df)

    # Structured console summary across dates
    _print_structured_summary(df)
    latex = _build_latex_summary_table(
        df,
        order_set=args.order_set,
        ci_mode=args.latex_ci,
        collapse_proxy_name=collapse_proxy_name,
    )
    if latex:
        print("\n=== LaTeX Summary Table ===")
        print(latex)
        if args.latex_out is not None:
            args.latex_out.parent.mkdir(parents=True, exist_ok=True)
            args.latex_out.write_text(latex + "\n", encoding="utf-8")
            print(f"Wrote LaTeX table to {args.latex_out}")
    if args.plot_metric:
        for metric_prefix in args.plot_metric:
            plot_out = args.plot_out_dir / f"sim_{args.order_set}_{metric_prefix}_by_date_bar.pdf"
            _plot_avg_ci_across_dates(
                df,
                metric_prefix=metric_prefix,
                ci_mode=args.plot_ci,
                out_path=plot_out,
                collapse_proxy_name=collapse_proxy_name,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
