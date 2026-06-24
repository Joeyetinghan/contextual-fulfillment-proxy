#!/usr/bin/env python3
"""
Dedicated policy-bound summarizer for simulation outputs.

Separates UB/LB reporting from collect_sim_summaries.py to keep workflows cleaner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.sim_summary_common import (
    _build_bound_vs_realized_ci_summary,
    _build_filtered_summary_df,
    _build_bounds_coverage_summary,
    _build_bounds_latex_table,
    _build_bounds_summary,
    _plot_bound_vs_realized_ci_comparison,
    _plot_bounds_comparison,
    _should_collapse_proxy_name,
)


def _dedupe_bound_rows(df):
    if df.empty or "simulation_date" not in df.columns:
        return df

    key_cols = ["simulation_date"] + [
        col for col in ("algo", "proxy_model_name", "proxy_repair_strategy") if col in df.columns
    ]
    if not key_cols:
        return df

    work = df.copy()
    work["_stem"] = work["path"].apply(lambda p: Path(str(p)).stem.lower()) if "path" in work.columns else ""
    if "algo" in work.columns:
        work["_algo"] = work["algo"].fillna("").astype(str).str.lower()
    else:
        work["_algo"] = ""
    work["_is_default"] = (
        (work["_stem"] == (work["_algo"] + "_summary"))
        | (work["_stem"] == (work["_algo"] + "_peak_summary"))
    )
    if "proxy_run_tag" in work.columns:
        work["_proxy_run_tag"] = work["proxy_run_tag"].fillna("").astype(str).str.strip()
    else:
        work["_proxy_run_tag"] = ""
    work["_has_tag"] = work["_proxy_run_tag"] != ""
    current_stage_total = (
        work["policy_current_stage_total"] if "policy_current_stage_total" in work.columns else None
    )
    current_stage_ci95 = (
        work["policy_current_stage_total_ci95"] if "policy_current_stage_total_ci95" in work.columns else None
    )
    if current_stage_total is not None and current_stage_ci95 is not None:
        work["_has_current_stage"] = (
            pd.to_numeric(current_stage_total, errors="coerce").notna()
            & pd.to_numeric(current_stage_ci95, errors="coerce").notna()
        )
    else:
        work["_has_current_stage"] = False
    if "path" in work.columns:
        work["_summary_mtime"] = work["path"].apply(
            lambda p: Path(str(p)).stat().st_mtime if Path(str(p)).exists() else float("-inf")
        )
    else:
        work["_summary_mtime"] = float("-inf")
    if "path" in work.columns:
        work["_path_len"] = work["path"].astype(str).str.len()
    else:
        work["_path_len"] = 0

    work = work.sort_values(
        key_cols + ["_has_current_stage", "_summary_mtime", "_is_default", "_has_tag", "_proxy_run_tag", "_path_len", "path"],
        ascending=[True] * len(key_cols) + [False, False, False, True, True, True, True],
    )
    before = len(work)
    work = work.drop_duplicates(subset=key_cols, keep="first")
    dropped = before - len(work)
    if dropped:
        print(
            "Applied per-date method deduplication: kept "
            f"{len(work)}/{before} rows (dropped {dropped} duplicate row(s))."
        )
    return work.drop(
        columns=["_stem", "_algo", "_is_default", "_proxy_run_tag", "_has_tag", "_has_current_stage", "_summary_mtime", "_path_len"],
        errors="ignore",
    )


def _format_coverage_printable(coverage_df: pd.DataFrame, coverage_cols: list[str]) -> pd.DataFrame:
    printable = coverage_df.loc[:, coverage_cols].copy()
    formatted: dict[str, pd.Series] = {"method_label": printable["method_label"].astype(str)}
    for col in coverage_cols[1:]:
        series = pd.to_numeric(printable[col], errors="coerce")
        formatted[col] = series.map(
            lambda v: f"{100.0 * float(v):.1f}%" if pd.notna(v) else "NA"
        )
    return pd.DataFrame(formatted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize simulation policy UB/LB bounds.")
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
        help="Exclude runs missing one or more dates in the selected date set (default: enabled).",
    )
    parser.add_argument(
        "--allow-partial-date-coverage",
        dest="require_full_date_coverage",
        action="store_false",
        help="Keep runs even when some dates are missing.",
    )
    parser.add_argument("--algo", type=str, action="append", default=None, help="Optional algo prefix filter (repeatable).")
    parser.add_argument("--proxy-model-name", type=str, action="append", default=None, help="Exact proxy_model_name to include (repeatable).")
    parser.add_argument("--proxy-model-contains", type=str, action="append", default=None, help="Substring match on proxy_model_name (repeatable, case-insensitive).")
    parser.add_argument("--include-max-orders", action="store_true", help="Include partial runs tagged with maxN.")
    parser.add_argument("--out-csv", type=Path, default=Path("logs/simulation_bounds_summary.csv"))
    parser.add_argument(
        "--out-latex",
        type=Path,
        default=None,
        help="Optional path to also save LaTeX table. When omitted, LaTeX is printed only.",
    )
    parser.add_argument("--out-plot", type=Path, default=None, help="Optional bounds plot path. Default: logs/figures/sim_<order_set>_bounds_ci.pdf")
    parser.add_argument(
        "--out-uncertainty-plot",
        type=Path,
        default=None,
        help="Optional plot path showing per-date realized replication totals against estimated UB intervals. "
        "Default: logs/figures/sim_<order_set>_bound_vs_realized_ci.pdf",
    )
    args = parser.parse_args()

    if args.date and (args.date_from or args.date_to):
        print("Use either --date or --date-from/--date-to, not both.")
        return 2

    try:
        df = _build_filtered_summary_df(
            root=args.root,
            order_set=args.order_set,
            date=args.date,
            date_from=args.date_from,
            date_to=args.date_to,
            algo=args.algo,
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

    df = _dedupe_bound_rows(df)
    collapse_proxy_name = _should_collapse_proxy_name(df)
    bounds_df = _build_bounds_summary(df, collapse_proxy_name=collapse_proxy_name)
    if bounds_df.empty:
        print("No policy bound fields found in summaries.")
        return 2

    uncertainty_df = _build_bound_vs_realized_ci_summary(df, collapse_proxy_name=collapse_proxy_name)
    if not uncertainty_df.empty and "ub_compare_metric" in uncertainty_df.columns:
        metric_values = set(uncertainty_df["ub_compare_metric"].dropna().astype(str).tolist())
        if metric_values == {"current_stage"}:
            print("Using current-stage totals for realized-cost comparison plots/coverage.")
        elif "current_stage" in metric_values:
            print(
                "Using current-stage totals for realized-cost comparison when available; "
                "falling back to full lookahead UB for older runs."
            )
    coverage_df = _build_bounds_coverage_summary(uncertainty_df)
    if not coverage_df.empty:
        join_cols = [
            col
            for col in ("algo", "proxy_model_name", "proxy_repair_strategy", "method_label")
            if col in bounds_df.columns and col in coverage_df.columns
        ]
        if join_cols:
            bounds_df = bounds_df.merge(coverage_df, on=join_cols, how="left", suffixes=("", "_coverage"))

    cols = ["method_label", "ub_total", "ub_total_ci95", "lb_total", "lb_total_ci95", "delta_total", "delta_total_ci95", "dates_covered"]
    print(bounds_df[cols].to_string(index=False))
    if "current_stage_total" in bounds_df.columns:
        current_stage_valid = bounds_df["current_stage_total"].notna() & bounds_df["current_stage_total_ci95"].notna()
        if current_stage_valid.any():
            print("\n=== Current-Stage Totals (comparable to realized cost) ===")
            current_stage_cols = ["method_label", "current_stage_total", "current_stage_total_ci95"]
            if "lb_current_stage_total" in bounds_df.columns and "lb_current_stage_total_ci95" in bounds_df.columns:
                current_stage_cols.extend(["lb_current_stage_total", "lb_current_stage_total_ci95"])
            current_stage_cols.append("dates_covered")
            print(bounds_df.loc[current_stage_valid, current_stage_cols].to_string(index=False))
    if "future_recourse_total" in bounds_df.columns:
        future_valid = bounds_df["future_recourse_total"].notna() & bounds_df["future_recourse_total_ci95"].notna()
        if future_valid.any():
            print("\n=== Future-Recourse Totals (lookahead tail) ===")
            future_cols = ["method_label", "future_recourse_total", "future_recourse_total_ci95"]
            if "lb_future_recourse_total" in bounds_df.columns and "lb_future_recourse_total_ci95" in bounds_df.columns:
                future_cols.extend(["lb_future_recourse_total", "lb_future_recourse_total_ci95"])
            future_cols.append("dates_covered")
            print(bounds_df.loc[future_valid, future_cols].to_string(index=False))

    if not coverage_df.empty:
        coverage_cols = [
            "method_label",
            "ub_point_cover_rate",
            "ub_ci_cover_rate",
            "ub_band_cover_rate",
            "ub_rep_point_cover_rate",
            "ub_rep_ci_cover_rate",
            "ub_rep_band_cover_rate",
            "lb_point_cover_rate",
            "lb_ci_cover_rate",
            "lb_band_cover_rate",
            "lb_rep_point_cover_rate",
            "lb_rep_ci_cover_rate",
            "lb_rep_band_cover_rate",
            "bracket_point_cover_rate",
            "bracket_ci_cover_rate",
            "bracket_rep_point_cover_rate",
            "bracket_rep_ci_cover_rate",
        ]
        printable = _format_coverage_printable(coverage_df, coverage_cols)
        print("\n=== Coverage Summary (date-mean and replication-level) ===")
        print(printable.to_string(index=False))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    bounds_df.to_csv(args.out_csv, index=False)
    print(f"Wrote bounds summary CSV to {args.out_csv}")

    latex = _build_bounds_latex_table(bounds_df, order_set=args.order_set)
    if latex:
        print("\n=== Bounds (LaTeX) ===")
        print(latex)
        if args.out_latex is not None:
            args.out_latex.parent.mkdir(parents=True, exist_ok=True)
            args.out_latex.write_text(latex + "\n", encoding="utf-8")
            print(f"Wrote bounds LaTeX table to {args.out_latex}")

    plot_out = args.out_plot or (Path("logs/figures") / f"sim_{args.order_set}_bounds_ci.pdf")
    _plot_bounds_comparison(bounds_df, out_path=plot_out)

    if not uncertainty_df.empty:
        uncertainty_out = args.out_uncertainty_plot or (
            Path("logs/figures") / f"sim_{args.order_set}_bound_vs_realized_ci.pdf"
        )
        _plot_bound_vs_realized_ci_comparison(uncertainty_df, out_path=uncertainty_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
