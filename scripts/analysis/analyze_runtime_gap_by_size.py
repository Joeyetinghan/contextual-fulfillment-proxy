#!/usr/bin/env python3
"""
Dedicated runtime/gap-by-instance-size summarizer for simulation outputs.
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
    _build_filtered_summary_df,
    _build_runtime_gap_by_size_summary,
    _build_runtime_gap_size_latex,
    _dedupe_latest_method_rows,
    _format_size_threshold_recommendations,
    _load_runtime_rows,
    _parse_size_cutoffs,
    _plot_runtime_gap_by_size,
    _should_collapse_proxy_name,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize runtime/gap vs instance size from simulation outputs.")
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
    parser.add_argument(
        "--size-binning",
        type=str,
        choices=["percentile", "quantile", "natural_breaks", "equal_range", "fixed"],
        default="natural_breaks",
        help=(
            "How to bin by instance size = eligible options x order quantity. "
            "Options: percentile, natural_breaks (k-means), equal_range, fixed. "
            "Default: natural_breaks."
        ),
    )
    parser.add_argument(
        "--size-cutoffs",
        type=str,
        default="128,384",
        help="Fixed cutoffs on instance size (= eligible options x order quantity), as 'small_cutoff,medium_cutoff'.",
    )
    parser.add_argument(
        "--size-transform",
        type=str,
        choices=["linear", "log1p"],
        default="log1p",
        help=(
            "Transform applied before deriving non-fixed thresholds. "
            "Default: log1p. "
            "'log1p' helps separate small/medium under heavy tails."
        ),
    )
    parser.add_argument(
        "--min-orders",
        type=int,
        default=1,
        help="Minimum number of orders required per (method,size-bin) cell.",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("logs/simulation_runtime_gap_by_size.csv"))
    parser.add_argument(
        "--out-latex",
        type=Path,
        default=None,
        help="Optional path to also save LaTeX table. When omitted, LaTeX is printed only.",
    )
    parser.add_argument(
        "--no-threshold-recommendations",
        action="store_true",
        help="Disable printing percentile/natural-breaks/equal-range threshold recommendations.",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="Optional output path for runtime/gap-by-size plot. Default: logs/figures/sim_<order_set>_runtime_gap_by_size.pdf",
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

    df = _dedupe_latest_method_rows(df)

    try:
        size_cutoffs = _parse_size_cutoffs(args.size_cutoffs)
    except ValueError as exc:
        print(f"Invalid --size-cutoffs: {exc}")
        return 2

    collapse_proxy_name = _should_collapse_proxy_name(df)
    runtime_rows = _load_runtime_rows(df, collapse_proxy_name=collapse_proxy_name)
    if not args.no_threshold_recommendations and not runtime_rows.empty:
        size_vals = (
            pd.to_numeric(runtime_rows.get("eligible_option_count"), errors="coerce")
            * pd.to_numeric(runtime_rows.get("order_total_qty"), errors="coerce").fillna(1.0).clip(lower=1.0)
        )
        print(_format_size_threshold_recommendations(size_vals, size_transform=args.size_transform))
    size_summary, bin_desc = _build_runtime_gap_by_size_summary(
        runtime_rows,
        size_binning=args.size_binning,
        fixed_cutoffs=size_cutoffs,
        size_transform=args.size_transform,
        min_orders_per_cell=args.min_orders,
    )
    if size_summary is None or size_summary.empty:
        print(
            "No runtime instance-size analysis generated. "
            "Need runtimes parquet files with eligible_option_count."
        )
        return 2

    preview_cols = [
        "method_label",
        "algo",
        "instance_size_bin",
        "instance_size_min",
        "instance_size_max",
        "eligible_options_mean",
        "order_qty_mean",
        "orders",
        "objective_per_order_mean",
        "objective_per_order_ci95",
        "objective_total_mean",
        "objective_total_ci95",
        "late_delivery_pct_mean",
        "late_delivery_pct_ci95",
        "cumulative_lateness_mean",
        "cumulative_lateness_ci95",
        "runtime_s_mean",
    ]
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
        print(size_summary[preview_cols].to_string(index=False))
    print(f"Instance size binning: {bin_desc}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    size_summary.to_csv(args.out_csv, index=False)
    print(f"Wrote runtime/gap-by-size CSV to {args.out_csv}")

    size_latex = _build_runtime_gap_size_latex(
        size_summary,
        order_set=args.order_set,
        bin_desc=bin_desc or args.size_binning,
    )
    if size_latex:
        print("\n=== Runtime/Gap by Size (LaTeX) ===")
        print(size_latex)
        if args.out_latex is not None:
            args.out_latex.parent.mkdir(parents=True, exist_ok=True)
            args.out_latex.write_text(size_latex + "\n", encoding="utf-8")
            print(f"Wrote runtime/gap-by-size LaTeX to {args.out_latex}")

    out_plot = args.out_plot or (Path("logs/figures") / f"sim_{args.order_set}_runtime_gap_by_size.pdf")
    _plot_runtime_gap_by_size(size_summary, out_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
