#!/usr/bin/env python3
"""
Build an intraday trajectory plot (bucketed by k minutes) for selected methods.

This is intentionally separate from collect_sim_summaries.py to keep the
summary collector focused on aggregate reporting.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.sim_summary_common import (  # noqa: E402
    _expand_cli_values,
    _method_label,
    _scan,
    _should_collapse_proxy_name,
)

_COLORBLIND_PALETTE = ["#d55e00", "#cc79a7", "#0072b2", "#f0e442", "#009e73"]


def _read_parquet_columns(
    path: Path,
    *,
    required: list[str],
    optional: list[str] | None = None,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    opt = optional or []
    requested = required + [c for c in opt if c not in required]
    try:
        return pd.read_parquet(path, columns=requested)
    except Exception:
        try:
            raw = pd.read_parquet(path)
        except Exception:
            return None
        missing_required = [c for c in required if c not in raw.columns]
        if missing_required:
            return None
        keep = [c for c in requested if c in raw.columns]
        return raw[keep].copy()


def _load_order_time_map(orders_csv: Path, simulation_date: str) -> pd.DataFrame:
    if not orders_csv.exists():
        print(f"Intraday trajectory: orders CSV not found: {orders_csv}")
        return pd.DataFrame(columns=["order_id", "order_time"])

    target_date = pd.to_datetime(simulation_date, errors="coerce")
    if pd.isna(target_date):
        print(f"Intraday trajectory: invalid date '{simulation_date}'")
        return pd.DataFrame(columns=["order_id", "order_time"])
    target_date = target_date.normalize()

    chunks: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(
            orders_csv,
            usecols=["order_ID", "order_time"],
            parse_dates=["order_time"],
            chunksize=500_000,
        )
    except Exception as exc:
        print(f"Intraday trajectory: failed to read {orders_csv}: {exc}")
        return pd.DataFrame(columns=["order_id", "order_time"])

    for chunk in reader:
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["order_ID", "order_time"]).copy()
        if chunk.empty:
            continue
        mask = chunk["order_time"].dt.normalize() == target_date
        if not mask.any():
            continue
        part = chunk.loc[mask, ["order_ID", "order_time"]].copy()
        part["order_id"] = part["order_ID"].astype(str)
        chunks.append(part[["order_id", "order_time"]])

    if not chunks:
        return pd.DataFrame(columns=["order_id", "order_time"])

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values("order_time").drop_duplicates("order_id", keep="first").reset_index(drop=True)
    return out


def _load_intraday_rows(
    df: pd.DataFrame,
    *,
    order_time_map: pd.DataFrame,
    collapse_proxy_name: bool = False,
) -> pd.DataFrame:
    if order_time_map.empty:
        return pd.DataFrame()

    date_key = pd.to_datetime(order_time_map["order_time"].iloc[0], errors="coerce")
    if pd.isna(date_key):
        return pd.DataFrame()
    simulation_date = date_key.strftime("%Y-%m-%d")

    date_df = df[df["simulation_date"].astype(str) == simulation_date].copy()
    if date_df.empty:
        return pd.DataFrame()

    lb_parts: list[pd.DataFrame] = []
    for _, summary_row in date_df.iterrows():
        rtp = summary_row.get("runtimes_parquet_path")
        if pd.isna(rtp):
            continue
        runtime_path = Path(str(rtp))
        runtime_raw = _read_parquet_columns(
            runtime_path,
            required=["order_id", "policy_lb_mean"],
        )
        if runtime_raw is None or runtime_raw.empty or "policy_lb_mean" not in runtime_raw.columns:
            continue
        runtime_raw = runtime_raw.dropna(subset=["order_id", "policy_lb_mean"]).copy()
        if runtime_raw.empty:
            continue
        runtime_raw["order_id"] = runtime_raw["order_id"].astype(str)
        lb_parts.append(runtime_raw[["order_id", "policy_lb_mean"]])

    if lb_parts:
        lb_lookup = (
            pd.concat(lb_parts, ignore_index=True)
            .groupby("order_id", as_index=False)["policy_lb_mean"]
            .mean()
            .rename(columns={"policy_lb_mean": "csaa_lb_mean"})
        )
    else:
        lb_lookup = pd.DataFrame(columns=["order_id", "csaa_lb_mean"])

    rows: list[pd.DataFrame] = []
    for _, summary_row in date_df.iterrows():
        rp = summary_row.get("parquet_path")
        rtp = summary_row.get("runtimes_parquet_path")
        if pd.isna(rp) or pd.isna(rtp):
            continue

        result_path = Path(str(rp))
        runtime_path = Path(str(rtp))

        result_raw = _read_parquet_columns(
            result_path,
            required=["order_id", "realized_cost"],
            optional=["replication"],
        )
        runtime_raw = _read_parquet_columns(
            runtime_path,
            required=["order_id", "runtime_seconds"],
            optional=["policy_ub_mean"],
        )
        if result_raw is None or runtime_raw is None or result_raw.empty or runtime_raw.empty:
            continue
        if "order_id" not in result_raw.columns or "order_id" not in runtime_raw.columns:
            continue

        result_raw = result_raw.dropna(subset=["order_id", "realized_cost"]).copy()
        runtime_raw = runtime_raw.dropna(subset=["order_id", "runtime_seconds"]).copy()
        if result_raw.empty or runtime_raw.empty:
            continue

        result_raw["order_id"] = result_raw["order_id"].astype(str)
        runtime_raw["order_id"] = runtime_raw["order_id"].astype(str)
        if "policy_ub_mean" not in runtime_raw.columns:
            runtime_raw["policy_ub_mean"] = np.nan

        per_order_cost = result_raw.groupby("order_id", as_index=False)["realized_cost"].mean()
        agg_cols = {"runtime_seconds": "mean"}
        if "policy_ub_mean" in runtime_raw.columns:
            agg_cols["policy_ub_mean"] = "mean"
        per_order_runtime = runtime_raw.groupby("order_id", as_index=False).agg(agg_cols)

        merged = per_order_cost.merge(per_order_runtime, on="order_id", how="inner")
        if merged.empty:
            continue
        merged = merged.merge(order_time_map, on="order_id", how="inner")
        if merged.empty:
            continue
        if not lb_lookup.empty:
            merged = merged.merge(lb_lookup, on="order_id", how="left")
            if "policy_ub_mean" in merged.columns:
                merged["ub_minus_csaa_lb"] = merged["policy_ub_mean"] - merged["csaa_lb_mean"]
        else:
            merged["csaa_lb_mean"] = np.nan
            merged["ub_minus_csaa_lb"] = np.nan

        merged["method_label"] = _method_label(summary_row, collapse_proxy_name=collapse_proxy_name)
        merged["algo"] = summary_row.get("algo", "")
        merged["run_id"] = summary_row.get("run_id", "")
        if "proxy_model_name" in df.columns:
            merged["proxy_model_name"] = summary_row.get("proxy_model_name", "-")
        rows.append(merged)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["order_time"] = pd.to_datetime(out["order_time"], errors="coerce")
    out = out.dropna(subset=["order_time"])
    return out


def _build_demand_profile(
    order_time_map: pd.DataFrame,
    *,
    bin_minutes: int,
) -> pd.DataFrame:
    if order_time_map.empty:
        return pd.DataFrame(columns=["bucket_time", "demand_orders"])

    k = max(1, int(bin_minutes))
    work = order_time_map.copy()
    work["bucket_time"] = pd.to_datetime(work["order_time"], errors="coerce").dt.floor(f"{k}min")
    work = work.dropna(subset=["bucket_time", "order_id"])
    if work.empty:
        return pd.DataFrame(columns=["bucket_time", "demand_orders"])

    demand = (
        work.groupby("bucket_time", dropna=False)["order_id"]
        .nunique()
        .reset_index(name="demand_orders")
        .sort_values("bucket_time")
        .reset_index(drop=True)
    )
    return demand


def _build_intraday_bucket_summary(
    intraday_rows: pd.DataFrame,
    *,
    bin_minutes: int,
) -> pd.DataFrame:
    if intraday_rows.empty:
        return pd.DataFrame()
    k = max(1, int(bin_minutes))
    work = intraday_rows.copy()
    work["bucket_time"] = work["order_time"].dt.floor(f"{k}min")

    agg = (
        work.groupby(["method_label", "bucket_time"], dropna=False)
        .agg(
            orders=("order_id", "nunique"),
            realized_cost_mean=("realized_cost", "mean"),
            runtime_s_mean=("runtime_seconds", "mean"),
            ub_minus_csaa_lb_mean=("ub_minus_csaa_lb", "mean"),
        )
        .reset_index()
    )
    agg = agg.sort_values(["method_label", "bucket_time"]).reset_index(drop=True)
    return agg


def _plot_intraday_trajectory(
    intraday_summary: pd.DataFrame,
    demand_profile: pd.DataFrame,
    *,
    simulation_date: str,
    bin_minutes: int,
    out_path: Path,
) -> None:
    if intraday_summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except Exception:
        print("Skipping intraday trajectory plot generation: matplotlib is not available.")
        return

    methods = (
        intraday_summary.groupby("method_label", dropna=False)["realized_cost_mean"]
        .max()
        .sort_values()
        .index.tolist()
    )
    if not methods:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
    demand_ax = axes[0].twinx()
    if not demand_profile.empty:
        dsub = demand_profile.sort_values("bucket_time")
        dx = pd.to_datetime(dsub["bucket_time"])
        dx_num = mdates.date2num(dx)
        bar_width_days = max(float(bin_minutes) / (24.0 * 60.0) * 0.85, 1e-6)
        demand_ax.bar(
            dx_num,
            dsub["demand_orders"].to_numpy(dtype=float),
            width=bar_width_days,
            color="#9e9e9e",
            alpha=0.28,
            align="center",
            label="Demand volume (orders)",
            zorder=1,
        )
        demand_ax.set_ylabel("Orders in bucket", color="#616161")
        demand_ax.tick_params(axis="y", labelcolor="#616161")
    else:
        demand_ax.set_yticks([])

    for i, method in enumerate(methods):
        sub = intraday_summary[intraday_summary["method_label"] == method].sort_values("bucket_time")
        if sub.empty:
            continue
        x = pd.to_datetime(sub["bucket_time"])
        color = _COLORBLIND_PALETTE[i % len(_COLORBLIND_PALETTE)]
        axes[0].plot(
            x,
            sub["realized_cost_mean"],
            marker="o",
            linewidth=1.9,
            markersize=3.0,
            label=method,
            color=color,
            zorder=3,
        )
        axes[1].plot(x, sub["runtime_s_mean"], marker="o", linewidth=1.6, markersize=2.8, label=method, color=color)

    axes[0].set_ylabel("Avg realized cost")
    axes[0].set_title(f"Intraday trend ({simulation_date}, {bin_minutes}-minute bins)")
    axes[0].grid(axis="y", alpha=0.25)
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    demand_handles, demand_labels = demand_ax.get_legend_handles_labels()
    axes[0].legend(line_handles + demand_handles, line_labels + demand_labels, loc="best", fontsize=8)

    axes[1].set_ylabel("Mean runtime (s)")
    axes[1].set_xlabel("Time bucket")
    axes[1].set_title("Per-bucket policy runtime")
    axes[1].grid(axis="y", alpha=0.25)

    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote intraday trajectory plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot intraday simulation trajectories by method.")
    parser.add_argument("--root", type=Path, default=Path("data/peak/simulation_results"))
    parser.add_argument("--order-set", type=str, default="test", choices=["test", "proxy_train"])
    parser.add_argument("--date", type=str, required=True, help="Single date (YYYY-MM-DD).")
    parser.add_argument(
        "--algo",
        type=str,
        action="append",
        default=None,
        help="Algo prefix filter (repeatable or comma-separated). Default: csaa, greedy, empirical, empirical_saa, pto, proxy.",
    )
    parser.add_argument(
        "--proxy-model-name",
        type=str,
        action="append",
        default=None,
        help="Exact proxy_model_name to include (repeatable or comma-separated).",
    )
    parser.add_argument(
        "--proxy-model-contains",
        type=str,
        action="append",
        default=None,
        help="Substring match on proxy_model_name (repeatable or comma-separated, case-insensitive).",
    )
    parser.add_argument(
        "--include-max-orders",
        action="store_true",
        help="Include partial runs tagged with maxN.",
    )
    parser.add_argument(
        "--bin-minutes",
        type=int,
        default=30,
        help="Bucket size in minutes (default: 30).",
    )
    parser.add_argument(
        "--orders-csv",
        type=Path,
        default=Path("data/processed/preprocessed_data_cs.csv"),
        help="Orders CSV used to map order_id to order_time.",
    )
    parser.add_argument(
        "--drop-first-order",
        dest="drop_first_order",
        action="store_true",
        default=True,
        help="Drop the first order of the day to reduce warm-start artifacts (default: enabled).",
    )
    parser.add_argument(
        "--keep-first-order",
        dest="drop_first_order",
        action="store_false",
        help="Keep the first order of the day.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for intraday bucket summary.",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="Optional output plot path. Default: logs/figures/sim_<order_set>_<date>_intraday_k<min>.png",
    )
    args = parser.parse_args()

    algo_prefixes = _expand_cli_values(args.algo) or ["csaa", "greedy", "empirical", "empirical_saa", "pto", "proxy"]
    if (args.proxy_model_name or args.proxy_model_contains) and "proxy" not in algo_prefixes:
        algo_prefixes.append("proxy")

    scan = _scan(args.root, args.order_set, args.date, algo_prefixes, args.include_max_orders)
    if not scan.records:
        print(f"No summaries found for date={args.date} under {args.root}/{args.order_set}.")
        return 2
    df = pd.DataFrame(scan.records)
    if "algo" in df.columns:
        print(f"Loaded summaries by algo: {df['algo'].value_counts(dropna=False).to_dict()}")

    proxy_name_filters = set(_expand_cli_values(args.proxy_model_name))
    proxy_contains_filters = [s.lower() for s in _expand_cli_values(args.proxy_model_contains) if str(s).strip()]
    if proxy_name_filters or proxy_contains_filters:
        if "algo" not in df.columns:
            print("Proxy model filtering requested, but 'algo' column is missing.")
            return 2
        proxy_mask = df["algo"].astype(str).eq("proxy")
        proxy_names = df["proxy_model_name"].fillna("").astype(str) if "proxy_model_name" in df.columns else pd.Series("", index=df.index, dtype=str)
        proxy_keep = pd.Series(False, index=df.index)
        if proxy_name_filters:
            proxy_keep |= proxy_names.isin(proxy_name_filters)
        if proxy_contains_filters:
            contains_mask = pd.Series(False, index=df.index)
            for token in proxy_contains_filters:
                contains_mask |= proxy_names.str.lower().str.contains(re.escape(token), regex=True)
            proxy_keep |= contains_mask
        df = df[(~proxy_mask) | proxy_keep].copy()
        if df.empty:
            print("No rows left after proxy model filtering.")
            return 2

    collapse_proxy_name = _should_collapse_proxy_name(df)
    order_time_map = _load_order_time_map(args.orders_csv, args.date)
    if order_time_map.empty:
        print("No order_time rows found for selected date.")
        return 2
    if args.drop_first_order:
        order_time_map = order_time_map.sort_values("order_time").reset_index(drop=True)
        dropped_id: str | None = None
        if not order_time_map.empty:
            dropped_id = str(order_time_map.loc[0, "order_id"])
            order_time_map = order_time_map.iloc[1:].reset_index(drop=True)
            print(f"Dropped first order of day for warm-start effect: {dropped_id}")
    else:
        dropped_id = None
    if order_time_map.empty:
        print("No order_time rows left after first-order drop.")
        return 2

    intraday_rows = _load_intraday_rows(
        df,
        order_time_map=order_time_map,
        collapse_proxy_name=collapse_proxy_name,
    )
    if intraday_rows.empty:
        print("No intraday rows generated. Check date/filters and parquet availability.")
        return 2
    if dropped_id is not None and "order_id" in intraday_rows.columns:
        intraday_rows = intraday_rows[intraday_rows["order_id"].astype(str) != dropped_id].copy()

    intraday_summary = _build_intraday_bucket_summary(intraday_rows, bin_minutes=args.bin_minutes)
    if intraday_summary.empty:
        print("No intraday bucket summary generated.")
        return 2
    methods = sorted(intraday_summary["method_label"].dropna().astype(str).unique().tolist())
    print(f"Methods in intraday summary: {methods}")
    demand_profile = _build_demand_profile(order_time_map, bin_minutes=args.bin_minutes)
    intraday_summary = intraday_summary.merge(demand_profile, on="bucket_time", how="left")

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = Path("logs/figures") / f"sim_{args.order_set}_{args.date}_intraday_k{int(args.bin_minutes)}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    intraday_summary.to_csv(out_csv, index=False)
    print(f"Wrote intraday trajectory CSV to {out_csv}")

    out_plot = args.out_plot
    if out_plot is None:
        out_plot = Path("logs/figures") / f"sim_{args.order_set}_{args.date}_intraday_k{int(args.bin_minutes)}.png"
    _plot_intraday_trajectory(
        intraday_summary,
        demand_profile,
        simulation_date=args.date,
        bin_minutes=args.bin_minutes,
        out_path=out_plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
