#!/usr/bin/env python3
"""
Shared helpers for simulation summary analysis scripts.

This module keeps reusable scan/filter and bounds/runtime-size utilities
out of the main collector script.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_CURRENCY_RE = re.compile(r"^\$([0-9,]*\.?[0-9]+)$")
_PCT_RE = re.compile(r"^([0-9]*\.?[0-9]+)%$")
_COLORBLIND_PALETTE = ["#d55e00", "#cc79a7", "#0072b2", "#f0e442", "#009e73"]


def _expand_cli_values(values: list[str] | None) -> list[str]:
    out: list[str] = []
    if not values:
        return out
    for raw in values:
        for token in str(raw).split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def _to_number(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return val

    m = _CURRENCY_RE.match(s)
    if m:
        return float(m.group(1).replace(",", ""))
    m = _PCT_RE.match(s)
    if m:
        return float(m.group(1))

    try:
        return float(s)
    except ValueError:
        return val


def _to_float_or_nan(val: Any) -> float:
    out = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    return float(out) if not pd.isna(out) else float("nan")


def _summary_to_results_parquet(path_val: Any) -> Path | None:
    if path_val is None or pd.isna(path_val):
        return None
    p = Path(str(path_val))
    if not p.exists():
        return None
    if p.name.endswith("_summary.json"):
        stem = p.name[: -len("_summary.json")]
        return p.with_name(f"{stem}.parquet")
    return p.with_suffix(".parquet")


def _load_total_realized_cost_rep_totals(results_path: Path) -> pd.Series | None:
    if not results_path.exists():
        return None
    try:
        raw = pd.read_parquet(results_path, columns=["replication", "realized_cost"])
    except Exception:
        return None
    if raw.empty:
        return None

    raw["replication"] = pd.to_numeric(raw["replication"], errors="coerce")
    raw["realized_cost"] = pd.to_numeric(raw["realized_cost"], errors="coerce")
    raw = raw.dropna(subset=["replication", "realized_cost"])
    if raw.empty:
        return None

    rep_totals = raw.groupby("replication")["realized_cost"].sum()
    if rep_totals.empty:
        return None
    return rep_totals.astype(float)


def _compute_total_realized_cost_rep_stats(results_path: Path) -> dict[str, float] | None:
    rep_totals = _load_total_realized_cost_rep_totals(results_path)
    if rep_totals is None or rep_totals.empty:
        return None

    values = rep_totals.to_numpy(dtype=float)
    n_rep = int(values.size)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if n_rep > 1 else 0.0
    ci95_val = float(1.96 * std_val / np.sqrt(n_rep))
    return {
        "mean": mean_val,
        "std": std_val,
        "ci95": ci95_val,
        "count": float(n_rep),
    }


def _parse_run_id(run_id: str) -> dict[str, Any]:
    parts = run_id.split("_")
    algo = parts[0] if parts else run_id
    # Preserve known multi-token algorithm ids before generic underscore splitting.
    if run_id.startswith("dtlp_bidprice"):
        algo = "dtlp_bidprice"
    elif run_id.startswith("primal_dual"):
        algo = "primal_dual"
    elif run_id.startswith("empirical_saa"):
        algo = "empirical_saa"
    out: dict[str, Any] = {
        "algo": algo,
        "run_id": run_id,
    }

    if out["algo"] == "proxy":
        if "rs" in parts:
            try:
                i = parts.index("rs")
                out["proxy_repair_strategy"] = parts[i + 1]
            except Exception:
                pass
        out["proxy_stochastic"] = "stochastic" in parts
        k_match = next((p for p in parts if p.startswith("k") and p[1:].isdigit()), None)
        if k_match:
            out["proxy_top_k"] = int(k_match[1:])

    out["peak_only"] = "peak" in parts
    max_match = next((p for p in parts if p.startswith("max") and p[3:].isdigit()), None)
    if max_match:
        out["max_orders"] = int(max_match[3:])
    return out


@dataclass(frozen=True)
class ScanResult:
    records: list[dict[str, Any]]
    files_scanned: int


def _scan(
    root: Path,
    order_set: str,
    date: str | None,
    algo_prefix: list[str] | None,
    include_max_orders: bool,
) -> ScanResult:
    base = root / order_set
    if date is not None:
        base = base / date
        paths = list((base / "solutions_eval").glob("*_summary.json"))
    else:
        paths = list(base.glob("*/solutions_eval/*_summary.json"))

    records: list[dict[str, Any]] = []
    for p in sorted(paths):
        run_id = p.stem
        if run_id.endswith("_summary"):
            run_id = run_id[: -len("_summary")]

        parsed = _parse_run_id(run_id)
        algo = str(parsed.get("algo", ""))
        if algo_prefix and not any(algo.startswith(pref) for pref in algo_prefix):
            continue
        if not include_max_orders and parsed.get("max_orders") is not None:
            continue

        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

        sim_date = p.parent.parent.name
        row: dict[str, Any] = {
            "simulation_date": sim_date,
            "order_set": order_set,
            "path": str(p),
            **parsed,
        }
        parquet_path = p.with_name(f"{run_id}.parquet")
        if parquet_path.exists():
            row["parquet_path"] = str(parquet_path)
        runtimes_parquet_path = p.with_name(f"{run_id}_runtimes.parquet")
        if runtimes_parquet_path.exists():
            row["runtimes_parquet_path"] = str(runtimes_parquet_path)

        for k, v in payload.items():
            row[k] = _to_number(v)
        if "proxy_model_name" in payload and payload["proxy_model_name"]:
            row["proxy_model_name"] = payload["proxy_model_name"]
        records.append(row)

    return ScanResult(records=records, files_scanned=len(paths))


def _group_columns(df: pd.DataFrame) -> list[str]:
    cols = ["algo"]
    if "proxy_model_name" in df.columns:
        cols.append("proxy_model_name")
    if "proxy_repair_strategy" in df.columns:
        cols.append("proxy_repair_strategy")
    return cols


def _method_label(row: pd.Series, collapse_proxy_name: bool = False) -> str:
    algo = str(row.get("algo", ""))
    pretty = {
        "csaa": "C-SAA",
        "dtlp_bidprice": "DTLP",
        "empirical": "Empirical-SAA",
        "empirical_saa": "Empirical-SAA",
        "greedy": "Greedy",
        "primal_dual": "Primal-Dual",
        "pto": "PTO",
        "proxy": "Proxy",
    }.get(algo, algo.upper())
    if algo != "proxy":
        return pretty
    if collapse_proxy_name:
        return "Proxy"

    model_name = row.get("proxy_model_name")
    if pd.isna(model_name) or str(model_name).strip() in {"", "-"}:
        return "Proxy"
    return f"Proxy ({model_name})"


def _should_collapse_proxy_name(df: pd.DataFrame) -> bool:
    if "algo" not in df.columns:
        return False
    proxy_mask = df["algo"].astype(str).eq("proxy")
    if not proxy_mask.any():
        return False
    if "proxy_model_name" not in df.columns:
        return True
    names = (
        df.loc[proxy_mask, "proxy_model_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    names = names[names != ""]
    return names.nunique() <= 1


def _filter_full_date_coverage(
    df: pd.DataFrame,
    *,
    key_cols: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty or "simulation_date" not in df.columns:
        return df

    date_count = int(df["simulation_date"].nunique())
    if date_count <= 1:
        return df

    if key_cols is None:
        if "run_id" in df.columns:
            key_cols = ["run_id"]
        else:
            key_cols = _group_columns(df)
    key_cols = [c for c in key_cols if c in df.columns]
    if not key_cols:
        return df

    date_coverage = (
        df.groupby(key_cols, dropna=False)["simulation_date"]
        .nunique()
        .reset_index(name="n_dates")
    )
    keep_keys = date_coverage[date_coverage["n_dates"] == date_count][key_cols]
    if keep_keys.empty:
        return df.iloc[0:0].copy()
    return df.merge(keep_keys, on=key_cols, how="inner")


def _build_filtered_summary_df(
    *,
    root: Path,
    order_set: str,
    date: str | None,
    date_from: str | None,
    date_to: str | None,
    algo: list[str] | None,
    include_max_orders: bool,
    require_full_date_coverage: bool,
    proxy_model_name: list[str] | None,
    proxy_model_contains: list[str] | None,
    proxy_strategy_preference: str | None = "inventory_weighted",
) -> pd.DataFrame:
    default_algo_prefixes = [
        "csaa",
        "dtlp_bidprice",
        "greedy",
        "empirical",
        "empirical_saa",
        "primal_dual",
        "pto",
        "proxy",
    ]
    algo_prefixes = _expand_cli_values(algo) or list(default_algo_prefixes)
    proxy_name_values = _expand_cli_values(proxy_model_name)
    proxy_contains_values = _expand_cli_values(proxy_model_contains)
    if (proxy_name_values or proxy_contains_values) and "proxy" not in algo_prefixes:
        algo_prefixes.append("proxy")

    scan = _scan(root, order_set, date, algo_prefixes, include_max_orders)
    if not scan.records:
        print(
            f"No summaries found under {root}/{order_set} "
            f"for algo filters={algo_prefixes}. Scanned {scan.files_scanned} files."
        )
        return pd.DataFrame()

    df = pd.DataFrame(scan.records)
    if date_from or date_to:
        sim_dates = pd.to_datetime(df["simulation_date"], errors="coerce")
        mask = pd.Series(True, index=df.index)
        if date_from:
            start = pd.to_datetime(date_from, errors="coerce")
            if pd.isna(start):
                raise ValueError(f"Invalid --date-from: {date_from}")
            mask &= sim_dates >= start
        if date_to:
            end = pd.to_datetime(date_to, errors="coerce")
            if pd.isna(end):
                raise ValueError(f"Invalid --date-to: {date_to}")
            mask &= sim_dates <= end
        before = len(df)
        df = df[mask].copy()
        print(f"Applied date range filter: kept {len(df)}/{before} rows.")

    proxy_name_filters = set(proxy_name_values)
    proxy_contains_filters = [s.lower() for s in proxy_contains_values if str(s).strip()]
    proxy_model_filter_active = bool(proxy_name_filters or proxy_contains_filters)
    if proxy_name_filters or proxy_contains_filters:
        if "algo" not in df.columns:
            raise ValueError("Proxy model filtering requested, but 'algo' column is missing.")
        proxy_mask = df["algo"].astype(str).eq("proxy")
        proxy_names = (
            df["proxy_model_name"].fillna("").astype(str)
            if "proxy_model_name" in df.columns
            else pd.Series("", index=df.index, dtype=str)
        )
        proxy_keep = pd.Series(False, index=df.index)
        if proxy_name_filters:
            proxy_keep |= proxy_names.isin(proxy_name_filters)
        if proxy_contains_filters:
            contains_mask = pd.Series(False, index=df.index)
            for token in proxy_contains_filters:
                contains_mask |= proxy_names.str.lower().str.contains(re.escape(token), regex=True)
            proxy_keep |= contains_mask
        before = len(df)
        df = df[(~proxy_mask) | proxy_keep].copy()
        print(f"Applied proxy model filter: kept {len(df)}/{before} rows.")

    # When a proxy model is explicitly selected for analysis, default to the
    # best inference strategy (inventory_weighted) unless caller opts out.
    if proxy_model_filter_active and proxy_strategy_preference:
        if "algo" in df.columns:
            proxy_mask = df["algo"].astype(str).eq("proxy")
            if proxy_mask.any():
                if "proxy_repair_strategy" in df.columns:
                    strategy_series = df["proxy_repair_strategy"].fillna("").astype(str).str.strip()
                    preferred_mask = strategy_series.eq(str(proxy_strategy_preference))
                    preferred_proxy_count = int((proxy_mask & preferred_mask).sum())
                    total_proxy_count = int(proxy_mask.sum())
                    if preferred_proxy_count > 0:
                        before = len(df)
                        df = df[(~proxy_mask) | preferred_mask].copy()
                        print(
                            "Applied proxy strategy preference "
                            f"('{proxy_strategy_preference}'): kept {len(df)}/{before} rows "
                            f"({preferred_proxy_count}/{total_proxy_count} proxy rows)."
                        )
                    else:
                        print(
                            "Preferred proxy strategy "
                            f"'{proxy_strategy_preference}' not found in filtered rows; "
                            "keeping all proxy strategies."
                        )
                else:
                    print(
                        "proxy_repair_strategy not present in summaries; "
                        "cannot apply proxy strategy preference."
                    )

    if require_full_date_coverage:
        before = len(df)
        before_runs = int(df["run_id"].nunique()) if "run_id" in df.columns else before
        df = _filter_full_date_coverage(df)
        after_runs = int(df["run_id"].nunique()) if "run_id" in df.columns else len(df)
        print(
            f"Applied full-date coverage filter: kept {len(df)}/{before} rows "
            f"({after_runs}/{before_runs} runs)."
        )
    return df


def _dedupe_latest_method_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest per-date method row when duplicate runs exist."""
    if df.empty or "simulation_date" not in df.columns or "algo" not in df.columns:
        return df

    key_cols = ["simulation_date", "algo"] + [
        col for col in ("proxy_model_name", "proxy_repair_strategy") if col in df.columns
    ]
    if not key_cols:
        return df

    work = df.copy()
    if "path" in work.columns:
        path_series = work["path"].fillna("").astype(str)
        work["_summary_mtime"] = path_series.map(
            lambda raw: Path(raw).stat().st_mtime if raw and Path(raw).exists() else float("-inf")
        )
    else:
        work["_summary_mtime"] = float("-inf")

    work["_row_order"] = np.arange(len(work), dtype=int)
    work = work.sort_values(
        key_cols + ["_summary_mtime", "_row_order"],
        ascending=[True] * len(key_cols) + [False, False],
        na_position="last",
    )

    before = len(work)
    work = work.drop_duplicates(subset=key_cols, keep="first")
    dropped = before - len(work)
    if dropped:
        print(
            "Applied per-date method deduplication: kept "
            f"{len(work)}/{before} rows (dropped {dropped} duplicate row(s))."
        )

    return work.drop(columns=["_summary_mtime", "_row_order"], errors="ignore")


def _latex_escape(text: str) -> str:
    escaped = text.replace("\\", "\\textbackslash{}")
    escaped = escaped.replace("_", "\\_")
    escaped = escaped.replace("%", "\\%")
    escaped = escaped.replace("&", "\\&")
    escaped = escaped.replace("#", "\\#")
    return escaped


def _format_pm(value: Any, ci: Any, digits: int = 3) -> str:
    if pd.isna(value) or pd.isna(ci):
        return "NA"
    return f"{float(value):,.{digits}f} $\\pm$ {float(ci):,.{digits}f}"


def _build_bounds_summary(
    df: pd.DataFrame,
    *,
    collapse_proxy_name: bool = False,
) -> pd.DataFrame:
    required_any = {
        "policy_ub_total",
        "policy_lb_total",
        "policy_ub_minus_lb_total",
        "policy_current_stage_total",
    }
    if df.empty or not any(col in df.columns for col in required_any):
        return pd.DataFrame()

    group_cols = _group_columns(df)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        rec = {col: row.get(col, "-") for col in group_cols}
        rec["simulation_date"] = row.get("simulation_date")
        rec["method_label"] = _method_label(row, collapse_proxy_name=collapse_proxy_name)

        ub_total = _to_float_or_nan(row.get("policy_ub_total"))
        ub_ci = _to_float_or_nan(row.get("policy_ub_total_ci95"))
        lb_total = _to_float_or_nan(row.get("policy_lb_total"))
        lb_ci = _to_float_or_nan(row.get("policy_lb_total_ci95"))
        delta_total = _to_float_or_nan(row.get("policy_ub_minus_lb_total"))
        delta_ci = _to_float_or_nan(row.get("policy_ub_minus_lb_total_ci95"))

        if pd.isna(delta_total) and not pd.isna(ub_total) and not pd.isna(lb_total):
            delta_total = float(ub_total - lb_total)
        if pd.isna(delta_ci) and not pd.isna(ub_ci) and not pd.isna(lb_ci):
            delta_ci = float(np.sqrt(ub_ci * ub_ci + lb_ci * lb_ci))

        current_stage_total = _to_float_or_nan(row.get("policy_current_stage_total"))
        current_stage_ci = _to_float_or_nan(row.get("policy_current_stage_total_ci95"))
        lb_current_stage_total = _to_float_or_nan(row.get("policy_lb_current_stage_total"))
        lb_current_stage_ci = _to_float_or_nan(row.get("policy_lb_current_stage_total_ci95"))
        future_recourse_total = _to_float_or_nan(row.get("policy_future_recourse_total"))
        future_recourse_ci = _to_float_or_nan(row.get("policy_future_recourse_total_ci95"))
        lb_future_recourse_total = _to_float_or_nan(row.get("policy_lb_future_recourse_total"))
        lb_future_recourse_ci = _to_float_or_nan(row.get("policy_lb_future_recourse_total_ci95"))

        rec["ub_total"] = ub_total
        rec["ub_ci95"] = ub_ci
        rec["current_stage_total"] = current_stage_total
        rec["current_stage_ci95"] = current_stage_ci
        rec["lb_current_stage_total"] = lb_current_stage_total
        rec["lb_current_stage_ci95"] = lb_current_stage_ci
        rec["future_recourse_total"] = future_recourse_total
        rec["future_recourse_ci95"] = future_recourse_ci
        rec["lb_future_recourse_total"] = lb_future_recourse_total
        rec["lb_future_recourse_ci95"] = lb_future_recourse_ci
        rec["lb_total"] = lb_total
        rec["lb_ci95"] = lb_ci
        rec["delta_total"] = delta_total
        rec["delta_ci95"] = delta_ci
        rec["ub_orders"] = _to_float_or_nan(row.get("policy_ub_orders"))
        rec["lb_orders"] = _to_float_or_nan(row.get("policy_lb_orders"))
        rec["paired_orders"] = _to_float_or_nan(row.get("policy_bound_paired_orders"))
        rows.append(rec)

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return pd.DataFrame()

    agg_rows: list[dict[str, Any]] = []
    for key, gdf in long_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = {col: key[idx] for idx, col in enumerate(group_cols)}
        rec["method_label"] = str(gdf["method_label"].iloc[0]) if "method_label" in gdf.columns else "method"
        rec["dates_covered"] = int(gdf["simulation_date"].nunique())

        for prefix in (
            "ub",
            "current_stage",
            "lb_current_stage",
            "future_recourse",
            "lb_future_recourse",
            "lb",
            "delta",
        ):
            v_col = f"{prefix}_total"
            c_col = f"{prefix}_ci95"
            valid = gdf[v_col].notna() & gdf[c_col].notna()
            rec[f"{prefix}_dates"] = int(valid.sum())
            if valid.any():
                vals = gdf.loc[valid, v_col].to_numpy(dtype=float)
                cis = gdf.loc[valid, c_col].to_numpy(dtype=float)
                rec[f"{prefix}_total"] = float(np.sum(vals))
                rec[f"{prefix}_total_ci95"] = float(np.sqrt(np.sum(np.square(cis))))
                rec[f"{prefix}_mean_per_date"] = float(np.mean(vals))
            else:
                rec[f"{prefix}_total"] = float("nan")
                rec[f"{prefix}_total_ci95"] = float("nan")
                rec[f"{prefix}_mean_per_date"] = float("nan")

        rec["ub_orders_total"] = float(np.nansum(pd.to_numeric(gdf["ub_orders"], errors="coerce")))
        rec["lb_orders_total"] = float(np.nansum(pd.to_numeric(gdf["lb_orders"], errors="coerce")))
        rec["paired_orders_total"] = float(np.nansum(pd.to_numeric(gdf["paired_orders"], errors="coerce")))
        agg_rows.append(rec)

    out = pd.DataFrame(agg_rows)
    if out.empty:
        return out
    out = out.sort_values(["ub_total", "method_label"], ascending=[True, True], na_position="last").reset_index(drop=True)
    return out


def _build_bounds_latex_table(bounds_df: pd.DataFrame, order_set: str) -> str | None:
    if bounds_df.empty:
        return None
    lines: list[str] = []
    lines.append("\\begin{table}[!ht]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Cumulative policy bounds across simulation dates for order set={_latex_escape(order_set)}. "
        "Totals are summed across dates; CI95 totals use root-sum-square aggregation across dates.}"
    )
    lines.append(f"\\label{{tab:sim_bounds_{_latex_escape(order_set)}}}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Total UB ($\\pm$ CI95)} & \\textbf{Total LB ($\\pm$ CI95)} & \\textbf{Total UB-LB ($\\pm$ CI95)} \\\\")
    lines.append("\\midrule")
    for _, row in bounds_df.iterrows():
        lines.append(
            f"{_latex_escape(str(row.get('method_label', '-')))} & "
            f"{_format_pm(row.get('ub_total'), row.get('ub_total_ci95'))} & "
            f"{_format_pm(row.get('lb_total'), row.get('lb_total_ci95'))} & "
            f"{_format_pm(row.get('delta_total'), row.get('delta_total_ci95'))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _plot_bounds_comparison(bounds_df: pd.DataFrame, out_path: Path) -> None:
    if bounds_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Skipping bounds plot generation: matplotlib is not available.")
        return

    plot_df = bounds_df.copy()
    methods = plot_df["method_label"].astype(str).tolist()
    y = np.arange(len(methods), dtype=float)

    ub = pd.to_numeric(plot_df["ub_total"], errors="coerce").to_numpy(dtype=float)
    ub_ci = pd.to_numeric(plot_df["ub_total_ci95"], errors="coerce").to_numpy(dtype=float)
    lb = pd.to_numeric(plot_df.get("lb_total"), errors="coerce").to_numpy(dtype=float)
    lb_ci = pd.to_numeric(plot_df.get("lb_total_ci95"), errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.45 * len(methods) + 1.5)))

    ub_valid = ~np.isnan(ub) & ~np.isnan(ub_ci)
    ax.errorbar(
        ub[ub_valid],
        y[ub_valid],
        xerr=ub_ci[ub_valid],
        fmt="o",
        capsize=3,
        color="#0072b2",
        ecolor="#0072b2",
        label="Upper bound",
    )

    lb_valid = ~np.isnan(lb) & ~np.isnan(lb_ci)
    if np.any(lb_valid):
        ax.errorbar(
            lb[lb_valid],
            y[lb_valid],
            xerr=lb_ci[lb_valid],
            fmt="s",
            capsize=3,
            color="#d55e00",
            ecolor="#d55e00",
            label="Lower bound",
        )
        for idx in np.where(lb_valid & ub_valid)[0]:
            ax.plot([lb[idx], ub[idx]], [y[idx], y[idx]], color="#009e73", alpha=0.45, linewidth=1.5)

    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative bound value")
    ax.set_title("Cumulative policy bounds with 95% CI")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote bounds comparison plot to {out_path}")


def _build_bound_vs_realized_ci_summary(
    df: pd.DataFrame,
    *,
    collapse_proxy_name: bool = False,
) -> pd.DataFrame:
    required_any = {"policy_ub_total", "policy_lb_total", "policy_ub_total_ci95", "policy_lb_total_ci95"}
    if df.empty or "path" not in df.columns or not any(col in df.columns for col in required_any):
        return pd.DataFrame()

    group_cols = _group_columns(df)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        results_path = None
        parquet_path = row.get("parquet_path")
        if parquet_path is not None and not pd.isna(parquet_path):
            results_path = Path(str(parquet_path))
        if results_path is None:
            results_path = _summary_to_results_parquet(row.get("path"))
        if results_path is None or not results_path.exists():
            continue

        rep_stats = _compute_total_realized_cost_rep_stats(results_path)
        if rep_stats is None:
            continue

        rec = {col: row.get(col, "-") for col in group_cols}
        rec["simulation_date"] = row.get("simulation_date")
        rec["method_label"] = _method_label(row, collapse_proxy_name=collapse_proxy_name)
        current_stage_total = _to_float_or_nan(row.get("policy_current_stage_total"))
        current_stage_ci95 = _to_float_or_nan(row.get("policy_current_stage_total_ci95"))
        if np.isfinite(current_stage_total) and np.isfinite(current_stage_ci95):
            rec["ub_total"] = current_stage_total
            rec["ub_est_ci95"] = current_stage_ci95
            rec["ub_compare_metric"] = "current_stage"
        else:
            rec["ub_total"] = _to_float_or_nan(row.get("policy_ub_total"))
            rec["ub_est_ci95"] = _to_float_or_nan(row.get("policy_ub_total_ci95"))
            rec["ub_compare_metric"] = "full_lookahead"
        lb_current_stage_total = _to_float_or_nan(row.get("policy_lb_current_stage_total"))
        lb_current_stage_ci95 = _to_float_or_nan(row.get("policy_lb_current_stage_total_ci95"))
        if np.isfinite(lb_current_stage_total) and np.isfinite(lb_current_stage_ci95):
            rec["lb_total"] = lb_current_stage_total
            rec["lb_est_ci95"] = lb_current_stage_ci95
            rec["lb_compare_metric"] = "current_stage"
        else:
            rec["lb_total"] = _to_float_or_nan(row.get("policy_lb_total"))
            rec["lb_est_ci95"] = _to_float_or_nan(row.get("policy_lb_total_ci95"))
            rec["lb_compare_metric"] = "full_lookahead"
        rec["realized_total_rep_mean"] = float(rep_stats["mean"])
        rec["realized_total_rep_ci95"] = float(rep_stats["ci95"])
        rec["replications"] = float(rep_stats["count"])
        rec["results_path"] = str(results_path)
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    sort_cols = ["method_label"]
    if "simulation_date" in out.columns:
        sort_cols.append("simulation_date")
    out = out.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return out


def _build_bounds_coverage_summary(ci_df: pd.DataFrame) -> pd.DataFrame:
    if ci_df.empty:
        return pd.DataFrame()

    group_cols = [col for col in _group_columns(ci_df) if col in ci_df.columns]
    if not group_cols:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for key, gdf in ci_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = {col: key[idx] for idx, col in enumerate(group_cols)}
        rec["method_label"] = str(gdf["method_label"].iloc[0]) if "method_label" in gdf.columns else "method"

        def _series(name: str) -> pd.Series:
            if name not in gdf.columns:
                return pd.Series(np.nan, index=gdf.index, dtype=float)
            return pd.to_numeric(gdf[name], errors="coerce")

        def _rate(mask: pd.Series) -> float:
            return float(mask.mean()) if bool(mask.size) else float("nan")

        realized = _series("realized_total_rep_mean")

        ub = _series("ub_total")
        ub_ci = _series("ub_est_ci95")
        ub_point_valid = realized.notna() & ub.notna()
        ub_ci_valid = ub_point_valid & ub_ci.notna()
        rec["ub_point_dates"] = int(ub_point_valid.sum())
        rec["ub_ci_dates"] = int(ub_ci_valid.sum())
        rec["ub_point_cover_rate"] = _rate(realized[ub_point_valid] <= ub[ub_point_valid]) if ub_point_valid.any() else float("nan")
        rec["ub_ci_cover_rate"] = _rate(realized[ub_ci_valid] <= (ub[ub_ci_valid] + ub_ci[ub_ci_valid])) if ub_ci_valid.any() else float("nan")
        rec["ub_band_cover_rate"] = _rate(
            (realized[ub_ci_valid] >= (ub[ub_ci_valid] - ub_ci[ub_ci_valid]))
            & (realized[ub_ci_valid] <= (ub[ub_ci_valid] + ub_ci[ub_ci_valid]))
        ) if ub_ci_valid.any() else float("nan")
        rec["ub_ci_slack_mean"] = (
            float(((ub[ub_ci_valid] + ub_ci[ub_ci_valid]) - realized[ub_ci_valid]).mean())
            if ub_ci_valid.any()
            else float("nan")
        )

        lb = _series("lb_total")
        lb_ci = _series("lb_est_ci95")
        lb_point_valid = realized.notna() & lb.notna()
        lb_ci_valid = lb_point_valid & lb_ci.notna()
        rec["lb_point_dates"] = int(lb_point_valid.sum())
        rec["lb_ci_dates"] = int(lb_ci_valid.sum())
        rec["lb_point_cover_rate"] = _rate(lb[lb_point_valid] <= realized[lb_point_valid]) if lb_point_valid.any() else float("nan")
        rec["lb_ci_cover_rate"] = _rate((lb[lb_ci_valid] - lb_ci[lb_ci_valid]) <= realized[lb_ci_valid]) if lb_ci_valid.any() else float("nan")
        rec["lb_band_cover_rate"] = _rate(
            (realized[lb_ci_valid] >= (lb[lb_ci_valid] - lb_ci[lb_ci_valid]))
            & (realized[lb_ci_valid] <= (lb[lb_ci_valid] + lb_ci[lb_ci_valid]))
        ) if lb_ci_valid.any() else float("nan")
        rec["lb_ci_slack_mean"] = (
            float((realized[lb_ci_valid] - (lb[lb_ci_valid] - lb_ci[lb_ci_valid])).mean())
            if lb_ci_valid.any()
            else float("nan")
        )

        bracket_point_valid = realized.notna() & ub.notna() & lb.notna()
        bracket_ci_valid = bracket_point_valid & ub_ci.notna() & lb_ci.notna()
        rec["bracket_point_dates"] = int(bracket_point_valid.sum())
        rec["bracket_ci_dates"] = int(bracket_ci_valid.sum())
        rec["bracket_point_cover_rate"] = (
            _rate((lb[bracket_point_valid] <= realized[bracket_point_valid]) & (realized[bracket_point_valid] <= ub[bracket_point_valid]))
            if bracket_point_valid.any()
            else float("nan")
        )
        rec["bracket_ci_cover_rate"] = (
            _rate(((lb[bracket_ci_valid] - lb_ci[bracket_ci_valid]) <= realized[bracket_ci_valid]) & (realized[bracket_ci_valid] <= (ub[bracket_ci_valid] + ub_ci[bracket_ci_valid])))
            if bracket_ci_valid.any()
            else float("nan")
        )

        ub_rep_below: list[np.ndarray] = []
        ub_rep_upper: list[np.ndarray] = []
        ub_rep_band: list[np.ndarray] = []
        lb_rep_above: list[np.ndarray] = []
        lb_rep_lower: list[np.ndarray] = []
        lb_rep_band: list[np.ndarray] = []
        bracket_rep_point: list[np.ndarray] = []
        bracket_rep_ci: list[np.ndarray] = []
        ub_rep_points = 0
        ub_rep_ci_points = 0
        lb_rep_points = 0
        lb_rep_ci_points = 0
        bracket_rep_points = 0
        bracket_rep_ci_points = 0

        for _, date_row in gdf.iterrows():
            results_path_val = date_row.get("results_path")
            if results_path_val is None or pd.isna(results_path_val):
                continue
            rep_totals = _load_total_realized_cost_rep_totals(Path(str(results_path_val)))
            if rep_totals is None or rep_totals.empty:
                continue
            rep_vals = rep_totals.to_numpy(dtype=float)

            ub_val = _to_float_or_nan(date_row.get("ub_total"))
            ub_ci_val = _to_float_or_nan(date_row.get("ub_est_ci95"))
            if np.isfinite(ub_val):
                ub_rep_points += int(rep_vals.size)
                ub_rep_below.append(rep_vals <= ub_val)
                if np.isfinite(ub_ci_val):
                    ub_rep_ci_points += int(rep_vals.size)
                    ub_rep_upper.append(rep_vals <= (ub_val + ub_ci_val))
                    ub_rep_band.append((rep_vals >= (ub_val - ub_ci_val)) & (rep_vals <= (ub_val + ub_ci_val)))

            lb_val = _to_float_or_nan(date_row.get("lb_total"))
            lb_ci_val = _to_float_or_nan(date_row.get("lb_est_ci95"))
            if np.isfinite(lb_val):
                lb_rep_points += int(rep_vals.size)
                lb_rep_above.append(lb_val <= rep_vals)
                if np.isfinite(lb_ci_val):
                    lb_rep_ci_points += int(rep_vals.size)
                    lb_rep_lower.append((lb_val - lb_ci_val) <= rep_vals)
                    lb_rep_band.append((rep_vals >= (lb_val - lb_ci_val)) & (rep_vals <= (lb_val + lb_ci_val)))

            if np.isfinite(ub_val) and np.isfinite(lb_val):
                bracket_rep_points += int(rep_vals.size)
                bracket_rep_point.append((lb_val <= rep_vals) & (rep_vals <= ub_val))
                if np.isfinite(ub_ci_val) and np.isfinite(lb_ci_val):
                    bracket_rep_ci_points += int(rep_vals.size)
                    bracket_rep_ci.append(((lb_val - lb_ci_val) <= rep_vals) & (rep_vals <= (ub_val + ub_ci_val)))

        rec["ub_rep_points"] = int(ub_rep_points)
        rec["ub_rep_ci_points"] = int(ub_rep_ci_points)
        rec["ub_rep_point_cover_rate"] = _rate(pd.Series(np.concatenate(ub_rep_below))) if ub_rep_below else float("nan")
        rec["ub_rep_ci_cover_rate"] = _rate(pd.Series(np.concatenate(ub_rep_upper))) if ub_rep_upper else float("nan")
        rec["ub_rep_band_cover_rate"] = _rate(pd.Series(np.concatenate(ub_rep_band))) if ub_rep_band else float("nan")

        rec["lb_rep_points"] = int(lb_rep_points)
        rec["lb_rep_ci_points"] = int(lb_rep_ci_points)
        rec["lb_rep_point_cover_rate"] = _rate(pd.Series(np.concatenate(lb_rep_above))) if lb_rep_above else float("nan")
        rec["lb_rep_ci_cover_rate"] = _rate(pd.Series(np.concatenate(lb_rep_lower))) if lb_rep_lower else float("nan")
        rec["lb_rep_band_cover_rate"] = _rate(pd.Series(np.concatenate(lb_rep_band))) if lb_rep_band else float("nan")

        rec["bracket_rep_points"] = int(bracket_rep_points)
        rec["bracket_rep_ci_points"] = int(bracket_rep_ci_points)
        rec["bracket_rep_point_cover_rate"] = _rate(pd.Series(np.concatenate(bracket_rep_point))) if bracket_rep_point else float("nan")
        rec["bracket_rep_ci_cover_rate"] = _rate(pd.Series(np.concatenate(bracket_rep_ci))) if bracket_rep_ci else float("nan")
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["method_label"], na_position="last").reset_index(drop=True)
    return out


def _plot_bound_vs_realized_ci_comparison(ci_df: pd.DataFrame, out_path: Path) -> None:
    if ci_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        print("Skipping bound-vs-realized CI plot generation: matplotlib is not available.")
        return

    plot_df = ci_df.dropna(subset=["ub_total", "ub_est_ci95"]).copy()
    if plot_df.empty:
        return

    plot_df["simulation_date"] = plot_df["simulation_date"].astype(str)
    methods = sorted(plot_df["method_label"].dropna().astype(str).unique().tolist())
    if not methods:
        return

    color_map = {
        method: _COLORBLIND_PALETTE[i % len(_COLORBLIND_PALETTE)]
        for i, method in enumerate(methods)
    }

    fig, axes = plt.subplots(len(methods), 1, figsize=(9.0, 4.2 * len(methods)), squeeze=False, sharex=False)
    axes_flat = axes.ravel()
    all_y: list[float] = []
    panel_payloads: list[tuple[Any, pd.DataFrame, list[tuple[float, np.ndarray, float, float, str]]]] = []

    for idx, method in enumerate(methods):
        ax = axes_flat[idx]
        sub = plot_df[plot_df["method_label"].astype(str) == method].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values("simulation_date").reset_index(drop=True)
        date_payloads: list[tuple[float, np.ndarray, float, float, str]] = []
        for pos, (_, row) in enumerate(sub.iterrows()):
            results_path_val = row.get("results_path")
            if results_path_val is None or pd.isna(results_path_val):
                continue
            rep_totals = _load_total_realized_cost_rep_totals(Path(str(results_path_val)))
            if rep_totals is None or rep_totals.empty:
                continue
            rep_vals = rep_totals.to_numpy(dtype=float)
            ub_val = _to_float_or_nan(row.get("ub_total"))
            ub_ci_val = _to_float_or_nan(row.get("ub_est_ci95"))
            if not np.isfinite(ub_val) or not np.isfinite(ub_ci_val):
                continue
            xpos = float(pos)
            date_payloads.append((xpos, rep_vals, float(ub_val), float(ub_ci_val), str(row.get("simulation_date"))))
            all_y.extend(rep_vals.tolist())
            all_y.extend([float(ub_val - ub_ci_val), float(ub_val), float(ub_val + ub_ci_val)])
        panel_payloads.append((ax, sub, date_payloads))

    if not panel_payloads or not all_y:
        plt.close(fig)
        return

    y_min = float(np.nanmin(np.asarray(all_y, dtype=float)))
    y_max = float(np.nanmax(np.asarray(all_y, dtype=float)))
    y_pad = max(1.0, 0.04 * max(1.0, y_max - y_min))

    for ax, sub, date_payloads in panel_payloads:
        method = str(sub["method_label"].iloc[0])
        method_color = color_map.get(method, _COLORBLIND_PALETTE[0])
        rep_band_hits: list[np.ndarray] = []
        compare_metric = (
            str(sub["ub_compare_metric"].iloc[0])
            if "ub_compare_metric" in sub.columns and not sub["ub_compare_metric"].empty
            else "full_lookahead"
        )

        for xpos, rep_vals, ub_val, ub_ci_val, _sim_date in date_payloads:
            band_half_width = 0.34
            band = Rectangle(
                (xpos - band_half_width, ub_val - ub_ci_val),
                2.0 * band_half_width,
                2.0 * ub_ci_val,
                facecolor=method_color,
                edgecolor="none",
                alpha=0.18,
                zorder=1,
            )
            ax.add_patch(band)
            ax.vlines(
                xpos,
                ub_val - ub_ci_val,
                ub_val + ub_ci_val,
                color=method_color,
                linewidth=1.3,
                alpha=0.75,
                zorder=1.5,
            )
            jitter = np.linspace(-0.18, 0.18, num=rep_vals.size) if rep_vals.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(rep_vals.size, xpos, dtype=float) + jitter,
                rep_vals,
                s=18,
                alpha=0.45,
                color="#4d4d4d",
                edgecolors="none",
                zorder=2,
            )
            ax.scatter(
                [xpos],
                [ub_val],
                s=34,
                color=method_color,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            rep_band_hits.append((rep_vals >= (ub_val - ub_ci_val)) & (rep_vals <= (ub_val + ub_ci_val)))

        ub_series = pd.to_numeric(sub["ub_total"], errors="coerce").to_numpy(dtype=float)
        x_line = np.arange(len(sub), dtype=float)
        valid_line = ~np.isnan(ub_series)
        if np.any(valid_line):
            ax.plot(
                x_line[valid_line],
                ub_series[valid_line],
                color=method_color,
                linewidth=1.2,
                alpha=0.9,
                zorder=2.5,
            )

        coverage_text = "Band coverage: NA"
        if rep_band_hits:
            band_rate = float(np.concatenate(rep_band_hits).mean())
            coverage_text = f"In shaded band: {100.0 * band_rate:.1f}%"
        ax.text(
            0.015,
            0.98,
            coverage_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
        )

        date_labels = sub["simulation_date"].astype(str).tolist()
        ax.set_xticks(np.arange(len(date_labels), dtype=float))
        ax.set_xticklabels(date_labels, rotation=30, ha="right")
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_ylabel("Total cost")
        if compare_metric == "current_stage":
            title_suffix = "estimated current-stage interval"
        else:
            title_suffix = "estimated UB interval"
        ax.set_title(f"{method}: realized replications vs {title_suffix}")
        ax.grid(axis="y", alpha=0.2)

    axes_flat[-1].set_xlabel("Simulation date")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote bounds-vs-realized replication plot to {out_path}")


def _parse_size_cutoffs(raw: str) -> tuple[float, float]:
    vals = [s.strip() for s in str(raw).split(",") if s.strip()]
    if len(vals) != 2:
        raise ValueError(f"Expected two comma-separated cutoffs, got '{raw}'")
    c1 = float(vals[0])
    c2 = float(vals[1])
    if c1 <= 0 or c2 <= 0 or c1 >= c2:
        raise ValueError(f"Invalid cutoffs '{raw}' (must satisfy 0 < c1 < c2)")
    return c1, c2


def _resolve_size_thresholds(
    size_series: pd.Series,
    *,
    size_binning: str,
    fixed_cutoffs: tuple[float, float],
    size_transform: str = "linear",
) -> tuple[float, float, str]:
    size = pd.to_numeric(size_series, errors="coerce").dropna().astype(float)
    if size.empty:
        raise ValueError("No valid size values available to compute thresholds.")
    size = size[size > 0]
    if size.empty:
        raise ValueError("Instance-size values must be positive.")

    method = str(size_binning).strip().lower()
    if method == "quantile":
        method = "percentile"
    transform = str(size_transform).strip().lower()
    if transform not in {"linear", "log1p"}:
        raise ValueError(f"Unsupported size transform: {size_transform}")

    if transform == "log1p":
        work_vals = np.log1p(size.to_numpy(dtype=float))
        to_original = lambda x: float(np.expm1(x))
        transform_desc = "log1p scale"
    else:
        work_vals = size.to_numpy(dtype=float)
        to_original = lambda x: float(x)
        transform_desc = "linear scale"

    if method == "fixed":
        c1, c2 = fixed_cutoffs
        desc = (
            "fixed thresholds by (eligible options x order qty) "
            f"(small <= {c1:g}, medium <= {c2:g}, large > {c2:g})"
        )
        return float(c1), float(c2), desc

    if method == "percentile":
        c1_t = float(np.quantile(work_vals, 0.33))
        c2_t = float(np.quantile(work_vals, 0.66))
        c1 = to_original(c1_t)
        c2 = to_original(c2_t)
        desc = (
            "percentile thresholds by (eligible options x order qty) "
            f"(q33={c1:.1f}, q66={c2:.1f}; computed on {transform_desc})"
        )
        return c1, c2, desc

    if method == "equal_range":
        min_t = float(np.min(work_vals))
        max_t = float(np.max(work_vals))
        if max_t <= min_t:
            eps = max(1e-6, abs(min_t) * 1e-6)
            c1_t = min_t + eps
            c2_t = min_t + 2.0 * eps
        else:
            step = (max_t - min_t) / 3.0
            c1_t = min_t + step
            c2_t = min_t + 2.0 * step
        c1 = to_original(c1_t)
        c2 = to_original(c2_t)
        min_val = to_original(min_t)
        max_val = to_original(max_t)
        desc = (
            "equal-range thresholds by (eligible options x order qty) "
            f"(min={min_val:.1f}, max={max_val:.1f}, c1={c1:.1f}, c2={c2:.1f}; "
            f"computed on {transform_desc})"
        )
        return float(c1), float(c2), desc

    if method == "natural_breaks":
        try:
            from sklearn.cluster import KMeans
        except Exception:
            c1_t = float(np.quantile(work_vals, 0.33))
            c2_t = float(np.quantile(work_vals, 0.66))
            c1 = to_original(c1_t)
            c2 = to_original(c2_t)
            desc = (
                "natural-breaks requested but sklearn unavailable; "
                f"fallback to percentile (q33={c1:.1f}, q66={c2:.1f}; computed on {transform_desc})"
            )
            return c1, c2, desc

        values = work_vals.reshape(-1, 1)
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(values)
            centroids = sorted(kmeans.cluster_centers_.flatten().tolist())
            c1_t = float((centroids[0] + centroids[1]) / 2.0)
            c2_t = float((centroids[1] + centroids[2]) / 2.0)
            c1 = to_original(c1_t)
            c2 = to_original(c2_t)
            if c1 >= c2:
                c1_t = float(np.quantile(work_vals, 0.33))
                c2_t = float(np.quantile(work_vals, 0.66))
                c1 = to_original(c1_t)
                c2 = to_original(c2_t)
                desc = (
                    "natural-breaks produced degenerate thresholds; "
                    f"fallback to percentile (q33={c1:.1f}, q66={c2:.1f}; computed on {transform_desc})"
                )
                return c1, c2, desc
            desc = (
                "natural-breaks (k-means) thresholds by (eligible options x order qty) "
                f"(c1={c1:.1f}, c2={c2:.1f}; computed on {transform_desc})"
            )
            return c1, c2, desc
        except Exception:
            c1_t = float(np.quantile(work_vals, 0.33))
            c2_t = float(np.quantile(work_vals, 0.66))
            c1 = to_original(c1_t)
            c2 = to_original(c2_t)
            desc = (
                "natural-breaks failed; "
                f"fallback to percentile (q33={c1:.1f}, q66={c2:.1f}; computed on {transform_desc})"
            )
            return c1, c2, desc

    raise ValueError(f"Unsupported size_binning method: {size_binning}")


def _format_size_threshold_recommendations(
    size_series: pd.Series,
    *,
    size_transform: str = "linear",
) -> str:
    size = pd.to_numeric(size_series, errors="coerce").dropna().astype(float)
    if size.empty:
        return "No valid instance-size values for threshold recommendations."

    methods = ["percentile", "natural_breaks", "equal_range"]
    lines = ["\n--- Threshold Recommendations ---"]
    lines.append(f"Using transform: {size_transform}")
    for method in methods:
        c1, c2, _desc = _resolve_size_thresholds(
            size,
            size_binning=method,
            fixed_cutoffs=(0.0, 1.0),
            size_transform=size_transform,
        )
        if c1 >= c2:
            eps = max(1e-6, abs(c1) * 1e-6, 1.0)
            c2 = c1 + eps
        small_pct = float((size <= c1).mean() * 100.0)
        medium_pct = float(((size > c1) & (size <= c2)).mean() * 100.0)
        large_pct = float((size > c2).mean() * 100.0)
        lines.append(f"\n{method.replace('_', ' ').title()}:")
        lines.append(f"  Small: <= {c1:.2f}")
        lines.append(f"  Medium: {c1:.2f} < value <= {c2:.2f}")
        lines.append(f"  Large: > {c2:.2f}")
        lines.append(
            f"  Distribution: Small {small_pct:.1f}%, Medium {medium_pct:.1f}%, Large {large_pct:.1f}%"
        )
    return "\n".join(lines)


def _load_runtime_rows(
    df: pd.DataFrame,
    *,
    collapse_proxy_name: bool = False,
) -> pd.DataFrame:
    if "runtimes_parquet_path" not in df.columns:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for _, summary_row in df.iterrows():
        path_val = summary_row.get("runtimes_parquet_path")
        if pd.isna(path_val):
            continue
        runtime_path = Path(str(path_val))
        if not runtime_path.exists():
            continue
        try:
            raw = pd.read_parquet(runtime_path)
        except Exception:
            continue
        if raw.empty or "runtime_seconds" not in raw.columns:
            continue

        raw = raw.copy()
        raw["order_id"] = raw.get("order_id", pd.Series("", index=raw.index)).astype(str)
        raw["summary_path"] = summary_row.get("path")
        raw["simulation_date"] = summary_row.get("simulation_date")
        raw["algo"] = summary_row.get("algo", "")
        raw["run_id"] = summary_row.get("run_id", "")
        if "proxy_model_name" in df.columns:
            raw["proxy_model_name"] = summary_row.get("proxy_model_name", "-")
        if "proxy_repair_strategy" in df.columns:
            raw["proxy_repair_strategy"] = summary_row.get("proxy_repair_strategy", "-")
        raw["method_label"] = _method_label(summary_row, collapse_proxy_name=collapse_proxy_name)
        rows.append(raw)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_runtime_gap_by_size_summary(
    runtime_df: pd.DataFrame,
    *,
    size_binning: str,
    fixed_cutoffs: tuple[float, float],
    size_transform: str = "linear",
    min_orders_per_cell: int = 1,
) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    if runtime_df.empty:
        return None, None
    if "eligible_option_count" not in runtime_df.columns:
        return None, None

    work = runtime_df.copy()
    work["eligible_option_count"] = pd.to_numeric(work["eligible_option_count"], errors="coerce")
    work["order_total_qty"] = pd.to_numeric(work.get("order_total_qty"), errors="coerce")
    work["order_total_qty"] = work["order_total_qty"].fillna(1.0).clip(lower=1.0)
    work["runtime_seconds"] = pd.to_numeric(work["runtime_seconds"], errors="coerce")
    work = work.dropna(subset=["eligible_option_count", "runtime_seconds"])
    work = work[work["eligible_option_count"] > 0]
    if work.empty:
        return None, None

    # Use total eligible options across all units in an order:
    #   effective size = eligible options per order * total order quantity.
    work["instance_size_value"] = work["eligible_option_count"].astype(float) * work["order_total_qty"].astype(float)
    size = work["instance_size_value"].astype(float)
    c1, c2, bin_desc = _resolve_size_thresholds(
        size,
        size_binning=size_binning,
        fixed_cutoffs=fixed_cutoffs,
        size_transform=size_transform,
    )
    if c1 >= c2:
        eps = max(1e-6, abs(c1) * 1e-6, 1.0)
        c2 = c1 + eps
    bins = [-np.inf, c1, c2, np.inf]
    work["instance_size_bin"] = pd.cut(
        size,
        bins=bins,
        labels=["small", "medium", "large"],
        include_lowest=True,
        right=True,
    )

    work["policy_ub_mean"] = pd.to_numeric(work.get("policy_ub_mean"), errors="coerce")
    work["policy_lb_mean"] = pd.to_numeric(work.get("policy_lb_mean"), errors="coerce")
    lb_lookup = (
        work.dropna(subset=["policy_lb_mean"])
        .groupby(["simulation_date", "order_id"], as_index=False)["policy_lb_mean"]
        .mean()
        .rename(columns={"policy_lb_mean": "csaa_lb_mean"})
    )
    if not lb_lookup.empty:
        work = work.merge(lb_lookup, on=["simulation_date", "order_id"], how="left")
    else:
        work["csaa_lb_mean"] = np.nan
    work["gap_to_csaa_lb"] = work["policy_ub_mean"] - work["csaa_lb_mean"]

    def _mean_std_ci(values: np.ndarray) -> tuple[float, float, float]:
        if values.size == 0:
            return float("nan"), float("nan"), float("nan")
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        ci_val = float(1.96 * std_val / np.sqrt(values.size))
        return mean_val, std_val, ci_val

    def _quantiles(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"p50": float("nan"), "p75": float("nan"), "p90": float("nan"), "p95": float("nan")}
        return {
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
        }

    group_cols = ["method_label", "algo"]
    if "proxy_model_name" in work.columns:
        group_cols.append("proxy_model_name")
    if "proxy_repair_strategy" in work.columns:
        group_cols.append("proxy_repair_strategy")
    group_cols.append("instance_size_bin")

    out_rows: list[dict[str, Any]] = []
    for key, gdf in work.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: key[i] for i, col in enumerate(group_cols)}
        n_orders = int(len(gdf))
        if n_orders < max(1, int(min_orders_per_cell)):
            continue

        runtime_vals = gdf["runtime_seconds"].to_numpy(dtype=float)
        rt_mean = float(np.mean(runtime_vals))
        rt_std = float(np.std(runtime_vals, ddof=1)) if n_orders > 1 else 0.0
        rt_ci95 = float(1.96 * rt_std / np.sqrt(n_orders))

        gap_vals = pd.to_numeric(gdf["gap_to_csaa_lb"], errors="coerce").dropna().to_numpy(dtype=float)
        n_gap = int(gap_vals.size)
        if n_gap > 0:
            gap_mean = float(np.mean(gap_vals))
            gap_std = float(np.std(gap_vals, ddof=1)) if n_gap > 1 else 0.0
            gap_ci95 = float(1.96 * gap_std / np.sqrt(n_gap))
        else:
            gap_mean = float("nan")
            gap_std = float("nan")
            gap_ci95 = float("nan")

        # Collect-sim-like per-bin metrics from raw results parquet.
        # Objective: per replication SUM over orders in bin; then aggregate across dates by replication id.
        # Late/cumulative: per replication MEAN over orders in bin (pooled across dates via sum/count).
        cost_rep_totals: pd.Series | None = None
        cost_sum_accum: pd.Series | None = None
        cost_cnt_accum: pd.Series | None = None
        late_sum_accum: pd.Series | None = None
        late_cnt_accum: pd.Series | None = None
        cum_sum_accum: pd.Series | None = None
        cum_cnt_accum: pd.Series | None = None

        if "summary_path" in gdf.columns:
            for summary_path_val, path_df in gdf.groupby("summary_path", dropna=False):
                results_path = _summary_to_results_parquet(summary_path_val)
                if results_path is None or not results_path.exists():
                    continue
                try:
                    raw = pd.read_parquet(
                        results_path,
                        columns=["replication", "order_id", "realized_cost", "late_delivery_pct", "cumulative_lateness"],
                    )
                except Exception:
                    continue
                if raw.empty:
                    continue
                raw["order_id"] = raw.get("order_id", pd.Series("", index=raw.index)).astype(str)
                bin_orders = set(path_df["order_id"].astype(str).tolist())
                if not bin_orders:
                    continue
                raw = raw[raw["order_id"].isin(bin_orders)].copy()
                if raw.empty:
                    continue
                raw["replication"] = pd.to_numeric(raw["replication"], errors="coerce")
                raw["realized_cost"] = pd.to_numeric(raw["realized_cost"], errors="coerce")
                raw["late_delivery_pct"] = pd.to_numeric(raw["late_delivery_pct"], errors="coerce")
                raw["cumulative_lateness"] = pd.to_numeric(raw["cumulative_lateness"], errors="coerce")
                raw = raw.dropna(subset=["replication"])
                if raw.empty:
                    continue
                raw["replication"] = raw["replication"].astype(int)

                cost_agg = raw.groupby("replication")["realized_cost"].agg(["sum", "count"])
                rep_cost = pd.to_numeric(cost_agg["sum"], errors="coerce")
                cost_rep_totals = rep_cost if cost_rep_totals is None else cost_rep_totals.add(rep_cost, fill_value=0.0)
                cost_sum = pd.to_numeric(cost_agg["sum"], errors="coerce")
                cost_cnt = pd.to_numeric(cost_agg["count"], errors="coerce")
                cost_sum_accum = cost_sum if cost_sum_accum is None else cost_sum_accum.add(cost_sum, fill_value=0.0)
                cost_cnt_accum = cost_cnt if cost_cnt_accum is None else cost_cnt_accum.add(cost_cnt, fill_value=0.0)

                late_agg = raw.groupby("replication")["late_delivery_pct"].agg(["sum", "count"])
                cum_agg = raw.groupby("replication")["cumulative_lateness"].agg(["sum", "count"])
                late_sum = pd.to_numeric(late_agg["sum"], errors="coerce")
                late_cnt = pd.to_numeric(late_agg["count"], errors="coerce")
                cum_sum = pd.to_numeric(cum_agg["sum"], errors="coerce")
                cum_cnt = pd.to_numeric(cum_agg["count"], errors="coerce")
                late_sum_accum = late_sum if late_sum_accum is None else late_sum_accum.add(late_sum, fill_value=0.0)
                late_cnt_accum = late_cnt if late_cnt_accum is None else late_cnt_accum.add(late_cnt, fill_value=0.0)
                cum_sum_accum = cum_sum if cum_sum_accum is None else cum_sum_accum.add(cum_sum, fill_value=0.0)
                cum_cnt_accum = cum_cnt if cum_cnt_accum is None else cum_cnt_accum.add(cum_cnt, fill_value=0.0)

        if cost_rep_totals is not None and not cost_rep_totals.empty:
            objective_vals = pd.to_numeric(cost_rep_totals, errors="coerce").dropna().to_numpy(dtype=float)
        else:
            objective_vals = np.array([], dtype=float)

        if cost_sum_accum is not None and cost_cnt_accum is not None:
            obj_cnt = pd.to_numeric(cost_cnt_accum, errors="coerce")
            obj_sum = pd.to_numeric(cost_sum_accum, errors="coerce")
            valid_obj = obj_cnt > 0
            objective_per_order_vals = (obj_sum[valid_obj] / obj_cnt[valid_obj]).dropna().to_numpy(dtype=float)
        else:
            objective_per_order_vals = np.array([], dtype=float)

        if late_sum_accum is not None and late_cnt_accum is not None:
            late_cnt = pd.to_numeric(late_cnt_accum, errors="coerce")
            late_sum = pd.to_numeric(late_sum_accum, errors="coerce")
            valid_late = late_cnt > 0
            late_vals = (late_sum[valid_late] / late_cnt[valid_late]).dropna().to_numpy(dtype=float)
        else:
            late_vals = np.array([], dtype=float)

        if cum_sum_accum is not None and cum_cnt_accum is not None:
            cum_cnt = pd.to_numeric(cum_cnt_accum, errors="coerce")
            cum_sum = pd.to_numeric(cum_sum_accum, errors="coerce")
            valid_cum = cum_cnt > 0
            cum_vals = (cum_sum[valid_cum] / cum_cnt[valid_cum]).dropna().to_numpy(dtype=float)
        else:
            cum_vals = np.array([], dtype=float)

        obj_mean, obj_std, obj_ci95 = _mean_std_ci(objective_vals)
        obj_po_mean, obj_po_std, obj_po_ci95 = _mean_std_ci(objective_per_order_vals)
        late_mean, late_std, late_ci95 = _mean_std_ci(late_vals)
        cum_mean, cum_std, cum_ci95 = _mean_std_ci(cum_vals)
        obj_q = _quantiles(objective_vals)
        obj_po_q = _quantiles(objective_per_order_vals)
        late_q = _quantiles(late_vals)
        cum_q = _quantiles(cum_vals)

        row.update(
            {
                "orders": n_orders,
                "gap_orders": n_gap,
                "instance_size_mean": float(np.mean(gdf["instance_size_value"])),
                "instance_size_p50": float(np.median(gdf["instance_size_value"])),
                "instance_size_min": float(np.min(gdf["instance_size_value"])),
                "instance_size_max": float(np.max(gdf["instance_size_value"])),
                "eligible_options_mean": float(np.mean(gdf["eligible_option_count"])),
                "eligible_options_p50": float(np.median(gdf["eligible_option_count"])),
                "eligible_options_min": float(np.min(gdf["eligible_option_count"])),
                "eligible_options_max": float(np.max(gdf["eligible_option_count"])),
                "order_qty_mean": float(np.mean(gdf["order_total_qty"])),
                "order_qty_p50": float(np.median(gdf["order_total_qty"])),
                "runtime_s_mean": rt_mean,
                "runtime_s_std": rt_std,
                "runtime_s_ci95": rt_ci95,
                "gap_to_csaa_lb_mean": gap_mean,
                "gap_to_csaa_lb_std": gap_std,
                "gap_to_csaa_lb_ci95": gap_ci95,
                "objective_total_mean": obj_mean,
                "objective_total_std": obj_std,
                "objective_total_ci95": obj_ci95,
                "objective_total_replications": int(objective_vals.size),
                "objective_total_p50": obj_q["p50"],
                "objective_total_p75": obj_q["p75"],
                "objective_total_p90": obj_q["p90"],
                "objective_total_p95": obj_q["p95"],
                "objective_per_order_mean": obj_po_mean,
                "objective_per_order_std": obj_po_std,
                "objective_per_order_ci95": obj_po_ci95,
                "objective_per_order_replications": int(objective_per_order_vals.size),
                "objective_per_order_p50": obj_po_q["p50"],
                "objective_per_order_p75": obj_po_q["p75"],
                "objective_per_order_p90": obj_po_q["p90"],
                "objective_per_order_p95": obj_po_q["p95"],
                "late_delivery_pct_mean": late_mean,
                "late_delivery_pct_std": late_std,
                "late_delivery_pct_ci95": late_ci95,
                "late_delivery_pct_replications": int(late_vals.size),
                "late_delivery_pct_p50": late_q["p50"],
                "late_delivery_pct_p75": late_q["p75"],
                "late_delivery_pct_p90": late_q["p90"],
                "late_delivery_pct_p95": late_q["p95"],
                "cumulative_lateness_mean": cum_mean,
                "cumulative_lateness_std": cum_std,
                "cumulative_lateness_ci95": cum_ci95,
                "cumulative_lateness_replications": int(cum_vals.size),
                "cumulative_lateness_p50": cum_q["p50"],
                "cumulative_lateness_p75": cum_q["p75"],
                "cumulative_lateness_p90": cum_q["p90"],
                "cumulative_lateness_p95": cum_q["p95"],
            }
        )
        out_rows.append(row)

    if not out_rows:
        return None, None

    out = pd.DataFrame(out_rows)
    size_order = {"small": 0, "medium": 1, "large": 2}
    out["instance_size_bin"] = out["instance_size_bin"].astype(str)
    out["size_order"] = out["instance_size_bin"].map(size_order).fillna(99)
    out = out.sort_values(["method_label", "size_order"]).drop(columns=["size_order"]).reset_index(drop=True)
    return out, bin_desc


def _build_runtime_gap_size_latex(
    size_df: pd.DataFrame,
    *,
    order_set: str,
    bin_desc: str,
) -> str | None:
    if size_df.empty:
        return None
    size_order = {"small": 0, "medium": 1, "large": 2}
    table_df = size_df.copy()
    table_df["_size_key"] = table_df["instance_size_bin"].astype(str).str.lower()
    table_df["_size_order"] = table_df["_size_key"].map(size_order).fillna(99)
    table_df = table_df.sort_values(["_size_order", "method_label"]).reset_index(drop=True)

    def _format_orders(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        return f"{int(value):,}"

    def _format_size_range(row: pd.Series) -> str:
        min_v = row.get("instance_size_min")
        max_v = row.get("instance_size_max")
        if pd.isna(min_v) or pd.isna(max_v):
            return "NA"
        return f"{float(min_v):,.0f}--{float(max_v):,.0f}"

    lines: list[str] = []
    lines.append("\\begin{table}[!ht]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Runtime, per-order objective, service by instance size for order set={_latex_escape(order_set)}. "
        f"Instance size is based on total eligible options across units (eligible options $\\times$ order quantity; {_latex_escape(bin_desc)}).}}"
    )
    lines.append(f"\\label{{tab:runtime_gap_size_{_latex_escape(order_set)}}}")
    lines.append("\\begin{adjustbox}{width=\\linewidth,center}")
    lines.append("\\begin{tabular}{llrlrrrr}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Size} & \\textbf{Method} & \\textbf{Orders} & \\textbf{Size Range} & "
        "\\textbf{Avg Runtime (s)} & \\textbf{Avg Obj / Order ($\\pm$ CI95)} & "
        "\\textbf{Late \\% ($\\pm$ CI95)} & \\textbf{Cum. Late ($\\pm$ CI95)} \\\\"
    )
    lines.append("\\midrule")
    grouped = list(table_df.groupby("_size_key", sort=False))
    for group_idx, (size_key, group) in enumerate(grouped):
        group = group.reset_index(drop=True)
        row_count = len(group)
        size_label = _latex_escape(size_key.title())
        shared_orders = group["orders"].nunique(dropna=False) == 1
        shared_range = (
            group["instance_size_min"].nunique(dropna=False) == 1
            and group["instance_size_max"].nunique(dropna=False) == 1
        )
        shared_orders_text = _format_orders(group.loc[0, "orders"]) if shared_orders else None
        shared_range_text = _format_size_range(group.loc[0]) if shared_range else None

        for row_idx, (_, row) in enumerate(group.iterrows()):
            runtime_text = "NA" if pd.isna(row.get("runtime_s_mean")) else f"{float(row.get('runtime_s_mean')):,.3f}"
            if row_idx == 0:
                size_cell = (
                    f"\\multirow{{{row_count}}}{{*}}{{{size_label}}}"
                    if row_count > 1
                    else size_label
                )
                if shared_orders:
                    orders_cell = (
                        f"\\multirow{{{row_count}}}{{*}}{{{shared_orders_text}}}"
                        if row_count > 1
                        else str(shared_orders_text)
                    )
                else:
                    orders_cell = _format_orders(row.get("orders"))
                if shared_range:
                    range_cell = (
                        f"\\multirow{{{row_count}}}{{*}}{{{_latex_escape(shared_range_text or 'NA')}}}"
                        if row_count > 1
                        else _latex_escape(shared_range_text or "NA")
                    )
                else:
                    range_cell = _latex_escape(_format_size_range(row))
            else:
                size_cell = ""
                orders_cell = "" if shared_orders else _format_orders(row.get("orders"))
                range_cell = "" if shared_range else _latex_escape(_format_size_range(row))

            lines.append(
                f"{size_cell} & "
                f"{_latex_escape(str(row.get('method_label', '-')))} & "
                f"{orders_cell} & "
                f"{range_cell} & "
                f"{runtime_text} & "
                f"{_format_pm(row.get('objective_per_order_mean'), row.get('objective_per_order_ci95'), digits=2)} & "
                f"{_format_pm(row.get('late_delivery_pct_mean'), row.get('late_delivery_pct_ci95'), digits=3)} & "
                f"{_format_pm(row.get('cumulative_lateness_mean'), row.get('cumulative_lateness_ci95'), digits=3)} \\\\"
            )
        if group_idx < len(grouped) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _plot_runtime_gap_by_size(size_df: pd.DataFrame, out_path: Path) -> None:
    if size_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Skipping runtime/gap-by-size plot generation: matplotlib is not available.")
        return

    bins = ["small", "medium", "large"]
    methods = size_df["method_label"].dropna().astype(str).unique().tolist()
    methods = sorted(methods)
    x = np.arange(len(bins), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    width = min(0.8 / max(1, len(methods)), 0.24)

    for i, method in enumerate(methods):
        sub = size_df[size_df["method_label"] == method].copy()
        sub["instance_size_bin"] = pd.Categorical(sub["instance_size_bin"], categories=bins, ordered=True)
        sub = sub.sort_values("instance_size_bin")

        runtime_means = np.full(len(bins), np.nan, dtype=float)
        runtime_errs = np.zeros(len(bins), dtype=float)
        gap_means = np.full(len(bins), np.nan, dtype=float)
        gap_errs = np.zeros(len(bins), dtype=float)

        for j, b in enumerate(bins):
            row = sub[sub["instance_size_bin"] == b]
            if row.empty:
                continue
            runtime_means[j] = float(row["runtime_s_mean"].iloc[0])
            runtime_errs[j] = float(row["runtime_s_ci95"].iloc[0]) if not pd.isna(row["runtime_s_ci95"].iloc[0]) else 0.0
            if not pd.isna(row["gap_to_csaa_lb_mean"].iloc[0]):
                gap_means[j] = float(row["gap_to_csaa_lb_mean"].iloc[0])
                gap_errs[j] = float(row["gap_to_csaa_lb_ci95"].iloc[0]) if not pd.isna(row["gap_to_csaa_lb_ci95"].iloc[0]) else 0.0

        offset = (i - (len(methods) - 1) / 2.0) * width
        r_valid = ~np.isnan(runtime_means)
        if np.any(r_valid):
            axes[0].bar(
                x[r_valid] + offset,
                runtime_means[r_valid],
                width=width,
                label=method,
                color=_COLORBLIND_PALETTE[i % len(_COLORBLIND_PALETTE)],
                alpha=0.9,
            )
        g_valid = ~np.isnan(gap_means)
        if np.any(g_valid):
            axes[1].bar(
                x[g_valid] + offset,
                gap_means[g_valid],
                width=width,
                yerr=gap_errs[g_valid],
                capsize=3,
                label=method,
                color=_COLORBLIND_PALETTE[i % len(_COLORBLIND_PALETTE)],
                alpha=0.9,
            )

    axes[0].set_ylabel("Runtime (seconds)")
    axes[0].set_title("Runtime by instance size")
    axes[1].set_ylabel("UB - CSAA LB")
    axes[1].set_title("Gap by instance size")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([b.title() for b in bins])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote runtime/gap-by-size plot to {out_path}")
