#!/usr/bin/env python3
"""
Summarize proxy ablation results into printable and LaTeX tables.

Supported ablation types:
  - architecture (test-simulation metrics from summary JSON/parquet)
  - loss (test-simulation metrics from summary JSON/parquet)
  - inference (test-simulation metrics from summary JSON/parquet)

For architecture/loss, use --source validation to force validation-log
summaries instead of simulation summaries.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.proxy.analyze_tuning_results import (
    collect_all_results,
    parse_log_file,
    resolve_selection_metric,
    select_best_per_group,
)
from scripts.analysis.sim_summary_common import _build_filtered_summary_df, _latex_escape
from scripts.proxy.summarize_proxy_ablation_validation import (
    _build_summary_df,
    _ensure_group_col,
    _load_full_model_row,
    _sort_by_metric,
)


DEFAULT_SELECTION_METRIC = "min:val_proxy_ub_mean"
_ARCH_ABLATION_TAGS = (
    "full_model",
    "full",
    "no_scenario_module",
    "no_dc_module",
    "single_tower",
)
_LOSS_ABLATION_TAGS = (
    "full_model",
    "full_loss",
    "no_constraint_loss",
    "no_cost_loss",
    "no_selection_loss",
    "no_carrier_loss",
    "no_label_smoothing",
    "no_dc_class_weights",
)

_TAG_SUFFIX_MARKERS = (
    "_cw",
    "_cost",
    "_tau",
    "_ew",
    "_sku",
    "_br",
    "_cemb",
    "_op",
    "_dcemb",
    "_th",
    "_job",
    "_cfg",
)


def _allowed_ablation_tags(run_tag_prefix: str | None) -> tuple[str, ...] | None:
    prefix = str(run_tag_prefix or "").strip()
    if prefix.startswith("ablation_arch"):
        return _ARCH_ABLATION_TAGS
    if prefix.startswith("ablation_loss"):
        return _LOSS_ABLATION_TAGS
    return None


def _ablation_tag_from_model_name(model_name: str, run_tag_prefix: str | None) -> str | None:
    text = str(model_name or "").lower().strip()
    if not text:
        return None
    m = re.search(r"_abl([a-z0-9_]+)", text)
    if not m:
        return None
    tag = m.group(1)
    prefix = str(run_tag_prefix or "").strip()
    if prefix.startswith("ablation_loss"):
        for pfx in ("fixed_loss_", "loss_"):
            if tag.startswith(pfx):
                tag = tag[len(pfx) :]
                break
    else:
        if tag.startswith("fixed_"):
            tag = tag[len("fixed_") :]
    for marker in _TAG_SUFFIX_MARKERS:
        idx = tag.find(marker)
        if idx > 0:
            tag = tag[:idx]
            break
    tag = re.sub(r"[^a-z0-9_]+", "_", tag).strip("_")
    return tag or None


def _split_tokens(raw_values: list[str] | None) -> list[str]:
    out: list[str] = []
    if not raw_values:
        return out
    for raw in raw_values:
        for token in str(raw).split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def _fmt_value(value: Any, digits: int) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):,.{digits}f}"


def _fmt_pm(mean: Any, ci: Any, digits: int) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(ci):
        return _fmt_value(mean, digits=digits)
    return f"{float(mean):,.{digits}f} +/- {float(ci):,.{digits}f}"


def _fmt_pm_latex(mean: Any, ci: Any, digits: int) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(ci):
        return _fmt_value(mean, digits=digits)
    return f"{float(mean):,.{digits}f} $\\pm$ {float(ci):,.{digits}f}"


def _mean_std_ci(values: pd.Series) -> tuple[float, float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    ci95_val = float(1.96 * std_val / math.sqrt(max(n, 1)))
    return mean_val, std_val, ci95_val


def _render_printable_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        if df.empty:
            f.write("No rows.\n")
            return
        f.write(df.to_string(index=False))
        f.write("\n")


def _build_arch_or_loss_summary(
    *,
    summary_csv: Path | None,
    log_dir: Path,
    run_names: list[str],
    ablation_type: str,
    group_col: str,
    selection_metric_spec: str,
    full_model_name: str | None,
    models_root: Path,
) -> pd.DataFrame:
    if summary_csv is not None:
        df = pd.read_csv(summary_csv)
        if df.empty:
            return df
        return df

    if not run_names:
        inferred = _infer_latest_run_name(
            log_dir=log_dir,
            require_ablation_rows=True,
            ablation_type=ablation_type,
        )
        if inferred is None:
            raise ValueError(
                "Could not infer a latest parseable ablation run from logs. "
                "Provide --run-name (or --summary-csv)."
            )
        run_names = [inferred]
        print(f"[summarize_proxy_ablations] --type={ablation_type}: inferred latest run-name='{inferred}'")

    df = collect_all_results(str(log_dir), run_names)
    if df.empty:
        return df
    df = _ensure_group_col(df, group_col)

    metric, maximize, use_abs, _label = resolve_selection_metric(df, selection_metric_spec)
    best_df = select_best_per_group(df, group_col, metric, maximize, use_abs=use_abs)
    best_df = _sort_by_metric(best_df, metric, maximize=maximize, use_abs=use_abs)
    full_row = _load_full_model_row(full_model_name, models_root, group_col)
    out = _build_summary_df(
        best_df=best_df,
        full_row=full_row,
        group_col=group_col,
        selection_metric=metric,
        maximize=maximize,
        use_abs=use_abs,
    )
    return out


def _row_matches_ablation_type(row: dict[str, Any], ablation_type: str) -> bool:
    name = str(row.get("model_name", "")).lower()
    if "_abl" not in name:
        return False
    if ablation_type == "loss":
        return ("_ablfixed_loss_" in name) or ("_ablloss_" in name)
    if ablation_type == "arch":
        if "_ablfixed_loss_" in name or "_ablloss_" in name:
            return False
        return "_ablfixed_" in name or "_abl" in name
    return "_abl" in name


def _infer_latest_run_name(
    log_dir: Path,
    require_ablation_rows: bool = True,
    ablation_type: str | None = None,
) -> str | None:
    if not log_dir.exists():
        return None
    files = sorted(
        log_dir.glob("*.out"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        try:
            rows = parse_log_file(str(path))
        except Exception:
            continue
        if not rows:
            continue
        if require_ablation_rows:
            if ablation_type in {"arch", "loss"}:
                has_ablation = any(_row_matches_ablation_type(r, ablation_type) for r in rows)
            else:
                has_ablation = any("_abl" in str(r.get("model_name", "")) for r in rows)
            if not has_ablation:
                continue
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return stem
    return None


def _infer_full_model_name_from_wrapper_logs(log_dir: Path, ablation_type: str) -> str | None:
    if not log_dir.exists():
        return None
    pattern = (
        "run_proxy_arch_ablation_fixed_*.out"
        if ablation_type == "arch"
        else "run_proxy_loss_ablation_fixed_*.out"
    )
    wrapper_logs = sorted(
        log_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in wrapper_logs:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in re.finditer(r"full_model=([^\s]+)", text):
            value = str(m.group(1)).strip()
            if value and value != "<unknown>":
                return value
    return None


def _canonicalize_arch_or_loss_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "rank_ablation" in out.columns:
        out = out.sort_values(["rank_ablation", "ablation_tag"], na_position="first")

    def _col_or_nan(col: str) -> pd.Series:
        if col in out.columns:
            return out[col]
        return pd.Series(np.nan, index=out.index)

    ub_mean = _col_or_nan("val_proxy_ub_mean")
    ub_ci95 = _col_or_nan("val_proxy_ub_ci95_mean")
    hit1_rep = _col_or_nan("val_joint_hit1_repaired")
    hit5_raw = _col_or_nan("val_joint_hit5_raw")
    proxy_le = _col_or_nan("val_proxy_le_csaa_frac")
    delta_ub = _col_or_nan("delta_vs_full_val_proxy_ub_mean")

    display = pd.DataFrame(
        {
            "Ablation": _col_or_nan("ablation_tag"),
            "Model": _col_or_nan("model_name"),
            "UB Mean +/- CI95": [
                _fmt_pm(m, c, digits=3)
                for m, c in zip(ub_mean, ub_ci95)
            ],
            "Hit@1 Repaired": [
                _fmt_value(v, digits=4) for v in hit1_rep
            ],
            "Hit@5 Raw": [
                _fmt_value(v, digits=4) for v in hit5_raw
            ],
            "Proxy<=CSAA": [
                _fmt_value(v, digits=4) for v in proxy_le
            ],
            "Delta UB vs Full": [
                _fmt_value(v, digits=3) for v in delta_ub
            ],
        }
    )
    return display


def _to_latex_arch_or_loss(df: pd.DataFrame, title: str) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[!ht]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{{_latex_escape(title)} ablation summary (validation metrics). "
        "Lower UB is better; higher Hit rates and Proxy<=CSAA are better.}}"
    )
    lines.append(f"\\label{{tab:{title.lower()}_ablation_summary}}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Ablation} & \\textbf{UB Mean ($\\pm$CI95)} & \\textbf{Hit@1 Rep} & "
        "\\textbf{Hit@5 Raw} & \\textbf{Proxy$\\leq$CSAA} & \\textbf{Delta UB vs Full} & \\textbf{Model} \\\\"
    )
    lines.append("\\midrule")
    for _, row in df.iterrows():
        ub_text = str(row.get("UB Mean +/- CI95", "NA"))
        lines.append(
            f"{_latex_escape(str(row.get('Ablation', '-')))} & "
            f"{_latex_escape(ub_text)} & "
            f"{_latex_escape(str(row.get('Hit@1 Repaired', 'NA')))} & "
            f"{_latex_escape(str(row.get('Hit@5 Raw', 'NA')))} & "
            f"{_latex_escape(str(row.get('Proxy<=CSAA', 'NA')))} & "
            f"{_latex_escape(str(row.get('Delta UB vs Full', 'NA')))} & "
            f"{_latex_escape(str(row.get('Model', '-')))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _infer_ablation_label(row: pd.Series, run_tag_prefix: str | None) -> str:
    def _normalize(label: str) -> str:
        out = str(label or "").strip()
        # Backward-compat: older runs used `fixed_...` tags.
        while out.startswith("fixed_"):
            out = out[len("fixed_") :]
        return out

    def canonicalize(label: str) -> str:
        label = _normalize(label)
        known = _allowed_ablation_tags(run_tag_prefix)
        if not known:
            return label
        for tag in sorted(known, key=len, reverse=True):
            if label == tag or label.startswith(f"{tag}_"):
                return tag
        return label

    run_tag = str(row.get("proxy_run_tag", "") or "").strip()
    if run_tag:
        if run_tag_prefix:
            prefix = f"{run_tag_prefix}_"
            if run_tag.startswith(prefix):
                return canonicalize(run_tag[len(prefix) :])
        return canonicalize(run_tag)
    model_tag = _ablation_tag_from_model_name(
        str(row.get("proxy_model_name", "") or row.get("run_id", "") or ""),
        run_tag_prefix,
    )
    if model_tag:
        return canonicalize(model_tag)
    strategy = str(row.get("proxy_repair_strategy", "") or "").strip()
    if strategy:
        return strategy
    return "default"


def _select_inference_baseline_mask(df: pd.DataFrame, run_tag_prefix: str | None) -> pd.Series:
    if df.empty or "proxy_repair_strategy" not in df.columns or "run_id" not in df.columns:
        return pd.Series(False, index=df.index)

    strategy = df["proxy_repair_strategy"].fillna("").astype(str).str.strip().str.lower()
    inv_mask = strategy.eq("inventory_weighted")
    if not inv_mask.any():
        return pd.Series(False, index=df.index)

    run_tags = df.get("proxy_run_tag", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    prefix = str(run_tag_prefix or "").strip()
    prefixed_mask = run_tags.str.startswith(f"{prefix}_") if prefix else pd.Series(False, index=df.index)

    candidates = df.loc[inv_mask & ~prefixed_mask].copy()
    if candidates.empty:
        candidates = df.loc[inv_mask].copy()
    if candidates.empty:
        return pd.Series(False, index=df.index)

    candidates["proxy_run_tag"] = run_tags.loc[candidates.index]
    if "path" in candidates.columns:
        candidates["summary_mtime"] = candidates["path"].apply(
            lambda p: Path(str(p)).stat().st_mtime if Path(str(p)).exists() else float("-inf")
        )
    else:
        candidates["summary_mtime"] = float("-inf")

    by_run = (
        candidates.groupby("run_id", dropna=False)
        .agg(
            has_untagged=("proxy_run_tag", lambda s: bool((s.astype(str).str.strip() == "").any())),
            max_summary_mtime=("summary_mtime", "max"),
            n_rows=("run_id", "size"),
        )
        .reset_index()
        .sort_values(
            ["has_untagged", "max_summary_mtime", "n_rows", "run_id"],
            ascending=[False, False, False, False],
            na_position="last",
        )
    )
    if by_run.empty:
        return pd.Series(False, index=df.index)
    chosen_run_id = by_run.iloc[0]["run_id"]
    return df["run_id"].astype(str) == str(chosen_run_id)


def _aggregate_inference_group(group_df: pd.DataFrame) -> dict[str, Any]:
    # Objective aligned with collect_sim_summaries: sum realized_cost over orders per replication, then across dates.
    # Replication-level stats aligned with collect_sim_summaries: aggregate raw
    # per-replication values across dates -- objective = summed realized cost;
    # late-delivery and cumulative-lateness = order-weighted means -- then take
    # mean/CI95 over replications (NOT 1.96*std/sqrt(6) over the daily means,
    # which would conflate between-date variability with sim uncertainty).
    rep_totals: pd.Series | None = None
    late_sum: pd.Series | None = None
    late_cnt: pd.Series | None = None
    cum_sum: pd.Series | None = None
    cum_cnt: pd.Series | None = None
    valid_dates = 0
    want_cols = ["replication", "realized_cost", "late_delivery_pct", "cumulative_lateness"]
    for parquet_path_val in group_df.get("parquet_path", pd.Series(dtype=str)).dropna().unique():
        parquet_path = Path(str(parquet_path_val))
        if not parquet_path.exists():
            continue
        try:
            raw = pd.read_parquet(parquet_path, columns=want_cols)
        except Exception:
            try:
                raw = pd.read_parquet(parquet_path, columns=["replication", "realized_cost"])
            except Exception:
                continue
        if raw.empty or "replication" not in raw.columns:
            continue
        raw["replication"] = pd.to_numeric(raw["replication"], errors="coerce")
        raw = raw.dropna(subset=["replication"])
        if raw.empty:
            continue
        raw["replication"] = raw["replication"].astype(int)
        rep = raw["replication"]
        if "realized_cost" in raw.columns:
            cost = pd.to_numeric(raw["realized_cost"], errors="coerce")
            valid = cost.notna()
            if valid.any():
                rep_day = cost[valid].groupby(rep[valid]).sum()
                rep_totals = rep_day if rep_totals is None else rep_totals.add(rep_day, fill_value=0.0)
        if "late_delivery_pct" in raw.columns:
            lv = pd.to_numeric(raw["late_delivery_pct"], errors="coerce")
            valid = lv.notna()
            if valid.any():
                ls = lv[valid].groupby(rep[valid]).sum()
                lc = lv[valid].groupby(rep[valid]).count()
                late_sum = ls if late_sum is None else late_sum.add(ls, fill_value=0.0)
                late_cnt = lc if late_cnt is None else late_cnt.add(lc, fill_value=0.0)
        if "cumulative_lateness" in raw.columns:
            cv = pd.to_numeric(raw["cumulative_lateness"], errors="coerce")
            valid = cv.notna()
            if valid.any():
                cs = cv[valid].groupby(rep[valid]).sum()
                cc = cv[valid].groupby(rep[valid]).count()
                cum_sum = cs if cum_sum is None else cum_sum.add(cs, fill_value=0.0)
                cum_cnt = cc if cum_cnt is None else cum_cnt.add(cc, fill_value=0.0)
        valid_dates += 1

    if rep_totals is not None and not rep_totals.empty:
        objective_vals = rep_totals.to_numpy(dtype=float)
        objective_n = int(objective_vals.size)
        objective_mean = float(np.mean(objective_vals))
        objective_std = float(np.std(objective_vals, ddof=1)) if objective_n > 1 else 0.0
        objective_ci95 = float(1.96 * objective_std / math.sqrt(max(objective_n, 1)))
        objective_source = "raw_rep_total"
    else:
        total_cost = pd.to_numeric(group_df.get("avg_realized_cost"), errors="coerce") * pd.to_numeric(
            group_df.get("orders_evaluated"), errors="coerce"
        )
        objective_mean, objective_std, objective_ci95 = _mean_std_ci(total_cost)
        objective_n = int(pd.to_numeric(total_cost, errors="coerce").notna().sum())
        objective_source = "summary_avg_times_orders"

    def _rep_rate_stats(sum_s: pd.Series | None, cnt_s: pd.Series | None):
        # Order-weighted per-replication rate (sum/count pooled across dates),
        # then mean/CI95 over replications. None when raw columns are unavailable
        # (callers fall back to the per-date summary aggregation).
        if sum_s is None or cnt_s is None or sum_s.empty:
            return None
        rate = (sum_s / cnt_s.replace(0, np.nan)).to_numpy(dtype=float)
        rate = rate[~np.isnan(rate)]
        n = int(rate.size)
        if n == 0:
            return None
        mean = float(np.mean(rate))
        std = float(np.std(rate, ddof=1)) if n > 1 else 0.0
        return mean, std, float(1.96 * std / math.sqrt(max(n, 1)))

    runtime_mean, runtime_std, runtime_ci95 = _mean_std_ci(group_df.get("avg_policy_runtime_ms", pd.Series(dtype=float)))
    late_stats = _rep_rate_stats(late_sum, late_cnt)
    if late_stats is not None:
        late_mean, late_std, late_ci95 = late_stats
    else:
        late_mean, late_std, late_ci95 = _mean_std_ci(group_df.get("avg_late_delivery_pct", pd.Series(dtype=float)))
    cum_stats = _rep_rate_stats(cum_sum, cum_cnt)
    if cum_stats is not None:
        cumlate_mean, cumlate_std, cumlate_ci95 = cum_stats
    else:
        cumlate_mean, cumlate_std, cumlate_ci95 = _mean_std_ci(group_df.get("avg_cumulative_lateness", pd.Series(dtype=float)))

    return {
        "objective_total_mean": objective_mean,
        "objective_total_std": objective_std,
        "objective_total_ci95": objective_ci95,
        "objective_reps": objective_n,
        "objective_source": objective_source,
        "runtime_ms_mean": runtime_mean,
        "runtime_ms_std": runtime_std,
        "runtime_ms_ci95": runtime_ci95,
        "late_delivery_pct_mean": late_mean,
        "late_delivery_pct_std": late_std,
        "late_delivery_pct_ci95": late_ci95,
        "cumulative_lateness_mean": cumlate_mean,
        "cumulative_lateness_std": cumlate_std,
        "cumulative_lateness_ci95": cumlate_ci95,
        "dates": int(group_df["simulation_date"].nunique()) if "simulation_date" in group_df.columns else int(len(group_df)),
    }


def _build_inference_summary(
    *,
    sim_root: Path,
    order_set: str,
    date_from: str | None,
    date_to: str | None,
    proxy_model_name: str | None,
    run_tag_prefix: str | None,
    include_default: bool,
    require_full_date_coverage: bool,
    inject_inventory_baseline: bool,
) -> pd.DataFrame:
    df = _build_filtered_summary_df(
        root=sim_root,
        order_set=order_set,
        date=None,
        date_from=date_from,
        date_to=date_to,
        algo=["proxy"],
        include_max_orders=False,
        require_full_date_coverage=require_full_date_coverage,
        proxy_model_name=[proxy_model_name] if proxy_model_name else None,
        proxy_model_contains=None,
        # Inference-ablation summary must keep all strategies for comparison.
        proxy_strategy_preference=None,
    )
    if df.empty:
        return df

    df = df.copy()
    df["proxy_run_tag"] = df.get("proxy_run_tag", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    df["proxy_repair_strategy"] = (
        df.get("proxy_repair_strategy", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    )
    df["ablation_tag"] = df.apply(lambda r: _infer_ablation_label(r, run_tag_prefix), axis=1)
    allowed_tags = _allowed_ablation_tags(run_tag_prefix)
    if allowed_tags is not None:
        df = df.loc[df["ablation_tag"].isin(allowed_tags)].copy()
        if df.empty:
            return df
    if inject_inventory_baseline:
        baseline_mask = _select_inference_baseline_mask(df, run_tag_prefix)
        if baseline_mask.any():
            df.loc[baseline_mask, "ablation_tag"] = "inventory_weighted"
    else:
        baseline_mask = pd.Series(False, index=df.index)

    tagged = (
        df["proxy_run_tag"].str.startswith(f"{run_tag_prefix}_")
        if run_tag_prefix
        else pd.Series(True, index=df.index)
    )
    # For arch/loss, keep all rows after ablation-tag canonicalization/filtering.
    if not inject_inventory_baseline:
        keep_mask = pd.Series(True, index=df.index)
    # Inference keeps tagged runs + one inventory_weighted baseline.
    elif run_tag_prefix:
        keep_mask = tagged | baseline_mask | (include_default & (df["proxy_run_tag"] == ""))
    else:
        keep_mask = pd.Series(True, index=df.index)
    df = df.loc[keep_mask].copy()
    if df.empty:
        return df

    rows: list[dict[str, Any]] = []
    for ablation_tag, gdf in df.groupby("ablation_tag", dropna=False):
        agg = _aggregate_inference_group(gdf)
        row = {
            "Ablation": ablation_tag,
            "Proxy Model": str(gdf["proxy_model_name"].dropna().astype(str).iloc[0])
            if "proxy_model_name" in gdf.columns and gdf["proxy_model_name"].notna().any()
            else "-",
            "Proxy Strategy": str(gdf["proxy_repair_strategy"].dropna().astype(str).iloc[0])
            if "proxy_repair_strategy" in gdf.columns and gdf["proxy_repair_strategy"].notna().any()
            else "-",
            "Proxy Run Tag": str(gdf["proxy_run_tag"].dropna().astype(str).iloc[0]) if gdf["proxy_run_tag"].notna().any() else "-",
            "Objective Total Mean": agg["objective_total_mean"],
            "Objective Total CI95": agg["objective_total_ci95"],
            "Objective Source": agg["objective_source"],
            "Runtime ms Mean": agg["runtime_ms_mean"],
            "Runtime ms CI95": agg["runtime_ms_ci95"],
            "Late Delivery % Mean": agg["late_delivery_pct_mean"],
            "Late Delivery % CI95": agg["late_delivery_pct_ci95"],
            "Cumulative Lateness Mean": agg["cumulative_lateness_mean"],
            "Cumulative Lateness CI95": agg["cumulative_lateness_ci95"],
            "Dates": agg["dates"],
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    full_mask = out["Ablation"].astype(str).eq("full_model")
    if full_mask.any():
        full_obj = pd.to_numeric(out.loc[full_mask, "Objective Total Mean"], errors="coerce").dropna()
        if not full_obj.empty:
            out["Delta Objective vs Full"] = pd.to_numeric(out["Objective Total Mean"], errors="coerce") - float(
                full_obj.iloc[0]
            )
        full_rt = pd.to_numeric(out.loc[full_mask, "Runtime ms Mean"], errors="coerce").dropna()
        if not full_rt.empty:
            out["Delta Runtime ms vs Full"] = pd.to_numeric(out["Runtime ms Mean"], errors="coerce") - float(
                full_rt.iloc[0]
            )
    out = out.sort_values(
        by=["Ablation", "Objective Total Mean"],
        key=lambda s: s.astype(str).ne("full_model") if s.name == "Ablation" else s,
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
    return out


def _canonicalize_inference_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Objective Total"] = [
        _fmt_pm(m, c, digits=2)
        for m, c in zip(out["Objective Total Mean"], out["Objective Total CI95"])
    ]
    out["Late Delivery %"] = [
        _fmt_pm(m, c, digits=3)
        for m, c in zip(out["Late Delivery % Mean"], out["Late Delivery % CI95"])
    ]
    out["Cumulative Lateness"] = [
        _fmt_pm(m, c, digits=3)
        for m, c in zip(out["Cumulative Lateness Mean"], out["Cumulative Lateness CI95"])
    ]
    return out[
        [
            "Ablation",
            "Objective Total",
            "Late Delivery %",
            "Cumulative Lateness",
        ]
    ]


def _to_latex_inference(df: pd.DataFrame, *, ablation_title: str = "Inference strategy") -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[!ht]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{{_latex_escape(ablation_title)} ablation: objective and service metrics on test simulation. All entries are mean $\\pm$ CI95.}}"
    )
    label_key = re.sub(r"[^a-z0-9]+", "_", ablation_title.lower()).strip("_")
    lines.append(f"\\label{{tab:proxy_{label_key}_ablation_summary}}")
    lines.append("\\begin{adjustbox}{width=\\linewidth,center}")
    col_spec = "l" + ("r" * int(len(df)))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header_cells = ["\\textbf{Metric}"] + [
        f"\\textbf{{{_latex_escape(str(row.get('Ablation', '-')))}}}" for _, row in df.iterrows()
    ]
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")
    value_cells = ["\\textbf{Objective Total}"] + [
        _fmt_pm_latex(row.get("Objective Total Mean"), row.get("Objective Total CI95"), digits=2)
        for _, row in df.iterrows()
    ]
    lines.append(" & ".join(value_cells) + " \\\\")
    late_cells = ["\\textbf{Late Delivery \\%}"] + [
        _fmt_pm_latex(row.get("Late Delivery % Mean"), row.get("Late Delivery % CI95"), digits=3)
        for _, row in df.iterrows()
    ]
    lines.append(" & ".join(late_cells) + " \\\\")
    cumlate_cells = ["\\textbf{Cumulative Lateness}"] + [
        _fmt_pm_latex(row.get("Cumulative Lateness Mean"), row.get("Cumulative Lateness CI95"), digits=3)
        for _, row in df.iterrows()
    ]
    lines.append(" & ".join(cumlate_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _infer_ablation_sim_root(log_dir: Path, ablation_type: str, run_name: str | None) -> Path | None:
    if not log_dir.exists():
        return None
    pattern = (
        "run_proxy_arch_ablation_fixed_*.out"
        if ablation_type == "arch"
        else "run_proxy_loss_ablation_fixed_*.out"
    )
    candidate_logs = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    run_token: str | None = None
    if run_name:
        run_token = str(run_name).strip().split("_")[-1]
        if not run_token.isdigit():
            run_token = None
    for path in candidate_logs:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if run_token and f"training array job: {run_token}" not in text:
            continue
        match = re.search(r"dedicated simulation root:\s*(\S+)", text)
        if match:
            return Path(match.group(1))
    return None


def _infer_inference_proxy_model(
    *,
    sim_root: Path,
    order_set: str,
    date_from: str | None,
    date_to: str | None,
    run_tag_prefix: str | None,
) -> str | None:
    prefix = str(run_tag_prefix or "").strip()
    if not prefix:
        return None

    df = _build_filtered_summary_df(
        root=sim_root,
        order_set=order_set,
        date=None,
        date_from=date_from,
        date_to=date_to,
        algo=["proxy"],
        include_max_orders=False,
        require_full_date_coverage=False,
        proxy_model_name=None,
        proxy_model_contains=None,
        proxy_strategy_preference=None,
    )
    if df.empty or "proxy_run_tag" not in df.columns or "proxy_model_name" not in df.columns:
        return None

    tags = df["proxy_run_tag"].fillna("").astype(str).str.strip()
    tagged = df.loc[tags.str.startswith(f"{prefix}_")].copy()
    if tagged.empty:
        return None

    tagged["proxy_model_name"] = tagged["proxy_model_name"].fillna("").astype(str).str.strip()
    tagged = tagged[tagged["proxy_model_name"] != ""]
    if tagged.empty:
        return None

    if "path" in tagged.columns:
        tagged["summary_mtime"] = tagged["path"].apply(
            lambda p: Path(str(p)).stat().st_mtime if Path(str(p)).exists() else float("-inf")
        )
    else:
        tagged["summary_mtime"] = float("-inf")

    grouped = (
        tagged.groupby("proxy_model_name", dropna=False)
        .agg(
            max_summary_mtime=("summary_mtime", "max"),
            n_rows=("proxy_model_name", "size"),
        )
        .reset_index()
        .sort_values(
            ["max_summary_mtime", "n_rows", "proxy_model_name"],
            ascending=[False, False, False],
            na_position="last",
        )
    )
    if grouped.empty:
        return None
    return str(grouped.iloc[0]["proxy_model_name"]).strip() or None


def _infer_latest_ablation_sim_root(base_dir: Path, ablation_type: str) -> Path | None:
    if not base_dir.exists():
        return None
    prefix = "arch_ablation_fixed_" if ablation_type == "arch" else "loss_ablation_fixed_"
    candidates = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize proxy ablation results into printable + LaTeX tables.")
    parser.add_argument(
        "--type",
        choices=["arch", "loss", "inference"],
        default=None,
        help="Ablation type to summarize.",
    )
    parser.add_argument(
        "--source",
        choices=["simulation", "validation"],
        default=None,
        help="Summary source for arch/loss. Defaults to simulation.",
    )

    # Shared outputs
    parser.add_argument("--out-dir", type=Path, default=Path("logs/summaries"))
    parser.add_argument("--out-prefix", type=str, default=None, help="Output filename prefix (without extension).")

    # Arch/loss inputs
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional precomputed summary CSV to format.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/tune"))
    parser.add_argument(
        "--run-name",
        action="append",
        default=None,
        help=(
            "Repeatable; supports comma-separated values. "
            "If omitted for arch/loss, the script auto-detects the latest parseable ablation run."
        ),
    )
    parser.add_argument("--group-col", type=str, default="ablation_tag")
    parser.add_argument("--selection-metric", type=str, default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--full-model-name", type=str, default=None)
    parser.add_argument("--models-root", type=Path, default=Path("data/models/proxy"))

    # Inference inputs
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=None,
        help="Simulation root. For arch/loss, if omitted, inferred from wrapper logs.",
    )
    parser.add_argument("--sim-root-base", type=Path, default=Path("data/peak/simulation_results_ablation"))
    parser.add_argument("--order-set", type=str, default="test", choices=["test", "proxy_train"])
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--proxy-model-name", type=str, default=None)
    parser.add_argument("--run-tag-prefix", type=str, default=None)
    parser.add_argument(
        "--include-default",
        action="store_true",
        help="Include additional untagged strategy runs (inventory_weighted baseline is always included).",
    )
    parser.add_argument(
        "--allow-partial-date-coverage",
        action="store_true",
        help="Do not require every ablation run to cover all selected dates.",
    )

    args = parser.parse_args()
    types_to_run = [args.type] if args.type else ["arch", "loss", "inference"]
    run_names = _split_tokens(args.run_name)
    run_name = run_names[0] if run_names else None
    had_failure = False

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for ablation_type in types_to_run:
        source = args.source or "simulation"
        if args.out_prefix and len(types_to_run) > 1:
            out_prefix = f"{args.out_prefix}_{ablation_type}"
        else:
            out_prefix = args.out_prefix or f"{ablation_type}_ablation"
        out_csv = out_dir / f"{out_prefix}_table.csv"
        out_txt = out_dir / f"{out_prefix}_table.txt"

        if ablation_type in {"arch", "loss"} and source == "validation":
            full_model_name = str(args.full_model_name or "").strip() or None
            if full_model_name is None:
                full_model_name = _infer_full_model_name_from_wrapper_logs(args.log_dir, ablation_type)
                if full_model_name:
                    print(
                        f"[summarize_proxy_ablations] --type={ablation_type}: "
                        f"inferred full model='{full_model_name}' from wrapper logs"
                    )
                else:
                    print(
                        f"[summarize_proxy_ablations] --type={ablation_type}: "
                        "no default full model inferred; pass --full-model-name to include baseline row"
                    )
            summary_df = _build_arch_or_loss_summary(
                summary_csv=args.summary_csv,
                log_dir=args.log_dir,
                run_names=run_names,
                ablation_type=ablation_type,
                group_col=args.group_col,
                selection_metric_spec=args.selection_metric,
                full_model_name=full_model_name,
                models_root=args.models_root,
            )
            if summary_df.empty:
                print(f"No rows found for requested {ablation_type} validation ablation summary.")
                had_failure = True
                continue
            printable = _canonicalize_arch_or_loss_table(summary_df)
            latex = _to_latex_arch_or_loss(printable, title="Architecture" if ablation_type == "arch" else "Loss")
            summary_df.to_csv(out_csv, index=False)
        else:
            default_prefix = (
                "ablation_arch_fixed"
                if ablation_type == "arch"
                else "ablation_loss_fixed"
                if ablation_type == "loss"
                else "infstrat"
            )
            run_tag_prefix = str(args.run_tag_prefix or default_prefix).strip() or None
            sim_root = args.sim_root
            if sim_root is None:
                if ablation_type in {"arch", "loss"}:
                    sim_root = _infer_ablation_sim_root(args.log_dir, ablation_type, run_name)
                    if sim_root is None:
                        sim_root = _infer_latest_ablation_sim_root(args.sim_root_base, ablation_type)
                    if sim_root is None:
                        print(
                            f"Could not infer simulation root for {ablation_type} ablation. "
                            "Pass --sim-root explicitly."
                        )
                        had_failure = True
                        continue
                    print(
                        f"[summarize_proxy_ablations] --type={ablation_type}: "
                        f"using simulation root '{sim_root}'"
                    )
                else:
                    sim_root = Path("data/peak/simulation_results")

            proxy_model_name = str(args.proxy_model_name or "").strip() or None
            if ablation_type in {"arch", "loss"} and proxy_model_name:
                print(
                    f"[summarize_proxy_ablations] --type={ablation_type}: "
                    "--proxy-model-name is ignored for arch/loss simulation summaries."
                )
            proxy_model_name_filter = proxy_model_name if ablation_type == "inference" else None

            if ablation_type == "inference" and proxy_model_name is None:
                inferred_model = _infer_inference_proxy_model(
                    sim_root=sim_root,
                    order_set=args.order_set,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    run_tag_prefix=run_tag_prefix,
                )
                if inferred_model:
                    proxy_model_name = inferred_model
                    proxy_model_name_filter = inferred_model
                    print(
                        "[summarize_proxy_ablations] --type=inference: "
                        f"inferred proxy model='{proxy_model_name}' from tagged simulation rows"
                    )

            summary_df = _build_inference_summary(
                sim_root=sim_root,
                order_set=args.order_set,
                date_from=args.date_from,
                date_to=args.date_to,
                proxy_model_name=proxy_model_name_filter,
                run_tag_prefix=run_tag_prefix,
                include_default=args.include_default,
                require_full_date_coverage=not args.allow_partial_date_coverage,
                inject_inventory_baseline=ablation_type == "inference",
            )
            if summary_df.empty:
                print(f"No rows found for requested {ablation_type} simulation ablation summary.")
                had_failure = True
                continue
            printable = _canonicalize_inference_table(summary_df)
            latex = _to_latex_inference(
                summary_df,
                ablation_title=(
                    "Architecture"
                    if ablation_type == "arch"
                    else "Loss"
                    if ablation_type == "loss"
                    else "Inference strategy"
                ),
            )
            summary_df.to_csv(out_csv, index=False)

        _render_printable_table(printable, out_txt)

        print(f"\n=== Printable Table ({ablation_type}) ===")
        print(printable.to_string(index=False))
        print(f"\n=== LaTeX Table ({ablation_type}) ===")
        print(latex)
        print(f"\nWrote CSV: {out_csv}")
        print(f"Wrote printable table: {out_txt}")

    return 2 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
