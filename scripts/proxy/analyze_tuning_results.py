#!/usr/bin/env python3
"""
Analyze proxy tuning results from training log files.

This parser is aligned with the current output format of:
  python -m src.training.proxy.train_proxy ...

Key focus: validation repaired metrics (what the simulator will execute).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import json
from typing import Any

import numpy as np
import pandas as pd


# Avoid truncating long model names in printed tables
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def _parse_scalar(raw: str) -> Any:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if any(ch in raw for ch in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_hyperparameters(param_str: str) -> dict[str, Any]:
    """Parse a CLI parameter string into a dict. Supports flags and --key value."""
    params: dict[str, Any] = {}
    tokens = shlex.split(param_str.strip())
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue

        key = tok[2:]
        value: Any = True
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            value = _parse_scalar(tokens[i + 1])
            i += 2
        else:
            i += 1

        # Canonicalize "--no-foo" into foo=False when the dest is "foo".
        if key.startswith("no-"):
            params[key[3:]] = False
        else:
            params[key] = value

    return params


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out) or np.isinf(out):
        return out
    return out


def _apply_final_best_eval_from_json(row: dict[str, Any]) -> None:
    """Override parsed metrics with final_best_eval.json when available."""
    out_dir = row.get("model_output_dir")
    if not out_dir:
        return
    eval_path = Path(str(out_dir)) / "final_best_eval.json"
    if not eval_path.exists():
        return

    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        return

    row["final_best_eval_json"] = str(eval_path)

    final_loss = _safe_float(payload.get("final_best_eval_val_loss"))
    if final_loss is not None:
        row["last_val_loss"] = final_loss
        row["val_loss_source"] = "final_best_eval_json"

    val_metrics = payload.get("final_best_eval_val_metrics")
    if isinstance(val_metrics, dict):
        mapping = {
            "hit1_dc": "val_dc_hit1_raw",
            "hit5_dc": "val_dc_hit5_raw",
            "mrr_dc": "val_dc_mrr_raw",
            "hit1_carrier": "val_carrier_hit1_raw",
            "hit5_carrier": "val_carrier_hit5_raw",
            "mrr_carrier": "val_carrier_mrr_raw",
            "joint_hit1": "val_joint_hit1_raw",
            "joint_hit5": "val_joint_hit5_raw",
            "mrr_joint": "val_joint_mrr_raw",
            "hit1_dc_repaired": "val_dc_hit1_repaired",
            "hit1_carrier_repaired": "val_carrier_hit1_repaired",
            "joint_hit1_repaired": "val_joint_hit1_repaired",
            "repair_changed_rate": "val_repair_changed_rate",
            "repair_over_inv_rate": "val_repair_over_inv_rate",
            "repair_over_inv_qty_frac": "val_repair_over_inv_qty_frac",
            "repair_raw_ineligible_rate": "val_repair_raw_ineligible_rate",
            "repair_raw_primary_ineligible_rate": "val_repair_raw_primary_ineligible_rate",
        }
        for src_k, dst_k in mapping.items():
            if src_k in val_metrics:
                v = _safe_float(val_metrics.get(src_k))
                if v is not None:
                    row[dst_k] = v
        row["val_metrics_source"] = "final_best_eval_json"

    ub_metrics = payload.get("final_best_eval_ub_metrics")
    if isinstance(ub_metrics, dict):
        ub_map = {
            "orders_evaluated": "val_ub_orders_evaluated",
            "orders_missing_artifacts": "val_ub_missing_artifacts",
            "orders_skipped": "val_ub_orders_skipped",
            "proxy_ub_mean": "val_proxy_ub_mean",
            "proxy_ub_ci95_mean": "val_proxy_ub_ci95_mean",
            "csaa_bar_mean": "val_csaa_bar_mean",
            "csaa_bar_ci95_mean": "val_csaa_bar_ci95_mean",
            "proxy_minus_csaa_bar": "val_proxy_minus_csaa_bar",
            "proxy_minus_csaa_median": "val_proxy_minus_csaa_median",
            "proxy_minus_csaa_p10": "val_proxy_minus_csaa_p10",
            "proxy_minus_csaa_p90": "val_proxy_minus_csaa_p90",
            "proxy_le_csaa_frac": "val_proxy_le_csaa_frac",
            "proxy_minus_csaa_z_mean": "val_proxy_minus_csaa_z_mean",
            "proxy_minus_csaa_z_median": "val_proxy_minus_csaa_z_median",
            "proxy_minus_csaa_z_count": "val_proxy_minus_csaa_z_count",
        }
        for src_k, dst_k in ub_map.items():
            if src_k not in ub_metrics:
                continue
            if dst_k.endswith("_count") or dst_k.endswith("_evaluated") or dst_k.endswith("_skipped") or dst_k.endswith("_artifacts"):
                try:
                    row[dst_k] = int(ub_metrics[src_k])
                except (TypeError, ValueError):
                    pass
            else:
                v = _safe_float(ub_metrics.get(src_k))
                if v is not None:
                    row[dst_k] = v
        row["val_ub_source"] = "final_best_eval_json"


def parse_log_file(file_path: str) -> list[dict[str, Any]]:
    """
    Parse one training .out file, which may contain multiple configs.
    We split on the public tuning delimiter: '--- Running config X / Y ---'.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as exc:
        print(f"Could not read {file_path}: {exc}")
        return []

    blocks = re.split(r"--- Running config \d+ / \d+ ---", content)
    if len(blocks) <= 1:
        return []

    # Current train_proxy.py patterns
    val_raw_pattern = (
        r"Val Metrics \(Raw\):\s+"
        r"DC\s+-\s+Hit@1:\s+([\d.]+)\s+\|\s+Hit@5:\s+([\d.]+)\s+\|\s+MRR:\s+([\d.]+)\s+"
        r"Carrier-\s+Hit@1:\s+([\d.]+)\s+\|\s+Hit@5:\s+([\d.]+)\s+\|\s+MRR:\s+([\d.]+)\s+"
        r"Joint\s+-\s+Hit@1:\s+([\d.]+)\s+\|\s+Hit@5:\s+([\d.]+)\s+\|\s+MRR:\s+([\d.]+)"
    )
    val_repaired_pattern = (
        r"Val Metrics \(Repaired\):\s+"
        r"DC\s+-\s+Hit@1:\s+([\d.]+)\s+"
        r"Carrier-\s+Hit@1:\s+([\d.]+)\s+"
        r"Joint\s+-\s+Hit@1:\s+([\d.]+)"
    )
    val_diag_1 = (
        r"Val Repair Diagnostics:\s+"
        r"Changed:\s+([\d.]+)\s+\|\s+"
        r"OverInv:\s+([\d.]+)\s+\(qty%\s+([\d.]+)\)\s+\|\s+"
        r"RawInelig:\s+([\d.]+)\s+\|\s+"
        r"RawPrimInelig:\s+([\d.]+)"
    )

    # CSAA-aligned UB diagnostics (optional; only present when ub_eval_orders > 0).
    ub_header_pattern = r"Val UB Metrics \(CSAA eval scenarios;\s+(\d+)\s+orders\):"
    ub_proxy_pattern = (
        r"Proxy UB - mean:\s+([^\s]+)\s+"
        r"\(avg CI95:\s+([^\s]+)\)\s+"
        r"\[missing=(\d+)\s+skipped=(\d+)\]"
    )
    ub_csaa_pattern = (
        r"CSAA bar_f - mean:\s+([^\s]+)\s+"
        r"\(avg CI95:\s+([^\s]+)\);\s+"
        r"Proxy - CSAA:\s+([^\s]+)"
    )
    ub_delta_pattern = (
        r"Proxy-CSAA median:\s+([^\s]+)\s+"
        r"\(p10=([^\s,]+),\s+p90=([^\s\)]+)\);\s+"
        r"P\(Proxy<=CSAA\):\s+([^\s]+)"
    )
    ub_z_pattern = (
        r"Proxy-CSAA z:\s+mean\s+([^\s,]+),\s+"
        r"median\s+([^\s]+)\s+\(n=(\d+)\)"
    )

    # Backward-compatible (older logs) - val raw metrics only
    val_simple_pattern = (
        r"Val Metrics:\s+"
        r"DC\s+-\s+Hit@1:\s+([\d.]+)\s+\|\s+Hit@5:\s+([\d.]+)\s+\|\s+MRR:\s+([\d.]+)\s+"
        r"Carrier-\s+Hit@1:\s+([\d.]+)\s+\|\s+Hit@5:\s+([\d.]+)\s+\|\s+MRR:\s+([\d.]+)\s+"
        r"Joint\s+-\s+Hit@1:\s+([\d.]+)\s+\|\s+Hit@5:\s+([\d.]+)\s+\|\s+MRR:\s+([\d.]+)"
    )

    results: list[dict[str, Any]] = []
    for block in blocks[1:]:
        params_match = re.search(r"Parameters:\s+(.*)", block)
        if not params_match:
            continue
        output_dir_match = re.search(r"Output Directory:\s+([^\r\n]+)", block)

        val_loss_matches = re.findall(
            r"va_loss:((?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)|(?:nan|inf|-inf))",
            block,
            flags=re.IGNORECASE,
        )
        val_raw = list(re.finditer(val_raw_pattern, block, re.DOTALL))
        val_rep = list(re.finditer(val_repaired_pattern, block, re.DOTALL))
        diag1 = list(re.finditer(val_diag_1, block, re.DOTALL))
        ub_header = list(re.finditer(ub_header_pattern, block, re.DOTALL))
        ub_proxy = list(re.finditer(ub_proxy_pattern, block, re.DOTALL))
        ub_csaa = list(re.finditer(ub_csaa_pattern, block, re.DOTALL))
        ub_delta = list(re.finditer(ub_delta_pattern, block, re.DOTALL))
        ub_z = list(re.finditer(ub_z_pattern, block, re.DOTALL))
        val_simple = list(re.finditer(val_simple_pattern, block, re.DOTALL))

        # Require at least val loss + either (raw+repaired) or older raw.
        if not val_loss_matches:
            continue

        row = parse_hyperparameters(params_match.group(1))
        if output_dir_match:
            out_dir_raw = output_dir_match.group(1).strip()
            row["model_output_dir"] = out_dir_raw
            row["model_output_name"] = Path(out_dir_raw).name
            row["model_best_pt"] = str(Path(out_dir_raw) / "best.pt")
            if "model_name" in row and row["model_name"]:
                row["model_name_short"] = row["model_name"]
            row["model_name_full"] = row["model_output_name"]
            row["model_name"] = row["model_output_name"]
        last_val_loss_raw = val_loss_matches[-1].lower()
        if last_val_loss_raw == "nan":
            row["last_val_loss"] = np.nan
        elif last_val_loss_raw == "inf":
            row["last_val_loss"] = np.inf
        elif last_val_loss_raw == "-inf":
            row["last_val_loss"] = -np.inf
        else:
            row["last_val_loss"] = float(last_val_loss_raw)

        if val_raw:
            m = val_raw[-1]
            row["val_dc_hit1_raw"] = float(m.group(1))
            row["val_dc_hit5_raw"] = float(m.group(2))
            row["val_dc_mrr_raw"] = float(m.group(3))
            row["val_carrier_hit1_raw"] = float(m.group(4))
            row["val_carrier_hit5_raw"] = float(m.group(5))
            row["val_carrier_mrr_raw"] = float(m.group(6))
            row["val_joint_hit1_raw"] = float(m.group(7))
            row["val_joint_hit5_raw"] = float(m.group(8))
            row["val_joint_mrr_raw"] = float(m.group(9))
        elif val_simple:
            m = val_simple[-1]
            row["val_dc_hit1_raw"] = float(m.group(1))
            row["val_dc_hit5_raw"] = float(m.group(2))
            row["val_dc_mrr_raw"] = float(m.group(3))
            row["val_carrier_hit1_raw"] = float(m.group(4))
            row["val_carrier_hit5_raw"] = float(m.group(5))
            row["val_carrier_mrr_raw"] = float(m.group(6))
            row["val_joint_hit1_raw"] = float(m.group(7))
            row["val_joint_hit5_raw"] = float(m.group(8))
            row["val_joint_mrr_raw"] = float(m.group(9))

        if val_rep:
            m = val_rep[-1]
            row["val_dc_hit1_repaired"] = float(m.group(1))
            row["val_carrier_hit1_repaired"] = float(m.group(2))
            row["val_joint_hit1_repaired"] = float(m.group(3))

        if diag1:
            m = diag1[-1]
            row["val_repair_changed_rate"] = float(m.group(1))
            row["val_repair_over_inv_rate"] = float(m.group(2))
            row["val_repair_over_inv_qty_frac"] = float(m.group(3))
            row["val_repair_raw_ineligible_rate"] = float(m.group(4))
            row["val_repair_raw_primary_ineligible_rate"] = float(m.group(5))

        if ub_proxy:
            m = ub_proxy[-1]
            if ub_header:
                row["val_ub_orders_evaluated"] = int(ub_header[-1].group(1))
            row["val_proxy_ub_mean"] = float(m.group(1))
            row["val_proxy_ub_ci95_mean"] = float(m.group(2))
            row["val_ub_missing_artifacts"] = int(m.group(3))
            row["val_ub_orders_skipped"] = int(m.group(4))

        if ub_csaa:
            m = ub_csaa[-1]
            row["val_csaa_bar_mean"] = float(m.group(1))
            row["val_csaa_bar_ci95_mean"] = float(m.group(2))
            row["val_proxy_minus_csaa_bar"] = float(m.group(3))
        if ub_delta:
            m = ub_delta[-1]
            row["val_proxy_minus_csaa_median"] = float(m.group(1))
            row["val_proxy_minus_csaa_p10"] = float(m.group(2))
            row["val_proxy_minus_csaa_p90"] = float(m.group(3))
            row["val_proxy_le_csaa_frac"] = float(m.group(4).rstrip("%")) / 100.0
        if ub_z:
            m = ub_z[-1]
            row["val_proxy_minus_csaa_z_mean"] = float(m.group(1))
            row["val_proxy_minus_csaa_z_median"] = float(m.group(2))
            row["val_proxy_minus_csaa_z_count"] = int(m.group(3))

        _apply_final_best_eval_from_json(row)
        row["log_file"] = os.path.basename(file_path)
        results.append(row)

    return results


def collect_all_results(log_dir: str, run_names: list[str] | None = None) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    print(f"Searching for log files in '{log_dir}'...")
    if run_names:
        print(f"  Filtering for run_name containing any of: {run_names}")

    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith(".out"):
            continue
        if run_names and not any(token in filename for token in run_names):
            continue
        path = os.path.join(log_dir, filename)
        results.extend(parse_log_file(path))

    if not results:
        print("No valid configs found in logs.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(f"Parsed {len(df)} configs across log files.")
    return df


def analyze_and_report(
    df: pd.DataFrame,
    top_n: int = 10,
    metric_specs: list[str] | None = None,
) -> None:
    if df.empty:
        return

    if "model_name" in df.columns:
        def _ablation_from_name(name: Any) -> str | None:
            if not isinstance(name, str):
                return None
            m = re.search(r"_abl([a-z0-9_]+)", name)
            return m.group(1) if m else None
        df["ablation_tag"] = df["model_name"].apply(_ablation_from_name)

    repaired_key = "val_joint_hit1_repaired" if "val_joint_hit1_repaired" in df.columns else None
    raw_key = "val_joint_hit1_raw" if "val_joint_hit1_raw" in df.columns else None

    display_cols = [
        c
        for c in [
            "model_name",
            "last_val_loss",
            "val_joint_hit1_repaired",
            "val_dc_hit1_repaired",
            "val_carrier_hit1_repaired",
            "val_proxy_ub_mean",
            "val_proxy_minus_csaa_bar",
            "val_proxy_minus_csaa_median",
            "val_proxy_minus_csaa_p10",
            "val_proxy_minus_csaa_p90",
            "val_proxy_le_csaa_frac",
            "val_proxy_minus_csaa_z_mean",
            "val_proxy_minus_csaa_z_median",
            "val_proxy_minus_csaa_z_count",
            "val_proxy_ub_ci95_mean",
            "val_ub_orders_evaluated",
            "val_joint_hit1_raw",
            "val_joint_hit5_raw",
            "val_dc_hit1_raw",
            "val_carrier_hit1_raw",
            "val_repair_changed_rate",
            "val_repair_over_inv_rate",
            "learning_rate",
            "weight_decay",
            "hidden_dim",
            "dropout_p",
            "cost_loss_weight",
            "constraint_loss_weight",
            "label_smoothing",
            "dc_embedding_dim",
            "carrier_emb_dim",
            "option_proj_dim",
        ]
        if c in df.columns
    ]

    requested_specs = metric_specs if metric_specs else ["auto"]
    seen: set[tuple[str, bool, bool]] = set()
    printed_any = False
    for metric_spec in requested_specs:
        try:
            metric, maximize, use_abs, label = resolve_selection_metric(df, metric_spec)
        except ValueError as exc:
            print(f"Skipping leaderboard for '{metric_spec}': {exc}")
            continue

        dedupe_key = (metric, maximize, use_abs)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        work = df.dropna(subset=[metric]).copy()
        if work.empty:
            continue
        if use_abs:
            work["_sort_metric"] = work[metric].abs()
            sort_col = "_sort_metric"
        else:
            sort_col = metric

        print("\n" + "=" * 90)
        print(f"Top {top_n} by [{label}] ({'higher is better' if maximize else 'lower is better'})")
        print("=" * 90)
        print(
            work
            .sort_values(sort_col, ascending=not maximize)
            .head(top_n)[display_cols]
            .to_string(index=False)
        )
        printed_any = True

    if not printed_any:
        print("\n" + "=" * 90)
        print(f"Top {top_n} by last_val_loss (lower is better)")
        print("=" * 90)
        print(df.sort_values("last_val_loss", ascending=True).head(top_n)[display_cols].to_string(index=False))

    if "ablation_tag" in df.columns and df["ablation_tag"].notna().any():
        group_cols = ["ablation_tag"]
        agg_map: dict[str, list[str]] = {"last_val_loss": ["mean", "min", "count"]}
        if repaired_key:
            agg_map[repaired_key] = ["mean", "max"]
        elif raw_key:
            agg_map[raw_key] = ["mean", "max"]
        summary = (
            df.dropna(subset=["ablation_tag"])
            .groupby(group_cols)
            .agg(agg_map)
            .sort_values(("last_val_loss", "mean"), ascending=True)
        )
        print("\n" + "=" * 90)
        print("Ablation Summary (grouped by model_name tag)")
        print("=" * 90)
        print(summary.to_string())

    # Hyperparameter impact on repaired joint (or raw fallback) + raw joint hit@5.
    primary_target = repaired_key or raw_key
    if primary_target:
        metric_cols = {
            "last_val_loss",
            "val_dc_hit1_raw",
            "val_dc_hit5_raw",
            "val_dc_mrr_raw",
            "val_carrier_hit1_raw",
            "val_carrier_hit5_raw",
            "val_carrier_mrr_raw",
            "val_joint_hit1_raw",
            "val_joint_hit5_raw",
            "val_joint_mrr_raw",
            "val_dc_hit1_repaired",
            "val_carrier_hit1_repaired",
            "val_joint_hit1_repaired",
            "val_proxy_ub_mean",
            "val_proxy_ub_ci95_mean",
            "val_csaa_bar_mean",
            "val_csaa_bar_ci95_mean",
            "val_proxy_minus_csaa_bar",
            "val_proxy_minus_csaa_median",
            "val_proxy_minus_csaa_p10",
            "val_proxy_minus_csaa_p90",
            "val_proxy_le_csaa_frac",
            "val_proxy_minus_csaa_z_mean",
            "val_proxy_minus_csaa_z_median",
            "val_proxy_minus_csaa_z_count",
            "val_ub_orders_evaluated",
            "val_ub_missing_artifacts",
            "val_ub_orders_skipped",
            "val_repair_changed_rate",
            "val_repair_over_inv_rate",
            "val_repair_over_inv_qty_frac",
            "val_repair_raw_ineligible_rate",
            "val_repair_raw_primary_ineligible_rate",
        }
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        params = [p for p in numeric if p not in metric_cols]

        impact_targets = [primary_target]
        if "val_joint_hit5_raw" in df.columns and "val_joint_hit5_raw" not in impact_targets:
            impact_targets.append("val_joint_hit5_raw")

        for target in impact_targets:
            print("\n" + "=" * 90)
            print(f"Hyperparameter impact on {target} (mean/std/count)")
            print("=" * 90)
            for p in params:
                if df[p].nunique() <= 1:
                    continue
                print(f"\n--- {p} ---")
                print(df.groupby(p)[target].agg(["mean", "std", "count"]).sort_values("mean", ascending=False))


def _sanitize_metric_key(metric_spec: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", metric_spec).strip("_")


def _with_metric_suffix(path: str, metric_spec: str, multi: bool) -> str:
    if not multi:
        return path
    p = Path(path)
    stem = p.stem
    suffix = p.suffix
    key = _sanitize_metric_key(metric_spec)
    return str(p.with_name(f"{stem}__{key}{suffix}"))


def resolve_selection_metric(df: pd.DataFrame, metric_spec: str) -> tuple[str, bool, bool, str]:
    """
    Return (metric_name, maximize, use_abs, display_label).

    metric_spec forms:
      - auto
      - metric_name
      - max:metric_name
      - min:metric_name
      - minabs:metric_name
    """
    if metric_spec == "auto":
        if "val_joint_hit1_repaired" in df.columns:
            return "val_joint_hit1_repaired", True, False, "auto(max repaired hit@1)"
        if "val_joint_hit1_raw" in df.columns:
            return "val_joint_hit1_raw", True, False, "auto(max raw hit@1)"
        return "last_val_loss", False, False, "auto(min val loss)"

    metric = metric_spec
    maximize: bool | None = None
    use_abs = False

    m = re.match(r"^(max|min|minabs):(.+)$", metric_spec)
    if m:
        mode = m.group(1)
        metric = m.group(2)
        if mode == "max":
            maximize = True
        elif mode == "min":
            maximize = False
        else:
            maximize = False
            use_abs = True

    if metric not in df.columns:
        raise ValueError(f"--selection_metric='{metric_spec}' resolved to '{metric}', which is not in parsed columns")

    if maximize is None:
        metric_l = metric.lower()
        if "minus_csaa" in metric_l:
            maximize = False
            use_abs = True
        elif any(tok in metric_l for tok in ["hit", "mrr", "le_csaa_frac"]):
            maximize = True
        elif any(tok in metric_l for tok in ["loss", "ub", "cost", "rate", "gap"]):
            maximize = False
        else:
            maximize = False

    label = f"{'max' if maximize else 'min'}"
    if use_abs:
        label += " abs"
    label += f" {metric}"
    return metric, maximize, use_abs, label


def select_best_per_group(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    maximize: bool,
    use_abs: bool = False,
) -> pd.DataFrame:
    work = df.dropna(subset=[metric]).copy()
    if work.empty:
        return work

    # Fallback: when group labels are missing (common outside ablation runs),
    # select one global best row so best_model_names_txt is never silently empty.
    has_groups = group_col in work.columns and work[group_col].notna().any()
    if has_groups:
        work = work.dropna(subset=[group_col]).copy()
    else:
        work[group_col] = "__all__"

    if work.empty:
        return work

    work["_sel_metric"] = work[metric].abs() if use_abs else work[metric]
    sort_cols = [group_col, "_sel_metric"]
    ascending = [True, True if use_abs else (not maximize)]
    if "last_val_loss" in work.columns and metric != "last_val_loss":
        sort_cols.append("last_val_loss")
        ascending.append(True)
    work = work.sort_values(sort_cols, ascending=ascending)
    best = work.groupby(group_col, as_index=False).first().drop(columns=["_sel_metric"], errors="ignore")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze hyperparameter tuning results from training log files.")
    parser.add_argument("--log_dir", type=str, default="logs/tune", help="Directory containing the .out log files.")
    parser.add_argument(
        "--run_name",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional substring filter for log filenames (repeatable). "
            "Also supports comma-separated values."
        ),
    )
    parser.add_argument("--top_n", type=int, default=10, help="How many rows to display in leaderboards.")
    parser.add_argument("--out_csv", type=str, default=None, help="Optional path to write parsed results as CSV.")
    parser.add_argument(
        "--selection_metric",
        type=str,
        action="append",
        default=None,
        help=(
            "Metric spec used for leaderboard printing and best-per-group selection. Repeatable. "
            "Formats: auto | metric | max:metric | min:metric | minabs:metric. "
            "Examples: --selection_metric max:val_joint_hit1_repaired "
            "--selection_metric max:val_joint_hit5_raw "
            "--selection_metric min:val_proxy_ub_mean "
            "--selection_metric max:val_proxy_le_csaa_frac. "
            "Default (when omitted): min:val_proxy_ub_mean + "
            "max:val_joint_hit1_repaired + max:val_joint_hit5_raw + "
            "max:val_proxy_le_csaa_frac."
        ),
    )
    parser.add_argument("--group_col", type=str, default="ablation_tag",
                        help="Column used to group runs when selecting best models.")
    parser.add_argument("--best_per_group_csv", type=str, default=None,
                        help="Optional path to write best row per group as CSV.")
    parser.add_argument("--best_model_names_txt", type=str, default=None,
                        help=(
                            "Path to write selected model names. "
                            "For one selection metric: one model per line. "
                            "For multiple metrics: a single TSV file with columns "
                            "metric_spec, group, model_name."
                        ))
    args = parser.parse_args()

    run_name_filters: list[str] | None = None
    if args.run_name:
        run_name_filters = []
        for raw in args.run_name:
            for token in str(raw).split(","):
                token = token.strip()
                if token:
                    run_name_filters.append(token)
        if not run_name_filters:
            run_name_filters = None

    df = collect_all_results(args.log_dir, run_name_filters)
    if df.empty:
        return 2

    if "model_name" in df.columns and "ablation_tag" not in df.columns:
        df["ablation_tag"] = df["model_name"].apply(
            lambda x: re.search(r"_abl([a-z0-9_]+)", x).group(1)
            if isinstance(x, str) and re.search(r"_abl([a-z0-9_]+)", x)
            else np.nan
        )

    if args.out_csv:
        out_path = args.out_csv
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote CSV: {out_path}")

    default_metric_specs = [
        "min:val_proxy_ub_mean",
        "max:val_joint_hit1_repaired",
        "max:val_joint_hit5_raw",
        "max:val_proxy_le_csaa_frac",
    ]
    metric_specs = args.selection_metric if args.selection_metric else default_metric_specs
    analyze_and_report(df, top_n=args.top_n, metric_specs=metric_specs)

    # Best-per-group export (end-to-end model selection for ablations).
    if args.best_per_group_csv or args.best_model_names_txt:
        multi_metric = len(metric_specs) > 1
        combined_name_rows: list[tuple[str, str, str]] = []
        for metric_spec in metric_specs:
            try:
                metric, maximize, use_abs, label = resolve_selection_metric(df, metric_spec)
            except ValueError as exc:
                # Keep default behavior robust when UB metrics are unavailable in older logs.
                if args.selection_metric is None:
                    print(f"Skipping default selection metric '{metric_spec}': {exc}")
                    continue
                raise
            best_df = select_best_per_group(
                df, args.group_col, metric, maximize, use_abs=use_abs
            )
            if best_df.empty:
                print(f"No rows available for best-per-group on {args.group_col} / {metric_spec}")
                continue

            cols = [c for c in [
                args.group_col,
                "model_name",
                "model_output_name",
                metric,
                "last_val_loss",
                "val_joint_hit1_repaired",
                "val_joint_hit5_raw",
                "val_dc_hit1_repaired",
                "val_carrier_hit1_repaired",
                "val_proxy_ub_mean",
                "val_proxy_minus_csaa_bar",
                "model_best_pt",
                "log_file",
            ] if c in best_df.columns]
            print("\n" + "=" * 90)
            print(f"Best Per {args.group_col} by [{label}]")
            print("=" * 90)
            print(best_df[cols].sort_values(args.group_col).to_string(index=False))

            if args.best_per_group_csv:
                out_path = _with_metric_suffix(args.best_per_group_csv, metric_spec, multi_metric)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                best_df.sort_values(args.group_col).to_csv(out_path, index=False)
                print(f"Wrote best-per-group CSV: {out_path}")

            if args.best_model_names_txt:
                names = (best_df.get("model_output_name", pd.Series(dtype=str)).fillna("")
                         .where(lambda s: s != "", best_df.get("model_name", pd.Series(dtype=str)).fillna("")))
                groups = best_df.get(args.group_col, pd.Series(["__all__"] * len(best_df))).fillna("__all__")
                if multi_metric:
                    for group, name in zip(groups.tolist(), names.tolist()):
                        if name:
                            combined_name_rows.append((metric_spec, str(group), str(name)))
                else:
                    out_path = args.best_model_names_txt
                    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        for name in names.tolist():
                            if name:
                                f.write(f"{name}\n")
                    print(f"Wrote model name list: {out_path}")

        if args.best_model_names_txt and multi_metric:
            out_path = args.best_model_names_txt
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            seen: set[tuple[str, str, str]] = set()
            unique_rows: list[tuple[str, str, str]] = []
            for row in combined_name_rows:
                if row in seen:
                    continue
                seen.add(row)
                unique_rows.append(row)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("metric_spec\tgroup\tmodel_name\n")
                for metric_spec, group, name in unique_rows:
                    f.write(f"{metric_spec}\t{group}\t{name}\n")
            print(f"Wrote combined model name list: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
