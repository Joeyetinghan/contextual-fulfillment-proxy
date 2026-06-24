#!/usr/bin/env python3
"""
Summarize scenario-count sensitivity runs (CSAA vs proxy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_money(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace("$", "").replace(",", "").strip()
    if not s:
        return None
    return float(s)


def _parse_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace("%", "").replace(",", "").strip()
    if not s:
        return None
    return float(s)


def _extract_n1_from_path(path: Path, root: Path):
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    for part in rel.parts:
        if part.startswith("n1_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _infer_algo(summary_path: Path, payload: dict):
    if payload.get("proxy_model_name"):
        return "proxy"
    stem = summary_path.stem.lower()
    if stem.startswith("csaa"):
        return "csaa"
    if stem.startswith("proxy"):
        return "proxy"
    if stem.startswith("empirical"):
        return "empirical_saa"
    if stem.startswith("greedy"):
        return "greedy"
    if stem.startswith("pto"):
        return "pto"
    return None


def _summary_to_parquet(summary_path: str | Path) -> Path:
    p = Path(summary_path)
    if p.name.endswith("_summary.json"):
        stem = p.name[: -len("_summary.json")]
        return p.with_name(f"{stem}.parquet")
    return p.with_suffix(".parquet")


def _compute_raw_objective_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute objective aligned with collect_sim_summaries raw_rep_total logic:
      - per date/file: sum realized_cost over orders per replication
      - across dates: sum those per-replication totals by replication id
      - report mean/std/ci95 across replications
    """
    rep_rows: list[pd.DataFrame] = []
    iter_cols = [col for col in ["summary_path", "algo", "n1", "date", "proxy_model_name"] if col in df.columns]
    work_df = df.loc[:, iter_cols].drop_duplicates()
    for _, row in work_df.iterrows():
        summary_path = row.get("summary_path")
        if not summary_path:
            continue
        parquet_path = _summary_to_parquet(summary_path)
        if not parquet_path.exists():
            continue
        try:
            raw = pd.read_parquet(parquet_path, columns=["replication", "realized_cost"])
        except Exception:
            continue
        if raw.empty:
            continue
        raw["replication"] = pd.to_numeric(raw["replication"], errors="coerce")
        raw["realized_cost"] = pd.to_numeric(raw["realized_cost"], errors="coerce")
        raw = raw.dropna(subset=["replication", "realized_cost"])
        if raw.empty:
            continue
        raw["replication"] = raw["replication"].astype(int)

        rep_totals = (
            raw.groupby("replication", as_index=False)["realized_cost"]
            .sum()
            .rename(columns={"realized_cost": "rep_total_cost"})
        )
        rep_totals["algo"] = row.get("algo")
        rep_totals["n1"] = int(row.get("n1"))
        rep_rows.append(rep_totals)

    if not rep_rows:
        return pd.DataFrame(columns=[
            "algo", "n1", "raw_objective_mean", "raw_objective_std",
            "raw_objective_ci95", "raw_objective_reps",
        ])

    rep_df = pd.concat(rep_rows, ignore_index=True)
    rep_df = (
        rep_df.groupby(["algo", "n1", "replication"], as_index=False)["rep_total_cost"]
        .sum()
    )

    out = (
        rep_df.groupby(["algo", "n1"], as_index=False)
        .agg(
            raw_objective_mean=("rep_total_cost", "mean"),
            raw_objective_std=("rep_total_cost", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
            raw_objective_reps=("rep_total_cost", "size"),
        )
    )
    out["raw_objective_ci95"] = (
        1.96 * out["raw_objective_std"] / np.sqrt(out["raw_objective_reps"].clip(lower=1))
    )
    return out


def _normalize_sensitivity_rows(
    df: pd.DataFrame,
    proxy_model_names: Iterable[str] | None,
) -> pd.DataFrame:
    if df.empty:
        return df

    proxy_name_set = {str(name).strip() for name in (proxy_model_names or []) if str(name).strip()}
    proxy_mask = df["algo"] == "proxy"

    proxy_df = df.loc[proxy_mask].copy()
    if not proxy_df.empty:
        proxy_df["proxy_model_name"] = proxy_df["proxy_model_name"].fillna("").astype(str).str.strip()
        proxy_df["proxy_run_tag"] = proxy_df["proxy_run_tag"].fillna("").astype(str).str.strip()

        if proxy_name_set:
            proxy_df = proxy_df[proxy_df["proxy_model_name"].isin(proxy_name_set)].copy()
        else:
            named_proxy_df = proxy_df[proxy_df["proxy_model_name"] != ""].copy()
            if not named_proxy_df.empty:
                named_proxy_df["_untagged"] = named_proxy_df["proxy_run_tag"] == ""
                model_dates = named_proxy_df.groupby("proxy_model_name")["date"].nunique().rename("dates")
                untagged_dates = (
                    named_proxy_df[named_proxy_df["_untagged"]]
                    .groupby("proxy_model_name")["date"]
                    .nunique()
                    .rename("untagged_dates")
                )
                model_score = (
                    pd.concat([model_dates, untagged_dates], axis=1)
                    .fillna(0)
                    .reset_index()
                    .sort_values(
                        ["untagged_dates", "dates", "proxy_model_name"],
                        ascending=[False, False, True],
                    )
                )
                if len(model_score) > 1:
                    chosen_model = str(model_score.iloc[0]["proxy_model_name"])
                    proxy_df = proxy_df[proxy_df["proxy_model_name"] == chosen_model].copy()
                    print(
                        "Note: multiple proxy models found in scenario-sensitivity rows; "
                        f"using '{chosen_model}'."
                    )

        if not proxy_df.empty:
            proxy_df["_has_tag"] = proxy_df["proxy_run_tag"] != ""
            proxy_df["_path_len"] = proxy_df["summary_path"].astype(str).str.len()
            proxy_df = proxy_df.sort_values(
                ["date", "n1", "proxy_model_name", "_has_tag", "proxy_run_tag", "_path_len", "summary_path"],
                ascending=[True, True, True, True, True, True, True],
            )
            proxy_before = len(proxy_df)
            proxy_df = proxy_df.drop_duplicates(subset=["date", "n1", "proxy_model_name"], keep="first")
            proxy_dropped = proxy_before - len(proxy_df)
            if proxy_dropped:
                print(
                    "Note: dropped "
                    f"{proxy_dropped} duplicate proxy scenario-sensitivity row(s) after date/n1 deduplication."
                )
            proxy_df = proxy_df.drop(columns=["_has_tag", "_path_len"], errors="ignore")

    non_proxy_df = df.loc[~proxy_mask].copy()
    if not non_proxy_df.empty:
        non_proxy_df["_stem"] = non_proxy_df["summary_path"].apply(lambda p: Path(p).stem.lower())
        non_proxy_df["_algo"] = non_proxy_df["algo"].fillna("").astype(str).str.lower()
        non_proxy_df["_is_default"] = (
            (non_proxy_df["_stem"] == (non_proxy_df["_algo"] + "_summary"))
            | (non_proxy_df["_stem"] == (non_proxy_df["_algo"] + "_peak_summary"))
        )
        non_proxy_df["_path_len"] = non_proxy_df["summary_path"].astype(str).str.len()
        non_proxy_df = non_proxy_df.sort_values(
            ["date", "algo", "n1", "_is_default", "_path_len", "summary_path"],
            ascending=[True, True, True, False, True, True],
        )
        non_proxy_before = len(non_proxy_df)
        non_proxy_df = non_proxy_df.drop_duplicates(subset=["date", "algo", "n1"], keep="first")
        non_proxy_dropped = non_proxy_before - len(non_proxy_df)
        if non_proxy_dropped:
            print(
                "Note: dropped "
                f"{non_proxy_dropped} duplicate non-proxy scenario-sensitivity row(s) after date/n1 deduplication."
            )
        non_proxy_df = non_proxy_df.drop(columns=["_stem", "_algo", "_is_default", "_path_len"], errors="ignore")

    out = pd.concat([non_proxy_df, proxy_df], ignore_index=True)
    return out


def load_rows(
    root: Path,
    order_set: str,
    algos: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    proxy_model_names: Iterable[str] | None = None,
):
    rows = []
    for summary_path in root.rglob("*_summary.json"):
        rel = summary_path.relative_to(root)
        if len(rel.parts) < 5:
            continue
        path_order_set = rel.parts[0]
        algo = rel.parts[1]
        date = rel.parts[3]
        if path_order_set != order_set:
            continue
        if algos and algo not in algos:
            continue
        if date_from and date < date_from:
            continue
        if date_to and date > date_to:
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        n1 = payload.get("saa_n1")
        if n1 is None:
            n1 = _extract_n1_from_path(summary_path, root)
        if n1 is None:
            continue
        row = {
            "summary_path": str(summary_path),
            "source": "sensitivity",
            "order_set": path_order_set,
            "algo": algo,
            "date": date,
            "n1": int(n1),
            "saa_q": payload.get("saa_q"),
            "saa_n2": payload.get("saa_n2"),
            "proxy_model_name": payload.get("proxy_model_name"),
            "proxy_run_tag": payload.get("proxy_run_tag"),
            "proxy_scenario_len": payload.get("proxy_scenario_len"),
            "avg_realized_cost": _parse_money(payload.get("avg_realized_cost")),
            "avg_policy_runtime_ms": _parse_float(payload.get("avg_policy_runtime_ms")),
            "total_policy_runtime_s": _parse_float(payload.get("total_policy_runtime_s")),
            "orders_evaluated": payload.get("orders_evaluated"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return _normalize_sensitivity_rows(df, proxy_model_names)


def load_standard_baseline_rows(
    standard_root: Path,
    order_set: str,
    baseline_n1: int,
    algos: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    proxy_model_names: Iterable[str] | None,
):
    rows = []
    root = standard_root / order_set
    if not root.exists():
        return pd.DataFrame(rows)
    proxy_name_set = {name for name in (proxy_model_names or []) if name}
    for summary_path in root.rglob("*_summary.json"):
        rel = summary_path.relative_to(standard_root)
        if len(rel.parts) < 4:
            continue
        path_order_set = rel.parts[0]
        date = rel.parts[1]
        if path_order_set != order_set:
            continue
        if date_from and date < date_from:
            continue
        if date_to and date > date_to:
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        algo = _infer_algo(summary_path, payload)
        if algo is None:
            continue
        if algos and algo not in algos:
            continue
        if algo == "proxy":
            model_name = payload.get("proxy_model_name")
            if proxy_name_set and model_name not in proxy_name_set:
                continue

        row = {
            "summary_path": str(summary_path),
            "source": "standard_baseline",
            "order_set": path_order_set,
            "algo": algo,
            "date": date,
            "n1": int(baseline_n1),
            "saa_q": payload.get("saa_q"),
            "saa_n2": payload.get("saa_n2"),
            "proxy_model_name": payload.get("proxy_model_name"),
            "proxy_run_tag": payload.get("proxy_run_tag"),
            "proxy_scenario_len": payload.get("proxy_scenario_len"),
            "avg_realized_cost": _parse_money(payload.get("avg_realized_cost")),
            "avg_policy_runtime_ms": _parse_float(payload.get("avg_policy_runtime_ms")),
            "total_policy_runtime_s": _parse_float(payload.get("total_policy_runtime_s")),
            "orders_evaluated": payload.get("orders_evaluated"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    proxy_mask = df["algo"] == "proxy"
    if not proxy_mask.any():
        return df

    proxy_df = df.loc[proxy_mask].copy()
    proxy_df["proxy_model_name"] = proxy_df["proxy_model_name"].fillna("").astype(str).str.strip()
    proxy_df = proxy_df[proxy_df["proxy_model_name"] != ""].copy()
    if proxy_df.empty:
        return df.loc[~proxy_mask].copy()

    if not proxy_name_set:
        proxy_df["_untagged"] = (
            proxy_df["proxy_run_tag"].fillna("").astype(str).str.strip() == ""
        )
        model_dates = proxy_df.groupby("proxy_model_name")["date"].nunique().rename("dates")
        untagged_dates = (
            proxy_df[proxy_df["_untagged"]]
            .groupby("proxy_model_name")["date"]
            .nunique()
            .rename("untagged_dates")
        )
        model_score = (
            pd.concat([model_dates, untagged_dates], axis=1)
            .fillna(0)
            .reset_index()
            .sort_values(["untagged_dates", "dates", "proxy_model_name"], ascending=[False, False, True])
        )
        chosen_model = str(model_score.iloc[0]["proxy_model_name"])
        proxy_df = proxy_df[proxy_df["proxy_model_name"] == chosen_model].copy()
        print(
            "Note: --proxy-model-name not provided; "
            f"using standard proxy baseline model '{chosen_model}'."
        )

    proxy_df["_proxy_run_tag"] = proxy_df["proxy_run_tag"].fillna("").astype(str)
    proxy_df["_has_tag"] = proxy_df["_proxy_run_tag"].str.strip() != ""
    proxy_df = proxy_df.sort_values(["date", "proxy_model_name", "_has_tag", "_proxy_run_tag", "summary_path"])
    proxy_df = proxy_df.drop_duplicates(subset=["date", "proxy_model_name"], keep="first")
    proxy_df = proxy_df.drop(columns=["_proxy_run_tag", "_has_tag"], errors="ignore")

    non_proxy_df = df.loc[~proxy_mask].copy()
    if not non_proxy_df.empty:
        non_proxy_df["_stem"] = non_proxy_df["summary_path"].apply(lambda p: Path(p).stem.lower())
        non_proxy_df["_algo"] = non_proxy_df["algo"].fillna("").astype(str).str.lower()
        non_proxy_df["_is_default"] = (
            (non_proxy_df["_stem"] == (non_proxy_df["_algo"] + "_summary"))
            | (non_proxy_df["_stem"] == (non_proxy_df["_algo"] + "_peak_summary"))
        )
        non_proxy_df["_path_len"] = non_proxy_df["summary_path"].astype(str).str.len()
        non_proxy_df = non_proxy_df.sort_values(
            ["date", "algo", "_is_default", "_path_len", "summary_path"],
            ascending=[True, True, False, True, True],
        )
        non_proxy_df = non_proxy_df.drop_duplicates(subset=["date", "algo"], keep="first")
        non_proxy_df = non_proxy_df.drop(columns=["_stem", "_algo", "_is_default", "_path_len"], errors="ignore")

    out = pd.concat([non_proxy_df, proxy_df], ignore_index=True)
    return out


def plot_metric(df_agg: pd.DataFrame, metric: str, ylabel: str, out_path: Path):
    palette = {
        "csaa": "#d55e00",
        "proxy": "#0072b2",
    }
    pretty_algo = {
        "csaa": "CSAA",
        "proxy": "Proxy",
    }
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for algo, g in df_agg.groupby("algo"):
        g = g.sort_values("n1")
        color = palette.get(algo, None)
        ax.errorbar(
            g["n1"],
            g[f"{metric}_mean"],
            yerr=g[f"{metric}_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=pretty_algo.get(algo, str(algo).upper()),
            color=color,
        )
    ax.set_xlabel("Scenario size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Scenario Sensitivity: {ylabel}")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined_cost_runtime(df_agg: pd.DataFrame, out_path: Path):
    palette = {
        "csaa": "#d55e00",
        "proxy": "#0072b2",
    }
    pretty_algo = {
        "csaa": "CSAA",
        "proxy": "Proxy",
    }
    fig, ax_cost = plt.subplots(figsize=(10.2, 4.9))
    ax_runtime = ax_cost.twinx()

    for algo, g in df_agg.groupby("algo"):
        g = g.sort_values("n1")
        color = palette.get(algo, None)
        if "objective_total_cost_mean" in g.columns:
            y = pd.to_numeric(g["objective_total_cost_mean"], errors="coerce").to_numpy(dtype=float)
            yerr = pd.to_numeric(g.get("objective_total_cost_std"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            valid = ~np.isnan(y)
            if np.any(valid):
                ax_cost.errorbar(
                    g.loc[valid, "n1"],
                    y[valid],
                    yerr=yerr[valid],
                    marker="o",
                    linewidth=2.1,
                    capsize=4,
                    label=f"{pretty_algo.get(algo, str(algo).upper())} obj",
                    color=color,
                )

        y_rt = pd.to_numeric(g.get("avg_policy_runtime_ms_mean"), errors="coerce").to_numpy(dtype=float)
        yerr_rt = pd.to_numeric(g.get("avg_policy_runtime_ms_std"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        valid_rt = ~np.isnan(y_rt)
        if np.any(valid_rt):
            ax_runtime.errorbar(
                g.loc[valid_rt, "n1"],
                y_rt[valid_rt],
                yerr=yerr_rt[valid_rt],
                marker="s",
                linestyle="--",
                linewidth=1.9,
                capsize=4,
                label=f"{pretty_algo.get(algo, str(algo).upper())} runtime",
                color=color,
                alpha=0.9,
            )

    ax_cost.set_xlabel("Scenario size")
    ax_cost.set_ylabel("Total objective value")
    ax_runtime.set_ylabel("Avg policy runtime (ms)", labelpad=12)
    fig.suptitle("Scenario sensitivity: objective and runtime vs scenario size", y=0.99)
    ax_cost.grid(alpha=0.25, linestyle="--")

    h1, l1 = ax_cost.get_legend_handles_labels()
    h2, l2 = ax_runtime.get_legend_handles_labels()
    fig.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=True,
    )

    proxy_runtime_note = _build_proxy_runtime_note(df_agg)
    if proxy_runtime_note:
        ax_runtime.text(
            0.88,
            0.10,
            proxy_runtime_note,
            transform=ax_runtime.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=palette["proxy"],
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": palette["proxy"],
                "alpha": 0.88,
            },
        )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_proxy_runtime_note(df_agg: pd.DataFrame) -> str | None:
    """Summarize proxy runtime in milliseconds for a compact annotation box."""
    if "algo" not in df_agg.columns or "avg_policy_runtime_ms_mean" not in df_agg.columns:
        return None
    proxy = df_agg[df_agg["algo"].astype(str).str.lower() == "proxy"].copy()
    if proxy.empty:
        return None
    runtime_ms = pd.to_numeric(proxy["avg_policy_runtime_ms_mean"], errors="coerce").dropna()
    if runtime_ms.empty:
        return None
    mean_ms = float(runtime_ms.mean())
    min_ms = float(runtime_ms.min())
    max_ms = float(runtime_ms.max())
    if abs(max_ms - min_ms) <= 0.5:
        return f"Proxy runtime\nwithin {max_ms:.1f} ms"
    return f"Proxy runtime\n{mean_ms:.1f} ms avg\n[{min_ms:.1f}, {max_ms:.1f}] ms"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize scenario sensitivity runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/peak/simulation_results/scenario_sensitivity"),
        help="Root directory containing scenario sensitivity outputs.",
    )
    parser.add_argument("--order-set", default="test", choices=["test", "proxy_train"])
    parser.add_argument(
        "--algos",
        nargs="*",
        default=["csaa", "proxy"],
        choices=["csaa", "proxy"],
        help="Algorithms to include (default: csaa proxy).",
    )
    parser.add_argument("--date-from", default=None, help="Optional YYYY-MM-DD lower bound (inclusive).")
    parser.add_argument("--date-to", default=None, help="Optional YYYY-MM-DD upper bound (inclusive).")
    parser.add_argument(
        "--standard-root",
        type=Path,
        default=Path("data/peak/simulation_results"),
        help="Standard simulation results root used to read baseline N1=50 runs.",
    )
    parser.add_argument("--baseline-n1", type=int, default=50, help="Baseline N1 label for standard runs.")
    parser.add_argument(
        "--proxy-model-name",
        default=None,
        help="Proxy model name to filter scenario-sensitivity and standard baseline rows.",
    )
    parser.add_argument(
        "--include-standard-baseline",
        action="store_true",
        default=True,
        help="Include baseline rows from --standard-root (default: enabled).",
    )
    parser.add_argument(
        "--no-include-standard-baseline",
        dest="include_standard_baseline",
        action="store_false",
        help="Disable loading baseline rows from --standard-root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/scratch/scenario_sensitivity"),
        help="Directory for CSV summaries and plots.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    requested_proxy_names = [args.proxy_model_name] if args.proxy_model_name else None
    df = load_rows(
        args.root,
        args.order_set,
        args.algos,
        args.date_from,
        args.date_to,
        requested_proxy_names,
    )
    proxy_names = []
    if args.proxy_model_name:
        proxy_names = [args.proxy_model_name]
    elif not df.empty:
        proxy_names = sorted(
            {
                name
                for name in df.loc[df["algo"] == "proxy", "proxy_model_name"].dropna().astype(str).tolist()
                if name
            }
        )
    if args.include_standard_baseline and ((args.algos is None) or ("proxy" in args.algos)):
        if not proxy_names:
            print(
                "Note: --proxy-model-name not provided. Baseline loader will auto-select one proxy model "
                "from standard results if needed."
            )

    if args.include_standard_baseline:
        baseline_df = load_standard_baseline_rows(
            standard_root=args.standard_root,
            order_set=args.order_set,
            baseline_n1=args.baseline_n1,
            algos=args.algos,
            date_from=args.date_from,
            date_to=args.date_to,
            proxy_model_names=proxy_names,
        )
        if not baseline_df.empty and not df.empty:
            key_cols = ["algo", "date", "n1", "proxy_model_name"]
            sens_keys = df[key_cols].fillna("-").astype(str).agg("|".join, axis=1)
            base_keys = baseline_df[key_cols].fillna("-").astype(str).agg("|".join, axis=1)
            baseline_df = baseline_df.loc[~base_keys.isin(set(sens_keys))]
        if not baseline_df.empty:
            df = pd.concat([df, baseline_df], ignore_index=True)

    if df.empty:
        print("No sensitivity summary files found for the requested filters.")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df["orders_evaluated"] = pd.to_numeric(df["orders_evaluated"], errors="coerce")
    df["total_realized_cost"] = df["avg_realized_cost"] * df["orders_evaluated"]
    rows_csv = args.out_dir / "scenario_sensitivity_rows.csv"
    df.sort_values(["algo", "n1", "date"]).to_csv(rows_csv, index=False)

    agg = (
        df.groupby(["algo", "n1"], as_index=False)
        .agg(
            dates=("date", "nunique"),
            total_realized_cost_mean=("total_realized_cost", "mean"),
            total_realized_cost_std=("total_realized_cost", "std"),
            avg_realized_cost_mean=("avg_realized_cost", "mean"),
            avg_realized_cost_std=("avg_realized_cost", "std"),
            avg_policy_runtime_ms_mean=("avg_policy_runtime_ms", "mean"),
            avg_policy_runtime_ms_std=("avg_policy_runtime_ms", "std"),
            total_policy_runtime_s_mean=("total_policy_runtime_s", "mean"),
            total_policy_runtime_s_std=("total_policy_runtime_s", "std"),
        )
        .sort_values(["algo", "n1"])
    )
    agg["total_realized_cost_ci95"] = 1.96 * agg["total_realized_cost_std"] / np.sqrt(agg["dates"].clip(lower=1))

    raw_obj = _compute_raw_objective_by_group(df)
    if not raw_obj.empty:
        agg = agg.merge(raw_obj, on=["algo", "n1"], how="left")
    else:
        agg["raw_objective_mean"] = np.nan
        agg["raw_objective_std"] = np.nan
        agg["raw_objective_ci95"] = np.nan
        agg["raw_objective_reps"] = np.nan

    agg["objective_total_cost_mean"] = agg["raw_objective_mean"].combine_first(agg["total_realized_cost_mean"])
    agg["objective_total_cost_std"] = agg["raw_objective_std"].combine_first(agg["total_realized_cost_std"])
    agg["objective_total_cost_ci95"] = agg["raw_objective_ci95"].combine_first(agg["total_realized_cost_ci95"])
    agg["objective_source"] = np.where(
        agg["raw_objective_mean"].notna(),
        "raw_rep_total",
        "summary_avg_times_orders",
    )
    # Keep per-date totals, but align total_realized_cost_* with collect_sim objective semantics.
    agg["total_realized_cost_per_date_mean"] = agg["total_realized_cost_mean"]
    agg["total_realized_cost_per_date_std"] = agg["total_realized_cost_std"]
    agg["total_realized_cost_per_date_ci95"] = agg["total_realized_cost_ci95"]
    agg["total_realized_cost_mean"] = agg["objective_total_cost_mean"]
    agg["total_realized_cost_std"] = agg["objective_total_cost_std"]
    agg["total_realized_cost_ci95"] = agg["objective_total_cost_ci95"]
    agg_csv = args.out_dir / "scenario_sensitivity_agg.csv"
    agg.to_csv(agg_csv, index=False)

    print("\n=== Scenario Sensitivity (obj/runtime vs scenario size) ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    non_raw = agg[agg["objective_source"] != "raw_rep_total"]
    if not non_raw.empty:
        print(
            "\nWarning: some objective rows are fallback (summary_avg_times_orders), "
            "so they may not exactly match collect_sim_summaries raw objective."
        )
        print(non_raw[["algo", "n1", "objective_source"]].to_string(index=False))
    print(
        "\nNote: total_realized_cost_* now matches collect_sim objective semantics "
        "(raw replication total across dates when available). "
        "Per-date means are kept in total_realized_cost_per_date_*."
    )
    print(f"\nWrote detailed rows: {rows_csv}")
    print(f"Wrote aggregated summary: {agg_csv}")

    if not args.no_plot:
        cost_plot = args.out_dir / "scenario_sensitivity_obj_vs_n1.pdf"
        runtime_plot = args.out_dir / "scenario_sensitivity_runtime_vs_n1.pdf"
        combined_plot = args.out_dir / "scenario_sensitivity_obj_runtime_combined_vs_n1.pdf"
        plot_metric(agg, "objective_total_cost", "Total Objective Value", cost_plot)
        plot_metric(agg, "avg_policy_runtime_ms", "Avg Policy Runtime (ms)", runtime_plot)
        plot_combined_cost_runtime(agg, combined_plot)
        print(f"Wrote plot: {cost_plot}")
        print(f"Wrote plot: {runtime_plot}")
        print(f"Wrote plot: {combined_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
