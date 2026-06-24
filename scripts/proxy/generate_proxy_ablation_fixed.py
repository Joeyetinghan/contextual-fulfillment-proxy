#!/usr/bin/env python3
"""
Generate fixed-hyperparameter proxy ablation grids (no tuning).

This script is intentionally separate from generate_and_tune_proxy.py so fixed
ablation workflows remain concise and explicit.

Generated ablations follow the refit-style training setup:
  - train on full train+validation data
  - reuse the selected epoch count from the base model when available
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.proxy.generate_and_tune_proxy import (  # noqa: E402
    _hierarchical_proxy_ablation_definitions,
    _hierarchical_proxy_loss_ablation_definitions,
    _load_base_from_model_checkpoint,
    _merge_base_config,
    _sanitize_config,
    _trim_or_pad,
    write_grid_file,
)
from scripts.proxy.generate_refit_proxy_grid import _load_best_epoch  # noqa: E402


def _build_fixed_configs(kind: str, include_full: bool) -> List[Dict[str, Any]]:
    if kind == "architecture":
        defs = _hierarchical_proxy_ablation_definitions(include_full=include_full)
    elif kind == "loss":
        defs = _hierarchical_proxy_loss_ablation_definitions(include_full=include_full)
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    configs: List[Dict[str, Any]] = []
    for tag, override in defs:
        cfg = dict(override)
        cfg["ablation_tag"] = tag
        configs.append(cfg)
    return configs


def _apply_refit_style_overrides(base_cfg: Dict[str, Any], base_model: Path | None) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    original_epochs = int(cfg.get("epochs", 500))
    refit_epochs = original_epochs
    if base_model is not None:
        refit_epochs = _load_best_epoch(base_model.parent) or original_epochs

    cfg.update(
        {
            "refit_full_data": True,
            "split_ratio": 0.0,
            "epochs": int(refit_epochs),
            "early_stopping_patience": max(
                int(refit_epochs) + 1,
                int(cfg.get("early_stopping_patience", 15)),
            ),
            "ub_eval_orders": 0,
            "final_eval_on_best": False,
            "final_ub_eval_orders": 0,
        }
    )
    return _sanitize_config(cfg)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate fixed-hyperparameter ablation config files for proxy training."
    )
    parser.add_argument("--kind", type=str, required=True, choices=["architecture", "loss"])
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--base-model", type=Path, default=None, help="Path to tuned checkpoint best.pt")
    parser.add_argument(
        "--components-only",
        dest="components_only",
        action="store_true",
        help="Exclude full baseline variant (default).",
    )
    parser.add_argument(
        "--include-full",
        dest="components_only",
        action="store_false",
        help="Include the full baseline variant.",
    )
    parser.add_argument("--n-jobs", type=int, default=None, help="Optional target number of jobs.")
    parser.add_argument("--configs-per-job", type=int, default=1, help="Configs per job line-batch.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle configs (off by default).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(components_only=True)
    args = parser.parse_args()

    if (args.base_config is None) == (args.base_model is None):
        raise ValueError("Specify exactly one of --base-config or --base-model.")

    include_full = not args.components_only
    configs = _build_fixed_configs(args.kind, include_full=include_full)

    if args.base_config is not None:
        with args.base_config.open("r", encoding="utf-8") as f:
            base_cfg = json.load(f)
        base_cfg.pop("model_name", None)
        base_cfg = _sanitize_config(base_cfg)
        base_cfg = _apply_refit_style_overrides(base_cfg, base_model=None)
        source_msg = f"base-config: {args.base_config}"
    else:
        base_cfg = _load_base_from_model_checkpoint(args.base_model)
        base_cfg = _apply_refit_style_overrides(base_cfg, base_model=args.base_model)
        source_msg = f"base-model: {args.base_model}"

    configs = [_sanitize_config(_merge_base_config(base_cfg, cfg)) for cfg in configs]

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(configs)
        print(f"Applied deterministic shuffle with seed={args.seed}")
    else:
        print("Shuffle disabled; using deterministic ablation order")

    if args.n_jobs is not None:
        target_count = args.n_jobs * args.configs_per_job
        configs = _trim_or_pad(configs, target_count)
    else:
        target_count = ((len(configs) + args.configs_per_job - 1) // args.configs_per_job) * args.configs_per_job
        configs = _trim_or_pad(configs, target_count)

    if args.output is None:
        suffix = "arch" if args.kind == "architecture" else "loss"
        args.output = Path("configs/proxy/paper/grids") / f"proxy_{suffix}_ablation.txt"

    print(f"Generating fixed ablation grid ({args.kind}) using {source_msg}")
    print(f"Generated {len(configs)} configs ({len(configs) // args.configs_per_job} jobs x {args.configs_per_job} cfg/job)")

    if args.dry_run:
        for i, cfg in enumerate(configs[: min(5, len(configs))], start=1):
            print(f"[sample {i}] ablation_tag={cfg.get('ablation_tag')}")
        return 0

    write_grid_file(configs, args.output, args.configs_per_job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
