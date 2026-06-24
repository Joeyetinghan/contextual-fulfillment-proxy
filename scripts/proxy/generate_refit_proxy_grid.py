#!/usr/bin/env python3
"""
Generate a refit grid for selected proxy models.

Refit policy:
  - load hyperparameters from each selected checkpoint
  - train on full data (--refit_full_data, --split_ratio 0)
  - disable held-out final eval in refit run
  - set epochs to best_epoch_by_val_loss from final_best_eval.json (default)

The output grid can be run line-by-line with `src.training.proxy.train_proxy`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _read_model_list(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _resolve_checkpoint(token: str, models_root: Path) -> tuple[str, Path]:
    raw = Path(token)
    if raw.suffix == ".pt":
        ckpt = raw
        model_name = ckpt.parent.name
    elif raw.exists():
        if raw.is_dir():
            ckpt = raw / "best.pt"
            model_name = raw.name
        else:
            raise FileNotFoundError(f"Expected directory or .pt checkpoint, got file: {raw}")
    else:
        ckpt = models_root / token / "best.pt"
        model_name = token
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return model_name, ckpt


def _load_flat_hparams(ckpt_path: Path) -> dict[str, Any]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = blob.get("hyperparams", {}) if isinstance(blob, dict) else {}
    mp = blob.get("model_params", {}) if isinstance(blob, dict) else {}
    out: dict[str, Any] = {}

    for section in ("training", "loss", "inference", "architecture"):
        payload = hp.get(section, {})
        if isinstance(payload, dict):
            out.update(payload)

    # Fallbacks for older checkpoints missing architecture keys in hyperparams.
    key_map = {
        "architecture": "model_variant",
        "carrier_embedding_dim": "carrier_emb_dim",
        "option_projection_dim": "option_proj_dim",
    }
    if isinstance(mp, dict):
        for src_k, value in mp.items():
            dst_k = key_map.get(src_k, src_k)
            if dst_k not in out and isinstance(value, (int, float, bool, str)):
                out[dst_k] = value
    return out


def _load_best_epoch(model_dir: Path) -> int | None:
    path = model_dir / "final_best_eval.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("best_epoch_by_val_loss")
    if isinstance(value, int) and value > 0:
        return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full-data refit configs and grid for selected proxy models.")
    parser.add_argument("--model-name", action="append", default=[], help="Selected model name (repeatable).")
    parser.add_argument("--model-list", type=Path, default=None, help="Optional txt file with model names (one per line).")
    parser.add_argument("--models-root", type=Path, default=Path("data/models/proxy"))
    parser.add_argument("--config-out-dir", type=Path, default=Path("configs/proxy/paper/refits"))
    parser.add_argument("--grid-out", type=Path, default=Path("configs/proxy/paper/grids/proxy_refit_selected.txt"))
    parser.add_argument("--name-suffix", type=str, default="refitfull")
    parser.add_argument(
        "--epochs-source",
        choices=["best_epoch", "original_epochs"],
        default="best_epoch",
        help="Refit epoch count source.",
    )
    parser.add_argument("--epochs", type=int, default=0, help="If >0, override refit epochs for all models.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_tokens = list(args.model_name)
    if args.model_list is not None:
        if not args.model_list.exists():
            raise FileNotFoundError(f"Model list file not found: {args.model_list}")
        model_tokens.extend(_read_model_list(args.model_list))
    model_tokens = [m.strip() for m in model_tokens if str(m).strip()]
    if not model_tokens:
        raise ValueError("No models provided. Use --model-name and/or --model-list.")

    args.config_out_dir.mkdir(parents=True, exist_ok=True)
    grid_lines: list[str] = []
    refit_names: list[str] = []

    for token in model_tokens:
        base_name, ckpt = _resolve_checkpoint(token, args.models_root)
        base_cfg = _load_flat_hparams(ckpt)
        if not base_cfg:
            raise ValueError(f"Could not recover hyperparameters from checkpoint: {ckpt}")

        original_epochs = int(base_cfg.get("epochs", 500))
        if args.epochs > 0:
            refit_epochs = int(args.epochs)
        elif args.epochs_source == "best_epoch":
            refit_epochs = _load_best_epoch(ckpt.parent) or original_epochs
        else:
            refit_epochs = original_epochs

        refit_name = f"{base_name}_{args.name_suffix}"
        refit_names.append(refit_name)

        cfg = dict(base_cfg)
        cfg.update(
            {
                "model_name": refit_name,
                "refit_full_data": True,
                "split_ratio": 0.0,
                "epochs": refit_epochs,
                "early_stopping_patience": max(refit_epochs + 1, int(base_cfg.get("early_stopping_patience", 15))),
                "final_eval_on_best": False,
                "ub_eval_orders": 0,
                "final_ub_eval_orders": 0,
            }
        )

        cfg_path = args.config_out_dir / f"{refit_name}.json"
        if args.dry_run:
            print(f"DRY-RUN config: {cfg_path} (epochs={refit_epochs}, src={base_name})")
        else:
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        grid_lines.append(f"--config {cfg_path.as_posix()}")

    if args.dry_run:
        print("\nDRY-RUN grid lines:")
        for line in grid_lines:
            print(f"  {line}")
    else:
        args.grid_out.parent.mkdir(parents=True, exist_ok=True)
        args.grid_out.write_text("\n".join(grid_lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(grid_lines)} refit configs to {args.config_out_dir}")
        print(f"Wrote refit grid: {args.grid_out}")
        print("\nRun refit jobs with:")
        print(f"  while read -r line; do python -m src.training.proxy.train_proxy $line; done < {args.grid_out.as_posix()}")
        print("\nThen run proxy test simulation for refit models:")
        print("  MODELS=(")
        for name in refit_names:
            print(f'    "{name}"')
        print("  )")
        print("  for m in \"${MODELS[@]}\"; do")
        print("    PROXY_MODEL=\"data/models/proxy/$m/best.pt\" bash scripts/reproduce/run_policy_suite.sh")
        print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
