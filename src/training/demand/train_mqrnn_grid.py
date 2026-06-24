#!/usr/bin/env python3
"""
Train MQRNN model with specific hyperparameters from grid search.
This script is called by SLURM array jobs, one job per hyperparameter combination.
"""
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import src.config as cfg
from src.model.mqrnn import MQRNN
from src.utils import pinball_loss

# Workaround: train_mqrnn.py calls parse_args() at module level.
# Save original argv and set a minimal one before importing to prevent argument conflicts.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]  # Minimal argv: just script name

# Now import from train_mqrnn (this will trigger its parse_args(), but with minimal args)
from src.training.demand.train_mqrnn import (
    load_npz, time_split, fit_scalers, apply_scalers, 
    epoch_loop, train_model
)

# Restore original argv for our own argument parsing
sys.argv = _original_argv

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Debug: Print sys.argv before parsing
    print("=" * 80, flush=True)
    print(f"DEBUG: sys.argv before parsing = {sys.argv}", flush=True)
    print(f"DEBUG: --train_on_proxy in sys.argv = {'--train_on_proxy' in sys.argv}", flush=True)
    print("=" * 80, flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--combination_idx", type=int, required=True,
                       help="Index of hyperparameter combination to use")
    parser.add_argument("--grid_file", type=Path, 
                       default=Path("data/demand_grid_search_combinations.json"),
                       help="Path to grid search combinations JSON file")
    parser.add_argument("--models_dir", type=Path, default=cfg.DEMAND_MODELS_DIR / "tune",
                       help="Directory to save models")
    parser.add_argument("--train_on_proxy", action="store_true",
                       help="Train on forecast+proxy data, evaluate on test data")
    parser.add_argument("--split_ratio", type=float, 
                       default=cfg.DEMAND_MODEL_VALIDATION_SPLIT_RATIO)
    
    args = parser.parse_args()
    
    # Debug: Print train_on_proxy after parsing
    print("=" * 80, flush=True)
    print(f"DEBUG: train_on_proxy after parsing = {args.train_on_proxy}", flush=True)
    print("=" * 80, flush=True)
    
    args.models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load grid combinations
    with open(args.grid_file, 'r') as f:
        grid_data = json.load(f)
    
    combinations = grid_data["combinations"]
    if args.combination_idx >= len(combinations):
        raise ValueError(f"Combination index {args.combination_idx} out of range "
                        f"(max: {len(combinations)-1})")
    
    params = combinations[args.combination_idx]
    
    # Print formatted hyperparameters
    print("=" * 80, flush=True)
    print(f"Training with combination {args.combination_idx}/{len(combinations)-1}", flush=True)
    print(f"train_on_proxy flag: {args.train_on_proxy}", flush=True)
    print("Hyperparameters:", flush=True)
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    
    # Load data
    root = cfg.PROCESSED_DATA_DIR
    forecast_data = load_npz(root / "mqrnn_forecast_train.npz")
    proxy_data = load_npz(root / "mqrnn_proxy_train.npz")
    test_data = load_npz(root / "mqrnn_test.npz")
    
    # Critical debug
    print("=" * 80, flush=True)
    print(f"CRITICAL DEBUG: args.train_on_proxy = {args.train_on_proxy} (type: {type(args.train_on_proxy)})", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    
    if args.train_on_proxy:
        print("Training on forecast + proxy data, evaluating on test data.", flush=True)
        train_raw = {k: np.concatenate([forecast_data[k], proxy_data[k]]) 
                     for k in forecast_data}
        train_raw, val_raw = time_split(train_raw, args.split_ratio)
        test_raw = test_data
    else:
        print("Training on forecast data, evaluating on proxy + test data.", flush=True)
        train_raw = forecast_data
        train_raw, val_raw = time_split(train_raw, args.split_ratio)
        test_raw = {k: np.concatenate([proxy_data[k], test_data[k]]) 
                   for k in proxy_data}
    
    # Fit scalers and prepare datasets
    sc_h, sc_p = fit_scalers(train_raw)
    train_set = apply_scalers(train_raw, sc_h, sc_p)
    val_set = apply_scalers(val_raw, sc_h, sc_p)
    
    # Print dataset sizes
    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set) if val_set is not None else 'None'}", flush=True)
    
    # Meta information
    full_raw = {k: np.concatenate([train_raw[k], val_raw[k], test_raw[k]]) 
               for k in train_raw}
    meta = {
        "num_cal": 4,
        "num_ord": len(cfg.DEMAND_ORDER_FEATURES),
        "num_skus": int(full_raw["sku_idx"].max()) + 1,
        "num_brands": int(full_raw["brand_idx"].max()) + 1,
        "Lh": cfg.DEMAND_MODEL_LOOKBACK,
        "Lp": cfg.DEMAND_FORECAST_HORIZON,
    }
    
    # Quantiles
    taus = torch.tensor(cfg.DEMAND_MODEL_QUANTILES, device=DEV).view(1, 1, -1)
    
    # Rename early_stopping_patience to early_stop for train_model compatibility
    train_params = params.copy()
    if "early_stopping_patience" in train_params:
        train_params["early_stop"] = train_params.pop("early_stopping_patience")
    
    # Train model
    train_mode = "_with_proxy" if args.train_on_proxy else ""
    model_filename = f"mqrnn_model{train_mode}_grid_{args.combination_idx:03d}.pt"
    model_path = args.models_dir / model_filename
    
    print(f"\nTraining model...", flush=True)
    best_val_loss = train_model(
        train_params, train_set, val_set, meta, taus,
        save_path=model_path, return_losses=False
    )
    
    # Evaluate on test set
    test_set = apply_scalers(test_raw, sc_h, sc_p)
    test_dl = DataLoader(test_set, batch_size=params["batch_size"] * 4,
                        shuffle=False, num_workers=0)
    
    # Load best model
    checkpoint = torch.load(model_path, weights_only=False)
    model = MQRNN(
        num_cal=meta["num_cal"], num_ord=meta["num_ord"],
        num_skus=meta["num_skus"], num_brands=meta["num_brands"],
        sku_emb=params.get("sku_emb", cfg.DEMAND_MODEL_SKU_EMBEDDING_DIM),
        brand_emb=params.get("brand_emb", cfg.DEMAND_MODEL_BRAND_EMBEDDING_DIM),
        hidden=params["hidden_dim"],
        ctx=params.get("context_dim", cfg.DEMAND_MODEL_CONTEXT_DIM),
        num_q=len(cfg.DEMAND_MODEL_QUANTILES),
        Lp=meta["Lp"], Lh=meta["Lh"],
        layers=params["lstm_n_layers"],
        dropout=params["dropout_p"] > 0,
        dropout_p=params["dropout_p"],
        bidirectional=params.get("bidirectional", cfg.DEMAND_MODEL_BIDIRECTIONAL)
    ).to(DEV)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Evaluate on test set
    test_loss = epoch_loop(model, test_dl, taus)
    
    # Save scalers
    train_mode = "_with_proxy" if args.train_on_proxy else ""
    scalers_filename = f"mqrnn_scalers{train_mode}_grid_{args.combination_idx:03d}.pt"
    scalers_path = args.models_dir / scalers_filename
    torch.save({"scaler_hist": sc_h, "scaler_pred": sc_p}, scalers_path)
    
    # Save results
    results = {
        "combination_idx": args.combination_idx,
        "hyperparameters": params,
        "validation_loss": float(best_val_loss) if best_val_loss else None,
        "test_loss": float(test_loss),
        "model_path": str(model_path),
        "scalers_path": str(scalers_path)
    }
    
    results_suffix = "_with_proxy" if args.train_on_proxy else ""
    results_file = args.models_dir / f"grid_results{results_suffix}_{args.combination_idx:03d}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Validation loss: {best_val_loss:.6f}" if best_val_loss else "N/A")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Model saved to {model_path}")
    print(f"Scalers saved to {scalers_path}")

if __name__ == "__main__":
    main()

