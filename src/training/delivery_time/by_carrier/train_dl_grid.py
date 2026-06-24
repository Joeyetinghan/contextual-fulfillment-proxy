#!/usr/bin/env python3
"""
Train delivery time DL model with specific hyperparameters from grid search.
This script is called by SLURM array jobs, one job per hyperparameter combination.
Trains models for all carriers with the same hyperparameter combination.
"""
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import src.config as cfg
from src.training.delivery_time.common import (
    prepare_categorical_encoders,
    load_split_data,
    format_carrier_id_for_path,
)
from src.training.delivery_time._train_dl import train_dl_model

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combination_idx", type=int, required=True,
                       help="Index of hyperparameter combination to use")
    parser.add_argument("--grid_file", type=Path, 
                       default=Path("data/delivery_dl_grid_search_combinations.json"),
                       help="Path to grid search combinations JSON file")
    parser.add_argument("--models_dir", type=Path, default=cfg.DELIVERY_MODELS_CS_DIR / "tune",
                       help="Directory to save models")
    parser.add_argument("--train_on_proxy", action="store_true",
                       help="Train on forecast+proxy data, evaluate on test data")
    
    args = parser.parse_args()
    
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
    print(f"Training with hyperparameter combination {args.combination_idx}", flush=True)
    print("=" * 80, flush=True)
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}", flush=True)
    print("=" * 80, flush=True)
    
    # Load data
    print("\nLoading data...", flush=True)
    df_train, df_eval = load_split_data(train_on_proxy=args.train_on_proxy, use_cs_data=True)
    
    # Get list of carrier services
    carrier_services = sorted(df_train['carrier_service_id_anon'].dropna().unique())
    print(f"Found {len(carrier_services)} carrier services: {carrier_services}")
    
    # Load cost models to ensure we train for all expected carriers
    cost_models_df = pd.read_csv('data/params/real_cost_models_cs.csv')
    expected_carriers = sorted(cost_models_df['carrier_service_id'].dropna().astype(int).unique())
    print(f"Expected carrier services from cost models: {expected_carriers}")
    
    # Filter to only train on carriers that exist in both
    carriers_to_train = sorted(set(carrier_services) & set(expected_carriers))
    if len(carriers_to_train) < len(expected_carriers):
        missing = set(expected_carriers) - set(carriers_to_train)
        print(f"  Warning: Missing carrier services in data: {missing}")
    
    print(f"\nWill train models for {len(carriers_to_train)} carrier services")
    
    # Prepare global categorical encoders (fit on all training data)
    print("\nPreparing categorical encoders...")
    categorical_encoders, vocab_sizes = prepare_categorical_encoders(df_train)
    for cat_feature, size in vocab_sizes.items():
        print(f"  - {cat_feature}: {size} unique values")
    
    # Determine mode suffix for model filenames
    mode_suffix = "_with_proxy" if args.train_on_proxy else ""
    
    # Train models for each carrier service
    for carrier_id in carriers_to_train:
        print(f"\n{'='*80}")
        print(f"Training DL model for Carrier Service {carrier_id} (combination {args.combination_idx})")
        print(f"{'='*80}")
        carrier_id_str = format_carrier_id_for_path(carrier_id)
        
        # Filter data for this carrier service
        df_train_cs = df_train[df_train['carrier_service_id_anon'] == carrier_id].copy()
        df_eval_cs = df_eval[df_eval['carrier_service_id_anon'] == carrier_id].copy()
        
        print(f"  Training samples: {len(df_train_cs):,}")
        print(f"  Evaluation samples: {len(df_eval_cs):,}")
        
        if len(df_train_cs) < 100:
            print(f"  Skipping - insufficient training data (< 100 samples)")
            continue
        
        # Prepare features
        features = cfg.DELIVERY_TIME_FEATURES
        X_train_df = df_train_cs[features]
        y_train_df = df_train_cs[cfg.DELIVERY_TIME_TARGET]
        
        valid_indices = y_train_df.dropna().index
        X_train_df = X_train_df.loc[valid_indices]
        y_train_df = y_train_df.loc[valid_indices]
        
        if X_train_df.empty:
            print(f"  Skipping - no valid training data after cleaning")
            continue
        
        # Separate numerical and categorical features
        X_numerical = X_train_df[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
        X_categorical = X_train_df[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
        
        # Scale numerical features
        x_scaler = StandardScaler()
        x_scaler.fit(X_numerical)
        X_numerical_scaled = x_scaler.transform(X_numerical)
        
        # Encode categorical features
        X_categorical_encoded = {}
        for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
            if cat_feature in X_categorical.columns:
                encoded = categorical_encoders[cat_feature].transform(X_categorical[cat_feature].astype(str))
                X_categorical_encoded[cat_feature] = encoded
        
        # Split for validation
        split_idx = int(len(X_numerical_scaled) * (1 - cfg.DELIVERY_DL_VALIDATION_SPLIT_RATIO))
        X_num_train, X_num_val = X_numerical_scaled[:split_idx], X_numerical_scaled[split_idx:]
        y_train, y_val = y_train_df.iloc[:split_idx].values, y_train_df.iloc[split_idx:].values
        
        X_cat_train, X_cat_val = {}, {}
        for cat_feature, encoded in X_categorical_encoded.items():
            X_cat_train[cat_feature] = encoded[:split_idx]
            X_cat_val[cat_feature] = encoded[split_idx:]
        
        # Convert to tensors
        X_num_train_t = torch.tensor(X_num_train, dtype=torch.float32)
        X_num_val_t = torch.tensor(X_num_val, dtype=torch.float32)
        X_cat_train_t = {k: torch.tensor(v, dtype=torch.long) for k, v in X_cat_train.items()}
        X_cat_val_t = {k: torch.tensor(v, dtype=torch.long) for k, v in X_cat_val.items()}
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        
        # Train DL model with grid hyperparameters
        try:
            dl_model, train_losses, val_losses = train_dl_model(
                params,
                X_num_train_t, X_cat_train_t, y_train_t,
                X_num_val_t, X_cat_val_t, y_val_t,
                vocab_sizes, DEV, return_model=True
            )

            # Build model_params in the same structure as non-grid training,
            # so evaluation code can treat tuned and non-tuned models uniformly.
            # Get embedding dimensions from params if provided, otherwise use config defaults
            dc_ori_emb_dim = params.get("dc_ori_embedding_dim", cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM)
            dc_des_emb_dim = params.get("dc_des_embedding_dim", cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM)
            
            model_params = {
                "numerical_dim": X_num_train.shape[1],
                "hidden_dim": params["hidden_dim"],
                "n_layers": params["n_layers"],
                "quantiles": cfg.DELIVERY_TIME_QUANTILES,
                "dropout": cfg.DELIVERY_DL_DROPOUT,
                "dropout_p": params["dropout_p"],
                "dc_ori_vocab_size": vocab_sizes.get("dc_ori"),
                "dc_des_vocab_size": vocab_sizes.get("dc_des"),
                "dc_ori_embedding_dim": dc_ori_emb_dim,
                "dc_des_embedding_dim": dc_des_emb_dim,
            }

            # Save DL model with combination index in filename
            model_path = args.models_dir / f"dl_model_{carrier_id_str}_combo_{args.combination_idx}{mode_suffix}.pt"
            torch.save(
                {
                    "model_state_dict": dl_model.state_dict(),
                    "x_scaler": x_scaler,
                    "categorical_encoders": categorical_encoders,
                    "vocab_sizes": vocab_sizes,
                    "quantiles": cfg.DELIVERY_TIME_QUANTILES,
                    "carrier_service_id": carrier_id,
                    "model_params": model_params,
                    # Keep full grid params & meta for reproducibility
                    "grid_params": params,
                    "combination_idx": args.combination_idx,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "best_val_loss": min(val_losses) if val_losses else None,
                },
                model_path,
            )
            print(f"  Saved model to {model_path}")
            print(f"  Best validation loss: {min(val_losses) if val_losses else 'N/A':.6f}")
        except Exception as e:
            print(f"  Error training DL model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

