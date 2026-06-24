"""
Train Delivery Time Models by Carrier Service (DL + Random Forest)

This script trains separate Deep Learning and/or Random Forest quantile regression 
models for each carrier service. It uses the preprocessed_data_cs.csv file which 
includes carrier_service_id_anon column.

The models are saved to data/models/delivery_time_cs/tune/ (cfg.DELIVERY_MODELS_CS_DIR)
with naming:
    - Deep Learning: dl_model_{carrier_service_id}{mode_suffix}.pt
    - Random Forest: rf_model_{carrier_service_id}{mode_suffix}.joblib
    where mode_suffix is "_with_proxy" if --train_on_proxy is used, otherwise ""
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import joblib
import argparse
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import StandardScaler
import src.config as cfg
from src.training.delivery_time.common import (
    plot_loss_curve,
    prepare_categorical_encoders,
    load_split_data,
    format_carrier_id_for_path,
)
from src.training.delivery_time._train_dl import train_dl_model
from src.training.delivery_time._train_qrf import train_rf_model


def main(args):
    """Main function to train carrier-specific DL/RF delivery time models."""
    print("=" * 80)
    print("Training Carrier-Specific Delivery Time Models (DL + Random Forest)")
    print("=" * 80)
    
    # Create output directories
    models_dir = cfg.DELIVERY_MODELS_CS_DIR
    plots_dir = Path(cfg.DATA_DIR) / 'plots' / 'delivery_time_cs'
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine mode suffix for model filenames
    mode_suffix = "_with_proxy" if args.train_on_proxy else ""
    
    # Load and engineer features for carrier-specific data
    # This will load preprocessed_data_cs.csv, engineer all features properly,
    # and split by time
    df_train, df_eval = load_split_data(train_on_proxy=args.train_on_proxy, use_cs_data=True)
    
    # Get list of carrier services
    carrier_services = sorted(df_train['carrier_service_id_anon'].dropna().unique())
    print(f"\nFound {len(carrier_services)} carrier services: {carrier_services}")
    
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
    
    # Setup device for DL
    device = None
    if args.use_dl:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
        print(f"\nUsing device: {device}")
    
    # Train models for each carrier service
    for carrier_id in carriers_to_train:
        print(f"\n{'='*80}")
        print(f"Training models for Carrier Service {carrier_id}")
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
        split_idx = int(len(X_numerical_scaled) * (1 - args.split_ratio))
        X_num_train, X_num_val = X_numerical_scaled[:split_idx], X_numerical_scaled[split_idx:]
        y_train, y_val = y_train_df.iloc[:split_idx].values, y_train_df.iloc[split_idx:].values
        
        X_cat_train, X_cat_val = {}, {}
        for cat_feature, encoded in X_categorical_encoded.items():
            X_cat_train[cat_feature] = encoded[:split_idx]
            X_cat_val[cat_feature] = encoded[split_idx:]
        
        # Train DL model if requested
        if args.use_dl:
            try:
                # Convert to tensors
                X_num_train_t = torch.tensor(X_num_train, dtype=torch.float32)
                X_num_val_t = torch.tensor(X_num_val, dtype=torch.float32)
                X_cat_train_t = {k: torch.tensor(v, dtype=torch.long) for k, v in X_cat_train.items()}
                X_cat_val_t = {k: torch.tensor(v, dtype=torch.long) for k, v in X_cat_val.items()}
                y_train_t = torch.tensor(y_train, dtype=torch.float32)
                y_val_t = torch.tensor(y_val, dtype=torch.float32)
                
                if args.n_trials > 0:
                    # Hyperparameter tuning with Optuna
                    import optuna
                    
                    def objective(trial):
                        params = {
                            'lr': trial.suggest_float('lr', *cfg.DELIVERY_DL_HYPERPARAMETER_GRID['lr'], log=True),
                            'hidden_dim': trial.suggest_categorical('hidden_dim', cfg.DELIVERY_DL_HYPERPARAMETER_GRID['hidden_dim']),
                            'n_layers': trial.suggest_categorical('n_layers', cfg.DELIVERY_DL_HYPERPARAMETER_GRID['n_layers']),
                            'dropout_p': trial.suggest_float('dropout_p', *cfg.DELIVERY_DL_HYPERPARAMETER_GRID['dropout_p']),
                            'weight_decay': trial.suggest_float('weight_decay', *cfg.DELIVERY_DL_HYPERPARAMETER_GRID['weight_decay'], log=True),
                            'batch_size': trial.suggest_categorical('batch_size', cfg.DELIVERY_DL_HYPERPARAMETER_GRID['batch_size']),
                            'epochs': args.epochs,
                        }
                        val_loss = train_dl_model(
                            params, X_num_train_t, X_cat_train_t, y_train_t,
                            X_num_val_t, X_cat_val_t, y_val_t,
                            vocab_sizes, device, trial=trial
                        )
                        return val_loss
                    
                    print(f"  Running Optuna hyperparameter tuning for carrier {carrier_id} ({args.n_trials} trials)...")
                    study = optuna.create_study(
                        direction='minimize',
                        sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
                    )
                    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
                    
                    best_params = study.best_params
                    best_epoch = study.best_trial.user_attrs.get('best_epoch', args.epochs)
                    best_params['epochs'] = best_epoch
                    print(f"  Best hyperparameters: {best_params}")
                    
                    # Train final model with best params
                    dl_model, train_losses, val_losses = train_dl_model(
                        best_params,
                        X_num_train_t, X_cat_train_t, y_train_t,
                        X_num_val_t, X_cat_val_t, y_val_t,
                        vocab_sizes, device, return_model=True
                    )
                    dl_params = best_params
                else:
                    # Use default or command-line parameters
                    dl_params = {
                        'lr': args.lr,
                        'hidden_dim': args.hidden_dim,
                        'n_layers': args.n_layers,
                        'dropout_p': args.dropout_p,
                        'weight_decay': args.weight_decay,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                    }
                    
                    dl_model, train_losses, val_losses = train_dl_model(
                        dl_params,
                        X_num_train_t, X_cat_train_t, y_train_t,
                        X_num_val_t, X_cat_val_t, y_val_t,
                        vocab_sizes, device, return_model=True
                    )
                
                # Save DL model
                model_path = models_dir / f"dl_model_{carrier_id_str}{mode_suffix}.pt"
                torch.save({
                    'model_state_dict': dl_model.state_dict(),
                    'x_scaler': x_scaler,
                    'categorical_encoders': categorical_encoders,
                    'vocab_sizes': vocab_sizes,
                    'quantiles': cfg.DELIVERY_TIME_QUANTILES,
                    'carrier_service_id': carrier_id,
                    'model_params': {
                        'numerical_dim': X_num_train.shape[1],
                        'hidden_dim': dl_params['hidden_dim'],
                        'n_layers': dl_params['n_layers'],
                        'quantiles': cfg.DELIVERY_TIME_QUANTILES,
                        'dropout': cfg.DELIVERY_DL_DROPOUT,
                        'dropout_p': dl_params['dropout_p'],
                        'dc_ori_vocab_size': vocab_sizes.get('dc_ori'),
                        'dc_des_vocab_size': vocab_sizes.get('dc_des'),
                        'dc_ori_embedding_dim': cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM,
                        'dc_des_embedding_dim': cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM
                    }
                }, model_path)
                print(f"  Saved DL model to {model_path}")
                
                # Plot loss curve
                plot_path = plots_dir / f"loss_curve_dl_carrier_{carrier_id_str}{mode_suffix}.png"
                plot_loss_curve(train_losses, val_losses, plot_path, 
                               title=f"DL Model Loss - Carrier {carrier_id}")
                
            except Exception as e:
                print(f"  Error training DL model: {e}")
        
        # Train RF model if requested
        if args.use_qrf:
            try:
                rf_model, train_loss, val_loss = train_rf_model(
                    X_num_train, y_train, X_num_val, y_val,
                    X_numerical_scaled, y_train_df.values, carrier_id
                )
                
                # Save RF model
                model_path = models_dir / f"rf_model_{carrier_id_str}{mode_suffix}.joblib"
                joblib.dump({
                    'model': rf_model,
                    'x_scaler': x_scaler,
                    'categorical_encoders': categorical_encoders,
                    'quantiles': cfg.DELIVERY_TIME_QUANTILES,
                    'carrier_service_id': carrier_id,
                }, model_path)
                print(f"  Saved RF model to {model_path}")
                
            except Exception as e:
                print(f"  Error training RF model: {e}")
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}")
    print(f"Models saved to: {models_dir}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train carrier-specific DL/RF delivery time models")
    parser.add_argument('--train_on_proxy', action='store_true',
                        help="Train on forecast+proxy data, evaluate on test data.")
    parser.add_argument('--use_dl', action='store_true',
                        help="Train Deep Learning models")
    parser.add_argument('--use_qrf', action='store_true',
                        help="Train Random Forest Quantile models")
    
    # DL hyperparameter arguments
    parser.add_argument('--n_trials', type=int, default=0, 
                        help="Number of Optuna trials for hyperparameter tuning. If 0, uses default parameters.")
    parser.add_argument('--epochs', type=int, default=cfg.DELIVERY_DL_EPOCHS)
    parser.add_argument('--lr', type=float, default=cfg.DELIVERY_DL_LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=cfg.DELIVERY_DL_BATCH_SIZE)
    parser.add_argument('--hidden_dim', type=int, default=cfg.DELIVERY_DL_HIDDEN_DIM)
    parser.add_argument('--n_layers', type=int, default=cfg.DELIVERY_DL_N_LAYERS)
    parser.add_argument('--dropout', action='store_true', help="Enable dropout.")
    parser.add_argument('--no-dropout', dest='dropout', action='store_false', help="Disable dropout.")
    parser.set_defaults(dropout=cfg.DELIVERY_DL_DROPOUT)
    parser.add_argument('--dropout_p', type=float, default=cfg.DELIVERY_DL_DROPOUT_P)
    parser.add_argument('--weight_decay', type=float, default=cfg.DELIVERY_DL_WEIGHT_DECAY, 
                        help="L2 regularization strength.")
    parser.add_argument('--dc_ori_embedding_dim', type=int, default=cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM, 
                        help="Embedding dimension for dc_ori.")
    parser.add_argument('--dc_des_embedding_dim', type=int, default=cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM, 
                        help="Embedding dimension for dc_des.")
    parser.add_argument('--split_ratio', type=float, default=cfg.DELIVERY_DL_VALIDATION_SPLIT_RATIO, 
                        help="Validation split ratio.")
    
    args = parser.parse_args()
    
    if not args.use_dl and not args.use_qrf:
        print("Error: Must specify at least one of --use_dl or --use_qrf")
        exit(1)
    
    main(args)
