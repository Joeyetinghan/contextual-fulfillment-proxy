"""
Train Delivery Time Quantile Models by Carrier Service

This script trains separate quantile regression models for each carrier service using
either XGBoost or scikit-learn QuantileRegressor.

It uses the preprocessed_data_cs.csv file which includes carrier_service_id_anon column.

The models are saved to data/models/delivery_time_cs/ directory with naming:
    delivery_model_{model_type}_{carrier_service_id}.joblib
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import joblib
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import src.config as cfg
from pathlib import Path

# Import common utilities
from src.training.delivery_time.common import (
    pinball_loss_np as _pinball_loss_np,
    plot_loss_curve,
    prepare_categorical_encoders,
    load_split_data,
    format_carrier_id_for_path,
)
from src.training.delivery_time._train_xgboost import train_xgboost_quantile
from src.training.delivery_time._train_sklearn import train_sklearn_quantile

def prepare_data(df_train, df_eval, categorical_encoders):
    """Prepare training and evaluation data with scaling and encoding."""
    features = cfg.DELIVERY_TIME_FEATURES
    
    # Prepare training data
    X_train_df = df_train[features]
    y_train_df = df_train[cfg.DELIVERY_TIME_TARGET]
    
    valid_indices_train = y_train_df.dropna().index
    X_train_df = X_train_df.loc[valid_indices_train]
    y_train_df = y_train_df.loc[valid_indices_train]
    
    if X_train_df.empty:
        return None, None, None, None, None, None, None, None, None
    
    # Prepare evaluation data
    X_eval = df_eval[features]
    y_eval = df_eval[cfg.DELIVERY_TIME_TARGET]
    valid_indices_eval = y_eval.dropna().index
    X_eval = X_eval.loc[valid_indices_eval]
    y_eval = y_eval.loc[valid_indices_eval]
    
    # Separate numerical and categorical features
    X_numerical = X_train_df[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
    X_categorical = X_train_df[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
    
    # Fit scaler for numerical features
    x_scaler = StandardScaler()
    x_scaler.fit(X_numerical)
    X_numerical_scaled = x_scaler.transform(X_numerical)
    
    # Encode categorical features
    X_categorical_encoded = {}
    for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
        if cat_feature in X_categorical.columns:
            encoded = categorical_encoders[cat_feature].transform(X_categorical[cat_feature].astype(str))
            X_categorical_encoded[cat_feature] = encoded
    
    # Helper to concatenate numerical and categorical
    def _concat(num_mat, cat_dict):
        cat_cols = [v.reshape(-1, 1) for v in cat_dict.values()]
        return np.concatenate([num_mat] + cat_cols, axis=1)
    
    # Data split for validation
    split_idx = int(len(X_numerical_scaled) * 0.8)
    
    X_numerical_train, X_numerical_val = X_numerical_scaled[:split_idx], X_numerical_scaled[split_idx:]
    
    X_categorical_train, X_categorical_val = {}, {}
    for cat_feature, encoded in X_categorical_encoded.items():
        X_categorical_train[cat_feature] = encoded[:split_idx]
        X_categorical_val[cat_feature] = encoded[split_idx:]
    
    y_train, y_val = y_train_df.iloc[:split_idx], y_train_df.iloc[split_idx:]
    
    # Prepare evaluation data
    X_eval_processed = None
    y_eval_np = None
    if not X_eval.empty:
        X_eval_numerical = X_eval[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
        X_eval_categorical = X_eval[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
        
        X_eval_numerical_scaled = x_scaler.transform(X_eval_numerical)
        
        X_eval_categorical_encoded = {}
        for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
            if cat_feature in X_eval_categorical.columns:
                encoded = categorical_encoders[cat_feature].transform(X_eval_categorical[cat_feature].astype(str))
                X_eval_categorical_encoded[cat_feature] = encoded
        
        X_eval_processed = _concat(X_eval_numerical_scaled, X_eval_categorical_encoded)
        y_eval_np = y_eval.values
    
    # Convert to numpy arrays
    X_train_np = _concat(X_numerical_train, X_categorical_train)
    y_train_np = y_train.values
    X_val_np = _concat(X_numerical_val, X_categorical_val)
    y_val_np = y_val.values
    
    X_full_np = _concat(X_numerical_scaled, X_categorical_encoded)
    y_full_np = y_train_df.values
    
    return X_train_np, y_train_np, X_val_np, y_val_np, X_full_np, y_full_np, X_eval_processed, y_eval_np, x_scaler

def main(args):
    """Main function to train carrier-specific delivery time models."""
    print("=" * 80)
    print(f"Training Carrier-Specific Delivery Time Models ({args.model_type.upper()})")
    print("=" * 80)
    
    # Create output directories
    models_dir = Path(cfg.DATA_DIR) / 'models' / 'delivery_time_cs'
    plots_dir = Path(cfg.DATA_DIR) / 'plots' / 'delivery_time_cs'
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    quantiles = cfg.DELIVERY_TIME_QUANTILES
    print(f"\nTraining {len(quantiles)} quantiles per carrier service")
    
    # Determine filename suffix
    file_suffix = "_with_proxy" if args.train_on_proxy else ""

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
        
        # Prepare data
        result = prepare_data(df_train_cs, df_eval_cs, categorical_encoders)
        if result[0] is None:
            print(f"  Skipping - no valid training data after cleaning")
            continue
        
        X_train_np, y_train_np, X_val_np, y_val_np, X_full_np, y_full_np, X_eval_np, y_eval_np, x_scaler = result
        
        # Train models
        if args.model_type == 'xgboost':
            models, train_losses, val_losses = train_xgboost_quantile(
                X_train_np, y_train_np, X_val_np, y_val_np, 
                X_full_np, y_full_np, quantiles, carrier_id
            )
        elif args.model_type == 'sklearn':
            models, train_losses, val_losses = train_sklearn_quantile(
                X_train_np, y_train_np, X_val_np, y_val_np, 
                X_full_np, y_full_np, quantiles
            )
        else:
            print(f"Unknown model type: {args.model_type}")
            continue
        
        # Save models
        model_path = models_dir / f"delivery_model_{args.model_type}_{carrier_id_str}{file_suffix}.joblib"
        joblib.dump({
            'models': models,
            'categorical_encoders': categorical_encoders,
            'x_scaler': x_scaler,
            'quantiles': quantiles,
            'carrier_service_id': carrier_id,
        }, model_path)
        print(f"  Saved model to {model_path}")
        
        # # Plot loss curves
        # plot_path = plots_dir / f"loss_curve_{args.model_type}_carrier_{carrier_id_str}.png"
        # plot_loss_curve(train_losses, val_losses, plot_path)
        # print(f"  Saved loss curve to {plot_path}")
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}")
    print(f"Models saved to: {models_dir}")
    print(f"Plots saved to: {plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train carrier-specific delivery time quantile models")
    parser.add_argument('--model_type', type=str, default='xgboost', choices=['xgboost', 'sklearn'],
                        help="Model type to train (xgboost or sklearn)")
    parser.add_argument('--train_on_proxy', action='store_true', 
                        help="Train on forecast+proxy data, evaluate on test data.")
    args = parser.parse_args()
    main(args)
