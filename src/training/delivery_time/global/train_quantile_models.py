"""
Delivery Time Model Training Script (XGBoost + sklearn)

This script trains delivery time models using traditional ML approaches:
- XGBoost with quantile regression objective
- sklearn QuantileRegressor (linear quantile regression)

QUANTILE REGRESSION APPROACHES:
1. ORIGINAL SCRIPT (train_delivery_time_model.py):
   - Deep Learning: Single TimeQuantileModel outputs all quantiles simultaneously
   - Random Forest: Single RandomForestQuantileRegressor (from quantile_forest package) outputs all quantiles simultaneously
   
2. THIS SCRIPT (train_delivery_time_xgboost.py):
   - XGBoost: Separate model for each quantile (limitation of XGBoost API)
   - sklearn: Separate QuantileRegressor for each quantile (limitation of sklearn API)

PLOT NAMING:
- Global plots: prediction_interval_global_{model_type}{mode_suffix}.png
- Per-DC plots: prediction_interval_dc_{dc}_{model_type}{mode_suffix}.png
- No conflicts with existing plots (which use 'dl' or 'qrf' suffixes)
"""

import sys
import os

# Add the project root to Python path so src module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import src.config as cfg

# Import common utilities
from src.training.delivery_time.common import (
    pinball_loss_np as _pinball_loss_np,
    plot_loss_curve,
    plot_sorted_interval,
    plot_time_series_interval,
    prepare_categorical_encoders,
    load_split_data
)

from src.training.delivery_time._train_xgboost import train_xgboost_quantile
from src.training.delivery_time._train_sklearn import train_sklearn_quantile

def predict_quantiles(models, X, quantiles):
    """Make predictions for all quantiles."""
    predictions = np.zeros((len(X), len(quantiles)))
    
    for i, tau in enumerate(quantiles):
        predictions[:, i] = models[tau].predict(X)
    
    return predictions

def main(args):
    """Main function to train and evaluate delivery time models using XGBoost and sklearn."""
    print("--- Training and Evaluating Delivery Time Models (XGBoost + sklearn) ---")
    cfg.DELIVERY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DELIVERY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mode_suffix = "_with_proxy" if args.train_on_proxy else ""
    
    # Load data splits using common utility
    df_train, df_eval = load_split_data(train_on_proxy=args.train_on_proxy, use_cs_data=False)

    # --- 1. Data Preparation ---
    print("Preparing data for global model...")
    
    features = cfg.DELIVERY_TIME_FEATURES

    X_train_df = df_train[features]
    y_train_df = df_train[cfg.DELIVERY_TIME_TARGET]
    
    valid_indices_train = y_train_df.dropna().index
    X_train_df = X_train_df.loc[valid_indices_train]
    y_train_df = y_train_df.loc[valid_indices_train]

    print(f"  - Found {len(X_train_df)} training samples with {len(features)} features.")

    if X_train_df.empty:
        print("Skipping training due to no training data after cleaning.")
        return

    # Prepare evaluation data
    X_eval = df_eval[features]
    y_eval = df_eval[cfg.DELIVERY_TIME_TARGET]
    valid_indices_eval = y_eval.dropna().index
    X_eval = X_eval.loc[valid_indices_eval]
    y_eval = y_eval.loc[valid_indices_eval]

    # Prepare categorical encoders
    print("Preparing categorical encoders...")
    categorical_encoders, vocab_sizes = prepare_categorical_encoders(X_train_df)

    # Separate numerical and categorical features (following DL script exactly)
    X_numerical = X_train_df[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
    X_categorical = X_train_df[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
    
    # Fit preprocessor for numerical features
    x_scaler = StandardScaler()
    x_scaler.fit(X_numerical)
    X_numerical_scaled = x_scaler.transform(X_numerical)

    # Encode categorical features
    X_categorical_encoded = {}
    for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
        if cat_feature in X_categorical.columns:
            encoded = categorical_encoders[cat_feature].transform(X_categorical[cat_feature].astype(str))
            X_categorical_encoded[cat_feature] = encoded
    
    # Helper to stack numerical matrix and categorical vector
    def _concat(num_mat, cat_dict):
        cat_cols = [v.reshape(-1, 1) for v in cat_dict.values()]
        return np.concatenate([num_mat] + cat_cols, axis=1)

    # Data Split for validation (following DL script exactly)
    split_idx = int(len(X_numerical_scaled) * (1 - args.split_ratio))
    
    # Split numerical features
    X_numerical_train, X_numerical_val = X_numerical_scaled[:split_idx], X_numerical_scaled[split_idx:]
    
    # Split categorical features
    X_categorical_train, X_categorical_val = {}, {}
    for cat_feature, encoded in X_categorical_encoded.items():
        X_categorical_train[cat_feature] = encoded[:split_idx]
        X_categorical_val[cat_feature] = encoded[split_idx:]
    
    # Split target
    y_train, y_val = y_train_df.iloc[:split_idx], y_train_df.iloc[split_idx:]
    
    # Prepare evaluation data (following DL script exactly)
    if not X_eval.empty:
        # Separate numerical and categorical features for evaluation
        X_eval_numerical = X_eval[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
        X_eval_categorical = X_eval[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
        
        # Scale numerical features
        X_eval_numerical_scaled = x_scaler.transform(X_eval_numerical)
        
        # Encode categorical features
        X_eval_categorical_encoded = {}
        for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
            if cat_feature in X_eval_categorical.columns:
                encoded = categorical_encoders[cat_feature].transform(X_eval_categorical[cat_feature].astype(str))
                X_eval_categorical_encoded[cat_feature] = encoded
        
        # Combine numerical and categorical features for evaluation
        X_eval_processed = _concat(X_eval_numerical_scaled, X_eval_categorical_encoded)
    else:
        X_eval_processed = np.array([])

    # Convert to numpy arrays for validation (following DL script exactly)
    X_train_np = _concat(X_numerical_train, X_categorical_train)
    y_train_np = y_train.values
    X_val_np = _concat(X_numerical_val, X_categorical_val)
    y_val_np = y_val.values
    
    # Prepare full dataset for final training (following DL script exactly)
    X_full_np = _concat(X_numerical_scaled, X_categorical_encoded)
    y_full_np = y_train_df.values
    X_eval_np = X_eval_processed
    y_eval_np = y_eval.values

    # Handle single quantile training if specified
    if args.quantile is not None:
        # Use numpy.isclose() for robust floating-point comparison
        found_quantile = None
        for q in cfg.DELIVERY_TIME_QUANTILES:
            if np.isclose(args.quantile, q, rtol=1e-10, atol=1e-10):
                found_quantile = q
                break
        
        if found_quantile is None:
            print(f"Error: Quantile {args.quantile} not found in config quantiles")
            print(f"Available quantiles: {cfg.DELIVERY_TIME_QUANTILES}")
            return
        
        quantiles = [found_quantile]
        print(f"Training only quantile: {found_quantile}")
    else:
        quantiles = cfg.DELIVERY_TIME_QUANTILES

    # --- 2. Model Training ---
    print("Training global models...")
    
    if args.model_type == 'xgboost':
        models, train_losses, val_losses = train_xgboost_quantile(X_train_np, y_train_np, X_val_np, y_val_np, X_full_np, y_full_np, quantiles)
        if len(quantiles) == 1:
            # Single quantile - include quantile in filename
            quantile_str = f"{int(quantiles[0] * 100)}"
            model_name = f"delivery_model_global_xgboost_{quantile_str}{mode_suffix}"
        else:
            # Multiple quantiles - use original name
            model_name = f"delivery_model_global_xgboost{mode_suffix}"
    elif args.model_type == 'sklearn':
        models, train_losses, val_losses = train_sklearn_quantile(X_train_np, y_train_np, X_val_np, y_val_np, X_full_np, y_full_np, quantiles)
        if len(quantiles) == 1:
            # Single quantile - include quantile in filename
            quantile_str = f"{int(quantiles[0] * 100)}"
            model_name = f"delivery_model_global_sklearn_{quantile_str}{mode_suffix}"
        else:
            # Multiple quantiles - use original name
            model_name = f"delivery_model_global_sklearn{mode_suffix}"
    else:
        print(f"Unknown model type: {args.model_type}")
        return

    # Save models
    model_path = cfg.DELIVERY_MODELS_DIR / f"{model_name}.joblib"
    joblib.dump({
        'models': models,
        'x_scaler': x_scaler,
        'categorical_encoders': categorical_encoders,
        'quantiles': quantiles
    }, model_path)
    print(f"Models saved to {model_path}")

    # Plot loss curves
    plot_loss_curve(train_losses, val_losses, cfg.DELIVERY_PLOTS_DIR / f"loss_curve_global_{args.model_type}{mode_suffix}.png")

    # --- 3. Evaluation ---
    print("Evaluating global model and generating plots...")
    
    if X_eval_np.size == 0:
        print("No evaluation data. Skipping evaluation plot.")
    else:
        # Make predictions
        predictions = predict_quantiles(models, X_eval_np, quantiles)
        
        # Create separate DataFrames for different plot types
        
        # Handle single quantile vs multiple quantiles
        if len(quantiles) == 1:
            # Single quantile training - create simple prediction plot
            day_plot_df = pd.DataFrame({
                'y_true': y_eval_np,
                'y_pred': predictions[:, 0],
                'dc_ori': df_eval.loc[valid_indices_eval, 'dc_ori'],
                'order_time': df_eval.loc[valid_indices_eval, 'order_time']
            })
            
            # For time-series plots - convert to hours
            y_true_hours = y_eval_np * 24
            hour_plot_df = pd.DataFrame({
                'y_true': y_true_hours,
                'y_pred': predictions[:, 0] * 24,
                'dc_ori': df_eval.loc[valid_indices_eval, 'dc_ori'],
                'order_time': df_eval.loc[valid_indices_eval, 'order_time']
            })
            
            print(f"Single quantile training ({quantiles[0]}): Skipping complex interval plots")
            skip_complex_plots = True
        else:
            # Multiple quantiles training - create full interval plots
            day_plot_df = pd.DataFrame({
                'y_true': y_eval_np,
                'y_pred_low': predictions[:, 0],
                'y_pred_upp': predictions[:, -1],
                'y_pred_5': predictions[:, 0],    # 5th percentile (90% interval)
                'y_pred_25': predictions[:, 4],   # 25th percentile (50% interval)
                'y_pred_50': predictions[:, 9],   # 50th percentile (median)
                'y_pred_75': predictions[:, 14],  # 75th percentile (50% interval)
                'y_pred_95': predictions[:, 18],  # 95th percentile (90% interval)
                'dc_ori': df_eval.loc[valid_indices_eval, 'dc_ori'],
                'order_time': df_eval.loc[valid_indices_eval, 'order_time']
            })
            
            # For time-series plots - convert to hours
            y_true_hours = y_eval_np * 24
            hour_plot_df = pd.DataFrame({
                'y_true': y_true_hours,
                'y_pred_low': predictions[:, 0] * 24,
                'y_pred_upp': predictions[:, -1] * 24,
                'y_pred_5': predictions[:, 0] * 24,    # 5th percentile (90% interval)
                'y_pred_25': predictions[:, 4] * 24,   # 25th percentile (50% interval)
                'y_pred_50': predictions[:, 9] * 24,   # 50th percentile (median)
                'y_pred_75': predictions[:, 14] * 24,  # 75th percentile (50% interval)
                'y_pred_95': predictions[:, 18] * 24,  # 95th percentile (90% interval)
                'dc_ori': df_eval.loc[valid_indices_eval, 'dc_ori'],
                'order_time': df_eval.loc[valid_indices_eval, 'order_time']
            })
            skip_complex_plots = False

        if not skip_complex_plots:
            # Global sorted interval plot
            fig, ax = plt.subplots(figsize=(15, 8))
            plot_sorted_interval(day_plot_df, ax, title=f"Global Model ({args.model_type.upper()}): Prediction Intervals vs. Actuals (Sorted Full Eval Set)")
            plot_path = cfg.DELIVERY_PLOTS_DIR / f"prediction_interval_global_{args.model_type}{mode_suffix}.png"
            plt.savefig(plot_path); plt.close()
            print(f"Global sorted interval plot saved to {plot_path}")

            # Global time-series plot
            fig, ax = plt.subplots(figsize=(35, 8))
            plot_time_series_interval(hour_plot_df, ax, title=f"Global Model ({args.model_type.upper()}): Prediction Intervals vs. Time of Day")
            plot_path = cfg.DELIVERY_PLOTS_DIR / f"time_series_interval_global_{args.model_type}{mode_suffix}.png"
            plt.savefig(plot_path); plt.close()
            print(f"Global time-series plot saved to {plot_path}")

            # Per-DC plots (both sorted and time-series)
            for dc in df_eval['dc_ori'].unique():
                # Sorted interval plot
                df_dc_eval = day_plot_df[day_plot_df['dc_ori'] == dc]
                if not df_dc_eval.empty:
                    fig, ax = plt.subplots(figsize=(15, 8))
                    plot_sorted_interval(df_dc_eval, ax, title=f"DC {dc}: Global Model ({args.model_type.upper()}) Predictions")
                    plot_path = cfg.DELIVERY_PLOTS_DIR / f"prediction_interval_dc_{dc}_{args.model_type}{mode_suffix}.png"
                    plt.savefig(plot_path); plt.close()

                # Time-series plot
                df_dc_eval = hour_plot_df[hour_plot_df['dc_ori'] == dc]
                if not df_dc_eval.empty:
                    fig, ax = plt.subplots(figsize=(35, 8))
                    plot_time_series_interval(df_dc_eval, ax, title=f"DC {dc} ({args.model_type.upper()})")
                    plot_path = cfg.DELIVERY_PLOTS_DIR / f"time_series_interval_dc_{dc}_{args.model_type}{mode_suffix}.png"
                    plt.savefig(plot_path); plt.close()
        else:
            # For single quantile, just save the model and skip complex plotting
            print(f"Single quantile training completed. Model saved for quantile {quantiles[0]}")

    print(f"\n--- Global {args.model_type} model has been trained and evaluated. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train delivery time models using XGBoost and sklearn quantile regression")
    parser.add_argument('--model_type', type=str, default='xgboost', choices=['xgboost', 'sklearn'], 
                        help="Model type to train (xgboost or sklearn)")
    parser.add_argument('--train_on_proxy', action='store_true', help="Train on forecast+proxy data, evaluate on test data.")
    parser.add_argument('--split_ratio', type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument('--quantile', type=float, help="Train only this specific quantile (e.g., 0.05, 0.50, 0.95)")
    args = parser.parse_args()
    main(args) 