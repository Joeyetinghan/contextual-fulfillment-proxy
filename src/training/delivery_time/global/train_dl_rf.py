import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import src.config as cfg
import optuna

# Import common utilities
from src.training.delivery_time.common import (
    plot_loss_curve,
    plot_sorted_interval,
    plot_time_series_interval,
    prepare_categorical_encoders,
    load_split_data
)
from src.training.delivery_time._train_dl import train_dl_model
from src.training.delivery_time._train_qrf import train_rf_model

def main(args):
    """Main function to train and evaluate a global delivery time model."""
    print("--- Training and Evaluating Global Delivery Time Model ---")
    cfg.DELIVERY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DELIVERY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mode_suffix = "_with_proxy" if args.train_on_proxy else ""
    model_name = f"delivery_model_global{mode_suffix}"

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

    # Prepare evaluation data for both DL and QRF
    X_eval = df_eval[features]
    y_eval = df_eval[cfg.DELIVERY_TIME_TARGET]
    valid_indices_eval = y_eval.dropna().index
    X_eval = X_eval.loc[valid_indices_eval]
    y_eval = y_eval.loc[valid_indices_eval]

    # Prepare categorical encoders for DL model (will be None for QRF)
    categorical_encoders = None
    vocab_sizes = None
    if args.use_dl:
        print("Preparing categorical encoders...")
        categorical_encoders, vocab_sizes = prepare_categorical_encoders(X_train_df)


    # --- 2. Model Training ---
    print("Training global model...")
    if args.use_dl:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)       

        # Separate numerical and categorical features
        X_numerical = X_train_df[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
        X_categorical = X_train_df[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
        
        # Fit preprocessor for numerical features
        x_scaler = StandardScaler()
        x_scaler.fit(X_numerical)
        X_numerical_scaled = x_scaler.transform(X_numerical)
        X_numerical_tensor = torch.tensor(X_numerical_scaled, dtype=torch.float32)
        
        # Encode categorical features
        X_categorical_encoded = {}
        for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
            if cat_feature in X_categorical.columns:
                encoded = categorical_encoders[cat_feature].transform(X_categorical[cat_feature].astype(str))
                X_categorical_encoded[cat_feature] = torch.tensor(encoded, dtype=torch.long)
        
        y_all = torch.tensor(y_train_df.values, dtype=torch.float32)

        # Data Split for validation
        split_idx = int(len(X_numerical_tensor) * (1 - args.split_ratio))
        
        # Split numerical features
        X_numerical_train, X_numerical_val = X_numerical_tensor[:split_idx], X_numerical_tensor[split_idx:]
        
        # Split categorical features
        X_categorical_train, X_categorical_val = {}, {}
        for cat_feature, tensor in X_categorical_encoded.items():
            X_categorical_train[cat_feature] = tensor[:split_idx]
            X_categorical_val[cat_feature] = tensor[split_idx:]
        
        # Split target
        y_train, y_val = y_all[:split_idx], y_all[split_idx:]
        
        # Prepare evaluation data tensors for validation during final retraining
        # (X_eval, y_eval, valid_indices_eval already defined above)
        if not X_eval.empty:
            # Separate numerical and categorical features for evaluation
            X_eval_numerical = X_eval[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
            X_eval_categorical = X_eval[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
            
            # Scale numerical features
            X_eval_numerical_scaled = x_scaler.transform(X_eval_numerical)
            X_eval_numerical_tensor = torch.tensor(X_eval_numerical_scaled, dtype=torch.float32, device=device)
            
            # Encode categorical features
            X_eval_categorical_encoded = {}
            for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
                if cat_feature in X_eval_categorical.columns:
                    encoded = categorical_encoders[cat_feature].transform(X_eval_categorical[cat_feature].astype(str))
                    X_eval_categorical_encoded[cat_feature] = torch.tensor(encoded, dtype=torch.long, device=device)
            
            # Convert y_eval to tensor for validation during retraining
            y_eval_tensor = torch.tensor(y_eval.values, dtype=torch.float32, device=device)
        else:
            X_eval_numerical_tensor = None
            X_eval_categorical_encoded = None
            y_eval_tensor = None
        
        if args.n_trials > 0:
            def objective(trial):
                params = {
                    'lr': trial.suggest_float('lr', *cfg.DELIVERY_DL_HYPERPARAMETER_GRID['lr'], log=True),
                    'hidden_dim': trial.suggest_categorical('hidden_dim', cfg.DELIVERY_DL_HYPERPARAMETER_GRID['hidden_dim']),
                    'n_layers': trial.suggest_categorical('n_layers', cfg.DELIVERY_DL_HYPERPARAMETER_GRID['n_layers']),
                    'dropout_p': trial.suggest_float('dropout_p', *cfg.DELIVERY_DL_HYPERPARAMETER_GRID['dropout_p']),
                    'weight_decay': trial.suggest_float('weight_decay', *cfg.DELIVERY_DL_HYPERPARAMETER_GRID['weight_decay'], log=True),
                    'batch_size': trial.suggest_categorical('batch_size', cfg.DELIVERY_DL_HYPERPARAMETER_GRID['batch_size']),
                }
                val_loss = train_dl_model(
                    params, X_numerical_train, X_categorical_train, y_train, 
                    X_numerical_val, X_categorical_val, y_val, vocab_sizes, device, trial=trial
                )
                return val_loss

            study = optuna.create_study(
                direction='minimize', 
                sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
            )
            study.optimize(objective, n_trials=args.n_trials)
            best_params = study.best_params
            best_epoch = study.best_trial.user_attrs.get("best_epoch", args.epochs)
            print(f"Best hyperparameters for global model: {best_params}")
            print(f"Best epoch: {best_epoch}")

            print("Retraining final model on the full dataset...")
            best_params['epochs'] = best_epoch
            final_model, train_losses, val_losses = train_dl_model(
                best_params, X_numerical_tensor, X_categorical_encoded, y_all, 
                X_eval_numerical_tensor, X_eval_categorical_encoded, y_eval_tensor, vocab_sizes, device, return_model=True
            )
        else:
            best_params = {
                'lr': args.lr, 'hidden_dim': args.hidden_dim, 'n_layers': args.n_layers,
                'dropout_p': args.dropout_p, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size,
                'epochs': args.epochs, 'dc_ori_embedding_dim': cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM,
                'dc_des_embedding_dim': cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM
            }
            print("Training final model with default params...")
            final_model, train_losses, val_losses = train_dl_model(
                best_params, X_numerical_train, X_categorical_train, y_train, 
                X_numerical_val, X_categorical_val, y_val, vocab_sizes, device, return_model=True
            )
        
        # Save DL model and preprocessor
        model_path = cfg.DELIVERY_MODELS_DIR / f"{model_name}.pt"
        torch.save({
            'state_dict': final_model.state_dict(),
            'x_scaler': x_scaler,
            'categorical_encoders': categorical_encoders,
            'vocab_sizes': vocab_sizes,
            'numerical_dim': X_numerical_tensor.shape[1],
            'hidden_dim': best_params['hidden_dim'],
            'n_layers': best_params['n_layers'],
            'dropout': True, 'dropout_p': best_params['dropout_p'],
            'dc_ori_embedding_dim': cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM,
            'dc_des_embedding_dim': cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM
        }, model_path)
        print(f"Model saved to {model_path}")
        
        plot_loss_curve(train_losses, val_losses, cfg.DELIVERY_PLOTS_DIR / f"loss_curve_global_dl{mode_suffix}.png")
        model = final_model # For evaluation

    else:
        # --- Random-Forest Quantile Regressor ---
        # For QRF, we don't need to scale the features.
        # Split data for validation
        split_idx = int(len(X_train_df) * (1 - args.split_ratio))
        X_train_split = X_train_df.iloc[:split_idx]
        X_val_split = X_train_df.iloc[split_idx:]
        y_train_split = y_train_df.iloc[:split_idx].values
        y_val_split = y_train_df.iloc[split_idx:].values
        
        # Train RF model using centralized function
        model, train_loss, val_loss = train_rf_model(
            X_train_split.values, y_train_split,
            X_val_split.values, y_val_split,
            X_train_df.values, y_train_df.values,
            carrier_id=None  # Global model, no carrier_id
        )
        
        model_path = cfg.DELIVERY_MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Plot loss curve (single point for RF since it doesn't have epoch-by-epoch losses)
        # Create simple plot showing train and val loss
        plot_loss_curve([train_loss], [val_loss], 
                       cfg.DELIVERY_PLOTS_DIR / f"loss_curve_global_qrf{mode_suffix}.png",
                       title="RF Model Loss")

    # --- 3. Evaluation ---
    print("Evaluating global model and generating plots...")
    
    if X_eval.empty:
        print("No evaluation data. Skipping evaluation plot.")
    else:
        if args.use_dl:
            model.eval()
            with torch.no_grad():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Make predictions
                preds = model(
                    X_eval_numerical_tensor,
                    X_eval_categorical_encoded.get('dc_ori'),
                    X_eval_categorical_encoded.get('dc_des')
                ).cpu().numpy()
            predictions = preds
        else: # QRF Pipeline
            predictions = model.predict(X_eval, quantiles=cfg.DELIVERY_TIME_QUANTILES)
        
        # Create separate DataFrames for different plot types
        
        # For interval plots (sorted by y_true) - keep in days
        day_plot_df = pd.DataFrame({
            'y_true': y_eval.values,
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
        y_true_hours = y_eval.values * 24
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

        # Global sorted interval plot
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_sorted_interval(day_plot_df, ax, title=f"Global Model: Prediction Intervals vs. Actuals (Sorted Full Eval Set)")
        suffix = "dl" if args.use_dl else "qrf"
        plot_path = cfg.DELIVERY_PLOTS_DIR / f"prediction_interval_global_{suffix}{mode_suffix}.png"
        plt.savefig(plot_path); plt.close()
        print(f"Global sorted interval plot saved to {plot_path}")

        # Global time-series plot
        fig, ax = plt.subplots(figsize=(35, 8))
        plot_time_series_interval(hour_plot_df, ax, title=f"Global Model: Prediction Intervals vs. Time of Day")
        plot_path = cfg.DELIVERY_PLOTS_DIR / f"time_series_interval_global_{suffix}{mode_suffix}.png"
        plt.savefig(plot_path); plt.close()
        print(f"Global time-series plot saved to {plot_path}")

        # Per-DC plots (both sorted and time-series)
        for dc in df_eval['dc_ori'].unique():
            # Sorted interval plot
            df_dc_eval = day_plot_df[day_plot_df['dc_ori'] == dc]
            if not df_dc_eval.empty:
                fig, ax = plt.subplots(figsize=(15, 8))
                plot_sorted_interval(df_dc_eval, ax, title=f"DC {dc}: Global Model Predictions")
                plot_path = cfg.DELIVERY_PLOTS_DIR / f"prediction_interval_dc_{dc}_{suffix}{mode_suffix}.png"
                plt.savefig(plot_path); plt.close()

            # Time-series plot
            df_dc_eval = hour_plot_df[hour_plot_df['dc_ori'] == dc]
            if not df_dc_eval.empty:
                fig, ax = plt.subplots(figsize=(35, 8))
                plot_time_series_interval(df_dc_eval, ax, title=f"DC {dc}")
            plot_path = cfg.DELIVERY_PLOTS_DIR / f"time_series_interval_dc_{dc}_{suffix}{mode_suffix}.png"
            plt.savefig(plot_path); plt.close()

    print("\n--- Global model has been trained and evaluated. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a global delivery-time model (QRF or DL-MLP)")
    parser.add_argument('--use_dl', action='store_true', help="Train deep-learning MLP instead of QRF")
    parser.add_argument('--train_on_proxy', action='store_true', help="Train on forecast+proxy data, evaluate on test data.")
    parser.add_argument('--n_trials', type=int, default=0, help="Number of Optuna trials for hyperparameter tuning. If 0, uses default parameters.")
    parser.add_argument('--epochs', type=int, default=cfg.DELIVERY_DL_EPOCHS)
    parser.add_argument('--lr', type=float, default=cfg.DELIVERY_DL_LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=cfg.DELIVERY_DL_BATCH_SIZE)
    parser.add_argument('--hidden_dim', type=int, default=cfg.DELIVERY_DL_HIDDEN_DIM)
    parser.add_argument('--n_layers', type=int, default=cfg.DELIVERY_DL_N_LAYERS)
    parser.add_argument('--dropout', action='store_true', help="Enable dropout.")
    parser.add_argument('--no-dropout', dest='dropout', action='store_false', help="Disable dropout.")
    parser.set_defaults(dropout=cfg.DELIVERY_DL_DROPOUT)
    parser.add_argument('--dropout_p', type=float, default=cfg.DELIVERY_DL_DROPOUT_P)
    parser.add_argument('--weight_decay', type=float, default=cfg.DELIVERY_DL_WEIGHT_DECAY, help="L2 regularization strength.")
    parser.add_argument('--dc_ori_embedding_dim', type=int, default=cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM, help="Embedding dimension for dc_ori.")
    parser.add_argument('--dc_des_embedding_dim', type=int, default=cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM, help="Embedding dimension for dc_des.")
    parser.add_argument('--split_ratio', type=float, default=cfg.DELIVERY_DL_VALIDATION_SPLIT_RATIO, help="Validation split ratio.")
    args = parser.parse_args()
    main(args)
