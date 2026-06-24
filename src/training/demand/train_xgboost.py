"""
Demand Model Training Script (XGBoost + sklearn) - Flexible Feature Processing

This script trains demand models using traditional ML approaches:
- XGBoost with quantile regression objective
- sklearn QuantileRegressor (linear quantile regression)

QUANTILE REGRESSION APPROACHES:
1. ORIGINAL SCRIPT (train_demand_model.py):
   - Deep Learning: MQRNN model with complex temporal structure
   
2. THIS SCRIPT (train_demand_xgb_meanstd.py):
   - XGBoost: Separate model for each quantile (limitation of XGBoost API)
   - sklearn: Separate QuantileRegressor for each quantile (limitation of sklearn API)

FEATURE PROCESSING OPTIONS:
- Mean/Std (--use_mean_std=True): Flatten features using statistical summaries (mean and std) across time dimensions
- Full Temporal (--use_mean_std=False): Flatten features by concatenating all time steps

DATA PROCESSING:
- Flattens the complex temporal structure (xh, yh, xp) into feature vectors
- Combines historical features, prediction features, and metadata (SKU, brand)
- Uses the last time step target (yp[:, -1]) as the prediction target

PLOT NAMING:
- Global plots: prediction_interval_global_{model_type}_{feature_suffix}{mode_suffix}.png
- Per-SKU plots: prediction_interval_sku_{sku}_{model_type}_{feature_suffix}{mode_suffix}.png
- Feature suffixes: "meanstd" or "fulltemporal"
- No conflicts with existing plots (which use 'mqrnn' suffix)
"""

import sys
import os

# Add the project root to Python path so src module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib, argparse, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import torch
import src.config as cfg
from src.utils import pinball_loss

import os

# ──────────────────── Scaling helpers (same as train_demand_model) ──────────

def fit_scalers(raw):
    """Fit StandardScalers on flattened historical and prediction feature blocks."""
    hist_flat = raw["xh"].reshape(-1, raw["xh"].shape[-1])
    pred_flat = raw["xp"].reshape(-1, raw["xp"].shape[-1])
    return StandardScaler().fit(hist_flat), StandardScaler().fit(pred_flat)


def apply_scalers(raw, sh, sp):
    """Apply previously fitted scalers to xh and xp blocks."""
    xh_scaled = sh.transform(raw["xh"].reshape(-1, raw["xh"].shape[-1])
                      ).reshape(raw["xh"].shape)
    xp_scaled = sp.transform(raw["xp"].reshape(-1, raw["xp"].shape[-1])
                      ).reshape(raw["xp"].shape)
    out = raw.copy()
    out["xh"], out["xp"] = xh_scaled, xp_scaled
    return out

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Global Model Loss Curve')
    plt.xlabel('Epoch'); plt.ylabel('Pinball Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_sorted_interval(df, ax, title):
    """Helper function to plot sorted prediction intervals."""
    sorted_predictions = df.sort_values(by=['y_true']).reset_index(drop=True)
    ax.fill_between(np.arange(len(sorted_predictions)),
                    sorted_predictions["y_pred_low"], 
                    sorted_predictions["y_pred_upp"], 
                    alpha=0.8, color="#e0f2ff", label="90% Prediction Interval")
    ax.plot(sorted_predictions["y_true"], 'o', markersize=3, label="Observed Value")
    ax.set_xticks([])
    ax.set_xlim([0, len(sorted_predictions)])
    ax.set_xlabel("Sorted Test Sample Index", fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel("Demand Quantity", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title(title, fontsize=14)

def plot_time_series_interval(df, ax, title):
    """Helper function to plot prediction intervals vs. time."""
    # Sort by time for proper time series visualization
    df_sorted = df.sort_values(by=['time']).reset_index(drop=True)
    
    # 90% Prediction Interval (darker blue)
    ax.fill_between(range(len(df_sorted)),
                    df_sorted["y_pred_5"], 
                    df_sorted["y_pred_95"], 
                    alpha=0.2, color="steelblue", label="90% Prediction Interval")
    
    # 50% Prediction Interval (lighter blue)  
    ax.fill_between(range(len(df_sorted)),
                    df_sorted["y_pred_25"], 
                    df_sorted["y_pred_75"], 
                    alpha=0.5, color="#e0f2ff", label="50% Prediction Interval")
    
    # Median prediction (black line)
    ax.plot(range(len(df_sorted)), df_sorted["y_pred_50"], 
            color='black', linewidth=1.5, alpha=0.7, label="Median")
    
    # Observations (black dots)
    ax.plot(range(len(df_sorted)), df_sorted["y_true"], 
            'o', markersize=2, alpha=0.5, color='black', label="Observation")
    
    ax.set_xlabel("Time Index", fontsize=12)
    ax.set_ylabel("Demand Quantity", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

def flatten_demand_features_mean_std(xh, yh, xp, sku_idx, brand_idx):
    """
    Flatten the complex temporal demand features into a feature matrix.
    Uses statistical summaries (mean and std) across time dimensions.
    
    Args:
        xh: Historical features (n_samples, lookback, n_features)
        yh: Historical targets (n_samples, lookback)
        xp: Prediction features (n_samples, forecast_horizon, n_features)
        sku_idx: SKU indices (n_samples,)
        brand_idx: Brand indices (n_samples,)
    
    Returns:
        X: Flattened feature matrix (n_samples, n_total_features)
    """
    n_samples = xh.shape[0]
    
    # Flatten historical features (mean and std across time)
    xh_mean = np.mean(xh, axis=1)  # (n_samples, n_features)
    xh_std = np.std(xh, axis=1)    # (n_samples, n_features)
    
    # Flatten historical targets (mean and std across time)
    yh_mean = np.mean(yh, axis=1, keepdims=True)  # (n_samples, 1)
    yh_std = np.std(yh, axis=1, keepdims=True)    # (n_samples, 1)
    
    # Flatten prediction features (mean and std across time)
    xp_mean = np.mean(xp, axis=1)  # (n_samples, n_features)
    xp_std = np.std(xp, axis=1)    # (n_samples, n_features)
    
    # Combine all features
    X = np.concatenate([
        xh_mean, xh_std,           # Historical features
        yh_mean, yh_std,           # Historical targets
        xp_mean, xp_std,           # Prediction features
        sku_idx.reshape(-1, 1),    # SKU indices
        brand_idx.reshape(-1, 1)   # Brand indices
    ], axis=1)
    
    return X

def flatten_demand_features_full_temporal(xh, yh, xp, sku_idx, brand_idx):
    """
    Flatten the complex temporal demand features into a feature matrix.
    Preserves all temporal information by concatenating all time steps.
    
    Args:
        xh: Historical features (n_samples, lookback, n_features)
        yh: Historical targets (n_samples, lookback)
        xp: Prediction features (n_samples, forecast_horizon, n_features)
        sku_idx: SKU indices (n_samples,)
        brand_idx: Brand indices (n_samples,)
    
    Returns:
        X: Flattened feature matrix (n_samples, n_total_features)
    """
    n_samples = xh.shape[0]
    
    # Flatten historical features by concatenating all time steps
    xh_flat = xh.reshape(n_samples, -1)  # (n_samples, lookback * n_features)
    
    # Flatten historical targets by concatenating all time steps
    yh_flat = yh.reshape(n_samples, -1)  # (n_samples, lookback)
    
    # Flatten prediction features by concatenating all time steps
    xp_flat = xp.reshape(n_samples, -1)  # (n_samples, forecast_horizon * n_features)
    
    # Combine all features
    X = np.concatenate([
        xh_flat,                    # Historical features (all time steps)
        yh_flat,                    # Historical targets (all time steps)
        xp_flat,                    # Prediction features (all time steps)
        sku_idx.reshape(-1, 1),     # SKU indices
        brand_idx.reshape(-1, 1)    # Brand indices
    ], axis=1)
    
    return X



def train_xgboost_quantile(X_train, y_train, X_val, y_val, quantiles):
    """Train XGBoost models for each quantile."""
    print("Training XGBoost quantile models...")
    
    models = {}
    train_losses = []
    val_losses = []
    
    for i, tau in enumerate(quantiles):
        print(f"  Training quantile {tau:.2f} ({i+1}/{len(quantiles)})")
        
        # Use all available CPU cores for XGBoost
        n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
        
        base_params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': tau,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': cfg.RANDOM_SEED,
            'n_jobs': n_jobs
        }
    
        model = MultiOutputRegressor(xgb.XGBRegressor(**base_params))
        model.fit(X_train, y_train)
        
        # Calculate losses
        val_pred = model.predict(X_val)
        train_pred = model.predict(X_train)
        
        # Print prediction shapes for verification
        print(f"    XGBoost quantile {tau:.2f} prediction shapes:")
        print(f"      Training predictions: {train_pred.shape}")
        print(f"      Validation predictions: {val_pred.shape}")
        print(f"      Training targets: {y_train.shape}")
        print(f"      Validation targets: {y_val.shape}")

        # For multi-output, we need to flatten predictions and targets for pinball loss
        val_pred_flat = val_pred.reshape(-1, 1)  # [N*L, 1]
        val_target_flat = y_val.reshape(-1)      # [N*L]
        train_pred_flat = train_pred.reshape(-1, 1)  # [N*L, 1]
        train_target_flat = y_train.reshape(-1)      # [N*L]

        val_loss = pinball_loss(torch.tensor(val_pred_flat, dtype=torch.float32),
                               torch.tensor(val_target_flat, dtype=torch.float32),
                               torch.tensor([tau], dtype=torch.float32)).item()

        train_loss = pinball_loss(torch.tensor(train_pred_flat, dtype=torch.float32),
                                 torch.tensor(train_target_flat, dtype=torch.float32),
                                 torch.tensor([tau], dtype=torch.float32)).item()
        
        models[tau] = model
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"  Completed quantile {tau:.2f}: Val Loss: {val_loss:.4f}, Train Loss: {train_loss:.4f}")
    
    return models, train_losses, val_losses

def train_sklearn_quantile(X_train, y_train, X_val, y_val, quantiles):
    """Train sklearn QuantileRegressor models for each quantile."""
    print("Training sklearn QuantileRegressor models...")
    
    models = {}
    train_losses = []
    val_losses = []
    
    for i, tau in enumerate(quantiles):
        print(f"  Training quantile {tau:.2f} ({i+1}/{len(quantiles)})")
        
        # Use available CPU cores (but limit to avoid memory issues)
        n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
        
        base = QuantileRegressor(quantile=tau, alpha=0.1, solver='highs')
        model = MultiOutputRegressor(base, n_jobs=n_jobs)
        
        # Fit model on training data
        print(f"    Fitting model on {len(X_train)} training samples...")
        model.fit(X_train, y_train)
        
        # Calculate losses
        print(f"    Calculating predictions and losses...")
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Print prediction shapes for verification
        print(f"    sklearn quantile {tau:.2f} prediction shapes:")
        print(f"      Training predictions: {train_pred.shape}")
        print(f"      Validation predictions: {val_pred.shape}")
        print(f"      Training targets: {y_train.shape}")
        print(f"      Validation targets: {y_val.shape}")
        
        # For multi-output, we need to flatten predictions and targets for pinball loss
        val_pred_flat = val_pred.reshape(-1, 1)  # [N*L, 1]
        val_target_flat = y_val.reshape(-1)      # [N*L]
        train_pred_flat = train_pred.reshape(-1, 1)  # [N*L, 1]
        train_target_flat = y_train.reshape(-1)      # [N*L]
        
        train_loss = pinball_loss(torch.tensor(train_pred_flat, dtype=torch.float32),
                                 torch.tensor(train_target_flat, dtype=torch.float32),
                                 torch.tensor([tau], dtype=torch.float32)).item()
        
        val_loss = pinball_loss(torch.tensor(val_pred_flat, dtype=torch.float32),
                               torch.tensor(val_target_flat, dtype=torch.float32),
                               torch.tensor([tau], dtype=torch.float32)).item()
        
        models[tau] = model
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"  Completed quantile {tau:.2f}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return models, train_losses, val_losses

def predict_quantiles(models, X, quantiles):
    """Make predictions for all quantiles."""
    n_samples = len(X)
    # Infer horizon length from first model
    sample_steps = models[quantiles[0]].predict(X[:1]).shape[1]
    preds = np.zeros((n_samples, sample_steps, len(quantiles)))
    
    for i, tau in enumerate(quantiles):
        preds[:, :, i] = models[tau].predict(X)   # (N , 24)
    
    return preds

def load_npz(path):
    """Load NPZ file and return dictionary."""
    arr = np.load(path)
    return {k: arr[k] for k in arr.files if k != 'order_id'}

def time_split(raw, split_ratio=0.15):
    """Split data by time."""
    n = len(raw['xh'])
    split_idx = int(n * (1 - split_ratio))
    
    train_raw = {k: v[:split_idx] for k, v in raw.items()}
    val_raw = {k: v[split_idx:] for k, v in raw.items()}
    
    return train_raw, val_raw

def main(args):
    """Main function to train and evaluate demand models using XGBoost and sklearn."""
    print("--- Training and Evaluating Demand Models (XGBoost + sklearn) ---")
    cfg.DEMAND_QR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DEMAND_QR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for meanstd models to match evaluation script expectations
    meanstd_dir = cfg.DEMAND_QR_MODELS_DIR / 'demand_qr_meanstd'
    meanstd_dir.mkdir(parents=True, exist_ok=True)
    mode_suffix = "_with_proxy" if args.train_on_proxy else ""
    
    # Load data splits - OPTIMIZED: Only load what we need
    print("Loading demand data...")
    if args.train_on_proxy:
        print("Training on forecast + proxy data, evaluating on test data.")
        forecast_data = load_npz(cfg.DEMAND_FORECAST_TRAIN_PATH)
        proxy_data = load_npz(cfg.DEMAND_PROXY_TRAIN_PATH)
        print(f"Forecast data shape: {forecast_data['xh'].shape}")
        print(f"Proxy data shape: {proxy_data['xh'].shape}")
        # Combine forecast and proxy data
        train_raw = {k: np.concatenate([forecast_data[k], proxy_data[k]]) for k in forecast_data}
        train_raw, val_raw = time_split(train_raw, args.split_ratio)
        
        # Load test data
        test_data = load_npz(cfg.DEMAND_TEST_PATH)
        test_raw = test_data
    else:
        print("Training on forecast data, evaluating on proxy + test data.")
        train_raw = load_npz(cfg.DEMAND_FORECAST_TRAIN_PATH)
        train_raw, val_raw = time_split(train_raw, args.split_ratio)
        
        # Load proxy and test data
        proxy_data = load_npz(cfg.DEMAND_PROXY_TRAIN_PATH)
        test_data = load_npz(cfg.DEMAND_TEST_PATH)
        test_raw = {k: np.concatenate([proxy_data[k], test_data[k]]) for k in proxy_data}

    # ---------------- 1. Data Preparation -----------------------------------
    print("Preparing data for demand models...")
    # Fit scalers on training split and scale all splits
    print("Fitting per-feature StandardScalers …")
    sc_hist, sc_pred = fit_scalers(train_raw)
    train_raw = apply_scalers(train_raw, sc_hist, sc_pred)
    val_raw   = apply_scalers(val_raw,   sc_hist, sc_pred)
    test_raw  = apply_scalers(test_raw,  sc_hist, sc_pred)

    # Flatten features for traditional ML models
    print("Flattening features …")
    if args.use_mean_std:
        flatten_func, feature_suffix = flatten_demand_features_mean_std, "meanstd"
    else:
        flatten_func, feature_suffix = flatten_demand_features_full_temporal, "fulltemporal"
    
    X_train = flatten_func(train_raw['xh'], train_raw['yh'], train_raw['xp'],
                           train_raw['sku_idx'], train_raw['brand_idx'])
    X_val   = flatten_func(val_raw['xh'], val_raw['yh'], val_raw['xp'],
                           val_raw['sku_idx'], val_raw['brand_idx'])
    
    # Targets are the full 24-step forecast horizon (shape: N × Lp)
    y_train = train_raw['yp']  # (N , 24)
    y_val   = val_raw['yp']    # (N , 24)

    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    print(f"  - Features: {X_train.shape[1]}")

    # Handle quantile training if specified
    if args.quantiles is not None:
        # Parse space-separated quantile string
        quantile_strs = args.quantiles.strip().split()
        quantiles = []
        for q_str in quantile_strs:
            try:
                q = float(q_str)
                # Use numpy.isclose() for robust floating-point comparison
                found_quantile = None
                for config_q in cfg.DEMAND_MODEL_QUANTILES:
                    if np.isclose(q, config_q, rtol=1e-10, atol=1e-10):
                        found_quantile = config_q
                        break
                
                if found_quantile is not None:
                    quantiles.append(found_quantile)
                else:
                    print(f"Warning: Quantile {q} not found in config, skipping")
            except ValueError:
                print(f"Warning: Invalid quantile value '{q_str}', skipping")
        
        if not quantiles:
            print("Error: No valid quantiles found")
            return
        
        print(f"Training quantiles: {quantiles}")
    elif args.quantile is not None:
        # Use numpy.isclose() for robust floating-point comparison
        found_quantile = None
        for q in cfg.DEMAND_MODEL_QUANTILES:
            if np.isclose(args.quantile, q, rtol=1e-10, atol=1e-10):
                found_quantile = q
                break
        
        if found_quantile is None:
            print(f"Error: Quantile {args.quantile} not found in config quantiles")
            return
        
        quantiles = [found_quantile]
        print(f"Training only quantile: {found_quantile}")
    else:
        # For array jobs, each job trains exactly one quantile
        # Get the job ID from environment variable if available
        job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1)) - 1
        
        # Map job ID to quantile index
        if job_id < len(cfg.DEMAND_MODEL_QUANTILES):
            quantiles = [cfg.DEMAND_MODEL_QUANTILES[job_id]]
            print(f"Job {job_id + 1}: Training quantile {cfg.DEMAND_MODEL_QUANTILES[job_id]}")
        else:
            print(f"Error: Job ID {job_id + 1} exceeds number of available quantiles")
            return

    # --- 2. Model Training ---
    print("Training demand models...")
    
    if args.model_type == 'xgboost':
        models, train_losses, val_losses = train_xgboost_quantile(X_train, y_train, X_val, y_val, quantiles)
        if len(quantiles) == 1:
            # Single quantile - include quantile in filename
            quantile_str = f"{int(quantiles[0] * 100)}"
            model_name = f"demand_model_global_xgboost_{feature_suffix}_{quantile_str}{mode_suffix}"
        else:
            # Multiple quantiles - use original name
            model_name = f"demand_model_global_xgboost_{feature_suffix}{mode_suffix}"
    elif args.model_type == 'sklearn':
        models, train_losses, val_losses = train_sklearn_quantile(X_train, y_train, X_val, y_val, quantiles)
        if len(quantiles) == 1:
            # Single quantile - include quantile in filename
            quantile_str = f"{int(quantiles[0] * 100)}"
            model_name = f"demand_model_global_sklearn_{feature_suffix}_{quantile_str}{mode_suffix}"
        else:
            # Multiple quantiles - use original name
            model_name = f"demand_model_global_sklearn_{feature_suffix}{mode_suffix}"
    else:
        print(f"Unknown model type: {args.model_type}")
        return

    # Save models and scalers in the meanstd subdirectory
    model_path = cfg.DEMAND_QR_MODELS_DIR / 'demand_qr_meanstd' / f"{model_name}.joblib"
    joblib.dump({
        'models': models,
        'hist_scaler': sc_hist,
        'pred_scaler': sc_pred,
        'quantiles': quantiles,
        'feature_suffix': feature_suffix
    }, model_path)
    print(f"Models saved to {model_path}")

    # Skip plotting for now to save memory
    # plot_loss_curve(train_losses, val_losses, cfg.DEMAND_QR_PLOTS_DIR / f"loss_curve_global_{args.model_type}_{feature_suffix}{mode_suffix}.png")

    # --- 3. Training Complete ---
    if len(quantiles) == 1:
        print(f"Single quantile training completed. Model saved for quantile {quantiles[0]}")
    print(f"\n--- Global {args.model_type} demand model has been trained. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand models using XGBoost and sklearn quantile regression")
    parser.add_argument('--model_type', type=str, default='xgboost', choices=['xgboost', 'sklearn'], 
                        help="Model type to train (xgboost or sklearn)")
    parser.add_argument('--train_on_proxy', action='store_true', help="Train on forecast+proxy data, evaluate on test data.")
    parser.add_argument('--split_ratio', type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument('--quantile', type=float, help="Train only this specific quantile (e.g., 0.05, 0.50, 0.95)")
    parser.add_argument('--quantiles', type=str, help="Train multiple quantiles (e.g., '0.05 0.10 0.15 0.20 0.25')")
    parser.add_argument('--use_mean_std', type=str, default='True', choices=['True', 'False'], 
                        help="Use mean/std feature flattening (True) or full temporal features (False). Default: True")
    args = parser.parse_args()
    # Convert string to boolean after parsing
    args.use_mean_std = args.use_mean_std == 'True'
    main(args) 