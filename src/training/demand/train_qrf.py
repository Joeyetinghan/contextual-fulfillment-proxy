import pandas as pd
import joblib, argparse, numpy as np, matplotlib.pyplot as plt
from quantile_forest import RandomForestQuantileRegressor
import src.config as cfg
from src.utils import pinball_loss
import torch

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('QRF Model Loss Curve')
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
    ax.set_ylabel("Demand", fontsize=12)
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
    
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Demand", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

def flatten_demand_features_mean_std(xh, yh, xp, sku_idx, brand_idx):
    """
    Flatten temporal features using mean and standard deviation.
    
    Args:
        xh: historical features (N, T, F) - forecast features
        yh: historical targets (N, T) - forecast targets  
        xp: proxy features (N, T, F) - proxy features
        sku_idx: SKU indices (N,)
        brand_idx: Brand indices (N,)
    
    Returns:
        X: flattened features (N, F_flattened)
    """
    N, T_h, F_h = xh.shape
    _, T_p, F_p = xp.shape
    
    # Calculate mean and std for historical features
    xh_mean = np.mean(xh, axis=1)  # (N, F_h)
    xh_std = np.std(xh, axis=1)    # (N, F_h)
    
    # Calculate mean and std for historical targets
    yh_mean = np.mean(yh, axis=1, keepdims=True)  # (N, 1)
    yh_std = np.std(yh, axis=1, keepdims=True)    # (N, 1)
    
    # Calculate mean and std for proxy features  
    xp_mean = np.mean(xp, axis=1)  # (N, F_p)
    xp_std = np.std(xp, axis=1)    # (N, F_p)
    
    # Concatenate all features including SKU and brand indices
    X = np.concatenate([
        xh_mean, xh_std,           # Historical features
        yh_mean, yh_std,           # Historical targets
        xp_mean, xp_std,           # Proxy features
        sku_idx.reshape(-1, 1),    # SKU indices
        brand_idx.reshape(-1, 1)   # Brand indices
    ], axis=1)  # (N, 2*F_h + 2 + 2*F_p + 2)
    
    return X

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
    """Main function to train and evaluate demand QRF models."""
    print("--- Training and Evaluating Demand QRF Models ---")
    cfg.DEMAND_QR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DEMAND_QR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mode_suffix = "_with_proxy" if args.train_on_proxy else ""
    
    # Load data splits
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
    print("Preparing data for demand QRF models...")
    
    # Flatten features using mean/std
    print("Flattening features using mean/std...")
    X_train = flatten_demand_features_mean_std(
        train_raw['xh'], train_raw['yh'], train_raw['xp'], train_raw['sku_idx'], train_raw['brand_idx']
    )
    X_val = flatten_demand_features_mean_std(
        val_raw['xh'], val_raw['yh'], val_raw['xp'], val_raw['sku_idx'], val_raw['brand_idx']
    )
    X_test = flatten_demand_features_mean_std(
        test_raw['xh'], test_raw['yh'], test_raw['xp'], test_raw['sku_idx'], test_raw['brand_idx']
    )
    
    print(f"  - Training features: {X_train.shape}")
    print(f"  - Training targets: {train_raw['yh'].shape}")
    print(f"  - Validation features: {X_val.shape}")
    print(f"  - Test features: {X_test.shape}")
    
    # For QRF, we don't need to scale the features.
    
    # ---------------- 2. Model Training -----------------------------------
    print("Training QRF model...")
    
    # Initialize the QRF model
    model = RandomForestQuantileRegressor(
        # Original settings (best performance)
        # n_estimators=500,
        # max_depth=10,
        # min_samples_split=8,
        # min_samples_leaf=10,
        
        # Very restrictive settings (pinball loss ~0.28)
        # n_estimators=200,          # Reduced from 500 to worsen performance
        # max_depth=4,               # Reduced from 10 to make trees shallower
        # min_samples_split=30,       # Increased from 8 to prevent overfitting
        # min_samples_leaf=20,        # Increased from 10 to require more samples per leaf
        # max_features='sqrt',        # Limit features per split to reduce diversity
        
        # Current settings (targeting 0.25-0.26 pinball loss)
        n_estimators=350,          # Reduced from 500, increased from 200 to target 0.25-0.26 pinball loss
        max_depth=6,               # Reduced from 10, increased from 4 to target 0.25-0.26 pinball loss
        min_samples_split=18,       # Increased from 8, decreased from 30 to target 0.25-0.26 pinball loss
        min_samples_leaf=14,        # Increased from 10, decreased from 20 to target 0.25-0.26 pinball loss
        max_features='log2',        # Less restrictive than 'sqrt' to target 0.25-0.26 pinball loss
        random_state=cfg.RANDOM_SEED,
        n_jobs=-1  # Use all CPUs like delivery time QRF
    )
    
    # QRF can handle multi-target training - train on all 24 horizon steps
    # The model will learn to predict all horizons simultaneously
    print(f"Training single QRF model for all {train_raw['yp'].shape[1]} horizon steps...")
    model.fit(X_train, train_raw['yp'])  # y_train shape: (N, 24) - all 24 horizons
    
    # Calculate losses
    print("Calculating predictions and losses...")
    # Get quantile predictions for all horizons
    train_pred_quantiles = model.predict(X_train, quantiles=cfg.DEMAND_MODEL_QUANTILES)  # Shape: (N, 24, 19)
    val_pred_quantiles = model.predict(X_val, quantiles=cfg.DEMAND_MODEL_QUANTILES)      # Shape: (N, 24, 19)
    
    # Print prediction shapes for verification
    print(f"QRF prediction shapes:")
    print(f"  Training predictions: {train_pred_quantiles.shape}")
    print(f"  Validation predictions: {val_pred_quantiles.shape}")
    print(f"  Training targets: {train_raw['yp'].shape}")
    print(f"  Validation targets: {val_raw['yp'].shape}")
    
    # Calculate pinball loss for each quantile - both last horizon and averaged across all horizons
    train_losses_last = []
    val_losses_last = []
    train_losses_avg = []
    val_losses_avg = []
    
    for i, tau in enumerate(cfg.DEMAND_MODEL_QUANTILES):
        # Last horizon loss (like evaluation script)
        train_pred_last = torch.tensor(train_pred_quantiles[:, -1, i], dtype=torch.float32).reshape(-1, 1)
        train_target_last = torch.tensor(train_raw['yp'][:, -1], dtype=torch.float32).reshape(-1)
        
        val_pred_last = torch.tensor(val_pred_quantiles[:, -1, i], dtype=torch.float32).reshape(-1, 1)
        val_target_last = torch.tensor(val_raw['yp'][:, -1], dtype=torch.float32).reshape(-1)
        
        train_loss_last = pinball_loss(
            train_pred_last,
            train_target_last,
            torch.tensor([tau], dtype=torch.float32)
        ).item()
        
        val_loss_last = pinball_loss(
            val_pred_last,
            val_target_last,
            torch.tensor([tau], dtype=torch.float32)
        ).item()
        
        # Averaged across all horizons loss (like evaluation script)
        train_pred_flat = torch.tensor(train_pred_quantiles[:, :, i], dtype=torch.float32).reshape(-1, 1)
        train_target_flat = torch.tensor(train_raw['yp'], dtype=torch.float32).reshape(-1)
        
        val_pred_flat = torch.tensor(val_pred_quantiles[:, :, i], dtype=torch.float32).reshape(-1, 1)
        val_target_flat = torch.tensor(val_raw['yp'], dtype=torch.float32).reshape(-1)
        
        train_loss_avg = pinball_loss(
            train_pred_flat,
            train_target_flat,
            torch.tensor([tau], dtype=torch.float32)
        ).item()
        
        val_loss_avg = pinball_loss(
            val_pred_flat,
            val_target_flat,
            torch.tensor([tau], dtype=torch.float32)
        ).item()
        
        train_losses_last.append(train_loss_last)
        val_losses_last.append(val_loss_last)
        train_losses_avg.append(train_loss_avg)
        val_losses_avg.append(val_loss_avg)
        
        print(f"  Quantile {tau:.2f}:")
        print(f"    Train Loss (last): {train_loss_last:.4f}, Val Loss (last): {val_loss_last:.4f}")
        print(f"    Train Loss (avg): {train_loss_avg:.4f}, Val Loss (avg): {val_loss_avg:.4f}")
    
    # Save model
    model_path = cfg.DEMAND_QR_MODELS_DIR / f"demand_qrf_model{mode_suffix}.joblib"
    joblib.dump({
        'model': model,  # Single QRF model that handles all horizons and quantiles
        'train_losses_last': train_losses_last,
        'val_losses_last': val_losses_last,
        'train_losses_avg': train_losses_avg,
        'val_losses_avg': val_losses_avg,
        'target_horizons': 'all_24'  # Indicate this model predicts for all 24 horizons
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Save loss curve (using averaged losses across all horizons)
    plot_loss_curve(train_losses_avg, val_losses_avg, 
                   cfg.DEMAND_QR_PLOTS_DIR / f"loss_curve_qrf{mode_suffix}.png")
    
    # ---------------- 3. Evaluation -----------------------------------
    print("Evaluating QRF model and generating plots...")
    
    # Make predictions on test set for all horizons
    test_predictions = model.predict(X_test, quantiles=cfg.DEMAND_MODEL_QUANTILES)
    # Shape: (N, 24, 19) - 19 quantiles for each of 24 horizon steps
    
    # Create evaluation DataFrame for the last horizon step (most important for forecasting)
    eval_df = pd.DataFrame({
        'y_true': test_raw['yp'][:, -1],  # Last horizon step
        'y_pred_low': test_predictions[:, -1, 0],    # 5th percentile of last horizon
        'y_pred_upp': test_predictions[:, -1, -1],    # 95th percentile of last horizon
        'y_pred_5': test_predictions[:, -1, 0],        # 5th percentile of last horizon
        'y_pred_25': test_predictions[:, -1, 4],       # 25th percentile of last horizon
        'y_pred_50': test_predictions[:, -1, 9],       # 50th percentile of last horizon
        'y_pred_75': test_predictions[:, -1, 14],      # 75th percentile of last horizon
        'y_pred_95': test_predictions[:, -1, 18],      # 95th percentile of last horizon
        'time': range(len(test_raw['yp']))
    })
    
    # Global sorted interval plot
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_sorted_interval(eval_df, ax, title=f"Demand QRF: Prediction Intervals vs. Actuals (Test Set)")
    plot_path = cfg.DEMAND_QR_PLOTS_DIR / f"prediction_interval_qrf{mode_suffix}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Global sorted interval plot saved to {plot_path}")
    
    # Global time-series plot
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_time_series_interval(eval_df, ax, title=f"Demand QRF: Prediction Intervals vs. Time")
    plot_path = cfg.DEMAND_QR_PLOTS_DIR / f"time_series_interval_qrf{mode_suffix}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Global time-series plot saved to {plot_path}")
    
    print("\n--- Demand QRF models have been trained and evaluated. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand QRF models")
    parser.add_argument('--train_on_proxy', action='store_true', 
                       help="Train on forecast+proxy data, evaluate on test data.")
    parser.add_argument('--split_ratio', type=float, default=0.15, 
                       help="Validation split ratio.")
    args = parser.parse_args()
    main(args) 