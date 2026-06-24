"""
Common utilities for delivery time model training.

This module contains shared functions used across different delivery time
training approaches (global vs carrier-specific, DL vs XGBoost vs RF vs CatBoost).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from sklearn.preprocessing import LabelEncoder
import src.config as cfg
from src.utils import pinball_loss
from src.data_utils import build_dc_event_snapshot


def format_carrier_id_for_path(carrier_id) -> str:
    """
    Normalize carrier IDs for filesystem usage.
    
    Converts float-like IDs such as 6.0 to "6" so saved artifacts align with the
    integer IDs referenced across the simulator. Non-integer values fall back to
    their string representation, and NaNs are labeled explicitly.
    """
    try:
        # Handle NaN (cannot use math.isnan on non-floats reliably)
        if carrier_id != carrier_id:
            return "nan"
        value = float(carrier_id)
        if value.is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return str(carrier_id)


def _ensure_unknown_token(encoder: LabelEncoder) -> None:
    """Ensure the encoder includes the configured unknown token."""
    unk = cfg.DELIVERY_DL_UNKNOWN_TOKEN
    classes = encoder.classes_.astype(str)
    if unk not in classes:
        encoder.classes_ = np.concatenate([classes, np.array([unk], dtype=classes.dtype)])


# ═══════════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════════

def pinball_loss_np(pred: np.ndarray, target: np.ndarray, tau: float) -> float:
    """
    Compute pinball loss for numpy arrays.
    
    Wrapper around utils.pinball_loss that works with numpy arrays.
    
    Parameters
    ----------
    pred : np.ndarray
        1-D array of predictions (length N)
    target : np.ndarray
        1-D array of true values (length N)
    tau : float
        Quantile in (0,1)
        
    Returns
    -------
    float
        Mean pinball loss across all samples
    """
    pred_t = torch.tensor(pred, dtype=torch.float32).unsqueeze(-1)  # [N,1]
    target_t = torch.tensor(target, dtype=torch.float32)  # [N]
    tau_t = torch.tensor([tau], dtype=torch.float32)  # [1]
    return pinball_loss(pred_t, target_t, tau_t).item()


# ═══════════════════════════════════════════════════════════════════════════
# Feature Engineering (from delivery_time_feature_engineering.py)
# ═══════════════════════════════════════════════════════════════════════════

def create_delivery_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features for delivery time prediction model.
    
    Args:
        data: Raw order data (sorted by order_time)
        
    Returns:
        DataFrame with delivery time features added
    """
    print("Generating delivery time features...")
    
    # === Time Effect ===
    data['order_hour'] = data['order_time'].dt.hour
    data['weekday'] = data['order_time'].dt.dayofweek

    # === Order Effect ===
    data['num_skus_in_order'] = data.groupby('order_ID')['sku_ID'].transform('nunique')
    data['has_bundle_discount'] = (data['bundle_discount_per_unit'] > 0).astype(int)
    data['has_coupon_discount'] = (data['coupon_discount_per_unit'] > 0).astype(int)
    data['has_gift_item'] = data.groupby('order_ID')['gift_item'].transform('max').astype(int)
    data['total_quantity_in_order'] = data.groupby('order_ID')['quantity'].transform('sum')

    # Calculate discount rate, handling potential division by zero
    data['discount_rate'] = -1 *(data['final_unit_price'] - data['original_unit_price']) / data['original_unit_price']
    data['discount_rate'] = data['discount_rate'].replace([np.inf, -np.inf], 0)
    data['discount_rate'] = data['discount_rate'].fillna(0)
    data['avg_discount_rate_in_order'] = data.groupby('order_ID')['discount_rate'].transform('mean').round(4)

    # === DC Operations ===
    data = build_dc_event_snapshot(data)

    # Verify all required delivery time features are present
    missing_features = [feat for feat in cfg.DELIVERY_TIME_FEATURES if feat not in data.columns]
    if missing_features:
        print(f"Warning: Missing delivery time features: {missing_features}")
    else:
        print(f"✓ All {len(cfg.DELIVERY_TIME_FEATURES)} delivery time features are present")
    
    print("Delivery time feature generation complete.")
    return data


def split_data_by_time(data: pd.DataFrame):
    """
    Splits data into forecast/proxy/test sets based on configured dates.
    
    Args:
        data: Feature-engineered data
        
    Returns:
        Tuple of (forecast_train_df, proxy_train_df, test_df)
    """
    print("Splitting data into threefold datasets...")
    
    forecast_train_start = pd.to_datetime(cfg.FORECAST_TRAIN_START_DATE) 
    forecast_train_end = pd.to_datetime(cfg.FORECAST_TRAIN_END_DATE)
    proxy_train_end = pd.to_datetime(cfg.PROXY_TRAIN_END_DATE)

    # Filter by start date to ensure enough history for lookback features
    forecast_train_df = data[(data['order_date'] >= forecast_train_start) & 
                           (data['order_date'] <= forecast_train_end)].copy()
    proxy_train_df = data[(data['order_date'] > forecast_train_end) & 
                        (data['order_date'] <= proxy_train_end)].copy()
    test_df = data[data['order_date'] > proxy_train_end].copy()

    print(f"Forecast training set shape: {forecast_train_df.shape}")
    print(f"Proxy training set shape: {proxy_train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    return forecast_train_df, proxy_train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════
# Categorical Feature Encoding
# ═══════════════════════════════════════════════════════════════════════════

def prepare_categorical_encoders(df_train):
    """
    Prepare label encoders for categorical features.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe containing categorical features
        
    Returns
    -------
    encoders : dict
        Dictionary mapping feature names to fitted LabelEncoder objects
    vocab_sizes : dict
        Dictionary mapping feature names to vocabulary sizes
    """
    encoders = {}
    vocab_sizes = {}
    
    for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
        if cat_feature in df_train.columns:
            le = LabelEncoder()
            le.fit(df_train[cat_feature].astype(str))
            _ensure_unknown_token(le)
            encoders[cat_feature] = le
            vocab_sizes[cat_feature] = len(le.classes_)
            print(f"  - {cat_feature}: {vocab_sizes[cat_feature]} unique values")
    
    return encoders, vocab_sizes


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading and Splitting
# ═══════════════════════════════════════════════════════════════════════════

def load_split_data(train_on_proxy: bool = False, use_cs_data: bool = False):
    """
    Load raw data, engineer features, and split delivery time data.
    
    This function integrates feature engineering directly into the data loading
    pipeline, ensuring proper features are always created.
    
    Parameters
    ----------
    train_on_proxy : bool, default=False
        If True, train on forecast+proxy and evaluate on test.
        If False, train on forecast and evaluate on proxy+test.
    use_cs_data : bool, default=False
        If True, load preprocessed_data_cs.csv (with carrier service info).
        If False, load preprocessed_data.csv (standard).
        
    Returns
    -------
    df_train : pd.DataFrame
        Training data with engineered features
    df_eval : pd.DataFrame
        Evaluation data with engineered features
    """
    # Load raw preprocessed data
    if use_cs_data:
        print("Loading preprocessed_data_cs.csv (with carrier service info)...")
        data_path = 'data/processed/preprocessed_data_cs.csv'
    else:
        print(f"Loading {cfg.PREPROCESSED_PATH}...")
        data_path = cfg.PREPROCESSED_PATH
    
    data = pd.read_csv(data_path, parse_dates=['order_time', 'order_date'])
    
    # Sort data for time-based features
    print("Sorting data by order_time for rolling features...")
    data.sort_values(by='order_time', inplace=True)
    
    # Engineer features
    data = create_delivery_time_features(data)
    
    # Split by time
    df_forecast, df_proxy, df_test = split_data_by_time(data)
    
    # Combine according to mode
    if train_on_proxy:
        print("  Mode: Training on forecast + proxy, evaluating on test")
        df_train = pd.concat([df_forecast, df_proxy])
        df_eval = df_test
    else:
        print("  Mode: Training on forecast, evaluating on proxy + test")
        df_train = df_forecast
        df_eval = pd.concat([df_proxy, df_test])
    
    # Clean column names
    df_train.columns = [c.strip() for c in df_train.columns]
    df_eval.columns = [c.strip() for c in df_eval.columns]
    
    print(f"  Training samples: {len(df_train):,}")
    print(f"  Evaluation samples: {len(df_eval):,}")
    
    return df_train, df_eval




# ═══════════════════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_loss_curve(train_losses, val_losses, save_path, title="Model Loss Curve"):
    """
    Plot training and validation loss curves.
    
    Parameters
    ----------
    train_losses : list
        Training losses over epochs/quantiles
    val_losses : list
        Validation losses over epochs/quantiles
    save_path : str or Path
        Path to save the plot
    title : str, default="Model Loss Curve"
        Title for the plot
    """
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch/Quantile'); plt.ylabel('Pinball Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_sorted_interval(df, ax, title):
    """
    Plot sorted prediction intervals.
    
    Shows 90% prediction interval vs observed values, sorted by true value.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: y_true, y_pred_low, y_pred_upp
    ax : matplotlib.axes.Axes
        Axes object to plot on
    title : str
        Title for the plot
    """
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
    ax.set_ylabel("Delivery Time (Days)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title(title, fontsize=14)


def plot_time_series_interval(df, ax, title):
    """
    Plot prediction intervals vs order time.
    
    Shows 90% and 50% prediction intervals, median, and observations over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: order_time, y_true, y_pred_5, y_pred_25, 
        y_pred_50, y_pred_75, y_pred_95
    ax : matplotlib.axes.Axes
        Axes object to plot on
    title : str
        Title for the plot
    """
    # Sort by order_time for proper time series visualization
    df_sorted = df.sort_values(by=['order_time']).reset_index(drop=True)
    
    # 90% Prediction Interval (darker blue)
    ax.fill_between(df_sorted["order_time"],
                    df_sorted["y_pred_5"], 
                    df_sorted["y_pred_95"], 
                    alpha=0.2, color="steelblue", label="90% Prediction Interval")
    
    # 50% Prediction Interval (lighter blue)  
    ax.fill_between(df_sorted["order_time"],
                    df_sorted["y_pred_25"], 
                    df_sorted["y_pred_75"], 
                    alpha=0.5, color="#e0f2ff", label="50% Prediction Interval")
    
    # Median prediction (black line)
    ax.plot(df_sorted["order_time"], df_sorted["y_pred_50"], 
            color='black', linewidth=1.5, alpha=0.7, label="Median")
    
    # Observations (black dots)
    ax.plot(df_sorted["order_time"], df_sorted["y_true"], 
            'o', markersize=2, alpha=0.5, color='black', label="Observation")
    
    ax.set_xlabel("Order Time", fontsize=12)
    ax.set_ylabel("Delivery time (hour)", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    ax.legend(frameon=False, fontsize=10, loc="upper left")


def plot_distribution(classes, true_dist, pred_dist, title, save_path):
    """
    Plot comparison of true vs predicted distributions.
    
    Used for CatBoost simulator evaluation.
    
    Parameters
    ----------
    classes : array-like
        Class labels (delivery time days)
    true_dist : array-like
        True empirical distribution
    pred_dist : array-like
        Predicted distribution
    title : str
        Plot title
    save_path : str or Path
        Path to save the plot
    """
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(classes, true_dist, width, label='True Empirical Distribution', 
            color='#4169E1', alpha=0.8)
    plt.bar(classes + width, pred_dist, width, label='Predicted Distribution', 
            color='#B3A369', alpha=0.8)
    plt.xticks(classes + width/2, [str(int(c)) for c in classes], 
               rotation=45, fontsize=12)
    plt.xlabel('Delivery Time (Days)')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
