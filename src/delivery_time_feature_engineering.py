# This script prepares features for the demand and delivery time forecasting models.

import pandas as pd
import numpy as np
import src.config as cfg
from src.data_utils import build_dc_event_snapshot

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

def verify_and_save_delivery_datasets(forecast_df, proxy_df, test_df):
    """
    Verifies target variable exists and saves delivery time datasets.
    
    Args:
        forecast_df, proxy_df, test_df: Split datasets
    """
    # Verify target variable
    print("\nVerifying target variable...")
    target_variable = cfg.DELIVERY_TIME_TARGET
    for name, df in [("Forecast Train", forecast_df), ("Proxy Train", proxy_df), ("Test", test_df)]:
        if target_variable not in df.columns:
            print(f"Error: Target variable '{target_variable}' not found in {name} dataset!")
        else:
            print(f"Success: Target variable '{target_variable}' is present in the {name} dataset.")

    # Save datasets
    print(f"\nSaving delivery time datasets...")
    forecast_df.to_csv(cfg.DELIVERY_FORECAST_TRAIN_PATH, index=False)
    proxy_df.to_csv(cfg.DELIVERY_PROXY_TRAIN_PATH, index=False)
    test_df.to_csv(cfg.DELIVERY_TEST_PATH, index=False)
    print(f"Saved to: {cfg.DELIVERY_FORECAST_TRAIN_PATH}, {cfg.DELIVERY_PROXY_TRAIN_PATH}, {cfg.DELIVERY_TEST_PATH}")


def main():
    """Main execution function."""
    print("=== FORECAST FEATURE ENGINEERING ===\n")
    
    # ---  DELIVERY TIME MODEL FEATURES ---
    print("DELIVERY TIME MODEL FEATURE ENGINEERING")
    
    # Load data
    print(f"Loading preprocessed data from {cfg.PREPROCESSED_PATH}...")
    data = pd.read_csv(cfg.PREPROCESSED_PATH, parse_dates=['order_time', 'order_date'])
    print("Data loaded.")
    
    # Sort data for time-based features
    print("Sorting data by order_time for rolling features...")
    data.sort_values(by='order_time', inplace=True)
    
    # Create delivery time features
    data = create_delivery_time_features(data)
    
    # Split data by time
    forecast_df, proxy_df, test_df = split_data_by_time(data)
    
    # Verify and save delivery time datasets
    verify_and_save_delivery_datasets(forecast_df, proxy_df, test_df)


if __name__ == "__main__":
    main()
