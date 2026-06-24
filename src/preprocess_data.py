import pandas as pd
import numpy as np
import os

import src.config as cfg

# --- Configuration ---
DATA_DIR = cfg.FULFILLMENT_DATA_DIR
OUTPUT_DIR = cfg.PROCESSED_DATA_DIR

# --- File Paths ---
order_file = os.path.join(DATA_DIR, 'JD_order_data.csv')
delivery_file = os.path.join(DATA_DIR, 'JD_delivery_data.csv')
sku_file = os.path.join(DATA_DIR, 'JD_sku_data.csv')
user_file = os.path.join(DATA_DIR, 'JD_user_data.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Data ---
print("Loading data...")
orders_df = pd.read_csv(order_file)
delivery_df = pd.read_csv(delivery_file)
sku_df = pd.read_csv(sku_file)
user_df = pd.read_csv(user_file)

# Rename 'promise' column to be more descriptive
orders_df.rename(columns={'promise': 'promise_delivery_days'}, inplace=True)

# Following your suggestion to clean the promise column by dropping nulls and converting to integer.
print("\n--- Cleaning Promise Delivery Days ---")
initial_order_count = orders_df['order_ID'].nunique()
print(f"Starting with {initial_order_count} unique orders before promise cleaning.")

# First, drop rows where the promise is missing entirely.
orders_df.dropna(subset=['promise_delivery_days'], inplace=True)
orders_after_initial_drop = orders_df['order_ID'].nunique()
print(f"Removed {initial_order_count - orders_after_initial_drop} orders with null promise values.")

# Convert to a numeric type, coercing errors will turn non-numeric text into NaN.
orders_df['promise_delivery_days'] = pd.to_numeric(orders_df['promise_delivery_days'], errors='coerce')

# Drop any rows that could not be converted.
orders_before_coerce_drop = orders_df['order_ID'].nunique()
orders_df.dropna(subset=['promise_delivery_days'], inplace=True)
orders_after_coerce_drop = orders_df['order_ID'].nunique()
print(f"Removed {orders_before_coerce_drop - orders_after_coerce_drop} orders with non-numeric promise values.")

# Now that the column is clean, convert it to an integer type.
orders_df['promise_delivery_days'] = orders_df['promise_delivery_days'].astype(int)
print(f"Finished promise cleaning. {orders_df['order_ID'].nunique()} orders remain.")

# --- 2. Merge Order, Delivery, and SKU Data ---
print("\nMerging datasets...")
# Step 2.1: Match orders with delivery info, dropping orders with missing shipments
# Both orders and delivery have a 'type' column. We specify suffixes to control the naming.
data = pd.merge(orders_df, delivery_df, on='order_ID', how='inner', suffixes=('_order', ''))
print(f"Initial merged orders: {data['order_ID'].nunique()}")

# Step 2.2: Merge with SKU data to get SKU type
data = pd.merge(data, sku_df[['sku_ID', 'type', 'brand_ID', 'attribute1', 'attribute2']], on='sku_ID', how='left', suffixes=('_delivery', '_sku'))
data.drop(columns=['type_delivery'], inplace=True)
data.rename(columns={'type_sku': 'type'}, inplace=True)

print("Merge complete.")

# Step 2.3: Merge with User data
data = pd.merge(data, user_df, on='user_ID', how='left')
print("Merge complete.")

# Step 2.4: Remove duplicate order-SKU pairs
print("\nRemoving duplicate order_ID-sku_ID pairs...")
initial_rows = len(data)
data.drop_duplicates(subset=['order_ID', 'sku_ID'], keep='first', inplace=True)
print(f"Removed {initial_rows - len(data)} duplicate rows.")
print(f"Orders remaining after deduplication: {data['order_ID'].nunique()}")

# --- 3. Initial Filtering ---
print("Applying initial filters...")

# Drop SKUs without brand_ID
print(f"Dropping SKUs without brand_ID: {data[data['brand_ID'].isna()]['sku_ID'].nunique()}")
data = data[data['brand_ID'].notna()]
print(f"Orders remaining: {data['order_ID'].nunique()}")
print(f"SKUs remaining: {data['sku_ID'].nunique()}")

# Step 3.2: Remove single-item gift orders
order_item_counts = data.groupby('order_ID')['sku_ID'].transform('count')
is_gift = data['gift_item'] > 0
single_gift_orders = data[(order_item_counts == 1) & is_gift]['order_ID'].unique()
initial_orders = data['order_ID'].nunique()
data = data[~data['order_ID'].isin(single_gift_orders)]
print(f"Removed {len(single_gift_orders)} single-item gift orders. Orders remaining: {data['order_ID'].nunique()} (from {initial_orders})")

data['order_date'] = pd.to_datetime(data['order_date'])

# Step 3.4: Exclude orders with multiple packages
package_counts = data.groupby('order_ID')['package_ID'].nunique()
multi_package_orders = package_counts[package_counts > 1].index
initial_orders = data['order_ID'].nunique()
data = data[~data['order_ID'].isin(multi_package_orders)]
print(f"Removed {len(multi_package_orders)} multi-package orders. Orders remaining: {data['order_ID'].nunique()} (from {initial_orders})")

# --- 4. Clean Delivery Time ---
print("Cleaning delivery time data...")
# Calculate end-to-end delivery time: from order placement to final arrival
data['order_time'] = pd.to_datetime(data['order_time'])
data['arr_time'] = pd.to_datetime(data['arr_time'])
data['delivery_time_hours'] = (data['arr_time'] - data['order_time']).dt.total_seconds() / 3600

# Calculate DC-to-DC travel time: from shipping out to arrival at station
data['ship_out_time'] = pd.to_datetime(data['ship_out_time'])
data['arr_station_time'] = pd.to_datetime(data['arr_station_time'])
data['travel_time_hours'] = (data['arr_station_time'] - data['ship_out_time']).dt.total_seconds() / 3600

# Remove negative delivery/travel times
initial_rows = len(data)
data = data[(data['delivery_time_hours'] >= 0) & (data['travel_time_hours'] >= 0)]
print(f"Removed {initial_rows - len(data)} records with negative time durations.")

# --- 4.1. Clean Delivery Promise Deviations ---
print("Cleaning delivery promise deviations...")
# Calculate delivery deviation based on promised vs actual delivery times
# Convert delivery hours to days using the same-day = 1 day rule
data['delivery_time_days'] = np.ceil(data['delivery_time_hours'] / 24)
data['delivery_deviation_days'] = data['delivery_time_days'] - data['promise_delivery_days']

# Remove orders with extreme delivery deviations that violate the 5-day maximum constraint
# This means: actual delivery time should not exceed 5 days
initial_rows = len(data)
data = data[data['delivery_time_days'] <= 5]
print(f"Removed {initial_rows - len(data)} records with delivery times exceeding 5 days.")

# Additional cleaning: remove extreme early deliveries (more than 5 days early)
# and extreme late deliveries (more than 5 days late)
initial_rows = len(data)
data = data[(data['delivery_deviation_days'] >= -5) & (data['delivery_deviation_days'] <= 5)]
print(f"Removed {initial_rows - len(data)} records with extreme delivery deviations (>5 days early or late).")
print(f"Orders remaining: {data['order_ID'].nunique()}")

# --- 5. Filter Based on Network and SKU Availability ---
print("Filtering based on network data and SKU availability...")

# Step 5.1: Filter dc_des based on network data
network_df = pd.read_csv(os.path.join(DATA_DIR, 'JD_network_data.csv'))
valid_dcs = np.sort(network_df['dc_ID'].unique())
initial_orders = data['order_ID'].nunique()
print(f"Number of unique dc_des before filtering: {data['dc_des'].nunique()}")
data = data[data['dc_des'].isin(valid_dcs)]
print(f"Number of unique dc_des after filtering: {data['dc_des'].nunique()}")
print(f"Removed orders with dc_des not in network data. Orders remaining: {data['order_ID'].nunique()} (from {initial_orders})")

# --- 6. Final Data Summary ---
print("\n--- Final Data Summary ---")
unique_dcs_count = data['dc_ori'].nunique()
unique_skus_count = data['sku_ID'].nunique()
unique_orders_count = data['order_ID'].nunique()
unique_dc_dest_count = data['dc_des'].nunique()

print(f"Number of unique origin DCs: {unique_dcs_count}")
print(f"Number of unique destination DCs: {unique_dc_dest_count}")
print(f"Number of unique SKUs: {unique_skus_count}")
print(f"Number of unique orders: {unique_orders_count}")

print("\nNumber of observations per origin DC:")
print(data['dc_ori'].value_counts())

print("\n--- Daily Sales Volume per SKU Summary ---")
daily_sku_sales = data.groupby(['sku_ID', 'order_date'])['quantity'].sum()
print(daily_sku_sales.describe())

# --- 7. Save Processed Data ---
# Train/test split will be handled later. Saving the full preprocessed dataset.
processed_file = os.path.join(OUTPUT_DIR, 'preprocessed_data.csv')
print(f"\nSaving preprocessed data to {processed_file}...")
data.to_csv(processed_file, index=False)

print("\nPreprocessing complete.")
