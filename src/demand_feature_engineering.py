# fe_mqrnn.py  – compact feature‑engineering for MQRNN
import numpy as np
import pandas as pd
import src.config as cfg
from tqdm import tqdm
from pathlib import Path
import json

def preprocess_features(df):
     # === Order Effect ===
    df['num_skus_in_order'] = df.groupby('order_ID')['sku_ID'].transform('nunique')
    df['has_bundle_discount'] = (df['bundle_discount_per_unit'] > 0).astype(int)
    df['has_coupon_discount'] = (df['coupon_discount_per_unit'] > 0).astype(int)
    df['has_gift_item'] = df.groupby('order_ID')['gift_item'].transform('max').astype(int)
    df['total_quantity_in_order'] = df.groupby('order_ID')['quantity'].transform('sum')

    # Calculate discount rate, handling potential division by zero
    df['discount_rate'] = -1 *(df['final_unit_price'] - df['original_unit_price']) / df['original_unit_price']
    df['discount_rate'] = df['discount_rate'].replace([np.inf, -np.inf], 0)
    df['discount_rate'] = df['discount_rate'].fillna(0)
    df['avg_discount_rate_in_order'] = df.groupby('order_ID')['discount_rate'].transform('mean').round(4)

    # Map gender
    df['gender'] = df['gender'].map({'M': 1, 'F': 0, 'U': -1})
    
    # Map age
    age_mapping = {'0-25': 0, '26-35': 1, '36-45': 2, '46-55': 3, '56+': 4, 'U': -1}
    df['age'] = df['age'].map(age_mapping)
    
    # Map marital_status
    df['marital_status'] = df['marital_status'].map({'S': 0, 'M': 1, 'U': -1})

    # Convert to a numeric type, coercing errors will turn non-numeric text into NaN.
    df['purchase_power'] = pd.to_numeric(df['purchase_power'], errors='coerce')
    df['attribute1'] = pd.to_numeric(df['attribute1'], errors='coerce')
    df['attribute2'] = pd.to_numeric(df['attribute2'], errors='coerce')
    df['education'] = pd.to_numeric(df['education'], errors='coerce')
    
    # Fill NaNs
    df = df.fillna(-1)

    return df


# ------------------------------------------------------------
# 1. CONFIG 
# ------------------------------------------------------------
OUT_DIR    = cfg.PROCESSED_DATA_DIR
OUT_DIR.mkdir(exist_ok=True)


TIME_FREQ  = cfg.DEMAND_MODEL_TIME_PERIOD_FREQ        # bucket width
HIST_LONG  = cfg.DEMAND_MODEL_LOOKBACK          # look‑back steps
PRED_LONG  = cfg.DEMAND_FORECAST_HORIZON          # forecast horizon
STRIDE     = 1
# *** calendar cut‑offs***
TRAIN_START = pd.Timestamp(cfg.FORECAST_TRAIN_START_DATE)
TRAIN_END = pd.Timestamp(cfg.FORECAST_TRAIN_END_DATE)   # window end <  A  => train
PROXY_TRAIN_END   = pd.Timestamp(cfg.PROXY_TRAIN_END_DATE)   # window end <  B  => val

ORDER_FEATS = cfg.DEMAND_ORDER_FEATURES
CATEG_FEATS = cfg.DEMAND_CATEGORICAL_ORDER_FEATURES

# ----------------------------  LOAD  ------------------------------
print("↳ Reading raw CSVs …")
df = pd.read_csv(cfg.PREPROCESSED_PATH , parse_dates=[cfg.DATE_COL]).sort_values(cfg.DATE_COL)
# Preprocess features
df = preprocess_features(df)
for c in CATEG_FEATS:
    df[c], _ = pd.factorize(df[c], sort=True)

# ---------- aggregation per‑SKU --------------------------
agg_dict = {cfg.DEMAND_TARGET_COL:"sum", **{c:"mean" for c in ORDER_FEATS}}
frames = []
for sku,g in tqdm(df.groupby(cfg.SKU_COL, sort=True), desc="Resample"):
    agg_df = (g.set_index(cfg.DATE_COL)
                .resample(TIME_FREQ).agg(agg_dict))
    agg_df = agg_df.fillna(0)
    agg_df[cfg.SKU_COL], agg_df[cfg.BRAND_COL] = sku, g[cfg.BRAND_COL].iat[0]
    frames.append(agg_df)
full = (pd.concat(frames).reset_index()
        .rename(columns={"index":cfg.DATE_COL})
        .sort_values([cfg.SKU_COL, cfg.DATE_COL]))

# ---------------------  CALENDAR NUMERICS -------------------------
t = full[cfg.DATE_COL]
cal_cols = ["hour_sin","hour_cos","dow_sin","dow_cos"]
full["hour_sin"], full["hour_cos"] = np.sin(2*np.pi*t.dt.hour/24), np.cos(2*np.pi*t.dt.hour/24)
full["dow_sin"],  full["dow_cos"]  = np.sin(2*np.pi*t.dt.dayofweek/7), np.cos(2*np.pi*t.dt.dayofweek/7)

# --------- ID label‑encoding for embeddings ---------------------
full["sku_idx"],   _ = pd.factorize(full[cfg.SKU_COL],   sort=True)
full["brand_idx"], _ = pd.factorize(full[cfg.BRAND_COL], sort=True)


# ---------- assemble window tensors -----------------------------
X_CAL = cal_cols
X_ORD = ORDER_FEATS
splits = { 'forecast_train': [], 'proxy_train': [], 'test': [] }
TOTAL  = HIST_LONG + PRED_LONG

def add_window(bucket, xh, yh, xp, yp, sku_id, brand_id, dt):
    bucket.append((xh, yh, xp, yp, sku_id, brand_id, dt))

for sku, g in tqdm(full.groupby(cfg.SKU_COL, sort=False), desc="Windows"):
    g = g.sort_values(cfg.DATE_COL).reset_index(drop=True)
    cal = g[X_CAL].values.astype("float32")
    ord_stats = g[X_ORD].values.astype("float32")
    qty = g[cfg.DEMAND_TARGET_COL].values.astype("float32")
    s, b = g["sku_idx"].iat[0], g["brand_idx"].iat[0]

    for pos in range(0, len(g) - TOTAL + 1, STRIDE):

        hist_start_ts  = g[cfg.DATE_COL].iloc[pos]             
        horiz_start_ts = g[cfg.DATE_COL].iloc[pos + HIST_LONG]
        end_ts         = g[cfg.DATE_COL].iloc[pos + TOTAL - 1]

        if end_ts.date() <= TRAIN_END.date():                    # forecast‑train
            bucket = 'forecast_train'
        elif horiz_start_ts.date() > TRAIN_END.date() and end_ts.date() <= PROXY_TRAIN_END.date():            # proxy‑train
            bucket = 'proxy_train'
        elif horiz_start_ts.date() > PROXY_TRAIN_END.date():                                      # test
            bucket = 'test'
        else:
            continue
        # ---- build window tensors ---------------------------------
        xh_num = np.hstack([cal[pos:pos+HIST_LONG],
                            ord_stats[pos:pos+HIST_LONG]])
        xp_num = cal[pos+HIST_LONG:pos+TOTAL]
        yh     = qty[pos:pos+HIST_LONG]
        yp     = qty[pos+HIST_LONG:pos+TOTAL]

        add_window(
            splits[bucket],
            xh_num, yh, xp_num, yp, s, b,
            np.datetime64(horiz_start_ts, 's')      # save horizon‑start
        )


# ---------------------  STACK & SAVE ------------------------------
def stack(bucket):
    if not bucket:
        return {k: np.array([]) for k in ['xh', 'yh', 'xp', 'yp', 'sku_idx', 'brand_idx', 'dt']}
    xh, yh, xp, yp, sku, brand, dt = zip(*bucket)
    return dict(xh=np.stack(xh), yh=np.stack(yh),
                xp=np.stack(xp), yp=np.stack(yp),
                sku_idx=np.array(sku), brand_idx=np.array(brand),
                dt=np.array(dt, dtype="datetime64[s]"))

np.savez(OUT_DIR/"mqrnn_forecast_train.npz", **stack(splits['forecast_train']))
np.savez(OUT_DIR/"mqrnn_proxy_train.npz",   **stack(splits['proxy_train']))
np.savez(OUT_DIR/"mqrnn_test.npz",  **stack(splits['test']))

print("✅ Done. Shapes:")
for k in ["forecast_train","proxy_train","test"]:
    data = np.load(OUT_DIR/f"mqrnn_{k}.npz")
    print(f"  {k:5s}", data["xh"].shape, data["xp"].shape, data["yh"].shape, data["yp"].shape, 
          data["sku_idx"].shape, data["brand_idx"].shape, data["dt"].shape)
    
    # Preview a few data points
    if len(data["xh"]) > 0:
        print(f"    {k} preview:")
        print(f"      First xh sample (shape {data['xh'][0].shape}): {data['xh'][0][:5]}")  # First 5 values
        print(f"      First yh sample (shape {data['yh'][0].shape}): {data['yh'][0]}")
        print(f"      First xp sample (shape {data['xp'][0].shape}): {data['xp'][0][:5]}")  # First 5 values  
        print(f"      First yp sample (shape {data['yp'][0].shape}): {data['yp'][0]}")
        print(f"      First sku_idx: {data['sku_idx'][0]}, brand_idx: {data['brand_idx'][0]}")
        print(f"      First datetime: {data['dt'][0]}")
        print()
    else:
        print(f"    {k} is empty")
        print()
# ------------------------------------------------------------
# WRITE LOOK‑UP TABLES FOR INFERENCE
# ------------------------------------------------------------
mappings_dir = OUT_DIR / "mappings"
mappings_dir.mkdir(exist_ok=True)

# SKU mapping
sku_uniques = pd.Series(full[cfg.SKU_COL].unique()).sort_values()
sku_to_idx  = {str(sku): int(i) for i, sku in enumerate(sku_uniques)}
(Path(mappings_dir / "sku_to_idx.json")
     .write_text(json.dumps(sku_to_idx)))

# Brand mapping  (optional but recommended)
brand_uniques = pd.Series(full[cfg.BRAND_COL].unique()).sort_values()
brand_to_idx  = {str(b): int(i) for i, b in enumerate(brand_uniques)}
(Path(mappings_dir / "brand_to_idx.json")
     .write_text(json.dumps(brand_to_idx)))

print("✓ Saved SKU & brand lookup tables to", mappings_dir)