from pathlib import Path
import numpy as np
import pandas as pd

# --- Randomness ---
RANDOM_SEED = 42

# --- Data Splitting ---
FORECAST_TRAIN_START_DATE = '2018-03-02'
FORECAST_TRAIN_END_DATE = '2018-03-18'
PROXY_TRAIN_END_DATE = '2018-03-25'
# The test set starts the day after the proxy training set ends.

# --- Directory and File Paths ---
DATA_DIR = Path('data')
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
INSTANCE_OUTPUT_DIR = DATA_DIR / 'fulfillment_instances'
FULFILLMENT_DATA_DIR = DATA_DIR / 'fulfillment'

DELIVERY_FORECAST_TRAIN_PATH = PROCESSED_DATA_DIR / 'delivery_time_forecast_train.csv'
DELIVERY_PROXY_TRAIN_PATH = PROCESSED_DATA_DIR / 'delivery_time_proxy_train.csv'
DELIVERY_TEST_PATH = PROCESSED_DATA_DIR / 'delivery_time_test.csv'
PREPROCESSED_LEGACY_PATH = PROCESSED_DATA_DIR / 'preprocessed_data.csv'
PREPROCESSED_PATH = PROCESSED_DATA_DIR / 'preprocessed_data_cs.csv'
NETWORK_PATH = FULFILLMENT_DATA_DIR / 'JD_network_data.csv'
SKU_DATA_PATH = FULFILLMENT_DATA_DIR / 'JD_sku_data.csv'
USER_DATA_PATH = FULFILLMENT_DATA_DIR / 'JD_user_data.csv'
COMPATIBILITY_MATRIX_PATH = PROCESSED_DATA_DIR / 'compatibility_matrix.csv'
DC_CARRIER_ELIGIBILITY_PATH = DATA_DIR / 'params' / 'dc_carrier_eligibility.csv'
HUB_BASED_COORDS_PATH = DATA_DIR / 'derived' / 'pseudo_coords' / 'hub_based_coords.csv'
LIMITED_COVERAGE_PATH = DATA_DIR / 'params' / 'limited_coverage.json'
REAL_COST_MODELS_CS_PATH = DATA_DIR / 'params' / 'real_cost_models_cs.csv'

# Carrier services to exclude (insufficient data for model training)
EXCLUDED_CARRIER_SERVICES = [3]

# Demand Model Paths (for MQRNN models)
DEMAND_FORECAST_TRAIN_PATH = PROCESSED_DATA_DIR / 'mqrnn_forecast_train.npz'
DEMAND_PROXY_TRAIN_PATH = PROCESSED_DATA_DIR / 'mqrnn_proxy_train.npz'
DEMAND_TEST_PATH = PROCESSED_DATA_DIR / 'mqrnn_test.npz'

# Proxy Training Dataset Paths
PROXY_DATA_DIR = DATA_DIR / 'proxy_data'
PROXY_TRAINING_DATA_DIR = PROXY_DATA_DIR / 'proxy_train' # Default for training script
PROXY_TEST_DATA_DIR = PROXY_DATA_DIR / 'test'

# Preprocessing Parameters
MIN_DC_OBS = 100 # Minimum observations to keep an origin DC
MIN_SKU_OBS = 100 # Minimum number of orders required for an SKU in the training set

STATIC_FEATURES = [
    "order_hour", "weekday", "num_skus_in_order",
    "has_bundle_discount", "has_coupon_discount", "type", "has_gift_item",
    "total_quantity_in_order", "avg_discount_rate_in_order",
    "user_level", "city_level", "plus", "promise_delivery_days", "dc_des", 
    "gender", "age", "marital_status", "education", "purchase_power", "attribute1", "attribute2"
]

DYNAMIC_FEATURES = [
    "shipped_orders_last_2h", "waiting_orders",
    "shipped_skus_last_2h", "waiting_skus"
]

# --- Simulation Processing Settings ---
PROCESSING_FALLBACK_MINUTES_PER_UNIT = 0.5  # default minutes needed to process one unit when no history
PROCESSING_MIN_SERVICE_MINUTES = 5.0        # minimum minutes spent per shipment even for small orders
PROCESSING_MAX_SERVICE_MINUTES = 720.0      # cap for historical handling minutes (12 hours)
PROCESSING_RATE_MIN_SAMPLES = 15            # minimum observations to trust a bucket
PROCESSING_QUEUE_SAFETY_FACTOR = 1.0        # multiplier applied to queue-derived delay
PROCESSING_SENTINEL_VALUE = -1              # sentinel for wildcard buckets (dc/hour/weekend)

# Delivery Time Model Settings
# Legacy paths (for backward compatibility with existing global models)
DELIVERY_MODELS_DIR = DATA_DIR / 'models' / 'delivery_time'
DELIVERY_PLOTS_DIR = DATA_DIR / 'plots' / 'delivery_time'

# New organized paths
DELIVERY_MODELS_GLOBAL_DIR = DATA_DIR / 'models' / 'delivery_time' / 'global'
DELIVERY_MODELS_CS_DIR = DATA_DIR / 'models' / 'delivery_time_cs' / 'tune'  # Carrier-specific
DELIVERY_TIME_SIMULATOR_PATH = DATA_DIR / 'models' / 'delivery_time_cs' # Carrier-specific simulator models
DELIVERY_PLOTS_GLOBAL_DIR = DATA_DIR / 'plots' / 'delivery_time' / 'global'
DELIVERY_PLOTS_CS_DIR = DATA_DIR / 'plots' / 'delivery_time_cs'  # Carrier-specific

DELIVERY_TIME_TARGET = "delivery_time_days"
# Features used to train the delivery time model.
DELIVERY_TIME_FEATURES = [
    "order_hour", "weekday", "shipped_orders_last_2h", "waiting_orders",
    "shipped_skus_last_2h", "waiting_skus", "num_skus_in_order",
    "has_bundle_discount", "has_coupon_discount", "type", "has_gift_item",
    "total_quantity_in_order", "avg_discount_rate_in_order",
    "user_level", "city_level", "plus", "promise_delivery_days", "dc_ori", "dc_des"
]

# ORIGIN_AGG_FEATURES = [
#     'dc_ori',"shipped_orders_last_2h", "waiting_orders",
#     "shipped_skus_last_2h",   "waiting_skus",
# ]

DELIVERY_TIME_QUANTILES = list(np.arange(0.05, 1, 0.05)) 
# + [0.92, 0.97, 0.99]

# --- Delivery-Time Quantile-Regression (QR) paths -----------------
DELIVERY_QR_MODELS_DIR = DATA_DIR / 'models' / 'delivery_time_qr'
DELIVERY_QR_PLOTS_DIR  = DATA_DIR / 'plots'  / 'delivery_time_qr'

# --- Delivery Time DL Model Hyperparameters ----------------------------------
DELIVERY_DL_EPOCHS = 500
DELIVERY_DL_LEARNING_RATE = 1e-4
DELIVERY_DL_BATCH_SIZE = 128
DELIVERY_DL_HIDDEN_DIM = 128
DELIVERY_DL_N_LAYERS = 3
DELIVERY_DL_DROPOUT = True
DELIVERY_DL_DROPOUT_P = 0.3
DELIVERY_DL_WEIGHT_DECAY = 1e-3
DELIVERY_DL_VALIDATION_SPLIT_RATIO = 0.2
DELIVERY_DL_EARLY_STOPPING_PATIENCE = 10
DELIVERY_DL_LR_SCHEDULER_PATIENCE = 5
DELIVERY_DL_LR_SCHEDULER_FACTOR = 0.5

# Embedding hyperparameters for categorical features
DELIVERY_DL_DC_ORI_EMBEDDING_DIM = 8
DELIVERY_DL_DC_DES_EMBEDDING_DIM = 8

# Categorical feature definitions
DELIVERY_DL_CATEGORICAL_FEATURES = ['dc_ori', 'dc_des']
DELIVERY_DL_UNKNOWN_TOKEN = "__UNK__"
DELIVERY_DL_NUMERICAL_FEATURES = [f for f in DELIVERY_TIME_FEATURES if f not in DELIVERY_DL_CATEGORICAL_FEATURES]

# Previous hyperparameter grid (commented out - was performing worse with pinball loss > 0.2)
# DELIVERY_DL_HYPERPARAMETER_GRID = {
#     # Learning rate: bias a bit lower than before to stabilise deeper/wider nets.
#     # (AdamW, log-uniform sampling in generate_grid.py)
#     'lr': (5e-4, 1.5e-3),
#
#     # Model capacity: focus on medium–large models that can match tree ensembles.
#     'hidden_dim': [192, 256, 384, 512],
#     'n_layers': [4, 5, 6],
#
#     # Dropout: slightly lighter regularisation than before.
#     'dropout_p': (0.10, 0.30),
#
#     # Weight decay: reduce upper bound so large models are not over-penalised.
#     'weight_decay': (1e-4, 8e-4),
#
#     # Batch size: emphasise 64/128; 32 is dropped to favour more stable gradients.
#     'batch_size': [64, 128],
#     
#     # Embedding dimensions: bias toward moderately larger embeddings for dc_ori/des.
#     'dc_ori_embedding_dim': [8, 12, 16],
#     'dc_des_embedding_dim': [8, 12, 16],
#     
#     # Early stopping patience: allow a bit more patience to fully fit harder carriers.
#     'early_stopping_patience': [10, 15, 20],
# }

# Restored hyperparameter grid (better performance - lower pinball loss)
DELIVERY_DL_HYPERPARAMETER_GRID = {
    'lr': (3e-4, 3e-3),                   # log=True
    'hidden_dim': [96, 128, 192],         # add 96 & 192; keep capacity options near 128
    'n_layers': [2, 3, 4],                # allow 2; 3 is often best but let Optuna confirm
    'dropout_p': (0.15, 0.40),            # trim the high-dropout tail (0.5–0.7 hurt in logs)
    'weight_decay': (3e-4, 3e-3),         # log=True; winners were ~1e-3
    'batch_size': [64, 128, 256],         # 128 won; optionally test 256 if VRAM allows
}

# Simulator-specific grid (broader capacity + higher lr range for outcome sampling)
DELIVERY_DL_SIMULATOR_HYPERPARAMETER_GRID = {
    'lr': (3e-4, 3e-2),                   # log=True (extend upper bound)
    'hidden_dim': [96, 128, 192, 256, 384],  # add 256/384
    'n_layers': [2, 3, 4, 5, 6],           # allow deeper nets
    'dropout_p': (0.15, 0.60),            # allow stronger regularization
    'weight_decay': (3e-4, 3e-2),         # log=True (extend upper bound)
    'batch_size': [64, 128, 256],
}

# Demand Model Settings
# Add new path for quantile regression models
DEMAND_QR_MODELS_DIR = DATA_DIR / 'models' / 'demand_qr'
DEMAND_QR_PLOTS_DIR = DATA_DIR / 'plots' / 'demand_qr'

# Add new path for MQRNN models
DEMAND_MODELS_DIR = DATA_DIR / 'models' / 'demand'
DEMAND_PLOTS_DIR = DATA_DIR / 'plots' / 'demand'

DEMAND_ORDER_FEATURES = [
    "num_skus_in_order",
    "has_bundle_discount", "has_coupon_discount", 
    "type", "has_gift_item",
    "total_quantity_in_order", "avg_discount_rate_in_order",
    "user_level", "plus", 
    "gender", "age", "marital_status", "education", 
    "city_level", "purchase_power",
    "promise_delivery_days",
    "attribute1", "attribute2",
    "dc_des"
]
DEMAND_CATEGORICAL_ORDER_FEATURES = ['dc_des']
SKU_COL    = "sku_ID"
BRAND_COL  = "brand_ID"
DEMAND_TARGET_COL = "quantity"
DATE_COL   = "order_time"

# Demand Model Hyperparameters
DEMAND_FORECAST_HORIZON = 24
DEMAND_MODEL_LOOKBACK = 36
DEMAND_MODEL_TIME_PERIOD_FREQ = 'h' # 'D' for day, 'h' for hour
DEMAND_MODEL_HIDDEN_DIM = 32
DEMAND_MODEL_QUANTILES = list(np.arange(0.05, 1, 0.05))
DEMAND_MODEL_LSTM_N_LAYERS = 2# Number of hidden layers
DEMAND_MODEL_MLP_LINEAR = True
DEMAND_MODEL_DROPOUT_P = 0.5
DEMAND_MODEL_CONTEXT_DIM = 32
DEMAND_MODEL_BIDIRECTIONAL = False
DEMAND_MODEL_SKU_EMBEDDING_DIM = 16
DEMAND_MODEL_BRAND_EMBEDDING_DIM = 12

DEMAND_MODEL_EARLY_STOPPING_PATIENCE = 8
DEMAND_MODEL_LR_SCHEDULER_PATIENCE = 4
DEMAND_MODEL_LR_SCHEDULER_FACTOR = 0.5
DEMAND_MODEL_VALIDATION_SPLIT_RATIO = 0.2
DEMAND_MODEL_EPOCHS = 200
DEMAND_MODEL_LEARNING_RATE = 1e-3
DEMAND_MODEL_BATCH_SIZE = 64
DEMAND_MODEL_WEIGHT_DECAY = 1e-4

# --- Demand Model Hyperparameter Tuning ---
# Old settings (commented out):
# "lr": (2e-4, 7e-4),            # log-uniform; best sat at the low end
# "hidden_dim": [128],           # fixed
# "lstm_n_layers": [2],          # fixed
# "dropout_p": (0.54, 0.60),     # very tight around winners
# "weight_decay": (3e-5, 2e-4),  # log-uniform; best ~3e-5–1e-4
# "batch_size": [32, 64, 128]    # keep 32/64; 128 as backup

DEMAND_MODEL_HYPERPARAMETER_GRID = {
    # Learning rate: Best was 8.89e-5 (very low), explore around that value
    "lr": (5e-5, 1.5e-4),                 # log-uniform; focus around best (8.89e-5)
    
    # Hidden dimension: Best was 384, explore around that
    "hidden_dim": [256, 384, 512],        # Best was 384, try 256 and 512
    
    # LSTM layers: Best was 3, keep that and try 4
    "lstm_n_layers": [3, 4],              # Best was 3, try 4 for more capacity
    
    # Dropout: Best was 0.697 (very high), explore even higher for regularization
    "dropout_p": (0.65, 0.80),            # Focus on high dropout; best was 0.697, try up to 0.8
    
    # Weight decay: Best was 0.0001, explore around that (lower than previous best)
    "weight_decay": (5e-5, 2e-4),         # log-uniform; focus around best (0.0001)
    
    # Batch size: Best was 16 (very small), keep that and try 32
    "batch_size": [16, 32],               # Best was 16, try 32
    
    # Context dimension: Best was 12 (smaller than expected), explore around that
    "context_dim": [12, 16, 20],         # Best was 12, try 16 and 20
    
    # Embeddings: Best was sku_emb=8, brand_emb=16
    "sku_emb": [8, 12],               # Best was 8, try 12 and 16
    "brand_emb": [12, 16],            # Best was 16, try 12 and 20
    
    # Early stopping patience: Best was 8, try more patience for better convergence
    "early_stopping_patience": [8, 12, 16] # Best was 8, try more patience
}

# Hybrid demand scenario generation threshold
# Mean cumulative demand over the first L periods used to classify fast/slow SKUs
DEMAND_HYBRID_THRESHOLD = 100.0

# --- Proxy Model Settings ---
PROXY_MODELS_DIR = DATA_DIR / 'models' / 'proxy'
PROXY_PLOTS_DIR = DATA_DIR / 'plots' / 'proxy'

# Reported proxy result (the proxy row shown in the main comparison table/figures).
# This is the `job28_cfg4` tuning leader, refit on full data, evaluated with the
# `inventory_weighted` inference strategy (total objective 829,805.22 +/- 3,591.60
# on the peak test set). Select it in analysis via:
#   --proxy-model-name <REPORTED_PROXY_MODEL>   (auto-pins the inventory_weighted strategy)
REPORTED_PROXY_MODEL = (
    'proxy_tune_lr4e-04_hd224_do0.22_nl2_wd1e-04_mvlhv2_cw0.2_cost0.01_ew0.00'
    '_sku8_br6_cemb16_op8_dcemb64_thNA_job28_cfg4_20260225_150952'
    '_refitfull_20260226_083409'
)
REPORTED_PROXY_STRATEGY = 'inventory_weighted'
PROXY_MODEL_HIDDEN_DIM = 160
PROXY_MODEL_N_LAYERS = 3
PROXY_MODEL_DROPOUT_P = 0.2
PROXY_MODEL_LEARNING_RATE = 3e-4
PROXY_MODEL_BATCH_SIZE = 64
PROXY_MODEL_WEIGHT_DECAY = 1e-3
PROXY_MODEL_CONSTRAINT_VIOLATION_WEIGHT = 0.2
PROXY_MODEL_CARDINALITY_PENALTY_WEIGHT = 0.0
PROXY_MODEL_ENTROPY_WEIGHT = 0.05
PROXY_MODEL_COST_LOSS_WEIGHT = 0.05
PROXY_MODEL_GUMBEL_TAU = 1.0
PROXY_MODEL_LABEL_SMOOTHING = 0.1  # Label smoothing factor (0.0 = no smoothing, 0.1 = typical value)
PROXY_MODEL_THRESHOLD = 0.0
# Default inference repair strategy:
# "argmax_then_split" = pick top-ranked DC/carrier by model score, then greedily split if inventory is short.
# "default" remains accepted as a legacy alias.
PROXY_MODEL_REPAIR_STRATEGY = "argmax_then_split"
# Power for inventory_weighted repair strategy during proxy inference.
# 1.0 = linear inventory/demand weighting; >1 favors high-inventory DCs more strongly.
PROXY_MODEL_INVENTORY_WEIGHT_POWER = 1.0
PROXY_INFERENCE_DEBUG = False

PROXY_MODEL_VALIDATION_SPLIT_RATIO = 0.2
PROXY_MODEL_EPOCHS = 500
PROXY_MODEL_EARLY_STOPPING_PATIENCE = 15
PROXY_MODEL_LR_SCHEDULER_PATIENCE = 10
PROXY_MODEL_LR_SCHEDULER_FACTOR = 0.5
PROXY_MODEL_SKU_EMBEDDING_DIM = 8
PROXY_MODEL_BRAND_EMBEDDING_DIM = 6
PROXY_MODEL_SELECTION_WEIGHT = 1.0
PROXY_MODEL_QUANTITY_WEIGHT = 0.0
PROXY_MODEL_SCENARIO_COMBINE = "add"
PROXY_ORDER_FEATURES = [
    "order_hour", "weekday", "num_skus_in_order",
    "has_coupon_discount",
    "type", "has_gift_item",
    "total_quantity_in_order", "avg_discount_rate_in_order",
    "user_level",
    "age", "education",
    "city_level", "purchase_power",
    "promise_delivery_days",
    "attribute1", "attribute2",
    "dc_des"
]
PROXY_CATEGORICAL_ORDER_FEATURES = ['dc_des']

# Optimization Model Cost Parameters (used for deterministic cost calculation)
COST_TIER_1 = 1.0 # Closest DC, not central warehouse
COST_TIER_2 = 1.6 # Closest DC, central warhouse
COST_TIER_3 = 1.5 # Same Region, not central warehouse
COST_TIER_4 = 2.2 # Same Region, central warehouse
COST_TIER_5 = 2.0 # Different Region, not central warehouse
COST_TIER_6 = 3.0 # Different Region, central warehouse
# COST_TIER_1 = 1.0  # Closest DC
# COST_TIER_2 = 1.15  # DC in the same region, not the closest
# COST_TIER_3 = 1.3  # Same Region, central warehousen # 3.0
# COST_TIER_4 = 1.6  # Different Region
COST_NOISE_SIGMA = 0.05           # log‑normal σ 
# --- Consolidation Discount & Penalty Parameters ---
BETA_DISCOUNT = 0.5  # 50% discount for shipping multiple items from the same DC
GAMMA_PLUS_LATE_PENALTY = 40.0 # Penalty for each day late per unit # 2.0
GAMMA_MINUS_EARLY_PENALTY = 0.2 # Penalty for each day early per unit # 0.2
# --- Stockout Penalty ---
STOCKOUT_PENALTY_PER_UNIT = 200.0 # Penalty for lost sales per unit
LOCAL_DC_STOCK_PROB = 0.75  # Probability that a SKU is carried at a local DC
CENTRAL_WAREHOUSE_STOCK_PROB = 1.0  # Probability that a SKU is carried at a central warehouse
LOCAL_DC_CSL = 0.50  # Cycle service level (probability) for local DCs
CENTRAL_WAREHOUSE_CSL = 0.80  # Cycle service level (probability) for central warehouses
LOCAL_DC_BASE_COST_PER_UNIT = 2.0  # Fixed per-unit shipping cost component for local DCs
CENTRAL_WAREHOUSE_BASE_COST_PER_UNIT = 4.0  # Fixed per-unit shipping cost for central warehouses
DELAY_BIAS_LEVEL = 1.0

# --- SAA ---
SAA_N1 = 50  # Samples for candidate generation
SAA_Q = 10 # Number of candidates to evaluate
SAA_N2 = 500 # Samples for candidate evaluation
# Optional sampled future-delivery penalties in 2nd-stage shipping objective.
# Default False keeps SAA objective on deterministic base shipping costs.
SAA_USE_FUTURE_PENALTIES = False
# SAA_N1 = 30  # Samples for candidate generation
# SAA_Q = 20 # Number of candidates to evaluate
# SAA_N2 = 300 # Samples for candidate evaluation
# --- Simulation and Model Parameters ---
MAX_ORDERS_TO_PROCESS = None # Set to None to process all orders

# --- Gurobi Parameters ---
GUROBI_MIP_FOCUS = 3
GUROBI_THREADS = 4 
GUROBI_TIME_LIMIT = 30 # 30 seconds

# --- Simulation Output ---
SIMULATION_RESULTS_DIR = DATA_DIR / 'simulation_results'
FULFILLMENT_PLANS_DIR = SIMULATION_RESULTS_DIR / 'fulfillment_plans'
PERFORMANCE_METRICS_DIR = SIMULATION_RESULTS_DIR / 'performance_metrics' 


PROCESSED_ORDERS_PATH = {
    "train": "data/processed_orders_train.parquet",
    "test": "data/processed_orders_test.parquet",
}
PRECOMPUTED_ORIGIN_STATS_DIR = "data/precomputed_origin_stats"
PRECOMPUTED_ORDER_DC_FEATURES_DIR = "data/precomputed_order_dc_features"

DEMAND_MODEL_PATH = "data/models/demand"
DELIVERY_TIME_MODEL_PATH = "data/models/delivery_time"

# Empirical distributions for empirical_saa
EMPIRICAL_DISTRIBUTIONS_DIR = DATA_DIR / "empirical" 

# --- Peak-hour defaults for peak-only evaluation/generation ---
PEAK_START_HOUR = 6   # inclusive
PEAK_END_HOUR = 18    # inclusive
PEAK_TIMEZONE = 'America/New_York'

# --- Simulator Configuration ---
SIM_PRECOMPUTE_DIR = DATA_DIR / 'derived' / 'sim'
SIM_USE_PRECOMPUTED_DIST = True
SIM_USE_PRECOMPUTED_COSTS = False
SIMULATOR_TYPE = 'dl'  # 'dl' or 'catboost'
SIM_MAX_QUEUE_WORKERS_PER_DC = None  # Optional parallelism limit

# --- RL Fine-Tuning ---
RL_K_CANDIDATES = 8
RL_TOP_M = 5
RL_NUM_REWARD_SAMPLES = 1
RL_KL_WEIGHT = 0.1
RL_LEARNING_RATE = 1e-5
RL_BATCH_SIZE = 16
RL_TRAIN_DAYS = ['2018-03-20', '2018-03-21', '2018-03-22', '2018-03-23']
RL_VAL_DAYS = ['2018-03-24']
