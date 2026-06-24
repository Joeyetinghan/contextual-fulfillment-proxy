# Global Delivery Time Model Training

This directory contains training scripts for global (carrier-agnostic) delivery time models.

## Available Training Scripts

### 1. `train_quantile_models.py`
Trains XGBoost and sklearn QuantileRegressor models on all deliveries (carrier-agnostic).

**Usage:**
```bash
python -m src.training.delivery_time.global.train_quantile_models [--model_type {xgboost,sklearn}] [--train_on_proxy] [--quantile 0.50]
```

**Options:**
- `--model_type`: Choose between 'xgboost' or 'sklearn' quantile regression
- `--train_on_proxy`: Train on forecast+proxy, evaluate on test (default: train on forecast only)
- `--quantile`: Train only a specific quantile (optional, default: all quantiles)

**Output:**
- Models: `data/models/delivery_time/delivery_model_global_{model_type}{mode_suffix}.joblib`
- Plots: `data/plots/delivery_time/`

### 2. `train_dl_rf.py`
Trains Deep Learning (PyTorch TimeQuantileModel) and/or Random Forest Quantile models.

**Usage:**
```bash
python -m src.training.delivery_time.global.train_dl_rf [--train_on_proxy] [--use_dl] [--use_qrf]
```

**Options:**
- `--train_on_proxy`: Train on forecast+proxy, evaluate on test
- `--use_dl`: Train deep learning model
- `--use_qrf`: Train quantile random forest model

**Output:**
- DL Models: `data/models/delivery_time/delivery_model_global{mode_suffix}.pt`
- RF Models: `data/models/delivery_time/delivery_model_global_qrf{mode_suffix}.joblib`

### 3. `train_simulator.py`
Trains a CatBoost classifier for discrete-event simulation (trained on test set to approximate true environment).

**Usage:**
```bash
python -m src.training.delivery_time.global.train_simulator [--n_trials NUM]
```

**Options:**
- `--n_trials`: Number of Optuna hyperparameter tuning trials (default: 0, no tuning)

**Output:**
- Models: `data/models/delivery_time/delivery_simulator_global.joblib`
- Plots: `data/plots/delivery_time/simulator/`

## Data Sources

Global models use the standard split files:
- `data/processed/delivery_time_forecast_train.csv`
- `data/processed/delivery_time_proxy_train.csv`
- `data/processed/delivery_time_test.csv`

## Common Utilities

All scripts use shared functions from `src/training/delivery_time/common.py` for:
- Data loading and splitting
- Categorical encoding
- Plotting functions
- Loss computation

## When to Use Global vs Carrier-Specific Models

**Use Global Models when:**
- You need a baseline for comparison
- Carrier service information is not available at prediction time
- Training time is limited
- You want to understand overall delivery time patterns

**Use Carrier-Specific Models when:**
- Carrier service is known at prediction time
- You need more accurate predictions per carrier
- Different carriers have significantly different delivery characteristics
- See: `src/training/delivery_time/by_carrier/` directory

