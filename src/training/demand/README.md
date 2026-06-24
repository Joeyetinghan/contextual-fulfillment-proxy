# Demand Forecasting Model Training

This directory contains training scripts for demand forecasting models.

## Available Training Scripts

### 1. `train_mqrnn.py`
Trains MQRNN (Multi-Quantile Recurrent Neural Network) for demand forecasting.

**Usage:**
```bash
python -m src.training.demand.train_mqrnn [options]
```

**Features:**
- LSTM-based architecture with quantile outputs
- Handles time series demand data
- SKU and brand embeddings
- Context features

**Output:**
- Models: `data/models/demand/`
- Plots: `data/plots/demand/`

---

### 2. `train_qrf.py`
Trains Quantile Random Forest for demand forecasting.

**Usage:**
```bash
python -m src.training.demand.train_qrf [options]
```

**Features:**
- Non-parametric approach
- Handles non-linear relationships
- Quantile predictions

**Output:**
- Models: `data/models/demand_qr/`
- Plots: `data/plots/demand_qr/`

---

### 3. `train_xgboost.py`
Trains XGBoost for demand forecasting.

**Usage:**
```bash
python -m src.training.demand.train_xgboost [options]
```

**Features:**
- Fast gradient boosting
- Handles tabular features well
- Quantile regression support

**Output:**
- Models: `data/models/demand_qr/`
- Plots: `data/plots/demand_qr/`

---

## Data Requirements

All demand training scripts use:
- Training data: `data/processed/mqrnn_forecast_train.npz`
- Proxy data: `data/processed/mqrnn_proxy_train.npz`
- Test data: `data/processed/mqrnn_test.npz`

## Model Selection

| Model | Best For | Speed | Accuracy | Recommendation |
|-------|----------|-------|----------|----------------|
| **MQRNN** | **Time series patterns** | Slow | **Highest** | **⭐ Production** |
| XGBoost | Tabular features | **Fast** | High | Quick experiments |
| QRF | Non-parametric | Medium | Medium-High | Alternative |

**⭐ Recommendation:** Use **MQRNN** (deep learning) for production deployments - it provides the best accuracy for demand forecasting with time series patterns.

## Configuration

All models use settings from `src/config.py`:
- `DEMAND_MODEL_*` - MQRNN hyperparameters
- `DEMAND_QR_*` - QR model paths
- `DEMAND_FORECAST_HORIZON` - Prediction horizon
- `DEMAND_MODEL_LOOKBACK` - Historical window

## Next Steps

After training demand models, they are used by:
- `src/scenario_generator.py` - Generate demand scenarios for optimization
- `src/algo/contextual_saa.py` - C-SAA algorithm
- `src/algo/empirical_saa.py` - Empirical SAA algorithm

