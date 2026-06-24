# Model Training

This directory contains the forecasting and proxy training entrypoints used by the paper workflow.

## Directory Structure

```text
src/training/
|-- demand/              # Demand forecasting models
|-- delivery_time/       # Delivery-time models
`-- proxy/               # Hierarchical proxy training
```

## Demand Forecasting

```bash
python -m src.training.demand.train_mqrnn
```

Tree-based alternatives remain available for ablations:

```bash
python -m src.training.demand.train_xgboost
python -m src.training.demand.train_qrf
```

Demand models are written under `data/models/demand/` or `data/models/demand_qr/`.

## Delivery-Time Forecasting

Carrier-specific delivery-time models are the paper-facing path:

```bash
python -m src.training.delivery_time.by_carrier.train_dl_rf --use_dl
```

Additional carrier-specific baselines:

```bash
python -m src.training.delivery_time.by_carrier.train_quantile_models
python -m src.training.delivery_time.by_carrier.train_simulator
```

Carrier-specific models are written under `data/models/delivery_time_cs/`.

## Proxy Training

Build proxy tensors from collected CSAA traces, then train the hierarchical proxy:

```bash
python -m src.proxy_feature_engineering --data_split proxy_train
python -m src.training.proxy.train_proxy --config configs/proxy/hierarchical_proxy_base.json
```

The public proxy architectures are `hierarchical_proxy_v2` and `single_tower`.
Trained checkpoints are written under `data/models/proxy/` and evaluated through:

```bash
python -m scripts.run_simulation --algo proxy --proxy-model data/models/proxy/<model_name>/best.pt
```
