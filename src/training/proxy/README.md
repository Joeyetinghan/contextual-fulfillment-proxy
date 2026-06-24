# Proxy Model Training

This directory contains the training entrypoint and shared helpers for the
paper-facing hierarchical proxy model.

## Main Entry Point

```bash
python -m src.training.proxy.train_proxy --config configs/proxy/hierarchical_proxy_base.json
```

The trainer consumes proxy feature tensors produced by:

```bash
python -m src.proxy_feature_engineering
```

Models and training metadata are written under `data/models/proxy/`.

## Public Architectures

- `hierarchical_proxy_v2`: the main hierarchical proxy used by the paper workflow.
- `single_tower`: compact baseline used for ablation studies.

The default proxy config is `configs/proxy/hierarchical_proxy_base.json`; the
paper tuning grids and selected refits live under `configs/proxy/paper/`.

## Simulation Use

Trained proxy checkpoints are evaluated through:

```bash
python -m scripts.run_simulation --algo proxy --proxy-model data/models/proxy/<model_name>/best.pt
```

The public simulation dispatcher also exposes the paper baselines: `greedy`,
`csaa`, `pto`, `empirical_saa`, `dtlp_bidprice`, and `primal_dual`.
