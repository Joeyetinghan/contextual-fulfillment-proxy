# Paper Proxy Config Manifest

This directory contains curated proxy configs used for paper-facing training,
ablation, and refit workflows.

## Directories

- `grids/`: parameter grids for full tuning, local search, and fixed ablations.
- `repair_strategies/`: retained paper-facing proxy repair config.
- `refits/`: selected refit configs plus `MANIFEST.json` mapping old generated filenames to stable public names.

## Main Configs

- `../hierarchical_proxy_base.json`: compact base config for new hierarchical proxy runs.
- `../hierarchical_proxy_main.json`: main cost-loss configuration used as the paper workflow anchor.
