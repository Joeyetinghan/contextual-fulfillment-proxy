# Reproduction Workflows

These shell scripts are portable wrappers around the public Python entrypoints.
They do not assume a particular cluster scheduler.

Run them from the repository root. Override settings with environment variables,
for example:

```bash
MODEL_NAME=hierarchical_proxy_main_public bash scripts/reproduce/train_proxy_main.sh
SIMULATION_DATES="2018-03-26 2018-03-27" bash scripts/reproduce/run_policy_suite.sh
```

## Scripts

- `build_precompute.sh`: build simulation precompute artifacts for `proxy_train` and `test`.
- `train_proxy_main.sh`: train the main hierarchical proxy config.
- `run_policy_suite.sh`: run the paper policy suite over one or more dates.
- `run_proxy_ablation_fixed.sh`: train fixed architecture/loss ablations from curated grid files.
- `collect_results.sh`: collect simulation summaries and bound tables.
