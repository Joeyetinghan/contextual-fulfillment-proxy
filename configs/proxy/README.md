Proxy configuration catalog and tuning workflow.

## Main configs

- `hierarchical_proxy_base.json`: primary hierarchical proxy v2 training config (recommended base for new runs).
- `hierarchical_proxy_main.json`: main cost-loss config used as the paper workflow anchor.

Run a single model:

```bash
python -m src.training.proxy.train_proxy --config configs/proxy/hierarchical_proxy_base.json
```

## Full vs ablations

Curated grid files live under `configs/proxy/paper/grids/`:

- `full` -> `hierarchical_proxy_paper_full` (cost-biased hybrid) -> `configs/proxy/paper/grids/proxy_full.txt`
- fixed architecture ablations -> `configs/proxy/paper/grids/proxy_arch_ablation.txt`
- fixed loss ablations -> `configs/proxy/paper/grids/proxy_loss_ablation.txt`

Run one grid line directly with:

```bash
python -m src.training.proxy.train_proxy <args from one grid line>
```

The portable wrappers in `scripts/reproduce/` show how to run the main model and
fixed ablations without assuming a particular scheduler.

## Ablation grids

Defined in `scripts/proxy/generate_and_tune_proxy.py`:

- `hierarchical_proxy_ablation_fair`
  - includes `full` + `single_tower` + one-module-off architecture ablations
  - shared hyperparameter candidate pool for fair comparison
- `hierarchical_proxy_ablation_modules_only`
  - same as above, but excludes `full`
- `hierarchical_proxy_loss_ablation_fair`
  - includes `loss_full_loss` baseline and loss ablations:
    - `loss_no_carrier_loss`
    - `loss_no_constraint_loss`
    - `loss_no_cost_loss`
    - `loss_no_label_smoothing`
    - `loss_no_dc_class_weights`
- `hierarchical_proxy_loss_ablation_components_only`
  - same as above, but excludes `loss_full_loss`

## Selecting best models after tuning

Use multiple selection metrics and export best-per-group candidates:

```bash
python -m scripts.proxy.analyze_tuning_results \
  --log_dir logs/tune \
  --run_name tune_proxy \
  --group_col ablation_tag \
  --selection_metric max:val_joint_hit1_repaired \
  --selection_metric min:val_proxy_ub_mean \
  --best_per_group_csv logs/scratch/proxy_best_by_group.csv \
  --best_model_names_txt logs/scratch/proxy_best_models.txt
```

This lets you compare ranking stability across repaired Hit@1 and UB-driven criteria.

## Repair-strategy config

The retained repair-strategy config is
`paper/repair_strategies/hierarchical_proxy_inventory_weighted.json`, matching
the paper-facing proxy repair mode. Other repair strategies remain available
through `scripts.run_simulation --proxy-repair-strategy` for ad hoc checks, but
their standalone training configs are not part of the public catalog.
