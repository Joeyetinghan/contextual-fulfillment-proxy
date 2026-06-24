# Public Scripts

This directory is limited to paper reproduction, proxy experiments, documented
result summaries, and portable reproduction workflows. Scheduler-specific local
wrappers live outside the tracked public script surface.

## Simulation

- `run_simulation.py`: unified simulator entrypoint.
- `reproduce/`: portable end-to-end shell workflows.

## Precompute

- `precompute/build_sim_precompute.py`: build simulation precompute artifacts.
- `precompute/precompute_base_cost_tensor.py`: precompute static base-cost tensors.
- `precompute/precompute_static_proxy_features.py`: precompute static proxy features.

## Proxy Experiments

- `proxy/generate_and_tune_proxy.py`: generate proxy training grids.
- `proxy/generate_proxy_ablation_fixed.py`: generate fixed proxy ablation grids.
- `proxy/generate_refit_proxy_grid.py`: generate selected refit configs.
- `proxy/analyze_tuning_results.py`: summarize tuning grid results.
- `proxy/summarize_proxy_ablations.py`: summarize ablation results.
- `proxy/summarize_proxy_ablation_validation.py`: validate ablation summaries.

## Analysis

- `analysis/collect_sim_summaries.py`: collect simulation summaries.
- `analysis/summarize_sim_bounds.py`: summarize bounds, uncertainty, and coverage.
- `analysis/analyze_runtime_gap_by_size.py`: analyze runtime and optimization gaps.
- `analysis/plot_intraday_trajectory.py`: plot intraday realized-cost trajectories.
- `analysis/summarize_scenario_sensitivity.py`: summarize scenario-count sensitivity.

Exploratory notebooks, one-off debugging scripts, and unused policy submission
helpers are intentionally excluded from the public script surface.
