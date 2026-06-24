# Learning Optimization Proxies for Sequential Contextual Stochastic Programs: An Order Fulfillment Application

This repository contains the research code for **Learning Optimization Proxies
for Sequential Contextual Stochastic Programs: An Order Fulfillment
Application**. It studies online fulfillment with stochastic demand, uncertain
delivery times, multiple distribution centers, and multiple carrier services.

The main policy is the fulfillment proxy: a neural surrogate that imitates
high-quality Contextual SAA (CSAA) teacher decisions and then serves fast
feasible decisions during simulation. CSAA is used as the teacher policy for
proxy data generation; the remaining policies are evaluation baselines.

## What Is Included

- Main policy: `proxy`.
- Teacher policy: `csaa`, used to generate high-quality training decisions.
- Baselines: `greedy`, `pto`, `empirical_saa`, `dtlp_bidprice`, and
  `primal_dual`.
- Forecasting code for demand and delivery-time models.
- Proxy feature engineering, training, ablations, and simulation entrypoints.
- Portable reproduction wrappers under `scripts/reproduce/`.

The repository is cleaned for public research release: local Slurm wrappers,
raw JD/MSOM data, large derived row-level artifacts, and internal audit/debug
scripts are not part of the tracked public workflow.

## Setup

```bash
mamba env create -f environment.yml
conda activate fulfillment_proxy
```

`dtlp_bidprice`, CSAA, and other exact optimization paths require a working
Gurobi installation and license. For CPU-only installs, replace
`pytorch-cuda=12.4` in `environment.yml` with the appropriate CPU PyTorch
package for your platform.

## Data

The JD/MSOM source data is not stored in this repository. Download it from:

<https://huggingface.co/datasets/a6687543/MSOM_Data_Driven_Challenge_2020/tree/main>

Expected local paths, external artifacts, and carrier-service augmentation are
documented in [DATA.md](DATA.md). Full row-level derived files such as
`data/processed/preprocessed_data_cs.csv` are generated or supplied locally and
should not be committed.

## Repository Layout

```text
src/
  algo/                    Online policies
  model/                   Proxy and forecasting model definitions
  simulator/               Simulation engine and precompute loaders
  training/                Demand, delivery-time, and proxy training code
  preprocess_data.py
  demand_feature_engineering.py
  delivery_time_feature_engineering.py
  proxy_feature_engineering.py
scripts/
  run_simulation.py        Main simulation CLI
  precompute/              Simulation/proxy precompute builders
  proxy/                   Proxy grid and ablation utilities
  analysis/                Result summaries and plots
  reproduce/               Portable paper workflow shell scripts
configs/proxy/             Proxy configs and curated paper grids
```

## Typical Workflow

### 1. Prepare data

```bash
python -m src.preprocess_data
```

Next, run the carrier-service augmentation to produce
`data/processed/preprocessed_data_cs.csv` — the carrier-aware order table that the
feature-engineering and simulation steps read. The full commands (travel-times →
pseudo DC coords → augmented table) are in
[DATA.md](DATA.md#reproducing-the-carrier-service-augmented-data). Then build the
forecasting feature tables:

```bash
python -m src.demand_feature_engineering
python -m src.delivery_time_feature_engineering
```

### 2. Train forecasting models

```bash
# demand model for the proxy_train period (writes mqrnn_model_tuned.pt)
python -m src.training.demand.train_mqrnn
# demand model for the test period (writes mqrnn_model_with_proxy_tuned.pt);
# test-set simulations load this one
python -m src.training.demand.train_mqrnn --train_on_proxy
python -m src.training.delivery_time.by_carrier.train_dl_rf --use_dl
python -m src.training.delivery_time.by_carrier.train_simulator
```

The delivery-time **simulator** model is required by `scripts.run_simulation`: its
outcome sampler uses `scenario_source='simulator'` and errors if the per-carrier /
global simulator artifacts are missing.

### 3. Build proxy training data

CSAA is run in collection mode here to generate teacher decisions for proxy
training.

```bash
bash scripts/reproduce/build_precompute.sh

python -m scripts.run_simulation \
  --algo csaa \
  --order_set proxy_train \
  --simulation_date 2018-03-20 \
  --csaa-debug-dir data/csaa_solutions \
  --collect-only --peak-only

python -m src.proxy_feature_engineering --data_split proxy_train
```

### 4. Train the proxy

```bash
python -m src.training.proxy.train_proxy \
  --config configs/proxy/hierarchical_proxy_base.json \
  --model_name hierarchical_proxy_main_public
```

(`hierarchical_proxy_main_public` is the name the reproduction wrappers in
`scripts/reproduce/` expect; override with `MODEL_NAME=...` if you change it.)

The default public proxy architecture is `hierarchical_proxy_v2`: it embeds the
order/DC/carrier context (with optional demand-scenario and DC modules) and
predicts scores over `(DC, carrier)` options. See `src/model/` and
`configs/proxy/hierarchical_proxy_base.json`.

The specific proxy reported in the paper's tables and figures is recorded in
`src/config.py` as `REPORTED_PROXY_MODEL` / `REPORTED_PROXY_STRATEGY` (the
`job28_cfg4` tuning leader, refit on full data, decoded with the
`inventory_weighted` strategy).

### 5. Run simulations

```bash
python -m scripts.run_simulation \
  --algo proxy \
  --proxy-model data/models/proxy/hierarchical_proxy_main_public/best.pt \
  --proxy-repair-strategy inventory_weighted \
  --order_set test \
  --simulation_date 2018-03-26
```

`--proxy-repair-strategy` controls how the proxy's `(DC, carrier)` scores are
decoded into a feasible assignment:

- `argmax_then_split` (library default) — take the top `(DC, carrier)` pair, then
  greedily split any remainder onto the next-best DCs.
- `inventory_weighted` (**reported setting**) — pick the DC maximizing
  `dc_prob * min(inventory/demand, 1)`, preserving the learned ranking while
  downweighting inventory-short DCs.
- `inventory_first` — among DCs that can fully cover the order, pick the highest-scoring one.
- `feasible_topk` / `feasible_joint_topk` — restrict the feasibility search to the
  top-`k` DCs / joint `(DC, carrier)` pairs.

All strategies share the same final greedy repair (assign to the chosen DC, pick
the cheapest eligible carrier, spill any remainder to the next DCs).

CSAA can be rerun on test dates as a high-quality reference policy. The same
entrypoint runs the baseline policies by changing `--algo` to `greedy`, `pto`,
`empirical_saa`, `dtlp_bidprice`, or `primal_dual`.

Portable wrappers:

```bash
SIMULATION_DATES="2018-03-26 2018-03-27" bash scripts/reproduce/run_policy_suite.sh
bash scripts/reproduce/collect_results.sh
```

## Simulation Setup

Each run is one `simulation_date` × one `order_set`, evaluated as an **independent
daily instance**: fresh initial inventory each day, no within-day replenishment, and
historical queue/backlog state seeded at the start of the day.

- **Feasible options.** Per order, eligible `(DC, carrier)` options come from
  `data/params/dc_carrier_eligibility.csv` (carrier `allowed_states`) and the DC
  pseudo-coordinates.
- **Option cost.** `base_cost = distance_km * coef_distance_km + fixed_dc_cost`,
  with `coef_distance_km` from `real_cost_models_cs.csv` and a fixed per-unit DC
  cost (2.0 local / 4.0 central). These coefficients are externally calibrated.
- **Initial inventory.** Built per date from demand strictly before that date
  (`target = mean + z·std`), with cycle service levels 0.50 (local) / 0.80
  (central); only a deterministic 75% subset of local `(sku, DC)` pairs is stocked,
  and on-hand is `ceil(target)`.
- **Processing delay.** `min(max(qty·m, 5) + waiting_units·m, 720)` minutes, where
  `m` is a historically-bucketed per-unit rate (fallback 0.5 min/unit).
- **Realized cost** = base shipping cost + stochastic delivery penalty −
  consolidation discount (`BETA_DISCOUNT = 0.5` on the base-cost portion when
  multiple units ship from one DC) + lost-sales penalty (`200` per unfilled unit).

The relevant constants (`LOCAL_DC_CSL`, `CENTRAL_WAREHOUSE_CSL`, `BETA_DISCOUNT`,
`STOCKOUT_PENALTY_PER_UNIT`, etc.) live in `src/config.py`.

## Useful References

- [DATA.md](DATA.md): data sources, local paths, and carrier-service augmentation.
- [configs/proxy/README.md](configs/proxy/README.md): retained proxy configs and paper grids.
- [scripts/README.md](scripts/README.md): public script inventory.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ye2025deep,
  title={Learning Optimization Proxies for Sequential Contextual Stochastic Programs: An Order Fulfillment Application},
  author={Ye, Tinghan and Tong, Shuaicheng and Guan, Changkun and Basciftci, Beste and Van Hentenryck, Pascal},
  booktitle={NeurIPS 2025 Workshop MLxOR: Mathematical Foundations and Operational Integration of Machine Learning for Uncertainty-Aware Decision-Making}
}
```

## License

This code is released under the MIT License. See [LICENSE](LICENSE).
