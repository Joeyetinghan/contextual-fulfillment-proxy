# Data Manifest

This repository does not track the JD/MSOM row-level data. It tracks only a
small set of calibration metadata files needed by the public carrier-service
augmentation workflow.

## Source Data

Download the original MSOM Data Driven Challenge 2020 data from:

<https://huggingface.co/datasets/a6687543/MSOM_Data_Driven_Challenge_2020/tree/main>

Place the raw JD/MSOM files under `data/fulfillment/` — the directory that
`src/config.py` and `src.preprocess_data` read from. The preprocessing pipeline
expects:

- `data/fulfillment/JD_order_data.csv`
- `data/fulfillment/JD_delivery_data.csv`
- `data/fulfillment/JD_sku_data.csv`
- `data/fulfillment/JD_user_data.csv`
- `data/fulfillment/JD_network_data.csv`

Most of the `data/` directory is otherwise treated as a local artifact directory.

## Tracked Calibration Metadata

The following small metadata/calibration files are included in Git:

- `data/params/calibration_dist_bins_km.json`
- `data/params/hubs.csv`
- `data/params/major_us_cities.csv`
- `data/params/real_cost_models_cs.csv`
- `data/params/real_ratios_cs.csv`
- `data/params/dc_carrier_eligibility.csv`

Other data files, including row-level JD/MSOM data, generated processed tables,
model checkpoints, simulation outputs, and ZIP geocoding caches, remain local
external artifacts.

## Data Provenance & Licensing

The repository **code** is released under the MIT License (see `LICENSE`).

The upstream **JD/MSOM source data is licensed separately** — the MSOM Data
Driven Challenge 2020 dataset is distributed under the Open Data Commons Open
Database License (ODbL), with JD.com's underlying copyrights reserved. ODbL
requires attribution and a license notice and may impose share-alike obligations
on derived databases. **This repository does not redistribute the row-level
source data**; obtain it directly from the source above under its own terms, and
review that license before publishing any derived data of your own.

The small files tracked here are reference/calibration aggregates or public
geographic data, not row-level orders:

| File | What it is | Provenance |
|------|------------|------------|
| `data/params/hubs.csv` | Hub city coordinates | Public US geographic reference data |
| `data/params/major_us_cities.csv` | Major US city coordinates / ZIP3 | Public US geographic reference data |
| `data/params/calibration_dist_bins_km.json` | Distance bin edges for carrier calibration | Externally supplied calibration constants |
| `data/params/real_cost_models_cs.csv` | Per-carrier distance cost coefficients | Externally supplied carrier calibration (not estimated from the JD order table) |
| `data/params/real_ratios_cs.csv` | Per-carrier delivery-time ratios by distance bin | Externally supplied carrier calibration (not estimated from the JD order table) |
| `data/params/dc_carrier_eligibility.csv` | DC × carrier × ZIP3 coverage matrix (`allowed_states`, `has_coverage`) | Aggregate derived from the fulfillment network and carrier-service definitions; no order-level data |

If you redistribute a fork, confirm that these aggregates are compatible with the
upstream ODbL terms for your use.

## Local Derived Artifacts

The pipeline expects these generated files during normal experiments:

- `data/processed/preprocessed_data.csv`
- `data/processed/preprocessed_data_cs.csv`
- `data/processed/aggregated_travel_times.csv`
- `data/derived/pseudo_coords/hub_based_coords.csv`
- `data/derived/sim/...`
- `data/training_data/proxy/...`
- `data/models/...`
- `data/csaa_solutions/...`

These files are not committed because they are row-level data or generated
experiment artifacts.

## Reproducing the Carrier-Service Augmented Data

`data/processed/preprocessed_data_cs.csv` is the carrier-aware order table used by
the carrier-specific delivery-time models and the carrier-aware simulator. It is a
**synthetic, calibrated** dataset: order rows come from the JD/MSOM data, while the
carrier-service labels and delivery-time scaling come from externally supplied
calibration files. The full table is a generated local artifact and is **not**
committed; regenerate it locally with the steps below.

Every step is deterministic given the same inputs and `--seed 42`, so the result
is fully reproducible.

### Prerequisites

- `data/processed/preprocessed_data.csv` — base preprocessed orders. Produce it by
  placing the raw JD/MSOM files under `data/fulfillment/` (see Source Data) and
  running `python -m src.preprocess_data`.
- `data/fulfillment/JD_network_data.csv` — DC-to-region network mapping (from the source data).
- The tracked calibration files above (`hubs.csv`, `major_us_cities.csv`,
  `real_ratios_cs.csv`, `calibration_dist_bins_km.json`).
- Optional: `data/params/zip5_geocoding.json` — a ZIP5 lookup that improves the
  synthetic customer geocoding. If absent, the augmentation falls back to generated
  ZIP5 values, so the pipeline still runs (results differ slightly).

### Step 1 — DC-to-DC travel-time table

`generate_hub_based_coords` places pseudo DC coordinates using an aggregated
DC-to-DC travel-time table. Build it from the base orders (winsorize travel times
at `p=0.05`, take the **median** per DC pair, keep pairs with `>= 5` events):

```bash
python -m src.data_augmentation.extract_ship_time \
  --input data/processed/preprocessed_data.csv \
  --output data/processed/aggregated_travel_times.csv
```

Output columns: `dc_ori, dc_des, travel_time_hours (median), n_events`.

### Step 2 — Pseudo DC coordinates

```bash
python -m src.data_augmentation.generate_hub_based_coords \
  --hubs data/params/hubs.csv \
  --regions data/fulfillment/JD_network_data.csv \
  --ship_times data/processed/aggregated_travel_times.csv \
  --cities data/params/major_us_cities.csv \
  --output data/derived/pseudo_coords/hub_based_coords.csv \
  --seed 42
```

Output columns include `dc_id, lat, lon, zip3`. City-based assignment (via
`--cities`) is recommended; it yields interpretable ZIP3 assignments.

### Step 3 — Carrier-service augmented order table

```bash
python -m src.data_augmentation.sample_carrier_services \
  --input data/processed/preprocessed_data.csv \
  --output data/processed/preprocessed_data_cs.csv \
  --ratios data/params/real_ratios_cs.csv \
  --bins data/params/calibration_dist_bins_km.json \
  --dc_coords data/derived/pseudo_coords/hub_based_coords.csv \
  --zip5_geocoding data/params/zip5_geocoding.json \
  --zip5_mapping data/params/zip5_geocoding_cache.json \
  --seed 42
```

(`--zip5_geocoding` / `--zip5_mapping` are optional; the latter is a local cache the
script may create or update.)

### What the augmentation does

For each order, `sample_carrier_services`:

1. generates a synthetic `customer_zip5` consistent with the destination region's ZIP3;
2. geocodes it to `customer_lat` / `customer_lon` / `customer_state`;
3. computes `distance_km` from the origin DC (`dc_ori`) to the customer (Haversine);
4. bins the distance via `calibration_dist_bins_km.json` (`dist_bin`);
5. samples a `carrier_service_id_anon` from the carrier pool for that bin
   (`real_ratios_cs.csv`) and reads its median delivery-time ratio (`delivery_ratio`);
6. multiplies the original delivery time by that ratio, preserving the originals as
   `delivery_time_{hours,days}_original`.

The resulting delivery time stays anchored to each order's observed value but is
re-scaled to the sampled carrier profile. `carrier_service_id_anon` is the grouping
key for the carrier-specific delivery-time models and the carrier-aware simulator.

Note: this is a **calibrated synthetic** dataset — the carrier label and re-scaled
delivery times are constructed, not observed. The calibration inputs
(`real_ratios_cs.csv`, `calibration_dist_bins_km.json`, `zip5_geocoding.json`) are
externally supplied and are not estimated from the JD order table.
