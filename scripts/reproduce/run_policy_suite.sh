#!/usr/bin/env bash
set -euo pipefail

ORDER_SET="${ORDER_SET:-test}"
SIMULATION_DATES="${SIMULATION_DATES:-2018-03-26}"
POLICIES="${POLICIES:-greedy pto empirical_saa dtlp_bidprice primal_dual proxy}"
NUM_REPLICATIONS="${NUM_REPLICATIONS:-50}"
SIMULATOR_TYPE="${SIMULATOR_TYPE:-dl}"
PEAK_ONLY="${PEAK_ONLY:-1}"
PROXY_MODEL="${PROXY_MODEL:-data/models/proxy/hierarchical_proxy_main_public/best.pt}"
PROXY_REPAIR_STRATEGY="${PROXY_REPAIR_STRATEGY:-inventory_weighted}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_ORDERS="${MAX_ORDERS:-}"

for simulation_date in $SIMULATION_DATES; do
  for policy in $POLICIES; do
    args=(
      python -m scripts.run_simulation
      --algo "$policy"
      --order_set "$ORDER_SET"
      --simulation_date "$simulation_date"
      --num_replications "$NUM_REPLICATIONS"
      --simulator_type "$SIMULATOR_TYPE"
    )

    if [ "$PEAK_ONLY" = "1" ]; then
      args+=(--peak-only)
    fi
    if [ -n "$OUTPUT_DIR" ]; then
      args+=(--output_dir "$OUTPUT_DIR")
    fi
    if [ -n "$MAX_ORDERS" ]; then
      args+=(--max-orders "$MAX_ORDERS")
    fi
    if [ "$policy" = "proxy" ]; then
      args+=(--proxy-model "$PROXY_MODEL")
      args+=(--proxy-repair-strategy "$PROXY_REPAIR_STRATEGY")
    fi

    "${args[@]}"
  done
done
