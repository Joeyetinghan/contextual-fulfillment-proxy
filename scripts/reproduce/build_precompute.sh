#!/usr/bin/env bash
set -euo pipefail

ORDER_SETS="${ORDER_SETS:-proxy_train test}"

for order_set in $ORDER_SETS; do
  python -m scripts.precompute.build_sim_precompute --order_set "$order_set"
done
