#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/proxy/hierarchical_proxy_main.json}"
MODEL_NAME="${MODEL_NAME:-hierarchical_proxy_main_public}"

python -m src.training.proxy.train_proxy \
  --config "$CONFIG" \
  --model_name "$MODEL_NAME"
