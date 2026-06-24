#!/usr/bin/env bash
set -euo pipefail

GRID="${GRID:-configs/proxy/paper/grids/proxy_arch_ablation.txt}"
MODEL_PREFIX="${MODEL_PREFIX:-}"
LINE_LIMIT="${LINE_LIMIT:-}"

line_no=0
while IFS= read -r line; do
  if [ -z "$line" ] || [[ "$line" == \#* ]]; then
    continue
  fi

  line_no=$((line_no + 1))
  if [ -n "$LINE_LIMIT" ] && [ "$line_no" -gt "$LINE_LIMIT" ]; then
    break
  fi

  read -r -a args <<< "$line"
  if [ -n "$MODEL_PREFIX" ]; then
    args+=(--model_name "${MODEL_PREFIX}_${line_no}")
  fi

  python -m src.training.proxy.train_proxy "${args[@]}"
done < "$GRID"
