#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/peak/simulation_results}"
ORDER_SET="${ORDER_SET:-test}"
DATE_FROM="${DATE_FROM:-}"
DATE_TO="${DATE_TO:-}"
OUT_DIR="${OUT_DIR:-logs/summaries}"

mkdir -p "$OUT_DIR"

summary_args=(
  python -m scripts.analysis.collect_sim_summaries
  --root "$ROOT"
  --order-set "$ORDER_SET"
  --out "$OUT_DIR/simulation_summary.csv"
  --latex-out "$OUT_DIR/simulation_summary.tex"
)

bounds_args=(
  python -m scripts.analysis.summarize_sim_bounds
  --root "$ROOT"
  --order-set "$ORDER_SET"
  --out-csv "$OUT_DIR/simulation_bounds.csv"
  --out-latex "$OUT_DIR/simulation_bounds.tex"
)

if [ -n "$DATE_FROM" ]; then
  summary_args+=(--date-from "$DATE_FROM")
  bounds_args+=(--date-from "$DATE_FROM")
fi
if [ -n "$DATE_TO" ]; then
  summary_args+=(--date-to "$DATE_TO")
  bounds_args+=(--date-to "$DATE_TO")
fi

"${summary_args[@]}"
"${bounds_args[@]}"
