#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

OUT_ROOT="${OUT_ROOT:-experiments/network_influence/plots_outputs}"

mapfile -t SWEEP_DIRS < <(
  find experiments/network_influence/outputs -mindepth 5 -maxdepth 5 -type d -path '*/runs/*/*' | sort
)

if [[ ${#SWEEP_DIRS[@]} -eq 0 ]]; then
  echo "No sweep dirs found under experiments/network_influence/outputs."
  exit 0
fi

for sweep_dir in "${SWEEP_DIRS[@]}"; do
  rel="${sweep_dir#experiments/network_influence/outputs/}"
  tag="${rel%%/*}"
  rest="${rel#*/}"
  ts="${rest%%/*}"
  rest2="${rest#*/}"  # runs/<model>/<sweep>
  rest2="${rest2#runs/}"
  model="${rest2%%/*}"
  sweep_name="${rest2#*/}"

  out_dir="${OUT_ROOT}/${tag}/${ts}/${model}/${sweep_name}"
  echo "Plotting ${sweep_dir} -> ${out_dir}"

  "${PYTHON_BIN}" -m experiments.network_influence.plots.generate_all \
    --sweep-dir "${sweep_dir}" \
    --out-dir "${out_dir}"
done

