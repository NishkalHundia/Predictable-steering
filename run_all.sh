#!/usr/bin/env bash
# End-to-end reproduction. Runs the experiment for all five paper behaviors,
# then renders paper-style figures and Tables 1 + 2.
#
# Hardware: needs a GPU with enough memory to fit Gemma-2-9B-IT in bf16
# (~20 GB). Total runtime ≈ 1–2h on a single H100, longer on smaller cards.
#
# Override the python entry point with:  PYTHON=/path/to/python ./run_all.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-google/gemma-2-9b-it}"
LAYERS="${LAYERS:-10-32}"
FACTORS="${FACTORS:-1,2,3,5,10}"
BATCH="${BATCH:-16}"
SEED="${SEED:-42}"

BEHAVIORS=(
    sycophancy
    survival-instinct
    corrigible-neutral-HHH
    hallucination
    myopic-reward
)

for b in "${BEHAVIORS[@]}"; do
    echo "=== ${b} ==="
    "${PYTHON}" scripts/mcqa_projection_link.py \
        --behavior "${b}" \
        --model_name "${MODEL}" \
        --layers "${LAYERS}" \
        --factors "${FACTORS}" \
        --batch_size "${BATCH}" \
        --seed "${SEED}"
done

echo "=== rendering paper figures ==="
"${PYTHON}" scripts/make_paper_plots.py --all

echo "=== rendering paper tables ==="
"${PYTHON}" scripts/make_paper_tables.py
