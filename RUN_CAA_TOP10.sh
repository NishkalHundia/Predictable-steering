#!/usr/bin/env bash

# ------------------------------------------------------------------
# AxBench DiffMean (CAA) pipeline on Gemma-2 2B layer-10
# ------------------------------------------------------------------
# This script documents the exact commands used in the session:
#   - environment setup
#   - dataset downloads
#   - training on a random subset of 10 Concept500 concepts
#   - latent & steering inference (factor = 1.0)
#   - steering evaluation with GPT-4o-mini judge
# Adjust paths or seeds as needed before executing.
# ------------------------------------------------------------------

set -euo pipefail

# ----- 0. Environment prep (per machine) -----
uv sync
export OPENAI_API_KEY="YOUR_KEY_HERE"
export OPENAI_BASE_URL="https://us.api.openai.com"

# ----- 1. Required dataset downloads (run once) -----
(cd axbench/data/ && uv run download-seed-sentences.py && uv run bash download-2b.sh && uv run bash download-alpaca.sh)

# ----- 2. Train DiffMean on 10 random Concept500 concepts (Gemma-2 2B, layer 10) -----
uv run torchrun --nproc_per_node=1 axbench/scripts/train.py --config axbench/sweep/custom/2b/l10/diffmean.yaml --dump_dir axbench/results/top100_diffmean --overwrite_data_dir axbench/concept500/prod_2b_l10_v1/generate --max_concepts 100 --output_length 128 --use_wandb False

# ----- 3. Latent (concept-detection) inference -----
uv run torchrun --nproc_per_node=1 axbench/scripts/inference.py --config axbench/sweep/custom/2b/l10/diffmean.yaml --dump_dir axbench/results/top10_diffmean --overwrite_metadata_dir axbench/results/top10_diffmean/train --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference --mode latent

# ----- 4. Steering inference (DiffMean, factor = 1.0) -----
uv run torchrun --nproc_per_node=1 axbench/scripts/inference.py --config axbench/sweep/custom/2b/l10/diffmean.yaml --dump_dir axbench/results/top10_diffmean --overwrite_metadata_dir axbench/results/top10_diffmean/train --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference --overwrite_inference_dump_dir axbench/results/top10_diffmean/inference --mode steering

# ----- 5. Steering evaluation (GPT-4o-mini judge only) -----
uv run axbench/scripts/evaluate.py --config axbench/sweep/custom/2b/l10/diffmean.yaml --dump_dir axbench/results/top10_diffmean --overwrite_inference_dump_dir axbench/results/top10_diffmean/inference --overwrite_evaluate_dump_dir axbench/results/top10_diffmean/eval --mode steering

# ------------------------------------------------------------------
# Notes:
#  • axbench/sweep/custom/2b/l10/diffmean.yaml limits models to DiffMean
#    and sets steering_factors: [1.0] plus LMJudgeEvaluator only.
#  • Training uses a shuffle; change --seed for different random subsets.
#  • Outputs:
#      - train artifacts:  axbench/results/top10_diffmean/train
#      - inference data:   axbench/results/top10_diffmean/inference
#      - evaluation:       axbench/results/top10_diffmean/eval
# ------------------------------------------------------------------

