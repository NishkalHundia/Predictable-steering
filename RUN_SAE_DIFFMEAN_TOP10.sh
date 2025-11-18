#!/usr/bin/env bash

set -euo pipefail

# AxBench GemmaScopeSAEDiffMean pipeline (Gemma-2 2B, layer 10)

uv sync
export OPENAI_API_KEY="YOUR_KEY_HERE"
export OPENAI_BASE_URL="https://us.api.openai.com"
export NP_API_KEY="YOUR_NEURONPEDIA_KEY"

uv run axbench/data/download-seed-sentences.py
uv run bash axbench/data/download-2b.sh
uv run bash axbench/data/download-alpaca.sh

uv run python axbench/scripts/sae_diffmean_train.py --config axbench/sweep/custom/2b/l10/sae_diffmean.yaml --dump_dir axbench/results/top10_sae_diffmean --overwrite_data_dir axbench/concept500/prod_2b_l10_v1/generate --max_concepts 10 --output_length 128

uv run torchrun --nproc_per_node=1 axbench/scripts/inference.py --config axbench/sweep/custom/2b/l10/sae_diffmean.yaml --dump_dir axbench/results/top10_sae_diffmean --overwrite_metadata_dir axbench/results/top10_sae_diffmean/train --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference --mode latent

uv run torchrun --nproc_per_node=1 axbench/scripts/inference.py --config axbench/sweep/custom/2b/l10/sae_diffmean.yaml --dump_dir axbench/results/top10_sae_diffmean --overwrite_metadata_dir axbench/results/top10_sae_diffmean/train --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference --overwrite_inference_dump_dir axbench/results/top10_sae_diffmean/inference --mode steering

uv run axbench/scripts/evaluate.py --config axbench/sweep/custom/2b/l10/sae_diffmean.yaml --dump_dir axbench/results/top10_sae_diffmean --overwrite_inference_dump_dir axbench/results/top10_sae_diffmean/inference --overwrite_evaluate_dump_dir axbench/results/top10_sae_diffmean/eval --mode steering

