# Predictable Steering

Reproduction code for **_Predictable Steering: Leveraging Geometric Proxies for Layer Selection and Per-Instance Success_** (ICML submission).

We extend the discriminability index `d'` (Braun et al., 2025), a geometric, training-time proxy for steering reliability. We demonstrate two further uses of `d'`:

- **Layer selection:** For a fixed dataset, `d'` predicts the optimal steering layer, enabling principled layer selection without exhaustive sweeps.
- **Individual test point success:** For datasets with high `d'`, the position of a test point along the steering vector direction predicts whether steering will succeed on that example.

The end-to-end pipeline is implemented in a single self-contained script: [`scripts/mcqa_projection_link.py`](scripts/mcqa_projection_link.py).

---

## What's in this repo

```
predictable-steering/
├── README.md                       # this file
├── pyproject.toml / requirements.txt
├── run_all.sh                      # end-to-end driver (5 behaviors → CSVs → paper figs + tables)
├── scripts/
│   ├── mcqa_projection_link.py     # the experiment: trains DiffMean vectors, runs phase A/B/val,
│   │                               # writes CSVs + JSON + plots/test/, plots/val/.
│   ├── make_paper_plots.py         # renders paper-style PNGs from those CSVs (no Modal needed)
│   └── make_paper_tables.py        # rebuilds Tables 1 & 2 (Pearson correlations) from CSVs
└── datasets/
    ├── raw/<behavior>/dataset.json            # contrastive A/B training pairs (Rimsky et al. 2024)
    └── test/<behavior>/test_dataset_ab.json   # 50 held-out test prompts per behavior
```

The five behaviors used in the paper:

* `sycophancy`
* `survival-instinct`
* `corrigible-neutral-HHH`  (referred to in the paper as `corrigibility`)
* `hallucination`
* `myopic-reward`

---

## Setup

Requires Python ≥ 3.10 and a CUDA GPU with enough memory to load Gemma-2-9B-IT in bf16 (≈ 20 GB). Tested on H100.

```bash
# pip
pip install -r requirements.txt

# or uv
uv pip install -r requirements.txt
```

Authenticate with Hugging Face once so the model weights can be downloaded:

```bash
huggingface-cli login
```

---

## Reproduction

The simplest path:

```bash
./run_all.sh
```

This loops over all five behaviors, calling `scripts/mcqa_projection_link.py` once per behavior, then renders paper-style figures and Tables 1 + 2.

**Outputs land here**:

```
results/mcqa_projection_link/gemma-2-9b-it/<behavior>/
    train_selection.json          # canonical seed → train/val raw indices
    steering_state.pt             # per-layer μ⁺, μ⁻, v
    dprime.json                   # per-layer d' on training projections
    train_projections.json        # per-layer scalar coords (κ) for train ±
    per_prompt_results.csv        # test split: κ_a, κ_a_postgen*, steered_correct_α  per (layer, prompt)
    val_prompt_results.csv        # validation split (same schema)
    per_layer_summary.csv         # one row per layer (incl. val-best-α columns)
    cross_layer_corr.csv          # predictor × target Pearson / Spearman
    summary.json                  # everything bundled
    plots/test/   plots/val/      # exploratory plots produced inline by the main script

paper_plots/<behavior>/
    <short>_steer_new.png         # per-layer accuracy + d'           (Figs 1, 4)
    <short>_mcc_new.png           # per-layer MCC + d'                (Figs 3, 7)
    projection_hist_postgen_layer_<L>.png   # train + post-gen κ histograms (Figs 2, 5, 6)
paper_plots/tables.md             # Tables 1 + 2 (markdown)
paper_plots/tables.csv            # same, machine-readable
```

### Running individual behaviors

```bash
python scripts/mcqa_projection_link.py \
    --behavior myopic-reward \
    --model_name google/gemma-2-9b-it \
    --layers 10-32 \
    --factors 1,2,3,5,10 \
    --batch_size 16 \
    --seed 42
```

The script is **resumable**: re-running with the same arguments reuses cached CSVs/tensors. Add `--force_recompute` (or `--force_recompute_val`) to override.

Then re-render figures / tables for that behavior:

```bash
python scripts/make_paper_plots.py --behavior myopic-reward --target_layer 21
python scripts/make_paper_tables.py
```

`make_paper_plots.py --all` covers all five behaviors at once. `--target_layer` defaults to layer 21 (layer 19 for `hallucination`), matching the paper.

---

## Mapping figures → outputs

| Paper figure / table | Script output |
|---|---|
| Figure 1 (myopic-reward, survival-instinct accuracy + d') | `paper_plots/{myopic,survival}/{*}_steer_new.png` |
| Figure 4 (sycophancy, hallucination, corrigibility — same)  | `paper_plots/{sycophancy,hallucination,corrigible}/{*}_steer_new.png` |
| Figure 2 (κ histograms at layer 21, myopic + survival)      | `paper_plots/{myopic,survival}/projection_hist_postgen_layer_21.png` |
| Figure 5 (κ histograms — corrigibility/halluc/sycophancy)   | `paper_plots/{corrigible,hallucination,sycophancy}/projection_hist_postgen_layer_{21,19,21}.png` |
| Figure 6 (myopic-reward unsteered, layer 21)                | (α=0 panel of) `paper_plots/myopic-reward/projection_hist_postgen_layer_21.png` |
| Figure 3 (per-layer MCC + d', myopic + survival)            | `paper_plots/{myopic,survival}/{*}_mcc_new.png` |
| Figure 7 (corrigibility / halluc / sycophancy MCC + d')     | `paper_plots/{corrigible,hallucination,sycophancy}/{*}_mcc_new.png` |
| Table 1 (per-layer d' vs per-layer test acc @ val-α*)        | `paper_plots/tables.md` (rebuilt from `per_layer_summary.csv`) |
| Table 2 (per-layer d' vs per-layer MCC of sign κ @ val-α*)   | `paper_plots/tables.md` (same) |

The Pearson correlations underlying Tables 1 and 2 are also stored in each behavior's `summary.json` (`pearson_dprime_vs_test_steered_acc_at_val_best_alpha`, `pearson_dprime_vs_sign_kappa_mcc_val_best_on_test`) and in `cross_layer_corr.csv`.

---

## Method (in brief)

For each behavior `b` and each contrastive pair `(prompt, r⁺, r⁻)`:

1. **Phase 0** — extract residual-stream activations at the answer-letter position for both `r⁺` and `r⁻` at every layer ℓ ∈ {10, …, 32}; the steering vector is `v⁽ℓ⁾ = μ⁺ − μ⁻`. Compute `d'⁽ℓ⁾` from the scalar projections of training pairs onto the difference-of-means line.
2. **Phase A** — unsteered greedy decode at the `(` position for every test prompt (and validation prompt). Record `κ_a` from the residual stream at the teacher-forced answer letter.
3. **Phase A2** — second forward on `prefix + [chosen-letter token]`; record `κ_a_postgen` per layer.
4. **Phase B** — at every (layer, α ∈ {1, 2, 3, 5, 10}), steered greedy decode + a second unsteered forward to record `κ_a_postgen_α`.
5. **Validation** — best α* per layer is chosen on validation prompts (sampled from `raw \ train`).
6. **Cross-layer** — Pearson(`d'`, test-acc @ val-α*) gives Table 1; Pearson(`d'`, MCC(sign `κ_a_postgen_α*`, steered_correct)) gives Table 2.

See [`scripts/mcqa_projection_link.py`](scripts/mcqa_projection_link.py) for the full implementation; the section headers in the file map 1-to-1 with the phases above.

---

## Datasets

The contrastive pairs and held-out MCQA prompts come from [Rimsky et al., 2024 — _Steering Llama 2 via Contrastive Activation Addition_](https://aclanthology.org/2024.acl-long.828/). The `datasets/` directory contains exactly the five behaviors used in the paper.

* `datasets/raw/<behavior>/dataset.json` — full set of contrastive A/B pairs. The script reproducibly samples 300 train + 50 val pairs (via the seed in `train_selection.json`).
* `datasets/test/<behavior>/test_dataset_ab.json` — held-out 50-prompt test split.

---

## References

```bibtex
@inproceedings{rimsky2024steering,
  title={Steering Llama 2 via Contrastive Activation Addition},
  author={Rimsky, Nina and Gabrieli, Nick and Schulz, Julian and Tong, Meg and Hubinger, Evan and Turner, Alexander Matt},
  booktitle={ACL},
  year={2024}
}

@inproceedings{braun2025understanding,
  title={Understanding (un)reliability of steering vectors in language models},
  author={Braun, Joschka and Eickhoff, Carsten and Krueger, David and Bahrainian, Seyed Ali and Krasheninnikov, Dmitrii},
  booktitle={ICLR 2025 Workshop on Building Trust in Language Models and Applications},
  year={2025}
}

@misc{wu2025axbenchsteeringllmssimple,
      title={AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders}, 
      author={Zhengxuan Wu and Aryaman Arora and Atticus Geiger and Zheng Wang and Jing Huang and Dan Jurafsky and Christopher D. Manning and Christopher Potts},
      year={2025},
      eprint={2501.17148},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.17148}, 
}
```
