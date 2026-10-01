# ARR open-ended Qwen3.5-9B runs

These jobs are for the ARR review of the paper. They are **not** written
into the Gemma production folders (`open_ended_projection_last_token/`,
`open_ended_projection_average/`, `open_ended_projection_link_prompted/`).

## Three evaluations

| Human name | Method | Script | Volume root |
|---|---|---|---|
| last-token open-ended ARR | Vanilla DiffMean at last response token | `axbench/scripts/open_ended_projection_link.py --diffmean_mode last_token` | `/vol/last_token_open_ended_ARR/Qwen3.5-9B/<behavior>/` |
| average-token open-ended ARR | Vanilla DiffMean, mean over response tokens | `axbench/scripts/open_ended_projection_link.py --diffmean_mode avg_token` | `/vol/average_token_open_ended_ARR/Qwen3.5-9B/<behavior>/` |
| open-ended ARR (prompted / cue DiffMean) | Cue+question DiffMean; no response teacher-forcing | `axbench/scripts/open_ended_projection_link_prompted.py` | `/vol/open_ended_ARR/Qwen3.5-9B/<behavior>/` |

Contrastive pairs (shared by all three):

```
/vol/prompted_datasets/Qwen3.5-9B/generated/<behavior>/
```

## Launch

```bash
MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_qwen35_oe_sweep.py
MODAL_PROFILE=nishkalhundia modal app logs qwen35-oe-arr
```

Protocol matches Gemma OE except layer range: Qwen3.5-9B is 32 layers
(valid 0–31), so we use layers 10–31. Factors 0,1,2,3,5,10, batch 32,
fluency ≥ 1.0, min 28 fluent examples, greedy OE decode, thinking off.
One A100-80GB per behavior; each worker does datagen then the three evals
in order.

Behaviors: myopic-reward, sycophancy, hallucination, survival-instinct,
corrigible-neutral-HHH.

## MCQA ARR

Single pipeline — there is no last-token / avg-token / cue-DiffMean split.
Teacher-force appends `" (A)"` / `" (B)"` onto the thinking-off generation
prefix (empty `<think></think>`) so train/eval are not OOD vs generate().

| Human name | Script | Volume root |
|---|---|---|
| MCQA ARR | `axbench/scripts/mcqa_projection_link.py` | `/vol/mcqa_ARR/Qwen3.5-9B/<behavior>/` |

```
/vol/mcqa_ARR/Qwen3.5-9B/<behavior>/
  train_selection.json
  steering_state.pt
  dprime.json
  train_projections.json
  per_prompt_results.csv
  val_prompt_results.csv
  per_layer_summary.csv
  cross_layer_corr.csv
  summary.json
  plots/test/
  plots/val/
```

Train: `datasets/raw/<behavior>/dataset.json`  
Test: `datasets/test/<behavior>/test_dataset_ab.json`  
Layers 10–31, factors 1,2,3,5,10, thinking off, A100-80GB × 5, profile `nishkalhundia`.

```bash
MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_qwen35_mcqa_smoke.py
MODAL_PROFILE=nishkalhundia modal app logs qwen35-mcqa-smoke

MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_qwen35_mcqa_sweep.py
MODAL_PROFILE=nishkalhundia modal app logs qwen35-mcqa-arr
```

Do **not** write to `/vol/mcqa_projection_link/` (Gemma production).

## Llama-3.1-8B-Instruct ARR MCQA

Same `mcqa_projection_link.py` as Gemma (completed-turn chat template, DiffMean
at the letter token, greedy at `'('`). No think tags, so no Qwen append path.
Secret: `huggingface-secret`. Layers 10–31.

```
/vol/mcqa_ARR/Llama-3.1-8B-Instruct/<behavior>/
```

```bash
MODAL_PROFILE=nishkalhundia python3 -m modal run --detach modal_runs/run_modal_llama31_mcqa_sweep.py
MODAL_PROFILE=nishkalhundia python3 -m modal app logs llama31-mcqa-arr
```

## Prompt-only cue site (pre_think)

Default prompted ARR reads cue DiffMean at the last generation-prompt token
(`</think>` / trailing newlines). The Gemma analogue is the token immediately
before `<think>` (assistant header):

```
/vol/open_ended_ARR_prethink/Qwen3.5-9B/<behavior>/
```

`--cue_site pre_think` in `open_ended_projection_link_prompted.py`. Reuses ARR
contrastive pairs. Does not touch `/vol/open_ended_ARR/`. Myopic-reward is
done; the other four are `modal_runs/run_modal_qwen35_oe_prompted_prethink.py`
(app `qwen35-oe-prethink`).

`plots/last_token/` and `plots/avg_token/` in this folder are **test-time κ
readouts** on the generated response (Table 3), not last-token / mean-pooled
DiffMean trainings.

## Table 2b when editing the canvas

OE scripts do **not** implement a val split. α\* on disk is test argmax
**match rate**. Paper leftover (and what the canvas Table 2b must use):

1. Hold out even `prompt_idx` as val (16), odd as test (16) from
   `per_prompt_results.csv`.
2. Fluency ≥ 1 on ≥ 14 of 16 (28/32 scaled).
3. Per layer, α\* = argmax val **mean behavior score** (not match rate).
4. y = **test** mean score at that α\*. Pearson r = corr(train d′, y).
5. Drop layers with no valid α\* or no valid test cell; show n when n ≠ 22.

Apply this to last-token, mean-pooled, last_prompt, **and** pre_think. Table 2a
stays peak fluency-valid score on all 32 prompts. Table 3 stays unsteered sign
MCC. Do not overwrite production CSVs; write analysis next to the canvas.

When the remaining pre_think behaviors finish, extend the canvas myopic cue-site
table (currently last_prompt vs pre_think) with the other four using this
protocol — do not paste `dprime_best_alpha_corr.csv` (that is leaky match-rate α\*).

## Llama-3.1-8B-Instruct ARR OE

Same three ARR roots, model short `Llama-3.1-8B-Instruct`. Secret:
`huggingface-secret`. Do not reuse Qwen contrastive pairs. Prompt-only uses
default `last_prompt` (assistant header; Llama has no `<think>`). Layers 10–31.
Table 2b still uses the post-hoc val split above.

```
/vol/prompted_datasets/Llama-3.1-8B-Instruct/generated/<behavior>/
/vol/last_token_open_ended_ARR/Llama-3.1-8B-Instruct/<behavior>/
/vol/average_token_open_ended_ARR/Llama-3.1-8B-Instruct/<behavior>/
/vol/open_ended_ARR/Llama-3.1-8B-Instruct/<behavior>/
```

```bash
MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_llama31_oe_smoke.py
MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_llama31_oe_sweep.py
```

Smoke writes `myopic-reward_smoke` only; delete those folders after a successful
smoke before trusting the full sweep.
