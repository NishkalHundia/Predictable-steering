"""
Train DiffMean steering vectors across a range of layers on MCQA data.

This is the multi-layer analogue of train_caa_mcqa.py. Exactly one pass over the
training data is done per example-pair direction (matching / not-matching); within
that pass, activations at every target layer are captured in a single forward via
per-layer hooks. For every layer we compute and save:

  * DiffMean_weight.pt   - steering vector v = mu_pos - mu_neg, shape [1, hidden]
  * DiffMean_bias.pt     - zeros([1])
  * mu_pos.pt / mu_neg.pt - per-class centroids (needed for kappa_a at test time)
  * separability.json    - {dprime, auroc, norm}
  * config.json          - full per-layer training config

MCQA data comes from `datasets/raw/<behavior>/dataset.json`. Since behavior dataset
sizes vary wildly (340 for corrigible-neutral-HHH, 20 184 for sycophancy), pass
--max_examples to randomly subsample the training pairs for a fair comparison.

Usage:
    uv run python axbench/scripts/mcqa_train_diffmean_sweep.py \
        --behavior corrigible-neutral-HHH \
        --model_name google/gemma-2-9b-it \
        --layers 10-32 \
        --max_examples 300 \
        --seed 42 \
        --output_dir results/mcqa_sweep/gemma-2-9b-it/corrigible-neutral-HHH
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)


BEHAVIOR_DATASET_MAP = {
    "sycophancy": "datasets/raw/sycophancy/dataset.json",
    "survival-instinct": "datasets/raw/survival-instinct/dataset.json",
    "corrigible-neutral-HHH": "datasets/raw/corrigible-neutral-HHH/dataset.json",
    "hallucination": "datasets/raw/hallucination/dataset.json",
    "refusal": "datasets/raw/refusal/dataset.json",
    "myopic-reward": "datasets/raw/myopic-reward/dataset.json",
    "coordinate-other-ais": "datasets/raw/coordinate-other-ais/dataset.json",
}


# ============================================================================
# Helpers
# ============================================================================
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def parse_layer_range(spec: str) -> list:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(x) for x in spec.split(","))


def gather_multi_layer_activations(model, target_layers, inputs):
    """One forward pass, capture outputs of every target decoder layer."""
    layer_acts = {}
    handles = []
    for layer_idx in target_layers:
        def _make_hook(l):
            def hook(mod, inp, out):
                layer_acts[l] = out[0].detach()
            return hook
        h = model.model.layers[layer_idx].register_forward_hook(
            _make_hook(layer_idx), always_call=True,
        )
        handles.append(h)
    _ = model.forward(**inputs)
    for h in handles:
        h.remove()
    return layer_acts


def find_answer_token_offset(tokenizer) -> int:
    """Offset from sequence end to the answer-letter (A/B) token.

    Matches train_caa_mcqa.py: on Gemma-2 chat the sequence ends
    ``... ( A ) <end_of_turn> \n``, so offset = 4. For base models, 2.
    """
    if supports_chat_template(tokenizer):
        messages = [
            {"role": "user", "content": "Test question?\n\nChoices:\n (A) Yes\n (B) No\n\nAnswer:"},
            {"role": "assistant", "content": " (A)"},
        ]
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )
    else:
        tokens = tokenizer.encode("Test? (A)")
    for offset in range(1, min(len(tokens), 10)):
        token_str = tokenizer.decode([tokens[-offset]]).strip()
        if token_str == "A":
            logger.warning(f"Auto-detected answer token at offset -{offset} from end")
            return offset
    fallback = 4 if supports_chat_template(tokenizer) else 2
    logger.warning(f"Could not auto-detect offset; using fallback -{fallback}")
    return fallback


def compute_dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    gap = abs(pos.mean() - neg.mean())
    pooled = np.sqrt(0.5 * (pos.var() + neg.var()))
    if pooled < 1e-12:
        return 0.0
    return float(gap / pooled)


def compute_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = np.concatenate([pos, neg])
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


# ============================================================================
# Data prep
# ============================================================================
def prepare_training_rows(data, tokenizer):
    """
    One row per (example, answer-class). full_text = prompt + assistant(' (X)').
    """
    rows = []
    for item in data:
        rows.append({
            "question": item["question"],
            "answer": item["answer_matching_behavior"],
            "labels": 1,
        })
        rows.append({
            "question": item["question"],
            "answer": item["answer_not_matching_behavior"],
            "labels": 0,
        })
    df = pd.DataFrame(rows)

    def _format(row):
        if supports_chat_template(tokenizer):
            msgs = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]
            tokens = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
            )[1:]  # drop BOS to match train_caa_mcqa.py
            return pd.Series({"full_text": tokenizer.decode(tokens)})
        return pd.Series({"full_text": row["question"] + row["answer"]})

    fmt = df.apply(_format, axis=1)
    df["full_text"] = fmt["full_text"]
    return df


# ============================================================================
# Training (all layers in one pass)
# ============================================================================
@torch.no_grad()
def train_all_layers(model, tokenizer, df, target_layers, device, answer_token_offset):
    """
    Single pass over every (example, answer-class); per layer accumulate pos/neg sums
    and per-example projections for d'/AUROC.
    """
    model.eval()
    hidden = model.config.hidden_size

    pos_sum = {l: torch.zeros(hidden, dtype=torch.float32, device=device) for l in target_layers}
    neg_sum = {l: torch.zeros(hidden, dtype=torch.float32, device=device) for l in target_layers}
    pos_acts = {l: [] for l in target_layers}
    neg_acts = {l: [] for l in target_layers}
    pos_count = 0
    neg_count = 0

    first = True
    for idx in tqdm(range(len(df)), desc="Collecting activations"):
        row = df.iloc[idx]
        inputs = tokenizer(
            row["full_text"], return_tensors="pt", truncation=True, max_length=512,
        ).to(device)
        seq_len = int(inputs["attention_mask"].sum().item())
        letter_pos = seq_len - answer_token_offset
        if letter_pos < 0:
            continue

        if first:
            logger.warning("\n=== SANITY CHECK: last 6 tokens of first example ===")
            for off in range(6, 0, -1):
                p = seq_len - off
                if p >= 0:
                    tok_id = inputs["input_ids"][0, p].item()
                    marker = " <-- letter" if off == answer_token_offset else ""
                    logger.warning(
                        f"  pos -{off}: '{tokenizer.decode([tok_id])}' (id={tok_id}){marker}"
                    )
            first = False

        layer_acts = gather_multi_layer_activations(
            model, target_layers,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )
        for l in target_layers:
            act = layer_acts[l][0, letter_pos].float()  # [hidden]
            if row["labels"] == 1:
                pos_sum[l] += act
                pos_acts[l].append(act.cpu())
            else:
                neg_sum[l] += act
                neg_acts[l].append(act.cpu())

        if row["labels"] == 1:
            pos_count += 1
        else:
            neg_count += 1

        if idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.warning(f"Collected {pos_count} positive / {neg_count} negative")

    results = {}
    for l in target_layers:
        mu_pos = (pos_sum[l] / max(pos_count, 1)).cpu()
        mu_neg = (neg_sum[l] / max(neg_count, 1)).cpu()
        steering = mu_pos - mu_neg

        s_norm_sq = float(steering.dot(steering))
        mu = 0.5 * (mu_pos + mu_neg)
        if s_norm_sq < 1e-12 or not pos_acts[l] or not neg_acts[l]:
            pos_proj = np.zeros(len(pos_acts[l]))
            neg_proj = np.zeros(len(neg_acts[l]))
        else:
            pos_proj = np.array([
                2.0 * float((a - mu).dot(steering)) / s_norm_sq for a in pos_acts[l]
            ])
            neg_proj = np.array([
                2.0 * float((a - mu).dot(steering)) / s_norm_sq for a in neg_acts[l]
            ])

        dprime = compute_dprime(pos_proj, neg_proj)
        auroc = compute_auroc(pos_proj, neg_proj)

        results[l] = {
            "steering": steering,
            "mu_pos": mu_pos,
            "mu_neg": mu_neg,
            "dprime": dprime,
            "auroc": auroc,
            "norm": float(steering.norm()),
        }
        logger.warning(
            f"Layer {l:2d}: norm={results[l]['norm']:.4f}  "
            f"d'={dprime:.3f}  AUROC={auroc:.3f}"
        )
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train MCQA DiffMean vectors across a range of layers.",
    )
    parser.add_argument("--behavior", type=str, required=True,
                        choices=list(BEHAVIOR_DATASET_MAP.keys()))
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--layers", type=str, default="10-32",
                        help="'10-32' or '10,15,20'")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Override MCQA raw dataset path.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: results/mcqa_sweep/<model>/<behavior>")
    parser.add_argument("--max_examples", type=int, default=300,
                        help="Max MCQA pairs (randomly sampled with --seed). Default 300.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer_token_offset", type=int, default=None,
                        help="Override auto-detected answer-letter offset.")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_layers = parse_layer_range(args.layers)
    logger.warning(f"Target layers: {target_layers}")

    dataset_path = args.dataset_path or BEHAVIOR_DATASET_MAP[args.behavior]
    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1]
        output_dir = Path("results") / "mcqa_sweep" / model_short / args.behavior
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    offset = args.answer_token_offset
    if offset is None:
        offset = find_answer_token_offset(tokenizer)
    logger.warning(f"Answer letter offset: -{offset}")

    logger.warning(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        device_map=device,
    )
    model.eval()

    logger.warning(f"Loading MCQA from {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)
    logger.warning(f"  {len(data)} total pairs")

    if args.max_examples is not None and args.max_examples < len(data):
        indices = random.sample(range(len(data)), args.max_examples)
        data = [data[i] for i in indices]
        logger.warning(f"  Randomly subsampled to {len(data)} pairs (seed={args.seed})")

    df = prepare_training_rows(data, tokenizer)
    logger.warning(f"Prepared {len(df)} rows (pos + neg)")

    layer_results = train_all_layers(model, tokenizer, df, target_layers, device, offset)

    # Save per-layer artifacts
    stats = []
    for l in target_layers:
        layer_dir = output_dir / f"layer_{l}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        res = layer_results[l]

        torch.save(res["steering"].unsqueeze(0), layer_dir / "DiffMean_weight.pt")
        torch.save(torch.zeros(1), layer_dir / "DiffMean_bias.pt")
        torch.save(res["mu_pos"], layer_dir / "mu_pos.pt")
        torch.save(res["mu_neg"], layer_dir / "mu_neg.pt")

        sep = {"dprime": res["dprime"], "auroc": res["auroc"], "norm": res["norm"]}
        with open(layer_dir / "separability.json", "w") as f:
            json.dump(sep, f, indent=2)

        cfg = {
            "model_name": args.model_name,
            "layer": l,
            "behavior": args.behavior,
            "dataset_path": dataset_path,
            "num_train_pairs": len(data),
            "max_examples": args.max_examples,
            "seed": args.seed,
            "answer_token_offset": offset,
            "steering_vector_norm": res["norm"],
            "dprime": res["dprime"],
            "auroc": res["auroc"],
            "method": "mcqa_diffmean_sweep",
        }
        with open(layer_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        stats.append({
            "layer": l,
            "dprime": res["dprime"],
            "auroc": res["auroc"],
            "norm": res["norm"],
        })

    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump({
            "behavior": args.behavior,
            "model_name": args.model_name,
            "layers": target_layers,
            "max_examples": args.max_examples,
            "seed": args.seed,
            "answer_token_offset": offset,
            "num_train_pairs": len(data),
            "dataset_path": dataset_path,
        }, f, indent=2)

    logger.warning(f"\nSweep training complete. Artifacts in {output_dir}")


if __name__ == "__main__":
    main()
