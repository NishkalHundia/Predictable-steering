"""
Token-Position Directional Agreement: Per-Token Cosine Similarity with Steering Vector.

For each training prompt i at layer l, computes:
  delta_i^t = pos_act^t - neg_act^t   for t = 0..min(len_pos, len_neg)-1
  avg_cos_sim_i = mean_t[ cos_sim(delta_i^t, s^l) ]

where s^l is the DiffMean steering vector (mean of per-prompt mean deltas).

This measures how consistently each token position's activation difference aligns
with the overall steering direction, averaged across positions.

Outputs:
  {sweep_dir}/cosine_similarity_token_position/
    layer_{N}/
      per_prompt_avg_cossim.csv
      bar_plot.png
    mean_across_layers.png
    summary.json

Usage:
    uv run python axbench/scripts/projection_tactics/token_position_agreement.py \
        --behavior corrigible-neutral-HHH \
        --model_name google/gemma-2-9b-it \
        --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep \
        --train_dataset_path datasets/generated/gemma-2-9b-it/corrigible-neutral-HHH/train_contrastive.json
"""
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")

BEHAVIORS = [
    "survival-instinct", "myopic-reward", "corrigible-neutral-HHH",
    "hallucination", "sycophancy", "refusal", "coordinate-other-ais",
]


# ============================================================================
# Helpers (shared with directional_agreement.py)
# ============================================================================
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def discover_layers(sweep_dir: Path) -> list[int]:
    layers = []
    for d in sweep_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            layers.append(int(d.name.split("_")[1]))
    return sorted(layers)


def gather_multi_layer_activations(model, target_layers: list[int], inputs: dict):
    layer_acts = {}
    handles = []
    for layer_idx in target_layers:
        def _make_hook(l):
            def hook(mod, inp, out):
                layer_acts[l] = out[0].detach()
            return hook
        h = model.model.layers[layer_idx].register_forward_hook(
            _make_hook(layer_idx), always_call=True
        )
        handles.append(h)
    _ = model.forward(**inputs)
    for h in handles:
        h.remove()
    return layer_acts


def _format_texts(tokenizer, use_chat, question, answer):
    if use_chat:
        q_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question},
             {"role": "assistant", "content": str(answer)}],
            tokenize=False, add_generation_prompt=False,
        )
    else:
        q_text = question
        full_text = question + "\n\n" + str(answer)
    return q_text, full_text


def _get_response_token_acts(model, tokenizer, target_layers, question, answer,
                             use_chat, device):
    """Forward pass -> per-token response activations at each target layer.

    Returns {layer: tensor[n_response_tokens, hidden]} or None if response is empty.
    """
    q_text, full_text = _format_texts(tokenizer, use_chat, question, answer)

    full_inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=1024,
    ).to(device)
    q_inputs = tokenizer(
        q_text, return_tensors="pt", truncation=True, max_length=1024,
    ).to(device)

    q_len = q_inputs["input_ids"].shape[1]
    f_len = full_inputs["input_ids"].shape[1]
    if f_len <= q_len:
        return None

    layer_acts = gather_multi_layer_activations(
        model, target_layers,
        {"input_ids": full_inputs["input_ids"],
         "attention_mask": full_inputs["attention_mask"]},
    )

    result = {}
    for layer in target_layers:
        # Keep all response token activations (not averaged)
        result[layer] = layer_acts[layer].float()[0, q_len:f_len].cpu()
    return result


# ============================================================================
# Core computation
# ============================================================================
@torch.no_grad()
def compute_token_position_agreement(model, tokenizer, train_dataset_path,
                                     target_layers, device):
    """
    For each training prompt:
      1. Get per-token response activations for pos and neg.
      2. Truncate both to min(len_pos, len_neg).
      3. Compute delta at each token position.
      4. Compute steering vector s^l = mean over all prompts of mean(delta per prompt).
      5. For each prompt, compute mean_t[ cos_sim(delta_t, s^l) ].

    Returns:
      results: {layer: {"avg_cos_sims": [float, ...], "mean": float, "std": float}}
      prompts: list of question strings
    """
    use_chat = supports_chat_template(tokenizer)
    model.eval()

    with open(train_dataset_path, encoding="utf-8") as f:
        train_data = json.load(f)

    logger.warning(f"Computing token-position agreement for {len(train_data)} pairs")

    # Phase 1: collect per-token deltas for each prompt at each layer
    # per_prompt_deltas[layer] = list of tensors, each [n_tokens_i, hidden]
    per_prompt_deltas = {l: [] for l in target_layers}
    prompts = []

    for pair_idx, item in enumerate(tqdm(train_data, desc="Collecting activations")):
        question = item["question"]
        pos_answer = item["answer_matching_behavior"]
        neg_answer = item["answer_not_matching_behavior"]

        pos_acts = _get_response_token_acts(
            model, tokenizer, target_layers, question, str(pos_answer),
            use_chat, device,
        )
        neg_acts = _get_response_token_acts(
            model, tokenizer, target_layers, question, str(neg_answer),
            use_chat, device,
        )

        if pos_acts is None or neg_acts is None:
            continue

        prompts.append(question)
        for layer in target_layers:
            p = pos_acts[layer]  # [len_pos, hidden]
            n = neg_acts[layer]  # [len_neg, hidden]
            min_len = min(p.shape[0], n.shape[0])
            delta = p[:min_len] - n[:min_len]  # [min_len, hidden]
            per_prompt_deltas[layer].append(delta)

        if pair_idx % 5 == 0:
            torch.cuda.empty_cache()

    # Phase 2: compute steering vector per layer (mean of per-prompt mean deltas)
    steering_vectors = {}
    for layer in target_layers:
        # Mean delta per prompt, then mean across prompts
        mean_deltas = torch.stack([d.mean(dim=0) for d in per_prompt_deltas[layer]])
        steering_vectors[layer] = mean_deltas.mean(dim=0)  # [hidden]

    # Phase 3: for each prompt, compute avg cosine sim across token positions
    results = {}
    for layer in target_layers:
        sv = steering_vectors[layer].unsqueeze(0)  # [1, hidden]
        avg_cos_sims = []

        for delta in per_prompt_deltas[layer]:
            # delta: [n_tokens, hidden], sv: [1, hidden]
            token_cos_sims = torch.nn.functional.cosine_similarity(delta, sv, dim=1)
            avg_cos_sims.append(token_cos_sims.mean().item())

        mean_val = np.mean(avg_cos_sims)
        std_val = np.std(avg_cos_sims)

        results[layer] = {
            "avg_cos_sims": avg_cos_sims,
            "mean": mean_val,
            "std": std_val,
        }

        logger.warning(
            f"  Layer {layer:2d}: mean avg_cos_sim = {mean_val:.4f} +/- {std_val:.4f}, "
            f"n = {len(avg_cos_sims)}"
        )

    return results, prompts


# ============================================================================
# Plots
# ============================================================================
def plot_bar(avg_cos_sims, mean_val, layer, behavior, output_path):
    """Bar plot: one bar per prompt showing its average token-position cosine sim."""
    n = len(avg_cos_sims)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.25), 5))

    indices = np.arange(n)
    colors = plt.cm.RdYlGn((np.array(avg_cos_sims) + 1) / 2)

    ax.bar(indices, avg_cos_sims, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=mean_val, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {mean_val:.4f}")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)

    ax.set_xlabel("Prompt index", fontsize=10)
    ax.set_ylabel("Avg token-position cos_sim(delta_t, s)", fontsize=10)
    ax.set_title(f"{behavior} - Layer {layer}: Token-Position Directional Agreement",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=10)

    if n <= 50:
        ax.set_xticks(indices)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mean_across_layers(layer_means, behavior, output_path):
    """Summary plot: mean avg_cos_sim across layers."""
    layers = sorted(layer_means.keys())
    means = [layer_means[l]["mean"] for l in layers]
    stds = [layer_means[l]["std"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(layers, means, yerr=stds, marker="o", linewidth=2, capsize=4,
                color="#2E86AB", markersize=7)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean avg token-position cos_sim", fontsize=10)
    ax.set_title(f"{behavior}: Token-Position Directional Agreement Across Layers",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Per-token-position directional agreement (avg cosine similarity with steering vector)"
    )
    parser.add_argument("--behavior", type=str, required=True, choices=BEHAVIORS)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Path to sweep output dir (contains layer_N/ subdirs)")
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="Path to train_contrastive.json")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers (default: discover from sweep_dir)")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--replot_only", action="store_true",
                        help="Skip model inference; reuse saved CSVs")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = sweep_dir / "cosine_similarity_token_position"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.layers:
        target_layers = sorted(int(x.strip()) for x in args.layers.split(","))
    else:
        target_layers = discover_layers(sweep_dir)
    if not target_layers:
        logger.error(f"No valid layers found in {sweep_dir}")
        sys.exit(1)
    logger.warning(f"Target layers: {target_layers}")

    if args.replot_only:
        results = {}
        prompts = None
        for layer in target_layers:
            layer_dir = out_dir / f"layer_{layer}"
            csv_path = layer_dir / "per_prompt_avg_cossim.csv"
            if not csv_path.exists():
                logger.error(f"Missing {csv_path}; run without --replot_only first")
                sys.exit(1)
            df = pd.read_csv(csv_path)
            avg_cos_sims = df["avg_cos_sim"].tolist()
            results[layer] = {
                "avg_cos_sims": avg_cos_sims,
                "mean": np.mean(avg_cos_sims),
                "std": np.std(avg_cos_sims),
            }
            if prompts is None:
                prompts = df["question"].tolist()
        logger.warning(f"Loaded token-position cosine sims for {len(target_layers)} layers")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning(f"Device: {device}")
        logger.warning(f"Loading model {args.model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, model_max_length=1024, padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.use_bf16 else None,
            device_map=device,
        )
        model.eval()

        results, prompts = compute_token_position_agreement(
            model, tokenizer, args.train_dataset_path, target_layers, device,
        )

        del model
        torch.cuda.empty_cache()

    # -- Save CSVs and per-layer plots ----------------------------------------
    layer_means = {}
    for layer in target_layers:
        layer_dir = out_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        avg_cos_sims = results[layer]["avg_cos_sims"]
        mean_val = results[layer]["mean"]
        std_val = results[layer]["std"]
        layer_means[layer] = {"mean": mean_val, "std": std_val}

        df = pd.DataFrame({
            "prompt_idx": list(range(len(avg_cos_sims))),
            "avg_cos_sim": avg_cos_sims,
            "question": prompts[:len(avg_cos_sims)] if prompts else [""] * len(avg_cos_sims),
        })
        df.to_csv(layer_dir / "per_prompt_avg_cossim.csv", index=False)

        plot_bar(avg_cos_sims, mean_val, layer, args.behavior,
                 layer_dir / "bar_plot.png")

        logger.warning(
            f"  Layer {layer:2d}: mean = {mean_val:.4f}, std = {std_val:.4f}, "
            f"saved to {layer_dir}"
        )

    # -- Summary plot across layers -------------------------------------------
    plot_mean_across_layers(layer_means, args.behavior,
                            out_dir / "mean_across_layers.png")

    # -- Summary JSON ---------------------------------------------------------
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "mode": "token_position",
        "sweep_dir": str(sweep_dir),
        "train_dataset_path": args.train_dataset_path,
        "n_prompts": len(results[target_layers[0]]["avg_cos_sims"]),
        "layers": {
            int(l): {
                "mean_avg_cos_sim": layer_means[l]["mean"],
                "std_avg_cos_sim": layer_means[l]["std"],
            }
            for l in target_layers
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()
