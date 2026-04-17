"""
Activation Spread: Per-Prompt Token-Position Dispersion of Raw Activations.

For each training prompt i at layer l, for positive and negative responses separately:
  Given response token activations A = [a_1, ..., a_T] (T = number of response tokens),
  compute the mean pairwise cosine distance between all token positions:

    spread_i = mean_{t < t'} [ 1 - cos_sim(a_t, a_{t'}) ]

  Higher spread means token activations point in more diverse directions.

Outputs:
  {sweep_dir}/activation_spread/
    positive/
      layer_{N}/
        per_prompt_spread.csv
        bar_plot.png
      mean_across_layers.png
    negative/
      layer_{N}/
        per_prompt_spread.csv
        bar_plot.png
      mean_across_layers.png
    summary.json

Usage:
    uv run python axbench/scripts/projection_tactics/activation_spread.py \\
        --behavior corrigible-neutral-HHH \\
        --model_name google/gemma-2-9b-it \\
        --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep \\
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
# Helpers
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
        result[layer] = layer_acts[layer].float()[0, q_len:f_len].cpu()
    return result


def compute_spread(token_acts: torch.Tensor) -> float:
    """Mean pairwise cosine distance among token-position activations.

    Args:
        token_acts: [n_tokens, hidden]

    Returns:
        spread = mean_{t < t'} [ 1 - cos_sim(a_t, a_{t'}) ]
        Returns 0.0 if fewer than 2 tokens.
    """
    n = token_acts.shape[0]
    if n < 2:
        return 0.0

    # Normalize rows
    norms = token_acts.norm(dim=1, keepdim=True).clamp(min=1e-12)
    normed = token_acts / norms

    # Full cosine similarity matrix [n, n]
    sim_matrix = normed @ normed.T

    # Upper triangle (i < j)
    indices = torch.triu_indices(n, n, offset=1)
    pairwise_sims = sim_matrix[indices[0], indices[1]]

    # Spread = mean(1 - cos_sim)
    return (1.0 - pairwise_sims).mean().item()


# ============================================================================
# Core computation
# ============================================================================
@torch.no_grad()
def compute_activation_spread(model, tokenizer, train_dataset_path,
                              target_layers, device):
    """
    For each training prompt, compute the activation spread (mean pairwise
    cosine distance) across response token positions, separately for positive
    and negative responses.

    Returns:
      pos_results: {layer: {"spreads": [float, ...], "mean": float, "std": float}}
      neg_results: same structure
      prompts: list of question strings
    """
    use_chat = supports_chat_template(tokenizer)
    model.eval()

    with open(train_dataset_path, encoding="utf-8") as f:
        train_data = json.load(f)

    logger.warning(f"Computing activation spread for {len(train_data)} pairs")

    pos_spreads = {l: [] for l in target_layers}
    neg_spreads = {l: [] for l in target_layers}
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
            pos_spreads[layer].append(compute_spread(pos_acts[layer]))
            neg_spreads[layer].append(compute_spread(neg_acts[layer]))

        if pair_idx % 5 == 0:
            torch.cuda.empty_cache()

    # Aggregate
    pos_results = {}
    neg_results = {}
    for layer in target_layers:
        ps = pos_spreads[layer]
        ns = neg_spreads[layer]
        pos_results[layer] = {
            "spreads": ps, "mean": np.mean(ps), "std": np.std(ps),
        }
        neg_results[layer] = {
            "spreads": ns, "mean": np.mean(ns), "std": np.std(ns),
        }
        logger.warning(
            f"  Layer {layer:2d}: pos spread = {np.mean(ps):.4f} +/- {np.std(ps):.4f}, "
            f"neg spread = {np.mean(ns):.4f} +/- {np.std(ns):.4f}"
        )

    return pos_results, neg_results, prompts


# ============================================================================
# Plots
# ============================================================================
def plot_bar(spreads, mean_val, layer, behavior, polarity, output_path):
    """Bar plot: one bar per prompt showing its activation spread."""
    n = len(spreads)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.25), 5))

    indices = np.arange(n)
    max_spread = max(spreads) if spreads else 1.0
    # Color: low spread = blue, high spread = red
    colors = plt.cm.coolwarm(np.array(spreads) / max(max_spread, 1e-6))

    ax.bar(indices, spreads, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=mean_val, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {mean_val:.4f}")

    ax.set_xlabel("Prompt index", fontsize=10)
    ax.set_ylabel("Spread (mean pairwise cosine distance)", fontsize=10)
    ax.set_title(
        f"{behavior} - Layer {layer}: Activation Spread ({polarity})",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, None)
    ax.legend(fontsize=10)

    if n <= 50:
        ax.set_xticks(indices)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mean_across_layers(layer_means, behavior, polarity, output_path):
    """Summary plot: mean spread across layers."""
    layers = sorted(layer_means.keys())
    means = [layer_means[l]["mean"] for l in layers]
    stds = [layer_means[l]["std"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(layers, means, yerr=stds, marker="o", linewidth=2, capsize=4,
                color="#2E86AB", markersize=7)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean spread (mean pairwise cosine distance)", fontsize=10)
    ax.set_title(
        f"{behavior}: Activation Spread Across Layers ({polarity})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(layers)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Per-prompt activation spread across token positions"
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
    out_dir = sweep_dir / "activation_spread"
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
        pos_results = {}
        neg_results = {}
        prompts = None
        for layer in target_layers:
            for polarity, results_dict in [("positive", pos_results),
                                           ("negative", neg_results)]:
                layer_dir = out_dir / polarity / f"layer_{layer}"
                csv_path = layer_dir / "per_prompt_spread.csv"
                if not csv_path.exists():
                    logger.error(f"Missing {csv_path}; run without --replot_only first")
                    sys.exit(1)
                df = pd.read_csv(csv_path)
                spreads = df["spread"].tolist()
                results_dict[layer] = {
                    "spreads": spreads,
                    "mean": np.mean(spreads),
                    "std": np.std(spreads),
                }
                if prompts is None and "question" in df.columns:
                    prompts = df["question"].tolist()
        logger.warning(f"Loaded activation spreads for {len(target_layers)} layers")
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

        pos_results, neg_results, prompts = compute_activation_spread(
            model, tokenizer, args.train_dataset_path, target_layers, device,
        )

        del model
        torch.cuda.empty_cache()

    # -- Save CSVs and per-layer plots for both polarities --------------------
    for polarity, results in [("positive", pos_results), ("negative", neg_results)]:
        pol_dir = out_dir / polarity
        pol_dir.mkdir(parents=True, exist_ok=True)

        layer_means = {}
        for layer in target_layers:
            layer_dir = pol_dir / f"layer_{layer}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            spreads = results[layer]["spreads"]
            mean_val = results[layer]["mean"]
            std_val = results[layer]["std"]
            layer_means[layer] = {"mean": mean_val, "std": std_val}

            df = pd.DataFrame({
                "prompt_idx": list(range(len(spreads))),
                "spread": spreads,
                "question": prompts[:len(spreads)] if prompts else [""] * len(spreads),
            })
            df.to_csv(layer_dir / "per_prompt_spread.csv", index=False)

            plot_bar(spreads, mean_val, layer, args.behavior, polarity,
                     layer_dir / "bar_plot.png")

            logger.warning(
                f"  [{polarity}] Layer {layer:2d}: mean = {mean_val:.4f}, "
                f"std = {std_val:.4f}, saved to {layer_dir}"
            )

        # Summary plot across layers for this polarity
        plot_mean_across_layers(layer_means, args.behavior, polarity,
                                pol_dir / "mean_across_layers.png")

    # -- Summary JSON ---------------------------------------------------------
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "sweep_dir": str(sweep_dir),
        "train_dataset_path": args.train_dataset_path,
        "n_prompts": len(pos_results[target_layers[0]]["spreads"]),
        "spread_metric": "mean pairwise cosine distance: mean_{t<t'}[1 - cos_sim(a_t, a_{t'})]",
        "layers": {
            int(l): {
                "positive": {
                    "mean_spread": pos_results[l]["mean"],
                    "std_spread": pos_results[l]["std"],
                },
                "negative": {
                    "mean_spread": neg_results[l]["mean"],
                    "std_spread": neg_results[l]["std"],
                },
            }
            for l in target_layers
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()
