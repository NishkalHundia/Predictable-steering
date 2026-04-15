"""
Directional Agreement: Per-Prompt Cosine Similarity with Steering Vector.

For each training prompt i at layer l, computes:
  Delta_i = mean(pos_response_acts) - mean(neg_response_acts)
  cos_sim(Delta_i, s^l)

where s^l is the DiffMean steering vector (computed from training data).

This measures how well each individual prompt's activation difference aligns
with the overall steering direction. A narrow, high-mean distribution means
the steering vector consistently captures individual examples' behavioral axes.

Outputs:
  {sweep_dir}/cosine_similarity/
    layer_{N}/
      cosine_similarities.csv
      bar_plot.png
      histogram.png
    summary.json

Usage:
    uv run python axbench/scripts/directional_agreement.py \\
        --behavior corrigible-neutral-HHH \\
        --model_name google/gemma-2-9b-it \\
        --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep \\
        --train_dataset_path datasets/generated/gemma-2-9b-it/corrigible-neutral-HHH/train_contrastive.json
"""
import sys
import json
import shutil
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


def _get_mean_response_act(model, tokenizer, target_layers, question, answer,
                           use_chat, device):
    """Forward pass -> mean response-token activation at each target layer."""
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
        resp_acts = layer_acts[layer].float()[0, q_len:f_len]
        result[layer] = resp_acts.mean(dim=0).cpu()
    return result


# ============================================================================
# Core computation
# ============================================================================
@torch.no_grad()
def compute_directional_agreement(model, tokenizer, train_dataset_path,
                                  target_layers, device):
    """
    For each training prompt, compute:
      Delta_i = mean_pos_act - mean_neg_act  (per-prompt activation difference)
      s^l = mean(Delta_i)                    (steering vector = DiffMean)
      cos_sim(Delta_i, s^l)                  (directional agreement)

    Returns:
      results: {layer: {"cos_sims": [float, ...], "mean_cos_sim": float,
                         "steering_vector": tensor, "deltas": [tensor, ...]}}
      prompts: list of question strings (for labeling)
    """
    use_chat = supports_chat_template(tokenizer)
    model.eval()

    with open(train_dataset_path, encoding="utf-8") as f:
        train_data = json.load(f)

    logger.warning(f"Computing directional agreement for {len(train_data)} pairs")

    # Collect per-prompt activation differences at each layer
    deltas = {l: [] for l in target_layers}
    prompts = []

    for pair_idx, item in enumerate(tqdm(train_data, desc="Collecting activations")):
        question = item["question"]
        pos_answer = item["answer_matching_behavior"]
        neg_answer = item["answer_not_matching_behavior"]

        pos_acts = _get_mean_response_act(
            model, tokenizer, target_layers, question, str(pos_answer),
            use_chat, device,
        )
        neg_acts = _get_mean_response_act(
            model, tokenizer, target_layers, question, str(neg_answer),
            use_chat, device,
        )

        if pos_acts is None or neg_acts is None:
            continue

        prompts.append(question)
        for layer in target_layers:
            delta = pos_acts[layer] - neg_acts[layer]
            deltas[layer].append(delta)

        if pair_idx % 5 == 0:
            torch.cuda.empty_cache()

    # Compute cosine similarities per layer
    results = {}
    for layer in target_layers:
        delta_stack = torch.stack(deltas[layer])  # [n_prompts, hidden]
        steering_vec = delta_stack.mean(dim=0)     # s^l = mean of differences

        # cos_sim(Delta_i, s^l) for each prompt
        cos_sims = torch.nn.functional.cosine_similarity(
            delta_stack, steering_vec.unsqueeze(0), dim=1
        ).tolist()

        mean_cos = np.mean(cos_sims)
        std_cos = np.std(cos_sims)

        results[layer] = {
            "cos_sims": cos_sims,
            "mean_cos_sim": mean_cos,
            "std_cos_sim": std_cos,
            "steering_norm": steering_vec.norm().item(),
            "deltas": delta_stack,
        }

        logger.warning(
            f"  Layer {layer:2d}: mean cos_sim = {mean_cos:.4f} +/- {std_cos:.4f}, "
            f"n = {len(cos_sims)}"
        )

    return results, prompts


def compute_pairwise_similarities(results, target_layers):
    """
    Compute cos_sim(Delta_i, Delta_j) for all pairs i != j.

    Returns:
      pairwise_results: {layer: {"cos_sims": [float, ...], "mean_cos_sim": float, ...}}
    """
    pairwise_results = {}
    for layer in target_layers:
        delta_stack = results[layer]["deltas"]  # [n, hidden]
        n = delta_stack.shape[0]

        # Normalize all deltas
        norms = delta_stack.norm(dim=1, keepdim=True).clamp(min=1e-12)
        normalized = delta_stack / norms

        # Full cosine similarity matrix: [n, n]
        sim_matrix = normalized @ normalized.T

        # Extract upper triangle (i < j), excludes diagonal
        indices = torch.triu_indices(n, n, offset=1)
        pairwise_sims = sim_matrix[indices[0], indices[1]].tolist()

        mean_cos = np.mean(pairwise_sims)
        std_cos = np.std(pairwise_sims)

        pairwise_results[layer] = {
            "cos_sims": pairwise_sims,
            "mean_cos_sim": mean_cos,
            "std_cos_sim": std_cos,
            "n_pairs": len(pairwise_sims),
        }

        logger.warning(
            f"  Layer {layer:2d}: pairwise mean cos_sim = {mean_cos:.4f} +/- {std_cos:.4f}, "
            f"n_pairs = {len(pairwise_sims)}"
        )

    return pairwise_results


# ============================================================================
# Plots
# ============================================================================
def plot_bar(cos_sims, mean_cos, layer, behavior, output_path):
    """Bar plot: one bar per prompt, colored by cos_sim value."""
    n = len(cos_sims)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.25), 5))

    indices = np.arange(n)
    colors = plt.cm.RdYlGn((np.array(cos_sims) + 1) / 2)  # map [-1, 1] to colormap

    ax.bar(indices, cos_sims, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=mean_cos, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {mean_cos:.4f}")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)

    ax.set_xlabel("Prompt index", fontsize=10)
    ax.set_ylabel("cos_sim(Delta_i, s)", fontsize=10)
    ax.set_title(f"{behavior} - Layer {layer}: Per-Prompt Directional Agreement",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=10)

    if n <= 50:
        ax.set_xticks(indices)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_histogram(cos_sims, mean_cos, layer, behavior, output_path, pairwise=False):
    """Histogram of cos_sim values with mean line."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(cos_sims, bins=25, color="#2E86AB", edgecolor="white",
            linewidth=0.5, alpha=0.8)
    ax.axvline(x=mean_cos, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {mean_cos:.4f}")

    if pairwise:
        ax.set_xlabel("cos_sim(Delta_i, Delta_j)", fontsize=10)
        title_suffix = "Pairwise"
    else:
        ax.set_xlabel("cos_sim(Delta_i, s)", fontsize=10)
        title_suffix = "Steering Vector"
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"{behavior} - Layer {layer}: Directional Agreement ({title_suffix})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-1.05, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mean_across_layers(layer_means, behavior, output_path):
    """Summary plot: mean cos_sim across layers."""
    layers = sorted(layer_means.keys())
    means = [layer_means[l]["mean"] for l in layers]
    stds = [layer_means[l]["std"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(layers, means, yerr=stds, marker="o", linewidth=2, capsize=4,
                color="#2E86AB", markersize=7)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean cos_sim(Delta_i, s)", fontsize=10)
    ax.set_title(f"{behavior}: Mean Directional Agreement Across Layers",
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
        description="Per-prompt directional agreement (cosine similarity with steering vector)"
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
    parser.add_argument("--pairwise", action="store_true",
                        help="Compute pairwise cos_sim(Delta_i, Delta_j) instead of steering vector similarity")
    parser.add_argument("--replot_only", action="store_true",
                        help="Skip model inference; reuse saved CSVs")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    if args.pairwise:
        out_dir = sweep_dir / "cosine_similarity_pairwise"
    else:
        out_dir = sweep_dir / "cosine_similarity"
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
        # Reload from saved CSVs
        results = {}
        prompts = None
        for layer in target_layers:
            layer_dir = out_dir / f"layer_{layer}"
            csv_path = layer_dir / "cosine_similarities.csv"
            if not csv_path.exists():
                logger.error(f"Missing {csv_path}; run without --replot_only first")
                sys.exit(1)
            df = pd.read_csv(csv_path)
            cos_sims = df["cos_sim"].tolist()
            results[layer] = {
                "cos_sims": cos_sims,
                "mean_cos_sim": np.mean(cos_sims),
                "std_cos_sim": np.std(cos_sims),
            }
            if prompts is None:
                prompts = df["question"].tolist()
        logger.warning(f"Loaded cosine similarities for {len(target_layers)} layers")
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

        results, prompts = compute_directional_agreement(
            model, tokenizer, args.train_dataset_path, target_layers, device,
        )

        del model
        torch.cuda.empty_cache()

    # -- Pairwise mode: compute pairwise similarities from deltas -----------
    if args.pairwise and not args.replot_only:
        plot_results = compute_pairwise_similarities(results, target_layers)
    elif args.pairwise and args.replot_only:
        plot_results = results  # already loaded from CSVs
    else:
        plot_results = results

    # -- Save CSVs and generate plots per layer ----------------------------
    layer_means = {}
    for layer in target_layers:
        layer_dir = out_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        if args.pairwise and not args.replot_only:
            cos_sims = plot_results[layer]["cos_sims"]
            mean_cos = plot_results[layer]["mean_cos_sim"]
            std_cos = plot_results[layer]["std_cos_sim"]
        else:
            cos_sims = plot_results[layer]["cos_sims"]
            mean_cos = plot_results[layer]["mean_cos_sim"]
            std_cos = plot_results[layer]["std_cos_sim"]

        layer_means[layer] = {"mean": mean_cos, "std": std_cos}

        # Save CSV
        if args.pairwise:
            df = pd.DataFrame({
                "pair_idx": list(range(len(cos_sims))),
                "cos_sim": cos_sims,
            })
        else:
            df = pd.DataFrame({
                "prompt_idx": list(range(len(cos_sims))),
                "cos_sim": cos_sims,
                "question": prompts[:len(cos_sims)] if prompts else [""] * len(cos_sims),
            })
        df.to_csv(layer_dir / "cosine_similarities.csv", index=False)

        # Plots — pairwise only gets histogram (too many pairs for bar plot)
        if args.pairwise:
            plot_histogram(cos_sims, mean_cos, layer, args.behavior,
                           layer_dir / "histogram.png", pairwise=True)
        else:
            plot_bar(cos_sims, mean_cos, layer, args.behavior,
                     layer_dir / "bar_plot.png")
            plot_histogram(cos_sims, mean_cos, layer, args.behavior,
                           layer_dir / "histogram.png")

        logger.warning(
            f"  Layer {layer:2d}: mean = {mean_cos:.4f}, std = {std_cos:.4f}, "
            f"saved to {layer_dir}"
        )

    # -- Summary plot across layers ----------------------------------------
    plot_mean_across_layers(layer_means, args.behavior,
                            out_dir / "mean_across_layers.png")

    # -- Summary JSON ------------------------------------------------------
    mode = "pairwise" if args.pairwise else "steering"
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "mode": mode,
        "sweep_dir": str(sweep_dir),
        "train_dataset_path": args.train_dataset_path,
        "n_prompts": len(results[target_layers[0]]["cos_sims"]),
        "layers": {
            int(l): {
                "mean_cos_sim": layer_means[l]["mean"],
                "std_cos_sim": layer_means[l]["std"],
            }
            for l in target_layers
        },
    }
    if args.pairwise:
        summary["n_pairs_per_layer"] = plot_results[target_layers[0]].get("n_pairs", 0)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()
