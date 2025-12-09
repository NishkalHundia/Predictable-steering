"""
Difference-of-Means Line Separability Analysis.

For each concept:
1. Load contrastive pairs from training data
2. Compute activations
3. Compute difference-of-means steering direction
4. Project activations onto the steering direction
5. Normalize so positive mean = 1, negative mean = -1
6. Compute discriminability index d'
7. Correlate with steerability
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from axbench.utils.constants import CHAT_MODELS, EMPTY_CONCEPT
from axbench.utils.model_utils import gather_residual_activations, get_prefix_length

import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_metadata(metadata_path):
    """Load metadata from a JSON lines file."""
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def get_concept_data(all_df, metadata, concept_id):
    """Get positive/negative examples for a concept from pre-loaded data."""
    concept = metadata[concept_id]["concept"]
    genre = metadata[concept_id]["concept_genres_map"][concept][0]
    
    concept_df = all_df[all_df['concept_id'] == concept_id]
    positive_df = concept_df[(concept_df["output_concept"] == concept) & (concept_df["category"] == "positive")]
    
    negative_df = all_df[(all_df["output_concept"] == EMPTY_CONCEPT) & (all_df["category"] == "negative")]
    negative_df = negative_df[negative_df["concept_genre"] == genre]
    
    return positive_df, negative_df, concept


def collect_activations(model, tokenizer, examples, layer, prefix_length, device):
    """Collect activations. Returns mean output vectors and labels."""
    vectors = []
    labels = []

    for row in examples.itertuples():
        prompt = row.input
        completion = row.output or ""

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        
        prompt_only = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(prompt_only, tokenize=True, add_generation_prompt=True)
        
        output_len = len(full_ids) - len(prompt_ids)
        if output_len <= 0:
            continue

        full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        inputs = {"input_ids": full_tensor, "attention_mask": torch.ones_like(full_tensor)}
        
        activations = gather_residual_activations(model, layer, inputs).squeeze(0).detach().cpu().float().numpy()
        output_activations = activations[prefix_length:]
        
        vectors.append(output_activations[-output_len:].mean(axis=0))
        labels.append(row.label)

    return np.stack(vectors), np.array(labels)


def compute_diffmean_metrics(vectors, labels):
    """
    Compute difference-of-means separability metrics.
    Returns d' (discriminability index) and other metrics.
    """
    positive_mask = labels == 1
    negative_mask = labels == 0
    
    pos_vecs = vectors[positive_mask]
    neg_vecs = vectors[negative_mask]
    
    if len(pos_vecs) == 0 or len(neg_vecs) == 0:
        return None
    
    # Difference-of-means (steering direction)
    mean_pos = pos_vecs.mean(axis=0)
    mean_neg = neg_vecs.mean(axis=0)
    diff_mean = mean_pos - mean_neg
    norm = np.linalg.norm(diff_mean)
    
    if norm < 1e-10:
        return None
    
    steering_dir = diff_mean / norm
    
    # Project onto steering direction
    pos_proj = pos_vecs @ steering_dir
    neg_proj = neg_vecs @ steering_dir
    
    # Normalize so pos mean = 1, neg mean = -1
    pos_mean, neg_mean = pos_proj.mean(), neg_proj.mean()
    scale = 2.0 / (pos_mean - neg_mean + 1e-10)
    shift = 1.0 - scale * pos_mean
    
    pos_proj_norm = scale * pos_proj + shift
    neg_proj_norm = scale * neg_proj + shift
    
    # d' = 2 / pooled_std (since means are now 1 and -1)
    pooled_std = np.sqrt(0.5 * (pos_proj_norm.std()**2 + neg_proj_norm.std()**2))
    d_prime = 2.0 / (pooled_std + 1e-10)
    
    # Overlap: fraction crossing decision boundary at 0
    overlap = ((pos_proj_norm < 0).sum() + (neg_proj_norm > 0).sum()) / len(labels)
    
    return {
        "d_prime": d_prime,
        "overlap_fraction": overlap,
        "pos_proj": pos_proj_norm,
        "neg_proj": neg_proj_norm,
    }


def plot_separability(pos_proj, neg_proj, concept_name, concept_id, d_prime, path):
    """Plot histogram of projections."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = np.linspace(min(pos_proj.min(), neg_proj.min()) - 0.5,
                       max(pos_proj.max(), neg_proj.max()) + 0.5, 30)
    
    ax.hist(pos_proj, bins=bins, alpha=0.6, label=f'Positive (n={len(pos_proj)})', color='blue', density=True)
    ax.hist(neg_proj, bins=bins, alpha=0.6, label=f'Negative (n={len(neg_proj)})', color='red', density=True)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(1, color='blue', linestyle='--', alpha=0.7)
    ax.axvline(-1, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Projection (normalized)')
    ax.set_ylabel('Density')
    ax.set_title(f'Concept {concept_id}: {concept_name[:40]}... (d\'={d_prime:.2f})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Difference-of-means separability analysis.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--metadata_jsonl", type=str, required=True)
    parser.add_argument("--steering_parquet", type=str, required=True)
    parser.add_argument("--start_concept", type=int, default=0)
    parser.add_argument("--end_concept", type=int, default=None)
    parser.add_argument("--plot_top_k", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load ALL data upfront (ONCE)
    logger.info("Loading data files...")
    steering_df = pd.read_parquet(args.steering_parquet)
    train_df = pd.read_parquet(args.train_parquet)
    metadata = load_metadata(args.metadata_jsonl)
    logger.info(f"Loaded {len(steering_df)} steering rows, {len(train_df)} train rows")

    # Pre-compute avg best LM score per concept
    best_per_prompt = steering_df.groupby(['concept_id', 'input_id'])['SteeringVector_LMJudgeEvaluator'].max()
    avg_best_scores = best_per_prompt.groupby('concept_id').mean().to_dict()

    # Get concept IDs
    all_concept_ids = sorted(steering_df['concept_id'].unique())
    if args.end_concept is not None:
        concept_ids = [c for c in all_concept_ids if args.start_concept <= c < args.end_concept]
    else:
        concept_ids = [c for c in all_concept_ids if c >= args.start_concept]
    logger.info(f"Processing {len(concept_ids)} concepts")

    # Load model
    logger.info(f"Loading model {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    is_chat_model = args.model_id in CHAT_MODELS
    prefix_length = get_prefix_length(tokenizer) if is_chat_model else 1

    # Process each concept
    results = []
    projection_data = {}
    
    for concept_id in tqdm(concept_ids, desc="Processing concepts"):
        if concept_id >= len(metadata):
            continue
            
        try:
            positive_df, negative_df, concept_name = get_concept_data(train_df, metadata, concept_id)
        except Exception as e:
            continue
        
        if positive_df.empty or negative_df.empty:
            continue
        
        if len(positive_df) > args.num_examples:
            positive_df = positive_df.sample(n=args.num_examples, random_state=args.seed)
        if len(negative_df) > args.num_examples:
            negative_df = negative_df.sample(n=args.num_examples, random_state=args.seed)
        
        positive_df = positive_df.copy()
        negative_df = negative_df.copy()
        positive_df["label"] = 1
        negative_df["label"] = 0
        examples = pd.concat([positive_df, negative_df], ignore_index=True)
        
        try:
            vectors, labels = collect_activations(model, tokenizer, examples, args.layer, prefix_length, device)
        except Exception as e:
            continue
        
        if len(vectors) < 4:
            continue
        
        metrics = compute_diffmean_metrics(vectors, labels)
        if metrics is None:
            continue
        
        results.append({
            "concept_id": concept_id,
            "concept_name": concept_name,
            "n_examples": len(vectors),
            "d_prime": metrics["d_prime"],
            "overlap_fraction": metrics["overlap_fraction"],
            "avg_best_lm_score": avg_best_scores.get(concept_id, np.nan),
        })
        
        projection_data[concept_id] = {
            "pos_proj": metrics["pos_proj"],
            "neg_proj": metrics["neg_proj"],
            "concept_name": concept_name,
            "d_prime": metrics["d_prime"],
        }

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "diffmean_results.csv", index=False)
    
    # Compute correlations
    valid_df = results_df.dropna(subset=["avg_best_lm_score"])
    
    if len(valid_df) < 10:
        logger.warning(f"Only {len(valid_df)} valid points")
        return
    
    corr_dprime, p1 = stats.pearsonr(valid_df["d_prime"], valid_df["avg_best_lm_score"])
    corr_overlap, p2 = stats.pearsonr(valid_df["overlap_fraction"], valid_df["avg_best_lm_score"])
    
    correlations = {
        "d_prime_vs_lm": {"r": corr_dprime, "p": p1},
        "overlap_vs_lm": {"r": corr_overlap, "p": p2},
        "n": len(valid_df),
    }
    
    with open(output_dir / "correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)
    
    print("\n" + "="*60)
    print(f"DIFF-MEAN SEPARABILITY vs STEERABILITY (n={len(valid_df)} concepts)")
    print("="*60)
    print(f"d' vs LM score:      r={corr_dprime:.4f} (p={p1:.2e})")
    print(f"Overlap vs LM score: r={corr_overlap:.4f} (p={p2:.2e})")
    print("="*60)
    
    # Summary plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(valid_df["d_prime"], valid_df["avg_best_lm_score"], alpha=0.5, s=15)
    axes[0].set_xlabel("d'")
    axes[0].set_ylabel("Avg Best LM Score")
    axes[0].set_title(f"d' vs Steerability (r={corr_dprime:.3f})")
    
    axes[1].scatter(valid_df["overlap_fraction"], valid_df["avg_best_lm_score"], alpha=0.5, s=15)
    axes[1].set_xlabel("Overlap Fraction")
    axes[1].set_ylabel("Avg Best LM Score")
    axes[1].set_title(f"Overlap vs Steerability (r={corr_overlap:.3f})")
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_plots.png", dpi=150)
    plt.close()
    
    # Individual plots for top/bottom concepts
    sorted_df = valid_df.sort_values("d_prime", ascending=False)
    
    for _, row in sorted_df.head(args.plot_top_k).iterrows():
        cid = row["concept_id"]
        if cid in projection_data:
            d = projection_data[cid]
            plot_separability(d["pos_proj"], d["neg_proj"], d["concept_name"], cid, d["d_prime"],
                            plots_dir / f"top_{cid}.png")
    
    for _, row in sorted_df.tail(args.plot_top_k).iterrows():
        cid = row["concept_id"]
        if cid in projection_data:
            d = projection_data[cid]
            plot_separability(d["pos_proj"], d["neg_proj"], d["concept_name"], cid, d["d_prime"],
                            plots_dir / f"bottom_{cid}.png")
    
    logger.info(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
