import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from openai import AsyncOpenAI
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from axbench.utils.constants import CHAT_MODELS
from axbench.utils.model_utils import gather_residual_activations
from axbench.utils.dataset import DatasetFactory
from axbench.scripts.generate import load_concepts

import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


def _build_openai_client(base_url: str | None):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set.")
    client_kwargs = {
        "api_key": api_key,
        "timeout": 60.0,
        "max_retries": 3,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    return AsyncOpenAI(**client_kwargs)


def _select_negative_examples(negative_df: pd.DataFrame, genre: str, num: int, seed: int) -> pd.DataFrame:
    genre_pool = negative_df[negative_df["concept_genre"] == genre]
    if genre_pool.empty:
        raise ValueError(f"No negative examples available for genre '{genre}'.")
    replace = len(genre_pool) < num
    return genre_pool.sample(n=num, replace=replace, random_state=seed).copy()


def _prepare_contrastive_examples(
    dataset_factory: DatasetFactory,
    concept: str,
    num_examples: int,
    output_length: int,
    seed: int,
) -> pd.DataFrame:
    concept_genres = dataset_factory.prepare_genre_concepts([concept])
    genre = concept_genres[concept][0]

    positive_df = dataset_factory.create_train_df(
        concept,
        num_examples,
        concept_genres,
        output_length=output_length,
    )
    negative_df = _select_negative_examples(dataset_factory.negative_df, genre, num_examples, seed)

    positive_df["label"] = 1
    negative_df["label"] = 0

    combined = pd.concat([positive_df, negative_df], ignore_index=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _collect_activations(
    model,
    tokenizer,
    examples: pd.DataFrame,
    layer: int,
    device,
    is_chat_model: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    last_vectors = []
    mean_vectors = []
    labels = []

    for row in examples.itertuples():
        prompt = row.input
        completion = row.output or ""

        if is_chat_model:
            messages_prompt = [{"role": "user", "content": prompt}]
            prompt_ids = tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=True,
                add_generation_prompt=True,
            )

            messages_full = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            full_ids = tokenizer.apply_chat_template(
                messages_full,
                tokenize=True,
                add_generation_prompt=False,
            )

            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        else:
            prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            completion_text = tokenizer.eos_token.join([prompt.rstrip(), completion.rstrip()])
            combined_tokens = tokenizer(
                completion_text,
                return_tensors="pt",
                add_special_tokens=True,
            )
            prompt_tensor = {k: v.to(device) for k, v in prompt_tokens.items()}
            full_tensor = {k: v.to(device) for k, v in combined_tokens.items()}

        if is_chat_model:
            prompt_len = prompt_tensor.shape[1]
            combined_len = full_tensor.shape[1]
            attention_mask = torch.ones_like(full_tensor)
            inputs = {"input_ids": full_tensor, "attention_mask": attention_mask}
        else:
            prompt_len = prompt_tensor["input_ids"].shape[1]
            combined_len = full_tensor["input_ids"].shape[1]
            inputs = full_tensor

        output_len = combined_len - prompt_len
        if output_len <= 0:
            logger.warning("Skipping example with non-positive generated length.")
            continue

        activations = gather_residual_activations(model, layer, inputs).squeeze(0).detach().cpu().numpy()
        last_vectors.append(activations[-1])
        mean_vectors.append(activations[-output_len:].mean(axis=0))
        labels.append(row.label)

    return np.stack(last_vectors), np.stack(mean_vectors), np.array(labels)


def _plot_pca(points: np.ndarray, labels: np.ndarray, title: str, output_path: Path):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(points)
    colors = ["#1f77b4" if label == 1 else "#d62728" for label in labels]

    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.75, edgecolors="k", linewidths=0.5)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", label="Positive", markerfacecolor="#1f77b4", markersize=8),
            plt.Line2D([0], [0], marker="o", color="w", label="Negative", markerfacecolor="#d62728", markersize=8),
        ],
        loc="best",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PCA analysis of contrastive pairs at a target model layer.")
    parser.add_argument("--model_id", required=True, help="HuggingFace model identifier (e.g. google/gemma-2-2b-it).")
    parser.add_argument("--layer", type=int, required=True, help="Target layer index for activations.")
    parser.add_argument("--concept_path", required=True, help="Path to concept list (txt/csv/json).")
    parser.add_argument("--concept_index", type=int, required=True, help="Index of the concept to analyze.")
    parser.add_argument("--dataset_category", default="instruction", choices=["instruction", "continuation"])
    parser.add_argument("--num_examples", type=int, default=20, help="Number of positive (and negative) examples.")
    parser.add_argument("--output_length", type=int, default=128, help="Max generation length for each example.")
    parser.add_argument("--output_dir", required=True, help="Directory to save PCA plots and CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_bf16", action="store_true", help="Load model weights in bfloat16.")
    parser.add_argument("--openai_model", default="gpt-4o-mini", help="Model used for generating contrastive pairs.")
    parser.add_argument("--master_data_dir", default="axbench/data", help="Directory with seed sentences/instructions.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)

    logger.warning(f"Loading base model {args.model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
    )
    base_model = base_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        base_model.resize_token_embeddings(len(tokenizer))

    is_chat_model = args.model_id in CHAT_MODELS

    concepts, refs = load_concepts(args.concept_path)
    if args.concept_index < 0 or args.concept_index >= len(concepts):
        raise ValueError(f"concept_index {args.concept_index} out of range (0-{len(concepts)-1})")
    concept = concepts[args.concept_index]

    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    client = _build_openai_client(base_url)
    cache_dir = output_dir / "contrastive_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_factory = DatasetFactory(
        base_model,
        client,
        tokenizer,
        args.dataset_category,
        args.num_examples,
        args.output_length,
        str(cache_dir),
        use_cache=True,
        master_data_dir=args.master_data_dir,
        lm_model=args.openai_model,
        seed=args.seed,
    )

    examples = _prepare_contrastive_examples(
        dataset_factory,
        concept,
        args.num_examples,
        args.output_length,
        args.seed,
    )
    logger.warning(f"Prepared {len(examples)} contrastive rows (concept={concept}).")

    last_vecs, mean_vecs, labels = _collect_activations(
        base_model,
        tokenizer,
        examples,
        args.layer,
        device,
        is_chat_model,
    )

    _plot_pca(
        last_vecs,
        labels,
        f"{concept} – Last Token PCA",
        output_dir / "pca_last_token.png",
    )
    _plot_pca(
        mean_vecs,
        labels,
        f"{concept} – Mean Output Tokens PCA",
        output_dir / "pca_mean_output.png",
    )

    last_coords = PCA(n_components=2).fit_transform(last_vecs)
    mean_coords = PCA(n_components=2).fit_transform(mean_vecs)
    csv_df = pd.DataFrame({
        "label": np.concatenate([labels, labels]),
        "type": ["last"] * len(labels) + ["mean"] * len(labels),
        "pc1": np.concatenate([last_coords[:, 0], mean_coords[:, 0]]),
        "pc2": np.concatenate([last_coords[:, 1], mean_coords[:, 1]]),
    })
    csv_df.to_csv(output_dir / "pca_projection.csv", index=False)
    logger.warning(f"PCA artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()

