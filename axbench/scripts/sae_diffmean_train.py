import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import axbench
from axbench.utils.constants import EMPTY_CONCEPT, CHAT_MODELS
from axbench.utils.model_utils import get_prefix_length
from axbench.scripts.args.training_args import TrainingArgs
from axbench.scripts.args.dataset_args import DatasetArgs
from axbench.scripts.train import load_metadata, prepare_df

import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


def _load_sae_params_from_ref(ref: str):
    sae_path = ref.split("https://www.neuronpedia.org/")[-1]
    url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
    api_key = os.environ.get("NP_API_KEY")
    if not api_key:
        raise ValueError("NP_API_KEY must be set to download SAE weights.")
    response = requests.get(url, headers={"X-Api-Key": api_key})
    response.raise_for_status()
    data = response.json()
    repo_id = data["source"]["hfRepoId"]
    folder_id = data["source"]["hfFolderId"]
    if repo_id is None or folder_id is None:
        raise RuntimeError(f"Unable to resolve SAE source for {ref}")
    filename = f"{folder_id}/params.npz"
    logger.warning(f"Downloading SAE parameters from {repo_id}/{filename}")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, force_download=False)
    logger.warning(f"Loaded SAE params from {local_path}")
    with np.load(local_path) as params:
        return {k: params[k] for k in params.files}


def main():
    training_args = TrainingArgs(section="train")
    dataset_args = DatasetArgs(section="generate")

    set_seed(training_args.seed)
    random.seed(training_args.seed)

    if training_args.overwrite_data_dir and Path(training_args.overwrite_data_dir).exists():
        data_dir = Path(training_args.overwrite_data_dir)
    else:
        data_dir = Path(training_args.dump_dir) / "generate"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_parquet = data_dir / "train_data.parquet"
    metadata_path = data_dir / "metadata.jsonl"
    if not train_parquet.exists():
        raise FileNotFoundError(f"Missing train data parquet at {train_parquet}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata jsonl at {metadata_path}")

    logger.warning(f"Loading training dataframe from {train_parquet}")
    all_df = pd.read_parquet(train_parquet)
    metadata = load_metadata(str(metadata_path))

    if "concept_id" not in all_df.columns:
        raise ValueError("train_data.parquet must include a 'concept_id' column.")

    negative_df = all_df[
        (all_df["output_concept"] == EMPTY_CONCEPT) & (all_df["category"] == "negative")
    ].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if training_args.use_bf16 else None

    logger.warning(f"Loading base model {training_args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        training_args.model_name,
        torch_dtype=dtype,
    )
    model_instance = model_instance.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model_instance.resize_token_embeddings(len(tokenizer))

    is_chat_model = training_args.model_name in CHAT_MODELS
    prefix_length = 1
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    sae_params = _load_sae_params_from_ref(metadata[0]["ref"])

    diffmean_params = training_args.models.GemmaScopeSAEDiffMean
    diffmean_model = axbench.GemmaScopeSAEDiffMean(
        model_instance,
        tokenizer,
        layer=training_args.layer,
        training_args=diffmean_params,
        device=device,
        seed=training_args.seed,
    )
    diffmean_model.make_model(
        mode="train",
        embed_dim=model_instance.config.hidden_size,
        low_rank_dimension=diffmean_params.low_rank_dimension,
        dtype=dtype,
        intervention_type=getattr(diffmean_params, "intervention_type", "addition"),
        sae_params=sae_params,
        metadata_path=str(metadata_path),
    )

    dump_dir = Path(training_args.dump_dir) / "train"
    dump_dir.mkdir(parents=True, exist_ok=True)
    metadata_out_path = dump_dir / "metadata.jsonl"
    if metadata_out_path.exists():
        metadata_out_path.unlink()
    for fname in ["GemmaScopeSAEDiffMean_weight.pt", "GemmaScopeSAEDiffMean_bias.pt"]:
        fpath = dump_dir / fname
        if fpath.exists():
            fpath.unlink()

    concept_entries = metadata.copy()
    random.shuffle(concept_entries)
    if training_args.max_concepts is not None:
        concept_entries = concept_entries[: int(training_args.max_concepts)]

    logger.warning(f"Training GemmaScopeSAEDiffMean for {len(concept_entries)} concepts.")
    for entry in concept_entries:
        concept_id = entry["concept_id"]
        concept_name = entry["concept"]
        logger.warning(f"Processing concept_id={concept_id} ({concept_name})")

        concept_df = all_df[all_df["concept_id"] == concept_id].copy()
        if concept_df.empty:
            logger.warning(f"No training data for concept_id {concept_id}, skipping.")
            continue

        prepared_df = prepare_df(
            concept_df,
            negative_df,
            concept_name,
            entry,
            tokenizer,
            binarize=diffmean_params.binarize_dataset,
            train_on_negative=diffmean_params.train_on_negative,
            use_dpo_loss=getattr(training_args, "use_dpo_loss", False),
            is_chat_model=is_chat_model,
            output_length=int(training_args.output_length),
            model_name=training_args.model_name,
            max_num_of_examples=training_args.max_num_of_examples,
            steering_prompt_type=getattr(diffmean_params, "steering_prompt_type", "prepend"),
            keep_orig_axbench_format=getattr(dataset_args, "keep_orig_axbench_format", False),
        )

        if prepared_df.empty:
            logger.warning(f"Prepared dataframe empty for concept_id {concept_id}, skipping.")
            continue

        diffmean_model.train(prepared_df, prefix_length=prefix_length, metadata_path=str(metadata_path))
        diffmean_model.save(dump_dir, model_name=diffmean_model.__str__())

        with open(metadata_out_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(entry) + "\n")

        torch.cuda.empty_cache()

    logger.warning("GemmaScopeSAEDiffMean training completed.")


if __name__ == "__main__":
    main()

