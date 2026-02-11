"""
Measure KL divergence between steered and unsteered first-token distributions.

For each test prompt and each steering factor, this script:
1. Runs a forward pass through the base model (no steering) to get P_base
2. Runs a forward pass through the steered model (via pyvene) to get P_steered
3. Computes KL(P_steered || P_base) at the first generated token position
4. Optionally reports which tokens shifted most

This is a model-internal metric that requires NO external judge — it directly 
measures how much the steering vector perturbs the next-token distribution.

Usage:
    # Single concept:
    uv run python axbench/scripts/eval_kl_divergence.py \
        --behavior sycophancy \
        --model_name google/gemma-2-9b-it \
        --layer 20 \
        --steering_vector_path results/gemma-2-9b-it/sycophancy-open-ended/DiffMean_weight.pt \
        --test_dataset_path datasets/generated/sycophancy/test_contrastive.json \
        --output_dir results/gemma-2-9b-it/sycophancy-open-ended/kl_eval

    # All AxBench concepts in a directory (from prepare_axbench_concepts.py):
    uv run python axbench/scripts/eval_kl_divergence.py \
        --behavior axbench \
        --model_name google/gemma-2-9b-it \
        --layer 20 \
        --axbench_dir results/gemma-2-9b-it/axbench_concepts \
        --output_dir results/gemma-2-9b-it/axbench_concepts
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyvene import IntervenableConfig, IntervenableModel
from axbench.models.interventions import AdditionIntervention
from axbench.utils.model_utils import get_prefix_length
from axbench.utils.constants import CHAT_MODELS

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)


# ============================================================================
# Behaviors (same set as eval_caa_open_ended.py, used only for labeling)
# ============================================================================
BEHAVIORS = [
    "sycophancy", "hallucination", "survival-instinct",
    "corrigible-neutral-HHH", "refusal", "coordinate-other-ais",
    "myopic-reward",
]


class SteeringModel:
    """Model with steering vector intervention, extended for KL divergence measurement."""

    def __init__(self, model, tokenizer, layer, steering_vector, device):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.device = device

        # Create intervention
        self.ax = AdditionIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=1
        )

        if steering_vector.dim() == 1:
            steering_vector = steering_vector.unsqueeze(0)

        self.ax.proj.weight.data = steering_vector.to(device)
        self.ax.proj.bias.data = torch.zeros(1, device=device)
        self.ax.to(device)
        self.ax.eval()

        self.vector_norm = self.ax.proj.weight.data.norm().item()
        logger.warning(f"Steering vector norm: {self.vector_norm:.4f}")

        # Create interventable model
        ax_config = IntervenableConfig(representations=[{
            "layer": layer,
            "component": f"model.layers[{layer}].output",
            "low_rank_dimension": 1,
            "intervention": self.ax
        }])
        self.ax_model = IntervenableModel(ax_config, model)
        self.ax_model.set_device(device)

    @torch.no_grad()
    def compute_first_token_kl(
        self, prompts, factor, prefix_length=1, top_k_tokens=10
    ):
        """
        Compute KL(P_steered || P_base) at the first generated token position.

        For each prompt, we:
        1. Forward pass through base model -> logits at last prompt position
        2. Forward pass through steered model -> logits at last prompt position
        3. Softmax both -> KL divergence

        Args:
            prompts: list of prompt strings
            factor: steering factor (scalar multiplier for the steering vector)
            prefix_length: prefix length for chat models
            top_k_tokens: number of top boosted/suppressed tokens to report

        Returns:
            dict with:
                kl_divergences: list of KL values per prompt
                js_divergences: list of JS divergence values per prompt
                top_boosted: list of (token, prob_change) for most boosted tokens
                top_suppressed: list of (token, prob_change) for most suppressed tokens
        """
        self.ax.eval()
        self.tokenizer.padding_side = "left"

        batch_size = len(prompts)

        # Tokenize
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        # ------------------------------------------------------------------
        # 1) Unsteered forward pass (raw model, no intervention)
        # ------------------------------------------------------------------
        base_outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=False,
        )
        # logits at the last position predict the first generated token
        # Shape: [batch, vocab_size]
        base_first_logits = base_outputs.logits[:, -1, :].float()

        # ------------------------------------------------------------------
        # 2) Steered forward pass (through pyvene IntervenableModel)
        # ------------------------------------------------------------------
        mag = torch.tensor([factor] * batch_size, dtype=torch.float32).to(self.device)
        idx = torch.tensor([0] * batch_size, dtype=torch.long).to(self.device)
        max_acts = torch.tensor([1.0] * batch_size, dtype=torch.float32).to(self.device)

        _, steered_outputs = self.ax_model(
            base={
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
            },
            unit_locations=None,
            subspaces=[{
                "idx": idx,
                "mag": mag,
                "max_act": max_acts,
            }],
            use_cache=False,
        )
        steered_first_logits = steered_outputs.logits[:, -1, :].float()

        # ------------------------------------------------------------------
        # 3) Compute distributions and divergences
        # ------------------------------------------------------------------
        base_log_probs = F.log_softmax(base_first_logits, dim=-1)
        steered_log_probs = F.log_softmax(steered_first_logits, dim=-1)
        base_probs = base_log_probs.exp()
        steered_probs = steered_log_probs.exp()

        # KL(P_steered || P_base) per example
        # = sum over vocab of P_steered * (log P_steered - log P_base)
        kl_per_example = F.kl_div(
            base_log_probs, steered_probs, reduction='none', log_target=False
        ).sum(dim=-1)  # [batch]

        # Jensen-Shannon divergence (symmetric, bounded [0, ln2])
        m_probs = 0.5 * (base_probs + steered_probs)
        m_log_probs = m_probs.log()
        js_per_example = 0.5 * (
            F.kl_div(m_log_probs, base_probs, reduction='none', log_target=False).sum(dim=-1) +
            F.kl_div(m_log_probs, steered_probs, reduction='none', log_target=False).sum(dim=-1)
        )

        # ------------------------------------------------------------------
        # 4) Top-k token shifts (which tokens got boosted / suppressed most)
        # ------------------------------------------------------------------
        prob_diff = steered_probs - base_probs  # [batch, vocab]

        all_top_boosted = []
        all_top_suppressed = []
        for i in range(batch_size):
            diff_i = prob_diff[i]
            # Top boosted
            top_vals, top_ids = diff_i.topk(top_k_tokens, largest=True)
            boosted = [
                (self.tokenizer.decode([tid]), round(tval.item(), 6))
                for tid, tval in zip(top_ids, top_vals)
            ]
            # Top suppressed
            bot_vals, bot_ids = diff_i.topk(top_k_tokens, largest=False)
            suppressed = [
                (self.tokenizer.decode([tid]), round(tval.item(), 6))
                for tid, tval in zip(bot_ids, bot_vals)
            ]
            all_top_boosted.append(boosted)
            all_top_suppressed.append(suppressed)

        return {
            "kl_divergences": kl_per_example.cpu().tolist(),
            "js_divergences": js_per_example.cpu().tolist(),
            "top_boosted": all_top_boosted,
            "top_suppressed": all_top_suppressed,
        }


def evaluate_kl_divergence(
    steering_model: SteeringModel,
    test_data: list,
    behavior: str,
    steering_factors: list,
    tokenizer,
    prefix_length: int = 1,
    batch_size: int = 4,
    top_k_tokens: int = 10,
) -> list:
    """Evaluate KL divergence at different steering factors."""

    all_results = []

    # Prepare prompts (same logic as eval_caa_open_ended.py)
    prompts = []
    use_chat_template = getattr(tokenizer, "chat_template", None) not in (None, "")
    for item in test_data:
        question = item["question"]
        if use_chat_template:
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = question + "\n\nAnswer:"
        prompts.append(prompt)

    for factor in steering_factors:
        logger.warning(f"\n{'='*50}")
        logger.warning(f"Computing KL divergence for steering factor: {factor}")
        logger.warning(f"{'='*50}")

        all_kl = []
        all_js = []
        all_boosted = []
        all_suppressed = []

        for batch_start in tqdm(
            range(0, len(prompts), batch_size), desc=f"Factor {factor}"
        ):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            result = steering_model.compute_first_token_kl(
                batch_prompts, factor, prefix_length, top_k_tokens
            )

            all_kl.extend(result["kl_divergences"])
            all_js.extend(result["js_divergences"])
            all_boosted.extend(result["top_boosted"])
            all_suppressed.extend(result["top_suppressed"])

        # Build per-prompt results
        for i, item in enumerate(test_data):
            result = {
                "question_idx": i,
                "question": item["question"],
                "behavior": behavior,
                "steering_factor": factor,
                "effective_strength": factor * steering_model.vector_norm,
                "vector_norm": steering_model.vector_norm,
                "kl_divergence": all_kl[i],
                "js_divergence": all_js[i],
                "top_boosted_tokens": json.dumps(all_boosted[i]),
                "top_suppressed_tokens": json.dumps(all_suppressed[i]),
            }
            all_results.append(result)

        # Print summary for this factor
        kl_arr = np.array(all_kl)
        js_arr = np.array(all_js)
        logger.warning(
            f"Factor {factor}: "
            f"KL mean={kl_arr.mean():.4f} ± {kl_arr.std():.4f}, "
            f"median={np.median(kl_arr):.4f}, "
            f"JS mean={js_arr.mean():.6f} ± {js_arr.std():.6f}"
        )

        # Print sample token shifts
        if len(all_boosted) > 0:
            logger.warning(f"  Sample prompt: {test_data[0]['question'][:80]}...")
            logger.warning(f"  Top boosted:    {all_boosted[0][:5]}")
            logger.warning(f"  Top suppressed: {all_suppressed[0][:5]}")

    return all_results


def run_single_concept(
    model, tokenizer, device, behavior, layer, steering_vector_path,
    test_dataset_path, output_dir, steering_factors, prefix_length,
    batch_size, top_k_tokens,
):
    """Run KL divergence evaluation for a single concept. Returns summary rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load steering vector
    logger.warning(f"Loading steering vector from {steering_vector_path}...")
    steering_vector = torch.load(steering_vector_path, map_location=device)
    logger.warning(f"Steering vector shape: {steering_vector.shape}")

    # Load test dataset
    logger.warning(f"Loading test dataset from {test_dataset_path}...")
    with open(test_dataset_path, 'r') as f:
        test_data = json.load(f)
    logger.warning(f"Loaded {len(test_data)} test examples")

    # Create steering model
    steering_model = SteeringModel(model, tokenizer, layer, steering_vector, device)

    # Run evaluation
    all_results = evaluate_kl_divergence(
        steering_model, test_data, behavior, steering_factors,
        tokenizer, prefix_length, batch_size, top_k_tokens
    )

    # Save results
    results_df = pd.DataFrame(all_results)

    parquet_path = output_dir / "kl_results.parquet"
    results_df.to_parquet(parquet_path, engine='pyarrow')
    logger.warning(f"Saved per-prompt results to {parquet_path}")

    # Summary per factor
    summary_rows = []
    for factor in steering_factors:
        factor_df = results_df[results_df["steering_factor"] == factor]
        kl_vals = factor_df["kl_divergence"]
        js_vals = factor_df["js_divergence"]
        summary_rows.append({
            "behavior": behavior,
            "steering_factor": factor,
            "effective_strength": factor * steering_model.vector_norm,
            "vector_norm": steering_model.vector_norm,
            "kl_mean": kl_vals.mean(),
            "kl_std": kl_vals.std(),
            "kl_median": kl_vals.median(),
            "kl_min": kl_vals.min(),
            "kl_max": kl_vals.max(),
            "js_mean": js_vals.mean(),
            "js_std": js_vals.std(),
            "n_examples": len(kl_vals),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = output_dir / "kl_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_json(output_dir / "kl_summary.json", orient='records', indent=2)
    logger.warning(f"Saved summary to {summary_path}")

    # Print summary table
    logger.warning("\n" + "=" * 70)
    logger.warning(f"KL DIVERGENCE SUMMARY: {behavior}")
    logger.warning("=" * 70)
    print(summary_df.to_string(index=False))

    # Clean up pyvene hooks so the model can be reused
    del steering_model
    torch.cuda.empty_cache()

    return summary_rows


def discover_axbench_concepts(axbench_dir):
    """
    Discover concept subdirectories produced by prepare_axbench_concepts.py.
    Each must contain DiffMean_weight.pt and test_prompts.json.
    Returns list of (concept_label, concept_dir) tuples.
    """
    axbench_dir = Path(axbench_dir)
    concepts = []

    for subdir in sorted(axbench_dir.iterdir()):
        if not subdir.is_dir():
            continue
        weight_path = subdir / "DiffMean_weight.pt"
        test_path = subdir / "test_prompts.json"
        if weight_path.exists() and test_path.exists():
            # Try to load config for a nice label
            config_path = subdir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                label = f"axbench_{config.get('concept_id', subdir.name)}"
            else:
                label = subdir.name
            concepts.append((label, subdir))

    return concepts


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute KL divergence between steered and unsteered first-token distributions."
    )
    parser.add_argument("--behavior", type=str, required=True,
                        help="Target behavior / concept name. Use 'axbench' to run all "
                             "concepts in --axbench_dir.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--steering_vector_path", type=str, default=None,
                        help="Path to saved DiffMean steering vector (.pt). "
                             "Not needed when --behavior axbench.")
    parser.add_argument("--test_dataset_path", type=str, default=None,
                        help="Path to test dataset JSON. "
                             "Not needed when --behavior axbench.")
    parser.add_argument("--axbench_dir", type=str, default=None,
                        help="Directory containing concept subdirs from "
                             "prepare_axbench_concepts.py. Required when --behavior axbench.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steering_factors", type=str, default="-2,-1,0,1,2",
                        help="Comma-separated steering factors to evaluate")
    parser.add_argument("--top_k_tokens", type=int, default=10,
                        help="Number of top boosted/suppressed tokens to report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Validate args
    if args.behavior == "axbench":
        if not args.axbench_dir:
            print("ERROR: --axbench_dir is required when --behavior is 'axbench'")
            sys.exit(1)
    else:
        if not args.steering_vector_path or not args.test_dataset_path:
            print("ERROR: --steering_vector_path and --test_dataset_path are required "
                  "for single-concept mode")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")

    # Parse steering factors
    steering_factors = [float(x.strip()) for x in args.steering_factors.split(",")]
    logger.warning(f"Steering factors: {steering_factors}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=1024)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (ONCE — reused across all concepts)
    logger.warning(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        device_map=device
    )
    model.eval()

    # Get prefix length
    prefix_length = 1
    if args.model_name in CHAT_MODELS:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    if args.behavior == "axbench":
        # ==================================================================
        # AXBENCH MODE: iterate over all concept subdirs
        # ==================================================================
        concepts = discover_axbench_concepts(args.axbench_dir)
        if not concepts:
            print(f"ERROR: No concept subdirs found in {args.axbench_dir}")
            print("Each subdir must contain DiffMean_weight.pt and test_prompts.json")
            sys.exit(1)

        logger.warning(f"Found {len(concepts)} AxBench concepts in {args.axbench_dir}")
        for label, cdir in concepts:
            logger.warning(f"  {label}: {cdir}")

        all_concept_summaries = []

        for i, (label, concept_dir) in enumerate(concepts):
            logger.warning(f"\n{'#'*70}")
            logger.warning(f"# Concept {i+1}/{len(concepts)}: {label}")
            logger.warning(f"{'#'*70}")

            kl_output_dir = concept_dir / "kl_eval"

            summary_rows = run_single_concept(
                model, tokenizer, device,
                behavior=label,
                layer=args.layer,
                steering_vector_path=str(concept_dir / "DiffMean_weight.pt"),
                test_dataset_path=str(concept_dir / "test_prompts.json"),
                output_dir=kl_output_dir,
                steering_factors=steering_factors,
                prefix_length=prefix_length,
                batch_size=args.batch_size,
                top_k_tokens=args.top_k_tokens,
            )
            all_concept_summaries.extend(summary_rows)

        # Save combined summary across all concepts
        combined_df = pd.DataFrame(all_concept_summaries)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        combined_path = output_dir / "kl_all_concepts_summary.csv"
        combined_df.to_csv(combined_path, index=False)
        combined_df.to_json(output_dir / "kl_all_concepts_summary.json", orient='records', indent=2)

        logger.warning(f"\n{'='*70}")
        logger.warning(f"COMBINED SUMMARY ({len(concepts)} concepts)")
        logger.warning(f"{'='*70}")
        print(combined_df.to_string(index=False))
        logger.warning(f"\nSaved combined summary to {combined_path}")

    else:
        # ==================================================================
        # SINGLE CONCEPT MODE (original behavior)
        # ==================================================================
        run_single_concept(
            model, tokenizer, device,
            behavior=args.behavior,
            layer=args.layer,
            steering_vector_path=args.steering_vector_path,
            test_dataset_path=args.test_dataset_path,
            output_dir=args.output_dir,
            steering_factors=steering_factors,
            prefix_length=prefix_length,
            batch_size=args.batch_size,
            top_k_tokens=args.top_k_tokens,
        )

    logger.warning("\nKL divergence evaluation complete!")


if __name__ == "__main__":
    main()
