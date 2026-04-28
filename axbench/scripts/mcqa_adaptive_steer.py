"""
MCQA per-prompt adaptive steering experiment.

For each test prompt j at layer l, computes a per-prompt steering factor:

    alpha_j(l) = max(0, (1 - kappa_a_j(l)) / 2)

This analytically moves the prompt's projection onto the steering vector exactly
to c+ (the positive training centroid). When steering adds alpha * v to the
activation, the projection shifts by alpha * ||v||. Setting the new projection
equal to c+ = mu_pos . v_hat and solving gives the formula above.

Prompts whose baseline projection is already at or beyond c+ (kappa_a >= 1) are
not steered backward; their factor is clamped to 0.

Compares on MCQA greedy accuracy:
  - Baseline (no steering)
  - Adaptive per-prompt steering (each prompt pushed to c+)
  - Fixed-factor steering at each specified alpha (for reference)

Usage:
    uv run python axbench/scripts/mcqa_adaptive_steer.py \
        --behavior corrigible-neutral-HHH \
        --model_name google/gemma-2-9b-it \
        --layers 10-32 \
        --max_examples 300 \
        --steering-factors 1,2,3,5,7,10
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from axbench.utils.constants import CHAT_MODELS
from axbench.utils.model_utils import get_prefix_length

import logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-whitegrid")


TEST_PATH_MAP = {
    b: f"datasets/test/{b}/test_dataset_ab.json"
    for b in [
        "sycophancy", "survival-instinct", "corrigible-neutral-HHH",
        "hallucination", "refusal", "myopic-reward", "coordinate-other-ais",
    ]
}

TRAIN_PATH_MAP = {
    b: f"datasets/raw/{b}/dataset.json"
    for b in [
        "sycophancy", "survival-instinct", "corrigible-neutral-HHH",
        "hallucination", "refusal", "myopic-reward", "coordinate-other-ais",
    ]
}


# ============================================================================
# Tokenization / layer helpers
# ============================================================================
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def parse_layer_range(spec: str) -> list:
    """Accept '10-32' or '10,15,20,32'."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(x) for x in spec.split(","))


def compute_dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    gap = abs(pos.mean() - neg.mean())
    pooled = np.sqrt(0.5 * (pos.var() + neg.var()))
    return float(gap / pooled) if pooled > 1e-12 else 0.0


def extract_matching_letter(answer_str: str) -> str:
    m = re.search(r"\(([A-Z])\)", answer_str.upper())
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse A/B letter from '{answer_str}'")


def build_prompt_and_full_ids(tokenizer, question: str, candidate: str):
    """Returns (prompt_ids, full_ids, letter_pos)."""
    if supports_chat_template(tokenizer):
        msgs_prompt = [{"role": "user", "content": question}]
        prompt_ids = tokenizer.apply_chat_template(
            msgs_prompt, tokenize=True, add_generation_prompt=True,
        )
        msgs_full = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": candidate},
        ]
        full_ids = tokenizer.apply_chat_template(
            msgs_full, tokenize=True, add_generation_prompt=False,
        )
    else:
        prompt_ids = tokenizer.encode(question)
        full_ids = tokenizer.encode(question + candidate)

    target_letter = extract_matching_letter(candidate)
    letter_pos = None
    for i in range(len(full_ids) - 1, max(len(prompt_ids) - 1, 0), -1):
        tok_str = tokenizer.decode([full_ids[i]]).strip()
        if tok_str == target_letter:
            letter_pos = i
            break
    if letter_pos is None:
        raise RuntimeError(
            f"Could not find '{target_letter}' token in suffix of full_ids "
            f"(tail decoded = {tokenizer.decode(full_ids[-8:])!r})"
        )
    return prompt_ids, full_ids, letter_pos


def answer_token_logprob_batch(logits: torch.Tensor, input_ids: torch.Tensor,
                                prompt_lens, lengths):
    """Vectorized sum of log-probs over answer tokens. Returns list of floats."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    results = []
    for i in range(logits.shape[0]):
        p_len = int(prompt_lens[i])
        L_i = int(lengths[i])
        if L_i <= p_len:
            results.append(0.0)
            continue
        idx = torch.arange(p_len, L_i, device=logits.device)
        token_ids = input_ids[i, idx]
        dist = log_probs[i, idx - 1]
        lp = dist.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).sum()
        results.append(float(lp.item()))
    return results


def pad_batch(token_lists, pad_id: int, device):
    """Right-pad a list of token id lists. Returns input_ids, attention_mask, lengths."""
    max_len = max(len(t) for t in token_lists)
    B = len(token_lists)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    lengths = []
    for i, t in enumerate(token_lists):
        L = len(t)
        input_ids[i, :L] = torch.tensor(t, dtype=torch.long, device=device)
        attention_mask[i, :L] = 1
        lengths.append(L)
    return input_ids, attention_mask, lengths


# ============================================================================
# Forward pass hooks
# ============================================================================
def make_capture_hook(storage: dict, layer_idx: int):
    def hook(mod, inp, out):
        storage[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook


def make_uniform_steering_hook(steering_vec: torch.Tensor, factor: float, prefix_length: int):
    """Adds factor * steering_vec uniformly to all batch items at positions >= prefix_length."""
    sv = steering_vec.detach().clone()

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = h.clone()
        new_h[:, prefix_length:] = (
            new_h[:, prefix_length:] + factor * sv.to(new_h.dtype).to(new_h.device)
        )
        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h

    return hook


def make_per_example_steering_hook(steering_vec: torch.Tensor, factors: list,
                                    prefix_length: int):
    """
    Adds factors[i] * steering_vec to positions >= prefix_length for batch item i.

    Enables per-prompt adaptive steering within a single batched forward pass:
    each sequence gets its own steering strength without separate forward passes.

    factors: list of floats of length == batch size (may include zeros).
    """
    sv = steering_vec.detach().clone()

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = h.clone()
        fa = torch.tensor(factors, dtype=new_h.dtype, device=new_h.device).view(-1, 1, 1)
        sv_dev = sv.to(new_h.dtype).to(new_h.device)
        new_h[:, prefix_length:] = new_h[:, prefix_length:] + fa * sv_dev.unsqueeze(0)
        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h

    return hook


@torch.no_grad()
def forward_capture_batch(model, input_ids, attention_mask, target_layers):
    """Batched forward pass with per-layer hidden-state capture.

    Returns logits [B, L, V] and layer_hiddens {layer: [B, L, H]}.
    """
    storage = {}
    handles = []
    for l in target_layers:
        h = model.model.layers[l].register_forward_hook(
            make_capture_hook(storage, l), always_call=True,
        )
        handles.append(h)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    for h in handles:
        h.remove()
    return out.logits, {l: storage[l] for l in target_layers}


@torch.no_grad()
def forward_uniform_steering_batch(model, input_ids, attention_mask,
                                    steering_layer: int, steering_vec: torch.Tensor,
                                    factor: float, prefix_length: int):
    """Batched forward pass with a uniform steering factor at one layer."""
    if factor == 0.0:
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    h = model.model.layers[steering_layer].register_forward_hook(
        make_uniform_steering_hook(steering_vec, factor, prefix_length),
        always_call=True,
    )
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        h.remove()
    return out.logits


@torch.no_grad()
def forward_per_example_steering_batch(model, input_ids, attention_mask,
                                        steering_layer: int, steering_vec: torch.Tensor,
                                        factors: list, prefix_length: int):
    """Batched forward pass where each item in the batch has its own steering factor."""
    if all(f == 0.0 for f in factors):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    h = model.model.layers[steering_layer].register_forward_hook(
        make_per_example_steering_hook(steering_vec, factors, prefix_length),
        always_call=True,
    )
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        h.remove()
    return out.logits


# ============================================================================
# Plots
# ============================================================================
def plot_accuracy_comparison(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    """
    Main result: baseline, adaptive, and fixed-factor greedy accuracy by layer.
    """
    if layer_df.empty:
        return
    layers = sorted(layer_df["layer"].values)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(layers, layer_df["baseline_greedy_acc"].values, "D--", color="gray",
            linewidth=1.5, markersize=6, label="Baseline (no steering)", alpha=0.8, zorder=3)
    ax.plot(layers, layer_df["adaptive_greedy_acc"].values, "o-", color="#E94F37",
            linewidth=2.5, markersize=7, label="Adaptive α_j = max(0, (1−κ)/2)", zorder=4)

    fixed_cols = sorted(c for c in layer_df.columns if c.startswith("fixed_greedy_acc_"))
    cmap = plt.get_cmap("viridis")
    for i, col in enumerate(fixed_cols):
        f_val = col.replace("fixed_greedy_acc_", "")
        color = cmap(i / max(1, len(fixed_cols) - 1))
        ax.plot(layers, layer_df[col].values, "s:", color=color,
                linewidth=1.5, markersize=4, label=f"Fixed α={f_val}", alpha=0.75)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Greedy accuracy", fontsize=11)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{behavior}: Adaptive per-prompt steering vs baseline vs fixed factors\n"
        f"Adaptive α_j moves each prompt's projection to the positive centroid c+",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_factor_distribution(per_prompt_df: pd.DataFrame, layers: list,
                              behavior: str, output_path: Path):
    """Box plot of adaptive factor distribution across layers."""
    if per_prompt_df.empty or "adaptive_factor" not in per_prompt_df.columns:
        return
    data = [
        per_prompt_df[per_prompt_df["layer"] == l]["adaptive_factor"].dropna().values
        for l in layers
    ]
    data = [d for d in data if len(d) > 0]
    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        data, positions=layers[:len(data)], widths=0.6, patch_artist=True,
        boxprops=dict(facecolor="#2E86AB", alpha=0.5),
        medianprops=dict(color="#E94F37", linewidth=2),
    )
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Adaptive factor α_j (clamped to ≥ 0)", fontsize=11)
    ax.set_xticks(layers[:len(data)])
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(
        f"{behavior}: Distribution of per-prompt adaptive steering factors by layer\n"
        f"α_j = max(0, (1−κ_a)/2). Zero = already at or beyond positive centroid.",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_clamped_fraction(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    """Per-layer fraction of prompts clamped (already at/beyond c+, kappa_a >= 1)."""
    if layer_df.empty or "frac_clamped" not in layer_df.columns:
        return
    layers = layer_df["layer"].values
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(layers, layer_df["frac_clamped"].values, color="#6B2D5C", alpha=0.7, width=0.6)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Fraction clamped (κ_a ≥ 1)", fontsize=11)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{behavior}: Fraction of prompts already at/beyond positive centroid per layer\n"
        f"These prompts receive α=0 (no steering needed or would push backward)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kappa_vs_adaptive_delta(per_prompt_df: pd.DataFrame, layer: int,
                                  behavior: str, output_path: Path):
    """Scatter: κ_a vs improvement (adaptive greedy match - baseline greedy match)."""
    ldf = per_prompt_df[per_prompt_df["layer"] == layer].dropna(
        subset=["kappa_a", "adaptive_greedy_match", "greedy_match_baseline"]
    )
    if len(ldf) < 4:
        return
    kappa = ldf["kappa_a"].values
    delta = ldf["adaptive_greedy_match"].values - ldf["greedy_match_baseline"].values

    colors = np.where(delta > 0, "#2ca02c", np.where(delta < 0, "#d62728", "gray"))
    n_improved = int((delta > 0).sum())
    n_degraded = int((delta < 0).sum())
    n_same = int((delta == 0).sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(kappa, delta, c=colors, s=70, edgecolors="k", linewidths=0.5, alpha=0.85)
    ax.axvline(x=-1, color="#d62728", linestyle=":", alpha=0.4, label="κ=−1 (μ⁻)")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="κ=0 (midpoint)")
    ax.axvline(x=+1, color="#2ca02c", linestyle=":", alpha=0.4, label="κ=+1 (μ⁺, clamp)")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("κ_a (baseline projection, normalized to [c−,c+])", fontsize=10)
    ax.set_ylabel("Δ greedy match (adaptive − baseline)", fontsize=10)
    ax.set_title(
        f"{behavior} — Layer {layer}: κ_a vs improvement from adaptive steering\n"
        f"Green=improved ({n_improved}), Red=degraded ({n_degraded}), "
        f"Gray=unchanged ({n_same})",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_adaptive_vs_best_fixed(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    """
    Per-layer: difference in greedy accuracy between adaptive and the best fixed factor.
    Positive = adaptive is better; negative = best fixed factor is better.
    """
    if layer_df.empty:
        return
    fixed_cols = sorted(c for c in layer_df.columns if c.startswith("fixed_greedy_acc_"))
    if not fixed_cols:
        return

    best_fixed = layer_df[fixed_cols].max(axis=1).values
    delta = layer_df["adaptive_greedy_acc"].values - best_fixed
    layers = layer_df["layer"].values

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]
    ax.bar(layers, delta, color=colors, alpha=0.7, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Adaptive − best fixed (greedy acc)", fontsize=11)
    ax.set_xticks(layers)
    ax.set_title(
        f"{behavior}: Adaptive steering vs best fixed factor, per layer\n"
        f"Green = adaptive wins; Red = best fixed factor wins",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MCQA per-prompt adaptive steering (project each prompt to c+). "
                    "Self-contained: computes DiffMean vectors from training data inline.",
    )
    parser.add_argument("--behavior", type=str, required=True,
                        choices=list(TEST_PATH_MAP.keys()))
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--train_path", type=str, default=None,
                        help="Default: datasets/raw/<behavior>/dataset.json")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Default: datasets/test/<behavior>/test_dataset_ab.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: results/mcqa_adaptive/<model>/<behavior>")
    parser.add_argument("--layers", type=str, default="10-32",
                        help="Layer range '10-32' or list '10,15,20'. Default: 10-32.")
    parser.add_argument("--max_examples", type=int, default=300,
                        help="Max training pairs for DiffMean (randomly sampled). Default: 300.")
    parser.add_argument(
        "--steering-factors", "--steering_factors", "--factors",
        dest="steering_factors",
        type=str,
        default="1,2,3",
        help="Comma-separated fixed factors for comparison (e.g. 1,2,3,5,10).",
    )
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Sequences per forward pass. Effective GPU batch for Phase B "
                             "is batch_size * (1 + len(fixed_factors)). Reduce if OOM.")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Limit number of test prompts (default: use all).")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fixed_factors = sorted(
        {float(x.strip()) for x in str(args.steering_factors).split(",") if x.strip()}
    )
    if not fixed_factors:
        logger.warning("No fixed factors specified; only adaptive steering will be run.")

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layers = parse_layer_range(args.layers)
    logger.warning(f"Target layers: {layers}")

    model_short = args.model_name.split("/")[-1]
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path("results") / "mcqa_adaptive" / model_short / args.behavior
    )
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path or TRAIN_PATH_MAP[args.behavior])
    test_path = Path(args.test_path or TEST_PATH_MAP[args.behavior])

    logger.warning(f"Loading training data: {train_path}")
    with open(train_path) as f:
        train_data = json.load(f)
    if args.max_examples is not None and args.max_examples < len(train_data):
        import random as _rnd
        indices = _rnd.sample(range(len(train_data)), args.max_examples)
        train_data = [train_data[i] for i in sorted(indices)]
        logger.warning(f"  Subsampled to {len(train_data)} pairs (seed={args.seed})")
    else:
        logger.warning(f"  {len(train_data)} training pairs")

    logger.warning(f"Loading test data: {test_path}")
    with open(test_path) as f:
        test_data = json.load(f)
    if args.max_test is not None:
        test_data = test_data[: args.max_test]
    logger.warning(f"  {len(test_data)} test prompts")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prefix_length = 1
    if args.model_name in CHAT_MODELS:
        prefix_length = get_prefix_length(tokenizer)
    logger.warning(f"Steering prefix_length = {prefix_length}")

    logger.warning(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        device_map=device,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Phase 0: Compute DiffMean vectors from training data.
    #
    # For each training pair (question, answer_matching, answer_not_matching):
    #   - tokenize both as [user: question | assistant: answer]
    #   - one batched forward pass captures all target layers at once
    #   - extract activation at the answer-letter token position
    #   - accumulate pos/neg sums per layer
    # Then: mu_pos = mean(pos_acts), mu_neg = mean(neg_acts), v = mu_pos - mu_neg.
    # ------------------------------------------------------------------
    logger.warning(
        f"\n=== Phase 0: DiffMean training pass "
        f"({len(train_data)} pairs × 2, {len(layers)} layers, B={args.batch_size}) ==="
    )

    # Build flat training items (one per (example, answer-class)).
    train_flat = []
    train_dropped = 0
    for item in train_data:
        q = item["question"]
        for answer, label in [
            (item["answer_matching_behavior"], 1),
            (item["answer_not_matching_behavior"], 0),
        ]:
            try:
                prompt_ids, full_ids, letter_pos = build_prompt_and_full_ids(
                    tokenizer, q, answer
                )
            except (RuntimeError, ValueError):
                train_dropped += 1
                continue
            train_flat.append({
                "full_ids": full_ids,
                "prompt_len": len(prompt_ids),
                "letter_pos": letter_pos,
                "label": label,
            })
    if train_dropped:
        logger.warning(f"  Dropped {train_dropped} malformed training items.")
    logger.warning(f"  {len(train_flat)} training items after filtering.")

    # Accumulate per-layer activations at the letter position.
    pos_acts = {l: [] for l in layers}
    neg_acts = {l: [] for l in layers}
    B = args.batch_size
    pad_id = tokenizer.pad_token_id

    for start in tqdm(range(0, len(train_flat), B), desc="Phase 0 batches"):
        batch = train_flat[start:start + B_train]
        token_lists = [b["full_ids"] for b in batch]
        input_ids, attn, _ = pad_batch(token_lists, pad_id, device)
        _, layer_hiddens = forward_capture_batch(model, input_ids, attn, layers)
        for i, b in enumerate(batch):
            for l in layers:
                act = layer_hiddens[l][i, b["letter_pos"], :].float().cpu()
                (pos_acts[l] if b["label"] == 1 else neg_acts[l]).append(act)
        del layer_hiddens, input_ids, attn

    # Compute steering vectors and d' per layer.
    steering_vecs = {}
    mu_poss, mu_negs, dprimes = {}, {}, {}
    for l in layers:
        if not pos_acts[l] or not neg_acts[l]:
            logger.warning(f"Layer {l}: no pos or neg activations — skipping.")
            continue
        mu_pos = torch.stack(pos_acts[l]).mean(0)
        mu_neg = torch.stack(neg_acts[l]).mean(0)
        v = mu_pos - mu_neg
        mu_poss[l] = mu_pos
        mu_negs[l] = mu_neg
        steering_vecs[l] = v.to(device)
        # d': project training acts onto v_hat, compute signal-to-noise ratio.
        v_norm_sq = float(v.dot(v))
        mu_mid = 0.5 * (mu_pos + mu_neg)
        if v_norm_sq > 1e-12:
            pos_proj = np.array([
                float(2.0 * (a - mu_mid).dot(v) / v_norm_sq) for a in pos_acts[l]
            ])
            neg_proj = np.array([
                float(2.0 * (a - mu_mid).dot(v) / v_norm_sq) for a in neg_acts[l]
            ])
            dprimes[l] = compute_dprime(pos_proj, neg_proj)
        else:
            dprimes[l] = 0.0
        logger.warning(
            f"  Layer {l:2d}: ||v||={float(v.norm()):.4f}  d'={dprimes[l]:.3f}  "
            f"n_pos={len(pos_acts[l])}  n_neg={len(neg_acts[l])}"
        )

    # Only keep layers where we successfully computed a steering vector.
    layers = [l for l in layers if l in steering_vecs]
    del pos_acts, neg_acts

    # ------------------------------------------------------------------
    # Pre-tokenize all (prompt, candidate) pairs
    # ------------------------------------------------------------------
    prompt_meta = []  # None for dropped prompts
    pre_tok = []      # None for dropped prompts
    dropped = 0
    for j, item in enumerate(test_data):
        q = item["question"]
        try:
            matching_letter = extract_matching_letter(item["answer_matching_behavior"])
            notmatch_letter = extract_matching_letter(item["answer_not_matching_behavior"])
        except (ValueError, KeyError):
            prompt_meta.append(None)
            pre_tok.append(None)
            dropped += 1
            continue
        if matching_letter == notmatch_letter:
            prompt_meta.append(None)
            pre_tok.append(None)
            dropped += 1
            continue
        row = {}
        for letter in (matching_letter, notmatch_letter):
            cand = f" ({letter})"
            prompt_ids, full_ids, letter_pos = build_prompt_and_full_ids(tokenizer, q, cand)
            row[letter] = {
                "full_ids": full_ids,
                "prompt_len": len(prompt_ids),
                "letter_pos": letter_pos,
            }
        pre_tok.append(row)
        prompt_meta.append({
            "matching_letter": matching_letter,
            "notmatch_letter": notmatch_letter,
        })
    if dropped:
        logger.warning(f"Dropped {dropped} malformed prompt(s).")

    flat_items = []
    for j, row in enumerate(pre_tok):
        if row is None:
            continue
        meta = prompt_meta[j]
        for role, letter in (
            ("match", meta["matching_letter"]),
            ("notmatch", meta["notmatch_letter"]),
        ):
            e = row[letter]
            flat_items.append({
                "prompt_idx": j,
                "letter": letter,
                "role": role,
                "full_ids": e["full_ids"],
                "prompt_len": e["prompt_len"],
                "letter_pos": e["letter_pos"],
            })

    n_items = len(flat_items)

    # ------------------------------------------------------------------
    # Phase A: Batched baseline forward + per-layer activation capture
    # ------------------------------------------------------------------
    logger.warning(f"\n=== Phase A: Baseline forward + activation capture "
                   f"(B={B}, {n_items} items across {len(layers)} layers) ===")

    baseline_logps = {}   # (j, role) -> float
    act_at_letter = {}    # (j, role, layer) -> tensor [hidden] on cpu

    for start in tqdm(range(0, n_items, B), desc="Baseline batches"):
        batch = flat_items[start:start + B]
        token_lists = [b["full_ids"] for b in batch]
        input_ids, attn, lens = pad_batch(token_lists, pad_id, device)
        logits, layer_hiddens = forward_capture_batch(model, input_ids, attn, layers)
        prompt_lens = [b["prompt_len"] for b in batch]
        logps = answer_token_logprob_batch(logits, input_ids, prompt_lens, lens)
        for i, b in enumerate(batch):
            baseline_logps[(b["prompt_idx"], b["role"])] = logps[i]
            for l in layers:
                act_at_letter[(b["prompt_idx"], b["role"], l)] = (
                    layer_hiddens[l][i, b["letter_pos"], :].float().cpu()
                )
        del logits, layer_hiddens, input_ids, attn

    # Collect baseline records (greedy letter, p_match, etc.)
    baseline_records = []
    for j, item in enumerate(test_data):
        if prompt_meta[j] is None:
            continue
        ml = prompt_meta[j]["matching_letter"]
        nl = prompt_meta[j]["notmatch_letter"]
        logp_m = baseline_logps[(j, "match")]
        logp_n = baseline_logps[(j, "notmatch")]
        p_match = float(np.exp(logp_m) / (np.exp(logp_m) + np.exp(logp_n)))
        is_match_greedy = logp_m > logp_n
        greedy_letter = ml if is_match_greedy else nl
        greedy_role = "match" if is_match_greedy else "notmatch"
        baseline_records.append({
            "prompt_idx": j,
            "question": item["question"],
            "answer_matching_behavior": item["answer_matching_behavior"],
            "answer_not_matching_behavior": item.get("answer_not_matching_behavior", ""),
            "matching_letter": ml,
            "notmatch_letter": nl,
            "greedy_letter": greedy_letter,
            "greedy_role": greedy_role,
            "baseline_p_match": p_match,
            "greedy_match_baseline": int(is_match_greedy),
            "baseline_logp_match": float(logp_m),
            "baseline_logp_notmatch": float(logp_n),
        })

    # ------------------------------------------------------------------
    # Compute kappa_a and adaptive factors per (layer, prompt)
    #
    # kappa_a = 2 * (act - mu) . v / ||v||^2
    #   where v = mu_pos - mu_neg, mu = (mu_pos + mu_neg) / 2
    #   kappa_a = +1 at mu_pos, -1 at mu_neg, 0 at midpoint.
    #
    # Adaptive factor: alpha_j = max(0, (1 - kappa_a_j) / 2)
    #   Derivation: steering adds alpha*v, shifting projection by alpha*||v||.
    #   Setting new projection = c+ (positive centroid projection) gives:
    #     p_j + alpha*||v|| = c+   =>   alpha = (c+ - p_j) / ||v||
    #   In kappa_a terms (where kappa_a=1 iff p=c+): alpha = (1 - kappa_a) / 2.
    #   Clamped to 0 for prompts already at or beyond c+ (kappa_a >= 1).
    # ------------------------------------------------------------------
    logger.warning("\n=== Computing kappa_a and adaptive factors ===")

    kappa_map = {}            # (l, j) -> float
    adaptive_factor_map = {}  # (l, j) -> float (clamped to >= 0)

    for l in layers:
        mu_pos = mu_poss[l]
        mu_neg = mu_negs[l]
        v = (mu_pos - mu_neg)
        v_norm_sq = float(v.dot(v))
        mu = 0.5 * (mu_pos + mu_neg)

        for rec in baseline_records:
            j = rec["prompt_idx"]
            act = act_at_letter[(j, rec["greedy_role"], l)]
            if v_norm_sq < 1e-12:
                kappa = float("nan")
                alpha = 0.0
            else:
                kappa = float(2.0 * (act - mu).dot(v).item() / v_norm_sq)
                alpha = max(0.0, (1.0 - kappa) / 2.0)
            kappa_map[(l, j)] = kappa
            adaptive_factor_map[(l, j)] = alpha

    del act_at_letter  # no longer needed

    # ------------------------------------------------------------------
    # Phase B: Combined adaptive + fixed-factor steering in one pass per layer.
    #
    # For each layer l and each batch of B flat_items, we replicate each
    # sequence (1 + len(fixed_factors)) times in a single larger batch:
    #
    #   slots per item:  [adaptive_factor_j, fixed_f0, fixed_f1, ...]
    #
    # The per-example hook applies a different alpha to each slot, so all
    # (adaptive + fixed) results come from ONE forward pass per layer instead
    # of (1 + F) separate passes.  This is a ~(1+F)x speedup for Phase B at
    # the cost of a (1+F)x larger batch size per pass.  Reduce --batch_size
    # if you hit OOM (the effective GPU batch is B * (1 + len(fixed_factors))).
    # ------------------------------------------------------------------
    n_modes = 1 + len(fixed_factors)   # adaptive + one slot per fixed factor
    eff_batch = B * n_modes
    logger.warning(
        f"\n=== Phase B: Combined adaptive + fixed steering "
        f"({len(layers)} layers, {n_modes} modes, effective batch={eff_batch}) ==="
    )

    adaptive_steered_map = {}               # (l, j) -> dict
    fixed_steered_map = {}                  # (factor, l, j) -> dict

    for l in tqdm(layers, desc="Phase B layers"):
        sv = steering_vecs[l]

        # Collect (prompt_idx, role, logp) per mode slot across all flat_items.
        # Shape: per_mode_logp[mode_idx][item_idx] = (prompt_idx, role, logp)
        per_mode_logp = [[None] * n_items for _ in range(n_modes)]

        for start in range(0, n_items, B):
            batch = flat_items[start:start + B]
            Bi = len(batch)

            # Build combined token list: for each item, one copy per mode.
            combined_tokens = []
            combined_factors = []
            combined_prompt_lens = []
            for b in batch:
                j = b["prompt_idx"]
                # Slot 0: adaptive factor for this prompt at this layer.
                combined_tokens.append(b["full_ids"])
                combined_factors.append(adaptive_factor_map.get((l, j), 0.0))
                combined_prompt_lens.append(b["prompt_len"])
                # Slots 1..F: fixed factors (same sequence, different alpha).
                for f in fixed_factors:
                    combined_tokens.append(b["full_ids"])
                    combined_factors.append(f)
                    combined_prompt_lens.append(b["prompt_len"])

            input_ids, attn, lens = pad_batch(combined_tokens, pad_id, device)
            logits = forward_per_example_steering_batch(
                model, input_ids, attn, l, sv, combined_factors, prefix_length,
            )
            logps = answer_token_logprob_batch(logits, input_ids, combined_prompt_lens, lens)
            del logits, input_ids, attn

            # Unpack: combined batch is laid out as
            #   [item0_mode0, item0_mode1, ..., item0_modeF,
            #    item1_mode0, item1_mode1, ..., item1_modeF, ...]
            for i, b in enumerate(batch):
                global_idx = start + i
                base = i * n_modes
                for m in range(n_modes):
                    per_mode_logp[m][global_idx] = (b["prompt_idx"], b["role"], logps[base + m])

        # Decode mode 0 (adaptive) into adaptive_steered_map.
        by_prompt = {}
        for pj, role, lp in per_mode_logp[0]:
            by_prompt.setdefault(pj, {})[role] = lp
        for rec in baseline_records:
            j = rec["prompt_idx"]
            logp_m = by_prompt[j]["match"]
            logp_n = by_prompt[j]["notmatch"]
            p_match_s = float(np.exp(logp_m) / (np.exp(logp_m) + np.exp(logp_n)))
            adaptive_steered_map[(l, j)] = {
                "adaptive_p_match": p_match_s,
                "adaptive_greedy_match": int(logp_m > logp_n),
                "adaptive_logp_match": float(logp_m),
                "adaptive_logp_notmatch": float(logp_n),
            }

        # Decode modes 1..F (fixed factors) into fixed_steered_map.
        for k, factor in enumerate(fixed_factors):
            by_prompt_f = {}
            for pj, role, lp in per_mode_logp[k + 1]:
                by_prompt_f.setdefault(pj, {})[role] = lp
            for rec in baseline_records:
                j = rec["prompt_idx"]
                logp_m = by_prompt_f[j]["match"]
                logp_n = by_prompt_f[j]["notmatch"]
                p_match_s = float(np.exp(logp_m) / (np.exp(logp_m) + np.exp(logp_n)))
                fixed_steered_map[(factor, l, j)] = {
                    "fixed_p_match": p_match_s,
                    "fixed_greedy_match": int(logp_m > logp_n),
                }

    # ------------------------------------------------------------------
    # Assemble per-prompt DataFrame
    # ------------------------------------------------------------------
    logger.warning("\n=== Assembling per-prompt results ===")
    per_prompt_rows = []
    for rec in baseline_records:
        j = rec["prompt_idx"]
        for l in layers:
            adp = adaptive_steered_map[(l, j)]
            row = {
                "layer": l,
                "prompt_idx": j,
                "question": rec["question"],
                "matching_letter": rec["matching_letter"],
                "notmatch_letter": rec["notmatch_letter"],
                "greedy_letter_baseline": rec["greedy_letter"],
                "baseline_logp_match": rec["baseline_logp_match"],
                "baseline_logp_notmatch": rec["baseline_logp_notmatch"],
                "baseline_p_match": rec["baseline_p_match"],
                "greedy_match_baseline": rec["greedy_match_baseline"],
                "kappa_a": kappa_map.get((l, j), float("nan")),
                "adaptive_factor": adaptive_factor_map.get((l, j), 0.0),
                "adaptive_logp_match": adp["adaptive_logp_match"],
                "adaptive_logp_notmatch": adp["adaptive_logp_notmatch"],
                "adaptive_p_match": adp["adaptive_p_match"],
                "adaptive_greedy_match": adp["adaptive_greedy_match"],
            }
            for factor in fixed_factors:
                fst = fixed_steered_map.get((factor, l, j), {})
                f_tag = f"{factor:g}"
                row[f"fixed_p_match_{f_tag}"] = fst.get("fixed_p_match", float("nan"))
                row[f"fixed_greedy_match_{f_tag}"] = fst.get("fixed_greedy_match", float("nan"))
            per_prompt_rows.append(row)

    per_prompt_df = pd.DataFrame(per_prompt_rows)
    per_prompt_df.to_csv(out_dir / "per_prompt_results.csv", index=False)
    logger.warning(
        f"Wrote {len(per_prompt_df)} per-prompt rows "
        f"({len(layers)} layers × {len(baseline_records)} prompts) "
        f"to {out_dir / 'per_prompt_results.csv'}"
    )

    # ------------------------------------------------------------------
    # Per-layer summaries
    # ------------------------------------------------------------------
    layer_rows = []
    for l in layers:
        ldf = per_prompt_df[per_prompt_df["layer"] == l]
        if ldf.empty:
            continue

        alpha_vals = ldf["adaptive_factor"].values
        kappa_vals = ldf["kappa_a"].values
        frac_clamped = float((kappa_vals >= 1.0).mean())

        row = {
            "layer": l,
            "dprime": dprimes.get(l, float("nan")),
            "n_prompts": int(len(ldf)),
            "baseline_greedy_acc": float(ldf["greedy_match_baseline"].mean()),
            "adaptive_greedy_acc": float(ldf["adaptive_greedy_match"].mean()),
            "baseline_p_match_mean": float(ldf["baseline_p_match"].mean()),
            "adaptive_p_match_mean": float(ldf["adaptive_p_match"].mean()),
            "delta_greedy_acc": float(
                ldf["adaptive_greedy_match"].mean() - ldf["greedy_match_baseline"].mean()
            ),
            "mean_kappa_a": float(np.nanmean(kappa_vals)),
            "mean_adaptive_factor": float(np.nanmean(alpha_vals)),
            "mean_adaptive_factor_unclamped": float(
                np.nanmean((1.0 - kappa_vals) / 2.0)
            ),
            "frac_clamped": frac_clamped,
        }

        for factor in fixed_factors:
            f_tag = f"{factor:g}"
            col = f"fixed_greedy_match_{f_tag}"
            if col in ldf.columns:
                row[f"fixed_greedy_acc_{f_tag}"] = float(ldf[col].mean())
                row[f"fixed_p_match_mean_{f_tag}"] = float(
                    ldf[f"fixed_p_match_{f_tag}"].mean()
                )

        # Cross-layer correlations: kappa_a vs baseline p_match (sanity check)
        valid = ldf[["kappa_a", "baseline_p_match"]].dropna()
        if len(valid) >= 4:
            rho, sp = scipy_stats.spearmanr(valid["kappa_a"], valid["baseline_p_match"])
            r, pr = scipy_stats.pearsonr(valid["kappa_a"], valid["baseline_p_match"])
            row["kappa_baseline_spearman_rho"] = float(rho)
            row["kappa_baseline_spearman_p"] = float(sp)
            row["kappa_baseline_pearson_r"] = float(r)
            row["kappa_baseline_pearson_p"] = float(pr)

        layer_rows.append(row)

    layer_df = pd.DataFrame(layer_rows).sort_values("layer").reset_index(drop=True)
    layer_df.to_csv(out_dir / "per_layer_summary.csv", index=False)

    # Print per-layer table
    fixed_greedy_cols = sorted(c for c in layer_df.columns if c.startswith("fixed_greedy_acc_"))
    logger.warning("\nPer-layer summary:")
    for _, r in layer_df.iterrows():
        fixed_str = "  ".join(
            f"fixed[{c.replace('fixed_greedy_acc_', '')}]={r[c]:.3f}"
            for c in fixed_greedy_cols if c in r and not pd.isna(r[c])
        )
        logger.warning(
            f"  L{int(r['layer']):2d}: d'={r['dprime']:.3f}  "
            f"base={r['baseline_greedy_acc']:.3f}  "
            f"adaptive={r['adaptive_greedy_acc']:.3f} "
            f"(Δ={r['delta_greedy_acc']:+.3f})  "
            f"mean_α={r['mean_adaptive_factor']:.3f}  "
            f"clamped={r['frac_clamped']:.2f}  "
            f"{fixed_str}"
        )

    # Aggregate summary across layers
    logger.warning("\nAggregate (mean across layers):")
    logger.warning(
        f"  Baseline greedy acc:  {layer_df['baseline_greedy_acc'].mean():.3f}"
    )
    logger.warning(
        f"  Adaptive greedy acc:  {layer_df['adaptive_greedy_acc'].mean():.3f}"
    )
    for col in fixed_greedy_cols:
        f_tag = col.replace("fixed_greedy_acc_", "")
        logger.warning(
            f"  Fixed α={f_tag} greedy acc: {layer_df[col].mean():.3f}"
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plots_dir = out_dir / "plots"
    plot_accuracy_comparison(
        layer_df, args.behavior,
        plots_dir / "accuracy_comparison_by_layer.png",
    )
    plot_factor_distribution(
        per_prompt_df, layers, args.behavior,
        plots_dir / "adaptive_factor_distribution_by_layer.png",
    )
    plot_clamped_fraction(
        layer_df, args.behavior,
        plots_dir / "clamped_fraction_by_layer.png",
    )
    plot_adaptive_vs_best_fixed(
        layer_df, args.behavior,
        plots_dir / "adaptive_vs_best_fixed_by_layer.png",
    )
    # Per-layer scatter plots (kappa_a vs delta)
    for l in layers:
        plot_kappa_vs_adaptive_delta(
            per_prompt_df, l, args.behavior,
            plots_dir / f"kappa_vs_delta_layer_{l}.png",
        )

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    fixed_greedy_cols_all = {
        c: float(layer_df[c].mean()) for c in fixed_greedy_cols if c in layer_df.columns
    }
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "sweep_dir": str(sweep_dir),
        "fixed_factors": fixed_factors,
        "n_layers": len(layers),
        "n_test_prompts": len(baseline_records),
        "layers": layers,
        "mean_across_layers": {
            "baseline_greedy_acc": float(layer_df["baseline_greedy_acc"].mean()),
            "adaptive_greedy_acc": float(layer_df["adaptive_greedy_acc"].mean()),
            "delta_greedy_acc": float(layer_df["delta_greedy_acc"].mean()),
            "mean_adaptive_factor": float(layer_df["mean_adaptive_factor"].mean()),
            "frac_clamped": float(layer_df["frac_clamped"].mean()),
            **fixed_greedy_cols_all,
        },
        "per_layer": layer_df.to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
