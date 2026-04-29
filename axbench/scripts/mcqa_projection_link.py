"""
MCQA Projection-Steering Link Analysis
=======================================

Core question: for a given behavior, does the side of the diff-in-means line
that an unsteered test activation lands on predict whether the model displays
the behavior? And does the layer where this prediction is strongest also happen
to be the layer where steering is most effective?

Pipeline (single self-contained script):
  Phase 0  — Compute DiffMean steering vectors from training data (inline).
  Phase A  — Baseline greedy decode + activation capture for all test prompts.
             Evaluation: argmax of lm_head at the '(' position (same approach
             as mcqa_adaptive_steer.py; off-format / gibberish → counted wrong).
  Phase B  — Fixed-factor steered greedy decode per (layer, factor).
  Analysis — Per layer:
               • sign(κ_a) vs baseline_correct  → MCC  (binary predictor)
               • κ_a vs baseline_correct         → Spearman ρ (continuous)
               • d' from training projections
               • steered greedy accuracy per factor
             Cross-layer:
               • Pearson/Spearman between projection MCC and steered accuracy
               • Same for d' vs steered accuracy

Outputs (under --output_dir):
  per_prompt_results.csv   — one row per (layer, prompt)
  per_layer_summary.csv    — one row per layer
  cross_layer_corr.csv     — predictor vs target correlations across layers
  plots/
    projection_quality_by_layer.png   — MCC + Spearman ρ across layers
    steering_acc_by_layer.png         — steered greedy acc, one line per factor
    projection_vs_steering.png        — scatter: projection MCC vs steered acc
    kappa_scatter_layer_{N}.png       — κ_a vs baseline_correct per prompt

Usage:
    uv run python axbench/scripts/mcqa_projection_link.py \\
        --behavior corrigible-neutral-HHH \\
        --model_name google/gemma-2-9b-it \\
        --layers 10-32 \\
        --factors 1,2,3,5,10

    # Loop over behaviors:
    for b in sycophancy hallucination corrigible-neutral-HHH myopic-reward survival-instinct; do
        uv run python axbench/scripts/mcqa_projection_link.py --behavior $b --factors 1,2,3,5,10
    done
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
from sklearn.metrics import matthews_corrcoef
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BEHAVIORS = [
    "sycophancy", "survival-instinct", "corrigible-neutral-HHH",
    "hallucination", "refusal", "myopic-reward", "coordinate-other-ais",
]
TEST_PATH_MAP  = {b: f"datasets/test/{b}/test_dataset_ab.json"  for b in BEHAVIORS}
TRAIN_PATH_MAP = {b: f"datasets/raw/{b}/dataset.json"           for b in BEHAVIORS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_layer_range(spec: str) -> list:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(x) for x in spec.split(","))


def extract_letter(answer_str: str) -> str:
    m = re.search(r"\(([A-Z])\)", answer_str.upper())
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse letter from '{answer_str}'")


def supports_chat_template(tok) -> bool:
    return getattr(tok, "chat_template", None) not in (None, "")


def build_full_ids(tokenizer, question: str, answer_letter: str):
    """
    Returns (prompt_ids, full_ids, open_paren_pos) where open_paren_pos is the
    index of the '(' token — the model predicts the next token here, giving the
    letter choice without needing any continuation.
    """
    cand = f" ({answer_letter})"
    if supports_chat_template(tokenizer):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question},
             {"role": "assistant", "content": cand}],
            tokenize=True, add_generation_prompt=False,
        )
    else:
        prompt_ids = tokenizer.encode(question)
        full_ids = tokenizer.encode(question + cand)

    # Find '(' position (the token just before the answer letter).
    letter_pos = None
    for i in range(len(full_ids) - 1, max(len(prompt_ids) - 1, 0), -1):
        if tokenizer.decode([full_ids[i]]).strip() == answer_letter:
            letter_pos = i
            break
    if letter_pos is None:
        raise RuntimeError(f"Cannot find '{answer_letter}' in suffix of full_ids")
    open_paren_pos = letter_pos - 1
    return prompt_ids, full_ids, letter_pos, open_paren_pos


def pad_batch(token_lists, pad_id, device):
    max_len = max(len(t) for t in token_lists)
    B = len(token_lists)
    ids  = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    lens = []
    for i, t in enumerate(token_lists):
        L = len(t)
        ids[i, :L]  = torch.tensor(t, dtype=torch.long, device=device)
        mask[i, :L] = 1
        lens.append(L)
    return ids, mask, lens


def compute_dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    gap = abs(pos.mean() - neg.mean())
    pooled = np.sqrt(0.5 * (pos.var() + neg.var()))
    return float(gap / pooled) if pooled > 1e-12 else 0.0


def safe_mcc(pred, actual):
    """MCC; returns nan if either array is constant (degenerate)."""
    pred, actual = np.asarray(pred, int), np.asarray(actual, int)
    if pred.std() < 1e-9 or actual.std() < 1e-9:
        return float("nan")
    return float(matthews_corrcoef(actual, pred))


def safe_spearman(x, y):
    """Spearman ρ; returns (nan, nan) if either array is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.nanstd(x) < 1e-9 or np.nanstd(y) < 1e-9:
        return float("nan"), float("nan")
    rho, p = scipy_stats.spearmanr(x, y)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# Forward-pass utilities (borrowed from mcqa_adaptive_steer.py approach)
# ---------------------------------------------------------------------------
def make_capture_hook(storage, layer_idx):
    def hook(mod, inp, out):
        storage[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook


def make_steering_hook(sv, factor, prefix_length):
    sv = sv.detach().clone()
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = h.clone()
        new_h[:, prefix_length:] += factor * sv.to(new_h.dtype).to(new_h.device)
        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h
    return hook


@torch.no_grad()
def forward_capture(model, input_ids, attention_mask, layers, open_paren_positions):
    """
    Run forward pass, capture hidden states at target layers, and decode the
    greedy next-token at each item's open_paren_position (= the letter choice).
    Returns: (next_token_ids [B], layer_hiddens {layer: tensor [B,L,H] cpu float32})
    """
    storage = {}
    handles = [
        model.model.layers[l].register_forward_hook(
            make_capture_hook(storage, l), always_call=True
        )
        for l in layers
    ]
    try:
        hs = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    finally:
        for h in handles:
            h.remove()

    next_toks = []
    for i in range(input_ids.shape[0]):
        pos = int(open_paren_positions[i])
        logit = model.lm_head(hs[i, pos, :].unsqueeze(0)).float().squeeze(0)
        next_toks.append(int(logit.argmax().item()))
    del hs
    return next_toks, {l: storage[l].float().cpu() for l in layers}


@torch.no_grad()
def forward_steered(model, input_ids, attention_mask, open_paren_positions,
                    layer, sv, factor, prefix_length):
    """Steered forward pass; returns greedy next-token ids at open_paren_positions."""
    handle = model.model.layers[layer].register_forward_hook(
        make_steering_hook(sv, factor, prefix_length), always_call=True
    )
    try:
        hs = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    finally:
        handle.remove()

    next_toks = []
    for i in range(input_ids.shape[0]):
        pos = int(open_paren_positions[i])
        logit = model.lm_head(hs[i, pos, :].unsqueeze(0)).float().squeeze(0)
        next_toks.append(int(logit.argmax().item()))
    del hs
    return next_toks


def decode_letter(tokenizer, token_id):
    s = tokenizer.decode([token_id]).strip()
    return s if (len(s) == 1 and s.isupper() and s.isalpha()) else None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_projection_quality(layer_df, behavior, out_path):
    """MCC of sign(κ_a) and Spearman ρ of κ_a, both vs baseline_correct, by layer."""
    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    ax1.plot(layers, layer_df["sign_kappa_mcc"].values, "o-",
             color="#2E86AB", linewidth=2.5, markersize=7,
             label="sign(κ_a) MCC vs baseline correct (left)")
    ax1.axhline(0, color="#2E86AB", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("MCC  [sign(κ_a) → correct?]", color="#2E86AB", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2E86AB")

    ax2.plot(layers, layer_df["kappa_spearman_rho"].values, "s--",
             color="#E94F37", linewidth=2, markersize=6,
             label="κ_a Spearman ρ vs baseline correct (right)")
    ax2.axhline(0, color="#E94F37", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Spearman ρ  [κ_a → correct?]", color="#E94F37", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#E94F37")

    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_xticks(layers)
    ax1.set_title(
        f"{behavior}: How well does the diff-in-means projection predict baseline behavior?\n"
        f"MCC uses sign(κ_a) as binary predictor; Spearman uses raw κ_a",
        fontsize=11, fontweight="bold",
    )
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_steering_acc(layer_df, factors, behavior, out_path):
    """Steered greedy accuracy by layer, one line per factor. Baseline shown as reference."""
    layers = layer_df["layer"].values
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(layers, layer_df["baseline_acc"].values, "D--", color="gray",
            linewidth=1.5, markersize=6, label="Baseline (α=0)", alpha=0.8)
    cmap = plt.get_cmap("plasma")
    for i, f in enumerate(factors):
        col = f"steered_acc_{f:g}"
        if col not in layer_df.columns:
            continue
        color = cmap(i / max(1, len(factors) - 1))
        ax.plot(layers, layer_df[col].values, "o-", color=color,
                linewidth=2, markersize=5, label=f"α={f:g}")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Greedy accuracy (correct letter generated)", fontsize=11)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{behavior}: Steered greedy accuracy by layer\n"
        f"Evaluation: actual generated token at '(' position (gibberish = wrong)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(title="Steering factor", fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_steering_and_dprime(layer_df, factors, behavior, out_path):
    """Steering accuracy for all factors + baseline + d' on one plot with dual y-axes."""
    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=(13, 5))

    # --- left axis: accuracy ---
    ax1.plot(layers, layer_df["baseline_acc"].values, "D--", color="gray",
             linewidth=1.5, markersize=6, label="Baseline (α=0)", alpha=0.85, zorder=3)
    cmap = plt.get_cmap("plasma")
    factor_cols = [f for f in factors if f"steered_acc_{f:g}" in layer_df.columns]
    for i, f in enumerate(factor_cols):
        color = cmap(i / max(1, len(factor_cols) - 1))
        ax1.plot(layers, layer_df[f"steered_acc_{f:g}"].values, "o-", color=color,
                 linewidth=2, markersize=5, label=f"α={f:g}", zorder=3)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Greedy accuracy", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(layers)

    # --- right axis: d' ---
    ax2 = ax1.twinx()
    if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d'", zorder=2)
        ax2.set_ylabel("d'  (training discriminability)", fontsize=11, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.set_ylim(bottom=0)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, title="Steering factor / metric",
               fontsize=9, loc="upper left", framealpha=0.85)

    ax1.set_title(
        f"{behavior}: Steering accuracy & training d' by layer",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_projection_vs_steering(layer_df, factors, behavior, out_path):
    """
    The key plot: scatter of projection MCC (x) vs steered accuracy (y),
    one point per layer, one panel per factor + one panel for best-α-per-layer.
    """
    panels = []
    for f in factors:
        if f == "best":
            if "best_steered_acc" in layer_df.columns:
                panels.append(("best_steered_acc", "Best α per layer"))
        else:
            col = f"steered_acc_{f:g}"
            if col in layer_df.columns:
                panels.append((col, f"α={f:g}"))
    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for idx, (col, label) in enumerate(panels):
        ax = axes[0][idx]
        x = layer_df["sign_kappa_mcc"].values
        y = layer_df[col].values
        lyrs = layer_df["layer"].values

        sc = ax.scatter(x, y, c=lyrs, cmap="viridis", s=80,
                        edgecolors="k", linewidths=0.4, zorder=3)
        plt.colorbar(sc, ax=ax, label="Layer")

        for xi, yi, li in zip(x, y, lyrs):
            if li % 3 == 0 and not (np.isnan(xi) or np.isnan(yi)):
                ax.annotate(str(li), (xi, yi), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")

        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 3 and np.std(x[mask]) > 1e-9 and np.std(y[mask]) > 1e-9:
            r, p = scipy_stats.pearsonr(x[mask], y[mask])
            ax.set_title(f"{label}   r={r:+.3f} (p={p:.3g})", fontsize=11, fontweight="bold")
        else:
            ax.set_title(label, fontsize=11, fontweight="bold")

        ax.set_xlabel("sign(κ_a) MCC vs baseline correct", fontsize=10)
        ax.set_ylabel("Steered greedy accuracy", fontsize=10)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(layer_df["baseline_acc"].mean(), color="gray",
                   linestyle="--", alpha=0.4, label="mean baseline acc")

    fig.suptitle(
        f"{behavior}: Projection quality → steering effectiveness (one point = one layer)\n"
        f"Rightmost panel = best steering factor at each layer",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kappa_scatter(prompt_df, layer, behavior, out_path):
    """κ_a (x) vs baseline_correct (0/1 jittered, y) for one layer."""
    ldf = prompt_df[prompt_df["layer"] == layer].dropna(subset=["kappa_a"])
    if len(ldf) < 4:
        return
    kappa = ldf["kappa_a"].values
    correct = ldf["baseline_correct"].astype(int).values

    jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(correct))
    colors = np.where(correct == 1, "#2ca02c", "#d62728")

    mcc = safe_mcc((kappa > 0).astype(int), correct)
    rho, _ = safe_spearman(kappa, correct)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(kappa, correct + jitter, c=colors, s=60,
               edgecolors="k", linewidths=0.4, alpha=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2,
               label="κ=0 (centroid boundary)")
    ax.set_xlabel("κ_a  (projection onto diff-in-means line)", fontsize=11)
    ax.set_ylabel("Baseline correct (1=yes, 0=no)", fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No behavior (0)", "Shows behavior (1)"])
    ax.set_title(
        f"{behavior} — Layer {layer}\n"
        f"sign(κ_a) MCC={mcc:+.3f}   Spearman ρ={rho:+.3f}   "
        f"(green=correct, red=incorrect)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True, choices=BEHAVIORS)
    parser.add_argument("--model_name", default="google/gemma-2-9b-it")
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--test_path",  default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--layers", default="10-32",
                        help="'10-32' or '10,15,20,32'")
    parser.add_argument("--max_examples", type=int, default=300,
                        help="Max training pairs for DiffMean (random sample). Default 300.")
    parser.add_argument("--factors", "--steering-factors", "--steering_factors",
                        dest="factors", default="1,2,3,5,10",
                        help="Comma-separated fixed steering factors. Default 1,2,3,5,10.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_recompute", action="store_true",
                        help="Ignore saved per_prompt_results.csv and rerun forward passes.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layers = parse_layer_range(args.layers)
    factors = sorted({float(x.strip()) for x in args.factors.split(",") if x.strip()})
    if not factors:
        logger.error("No steering factors given.")
        sys.exit(1)

    model_short = args.model_name.split("/")[-1]
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path("results") / "mcqa_projection_link" / model_short / args.behavior
    )
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path or TRAIN_PATH_MAP[args.behavior])
    test_path  = Path(args.test_path  or TEST_PATH_MAP[args.behavior])

    with open(train_path) as f:
        train_data = json.load(f)
    if args.max_examples and args.max_examples < len(train_data):
        idxs = random.sample(range(len(train_data)), args.max_examples)
        train_data = [train_data[i] for i in sorted(idxs)]
    logger.warning(f"Training pairs: {len(train_data)}")

    with open(test_path) as f:
        test_data = json.load(f)
    if args.max_test:
        test_data = test_data[:args.max_test]
    logger.warning(f"Test prompts: {len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Check if we can skip forward passes (reuse saved CSV).
    # d' is stored separately in dprime.json so it survives CSV reuse.
    # ------------------------------------------------------------------
    per_prompt_csv = out_dir / "per_prompt_results.csv"
    dprime_json    = out_dir / "dprime.json"
    per_prompt_df  = None
    dprimes        = {}  # populated either from file or Phase 0

    if (not args.force_recompute) and per_prompt_csv.exists():
        try:
            df = pd.read_csv(per_prompt_csv)
            have_layers  = set(int(l) for l in df["layer"].unique())
            have_factors = set(
                float(c.replace("steered_correct_", ""))
                for c in df.columns if c.startswith("steered_correct_")
            )
            need_layers  = set(layers)
            need_factors = set(factors)
            if need_layers <= have_layers and need_factors <= have_factors:
                per_prompt_df = df
                logger.warning(
                    f"Reusing {per_prompt_csv} — covers all requested layers/factors. "
                    f"Pass --force_recompute to redo."
                )
            else:
                missing_l = need_layers - have_layers
                missing_f = need_factors - have_factors
                logger.warning(
                    f"CSV exists but missing layers={missing_l or 'none'} "
                    f"factors={missing_f or 'none'}; recomputing."
                )
        except Exception as e:
            logger.warning(f"Could not reuse CSV: {e}")

    # Load d' from file if available (may exist even when CSV does too).
    if dprime_json.exists():
        try:
            with open(dprime_json) as f:
                dprimes = {int(k): float(v) for k, v in json.load(f).items()}
            logger.warning(f"Loaded d' for {len(dprimes)} layers from {dprime_json}")
        except Exception as e:
            logger.warning(f"Could not load {dprime_json}: {e}")

    # If CSV is reusable but d' is still missing, run Phase 0 only (no Phase A/B).
    need_dprime = not dprimes and per_prompt_df is not None
    need_full   = per_prompt_df is None

    if need_dprime or need_full:
        # ------------------------------------------------------------------
        # Load tokenizer + model.
        # ------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prefix_length = get_prefix_length(tokenizer) if args.model_name in CHAT_MODELS else 1
        logger.warning(f"prefix_length = {prefix_length}")

        logger.warning(f"Loading model {args.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.use_bf16 else None,
            device_map=device,
        )
        model.eval()
        pad_id = tokenizer.pad_token_id
        B = args.batch_size

        # ------------------------------------------------------------------
        # Phase 0: DiffMean vectors from training data.
        # Activation captured at the answer-letter token position.
        # ------------------------------------------------------------------
        logger.warning(f"\n=== Phase 0: DiffMean ({len(train_data)} pairs, {len(layers)} layers) ===")

        train_flat = []
        for item in train_data:
            q = item["question"]
            for answer, label in [
                (item["answer_matching_behavior"], 1),
                (item["answer_not_matching_behavior"], 0),
            ]:
                try:
                    letter = extract_letter(answer)
                    _, full_ids, letter_pos, open_paren_pos = build_full_ids(tokenizer, q, letter)
                    train_flat.append({"full_ids": full_ids, "letter_pos": letter_pos, "label": label})
                except Exception:
                    pass

        pos_acts = {l: [] for l in layers}
        neg_acts = {l: [] for l in layers}

        for start in tqdm(range(0, len(train_flat), B), desc="Phase 0"):
            batch = train_flat[start:start + B]
            ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
            dummy_pos = [0] * len(batch)
            _, hiddens = forward_capture(model, ids, mask, layers, dummy_pos)
            for i, b in enumerate(batch):
                for l in layers:
                    act = hiddens[l][i, b["letter_pos"], :]
                    (pos_acts[l] if b["label"] == 1 else neg_acts[l]).append(act)
            del hiddens, ids, mask

        steering_vecs, mu_poss, mu_negs, dprimes = {}, {}, {}, {}
        for l in layers:
            if not pos_acts[l] or not neg_acts[l]:
                continue
            mu_pos = torch.stack(pos_acts[l]).mean(0)
            mu_neg = torch.stack(neg_acts[l]).mean(0)
            v = mu_pos - mu_neg
            v_norm_sq = float(v.dot(v))
            mu_poss[l] = mu_pos
            mu_negs[l] = mu_neg
            steering_vecs[l] = v.to(device)
            mu_mid = 0.5 * (mu_pos + mu_neg)
            if v_norm_sq > 1e-12:
                pp = np.array([float(2.0*(a-mu_mid).dot(v)/v_norm_sq) for a in pos_acts[l]])
                np_ = np.array([float(2.0*(a-mu_mid).dot(v)/v_norm_sq) for a in neg_acts[l]])
                dprimes[l] = compute_dprime(pp, np_)
            else:
                dprimes[l] = 0.0
            logger.warning(f"  L{l:2d}: d'={dprimes[l]:.3f}  ||v||={float(v.norm()):.3f}")

        layers = [l for l in layers if l in steering_vecs]
        del pos_acts, neg_acts

        # Save d' so it survives future CSV-reuse runs.
        with open(dprime_json, "w") as f:
            json.dump({str(l): float(v) for l, v in dprimes.items()}, f, indent=2)
        logger.warning(f"Saved d' to {dprime_json}")

        if not need_full:
            logger.warning("Phase 0 complete (d' only). Skipping Phase A/B — CSV reused.")
        else:
            # ----------------------------------------------------------------
            # Tokenize test prompts.
            # ----------------------------------------------------------------
            flat_items = []
            for j, item in enumerate(test_data):
                try:
                    ml = extract_letter(item["answer_matching_behavior"])
                    nl = extract_letter(item["answer_not_matching_behavior"])
                except Exception:
                    continue
                if ml == nl:
                    continue
                try:
                    _, full_ids, letter_pos, open_paren_pos = build_full_ids(tokenizer, item["question"], ml)
                except Exception:
                    continue
                flat_items.append({
                    "prompt_idx": j,
                    "matching_letter": ml,
                    "notmatch_letter": nl,
                    "full_ids": full_ids,
                    "letter_pos": letter_pos,
                    "open_paren_pos": open_paren_pos,
                })
            n_items = len(flat_items)
            logger.warning(f"Valid test items: {n_items}")

            # ----------------------------------------------------------------
            # Phase A: Baseline greedy decode + activation capture.
            # ----------------------------------------------------------------
            logger.warning(f"\n=== Phase A: Baseline ({n_items} prompts) ===")

            baseline_tok = {}    # j -> int (token id)
            act_at_letter = {}   # (j, l) -> tensor [H]

            for start in tqdm(range(0, n_items, B), desc="Phase A"):
                batch = flat_items[start:start + B]
                ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
                toks, hiddens = forward_capture(
                    model, ids, mask, layers,
                    [b["open_paren_pos"] for b in batch],
                )
                for i, b in enumerate(batch):
                    j = b["prompt_idx"]
                    baseline_tok[j] = toks[i]
                    for l in layers:
                        act_at_letter[(j, l)] = hiddens[l][i, b["letter_pos"], :]
                del hiddens, ids, mask

            # Build baseline records.
            baseline_records = {}
            for b in flat_items:
                j = b["prompt_idx"]
                g = decode_letter(tokenizer, baseline_tok[j])
                baseline_records[j] = {
                    "question": test_data[j]["question"],
                    "answer_matching_behavior": test_data[j]["answer_matching_behavior"],
                    "matching_letter": b["matching_letter"],
                    "notmatch_letter": b["notmatch_letter"],
                    "baseline_token": g if g else tokenizer.decode([baseline_tok[j]]).strip(),
                    "baseline_on_format": g is not None,
                    "baseline_correct": g == b["matching_letter"],
                }

            # Compute κ_a per (layer, prompt).
            kappa_map = {}
            for l in layers:
                mu_pos, mu_neg = mu_poss[l], mu_negs[l]
                v = mu_pos - mu_neg
                v_norm_sq = float(v.dot(v))
                mu = 0.5 * (mu_pos + mu_neg)
                for b in flat_items:
                    j = b["prompt_idx"]
                    act = act_at_letter[(j, l)]
                    kappa_map[(l, j)] = float(2.0 * (act - mu).dot(v) / v_norm_sq) \
                        if v_norm_sq > 1e-12 else float("nan")
            del act_at_letter

            # ----------------------------------------------------------------
            # Phase B: Fixed-factor steered greedy decode per (layer, factor).
            # ----------------------------------------------------------------
            logger.warning(f"\n=== Phase B: Steered ({len(layers)} layers × {len(factors)} factors) ===")

            steered_tok = {}  # (factor, l, j) -> int

            for l in tqdm(layers, desc="Phase B layers"):
                sv = steering_vecs[l]
                for factor in factors:
                    results = []
                    for start in range(0, n_items, B):
                        batch = flat_items[start:start + B]
                        ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
                        toks = forward_steered(
                            model, ids, mask,
                            [b["open_paren_pos"] for b in batch],
                            l, sv, factor, prefix_length,
                        )
                        for i, b in enumerate(batch):
                            results.append((b["prompt_idx"], toks[i]))
                        del ids, mask
                    for j, tid in results:
                        steered_tok[(factor, l, j)] = tid

            # ----------------------------------------------------------------
            # Assemble per-prompt DataFrame.
            # ----------------------------------------------------------------
            rows = []
            for l in layers:
                for b in flat_items:
                    j = b["prompt_idx"]
                    rec = baseline_records[j]
                    row = {
                        "layer": l,
                        "prompt_idx": j,
                        "question": rec["question"],
                        "answer_matching_behavior": rec["answer_matching_behavior"],
                        "matching_letter": rec["matching_letter"],
                        "notmatch_letter": rec["notmatch_letter"],
                        "kappa_a": kappa_map.get((l, j), float("nan")),
                        "baseline_token": rec["baseline_token"],
                        "baseline_on_format": rec["baseline_on_format"],
                        "baseline_correct": rec["baseline_correct"],
                    }
                    for factor in factors:
                        tid = steered_tok.get((factor, l, j))
                        if tid is None:
                            row[f"steered_token_{factor:g}"] = ""
                            row[f"steered_on_format_{factor:g}"] = False
                            row[f"steered_correct_{factor:g}"] = False
                            row[f"steering_factor"] = factor
                        else:
                            g = decode_letter(tokenizer, tid)
                            row[f"steered_token_{factor:g}"] = g or tokenizer.decode([tid]).strip()
                            row[f"steered_on_format_{factor:g}"] = g is not None
                            row[f"steered_correct_{factor:g}"] = g == b["matching_letter"]
                    rows.append(row)

            per_prompt_df = pd.DataFrame(rows)
            per_prompt_df.to_csv(per_prompt_csv, index=False)
            logger.warning(f"Saved {len(per_prompt_df)} rows to {per_prompt_csv}")

    # ------------------------------------------------------------------
    # Analysis: per-layer summary.
    # ------------------------------------------------------------------

    layer_rows = []
    for l in sorted(per_prompt_df["layer"].unique()):
        ldf = per_prompt_df[per_prompt_df["layer"] == l]
        kappa = ldf["kappa_a"].values
        correct = ldf["baseline_correct"].astype(int).values

        sign_pred = (kappa > 0).astype(int)
        mcc = safe_mcc(sign_pred[~np.isnan(kappa)], correct[~np.isnan(kappa)])
        rho, rho_p = safe_spearman(kappa, correct.astype(float))

        row = {
            "layer": int(l),
            "dprime": dprimes.get(int(l), float("nan")),
            "n_prompts": int(len(ldf)),
            "baseline_acc": float(ldf["baseline_correct"].mean()),
            "baseline_off_format": float((~ldf["baseline_on_format"]).mean()),
            "sign_kappa_mcc": mcc,
            "kappa_spearman_rho": rho,
            "kappa_spearman_p": rho_p,
        }
        for factor in factors:
            col_c = f"steered_correct_{factor:g}"
            col_f = f"steered_on_format_{factor:g}"
            if col_c in ldf.columns:
                row[f"steered_acc_{factor:g}"] = float(ldf[col_c].mean())
                row[f"steered_off_format_{factor:g}"] = float((~ldf[col_f]).mean())

        layer_rows.append(row)

    layer_df = pd.DataFrame(layer_rows).sort_values("layer").reset_index(drop=True)

    # Add best-across-factors steered accuracy per layer.
    acc_cols = [f"steered_acc_{f:g}" for f in factors if f"steered_acc_{f:g}" in layer_df.columns]
    if acc_cols:
        layer_df["best_steered_acc"] = layer_df[acc_cols].max(axis=1)
        layer_df["best_factor"] = layer_df[acc_cols].idxmax(axis=1).str.replace("steered_acc_", "")

    layer_df.to_csv(out_dir / "per_layer_summary.csv", index=False)

    logger.warning("\nPer-layer summary:")
    for _, r in layer_df.iterrows():
        steered_str = "  ".join(
            f"α={f:g}={r[f'steered_acc_{f:g}']:.3f}"
            for f in factors if f"steered_acc_{f:g}" in r and not pd.isna(r[f"steered_acc_{f:g}"])
        )
        best_str = f"  best={r['best_steered_acc']:.3f}@α={r['best_factor']}" \
            if "best_steered_acc" in r else ""
        logger.warning(
            f"  L{int(r['layer']):2d}: d'={r['dprime']:.3f}  "
            f"base={r['baseline_acc']:.3f}  "
            f"MCC={r['sign_kappa_mcc']:+.3f}  ρ={r['kappa_spearman_rho']:+.3f}  "
            f"{steered_str}{best_str}"
        )

    # Cross-layer correlations: predictor vs each factor AND vs best-across-factors.
    cross_rows = []
    predictors = [("sign_kappa_mcc", "sign(κ_a) MCC"), ("kappa_spearman_rho", "κ_a Spearman ρ"),
                  ("dprime", "d'")]

    targets = [(f"steered_acc_{f:g}", f"α={f:g}") for f in factors
               if f"steered_acc_{f:g}" in layer_df.columns]
    if "best_steered_acc" in layer_df.columns:
        targets.append(("best_steered_acc", "best α per layer"))

    for pred_col, pred_label in predictors:
        if pred_col not in layer_df.columns:
            continue
        pv = layer_df[pred_col].values
        for tgt_col, tgt_label in targets:
            if tgt_col not in layer_df.columns:
                continue
            tv = layer_df[tgt_col].values
            mask = ~(np.isnan(pv) | np.isnan(tv))
            if mask.sum() < 3 or np.std(pv[mask]) < 1e-9 or np.std(tv[mask]) < 1e-9:
                continue
            r, p = scipy_stats.pearsonr(pv[mask], tv[mask])
            rho, sp = scipy_stats.spearmanr(pv[mask], tv[mask])
            cross_rows.append({
                "predictor": pred_label, "target": tgt_label,
                "pearson_r": float(r), "pearson_p": float(p),
                "spearman_rho": float(rho), "spearman_p": float(sp),
                "n_layers": int(mask.sum()),
            })
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(out_dir / "cross_layer_corr.csv", index=False)
    if not cross_df.empty:
        logger.warning("\nCross-layer correlations (predictor → steered accuracy):")
        for _, r in cross_df.iterrows():
            sig = "*" if r["pearson_p"] < 0.05 else " "
            logger.warning(
                f"  {r['predictor']:20s} → {r['target']:20s}: "
                f"r={r['pearson_r']:+.3f} (p={r['pearson_p']:.3g}){sig}  "
                f"ρ={r['spearman_rho']:+.3f} (p={r['spearman_p']:.3g})"
            )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plots = out_dir / "plots"
    plot_projection_quality(layer_df, args.behavior, plots / "projection_quality_by_layer.png")
    plot_steering_acc(layer_df, factors, args.behavior, plots / "steering_acc_by_layer.png")
    plot_steering_and_dprime(layer_df, factors, args.behavior, plots / "steering_acc_and_dprime.png")
    plot_projection_vs_steering(
        layer_df,
        factors + (["best"] if "best_steered_acc" in layer_df.columns else []),
        args.behavior,
        plots / "projection_vs_steering.png",
    )
    for l in sorted(per_prompt_df["layer"].unique()):
        plot_kappa_scatter(per_prompt_df, int(l), args.behavior,
                           plots / f"kappa_scatter_layer_{int(l)}.png")

    # Summary JSON.
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "layers": list(map(int, layer_df["layer"].values)),
        "factors": factors,
        "n_test_prompts": int(len(per_prompt_df["prompt_idx"].unique())),
        "per_layer": layer_df.to_dict(orient="records"),
        "cross_layer_corr": cross_df.to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
