"""
MCQA test-time steering + projection analysis.

Consumes artifacts produced by `mcqa_train_diffmean_sweep.py` (per-layer DiffMean
weight, mu_pos, mu_neg, d').

For each of the 50 MCQA test prompts in `datasets/test/<behavior>/test_dataset_ab.json`:

  1. (Baseline) Score both candidate strings ' (A)' and ' (B)' by log P(candidate tokens |
     prompt). Softmax between the two -> P(matching). Record greedy_match.
  2. (Projection) During those baseline forward passes, capture the hidden state at the
     answer-letter position at every target layer. The letter whose forward pass is
     used is the GREEDY choice (whichever candidate the model assigned higher log-prob).
     For each layer l, compute kappa_a with the saved mu_pos, mu_neg:
         kappa_a = 2 * (a - mu) . v / ||v||^2     where v = mu_pos - mu_neg, mu = (mu_pos + mu_neg)/2
  3. (Steered, factor=alpha) Per layer l, reinstall a forward hook that adds
     alpha * v to layer l's output at positions >= prefix_length, rescore both
     candidates, compute P_steered(matching).

EXPERIMENT 1 (cross-layer link):
  Plot d'_l (training) and steered_acc_l (test, at alpha=1) as two lines vs layer.
  Report Pearson/Spearman between the two series across layers.

EXPERIMENT 2a (sanity check — does the diff-in-means line encode behavior?):
  Per layer l, scatter kappa_a_j (x) vs baseline P(matching)_j (y) over j = 1..50.
  Report per-layer Spearman / Pearson.

Usage:
    uv run python axbench/scripts/mcqa_test_steer_and_project.py \
        --behavior corrigible-neutral-HHH \
        --model_name google/gemma-2-9b-it \
        --sweep_dir results/mcqa_sweep/gemma-2-9b-it/corrigible-neutral-HHH \
        --steering-factors=1,2,3,5,10
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


# ============================================================================
# Helpers
# ============================================================================
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def discover_layers(sweep_dir: Path) -> list:
    layers = []
    for d in sweep_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            if (d / "DiffMean_weight.pt").exists() and (d / "mu_pos.pt").exists():
                layers.append(int(d.name.split("_")[1]))
    return sorted(layers)


def extract_matching_letter(answer_str: str) -> str:
    m = re.search(r"\(([A-Z])\)", answer_str.upper())
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse A/B letter from '{answer_str}'")


# ============================================================================
# Tokenization: prompt + candidate (' (A)' or ' (B)')
# ============================================================================
def build_prompt_and_full_ids(tokenizer, question: str, candidate: str):
    """
    Returns (prompt_ids, full_ids, letter_pos) for the single example.

    `letter_pos` is the absolute index into full_ids pointing at the A/B token.
    """
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

    # Find letter position: search backwards from end for the A/B token.
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


def answer_token_logprob(logits: torch.Tensor, full_ids, prompt_len: int) -> float:
    """
    logits: [seq_len, vocab] on any device. full_ids: tensor or list of ids on CPU.
    Returns sum of log P(token_i | prefix) for i in [prompt_len, seq_len).
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1).cpu()
    if isinstance(full_ids, torch.Tensor):
        ids = full_ids.cpu().tolist()
    else:
        ids = list(full_ids)
    total = 0.0
    for i in range(prompt_len, len(ids)):
        total += float(log_probs[i - 1, int(ids[i])].item())
    return total


def batched_answer_logprobs(logits: torch.Tensor, input_ids: torch.Tensor,
                            prompt_lens, lengths):
    """
    Vectorized sum of log-probs over answer tokens.

    logits:     [B, L, V]  (on device)
    input_ids:  [B, L]     (on device)
    prompt_lens: list[int] length B
    lengths:     list[int] length B  (= len(full_ids) before padding)

    Returns a list of Python floats of length B.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [B, L]
    # log P(tok_k) corresponds to gathered[:, k] using logits[:, k-1]
    # so we want sum_{k=prompt_lens[i]}^{lengths[i]-1} gathered_shifted[i, k]
    # where gathered_shifted[i, k] = log_probs[i, k-1, input_ids[i, k]].
    results = []
    for i in range(logits.shape[0]):
        p_len = int(prompt_lens[i])
        L_i = int(lengths[i])
        if L_i <= p_len:
            results.append(0.0)
            continue
        # Index k runs over [p_len, L_i) for answer tokens.
        # log P(tok at k | prefix) uses distribution at position k-1.
        idx = torch.arange(p_len, L_i, device=logits.device)
        token_ids = input_ids[i, idx]
        dist = log_probs[i, idx - 1]  # [n_answer, V]
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
# Forward passes: baseline (with multi-layer activation capture) + steered
# ============================================================================
def make_steering_hook(steering_vec: torch.Tensor, factor: float, prefix_length: int):
    sv = steering_vec.detach().clone()

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            new_h = h.clone()
            new_h[:, prefix_length:] = (
                new_h[:, prefix_length:] + factor * sv.to(new_h.dtype).to(new_h.device)
            )
            return (new_h,) + out[1:]
        new_h = out.clone()
        new_h[:, prefix_length:] = (
            new_h[:, prefix_length:] + factor * sv.to(new_h.dtype).to(new_h.device)
        )
        return new_h

    return hook


def make_capture_hook(storage: dict, layer_idx: int):
    def hook(mod, inp, out):
        storage[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook


@torch.no_grad()
def forward_capture_batch(model, input_ids, attention_mask, target_layers):
    """Batched forward pass with per-layer hidden-state capture.

    Returns:
        logits: [B, L, V]
        layer_hiddens: {layer_idx: [B, L, hidden]}
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
def forward_with_steering_batch(model, input_ids, attention_mask,
                                 steering_layer: int, steering_vec: torch.Tensor,
                                 factor: float, prefix_length: int):
    """Batched forward pass with a steering addition hook at one layer."""
    if factor == 0.0:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits
    h = model.model.layers[steering_layer].register_forward_hook(
        make_steering_hook(steering_vec, factor, prefix_length),
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
def _format_factor_tag(layer_df: pd.DataFrame) -> str:
    """Return an α-label based on the steering_factor column (if present & unique)."""
    if "steering_factor" in layer_df.columns:
        uniq = sorted(set(float(x) for x in layer_df["steering_factor"].dropna().unique()))
        if len(uniq) == 1:
            return f"α={uniq[0]:g}"
    return "α=steering"


def _plot_predictor_vs_steering(
    layer_df: pd.DataFrame, pred_col: str, pred_label: str, pred_color: str,
    behavior: str, output_path: Path, title_suffix: str,
):
    """Dual-y-axis line plot: a per-layer predictor + steered/baseline acc by layer."""
    if layer_df.empty or pred_col not in layer_df.columns:
        return
    sub = layer_df.dropna(subset=[pred_col, "steered_acc_mean"])
    if len(sub) < 3:
        return
    alpha_tag = _format_factor_tag(sub)
    layers = sub["layer"].values
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(layers, sub[pred_col].values, "o-", color=pred_color,
             linewidth=2.5, markersize=7, label=pred_label)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel(pred_label, color=pred_color, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=pred_color)
    ax1.set_xticks(layers)

    color_st = "#2E86AB"
    ax2 = ax1.twinx()
    ax2.plot(layers, sub["steered_acc_mean"].values, "s-", color=color_st,
             linewidth=2, markersize=6, label=f"Steered acc (test, {alpha_tag})")
    ax2.plot(layers, sub["baseline_acc_mean"].values, "D--", color="gray",
             linewidth=1.5, markersize=5, alpha=0.7, label="Baseline acc (test, α=0)")
    ax2.set_ylabel("Test P(matching)", color=color_st, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_st)
    ax2.set_ylim(0, 1.05)

    pred = sub[pred_col].values
    st = sub["steered_acc_mean"].values
    delta = st - sub["baseline_acc_mean"].values
    corr_lines = []
    if len(sub) >= 3:
        r, p = scipy_stats.pearsonr(pred, st)
        rho, sp = scipy_stats.spearmanr(pred, st)
        corr_lines.append(
            f"{pred_label} vs steered_acc:  r={r:+.3f} (p={p:.3f}), ρ={rho:+.3f} (p={sp:.3f})"
        )
        rd, pd_ = scipy_stats.pearsonr(pred, delta)
        rhod, spd = scipy_stats.spearmanr(pred, delta)
        corr_lines.append(
            f"{pred_label} vs Δacc:         r={rd:+.3f} (p={pd_:.3f}), ρ={rhod:+.3f} (p={spd:.3f})"
        )
    if corr_lines:
        ax1.text(0.02, 0.98, "\n".join(corr_lines), transform=ax1.transAxes,
                 fontsize=8, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax1.set_title(f"{behavior}: {title_suffix.format(alpha=alpha_tag)}",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dprime_vs_steering(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    _plot_predictor_vs_steering(
        layer_df, "dprime", "d' (train)", "#E94F37",
        behavior, output_path,
        title_suffix="d' (train) vs Steered Accuracy (test, {alpha}) by Layer",
    )


def plot_kappa_rho_vs_steering(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    _plot_predictor_vs_steering(
        layer_df, "proj_spearman_rho", "ρ(κ_a, P_base) (test, N=50)", "#6B2D5C",
        behavior, output_path,
        title_suffix="Projection quality (κ_a → baseline) vs Steered Accuracy ({alpha}) by Layer",
    )


def plot_combined_predictors(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    """Overlay both predictors (d' and κ_a ρ) alongside steered accuracy for comparison."""
    if layer_df.empty:
        return
    sub = layer_df.dropna(subset=["steered_acc_mean"])
    if len(sub) < 3:
        return
    alpha_tag = _format_factor_tag(sub)
    layers = sub["layer"].values

    fig, ax1 = plt.subplots(figsize=(11, 5))
    color_dp, color_kappa, color_st = "#E94F37", "#6B2D5C", "#2E86AB"

    def _z(x):
        x = np.asarray(x, dtype=float)
        mask = ~np.isnan(x)
        if mask.sum() < 2 or np.nanstd(x) < 1e-12:
            return np.zeros_like(x)
        return (x - np.nanmean(x)) / np.nanstd(x)

    if "dprime" in sub.columns and sub["dprime"].notna().any():
        ax1.plot(layers, _z(sub["dprime"].values), "o-", color=color_dp,
                 linewidth=2, markersize=6, label="d' (train, z-scored)")
    if "proj_spearman_rho" in sub.columns and sub["proj_spearman_rho"].notna().any():
        ax1.plot(layers, _z(sub["proj_spearman_rho"].values), "^-", color=color_kappa,
                 linewidth=2, markersize=6, label="ρ(κ_a, P_base) (test, z-scored)")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Predictor (z-scored)", fontsize=11)
    ax1.set_xticks(layers)

    ax2 = ax1.twinx()
    ax2.plot(layers, sub["steered_acc_mean"].values, "s-", color=color_st,
             linewidth=2.5, markersize=6, label=f"Steered acc (test, {alpha_tag})")
    ax2.set_ylabel("Steered P(matching)", color=color_st, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_st)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax1.set_title(
        f"{behavior}: Train-side (d') vs Test-side (κ_a ρ) predictors of Steering "
        f"({alpha_tag})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sign_agreement_by_layer(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    """Per-layer accuracy of sign(κ_a) as a predictor of greedy_match_baseline.

    Y axis: agreement rate = mean( (κ_a > 0) == greedy_match_baseline ).
    Dashed line = majority-class baseline (what you'd get predicting the majority
    label every time); solid gray = chance (0.5). A secondary line shows MCC on a
    right axis (robust to class imbalance).
    """
    if layer_df.empty or "proj_sign_agreement" not in layer_df.columns:
        return
    sub = layer_df.dropna(subset=["proj_sign_agreement"])
    if len(sub) < 3:
        return
    layers = sub["layer"].values
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(layers, sub["proj_sign_agreement"].values, "o-",
             color="#2E86AB", linewidth=2.5, markersize=7,
             label="Agreement: sign(κ_a) = greedy_match")
    if "proj_sign_majority_baseline" in sub.columns:
        ax1.plot(layers, sub["proj_sign_majority_baseline"].values, "D--",
                 color="gray", linewidth=1.5, markersize=5, alpha=0.7,
                 label="Majority-class baseline")
    ax1.axhline(0.5, color="red", linestyle=":", alpha=0.4, label="Chance (0.5)")
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Binary sign agreement (N=50)", color="#2E86AB", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2E86AB")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(layers)

    if "proj_sign_mcc" in sub.columns:
        ax2 = ax1.twinx()
        ax2.plot(layers, sub["proj_sign_mcc"].values, "s-",
                 color="#E94F37", linewidth=2, markersize=6,
                 label="MCC (sign(κ_a) vs greedy_match)")
        ax2.axhline(0, color="#E94F37", linestyle=":", alpha=0.3)
        ax2.set_ylabel("Matthews correlation", color="#E94F37", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#E94F37")
        ax2.set_ylim(-1.05, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if "proj_sign_mcc" in sub.columns:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    else:
        ax1.legend(lines1, labels1, loc="lower right", fontsize=8)
    ax1.set_title(
        f"{behavior}: Does sign(κ_a) predict the unsteered model's greedy choice?",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_projection_rho_by_layer(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    if layer_df.empty:
        return
    layers = layer_df["layer"].values
    fig, ax = plt.subplots(figsize=(10, 5))

    for col, label, color, marker in [
        ("proj_spearman_rho", "Spearman(κ_a, P_base(matching))", "#2E86AB", "o"),
        ("proj_pearson_r", "Pearson(κ_a, P_base(matching))", "#44AF69", "s"),
    ]:
        if col not in layer_df.columns:
            continue
        vals = layer_df[col].values
        p_col = "proj_spearman_p" if col == "proj_spearman_rho" else "proj_pearson_p"
        pvals = layer_df[p_col].values if p_col in layer_df.columns else np.ones_like(vals)
        ax.plot(layers, vals, f"{marker}-", color=color, linewidth=2, markersize=6,
                label=label, zorder=2)
        sig = pvals < 0.05
        ax.scatter(layers[sig], vals[sig], s=140, facecolors="none",
                   edgecolors=color, linewidths=2.5, zorder=3)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Per-layer correlation (N=50)", fontsize=10)
    ax.set_title(f"{behavior}: Projection κ_a vs Baseline P(matching), per layer",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(fontsize=9, loc="best")
    ax.text(0.02, 0.02, "Circled = p < 0.05",
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter_for_layer(per_prompt: pd.DataFrame, layer: int, behavior: str,
                           output_path: Path):
    ldf = per_prompt[per_prompt["layer"] == layer]
    valid = ldf[["kappa_a", "baseline_p_match"]].dropna()
    if len(valid) < 4:
        return
    rho, sp = scipy_stats.spearmanr(valid["kappa_a"], valid["baseline_p_match"])
    r, pr = scipy_stats.pearsonr(valid["kappa_a"], valid["baseline_p_match"])

    # Binary-agreement stats: sign(κ_a) vs greedy_match_baseline.
    sign_df = ldf[["kappa_a", "greedy_match_baseline"]].dropna()
    pred = (sign_df["kappa_a"].values > 0).astype(int)
    actual = sign_df["greedy_match_baseline"].astype(int).values
    sign_acc = float((pred == actual).mean()) if len(actual) else float("nan")
    tp = int(((pred == 1) & (actual == 1)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    majority = (
        max(actual.sum(), len(actual) - actual.sum()) / len(actual)
        if len(actual) else float("nan")
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade quadrants: top-right and bottom-left are "sign(κ_a) agrees with
    # greedy_match (using P_base>0.5 as the continuous proxy for match=1)".
    xlim = (float(min(valid["kappa_a"].min(), -1.2)),
            float(max(valid["kappa_a"].max(), 1.2)))
    ax.axvspan(0, xlim[1], ymin=0.5, ymax=1.0, color="#2ca02c", alpha=0.06, zorder=0)
    ax.axvspan(xlim[0], 0, ymin=0.0, ymax=0.5, color="#d62728", alpha=0.06, zorder=0)

    match_col = np.where(ldf["greedy_match_baseline"] > 0.5, "#2ca02c", "#d62728")
    ax.scatter(ldf["kappa_a"], ldf["baseline_p_match"], s=60, c=match_col,
               edgecolors="k", linewidths=0.5, zorder=3, alpha=0.85)
    z = np.polyfit(valid["kappa_a"], valid["baseline_p_match"], 1)
    xr = np.linspace(valid["kappa_a"].min(), valid["kappa_a"].max(), 50)
    ax.plot(xr, np.polyval(z, xr), "--", color="black", alpha=0.5, linewidth=1.5)

    ax.axvline(x=-1, color="#d62728", linestyle=":", alpha=0.4, label="κ=−1 (μ⁻)")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.6,
               label="κ=0 decision boundary")
    ax.axvline(x=+1, color="#2ca02c", linestyle=":", alpha=0.4, label="κ=+1 (μ⁺)")
    ax.axhline(y=0.5, color="gray", linestyle="-", alpha=0.25)
    ax.set_xlim(xlim)

    ax.set_xlabel("κ_a  (projection onto diff-in-means line)", fontsize=10)
    ax.set_ylabel("Baseline P(matching)", fontsize=10)
    ax.set_title(
        f"{behavior} — Layer {layer}: κ_a vs baseline behavior score\n"
        f"Continuous: Spearman ρ={rho:+.3f} (p={sp:.3g}), Pearson r={r:+.3f} (p={pr:.3g})\n"
        f"Binary sign(κ_a) vs greedy_match: acc={sign_acc:.2f} "
        f"(majority baseline={majority:.2f}); "
        f"TP={tp} TN={tn} FP={fp} FN={fn}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Cross-factor plots
# ============================================================================
def plot_steered_sign_agreement_for_factor(
    prompt_df_factor: pd.DataFrame, factor: float,
    behavior: str, output_path: Path,
):
    """
    Per-factor version of projection_sign_agreement_by_layer.

    For a single steering factor α, per layer:
      - κ_a_steered = kappa_a + 2α   (exact: steering shifts projection by 2α)
      - pred  = sign(κ_a_steered) > 0
      - actual = greedy_match_steered   (did the STEERED model pick matching?)

    This is different for every factor folder because:
      - At α=0: same as baseline (κ_a_steered = κ_a)
      - At α=3: points that were near the centroid get pushed past it; if their
        steered score also flips to 1, agreement goes up. If the score doesn't
        flip, they become disagreements.

    Both agreement and MCC are plotted alongside the unsteered baseline for
    comparison.
    """
    if prompt_df_factor.empty or "kappa_a" not in prompt_df_factor.columns:
        return
    layers = sorted(prompt_df_factor["layer"].unique())

    agreements, mccs, majorities = [], [], []
    baseline_agreements, baseline_mccs = [], []

    for l in layers:
        sub = prompt_df_factor[prompt_df_factor["layer"] == l][
            ["kappa_a", "greedy_match_steered", "greedy_match_baseline"]
        ].dropna()
        if len(sub) < 4:
            agreements.append(float("nan"))
            mccs.append(float("nan"))
            majorities.append(float("nan"))
            baseline_agreements.append(float("nan"))
            baseline_mccs.append(float("nan"))
            continue

        ka_steered = sub["kappa_a"].values + 2.0 * factor
        pred_s = (ka_steered > 0).astype(int)
        actual_s = sub["greedy_match_steered"].astype(int).values
        agreements.append(float((pred_s == actual_s).mean()))
        n_pos = int(actual_s.sum())
        majorities.append(float(max(n_pos, len(actual_s) - n_pos) / len(actual_s)))
        tp = int(((pred_s == 1) & (actual_s == 1)).sum())
        tn = int(((pred_s == 0) & (actual_s == 0)).sum())
        fp = int(((pred_s == 1) & (actual_s == 0)).sum())
        fn = int(((pred_s == 0) & (actual_s == 1)).sum())
        denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mccs.append(float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0)

        # Baseline comparison (α=0: κ_a vs greedy_match_baseline).
        pred_b = (sub["kappa_a"].values > 0).astype(int)
        actual_b = sub["greedy_match_baseline"].astype(int).values
        baseline_agreements.append(float((pred_b == actual_b).mean()))
        tp = int(((pred_b == 1) & (actual_b == 1)).sum())
        tn = int(((pred_b == 0) & (actual_b == 0)).sum())
        fp = int(((pred_b == 1) & (actual_b == 0)).sum())
        fn = int(((pred_b == 0) & (actual_b == 1)).sum())
        denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        baseline_mccs.append(float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    layers_arr = np.array(layers)
    # Steered (this factor).
    ax1.plot(layers_arr, agreements, "o-", color="#2E86AB", linewidth=2.5,
             markersize=7, label=f"Agreement: sign(κ_a+2α) = greedy_match_steered  α={factor:g}")
    ax1.plot(layers_arr, majorities, "D--", color="gray", linewidth=1.5,
             markersize=5, alpha=0.6, label="Majority-class baseline (steered)")
    ax1.axhline(0.5, color="red", linestyle=":", alpha=0.4, label="Chance (0.5)")
    # Baseline overlay for comparison.
    ax1.plot(layers_arr, baseline_agreements, "o:", color="#2E86AB", linewidth=1.2,
             markersize=4, alpha=0.4, label="Agreement: baseline (α=0, for ref)")

    ax2.plot(layers_arr, mccs, "s-", color="#E94F37", linewidth=2,
             markersize=6, label=f"MCC steered α={factor:g}")
    ax2.plot(layers_arr, baseline_mccs, "s:", color="#E94F37", linewidth=1.2,
             markersize=4, alpha=0.4, label="MCC baseline (α=0, for ref)")
    ax2.axhline(0, color="#E94F37", linestyle=":", alpha=0.3)
    ax2.set_ylabel("Matthews correlation", color="#E94F37", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#E94F37")
    ax2.set_ylim(-1.05, 1.05)

    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Binary sign agreement", color="#2E86AB", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2E86AB")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(layers_arr)
    ax1.set_title(
        f"{behavior} (α={factor:g}): sign(κ_a + 2α) vs steered greedy choice\n"
        f"Solid = steered  ·  Dotted = unsteered baseline (for comparison)",
        fontsize=11, fontweight="bold",
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_steered_sign_mcc_by_layer(per_prompt_df: pd.DataFrame, behavior: str,
                                    output_path: Path):
    """
    For each steering factor α (including α=0 = baseline), compute:
        κ_a_steered(j, l, α) = κ_a_baseline(j, l) + 2α
    (exact: steering adds α·v to the activation, which shifts the projection
    along v by exactly 2α in κ_a units).
    Then compute MCC(sign(κ_a_steered), greedy_match_steered) per (layer, factor).

    This answers: "does the side of the diff-in-means line that the STEERED
    activation lands on predict the steered model's greedy choice, and does
    this relationship hold / improve as α increases?"
    """
    if per_prompt_df.empty or "kappa_a" not in per_prompt_df.columns:
        return
    factors = sorted(per_prompt_df["steering_factor"].unique())
    layers = sorted(per_prompt_df["layer"].unique())

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("plasma")

    for i, f in enumerate(factors):
        color = cmap(i / max(1, len(factors) - 1))
        mccs = []
        for l in layers:
            sub = per_prompt_df[
                (per_prompt_df["steering_factor"] == f) &
                (per_prompt_df["layer"] == l)
            ][["kappa_a", "greedy_match_steered"]].dropna()
            if len(sub) < 4:
                mccs.append(float("nan"))
                continue
            kappa_steered = sub["kappa_a"].values + 2.0 * f
            pred = (kappa_steered > 0).astype(int)
            actual = sub["greedy_match_steered"].astype(int).values
            tp = int(((pred == 1) & (actual == 1)).sum())
            tn = int(((pred == 0) & (actual == 0)).sum())
            fp = int(((pred == 1) & (actual == 0)).sum())
            fn = int(((pred == 0) & (actual == 1)).sum())
            denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
            mccs.append(mcc)

        # α=0 is the baseline — draw it thicker for reference.
        lw = 3.0 if f == 0 else 1.8
        ls = "-" if f == 0 else "--"
        ax.plot(layers, mccs, marker="o", linestyle=ls, linewidth=lw,
                markersize=5, color=color, label=f"α={f:g}")

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("MCC  [sign(κ_a_steered) vs greedy_match_steered]", fontsize=10)
    ax.set_title(
        f"{behavior}: MCC of sign(κ_a + 2α) predicting steered greedy choice, by layer\n"
        f"κ_a_steered = κ_a_baseline + 2α  (exact shift from steering)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(title="Steering factor α", fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sign_mcc_vs_steered_acc(layer_df_all: pd.DataFrame, behavior: str,
                                  output_path: Path):
    """
    The key cross-factor question:
    "At each layer, if sign(κ_a) already predicted baseline behavior (high MCC),
    does steering at that layer also work well — and does this hold across factors?"

    Left axis:  sign(κ_a) MCC vs greedy_match_baseline  (factor-independent,
                drawn once as a thick black line)
    Right axis: steered greedy accuracy, one colored line per steering factor.

    If the two sets of curves peak at the same layers, the diff-in-means direction
    at those layers both encodes and steers behavior.
    """
    if layer_df_all.empty:
        return
    factors = sorted(layer_df_all["steering_factor"].unique())
    if not factors:
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    # sign(κ_a) MCC — factor-independent; use any factor's rows (same values).
    ref = layer_df_all[layer_df_all["steering_factor"] == factors[0]].sort_values("layer")
    if "proj_sign_mcc" in ref.columns:
        ax1.plot(ref["layer"].values, ref["proj_sign_mcc"].values,
                 "k-", linewidth=3, marker="s", markersize=6,
                 label="sign(κ_a) MCC (baseline, left axis)", zorder=5)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("sign(κ_a) MCC  [baseline, unsteered]", fontsize=11, color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Steered greedy accuracy per factor.
    cmap = plt.get_cmap("plasma")
    for i, f in enumerate(factors):
        sub = layer_df_all[layer_df_all["steering_factor"] == f].sort_values("layer")
        if "steered_greedy_acc" not in sub.columns:
            continue
        color = cmap(i / max(1, len(factors) - 1))
        ax2.plot(sub["layer"].values, sub["steered_greedy_acc"].values,
                 "o--", color=color, linewidth=1.8, markersize=4,
                 label=f"steered greedy acc α={f:g} (right axis)", alpha=0.85)

    ax2.set_ylabel("Steered greedy accuracy", fontsize=11, color="#8B0000")
    ax2.tick_params(axis="y", labelcolor="#8B0000")

    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_title(
        f"{behavior}: sign(κ_a) MCC [left] vs steered greedy accuracy [right] by layer\n"
        f"Do the same layers where projection is informative also respond best to steering?",
        fontsize=11, fontweight="bold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_factor_sweep(layer_df_all: pd.DataFrame, metric: str, metric_label: str,
                      behavior: str, output_path: Path):
    """One line per steering factor, y = metric by layer."""
    if layer_df_all.empty or metric not in layer_df_all.columns:
        return
    factors = sorted(layer_df_all["steering_factor"].unique())
    if not factors:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(factors):
        sub = layer_df_all[layer_df_all["steering_factor"] == f].sort_values("layer")
        color = cmap(i / max(1, len(factors) - 1)) if len(factors) > 1 else cmap(0.5)
        ax.plot(sub["layer"].values, sub[metric].values, "o-",
                color=color, linewidth=2, markersize=5, label=f"α={f:g}")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"{behavior}: {metric_label} by layer, across α",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Steering factor", fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MCQA test-time steering + projection analysis "
                    "(multi-factor, batched).",
    )
    parser.add_argument("--behavior", type=str, required=True,
                        choices=list(TEST_PATH_MAP.keys()))
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Output of mcqa_train_diffmean_sweep.py")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Default: datasets/test/<behavior>/test_dataset_ab.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: <sweep_dir>/mcqa_analysis")
    parser.add_argument(
        "--steering-factors", "--steering_factors", "--factors",
        dest="steering_factors",
        type=str,
        default="1",
        metavar="STR",
        help="Comma-separated list of unnormalized steering factors α "
             "(e.g. 1,2,3,5,10). Pass as one string; default 1.",
    )
    parser.add_argument(
        "--steering_factor", type=float, default=None,
        help="Single α only; equivalent to --steering-factors STR with one value. "
             "If set, overrides --steering-factors.",
    )
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for baseline and steered forward passes.")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Limit number of test prompts (default: use all).")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Ignore any existing per_prompt_results.csv in out_dir and redo "
             "all baseline + steered forward passes.",
    )
    args = parser.parse_args()

    if args.steering_factor is not None:
        factors = [float(args.steering_factor)]
    else:
        factors = sorted(
            {float(x.strip()) for x in str(args.steering_factors).split(",") if x.strip()}
        )
    if not factors:
        logger.error("No steering factors (use --steering-factors STR or --steering_factor α)")
        sys.exit(1)
    # Always include α=0 (baseline) so all sweep plots have an unsteered reference.
    if 0.0 not in factors:
        factors = sorted([0.0] + factors)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sweep_dir = Path(args.sweep_dir)
    layers = discover_layers(sweep_dir)
    if not layers:
        logger.error(f"No trained layers in {sweep_dir}")
        sys.exit(1)
    logger.warning(f"Discovered layers: {layers}")

    out_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "mcqa_analysis"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    test_path = Path(args.test_path or TEST_PATH_MAP[args.behavior])
    logger.warning(f"Loading test set: {test_path}")
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

    # ----------------------------------------------------------------------
    # Reuse previous results if per_prompt_results.csv already covers every
    # requested (factor, layer) pair. No separate cache files needed -- that
    # CSV already contains log-probs, greedy letters, kappa_a, etc.
    # ----------------------------------------------------------------------
    per_prompt_csv = out_dir / "per_prompt_results.csv"
    cached_df = None
    if (not args.force_recompute) and per_prompt_csv.exists():
        try:
            df = pd.read_csv(per_prompt_csv)
            have = set(
                (float(f), int(l))
                for f, l in zip(df["steering_factor"], df["layer"])
            )
            want = set((float(f), int(l)) for f in factors for l in layers)
            missing = want - have

            # Special case: if the only missing pairs are factor 0, synthesize
            # them from the baseline columns already in the CSV (no GPU needed).
            missing_nonzero = {(f, l) for f, l in missing if f != 0.0}
            if missing and not missing_nonzero:
                logger.warning(
                    f"factor=0 rows missing from {per_prompt_csv.name}; "
                    f"synthesizing from baseline columns (no GPU needed)."
                )
                base_rows = df.drop_duplicates(
                    subset=["layer", "prompt_idx"]
                ).copy()
                # Pick one representative factor's rows to clone as factor=0.
                rep_factor = df["steering_factor"].iloc[0]
                base_rows = df[df["steering_factor"] == rep_factor].copy()
                base_rows["steering_factor"] = 0.0
                base_rows["greedy_letter_steered"] = base_rows["greedy_letter_baseline"]
                base_rows["steered_logp_match"] = base_rows["baseline_logp_match"]
                base_rows["steered_logp_notmatch"] = base_rows["baseline_logp_notmatch"]
                base_rows["steered_p_match"] = base_rows["baseline_p_match"]
                base_rows["greedy_match_steered"] = base_rows["greedy_match_baseline"]
                df = pd.concat([df, base_rows], ignore_index=True)
                df.to_csv(per_prompt_csv, index=False)
                missing = set()

            if not missing:
                cached_df = df
                logger.warning(
                    f"Reusing existing {per_prompt_csv} "
                    f"(covers all {len(want)} (factor, layer) pairs). "
                    f"Pass --force_recompute to redo forward passes."
                )
            else:
                logger.warning(
                    f"{per_prompt_csv} exists but is missing "
                    f"{len(missing_nonzero)} non-zero (factor, layer) pairs "
                    f"(e.g. {sorted(missing_nonzero)[:3]}); recomputing from scratch."
                )
        except Exception as e:
            logger.warning(f"Could not reuse {per_prompt_csv}: {e}")
    need_model = cached_df is None

    # Steering artifacts: always needed (kappa recompute if need_model; dprime
    # for summaries/plots regardless).
    steering_vecs = {}
    mu_poss, mu_negs, dprimes = {}, {}, {}
    for l in layers:
        layer_dir = sweep_dir / f"layer_{l}"
        w = torch.load(layer_dir / "DiffMean_weight.pt", map_location="cpu")
        if w.dim() == 2:
            w = w.squeeze(0)
        steering_vecs[l] = w.to(device) if need_model else w
        mu_poss[l] = torch.load(layer_dir / "mu_pos.pt", map_location="cpu")
        mu_negs[l] = torch.load(layer_dir / "mu_neg.pt", map_location="cpu")
        with open(layer_dir / "separability.json") as f:
            dprimes[l] = json.load(f).get("dprime")

    model = None
    if need_model:
        logger.warning(f"Loading model {args.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.use_bf16 else None,
            device_map=device,
        )
        model.eval()
    else:
        logger.warning(
            "All requested (factor, layer) pairs already present in "
            f"{per_prompt_csv.name}; skipping model load + forward passes."
        )

    if not need_model:
        # Reuse everything from per_prompt_results.csv; skip tokenization,
        # Phase A, Phase B, kappa recompute, assembly, and generations dumps
        # (the old jsonls on disk remain valid).
        per_prompt_df = cached_df
    else:
        # ------------------------------------------------------------------
        # Pre-tokenize all (prompt, candidate) pairs once.
        # MCQA rows are *not* always (A, B): e.g. some survival-instinct rows
        # are (A, C) or (A, E). Candidates come from this row's actual
        # answer_matching_behavior / answer_not_matching_behavior.
        # ------------------------------------------------------------------
        prompt_meta = []
        pre_tok = []
        dropped = 0
        for j, item in enumerate(test_data):
            q = item["question"]
            matching_letter = extract_matching_letter(item["answer_matching_behavior"])
            notmatch_letter = extract_matching_letter(item["answer_not_matching_behavior"])
            if matching_letter == notmatch_letter:
                logger.warning(
                    f"Prompt {j}: matching and not-matching letters both "
                    f"'{matching_letter}'; skipping."
                )
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
            logger.warning(
                f"Dropped {dropped} malformed prompt(s); "
                f"continuing with {len(test_data) - dropped} prompts."
            )

        # Flatten into a single list of "items" we batch: (prompt_idx, letter, role).
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

        B = args.batch_size
        n_items = len(flat_items)

        # ----------------------------------------------------------------------
        # Phase A - batched baseline forward + per-layer activation capture.
        # Activations are stored per (prompt_idx, letter) at the letter token position.
        # ----------------------------------------------------------------------
        logger.warning(
            f"\n=== Phase A: batched baseline forward (B={B}, {n_items} items) ==="
        )
        pad_id = tokenizer.pad_token_id
        baseline_logps = {}  # (j, role) -> float, role ∈ {"match","notmatch"}
        act_at_letter = {}   # (j, role, layer) -> tensor [hidden] on cpu
        for start in tqdm(range(0, n_items, B), desc="Baseline batches"):
            batch = flat_items[start:start + B]
            token_lists = [b["full_ids"] for b in batch]
            input_ids, attn, lens = pad_batch(token_lists, pad_id, device)
            logits, layer_hiddens = forward_capture_batch(model, input_ids, attn, layers)
            prompt_lens = [b["prompt_len"] for b in batch]
            logps = batched_answer_logprobs(logits, input_ids, prompt_lens, lens)
            for i, b in enumerate(batch):
                baseline_logps[(b["prompt_idx"], b["role"])] = logps[i]
                for l in layers:
                    act_at_letter[(b["prompt_idx"], b["role"], l)] = (
                        layer_hiddens[l][i, b["letter_pos"], :].float().cpu()
                    )
            del logits, layer_hiddens, input_ids, attn

        # Build baseline_records from the log-probs; pick greedy-letter activation
        # per layer.
        baseline_records = []
        for j, item in enumerate(test_data):
            if prompt_meta[j] is None:
                continue
            matching_letter = prompt_meta[j]["matching_letter"]
            notmatch_letter = prompt_meta[j]["notmatch_letter"]
            logp_m = baseline_logps[(j, "match")]
            logp_n = baseline_logps[(j, "notmatch")]
            p_match = float(np.exp(logp_m) / (np.exp(logp_m) + np.exp(logp_n)))
            is_match_greedy = logp_m > logp_n
            greedy_letter = matching_letter if is_match_greedy else notmatch_letter
            greedy_role = "match" if is_match_greedy else "notmatch"
            greedy_match = int(is_match_greedy)
            baseline_records.append({
                "prompt_idx": j,
                "question": item["question"],
                "answer_matching_behavior": item["answer_matching_behavior"],
                "answer_not_matching_behavior": item.get("answer_not_matching_behavior", ""),
                "matching_letter": matching_letter,
                "notmatch_letter": notmatch_letter,
                "greedy_letter": greedy_letter,
                "baseline_p_match": p_match,
                "greedy_match_baseline": greedy_match,
                "baseline_logp_match": float(logp_m),
                "baseline_logp_notmatch": float(logp_n),
                "activations": {
                    l: act_at_letter[(j, greedy_role, l)] for l in layers
                },
            })
        del act_at_letter


        # ----------------------------------------------------------------------
        # Phase B - batched steered forward per factor per layer (cached per pair).
        # ----------------------------------------------------------------------
        # steered_map[factor][(layer, prompt_idx)] = dict(logp_match, logp_notmatch,
        #                                                  p_match_steered, ...).
        # We first populate from disk-cached (factor, layer) files, then compute
        # any missing pairs and persist each one immediately.
        steered_map = {f: {} for f in factors}
        pad_id = tokenizer.pad_token_id

        # Factor 0 = no steering: steered scores == baseline scores. No GPU work.
        if 0.0 in factors:
            for l in layers:
                for rec in baseline_records:
                    j = rec["prompt_idx"]
                    steered_map[0.0][(l, j)] = {
                        "p_match_steered": rec["baseline_p_match"],
                        "greedy_match_steered": rec["greedy_match_baseline"],
                        "greedy_letter_steered": rec["greedy_letter"],
                        "steered_logp_match": rec["baseline_logp_match"],
                        "steered_logp_notmatch": rec["baseline_logp_notmatch"],
                    }

        nonzero_factors = [f for f in factors if f != 0.0]
        total_fwds = len(nonzero_factors) * len(layers)
        logger.warning(
            f"\n=== Phase B: batched steered scoring (factors={nonzero_factors}, "
            f"{total_fwds} layer*factor combos, B={B}) ==="
        )
        outer = tqdm(total=total_fwds, desc="Steered fwds (factor*layer)")
        for factor in nonzero_factors:
            for l in layers:
                sv = steering_vecs[l]
                per_item_logp = [None] * n_items
                for start in range(0, n_items, B):
                    batch = flat_items[start:start + B]
                    token_lists = [b["full_ids"] for b in batch]
                    input_ids, attn, lens = pad_batch(token_lists, pad_id, device)
                    logits = forward_with_steering_batch(
                        model, input_ids, attn, l, sv, factor, prefix_length,
                    )
                    prompt_lens = [b["prompt_len"] for b in batch]
                    logps = batched_answer_logprobs(logits, input_ids, prompt_lens, lens)
                    for i, b in enumerate(batch):
                        per_item_logp[start + i] = (b["prompt_idx"], b["role"], logps[i])
                    del logits, input_ids, attn
                by_prompt = {}
                for pj, role, lp in per_item_logp:
                    by_prompt.setdefault(pj, {})[role] = lp
                for j, item in enumerate(test_data):
                    if prompt_meta[j] is None:
                        continue
                    ml = prompt_meta[j]["matching_letter"]
                    nl = prompt_meta[j]["notmatch_letter"]
                    logp_m = by_prompt[j]["match"]
                    logp_n = by_prompt[j]["notmatch"]
                    p_match_s = float(np.exp(logp_m) / (np.exp(logp_m) + np.exp(logp_n)))
                    is_match_greedy = logp_m > logp_n
                    greedy_letter_s = ml if is_match_greedy else nl
                    steered_map[factor][(l, j)] = {
                        "p_match_steered": p_match_s,
                        "greedy_match_steered": int(is_match_greedy),
                        "greedy_letter_steered": greedy_letter_s,
                        "steered_logp_match": float(logp_m),
                        "steered_logp_notmatch": float(logp_n),
                    }
                outer.update(1)
        outer.close()

        # ----------------------------------------------------------------------
        # Precompute kappa_a per (layer, prompt) -- factor-independent.
        # ----------------------------------------------------------------------
        kappa_map = {}  # (layer, j) -> float
        for l in layers:
            mu_pos, mu_neg = mu_poss[l].float(), mu_negs[l].float()
            v = (mu_pos - mu_neg)
            v_norm_sq = float(v.dot(v))
            mu = 0.5 * (mu_pos + mu_neg)
            for rec in baseline_records:
                j = rec["prompt_idx"]
                act = rec["activations"][l]
                if v_norm_sq < 1e-12:
                    kappa_map[(l, j)] = float("nan")
                else:
                    kappa_map[(l, j)] = float(
                        2.0 * (act - mu).dot(v).item() / v_norm_sq
                    )

        # ----------------------------------------------------------------------
        # Assemble per-prompt × per-layer × per-factor rows.
        # ----------------------------------------------------------------------
        per_prompt_rows = []
        for factor in factors:
            for rec in baseline_records:
                j = rec["prompt_idx"]
                for l in layers:
                    st = steered_map[factor][(l, j)]
                    per_prompt_rows.append({
                        "steering_factor": factor,
                        "layer": l,
                        "prompt_idx": j,
                        "question": rec["question"],
                        "answer_matching_behavior": rec["answer_matching_behavior"],
                        "answer_not_matching_behavior": rec["answer_not_matching_behavior"],
                        "matching_letter": rec["matching_letter"],
                        "notmatch_letter": rec["notmatch_letter"],
                        "greedy_letter_baseline": rec["greedy_letter"],
                        "baseline_logp_match": rec["baseline_logp_match"],
                        "baseline_logp_notmatch": rec["baseline_logp_notmatch"],
                        "baseline_p_match": rec["baseline_p_match"],
                        "greedy_match_baseline": rec["greedy_match_baseline"],
                        "greedy_letter_steered": st["greedy_letter_steered"],
                        "steered_logp_match": st["steered_logp_match"],
                        "steered_logp_notmatch": st["steered_logp_notmatch"],
                        "steered_p_match": st["p_match_steered"],
                        "greedy_match_steered": st["greedy_match_steered"],
                        "kappa_a": kappa_map[(l, j)],
                    })

        per_prompt_df = pd.DataFrame(per_prompt_rows)
        per_prompt_df.to_csv(out_dir / "per_prompt_results.csv", index=False)
        logger.warning(
            f"Wrote {len(per_prompt_df)} per-prompt rows "
            f"({len(factors)} factors × {len(layers)} layers × {len(test_data)} prompts) "
            f"to {out_dir/'per_prompt_results.csv'}"
        )

        # ----------------------------------------------------------------------
        # Generations dumps (human-readable).
        # ----------------------------------------------------------------------
        gens_dir = out_dir / "generations"
        gens_dir.mkdir(exist_ok=True)
        with open(gens_dir / "baseline.jsonl", "w", encoding="utf-8") as f:
            for rec in baseline_records:
                f.write(json.dumps({
                    "prompt_idx": rec["prompt_idx"],
                    "question": rec["question"],
                    "answer_matching_behavior": rec["answer_matching_behavior"],
                    "answer_not_matching_behavior": rec["answer_not_matching_behavior"],
                    "matching_letter": rec["matching_letter"],
                    "notmatch_letter": rec["notmatch_letter"],
                    "baseline_greedy_letter": rec["greedy_letter"],
                    "baseline_logp_match": rec["baseline_logp_match"],
                    "baseline_logp_notmatch": rec["baseline_logp_notmatch"],
                    "baseline_p_match": rec["baseline_p_match"],
                    "greedy_match_baseline": rec["greedy_match_baseline"],
                }, ensure_ascii=False) + "\n")
        for factor in factors:
            f_tag = f"{factor:g}"
            for l in layers:
                out_path = gens_dir / f"steered_layer_{l}_factor_{f_tag}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for rec in baseline_records:
                        j = rec["prompt_idx"]
                        st = steered_map[factor][(l, j)]
                        f.write(json.dumps({
                            "prompt_idx": j,
                            "layer": l,
                            "steering_factor": factor,
                            "question": rec["question"],
                            "matching_letter": rec["matching_letter"],
                            "notmatch_letter": rec["notmatch_letter"],
                            "baseline_greedy_letter": rec["greedy_letter"],
                            "steered_greedy_letter": st["greedy_letter_steered"],
                            "flipped": bool(rec["greedy_letter"] != st["greedy_letter_steered"]),
                            "baseline_p_match": rec["baseline_p_match"],
                            "steered_p_match": st["p_match_steered"],
                            "steered_logp_match": st["steered_logp_match"],
                            "steered_logp_notmatch": st["steered_logp_notmatch"],
                            "greedy_match_baseline": rec["greedy_match_baseline"],
                            "greedy_match_steered": st["greedy_match_steered"],
                        }, ensure_ascii=False) + "\n")
        logger.warning(
            f"Wrote generations to {gens_dir}/ "
            f"(baseline.jsonl + {len(factors) * len(layers)} steered_layer_*_factor_*.jsonl)"
        )

    # ----------------------------------------------------------------------
    # Per-layer summaries, per factor.
    # ----------------------------------------------------------------------
    all_layer_rows = []
    for factor in factors:
        fdf = per_prompt_df[per_prompt_df["steering_factor"] == factor]
        for l in layers:
            ldf = fdf[fdf["layer"] == l]
            if ldf.empty:
                continue
            row = {
                "steering_factor": factor,
                "layer": l,
                "dprime": dprimes.get(l),
                "baseline_acc_mean": float(ldf["baseline_p_match"].mean()),
                "baseline_greedy_acc": float(ldf["greedy_match_baseline"].mean()),
                "steered_acc_mean": float(ldf["steered_p_match"].mean()),
                "steered_greedy_acc": float(ldf["greedy_match_steered"].mean()),
                "delta_acc": float(ldf["steered_p_match"].mean() - ldf["baseline_p_match"].mean()),
                "delta_greedy_acc": float(
                    ldf["greedy_match_steered"].mean() - ldf["greedy_match_baseline"].mean()
                ),
                "n_prompts": int(len(ldf)),
            }
            valid = ldf[["kappa_a", "baseline_p_match"]].dropna()
            if len(valid) >= 4:
                rho, sp = scipy_stats.spearmanr(valid["kappa_a"], valid["baseline_p_match"])
                r, pr = scipy_stats.pearsonr(valid["kappa_a"], valid["baseline_p_match"])
                row.update({
                    "proj_spearman_rho": float(rho), "proj_spearman_p": float(sp),
                    "proj_pearson_r": float(r), "proj_pearson_p": float(pr),
                })
            else:
                row.update({
                    "proj_spearman_rho": np.nan, "proj_spearman_p": np.nan,
                    "proj_pearson_r": np.nan, "proj_pearson_p": np.nan,
                })

            # Sign classification: does sign(κ_a) predict greedy_match_baseline?
            # Encoding: pred = 1 iff κ_a > 0 (positive side of centroid ⇒ "should
            # show behavior"); actual = greedy_match_baseline (1 if unsteered model
            # greedily picked matching answer).
            sign_df = ldf[["kappa_a", "greedy_match_baseline"]].dropna()
            if len(sign_df) >= 4:
                pred = (sign_df["kappa_a"].values > 0).astype(int)
                actual = sign_df["greedy_match_baseline"].astype(int).values
                n_pos_pred = int(pred.sum())
                n_pos_actual = int(actual.sum())
                acc = float((pred == actual).mean())
                majority = float(max(n_pos_actual, len(actual) - n_pos_actual) / len(actual))
                # Matthews correlation coefficient (binary); robust to class imbalance.
                # TP,FP,FN,TN
                tp = int(((pred == 1) & (actual == 1)).sum())
                tn = int(((pred == 0) & (actual == 0)).sum())
                fp = int(((pred == 1) & (actual == 0)).sum())
                fn = int(((pred == 0) & (actual == 1)).sum())
                denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                mcc = float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
                # Precision/recall for the positive class.
                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
                recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
                row.update({
                    "proj_sign_agreement": acc,
                    "proj_sign_majority_baseline": majority,
                    "proj_sign_mcc": mcc,
                    "proj_sign_precision_pos": precision,
                    "proj_sign_recall_pos": recall,
                    "proj_sign_n_pred_pos": n_pos_pred,
                    "proj_sign_n_actual_pos": n_pos_actual,
                    "proj_sign_tp": tp, "proj_sign_tn": tn,
                    "proj_sign_fp": fp, "proj_sign_fn": fn,
                })
            else:
                row.update({
                    "proj_sign_agreement": np.nan,
                    "proj_sign_majority_baseline": np.nan,
                    "proj_sign_mcc": np.nan,
                    "proj_sign_precision_pos": np.nan,
                    "proj_sign_recall_pos": np.nan,
                    "proj_sign_n_pred_pos": 0, "proj_sign_n_actual_pos": 0,
                    "proj_sign_tp": 0, "proj_sign_tn": 0,
                    "proj_sign_fp": 0, "proj_sign_fn": 0,
                })
            all_layer_rows.append(row)

    layer_df_all = pd.DataFrame(all_layer_rows).sort_values(
        ["steering_factor", "layer"]
    ).reset_index(drop=True)
    layer_df_all.to_csv(out_dir / "per_layer_summary.csv", index=False)

    for factor in factors:
        logger.warning(f"\nPer-layer summary — α={factor:g}:")
        sub = layer_df_all[layer_df_all["steering_factor"] == factor]
        for _, r in sub.iterrows():
            logger.warning(
                f"  L{int(r['layer']):2d}: d'={r['dprime']:.3f}  "
                f"base_acc={r['baseline_acc_mean']:.3f} "
                f"(greedy={r['baseline_greedy_acc']:.2f})  "
                f"steered_acc={r['steered_acc_mean']:.3f} "
                f"(greedy={r['steered_greedy_acc']:.2f})  "
                f"Δ={r['delta_acc']:+.3f}  "
                f"proj_ρ={r['proj_spearman_rho']:+.3f} (p={r['proj_spearman_p']:.2g})  "
                f"sign_acc={r['proj_sign_agreement']:.2f} "
                f"(maj={r['proj_sign_majority_baseline']:.2f}) "
                f"mcc={r['proj_sign_mcc']:+.3f}"
            )

    # ----------------------------------------------------------------------
    # Cross-layer correlations per factor.
    # ----------------------------------------------------------------------
    cross_rows = []
    targets = [
        "baseline_acc_mean", "steered_acc_mean", "delta_acc",
        "baseline_greedy_acc", "steered_greedy_acc", "delta_greedy_acc",
        "proj_spearman_rho", "proj_pearson_r",
        "proj_sign_agreement", "proj_sign_mcc",
    ]
    predictors = [
        ("dprime", "d'"),
        ("proj_spearman_rho", "κ_a→P_base ρ"),
        ("proj_pearson_r", "κ_a→P_base r"),
        ("proj_sign_agreement", "sign(κ_a) acc"),
        ("proj_sign_mcc", "sign(κ_a) MCC"),
    ]
    for factor in factors:
        sub = layer_df_all[layer_df_all["steering_factor"] == factor]
        for pred_col, pred_label in predictors:
            if pred_col not in sub.columns:
                continue
            pv = sub[pred_col].values
            for tgt in targets:
                if tgt not in sub.columns or tgt == pred_col:
                    continue
                tv = sub[tgt].values
                mask = ~(pd.isna(pv) | pd.isna(tv))
                if mask.sum() < 3:
                    continue
                r, pr = scipy_stats.pearsonr(pv[mask], tv[mask])
                rho, sp = scipy_stats.spearmanr(pv[mask], tv[mask])
                cross_rows.append({
                    "steering_factor": factor,
                    "predictor": pred_col, "predictor_label": pred_label, "target": tgt,
                    "n": int(mask.sum()),
                    "pearson_r": float(r), "pearson_p": float(pr),
                    "spearman_rho": float(rho), "spearman_p": float(sp),
                })
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(out_dir / "cross_layer_correlation.csv", index=False)
    for factor in factors:
        logger.warning(f"\nCross-layer correlations (α={factor:g}, N = #layers):")
        sub = cross_df[cross_df["steering_factor"] == factor]
        for _, r in sub.iterrows():
            sig = "*" if r["pearson_p"] < 0.05 else " "
            logger.warning(
                f"  {r['predictor_label']:14s} vs {r['target']:25s} (n={int(r['n'])}): "
                f"r={r['pearson_r']:+.3f} (p={r['pearson_p']:.3g}){sig}  "
                f"ρ={r['spearman_rho']:+.3f} (p={r['spearman_p']:.3g})"
            )

    # ----------------------------------------------------------------------
    # Plots — per-factor subfolders + cross-factor sweep plots.
    # ----------------------------------------------------------------------
    plots_dir = out_dir / "plots"
    for factor in factors:
        f_tag = f"{factor:g}"
        fdir = plots_dir / f"factor_{f_tag}"
        fdir.mkdir(parents=True, exist_ok=True)
        sub_layer = layer_df_all[layer_df_all["steering_factor"] == factor].copy()
        sub_prompt = per_prompt_df[per_prompt_df["steering_factor"] == factor].copy()
        # Inner plots now auto-detect α from the filtered frame; no need to bake it
        # into the behavior string.
        plot_dprime_vs_steering(
            sub_layer, args.behavior,
            fdir / "dprime_vs_steering_by_layer.png",
        )
        plot_kappa_rho_vs_steering(
            sub_layer, args.behavior,
            fdir / "kappa_rho_vs_steering_by_layer.png",
        )
        plot_combined_predictors(
            sub_layer, args.behavior,
            fdir / "combined_predictors_vs_steering.png",
        )
        plot_projection_rho_by_layer(
            sub_layer, args.behavior + f" (α={f_tag})",
            fdir / "projection_rho_by_layer.png",
        )
        plot_sign_agreement_by_layer(
            sub_layer, args.behavior + f" (α={f_tag})",
            fdir / "projection_sign_agreement_by_layer_baseline.png",
        )
        plot_steered_sign_agreement_for_factor(
            sub_prompt, factor, args.behavior,
            fdir / "projection_sign_agreement_by_layer_steered.png",
        )
        for l in layers:
            plot_scatter_for_layer(
                sub_prompt, l, args.behavior + f" (α={f_tag})",
                fdir / f"projection_scatter_layer_{l}.png",
            )

    # Cross-factor sweep plots (one line per factor).
    plot_steered_sign_mcc_by_layer(
        per_prompt_df, args.behavior,
        plots_dir / "steered_sign_mcc_by_layer.png",
    )
    plot_sign_mcc_vs_steered_acc(
        layer_df_all, args.behavior,
        plots_dir / "sign_mcc_vs_steered_acc.png",
    )
    plot_factor_sweep(layer_df_all, "steered_acc_mean",
                      "Steered P(matching)", args.behavior,
                      plots_dir / "factor_sweep_steered_acc.png")
    plot_factor_sweep(layer_df_all, "delta_acc",
                      "Δ P(matching) (steered − baseline)", args.behavior,
                      plots_dir / "factor_sweep_delta_acc.png")
    plot_factor_sweep(layer_df_all, "steered_greedy_acc",
                      "Steered greedy accuracy", args.behavior,
                      plots_dir / "factor_sweep_steered_greedy_acc.png")

    # ----------------------------------------------------------------------
    # Summary JSON.
    # ----------------------------------------------------------------------
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "sweep_dir": str(sweep_dir),
        "steering_factors": factors,
        "batch_size": args.batch_size,
        "n_layers": len(layers),
        "n_test_prompts": len(test_data),
        "layers": layers,
        "per_layer": layer_df_all.to_dict(orient="records"),
        "cross_layer_correlation": cross_df.to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
