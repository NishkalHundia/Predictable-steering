"""
Test-Time Projection Analysis using logistic regression probe direction.

Same analysis as projection_vs_steering_factor.py, but projects onto the
logistic regression probe direction instead of the DiffMean direction.

Probe direction (from thesis Section 4.7):
  - Train a logistic regression classifier on layer-l activations
  - Positive activations a^l(x_i, y_i^+) labeled 1, negative labeled 0
  - The weight vector w_logreg defines the probe direction
  - Projection: proj_logreg(a) = a . w_logreg

Unlike DiffMean (which only uses class means), the probe optimizes a
classification objective across all training activations, so it can find
a direction offering superior class separation — particularly if
distributions are not best separated by their means.

Centroid normalization is the same:
  c+ = mu_pos . w_hat,  c- = mu_neg . w_hat
  p_tilde = (p - m) / h  where m = (c+ + c-)/2, h = (c+ - c-)/2

Requires:
  - Completed sweep (sweep_layers_open_ended.py) with eval results
  - Training contrastive dataset (for probe fitting + centroid computation)

Outputs:
  {sweep_dir}/projection_analysis_probe/
    test_projections.csv
    train_projections.csv
    centroids.csv
    correlations.csv
    analysis_summary.json
    plots/
      scatter_layer_{N}.png
      distribution_layer_{N}.png
      correlation_across_layers.png

Usage:
    uv run python axbench/scripts/projection_tactics/projection_probe.py \\
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
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
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

BEHAVIOR_SCALES = {
    "survival-instinct": (-5, 5, 0),
    "myopic-reward": (-5, 5, 0),
    "corrigible-neutral-HHH": (-5, 5, 0),
    "hallucination": (0, 5, None),
    "sycophancy": (0, 10, 5),
    "refusal": (0, 10, 5),
    "coordinate-other-ais": (0, 10, 5),
}

BEHAVIORS = list(BEHAVIOR_SCALES.keys())


# ============================================================================
# Helpers
# ============================================================================
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def discover_layers(sweep_dir: Path) -> list[int]:
    layers = []
    for d in sweep_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            has_eval = (d / "eval" / "eval_results.parquet").exists()
            if has_eval:
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
# Probe direction computation
# ============================================================================
def compute_probe_directions(train_acts: dict, target_layers: list[int],
                             C: float = 1.0, max_iter: int = 1000):
    """
    Train a logistic regression probe per layer and extract the weight vector
    as the probe direction.

    Args:
        train_acts: {layer: {"pos": [tensor, ...], "neg": [tensor, ...]}}
        target_layers: list of layer indices
        C: inverse regularization strength for LogisticRegression
        max_iter: maximum iterations for solver convergence

    Returns:
        {layer: w_probe_unit_vector} (cpu tensors)
    """
    directions = {}
    for layer in target_layers:
        pos_acts = torch.stack(train_acts[layer]["pos"]).numpy()  # [n_pos, hidden]
        neg_acts = torch.stack(train_acts[layer]["neg"]).numpy()  # [n_neg, hidden]

        X = np.concatenate([pos_acts, neg_acts], axis=0)
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

        clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        clf.fit(X, y)
        train_acc = clf.score(X, y)

        w_probe = torch.from_numpy(clf.coef_[0]).float()

        # Compute cosine similarity with DiffMean for reference
        mu_pos = torch.from_numpy(pos_acts.mean(axis=0)).float()
        mu_neg = torch.from_numpy(neg_acts.mean(axis=0)).float()
        diff = mu_pos - mu_neg

        # Normalize to unit vector
        norm = w_probe.norm()
        if norm > 1e-12:
            w_probe = w_probe / norm

        directions[layer] = w_probe
        cos_sim = torch.cosine_similarity(
            w_probe.unsqueeze(0), (diff / diff.norm()).unsqueeze(0)
        ).item()
        logger.warning(
            f"  Layer {layer:2d}: train_acc = {train_acc:.4f}, "
            f"probe norm (pre-normalize) = {norm:.4f}, "
            f"cos(Probe, DiffMean) = {cos_sim:.4f}"
        )

    return directions


# ============================================================================
# Extraction: raw training activations
# ============================================================================
@torch.no_grad()
def extract_train_activations(model, tokenizer, train_dataset_path,
                              target_layers, device):
    """
    Collect full mean-response-activation vectors for each training example.

    Returns:
        train_acts: {layer: {"pos": [tensor, ...], "neg": [tensor, ...]}}
        train_meta: list of dicts with pair_idx, label, label_name
    """
    use_chat = supports_chat_template(tokenizer)
    model.eval()

    with open(train_dataset_path, encoding="utf-8") as f:
        train_data = json.load(f)

    logger.warning(f"Extracting training activations for {len(train_data)} pairs")

    train_acts = {l: {"pos": [], "neg": []} for l in target_layers}
    train_meta = []

    for pair_idx, item in enumerate(tqdm(train_data, desc="Train activations")):
        for label, answer_key in [(1, "answer_matching_behavior"),
                                  (0, "answer_not_matching_behavior")]:
            question = item["question"]
            answer = item[answer_key]

            mean_acts = _get_mean_response_act(
                model, tokenizer, target_layers, question, str(answer),
                use_chat, device,
            )
            if mean_acts is None:
                continue

            key = "pos" if label == 1 else "neg"
            for layer in target_layers:
                train_acts[layer][key].append(mean_acts[layer])

            train_meta.append({
                "pair_idx": pair_idx,
                "label": label,
                "label_name": "pos" if label == 1 else "neg",
            })

        if pair_idx % 5 == 0:
            torch.cuda.empty_cache()

    for layer in target_layers:
        logger.warning(
            f"  Layer {layer}: {len(train_acts[layer]['pos'])} pos, "
            f"{len(train_acts[layer]['neg'])} neg activations"
        )

    return train_acts, train_meta


# ============================================================================
# Projection helpers
# ============================================================================
def project_train_activations(train_acts, train_meta, directions, target_layers):
    """Project raw training activations onto probe directions -> DataFrame."""
    rows = []
    for layer in target_layers:
        w = directions[layer]
        for key, label in [("pos", 1), ("neg", 0)]:
            for i, act in enumerate(train_acts[layer][key]):
                proj = act.dot(w).item()
                rows.append({
                    "pair_idx": i,
                    "label": label,
                    "label_name": key,
                    "layer": int(layer),
                    "projection": proj,
                })
    return pd.DataFrame(rows)


@torch.no_grad()
def extract_test_projections(model, tokenizer, sweep_dir, directions,
                             target_layers, device, return_raw=False):
    """
    For each test prompt, get the unsteered generation (factor=0), forward pass,
    mean response-token activation, project onto probe direction.

    If return_raw=True, also returns:
        test_mean_acts: {(question_idx, layer): tensor} for caching
        test_metadata: list of dicts with question_idx, layer, baseline_score, etc.
    """
    use_chat = supports_chat_template(tokenizer)
    model.eval()
    sweep_dir = Path(sweep_dir)

    first_layer = target_layers[0]
    eval_df = pd.read_parquet(
        sweep_dir / f"layer_{first_layer}" / "eval" / "eval_results.parquet"
    )
    baseline_df = eval_df[eval_df["steering_factor"] == 0].copy().reset_index(drop=True)
    if baseline_df.empty:
        closest = eval_df.loc[eval_df["steering_factor"].abs().idxmin(), "steering_factor"]
        baseline_df = eval_df[eval_df["steering_factor"] == closest].copy().reset_index(drop=True)
        logger.warning(f"No factor=0 found; using factor={closest}")

    baseline_scores = {}
    for layer in target_layers:
        layer_eval = pd.read_parquet(
            sweep_dir / f"layer_{layer}" / "eval" / "eval_results.parquet"
        )
        f0 = layer_eval[layer_eval["steering_factor"] == 0]
        baseline_scores[layer] = dict(zip(f0["question_idx"], f0["behavior_score"]))

    logger.warning(f"Extracting probe projections for {len(baseline_df)} test prompts")

    rows = []
    raw_acts = {}
    raw_meta = []

    for idx in tqdm(range(len(baseline_df)), desc="Test projections (Probe)"):
        row = baseline_df.iloc[idx]
        question = row["question"]
        generation = row["generation"]
        question_idx = row.get("question_idx", idx)

        if pd.isna(generation) or str(generation).strip() == "":
            continue

        mean_acts = _get_mean_response_act(
            model, tokenizer, target_layers, question, str(generation),
            use_chat, device,
        )
        if mean_acts is None:
            continue

        for layer in target_layers:
            act = mean_acts[layer]
            w = directions[layer].to(act.device)
            proj = act.dot(w).item()
            bs = baseline_scores.get(layer, {}).get(question_idx, np.nan)

            rows.append({
                "question_idx": int(question_idx),
                "layer": int(layer),
                "projection": proj,
                "baseline_score": bs,
                "question": question,
                "generation": str(generation),
            })

            if return_raw:
                raw_acts[(int(question_idx), int(layer))] = act.cpu()
                raw_meta.append({
                    "question_idx": int(question_idx),
                    "layer": int(layer),
                    "idx": idx,
                    "baseline_score": bs,
                    "question": question,
                    "generation": str(generation),
                })

        if idx % 5 == 0:
            torch.cuda.empty_cache()

    logger.warning(f"Extracted {len(rows)} test projection values")
    df = pd.DataFrame(rows)
    if return_raw:
        return df, raw_acts, raw_meta
    return df


# ============================================================================
# Centroid normalization
# ============================================================================
def compute_centroids(train_df, target_layers):
    rows = []
    for layer in target_layers:
        ldf = train_df[train_df["layer"] == layer]
        c_pos = ldf[ldf["label"] == 1]["projection"].mean()
        c_neg = ldf[ldf["label"] == 0]["projection"].mean()
        midpoint = (c_pos + c_neg) / 2.0
        half_range = (c_pos - c_neg) / 2.0
        rows.append({
            "layer": layer,
            "c_pos": c_pos,
            "c_neg": c_neg,
            "midpoint": midpoint,
            "half_range": half_range,
        })
    return pd.DataFrame(rows)


def normalize_projections(df, centroids_df, proj_col="projection"):
    centroid_map = {
        row["layer"]: (row["midpoint"], row["half_range"])
        for _, row in centroids_df.iterrows()
    }
    norms = []
    for _, row in df.iterrows():
        m, h = centroid_map.get(row["layer"], (0, 1))
        norms.append((row[proj_col] - m) / h if abs(h) > 1e-12 else 0.0)
    df = df.copy()
    df["normalized_projection"] = norms
    return df


# ============================================================================
# Correlation
# ============================================================================
def compute_correlations(test_df, target_layers):
    rows = []
    for layer in target_layers:
        ldf = test_df[test_df["layer"] == layer]
        valid = ldf[["normalized_projection", "baseline_score"]].dropna()
        if len(valid) < 4:
            continue
        r, p_r = scipy_stats.pearsonr(valid["normalized_projection"], valid["baseline_score"])
        rho, p_rho = scipy_stats.spearmanr(valid["normalized_projection"], valid["baseline_score"])
        rows.append({
            "layer": layer,
            "pearson_r": r, "pearson_p": p_r,
            "spearman_rho": rho, "spearman_p": p_rho,
            "n": len(valid),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================
def plot_scatter(test_df, layer, behavior, output_path):
    ldf = test_df[test_df["layer"] == layer]
    valid = ldf[["normalized_projection", "baseline_score"]].dropna()
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid["normalized_projection"], valid["baseline_score"],
               alpha=0.8, s=60, edgecolors="k", linewidths=0.4, color="#2E86AB",
               zorder=3)

    ax.axvline(x=-1, color="#d62728", linestyle="--", alpha=0.6, label="Neg centroid (-1)")
    ax.axvline(x=+1, color="#2ca02c", linestyle="--", alpha=0.6, label="Pos centroid (+1)")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.4, label="Midpoint (0)")

    if len(valid) >= 4:
        r, p_r = scipy_stats.pearsonr(valid["normalized_projection"], valid["baseline_score"])
        rho, p_rho = scipy_stats.spearmanr(valid["normalized_projection"], valid["baseline_score"])
        z = np.polyfit(valid["normalized_projection"], valid["baseline_score"], 1)
        xr = np.linspace(valid["normalized_projection"].min(),
                         valid["normalized_projection"].max(), 50)
        ax.plot(xr, np.polyval(z, xr), "--", color="red", alpha=0.7, linewidth=1.5,
                zorder=2)
        txt = f"r = {r:.3f} (p = {p_r:.1e})\nrho = {rho:.3f} (p = {p_rho:.1e})\nn = {len(valid)}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Normalized probe projection  (neg centroid = -1, pos centroid = +1)",
                  fontsize=10)
    ax.set_ylabel("Baseline behavior score (unsteered)", fontsize=10)
    ax.set_title(f"{behavior} - Layer {layer} (Probe)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_distribution(train_df, test_df, layer, behavior, output_path):
    train_l = train_df[train_df["layer"] == layer]
    test_l = test_df[test_df["layer"] == layer]

    fig, ax = plt.subplots(figsize=(10, 5))

    groups, labels, colors = [], [], []
    if not train_l.empty:
        pos = train_l[train_l["label"] == 1]["normalized_projection"].dropna()
        neg = train_l[train_l["label"] == 0]["normalized_projection"].dropna()
        if len(pos):
            groups.append(pos); labels.append(f"Train pos (n={len(pos)})"); colors.append("#2ca02c")
        if len(neg):
            groups.append(neg); labels.append(f"Train neg (n={len(neg)})"); colors.append("#d62728")
    if not test_l.empty:
        test_vals = test_l["normalized_projection"].dropna()
        if len(test_vals):
            groups.append(test_vals); labels.append(f"Test (n={len(test_vals)})"); colors.append("#1f77b4")

    if not groups:
        plt.close()
        return

    all_vals = pd.concat(groups)
    bins = np.linspace(all_vals.min() - 0.5, all_vals.max() + 0.5, 35)
    for vals, lbl, clr in zip(groups, labels, colors):
        ax.hist(vals, bins=bins, alpha=0.45, label=lbl, color=clr,
                edgecolor="white", linewidth=0.5)

    ax.axvline(x=-1, color="#d62728", linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(x=+1, color="#2ca02c", linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    ylim = ax.get_ylim()
    ax.text(-1, ylim[1] * 0.95, " c- = -1", color="#d62728", fontsize=9,
            fontweight="bold", va="top")
    ax.text(+1, ylim[1] * 0.95, " c+ = +1", color="#2ca02c", fontsize=9,
            fontweight="bold", va="top")

    ax.set_xlabel("Normalized probe projection  (neg centroid = -1, pos centroid = +1)",
                  fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"{behavior} - Layer {layer}: Train vs Test (Probe)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_across_layers(corr_df, behavior, output_path):
    if corr_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    layers = corr_df["layer"].values
    rhos = corr_df["spearman_rho"].values
    pvals = corr_df["spearman_p"].values

    sig_mask = pvals < 0.05
    ax.plot(layers, rhos, "o-", color="#2E86AB", linewidth=2, markersize=7,
            zorder=2, label="Spearman rho")
    ax.scatter(layers[sig_mask], rhos[sig_mask], s=120, facecolors="none",
               edgecolors="red", linewidths=2, zorder=3, label="p < 0.05")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Spearman rho (normalized probe proj. vs baseline score)", fontsize=10)
    ax.set_title(f"{behavior}: Probe Projection-Score Correlation Across Layers",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Test-time projection analysis using logistic regression probe direction"
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
    parser.add_argument("--probe_C", type=float, default=1.0,
                        help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--probe_max_iter", type=int, default=1000,
                        help="Max iterations for logistic regression solver")
    parser.add_argument("--replot_only", action="store_true",
                        help="Skip model inference; reuse saved CSVs")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    out_dir = sweep_dir / "projection_analysis_probe"
    plot_dir = out_dir / "plots"
    # Clean plots/CSVs but preserve cached activation .pt files
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    for csv_file in out_dir.glob("*.csv"):
        csv_file.unlink()
    for json_file in out_dir.glob("*.json"):
        json_file.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.layers:
        target_layers = sorted(int(x.strip()) for x in args.layers.split(","))
    else:
        target_layers = discover_layers(sweep_dir)
    if not target_layers:
        logger.error(f"No valid layers found in {sweep_dir}")
        sys.exit(1)
    logger.warning(f"Target layers: {target_layers}")

    # -- Paths for cached CSVs ---------------------------------------------
    test_proj_path = out_dir / "test_projections.csv"
    train_proj_path = out_dir / "train_projections.csv"

    # -- Paths for cached activations (.pt) --------------------------------
    train_acts_path = out_dir / "train_activations.pt"
    test_acts_path = out_dir / "test_activations.pt"

    if args.replot_only:
        for p in [test_proj_path, train_proj_path]:
            if not p.exists():
                logger.error(f"Missing {p}; run without --replot_only first")
                sys.exit(1)
        test_df = pd.read_csv(test_proj_path)
        train_df = pd.read_csv(train_proj_path)
        logger.warning(f"Loaded {len(test_df)} test, {len(train_df)} train projections")
    else:
        model = None
        tokenizer = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _ensure_model_loaded():
            nonlocal model, tokenizer
            if model is not None:
                return
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

        # -- Phase 1: get raw training activations (from cache or model) ---
        if train_acts_path.exists():
            logger.warning(f"Loading cached training activations from {train_acts_path}")
            cached = torch.load(train_acts_path, map_location="cpu", weights_only=False)
            train_acts = cached["train_acts"]
            train_meta = cached["train_meta"]
        else:
            _ensure_model_loaded()
            logger.warning("Phase 1: Collecting raw training activations ...")
            train_acts, train_meta = extract_train_activations(
                model, tokenizer, args.train_dataset_path, target_layers, device,
            )
            logger.warning(f"Saving training activations to {train_acts_path}")
            torch.save({"train_acts": train_acts, "train_meta": train_meta}, train_acts_path)

        # Phase 2: compute probe directions
        logger.warning("Phase 2: Training logistic regression probes ...")
        directions = compute_probe_directions(
            train_acts, target_layers, C=args.probe_C, max_iter=args.probe_max_iter,
        )

        # Phase 3: project training activations onto probe directions
        logger.warning("Phase 3: Projecting training activations ...")
        train_df = project_train_activations(train_acts, train_meta, directions, target_layers)

        # -- Phase 4: get test activations (from cache or model) -----------
        if test_acts_path.exists():
            logger.warning(f"Loading cached test activations from {test_acts_path}")
            cached_test = torch.load(test_acts_path, map_location="cpu", weights_only=False)
            test_mean_acts = cached_test["test_mean_acts"]
            test_metadata = cached_test["test_metadata"]
            # project cached test activations onto probe directions
            test_rows = []
            for entry in test_metadata:
                layer = entry["layer"]
                w = directions[layer]
                proj = test_mean_acts[(entry["question_idx"], layer)].dot(w).item()
                test_rows.append({
                    "question_idx": entry["question_idx"],
                    "layer": layer,
                    "projection": proj,
                    "baseline_score": entry["baseline_score"],
                    "question": entry["question"],
                    "generation": entry["generation"],
                })
            test_df = pd.DataFrame(test_rows)
        else:
            _ensure_model_loaded()
            logger.warning("Phase 4: Extracting test activations ...")
            test_df, test_mean_acts, test_metadata = extract_test_projections(
                model, tokenizer, sweep_dir, directions, target_layers, device,
                return_raw=True,
            )
            logger.warning(f"Saving test activations to {test_acts_path}")
            torch.save({"test_mean_acts": test_mean_acts, "test_metadata": test_metadata}, test_acts_path)

        if model is not None:
            del model
            torch.cuda.empty_cache()

    # -- Compute centroids and normalize --------------------------------
    logger.warning("Computing centroids and normalizing ...")
    centroids_df = compute_centroids(train_df, target_layers)
    centroids_df.to_csv(out_dir / "centroids.csv", index=False)

    for _, c in centroids_df.iterrows():
        logger.warning(
            f"  Layer {int(c['layer']):2d}: c+ = {c['c_pos']:.2f}, "
            f"c- = {c['c_neg']:.2f}, gap = {2*c['half_range']:.2f}"
        )

    test_df = normalize_projections(test_df, centroids_df)
    train_df = normalize_projections(train_df, centroids_df)

    test_df.to_csv(test_proj_path, index=False)
    train_df.to_csv(train_proj_path, index=False)

    # -- Correlations ---------------------------------------------------
    corr_df = compute_correlations(test_df, target_layers)
    corr_df.to_csv(out_dir / "correlations.csv", index=False)

    logger.warning("Correlations (normalized probe projection vs baseline_score):")
    for _, row in corr_df.iterrows():
        sig = " *" if row["spearman_p"] < 0.05 else ""
        logger.warning(
            f"  Layer {int(row['layer']):2d}: "
            f"rho = {row['spearman_rho']:+.3f} (p = {row['spearman_p']:.1e}), "
            f"r = {row['pearson_r']:+.3f} (p = {row['pearson_p']:.1e}), "
            f"n = {int(row['n'])}{sig}"
        )

    # -- Plots ----------------------------------------------------------
    logger.warning("Generating plots ...")
    for layer in target_layers:
        plot_scatter(test_df, layer, args.behavior,
                     plot_dir / f"scatter_layer_{layer}.png")
        plot_distribution(train_df, test_df, layer, args.behavior,
                          plot_dir / f"distribution_layer_{layer}.png")

    plot_correlation_across_layers(corr_df, args.behavior,
                                   plot_dir / "correlation_across_layers.png")

    # -- Summary --------------------------------------------------------
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "direction": "Probe",
        "probe_C": args.probe_C,
        "probe_max_iter": args.probe_max_iter,
        "sweep_dir": str(sweep_dir),
        "train_dataset_path": args.train_dataset_path,
        "layers": target_layers,
        "n_test_prompts": int(test_df.groupby("layer")["question_idx"].nunique().median()),
        "n_train_pairs": int(train_df.groupby("layer")["pair_idx"].nunique().median()) if "pair_idx" in train_df.columns else 0,
        "centroids": centroids_df.to_dict(orient="records"),
        "correlations": corr_df.to_dict(orient="records"),
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()
