"""
Modal app: re-render a focused set of mcqa_projection_link figures from CSVs
already stored in the `steering` Modal volume, **without** re-running the
experiment, and pull the new PNGs back into `paper_plots/<behavior>/`.

Renders only three figures per behavior (paper-ready subset):
    - steering_acc_and_dprime.png
    - mcc_best_alpha_vs_dprime.png
    - projection_hist_postgen_layer_<target_layer>.png   (default layer 21)

Volume layout expected (inside the `steering` volume):
    mcqa_projection_link/<model_short>/<behavior>/per_layer_summary.csv
    ...                                          /per_prompt_results.csv
    ...                                          /train_projections.json
    ...                                          /summary.json

Replot-only overrides (do NOT touch axbench/scripts/mcqa_projection_link.py):
    * MCC dual-axis plot:
        - legend removed
        - x/y axis labels enlarged (fontsize 15)
        - tick label fontsize bumped to 12
        - right-axis label renamed: "d' (training discriminability)" -> "d' (train)"

Run:
    uv run modal run axbench/scripts/replot_mcqa_modal.py --behavior corrigible-neutral-HHH
    uv run modal run axbench/scripts/replot_mcqa_modal.py --all-behaviors
    uv run modal run axbench/scripts/replot_mcqa_modal.py --all-behaviors --target-layer 24
"""

import sys
import types
from pathlib import Path

import modal

APP_NAME = "mcqa-replot"
VOLUME_NAME = "steering"
DEFAULT_MODEL_SHORT = "gemma-2-9b-it"
BEHAVIORS = [
    "sycophancy", "survival-instinct", "corrigible-neutral-HHH",
    "hallucination", "myopic-reward",
]
# Single-word short label per behavior, used when renaming the MCC / steering
# figures locally to "<short>_mcc_new.png" / "<short>_steer_new.png".
BEHAVIOR_SHORT = {
    "sycophancy": "sycophancy",
    "survival-instinct": "survival",
    "corrigible-neutral-HHH": "corrigible",
    "hallucination": "hallucination",
    "myopic-reward": "myopic",
}
# Per-behavior default layer for the post-gen projection histogram. Used
# when --target-layer is not explicitly passed (sentinel value < 0).
DEFAULT_LAYER = 21
BEHAVIOR_DEFAULT_LAYER = {
    "hallucination": 19,
}

SCRIPTS_DIR = Path(__file__).resolve().parent  # axbench/scripts/

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME)

# Lightweight image: only what plot funcs actually need.
# Heavy deps (torch / transformers / sklearn / tqdm / axbench.utils) are stubbed
# at runtime so we can import mcqa_projection_link's plot helpers without
# pulling them in.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "matplotlib==3.9.2",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scipy==1.13.1",
    )
    .add_local_dir(SCRIPTS_DIR.as_posix(), remote_path="/root/scripts")
)


# ---------------------------------------------------------------------------
# Container-side helpers
# ---------------------------------------------------------------------------
def _install_stubs():
    """Stub heavy deps that mcqa_projection_link imports at module load.

    Plot functions only touch matplotlib / numpy / pandas / scipy. Everything
    else (torch, transformers, sklearn, tqdm, axbench.utils.*) is referenced
    only by training / inference paths we never call from the replot script.
    """

    class _NoGradCtx:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __call__(self, fn): return fn  # @torch.no_grad() passthrough

    def _no_grad(*a, **k):
        return _NoGradCtx()

    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = _no_grad
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.is_tensor = lambda x: False
    torch_stub.long = "long"
    torch_stub.bfloat16 = "bfloat16"
    for name in ("zeros", "full", "tensor", "stack", "save", "load",
                 "device", "manual_seed"):
        setattr(torch_stub, name, lambda *a, _n=name, **k: None)
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None,
    )
    sys.modules["torch"] = torch_stub

    tr_stub = types.ModuleType("transformers")
    tr_stub.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
    tr_stub.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules["transformers"] = tr_stub

    sk_stub = types.ModuleType("sklearn")
    sk_metrics = types.ModuleType("sklearn.metrics")
    sk_metrics.matthews_corrcoef = lambda *a, **k: 0.0
    sk_stub.metrics = sk_metrics
    sys.modules["sklearn"] = sk_stub
    sys.modules["sklearn.metrics"] = sk_metrics

    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda x=None, *a, **k: x if x is not None else (lambda y: y)
    sys.modules["tqdm"] = tqdm_stub

    ax_pkg = types.ModuleType("axbench")
    ax_utils = types.ModuleType("axbench.utils")
    ax_constants = types.ModuleType("axbench.utils.constants")
    ax_constants.CHAT_MODELS = []
    ax_model_utils = types.ModuleType("axbench.utils.model_utils")
    ax_model_utils.get_prefix_length = lambda *a, **k: 1
    ax_pkg.utils = ax_utils
    ax_utils.constants = ax_constants
    ax_utils.model_utils = ax_model_utils
    sys.modules.setdefault("axbench", ax_pkg)
    sys.modules["axbench.utils"] = ax_utils
    sys.modules["axbench.utils.constants"] = ax_constants
    sys.modules["axbench.utils.model_utils"] = ax_model_utils


def _patch_mcc_no_legend(m, plt, np):
    """Replace the shared MCC dual-axis helper with a no-legend / big-font /
    'd' (train)' version. Both `plot_mcc_best_alpha_vs_dprime` and
    `plot_mcc_val_best_alpha_on_test_vs_dprime` resolve this name from the
    module namespace at call time, so a single patch covers both wrappers.
    """
    def _mcc_no_legend(layer_df, behavior, out_path, mcc_col, title_line1,
                       mcc_label="MCC(sign κ vs actual match)"):
        if mcc_col not in layer_df.columns:
            return
        mcc = layer_df[mcc_col].values.astype(float)
        if not np.any(np.isfinite(mcc)):
            return
        from matplotlib.ticker import MaxNLocator
        layers = layer_df["layer"].values
        even_layers = [int(l) for l in layers if int(l) % 2 == 0]
        fig, ax1 = plt.subplots(figsize=(13, 5))
        ax1.plot(layers, mcc, "o-", color="#C73E1D", linewidth=2, markersize=6,
                 label=mcc_label, zorder=3)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
        ax1.set_xlabel("Layer", fontsize=16)
        ax1.set_ylabel("MCC", fontsize=16)
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_xticks(even_layers)
        ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax1.tick_params(axis="x", labelsize=14)
        ax1.tick_params(axis="y", labelcolor="#C73E1D", labelsize=14)
        ax2 = ax1.twinx()
        if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
            ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12,
                             color="steelblue")
            ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                     linewidth=1.5, markersize=5, label="d' (train)", zorder=2)
            ax2.set_ylabel("d' (train)", fontsize=16, color="steelblue")
            ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=14)
            ax2.set_ylim(bottom=0)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        # Legend and title intentionally omitted (replot-only override).
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    m.plot_mcc_vs_dprime_for_column = _mcc_no_legend


def _patch_steering_and_dprime(m, plt):
    """Replot-only override: bump axis label font sizes by 2pt (11→13) and drop the title."""
    def _steering_and_dprime(layer_df, factors, behavior, out_path):
        from matplotlib.ticker import MaxNLocator
        layers = layer_df["layer"].values
        even_layers = [int(l) for l in layers if int(l) % 2 == 0]
        fig, ax1 = plt.subplots(figsize=(13, 5))

        ax1.plot(layers, layer_df["baseline_acc"].values, "D--", color="gray",
                 linewidth=1.5, markersize=6, label="Baseline (α=0)", alpha=0.85, zorder=3)
        cmap = plt.get_cmap("plasma")
        factor_cols = [f for f in factors if f"steered_acc_{f:g}" in layer_df.columns]
        for i, f in enumerate(factor_cols):
            color = cmap(i / max(1, len(factor_cols) - 1))
            ax1.plot(layers, layer_df[f"steered_acc_{f:g}"].values, "o-", color=color,
                     linewidth=2, markersize=5, label=f"α={f:g}", zorder=3)
        ax1.set_xlabel("Layer", fontsize=16)
        ax1.set_ylabel("Greedy accuracy", fontsize=16)
        ax1.set_ylim(0, 1.05)
        ax1.set_xticks(even_layers)
        ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax1.tick_params(axis="x", labelsize=14)
        ax1.tick_params(axis="y", labelsize=14)

        ax2 = ax1.twinx()
        if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
            ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
            ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                     linewidth=1.5, markersize=5, label="d'", zorder=2)
            ax2.set_ylabel("d' (train)", fontsize=16, color="steelblue")
            ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=14)
            ax2.set_ylim(bottom=0)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

        # Legend intentionally omitted (replot-only override).
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    m.plot_steering_and_dprime = _steering_and_dprime


def _patch_projection_histograms_no_title(m):
    """Wrap plot_projection_histograms with consistent typography across
    behaviors:
      - figure suptitle is dropped, per-axes legend is dropped
      - "# examples" ylabel rewritten to "# prompts"
      - tick fontsize forced to TICK_FS on every axes
      - axes labels (xlabel/ylabel) forced to TICK_FS + 1
      - subplot titles (Train, α=…) forced to TICK_FS + 1 and bolded
      - savefig drops bbox_inches="tight" so every PNG comes out at
        figsize×dpi (2400×1050) regardless of how many subpanels are
        visible — eliminates the "fontsize looks different across
        behaviors" issue from variable cropping.
    """
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    orig_fn = m.plot_projection_histograms

    TICK_FS = 14
    LABEL_FS = TICK_FS + 1
    TITLE_FS = TICK_FS + 1

    def _wrapped(*args, **kwargs):
        original_suptitle    = Figure.suptitle
        original_set_xlabel  = Axes.set_xlabel
        original_set_ylabel  = Axes.set_ylabel  # used directly in savefig hook
        original_set_title   = Axes.set_title
        original_legend      = Axes.legend
        original_savefig     = Figure.savefig

        def _xlabel_override(self, *a, **kw):
            kw["fontsize"] = LABEL_FS
            return original_set_xlabel(self, *a, **kw)

        def _title_override(self, *a, **kw):
            kw["fontsize"] = TITLE_FS
            kw["fontweight"] = "bold"
            return original_set_title(self, *a, **kw)

        def _savefig_override(self, *a, **kw):
            from matplotlib.ticker import MaxNLocator
            # Force a uniform subplot layout across all behaviors so the saved
            # PNGs come out at identical pixel dimensions (figsize × dpi)
            # regardless of which factor panels were hidden.
            self.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.10,
                                 wspace=0.30, hspace=0.40)
            # `plot_projection_histograms` creates axes via plt.subplots(2, 4)
            # in row-major order, so fig.axes[0] is the train panel.
            for i, ax in enumerate(self.axes):
                ax.tick_params(axis="both", labelsize=TICK_FS)
                # y-axis is a count of prompts -> always integer ticks.
                ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                if i == 0:
                    original_set_ylabel(ax, "# prompts", fontsize=LABEL_FS)
                else:
                    # Drop ylabel on every test panel (original code may have
                    # set "# prompts" on factor_axes[0]).
                    original_set_ylabel(ax, "")
            kw.pop("bbox_inches", None)
            kw.pop("pad_inches", None)
            return original_savefig(self, *a, **kw)

        Figure.suptitle  = lambda self, *a, **kw: None
        Axes.set_xlabel  = _xlabel_override
        Axes.set_title   = _title_override
        Axes.legend      = lambda self, *a, **kw: None
        Figure.savefig   = _savefig_override
        try:
            return orig_fn(*args, **kwargs)
        finally:
            Figure.suptitle  = original_suptitle
            Axes.set_xlabel  = original_set_xlabel
            Axes.set_title   = original_set_title
            Axes.legend      = original_legend
            Figure.savefig   = original_savefig

    m.plot_projection_histograms = _wrapped


def _load_factors(out_dir: Path, layer_df) -> list:
    import json as _json
    sp = out_dir / "summary.json"
    if sp.exists():
        try:
            blob = _json.loads(sp.read_text())
            f_list = blob.get("factors")
            if f_list:
                return sorted(float(x) for x in f_list)
        except Exception:
            pass
    factors = set()
    for c in layer_df.columns:
        if c.startswith("steered_acc_factor_"):
            try:
                factors.add(float(c.replace("steered_acc_factor_", "")))
            except ValueError:
                pass
    return sorted(factors)


def _replot_one(model_short: str, behavior: str, target_layer: int) -> dict:
    """Run inside the Modal container. Returns {relative_png_path: bytes}.

    Only renders the three figures requested:
      - projection_hist_postgen_layer_<target_layer>.png
      - mcc_best_alpha_vs_dprime.png
      - steering_acc_and_dprime.png
    """
    import json as _json
    import shutil
    import tempfile

    _install_stubs()
    sys.path.insert(0, "/root/scripts")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import mcqa_projection_link as m  # noqa: E402
    _patch_mcc_no_legend(m, plt, np)
    _patch_steering_and_dprime(m, plt)
    _patch_projection_histograms_no_title(m)

    src = Path("/vol") / "mcqa_projection_link" / model_short / behavior
    if not src.exists():
        raise FileNotFoundError(f"Behavior dir not found in volume: {src}")

    per_layer_csv = src / "per_layer_summary.csv"
    per_prompt_csv = src / "per_prompt_results.csv"
    if not per_layer_csv.exists() or not per_prompt_csv.exists():
        raise FileNotFoundError(
            f"Missing CSVs under {src}. Have: {sorted(p.name for p in src.iterdir())}"
        )

    per_layer = pd.read_csv(per_layer_csv)
    per_prompt = pd.read_csv(per_prompt_csv)

    train_projections = {}
    tp_path = src / "train_projections.json"
    if tp_path.exists():
        raw = _json.loads(tp_path.read_text())
        train_projections = {
            int(k): {"pos": v["pos"], "neg": v["neg"]} for k, v in raw.items()
        }

    factors = _load_factors(src, per_layer)

    available_layers = {int(x) for x in per_layer["layer"].values}
    if target_layer not in available_layers:
        raise ValueError(
            f"Layer {target_layer} not in per_layer_summary.csv "
            f"(have: {sorted(available_layers)})"
        )

    workdir = Path(tempfile.mkdtemp())
    plots_root = workdir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    # Only the three requested figures.
    m.plot_steering_and_dprime(
        per_layer, factors, behavior,
        plots_root / "steering_acc_and_dprime.png",
    )
    m.plot_mcc_best_alpha_vs_dprime(
        per_layer, behavior,
        plots_root / "mcc_best_alpha_vs_dprime.png",
    )
    m.plot_projection_histograms(
        per_prompt, factors, behavior, target_layer, train_projections,
        plots_root / f"projection_hist_postgen_layer_{target_layer}.png",
        postgen=True,
    )

    out: dict[str, bytes] = {}
    for png in plots_root.rglob("*.png"):
        rel = png.relative_to(plots_root).as_posix()
        out[rel] = png.read_bytes()
    shutil.rmtree(workdir, ignore_errors=True)
    return out


# ---------------------------------------------------------------------------
# Modal entrypoints
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def replot(model_short: str, behavior: str, target_layer: int) -> dict:
    return _replot_one(model_short, behavior, target_layer)


@app.local_entrypoint()
def main(behavior: str = "",
         model_short: str = DEFAULT_MODEL_SHORT,
         all_behaviors: bool = False,
         target_layer: int = -1,
         out_dir: str = "paper_plots"):
    if not behavior and not all_behaviors:
        raise SystemExit("Pass --behavior <name> or --all-behaviors")
    targets = BEHAVIORS if all_behaviors else [behavior]
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    for b in targets:
        # If user didn't pass --target-layer (sentinel < 0), pick per-behavior default.
        layer = target_layer if target_layer >= 0 else BEHAVIOR_DEFAULT_LAYER.get(b, DEFAULT_LAYER)
        print(f"[replot] {b} (layer={layer}) ...")
        try:
            files = replot.remote(model_short, b, layer)
        except Exception as e:
            print(f"  ! failed: {e}")
            continue
        b_dir = base / b
        b_dir.mkdir(parents=True, exist_ok=True)
        short = BEHAVIOR_SHORT.get(b, b.split("-")[0])
        for rel, data in files.items():
            # Local renames using single-word behavior short label.
            if rel == "mcc_best_alpha_vs_dprime.png":
                rel = f"{short}_mcc_new.png"
            elif rel == "steering_acc_and_dprime.png":
                rel = f"{short}_steer_new.png"
            target = b_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            print(f"  wrote {target} ({len(data):,} bytes)")
    print(f"Done. Figures in {base.resolve()}")
