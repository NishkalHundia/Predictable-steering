"""
Self-contained Modal runner for the open-ended projection-link sweep.

Everything open_ended_projection_link.py needs is baked into the image:
  - the repo code (copied in at build time, so the content-token fix is included)
  - all Python deps (uv sync --frozen from the repo's pyproject/uv.lock)
Datasets + outputs live on the "steering" volume, mounted at /vol.

The sweep runs as the function body, so the container — and the GPU bill —
STOPS the moment the sweep returns. Run it detached so it survives your laptop
sleeping/closing:

    MODAL_PROFILE=nishkalhundia modal run --detach run_modal_sweep.py

Monitor (read-only, no new container):
    MODAL_PROFILE=nishkalhundia modal app logs steering-sweep
    MODAL_PROFILE=nishkalhundia modal app list
"""
import os
import subprocess

import modal

app = modal.App("steering-sweep")

# Volume we rely on (datasets at /vol/expanded_datasets, outputs at /vol/...).
vol = modal.Volume.from_name("steering")

# Bake code + deps into the image. Copy only the code (datasets/results live on
# the volume, not in the image) and uv-sync the locked dependency set.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .add_local_dir(
        ".",
        "/root/axbench",
        copy=True,  # available during the build so `uv sync` can run below
        ignore=[
            "datasets", "results", "gemma2_2b_l10_steering", "paper_plots",
            ".git", ".venv", "__pycache__", "*.pyc", "*.png", "wandb",
            # exclude the Modal runners themselves so editing them doesn't
            # invalidate the (expensive) uv-sync image layer
            "run_modal_sweep.py", "run_modal_sleep.py",
        ],
    )
    .run_commands("cd /root/axbench && uv sync --frozen")
)

# Gemma is gated → HF token; judge → OpenAI key.
SECRETS = [
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("hf-gemma-token"),
]

# Fixed config for this run (layers 10-32, factors 0,1,2,3,5,10).
# NOTE: these MUST match both the dataset dir names on the volume
# (/vol/expanded_datasets/generated/<behavior>/) AND the --behavior choices in
# open_ended_projection_link.py (e.g. "corrigible-neutral-HHH", not "corrigibility").
BEHAVIORS = [
    "myopic-reward",
]

LAYERS = "10-32"
FACTORS = "0,1,2,3,5,10"

# Phase 0 only: pass --prompted_diffmean (cue+question last-token DiffMean;
# no response teacher-forcing). Writes under
# /vol/open_ended_projection_link_prompted/ instead of _sws/.
PROMPTED_DIFFMEAN = True

# Phase 0 (steering_state.pt/dprime.json/train_projections.json) was already
# rebuilt with the fixed κ-position code in the last run, and layers are
# unchanged — so there's nothing to recompute this time (factor 0 is a no-op:
# the unsteered baseline is always generated regardless of --factors). Reuse
# the cache. Flip to True only when the underlying code/layers actually change.
FORCE_RECOMPUTE = True

# REPLOT-ONLY mode: redraw plots from the cached per_prompt_results.csv WITHOUT
# any generation/judge — so run_one needs no GPU (the container drops to CPU
# below) and no secrets. Use it to regenerate the per-layer projection
# histograms (or any plot) after a completed sweep. HIST_LAYERS controls which
# layers get a histogram file; here every layer in LAYERS (10..32).
REPLOT_ONLY = False
HIST_LAYERS = ",".join(str(l) for l in range(10, 33))

# Guard: replotting must not also try to force-recompute (which would delete the
# cache the replot depends on).
assert not (REPLOT_ONLY and FORCE_RECOMPUTE), \
    "REPLOT_ONLY needs the cached CSV — set FORCE_RECOMPUTE = False."

OUTPUT_ROOT = (
    "/vol/open_ended_projection_link_prompted" if PROMPTED_DIFFMEAN
    else "/vol/open_ended_projection_link_sws"
)


def _paths(behavior):
    train = f"/vol/expanded_datasets/generated/{behavior}/train_contrastive.json"
    test = f"/vol/expanded_datasets/generated/{behavior}/test_contrastive.json"
    output_dir = f"{OUTPUT_ROOT}/{behavior}"
    return train, test, output_dir


def _normalize_hf_token():
    """Gemma is gated. Normalize whatever key the secret uses into the names
    transformers/huggingface_hub actually read."""
    hf_keys = [k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper()]
    print("HF-ish env keys present:", hf_keys, flush=True)
    hf_tok = None
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
              "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(k):
            hf_tok = os.environ[k]
            break
    if hf_tok is None:  # fall back to any HF-ish var that looks like a token
        for k in hf_keys:
            if os.environ.get(k, "").startswith("hf_"):
                hf_tok = os.environ[k]
                break
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_tok
        print("HF token normalized into HF_TOKEN / HUGGING_FACE_HUB_TOKEN", flush=True)
    else:
        print("WARNING: no HF token found in env — gated model load will 401", flush=True)
    # Persist the model download on the volume so retries don't re-fetch 18 GB.
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")


@app.function(
    image=image,
    volumes={"/vol": vol},
    # Replot redraws from the cached CSV — no model, no GPU. Full sweeps need one.
    gpu=(None if REPLOT_ONLY else "A100"),
    timeout=86400,        # 24 h ceiling; exits as soon as this behavior returns
    secrets=SECRETS,
    # Each behavior gets its OWN container — map() below fans out 4 in parallel.
    max_containers=len(BEHAVIORS),
)
def run_one(behavior: str) -> tuple[str, str]:
    """Run the projection-link pipeline for a single behavior in its own container.
    Full sweep uses a GPU; REPLOT_ONLY runs on CPU (no generation/judge).
    Never raises — returns (behavior, status) so one failure can't abort siblings."""
    _normalize_hf_token()
    train_path, test_path, output_dir = _paths(behavior)
    verb = "REPLOT" if REPLOT_ONLY else "START"
    print(f"\n=== {verb} {behavior} → {output_dir} ===", flush=True)

    if FORCE_RECOMPUTE:
        # Clear stale Phase-0 cache so a code/layer change actually rebuilds μ±/v/d'.
        # (--force_recompute alone does NOT rebuild Phase 0 — it keys off these files.)
        for fname in ("steering_state.pt", "dprime.json",
                      "train_projections.json", "per_prompt_results.csv"):
            p = os.path.join(output_dir, fname)
            if os.path.exists(p):
                os.remove(p)
                print(f"removed stale {p}", flush=True)

    cmd = [
        "uv", "run", "python",
        "axbench/scripts/open_ended_projection_link.py",
        "--behavior", behavior,
        "--train_path", train_path,
        "--test_path", test_path,
        "--output_dir", output_dir,
        "--layers", LAYERS,
        "--factors", FACTORS,
        "--batch_size", "32",
    ]
    if REPLOT_ONLY:
        # Skip GPU/judge; just redraw plots (histograms for every HIST_LAYERS layer)
        # from the cached per_prompt_results.csv already on the volume.
        cmd += ["--replot_only", "--hist_layers", HIST_LAYERS]
    else:
        # Full sweep: still plot histograms for every layer (cheap vs generation).
        cmd += ["--hist_layers", HIST_LAYERS]
    if PROMPTED_DIFFMEAN:
        cmd.append("--prompted_diffmean")
    if FORCE_RECOMPUTE:
        cmd.append("--force_recompute")
    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(
            cmd,
            cwd="/root/axbench",
            check=True,
            env={**os.environ, "PYTHONPATH": "/root/axbench"},
        )
    except subprocess.CalledProcessError as e:
        print(f"!!! FAILED {behavior}: exit {e.returncode}", flush=True)
        return (behavior, f"failed(exit {e.returncode})")
    vol.commit()  # persist this behavior's results to the volume
    print(f"=== DONE {behavior} → {output_dir} ===", flush=True)
    return (behavior, "done")


@app.function(
    image=image,
    volumes={"/vol": vol},
    timeout=86400,        # cheap CPU orchestrator; just waits on the 4 GPU workers
    secrets=SECRETS,
)
def run_sweep():
    """Orchestrator: fan out one container per behavior and wait for all.
    Runs as the detached function so all 4 workers survive client disconnect.
    Workers are GPU for a full sweep, CPU when REPLOT_ONLY."""
    # .map() dispatches all behaviors at once → 4 containers in parallel.
    results = list(run_one.map(BEHAVIORS, order_outputs=False))
    done   = [b for b, s in results if s == "done"]
    failed = [f"{b}:{s}" for b, s in results if s != "done"]
    print(
        f"\n=== {'REPLOT' if REPLOT_ONLY else 'SWEEP'} SUMMARY ===\n"
        f"  done:   {done or '-'}\n"
        f"  failed: {failed or '-'}",
        flush=True,
    )


@app.local_entrypoint()
def main():
    # .spawn() fires the orchestrator and returns immediately — combined with
    # `modal run --detach`, the app + the 4 parallel workers run to completion
    # in Modal's cloud fully independent of this machine.
    kind = "CPU replot" if REPLOT_ONLY else "GPU"
    fc = run_sweep.spawn()
    print(f"Spawned run_sweep (orchestrator) call id={fc.object_id}")
    print(f"Fans out {len(BEHAVIORS)} parallel {kind} workers (one per behavior).")
    if REPLOT_ONLY:
        print(f"Replot only — redraws plots (histograms for layers {HIST_LAYERS}).")
    if PROMPTED_DIFFMEAN:
        print("PROMPTED_DIFFMEAN=True → Phase 0 uses --prompted_diffmean.")
    print("Running detached in Modal cloud. Safe to disconnect.")
    print("Watch:  MODAL_PROFILE=nishkalhundia modal app logs steering-sweep")
    print(f"Results land under {OUTPUT_ROOT}/<behavior> for:")
    print("  " + ", ".join(BEHAVIORS))
