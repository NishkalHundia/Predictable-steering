"""
Self-contained Modal runner for the vanilla open-ended projection-link sweep.

Calls open_ended_projection_link.py with --diffmean_mode last_token:
  Phase 0: last response token → DiffMean
  Phase B: inject steering on generated tokens (prefix_length onward)
  Post-gen κ: last generated content token

Uses prompted_datasets (Gemma-cued pos/neg answers).
Outputs: /vol/open_ended_projection_last_token/<behavior>

    MODAL_PROFILE=nishkalhundia modal run --detach run_modal_sweep.py

Monitor:
    MODAL_PROFILE=nishkalhundia modal app logs steering-last-sweep
    MODAL_PROFILE=nishkalhundia modal app list
"""
import os
import subprocess

import modal

app = modal.App("steering-last-sweep")

vol = modal.Volume.from_name("steering")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .add_local_dir(
        ".",
        "/root/axbench",
        copy=True,
        ignore=[
            "datasets", "results", "gemma2_2b_l10_steering", "paper_plots",
            ".git", ".venv", "__pycache__", "*.pyc", "*.png", "wandb",
            "run_modal_sweep.py", "run_modal_avg_sweep.py",
            "run_modal_prompted_sweep.py", "run_modal_sleep.py",
            "run_modal_prompted_contrastive.py",
        ],
    )
    .run_commands("cd /root/axbench && uv sync --frozen")
)

SECRETS = [
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("hf-gemma-token"),
]

BEHAVIORS = [
    "myopic-reward",
    "hallucination",
    "survival-instinct",
    "corrigible-neutral-HHH",
]

LAYERS = "10-32"
FACTORS = "0,1,2,3,5,10"
DIFFMEAN_MODE = "last_token"
FORCE_RECOMPUTE = True
REPLOT_ONLY = False
HIST_LAYERS = ",".join(str(l) for l in range(10, 33))
BATCH_SIZE = "32"
FLUENCY_THRESHOLD = "1.0"
MIN_EXAMPLES = "28"

assert not (REPLOT_ONLY and FORCE_RECOMPUTE), \
    "REPLOT_ONLY needs the cached CSV — set FORCE_RECOMPUTE = False."

OUTPUT_ROOT = "/vol/open_ended_projection_last_token"


def _paths(behavior):
    train = f"/vol/prompted_datasets/generated/{behavior}/train_contrastive.json"
    test = f"/vol/prompted_datasets/generated/{behavior}/test_contrastive.json"
    val = f"/vol/prompted_datasets/generated/{behavior}/val_contrastive.json"
    output_dir = f"{OUTPUT_ROOT}/{behavior}"
    return train, test, val, output_dir


def _normalize_hf_token():
    hf_keys = [k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper()]
    print("HF-ish env keys present:", hf_keys, flush=True)
    hf_tok = None
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
              "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(k):
            hf_tok = os.environ[k]
            break
    if hf_tok is None:
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
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")


@app.function(
    image=image,
    volumes={"/vol": vol},
    gpu=(None if REPLOT_ONLY else "A100"),
    timeout=86400,
    secrets=SECRETS,
    max_containers=len(BEHAVIORS),
)
def run_one(behavior: str, val_only: bool = False) -> tuple[str, str]:
    _normalize_hf_token()
    train_path, test_path, val_path, output_dir = _paths(behavior)
    verb = "VAL-ONLY" if val_only else ("REPLOT" if REPLOT_ONLY else "START")
    print(f"\n=== {verb} {DIFFMEAN_MODE} {behavior} → {output_dir} ===", flush=True)

    # --val_only reuses cached Phase 0 + test — never wipe those caches for it.
    force_recompute = FORCE_RECOMPUTE and not val_only
    if force_recompute:
        for fname in ("steering_state.pt", "dprime.json",
                      "train_projections.json", "per_prompt_results.csv",
                      "per_prompt_results_val.csv"):
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
        "--val_path", val_path,
        "--output_dir", output_dir,
        "--layers", LAYERS,
        "--factors", FACTORS,
        "--diffmean_mode", DIFFMEAN_MODE,
        "--batch_size", BATCH_SIZE,
        "--fluency_threshold", FLUENCY_THRESHOLD,
        "--min_examples", MIN_EXAMPLES,
        "--hist_layers", HIST_LAYERS,
    ]
    if val_only:
        cmd.append("--val_only")
    elif REPLOT_ONLY:
        cmd += ["--replot_only"]
    if force_recompute:
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
    vol.commit()
    print(f"=== DONE {behavior} → {output_dir} ===", flush=True)
    return (behavior, "done")


@app.function(
    image=image,
    volumes={"/vol": vol},
    timeout=86400,
    secrets=SECRETS,
)
def run_sweep():
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
    kind = "CPU replot" if REPLOT_ONLY else "GPU"
    fc = run_sweep.spawn()
    print(f"Spawned run_sweep (orchestrator) call id={fc.object_id}")
    print(f"Fans out {len(BEHAVIORS)} parallel {kind} workers (one per behavior).")
    print(f"diffmean_mode={DIFFMEAN_MODE}, batch={BATCH_SIZE}, "
          f"fluency>={FLUENCY_THRESHOLD}, min_examples={MIN_EXAMPLES}")
    print(f"data=/vol/prompted_datasets/generated/<behavior>/")
    if REPLOT_ONLY:
        print(f"Replot only — redraws plots (histograms for layers {HIST_LAYERS}).")
    print("Running detached in Modal cloud. Safe to disconnect.")
    print("Watch:  MODAL_PROFILE=nishkalhundia modal app logs steering-last-sweep")
    print(f"Results land under {OUTPUT_ROOT}/<behavior> for:")
    print("  " + ", ".join(BEHAVIORS))
