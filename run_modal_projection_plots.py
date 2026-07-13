"""
Modal runner for open_ended_projection_plots.py.

Computes steered κ (batched GPU forward passes) plus projection histograms,
match-rate, avg-score, and MCC-vs-d' plots for the completed filtered_sweep
results already on the "steering" volume:

    /vol/filtered_sweep/gemma-2-9b-it/<behavior>-sweep/layer_{10..32}/...

Everything the analysis script needs — DiffMean_weight.pt, separability.json,
eval/eval_results.parquet, projection_analysis/{centroids,train_projections}.csv
— is already on disk; this job does NOT re-run the sweep or re-derive train
statistics, it only forward-passes the existing steered generations to get κ.

Unlike run_modal_sweep.py (which processed behaviors sequentially inside one
container), this spins up ONE container per behavior so all 4 run in parallel.

Run it detached so it survives your laptop sleeping/closing:

    modal run --detach run_modal_projection_plots.py

Monitor (read-only, no new container):
    modal app logs open-ended-projection-plots
    modal app list
"""
import os
import subprocess

import modal

app = modal.App("open-ended-projection-plots")

# Datasets + outputs live on the "steering" volume, mounted at /vol.
vol = modal.Volume.from_name("steering")

# Bake code + deps into the image (same recipe as run_modal_sweep.py).
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
            "run_modal_sweep.py", "run_modal_sleep.py",
            "run_modal_projection_plots.py",
        ],
    )
    .run_commands("cd /root/axbench && uv sync --frozen")
)

# Gemma is gated -> HF token. No OpenAI key needed: judge scores already exist
# in eval_results.parquet, this job only computes κ from existing generations.
SECRETS = [modal.Secret.from_name("hf-gemma-token")]

BEHAVIORS = [
    "myopic-reward",
    "survival-instinct",
    "hallucination",
    "corrigible-neutral-HHH",
]
LAYERS = "10-32"
FACTORS = "0,1,2,3,5,10"
BATCH_SIZE = 16


def _paths(behavior: str) -> tuple[str, str]:
    sweep_dir = f"/vol/filtered_sweep/gemma-2-9b-it/{behavior}-sweep"
    out_dir = f"{sweep_dir}/open_ended_projection_plots"
    return sweep_dir, out_dir


@app.function(
    image=image,
    volumes={"/vol": vol},
    gpu="A100",
    timeout=86400,  # 24h ceiling; exits as soon as this behavior's run returns
    secrets=SECRETS,
)
def run_behavior(behavior: str):
    # HuggingFace auth: Gemma is gated. Normalize whatever key the secret uses.
    hf_keys = [k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper()]
    print(f"[{behavior}] HF-ish env keys present:", hf_keys, flush=True)
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
        print(f"[{behavior}] HF token normalized", flush=True)
    else:
        print(f"[{behavior}] WARNING: no HF token found — gated model load will 401", flush=True)

    # Reuse the model weights already cached on the volume by run_modal_sweep.py.
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")

    sweep_dir, out_dir = _paths(behavior)
    print(f"\n=== START {behavior} → {out_dir} ===", flush=True)

    cmd = [
        "uv", "run", "python",
        "axbench/scripts/open_ended_projection_plots.py",
        "--behavior", behavior,
        "--sweep_dir", sweep_dir,
        "--layers", LAYERS,
        "--factors", FACTORS,
        "--batch_size", str(BATCH_SIZE),
    ]
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
        raise

    vol.commit()
    print(f"=== DONE {behavior} → {out_dir} ===", flush=True)


@app.local_entrypoint()
def main():
    # .spawn() per behavior fires 4 independent containers that run
    # concurrently on Modal — this call returns immediately and does not
    # block, so laptop sleep / client death cannot cancel the jobs.
    calls = []
    for behavior in BEHAVIORS:
        fc = run_behavior.spawn(behavior)
        calls.append((behavior, fc))
        print(f"Spawned {behavior}: call id={fc.object_id}")

    print("\nAll 4 behaviors running in parallel, detached in Modal cloud.")
    print("Safe to disconnect. Watch:  modal app logs open-ended-projection-plots")
    print("\nResults will land under /vol/filtered_sweep/gemma-2-9b-it/<behavior>-sweep/"
          "open_ended_projection_plots/ for:")
    for behavior, fc in calls:
        print(f"  {behavior}  (call id={fc.object_id})")
