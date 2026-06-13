"""
Run open-ended projection-link sweeps on Modal and EXIT when done. HARD CODED IN TERMS OF ARGS, NEED TO EDIT FOR GENERALISIBIKITY.

Unlike run_modal_sleep.py (which sleeps 24 h and must be killed by hand), this
runs the sweep(s) as the function body. Modal tears the container — and the GPU
bill — down the moment the function returns. No manual `modal app stop` needed.

Per behavior it runs, e.g.:
    uv run python axbench/scripts/open_ended_projection_link.py \
        --behavior myopic-reward \
        --train_path /vol/expanded_datasets/generated/myopic-reward/train_contrastive.json \
        --test_path  /vol/expanded_datasets/generated/myopic-reward/test_contrastive.json \
        --output_dir /vol/open_ended_projection_link_sws/myopic-reward \
        --factors 1,2,3,5

Usage (from your laptop):
    modal run run_modal_sweep.py --behaviors myopic-reward
    modal run run_modal_sweep.py --behaviors "myopic-reward,sycophancy,survival-instinct"

Knobs you may need to confirm for your machine are marked CONFIRM below.
"""
import subprocess

import modal

app = modal.App("steering-sweep")

# Same volume as run_modal_sleep.py, mounted at /vol.
vol = modal.Volume.from_name("steering")

# CONFIRM: this image must have `uv` available so `uv run` works inside /vol.
# If your venv already lives on the volume, uv just reuses it; otherwise it syncs.
image = modal.Image.debian_slim().apt_install("git").pip_install("uv")

# CONFIRM: repo root on the volume (the dir containing axbench/ and datasets/).
REPO_DIR = "/vol/axbench"

# Per-behavior path roots (override via the local-entrypoint flags below).
DATASET_ROOT = "/vol/expanded_datasets/generated"     # {root}/{behavior}/{train,test}_contrastive.json
OUTPUT_ROOT = "/vol/open_ended_projection_link_sws"   # {root}/{behavior}

# CONFIRM: a Modal secret holding OPENAI_API_KEY (+ optional OPENAI_BASE_URL).
#   modal secret create openai-secret OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://us.api.openai.com
SECRET = modal.Secret.from_name("openai-secret")


@app.function(
    image=image,
    volumes={"/vol": vol},
    gpu="A100",
    timeout=86400,          # 24 h ceiling; the container exits as soon as the sweep returns
    secrets=[SECRET],
)
def run_sweep(
    behaviors: str,
    layers: str = "10-32",
    factors: str = "1,2,3,5",
    batch_size: int = 16,
    dataset_root: str = DATASET_ROOT,
    output_root: str = OUTPUT_ROOT,
):
    """Run the sweep for each behavior, then return → container auto-stops."""
    for behavior in [b.strip() for b in behaviors.split(",") if b.strip()]:
        train_path = f"{dataset_root}/{behavior}/train_contrastive.json"
        test_path = f"{dataset_root}/{behavior}/test_contrastive.json"
        out_dir = f"{output_root}/{behavior}"
        print(f"\n=== Sweeping {behavior} → {out_dir} ===", flush=True)
        subprocess.run(
            [
                "uv", "run", "python",
                "axbench/scripts/open_ended_projection_link.py",
                "--behavior", behavior,
                "--train_path", train_path,
                "--test_path", test_path,
                "--output_dir", out_dir,
                "--layers", layers,
                "--factors", factors,
                "--batch_size", str(batch_size),
            ],
            cwd=REPO_DIR,
            check=True,          # raise → fail loudly (still exits the container)
        )
        vol.commit()             # flush results to the volume after each behavior
        print(f"=== Done {behavior} ===", flush=True)


@app.local_entrypoint()
def main(
    behaviors: str = "myopic-reward",
    layers: str = "10-32",
    factors: str = "1,2,3,5",
    batch_size: int = 16,
    dataset_root: str = DATASET_ROOT,
    output_root: str = OUTPUT_ROOT,
):
    run_sweep.remote(behaviors, layers, factors, batch_size, dataset_root, output_root)
