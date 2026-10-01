import modal
import time

app = modal.App("sleep-container")

vol = modal.Volume.from_name("steering")

@app.function(volumes={"/vol": vol}, timeout=86400, gpu="A100")
def sleep_forever():
    print("Container started. Volume 'steering' mounted at /vol")
    print("Sleeping for 86400 seconds (24 hours)...")
    time.sleep(86400)
    print("Done sleeping")

@app.local_entrypoint()
def main():
    sleep_forever.remote()

