#!/usr/bin/env python3
"""
Warmup LTX 2.3 into VRAM by running a minimal T2V workflow once.
ComfyUI loads models on-demand; until a workflow runs, the checkpoint stays on disk.
Run this after ComfyUI is up (e.g. from run_ltx_rig.sh or manually) so the model is
loaded and ready for the first grid job.

Usage:
  python warmup_ltx_vram.py [--comfy-url http://127.0.0.1:8188] [--no-wait]
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKFLOW_PATH = SCRIPT_DIR / "workflows" / "ltx_2_3_t2v_multigpu.json"
COMFY_URL_DEFAULT = "http://127.0.0.1:8188"


def wait_for_comfy(comfy_url: str, timeout: float = 120, interval: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{comfy_url}/system_stats", timeout=5)
            if r.ok:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(interval)
    return False


def queue_prompt(comfy_url: str, workflow: dict) -> str:
    payload = {"prompt": workflow, "client_id": "warmup_ltx_vram"}
    resp = requests.post(f"{comfy_url}/prompt", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_done(comfy_url: str, prompt_id: str, poll_interval: float = 2.0) -> dict:
    while True:
        resp = requests.get(f"{comfy_url}/history/{prompt_id}", timeout=10)
        resp.raise_for_status()
        hist = resp.json()
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(poll_interval)


def main() -> int:
    ap = argparse.ArgumentParser(description="Warmup LTX 2.3 into VRAM via minimal T2V run")
    ap.add_argument("--comfy-url", default=COMFY_URL_DEFAULT, help="ComfyUI base URL")
    ap.add_argument("--no-wait", action="store_true", help="Do not wait for ComfyUI; fail if not ready")
    args = ap.parse_args()
    comfy_url = args.comfy_url.rstrip("/")

    if not WORKFLOW_PATH.exists():
        print(f"Workflow not found: {WORKFLOW_PATH}", file=sys.stderr)
        return 1

    if not args.no_wait:
        print("Waiting for ComfyUI...", flush=True)
        if not wait_for_comfy(comfy_url):
            print("ComfyUI did not become ready in time.", file=sys.stderr)
            return 1
        print("ComfyUI is up.", flush=True)

    with open(WORKFLOW_PATH) as f:
        workflow = json.load(f)

    # Minimal run: 17 frames (8n+1), 1 step, so model loads but finishes fast
    if "6" in workflow and workflow["6"].get("class_type") == "EmptyLTXVLatentVideo":
        workflow["6"]["inputs"]["length"] = 17
    if "9" in workflow and workflow["9"].get("class_type") == "KSampler":
        workflow["9"]["inputs"]["steps"] = 1

    print("Queueing minimal T2V warmup (loads LTX 2.3 into VRAM)...", flush=True)
    try:
        prompt_id = queue_prompt(comfy_url, workflow)
    except requests.exceptions.RequestException as e:
        print(f"Failed to queue warmup: {e}", file=sys.stderr)
        return 1

    print("Waiting for warmup to finish...", flush=True)
    try:
        wait_for_done(comfy_url, prompt_id)
    except requests.exceptions.RequestException as e:
        print(f"Warmup request failed: {e}", file=sys.stderr)
        return 1

    print("Warmup done. LTX 2.3 is loaded into VRAM.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
