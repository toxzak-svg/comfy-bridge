#!/usr/bin/env python3
"""
Run LTX 2.3 image-to-video locally via ComfyUI API.

Prerequisites:
- ComfyUI running with ComfyUI-LTXVideo and LTX 2.3 models (see ComfyUI/download_ltx_supporting_models.py)
- ComfyUI-VideoHelperSuite for VHS_VideoCombine (or use a workflow that uses SaveVideo)
- Start ComfyUI without --cuda-device so both GPUs are visible (required for MultiGPU workflow).

Usage:
  python run_ltx23_i2v_local.py --image path/to/image.png --prompt "Smooth motion, wind in the trees"
  python run_ltx23_i2v_local.py --image frame.png --prompt "Camera slowly zooming in" --out my_video.mp4 --duration 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).resolve().parent
# LTX 2.3 I2V: MultiGPU loader + Gemma text encoder (no VHS required).
# Use ltx_2_3_i2v_createvideo.json if you have a checkpoint with built-in CLIP; use ltx_2_3_i2v.json if you have VideoHelperSuite.
DEFAULT_WORKFLOW = SCRIPT_DIR / "workflows" / "ltx_2_3_i2v_createvideo_multigpu_comfyorg.json"
COMFY_URL = "http://127.0.0.1:8188"


def check_comfy_reachable(comfy_url: str) -> None:
    """Verify ComfyUI is reachable; raise on failure."""
    try:
        r = requests.get(f"{comfy_url}/system_stats", timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"ComfyUI is not reachable at {comfy_url}. Start ComfyUI first (e.g. in tmux with run_ltx_rig.sh)."
        )
    except requests.exceptions.Timeout:
        raise SystemExit(f"ComfyUI at {comfy_url} did not respond in time.")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"ComfyUI check failed: {e}")


def upload_image(comfy_url: str, image_path: Path) -> str:
    """Upload image to ComfyUI; return filename as stored."""
    with open(image_path, "rb") as f:
        data = f.read()
    name = image_path.name
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        name = name + ".png"
    resp = requests.post(
        f"{comfy_url}/upload/image",
        files={"image": (name, data, "image/png")},
        data={"overwrite": "true"},
        timeout=60,
    )
    resp.raise_for_status()
    out = resp.json()
    return out.get("name", name)


def queue_prompt(comfy_url: str, workflow: dict) -> str:
    """Submit workflow and return prompt_id."""
    payload = {"prompt": workflow, "client_id": "ltx23_i2v_local"}
    resp = requests.post(f"{comfy_url}/prompt", json=payload, timeout=30)
    if not resp.ok:
        try:
            body = resp.json()
            err = body.get("error", body)
            if isinstance(err, dict):
                msg = err.get("message", err.get("details", ""))
                print(f"ComfyUI error ({resp.status_code}): {msg}")
                for node_id, ne in (body.get("node_errors") or {}).items():
                    for e in ne.get("errors", []):
                        print(f"  Node {node_id} ({ne.get('class_type', '')}): {e.get('message', '')} - {e.get('details', '')}")
            else:
                print(f"ComfyUI error ({resp.status_code}): {body}")
        except Exception:
            print(f"ComfyUI error ({resp.status_code}): {resp.text[:500]}")
        resp.raise_for_status()
    out = resp.json()
    return out["prompt_id"]


def wait_for_done(comfy_url: str, prompt_id: str, poll_interval: float = 2.0) -> dict:
    """Poll history until prompt is done; return history entry."""
    while True:
        resp = requests.get(f"{comfy_url}/history/{prompt_id}", timeout=10)
        resp.raise_for_status()
        hist = resp.json()
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(poll_interval)


def get_video_output(history_outputs: dict) -> tuple:
    """Get first video filename and subfolder from ComfyUI result. Returns (filename, subfolder)."""
    def _filename_and_subfolder(info: dict) -> tuple:
        if not info:
            return None, ""
        # Support both "filename" (SavedResult) and "name" (some APIs)
        name = info.get("filename") or info.get("name")
        sub = info.get("subfolder", "")
        return name, sub

    for node_id, out in history_outputs.items():
        if not isinstance(out, dict):
            continue
        if "gifs" in out and out["gifs"]:
            return _filename_and_subfolder(out["gifs"][0])
        if "videos" in out and out["videos"]:
            return _filename_and_subfolder(out["videos"][0])
        # SaveVideo/CreateVideo (comfy_extras) use PreviewVideo -> "images" + "animated"
        if "images" in out and out.get("animated") and out["images"]:
            return _filename_and_subfolder(out["images"][0])
    return None, ""


def download_output(comfy_url: str, filename: str, subfolder: str = "") -> bytes:
    """Download file from ComfyUI output."""
    params = {"filename": filename, "type": "output"}
    if subfolder:
        params["subfolder"] = subfolder
    resp = requests.get(f"{comfy_url}/view", params=params, timeout=120)
    resp.raise_for_status()
    return resp.content


def main():
    ap = argparse.ArgumentParser(description="LTX 2.3 image-to-video via local ComfyUI")
    ap.add_argument("--image", "-i", required=True, type=Path, help="Input image path")
    ap.add_argument("--prompt", "-p", default="Smooth motion, subtle movement, cinematic", help="Text prompt")
    ap.add_argument("--out", "-o", type=Path, default=None, help="Output video path (default: input name + _ltx23.mp4)")
    ap.add_argument("--comfy-url", default=COMFY_URL, help=f"ComfyUI URL (default: {COMFY_URL})")
    ap.add_argument("--workflow", "-w", type=Path, default=DEFAULT_WORKFLOW, help="Workflow JSON (LTX 2.3 I2V)")
    ap.add_argument("--duration", type=float, default=20, help="Video duration in seconds (default: 20)")
    ap.add_argument("--length", type=int, default=None, help="Number of frames (overrides --duration if set)")
    ap.add_argument("--fps", type=float, default=24, help="Output FPS (default: 24)")
    ap.add_argument("--steps", type=int, default=20, help="Sampling steps (default: 20)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}")
        print("  Use a path to a real image, e.g. /home/zach/ComfyUI/input/example.png")
        sys.exit(1)
    if not args.workflow.exists():
        print(f"Error: workflow not found: {args.workflow}")
        sys.exit(1)

    # Frames = duration * fps (default 20s @ 24fps = 480 frames)
    num_frames = args.length if args.length is not None else int(round(args.duration * args.fps))

    comfy_url = args.comfy_url.rstrip("/")
    print(f"ComfyUI: {comfy_url}")
    try:
        check_comfy_reachable(comfy_url)
    except SystemExit as e:
        print(e.args[0] if e.args else "ComfyUI not reachable.")
        sys.exit(1)
    print(f"Uploading image: {args.image}")
    try:
        filename = upload_image(comfy_url, args.image)
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)
    print(f"  -> {filename}")

    with open(args.workflow) as f:
        workflow = json.load(f)

    # Remove _bridge so ComfyUI gets a valid prompt
    if "_bridge" in workflow:
        bridge = workflow.pop("_bridge")
        nodes = bridge.get("nodes", {})
        fields = bridge.get("fields", {})
    else:
        nodes = {}
        fields = {}

    # Inject image into LoadImage node (node 6 in ltx_2_3_i2v.json)
    source_image_node = nodes.get("source_image", "6")
    if source_image_node in workflow:
        workflow[source_image_node]["inputs"]["image"] = filename

    # Prompt/negative
    prompt_node = nodes.get("prompt", "2")
    neg_node = nodes.get("negative_prompt", "3")
    if prompt_node in workflow:
        workflow[prompt_node]["inputs"]["text"] = args.prompt
    if neg_node in workflow:
        workflow[neg_node]["inputs"]["text"] = workflow[neg_node]["inputs"].get("text", "")

    # Video length (EmptyLTXVLatentVideo)
    latent_node = nodes.get("video_latent", "5")
    if latent_node in workflow:
        workflow[latent_node]["inputs"]["length"] = num_frames

    # FPS: set on output node (e.g. VHS_VideoCombine) and on conditioning if present (LTXVConditioning)
    fps_node_id = nodes.get("fps")
    if fps_node_id and fps_node_id in workflow:
        fps_field = fields.get("fps", "frame_rate")
        if fps_field in workflow[fps_node_id].get("inputs", {}):
            workflow[fps_node_id]["inputs"][fps_field] = args.fps
    for nid, node in workflow.items():
        if nid.startswith("_") or not isinstance(node, dict):
            continue
        if node.get("class_type") == "LTXVConditioning" and "frame_rate" in node.get("inputs", {}):
            node["inputs"]["frame_rate"] = args.fps
            break

    # Sampler: steps, seed
    sampler_node = nodes.get("sampler", "8")
    if sampler_node in workflow:
        workflow[sampler_node]["inputs"]["steps"] = args.steps
        if args.seed is not None:
            workflow[sampler_node]["inputs"]["seed"] = args.seed

    print("Queueing prompt...")
    try:
        prompt_id = queue_prompt(comfy_url, workflow)
    except Exception as e:
        print(f"Queue failed: {e}")
        sys.exit(1)
    print(f"  prompt_id: {prompt_id}")

    print("Waiting for completion (this may take several minutes)...")
    try:
        result = wait_for_done(comfy_url, prompt_id)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    outputs = result.get("outputs", {})
    status = result.get("status", {})
    video_filename, subfolder = get_video_output(outputs)
    if not video_filename:
        print("No video output in result. Check ComfyUI for errors.")
        if status.get("status_str") == "error":
            for msg in (status.get("messages") or []):
                if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
                    info = msg[1]
                    exc = info.get("exception_message", "")
                    print(f"  Execution error on node {info.get('node_id')} ({info.get('node_type', '')}): {exc}")
                    if "invalid tokenizer" in exc and "LTXV2AVTextEncoderLoaderMultiGPU" in str(info.get("node_type", "")):
                        print("  Fix: Your Gemma text encoder file does not include the tokenizer. Use the Comfy-Org encoder.")
                        print("  See: workflows/README_LTX_TEXT_ENCODER.md")
                        print("  Then run with: --workflow workflows/ltx_2_3_i2v_createvideo_multigpu_comfyorg.json")
        if not outputs:
            print("  (outputs were empty - run may have failed before any save node)")
        else:
            print("  Output keys per node:", {k: list(v.keys()) if isinstance(v, dict) else type(v).__name__ for k, v in outputs.items()})
        sys.exit(1)

    print(f"Downloading: {video_filename}")
    try:
        data = download_output(comfy_url, video_filename, subfolder)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        out_path = args.image.with_stem(args.image.stem + "_ltx23").with_suffix(".mp4")
    out_path = Path(out_path)
    out_path.write_bytes(data)
    print(f"Saved: {out_path} ({len(data)} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
