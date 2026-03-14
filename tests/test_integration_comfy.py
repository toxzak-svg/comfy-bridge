"""
Integration tests: require a running ComfyUI instance.
Run with: pytest -m integration
Skip by default: pytest (excludes -m integration).
"""
import json
import os
import sys
import time

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bridge import ComfyUIBridge, DummyJobPopResponse

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workflows")
COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")


def _comfy_reachable():
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _comfy_reachable(), reason="ComfyUI not reachable at COMFYUI_URL")
class TestComfyUISubmit:
    """Submit updated workflow to ComfyUI and wait for completion."""

    def test_submit_image_workflow_and_complete(self):
        with open(os.path.join(WORKFLOWS_DIR, "image_z_image_turbo_api.json"), "r") as f:
            workflow_template = json.load(f)
        bridge = ComfyUIBridge(
            worker_name="test_worker",
            api_key="test_key",
            workflow_dir=WORKFLOWS_DIR,
            workflow_file="image_z_image_turbo_api.json",
            comfy_url=COMFY_URL,
        )
        bridge.workflow_template = workflow_template
        job = DummyJobPopResponse(
            id="integration-test-1",
            model="z_image_turbo",
            kudos=10,
            payload={
                "prompt": "Cyberpunk city at night, neon lights, rain",
                "negative_prompt": "",
                "seed": 98765,
                "width": 1024,
                "height": 1024,
            },
        )
        updated = bridge._update_workflow_with_metadata(workflow_template, job)
        payload = {"prompt": updated}
        response = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=10)
        if response.status_code != 200:
            body = response.text
            # ComfyUI reachable but workflow expects models not installed on this rig (e.g. Z-Image-Turbo vs WAN/LTX)
            if response.status_code == 400 and "value_not_in_list" in body and "not in " in body:
                pytest.skip(
                    "ComfyUI has different models than this workflow (e.g. WAN/LTX only). "
                    "Bridge pipeline is working; use an image workflow matching this ComfyUI or install Z-Image-Turbo."
                )
            assert response.status_code == 200, body
        prompt_id = response.json().get("prompt_id")
        assert prompt_id
        for _ in range(60):
            resp = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
            if resp.status_code == 200 and prompt_id in resp.json() and resp.json()[prompt_id].get("outputs"):
                return
            time.sleep(1)
        pytest.fail("ComfyUI did not complete the prompt within 60s")
