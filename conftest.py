"""
Pytest configuration and shared fixtures for comfy-bridge tests.
"""
import json
import os
import sys

import pytest

# Ensure package root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge import ComfyUIBridge, DummyJobPopResponse


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKFLOWS_DIR = os.path.join(os.path.dirname(__file__), "workflows")


def _load_workflow(name: str) -> dict:
    path = os.path.join(WORKFLOWS_DIR, name)
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bridge fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def workflow_dir():
    """Workflows directory (absolute)."""
    return WORKFLOWS_DIR


@pytest.fixture
def bridge_image_only(workflow_dir):
    """Bridge configured for image workflow only (no LTX video)."""
    return ComfyUIBridge(
        worker_name="test_worker",
        api_key="test_key",
        workflow_dir=workflow_dir,
        workflow_file="image_z_image_turbo_api.json",
    )


@pytest.fixture
def bridge_with_video(workflow_dir):
    """Bridge configured with both image and LTX video workflows."""
    bridge = ComfyUIBridge(
        worker_name="test_worker",
        api_key="test_key",
        workflow_dir=workflow_dir,
        workflow_file="image_z_image_turbo_api.json",
        workflow_video_file="ltx_2_3_t2v.json",
        grid_video_model="ltx-2.3",
    )
    return bridge


@pytest.fixture
def image_workflow_template():
    """Z-Image-Turbo style workflow with _bridge (image)."""
    return _load_workflow("image_z_image_turbo_api.json")


@pytest.fixture
def ltx_video_workflow_template():
    """LTX 2.3 T2V workflow with _bridge (video)."""
    return _load_workflow("ltx_2_3_t2v.json")


# ---------------------------------------------------------------------------
# Job fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def image_job():
    """Sample image generation job from the grid."""
    return DummyJobPopResponse(
        id="test-job-image-1",
        model="z_image_turbo",
        media_type="image",
        kudos=10,
        payload={
            "prompt": "A beautiful sunset over the ocean, golden hour",
            "negative_prompt": "blurry, low quality",
            "seed": 12345,
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "cfg_scale": 1,
            "sampler_name": "res_multistep",
        },
    )


@pytest.fixture
def video_job():
    """Sample video (LTX) job from the grid."""
    return DummyJobPopResponse(
        id="test-job-video-1",
        model="ltx-2.3",
        media_type="video",
        kudos=50,
        payload={
            "prompt": "A cat walking through a futuristic city",
            "negative_prompt": "",
            "length": 97,
            "fps": 24,
            "seed": 42,
            "steps": 20,
            "cfg_scale": 3.0,
        },
    )


@pytest.fixture
def video_job_with_fps():
    """Video job with explicit length and fps for metadata tests."""
    return DummyJobPopResponse(
        id="fps-test-job",
        model="ltx-2.3",
        media_type="video",
        payload={
            "prompt": "Test prompt",
            "negative_prompt": "",
            "length": 121,
            "fps": 25.0,
        },
    )
