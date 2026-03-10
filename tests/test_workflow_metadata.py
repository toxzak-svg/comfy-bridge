"""
Unit tests: workflow update via _bridge metadata (image and video).
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bridge import ComfyUIBridge, DummyJobPopResponse

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workflows")


def load_workflow(name):
    with open(os.path.join(WORKFLOWS_DIR, name), "r") as f:
        return json.load(f)


class TestImageWorkflowMetadata:
    """Image workflow: _update_workflow_with_metadata sets prompt, seed, dimensions, output prefix."""

    def test_prompt_and_seed_set(
        self, bridge_image_only, image_workflow_template, image_job
    ):
        bridge_image_only.workflow_template = image_workflow_template
        updated = bridge_image_only._update_workflow_with_metadata(
            image_workflow_template, image_job
        )
        assert "_bridge" not in updated
        # Z-Image uses PrimitiveStringMultiline for prompt
        for nid, node in updated.items():
            if isinstance(node, dict) and node.get("class_type") == "PrimitiveStringMultiline":
                assert node["inputs"].get("value") == image_job.payload.prompt
                break
        for nid, node in updated.items():
            if isinstance(node, dict) and node.get("class_type") == "KSampler":
                assert node["inputs"].get("seed") == image_job.payload.seed
                break

    def test_output_filename_prefix_contains_job_id(
        self, bridge_image_only, image_workflow_template, image_job
    ):
        bridge_image_only.workflow_template = image_workflow_template
        updated = bridge_image_only._update_workflow_with_metadata(
            image_workflow_template, image_job
        )
        for nid, node in updated.items():
            if isinstance(node, dict) and node.get("class_type") == "SaveImage":
                assert image_job.id in node["inputs"].get("filename_prefix", "")
                return
        pytest.fail("No SaveImage node found")

    def test_workflow_sent_to_comfy_has_no_bridge_key(
        self, bridge_image_only, image_workflow_template, image_job
    ):
        updated = bridge_image_only._update_workflow_with_metadata(
            image_workflow_template, image_job
        )
        assert "_bridge" not in updated


class TestVideoWorkflowMetadata:
    """Video (LTX) workflow: length, fps, checkpoint overwrite."""

    def test_video_length_set(self, bridge_with_video, ltx_video_workflow_template, video_job):
        bridge_with_video.workflow_video_template = ltx_video_workflow_template
        updated = bridge_with_video._update_workflow_with_metadata(
            ltx_video_workflow_template, video_job
        )
        assert "_bridge" not in updated
        # Node 5 = EmptyLTXVLatentVideo
        assert updated.get("5", {}).get("inputs", {}).get("length") == 97

    def test_video_fps_set(
        self, bridge_with_video, ltx_video_workflow_template, video_job_with_fps
    ):
        bridge_with_video.workflow_video_template = ltx_video_workflow_template
        updated = bridge_with_video._update_workflow_with_metadata(
            ltx_video_workflow_template, video_job_with_fps
        )
        # Node 4 = LTXVConditioning with frame_rate
        assert updated.get("4", {}).get("inputs", {}).get("frame_rate") == 25.0

    def test_video_length_121(self, bridge_with_video, ltx_video_workflow_template, video_job_with_fps):
        bridge_with_video.workflow_video_template = ltx_video_workflow_template
        updated = bridge_with_video._update_workflow_with_metadata(
            ltx_video_workflow_template, video_job_with_fps
        )
        assert updated.get("5", {}).get("inputs", {}).get("length") == 121
