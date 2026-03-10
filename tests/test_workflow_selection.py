"""
Unit tests: workflow selection by job type (image vs video) and fail-fast for video without template.
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


class TestWorkflowSelection:
    """_convert_job_to_workflow chooses image or video template by job.media_type."""

    def test_image_job_uses_image_template(
        self, bridge_image_only, image_workflow_template, image_job
    ):
        bridge_image_only.workflow_template = image_workflow_template
        workflow = bridge_image_only._convert_job_to_workflow(image_job)
        assert "_bridge" not in workflow
        # Image workflow has PrimitiveStringMultiline (Z-Image style)
        has_primitive = any(
            isinstance(n, dict) and n.get("class_type") == "PrimitiveStringMultiline"
            for n in workflow.values()
        )
        assert has_primitive

    def test_video_job_uses_video_template_when_configured(
        self, bridge_with_video, image_workflow_template, ltx_video_workflow_template, video_job
    ):
        bridge_with_video.workflow_template = image_workflow_template
        bridge_with_video.workflow_video_template = ltx_video_workflow_template
        workflow = bridge_with_video._convert_job_to_workflow(video_job)
        assert "_bridge" not in workflow
        # Video template has EmptyLTXVLatentVideo (node 5) with length
        assert "5" in workflow
        assert workflow["5"].get("inputs", {}).get("length") == 97

    def test_video_job_without_video_template_raises(self, workflow_dir):
        bridge = ComfyUIBridge(
            worker_name="test_worker",
            api_key="test_key",
            workflow_dir=workflow_dir,
            workflow_file="turbovision.json",
        )
        bridge.workflow_template = {"_bridge": {"nodes": {}}}
        bridge.workflow_video_template = None
        job = DummyJobPopResponse(
            id="video-no-template",
            model="ltx-2.3",
            media_type="video",
            payload={"prompt": "Test"},
        )
        with pytest.raises(ValueError) as exc_info:
            bridge._convert_job_to_workflow(job)
        assert "WORKFLOW_LTX_FILE" in str(exc_info.value) or "no LTX" in str(exc_info.value)

    def test_media_type_defaults_to_image(self, bridge_image_only, image_workflow_template):
        bridge_image_only.workflow_template = image_workflow_template
        job = DummyJobPopResponse(
            id="no-media-type",
            model="sdxl",
            payload={"prompt": "Hello"},
        )
        assert getattr(job, "media_type", None) in (None, "image")
        workflow = bridge_image_only._convert_job_to_workflow(job)
        assert "_bridge" not in workflow
