"""
Unit tests: bridge initialization, video model advertising, workflow loading.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bridge import ComfyUIBridge

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workflows")


class TestBridgeInit:
    """Bridge constructor and workflow loading."""

    def test_video_workflow_loaded_when_file_set(self, workflow_dir):
        bridge = ComfyUIBridge(
            worker_name="t",
            api_key="k",
            workflow_dir=workflow_dir,
            workflow_file="turbovision.json",
            workflow_video_file="ltx_2_3_t2v.json",
        )
        assert bridge.workflow_video_template is not None
        assert "_bridge" in bridge.workflow_video_template
        assert bridge.workflow_video_template["_bridge"].get("media_type") == "video"

    def test_video_template_none_when_file_not_set(self, workflow_dir):
        bridge = ComfyUIBridge(
            worker_name="t",
            api_key="k",
            workflow_dir=workflow_dir,
            workflow_file="turbovision.json",
        )
        assert bridge.workflow_video_template is None

    def test_grid_video_model_defaults_when_ltx_workflow_set(self, workflow_dir):
        bridge = ComfyUIBridge(
            worker_name="t",
            api_key="k",
            workflow_dir=workflow_dir,
            workflow_video_file="ltx_2_3_t2v.json",
            grid_video_model=None,
        )
        assert bridge.grid_video_model == "ltx-2.3"

    def test_grid_video_model_defaults_when_i2v_only_set(self, workflow_dir):
        """I2V-only config (no T2V workflow) still defaults grid_video_model to ltx-2.3."""
        bridge = ComfyUIBridge(
            worker_name="t",
            api_key="k",
            workflow_dir=workflow_dir,
            workflow_file="turbovision.json",
            workflow_video_i2v_file="ltx_2_3_i2v_createvideo_multigpu_comfyorg.json",
            grid_video_model=None,
        )
        assert bridge.grid_video_model == "ltx-2.3"
        assert bridge.workflow_i2v_template is not None


class TestModelsAdvertising:
    """Video models are added to self.models when video workflow is configured."""

    @pytest.mark.asyncio
    async def test_video_model_added_to_models_when_configured(self, workflow_dir):
        bridge = ComfyUIBridge(
            worker_name="t",
            api_key="k",
            workflow_dir=workflow_dir,
            workflow_file="turbovision.json",
            workflow_video_file="ltx_2_3_t2v.json",
            grid_video_model="ltx-2.3",
        )
        await bridge.initialize_models()
        assert "ltx-2.3" in bridge.models

    @pytest.mark.asyncio
    async def test_image_only_bridge_does_not_advertise_video_model(self, bridge_image_only):
        bridge_image_only.workflow_template = {"_bridge": {}}
        await bridge_image_only.initialize_models()
        assert "ltx-2.3" not in bridge_image_only.models

    @pytest.mark.asyncio
    async def test_i2v_only_adds_video_model_to_available(self, workflow_dir):
        """When only I2V workflow is configured, ltx-2.3 is still added to available models list."""
        bridge = ComfyUIBridge(
            worker_name="t",
            api_key="k",
            workflow_dir=workflow_dir,
            workflow_file="turbovision.json",
            workflow_video_i2v_file="ltx_2_3_i2v_createvideo_multigpu_comfyorg.json",
            grid_video_model=None,
        )
        await bridge.initialize_models()
        assert "ltx-2.3" in bridge.models
        assert bridge.workflow_i2v_template is not None
        assert bridge.workflow_video_template is None


class TestModelsAdvertisingSync:
    """Sync: bridge configured for video has grid_video_model and template loaded."""

    def test_video_configured_state(self, bridge_with_video):
        assert bridge_with_video.grid_video_model == "ltx-2.3"
        assert bridge_with_video.workflow_video_template is not None
        assert bridge_with_video.workflow_video_template.get("_bridge", {}).get("media_type") == "video"
