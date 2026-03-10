#!/usr/bin/env python3
"""
Test the bridge's workflow update logic for Z-Image-Turbo.
This simulates what happens when a job comes in from the grid.
"""

import json
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge import ComfyUIBridge, DummyJobPopResponse

def test_bridge_logic():
    """Test the bridge's _update_workflow_with_job_params method."""
    
    print("=" * 60)
    print("Bridge Logic Test for Z-Image-Turbo")
    print("=" * 60)
    
    # Load the Z-Image-Turbo workflow
    workflow_path = "workflows/image_z_image_turbo_api.json"
    print(f"\nLoading workflow: {workflow_path}")
    
    with open(workflow_path, "r") as f:
        workflow_template = json.load(f)
    
    print(f"✓ Loaded {len(workflow_template)} nodes")
    
    # Create a mock bridge instance
    bridge = ComfyUIBridge(
        worker_name="test_worker",
        api_key="test_key",
        workflow_dir="workflows",
        workflow_file="image_z_image_turbo_api.json"
    )
    bridge.workflow_template = workflow_template
    
    # Create a mock job from the grid
    test_prompt = "A beautiful sunset over the ocean, golden hour lighting, photorealistic"
    test_negative = "blurry, low quality"  # Note: Z-Image doesn't use negative, but we test it doesn't break
    test_seed = 12345
    
    print(f"\nSimulating grid job:")
    print(f"  Prompt: {test_prompt[:50]}...")
    print(f"  Negative: {test_negative}")
    print(f"  Seed: {test_seed}")
    
    job_data = {
        "id": "test-job-123",
        "model": "z_image_turbo",
        "kudos": 10,
        "payload": {
            "prompt": test_prompt,
            "negative_prompt": test_negative,
            "seed": test_seed,
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "cfg_scale": 1,
            "sampler_name": "res_multistep"
        }
    }
    
    job = DummyJobPopResponse(**job_data)
    
    # Run the bridge's workflow update logic
    print("\nRunning bridge._update_workflow_with_metadata()...")
    print("-" * 40)
    
    updated_workflow = bridge._update_workflow_with_metadata(workflow_template, job)
    
    print("-" * 40)
    
    # Verify the results
    print("\nVerification:")
    
    errors = []
    
    # Check PrimitiveStringMultiline got the prompt
    for node_id, node in updated_workflow.items():
        if isinstance(node, dict) and node.get("class_type") == "PrimitiveStringMultiline":
            actual_prompt = node.get("inputs", {}).get("value", "")
            if actual_prompt == test_prompt:
                print(f"✓ PrimitiveStringMultiline ({node_id}): prompt set correctly")
            else:
                errors.append(f"PrimitiveStringMultiline prompt mismatch: got '{actual_prompt[:30]}...'")
                print(f"✗ PrimitiveStringMultiline ({node_id}): WRONG - got '{actual_prompt[:30]}...'")
    
    # Check KSampler got the seed
    for node_id, node in updated_workflow.items():
        if isinstance(node, dict) and node.get("class_type") == "KSampler":
            actual_seed = node.get("inputs", {}).get("seed")
            if actual_seed == test_seed:
                print(f"✓ KSampler ({node_id}): seed set correctly to {actual_seed}")
            else:
                errors.append(f"KSampler seed mismatch: expected {test_seed}, got {actual_seed}")
                print(f"✗ KSampler ({node_id}): WRONG - expected {test_seed}, got {actual_seed}")
    
    # Check CLIPTextEncode didn't get overwritten (it should get text from connection)
    for node_id, node in updated_workflow.items():
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            text_input = node.get("inputs", {}).get("text")
            if isinstance(text_input, list):
                print(f"✓ CLIPTextEncode ({node_id}): text is connection reference (not overwritten)")
            else:
                # It might be okay if it was a direct text input workflow
                print(f"  CLIPTextEncode ({node_id}): text is direct value: '{str(text_input)[:30]}...'")
    
    # Check SaveImage got job ID prefix
    for node_id, node in updated_workflow.items():
        if isinstance(node, dict) and node.get("class_type") == "SaveImage":
            prefix = node.get("inputs", {}).get("filename_prefix", "")
            if "test-job-123" in prefix:
                print(f"✓ SaveImage ({node_id}): filename_prefix set to '{prefix}'")
            else:
                print(f"  SaveImage ({node_id}): filename_prefix is '{prefix}'")
    
    print()
    if errors:
        print(f"✗ FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("✓ All checks passed!")
        return True

def test_with_comfy_api():
    """Actually submit the updated workflow to ComfyUI."""
    import requests
    import time
    
    print("\n" + "=" * 60)
    print("Full Integration Test (via ComfyUI API)")
    print("=" * 60)
    
    # Load workflow
    with open("workflows/image_z_image_turbo_api.json", "r") as f:
        workflow_template = json.load(f)
    
    # Create bridge
    bridge = ComfyUIBridge(
        worker_name="test_worker",
        api_key="test_key",
        workflow_dir="workflows",
        workflow_file="image_z_image_turbo_api.json"
    )
    bridge.workflow_template = workflow_template
    
    # Create job
    job_data = {
        "id": "integration-test-456",
        "model": "z_image_turbo",
        "kudos": 10,
        "payload": {
            "prompt": "Cyberpunk city at night, neon lights, rain reflections, cinematic",
            "negative_prompt": "",
            "seed": 98765,
            "width": 1024,
            "height": 1024,
        }
    }
    job = DummyJobPopResponse(**job_data)
    
    # Update workflow
    print("\nUpdating workflow with job parameters...")
    updated_workflow = bridge._update_workflow_with_metadata(workflow_template, job)
    
    # Submit to ComfyUI
    print(f"Submitting to ComfyUI...")
    payload = {"prompt": updated_workflow}
    response = requests.post("http://127.0.0.1:8188/prompt", json=payload)
    
    if response.status_code != 200:
        print(f"✗ Error: {response.status_code} - {response.text}")
        return False
    
    prompt_id = response.json().get("prompt_id")
    print(f"✓ Queued with prompt_id: {prompt_id}")
    
    # Wait for completion
    print("Waiting for generation...")
    for i in range(60):
        resp = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}")
        if resp.status_code == 200:
            history = resp.json()
            if prompt_id in history and history[prompt_id].get("outputs"):
                print(f"✓ Generation completed!")
                # Find image
                for node_id, output in history[prompt_id]["outputs"].items():
                    if "images" in output:
                        for img in output["images"]:
                            print(f"✓ Generated: {img['filename']}")
                return True
        time.sleep(1)
        print(f"  {i+1}s...", end="\r")
    
    print("✗ Timeout")
    return False


def test_video_workflow_selection():
    """When media_type is 'video' and workflow_video_template is set, _convert_job_to_workflow uses video template."""
    print("\n" + "=" * 60)
    print("LTX Video Workflow Selection Test")
    print("=" * 60)
    
    with open("workflows/ltx_2_3_t2v.json", "r") as f:
        video_template = json.load(f)
    
    bridge = ComfyUIBridge(
        worker_name="test_worker",
        api_key="test_key",
        workflow_dir="workflows",
        workflow_file="image_z_image_turbo_api.json",
        workflow_video_file="ltx_2_3_t2v.json",
    )
    bridge.workflow_template = {"_bridge": {"nodes": {}, "media_type": "image"}}  # minimal image template
    bridge.workflow_video_template = video_template
    
    job_data = {
        "id": "video-job-1",
        "model": "ltx-2.3",
        "media_type": "video",
        "kudos": 50,
        "payload": {"prompt": "A cat walking", "length": 97, "fps": 24},
    }
    job = DummyJobPopResponse(**job_data)
    
    workflow = bridge._convert_job_to_workflow(job)
    # Video template has EmptyLTXVLatentVideo (node 5) and _bridge was popped
    assert "_bridge" not in workflow, "Workflow sent to ComfyUI must not contain _bridge"
    # Node 5 in LTX template is EmptyLTXVLatentVideo with length
    if "5" in workflow and "inputs" in workflow["5"]:
        assert workflow["5"]["inputs"].get("length") == 97, "Video length should be set to 97"
    print("✓ Video job uses video template and length is set")
    return True


def test_video_job_no_template_fails():
    """Video job without workflow_video_template raises clear error."""
    print("\n" + "=" * 60)
    print("Video Job Without Template (Fail Fast) Test")
    print("=" * 60)
    
    bridge = ComfyUIBridge(
        worker_name="test_worker",
        api_key="test_key",
        workflow_dir="workflows",
        workflow_file="turbovision.json",
    )
    bridge.workflow_template = {"_bridge": {"nodes": {}}}
    bridge.workflow_video_template = None
    
    job_data = {
        "id": "video-job-2",
        "model": "ltx-2.3",
        "media_type": "video",
        "payload": {"prompt": "Test"},
    }
    job = DummyJobPopResponse(**job_data)
    
    try:
        bridge._convert_job_to_workflow(job)
        print("✗ Expected ValueError for video job without video template")
        return False
    except ValueError as e:
        if "WORKFLOW_LTX_FILE" in str(e) or "no LTX" in str(e):
            print("✓ Video job without template raises clear ValueError")
            return True
        raise
    return False


def test_video_length_and_fps_in_metadata():
    """_update_workflow_with_metadata sets length and fps for video _bridge workflow."""
    with open("workflows/ltx_2_3_t2v.json", "r") as f:
        workflow_template = json.load(f)
    
    bridge = ComfyUIBridge(
        worker_name="test_worker",
        api_key="test_key",
        workflow_dir="workflows",
        workflow_video_file="ltx_2_3_t2v.json",
    )
    bridge.workflow_video_template = workflow_template
    
    job_data = {
        "id": "fps-test",
        "model": "ltx-2.3",
        "payload": {"prompt": "Test", "negative_prompt": "", "length": 121, "fps": 25.0},
    }
    job = DummyJobPopResponse(**job_data)
    
    out = bridge._update_workflow_with_metadata(workflow_template, job)
    assert "_bridge" not in out
    # video_latent is node 5
    assert out.get("5", {}).get("inputs", {}).get("length") == 121
    # fps is node 4 (LTXVConditioning frame_rate)
    assert out.get("4", {}).get("inputs", {}).get("frame_rate") == 25.0
    print("✓ Length and FPS set correctly in video workflow")
    return True


if __name__ == "__main__":
    # Prefer pytest when available: run tests in tests/
    try:
        import pytest
        tests_dir = os.path.join(os.path.dirname(__file__), "tests")
        sys.exit(pytest.main([tests_dir, "-v", "--tb=short", "-m", "not integration"]))
    except ImportError:
        pass
    # Fallback: run legacy script-style tests
    success = test_bridge_logic()
    if success:
        try:
            test_video_workflow_selection()
            test_video_job_no_template_fails()
            test_video_length_and_fps_in_metadata()
        except Exception as e:
            import traceback
            print(f"\nVideo tests error: {e}")
            traceback.print_exc()
            success = False
    if success:
        try:
            test_with_comfy_api()
        except Exception as e:
            print(f"\nIntegration test skipped: {e}")
    print("\n" + "=" * 60)
    sys.exit(0 if success else 1)
