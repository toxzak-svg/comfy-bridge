#!/usr/bin/env python3
"""
Start script for the ComfyUI Bridge for AI Power Grid.

This script loads environment variables from a .env file and starts the bridge.
"""

import os
import sys
import logging
import asyncio
import argparse
import signal
from dotenv import load_dotenv

# Import the bridge module
from bridge import ComfyUIBridge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for shutdown handling
bridge = None
shutdown_event = None

def print_header():
    """Print a banner for the ComfyUI Bridge."""
    print("\n" + "=" * 80)
    print(" " * 25 + "ComfyUI Bridge for AI Power Grid")
    print("=" * 80)
    print("This bridge connects your local ComfyUI installation to the AI Power Grid,")
    print("allowing it to work as an image worker for the distributed network.")
    print("=" * 80 + "\n")

def handle_shutdown(sig, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_event
    
    if shutdown_event is None:
        return
        
    logger.info("Shutdown signal received. Gracefully stopping worker...")
    shutdown_event.set()

async def main():
    """Main entry point for the bridge."""
    global bridge, shutdown_event
    
    # Create shutdown event for graceful termination
    shutdown_event = asyncio.Event()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start ComfyUI bridge')
    parser.add_argument('--workflow', '-w', help='Workflow JSON file to use from workflows directory')
    parser.add_argument('--grid-model', help='Model to advertise to the grid (overrides GRID_MODEL in .env)')
    parser.add_argument('--workflow-ltx', help='LTX 2.3 video (T2V) workflow JSON (e.g. ltx_2_3_t2v.json)')
    parser.add_argument('--workflow-ltx-i2v', help='LTX 2.3 image-to-video workflow JSON (e.g. ltx_2_3_i2v.json)')
    parser.add_argument('--grid-video-model', help='Video model name(s) to advertise (e.g. ltx-2.3)')
    args = parser.parse_args()
    
    # Load environment variables: home .env first (e.g. AIPG_API_KEY), then local .env
    load_dotenv(os.path.expanduser("~/.env"))
    load_dotenv()
    
    # Get configuration from environment variables or use defaults (prefer AIPG_API_KEY for job worker)
    api_key = os.environ.get("AIPG_API_KEY") or os.environ.get("GRID_API_KEY", "")
    worker_name = os.environ.get("GRID_WORKER_NAME", "ComfyUI-Bridge-Worker")
    comfy_url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")  # ComfyUI default port
    ltx_desktop_url = os.environ.get("LTX_DESKTOP_URL", "http://127.0.0.1:3000")  # LTX Desktop local server
    nsfw = os.environ.get("GRID_NSFW", "false").lower() == "true"
    threads = int(os.environ.get("GRID_THREADS", "1"))
    max_pixels = int(os.environ.get("GRID_MAX_PIXELS", "1048576"))
    api_url = os.environ.get("GRID_API_URL", "https://api.aipowergrid.io/api")
    workflow_dir = os.environ.get("WORKFLOW_DIR", os.path.join(os.getcwd(), "workflows"))
    
    # Use command line argument for workflow file if provided, otherwise use environment variable
    workflow_file = args.workflow or os.environ.get("WORKFLOW_FILE")
    
    # Use command line argument for grid model if provided, otherwise use environment variable
    grid_model = args.grid_model or os.environ.get("GRID_MODEL")
    
    # LTX 2.3 video: T2V workflow (default multi-GPU), optional I2V workflow, and model to advertise
    workflow_video_file = args.workflow_ltx or os.environ.get("WORKFLOW_LTX_FILE") or "ltx_2_3_t2v_multigpu.json"
    workflow_video_i2v_file = args.workflow_ltx_i2v or os.environ.get("WORKFLOW_LTX_I2V_FILE")
    grid_video_model = args.grid_video_model or os.environ.get("GRID_VIDEO_MODEL")
    
    # LTX API (optional): when set, video jobs use LTX-2.3 HTTP API instead of ComfyUI
    ltx_base_url = os.environ.get("LTX_API_URL", "").strip() or None
    ltx_api_key = os.environ.get("LTX_API_KEY", "").strip() or None
    ltx_async = os.environ.get("LTX_ASYNC", "false").lower() == "true"
    
    if not api_key:
        print("Error: GRID_API_KEY environment variable is required")
        sys.exit(1)
    
    if workflow_file:
        print(f"Using workflow file: {workflow_file}")
    else:
        print("No workflow file specified. Using default workflow.")
    
    if workflow_video_file:
        print(f"Using LTX video (T2V) workflow: {workflow_video_file}")
    if workflow_video_i2v_file:
        print(f"Using LTX image-to-video workflow: {workflow_video_i2v_file}")
    if grid_video_model:
        print(f"Advertising video model(s) to grid: {grid_video_model}")
    if ltx_base_url:
        print(f"LTX API enabled: {ltx_base_url} (async={ltx_async})")
        
    if grid_model:
        print(f"Advertising model to grid: {grid_model}")
    
    print_header()
    
    print("Starting bridge with the following configuration:")
    print(f"- Worker name: {worker_name}")
    print(f"- ComfyUI URL: {comfy_url}")
    print(f"- LTX Desktop URL: {ltx_desktop_url}")
    print(f"- API URL: {api_url}")
    print(f"- NSFW allowed: {nsfw}")
    print(f"- Concurrent threads: {threads}")
    print(f"- Max pixels: {max_pixels}\n")
    
    try:
        # Create and start the bridge
        bridge = ComfyUIBridge(
            api_key=api_key,
            worker_name=worker_name,
            base_url=api_url,
            comfy_url=comfy_url,
            ltx_desktop_url=ltx_desktop_url,
            nsfw=nsfw,
            threads=threads,
            max_pixels=max_pixels,
            workflow_dir=workflow_dir,
            workflow_file=workflow_file,
            grid_model=grid_model,
            workflow_video_file=workflow_video_file,
            grid_video_model=grid_video_model,
            workflow_video_i2v_file=workflow_video_i2v_file,
            ltx_base_url=ltx_base_url,
            ltx_api_key=ltx_api_key,
            ltx_async=ltx_async,
        )
        
        # Start the bridge
        bridge_task = asyncio.create_task(bridge.start())
        
        # Wait for either bridge to complete or shutdown event
        await asyncio.wait(
            [bridge_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # If we're here because of shutdown event, ensure worker is properly shut down
        if shutdown_event.is_set() and bridge.running:
            logger.info("Gracefully shutting down worker...")
            await bridge._unregister_worker()
            await bridge._cleanup()
            logger.info("Worker gracefully stopped. Note: It may still appear online in the AI Power Grid for a while.")
    except Exception as e:
        logger.error(f"Error starting bridge: {e}")
    finally:
        # Ensure we clean up resources
        if bridge:
            await bridge._cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 