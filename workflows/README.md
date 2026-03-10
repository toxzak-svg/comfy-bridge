# ComfyUI Bridge Workflow Templates

This directory contains workflow templates for the ComfyUI Bridge. Each workflow file is a JSON representation of a ComfyUI workflow.

## How to Use

1. Create a JSON file with your ComfyUI workflow
2. Use placeholder values that will be replaced by the bridge:
   - `POSITIVE_PROMPT_PLACEHOLDER` - Will be replaced with the user's prompt
   - `NEGATIVE_PROMPT_PLACEHOLDER` - Will be replaced with the user's negative prompt
3. Run the bridge with your selected workflow file:
   ```
   python start_bridge.py --workflow your_workflow.json
   ```
   
   Alternatively, you can set the workflow file in your `.env` file:
   ```
   WORKFLOW_FILE=your_workflow.json
   ```

## Dynamic Values

The bridge will automatically update the following values in your workflow:

- `seed` - Replaced with the job's seed value
- `steps` - Replaced with the job's steps value
- `cfg` - Replaced with the job's CFG scale value
- `sampler_name` - Replaced with the job's sampler name (mapped to ComfyUI compatibility)
- `width` and `height` - Replaced with the job's requested dimensions
- `filename_prefix` - Replaced with a job-specific ID for the output file

## Available Workflow Templates

The repository includes several pre-made templates:

- `sd15_workflow.json` - For Stable Diffusion 1.5
- `sdxl_workflow.json` - For SDXL Base 1.0
- `sdxl_turbo_workflow.json` - For SDXL Turbo (optimized parameters)
- `turbovision.json` - For TurboVision XL
- `sdxl-lightning.json` - For SDXL Lightning
- `ltx_2_3_t2v.json` - LTX 2.3 text-to-video (fp8, 20s default, single-GPU)
- `ltx_2_3_t2v_multigpu.json` - LTX 2.3 text-to-video (distilled, 20s, **dual-GPU**, default for T2V)
- `ltx_2_3_i2v_createvideo_multigpu.json` - LTX 2.3 image-to-video (multi-GPU, 20s)

### LTX 2.3: speed and dual-GPU

- **20s videos**: Default is 481 frames @ 24 fps (LTX uses frame count `8n+1`). Override via job `length` / `fps`.
- **Faster without hurting quality**: T2V uses 16 steps; keep scheduler `simple`. Avoid going below ~14 steps for quality.
- **Dual-GPU (default)**:  
  - **T2V**: Default workflow is `ltx_2_3_t2v_multigpu.json` (uses `LTXV2CheckpointLoaderMultiGPU` + distilled checkpoint + Gemma text encoder). Start ComfyUI **without** `--cuda-device` so both GPUs are visible.  
  - **I2V**: `ltx_2_3_i2v_createvideo_multigpu.json` uses the same multi-GPU loaders.  
  Single-GPU T2V: set `WORKFLOW_LTX_FILE=ltx_2_3_t2v.json` (fp8 checkpoint).

## How to Get a Workflow File

1. Create your workflow in ComfyUI
2. Click the "Save" button to download the workflow as a JSON file
3. Copy the JSON file to this directory
4. Run the bridge with your selected workflow

## Special Node Requirements

For the bridge to properly update your workflow, it should contain:

1. A `KSampler` node (for updating sampling parameters)
2. `CLIPTextEncode` nodes with the placeholders mentioned above
3. An `EmptyLatentImage` node (for setting dimensions)
4. A `SaveImage` node (for saving the output with the correct filename)

The bridge will try to find these nodes by their class type, regardless of the node IDs used in your workflow. 