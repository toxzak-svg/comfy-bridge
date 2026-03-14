# ComfyUI Bridge for AI Power Grid

Connect your local ComfyUI installation to the AI Power Grid network and run it as a distributed image generation worker.

**Quick start for [aipg.art/create](https://aipg.art/create):** Start ComfyUI with your LTX workflow → set `AIPG_API_KEY`, `WORKFLOW_LTX_FILE`, `WORKFLOW_LTX_I2V_FILE` in `.env` → run `python start_bridge.py` → your worker will receive video jobs from the Create page. See [Using with aipg.art/create](#-using-with-aipgartcreate) for the full checklist.

---

## 🚀 Overview

- **Bridge**: Receives image-generation jobs from AI Power Grid.  
- **Worker**: Executes jobs via your local ComfyUI instance.  
- **Return**: Uploads generated images back to the network.  

This allows you to contribute GPU cycles to a decentralized AI rendering network while leveraging your local ComfyUI setup.

---

## 🎯 Features

- Auto-detects installed ComfyUI model checkpoints and maps them to AI Power Grid model names.  
- Customizable: override advertised models via `GRID_MODEL` (supports comma-separated lists).  
- Workflow templating: use your own ComfyUI `.json` workflow files.  
- Async, multi-threaded job polling and processing.  

---

## 🛠 Prerequisites

1. **Python 3.9+**  
2. **ComfyUI** running locally (default: `http://127.0.0.1:8000`).  
3. **AI Power Grid** account + API key: https://dashboard.aipowergrid.io  

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/youruser/comfy-bridge.git
cd comfy-bridge

# 2. Create & activate a virtual environment
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -e .
````

---

## ⚙️ Configuration

Copy the example `.env` and adjust values:

```ini
# .env
GRID_API_KEY=your_powergrid_api_key          # required
GRID_WORKER_NAME=MyComfyWorker.APIG_Wallet   # optional
COMFYUI_URL=http://127.0.0.1:8000            # optional
GRID_API_URL=https://api.aipowergrid.io/api  # optional
GRID_NSFW=false                              # allow NSFW? true/false
GRID_THREADS=2                               # concurrent jobs
GRID_MAX_PIXELS=1048576                      # max output resolution (pixels)
GRID_MODEL=stable_diffusion, Flux.1-Krea-dev Uncensored (fp8+CLIP+VAE)  # comma-separated model names
WORKFLOW_FILE=my_workflow.json               # ComfyUI JSON export template
```

* **`GRID_MODEL`** supports one or more model keys (comma-separated). If unset, the bridge auto-detects from your ComfyUI checkpoints.
* **`WORKFLOW_FILE`** points to a JSON workflow in your `workflows/` directory.
* **`WORKFLOW_LTX_FILE`** (optional) — LTX 2.3 **text-to-video** workflow JSON (e.g. `ltx_2_3_t2v.json`). When set, video jobs use this workflow. See [LTX 2.3 on this GPU rig](#ltx-23-on-this-gpu-rig) below.
* **`WORKFLOW_LTX_I2V_FILE`** (optional) — LTX 2.3 **image-to-video** workflow (e.g. `ltx_2_3_i2v_createvideo_multigpu_comfyorg.json`). When set, video jobs that include a source image use this I2V workflow; the workflow must have `_bridge` with `source_image` node.
* **`GRID_VIDEO_MODEL`** (optional) — Video model name(s) to advertise (e.g. `ltx-2.3`). If unset and `WORKFLOW_LTX_FILE` is set, defaults to `ltx-2.3`.

---

## LTX 2.3 on this GPU rig

You can run **LTX 2.3** video jobs on this machine by using ComfyUI with [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) and configuring the bridge to use an LTX 2.3 workflow for video jobs.

**Prerequisites**

- ComfyUI with the [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) custom nodes.
- LTX 2.3 models (e.g. Gemma text encoder, VAEs, checkpoints, optional distilled LoRA) — see [LTX ComfyUI docs](https://docs.ltx.video/open-source-model/integration-tools/comfy-ui). On this rig you can use `ComfyUI/download_ltx_supporting_models.py` to fetch supporting models.

**Configuration**

1. Export your LTX 2.3 workflow from ComfyUI in **API format**: Dev Mode → Save (API Format). Place it under `workflows/` (e.g. `workflows/ltx_2_3_t2v.json`).
2. Add a **`_bridge`** section to the workflow JSON so the bridge can inject job parameters. Include at least: `nodes` (e.g. `prompt`, `sampler`, `video_latent`, `output`, optionally `checkpoint`, `fps`), `fields`, `supports_negative`, and `media_type: "video"`.
3. In `.env` set:
   - `WORKFLOW_LTX_FILE=ltx_2_3_t2v.json` (or your filename)
   - Optionally `GRID_VIDEO_MODEL=ltx-2.3` (defaults to `ltx-2.3` when the LTX workflow is set)

**Behavior**

- One ComfyUI instance and one bridge process run on this machine. The bridge connects to `COMFYUI_URL` and can serve both image and video jobs.
- For **image** jobs the bridge uses `WORKFLOW_FILE` (image workflow). For **video** jobs (grid sends `media_type: "video"`) it uses `WORKFLOW_LTX_FILE` (LTX 2.3 T2V) or `WORKFLOW_LTX_I2V_FILE` (I2V when job has source image). If a video job arrives and no LTX workflow is configured, the bridge fails the job with a clear error.
- The bridge sets `length` and `fps` from the job payload into the video latent and FPS nodes when the workflow’s `_bridge` defines them.

**CLI**

```bash
python start_bridge.py --workflow turbovision.json --workflow-ltx ltx_2_3_t2v.json --workflow-ltx-i2v ltx_2_3_i2v.json --grid-video-model ltx-2.3
```

**LTX 2.3 HTTP API (alternative to ComfyUI for video)**

You can send **video** jobs to an external HTTP API instead of ComfyUI by setting in `.env`:

- **`LTX_API_URL`** — Base URL of the API (e.g. `https://api.ltx.video` or a self-hosted server at `http://127.0.0.1:7000`).
- **`LTX_API_KEY`** — (Optional) Bearer token if the API requires auth.
- **`LTX_ASYNC`** — Set to `true` if the API returns a `task_id` and you must poll for completion (e.g. `GET /v1/video/status/{task_id}`, then `GET /v1/video/result/{task_id}`).

The API must support:

- `POST /v1/upload` — I2V: upload source image; response: `upload_url`, `storage_uri`.
- `POST /v1/text-to-video` — T2V: body `prompt`, `model`, `resolution`, `duration`, `fps`; sync: return video bytes or JSON with `video_url`; async: return `task_id`.
- `POST /v1/image-to-video` — I2V: same + `image_uri`.
- If async: `GET /v1/video/status/{task_id}`, `GET /v1/video/result/{task_id}`.

**Options:**

1. **ComfyUI (default)** — Leave `LTX_API_URL` unset. Start ComfyUI with LTX (e.g. `./run_comfyui.sh` in ComfyUI), set `WORKFLOW_LTX_FILE` and `WORKFLOW_LTX_I2V_FILE`. The bridge sends video jobs to ComfyUI.
2. **Lightricks cloud API** — Set `LTX_API_URL=https://api.ltx.video` and `LTX_API_KEY=<your_key>` from [LTX Developer Console](https://docs.ltx.video). Video jobs go to their API; no local ComfyUI needed for video.
3. **Self-hosted API** — If you run a server that implements the same endpoints (e.g. on port 7000), set `LTX_API_URL=http://127.0.0.1:7000`. There is no standard “LTX API server” package; this is for custom or third-party implementations.

To run **LTX 2.3 image-to-video locally** (no grid): ensure ComfyUI is running with ComfyUI-LTXVideo and ComfyUI-VideoHelperSuite, then:

```bash
python run_ltx23_i2v_local.py --image path/to/image.png --prompt "Smooth motion, wind in the trees" --out my_video.mp4
```

**Docker**

Use the same `workflows/` mount and set `WORKFLOW_LTX_FILE` and `GRID_VIDEO_MODEL` in your env; no Dockerfile change is required unless you add extra dependencies for LTX.

---

## 🎨 Using with [aipg.art/create](https://aipg.art/create)

To have your **locally deployed LTX model and workflow** serve jobs from the [AI Power Grid Create page](https://aipg.art/create):

1. **Start ComfyUI** with your LTX 2.3 setup (both GPUs visible if using multi-GPU workflows). Ensure the LTX checkpoint and Gemma text encoder are in place (see [LTX 2.3 I2V models](workflows/README.md#ltx-23-i2v-models)).
2. **Configure the bridge** so it advertises the video model and points at your workflows. In `.env` (or via CLI):
   - `AIPG_API_KEY` or `GRID_API_KEY` — your key from [dashboard.aipowergrid.io](https://dashboard.aipowergrid.io).
   - `COMFYUI_URL` — your ComfyUI URL (e.g. `http://127.0.0.1:8188`).
   - `WORKFLOW_LTX_FILE=ltx_2_3_t2v_multigpu.json` (or your T2V workflow).
   - `WORKFLOW_LTX_I2V_FILE=ltx_2_3_i2v_createvideo_multigpu_comfyorg.json` (for image-to-video).
   - `GRID_VIDEO_MODEL=ltx-2.3` (optional; defaults to `ltx-2.3` when either LTX workflow is set).
3. **Start the bridge**: `python start_bridge.py` (or `./run_bridge.sh`). It registers with the grid and lists `ltx-2.3` in its available models.
4. **Use [aipg.art/create](https://aipg.art/create)** — when someone (or you) creates a video or image-to-video with a model your worker supports, the grid can assign the job to your bridge; your local ComfyUI runs the LTX workflow and the result is returned to the Create page.

The bridge and aipg.art both use the same AI Power Grid API; advertising `ltx-2.3` is what allows your worker to receive those jobs from the Create UI.

---

## ▶️ Running the Bridge

Start your ComfyUI web server, then:

```bash
# Via CLI module
python -m comfy_bridge.cli
```

Or directly (legacy):

```bash
start_bridge.py
```

The bridge will:

1. Register as a worker with AI Power Grid.
2. Poll for jobs every few seconds.
3. Render in ComfyUI.
4. Submit results back to the network.

---

## 🐳 Docker

### Build & Run the Container

1. **Build** the Docker image:

   ```bash
   docker build -t comfy-bridge .
   ```

2. **Run** the container:

   - **Linux** (host networking):
     ```bash
     docker run --rm --network host --env-file .env comfy-bridge
     ```

   - **macOS/Windows** (using `host.docker.internal`):
     ```bash
     docker run --rm \
       -v "$(pwd)/workflows:/app/workflows" \
       --env-file .env \
       -e COMFYUI_URL=http://host.docker.internal:8000 \
       comfy-bridge
     ```

---

## 🐳 Docker Compose

### If you prefer using Docker Compose to run your Container

**Linux**

Build & run:
   ```bash
   docker-compose -f docker-compose.linux.yml up --build
   ```

**macOS/Windows**

Build & run:
   ```bash
   docker-compose -f docker-compose.win-macos.yml up --build
   ```

---

## ✅ Testing

All core modules include unit and async tests. To run them:

```bash
pytest
```

Tests use `pytest-asyncio` for async routines and `respx` for HTTP mocking.

---

## 🐞 Troubleshooting

* **No jobs found?** Check `Advertising models:` log; ensure `GRID_MODEL` is set or your checkpoints match default mappings.
* **400 Bad Request**: unrecognized models—verify model key names or adjust `GRID_MODEL`.
* **ComfyUI unreachable**: confirm `COMFYUI_URL` and that the server is running.
* **API auth errors**: verify `GRID_API_KEY` and network access.

Logs are printed at INFO (bridge flow) and DEBUG (detailed payloads) levels. Adjust via:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgements

* **AI Power Grid** ([https://aipowergrid.io](https://aipowergrid.io)) - For the API
* **ComfyUI** ([https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)) - For the local image generation backend
* **httpx**, **aiohttp**, **pytest**, **pytest-asyncio** ❤️

```
```
