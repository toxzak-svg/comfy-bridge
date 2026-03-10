#!/usr/bin/env bash
# Activate and run the ComfyUI bridge (image + optional LTX video).
# Prerequisites: pip install -r requirements.txt, .env with GRID_API_KEY set.

set -e
cd "$(dirname "$0")"

# Load AIPG_API_KEY and other vars from home .env if present (job worker config)
if [ -f "${HOME}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  . "${HOME}/.env"
  set +a
fi

# Optional: use venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Default: image workflow + LTX video (T2V + optional I2V) if files exist
WORKFLOW="${WORKFLOW_FILE:-turbovision.json}"
LTX_WORKFLOW="${WORKFLOW_LTX_FILE:-}"
LTX_I2V_WORKFLOW="${WORKFLOW_LTX_I2V_FILE:-}"
if [ -n "$LTX_WORKFLOW" ] || [ -f "workflows/ltx_2_3_t2v.json" ]; then
  LTX_WORKFLOW="${LTX_WORKFLOW:-ltx_2_3_t2v.json}"
  EXTRA_ARGS="--workflow-ltx $LTX_WORKFLOW"
  if [ -n "$LTX_I2V_WORKFLOW" ] || [ -f "workflows/ltx_2_3_i2v_createvideo_multigpu_comfyorg.json" ] || [ -f "workflows/ltx_2_3_i2v.json" ]; then
    if [ -z "$LTX_I2V_WORKFLOW" ]; then
      if [ -f "workflows/ltx_2_3_i2v_createvideo_multigpu_comfyorg.json" ]; then
        LTX_I2V_WORKFLOW="ltx_2_3_i2v_createvideo_multigpu_comfyorg.json"
      else
        LTX_I2V_WORKFLOW="ltx_2_3_i2v.json"
      fi
    fi
    EXTRA_ARGS="$EXTRA_ARGS --workflow-ltx-i2v $LTX_I2V_WORKFLOW"
  fi
  python3 start_bridge.py --workflow "$WORKFLOW" $EXTRA_ARGS "$@"
else
  python3 start_bridge.py --workflow "$WORKFLOW" "$@"
fi
echo "Bridge exited with code $?"
