# LTX 2.3 I2V – "invalid tokenizer" fix

If you see:

**Execution error on node 2 (LTXV2AVTextEncoderLoaderMultiGPU): invalid tokenizer**

the Gemma 3 12B text encoder you have does **not** include the tokenizer. ComfyUI expects a single `.safetensors` file that contains both the model weights and the `spiece_model` (tokenizer).

## Fix: use the correct text encoder

1. Download a **single-file** Gemma 3 12B text encoder that includes the tokenizer from **Comfy-Org**:

   - **Smaller (recommended):**  
     [gemma_3_12B_it_fp4_mixed.safetensors](https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors) (~9.5 GB)

   - **Larger options:**  
     [gemma_3_12B_it_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors) (~13.2 GB)  
     [gemma_3_12B_it.safetensors](https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it.safetensors) (~24.4 GB, full precision)

2. Put the file in your ComfyUI text encoders folder:
   ```text
   ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
   ```
   (Use the same filename as in the download.)

3. Use the workflow that references this file:
   ```bash
   python3 run_ltx23_i2v_local.py --image /path/to/image.png --prompt "Your prompt" \
     --workflow workflows/ltx_2_3_i2v_createvideo_multigpu_comfyorg.json
   ```

The multi-shard `gemma-3-12b-it-qat-q4_0-unquantized/model-0000X-of-00005.safetensors` files from other sources often do **not** include the tokenizer; those will keep giving "invalid tokenizer" until you switch to a Comfy-Org single-file encoder above.
