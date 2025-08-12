# ComfyUI-LipSyncNodes

Nodes for turning audio into phoneme cues (Rhubarb) and composing PNG/MP4/MOV with optional audio mux.

## Install
1. Copy this folder to `ComfyUI/custom_nodes/ComfyUI-LipSyncNodes`.
2. `pip install -r custom_nodes/ComfyUI-LipSyncNodes/requirements.txt`
3. Install ffmpeg (e.g. `brew install ffmpeg`).
4. Get Rhubarb binary via `scripts/get_rhubarb_mac.sh` or `scripts/get_rhubarb_win.ps1` and set the node's `rhubarb_path`.

## Output
Set `LIPSYNC_OUTPUT_DIR` env var or use node's `output_dir`. Files auto-increment to avoid overwrite.

Built 2025-08-12.
