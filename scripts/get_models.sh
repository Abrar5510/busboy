#!/usr/bin/env bash
# Download the pre-converted OpenVINO INT4 planner LLM into models/ (~1.1 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('OpenVINO/Qwen3-1.7B-int4-ov', local_dir='models/qwen3-1.7b-int4-ov')"
