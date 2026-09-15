#!/usr/bin/env bash
# One-command reproduction on an Intel system (e.g. the hackathon Core Ultra box, Ubuntu 24.04):
# builds the exact locked environment (uv.lock, Python 3.12), self-checks the sim, exports ACT to
# OpenVINO with a parity check, benchmarks every OpenVINO device (CPU / Arc GPU / NPU), and runs
# closed-loop MuJoCo episodes on the IR on each device.
#
#   bash scripts/intel_quickstart.sh <act pretrained_model dir or HF repo id>
#   EPISODES=10 bash scripts/intel_quickstart.sh ...
set -euo pipefail
ACT_CKPT=${1:?usage: $0 <act checkpoint dir or HF repo id>}
EPISODES=${EPISODES:-5}
export MUJOCO_GL=${MUJOCO_GL:-egl}  # headless rendering; use osmesa if EGL is unavailable
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --frozen                                   # exact versions from uv.lock, own Python 3.12
uv run python -m dinner.env                        # sim self-check
uv run python -m scripts.export_openvino --act-ckpt "$ACT_CKPT"
uv run python -m scripts.bench_intel --ckpt "$ACT_CKPT"

DEVICES=$(uv run python -c "import openvino as ov; print(' '.join(d for d in ov.Core().available_devices if d.split('.')[0] in ('CPU', 'GPU', 'NPU')))")
for dev in $DEVICES; do
    uv run python -m scripts.eval --policy act --ckpt "$ACT_CKPT" --backend openvino --ov-device "$dev" \
        --episodes "$EPISODES" --video --video-episodes 1 || echo "closed-loop eval on $dev failed; continuing"
done

echo "Done. See results/openvino_export.json, results/intel_bench.json, results/act_openvino-*_nominal_train.json, results/videos/"
