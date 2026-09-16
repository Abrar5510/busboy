#!/usr/bin/env bash
# One-command reproduction on an Intel system (e.g. a Core Ultra box, Ubuntu 24.04):
# builds the exact locked environment (uv.lock, Python 3.12), self-checks the sim, downloads the INT4 planner,
# exports ACT (and SmolVLA, if given) to OpenVINO with parity checks, benchmarks every OpenVINO device
# (CPU / Arc GPU / NPU), runs the agent eval with the planner on this machine, and closed-loop policy
# episodes on the IRs.
#
#   bash scripts/intel_quickstart.sh <act pretrained_model dir or HF repo id> [smolvla pretrained_model dir]
#   EPISODES=10 bash scripts/intel_quickstart.sh ...
set -euo pipefail
ACT_CKPT=${1:?usage: $0 <act checkpoint> [smolvla checkpoint]}
SMOLVLA_CKPT=${2:-}
EPISODES=${EPISODES:-5}
export MUJOCO_GL=${MUJOCO_GL:-egl}  # headless rendering; use osmesa if EGL is unavailable
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --frozen                                   # exact versions from uv.lock, own Python 3.12
uv run python -m dinner.env                        # sim self-check
uv run python -m dinner.perception                 # perception accuracy self-check
bash scripts/get_models.sh
uv run python -m scripts.export_openvino --act-ckpt "$ACT_CKPT" ${SMOLVLA_CKPT:+--smolvla-ckpt "$SMOLVLA_CKPT"}
uv run python -m scripts.bench_intel --ckpt "$ACT_CKPT" --planner ${SMOLVLA_CKPT:+--smolvla-ckpt "$SMOLVLA_CKPT"}

DEVICES=$(uv run python -c "import openvino as ov; print(' '.join(d for d in ov.Core().available_devices if d.split('.')[0] in ('CPU', 'GPU', 'NPU')))")
for dev in $DEVICES; do  # planner on each device (the cache is keyed by model+prompt, so clear it per device)
    rm -f outputs/planner_cache.json
    uv run python -m scripts.agent --eval --episodes "$EPISODES" --planner-device "$dev" --out "results/intel_$dev" \
        || echo "agent eval with planner on $dev failed; continuing"
done
for dev in $DEVICES; do
    uv run python -m scripts.eval --policy act --ckpt "$ACT_CKPT" --backend openvino --ov-device "$dev" \
        --episodes "$EPISODES" --video --video-episodes 1 || echo "ACT closed-loop eval on $dev failed; continuing"
done
if [ -n "$SMOLVLA_CKPT" ]; then
    uv run python -m scripts.eval --policy smolvla --ckpt "$SMOLVLA_CKPT" --backend openvino --ov-device GPU \
        --episodes "$EPISODES" || echo "SmolVLA closed-loop eval on GPU failed; continuing"
fi

echo "Done. See results/openvino_export.json, results/intel_bench.json, results/intel_*/, results/*_openvino-*.json"
