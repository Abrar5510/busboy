# CPU inference image: closed-loop MuJoCo evaluation of a trained policy, rendered headless.
# intel bench must run on x86 Intel hardware (see scripts/bench_intel.py)
#
#   docker build -t dinner-vla .
#   docker run --rm \
#     -v $PWD/outputs/train/smolvla_dinner/checkpoints/last/pretrained_model:/app/ckpt:ro \
#     -v $PWD/results:/app/results \
#     dinner-vla --policy smolvla --ckpt /app/ckpt --episodes 1 --video
#
# The agent (planner model mounted from scripts/get_models.sh):
#   docker run --rm -v $PWD/models:/app/models:ro -v $PWD/results:/app/results \
#     --entrypoint uv dinner-vla run --frozen python -m scripts.agent --eval --episodes 10
#
# If EGL cannot initialize without a GPU, add: -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libegl-mesa0 libgl1 libosmesa6 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=surfaceless \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY sim sim
COPY dinner dinner
COPY scripts scripts

ENTRYPOINT ["uv", "run", "--frozen", "python", "-m", "scripts.eval"]
