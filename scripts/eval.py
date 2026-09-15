"""Closed-loop evaluation of ACT / SmolVLA checkpoints in the dinner-table MuJoCo env.

    python scripts/eval.py --policy smolvla --ckpt outputs/train/smolvla_dinner/checkpoints/last/pretrained_model
    python scripts/eval.py --policy smolvla --ckpt ... --paraphrases eval          # held-out instruction wording
    python scripts/eval.py --policy smolvla --ckpt ... --rand heavy                # 1.5x randomization
    python scripts/eval.py --policy act --ckpt ... --backend openvino --ir results/act_dinner.xml
    mjpython scripts/eval.py ... --viewer                                          # live MuJoCo viewer (macOS)

Writes results/{policy}_{backend}_{rand}_{paraphrases}.json and, with --video, mp4s under results/videos/.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from lerobot.policies import get_policy_class, make_pre_post_processors

from dinner.env import CAMERAS, FPS, TASKS, DinnerEnv

EVAL_SEED_START = 10000  # collection uses seeds below this
TIMEOUT_S = 20


def to_batch(obs, text):
    """Env observation (uint8 HWC images) -> unbatched-policy input, as lerobot's eval loop builds it."""
    batch = {k: torch.from_numpy(v).permute(2, 0, 1)[None].float().div(255)
             for k, v in obs.items() if k.startswith("observation.images.")}
    batch["observation.state"] = torch.from_numpy(obs["observation.state"])[None]
    batch["task"] = [text]
    return batch


class OpenVINOACT:
    """ACT action-chunk IR behind the policy's select_action/reset interface.
    Input order must match scripts/export_openvino.py: state, then config.image_features order."""

    def __init__(self, ir_path, config):
        import openvino as ov

        # f32 so closed-loop actions match torch (CPU default is f16 on ARM, bf16 on AMX Xeons).
        self.model = ov.Core().compile_model(str(ir_path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        self.config = config
        self.image_keys = list(config.image_features)
        self.queue = []

    def reset(self):
        self.queue.clear()

    def select_action(self, batch):
        if not self.queue:
            inputs = [batch["observation.state"].cpu().numpy()] + [batch[k].cpu().numpy() for k in self.image_keys]
            chunk = self.model(inputs)[0]  # (1, chunk_size, action_dim)
            self.queue = [torch.from_numpy(chunk[:, i]) for i in range(self.config.n_action_steps)]
        return self.queue.pop(0)


def overlay(obs, text, p50_ms, t):
    frame = np.hstack([obs[f"observation.images.{k}"] for k in CAMERAS])
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    label = f"{text} | p50 {p50_ms:.0f} ms | t={t:.1f}s"
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def run_episode(env, policy, pre, post, task, text, seed, level, infer_every, writer=None, viewer=None):
    obs = env.reset(seed, level)
    policy.reset()
    lat = []
    for step in range(TIMEOUT_S * FPS):
        batch = pre(to_batch(obs, text))
        tic = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(batch)
        if action.device.type == "mps":
            torch.mps.synchronize()
        lat.append(time.perf_counter() - tic)
        action = post(action)
        env.step(action.squeeze(0).cpu().numpy())
        obs = env.obs()
        if viewer is not None:
            viewer.sync()
        if writer is not None:
            infer = lat[::infer_every]
            writer.append_data(overlay(obs, text, 1000 * float(np.median(infer)), (step + 1) / FPS))
        if env.success(task):
            return True, (step + 1) / FPS, lat
    return False, None, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["act", "smolvla"])
    ap.add_argument("--ckpt", required=True, help="pretrained_model dir or HF repo id")
    ap.add_argument("--backend", default="torch", choices=["torch", "openvino"])
    ap.add_argument("--ir", default="results/act_dinner.xml", help="OpenVINO IR for --backend openvino")
    ap.add_argument("--episodes", type=int, default=20, help="per task")
    ap.add_argument("--rand", default="nominal", choices=["nominal", "heavy"])
    ap.add_argument("--paraphrases", default="train", choices=["train", "eval", "all"])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-episodes", type=int, default=2, help="episodes per task to record")
    ap.add_argument("--viewer", action="store_true", help="live viewer; run with mjpython on macOS")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.backend == "openvino":
        if args.policy != "act":
            raise SystemExit("--backend openvino is only supported for ACT")
        args.device = "cpu"
    tasks = ["set_table"] if args.policy == "act" else args.tasks  # ACT has no language input: set_table only

    policy = get_policy_class(args.policy).from_pretrained(args.ckpt).to(args.device).eval()
    pre, post = make_pre_post_processors(policy.config, args.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    if args.backend == "openvino":
        policy = OpenVINOACT(args.ir, policy.config)
    infer_every = policy.config.n_action_steps  # one model call per chunk; the rest are queue pops

    env = DinnerEnv()
    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    tag = f"{args.policy}_{args.backend}_{args.rand}_{args.paraphrases}"
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = {t: "n/a" for t in TASKS}
    all_lat = []
    for ti, task in enumerate(tasks):
        texts = TASKS[task]["train"] + TASKS[task]["eval"] if args.paraphrases == "all" else TASKS[task][args.paraphrases]
        wins, times = 0, []
        for ep in range(args.episodes):
            text = str(rng.choice(texts))
            writer = None
            if args.video and ep < args.video_episodes:
                writer = imageio.get_writer(out / "videos" / f"{tag}_{task}_ep{ep}.mp4", fps=FPS, macro_block_size=1)
            ok, t, lat = run_episode(env, policy, pre, post, task, text, EVAL_SEED_START + 1000 * ti + ep,
                                     args.rand, infer_every, writer, viewer)
            if writer is not None:
                writer.close()
            all_lat.extend(lat[::infer_every])
            wins += ok
            if ok:
                times.append(t)
            print(f"{task} ep{ep} {'OK ' if ok else 'FAIL'} {text!r}", flush=True)
        results[task] = {"success_rate": wins / args.episodes, "successes": wins, "episodes": args.episodes,
                         "mean_time_to_success_s": float(np.mean(times)) if times else None}
        print(f"== {task}: {wins}/{args.episodes}", flush=True)

    summary = {
        "policy": args.policy, "backend": args.backend, "ckpt": str(args.ckpt), "rand": args.rand,
        "paraphrases": args.paraphrases, "device": args.device, "tasks": results,
        "p50_infer_ms": 1000 * float(np.median(all_lat)), "p95_infer_ms": 1000 * float(np.percentile(all_lat, 95)),
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
    }
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
