"""Closed-loop evaluation of ACT / SmolVLA checkpoints in the dinner-table MuJoCo env.

    python -m scripts.eval --policy smolvla --ckpt outputs/train/smolvla_dinner/checkpoints/last/pretrained_model
    python -m scripts.eval --policy smolvla --ckpt ... --paraphrases eval          # held-out instruction wording
    python -m scripts.eval --policy smolvla --ckpt ... --rand heavy                # 1.5x randomization
    python -m scripts.eval --policy act --ckpt ... --act-temporal-ensemble 0.01    # ACT with temporal ensembling
    python -m scripts.eval --policy act --ckpt ... --backend openvino --ov-device GPU   # CPU | GPU | NPU
    mjpython -m scripts.eval ... --viewer                                          # live MuJoCo viewer (macOS)

Writes results/{policy}_{backend}_{rand}_{paraphrases}.json (task success plus per-object placement counts) and,
with --video, mp4s under results/videos/.
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
    """ACT action-chunk IR behind the policy's select_action/reset interface, with optional temporal
    ensembling (same weighting as lerobot's ACTTemporalEnsembler). Input order must match
    scripts/export_openvino.py: state, then config.image_features order."""

    def __init__(self, ir_path, config, device="CPU", ensemble_coeff=None):
        import openvino as ov

        # f32 so closed-loop actions match torch (CPU default is f16 on ARM, bf16 on AMX). NPU runs f16 only.
        cfg = {} if device.startswith("NPU") else {"INFERENCE_PRECISION_HINT": "f32"}
        self.model = ov.Core().compile_model(str(ir_path), device, cfg)
        self.config = config
        self.image_keys = list(config.image_features)
        self.coeff = ensemble_coeff
        self.reset()

    def reset(self):
        self.queue = []
        self.chunks = []  # (start_step, chunk) for temporal ensembling
        self.t = 0

    def _infer(self, batch):
        inputs = [batch["observation.state"].cpu().numpy()] + [batch[k].cpu().numpy() for k in self.image_keys]
        return self.model(inputs)[0]  # (1, chunk_size, action_dim)

    def select_action(self, batch):
        if self.coeff is None:
            if not self.queue:
                chunk = self._infer(batch)
                self.queue = [torch.from_numpy(chunk[:, i]) for i in range(self.config.n_action_steps)]
            return self.queue.pop(0)
        # Temporal ensembling: predict every step, average all chunks covering this step, older weighted more.
        self.chunks.append((self.t, self._infer(batch)[0]))
        self.chunks = [(s, c) for s, c in self.chunks if self.t - s < len(c)]
        preds = np.stack([c[self.t - s] for s, c in self.chunks])  # oldest first
        w = np.exp(-self.coeff * np.arange(len(preds)))
        self.t += 1
        return torch.from_numpy((w[:, None] * preds).sum(0) / w.sum())[None]


def _text(frame, s, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def overlay(obs, text, p50_ms, t, seed, level, banner=None):
    """Front + both wrist cameras, with the command, latency, seed and (at the end) the outcome."""
    frame = np.hstack([obs[f"observation.images.{k}"] for k in CAMERAS])
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    _text(frame, f"{text} | p50 {p50_ms:.0f} ms | t={t:.1f}s", (10, 24))
    _text(frame, f"seed {seed} | randomization: {level}", (10, 50))
    if banner is not None:
        ok = banner.startswith("SUCCESS")
        _text(frame, banner, (10, frame.shape[0] - 24), 1.4, (80, 220, 80) if ok else (80, 80, 240))
    return frame


def run_episode(env, policy, pre, post, task, text, seed, level, infer_every, writer=None, viewer=None):
    """Returns (task success, time to success, per-call latencies, {object: placed at episode end})."""
    obs = env.reset(seed, level, task)
    policy.reset()
    lat, ok, t_done = [], False, None
    for step in range(int(TASKS[task].get("timeout_s", TIMEOUT_S) * FPS)):
        batch = pre(to_batch(obs, text))
        tic = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(batch)
        if action.device.type == "mps":
            torch.mps.synchronize()
        elif action.device.type == "xpu":  # Intel Arc iGPU via PyTorch XPU
            torch.xpu.synchronize()
        lat.append(time.perf_counter() - tic)
        action = post(action)
        env.step(action.squeeze(0).cpu().numpy())
        obs = env.obs()
        if viewer is not None:
            viewer.sync()
        if writer is not None:
            writer.append_data(overlay(obs, text, 1000 * float(np.median(lat[::infer_every])), (step + 1) / FPS,
                                       seed, level))
        if env.success(task):
            ok, t_done = True, (step + 1) / FPS
            break
    placed = {o: bool(env._place_ok(o, site)) for o, site, _ in TASKS[task]["place"]}
    if writer is not None:  # hold the final state for a second with the outcome
        banner = f"SUCCESS in {t_done:.1f}s" if ok else \
            "FAIL (placed: " + (", ".join(o for o, p in placed.items() if p) or "none") + ")"
        final = overlay(obs, text, 1000 * float(np.median(lat[::infer_every])), (step + 1) / FPS, seed, level, banner)
        for _ in range(FPS):
            writer.append_data(final)
    return ok, t_done, lat, placed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["act", "smolvla"])
    ap.add_argument("--ckpt", required=True, help="pretrained_model dir or HF repo id")
    ap.add_argument("--backend", default="torch", choices=["torch", "openvino"])
    ap.add_argument("--ir", default="results/act_dinner.xml", help="OpenVINO IR for --backend openvino")
    ap.add_argument("--ov-device", default="CPU", help="OpenVINO device for --backend openvino: CPU, GPU, NPU")
    ap.add_argument("--act-temporal-ensemble", type=float, default=None,
                    help="ACT: predict every step and exponentially average overlapping chunks (e.g. 0.01)")
    ap.add_argument("--episodes", type=int, default=10, help="per task")
    ap.add_argument("--rand", default="nominal", choices=["nominal", "heavy"])
    ap.add_argument("--paraphrases", default="train", choices=["train", "eval", "all"])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-episodes", type=int, default=2, help="episodes per task to record")
    ap.add_argument("--viewer", action="store_true", help="live viewer; run with mjpython on macOS")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    te = args.policy == "act" and args.act_temporal_ensemble is not None
    if args.backend == "openvino":
        if args.policy != "act":
            raise SystemExit("--backend openvino is only supported for ACT")
        args.device = "cpu"  # pre/post processors run on CPU around the IR
    tasks = ["set_table"] if args.policy == "act" else args.tasks  # ACT has no language input: set_table only

    policy_cls = get_policy_class(args.policy)
    if te and args.backend == "torch":
        from lerobot.configs.policies import PreTrainedConfig

        cfg = PreTrainedConfig.from_pretrained(args.ckpt)
        cfg.temporal_ensemble_coeff, cfg.n_action_steps = args.act_temporal_ensemble, 1
        policy = policy_cls.from_pretrained(args.ckpt, config=cfg).to(args.device).eval()
    else:
        policy = policy_cls.from_pretrained(args.ckpt).to(args.device).eval()
    pre, post = make_pre_post_processors(policy.config, args.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    if args.backend == "openvino":
        policy = OpenVINOACT(args.ir, policy.config, args.ov_device, args.act_temporal_ensemble if te else None)
    infer_every = 1 if te else policy.config.n_action_steps  # one model call per chunk; the rest are queue pops

    env = DinnerEnv()
    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    backend = args.backend if args.backend == "torch" else \
        f"openvino-{args.ov_device.lower()}" + ("-int8" if "int8" in Path(args.ir).stem else "")
    backend += "-te" if te else ""
    tag = f"{args.policy}_{backend}_{args.rand}_{args.paraphrases}"
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = {t: "n/a" for t in TASKS}
    all_lat = []
    for ti, task in enumerate(tasks):
        texts = TASKS[task]["train"] + TASKS[task]["eval"] if args.paraphrases == "all" else TASKS[task][args.paraphrases]
        wins, times, placed_counts = 0, [], {o: 0 for o, _, _ in TASKS[task]["place"]}
        for ep in range(args.episodes):
            text = str(rng.choice(texts))
            writer = None
            if args.video and ep < args.video_episodes:
                writer = imageio.get_writer(out / "videos" / f"{tag}_{task}_ep{ep}.mp4", fps=FPS, macro_block_size=1)
            ok, t, lat, placed = run_episode(env, policy, pre, post, task, text, EVAL_SEED_START + 1000 * ti + ep,
                                             args.rand, infer_every, writer, viewer)
            if writer is not None:
                writer.close()
            all_lat.extend(lat[::infer_every])
            wins += ok
            if ok:
                times.append(t)
            for o, p in placed.items():
                placed_counts[o] += p
            print(f"{task} ep{ep} {'OK ' if ok else 'FAIL'} placed={[o for o, p in placed.items() if p]} {text!r}", flush=True)
        results[task] = {"success_rate": wins / args.episodes, "successes": wins, "episodes": args.episodes,
                         "mean_time_to_success_s": float(np.mean(times)) if times else None,
                         "object_placed": placed_counts}
        print(f"== {task}: {wins}/{args.episodes} objects placed {placed_counts}", flush=True)

    summary = {
        "policy": args.policy, "backend": backend, "ckpt": str(args.ckpt), "rand": args.rand,
        "paraphrases": args.paraphrases, "device": args.device if args.backend == "torch" else args.ov_device,
        "act_temporal_ensemble": args.act_temporal_ensemble if te else None,
        "tasks": results,
        "p50_infer_ms": 1000 * float(np.median(all_lat)), "p95_infer_ms": 1000 * float(np.percentile(all_lat, 95)),
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
    }
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
