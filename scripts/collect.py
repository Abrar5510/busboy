"""Collect expert demonstrations into LeRobot v3 datasets.

One pass, two datasets: every successful episode goes to data/dinner_table (SmolVLA, all tasks);
set_table episodes are also written to data/dinner_set_table (ACT baseline). No duplicate simulation.

    python scripts/collect.py --episodes-per-task 50
    python scripts/collect.py --episodes-per-task 3 --overwrite     # smoke
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dinner.env import ARMS, CAMERAS, FPS, IMG_HW, JOINTS, TASKS, DinnerEnv, task_text
from dinner.expert import Expert, rollout

MAX_ATTEMPTS = 3  # per seed; then skip seed and move on
TIME_BUDGET_H = 4.0  # stop collecting when exceeded, train on what you have
EVAL_SEED_START = 10000  # collection seeds must stay below; eval uses >= this
NAMES = [f"{a}_{j}" for a in ARMS for j in JOINTS]


def features():
    f = {f"observation.images.{k}": {"dtype": "video", "shape": (*IMG_HW, 3), "names": ["height", "width", "channels"]}
         for k in CAMERAS}
    f["observation.state"] = {"dtype": "float32", "shape": (len(NAMES),), "names": NAMES}
    f["action"] = {"dtype": "float32", "shape": (len(NAMES),), "names": NAMES}
    return f


def create(repo_id, root, overwrite):
    root = Path(root)
    if root.exists():
        if not overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    return LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features(), root=root,
                                 robot_type="so101_dual", use_videos=True, image_writer_threads=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-task", type=int, default=50)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--level", default="nominal")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--root", default="data")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = create("local/dinner_table", Path(args.root) / "dinner_table", args.overwrite)
    ds_act = create("local/dinner_set_table", Path(args.root) / "dinner_set_table", args.overwrite) \
        if "set_table" in args.tasks else None

    env = DinnerEnv()
    expert = Expert(env)
    rng = np.random.default_rng(args.seed_start)
    counts = {t: 0 for t in args.tasks}
    target = args.episodes_per_task * len(args.tasks)
    seed, attempts, t0 = args.seed_start, 0, time.time()
    try:
        # Round-robin over tasks so a budget stop still leaves a balanced dataset.
        while any(c < args.episodes_per_task for c in counts.values()):
            for task in args.tasks:
                if counts[task] >= args.episodes_per_task:
                    continue
                hours = (time.time() - t0) / 3600
                if hours > TIME_BUDGET_H:
                    print(f"time budget {TIME_BUDGET_H} h exceeded; stopping with {counts}")
                    return
                assert seed < EVAL_SEED_START, "ran into eval seed range"
                for _ in range(MAX_ATTEMPTS):
                    attempts += 1
                    buf = []
                    ok = rollout(env, expert, task, seed, args.level, on_step=lambda o, a, buf=buf: buf.append((o, a)))
                    if ok:
                        break
                seed += 1
                if not ok:
                    continue
                text = task_text(task, "train", rng)
                for obs, action in buf:
                    frame = {**obs, "action": action.astype(np.float32), "task": text}
                    ds.add_frame(frame)
                    if ds_act is not None and task == "set_table":
                        ds_act.add_frame(frame)
                ds.save_episode()
                if ds_act is not None and task == "set_table":
                    ds_act.save_episode()
                counts[task] += 1
                done = sum(counts.values())
                elapsed = time.time() - t0
                print(f"[{done}/{target}] {task} seed={seed - 1} frames={len(buf)} attempts={attempts} "
                      f"{done / elapsed * 3600:.0f} eps/h, projected total {elapsed / done * target / 3600:.2f} h",
                      flush=True)
    finally:
        ds.finalize()
        if ds_act is not None:
            ds_act.finalize()
        print("episodes:", counts, "seeds used:", seed - args.seed_start, "attempts:", attempts)


if __name__ == "__main__":
    main()
