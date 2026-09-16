"""Hierarchical agent: language planner (LLM on OpenVINO) -> camera perception -> bimanual skills, closed-loop.

    python -m scripts.agent --command "Hand the fork from arm B to arm A and put it left of the plate." --video
    python -m scripts.agent --eval --episodes 10 --video                 # every task, held-out seeds, JSON report
    python -m scripts.agent --eval --paraphrases eval --rand heavy       # unseen wording, 1.5x randomization
    python -m scripts.agent --eval --disturb                             # knock a placed object off mid-task
    python -m scripts.agent --eval --preplaced                           # one object already in place at start

Loop per command:
1. Observe: the overhead camera gives object poses (dinner.perception); the planner gets a scene summary.
2. Understand: the LLM turns command + scene into steps; the scheduler groups them into bimanual phases.
3. Plan: before each phase the scene is re-perceived; steps already satisfied are skipped. If a step can't be
   planned (IK/collision), repair tries the other arm, then a hand-off over the relay pad in either direction.
4. Act: the phase's arms move together (mink IK + OMPL skills from dinner.expert, planned on perceived poses).
5. Verify: the camera checks every step; a failed or disturbed step is re-queued from where the object now is.

Success is judged by the simulator (dinner.env success check), never by the agent's own perception.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import mujoco
import imageio.v2 as imageio
import numpy as np

from dinner.env import FPS, HOME, PLACE_TOL, RELAY_YAW, TASKS, DinnerEnv, _yaw_quat, task_phases
from dinner.expert import Expert
from dinner.perception import Perception
from dinner.planner import MODEL_DIR, Planner

EVAL_SEED_START = 10000  # same held-out seeds as scripts.eval
MAX_RETRIES = 2  # per step
MAX_REPAIRS = 4  # per episode; a repair that planned once can fail to re-plan, so bound the loop
YAW_TOL = np.deg2rad(45)
TAIL_STEPS = 15
WHITE, GREY, GREEN, RED, CYAN = (255, 255, 255), (170, 170, 170), (90, 220, 90), (240, 80, 80), (0, 230, 255)
STATUS = {"done": GREEN, "running": CYAN, "failed": RED, "retry": (255, 170, 60), "replanned": (255, 170, 60)}
SITE_LABEL = {"fork_target": "left of plate", "spoon_target": "right of plate", "cup_target": "top right",
              "relay": "hand-off pad"}


def _ang(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def step_label(arm, obj, site):
    return f"{'A' if arm == 'left' else 'B'}({arm}): {obj} -> {SITE_LABEL[site]}"


class Agent:
    def __init__(self, env, planner, writer=None):
        self.env, self.planner, self.writer = env, planner, writer
        self.expert = Expert(env)
        self.per = Perception(env)
        self.video_cam = mujoco.Renderer(env.model, 512, 512) if writer is not None else None
        self.log = []
        self.plan_rows = []  # [label, status] for the overlay

    # ------------------------------------------------------------------ perception helpers
    def verified(self, scene, obj, site):
        o, t = scene["objects"].get(obj), scene["targets"].get(site)
        if o is None or t is None or np.linalg.norm(o["pos"][:2] - t[:2]) > PLACE_TOL or not o["upright"]:
            return False
        if obj in ("fork", "spoon"):
            want = RELAY_YAW if site == "relay" else self.env.target_yaw[obj]
            return _ang(o["yaw"], want) < YAW_TOL
        return True

    def perceived_world(self, scene):
        """Planning world: live arm joints (proprioception) + object poses from the camera."""
        env, w = self.env, self.env.data.qpos.copy()
        for name, o in scene["objects"].items():
            a = env.free_adr[name]
            w[a:a + 3] = o["pos"]
            yaw = self.env.target_yaw["fork"] - np.pi / 2 if name == "plate" else o["yaw"]
            w[a + 3:a + 7] = _yaw_quat(yaw)
        return w

    # ------------------------------------------------------------------ io
    def note(self, msg):
        self.log.append((round(self.env.data.time, 2), msg))
        print(f"  [{self.env.data.time:5.1f}s] {msg}", flush=True)

    def frame(self, scene=None, banner=None):
        if self.writer is None:
            return
        self.video_cam.update_scene(self.env.data, camera="front")
        front = self.video_cam.render().copy()
        top = self.per.render_rgb().copy()
        if scene is not None:
            self.last_scene = scene
        sc = getattr(self, "last_scene", None)
        if sc is not None:  # perceived poses and targets drawn on the overhead view
            for site, t in sc["targets"].items():
                cv2.circle(top, self.per.project(t), 11 if site != "relay" else 8, (60, 220, 60), 1)
            for name, o in sc["objects"].items():
                p = self.per.project(o["pos"])
                cv2.circle(top, p, 4, CYAN, -1)
                if name in ("fork", "spoon"):
                    q = self.per.project(o["pos"] + 0.05 * np.r_[np.cos(o["yaw"]), np.sin(o["yaw"]), 0])
                    cv2.arrowedLine(top, p, q, CYAN, 2, tipLength=0.3)
                _txt(top, name, (p[0] + 8, p[1] - 8), 0.4, CYAN)
        top = cv2.resize(top, (512, 512))
        _txt(top, "overhead camera: perceived poses (cyan), targets (green)", (8, 500), 0.42)
        _txt(front, "front camera", (8, 500), 0.42)
        panel = np.full((210, 1024, 3), 20, np.uint8)
        _txt(panel, f'Command: "{self.command}"', (12, 26), 0.58, WHITE)
        _txt(panel, self.header, (12, 52), 0.45, GREY)
        _txt(panel, "Plan (LLM on OpenVINO -> bimanual phases)", (12, 80), 0.45, GREY)
        _txt(panel, "Agent log (camera checks, repairs)", (520, 80), 0.45, GREY)
        for i, (label, status) in enumerate(self.plan_rows[:5]):
            _txt(panel, f"{i + 1}. {label}  [{status}]", (12, 106 + 22 * i), 0.47, STATUS.get(status, GREY))
        for i, (t, msg) in enumerate(self.log[-5:]):
            _txt(panel, f"{t:5.1f}s {msg}"[:58], (520, 106 + 22 * i), 0.43, (230, 220, 150))
        img = np.vstack([np.hstack([front, top]), panel])
        if banner is not None:
            _txt(img, banner, (20, 470), 1.2, GREEN if banner.startswith("SUCCESS") else RED, 2)
        self.writer.append_data(img)

    def execute(self, traj):
        for a in traj:
            self.env.step(a)
            self.frame()

    # ------------------------------------------------------------------ planning with repair
    def plan_phase(self, phase, last, scene):
        self.expert.scene, self.expert.world = scene, self.perceived_world(scene)
        try:
            return self.expert.plan_phase(phase, last)
        finally:
            self.expert.scene = self.expert.world = None

    def repair(self, arm, obj, site, last, scene):
        """Alternatives for a single step whose plan failed: other arm, then relay hand-offs."""
        other = "right" if arm == "left" else "left"
        options = [[{other: (obj, site)}]]
        if site != "relay":
            options += [[{arm: (obj, "relay")}, {other: (obj, site)}], [{other: (obj, "relay")}, {arm: (obj, site)}]]
        at_pad = {**scene, "objects": {**scene["objects"], obj: {
            **scene["objects"][obj], "pos": np.r_[scene["targets"]["relay"][:2], scene["objects"][obj]["pos"][2]],
            "yaw": RELAY_YAW}}}
        for opt in options:  # a hand-off only helps if the receiving arm can finish from the pad
            if self.plan_phase(opt[0], last, scene) is not None and \
                    (len(opt) == 1 or self.plan_phase(opt[1], last, at_pad) is not None):
                return opt
        return None

    # ------------------------------------------------------------------ main loop
    def run(self, command, seed, level="nominal", layout_task=None, disturb=False, preplaced=None):
        env = self.env
        env.reset(seed, level, layout_task)
        self.start_pos = {o: env.data.xpos[b].copy() for o, b in env.body_id.items()}
        self.command, self.log, self.plan_rows, self.executed = command, [], [], []
        self.header = f"seed {seed} | randomization {level}"
        if preplaced:  # scene starts with this object already at its target
            obj, site = preplaced
            t = env.data.site_xpos[env.model.site(site).id]
            env.set_free(obj, [t[0], t[1], env.model.qpos0[env.free_adr[obj] + 2]], _yaw_quat(env.target_yaw.get(obj, 0)))
            for _ in range(10):
                env.step(np.tile(HOME, 2))
        scene = self.per.observe()
        done = [o for o, s in (("fork", "fork_target"), ("spoon", "spoon_target"), ("cup", "cup_target"))
                if self.verified(scene, o, s)]
        for _ in range(FPS):
            self.frame(scene)

        n_gen = len(self.planner.latency_s)
        steps, phases, notes = self.planner.plan(command, scene, done)
        lat = sum(self.planner.latency_s[n_gen:]) or None
        for n in notes:
            self.note(n)
        self.note(f"plan: {len(steps)} steps, {len(phases)} phases" + (f" ({lat:.1f}s LLM)" if lat else " (cached)"))
        rows = {}
        for ph in phases:
            for arm, (obj, site) in ph.items():
                rows[(arm, obj, site)] = len(self.plan_rows)
                self.plan_rows.append([step_label(arm, obj, site), "pending"])
        goals = [(o, s) for ph in phases for _, (o, s) in ph.items()]
        goals = [(o, s) for i, (o, s) in enumerate(goals) if s != "relay" and all(o2 != o for o2, _ in goals[i + 1:])]

        queue = [dict(ph) for ph in phases]
        retries, repairs = {}, 0
        last = np.tile(HOME, 2)
        disturbed = not disturb
        while queue:
            phase = queue.pop(0)
            scene = self.per.observe()
            for arm, (obj, site) in list(phase.items()):
                if self.verified(scene, obj, site):
                    self.note(f"{obj} already at {SITE_LABEL[site]}: skip")
                    self._status(rows, arm, obj, site, "done")
                    del phase[arm]
            for arm, (obj, site) in list(phase.items()):
                if obj not in scene["objects"]:
                    self.note(f"camera: {obj} not visible; can't plan it")
                    self._status(rows, arm, obj, site, "failed")
                    del phase[arm]
            if not phase:
                continue
            traj = self.plan_phase(phase, last, scene)
            if traj is None and len(phase) > 1:  # split a failed parallel phase into sequential steps
                self.note(f"parallel plan failed ({self.expert.fail}); running sequentially")
                queue[:0] = [{a: s} for a, s in phase.items()]
                continue
            if traj is None:
                (arm, (obj, site)), = phase.items()
                repairs += 1
                self.note(f"cannot plan {obj} with {arm} arm ({self.expert.fail}); repairing")
                fix = self.repair(arm, obj, site, last, scene) if repairs <= MAX_REPAIRS else None
                if fix is None:
                    self.note(f"no feasible repair for {obj}")
                    self._status(rows, arm, obj, site, "failed")
                    continue
                self.note("repair: " + " then ".join(step_label(a, *st) for p in fix for a, st in p.items()))
                self._status(rows, arm, obj, site, "replanned")
                for p in fix:
                    (a2, (o2, s2)), = p.items()
                    rows[(a2, o2, s2)] = len(self.plan_rows)
                    self.plan_rows.append([step_label(a2, o2, s2), "pending"])
                queue[:0] = fix
                continue
            for arm, (obj, site) in phase.items():
                self._status(rows, arm, obj, site, "running")
            self.execute(traj)
            last = traj[-1]
            if not disturbed and any(s != "relay" for _, s in phase.values()):
                disturbed = True
                obj = next(o for o, s in phase.values() if s != "relay")
                self.disturb(obj, seed)
            scene = self.per.observe()
            self.frame(scene)
            # Earlier placements can be undone by later motion: re-check them with this phase's steps.
            knocked = [(a, o, st) for a, o, st in self.executed if st != "relay"
                       and all(o != o2 for o2, _ in phase.values()) and not self.verified(scene, o, st)]
            for a, o, st in knocked:
                self.executed.remove((a, o, st))
                self.note(f"camera: {o} was knocked off {SITE_LABEL[st]}")
            for arm, obj, site in [(a, o, st) for a, (o, st) in phase.items()] + knocked:
                if self.verified(scene, obj, site):
                    self._status(rows, arm, obj, site, "done")
                    self.executed.append((arm, obj, site))
                    continue
                retries[(obj, site)] = retries.get((obj, site), 0) + 1
                if not scene["objects"].get(obj, {"upright": True})["upright"]:
                    self.note(f"camera: {obj} tipped over; top-down grasp can't recover it")
                    self._status(rows, arm, obj, site, "failed")
                    continue
                if retries[(obj, site)] > MAX_RETRIES:
                    self.note(f"{obj} not at {SITE_LABEL[site]} after {MAX_RETRIES} retries; giving up")
                    self._status(rows, arm, obj, site, "failed")
                    continue
                self.note(f"camera: {obj} not at {SITE_LABEL[site]}; re-planning from its new pose")
                self._status(rows, arm, obj, site, "retry")
                queue.insert(0, {arm: (obj, site)})
        for _ in range(TAIL_STEPS):
            env.step(last)
            self.frame()
        ok = all(env._place_ok(o, s) for o, s in goals) and all(env.released(a) for a, _, _ in self.executed)
        placed = {o: bool(env._place_ok(o, s)) for o, s in goals}
        banner = f"SUCCESS in {env.data.time:.1f}s" if ok else \
            "FAIL (placed: " + (", ".join(o for o, p in placed.items() if p) or "none") + ")"
        for _ in range(FPS * 2):
            self.frame(banner=banner)
        return {"success": bool(ok), "placed": placed, "executed": self.executed, "goals": goals, "steps": steps,
                "phases": [{a: list(s) for a, s in ph.items()} for ph in phases],
                "log": self.log, "sim_time_s": float(env.data.time), "planner_latency_s": lat}

    def _status(self, rows, arm, obj, site, status):
        if (arm, obj, site) in rows:
            self.plan_rows[rows[(arm, obj, site)]][1] = status

    def disturb(self, obj, seed):
        """Someone knocks the just-placed object back toward where it started."""
        env = self.env
        rng = np.random.default_rng(seed + 7)
        pos = env.data.xpos[env.body_id[obj]].copy()
        start = self.start_pos[obj]
        new = start[:2] + rng.uniform(-0.02, 0.02, 2)
        yaw = float(np.arctan2(env.data.xmat[env.body_id[obj]][3], env.data.xmat[env.body_id[obj]][0])) \
            + rng.uniform(-0.3, 0.3)
        env.set_free(obj, [new[0], new[1], pos[2]], _yaw_quat(yaw) if obj != "cup" else [1, 0, 0, 0])
        self.note(f"DISTURBANCE: {obj} pushed off its target")
        for _ in range(10):
            env.step(env.data.ctrl[self.env.act_ids].copy())
            self.frame()


def _txt(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    """Text on a dark box (an outline stroke would be wider than the text and ghost). Colors are RGB."""
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + base + 1), (20, 20, 20), -1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


EXTRA_COMMANDS = [  # free-form commands beyond the task paraphrases, run with --eval --extra
    ("Hand the fork from arm B to arm A and place it left of the plate.", "handoff_fork"),
    ("Arm B, put the cup at the top right, and arm A, put the fork left of the plate.", None),
    ("Set the table, then put the cup at the top right.", None),
    ("Pass the spoon to arm A and have it place the spoon left of the plate.", "set_table"),
    ("Arm A, put the fork left of the plate. Arm B, put the spoon on the right, then the cup at the top right.", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--command")
    ap.add_argument("--layout", default=None, choices=list(TASKS), help="scene layout (handoff_fork swaps cutlery)")
    ap.add_argument("--seed", type=int, default=EVAL_SEED_START)
    ap.add_argument("--eval", action="store_true", help="every task x --episodes held-out seeds")
    ap.add_argument("--extra", action="store_true", help="with --eval: EXTRA_COMMANDS instead of the task set")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--rand", default="nominal", choices=["nominal", "heavy"])
    ap.add_argument("--paraphrases", default="train", choices=["train", "eval"])
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--preplaced", action="store_true", help="first goal object starts already in place")
    ap.add_argument("--planner-device", default="CPU")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-episodes", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    env = DinnerEnv()
    planner = Planner(device=args.planner_device)
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    tag = "agent_" + args.rand + "_" + args.paraphrases + ("_disturb" if args.disturb else "") \
        + ("_preplaced" if args.preplaced else "") + ("_extra" if args.extra else "")

    if args.eval:
        rng = np.random.default_rng(0)
        if args.extra:
            jobs = [(f"extra{i}", cmd, layout) for i, (cmd, layout) in enumerate(EXTRA_COMMANDS)]
        else:
            jobs = [(t, None, t) for t in args.tasks]
    else:
        jobs = [("command", args.command, args.layout)]
        args.episodes = 1

    results, wall = {}, []
    for ti, (name, cmd, layout) in enumerate(jobs):
        wins, eps = 0, []
        for ep in range(args.episodes):
            text = cmd or str(rng.choice(TASKS[name][args.paraphrases]))
            if not args.eval:
                seed = args.seed
            elif args.extra:
                seed = EVAL_SEED_START + 7000 + 100 * ti + ep
            else:  # the seeds scripts.eval uses for this task
                seed = EVAL_SEED_START + 1000 * list(TASKS).index(name) + ep
            pre = None  # --preplaced: multi-object tasks start with their first object already in place
            if args.preplaced and name in TASKS and len(TASKS[name]["place"]) > 1:
                pre = TASKS[name]["place"][0][:2]
            writer = None
            if args.video and ep < args.video_episodes:
                writer = imageio.get_writer(out / "videos" / f"{tag}_{name}_ep{ep}.mp4", fps=FPS, macro_block_size=1)
            agent = Agent(env, planner, writer)
            print(f"{name} ep{ep} seed {seed}: {text!r}", flush=True)
            tic = time.perf_counter()
            r = agent.run(text, seed, args.rand, layout, args.disturb, pre)
            wall.append(time.perf_counter() - tic)
            if writer is not None:
                writer.close()
            if name in TASKS:  # judge the task's own success check, not the agent's plan
                want = {o for o, _, _ in TASKS[name]["place"]}
                r["plan_success"] = r["success"]
                r["extra_objects"] = sorted({o for o, _ in r["goals"]} - want)
                # env.success also wants every task arm's gripper open; an arm the agent never used (a pre-placed
                # object, or a repair that switched arms) still rests at HOME with the jaw shut, so check the
                # arms that acted instead.
                used = {a for a, _, _ in r["executed"]}
                r["success"] = (all(env._place_ok(o, st) for o, st, _ in TASKS[name]["place"])
                                and all(env.released(a) for a in used) and not r["extra_objects"])
                if name == "handoff_fork":  # the instruction is the hand-off itself: both arm roles must happen
                    r["success"] = r["success"] and all(
                        (a, o, st) in r["executed"] for ph in task_phases(name) for a, (o, st) in ph.items())
            wins += r["success"]
            eps.append({"seed": seed, "command": text, "preplaced": pre, **r})
            print(f"== {name} ep{ep}: {'OK' if r['success'] else 'FAIL'} placed={r['placed']} "
                  f"extra={r.get('extra_objects')} wall={wall[-1]:.0f}s", flush=True)
        results[name] = {"successes": wins, "episodes": args.episodes, "success_rate": wins / args.episodes,
                         "episodes_detail": eps}
        print(f"==== {name}: {wins}/{args.episodes}", flush=True)

    summary = {"policy": "agent (LLM planner + perception + mink/OMPL skills)", "planner_model": MODEL_DIR.name,
               "planner_device": args.planner_device, "rand": args.rand, "paraphrases": args.paraphrases,
               "disturb": args.disturb, "preplaced": args.preplaced,
               "tasks": {k: {kk: vv for kk, vv in v.items() if kk != "episodes_detail"} for k, v in results.items()},
               "planner_latency_s_uncached": planner.latency_s, "episode_wall_s_mean": float(np.mean(wall)),
               "machine": {"platform": platform.platform(), "processor": platform.processor()},
               "episodes": {k: v["episodes_detail"] for k, v in results.items()}}
    if args.eval:
        (out / f"{tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["tasks"], indent=1))


if __name__ == "__main__":
    main()
