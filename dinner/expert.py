"""Scripted expert: mink IK for grasp/place poses, OMPL RRTConnect for joint-space transits.

Per pick-place: home -> pregrasp (OMPL) -> grasp (linear) -> close -> lift (linear)
-> above target (OMPL, carrying) -> place (linear) -> open -> retreat (linear) -> home (OMPL).
Dual-arm tasks: each arm is planned with the other held at its start pose, then both
trajectories are padded to equal length and executed simultaneously; arm-arm conflicts
are caught by the rollout + success filter.

Run `python -m dinner.expert --seeds 20` for per-task expert success.
"""

import argparse
import time

import mink
import mujoco
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from dinner.env import ARMS, GRIP_CLOSED, GRIP_OPEN, HOME, OBJECTS, TASKS, DinnerEnv

ou.setLogLevel(ou.LogLevel.LOG_WARN)

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
DT = 1 / 30
MAX_VEL = 1.5  # rad/s, OMPL transits
FINE_VEL = 0.5  # rad/s, approach / lift / lower / retreat
GRIP_STEPS = 10  # control steps to close or open
PLAN_TIME = 2.0  # s per OMPL query

# Grasp/place heights of gripperframe (fingertip) relative to object body / target site.
PREGRASP_DZ = 0.035  # clears cup rim (tip 1 cm above) and handles; lower = more reach near and far
GRASP_DZ = {"fork": 0.002, "spoon": 0.002, "cup": 0.015}
PLACE_DZ = {"fork": 0.015, "spoon": 0.015, "cup": 0.06}
POS_TOL = 0.004
ORI_TOL = {"fork": 0.05, "spoon": 0.05, "cup": 0.6}  # cylinder tolerates a tilted approach
ORI_COST = {"fork": 0.3, "spoon": 0.3, "cup": 0.05}
IK_ITERS = 400
# IK seeds relative to the pan toward the target: reaching, extended, folded (near base), roll variants.
IK_SEEDS = ([-0.5, 0.5, 1.0, 0.0], [0.0, 1.0, 0.5, 0.0], [-1.2, 1.5, 1.2, 0.0],
            [-1.0, 1.4, 1.0, 1.5], [-1.0, 1.4, 1.0, -1.5])


def _grasp_rot(close_dir):
    """Top-down: site x (approach) -> world -z, site z (jaw closing axis) -> close_dir."""
    x = np.array([0.0, 0.0, -1.0])
    z = np.array([close_dir[0], close_dir[1], 0.0])
    z /= np.linalg.norm(z)
    return mink.SO3.from_matrix(np.column_stack([x, np.cross(z, x), z]))


def _across(yaw):
    """Horizontal direction perpendicular to a long axis at `yaw`."""
    return np.array([-np.sin(yaw), np.cos(yaw)])


def _arm_collision(model, data, arm_geoms, forbidden_ids):
    for c in data.contact[:data.ncon]:
        g1, g2 = c.geom1, c.geom2
        if g1 in arm_geoms and (g2 in forbidden_ids or g2 in arm_geoms):
            return True
        if g2 in arm_geoms and (g1 in forbidden_ids or g1 in arm_geoms):
            return True
    return False


def make_is_valid(model, data, arm_qadr, arm_geoms, forbidden_ids):
    n = len(arm_qadr)

    def is_valid(state):
        saved = data.qpos.copy()
        data.qpos[arm_qadr] = [state[i] for i in range(n)]  # nanobind State is indexable, not sliceable
        mujoco.mj_forward(model, data)
        ok = not _arm_collision(model, data, arm_geoms, forbidden_ids)
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        return ok

    return is_valid


def _interp(waypoints, grip_from, grip_to, vel):
    """Resample a joint path at 30 Hz under a max joint speed; gripper ramps across the whole segment."""
    pts = np.asarray(waypoints, dtype=float)
    steps = [max(1, int(np.ceil(np.abs(b - a).max() / (vel * DT)))) for a, b in zip(pts[:-1], pts[1:])]
    total, k, out = sum(steps), 0, []
    for (a, b), n in zip(zip(pts[:-1], pts[1:]), steps):
        for i in range(1, n + 1):
            k += 1
            out.append(np.r_[a + (b - a) * i / n, grip_from + (grip_to - grip_from) * k / total])
    return out


def _hold(q, grip_from, grip_to, n=GRIP_STEPS):
    return [np.r_[q, grip_from + (grip_to - grip_from) * i / n] for i in range(1, n + 1)]


class Expert:
    def __init__(self, env):
        self.env = env
        m = env.model
        self.scratch = mujoco.MjData(m)
        self.cfg = mink.Configuration(m)
        jid = lambda a, j: m.joint(f"{a}_{j}").id
        self.arm_q = {a: np.array([m.jnt_qposadr[jid(a, j)] for j in ARM_JOINTS]) for a in ARMS}
        self.grip_q = {a: m.jnt_qposadr[jid(a, "gripper")] for a in ARMS}
        self.ranges = {a: m.jnt_range[[jid(a, j) for j in ARM_JOINTS]].copy() for a in ARMS}
        arm_dofs = {a: {m.jnt_dofadr[jid(a, j)] for j in ARM_JOINTS} for a in ARMS}
        self.freeze = {a: mink.DofFreezingTask(model=m, dof_indices=[i for i in range(m.nv) if i not in arm_dofs[a]])
                       for a in ARMS}
        self.limits = [mink.ConfigurationLimit(m)]
        geoms_of = lambda pred: {g for g in range(m.ngeom) if pred(m.body(m.geom_bodyid[g]).name)}
        self.arm_geoms = {a: geoms_of(lambda n, a=a: n.startswith(a + "_")) for a in ARMS}
        self.obj_geoms = {o: geoms_of(lambda n, o=o: n == o) for o in OBJECTS}
        self.table_geoms = {m.geom("table_top").id}
        self.fail = None  # reason the last plan_task returned None

    def world_valid(self, arm, held=None, grip=GRIP_OPEN):
        """Validity checker for `arm` against the world at env.data: table, the other arm (at its
        current pose), itself, and every object except `held`."""
        d = self.scratch
        d.qpos[:] = self.env.data.qpos
        d.qpos[self.grip_q[arm]] = grip
        other = [a for a in ARMS if a != arm][0]
        forbidden = set(self.table_geoms) | self.arm_geoms[other]
        for o, gs in self.obj_geoms.items():
            if o != held:
                forbidden |= gs
        return make_is_valid(self.env.model, d, self.arm_q[arm], self.arm_geoms[arm], forbidden)

    # ------------------------------------------------------------------ IK
    def ik(self, arm, pos, close_dir, obj, prev=None, valid=None):
        """First collision-free IK solution within tolerance over the seed set, else None."""
        d = self.env.data
        base = d.xpos[self.env.model.body(f"{arm}_base").id]
        pan = np.arctan2(pos[0] - base[0], pos[1] - base[1])
        seeds = ([prev] if prev is not None else []) + [np.r_[pan, s] for s in IK_SEEDS]
        q = d.qpos.copy()
        q[self.grip_q[arm]] = GRIP_OPEN
        task = mink.FrameTask(f"{arm}_gripperframe", "site", position_cost=1.0,
                              orientation_cost=ORI_COST[obj], lm_damping=1.0)
        task.set_target(mink.SE3.from_rotation_and_translation(_grasp_rot(close_dir), np.asarray(pos)))
        for sd in seeds:
            q[self.arm_q[arm]] = sd
            self.cfg.update(q)
            for it in range(IK_ITERS):
                v = mink.solve_ik(self.cfg, [task], 0.02, "daqp", damping=1e-3,
                                  limits=self.limits, constraints=[self.freeze[arm]])
                self.cfg.integrate_inplace(v, 0.02)
                if it % 50 == 49:
                    e = task.compute_error(self.cfg)
                    if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                        break
            e = task.compute_error(self.cfg)
            sol = self.cfg.q[self.arm_q[arm]].copy()
            if np.linalg.norm(e[:3]) < POS_TOL and np.linalg.norm(e[3:]) < ORI_TOL[obj] and (valid is None or valid(sol)):
                return sol
        return None

    # ------------------------------------------------------------------ OMPL
    def plan(self, arm, q_from, q_to, held=None):
        """Joint-space RRTConnect for one arm; world as at env.data (other arm at its current pose)."""
        is_valid = self.world_valid(arm, held, GRIP_CLOSED if held else GRIP_OPEN)
        if not (is_valid(q_from) and is_valid(q_to)):
            return None
        space = ob.RealVectorStateSpace(5)
        bounds = ob.RealVectorBounds(5)
        for i, (lo, hi) in enumerate(self.ranges[arm]):
            bounds.setLow(i, lo)
            bounds.setHigh(i, hi)
        space.setBounds(bounds)
        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(is_valid)
        si.setStateValidityCheckingResolution(0.005)
        si.setup()
        start, goal = si.allocState(), si.allocState()
        for i in range(5):
            start[i], goal[i] = float(q_from[i]), float(q_to[i])
        ss = og.SimpleSetup(si)
        ss.setStartAndGoalStates(start, goal)
        ss.setPlanner(og.RRTConnect(si))
        if not ss.solve(PLAN_TIME):
            return None
        ss.simplifySolution()
        path = ss.getSolutionPath()
        return [np.array([path.getState(k)[i] for i in range(5)]) for k in range(path.getStateCount())]

    # ------------------------------------------------------------------ tasks
    def plan_pick_place(self, arm, obj, site):
        env = self.env
        d, m = env.data, env.model
        opos = d.xpos[env.body_id[obj]].copy()
        tpos = d.site_xpos[m.site(site).id].copy()
        if obj == "cup":
            dirs = [np.array(v, float) for v in ([1, 0], [0, 1], [-1, 0], [0, -1])]
            pairs = [(cg, cp) for cg in dirs for cp in dirs]
        else:  # the jaw axis is fixed relative to the handle; rotate it by the yaw change to keep head direction
            xm = d.xmat[env.body_id[obj]]
            oyaw = np.arctan2(xm[3], xm[0])
            dyaw = env.target_yaw[obj] - oyaw
            rot = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
            pairs = [(s * _across(oyaw), rot @ (s * _across(oyaw))) for s in (1, -1)]

        cache = {}

        def ik_cached(key, pos, cdir, held, prev=None):
            k = (key, tuple(np.round(cdir, 6)))
            if k not in cache:
                cache[k] = self.ik(arm, pos, cdir, obj, prev, valid=self.world_valid(arm, held))
            return cache[k]

        up = np.array([0, 0, PREGRASP_DZ])
        for cg, cp in pairs:
            q_pre = ik_cached("pre", opos + [0, 0, GRASP_DZ[obj]] + up, cg, None)
            q_grasp = q_pre is not None and ik_cached("grasp", opos + [0, 0, GRASP_DZ[obj]], cg, obj, q_pre)
            q_above = q_grasp is not None and q_grasp is not False and ik_cached("above", tpos + [0, 0, PLACE_DZ[obj]] + up, cp, obj)
            q_place = q_above is not None and q_above is not False and ik_cached("place", tpos + [0, 0, PLACE_DZ[obj]], cp, obj, q_above)
            if q_place is not None and q_place is not False:
                break
        else:
            self.fail = f"{obj}: no collision-free IK chain"
            return None

        home = HOME[:5]
        p1 = self.plan(arm, home, q_pre)
        p2 = p1 and self.plan(arm, q_pre, q_above, held=obj)
        p3 = p2 and self.plan(arm, q_above, home)
        if not p3:
            self.fail = f"{obj}: OMPL " + ("home->pregrasp" if not p1 else "carry" if not p2 else "->home")
            return None
        return (_interp(p1, HOME[5], GRIP_OPEN, MAX_VEL)
                + _interp([q_pre, q_grasp], GRIP_OPEN, GRIP_OPEN, FINE_VEL)
                + _hold(q_grasp, GRIP_OPEN, GRIP_CLOSED)
                + _interp([q_grasp, q_pre], GRIP_CLOSED, GRIP_CLOSED, FINE_VEL)
                + _interp(p2, GRIP_CLOSED, GRIP_CLOSED, MAX_VEL)
                + _interp([q_above, q_place], GRIP_CLOSED, GRIP_CLOSED, FINE_VEL)
                + _hold(q_place, GRIP_CLOSED, GRIP_OPEN)
                + _interp([q_place, q_above], GRIP_OPEN, GRIP_OPEN, FINE_VEL)
                + _interp(p3, GRIP_OPEN, GRIP_OPEN, MAX_VEL))

    def plan_task(self, task):
        """(T, 12) joint-target trajectory for `task` from the current env state, or None (see self.fail)."""
        self.fail = None
        per_arm = {}
        for obj, site, arm in TASKS[task]["place"]:
            traj = self.plan_pick_place(arm, obj, site)
            if traj is None:
                return None
            per_arm[arm] = traj
        T = max(len(t) for t in per_arm.values())
        out = np.zeros((T, 12))
        for i, arm in enumerate(ARMS):
            traj = per_arm.get(arm, [HOME.copy()])
            out[:, 6 * i:6 * (i + 1)] = traj + [traj[-1]] * (T - len(traj))
        return out


def rollout(env, expert, task, seed, level="nominal", on_step=None, tail=15):
    """Reset, plan, execute. Returns success bool, or None if planning failed.
    on_step(obs, action) is called before each env.step with the observation the action is taken from."""
    obs = env.reset(seed, level)
    traj = expert.plan_task(task)
    if traj is None:
        return None
    traj = np.vstack([traj, np.repeat(traj[-1:], tail, axis=0)])
    for a in traj:
        if on_step is not None:
            on_step(obs, a)
        env.step(a)
        if on_step is not None:
            obs = env.obs()
    return env.success(task)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--level", default="nominal")
    args = ap.parse_args()

    env = DinnerEnv(render=False)
    ex = Expert(env)

    # Checker hygiene: validity checks restore state; planning never touches env.data.
    env.reset(0)
    before = env.data.qpos.copy()
    is_valid = ex.world_valid("left")
    saved = ex.scratch.qpos.copy()
    is_valid(HOME[:5] + 0.3)
    assert np.array_equal(saved, ex.scratch.qpos), "is_valid leaked state"
    assert ex.plan("left", HOME[:5], HOME[:5] + np.array([0.4, 0.3, -0.3, 0.0, 0.5])) is not None, "trivial plan failed"
    assert np.array_equal(before, env.data.qpos), "planning mutated env.data"

    for task in args.tasks:
        ok, fails, t0 = 0, {}, time.time()
        for seed in range(args.seeds):
            r = rollout(env, ex, task, seed, args.level)
            ok += bool(r)
            if not r:
                reason = ex.fail if r is None else "execution (success check failed)"
                fails[reason] = fails.get(reason, 0) + 1
        print(f"{task:12s} success {ok}/{args.seeds}  {(time.time() - t0) / args.seeds:.1f}s/ep  fails {fails}", flush=True)
