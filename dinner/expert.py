"""Scripted expert: mink IK for grasp/place poses, OMPL RRTConnect for joint-space transits.

Per pick-place: home -> pregrasp (OMPL) -> grasp (linear) -> close -> lift (linear)
-> above target (OMPL, carrying) -> place (linear) -> open -> retreat (linear) -> home (OMPL, around the
object now at its target).

Bimanual strategy: a task is a list of phases (dinner.env.task_phases). Arms within a phase move
simultaneously, each planned with the other arm at its current pose; phases run in order, and each phase
is planned from the simulator state the previous one left (closed-loop between phases). This gives:
- simultaneous dual-arm action (set_table),
- multi-step sequences (full_setting: fork + spoon together, then the cup),
- hand-off through a relay (handoff_fork: the right arm sets the fork on a relay both arms reach,
  the left arm picks it up there and places it).

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

from dinner.env import ARMS, GRIP_CLOSED, GRIP_OPEN, HOME, OBJECTS, RELAY_YAW, TASKS, DinnerEnv, _rot2, _yaw_quat, task_phases

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


def _sync(phase, last):
    """{arm: [6-vectors]} -> (T, 12) run in parallel; an arm absent or finished holds its last command.
    `last` is the 12-D command the previous phase ended on."""
    cur = {"left": last[:6].copy(), "right": last[6:].copy()}
    rows = []
    for i in range(max(len(s) for s in phase.values())):
        for a, s in phase.items():
            if i < len(s):
                cur[a] = s[i]
        rows.append(np.r_[cur["left"], cur["right"]])
    return np.array(rows)


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
        self.fail = None  # reason the last plan returned None
        self.world = None  # qpos the planner treats as the world; None = env.data (the live sim state)

    def _world_qpos(self):
        return self.world if self.world is not None else self.env.data.qpos

    def world_valid(self, arm, held=None, grip=GRIP_OPEN):
        """Validity checker for `arm` against the planning world: table, the other arm (at its pose in
        that world), itself, and every object except `held`."""
        d = self.scratch
        d.qpos[:] = self._world_qpos()
        d.qpos[self.grip_q[arm]] = grip
        other = [a for a in ARMS if a != arm][0]
        forbidden = set(self.table_geoms) | self.arm_geoms[other]
        for o, gs in self.obj_geoms.items():
            if o != held:
                forbidden |= gs
        return make_is_valid(self.env.model, d, self.arm_q[arm], self.arm_geoms[arm], forbidden)

    def _target_yaw(self, obj, site):
        return RELAY_YAW if site == "relay" else self.env.target_yaw.get(obj, 0.0)

    def _placed(self, world, obj, site):
        """Copy of a world qpos with `obj` resting at `site`."""
        env = self.env
        w, a = world.copy(), env.free_adr[obj]
        t = env.data.site_xpos[env.model.site(site).id]
        w[a:a + 3] = [t[0], t[1], env.model.qpos0[a + 2]]
        w[a + 3:a + 7] = _yaw_quat(self._target_yaw(obj, site))
        return w

    # ------------------------------------------------------------------ IK
    def ik(self, arm, pos, close_dir, obj, prev=None, valid=None):
        """First collision-free IK solution within tolerance over the seed set, else None."""
        base = self.env.data.xpos[self.env.model.body(f"{arm}_base").id]
        pan = np.arctan2(pos[0] - base[0], pos[1] - base[1])
        seeds = ([prev] if prev is not None else []) + [np.r_[pan, s] for s in IK_SEEDS]
        m = self.env.model
        q = self._world_qpos().copy()
        # Live sim state can sit a hair past a joint limit (e.g. a gripper squeezed shut), which makes the
        # ConfigurationLimit QP infeasible. Clamp every limited hinge into range first.
        lim = m.jnt_limited.astype(bool) & (m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        adr = m.jnt_qposadr[lim]
        q[adr] = np.clip(q[adr], m.jnt_range[lim, 0] + 1e-6, m.jnt_range[lim, 1] - 1e-6)
        q[self.grip_q[arm]] = GRIP_OPEN
        task = mink.FrameTask(f"{arm}_gripperframe", "site", position_cost=1.0,
                              orientation_cost=ORI_COST[obj], lm_damping=1.0)
        task.set_target(mink.SE3.from_rotation_and_translation(_grasp_rot(close_dir), np.asarray(pos)))
        for sd in seeds:
            q[self.arm_q[arm]] = np.clip(sd, self.ranges[arm][:, 0] + 1e-6, self.ranges[arm][:, 1] - 1e-6)
            self.cfg.update(q)
            try:
                for it in range(IK_ITERS):
                    v = mink.solve_ik(self.cfg, [task], 0.02, "daqp", damping=1e-3,
                                      limits=self.limits, constraints=[self.freeze[arm]])
                    self.cfg.integrate_inplace(v, 0.02)
                    if it % 50 == 49:
                        e = task.compute_error(self.cfg)
                        if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                            break
            except mink.exceptions.NoSolutionFound:
                continue  # this seed is infeasible; try the next
            e = task.compute_error(self.cfg)
            sol = self.cfg.q[self.arm_q[arm]].copy()
            if np.linalg.norm(e[:3]) < POS_TOL and np.linalg.norm(e[3:]) < ORI_TOL[obj] and (valid is None or valid(sol)):
                return sol
        return None

    # ------------------------------------------------------------------ OMPL
    def plan(self, arm, q_from, q_to, held=None):
        """Joint-space RRTConnect for one arm in the planning world."""
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

    # ------------------------------------------------------------------ pick-place
    def plan_pick_place(self, arm, obj, site):
        env = self.env
        m = env.model
        b = env.body_id[obj]
        opos, xm = env.data.xpos[b].copy(), env.data.xmat[b].copy()  # live pose: phases plan from the real state
        tpos = env.data.site_xpos[m.site(site).id].copy()
        if obj == "cup":
            dirs = [np.array(v, float) for v in ([1, 0], [0, 1], [-1, 0], [0, -1])]
            pairs = [(cg, cp) for cg in dirs for cp in dirs]
        else:  # the jaw axis is fixed relative to the handle; rotate it by the yaw change to keep head direction
            oyaw = np.arctan2(xm[3], xm[0])
            rot = _rot2(self._target_yaw(obj, site) - oyaw)
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
            if q_pre is None:
                continue
            q_grasp = ik_cached("grasp", opos + [0, 0, GRASP_DZ[obj]], cg, obj, q_pre)
            if q_grasp is None:
                continue
            q_above = ik_cached("above", tpos + [0, 0, PLACE_DZ[obj]] + up, cp, obj)
            if q_above is None:
                continue
            q_place = ik_cached("place", tpos + [0, 0, PLACE_DZ[obj]], cp, obj, q_above)
            if q_place is not None:
                break
        else:
            self.fail = f"{obj}->{site}: no collision-free IK chain"
            return None

        home = HOME[:5]
        before = self.world
        p1 = self.plan(arm, home, q_pre)
        p2 = p1 and self.plan(arm, q_pre, q_above, held=obj)
        self.world = self._placed(self._world_qpos(), obj, site)  # go home around the object where it now is
        p3 = p2 and self.plan(arm, q_above, home)
        self.world = before
        if not p3:
            self.fail = f"{obj}->{site}: OMPL " + ("home->pregrasp" if not p1 else "carry" if not p2 else "->home")
            return None
        # The cup is gripped with a tilted approach; at 0.8 the jaws brush it on approach and drag it over on
        # retreat. 1.2 (~95 mm gap) clears a 40 mm cup. The pause after opening lets it settle before retreat.
        go = 1.2 if obj == "cup" else GRIP_OPEN
        return (_interp(p1, HOME[5], go, MAX_VEL)
                + _interp([q_pre, q_grasp], go, go, FINE_VEL)
                + _hold(q_grasp, go, GRIP_CLOSED)
                + _interp([q_grasp, q_pre], GRIP_CLOSED, GRIP_CLOSED, FINE_VEL)
                + _interp(p2, GRIP_CLOSED, GRIP_CLOSED, MAX_VEL)
                + _interp([q_above, q_place], GRIP_CLOSED, GRIP_CLOSED, FINE_VEL)
                + _hold(q_place, GRIP_CLOSED, go)
                + _hold(q_place, go, go)
                + _interp([q_place, q_above], go, go, FINE_VEL)
                + _interp(p3, go, go, MAX_VEL))

    def plan_phase(self, phase, last):
        """(T, 12) trajectory for one phase {arm: (obj, site)} from the live env state, or None (see self.fail).
        Each arm is planned with the other at its current pose; `last` is the previous 12-D command."""
        self.fail = None
        trajs = {}
        for arm, (obj, site) in phase.items():
            traj = self.plan_pick_place(arm, obj, site)
            if traj is None:
                return None
            trajs[arm] = traj
        return _sync(trajs, last)


def rollout(env, expert, task, seed, level="nominal", on_step=None, tail=15):
    """Reset, then plan and execute each phase from the state the previous one left.
    Returns success bool, or None if planning failed. on_step(obs, action) is called before each env.step
    with the observation the action is taken from."""
    obs = env.reset(seed, level, task)
    last = np.tile(HOME, 2)

    def run(traj):
        nonlocal obs
        for a in traj:
            if on_step is not None:
                on_step(obs, a)
            env.step(a)
            if on_step is not None:
                obs = env.obs()

    for phase in task_phases(task):
        traj = expert.plan_phase(phase, last)
        if traj is None:
            return None
        run(traj)
        last = traj[-1]
    run(np.repeat(last[None], tail, axis=0))
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
