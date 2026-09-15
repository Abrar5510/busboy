"""Dual SO-101 dinner-table environment (MuJoCo).

State/action: 12-D, [left 6 joints, right 6 joints] = 5 arm joints + gripper each.
Action is a joint position target written straight to data.ctrl.
Run `python -m dinner.env` for the self-check.
"""

from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "sim" / "scene.xml"

FPS = 30
N_SUBSTEPS = 16  # 16 x (1/480 s) = exactly 1/30 s
IMG_HW = (256, 256)
CAMERAS = {"front": "front", "left_wrist": "left_wrist_cam", "right_wrist": "right_wrist_cam"}
ARMS = ("left", "right")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# Calibration knobs (tune against the sim, not in code paths).
HOME = np.array([0.0, -1.74, 1.2, 1.4, 0.0, 0.0])  # per arm; tucked, no self-contact
GRIP_OPEN = 0.8
GRIP_CLOSED = -0.17
GRIP_OPEN_BAND = 0.3  # gripper ctrl >= this counts as released
JAW_MAX_GAP = 0.1115  # measured: max jaw gap (mid-finger pads) at gripper=1.745
PLACE_TOL = 0.03
SETTLE_STEPS = 15  # control steps after reset so objects come to rest

# Base randomization ranges; every one is multiplied by RAND_LEVELS[level].
RAND_LEVELS = {"nominal": 1.0, "heavy": 1.5}
OBJ_XY = 0.04
OBJ_YAW = np.deg2rad(30)
CAM_POS = 0.02
CAM_LOOK = 0.04  # look-at jitter; ~3 deg at the front camera's range
CAM_FOVY = 5.0
MASS_FRICTION = 0.2
LIGHT_POS = 0.3
LIGHT_DIFFUSE = 0.3

OBJECTS = ("plate", "fork", "spoon", "cup")
FOOTPRINT = {"plate": 0.065, "fork": 0.05, "spoon": 0.05, "cup": 0.025}  # for layout rejection sampling
START_TARGET = {"fork": "fork_target", "spoon": "spoon_target", "cup": "cup_target"}
TARGET_YAW_OFFSET = {"fork": np.pi / 2, "spoon": np.pi / 2}  # across the table, head away

TASKS = {
    "fork_left": {
        "arms": ["left"], "place": [("fork", "fork_target", "left")],
        "train": ["Place the fork to the left of the plate.",
                  "Put the fork down to the plate's left.",
                  "Set the fork on the left side of the plate."],
        "eval": ["Position the fork left of the plate."],
    },
    "spoon_right": {
        "arms": ["right"], "place": [("spoon", "spoon_target", "right")],
        "train": ["Place the spoon to the right of the plate.",
                  "Put the spoon down to the plate's right.",
                  "Set the spoon on the right side of the plate."],
        "eval": ["Position the spoon right of the plate."],
    },
    "cup_tr": {
        "arms": ["right"], "place": [("cup", "cup_target", "right")],
        "train": ["Put the cup at the top right of the plate.",
                  "Place the cup above and to the right of the plate.",
                  "Set the cup at the plate's upper right."],
        "eval": ["Move the cup to the upper right corner of the place setting."],
    },
    "set_table": {
        "arms": ["left", "right"], "subtasks": ["fork_left", "spoon_right"],
        "place": [("fork", "fork_target", "left"), ("spoon", "spoon_target", "right")],
        "train": ["Set the table.",
                  "Set the table for dinner.",
                  "Lay out the fork and spoon around the plate."],
        "eval": ["Arrange the cutlery for a meal."],
    },
}


def task_text(task, split, rng):
    return str(rng.choice(TASKS[task][split]))


def _ang_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class DinnerEnv:
    def __init__(self, render=True):
        self.model = m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(m)
        self.state_idx = np.array([m.jnt_qposadr[m.joint(f"{a}_{j}").id] for a in ARMS for j in JOINTS])
        self.act_ids = np.array([m.actuator(f"{a}_{j}").id for a in ARMS for j in JOINTS])
        self.grip_act = {a: m.actuator(f"{a}_gripper").id for a in ARMS}
        self.free_adr = {o: m.jnt_qposadr[m.joint(o).id] for o in OBJECTS}
        self.body_id = {o: m.body(o).id for o in OBJECTS}
        self.obj_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.body_id.values()]

        self.table_geom_id = m.geom("table_top").id
        self.table_top_z = m.geom_pos[self.table_geom_id][2] + m.geom_size[self.table_geom_id][2]
        self.table_mat_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n) for n in ("mat_a", "mat_b", "mat_c")]
        self.front_cam = m.camera("front").id
        self.look_at = m.body("look_at").id

        # Nominal copies: every reset restores these before randomizing, so nothing drifts.
        self._nominal = {k: getattr(m, k).copy() for k in (
            "body_mass", "geom_friction", "geom_rgba", "geom_matid", "cam_pos", "cam_fovy",
            "body_pos", "light_pos", "light_diffuse", "light_castshadow")}

        self.renderer = mujoco.Renderer(m, *IMG_HW) if render else None
        self.target_yaw = {}

    # ---------------------------------------------------------------- reset
    def _sample_layout(self, rng, s):
        """Object xy/yaw around the scene's nominal layout, rejecting overlaps and objects
        that start near any target zone (a blocker there makes the place pose unreachable).
        ponytail: rejection sampling, 50 tries then accept; fine at these densities."""
        m = self.model
        nominal = {}
        for o in OBJECTS:
            q0 = m.qpos0[self.free_adr[o]:self.free_adr[o] + 7]
            nominal[o] = (q0[:2].copy(), 2 * np.arctan2(q0[6], q0[3]))
        for _ in range(50):
            lay = {o: (xy + rng.uniform(-OBJ_XY * s, OBJ_XY * s, 2), yaw + rng.uniform(-OBJ_YAW * s, OBJ_YAW * s))
                   for o, (xy, yaw) in nominal.items()}
            pxy, pyaw = lay["plate"]
            ok = all(np.linalg.norm(lay[a][0] - lay[b][0]) > FOOTPRINT[a] + FOOTPRINT[b] + 0.01
                     for i, a in enumerate(OBJECTS) for b in OBJECTS[i + 1:])
            for site in START_TARGET.values():
                target_xy = pxy + _rot2(pyaw) @ m.site_pos[m.site(site).id][:2]
                ok &= all(np.linalg.norm(lay[o][0] - target_xy) > FOOTPRINT[o] + 2 * PLACE_TOL for o in START_TARGET)
            if ok:
                break
        return lay

    def reset(self, seed, level="nominal"):
        m, d = self.model, self.data
        rng = np.random.default_rng(seed)
        s = RAND_LEVELS[level]
        u = lambda r, n=None: rng.uniform(-r * s, r * s, n)

        for k, v in self._nominal.items():
            getattr(m, k)[:] = v
        mujoco.mj_resetData(m, d)

        home = np.tile(HOME, 2)
        d.qpos[self.state_idx] = home
        d.ctrl[self.act_ids] = home

        lay = self._sample_layout(rng, s)
        for o, (xy, yaw) in lay.items():
            a = self.free_adr[o]
            d.qpos[a:a + 2] = xy
            d.qpos[a + 2] = m.qpos0[a + 2]
            d.qpos[a + 3:a + 7] = _yaw_quat(yaw)
        plate_yaw = lay["plate"][1]

        # Appearance and physics.
        m.geom_matid[self.table_geom_id] = int(rng.choice(self.table_mat_ids))
        for g in self.obj_geoms:
            m.geom_rgba[g, :3] = np.clip(m.geom_rgba[g, :3] + u(0.15, 3), 0, 1)
            m.geom_friction[g, 0] *= 1 + u(MASS_FRICTION)
        for b in self.body_id.values():
            m.body_mass[b] *= 1 + u(MASS_FRICTION)
        m.light_pos[:] += u(LIGHT_POS, m.light_pos.shape)
        m.light_diffuse[:] = np.clip(m.light_diffuse * (1 + u(LIGHT_DIFFUSE)), 0.1, 1.0)
        m.light_castshadow[:] = rng.random(m.nlight) < 0.8
        m.cam_pos[self.front_cam] += u(CAM_POS, 3)
        m.cam_fovy[self.front_cam] += u(CAM_FOVY)
        m.body_pos[self.look_at] += u(CAM_LOOK, 3)

        mujoco.mj_forward(m, d)
        for _ in range(SETTLE_STEPS):
            self.step(home)
        self.target_yaw = {n: plate_yaw + off for n, off in TARGET_YAW_OFFSET.items()}
        return self.obs()

    # ---------------------------------------------------------------- io
    def obs(self):
        o = {"observation.state": self.data.qpos[self.state_idx].astype(np.float32)}
        if self.renderer is not None:
            for key, cam in CAMERAS.items():
                self.renderer.update_scene(self.data, camera=cam)
                o[f"observation.images.{key}"] = self.renderer.render().copy()
        return o

    def step(self, action):
        self.data.ctrl[self.act_ids] = action
        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)

    # ---------------------------------------------------------------- success
    def _place_ok(self, name, target_site):
        m, d = self.model, self.data
        b = self.body_id[name]
        pos, xmat = d.xpos[b], d.xmat[b].reshape(3, 3)
        target = d.site_xpos[m.site(target_site).id]
        if np.linalg.norm(pos[:2] - target[:2]) > PLACE_TOL:
            return False
        if pos[2] < self.table_top_z + 0.002:  # actually on table
            return False
        if name in ("fork", "spoon"):  # yaw of long axis
            yaw = np.arctan2(xmat[1, 0], xmat[0, 0])
            if abs(_ang_diff(yaw, self.target_yaw[name])) > np.deg2rad(45):
                return False
        if name == "cup":  # no abs: upside-down must fail
            if np.degrees(np.arccos(np.clip(xmat[2, 2], -1.0, 1.0))) > 30:
                return False
        return True

    def released(self, arm):
        return self.data.ctrl[self.grip_act[arm]] >= GRIP_OPEN_BAND

    def success(self, task):
        return all(self._place_ok(o, site) and self.released(arm) for o, site, arm in TASKS[task]["place"])

    def set_free(self, name, pos, quat):
        a = self.free_adr[name]
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = quat
        dof = self.model.jnt_dofadr[self.model.joint(name).id]
        self.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)


if __name__ == "__main__":
    env = DinnerEnv()
    m, d = env.model, env.data
    assert abs(N_SUBSTEPS * m.opt.timestep - 1 / 30) < 1e-9

    yaws = set()
    for level in RAND_LEVELS:
        for seed in range(5):
            o = env.reset(seed, level)
            assert np.isfinite(d.qpos).all()
            for cam in CAMERAS:
                img = o[f"observation.images.{cam}"]
                assert img.shape == (*IMG_HW, 3) and img.dtype == np.uint8 and img.std() > 1, cam
            arm_err = np.abs(d.qpos[env.state_idx] - np.tile(HOME, 2)).reshape(2, 6)[:, :5]
            assert arm_err.max() < np.deg2rad(1), f"arms not at home: {np.degrees(arm_err.max()):.2f} deg"
            for t in TASKS:
                assert not env.success(t), f"{t} trivially solved at reset (seed {seed}, {level})"
            yaws.add(round(env.target_yaw["fork"], 4))
    assert len(yaws) > 1, "target_yaw does not follow plate yaw"

    # _place_ok: teleport onto targets.
    env.reset(0)
    z = env.table_top_z
    fork_t = d.site_xpos[m.site("fork_target").id]
    env.set_free("fork", [fork_t[0], fork_t[1], z + 0.006], _yaw_quat(env.target_yaw["fork"]))
    assert env._place_ok("fork", "fork_target")
    env.set_free("fork", [fork_t[0], fork_t[1], z + 0.006], _yaw_quat(env.target_yaw["fork"] + np.pi / 2))
    assert not env._place_ok("fork", "fork_target"), "yawed 90 deg must fail"
    env.set_free("fork", [fork_t[0] + 0.05, fork_t[1], z + 0.006], _yaw_quat(env.target_yaw["fork"]))
    assert not env._place_ok("fork", "fork_target"), "off target must fail"
    cup_t = d.site_xpos[m.site("cup_target").id]
    env.set_free("cup", [cup_t[0], cup_t[1], z + 0.04], [1, 0, 0, 0])
    assert env._place_ok("cup", "cup_target")
    env.set_free("cup", [cup_t[0], cup_t[1], z + 0.04], [0, 1, 0, 0])  # flipped 180 deg about x
    assert not env._place_ok("cup", "cup_target"), "upside-down cup must fail"
    env.set_free("cup", [cup_t[0], cup_t[1], z + 0.04], [1, 0, 0, 0])
    d.ctrl[env.grip_act["right"]] = GRIP_CLOSED
    assert not env.success("cup_tr"), "success requires released gripper"
    d.ctrl[env.grip_act["right"]] = GRIP_OPEN
    assert env.success("cup_tr")

    rng = np.random.default_rng(0)
    assert task_text("fork_left", "eval", rng) == "Position the fork left of the plate."
    print("dinner.env self-check OK")
