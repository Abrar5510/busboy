"""Camera perception: object poses from the overhead camera's depth + instance masks.

Instance masks come from MuJoCo's segmentation buffer, standing in for a trained detector (the same labeled
renders are what you would train one on, e.g. with Intel Geti). Everything downstream is real geometry: pixels
are back-projected through the camera's intrinsics/extrinsics, positions come from the visible top surface,
cutlery yaw from PCA over the handle with the head deciding direction, cup uprightness from its top height.

The plate is a plain disc, so its heading is unobservable; the place-setting frame's yaw is taken from the
scene (like a known table layout). Its position is perceived.

Run `python -m dinner.perception` for the accuracy self-check against simulator ground truth.
"""

import mujoco
import numpy as np

from dinner.env import OBJECTS, DinnerEnv

CAM = "top"
HW = (480, 480)
HALF_H = {"plate": 0.006, "fork": 0.006, "spoon": 0.006, "cup": 0.04}  # body origin above the table (object models)
HANDLE_HALF_LEN = 0.05 - 0.0012  # model half-length minus about half a pixel (edge pixels count at their centre)
TOP_BAND = 0.003  # points within this of an object's highest point form its top surface
CUP_UPRIGHT_MIN_Z = 0.06  # cup top above table when upright is 0.08; lying on its side it is 0.04


class Perception:
    def __init__(self, env: DinnerEnv):
        self.env = env
        m = env.model
        self.cam = m.camera(CAM).id
        self.rgb = mujoco.Renderer(m, *HW)
        self.depth = mujoco.Renderer(m, *HW)
        self.depth.enable_depth_rendering()
        self.seg = mujoco.Renderer(m, *HW)
        self.seg.enable_segmentation_rendering()
        geoms = {o: [g for g in range(m.ngeom) if m.geom_bodyid[g] == env.body_id[o]] for o in OBJECTS}
        self.geoms = geoms
        self.handle = {o: m.geom(f"{o}_handle").id for o in ("fork", "spoon")}
        self.head = {o: m.geom(f"{o}_head").id for o in ("fork", "spoon")}
        self.site_local = {s: m.site_pos[m.site(s).id].copy() for s in ("fork_target", "spoon_target", "cup_target")}

    def _points(self, depth):
        m, d = self.env.model, self.env.data
        h, w = HW
        f = 0.5 * h / np.tan(np.deg2rad(m.cam_fovy[self.cam]) / 2)
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        cam_pts = np.stack([(u - w / 2) / f * depth, -(v - h / 2) / f * depth, -depth], -1)
        return cam_pts @ d.cam_xmat[self.cam].reshape(3, 3).T + d.cam_xpos[self.cam]

    def project(self, xyz):
        """World point -> (u, v) pixel in the perception image (for drawing)."""
        d = self.env.data
        c = d.cam_xmat[self.cam].reshape(3, 3).T @ (np.asarray(xyz) - d.cam_xpos[self.cam])
        f = 0.5 * HW[0] / np.tan(np.deg2rad(self.env.model.cam_fovy[self.cam]) / 2)
        return int(HW[1] / 2 + f * c[0] / -c[2]), int(HW[0] / 2 - f * c[1] / -c[2])

    def render_rgb(self):
        self.rgb.update_scene(self.env.data, camera=CAM)
        return self.rgb.render()

    def observe(self):
        """-> {"image": rgb, "objects": {name: {"pos", "yaw", "upright", "visible_px"}}, "targets": {site: xyz}}.
        Objects with no visible pixels are omitted."""
        d = self.env.data
        for r in (self.rgb, self.depth, self.seg):
            r.update_scene(d, camera=CAM)
        rgb, depth, seg = self.rgb.render(), self.depth.render(), self.seg.render()
        pts = self._points(depth)
        geom_px = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        table_z = self.env.table_top_z

        objects = {}
        for o, gs in self.geoms.items():
            mask = np.isin(geom_px, gs)
            if not mask.any():
                continue
            p = pts[mask]
            top_mask = mask & (pts[..., 2] > p[:, 2].max() - TOP_BAND)
            if o in self.handle:  # handle top surface: origin + axis; head centroid gives direction
                hp = pts[geom_px == self.handle[o]][:, :2]  # all handle pixels: a top-band cut is too depth-noisy
                mean = hp.mean(0)
                axis = np.linalg.eigh(np.cov((hp - mean).T))[1][:, -1]
                head = pts[geom_px == self.head[o]]
                if len(head) and np.dot(head[:, :2].mean(0) - mean, axis) < 0:
                    axis = -axis
                # The home-pose arm can hide the handle's base end, which biases the centroid; the head end is
                # the one facing away from the robot, so anchor on it (handle half-length from the object model).
                xy = mean + ((hp - mean) @ axis).max() * axis - HANDLE_HALF_LEN * axis
                yaw = float(np.arctan2(axis[1], axis[0]))
            else:
                xy, yaw = pts[top_mask][:, :2].mean(0), 0.0
            objects[o] = {"pos": np.r_[xy, table_z + HALF_H[o]], "yaw": yaw,
                          "upright": bool(p[:, 2].max() - table_z > CUP_UPRIGHT_MIN_Z) if o == "cup" else True,
                          "top_z": float(p[:, 2].max()), "visible_px": int(mask.sum())}

        targets = {"relay": d.site_xpos[self.env.model.site("relay").id].copy()}  # fixed pad on the table
        if "plate" in objects:
            pyaw = self.env.target_yaw["fork"] - np.pi / 2  # setting-frame heading (see module docstring)
            c, s = np.cos(pyaw), np.sin(pyaw)
            for site, loc in self.site_local.items():
                targets[site] = np.r_[objects["plate"]["pos"][:2] + [c * loc[0] - s * loc[1], s * loc[0] + c * loc[1]],
                                      table_z]
        return {"image": rgb, "objects": objects, "targets": targets}


if __name__ == "__main__":
    env = DinnerEnv(render=False)
    per = Perception(env)
    pos_err, yaw_err = [], []
    for seed in range(20):
        env.reset(10000 + seed, "heavy", "handoff_fork" if seed % 2 else None)
        obs = per.observe()
        for o in OBJECTS:
            assert o in obs["objects"], f"{o} not visible (seed {seed})"
            est, b = obs["objects"][o], env.body_id[o]
            pos_err.append(np.linalg.norm(est["pos"][:2] - env.data.xpos[b][:2]))
            if o in ("fork", "spoon"):
                xm = env.data.xmat[b]
                yaw_err.append(abs((est["yaw"] - np.arctan2(xm[3], xm[0]) + np.pi) % (2 * np.pi) - np.pi))
        for site in ("fork_target", "spoon_target", "cup_target"):
            pos_err.append(np.linalg.norm(obs["targets"][site][:2] - env.data.site_xpos[env.model.site(site).id][:2]))
        assert obs["objects"]["cup"]["upright"]
    print(f"position error mean {1000 * np.mean(pos_err):.1f} mm, max {1000 * np.max(pos_err):.1f} mm; "
          f"cutlery yaw error mean {np.degrees(np.mean(yaw_err)):.1f} deg, max {np.degrees(np.max(yaw_err)):.1f} deg")
    assert np.max(pos_err) < 0.006 and np.degrees(np.max(yaw_err)) < 5

    env.reset(10000)  # a cup knocked over must read as not upright
    env.set_free("cup", env.data.xpos[env.body_id["cup"]] + [0, 0, -0.02], [0.7071, 0.7071, 0, 0])
    assert not per.observe()["objects"]["cup"]["upright"]
    print("dinner.perception self-check OK")
