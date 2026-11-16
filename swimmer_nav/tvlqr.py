"""Time-varying LQR tracking of a trajectory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import casadi as ca
import mujoco
import mujoco.viewer
import numpy as np

from .body_path import body_points
from .config import SwimmerConfig
from .dynamics import NQ, NU, NX, SwimmerModel, sim_state
from .gcs import GcsMap
from .physics import SwimmerSim
from .trajopt import Trajectory, plan_trajectory, plot, rollout, route_regions
from .world import body_clearances, wall_clearances

# state weights: x, y, theta, phi1, phi2, xd, yd, thetad, phi1d, phi2d, F
Q_DEFAULT = np.array([20, 20, 5, 0.5, 0.5, 2, 2, 0.5, 0.05, 0.05, 0.01])
R_DEFAULT = np.array([1.0, 1.0, 2.0])  # phi1*, phi2* [rad], thrust command [0, 1]


class TVLQR:
    def __init__(self, traj: Trajectory, cfg: SwimmerConfig | None = None,
                 Q=Q_DEFAULT, R=R_DEFAULT, Qf_scale: float = 10.0):
        self.cfg, self.traj = cfg or SwimmerConfig(), traj
        t0 = time.perf_counter()
        Phi = SwimmerModel(self.cfg).discrete(traj.h)
        x, u = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
        xn = Phi(x, u)
        jac = ca.Function("linearize", [x, u], [ca.jacobian(xn, x), ca.jacobian(xn, u)])
        N = traj.N
        self.A, self.B = np.empty((N, NX, NX)), np.empty((N, NX, NU))
        for k in range(N):
            a, b = jac(traj.X[k], traj.U[k])
            self.A[k], self.B[k] = np.array(a), np.array(b)
        Q, R = np.diag(Q), np.diag(R)
        P = Qf_scale * Q
        self.K = np.empty((N, NU, NX))
        for k in reversed(range(N)):
            BtP = self.B[k].T @ P
            self.K[k] = np.linalg.solve(R + BtP @ self.B[k], BtP @ self.A[k])
            P = Q + self.A[k].T @ P @ (self.A[k] - self.B[k] @ self.K[k])
            P = 0.5 * (P + P.T)
        self.gain_seconds = time.perf_counter() - t0
        tl = self.cfg.joint_target_limit
        self.lo, self.hi = np.array([-tl, -tl, 0.0]), np.array([tl, tl, 1.0])

    def reference(self, k: int) -> np.ndarray:
        return self.traj.X[min(k, self.traj.N)]

    def __call__(self, k: int, x: np.ndarray) -> np.ndarray:
        kk = min(k, self.traj.N - 1)
        dx = x - self.reference(k)
        dx[2] = (dx[2] + np.pi) % (2 * np.pi) - np.pi
        return np.clip(self.traj.U[kk] - self.K[kk] @ dx, self.lo, self.hi)


def tracking_error(traj: Trajectory, X: np.ndarray, L: float) -> float:
    """RMS head-position error against the plan over the plan's duration [m]."""
    n = min(len(X), traj.N + 1)
    e = [np.linalg.norm(body_points(X[k, :NQ], L)[0] - body_points(traj.X[k, :NQ], L)[0]) for k in range(n)]
    return float(np.sqrt(np.mean(np.square(e))))


def evaluate(m: GcsMap, traj: Trajectory, cfg: SwimmerConfig, drag_scales=(0.7, 1.0, 1.3)):
    ctrl = TVLQR(traj, cfg)
    runs = {}
    for s in drag_scales:
        runs[f"open loop, drag x{s}"] = rollout(traj, m.circles, cfg, None, s)
        runs[f"TVLQR, drag x{s}"] = rollout(traj, m.circles, cfg, ctrl, s)
    for r in runs.values():
        r["rms_error"] = tracking_error(traj, r["X"], cfg.link_length)
    return runs, ctrl


# ------------------------------------------------------------ visualization
def _add_sphere(scn, pos, radius, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.array([pos[0], pos[1], 0.06]), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def draw_reference(scn, ref_heads: np.ndarray, k: int, start: int = 0) -> None:
    """Planned head path (small purple dots) and the plan's head at step k (big dot).
    Geoms are appended after index `start` (the viewer's user scene starts empty)."""
    scn.ngeom = start
    for p in ref_heads[::2]:
        _add_sphere(scn, p, 0.015, (0.48, 0.12, 0.64, 0.8))
    _add_sphere(scn, ref_heads[min(k, len(ref_heads) - 1)], 0.05, (0.48, 0.12, 0.64, 0.5))


def play(m: GcsMap, traj: Trajectory, cfg: SwimmerConfig, open_loop: bool, drag: float,
         speed: float = 1.0, video: str | None = None, extra_steps: int = 20) -> None:
    from .evaluate import top_camera

    sim = SwimmerSim(cfg)
    sim.set_drag(cfg.with_drag_scale(drag))
    sim.reset(traj.X[0, :NQ])
    sim.show_obstacles(m.circles)
    sim.show_goal(traj.goal, 0.25)
    ctrl = None if open_loop else TVLQR(traj, cfg)
    ref_heads = np.array([body_points(x[:NQ], cfg.link_length)[0] for x in traj.X])
    dt, L, A = traj.h, cfg.link_length, cfg.arena_half
    label = f"{'open loop' if open_loop else 'TVLQR'}, drag x{drag}"

    def run(frame):
        status, k = "missed goal", 0
        for k in range(traj.N + extra_steps):
            frame(k)
            u = traj.U[min(k, traj.N - 1)] if ctrl is None else ctrl(k, sim_state(sim))
            sim.step(u[:2], u[2])
            q = sim.data.qpos
            c = wall_clearances(q, A, L, cfg.link_radius).min()
            if len(m.circles):
                c = min(c, body_clearances(q, m.circles, L, cfg.link_radius).min())
            if c < 0:
                status = f"COLLISION at {(k + 1) * dt:.1f} s"
                break
            if np.linalg.norm(sim.head_pos() - traj.goal) <= 0.25:
                status = f"goal reached at {(k + 1) * dt:.1f} s"
                break
        for _ in range(10):  # linger on the final state
            frame(k)
        print(f"{label}: {status}")

    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        renderer, cam = mujoco.Renderer(sim.model, 720, 720), mujoco.MjvCamera()
        top_camera(cam)
        writer = imageio.get_writer(video, fps=round(speed / dt), codec="libx264", macro_block_size=1)

        def frame(k):
            renderer.update_scene(sim.data, camera=cam)
            draw_reference(renderer.scene, ref_heads, k, start=renderer.scene.ngeom)
            writer.append_data(renderer.render())

        run(frame)
        writer.close()
        print(f"wrote {video}")
        return

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        top_camera(viewer.cam)

        def frame(k):
            t0 = time.perf_counter()
            with viewer.lock():
                draw_reference(viewer.user_scn, ref_heads, k)
            viewer.sync()
            time.sleep(max(0.0, dt / speed - (time.perf_counter() - t0)))

        run(frame)
        while viewer.is_running():  # keep the window open on the final state
            time.sleep(0.05)


# --------------------------------------------------------------------- CLI
def load_or_plan(map_path: str, cfg: SwimmerConfig) -> tuple[GcsMap, Trajectory, dict]:
    m = GcsMap.load(map_path)
    npz = Path(f"{Path(map_path).with_suffix('')}_traj.npz")
    q0, goal = np.array(m.meta["start_qpos"]), np.array(m.meta["goal"])
    if npz.exists():
        traj = Trajectory.load(npz)
        plan = m.plan_bezier(body_points(q0, cfg.link_length)[0], goal)
        info = {"plan": plan, "route": route_regions(m, plan)}
        print(f"loaded {npz}")
    else:
        traj, info = plan_trajectory(m, q0, goal, cfg)
        if traj is None:
            raise SystemExit(f"NO PLAN: {info['reason']}")
        traj.save(npz)
    return m, traj, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="2D GCS map (maps/...json)")
    ap.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer")
    ap.add_argument("--video", default=None, help="record an mp4 instead")
    ap.add_argument("--open-loop", action="store_true", help="(view/video) no feedback, plain plan controls")
    ap.add_argument("--drag", type=float, default=1.0, help="(view/video) drag scale of the simulated water")
    ap.add_argument("--speed", type=float, default=1.0, help="(view/video) playback speed")
    a = ap.parse_args()
    cfg = SwimmerConfig()
    m, traj, info = load_or_plan(a.map, cfg)
    if a.view or a.video:
        play(m, traj, cfg, a.open_loop, a.drag, a.speed, a.video)
        return
    runs, ctrl = evaluate(m, traj, cfg)
    print(f"plan: {traj.duration:.1f} s; TVLQR gains for {traj.N} steps computed in {ctrl.gain_seconds:.2f} s")
    print(f"{'':24s} {'result':>12s} {'time':>6s} {'head-goal':>9s} {'min clr':>8s} {'RMS track':>9s}")
    for k, r in runs.items():
        res = "success" if r["success"] else ("collision" if r["collision"] else "missed goal")
        print(f"{k:24s} {res:>12s} {r['steps'] * traj.h:5.1f}s {r['final_goal_dist']:8.2f}m "
              f"{r['min_clearance']:+7.3f}m {r['rms_error']:8.3f}m")
    out = f"{Path(a.map).with_suffix('')}_tvlqr.png"
    plot(out, m, info, traj, runs, cfg)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
