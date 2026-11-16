"""Evaluate a trained policy
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from .config import SwimmerConfig
from .env import SwimmerNavEnv
from .world import TASKSETS, make_taskset


def run_config(run_dir: str | Path) -> SwimmerConfig:
    """The physics (incl. thruster options) the run was trained with."""
    c = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())
    return SwimmerConfig.from_dict(c.get("physics"))


def load_obs_normalization(run_dir: str | Path, which: str = "best") -> dict:
    """Frozen observation statistics of a checkpoint: from the full saved
    VecNormalize, or from the older obs-only pickle."""
    run_dir = Path(run_dir)
    full = run_dir / f"{which}_vecnormalize.pkl"
    if full.exists():
        with open(full, "rb") as f:
            vn = pickle.load(f)
        return {"mean": vn.obs_rms.mean, "var": vn.obs_rms.var, "clip": vn.clip_obs, "eps": vn.epsilon}
    with open(run_dir / f"{which}_obs_rms.pkl", "rb") as f:
        return pickle.load(f)


def load_policy(run_dir: str | Path, which: str = "best"):
    """Deterministic policy act(obs_batch) -> action_batch, with the frozen
    observation normalization saved next to the checkpoint."""
    from stable_baselines3 import PPO

    run_dir = Path(run_dir)
    model = PPO.load(run_dir / f"{which}_model.zip", device="cpu")
    rms = load_obs_normalization(run_dir, which)
    scale = 1.0 / np.sqrt(rms["var"] + rms["eps"])

    def act(obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs - rms["mean"]) * scale, -rms["clip"], rms["clip"])
        return model.predict(x, deterministic=True)[0]

    return act


# ------------------------------------------------------------------- rollouts
def run_episodes(act, tasks, reset_options: dict | None = None, n_parallel: int = 16,
                 cfg: SwimmerConfig | None = None) -> list[dict]:
    """Run each task once with a pool of in-process envs (policy calls batched).
    Returns the final info dict of every episode, in task order."""
    envs = [SwimmerNavEnv(cfg=cfg) for _ in range(min(n_parallel, len(tasks)))]
    results: list[dict] = [{} for _ in tasks]
    owner = [-1] * len(envs)  # task index running in each env
    obs = np.zeros((len(envs), SwimmerNavEnv.obs_dim), np.float32)
    queue = iter(range(len(tasks)))

    def start(i: int) -> None:
        k = next(queue, -1)
        owner[i] = k
        if k >= 0:
            obs[i], _ = envs[i].reset(options={**(reset_options or {}), "task": tasks[k].copy()})

    for i in range(len(envs)):
        start(i)
    while (active := [i for i, k in enumerate(owner) if k >= 0]):
        actions = act(obs[active])
        for a, i in zip(actions, active):
            obs[i], _, term, trunc, info = envs[i].step(a)
            if term or trunc:
                results[owner[i]] = info
                start(i)
    return results


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if r["success"]]
    out = {"episodes": len(results),
           "success": float(np.mean([r["success"] for r in results])),
           "collision": float(np.mean([r["collision"] for r in results])),
           "timeout": float(np.mean([r["timeout"] for r in results]))}
    if ok:  # efficiency is only meaningful on successful episodes
        out["time_to_goal_s"] = float(np.mean([r["time"] for r in ok]))
        out["thrust_impulse_Ns"] = float(np.mean([r["thrust_impulse"] for r in ok]))
        out["joint_effort_N2m2s"] = float(np.mean([r["joint_effort"] for r in ok]))
        out["path_over_geodesic"] = float(np.mean([r["path_length"] / r["geodesic_start"] for r in ok]))
    return out


def inference_latency(act, n: int = 2000) -> dict:
    """Wall-clock time of one policy call on a single observation (what a robot
    would pay every control step)."""
    obs = SwimmerNavEnv().reset(seed=0)[0][None]
    for _ in range(50):
        act(obs)
    t = np.empty(n)
    for k in range(n):
        t0 = time.perf_counter()
        act(obs)
        t[k] = time.perf_counter() - t0
    return {"mean_us": float(t.mean() * 1e6), "p99_us": float(np.percentile(t, 99) * 1e6)}


# -------------------------------------------------------------------- report
SUITES = {  # name: (task set, number of tasks, reset options)
    "test": ("test", 200, None),
    "test, drag -30%": ("test", 100, {"drag_scale": (0.7, 0.7)}),
    "test, drag +30%": ("test", 100, {"drag_scale": (1.3, 1.3)}),
}


def report(run_dir: str, which: str) -> dict:
    act, cfg = load_policy(run_dir, which), run_config(run_dir)
    out = {"inference": inference_latency(act)}
    print(f"policy inference: {out['inference']['mean_us']:.0f} us/step (p99 {out['inference']['p99_us']:.0f} us)")
    print(f"{'suite':32s} {'succ':>5s} {'coll':>5s} {'tout':>5s} {'time':>6s} {'thrust':>10s} {'effort':>7s}")
    for name, (tset, n, opts) in SUITES.items():
        s = summarize(run_episodes(act, make_taskset(tset, n, cfg), opts, cfg=cfg))
        out[name] = s
        print(f"{name:32s} {s['success']:5.2f} {s['collision']:5.2f} {s['timeout']:5.2f} "
              f"{s.get('time_to_goal_s', np.nan):6.1f} {s.get('thrust_impulse_Ns', np.nan):10.1f} "
              f"{s.get('joint_effort_N2m2s', np.nan):7.1f}")
    path = Path(run_dir) / f"eval_{which}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    return out


# ------------------------------------------------------------- view / video
def describe(info: dict) -> str:
    if info["success"]:
        return f"SUCCESS in {info['time']:.1f}s"
    return "COLLISION" if info["collision"] else "TIMEOUT"


def top_camera(cam) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = 0.0
    cam.distance, cam.elevation, cam.azimuth = 9.5, -90.0, 90.0


def play(run_dir, which, tset, start, n, drag, speed, video=None) -> None:
    act, cfg = load_policy(run_dir, which), run_config(run_dir)
    tasks = make_taskset(tset, start + n, cfg)[start:]
    play_tasks(act, cfg, tasks, drag, speed, video, labels=[f"task {start + k}" for k in range(n)])


def play_tasks(act, cfg, tasks, drag=1.0, speed=1.0, video=None, labels=None) -> None:
    """Run the policy on the given tasks one after another, in the viewer or to an mp4."""
    labels = labels or [f"task {k}" for k in range(len(tasks))]
    opts = {"drag_scale": (drag, drag)}
    env = SwimmerNavEnv(cfg=cfg)
    dt = env.cfg.control_dt

    def episodes(on_frame, keep_going=lambda: True):
        for k, task in enumerate(tasks):
            if not keep_going():
                return
            obs, _ = env.reset(options={**opts, "task": task.copy()})
            print(f"{labels[k]}: {len(task.circles)} obstacles ... ", end="", flush=True)
            done = False
            while not done and keep_going():
                on_frame()
                obs, _, term, trunc, info = env.step(act(obs[None])[0])
                done = term or trunc
            if done:
                print(describe(info))
                for _ in range(10):  # linger on the final state
                    on_frame()

    if video:
        import imageio.v2 as imageio

        Path(video).parent.mkdir(parents=True, exist_ok=True)
        renderer, cam = mujoco.Renderer(env.sim.model, 720, 720), mujoco.MjvCamera()
        top_camera(cam)
        writer = imageio.get_writer(video, fps=round(speed / dt), codec="libx264", macro_block_size=1)

        def frame():
            renderer.update_scene(env.sim.data, camera=cam)
            writer.append_data(renderer.render())

        episodes(frame)
        writer.close()
        print(f"wrote {video}")
        return

    with mujoco.viewer.launch_passive(env.sim.model, env.sim.data) as viewer:
        top_camera(viewer.cam)

        def frame():
            t0 = time.perf_counter()
            viewer.sync()
            time.sleep(max(0.0, dt / speed - (time.perf_counter() - t0)))

        episodes(frame, viewer.is_running)
        while viewer.is_running():  # keep the window open on the final state
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/ppo")
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--video", default=None, help="record an mp4 instead")
    ap.add_argument("--set", default="test", choices=list(TASKSETS))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--drag", type=float, default=1.0, help="drag scale, e.g. 0.7 or 1.3")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    a = ap.parse_args()
    if a.view or a.video:
        play(a.run, a.which, a.set, a.start, a.n, a.drag, a.speed, a.video)
    else:
        report(a.run, a.which)


if __name__ == "__main__":
    main()
