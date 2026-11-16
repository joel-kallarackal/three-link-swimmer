"""Train PPO policy.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from .config import SwimmerConfig
from .env import EnvParams, SwimmerNavEnv
from .evaluate import run_episodes, summarize
from .world import LayoutSpec, make_taskset


def make_env(seed: int, params: EnvParams, cfg: SwimmerConfig):
    def thunk():
        env = Monitor(SwimmerNavEnv(params, cfg), info_keywords=("success", "n_obstacles"))
        env.reset(seed=seed)
        return env
    return thunk


def make_vecnormalize(raw, gamma: float, init: Path | None = None, which: str = "last",
                      reset_reward_stats: bool = False) -> VecNormalize:
    """New VecNormalize, or the exact one saved with a run (fine-tuning)."""
    if init is None:
        return VecNormalize(raw, gamma=gamma, clip_obs=10.0)
    full = init / f"{which}_vecnormalize.pkl"
    if full.exists():
        venv = VecNormalize.load(str(full), raw)
        venv.training, venv.norm_reward = True, True
        print(f"restored VecNormalize from {full}: obs count {venv.obs_rms.count:.0f}, "
              f"return var {float(venv.ret_rms.var):.3g}")
    else:  # runs saved before full VecNormalize checkpoints: observation stats only
        with open(init / f"{which}_obs_rms.pkl", "rb") as f:
            rms = pickle.load(f)
        venv = VecNormalize(raw, gamma=gamma, clip_obs=10.0)
        venv.obs_rms.mean, venv.obs_rms.var = rms["mean"].copy(), rms["var"].copy()
        venv.obs_rms.count = 1e6  # unknown true count: trust the loaded statistics, adapt slowly
        print(f"WARNING: {full.name} not found; restored observation mean/var only "
              f"(count assumed, reward statistics fresh)")
    if reset_reward_stats:
        venv.ret_rms = RunningMeanStd(shape=())
        venv.returns = np.zeros(venv.num_envs)
        print("reward normalization statistics reset")
    return venv


class SuccessGatedCurriculum:
    """Level k = episodes have up to k obstacles. Advance one level when the
    success rate over the last `window` episodes with exactly k obstacles
    reaches the level's threshold, which falls linearly from `threshold` at
    level 0 to `threshold_min` at `max_level` (denser layouts are inherently
    harder; a fixed bar stalled the curriculum at 5-6 obstacles). The first
    `skip` episodes after a change are ignored: they were already running
    (sampled at the old level) when the level changed."""

    def __init__(self, max_level: int, threshold: float, window: int, skip: int, start_level: int = 0,
                 threshold_min: float | None = None):
        self.max_level, self.skip = max_level, skip
        self.threshold_max = threshold
        self.threshold_min = threshold if threshold_min is None else threshold_min
        self.level = start_level
        self.results: deque[bool] = deque(maxlen=window)
        self._to_skip = 0

    @property
    def threshold(self) -> float:
        """Success needed to leave the current level."""
        frac = self.level / max(self.max_level, 1)
        return self.threshold_max - (self.threshold_max - self.threshold_min) * frac

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.results)) if self.results else 0.0

    def record(self, success: bool, n_obstacles: int) -> bool:
        """Record a finished training episode; returns True if the level went up."""
        if self._to_skip > 0:
            self._to_skip -= 1
            return False
        if n_obstacles != self.level:  # only the hardest case counts toward the gate
            return False
        self.results.append(bool(success))
        if (self.level < self.max_level and len(self.results) == self.results.maxlen
                and self.success_rate >= self.threshold):
            self.level += 1
            self.results.clear()
            self._to_skip = self.skip
            return True
        return False


class CurriculumAndValidation(BaseCallback):
    def __init__(self, out: Path, curriculum: SuccessGatedCurriculum, eval_every: int,
                 cfg: SwimmerConfig, val_set: str = "val"):
        super().__init__()
        self.cfg, self.out, self.cur = cfg, out, curriculum
        self.max_obstacles = curriculum.max_level
        self.eval_every, self.next_eval = eval_every, eval_every
        self.best, self.t0 = -1.0, time.time()
        self.val_tasks = make_taskset(val_set, 100, cfg)
        self.log = open(out / "progress.csv", "w", newline="")
        self.csv = csv.writer(self.log)
        self.csv.writerow(["steps", "minutes", "max_obstacles", "train_success", "val_success",
                           "val_collision", "val_timeout", "val_time_to_goal_s"])

    @property
    def level(self) -> int:
        return self.cur.level

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_max_obstacles", self.level)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info and self.cur.record(info["success"], info["n_obstacles"]):
                self.training_env.env_method("set_max_obstacles", self.level)
                print(f"[curriculum] level {self.level}: 0-{self.level} obstacles at "
                      f"{self.num_timesteps:,} steps (next gate {self.cur.threshold:.2f})", flush=True)
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            self.validate()
        return True

    def save(self, tag: str) -> None:
        self.model.save(self.out / f"{tag}_model.zip")
        self.training_env.save(str(self.out / f"{tag}_vecnormalize.pkl"))  # full VecNormalize state

    def validate(self) -> None:
        venv: VecNormalize = self.training_env
        act = lambda obs: self.model.predict(venv.normalize_obs(obs), deterministic=True)[0]  # noqa: E731
        s = summarize(run_episodes(act, self.val_tasks, cfg=self.cfg))
        train_succ = float(np.mean([e["success"] for e in self.model.ep_info_buffer] or [0]))
        minutes = (time.time() - self.t0) / 60
        self.csv.writerow([self.num_timesteps, f"{minutes:.1f}", self.level, f"{train_succ:.3f}",
                           s["success"], s["collision"], s["timeout"], s.get("time_to_goal_s", "")])
        self.log.flush()
        for k in ("success", "collision", "timeout"):
            self.logger.record(f"val/{k}", s[k])
        self.logger.record("train/max_obstacles", self.level)
        print(f"[val] {self.num_timesteps:>10,} steps {minutes:6.1f} min | obstacles 0-{self.level:<2d} "
              f"(level succ {self.cur.success_rate:.2f}/{self.cur.threshold:.2f}) | train succ {train_succ:.2f} | val succ {s['success']:.2f} coll {s['collision']:.2f} "
              f"tout {s['timeout']:.2f} T {s.get('time_to_goal_s', float('nan')):.1f}s", flush=True)
        self.save("last")
        self.last_val = s["success"]
        # only checkpoints trained on the full obstacle range compete for "best"
        if self.level == self.max_obstacles and s["success"] > self.best:
            self.best = s["success"]
            self.save("best")

    def _on_training_end(self) -> None:
        self.validate()
        if self.best < 0:  # never reached the full obstacle range
            print(f"WARNING: curriculum stopped at level {self.level}/{self.max_obstacles}; "
                  "saving the final policy as best", flush=True)
            self.best = self.last_val
            self.save("best")
        self.log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--steps", type=float, default=None, help="override total steps")
    ap.add_argument("--out", default=None, help="override output directory")
    ap.add_argument("--init", default=None, help="fine-tune: start from this run's weights")
    ap.add_argument("--init-which", default="last", choices=["best", "last"])
    ap.add_argument("--start-level", type=int, default=None, help="initial curriculum level")
    ap.add_argument("--reset-reward-stats", action="store_true",
                    help="fine-tune: restart reward normalization (reward function changed)")
    args = ap.parse_args()

    c = yaml.safe_load(open(args.config))
    tc, pc = c["train"], c["ppo"]
    if args.init is None and tc.get("init"):  # fine-tuning given in the config (e.g. configs/final/)
        args.init = tc["init"]["run"]
        args.init_which = tc["init"].get("which", "last")
        args.reset_reward_stats = bool(tc["init"].get("reset_reward_stats", False))
    total = int(args.steps or tc["total_steps"])
    out = Path(args.out or tc["out"])
    out.mkdir(parents=True, exist_ok=True)
    c["train"]["init"] = ({"run": args.init, "which": args.init_which,
                           "reset_reward_stats": args.reset_reward_stats} if args.init else None)
    if args.start_level is not None:
        c["curriculum"]["start_level"] = args.start_level
    (out / "config.yaml").write_text(yaml.safe_dump(c))
    torch.set_num_threads(tc["torch_threads"])

    cc = c["curriculum"]
    params = EnvParams(**c.get("reward", {}), p_hardest=cc["p_hardest"],
                       layout=LayoutSpec(**{k: tuple(v) if isinstance(v, list) else v
                                            for k, v in c.get("layout", {}).items()}))
    cfg = SwimmerConfig.from_dict(c.get("physics"))
    print(f"env params: {params}")
    print(f"thruster: {cfg.thruster}")
    raw = SubprocVecEnv([make_env(tc["seed"] * 1000 + i, params, cfg) for i in range(tc["n_envs"])])
    venv = make_vecnormalize(raw, pc["gamma"], args.init and Path(args.init), args.init_which,
                             args.reset_reward_stats)
    model = PPO(
        "MlpPolicy", venv, seed=tc["seed"], device="cpu", verbose=0, tensorboard_log=str(out / "tb"),
        n_steps=pc["n_steps"], batch_size=pc["batch_size"], n_epochs=pc["n_epochs"],
        learning_rate=pc["learning_rate"], gamma=pc["gamma"], gae_lambda=pc["gae_lambda"],
        clip_range=pc["clip_range"], ent_coef=pc["ent_coef"], target_kl=pc["target_kl"],
        policy_kwargs=dict(net_arch=dict(pi=pc["net"], vf=pc["net"]), activation_fn=torch.nn.Tanh,
                           log_std_init=pc["log_std_init"]),
    )
    curriculum = SuccessGatedCurriculum(params.layout.n_obstacles[1], cc["threshold"], cc["window"],
                                        skip=tc["n_envs"], start_level=cc.get("start_level", 0),
                                        threshold_min=cc.get("threshold_min"))
    if args.init:
        init = Path(args.init)
        model.policy.load_state_dict(PPO.load(init / f"{args.init_which}_model.zip", device="cpu").policy.state_dict())
        print(f"fine-tuning from {init}/{args.init_which}_model.zip, curriculum level {curriculum.level}")
    cb = CurriculumAndValidation(out, curriculum, tc["eval_every"], cfg, tc.get("val_set", "val"))
    t0 = time.time()
    model.learn(total_timesteps=total, callback=cb)
    minutes = (time.time() - t0) / 60
    (out / "train_summary.json").write_text(json.dumps(
        {"total_steps": total, "train_minutes": minutes, "best_val_success": cb.best,
         "final_curriculum_level": cb.level}, indent=2))
    print(f"done: {total:,} steps in {minutes:.1f} min, best validation success {cb.best:.2f}")
    venv.close()


if __name__ == "__main__":
    main()
