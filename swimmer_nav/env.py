from __future__ import annotations

from dataclasses import dataclass, field, replace

import gymnasium as gym
import numpy as np

from .config import SwimmerConfig
from .physics import SwimmerSim
from .world import (GeodesicField, LayoutSpec, Task, body_clearances, body_points,
                    point_clearances, sample_task, wall_clearances)

N_VISIBLE = 3  # obstacles the robot can see
SLOT = 5  # features per visible obstacle


@dataclass
class EnvParams:
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    goal_tolerance: float = 0.25  # m, head to goal
    max_steps: int = 400  # 40 s
    # reward
    progress: str = "geodesic"  # "geodesic" (obstacle-aware) | "euclidean" (straight line)
    shaping_inflate: float | None = None
    w_progress: float = 1.0  # per meter of progress toward the goal
    r_success: float = 10.0
    r_collision: float = -10.0
    w_thrust: float = 0.01  # x (mean |F| / F_max)^2 per step
    w_torque: float = 0.005  # x sum (tau / 4 Nm)^2 per step
    w_action_rate: float = 0.01
    w_time: float = 0.005
    clearance: str = "closest"
    w_clearance: float = 0.1
    w_approach: float = 0.0
    approach_rate: float = 1.0  # 1/s
    safe_distance: float = 0.3  # m
    p_hardest: float = 0.0
    drag_scale_range: tuple[float, float] | None = None
    current_max: float = 0.0  # m/s


class SwimmerNavEnv(gym.Env):
    metadata = {"render_modes": []}
    obs_dim = 3 + 4 + 7 + 3 + 1 + 4 + N_VISIBLE * SLOT
    OBS0 = 22  # first obstacle slot

    def __init__(self, params: EnvParams | None = None, cfg: SwimmerConfig | None = None):
        self.p = params or EnvParams()
        self.cfg = cfg or SwimmerConfig()
        self.sim = SwimmerSim(self.cfg)
        self.A = self.cfg.arena_half
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.max_obstacles = self.p.layout.n_obstacles[1]  # curriculum hook
        self.task: Task | None = None

    def set_max_obstacles(self, n: int) -> None:
        """Curriculum: subsequent episodes sample 0..n obstacles."""
        self.max_obstacles = int(n)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "task" in options:
            task = options["task"]
        else:
            k = self.max_obstacles
            lo = k if self.np_random.random() < self.p.p_hardest else self.p.layout.n_obstacles[0]
            spec = replace(self.p.layout, n_obstacles=(lo, k))
            task = sample_task(self.np_random, spec, self.cfg)
        if task.field is None:
            task.field = GeodesicField(task.circles, task.goal, self.p.layout.inflate, self.A)
        self.task = task

        rng = self.np_random
        if "drag_scale" in options:
            scale = options["drag_scale"]
        elif self.p.drag_scale_range:
            scale = rng.uniform(*self.p.drag_scale_range, 2)
        else:
            scale = (1.0, 1.0)
        self.sim.set_drag(self.cfg.with_drag_scale(*scale))
        if "current" in options:
            self.sim.current = np.asarray(options["current"], float)
        else:
            a = rng.uniform(-np.pi, np.pi)
            self.sim.current = rng.uniform(0, self.p.current_max) * np.array([np.cos(a), np.sin(a)])

        self.sim.reset(task.start_qpos)
        self.sim.show_obstacles(task.circles)
        self.sim.show_goal(task.goal, self.p.goal_tolerance)
        self.steps = 0
        self.prev_action = np.zeros(3)
        self._head = self.sim.head_pos()
        self._build_shaping_field()
        self.d_goal = self._distance_to_goal(self._head)
        self.d_geo_start = task.field(self._head)  # path-efficiency metric
        self.path_length = 0.0
        self._update_clearances()
        self._h = self.min_clearance()
        return self._obs(), {}

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        p, sim, dt = self.p, self.sim, self.cfg.control_dt
        imp0, eff0 = sim.thrust_impulse, sim.joint_effort
        sim.step(a[:2] * self.cfg.joint_target_limit, a[2])
        self.steps += 1

        head = sim.head_pos()
        self.path_length += float(np.linalg.norm(head - self._head))
        self._head = head
        d_new = self._distance_to_goal(head)
        progress, self.d_goal = self.d_goal - d_new, d_new
        f_mean = (sim.thrust_impulse - imp0) / dt
        tau2_mean = (sim.joint_effort - eff0) / dt
        reward = (p.w_progress * progress
                  - p.w_thrust * (f_mean / self.cfg.thruster.max_force) ** 2
                  - p.w_torque * tau2_mean / 16.0
                  - p.w_action_rate * float(np.sum((a - self.prev_action) ** 2))
                  - p.w_time)
        self.prev_action = a

        h_prev = self._h
        self._update_clearances()
        reward -= self.clearance_penalty()
        self._h = self.min_clearance()
        if p.w_approach > 0:
            h_dot = (self._h - h_prev) / dt
            reward -= p.w_approach * max(0.0, -(h_dot + p.approach_rate * max(self._h, 0.0)))
        success = bool(np.linalg.norm(head - self.task.goal) < p.goal_tolerance)
        collision = bool(self._walls.min() < 0 or (len(self._clr) and self._clr.min() < 0))
        terminated = success or collision
        truncated = not terminated and self.steps >= p.max_steps
        reward += p.r_success * success + p.r_collision * collision
        if not np.isfinite(reward):
            raise FloatingPointError(f"non-finite reward at head={head}")

        info = {}
        if terminated or truncated:
            info = {"success": success, "collision": collision, "timeout": truncated,
                    "time": self.steps * dt, "thrust_impulse": sim.thrust_impulse,
                    "joint_effort": sim.joint_effort, "path_length": self.path_length,
                    "geodesic_start": self.d_geo_start, "n_obstacles": len(self.task.circles)}
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------ observation
    def _build_shaping_field(self) -> None:
        si = self.p.shaping_inflate
        if self.p.progress != "geodesic" or si is None or si == self.p.layout.inflate:
            self._shaping = self.task.field
        else:
            self._shaping = GeodesicField(self.task.circles, self.task.goal, si, self.A)

    def _distance_to_goal(self, head) -> float:
        if self.p.progress == "geodesic":
            return self._shaping(head)
        if self.p.progress == "euclidean":
            return float(np.linalg.norm(self.task.goal - head))
        raise ValueError(f"unknown progress {self.p.progress!r}")

    def clearance_penalty(self) -> float:
        """w * (1 - c / d_safe)^2 for the closest body point, or summed over 7 body points."""
        p, cfg = self.p, self.cfg
        if p.clearance == "closest":
            c = max(self.min_clearance(), 0.0)
            return p.w_clearance * (1.0 - c / p.safe_distance) ** 2 if c < p.safe_distance else 0.0
        if p.clearance != "per_point":
            raise ValueError(f"unknown clearance {p.clearance!r}")
        pts = body_points(self.sim.data.qpos, cfg.link_length)
        c = np.clip(point_clearances(pts, self.task.circles, self.A, cfg.link_radius), 0.0, None)
        near = c < p.safe_distance
        return float(p.w_clearance * np.sum((1.0 - c[near] / p.safe_distance) ** 2))

    def min_clearance(self) -> float:
        """Closest distance from the body to any obstacle or wall [m]."""
        return float(min(self._walls.min(), self._clr.min() if len(self._clr) else np.inf))

    def _update_clearances(self) -> None:
        q, cfg = self.sim.data.qpos, self.cfg
        self._clr = body_clearances(q, self.task.circles, cfg.link_length, cfg.link_radius)
        self._walls = wall_clearances(q, self.A, cfg.link_length, cfg.link_radius)

    def _obs(self) -> np.ndarray:
        A, L = self.A, self.cfg.link_length
        q, qd = self.sim.data.qpos, self.sim.data.qvel
        c, s = np.cos(q[2]), np.sin(q[2])
        to_head = np.array([[c, s], [-s, c]])  # world -> head frame
        head = self._head
        g = to_head @ (self.task.goal - head)
        v_link0 = qd[:2] + qd[2] * 0.5 * L * np.array([-s, c])

        obs = np.zeros(self.obs_dim, np.float32)
        obs[0:2] = g / A
        obs[2] = np.hypot(*g) / (2 * A)
        obs[3:5] = head / A
        obs[5:7] = c, s
        obs[7:9] = to_head @ v_link0
        obs[9:14] = qd[2], q[3], q[4], qd[3], qd[4]
        obs[14:17] = self.prev_action
        obs[17] = self.sim.thrust / self.cfg.thruster.max_force
        obs[18:22] = self._walls / A
        if len(self._clr):
            near = np.argsort(self._clr)[:N_VISIBLE]
            circles = self.task.circles[near]
            slots = obs[self.OBS0:].reshape(N_VISIBLE, SLOT)
            k = len(near)
            slots[:k, 0:2] = (circles[:, :2] - head) @ to_head.T / A
            slots[:k, 2] = circles[:, 2]
            slots[:k, 3] = self._clr[near] / A
            slots[:k, 4] = 1.0
        return obs

    # ----------------------------------------- mid-episode changes (scenarios)
    def set_goal(self, goal) -> None:
        self.task.goal = np.asarray(goal, float)
        self.task.field = GeodesicField(self.task.circles, self.task.goal, self.p.layout.inflate, self.A)
        self._build_shaping_field()
        self.sim.show_goal(self.task.goal, self.p.goal_tolerance)
        self.d_goal = self._distance_to_goal(self._head)

    def set_circles(self, circles) -> None:
        self.task.circles = np.asarray(circles, float).reshape(-1, 3)
        self.task.field = GeodesicField(self.task.circles, self.task.goal, self.p.layout.inflate, self.A)
        self._build_shaping_field()
        self.sim.show_obstacles(self.task.circles)
        self._update_clearances()
        self._h = self.min_clearance()
        self.d_goal = self._distance_to_goal(self._head)
