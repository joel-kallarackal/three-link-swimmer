"""The world around the swimmer: body geometry and collisions, obstacle layouts,
start/goal sampling, and the geodesic distance-to-goal used for reward shaping.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .config import SwimmerConfig


# ------------------------------------------------------------------ geometry
def link_segments(q: np.ndarray, L: float) -> np.ndarray:
    """(3, 2, 2): [rear end, front end] of each link's center line, head link first."""
    x, y, th, q1, q2 = q[:5]
    segs = np.empty((3, 2, 2))
    front = np.array([x, y]) + L * np.array([np.cos(th), np.sin(th)])  # head tip
    joint = np.array([x, y])
    for i, phi in enumerate((th, th + q1, th + q1 + q2)):
        t = L * np.array([np.cos(phi), np.sin(phi)])
        if i == 0:
            segs[0] = (joint, front)
        else:
            segs[i] = (joint - t, joint)
            joint = joint - t
    return segs


def body_clearances(q: np.ndarray, circles: np.ndarray, L: float, r_link: float) -> np.ndarray:
    """Signed distance between the swimmer body (3 capsules) and each circle (k,)."""
    if len(circles) == 0:
        return np.zeros(0)
    segs = link_segments(q, L)
    a, b, p = segs[:, 0], segs[:, 1], circles[:, :2]
    ab = b - a
    t = np.clip(((p[None] - a[:, None]) * ab[:, None]).sum(-1) / (ab * ab).sum(-1)[:, None], 0, 1)
    d = np.linalg.norm(p[None] - (a[:, None] + t[..., None] * ab[:, None]), axis=-1)  # (3, k)
    return d.min(0) - circles[:, 2] - r_link


def body_points(q: np.ndarray, L: float) -> np.ndarray:
    """7 points along the body center line, head tip to tail tip, 0.15 m apart:
    head tip, link0 middle, joint 1, link1 middle, joint 2, link2 middle, tail tip."""
    segs = link_segments(q, L)  # (3, [rear, front], 2), head link first
    ends = np.array([segs[0, 1], segs[0, 0], segs[1, 0], segs[2, 0]])  # head, j1, j2, tail
    mids = segs.mean(axis=1)
    return np.array([ends[0], mids[0], ends[1], mids[1], ends[2], mids[2], ends[3]])


def point_clearances(pts: np.ndarray, circles: np.ndarray, arena_half: float, r_link: float) -> np.ndarray:
    """Clearance of each body point (with the link radius) to the nearest obstacle or wall (n,)."""
    c = arena_half - np.abs(pts).max(axis=1)
    if len(circles):
        d = np.linalg.norm(pts[:, None] - circles[None, :, :2], axis=-1) - circles[None, :, 2]
        c = np.minimum(c, d.min(axis=1))
    return c - r_link


def wall_clearances(q: np.ndarray, arena_half: float, L: float, r_link: float) -> np.ndarray:
    """Body clearance to the [+x, -x, +y, -y] walls."""
    pts = link_segments(q, L).reshape(-1, 2)
    return np.array([arena_half - pts[:, 0].max(), arena_half + pts[:, 0].min(),
                     arena_half - pts[:, 1].max(), arena_half + pts[:, 1].min()]) - r_link


# ------------------------------------------------------------ geodesic field
class GeodesicField:
    """Backward Dijkstra from the goal on a 16-connected grid (5 cm) whose
    obstacles are inflated by `inflate`. Knight moves keep the metrication
    error below ~3 % (8-connected would be ~8 %)."""

    RES = 0.05

    def __init__(self, circles: np.ndarray, goal: np.ndarray, inflate: float, arena_half: float):
        self.A, res = arena_half, self.RES
        self.n = n = int(round(2 * arena_half / res)) + 1
        xs = -arena_half + res * np.arange(n)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        free = (np.abs(X) <= arena_half - inflate) & (np.abs(Y) <= arena_half - inflate)
        for cx, cy, r in circles:
            free &= (X - cx) ** 2 + (Y - cy) ** 2 > (r + inflate) ** 2
        self.free = free

        idx = np.arange(n * n).reshape(n, n)
        fr = free.ravel()
        s2, s5 = res * np.sqrt(2), res * np.sqrt(5)
        # (di, dj, cost, cells a knight move passes over and must also be free)
        moves = [(1, 0, res, []), (0, 1, res, []), (1, 1, s2, []), (1, -1, s2, []),
                 (1, 2, s5, [(0, 1), (1, 1)]), (2, 1, s5, [(1, 0), (1, 1)]),
                 (2, -1, s5, [(1, 0), (1, -1)]), (1, -2, s5, [(0, -1), (1, -1)])]
        rows, cols, wts = [], [], []
        for di, dj, c, via in moves:
            a = idx[max(0, -di):n - max(0, di), max(0, -dj):n - max(0, dj)]
            b = idx[max(0, di):n + min(0, di), max(0, dj):n + min(0, dj)]
            ok = fr[a.ravel()] & fr[b.ravel()]
            for vi, vj in via:
                ok &= fr[(a + vi * n + vj).ravel()]
            rows.append(a.ravel()[ok])
            cols.append(b.ravel()[ok])
            wts.append(np.full(ok.sum(), c))
        G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n * n, n * n)).tocsr()
        self.dist = dijkstra(G, directed=False, indices=self._nearest_free(goal)).reshape(n, n)

        # Reward shaping must always be finite (inf - inf = NaN poisons PPO):
        # blocked / disconnected cells get the nearest reachable value + the gap.
        fin = np.isfinite(self.dist)
        gap, (ii, jj) = distance_transform_edt(~fin, return_indices=True)
        self.shaping = np.where(fin, self.dist, self.dist[ii, jj] + res * gap)

    def _ij(self, p) -> tuple[int, int]:
        i = int(np.clip(round((p[0] + self.A) / self.RES), 0, self.n - 1))
        j = int(np.clip(round((p[1] + self.A) / self.RES), 0, self.n - 1))
        return i, j

    def _nearest_free(self, p) -> int:
        i, j = self._ij(p)
        if not self.free[i, j]:
            fi, fj = np.nonzero(self.free)
            k = np.argmin((fi - i) ** 2 + (fj - j) ** 2)
            i, j = fi[k], fj[k]
        return i * self.n + j

    def __call__(self, p) -> float:
        """Distance-to-goal [m] (bilinear interpolation, always finite)."""
        u = min(max((p[0] + self.A) / self.RES, 0.0), self.n - 1 - 1e-9)
        v = min(max((p[1] + self.A) / self.RES, 0.0), self.n - 1 - 1e-9)
        i, j = int(u), int(v)
        fu, fv, c = u - i, v - j, self.shaping
        return float((1 - fu) * ((1 - fv) * c[i, j] + fv * c[i, j + 1])
                     + fu * ((1 - fv) * c[i + 1, j] + fv * c[i + 1, j + 1]))

    def reachable(self, p) -> bool:
        i, j = self._ij(p)
        return bool(self.free[i, j] and np.isfinite(self.dist[i, j]))


# --------------------------------------------------------------------- tasks
@dataclass(frozen=True)
class LayoutSpec:
    n_obstacles: tuple[int, int] = (0, 10)  # inclusive
    radius: tuple[float, float] = (0.15, 0.5)  # m
    min_start_goal: float = 2.0  # m, head to goal
    goal_clearance: float = 0.35  # m, goal to obstacle surface
    start_clearance: float = 0.15  # m, body to obstacles and walls
    inflate: float = 0.12  # m, geodesic grid inflation (link radius + margin)
    wall_margin: float = 0.35  # m, keep obstacles off the walls
    p_blocked: float = 0.0


@dataclass
class Task:
    circles: np.ndarray  # (k, 3) [x, y, radius]
    start_qpos: np.ndarray  # (5,)
    goal: np.ndarray  # (2,)
    field: GeodesicField | None = field(default=None, repr=False)

    def copy(self) -> "Task":
        return replace(self, circles=self.circles.copy(), start_qpos=self.start_qpos.copy(),
                       goal=self.goal.copy())


def sample_circles(rng: np.random.Generator, spec: LayoutSpec, arena_half: float) -> np.ndarray:
    """A random obstacle layout (k, 3) [x, y, radius] (overlaps allowed)."""
    n = rng.integers(spec.n_obstacles[0], spec.n_obstacles[1] + 1)
    rad = rng.uniform(*spec.radius, n)
    lo = -(arena_half - spec.wall_margin) + rad
    return np.c_[rng.uniform(lo, -lo), rng.uniform(lo, -lo), rad].reshape(-1, 3)


def line_blocked(a: np.ndarray, b: np.ndarray, circles: np.ndarray, margin: float) -> bool:
    """Does the straight segment a->b pass within `margin` of an obstacle?"""
    if not len(circles):
        return False
    ab = b - a
    s = np.clip(((circles[:, :2] - a) @ ab) / (ab @ ab), 0.0, 1.0)
    d = np.linalg.norm(circles[:, :2] - (a + s[:, None] * ab), axis=1)
    return bool((d < circles[:, 2] + margin).any())


def _block_line(rng, circles, head, goal, q, spec, cfg) -> np.ndarray | None:
    """Move one random obstacle onto the head->goal line (centre 35-65% of the
    way along, sideways offset up to half its radius, so either side can be the
    shorter detour). None if the moved obstacle breaks a layout constraint."""
    L, r, A = cfg.link_length, cfg.link_radius, cfg.arena_half
    circles = circles.copy()
    k = rng.integers(len(circles))
    ab = goal - head
    perp = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
    rad = circles[k, 2]
    c = head + rng.uniform(0.35, 0.65) * ab + rng.uniform(-0.5, 0.5) * rad * perp
    circles[k, :2] = c
    if (np.abs(c).max() + rad > A - spec.wall_margin
            or np.linalg.norm(c - goal) - rad < spec.goal_clearance
            or body_clearances(q, circles, L, r).min() < spec.start_clearance):
        return None
    return circles


def sample_start_goal(rng: np.random.Generator, circles: np.ndarray, spec: LayoutSpec,
                      cfg: SwimmerConfig) -> Task | None:
    """A feasible start pose and goal on a given layout, or None if this
    attempt failed (goal too close to an obstacle, no valid start, unreachable)."""
    L, r, A = cfg.link_length, cfg.link_radius, cfg.arena_half
    lim = A - 0.5
    goal = rng.uniform(-lim, lim, 2)
    if len(circles) and (np.linalg.norm(circles[:, :2] - goal, axis=1) - circles[:, 2]).min() < spec.goal_clearance:
        return None
    for _ in range(50):
        head = rng.uniform(-lim, lim, 2)
        th = rng.uniform(-np.pi, np.pi)
        q = np.r_[head - L * np.array([np.cos(th), np.sin(th)]), th, 0.0, 0.0]
        if (np.linalg.norm(head - goal) >= spec.min_start_goal
                and wall_clearances(q, A, L, r).min() >= spec.start_clearance
                and (len(circles) == 0 or body_clearances(q, circles, L, r).min() >= spec.start_clearance)):
            break
    else:
        return None
    force = spec.p_blocked >= 1.0 or (spec.p_blocked > 0.0 and rng.random() < spec.p_blocked)
    if force and len(circles) and not line_blocked(head, goal, circles, r):
        circles = _block_line(rng, circles, head, goal, q, spec, cfg)
        if circles is None:
            return None
    fld = GeodesicField(circles, goal, spec.inflate, A)
    return Task(circles, q, goal, fld) if fld.reachable(head) else None


def sample_task(rng: np.random.Generator, spec: LayoutSpec, cfg: SwimmerConfig) -> Task:
    """A random layout with a feasible start and goal (goal reachable in the grid)."""
    for _ in range(200):
        task = sample_start_goal(rng, sample_circles(rng, spec, cfg.arena_half), spec, cfg)
        if task is not None:
            return task
    raise RuntimeError("could not sample a feasible task")


TASKSETS = {
    "val": (1_000_000, LayoutSpec()),
    "test": (2_000_000, LayoutSpec()),
    # out of distribution: denser and larger obstacles than anything in training
    "ood": (3_000_000, LayoutSpec(n_obstacles=(10, 14), radius=(0.2, 0.6))),
    # every layout has an obstacle across the straight start->goal line
    "blocked": (4_000_000, LayoutSpec(n_obstacles=(1, 10), p_blocked=1.0)),
    # sparse setting (training with at most 5 obstacles) and its clutter failure case
    "val_sparse": (5_000_000, LayoutSpec(n_obstacles=(0, 5))),
    "test_sparse": (6_000_000, LayoutSpec(n_obstacles=(0, 5))),
    "blocked_sparse": (7_000_000, LayoutSpec(n_obstacles=(1, 5), p_blocked=1.0)),
    "dense": (8_000_000, LayoutSpec(n_obstacles=(6, 10), p_blocked=0.5)),
}


def make_taskset(name: str, n: int, cfg: SwimmerConfig | None = None) -> list[Task]:
    seed0, spec = TASKSETS[name]
    cfg = cfg or SwimmerConfig()
    return [sample_task(np.random.default_rng(seed0 + i), spec, cfg) for i in range(n)]
