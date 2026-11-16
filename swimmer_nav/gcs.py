"""Graph of convex sets (GCS) over the swimmer's free space, built offline and
stored with its map.
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron, Point, VPolytope
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import MathematicalProgram, ScsSolver

logging.getLogger("drake").setLevel(logging.WARNING)  # silence per-solve INFO lines

POLYGON_SIDES = 16  # obstacles are circumscribed polygons (conservative)
FORMAT_VERSION = 1
ROUNDED_PATHS = 200  # GCS rounding budget (offline: quality over speed)


def solve_options(rounded_paths: int = ROUNDED_PATHS) -> GraphOfConvexSetsOptions:
    """GCS is solved as a convex relaxation + randomized rounding (no licensed
    MICP solver). With many overlapping regions the relaxation is loose and
    Drake's default 5 rounded paths miss the best route (up to 1.68x the
    geodesic on dense maps); 200 brings every tested map within ~2 %."""
    o = GraphOfConvexSetsOptions()
    o.max_rounded_paths = rounded_paths
    o.max_rounding_trials = 4 * rounded_paths
    return o


# ====================================================================== 2D IRIS
def obstacle_polygons(circles: np.ndarray, inflate: float) -> list[np.ndarray]:
    """Each inflated circle as a regular polygon (n, 2) that *contains* it
    (vertex radius R / cos(pi/n)), so regions avoiding the polygons are
    guaranteed to avoid the circles."""
    n = POLYGON_SIDES
    ang = 2 * np.pi * np.arange(n) / n
    unit = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return [np.array([x, y]) + (r + inflate) / np.cos(np.pi / n) * unit for x, y, r in circles]


def _closest_point_to_origin(poly: np.ndarray) -> np.ndarray:
    """Closest point of a convex polygon (ordered vertices) to the origin,
    assuming the origin is outside it."""
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(-(a * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-18), 0.0, 1.0)
    p = a + t[:, None] * ab
    return p[np.argmin((p * p).sum(1))]


def _separating_hyperplanes(polys, C, d, lim):
    """IRIS step 1: for the ellipse {C u + d : |u| <= 1}, add hyperplanes tangent
    to scaled copies of it that separate it from every obstacle, nearest
    obstacle (in the ellipse metric) first. The box [-lim, lim]^2 is the domain."""
    Cinv = np.linalg.inv(C)
    A = [np.array([1.0, 0]), np.array([-1.0, 0]), np.array([0, 1.0]), np.array([0, -1.0])]
    b = [lim] * 4
    remaining = list(range(len(polys)))
    ball = [(P - d) @ Cinv.T for P in polys]  # obstacles in ellipse coordinates
    while remaining:
        near = [_closest_point_to_origin(ball[k]) for k in remaining]
        k = int(np.argmin([q @ q for q in near]))
        x_star = C @ near[k] + d  # closest obstacle point, world frame
        a = Cinv.T @ Cinv @ (x_star - d)  # normal of the scaled ellipse at x_star
        a /= np.linalg.norm(a)
        A.append(a)
        b.append(a @ x_star)
        # drop obstacles entirely on the far side of the new hyperplane
        remaining = [j for j in remaining if not np.all(polys[j] @ a >= b[-1] - 1e-12)]
    return np.array(A), np.array(b)


def _max_volume_ellipse(A: np.ndarray, b: np.ndarray):
    """IRIS step 2: largest ellipse {C u + d} inside {A x <= b} (log-det, SCS)."""
    prog = MathematicalProgram()
    C = prog.NewSymmetricContinuousVariables(2, "C")
    d = prog.NewContinuousVariables(2, "d")
    prog.AddMaximizeLogDeterminantCost(C)
    for ai, bi in zip(A, b):  # |C a_i| <= b_i - a_i . d
        prog.AddLorentzConeConstraint(np.r_[bi - ai @ d, C @ ai])
    result = ScsSolver().Solve(prog)
    if not result.is_success():
        raise RuntimeError(f"max-volume ellipse failed: {result.get_solution_result()}")
    return result.GetSolution(C), result.GetSolution(d)


def iris_2d(polys: list[np.ndarray], seed: np.ndarray, lim: float, max_iters: int = 10,
            rel_tol: float = 0.02) -> HPolyhedron:
    """Large convex obstacle-free region around `seed` inside [-lim, lim]^2
    (Deits & Tedrake 2015). The returned region always contains `seed`."""
    seed = np.asarray(seed, float)
    C, d = 1e-3 * np.eye(2), seed
    best, last_det = None, np.linalg.det(C)
    for _ in range(max_iters):
        A, b = _separating_hyperplanes(polys, C, d, lim)
        if not np.all(A @ seed <= b + 1e-9):
            break  # the region drifted off the seed: keep the last one containing it
        best = (A, b)
        C, d = _max_volume_ellipse(A, b)
        det = np.linalg.det(C)
        if det <= last_det * (1 + rel_tol):
            break
        last_det = det
    return HPolyhedron(*best)


# ======================================================= free-space sampling
def _free_mask(pts: np.ndarray, circles: np.ndarray, inflate: float) -> np.ndarray:
    if len(circles) == 0:
        return np.ones(len(pts), bool)
    d = np.linalg.norm(pts[:, None] - circles[None, :, :2], axis=-1)
    return np.all(d > circles[None, :, 2] + inflate, axis=1)


def _clearance(pts: np.ndarray, circles: np.ndarray, inflate: float, lim: float) -> np.ndarray:
    """Distance from each point to the nearest inflated obstacle or wall."""
    c = lim - np.abs(pts).max(axis=1)
    if len(circles):
        d = np.linalg.norm(pts[:, None] - circles[None, :, :2], axis=-1) - circles[None, :, 2] - inflate
        c = np.minimum(c, d.min(axis=1))
    return c


def medial_axis(circles: np.ndarray, lim: float, inflate: float, res: float = 0.05):
    """Grid points on the medial axis of free space (ridge of the clearance
    function: a local maximum along at least one of 4 directions), and their
    clearance. The medial axis passes through the middle of every gap."""
    xs = np.arange(-lim, lim + 1e-9, res)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel()]
    c = _clearance(pts, circles, inflate, lim).reshape(X.shape)
    P = np.pad(c, 1, constant_values=-np.inf)
    n = c.shape[0]
    ridge = np.zeros_like(c, bool)
    for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        fwd = P[1 + di:1 + di + n, 1 + dj:1 + dj + n]
        bwd = P[1 - di:1 - di + n, 1 - dj:1 - dj + n]
        ridge |= (c >= fwd) & (c >= bwd) & ((c > fwd) | (c > bwd))
    ridge &= c > 0.5 * res
    return pts[ridge.ravel()], c[ridge]


def _contains(region: HPolyhedron, pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    return np.all(pts @ region.A().T <= region.b() + tol, axis=1)


# ===================================================================== GcsMap
@dataclass
class GcsMap:
    arena_half: float  # m, square arena [-A, A]^2
    circles: np.ndarray  # (k, 3) [x, y, radius]
    inflate: float  # m, added to obstacles and walls for the head point
    regions: list[HPolyhedron] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)  # overlapping region pairs
    seeds: list[np.ndarray] = field(default_factory=list)
    coverage: float = 0.0  # fraction of sampled free area inside some region
    build_seconds: float = 0.0  # offline cost of the decomposition
    meta: dict = field(default_factory=dict)  # provenance: seed, layout spec, task, ...

    @property
    def lim(self) -> float:
        return self.arena_half - self.inflate

    # ------------------------------------------------------------- building
    @classmethod
    def build(cls, circles: np.ndarray, arena_half: float, inflate: float = 0.12,
              coverage_target: float = 0.97, max_regions: int = 80, n_samples: int = 4000,
              prune: bool = True, seed: int = 0, meta: dict | None = None) -> "GcsMap":
        t0 = time.perf_counter()
        circles = np.asarray(circles, float).reshape(-1, 3)
        m = cls(arena_half, circles, inflate, meta=dict(meta or {}))
        polys, lim = obstacle_polygons(circles, inflate), m.lim

        axis_pts, axis_clr = medial_axis(circles, lim, inflate)
        rng = np.random.default_rng(seed)
        smp = rng.uniform(-lim, lim, (n_samples, 2))
        smp = smp[_free_mask(smp, circles, inflate)]
        pts = np.r_[axis_pts, smp]  # everything we try to cover
        clr = np.r_[axis_clr, _clearance(smp, circles, inflate, lim)]
        is_axis = np.r_[np.ones(len(axis_pts), bool), np.zeros(len(smp), bool)]
        covered, failed = np.zeros(len(pts), bool), np.zeros(len(pts), bool)

        def grow(k: int) -> None:
            try:
                r = iris_2d(polys, pts[k], lim)
            except RuntimeError as e:  # numerical failure: skip this seed
                warnings.warn(f"IRIS failed at seed {pts[k]}: {e}")
                failed[k] = True
                return
            m.regions.append(r)
            m.seeds.append(pts[k])
            covered[:] |= _contains(r, pts)

        def next_seed(mask):  # uncovered point in `mask` with the largest clearance
            open_ = mask & ~covered & ~failed
            return int(np.argmax(np.where(open_, clr, -np.inf))) if open_.any() else None

        while len(m.regions) < max_regions and (k := next_seed(is_axis)) is not None:
            grow(k)  # 1. medial axis
        while (len(m.regions) < max_regions and covered[~is_axis].mean() < coverage_target
               and (k := next_seed(~is_axis)) is not None):
            grow(k)  # 2. area coverage
        if prune and len(m.regions) > 1:  # 3. set cover keeps everything covered so far
            m._prune(pts, covered)

        m.coverage = float(np.any([_contains(r, smp) for r in m.regions], axis=0).mean()) if m.regions else 0.0
        m._compute_edges()
        m.build_seconds = time.perf_counter() - t0
        m.meta.update({"coverage_target": coverage_target, "max_regions": max_regions,
                       "n_samples": n_samples, "prune": prune, "sample_seed": seed,
                       "medial_axis_points": int(len(axis_pts))})
        return m

    def _prune(self, pts: np.ndarray, target: np.ndarray) -> None:
        cover = np.array([_contains(r, pts) for r in self.regions])  # (R, N)
        keep, covered = [], np.zeros(len(pts), bool)
        while (target & ~covered).any():
            gain = (cover & ~covered).sum(axis=1)
            gain[keep] = -1
            k = int(np.argmax(gain))
            keep.append(k)
            covered |= cover[k]
        self.regions = [self.regions[k] for k in keep]
        self.seeds = [self.seeds[k] for k in keep]

    def _compute_edges(self) -> None:
        self.edges = [(i, j) for i in range(len(self.regions)) for j in range(i + 1, len(self.regions))
                      if self.regions[i].IntersectsWith(self.regions[j])]

    def is_free(self, p) -> bool:
        p = np.asarray(p, float)
        return bool(np.abs(p).max() <= self.lim and _free_mask(p[None], self.circles, self.inflate)[0])

    def regions_containing_all(self, pts, tol: float = 1e-6) -> list[int]:
        """Regions that contain every point in pts (k, 2). For a Bezier segment's
        control points this certifies the whole segment is in that region."""
        pts = np.atleast_2d(pts)
        return [k for k, r in enumerate(self.regions) if _contains(r, pts, tol).all()]

    def region_containing(self, p) -> int | None:
        for k, r in enumerate(self.regions):
            if r.PointInSet(np.asarray(p, float)):
                return k
        return None

    def ensure_contains(self, p) -> int:
        """Index of a region containing p, growing a new one (query time) if p
        lies in the small uncovered part of free space."""
        k = self.region_containing(p)
        if k is not None:
            return k
        if not self.is_free(p):
            raise ValueError(f"point {p} is inside an (inflated) obstacle or wall")
        r = iris_2d(obstacle_polygons(self.circles, self.inflate), p, self.lim)
        self.regions.append(r)
        self.seeds.append(np.asarray(p, float))
        n = len(self.regions) - 1
        self.edges += [(i, n) for i in range(n) if self.regions[i].IntersectsWith(r)]
        return n

    # ------------------------------------------------------------ planning
    def build_gcs(self, start, goal, order: int = 1):
        """Drake GcsTrajectoryOptimization over the stored regions, with start
        and goal as point vertices. Returns (gcs, source, regions, target)."""
        self.ensure_contains(start)
        self.ensure_contains(goal)
        gcs = GcsTrajectoryOptimization(2)
        both_ways = self.edges + [(j, i) for i, j in self.edges]  # Drake edges are directed
        regions = gcs.AddRegions(self.regions, both_ways, order=order)
        source = gcs.AddRegions([Point(np.asarray(start, float))], order=0)
        target = gcs.AddRegions([Point(np.asarray(goal, float))], order=0)
        gcs.AddEdges(source, regions)
        gcs.AddEdges(regions, target)
        return gcs, source, regions, target

    def shortest_path(self, start, goal, rounded_paths: int = ROUNDED_PATHS):
        """Minimum-length piecewise-linear head path (sanity check / route).
        Returns (points (N, 2) or None, solve seconds)."""
        gcs, source, _, target = self.build_gcs(start, goal, order=1)
        gcs.AddPathLengthCost()
        t0 = time.perf_counter()
        traj, result = gcs.SolvePath(source, target, solve_options(rounded_paths))
        dt = time.perf_counter() - t0
        if not result.is_success():
            return None, dt
        pts = np.array([traj.value(t).ravel() for t in np.linspace(traj.start_time(), traj.end_time(), 400)])
        return pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]], dt

    def plan_bezier(self, start, goal, order: int = 5, continuity: int = 2, v_max: float = 0.5,
                    w_time: float = 1.0, w_length: float = 1.0,
                    rounded_paths: int = ROUNDED_PATHS) -> "BezierPlan | None":
        """Collision-free head path through the regions (initial guess for
        trajectory optimization), with Drake's GcsTrajectoryOptimization.

        Each visited region gets a Bezier segment of degree `order` whose
        control points lie in that region; by the convex-hull property the
        whole segment is then inside the (convex, obstacle-free) region.
        Segments are joined C^`continuity` (in the path parameter: Drake 1.26
        does not enforce continuity of the time-scaled velocity, so speed may
        jump at region changes - irrelevant for an initial guess). Per-axis
        speed <= v_max; cost = w_time * T + w_length * length. Region choice
        and shape are one convex program (relaxation + rounding).
        """
        gcs, source, regions, target = self.build_gcs(start, goal, order=order)
        gcs.AddTimeCost(w_time)
        gcs.AddPathLengthCost(w_length)
        gcs.AddVelocityBounds(-v_max * np.ones(2), v_max * np.ones(2))
        for k in range(1, continuity + 1):
            gcs.AddPathContinuityConstraints(k)
        t0 = time.perf_counter()
        traj, result = gcs.SolvePath(source, target, solve_options(rounded_paths))
        if not result.is_success():
            return None
        return BezierPlan(traj, time.perf_counter() - t0, order, continuity, v_max)

    # ------------------------------------------------------------ storage
    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "map": {"arena_half": self.arena_half, "inflate": self.inflate,
                    "polygon_sides": POLYGON_SIDES,
                    "obstacles": [{"x": x, "y": y, "radius": r} for x, y, r in self.circles.tolist()]},
            "gcs": {"regions": [{"A": r.A().tolist(), "b": r.b().tolist()} for r in self.regions],
                    "edges": [list(e) for e in self.edges],
                    "seeds": [s.tolist() for s in self.seeds],
                    "coverage": self.coverage, "build_seconds": self.build_seconds},
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GcsMap":
        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported map format {d.get('format_version')}")
        mp, g = d["map"], d["gcs"]
        circles = np.array([[o["x"], o["y"], o["radius"]] for o in mp["obstacles"]], float).reshape(-1, 3)
        return cls(arena_half=mp["arena_half"], circles=circles, inflate=mp["inflate"],
                   regions=[HPolyhedron(np.array(r["A"]), np.array(r["b"])) for r in g["regions"]],
                   edges=[tuple(e) for e in g["edges"]], seeds=[np.array(s) for s in g["seeds"]],
                   coverage=g["coverage"], build_seconds=g["build_seconds"], meta=d.get("meta", {}))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GcsMap":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------- analysis
    def connected_components(self) -> list[set[int]]:
        parent = list(range(len(self.regions)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j in self.edges:
            parent[find(i)] = find(j)
        comps: dict[int, set[int]] = {}
        for i in range(len(self.regions)):
            comps.setdefault(find(i), set()).add(i)
        return list(comps.values())

    def plot(self, ax, start=None, goal=None, path=None, title=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon, Rectangle

        A = self.arena_half
        ax.add_patch(Rectangle((-A, -A), 2 * A, 2 * A, fc="#eef5f9", ec="#455a64", lw=2))
        cmap = plt.get_cmap("tab20")
        for k, r in enumerate(self.regions):
            ax.add_patch(Polygon(region_polygon(r), fc=cmap(k % 20), ec=cmap(k % 20), alpha=0.25, lw=1))
        for x, y, r in self.circles:
            ax.add_patch(Circle((x, y), r + self.inflate, fc="none", ec="#8d6e63", ls=":", lw=0.8))
            ax.add_patch(Circle((x, y), r, fc="#8d6e63", ec="#5d4037"))
        if self.regions:
            centers = np.array([r.ChebyshevCenter() for r in self.regions])
            for i, j in self.edges:
                ax.plot(*centers[[i, j]].T, color="#37474f", lw=0.6, alpha=0.6)
            ax.scatter(*centers.T, s=8, color="#37474f", zorder=3)
        if path is not None:
            ax.plot(*np.asarray(path).T, color="#ad1457", lw=2, zorder=4)
        if start is not None:
            ax.plot(*start, "o", color="#ef6c00", ms=8, zorder=5)
        if goal is not None:
            ax.plot(*goal, "*", color="#2e7d32", ms=14, zorder=5)
        ax.set_xlim(-A - 0.1, A + 0.1)
        ax.set_ylim(-A - 0.1, A + 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9)


@dataclass
class BezierPlan:
    """Result of GcsMap.plan_bezier: a piecewise Bezier head trajectory."""

    traj: object  # pydrake CompositeTrajectory of BezierCurves
    solve_seconds: float
    order: int
    continuity: int
    v_max: float

    @property
    def duration(self) -> float:
        return self.traj.end_time() - self.traj.start_time()

    @property
    def segments(self) -> list:
        """Bezier segments with nonzero duration (one per visited region)."""
        segs = [self.traj.segment(i) for i in range(self.traj.get_number_of_segments())]
        return [g for g in segs if g.end_time() - g.start_time() > 1e-9]

    def sample(self, n: int = 400):
        """Times (n,), positions (n, 2), velocities (n, 2)."""
        t = np.linspace(self.traj.start_time(), self.traj.end_time(), n)
        f = lambda k: np.array([self.traj.EvalDerivative(x, k).ravel() for x in t])  # noqa: E731
        return t, f(0), f(1)

    @property
    def length(self) -> float:
        _, p, _ = self.sample(2000)
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def region_polygon(region: HPolyhedron) -> np.ndarray:
    """Vertices of a 2D region in counter-clockwise order."""
    v = VPolytope(region).vertices().T
    c = v.mean(axis=0)
    return v[np.argsort(np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0]))]
