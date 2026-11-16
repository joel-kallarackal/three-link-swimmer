"""Full-body initial guess from a 2D GCS head path ("follow the leader").
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SwimmerConfig
from .gcs import BezierPlan, GcsMap
from .world import body_clearances, link_segments, wall_clearances


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def body_points(q: np.ndarray, L: float) -> np.ndarray:
    """(4, 2): head tip, joint 1, joint 2, tail tip."""
    segs = link_segments(q, L)  # [rear, front] per link, head link first
    return np.array([segs[0, 1], segs[0, 0], segs[1, 0], segs[2, 0]])


def config_from_points(P: np.ndarray) -> np.ndarray:
    """q from the 4 body points (head, joint 1, joint 2, tail)."""
    head, j1, j2, tail = P
    th = np.arctan2(*(head - j1)[::-1])
    a1 = np.arctan2(*(j1 - j2)[::-1])  # heading of link 1 = theta + phi1
    a2 = np.arctan2(*(j2 - tail)[::-1])  # heading of link 2 = theta + phi1 + phi2
    return np.r_[j1, th, wrap(a1 - th), wrap(a2 - a1)]


class Polyline:
    """A path parametrized by arc length."""

    def __init__(self, pts: np.ndarray):
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
        self.p = pts[keep]
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))]

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length)
        return np.array([np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])])

    def behind(self, s: float, dist: float) -> float:
        """Arc length of the first point behind s (going backwards) at straight-line
        distance `dist` from the point at s."""
        p0 = self.at(s)
        i = np.searchsorted(self.s, s) - 1  # last vertex before s
        prev_s, prev_d = s, 0.0
        while i >= 0:
            d = np.linalg.norm(self.p[i] - p0)
            if d >= dist:  # crossing between vertex i and the previous sample: bisect
                lo, hi = self.s[i], prev_s
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if np.linalg.norm(self.at(mid) - p0) >= dist:
                        lo = mid
                    else:
                        hi = mid
                return 0.5 * (lo + hi)
            prev_s, prev_d = self.s[i], d
            i -= 1
        raise ValueError("path too short behind this point")


@dataclass
class BodyGuess:
    t: np.ndarray  # (K,) times [s]
    q: np.ndarray  # (K, 5) configurations
    qd: np.ndarray  # (K, 5) finite-difference velocities
    speed: float
    path_length: float

    def points(self, L: float) -> np.ndarray:
        """(K, 4, 2) head, joint 1, joint 2, tail for every knot."""
        return np.array([body_points(q, L) for q in self.q])


def follow_the_leader(plan: BezierPlan, q_start: np.ndarray, speed: float = 0.4, dt: float = 0.1,
                      L: float = 0.3) -> BodyGuess:
    """Body poses every dt seconds with the head moving along the GCS path at `speed`."""
    _, head_path, _ = plan.sample(2000)
    start_pts = body_points(q_start, L)  # head, j1, j2, tail
    behind_tail = start_pts[3] + (start_pts[3] - start_pts[2])
    poly = Polyline(np.r_[behind_tail[None], start_pts[:0:-1], head_path])
    s0 = poly.length - Polyline(head_path).length  # arc length where the head path starts
    S = poly.length - s0
    n = int(np.ceil(S / speed / dt))
    t = np.arange(n + 1) * dt
    q = []
    for tk in t:
        s_head = s0 + min(speed * tk, S)
        s1 = poly.behind(s_head, L)
        s2 = poly.behind(s1, L)
        s3 = poly.behind(s2, L)
        q.append(config_from_points(np.array([poly.at(s_head), poly.at(s1), poly.at(s2), poly.at(s3)])))
    q = np.array(q)
    q[:, 2] = np.unwrap(q[:, 2])  # continuous heading (no jumps at +-pi)
    qd = np.gradient(q, dt, axis=0)
    return BodyGuess(t, q, qd, speed, S)


def check(guess: BodyGuess, m: GcsMap, cfg: SwimmerConfig) -> dict:
    """Kinematic validity of the guess."""
    L, r = cfg.link_length, cfg.link_radius
    P = guess.points(L)
    link_ok = np.zeros((len(P), 3), bool)
    for k, pts in enumerate(P):
        for i in range(3):  # link i spans points i and i+1
            link_ok[k, i] = bool(m.regions_containing_all(pts[i:i + 2]))
    clr = np.array([min(wall_clearances(q, cfg.arena_half, L, r).min(),
                        body_clearances(q, m.circles, L, r).min() if len(m.circles) else np.inf)
                    for q in guess.q])
    return {"max_abs_joint": float(np.abs(guess.q[:, 3:5]).max()), "joint_limit": cfg.joint_limit,
            "links_in_a_region": float(link_ok.mean()), "link_ok": link_ok,
            "min_true_clearance": float(clr.min()), "clearance": clr}


def plot(path, m: GcsMap, plan: BezierPlan, guess: BodyGuess, chk: dict, cfg: SwimmerConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    A, L = m.arena_half, cfg.link_length
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.add_patch(Rectangle((-A, -A), 2 * A, 2 * A, fc="#eef5f9", ec="#455a64", lw=2))
    for x, y, r in m.circles:
        ax.add_patch(Circle((x, y), r + m.inflate, fc="none", ec="#8d6e63", ls=":", lw=0.8))
        ax.add_patch(Circle((x, y), r, fc="#8d6e63", ec="#5d4037"))
    _, hp, _ = plan.sample(500)
    ax.plot(*hp.T, color="#7b1fa2", lw=1.2, label="GCS head path")
    P = guess.points(L)
    for k in range(0, len(P), 5):
        ok = chk["link_ok"][k].all()
        ax.plot(*P[k].T, "-", color="#2e7d32" if ok else "#c62828", lw=1.6, alpha=0.8)
        ax.plot(*P[k, 0], "o", color="#2e7d32" if ok else "#c62828", ms=2.5)
    ax.plot([], [], color="#2e7d32", label="body every 0.5 s: all links in a region")
    ax.plot([], [], color="#c62828", label="body: some link not in any region")
    ax.set_xlim(-A - 0.1, A + 0.1); ax.set_ylim(-A - 0.1, A + 0.1); ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"follow the leader at {guess.speed} m/s: {guess.t[-1]:.1f} s", fontsize=9)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(guess.t, guess.q[:, 3], label="phi1"); ax2.plot(guess.t, guess.q[:, 4], label="phi2")
    for s in (-1, 1):
        ax2.axhline(s * cfg.joint_limit, color="r", ls=":", lw=0.8)
        ax2.axhline(s * cfg.joint_target_limit, color="0.5", ls=":", lw=0.8)
    ax2.set_ylabel("joint angle [rad]"); ax2.legend(fontsize=7)
    ax2.set_title("red: joint limit, grey: servo target limit", fontsize=8)
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(guess.t, chk["clearance"], label="true body clearance [m]")
    ax3.plot(guess.t, chk["link_ok"].all(axis=1).astype(float) * 0.1, ".", ms=2, label="all links in a region (0.1 = yes)")
    ax3.axhline(0, color="r", lw=0.8); ax3.set_xlabel("t [s]"); ax3.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="2D GCS map (maps/...json)")
    ap.add_argument("--speed", type=float, default=0.4, help="head speed along the path [m/s]")
    ap.add_argument("--out", default=None, help="figure (default: <map>_body.png)")
    a = ap.parse_args()
    cfg = SwimmerConfig()
    m = GcsMap.load(a.map)
    q0, goal = np.array(m.meta["start_qpos"]), np.array(m.meta["goal"])
    plan = m.plan_bezier(np.array(m.meta["start_head"]), goal)
    if plan is None:
        raise SystemExit("GCS found no path")
    print(f"GCS head path: {plan.length:.2f} m, {len(plan.segments)} segments, solved in {plan.solve_seconds:.1f} s")
    guess = follow_the_leader(plan, q0, a.speed, cfg.control_dt, cfg.link_length)
    chk = check(guess, m, cfg)
    print(f"body guess: {len(guess.t)} knots over {guess.t[-1]:.1f} s at {a.speed} m/s")
    print(f"  max |joint angle| {chk['max_abs_joint']:.2f} rad (limit {cfg.joint_limit}, servo targets "
          f"{cfg.joint_target_limit})")
    print(f"  links with both ends in a common region: {chk['links_in_a_region']:.1%}")
    print(f"  min true body clearance {chk['min_true_clearance']:+.3f} m")
    out = a.out or str(Path(a.map).with_suffix("")) + "_body.png"
    plot(out, m, plan, guess, chk, cfg)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
