"""Plan a collision-free head path on a stored map with Drake GCS (Bezier
curves + convex optimization). Used as the initial guess for
trajectory optimization.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .gcs import GcsMap, region_polygon  # noqa: E402
from .world import GeodesicField  # noqa: E402


def clearance_along(m: GcsMap, pos: np.ndarray) -> np.ndarray:
    """Distance from each trajectory point to the nearest inflated obstacle or wall."""
    c = m.lim - np.abs(pos).max(axis=1)
    if len(m.circles):
        d = np.linalg.norm(pos[:, None] - m.circles[None, :, :2], axis=-1) - m.circles[None, :, 2] - m.inflate
        c = np.minimum(c, d.min(axis=1))
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--start", type=float, nargs=2, default=None)
    ap.add_argument("--goal", type=float, nargs=2, default=None)
    ap.add_argument("--order", type=int, default=5, help="Bezier degree per region")
    ap.add_argument("--continuity", type=int, default=2, help="C^k continuity between segments")
    ap.add_argument("--vmax", type=float, default=0.5, help="per-axis speed bound [m/s]")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    m = GcsMap.load(a.map)
    start = np.array(a.start if a.start else m.meta["start_head"])
    goal = np.array(a.goal if a.goal else m.meta["goal"])
    plan = m.plan_bezier(start, goal, order=a.order, continuity=a.continuity, v_max=a.vmax)
    if plan is None:
        raise SystemExit("GCS found no path")

    t, pos, vel = plan.sample(1000)
    clr = clearance_along(m, pos)
    geo = GeodesicField(m.circles, goal, m.inflate, m.arena_half)(start)
    print(f"map {a.map}: {len(m.circles)} obstacles, {len(m.regions)} regions, {len(m.edges)} edges")
    print(f"start {start}  goal {goal}")
    print(f"GCS solve {plan.solve_seconds:.2f} s | duration {plan.duration:.2f} s | length {plan.length:.2f} m "
          f"(grid geodesic {geo:.2f} m) | {len(plan.segments)} Bezier segments, degree {a.order}, C{a.continuity}")
    owners = [m.regions_containing_all(seg.control_points().T) for seg in plan.segments]
    print(f"min clearance to inflated obstacles {clr.min():.3f} m (must be >= 0)")
    print("segment -> regions containing all its control points: "
          + ", ".join(f"{i}:{o}" for i, o in enumerate(owners)))
    if not all(owners):
        print("WARNING: a segment is not certified inside any single region")

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.03, 0.05, 0.55, 0.88])
    m.plot(ax, start=start, goal=goal)
    for seg, own in zip(plan.segments, owners):
        # outline a region that contains ALL of the segment's control points (by
        # the convex-hull property the whole segment then lies inside it)
        cp = seg.control_points().T
        if own:
            ax.add_patch(Polygon(region_polygon(m.regions[own[0]]), fc="none", ec="#ad1457", lw=1.5, zorder=3))
        ax.plot(*cp.T, "o", ms=2.5, color="#6a1b9a", zorder=5)
    ax.plot(*pos.T, color="#ad1457", lw=2.5, zorder=6)
    ax.set_title(f"GCS Bezier path: {plan.length:.2f} m, {len(plan.segments)} segments, "
                 f"solve {plan.solve_seconds:.1f} s", fontsize=10)

    cx = fig.add_axes([0.65, 0.3, 0.32, 0.4])
    cx.plot(t, clr)
    cx.axhline(0, color="#c62828", lw=0.8)
    cx.set_xlabel("time along GCS trajectory [s]")
    cx.set_ylabel("m")
    cx.set_title("clearance to inflated obstacles / walls", fontsize=10)

    out = Path(a.out or f"figures/plan_{Path(a.map).stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
