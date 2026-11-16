"""Generate maps and their graph of convex sets, and store them for planners.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import SwimmerConfig  # noqa: E402
from .gcs import GcsMap  # noqa: E402
from .world import LayoutSpec, TASKSETS, make_taskset, sample_circles  # noqa: E402


def head_position(qpos: np.ndarray, L: float) -> np.ndarray:
    return qpos[:2] + L * np.array([np.cos(qpos[2]), np.sin(qpos[2])])


def save_with_preview(m: GcsMap, path: Path, start=None, goal=None) -> None:
    m.save(path)
    fig, ax = plt.subplots(figsize=(5, 5))
    m.plot(ax, start=start, goal=goal,
           title=f"{path.stem}: {len(m.circles)} obstacles, {len(m.regions)} regions, "
                 f"{len(m.edges)} edges, {m.coverage:.0%} covered, {m.build_seconds:.2f} s")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=100)
    plt.close(fig)
    print(f"{path}: {len(m.circles):2d} obstacles | {len(m.regions):2d} regions {len(m.edges):3d} edges "
          f"{len(m.connected_components())} component(s) | coverage {m.coverage:.3f} | "
          f"built in {m.build_seconds:.2f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--seed", type=int, default=0, help="random map seed")
    src.add_argument("--taskset", choices=list(TASKSETS), help="build maps for evaluation tasks")
    ap.add_argument("--obstacles", type=int, nargs=2, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--radius", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--start", type=int, default=0, help="first task index (with --taskset)")
    ap.add_argument("--count", type=int, default=1, help="number of tasks (with --taskset)")
    ap.add_argument("--out", default=None, help="output file (random map)")
    ap.add_argument("--out-dir", default="maps", help="output directory (task maps)")
    ap.add_argument("--coverage", type=float, default=0.97, help="target free-area coverage")
    ap.add_argument("--inflate", type=float, default=0.12, help="obstacle/wall inflation [m]")
    args = ap.parse_args()

    cfg = SwimmerConfig()
    build = dict(inflate=args.inflate, coverage_target=args.coverage)

    if args.taskset:
        tasks = make_taskset(args.taskset, args.start + args.count, cfg)[args.start:]
        for k, t in enumerate(tasks, start=args.start):
            head = head_position(t.start_qpos, cfg.link_length)
            meta = {"source": f"taskset:{args.taskset}", "task_index": k,
                    "start_qpos": t.start_qpos.tolist(), "start_head": head.tolist(),
                    "goal": t.goal.tolist()}
            m = GcsMap.build(t.circles, cfg.arena_half, meta=meta, **build)
            save_with_preview(m, Path(args.out_dir) / f"{args.taskset}_{k:03d}.json", head, t.goal)
        return

    spec = LayoutSpec()
    if args.obstacles:
        spec = replace(spec, n_obstacles=tuple(args.obstacles))
    if args.radius:
        spec = replace(spec, radius=tuple(args.radius))
    circles = sample_circles(np.random.default_rng(args.seed), spec, cfg.arena_half)
    meta = {"source": "random", "seed": args.seed,
            "layout_spec": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(spec).items()}}
    m = GcsMap.build(circles, cfg.arena_half, meta=meta, **build)
    save_with_preview(m, Path(args.out or f"maps/random_{args.seed:03d}.json"))


if __name__ == "__main__":
    main()
