"""Kinodynamic planning: 2D GCS head path -> body guess -> direct collocation.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np

from .body_path import BodyGuess, body_points, follow_the_leader
from .config import SwimmerConfig
from .dynamics import NQ, NU, NX, SwimmerModel
from .gcs import BezierPlan, GcsMap
from .world import body_clearances, wall_clearances

RADAU_DEGREE = 3
SOLVED = ("Solve_Succeeded", "Solved_To_Acceptable_Level")


def radau_coefficients(d: int = RADAU_DEGREE):
    """Collocation matrices for Lagrange polynomials on [0, 1] (CasADi example)."""
    tau = np.r_[0.0, ca.collocation_points(d, "radau")]
    C, D = np.zeros((d + 1, d + 1)), np.zeros(d + 1)
    for j in range(d + 1):
        p = np.poly1d([1.0])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1.0, -tau[r]]) / (tau[j] - tau[r])
        D[j] = p(1.0)
        dp = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = dp(tau[r])
    return tau, C, D


def body_points_ca(q, L):
    """CasADi (2, 4): head tip, joint 1, joint 2, tail tip."""
    th, a1, a2 = q[2], q[2] + q[3], q[2] + q[3] + q[4]
    j1 = q[0:2]
    head = j1 + L * ca.vertcat(ca.cos(th), ca.sin(th))
    j2 = j1 - L * ca.vertcat(ca.cos(a1), ca.sin(a1))
    tail = j2 - L * ca.vertcat(ca.cos(a2), ca.sin(a2))
    return ca.horzcat(head, j1, j2, tail)


@dataclass
class Trajectory:
    """Optimized trajectory on the control grid (h = control period)."""
    h: float
    X: np.ndarray  # (N+1, 11) states at knots
    U: np.ndarray  # (N, 3) controls held over each interval
    link_regions: np.ndarray  # (N+1, 3) region index of each link at each knot
    goal: np.ndarray
    status: str = ""
    solve_seconds: float = 0.0
    cost: float = float("nan")
    meta: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return len(self.U)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.N + 1) * self.h

    @property
    def duration(self) -> float:
        return self.N * self.h

    def save(self, path) -> None:
        np.savez(path, h=self.h, X=self.X, U=self.U, link_regions=self.link_regions, goal=self.goal,
                 status=self.status, solve_seconds=self.solve_seconds, cost=self.cost,
                 iterations=self.meta.get("iterations", -1))

    @classmethod
    def load(cls, path) -> "Trajectory":
        z = np.load(path)
        return cls(float(z["h"]), z["X"], z["U"], z["link_regions"], z["goal"], str(z["status"]),
                   float(z["solve_seconds"]), float(z["cost"]),
                   meta={"iterations": int(z["iterations"]) if "iterations" in z else "?"})


def route_regions(m: GcsMap, plan: BezierPlan) -> list[int]:
    """Regions of the GCS route: every region that contains a whole segment."""
    out = []
    for g in plan.segments:
        for k in m.regions_containing_all(np.asarray(g.control_points()).T):
            if k not in out:
                out.append(k)
    return out


def assign_link_regions(m: GcsMap, route: list[int], P: np.ndarray) -> tuple[np.ndarray, float]:
    """P (K, 4, 2) body points per knot -> (K, 3) region per link, and the
    fraction of links that already fit their region in the guess."""
    K = len(P)
    reg, fits = np.zeros((K, 3), int), 0
    for k in range(K):
        for i in range(3):
            ends = P[k, i:i + 2]
            viol = [max(0.0, float((m.regions[r].A() @ ends.T - m.regions[r].b()[:, None]).max())) for r in route]
            j = int(np.argmin(viol))
            reg[k, i] = route[j]
            fits += viol[j] <= 1e-9
    return reg, fits / (3 * K)


class Collocation:
    def __init__(self, m: GcsMap, cfg: SwimmerConfig | None = None, goal_radius: float = 0.1,
                 w_thrust: float = 1.0, w_torque: float = 1.0, w_smooth: float = 0.1, joint_margin: float = 0.1):
        self.m, self.cfg = m, cfg or SwimmerConfig()
        self.model = SwimmerModel(self.cfg)
        self.goal_radius, self.joint_margin = goal_radius, joint_margin
        self.w = (w_thrust, w_torque, w_smooth)
        self.Ab = []  # row-normalized region inequalities
        for r in m.regions:
            n = np.linalg.norm(r.A(), axis=1)
            self.Ab.append((r.A() / n[:, None], r.b() / n))

    def solve(self, x0, goal, guess: BodyGuess, link_regions: np.ndarray, max_iter: int = 1000,
              verbose: bool = False) -> Trajectory:
        cfg, h = self.cfg, self.cfg.control_dt
        f, tau_fn = self.model.f, self.model.joint_torque
        L, Fmax = cfg.link_length, cfg.thruster.max_force
        tau, C, D = radau_coefficients()
        d = RADAU_DEGREE
        N = len(guess.t) - 1
        jl, tl = cfg.joint_limit - self.joint_margin, cfg.joint_target_limit
        w_thr, w_tq, w_du = self.w

        opti = ca.Opti()
        X = opti.variable(NX, N + 1)
        Xc = [opti.variable(NX, d) for _ in range(N)]
        U = opti.variable(NU, N)

        def in_regions(x, regs):  # each link's two end points inside its region
            P = body_points_ca(x[:NQ], L)
            for i in range(3):
                A_r, b_r = self.Ab[regs[i]]
                opti.subject_to(ca.vec(ca.mtimes(A_r, P[:, i:i + 2])) <= np.tile(b_r, 2))

        J = 0
        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            xk, uk = X[:, k], U[:, k]
            xs = [xk] + [Xc[k][:, j] for j in range(d)]
            for j in range(1, d + 1):
                opti.subject_to(h * f(xs[j], uk) == sum(C[r, j] * xs[r] for r in range(d + 1)))
            opti.subject_to(X[:, k + 1] == sum(D[r] * xs[r] for r in range(d + 1)))
            for j in range(d - 1):  # interior collocation points: regions of the interval's end knot
                in_regions(Xc[k][:, j], link_regions[k + 1])
            for xj in xs[1:]:
                opti.subject_to(opti.bounded(-jl, xj[3:5], jl))
            opti.subject_to(opti.bounded(-tl, uk[0:2], tl))
            opti.subject_to(opti.bounded(0.0, uk[2], 1.0))
            tq = tau_fn(Xc[k][:, d - 1], uk)
            J += h * (w_thr * (Xc[k][NX - 1, d - 1] / Fmax) ** 2 + w_tq * ca.sumsqr(tq))
            if k > 0:
                J += w_du * ca.sumsqr(U[:, k] - U[:, k - 1])
        for k in range(1, N + 1):  # knots (the start is fixed)
            in_regions(X[:, k], link_regions[k])
        head_N = body_points_ca(X[:NQ, N], L)[:, 0]
        opti.subject_to(ca.sumsqr(head_N - goal) <= self.goal_radius ** 2)
        opti.minimize(J)

        # initial guess: body guess with joints inside the servo range
        q = guess.q.copy()
        q[:, 3:5] = np.clip(q[:, 3:5], -tl, tl)
        Xg = np.c_[q, np.gradient(q, h, axis=0), np.full(N + 1, 0.5 * Fmax)]
        Xg[0] = x0
        opti.set_initial(X, Xg.T)
        for k in range(N):
            for j in range(d):
                opti.set_initial(Xc[k][:, j], (1 - tau[j + 1]) * Xg[k] + tau[j + 1] * Xg[k + 1])
        opti.set_initial(U, np.c_[q[1:, 3:5], np.full(N, 0.5)].T)

        opti.solver("ipopt", {"expand": True, "print_time": False},
                    {"max_iter": max_iter, "print_level": 5 if verbose else 0, "tol": 1e-6,
                     "acceptable_tol": 1e-4, "acceptable_iter": 10, "mu_strategy": "adaptive"})
        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            get = sol.value
        except RuntimeError:
            get = opti.debug.value
        status = opti.stats()["return_status"]
        return Trajectory(h, np.array(get(X)).T, np.array(get(U)).reshape(NU, N).T, link_regions,
                          np.asarray(goal, float), status, time.perf_counter() - t0, float(get(J)),
                          meta={"iterations": opti.stats()["iter_count"]})

    def verify(self, traj: Trajectory, tol: float = 1e-5) -> dict:
        """Re-check the solution at the knots: regions, joint limits, goal, true clearance."""
        cfg, L = self.cfg, self.cfg.link_length
        worst_region, worst_clr = -np.inf, np.inf
        for k in range(1, traj.N + 1):
            P = body_points(traj.X[k, :NQ], L)
            for i in range(3):
                A_r, b_r = self.Ab[traj.link_regions[k, i]]
                worst_region = max(worst_region, float((A_r @ P[i:i + 2].T - b_r[:, None]).max()))
            q = traj.X[k, :NQ]
            c = wall_clearances(q, cfg.arena_half, L, cfg.link_radius).min()
            if len(self.m.circles):
                c = min(c, body_clearances(q, self.m.circles, L, cfg.link_radius).min())
            worst_clr = min(worst_clr, c)
        goal_dist = float(np.linalg.norm(body_points(traj.X[-1, :NQ], L)[0] - traj.goal))
        joints = float(np.abs(traj.X[:, 3:5]).max())
        ok = (worst_region <= tol and joints <= cfg.joint_limit - self.joint_margin + tol
              and goal_dist <= self.goal_radius + tol)
        return {"ok": ok, "max_region_violation": worst_region, "min_true_clearance": worst_clr,
                "goal_distance": goal_dist, "max_abs_joint": joints}


def plan_trajectory(m: GcsMap, q_start, goal, cfg: SwimmerConfig | None = None, speed: float = 0.4,
                    verbose: bool = True, ipopt_verbose: bool = False, x0=None, **kw):
    """GCS -> body guess -> collocation. Returns (trajectory or None, info dict).
    x0: full initial state (11,) when starting in motion (replanning); default:
    at rest at q_start with the thruster off."""
    cfg = cfg or SwimmerConfig()
    q_start, goal = np.asarray(q_start, float), np.asarray(goal, float)
    L = cfg.link_length
    info = {}
    try:
        plan = m.plan_bezier(body_points(q_start, L)[0], goal)
    except ValueError as e:  # start or goal inside an (inflated) obstacle
        info["reason"] = f"GCS: {e}"
        return None, info
    if plan is None:
        info["reason"] = "GCS: no route through overlapping regions"
        return None, info
    route = route_regions(m, plan)
    guess = follow_the_leader(plan, q_start, speed, cfg.control_dt, L)
    regs, fit = assign_link_regions(m, route, guess.points(L))
    info.update(plan=plan, guess=guess, route=route, gcs_seconds=plan.solve_seconds, guess_fit=fit)
    if verbose:
        print(f"GCS: route through regions {route}, head path {plan.length:.2f} m, {plan.solve_seconds:.1f} s")
        print(f"guess: {len(guess.t) - 1} intervals ({guess.t[-1]:.1f} s at {speed} m/s), "
              f"{fit:.1%} of links already inside their assigned region")
    prob = Collocation(m, cfg, **kw)
    x0 = np.r_[q_start, np.zeros(NQ), 0.0] if x0 is None else np.asarray(x0, float)
    traj = prob.solve(x0, goal, guess, regs, verbose=ipopt_verbose)
    chk = prob.verify(traj)
    info.update(check=chk)
    if verbose:
        print(f"collocation: {traj.status} after {traj.meta['iterations']} iterations, "
              f"{traj.solve_seconds:.1f} s, cost {traj.cost:.3f}")
        print(f"  re-check: region violation {chk['max_region_violation']:.1e}, "
              f"max |joint| {chk['max_abs_joint']:.2f}, head-goal {chk['goal_distance']:.3f} m, "
              f"min true clearance {chk['min_true_clearance']:+.3f} m -> {'OK' if chk['ok'] else 'FAILED'}")
    info["candidate"] = traj
    if traj.status not in SOLVED or not chk["ok"]:
        info["reason"] = f"collocation: {traj.status}" if traj.status not in SOLVED else "re-check failed"
        return None, info
    return traj, info


# ------------------------------------------------------------- simulation
def rollout(traj: Trajectory, circles: np.ndarray, cfg: SwimmerConfig | None = None, controller=None,
            drag_scale: float = 1.0, extra_steps: int = 20, goal_tolerance: float = 0.25) -> dict:
    """Run the plan in MuJoCo. controller(k, x) -> u; default open loop (u = U_k).
    Success = head within goal_tolerance at any step, without collision."""
    from .dynamics import sim_state
    from .physics import SwimmerSim

    cfg = cfg or SwimmerConfig()
    L = cfg.link_length
    sim = SwimmerSim(cfg)
    sim.set_drag(cfg.with_drag_scale(drag_scale))
    sim.reset(traj.X[0, :NQ])
    xs, us, min_clr = [sim_state(sim)], [], np.inf
    res = dict(success=False, collision=False)
    for k in range(traj.N + extra_steps):
        u = traj.U[min(k, traj.N - 1)] if controller is None else controller(k, xs[-1])
        us.append(u)
        sim.step(u[:2], u[2])
        xs.append(sim_state(sim))
        q = sim.data.qpos
        c = wall_clearances(q, cfg.arena_half, L, cfg.link_radius).min()
        if len(circles):
            c = min(c, body_clearances(q, circles, L, cfg.link_radius).min())
        min_clr = min(min_clr, c)
        if c < 0:
            res["collision"] = True
            break
        if np.linalg.norm(sim.head_pos() - traj.goal) <= goal_tolerance:
            res["success"] = True
            break
    res.update(steps=len(us), X=np.array(xs), U=np.array(us), min_clearance=float(min_clr),
               final_goal_dist=float(np.linalg.norm(sim.head_pos() - traj.goal)),
               time=len(us) * traj.h, thrust_impulse=sim.thrust_impulse, joint_effort=sim.joint_effort,
               timeout=not res["success"] and not res["collision"])
    return res


# ------------------------------------------------------------------ figure
def plot(path, m: GcsMap, info: dict, traj: Trajectory, runs: dict | None = None, cfg=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    from .gcs import region_polygon

    cfg = cfg or SwimmerConfig()
    A, L = m.arena_half, cfg.link_length
    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.add_patch(Rectangle((-A, -A), 2 * A, 2 * A, fc="#eef5f9", ec="#455a64", lw=2))
    for k in info["route"]:
        ax.add_patch(Polygon(region_polygon(m.regions[k]), fc="#7b1fa2", alpha=0.06, ec="#7b1fa2", lw=0.6))
    for x, y, r in m.circles:
        ax.add_patch(Circle((x, y), r + m.inflate, fc="none", ec="#8d6e63", ls=":", lw=0.8))
        ax.add_patch(Circle((x, y), r, fc="#8d6e63", ec="#5d4037"))
    ax.add_patch(Circle(traj.goal, 0.25, fc="#2e7d32", alpha=0.2))
    _, hp, _ = info["plan"].sample(400)
    ax.plot(*hp.T, "--", color="#7b1fa2", lw=1, label="GCS head path")
    P = np.array([body_points(x[:NQ], L) for x in traj.X])
    ax.plot(*P[:, 0].T, color="#ef6c00", lw=2, label=f"optimized head, {traj.duration:.1f} s")
    for k in range(0, traj.N + 1, 5):
        ax.plot(*P[k].T, color="#ef6c00", lw=1, alpha=0.7)
    for name, r in (runs or {}).items():
        res = "success" if r["success"] else ("collision" if r["collision"] else "missed goal")
        Ps = np.array([body_points(x[:NQ], L)[0] for x in r["X"]])
        ax.plot(*Ps.T, lw=1, label=f"MuJoCo {name}: {res}")
    ax.set_xlim(-A - 0.1, A + 0.1); ax.set_ylim(-A - 0.1, A + 0.1); ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("route regions (purple), body every 0.5 s", fontsize=9)

    t, tk = traj.t[:-1], traj.t
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.step(t, traj.U[:, 0], where="post", label="phi1 target"); ax2.step(t, traj.U[:, 1], where="post", label="phi2 target")
    ax2.plot(tk, traj.X[:, 3], "C0:", lw=0.8, label="phi1"); ax2.plot(tk, traj.X[:, 4], "C1:", lw=0.8, label="phi2")
    ax2.set_ylabel("rad"); ax2.legend(fontsize=6, ncol=2)
    ax3 = fig.add_subplot(3, 2, 4)
    ax3.step(t, traj.U[:, 2], where="post", label="thrust command")
    ax3.plot(tk, traj.X[:, 10] / cfg.thruster.max_force, label="thrust force / max")
    ax3.legend(fontsize=7)
    ax4 = fig.add_subplot(3, 2, 6)
    ax4.plot(tk, np.linalg.norm(traj.X[:, 5:7], axis=1), label="speed of joint 1 [m/s]")
    ax4.set_xlabel("t [s]"); ax4.legend(fontsize=7)
    fig.suptitle(f"collocation: {traj.status}, {traj.solve_seconds:.1f} s, {traj.meta.get('iterations', '?')} iterations",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="2D GCS map (maps/...json)")
    ap.add_argument("--speed", type=float, default=0.4, help="head speed of the initial guess [m/s]")
    ap.add_argument("--verbose", action="store_true", help="print IPOPT iterations")
    ap.add_argument("--out", default=None, help="output prefix (default: next to the map)")
    a = ap.parse_args()
    cfg = SwimmerConfig()
    m = GcsMap.load(a.map)
    q0, goal = np.array(m.meta["start_qpos"]), np.array(m.meta["goal"])
    traj, info = plan_trajectory(m, q0, goal, cfg, speed=a.speed, ipopt_verbose=a.verbose)
    out = Path(a.out or Path(a.map).with_suffix(""))
    cand = info.get("candidate")
    if traj is None:
        print(f"NO PLAN: {info['reason']}")
        if cand is not None:
            plot(f"{out}_traj_failed.png", m, info, cand, cfg=cfg)
            print(f"saved {out}_traj_failed.png (the solver's last iterate)")
        return
    traj.save(f"{out}_traj.npz")
    r = rollout(traj, m.circles, cfg)
    print(f"open loop in MuJoCo: {'success' if r['success'] else 'collision' if r['collision'] else 'missed goal'} "
          f"after {r['steps'] * traj.h:.1f} s, head-goal {r['final_goal_dist']:.2f} m, "
          f"min clearance {r['min_clearance']:+.3f} m")
    plot(f"{out}_traj.png", m, info, traj, {"open loop": r}, cfg)
    print(f"saved {out}_traj.npz, {out}_traj.png")


if __name__ == "__main__":
    main()
