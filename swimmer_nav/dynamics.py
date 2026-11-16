"""CasADi model of the swimmer for trajectory optimization and LQR.
"""
from __future__ import annotations

from functools import cached_property

import casadi as ca
import numpy as np

from .config import SwimmerConfig

NQ, NX, NU = 5, 11, 3


def capsule_mass_inertia(L: float, r: float, rho: float) -> tuple[float, float]:
    """Mass and inertia about the transverse axis through the centre."""
    m_cyl = rho * np.pi * r * r * L
    m_hemi = rho * (2.0 / 3.0) * np.pi * r ** 3
    I_cyl = m_cyl * (r * r / 4.0 + L * L / 12.0)
    d = L / 2.0 + 3.0 * r / 8.0  # hemisphere COM offset from the capsule centre
    I_hemi = m_hemi * (83.0 / 320.0 * r * r + d * d)
    return m_cyl + 2 * m_hemi, I_cyl + 2 * I_hemi


def kinematics(q, L, cos=ca.cos, sin=ca.sin):
    """Link centres, link headings, head tip and tail tip (CasADi or numpy)."""
    x, y, th, q1, q2 = q[0], q[1], q[2], q[3], q[4]
    phi = [th, th + q1, th + q1 + q2]
    t = [(cos(p), sin(p)) for p in phi]
    c0 = (x + 0.5 * L * t[0][0], y + 0.5 * L * t[0][1])
    head = (x + L * t[0][0], y + L * t[0][1])
    c1 = (x - 0.5 * L * t[1][0], y - 0.5 * L * t[1][1])
    j2 = (x - L * t[1][0], y - L * t[1][1])
    c2 = (j2[0] - 0.5 * L * t[2][0], j2[1] - 0.5 * L * t[2][1])
    tail = (j2[0] - L * t[2][0], j2[1] - L * t[2][1])
    return [c0, c1, c2], phi, head, tail


class SwimmerModel:
    def __init__(self, cfg: SwimmerConfig | None = None):
        self.cfg = cfg or SwimmerConfig()

    @cached_property
    def f(self) -> ca.Function:
        """Continuous dynamics xdot = f(x, u)."""
        c = self.cfg
        L, r = c.link_length, c.link_radius
        m, I = capsule_mass_inertia(L, r, c.density)
        d = 2 * r
        lt, ln = c.c_lin_tangential, c.c_lin_normal
        qt, qn = 0.5 * c.density * c.cd_tangential * d, 0.5 * c.density * c.cd_normal * d
        eps2 = c.abs_smoothing ** 2
        xi, wi = np.polynomial.legendre.leggauss(c.n_quad_points)
        s_pts, w_pts = xi * L / 2, wi * L / 2

        x, u = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
        q, qd, F = x[:NQ], x[NQ:2 * NQ], x[2 * NQ]
        centers, phis, _, tail = kinematics(q, L)
        centers = [ca.vertcat(*p) for p in centers]

        # Lagrangian: kinetic energy of 3 rigid links + joint armature
        T = 0
        for p, phi in zip(centers, phis):
            vc, om = ca.jtimes(p, q, qd), ca.jtimes(phi, q, qd)
            T += 0.5 * m * ca.dot(vc, vc) + 0.5 * I * om * om
        T += 0.5 * c.hinge_armature * (qd[3] ** 2 + qd[4] ** 2)
        M = ca.hessian(T, qd)[0]
        bias = ca.jtimes(ca.gradient(T, qd), q, qd) - ca.gradient(T, q)

        # generalized forces: drag (quadrature points), servos, thruster
        Q = ca.SX.zeros(NQ)
        for p, phi in zip(centers, phis):
            vc, om = ca.jtimes(p, q, qd), ca.jtimes(phi, q, qd)
            tx, ty = ca.cos(phi), ca.sin(phi)
            for sk, wk in zip(s_pts, w_pts):
                pt = p + float(sk) * ca.vertcat(tx, ty)
                vx = vc[0] - om * float(sk) * ty
                vy = vc[1] + om * float(sk) * tx
                vt, vn = vx * tx + vy * ty, -vx * ty + vy * tx
                ft = -(lt + qt * ca.sqrt(vt * vt + eps2)) * vt * float(wk)
                fn = -(ln + qn * ca.sqrt(vn * vn + eps2)) * vn * float(wk)
                f_world = ca.vertcat(ft * tx - fn * ty, ft * ty + fn * tx)
                Q += ca.mtimes(ca.jacobian(pt, q).T, f_world)
        tau = c.kp * (u[0:2] - q[3:5]) - c.kv * qd[3:5]
        Q[3] += tau[0]
        Q[4] += tau[1]
        tail = ca.vertcat(*tail)
        Q += ca.mtimes(ca.jacobian(tail, q).T, F * ca.vertcat(ca.cos(phis[2]), ca.sin(phis[2])))

        qdd = ca.solve(M, Q - bias)
        thr = c.thruster
        Fdot = (thr.max_force * u[2] - F) / thr.time_constant if thr.has_dynamics else 0 * F
        return ca.Function("f", [x, u], [ca.vertcat(qd, qdd, Fdot)], ["x", "u"], ["xdot"])

    @cached_property
    def joint_torque(self) -> ca.Function:
        x, u = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
        c = self.cfg
        return ca.Function("tau", [x, u], [c.kp * (u[0:2] - x[3:5]) - c.kv * x[8:10]])

    @cached_property
    def head(self) -> ca.Function:
        q = ca.SX.sym("q", NQ)
        return ca.Function("head", [q], [ca.vertcat(*kinematics(q, self.cfg.link_length)[2])])

    def discrete(self, dt: float, n_sub: int = 10) -> ca.Function:
        """x+ = Phi(x, u): RK4 with n_sub substeps, u held (zero-order hold).
        dt / n_sub must stay <= ~0.02 s (the servo mode is stiff)."""
        x, u = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
        h, xk = dt / n_sub, x
        for _ in range(n_sub):
            k1 = self.f(xk, u)
            k2 = self.f(xk + 0.5 * h * k1, u)
            k3 = self.f(xk + 0.5 * h * k2, u)
            k4 = self.f(xk + h * k3, u)
            xk = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return ca.Function("Phi", [x, u], [xk], ["x", "u"], ["xnext"])


def sim_state(sim) -> np.ndarray:
    """Model state x from a SwimmerSim (qpos, qvel, thruster force)."""
    return np.r_[sim.data.qpos, sim.data.qvel, sim.thrust]
