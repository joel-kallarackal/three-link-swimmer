"""Swimmer dynamics: MuJoCo rigid bodies + anisotropic drag + tail thruster.
"""
from __future__ import annotations

import mujoco
import numba
import numpy as np

from .config import SwimmerConfig
from .thruster import thruster_step

MAX_OBSTACLES = 16  # visual obstacle geoms preallocated in the model


# ---------------------------------------------------------------------- model
def build_mjcf(cfg: SwimmerConfig) -> str:
    L, r, A = cfg.link_length, cfg.link_radius, cfg.arena_half
    jl, tl, w = cfg.joint_limit, cfg.joint_target_limit, 0.05
    hinge = f'type="hinge" axis="0 0 1" limited="true" range="{-jl} {jl}" armature="{cfg.hinge_armature}"'
    walls = "\n".join(
        f'<geom type="box" size="{sx} {sy} 0.08" pos="{px} {py} 0" rgba="0.3 0.35 0.4 1"/>'
        for sx, sy, px, py in [(A + w, w, 0, A + w), (A + w, w, 0, -A - w), (w, A + w, A + w, 0), (w, A + w, -A - w, 0)])
    obstacles = "\n".join(
        f'<geom name="obs{i}" type="cylinder" size="0.3 0.05" pos="0 0 -10" rgba="0.55 0.35 0.25 0"/>'
        for i in range(MAX_OBSTACLES))
    servo = f'ctrllimited="true" ctrlrange="{-tl} {tl}" gainprm="{cfg.kp}" biastype="affine" biasprm="0 {-cfg.kp} {-cfg.kv}"'
    return f"""
<mujoco model="swimmer">
  <compiler angle="radian"/>
  <option timestep="{cfg.timestep}" gravity="0 0 0" integrator="implicitfast"/>
  <visual><global offwidth="960" offheight="960"/></visual>
  <default><geom contype="0" conaffinity="0" density="{cfg.density}"/></default>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom type="plane" size="{A} {A} 0.1" pos="0 0 -0.1" rgba="0.8 0.92 0.97 1"/>
    {walls}
    {obstacles}
    <geom name="goal" type="cylinder" size="0.25 0.002" pos="0 0 -0.05" rgba="0.2 0.8 0.3 0.6"/>
    <body name="link0">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="{L} 0 0 0 0 0" size="{r}" rgba="0.95 0.6 0.1 1"/>
      <site name="head" pos="{L} 0 0"/>
      <body name="link1">
        <joint name="j1" {hinge}/>
        <geom type="capsule" fromto="0 0 0 {-L} 0 0" size="{r}" rgba="0.85 0.85 0.85 1"/>
        <body name="link2" pos="{-L} 0 0">
          <joint name="j2" {hinge}/>
          <geom type="capsule" fromto="0 0 0 {-L} 0 0" size="{r}" rgba="0.85 0.85 0.85 1"/>
          <site name="thruster" pos="{-L} 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="servo1" joint="j1" {servo}/>
    <general name="servo2" joint="j2" {servo}/>
    <general name="thrust" site="thruster" gear="1 0 0 0 0 0" ctrllimited="true"
             ctrlrange="{-cfg.thruster.max_force} {cfg.thruster.max_force}"/>
  </actuator>
</mujoco>"""


# ----------------------------------------------------------------------- drag
@numba.njit(cache=True)
def drag_wrenches(phis, v_centers, omegas, current, coef, s, w, F, tau):
    """Anisotropic drag on each link, integrated with Gauss-Legendre quadrature.

    phis (N,) link headings; v_centers (N,2) link-center velocities; omegas (N,)
    current (2,) water velocity; coef = [lin_t, lin_n, quad_t, quad_n, eps^2]
    s, w: quadrature points (offset from link center) and length weights.
    Writes F (N,2) world-frame force and tau (N,) torque about each link center.
    """
    for i in range(phis.shape[0]):
        c, sn = np.cos(phis[i]), np.sin(phis[i])
        fx = fy = tz = 0.0
        for k in range(s.shape[0]):
            # point velocity relative to the water: v_center + omega * s_k * n - current
            vx = v_centers[i, 0] - omegas[i] * s[k] * sn - current[0]
            vy = v_centers[i, 1] + omegas[i] * s[k] * c - current[1]
            vt = vx * c + vy * sn  # along the link
            vn = -vx * sn + vy * c  # across the link
            ft = -(coef[0] + coef[2] * np.sqrt(vt * vt + coef[4])) * vt * w[k]
            fn = -(coef[1] + coef[3] * np.sqrt(vn * vn + coef[4])) * vn * w[k]
            fx += ft * c - fn * sn
            fy += ft * sn + fn * c
            tz += s[k] * fn
        F[i, 0], F[i, 1], tau[i] = fx, fy, tz


# ------------------------------------------------------------------ simulator
class SwimmerSim:
    def __init__(self, cfg: SwimmerConfig | None = None):
        self.cfg = cfg or SwimmerConfig()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.bodies = np.array([self.model.body(f"link{i}").id for i in range(3)])
        self.obs_geoms = [self.model.geom(f"obs{i}").id for i in range(MAX_OBSTACLES)]
        self.goal_geom = self.model.geom("goal").id
        self.head_site = self.model.site("head").id
        xi, wi = np.polynomial.legendre.leggauss(self.cfg.n_quad_points)
        self._s, self._w = xi * self.cfg.link_length / 2, wi * self.cfg.link_length / 2
        self.current = np.zeros(2)
        self.set_drag(self.cfg)
        self._vel6, self._vc, self._om = np.zeros(6), np.zeros((3, 2)), np.zeros(3)
        self._F, self._tau = np.zeros((3, 2)), np.zeros(3)

    def set_drag(self, cfg: SwimmerConfig) -> None:
        """Drag coefficients from `cfg` (lets experiments perturb drag only)."""
        d = 2 * cfg.link_radius
        self.drag_cfg = cfg
        self._coef = np.array([cfg.c_lin_tangential, cfg.c_lin_normal,
                               0.5 * cfg.density * cfg.cd_tangential * d,
                               0.5 * cfg.density * cfg.cd_normal * d, cfg.abs_smoothing ** 2])

    def reset(self, qpos: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.ctrl[:2] = qpos[3:5]
        self.thrust = 0.0  # thruster force state [N]
        self.thrust_impulse = 0.0  # integral |F| dt         [N s]
        self.joint_effort = 0.0  # integral sum tau^2 dt    [N^2 m^2 s]
        mujoco.mj_forward(self.model, self.data)

    def _apply_drag(self) -> None:
        m, d = self.model, self.data
        for i, b in enumerate(self.bodies):
            # [angular; linear] velocity at the body COM (= link center), world frame
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, self._vel6, 0)
            self._om[i] = self._vel6[2]
            self._vc[i] = self._vel6[3:5]
        R = d.xmat[self.bodies]
        drag_wrenches(np.arctan2(R[:, 3], R[:, 0]), self._vc, self._om, self.current,
                      self._coef, self._s, self._w, self._F, self._tau)
        d.xfrc_applied[self.bodies, 0:2] = self._F
        d.xfrc_applied[self.bodies, 5] = self._tau

    def step(self, joint_targets, thrust_cmd: float) -> None:
        """Hold joint targets [rad] and thrust command [-1, 1] for one control
        period. The thruster force follows its spin-up lag every substep."""
        m, d = self.model, self.data
        tl, dt, u = self.cfg.joint_target_limit, m.opt.timestep, float(thrust_cmd)
        thr = self.cfg.thruster
        d.ctrl[0:2] = np.clip(joint_targets, -tl, tl)
        if not thr.has_dynamics:  # no lag: constant over the period, compute once
            self.thrust = thruster_step(self.thrust, u, dt, thr)
        for _ in range(self.cfg.frame_skip):
            if thr.has_dynamics:
                self.thrust = thruster_step(self.thrust, u, dt, thr)
            f = self.thrust
            d.ctrl[2] = f
            mujoco.mj_step1(m, d)  # kinematics and velocities at the current state
            self._apply_drag()
            mujoco.mj_step2(m, d)
            tau = d.actuator_force[0:2]
            self.thrust_impulse += abs(f) * dt
            self.joint_effort += float(tau @ tau) * dt
        mujoco.mj_kinematics(m, d)

    # ------------------------------------------------------------ accessors
    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel.copy()

    def head_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.head_site, :2].copy()

    # -------------------------------------------------- visuals (no physics)
    def show_obstacles(self, circles: np.ndarray) -> None:
        for i, g in enumerate(self.obs_geoms):
            if i < len(circles):
                self.model.geom_pos[g] = (circles[i, 0], circles[i, 1], 0.0)
                self.model.geom_size[g, 0] = circles[i, 2]
                self.model.geom_rgba[g, 3] = 1.0
            else:
                self.model.geom_pos[g] = (0, 0, -10)
                self.model.geom_rgba[g, 3] = 0.0
        mujoco.mj_kinematics(self.model, self.data)  # world-frame geom poses for rendering

    def show_goal(self, goal: np.ndarray, radius: float) -> None:
        self.model.geom_pos[self.goal_geom] = (goal[0], goal[1], -0.05)
        self.model.geom_size[self.goal_geom, 0] = radius
        mujoco.mj_kinematics(self.model, self.data)
