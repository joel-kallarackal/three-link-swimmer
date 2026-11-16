"""All physical parameters of the swimmer, in one place."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ThrusterConfig:
    """Tail thruster (see thruster.py): forward only, with a first-order spin-up lag."""

    max_force: float = 3.0  # N, at u = 1
    time_constant: float = 0.15  # s, spin-up lag; 0 = instantaneous

    @property
    def has_dynamics(self) -> bool:
        return self.time_constant > 0


@dataclass(frozen=True)
class SwimmerConfig:
    # body: 3 capsule links, neutrally buoyant
    link_length: float = 0.30  # m
    link_radius: float = 0.04  # m
    density: float = 1000.0  # kg/m^3 (water)
    hinge_armature: float = 0.005  # kg m^2
    joint_limit: float = 2.0  # rad, hard stop
    joint_target_limit: float = 1.4  # rad, commanded target range

    # joint position servos: tau = kp (q* - q) - kv qdot
    kp: float = 4.0  # N m / rad
    kv: float = 0.4  # N m s / rad

    # thruster on the tail tip, along the tail axis
    thruster: ThrusterConfig = field(default_factory=ThrusterConfig)

    # anisotropic drag per unit length: f = -(c_lin v + c_quad |v| v), split into
    # tangential and normal components; c_quad = 0.5 rho Cd D
    cd_normal: float = 1.2
    cd_tangential: float = 0.15
    c_lin_normal: float = 2.0  # N s / m^2
    c_lin_tangential: float = 0.25  # N s / m^2
    n_quad_points: int = 4  # Gauss-Legendre points per link
    abs_smoothing: float = 1e-3  # m/s, |v| ~ sqrt(v^2 + eps^2)

    # simulation
    timestep: float = 0.01  # s
    frame_skip: int = 10  # control at 10 Hz
    arena_half: float = 3.0  # m, square tank [-3, 3]^2

    @property
    def control_dt(self) -> float:
        return self.timestep * self.frame_skip

    def with_drag_scale(self, normal: float, tangential: float | None = None) -> "SwimmerConfig":
        """Copy with drag coefficients multiplied (robustness experiments)."""
        tangential = normal if tangential is None else tangential
        return replace(self,
                       cd_normal=self.cd_normal * normal, c_lin_normal=self.c_lin_normal * normal,
                       cd_tangential=self.cd_tangential * tangential,
                       c_lin_tangential=self.c_lin_tangential * tangential)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SwimmerConfig":
        """Build from a config-file dict: {'thruster': {...}, other fields...}."""
        d = dict(d or {})
        thruster = ThrusterConfig(**d.pop("thruster", {}) or {})
        return cls(thruster=thruster, **d)
