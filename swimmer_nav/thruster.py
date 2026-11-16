"""Thruster model
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .config import ThrusterConfig


@dataclass(frozen=True)
class Ops:
    fmin: Callable
    fmax: Callable
    exp: Callable


FLOAT_OPS = Ops(fmin=min, fmax=max, exp=math.exp)


def casadi_ops() -> Ops:
    import casadi as ca

    return Ops(fmin=ca.fmin, fmax=ca.fmax, exp=ca.exp)


def commanded_force(u, c: ThrusterConfig, ops: Ops = FLOAT_OPS):
    """Steady-state force for command u: forward only, u clipped to [0, 1]
    (the policy's action stays in [-1, 1]; any u <= 0 switches the thruster off)."""
    return ops.fmin(ops.fmax(u, 0.0), 1.0) * c.max_force


def thruster_step(F, u, dt, c: ThrusterConfig, ops: Ops = FLOAT_OPS):
    """Advance the thruster force F by dt under a held command u."""
    F_cmd = commanded_force(u, c, ops)
    if c.time_constant <= 0:
        return F_cmd
    return F + (1.0 - ops.exp(-dt / c.time_constant)) * (F_cmd - F)
