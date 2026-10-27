"""Six-axis diagonal admittance with the thesis' semi-implicit update.

The independent hard limits belong to the execution supervisor. This local
model does not establish stability of robot dynamics, delay or contact.
"""
from __future__ import annotations

import numpy as np


def _six(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite six-vector")
    return vector.copy()


def advance_scale(rho: float, low: float, high: float) -> float:
    """Unit nominal advance below low; smoothly zero at and above high."""
    if not np.all(np.isfinite([rho, low, high])) or rho < 0 or low < 0 or high <= low:
        raise ValueError("advance thresholds require finite rho>=0 and 0<=low<high")
    s = float(np.clip((rho - low) / (high - low), 0.0, 1.0))
    return 1.0 - 3.0 * s**2 + 2.0 * s**3


class Admittance:
    def __init__(self, config: dict):
        self.mass = _six(config["mass"], "mass")
        self.stiffness = _six(config["stiffness"], "stiffness")
        self.damping_min = _six(config["damping_min"], "damping_min")
        self.damping_max = _six(config["damping_max"], "damping_max")
        self.damping_rate = _six(config["damping_rate"], "damping_rate")
        if (np.any(self.mass <= 0) or np.any(self.stiffness <= 0)
                or np.any(self.damping_min <= 0)
                or np.any(self.damping_max < self.damping_min)
                or np.any(self.damping_rate < 0)):
            raise ValueError("M,K,D must be positive; Dmax>=Dmin and damping_rate>=0")
        margin = float(config["dt_margin"])
        if not np.isfinite(margin) or not 0 < margin <= 1:
            raise ValueError("dt_margin must be in (0,1]")
        # The energy bound may attain its minimum at either damping endpoint.
        endpoints = np.stack([self.damping_min, self.damping_max])
        intervals = 2 * self.mass * endpoints / (self.mass * self.stiffness + endpoints**2)
        self.max_dt = float(margin * np.min(intervals))
        self.reset()

    def reset(self, velocity=None) -> None:
        initial_velocity = np.zeros(6) if velocity is None else _six(velocity, "velocity")
        self.displacement = np.zeros(6)
        self.velocity = initial_velocity
        self.damping = self.damping_max.copy()

    def step(self, wrench, dt: float, risk=None) -> np.ndarray:
        """Update from w-w_ref using the measured elapsed interval, in seconds."""
        load = _six(wrench, "wrench difference")
        if not np.isfinite(dt) or dt <= 0 or dt > self.max_dt:
            raise ValueError(f"admittance dt must be in (0,{self.max_dt:.9g}] seconds")
        if risk is None:
            target = self.damping_max
        else:
            risk = _six(risk, "risk")
            if np.any(risk < 0) or np.any(risk > 1):
                raise ValueError("risk must be in [0,1]")
            # Thesis eq. (4-15): higher risk lowers damping to its positive
            # lower bound. Independent load gating reduces/stops advancement.
            target = self.damping_max - (self.damping_max - self.damping_min) * risk
        damping = self.damping + np.clip(target - self.damping,
                                        -self.damping_rate * dt, self.damping_rate * dt)
        acceleration = (load - damping * self.velocity - self.stiffness * self.displacement) / self.mass
        velocity = self.velocity + dt * acceleration
        displacement = self.displacement + dt * velocity
        if not np.all(np.isfinite(np.r_[velocity, displacement, damping])):
            raise ValueError("admittance state overflow")
        self.damping, self.velocity, self.displacement = damping, velocity, displacement
        return displacement.copy()

    def sync(self, displacement, velocity) -> None:
        """Synchronize with the feasible correction after command limiting.

        The caller must subtract its feasible nominal reference and bias from
        the limited command before using this method; sending a clipped command
        while leaving an accumulated internal displacement is not sufficient.
        """
        new_displacement = _six(displacement, "displacement")
        new_velocity = _six(velocity, "velocity")
        self.displacement, self.velocity = new_displacement, new_velocity
