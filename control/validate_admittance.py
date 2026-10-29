"""Analytic versus discrete admittance response, using synthetic parameters only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .admittance import Admittance


def exact_step(time, mass, damping, stiffness, force):
    """Zero-state displacement of M*x''+D*x'+K*x=force for constant force."""
    t = np.asarray(time, dtype=float)
    if (not np.isfinite([mass, damping, stiffness, force]).all()
            or min(mass, damping, stiffness) <= 0 or not np.isfinite(t).all() or np.any(t < 0)):
        raise ValueError("finite_positive_M_D_K_and_nonnegative_time_required")
    discriminant = damping*damping-4*mass*stiffness
    equilibrium = force/stiffness
    if np.isclose(discriminant, 0, atol=1e-12, rtol=1e-12):
        rate = damping/(2*mass)
        return equilibrium*(1-(1+rate*t)*np.exp(-rate*t))
    root = np.sqrt(complex(discriminant))
    r1, r2 = (-damping+root)/(2*mass), (-damping-root)/(2*mass)
    return np.real(equilibrium*(1+(r2*np.exp(r1*t)-r1*np.exp(r2*t))/(r1-r2)))


def validate_responses():
    rows = []
    for name, damping in (("underdamped", 2.), ("critical", 10.), ("overdamped", 20.)):
        for dt in (.01, .005, .0025):
            config = {"mass": [1.]*6, "stiffness": [25.]*6,
                      "damping_min": [damping]*6, "damping_max": [damping]*6,
                      "damping_rate": [0.]*6, "dt_margin": .8}
            model = Admittance(config)
            count = round(4/dt)
            observed = np.array([model.step([2., 0, 0, 0, 0, 0], dt)[0] for _ in range(count)])
            expected = exact_step(np.arange(1, count+1)*dt, 1., damping, 25., 2.)
            error = observed-expected
            rows.append({"response": name, "dt_s": dt, "max_allowed_dt_s": model.max_dt,
                         "max_abs_displacement_error": float(np.max(np.abs(error))),
                         "rms_displacement_error": float(np.sqrt(np.mean(error**2))),
                         "final_displacement": float(observed[-1]), "static_equilibrium": .08})
    convergence = all(rows[i+1]["max_abs_displacement_error"] < rows[i]["max_abs_displacement_error"]
                      for i in (0, 1, 3, 4, 6, 7))
    return {"schema_version": 1, "source": "synthetic_analytic_ODE", "hardware_accessed": False,
            "passes_dt_refinement": convergence, "cases": rows,
            "interpretation": "Constant-damping ODE discretization check; does not validate robot/contact stability or physical limits."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_responses()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"passes_dt_refinement": report["passes_dt_refinement"], "cases": len(report["cases"])}))
    if not report["passes_dt_refinement"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
