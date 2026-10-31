"""Deterministic software fixture. It is not a validated connector simulator."""
import argparse
import json
import sys
from pathlib import Path
from collections import deque

import numpy as np

from .config import METHODS, load_config, simulation_config
from .geometry import invert, log_so3
from .logging import TaskLog, finite_json, source_digest
from .admittance import Admittance
from .state_machine import InsertionController
from .types import RobotSample


SCENARIOS = ("nominal", "blocked", "overload", "stale", "disconnect", "no_contact", "bad_vision", "risk_outside",
             "feedback_lag", "period_jitter", "trial_timeout", "observation_drift", "confirmation_blocked")


class SimulatedRobot:
    def __init__(self, config, scenario="nominal"):
        if config.get("simulation_only") is not True:
            raise ValueError("simulation requires an explicitly synthetic configuration")
        if scenario not in SCENARIOS:
            raise ValueError("unknown scenario")
        self.config, self.scenario = config, scenario
        self.pose = np.eye(4)
        self.pose[2, 3] = -config["task"]["pre_distance"]
        self.seq = 0
        self.stopped = False
        self.sent = 0
        self.pending = deque()
        self.trial_phase = None

    def read(self, now):
        if self.scenario == "feedback_lag" and len(self.pending) >= 3:
            self.pose = self.pending.popleft()
        self.seq += 1
        wrench = np.zeros(6)
        if self.pose[2, 3] >= 0 and self.scenario != "no_contact":
            wrench[2] = -.22 - 20 * self.pose[2, 3]
            wrench[:2] = -30 * self.pose[:2, 3]
            wrench[3:] = -.1 * log_so3(self.pose[:3, :3])
        if now >= 2. and self.scenario == "overload":
            wrench[2] = -30.
        if now >= 2. and self.scenario == "risk_outside":
            wrench[2] = -7.
        return RobotSample(
            self.seq, now - (.2 if now >= 2. and self.scenario == "stale" else 0.),
            self.pose.copy(), wrench, tuple([0.] * 7),
            not (now >= 2. and self.scenario == "disconnect"), "simulated_disconnect",
            "task", {"simulation": True, "external_compensated": True,
                     "force_channel": "synthetic_external", "moment_origin": "tool"},
        )

    def observation(self, sample):
        return {
            "schema_version": 1, "valid": self.scenario != "bad_vision",
            "reason": "synthetic_fixture", "frame_id": f"synthetic-{sample.seq}",
            "captured_at": sample.timestamp, "received_at": sample.timestamp,
            "processed_at": sample.timestamp, "target_id": self.config["target_id"],
            "source": "simulation", "time_sync_valid": True,
            "clock": "host_monotonic", "camera_serial": "SIMULATED",
            "camera_frame": "camera_color_optical", "translation_unit": "m",
            "T_camera_target": invert(sample.pose).tolist(), "calibration_version": "synthetic-v1",
            "quality": {"reprojection_rms_px": .2, "max_error_px": .5, "inlier_count": 9,
                        "all_in_front": True, "camera_calibrated": True,
                        "visual_budget": [.004, .004, .08, .08],
                        "budget_version": "synthetic-v1"},
        }

    def command(self, target, dt):
        if self.stopped:
            raise RuntimeError("command after stop")
        self.sent += 1
        if self.scenario == "feedback_lag":
            self.pending.append(target.copy())
            return
        previous = self.pose.copy()
        self.pose = target.copy()
        if self.scenario == "blocked":
            self.pose[2, 3] = min(self.pose[2, 3], .001)
        if self.scenario == "confirmation_blocked" and self.trial_phase == "CONFIRM":
            self.pose[2, 3] = previous[2, 3]
        if self.scenario == "trial_timeout":
            self.pose[0, 3] = previous[0, 3]
        if self.scenario == "observation_drift" and self.trial_phase == "OBSERVE":
            self.pose[0, 3] += .00002

    def stop(self):
        self.stopped = True


def make_policy(config):
    if not METHODS[config["method"]]["learning"]:
        return None
    from .learning import ForcePolicy
    learning = {**config["learning"], "allow_synthetic": config.get("simulation_only") is True,
                "require_hardware_validation": config.get("simulation_only") is not True}
    policy = ForcePolicy(learning, learning.get("model_path"))
    if not policy.model_ready:
        raise ValueError("GRU checkpoint is not ready: " + policy.last_reason)
    return policy


def run_simulation(config=None, scenario="nominal", output=None, max_steps=10000, policy=None):
    config = config or simulation_config()
    controller = InsertionController(config, policy if policy is not None else make_policy(config))
    robot = SimulatedRobot(config, scenario)
    logger = TaskLog(output, config, "simulation") if output else None
    decisions = []
    try:
        now = 0.
        for k in range(max_steps):
            dt = config["period"] * ((.6, 1.4, .8, 1.2)[k % 4] if scenario == "period_jitter" else 1.)
            sample = robot.read(now)
            observation = robot.observation(sample) if controller.state == "VISION" else None
            decision = controller.step(sample, now, observation)
            decisions.append(decision)
            if logger:
                logger.write({"kind": "cycle", "at": now, "target_id": config["target_id"],
                              "sample": sample.to_dict(), "observation": observation,
                              "decision": decision.to_dict()})
            if decision.stop:
                robot.stop()
                controller.acknowledge_stop()
            elif decision.command is not None:
                robot.trial_phase = controller.trial.phase if controller.trial else None
                robot.command(decision.command, dt)
            if decision.finished:
                break
            now += dt
        else:
            robot.stop()
            raise RuntimeError("simulation exceeded max_steps without a terminal decision")
        result = {"source": "simulation", "scenario": scenario, "method": config["method"],
                  "success": decision.success, "reason": decision.reason,
                  "duration": now, "cycles": len(decisions), "commands": robot.sent,
                  "stop_confirmed": robot.stopped, "states": [e["to"] for e in controller.events]}
        if logger:
            logger.write({"kind": "result", **result, "events": controller.events})
        return result, decisions
    finally:
        robot.stop()
        if logger:
            logger.close()


def load_log(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or rows[0].get("kind") != "manifest" or rows[0].get("schema_version") not in (1, 2):
        raise ValueError("not a versioned control task log")
    ids = {r.get("task_id") for r in rows}
    if len(ids) != 1 or None in ids:
        raise ValueError("a task log must contain exactly one task identity")
    return rows


def replay(path):
    rows = load_log(path)
    config = rows[0]["config"]
    probe = config.get("supervised_probe")
    if probe is not None:
        from .probe import OperatorProbePolicy, prepare_probe
        config = prepare_probe(config, probe["action"])
        controller = InsertionController(config, OperatorProbePolicy(config["learning"], probe["action"]),
                                         probe_action=probe["action"])
    else:
        controller = InsertionController(config, make_policy(config))
    mismatches = []
    count = 0
    last = None
    for row in rows:
        if row.get("kind") != "cycle":
            continue
        count += 1
        sample = RobotSample.from_dict(row["sample"])
        last = controller.step(sample, float(row["at"]), row.get("observation"))
        original = row["decision"]
        difference = [k for k in ("state", "stop", "finished", "success", "reason") if last.to_dict()[k] != original[k]]
        old_command = original.get("command")
        if (last.command is None) != (old_command is None):
            difference.append("command_presence")
        elif last.command is not None and not np.allclose(last.command, old_command, atol=1e-9, rtol=0):
            difference.append("command")
        if difference:
            mismatches.append({"cycle": count, "at": row["at"], "fields": difference})
        if last.stop:
            controller.acknowledge_stop()
    if not count:
        raise ValueError("log has no controller cycles to replay")
    return {"hardware_accessed": False, "source": str(path), "cycles": count,
            "source_code_matches": rows[0].get("source_sha256") == source_digest(),
            "decision_mismatches": mismatches, "reproduced": not mismatches,
            "final_state": last.state, "final_reason": last.reason,
            "interpretation": "recorded feedback replay; no claim of alternative closed-loop physical outcome"}


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


def _simulate_main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else simulation_config()
    if args.method:
        config["method"] = args.method
    if args.model:
        config["learning"]["model_path"] = str(args.model)
    try:
        result, _ = run_simulation(config, args.scenario, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Simulation failed: {exc}\n")


def _replay_main(argv):
    parser = argparse.ArgumentParser(description="Recompute recorded feedback; no hardware access.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = replay(args.input)
        text = json.dumps(finite_json(report), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        print(text)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Replay rejected: {exc}\n")


def _admittance_main(argv):
    parser = argparse.ArgumentParser(description="Analytic versus discrete admittance response with synthetic parameters.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_responses()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"passes_dt_refinement": report["passes_dt_refinement"], "cases": len(report["cases"])}))
    if not report["passes_dt_refinement"]:
        raise SystemExit(1)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"simulate": _simulate_main, "replay": _replay_main, "admittance": _admittance_main}
    parser = argparse.ArgumentParser(description="Offline simulation, feedback replay and admittance checks; no hardware access.")
    parser.add_argument("command", nargs="?", choices=commands, default="simulate",
                        help="Use COMMAND --help for its options; omitted command runs simulation.")
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    if argv and argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        return
    if argv and not argv[0].startswith("-"):
        parser.error("unknown offline command: " + argv[0])
    return _simulate_main(argv)


if __name__ == "__main__":
    main()
