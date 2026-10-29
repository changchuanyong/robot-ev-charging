"""Deterministic software fixture. It is not a validated connector simulator."""
import argparse
import json
from pathlib import Path
from collections import deque

import numpy as np

from .config import METHODS, load_config, simulation_config
from .geometry import invert, log_so3
from .logging import TaskLog
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
