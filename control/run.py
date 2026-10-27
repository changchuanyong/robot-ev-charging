"""Explicit real-device entrypoint. The default only validates local files."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from .config import METHODS, load_config, validate_config
from .logging import TaskLog
from .simulate import make_policy
from .state_machine import InsertionController


def preflight(config):
    errors = validate_config(config, hardware=True)
    policy = None
    if not errors:
        try:
            if METHODS[config["method"]]["learning"]:
                from .learning import load_checkpoint
                metadata, _ = load_checkpoint(config["learning"]["model_path"])
                if metadata["synthetic"]:
                    errors.append("synthetic GRU checkpoints cannot control real hardware")
                if metadata.get("validated_for_hardware") is not True:
                    errors.append("GRU checkpoint lacks independent hardware acceptance")
                acceptance = metadata.get("hardware_acceptance", {})
                for name, expected in (("calibration_version", config["calibration"]["version"]),
                                       ("wrench_version", config["wrench"]["version"]),
                                       ("robot_model", config["hardware"]["expected_arm_model"])):
                    if acceptance.get(name) != expected:
                        errors.append(f"GRU hardware acceptance {name} mismatch")
                policy = make_policy(config)
            InsertionController(config, policy)
            from .hardware import RealManRobot
            driver_config = hardware_config(config)
            # This validates pure configuration. It does not construct the SDK
            # RoboticArm instance or initiate sockets.
            RealManRobot(driver_config)._validate_config()
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as exc:
            errors.append(str(exc))
    return errors, policy


def hardware_config(config):
    h = deepcopy(config["hardware"])
    h["T_flange_tool"] = config["calibration"]["T_flange_tool"]
    h["wrench_version"] = config["wrench"]["version"]
    return h


def read_observation(path, *, published_after_ns=None, captured_after=None):
    try:
        path = Path(path)
        before = path.stat()
        if published_after_ns is not None and before.st_mtime_ns <= published_after_ns:
            return {"valid": False, "reason": "observation_predates_this_run"}
        with path.open(encoding="utf-8") as stream:
            observation = json.load(stream)
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            return {"valid": False, "reason": "observation_changed_during_read"}
        if captured_after is not None and observation.get("captured_at", -1) < captured_after:
            return {"valid": False, "reason": "capture_predates_this_run"}
        return observation
    except (OSError, ValueError, TypeError, AttributeError):
        return {"valid": False, "reason": "observation_unavailable"}


def run_device(config, observation_path, output, *, collect_only=False, duration=10.):
    errors, policy = preflight(config)
    if errors:
        raise ValueError("Preflight rejected:\n- " + "\n- ".join(errors))
    from .hardware import RealManRobot
    robot = RealManRobot(hardware_config(config))
    controller = InsertionController(config, policy)
    stop_command_ack = False
    stop_feedback_confirmed = False
    failure = None
    with TaskLog(output, config, "hardware") as log:
        try:
            robot.connect()
            log.write({"kind": "device", "identity": robot.identity, "collect_only": collect_only})
            start = deadline = time.monotonic()
            observation_start_ns = time.time_ns()
            previous_seq = None
            while True:
                now = time.monotonic()
                # No motion has been sent while waiting for the first UDP data.
                try:
                    sample = robot.read()
                except RuntimeError as exc:
                    if previous_seq is None and now - start < config["timing"]["max_sample_age"]:
                        time.sleep(min(.001, config["period"]))
                        continue
                    raise RuntimeError(f"feedback unavailable: {exc}") from exc
                if collect_only:
                    if sample.seq != previous_seq:
                        log.write({"kind": "sample", "at": now, "sample": sample.to_dict()})
                    if not sample.valid:
                        raise RuntimeError("invalid collection feedback: " + sample.reason)
                    if now - start >= duration:
                        log.write({"kind": "result", "success": False, "reason": "collection_only_no_motion",
                                   "duration": now - start})
                        break
                else:
                    observation = read_observation(observation_path, published_after_ns=observation_start_ns,
                                                   captured_after=start) if controller.state == "VISION" else None
                    compute_start = time.monotonic()
                    decision = controller.step(sample, now, observation)
                    compute_end = time.monotonic()
                    # Excess computation time invalidates the command before
                    # sending, regardless of the previous period's validity.
                    if compute_end - sample.timestamp > config["timing"]["max_sample_age"]:
                        raise RuntimeError("feedback became stale during control computation")
                    sent_at = None
                    if decision.stop:
                        robot.stop()
                        stop_command_ack = robot.stop_command_ack
                        stop_feedback_confirmed = robot.wait_stopped(
                            config["hardware"]["stop_feedback_timeout"], config["hardware"]["stop_tolerance"])
                        if stop_feedback_confirmed:
                            controller.acknowledge_stop()
                        else:
                            decision.success = False
                            decision.reason = "stop_feedback_unconfirmed"
                            decision.finished = True
                            failure = RuntimeError("stop command acknowledged but measured stop was not confirmed")
                    elif decision.command is not None:
                        robot.command(decision.command, decision.details["dt"])
                        sent_at = time.monotonic()
                    log.write({"kind": "cycle", "at": now, "sample": sample.to_dict(),
                               "observation": observation, "decision": decision.to_dict(),
                               "compute_start": compute_start, "compute_end": compute_end,
                               "sent_at": sent_at, "stop_command_ack": stop_command_ack,
                               "stop_feedback_confirmed": stop_feedback_confirmed})
                    if decision.finished:
                        log.write({"kind": "result", "success": decision.success, "reason": decision.reason,
                                   "duration": now - start, "stop_command_ack": stop_command_ack,
                                   "stop_feedback_confirmed": stop_feedback_confirmed,
                                   "events": controller.events})
                        break
                previous_seq = sample.seq
                deadline += config["period"]
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif -delay > config["timing"]["max_period"]:
                    raise RuntimeError("control scheduler missed its accepted deadline")
        except BaseException as exc:
            failure = exc
            if not collect_only:
                try:
                    robot.stop()
                    stop_command_ack = robot.stop_command_ack
                    stop_feedback_confirmed = robot.wait_stopped(
                        config["hardware"]["stop_feedback_timeout"], config["hardware"]["stop_tolerance"])
                except Exception as stop_exc:
                    log.write({"kind": "stop_error", "error": str(stop_exc),
                               "stop_command_ack": robot.stop_command_ack, "stop_feedback_confirmed": False})
            log.write({"kind": "result", "success": False, "reason": type(exc).__name__,
                       "error": str(exc), "stop_command_ack": stop_command_ack,
                       "stop_feedback_confirmed": stop_feedback_confirmed})
        finally:
            try:
                robot.close()
            except Exception as exc:
                log.write({"kind": "close_error", "error": str(exc)})
                if failure is None:
                    failure = exc
        if failure:
            raise failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/control.template.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="local readiness checks only (default)")
    mode.add_argument("--collect", action="store_true", help="connect and log feedback; no motion commands")
    mode.add_argument("--execute-motion", action="store_true", help="explicitly enable one real insertion attempt")
    parser.add_argument("--observation", type=Path, default=Path("dataset/live/latest_control_pose.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=float, default=10.)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if not (args.collect or args.execute_motion):
            errors, _ = preflight(config)
            print(json.dumps({"ready": not errors, "hardware_accessed": False, "errors": errors}, ensure_ascii=False, indent=2))
            return 2 if errors else 0
        if args.output is None or args.duration <= 0:
            parser.error("device runs require a new --output log path and a positive duration")
        run_device(config, args.observation, args.output, collect_only=args.collect, duration=args.duration)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Run refused/terminated: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
