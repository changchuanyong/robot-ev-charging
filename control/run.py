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
from .geometry import validate_transform
from .wrench import WrenchProcessor
from .types import Decision
from .synergy import ACTIONS
from .probe import OperatorProbePolicy, prepare_probe

import numpy as np


def preflight(config, *, probe_action=None):
    if not isinstance(config, dict):
        return validate_config(config, hardware=True), None
    if probe_action is not None:
        try:
            config = prepare_probe(config, probe_action)
        except (ValueError, KeyError, TypeError) as exc:
            return [str(exc)], None
    elif config.get("supervised_probe") is not None:
        return ["probe configuration requires explicit --probe-action; ordinary C/F execution cannot bypass GRU acceptance"], None
    errors = validate_config(config, hardware=True, supervised_probe=probe_action is not None)
    policy = None
    if not errors:
        try:
            if probe_action is not None:
                policy = OperatorProbePolicy(config["learning"], probe_action)
            elif METHODS[config["method"]]["learning"]:
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
            InsertionController(config, policy, probe_action=probe_action)
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


def run_device(config, observation_path, output, *, collect_only=False, duration=10., collection_frame=None,
               probe_action=None):
    if probe_action is not None:
        if collect_only or collection_frame is not None:
            raise ValueError("supervised probes require explicit motion mode, not collection-only mode")
        config = prepare_probe(config, probe_action)
    errors, policy = preflight(config, probe_action=probe_action)
    if errors:
        raise ValueError("Preflight rejected:\n- " + "\n- ".join(errors))
    processor, task_rotation = None, None
    if collection_frame is not None:
        if (not isinstance(collection_frame, dict) or not collect_only or not collection_frame.get("version")
                or collection_frame.get("validated") is not True):
            raise ValueError("collection task frame needs a validated version and --collect mode")
        task_rotation = validate_transform(collection_frame["T_base_task"])[:3, :3]
        processor = WrenchProcessor(config["wrench"])
    if collect_only and config.get("experiment", {}).get("formal"):
        raise ValueError("collection-only runs cannot consume a formal insertion task")
    from .hardware import RealManRobot
    robot = RealManRobot(hardware_config(config))
    controller = InsertionController(config, policy, probe_action=probe_action)
    stop_command_ack = False
    stop_feedback_confirmed = False
    failure = None
    with TaskLog(output, config, "hardware") as log:
        try:
            robot.connect()
            log.write({"kind": "device", "identity": robot.identity, "collect_only": collect_only,
                       "collection_frame": collection_frame, "supervised_probe": config.get("supervised_probe")})
            start = deadline = time.monotonic()
            observation_start_ns = time.time_ns()
            previous_seq = None
            previous_sample_at = None
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
                    if processor is not None and sample.seq != previous_seq:
                        dt = config["period"] if previous_sample_at is None else sample.timestamp - previous_sample_at
                        if (not np.isfinite(sample.timestamp)
                                or not -config["timing"]["future_tolerance"] <= now - sample.timestamp <= config["timing"]["max_sample_age"]
                                or (previous_seq is not None and sample.seq <= previous_seq)):
                            raise RuntimeError("stale collection feedback")
                        current, filtered = processor.process(sample, task_rotation, dt)
                        decision = Decision("COLLECT", details={"dt": dt, "wrench": current.tolist(),
                            "filtered_wrench": filtered.tolist(), "task_frame_version": collection_frame["version"],
                            "collection_only": True})
                        log.write({"kind": "cycle", "at": now, "sample": sample.to_dict(),
                                   "observation": None, "decision": decision.to_dict(), "sent_at": None})
                        previous_sample_at = sample.timestamp
                    if now - start >= duration:
                        log.write({"kind": "result", "success": False, "reason": "collection_only_no_motion",
                                   "duration": now - start})
                        break
                else:
                    observation = read_observation(observation_path, published_after_ns=observation_start_ns,
                                                   captured_after=start) if controller.state == "VISION" else None
                    compute_start = time.monotonic()
                    decision = controller.step(sample, now, observation)
                    if probe_action is not None:
                        # A calibration outcome always terminates this run;
                        # it must never become an insertion-success record.
                        decision.success = False
                        if decision.details.get("trial_result") is not None:
                            decision.command = None
                            decision.stop = decision.finished = True
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
                            if probe_action is not None and decision.details.get("trial_result"):
                                trial = deepcopy(decision.details["trial_result"])
                                trial.update(pre_stop_outcome=trial.get("outcome"), outcome="invalid",
                                             reason="stop_feedback_unconfirmed")
                                decision.details.update(trial_result=trial, probe_completed=False)
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
                                   "calibration_only": probe_action is not None, "probe_action": probe_action,
                                   "probe_completed": decision.details.get("probe_completed", False),
                                   "probe_outcome": (decision.details.get("trial_result") or {}).get("outcome"),
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
    parser.add_argument("--collection-frame", type=Path, help="validated T_base_task JSON; adds transformed COLLECT cycle logs")
    parser.add_argument("--probe-action", choices=ACTIONS[:-1],
                        help="explicit single supervised calibration direction; requires method C and session/specimen/printing batch identities")
    parser.add_argument("--schedule", type=Path, help="frozen formal experiment schedule")
    parser.add_argument("--scheduled-task", help="unique complete task ID from schedule")
    parser.add_argument("--setup-record", type=Path, help="verified physical setup/session/wear record for formal task")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.probe_action and (args.collect or any((args.schedule, args.scheduled_task, args.setup_record))):
            parser.error("--probe-action is calibration-only; cannot combine with --collect or a formal schedule")
        if any((args.schedule, args.scheduled_task, args.setup_record)):
            if not all((args.schedule, args.scheduled_task, args.setup_record)) or args.collect:
                parser.error("formal tasks require --schedule, --scheduled-task and --setup-record; cannot use --collect")
            from .experiments import bind_task
            config["experiment"] = bind_task(json.loads(args.schedule.read_text(encoding="utf-8")), args.scheduled_task,
                                              config, json.loads(args.setup_record.read_text(encoding="utf-8")))
        if not (args.collect or args.execute_motion):
            errors, _ = preflight(config, probe_action=args.probe_action)
            print(json.dumps({"ready": not errors, "hardware_accessed": False, "errors": errors}, ensure_ascii=False, indent=2))
            return 2 if errors else 0
        if args.output is None or args.duration <= 0:
            parser.error("device runs require a new --output log path and a positive duration")
        frame = json.loads(args.collection_frame.read_text(encoding="utf-8")) if args.collection_frame else None
        run_device(config, args.observation, args.output, collect_only=args.collect, duration=args.duration,
                   collection_frame=frame, probe_action=args.probe_action)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Run refused/terminated: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
