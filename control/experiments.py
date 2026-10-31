"""Plan and bind complete printed-part insertion tasks; never command hardware."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from .simulate import load_log

METHODS = ("A", "B", "C", "F")
AXES = ("x", "y", "Rx", "Ry")
CONDITIONS = {"zero": [0, 0, 0, 0]}
for _index, _axis in enumerate(AXES):
    for _sign, _word in ((1, "plus"), (-1, "minus")):
        _value = [0, 0, 0, 0]
        _value[_index] = _sign
        CONDITIONS[f"{_axis}_{_word}"] = _value
for _x in (-1, 1):
    for _ry in (-1, 1):
        CONDITIONS[f"x_{'plus' if _x > 0 else 'minus'}_Ry_{'plus' if _ry > 0 else 'minus'}"] = [_x, 0, 0, _ry]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def configuration_digest(config):
    """Bind controller settings while allowing a new task's bookkeeping."""
    return digest({k: v for k, v in config.items() if k != "experiment"})


def protocol_template():
    return {"schema_version": 2, "kind": "printed_insertion_protocol", "status": "draft",
            "protocol_version": None, "rules_version": "finite-trial-v2",
            "scope": "printed mechanical insertion only", "seed": 20261029,
            "methods": list(METHODS), "method_aliases": {"E": "C"}, "repeats_per_cell": 10,
            "offset_axes": list(AXES), "offset_units": ["m", "m", "rad", "rad"],
            "offset_amplitudes": [None] * 4, "setup_tolerance": [None] * 4,
            "geometry_version": None, "geometry_references": {}, "pilot_evidence": None,
            "specimens": [{"specimen_id": "test-01", "printing_batch_id": None},
                          {"specimen_id": "test-02", "printing_batch_id": None}],
            "method_config_sha256": {}, "frozen_configuration_files": {},
            "source_sha256": None, "model_sha256": None,
            "note": "Fill measured amplitudes and setup tolerances after independent pilot trials; freeze binds validated hardware configurations. No physical result is generated."}


def validate_protocol(protocol, *, frozen=False):
    if protocol.get("schema_version") != 2 or protocol.get("kind") != "printed_insertion_protocol":
        raise ValueError("expected printed insertion protocol schema 2")
    if protocol.get("rules_version") != "finite-trial-v2":
        raise ValueError("protocol control rules mismatch")
    if frozen and protocol.get("status") != "frozen":
        raise ValueError("formal execution requires a frozen protocol")
    if protocol.get("methods") != list(METHODS) or protocol.get("repeats_per_cell") != 10:
        raise ValueError("core design must be A/B/C/F x 13 conditions x 10 complete tasks")
    specimens = protocol.get("specimens", [])
    if len(specimens) != 2 or len({s.get("specimen_id") for s in specimens}) != 2:
        raise ValueError("exactly two distinct formal test specimens are required")
    if any(not s.get("specimen_id") for s in specimens):
        raise ValueError("specimen identities cannot be empty")
    if protocol.get("offset_axes") != list(AXES) or protocol.get("offset_units") != ["m", "m", "rad", "rad"]:
        raise ValueError("offsets must use x/y/Rx/Ry in m/m/rad/rad")
    for field in ("offset_amplitudes", "setup_tolerance"):
        vector = protocol.get(field)
        if not isinstance(vector, list) or len(vector) != 4:
            raise ValueError(f"{field} needs four values")
        if frozen or any(v is not None for v in vector):
            a = np.asarray(vector, float)
            if not np.isfinite(a).all() or np.any(a <= 0):
                raise ValueError(f"{field} must contain measured finite positive values")
    if frozen:
        for field in ("protocol_version", "geometry_version", "pilot_evidence"):
            if not isinstance(protocol.get(field), str) or not protocol[field].strip():
                raise ValueError(f"frozen protocol requires {field}")
        if any(not s.get("printing_batch_id") for s in specimens):
            raise ValueError("frozen protocol requires specimen printing batches")
        references = protocol.get("geometry_references", {})
        if set(references) != {s["specimen_id"] for s in specimens}:
            raise ValueError("frozen protocol requires measured geometry for both test specimens")
        for reference in references.values():
            if (not reference.get("path") or not reference.get("version")
                    or not isinstance(reference.get("sha256"), str) or len(reference["sha256"]) != 64):
                raise ValueError("geometry reference requires a versioned path and SHA256")
        hashes = protocol.get("method_config_sha256", {})
        if set(hashes) != set(METHODS) or any(not isinstance(v, str) or len(v) != 64 for v in hashes.values()):
            raise ValueError("frozen protocol requires all four configuration hashes")
        for field in ("source_sha256", "model_sha256"):
            if not isinstance(protocol.get(field), str) or len(protocol[field]) != 64:
                raise ValueError(f"frozen protocol requires {field}")


def freeze_protocol(protocol, config_paths, geometry_paths=None):
    from .config import validate_config
    from .logging import source_digest
    from vision.geometry_reference import load_geometry_reference
    result = deepcopy(protocol)
    if set(config_paths) != set(METHODS):
        raise ValueError("freeze requires one A/B/C/F hardware configuration each")
    if not geometry_paths or len(geometry_paths) != 2:
        raise ValueError("freeze requires measured geometry files for both test specimens")
    result["geometry_references"] = {}
    specimens = {s["specimen_id"]: s for s in protocol["specimens"]}
    for path in geometry_paths:
        geometry = load_geometry_reference(path)
        specimen = specimens.get(geometry.get("specimen_id"))
        if (geometry["status"] != "measured_print" or geometry["validated"] is not True
                or specimen is None or specimen["printing_batch_id"] != geometry["printing_batch_id"]
                or geometry["specimen_id"] in result["geometry_references"]):
            raise ValueError("formal geometry must identify each measured, validated test specimen and printing batch")
        result["geometry_references"][geometry["specimen_id"]] = {
            "path": geometry["source_path"], "sha256": geometry["source_sha256"], "version": geometry["version"]}
    result["method_config_sha256"], result["frozen_configuration_files"] = {}, {}
    shared = None
    for method, path in config_paths.items():
        path = Path(path)
        config = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_config(config, hardware=True)
        if config.get("method") != method:
            errors.append("method does not match configuration assignment")
        if errors:
            raise ValueError(f"{method}: cannot freeze unvalidated hardware configuration: {'; '.join(errors)}")
        comparable = {k: v for k, v in config.items() if k not in ("method", "experiment")}
        if shared is not None and shared != comparable:
            raise ValueError("comparison configurations must share all parameters except method; method flags implement the ablation")
        shared = comparable
        result["method_config_sha256"][method] = configuration_digest(config)
        result["frozen_configuration_files"][method] = str(path.resolve())
    model = Path(shared["learning"]["model_path"])
    result["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    result["source_sha256"] = source_digest()
    result["status"] = "frozen"
    validate_protocol(result, frozen=True)
    return result


def make_schedule(protocol):
    validate_protocol(protocol, frozen=protocol.get("status") == "frozen")
    rng = random.Random(protocol["seed"])
    protocol_hash = digest(protocol)
    tasks = []
    # Each repeat is a matched block of all four methods on one specimen.
    # Five repeats per specimen balance all 52 condition/method cells.
    for repeat in range(10):
        condition_order = list(CONDITIONS)
        rng.shuffle(condition_order)
        specimen = protocol["specimens"][repeat % 2]
        for condition in condition_order:
            block_id = f"{condition}-r{repeat + 1:02d}"
            methods = list(METHODS)
            rng.shuffle(methods)
            for method in methods:
                signs = CONDITIONS[condition]
                amplitudes = protocol["offset_amplitudes"]
                offset = [0.0 if sign == 0 else (sign * amp if amp is not None else None)
                          for sign, amp in zip(signs, amplitudes)]
                task_id = f"{protocol_hash[:12]}-{block_id}-{method}"
                tasks.append({"task_id": task_id, "order": len(tasks) + 1, "status": "pending",
                              "method": method, "condition": condition, "repeat": repeat + 1,
                              "pair_id": block_id, "block_id": block_id, **specimen,
                              "planned_initial_offset": offset, "session_id": None,
                              "fixture_event": None, "wear_cycles_before": None,
                              "result_log": None})
    return {"schema_version": 2, "kind": "printed_insertion_schedule", "protocol": deepcopy(protocol),
            "protocol_sha256": protocol_hash, "task_count": len(tasks), "tasks": tasks,
            "interpretation": "520 pending complete tasks; no trials have been performed"}


def bind_task(schedule, task_id, config, setup):
    from .logging import source_digest
    protocol = schedule["protocol"]
    validate_protocol(protocol, frozen=True)
    if schedule.get("protocol_sha256") != digest(protocol):
        raise ValueError("schedule protocol hash mismatch")
    expected = make_schedule(protocol)
    matches = [t for t in expected["tasks"] if t["task_id"] == task_id]
    supplied = [t for t in schedule["tasks"] if t["task_id"] == task_id]
    if len(matches) != 1 or len(supplied) != 1:
        raise ValueError("task must occur exactly once in the frozen schedule")
    task = matches[0]
    for field in ("method", "condition", "specimen_id", "printing_batch_id", "pair_id", "planned_initial_offset"):
        if supplied[0].get(field) != task[field]:
            raise ValueError(f"scheduled {field} was changed after freezing")
    if task["method"] != config.get("method") or protocol["method_config_sha256"][task["method"]] != configuration_digest(config):
        raise ValueError("task method/configuration differs from frozen protocol")
    if protocol["source_sha256"] != source_digest():
        raise ValueError("source code changed after formal protocol freeze")
    model = Path(config["learning"]["model_path"])
    if protocol["model_sha256"] != hashlib.sha256(model.read_bytes()).hexdigest():
        raise ValueError("GRU model changed after formal protocol freeze")
    geometry = protocol["geometry_references"][task["specimen_id"]]
    if geometry["sha256"] != hashlib.sha256(Path(geometry["path"]).read_bytes()).hexdigest():
        raise ValueError("specimen geometry changed after formal protocol freeze")
    if setup.get("task_id") != task_id or setup.get("verified") is not True:
        raise ValueError("task needs a separately verified setup record")
    for field in ("session_id", "fixture_event", "measurement_basis"):
        if not isinstance(setup.get(field), str) or not setup[field].strip():
            raise ValueError(f"setup requires {field}")
    if not isinstance(setup.get("wear_cycles_before"), int) or isinstance(setup["wear_cycles_before"], bool) or setup["wear_cycles_before"] < 0:
        raise ValueError("setup requires nonnegative cumulative specimen insertion count")
    measured = np.asarray(setup.get("measured_initial_offset"), float)
    if measured.shape != (4,) or not np.isfinite(measured).all():
        raise ValueError("setup requires measured initial x/y/Rx/Ry offset")
    if np.any(np.abs(measured - task["planned_initial_offset"]) > protocol["setup_tolerance"]):
        raise ValueError("measured initial deviation exceeds the frozen setup tolerance")
    setup_fields = ("verified", "session_id", "fixture_event", "measurement_basis", "wear_cycles_before", "measured_initial_offset")
    return {**task, **{key: setup[key] for key in setup_fields}, "formal": True, "protocol_version": protocol["protocol_version"],
            "protocol_sha256": schedule["protocol_sha256"], "geometry_version": protocol["geometry_version"],
            "geometry_reference": geometry,
            "method_config_sha256": configuration_digest(config), "rules_version": protocol["rules_version"]}


def wilson(successes, count, z=1.959963984540054):
    if not count:
        return None
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def task_metrics(path):
    rows = load_log(path)
    manifest = rows[0]
    if (manifest["config"].get("supervised_probe") is not None
            or manifest.get("experiment", {}).get("calibration_only") is True):
        raise ValueError(f"{path}: supervised probes are calibration-only, excluded from insertion method statistics and the 520 tasks")
    cycles = [r for r in rows if r.get("kind") == "cycle"]
    results = [r for r in rows if r.get("kind") == "result"]
    if (any(r.get("kind") == "device" and r.get("collect_only") for r in rows)
            or (not cycles and any(r.get("kind") == "sample" for r in rows))
            or any(r.get("decision", {}).get("state") == "COLLECT" for r in cycles)):
        raise ValueError(f"{path}: no insertion cycles (collection-only logs are not trials)")
    result = results[-1] if results else {"success": False, "reason": "incomplete_log"}
    if not cycles and not results:
        result = {"success": False, "reason": "incomplete_log_before_feedback"}
    loads, intervals, ages, pauses, limited = [], [], [], 0., 0
    trials, vision_checks, vision_rejects = {}, 0, 0
    for row in cycles:
        d = row["decision"]["details"]
        if d.get("dt") is not None:
            intervals.append(d["dt"])
        ages.append(row["at"] - row["sample"]["timestamp"])
        if d.get("wrench") is not None:
            a = np.asarray(d["wrench"], float)
            if a.shape == (6,) and np.isfinite(a).all():
                loads.append(a)
        if row["decision"]["state"] == "INSERT" and d.get("advance_scale") == 0:
            pauses += d.get("dt", 0.)
        limited += int(d.get("limited", False))
        for name in ("finite_trial", "trial_result"):
            trial = d.get(name) or {}
            if trial.get("trial_id"):
                trials.setdefault(trial["trial_id"], {}).update(trial)
        if row.get("observation") is not None:
            vision_checks += 1
            vision_rejects += int(row["observation"].get("valid") is not True
                                  or row["decision"].get("reason", "").startswith("vision_"))
    peaks = None
    if loads:
        a = np.asarray(loads)
        peaks = {"axial_N": float(np.abs(a[:, 2]).max()),
                 "lateral_N": float(np.linalg.norm(a[:, :2], axis=1).max()),
                 "moment_Nm": float(np.linalg.norm(a[:, 3:], axis=1).max())}
    experiment = manifest.get("experiment", manifest["config"].get("experiment", {}))
    outcomes = {}
    for trial in trials.values():
        outcome = trial.get("outcome") or "unconfirmed"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    duration = result.get("duration")
    if duration is None and cycles:
        duration = cycles[-1]["at"] - cycles[0]["at"]
    return {"task_id": manifest["task_id"], "method": "C" if manifest["config"]["method"] == "E" else manifest["config"]["method"],
            "recorded_method": manifest["config"]["method"],
            "source": manifest["source"], "target_id": manifest["config"]["target_id"],
            "config_sha256": manifest["config_sha256"], "success": result.get("success") is True,
            "rules_version": manifest.get("rules_version", "legacy-unversioned"),
            "source_sha256": manifest.get("source_sha256"), "model_sha256": manifest.get("model_sha256"),
            "protocol_sha256": experiment.get("protocol_sha256"), "condition": experiment.get("condition"),
            "specimen_id": experiment.get("specimen_id"), "printing_batch_id": experiment.get("printing_batch_id"),
            "session_id": experiment.get("session_id"), "pair_id": experiment.get("pair_id"),
            "wear_cycles_before": experiment.get("wear_cycles_before"), "fixture_event": experiment.get("fixture_event"),
            "formal": experiment.get("formal", False),
            "method_config_sha256": experiment.get("method_config_sha256", manifest["config_sha256"]),
            "reason": result.get("reason"), "duration": duration,
            "peak_current_wrench_at_tool_in_task_frame": peaks, "pause_seconds": pauses,
            "limited_fraction": limited / len(cycles) if cycles else None,
            "finite_trials": len(trials), "trial_outcomes": outcomes,
            "trial_effectiveness": outcomes.get("effective", 0) / len(trials) if trials else None,
            "visual_checks": vision_checks, "visual_rejects": vision_rejects,
            "visual_reject_rate": vision_rejects / vision_checks if vision_checks else None,
            "period_p50_p95_max": np.percentile(intervals, [50,95,100]).tolist() if intervals else None,
            "sample_age_max": max(ages) if ages else None, "path": str(path)}


def _summarize(tasks, fields):
    groups = {}
    for task in tasks:
        groups.setdefault(tuple(task[field] for field in fields), []).append(task)
    summaries = []
    for key, group in groups.items():
        count, successes = len(group), sum(t["success"] for t in group)
        durations = [t["duration"] for t in group if t["duration"] is not None]
        trials = sum(t["finite_trials"] for t in group)
        effective = sum(t["trial_outcomes"].get("effective", 0) for t in group)
        checks, rejects = sum(t["visual_checks"] for t in group), sum(t["visual_rejects"] for t in group)
        peaks = [t["peak_current_wrench_at_tool_in_task_frame"] for t in group if t["peak_current_wrench_at_tool_in_task_frame"]]
        reasons = {}
        for task in group:
            reason = task["reason"] or "unspecified"
            reasons[reason] = reasons.get(reason, 0) + 1
        summaries.append({**dict(zip(fields, key)), "tasks": count, "successes": successes,
                          "success_rate": successes / count, "wilson_95": wilson(successes, count),
                          "duration_count": len(durations),
                          "duration_mean": float(np.mean(durations)) if durations else None,
                          "duration_sample_std": float(np.std(durations, ddof=1)) if len(durations) > 1 else None,
                          "peak_wrench_max": {axis: max(p[axis] for p in peaks) for axis in peaks[0]} if peaks else None,
                          "finite_trials": trials, "effective_trials": effective,
                          "trial_effectiveness": effective / trials if trials else None,
                          "visual_checks": checks, "visual_rejects": rejects,
                          "visual_reject_rate": rejects / checks if checks else None, "outcomes": reasons})
    return summaries


def evaluate(paths):
    tasks = [task_metrics(p) for p in paths]
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate task IDs: repeated files/attempts cannot inflate sample size")
    paired = [(t["protocol_sha256"], t["pair_id"], t["method"]) for t in tasks if t["formal"]]
    if len(set(paired)) != len(paired):
        raise ValueError("duplicate formal method/block: E reuses C and retries cannot inflate the planned sample size")
    base = ("source", "target_id", "method", "rules_version", "protocol_sha256", "method_config_sha256", "source_sha256", "model_sha256")
    return {"schema_version": 2, "groups": _summarize(tasks, base + ("condition", "specimen_id")),
            "method_summary": _summarize(tasks, base), "tasks": tasks,
            "note": "Counts are complete task identities, including stops/timeouts/incomplete logs. E aliases C. Source, control rules, protocol and configurations remain separate; printed mechanical insertion only."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--protocol", type=Path, required=True)
    for method in METHODS:
        freeze.add_argument(f"--config-{method.lower()}", type=Path, required=True)
    freeze.add_argument("--geometry-references", type=Path, nargs=2, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--protocol", type=Path, required=True)
    schedule.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("evaluate", help="Summarize complete task logs, including failed tasks")
    evaluation.add_argument("logs", type=Path, nargs="+")
    evaluation.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            report = evaluate(args.logs)
            text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(text + "\n")
            print(text)
            return
        if args.command == "template":
            result = protocol_template()
        else:
            protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
            result = (make_schedule(protocol) if args.command == "schedule" else
                      freeze_protocol(protocol, {m: getattr(args, f"config_{m.lower()}") for m in METHODS}, args.geometry_references))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "status": result.get("status", "pending"),
                          "task_count": result.get("task_count", 0), "hardware_accessed": False}))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        operation = "Evaluation" if args.command == "evaluate" else "Experiment preparation"
        parser.exit(2, f"{operation} rejected: {exc}\n")


if __name__ == "__main__":
    main()
