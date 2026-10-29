"""Plan and bind complete printed-part insertion tasks; never command hardware."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np

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


def main():
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
    args = parser.parse_args()
    try:
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
        parser.exit(2, f"Experiment preparation rejected: {exc}\n")


if __name__ == "__main__":
    main()
