"""Offline integrity and nominal-simulation checks; no camera or robot access."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_entry_configs(root):
    """Check the files consumed by default preview, collect and run entrypoints."""
    from .model_config import resolve_model_config

    defaults = ("data_capture_d415.json", "d415_preview_usb3.json", "d415_depth_usb3.json",
                "d415_preview.json", "geometry_nominal.json", "realsense_config.json")
    configs, errors = {}, []
    for name in defaults:
        path = root / "config" / name
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("config_object_required")
            configs[name] = value
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(f"entry_config {name}: {exc}")
    for name in ("d415_preview.json", "realsense_config.json"):
        if name not in configs:
            continue
        config = configs[name]
        # preview's --geometry default overrides the pipeline's geometry field.
        if name == "d415_preview.json":
            config["geometry_reference"] = "config/geometry_nominal.json"
        try:
            resolve_model_config(config, root=root, verify=True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f"entry_model {name}: {exc}")
        geometry_key = "geometry_reference" if config.get("geometry_reference") else "model_points"
        for key in (geometry_key, "observed_layout_prior", "intrinsics_override_file"):
            value = config.get(key)
            if value is None and key != geometry_key:
                continue
            if not isinstance(value, str) or not value.strip():
                errors.append(f"entry_dependency {name}.{key}: path_required")
                continue
            path = Path(value)
            path = path if path.is_absolute() else root / path
            if not path.is_file():
                errors.append(f"entry_dependency {name}.{key}: missing {path}")
    return {"default_configs": len(defaults), "loaded_configs": len(configs)}, errors


def check_library(library):
    from .training_library import calibration_status, depth_status, read_entries, validate_library

    library = Path(library).resolve()
    if not library.is_dir() or not (library / "records").is_dir():
        raise ValueError(f"library_missing: {library}")
    entries = read_entries(library)
    errors, warnings = [], []
    counts = dict(Counter(entry["status"] for entry in entries))
    training_ready = True
    try:
        validate_library(library)
    except ValueError as exc:
        training_ready = False
        message = str(exc)
        if not message.startswith("library_not_ready:\n"):
            raise
        for issue in message.splitlines()[1:]:
            reason = issue.split(": ", 1)[-1]
            if reason in {"annotation_pending", "annotation_conflict", "no_ready_training_images"}:
                warnings.append(issue)
            else:
                errors.append(issue)
    from train.python.dataset_split import read_split, group_for
    split = read_split(ROOT / "config/port_dataset_split.json")
    pending_groups = sorted({group_for(entry, split) for entry in entries
                             if entry["status"] != "excluded"
                             and group_for(entry, split) not in split["groups"]})
    if pending_groups:
        training_ready = False
        warnings.append(f"pending_split: {pending_groups}")
    depth = depth_status(library)
    errors.extend(depth["problems"])
    if depth["conflicts"]:
        warnings.append(f"depth_conflicts: {depth['conflicts']}")
    calibration = calibration_status(library)
    errors.extend(calibration["problems"])
    for kind in ("rgb", "depth"):
        missing = {state: count for state, count in calibration[kind].items()
                   if state in {"source_only", "missing", "unverifiable"} and count}
        if missing:
            warnings.append(f"{kind}_calibration_not_preserved: {missing}")
    if calibration["records_without_rgb_reference"]:
        warnings.append(f"records_without_rgb_reference: {calibration['records_without_rgb_reference']}")
    return {
        "path": str(library), "records": len(entries), "status_counts": counts,
        "training_ready": training_ready, "pending_split_groups": pending_groups,
        "depth_records": depth["with_depth"],
        "calibration": {key: calibration[key] for key in
                        ("rgb", "depth", "records_without_rgb_reference")},
    }, errors, warnings


def check_project(library=None):
    report = {"scope": "offline_integrity_and_nominal_simulation",
              "checks": {}, "errors": [], "warnings": []}
    checks, errors, warnings = report["checks"], report["errors"], report["warnings"]
    sources = [path for folder in ("vision", "control", "live", "train")
               for path in (ROOT / folder).rglob("*.py")]
    for path in sources:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"python_syntax {path.relative_to(ROOT)}: {exc}")
    checks["python_files"] = len(sources)
    configs = list((ROOT / "config").glob("*.json"))
    for path in configs:
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(f"config_json {path.relative_to(ROOT)}: {exc}")
    checks["json_configs"] = len(configs)
    checks["entry_configs"], entry_errors = check_entry_configs(ROOT)
    errors.extend(entry_errors)
    try:
        from .model_config import current_model_path
        checks["model"] = {"weights": str(current_model_path(root=ROOT, verify=True)), "sha256_verified": True}
    except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(f"model: {exc}")
    try:
        result, library_errors, library_warnings = check_library(library or ROOT / "dataset/vision")
        checks["library"] = result
        errors.extend(library_errors)
        warnings.extend(library_warnings)
    except (ImportError, OSError, ValueError, KeyError, TypeError, TimeoutError, RuntimeError) as exc:
        errors.append(f"library: {exc}")
    try:
        from control.simulate import run_simulation
        result, _ = run_simulation(scenario="nominal", output=None)
        checks["simulation"] = result
        if result["success"] is not True or result["stop_confirmed"] is not True:
            errors.append("nominal_simulation_did_not_succeed_and_stop")
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        errors.append(f"simulation: {exc}")
    report["status"] = "failed" if errors else "passed_with_warnings" if warnings else "passed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="离线项目自检：源码、配置、模型、图库和标称仿真；不访问硬件。")
    parser.add_argument("--library", type=Path, help="待检查图库，默认 dataset/vision")
    args = parser.parse_args(argv)
    report = check_project(args.library)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
