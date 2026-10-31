"""Resolve the shared detector once, while keeping experiment weights explicit."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .contracts import atomic_capture_json


ROOT = Path(__file__).resolve().parents[1]
CURRENT_MODEL_CONFIG = "config/vision_model.json"


def _path(value, root):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model_path_required")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def resolve_model_config(config: dict, *, root=ROOT, verify=False) -> dict:
    """Return a pinned copy; a missing/broken current model never selects an old one."""
    root = Path(root).resolve()
    result = deepcopy(config)
    if "model_registry" in result:
        if "weights" in result:
            raise ValueError("choose_model_registry_or_explicit_weights_not_both")
        registry_path = _path(result.pop("model_registry"), root)
        content = registry_path.read_bytes()
        registry = json.loads(content.decode("utf-8-sig"))
        if registry.get("schema_version") != 1:
            raise ValueError("unsupported_vision_model_schema")
        expected = registry.get("weights_sha256", "")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("vision_model_weights_sha256_required")
        result["weights"] = str(_path(registry.get("weights"), root))
        result["weights_sha256"] = expected
        result["model_source"] = {
            "kind": "registry", "registry_path": str(registry_path),
            "registry_sha256": hashlib.sha256(content).hexdigest(),
            "training_run": registry.get("training_run"),
        }
    elif result.get("weights"):
        result["weights"] = str(_path(result["weights"], root))
    if verify:
        weights = _path(result.get("weights"), root)
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights missing: {weights}")
        expected = result.get("weights_sha256")
        if expected is not None and hashlib.sha256(weights.read_bytes()).hexdigest() != expected:
            raise ValueError(f"YOLO weights hash mismatch: {weights}")
    return result


def current_model_path(*, root=ROOT, verify=False) -> Path:
    config = resolve_model_config({"model_registry": CURRENT_MODEL_CONFIG}, root=root, verify=verify)
    return Path(config["weights"])


def publish_current_model(weights, *, root=ROOT, training_report) -> dict:
    """Publish only the best weights of a fully completed, checked training run."""
    root = Path(root).resolve()
    weights = _path(str(weights), root)
    report_path = _path(str(training_report), root)
    content = report_path.read_bytes()
    report = json.loads(content.decode("utf-8-sig"))
    epochs = report.get("requested_epochs")
    if (report.get("status") != "completed" or type(epochs) is not int or epochs <= 0
            or report.get("completed_epochs") != epochs
            or report.get("snapshot_posttrain_check", {}).get("status") != "passed"):
        raise ValueError("completed_checked_training_report_required_for_publication")
    if _path(report.get("best_weights"), root) != weights:
        raise ValueError("published_weights_must_match_completed_training_report")
    expected = report.get("best_weights_sha256", "")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("training_report_best_weights_sha256_required")
    resolve_model_config({"weights": str(weights), "weights_sha256": expected}, root=root, verify=True)

    def portable(path):
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)

    registry = {
        "schema_version": 1,
        "description": "Shared active YOLOv8n detector; camera calibration and control acceptance are configured separately.",
        "weights": portable(weights), "weights_sha256": expected,
        "training_run": portable(report_path.parent), "training_report": portable(report_path),
        "training_report_sha256": hashlib.sha256(content).hexdigest(),
        "evaluation": report.get("evaluation", "held_out_development"),
        "independent_validation": report.get("independent_validation", False),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_capture_json(root / CURRENT_MODEL_CONFIG, registry)
    return registry
