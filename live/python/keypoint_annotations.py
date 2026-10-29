"""Schema checks for development-only nine-hole annotations in original pixels."""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
from pathlib import Path, PurePosixPath
import re


LABELS = ("S_neg", "CC2", "S_pos", "CC1", "DC_neg", "DC_pos", "A_neg", "PE", "A_pos")
VISIBILITIES = ("unlabeled", "visible", "uncertain", "occluded", "out_of_frame")
SOURCES = ("unlabeled", "ai_visual", "human")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_image_key(path: str) -> str:
    """Accept stored Windows separators, but require a repository-relative path."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Image/path must be a nonempty string")
    value = path.replace("\\", "/")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or ":" in value:
        raise ValueError(f"Expected repository-relative path: {path}")
    return str(parsed)


def _hash(value, description):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid SHA256: {description}")


def _xy(value, width, height, description):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        raise ValueError(f"Invalid finite pixel coordinate: {description}")
    if not (0 <= value[0] < width and 0 <= value[1] < height):
        raise ValueError(f"Pixel coordinate outside image: {description}")


def visible_points(image: dict) -> dict[str, list[float]]:
    """Only directly visible, named observations may enter PnP."""
    return {name: point["xy"] for name, point in image["points"].items()
            if point["visibility"] == "visible"}


def validate_annotations(data: dict) -> None:
    """Validate the schema without reading files or trusting a draft as human truth."""
    if not isinstance(data, dict) or data.get("schema_version") not in (1, 2):
        raise ValueError("Expected annotation schema_version 1 or 2")
    if data.get("purpose") != "development_diagnostic" or data.get("independent_test_set") is not False:
        raise ValueError("This annotation set must remain a development diagnostic")
    if data.get("labels") != list(LABELS):
        raise ValueError("Annotation labels/order differ from the nine-hole schema")
    if not isinstance(data.get("feature_definition"), str) or not data["feature_definition"].strip():
        raise ValueError("Missing feature_definition")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Missing provenance")
    if data["schema_version"] == 1:
        for field in ("batch_report_path", "camera_config_path", "model_points_path"):
            normalized_image_key(provenance.get(field))
        for field in ("batch_report_sha256", "camera_config_sha256", "model_points_sha256", "pnp_source_sha256"):
            _hash(provenance.get(field), field)
    else:
        if provenance.get("kind") != "capture_session":
            raise ValueError("Capture annotations require capture_session provenance")
        for field in ("manifest_path", "camera_config_path"):
            normalized_image_key(provenance.get(field))
        for field in ("manifest_sha256", "camera_config_sha256"):
            _hash(provenance.get(field), field)
        for field in ("session_id", "batch_id", "camera_serial"):
            if not isinstance(provenance.get(field), str) or not provenance[field].strip():
                raise ValueError(f"Capture provenance requires {field}")
    if not isinstance(data.get("images"), list) or not data["images"]:
        raise ValueError("Expected a nonempty images list")
    paths, hashes = set(), set()
    for entry in data["images"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid image entry")
        key = normalized_image_key(entry.get("image"))
        digest = entry.get("image_sha256")
        _hash(digest, key)
        if key in paths or digest in hashes:
            raise ValueError(f"Duplicate image identity or content: {key}")
        paths.add(key)
        hashes.add(digest)
        width, height = entry.get("width"), entry.get("height")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
            raise ValueError(f"Invalid image dimensions: {key}")
        if entry.get("split") not in ("train", "val", "test", "unknown"):
            raise ValueError(f"Invalid split: {key}")
        if data["schema_version"] == 2:
            if entry["split"] != data["images"][0]["split"]:
                raise ValueError("Capture session must use a single split")
            for field in ("session_id", "batch_id", "camera_serial"):
                if entry.get(field) != provenance[field]:
                    raise ValueError(f"Capture image {field} differs from provenance: {key}")
        if not isinstance(entry.get("selection_reasons"), list) or any(
                not isinstance(reason, str) for reason in entry["selection_reasons"]):
            raise ValueError(f"Invalid selection_reasons: {key}")
        if not isinstance(entry.get("notes"), str):
            raise ValueError(f"Invalid notes: {key}")
        if entry.get("review_status") not in ("draft", "reviewed"):
            raise ValueError(f"Invalid review_status: {key}")
        if not isinstance(entry.get("reviewer"), str):
            raise ValueError(f"Invalid reviewer: {key}")
        if entry["review_status"] == "reviewed":
            if not entry["reviewer"].strip() or not isinstance(entry.get("reviewed_at"), str):
                raise ValueError(f"Reviewed image requires reviewer and reviewed_at: {key}")
            try:
                timestamp = datetime.fromisoformat(entry["reviewed_at"].replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"Invalid reviewed_at timestamp: {key}") from exc
            if timestamp.tzinfo is None:
                raise ValueError(f"reviewed_at must include timezone: {key}")
        elif entry.get("reviewed_at") is not None or entry["reviewer"]:
            raise ValueError(f"Draft image must clear reviewer and reviewed_at: {key}")
        roi = entry.get("roi_bbox_xyxy")
        if not isinstance(roi, list) or len(roi) not in (0, 4):
            raise ValueError(f"ROI must be empty or four coordinates: {key}")
        if roi and (any(isinstance(v, bool) or not isinstance(v, (int, float))
                        or not math.isfinite(v) for v in roi)
                    or not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height)):
            raise ValueError(f"Invalid ROI bounds: {key}")
        points = entry.get("points")
        if not isinstance(points, dict) or set(points) != set(LABELS):
            raise ValueError(f"Every image must record all nine point statuses: {key}")
        for name, point in points.items():
            context = f"{key}/{name}"
            if not isinstance(point, dict) or point.get("visibility") not in VISIBILITIES:
                raise ValueError(f"Invalid visibility: {context}")
            if point.get("source") not in SOURCES or not isinstance(point.get("note"), str):
                raise ValueError(f"Invalid point source/note: {context}")
            visibility = point["visibility"]
            if visibility != "unlabeled" and point["source"] == "unlabeled":
                raise ValueError(f"Decided point status needs an observation source: {context}")
            if visibility == "visible":
                _xy(point.get("xy"), width, height, context)
            elif point.get("xy") is not None:
                raise ValueError(f"Nonvisible point cannot supply coordinates: {context}")
            if entry["review_status"] == "reviewed" and visibility == "unlabeled":
                raise ValueError(f"Reviewed image still has unlabeled point: {context}")
        auto = entry.get("auto_points")
        if not isinstance(auto, dict) or not set(auto).issubset(LABELS):
            raise ValueError(f"Invalid automatic point labels: {key}")
        for name, value in auto.items():
            _xy(value, width, height, f"automatic {key}/{name}")
