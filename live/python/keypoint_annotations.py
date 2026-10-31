"""Schema checks for development-only nine-hole annotations in original pixels."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np


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


def compare_passes(first: dict, second: dict, blind_protocol: dict) -> dict:
    for data in (first, second):
        validate_annotations(data)
        if any(image["review_status"] != "reviewed" for image in data["images"]):
            raise ValueError("both_passes_must_be_human_reviewed")
    if first["feature_definition"] != second["feature_definition"]:
        raise ValueError("feature_definition_must_match")
    if (blind_protocol.get("independent_blind_passes") is not True
            or not isinstance(blind_protocol.get("record"), str) or not blind_protocol["record"].strip()):
        raise ValueError("explicit_blind_annotation_protocol_record_required")
    a = {row["image_sha256"]: row for row in first["images"]}
    b = {row["image_sha256"]: row for row in second["images"]}
    if set(a) != set(b):
        raise ValueError("blind_passes_require_identical_image_contents")
    rows, distances, disagreements = [], [], []
    for digest in sorted(a):
        left, right = a[digest], b[digest]
        if (left["width"], left["height"]) != (right["width"], right["height"]):
            raise ValueError("annotation_image_dimensions_mismatch")
        if left["split"] != "train" or right["split"] != "train":
            raise ValueError("repeatability_tolerance_uses_development_train_images_only")
        points = {}
        for name in first["labels"]:
            p, q = left["points"][name], right["points"][name]
            if p["visibility"] != q["visibility"]:
                disagreements.append({"image_sha256": digest, "label": name,
                                      "first": p["visibility"], "second": q["visibility"]})
            if p["visibility"] == q["visibility"] == "visible":
                if p["source"] != "human" or q["source"] != "human":
                    raise ValueError("repeatability_requires_human_point_sources")
                delta = np.asarray(q["xy"])-np.asarray(p["xy"])
                distance = float(np.linalg.norm(delta))
                points[name] = {"delta_xy_px": delta.tolist(), "distance_px": distance}
                distances.append(distance)
        rows.append({"image": left["image"], "image_sha256": digest, "points": points})
    if not distances:
        raise ValueError("no_common_visible_points")
    values = np.asarray(distances)
    return {"schema_version": 1, "kind": "blind_annotation_repeatability", "source": "human_annotations",
            "image_count": len(rows), "planned_image_count": 20, "count_budget_met": len(rows) >= 20,
            "point_count": len(values), "mean_distance_px": float(values.mean()),
            "rms_distance_px": float(np.sqrt(np.mean(values**2))), "median_distance_px": float(np.median(values)),
            "p95_distance_px": float(np.quantile(values, .95)), "max_distance_px": float(values.max()),
            "visibility_disagreements": disagreements, "blind_protocol": blind_protocol,
            "tolerance_frozen": False, "tolerance_px": None, "rows": rows,
            "interpretation": "Inter-pass differences measure repeatability, not absolute accuracy. Review disagreements and freeze a separate tolerance before held-out evaluation."}


def repeatability_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--blind-record", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    first = json.loads(args.first.read_text(encoding="utf-8-sig"))
    second = json.loads(args.second.read_text(encoding="utf-8-sig"))
    report = compare_passes(first, second, {"independent_blind_passes": True, "record": args.blind_record})
    report["input_sha256"] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.first, args.second)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("image_count", "point_count", "p95_distance_px", "tolerance_frozen")}))


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "repeatability":
        return repeatability_main(arguments[1:])
    parser = argparse.ArgumentParser(description="Nine-hole annotation validation and blind-pass repeatability")
    parser.add_argument("command", choices=("repeatability",))
    parser.parse_args(arguments)


if __name__ == "__main__":
    main()
