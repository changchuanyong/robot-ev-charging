"""Build a manual nine-hole review page for a finalized RGB capture session."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from keypoint_annotations import LABELS, normalized_image_key, validate_annotations
from prepare_keypoint_review import render_review, repo_path, require_same_manifest
from vision.contracts import intrinsics_signature, validate_camera


def _timestamp(value, name):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid capture timestamp: {name}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Capture timestamp needs timezone: {name}")


def build_capture_annotations(manifest_path: Path, root: Path = ROOT_DIR) -> dict:
    root, manifest_path = root.resolve(), manifest_path.resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("Capture session must be inside repository")
    session = manifest_path.parent
    content = manifest_path.read_bytes()
    manifest = json.loads(content)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("kind") != "realsense_rgb_capture"):
        raise ValueError("Expected realsense_rgb_capture manifest schema_version 1")
    if manifest.get("status") not in ("completed", "failed", "interrupted"):
        raise ValueError("Capture must be finalized; recording sessions cannot be reviewed")
    for field in ("session_id", "batch_id"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Missing capture {field}")
    if manifest.get("split") not in ("train", "val", "test", "unknown"):
        raise ValueError("Invalid capture split")
    if not isinstance(manifest.get("notes"), str):
        raise ValueError("Invalid capture notes")
    _timestamp(manifest.get("created_at"), "created_at")
    if manifest.get("camera_file") != "camera.json":
        raise ValueError("Capture camera_file must be camera.json")
    camera_path = repo_path(session, manifest["camera_file"])
    camera_content = camera_path.read_bytes()
    camera_hash = hashlib.sha256(camera_content).hexdigest()
    if camera_hash != manifest.get("camera_sha256"):
        raise ValueError("Capture camera hash changed")
    camera = json.loads(camera_content)
    if not isinstance(camera, dict):
        raise ValueError("Invalid capture camera")
    for field in ("serial", "device_name"):
        if not isinstance(camera.get(field), str) or not camera[field].strip():
            raise ValueError(f"Missing capture camera {field}")
    if camera.get("stream") != "color":
        raise ValueError("Capture requires color camera intrinsics")
    if any(isinstance(camera.get(field), bool) or not isinstance(camera.get(field), int)
           or camera[field] <= 0 for field in ("image_width", "image_height")):
        raise ValueError("Invalid capture camera dimensions")
    validate_camera(camera, (camera["image_height"], camera["image_width"], 3))
    if camera.get("intrinsics_sha256") != intrinsics_signature(camera):
        raise ValueError("Capture intrinsics signature changed")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Capture has no frames to review")
    result = {
        "schema_version": 2, "purpose": "development_diagnostic", "independent_test_set": False,
        "labels": list(LABELS),
        "feature_definition": "visible inner aperture center; physical reference unverified",
        "provenance": {
            "kind": "capture_session",
            "manifest_path": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": hashlib.sha256(content).hexdigest(),
            "camera_config_path": camera_path.relative_to(root).as_posix(),
            "camera_config_sha256": camera_hash,
            "session_id": manifest["session_id"], "batch_id": manifest["batch_id"],
            "camera_serial": camera["serial"],
        },
        "images": [],
    }
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Invalid capture frame")
        name = normalized_image_key(frame.get("image"))
        if not name.startswith("images/") or Path(name).suffix.lower() != ".png":
            raise ValueError("Capture frames must be PNG files under session/images")
        path = repo_path(session, name)
        image_bytes = path.read_bytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest != frame.get("image_sha256"):
            raise ValueError(f"Capture image hash changed: {name}")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable capture image: {name}")
        height, width = image.shape[:2]
        if (frame.get("width"), frame.get("height")) != (width, height):
            raise ValueError(f"Capture image dimensions changed: {name}")
        validate_camera(camera, image.shape)
        number, timestamp = frame.get("frame_number"), frame.get("device_timestamp_ms")
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"Invalid capture frame_number: {name}")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"Invalid device timestamp: {name}")
        if not isinstance(frame.get("timestamp_domain"), str) or not frame["timestamp_domain"].strip():
            raise ValueError(f"Invalid timestamp domain: {name}")
        _timestamp(frame.get("received_at_utc"), f"{name}/received_at_utc")
        result["images"].append({
            "image": path.relative_to(root).as_posix(), "image_sha256": digest,
            "width": width, "height": height, "split": manifest["split"],
            "session_id": manifest["session_id"], "batch_id": manifest["batch_id"],
            "camera_serial": camera["serial"], "selection_reasons": [],
            "review_status": "draft", "reviewer": "", "reviewed_at": None,
            "notes": "", "roi_bbox_xyxy": [], "auto_points": {},
            "points": {label: {"xy": None, "visibility": "unlabeled", "source": "unlabeled", "note": ""}
                       for label in LABELS},
        })
    if "metadata" in manifest:
        result["provenance"]["capture_metadata"] = manifest["metadata"]
        for entry in result["images"]:
            entry["capture_metadata"] = manifest["metadata"]
    validate_annotations(result)
    return result


def prepare(manifest_path: Path, output_dir: Path | None = None, root: Path = ROOT_DIR,
            annotations_path: Path | None = None) -> tuple[Path, Path]:
    expected = build_capture_annotations(manifest_path, root)
    output_dir = (output_dir or manifest_path.resolve().parent / "review").resolve()
    if not output_dir.is_relative_to(root.resolve()):
        raise ValueError("Review output must be inside repository")
    source = annotations_path or output_dir / "annotations.json"
    if annotations_path is not None and not source.exists():
        raise FileNotFoundError(source)
    if source.exists():
        annotations = json.loads(source.read_text(encoding="utf-8"))
        require_same_manifest(annotations, expected)
    else:
        annotations = expected
    page = output_dir / "review.html"
    protected = {manifest_path.resolve(), source.resolve(),
                 repo_path(root, expected["provenance"]["camera_config_path"]),
                 *(repo_path(root, row["image"]) for row in expected["images"])}
    if not page.resolve().is_relative_to(root.resolve()) or page.resolve() in protected:
        raise ValueError("Review page must not replace capture or annotation inputs")
    # All bound inputs are checked before any output is created; an existing
    # annotation JSON is read-only, including an explicitly supplied review export.
    html = render_review(annotations, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        with source.open("x", encoding="utf-8") as handle:
            json.dump(annotations, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    if page.exists():
        page.read_text(encoding="utf-8")
    page.write_text(html, encoding="utf-8")
    return source, page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--annotations", type=Path, help="Existing reviewed JSON to render without overwriting it")
    args = parser.parse_args()
    annotations, page = prepare(args.manifest, args.output_dir, annotations_path=args.annotations)
    print(json.dumps({"annotations": str(annotations), "review_page": str(page)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
