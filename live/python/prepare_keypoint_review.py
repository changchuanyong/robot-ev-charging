"""Prepare an offline, original-pixel nine-hole annotation review page."""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from keypoint_annotations import LABELS, normalized_image_key, sha256_file, validate_annotations


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from vision.photo_sources import resolve_photo
from vision.contracts import intrinsics_signature, validate_camera
TEMPLATE_PATH = Path(__file__).with_name("keypoint_review.html")
EDITABLE_FIELDS = {"points", "review_status", "reviewer", "reviewed_at", "notes"}


def repo_path(root: Path, value: str) -> Path:
    path = (root / normalized_image_key(value)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {value}")
    return resolve_photo(value, root=root) if str(value).replace("\\", "/").startswith("yolo_port/") else path


def build_annotations(selection: dict, batch: dict, root: Path, batch_report_path: Path) -> dict:
    if selection.get("purpose") != "development_diagnostic" or selection.get("independent_test_set") is not False:
        raise ValueError("Selection must explicitly identify development diagnostic data")
    selected = selection.get("images", [])
    if not selected:
        raise ValueError("Selection is empty")
    rows = {normalized_image_key(row["image"]): row for row in batch["rows"]}
    if len(rows) != len(batch["rows"]):
        raise ValueError("Duplicate image identities in batch report")
    provenance = batch["provenance"]
    camera_path = root / "config/camera_intrinsics.json"
    model_path = root / "config/charging_port_model.csv"
    source_path = root / "live/python/pnp2.py"
    for path, expected in (
        (camera_path, provenance["camera_config_sha256"]),
        (model_path, provenance["model_points_sha256"]),
        (source_path, provenance["source_sha256"]["pnp2.py"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Batch input changed: {path.name}")
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "purpose": "development_diagnostic",
        "independent_test_set": False,
        "labels": list(LABELS),
        "feature_definition": "visible inner aperture center; physical reference unverified",
        "provenance": {
            "batch_report_path": batch_report_path.resolve().relative_to(root.resolve()).as_posix(),
            "batch_report_sha256": sha256_file(batch_report_path),
            "camera_config_path": "config/camera_intrinsics.json",
            "camera_config_sha256": sha256_file(camera_path),
            "model_points_path": "config/charging_port_model.csv",
            "model_points_sha256": sha256_file(model_path),
            "pnp_source_sha256": sha256_file(source_path),
        },
        "images": [],
    }
    identities, contents = set(), set()
    for item in selected:
        name = normalized_image_key(item["image"])
        if name in identities:
            raise ValueError(f"Duplicate selected image: {name}")
        identities.add(name)
        if name not in rows:
            raise ValueError(f"Image absent from batch report: {name}")
        path = repo_path(root, name)
        image_bytes = path.read_bytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        row = rows[name]
        if digest != row.get("image_sha256"):
            raise ValueError(f"Image changed: {name}")
        if digest in contents:
            raise ValueError(f"Duplicate selected image content: {name}")
        contents.add(digest)
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unreadable image: {name}")
        height, width = frame.shape[:2]
        if (width, height) != (camera.get("image_width"), camera.get("image_height")):
            raise ValueError(f"Image size does not match camera intrinsics: {name}")
        result["images"].append({
            "image": name,
            "image_sha256": digest,
            "width": width,
            "height": height,
            "split": row.get("split", "unknown"),
            "selection_reasons": list(item.get("selection_reasons", [])),
            "review_status": "draft",
            "reviewer": "",
            "reviewed_at": None,
            "notes": "",
            "roi_bbox_xyxy": copy.deepcopy(row.get("roi_bbox_xyxy", [])),
            "auto_points": copy.deepcopy(row.get("image_points_full", {})),
            "points": {label: {"xy": None, "visibility": "unlabeled", "source": "unlabeled", "note": ""}
                       for label in LABELS},
        })
    validate_annotations(result)
    return result


def require_same_manifest(annotations: dict, expected: dict) -> None:
    """Allow edits only to review records, never image/provenance bindings."""
    validate_annotations(annotations)
    header = {key: value for key, value in expected.items() if key != "images"}
    other_header = {key: value for key, value in annotations.items() if key != "images"}
    if header != other_header:
        raise ValueError("Annotation header/provenance differs from selected inputs")
    if len(annotations["images"]) != len(expected["images"]):
        raise ValueError("Annotation image count differs from selection")
    for actual, original in zip(annotations["images"], expected["images"]):
        actual_bound = {key: value for key, value in actual.items() if key not in EDITABLE_FIELDS}
        original_bound = {key: value for key, value in original.items() if key not in EDITABLE_FIELDS}
        if actual_bound != original_bound:
            raise ValueError(f"Annotation image metadata differs: {original['image']}")


def render_review(annotations: dict, root: Path, template_path: Path = TEMPLATE_PATH, *,
                  review_images: list[str] | None = None) -> str:
    validate_annotations(annotations)
    indices = list(range(len(annotations["images"])))
    if review_images is not None:
        if not isinstance(review_images, list) or not review_images:
            raise ValueError("review_images must be a nonempty list")
        names = [normalized_image_key(name) for name in review_images]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate review_images selection")
        by_name = {normalized_image_key(row["image"]): index
                   for index, row in enumerate(annotations["images"])}
        if any(name not in by_name for name in names):
            raise ValueError("review_images must belong to the annotation seed")
        indices = [by_name[name] for name in names]
    selected = set(indices)
    images = []
    for index, row in enumerate(annotations["images"]):
        path = repo_path(root, row["image"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image changed: {row['image']}")
        if index in selected:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            images.append({"image": row["image"], "src": f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"})
    annotation_json = json.dumps(annotations, ensure_ascii=False, sort_keys=True, allow_nan=False)
    payload = {"annotations": annotations, "images": images,
               "seed_sha256": hashlib.sha256(annotation_json.encode("utf-8")).hexdigest()}
    if review_images is not None:
        payload["review_indices"] = indices
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = template_path.read_text(encoding="utf-8")
    if template.count("__KEYPOINT_PAYLOAD__") != 1:
        raise ValueError("Review template requires exactly one payload marker")
    return template.replace("__KEYPOINT_PAYLOAD__", serialized)


def prepare(selection_path: Path, batch_report_path: Path, output_dir: Path,
            root: Path = ROOT_DIR, annotations_path: Path | None = None) -> tuple[Path, Path]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    batch = json.loads(batch_report_path.read_text(encoding="utf-8"))
    expected = build_annotations(selection, batch, root, batch_report_path)
    source = annotations_path or output_dir / "annotations.json"
    if annotations_path is not None and not source.exists():
        raise FileNotFoundError(source)
    if source.exists():
        annotations = json.loads(source.read_text(encoding="utf-8"))
        require_same_manifest(annotations, expected)
    else:
        annotations = expected
    # Validate and render before creating any output. Existing annotation files are never overwritten.
    html = render_review(annotations, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        with source.open("x", encoding="utf-8") as handle:
            json.dump(annotations, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    page = output_dir / "review.html"
    if page.exists():
        page.read_text(encoding="utf-8")  # Generated page only; annotations remain in their JSON file.
    page.write_text(html, encoding="utf-8")
    return source, page


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


def prepare_capture(manifest_path: Path, output_dir: Path | None = None, root: Path = ROOT_DIR,
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--selection", type=Path, help="Historical image selection; defaults to config/vision_annotation_pilot.json")
    source.add_argument("--manifest", type=Path, help="Finalized capture manifest; uses this session's camera intrinsics")
    parser.add_argument("--batch-report", type=Path, help="Historical batch report; cannot be combined with --manifest")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the capture session's review/ or artifacts/keypoint_review_pilot")
    parser.add_argument("--annotations", type=Path, help="Existing annotation JSON to embed; never overwritten")
    args = parser.parse_args(argv)
    if args.manifest is not None:
        if args.batch_report is not None:
            parser.error("--batch-report cannot be combined with --manifest")
        source, page = prepare_capture(args.manifest, args.output_dir, annotations_path=args.annotations)
        print(json.dumps({"annotations": str(source), "review_page": str(page)}, ensure_ascii=False))
        return
    source, page = prepare(args.selection or ROOT_DIR / "config/vision_annotation_pilot.json",
                           args.batch_report or ROOT_DIR / "artifacts/batch_eval_optimized/latest_details.json",
                           args.output_dir or ROOT_DIR / "artifacts/keypoint_review_pilot",
                           annotations_path=args.annotations)
    print(f"Annotations: {source.resolve()}")
    print(f"Review page: {page.resolve()}")
    print("Offline review only. Export reviewed_annotations.json after human review; AI drafts are not ground truth.")


if __name__ == "__main__":
    main()
