"""Synchronize original connector photos and prepare a reviewable box inventory."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.capture_files import atomic_capture_json
from vision.model_config import current_model_path
from vision.training_library import read_entries, register_image
from vision.photo_sources import resolve_photo

CAPTURE_ROOTS = (
    "artifacts/captures", "artifacts/d415_preview", "artifacts/d415_sdk_apply",
    "artifacts/d415_live_check", "artifacts/d415_usb3_recheck",
    "artifacts/d415_preview_upgrade",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LATEST_LABELS = "artifacts/black_port_finetune/20261030_v2/training_manifest.json"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(base, value):
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"source_path_outside_root: {value}")
    return path


def _date(path):
    match = re.search(r"(20\d{2})(\d{2})(\d{2})T", str(path))
    return "-".join(match.groups()) if match else None


def sync_library(root=ROOT, library=None, extra_sources=()):
    """Idempotent import; known old label candidates cannot undo a review decision."""
    root = Path(root).resolve()
    library = Path(library or root / "dataset/vision").resolve()
    registered = 0
    upgraded_sources = set()
    manifest_path = root / LATEST_LABELS
    if manifest_path.exists():
        for item in _json(manifest_path)["entries"]:
            image = _inside(manifest_path.parent, item["image"])
            label = _inside(manifest_path.parent, item["label"])
            if _sha(image) != item["image_sha256"] or _sha(label) != item["label_sha256"]:
                raise ValueError(f"historical_snapshot_hash_mismatch: {image}")
            original = _inside(root, item["source_image"])
            if original.exists() and _sha(original) != item["image_sha256"]:
                raise ValueError(f"original_image_missing_or_changed: {original}")
            try:
                available = resolve_photo(item["source_image"], root=root, library=library,
                                          expected_sha256=item["image_sha256"])
            except (OSError, ValueError) as exc:
                raise ValueError(f"original_image_missing_or_changed: {original}") from exc
            register_image(available, library_root=library, date=_date(original), label_path=label,
                           source={"path": str(original), "training_manifest": str(manifest_path),
                                   "original_image": str(original),
                                   "original_split": item["split"],
                                   "original_group": item.get("group"),
                                   "snapshot_label": str(label)},
                           annotation_source=item.get("annotation_source", "existing_yolo_label"),
                           human_reviewed=item.get("human_reviewed", False))
            upgraded_sources.add(original)
            registered += 1

    historical = root / "yolo_port/images"
    for image in sorted(historical.rglob("*")) if historical.exists() else []:
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image.relative_to(historical)
        label = root / "yolo_port/labels" / relative.with_suffix(".txt")
        register_image(image, library_root=library,
                       source={"path": str(image.resolve()), "original_split": relative.parts[0]},
                       label_path=label if label.exists() and image.resolve() not in upgraded_sources else None,
                       annotation_source="existing_yolo_port_label")
        registered += 1

    for directory in CAPTURE_ROOTS:
        folder = root / directory
        if not folder.exists():
            continue
        registered_images = set()
        opted_out_sessions = set()
        for path in sorted(folder.rglob("manifest.json")):
            manifest = _json(path)
            if manifest.get("kind") != "realsense_rgb_capture":
                continue
            if manifest.get("metadata", {}).get("training_library_enabled") is False:
                opted_out_sessions.add(path.parent.resolve())
                continue
            camera_path = _inside(path.parent, manifest["camera_file"])
            camera = _json(camera_path)
            if _sha(camera_path) != manifest["camera_sha256"]:
                raise ValueError(f"capture_camera_hash_mismatch: {path}")
            if not camera.get("serial") or camera.get("stream") != "color":
                raise ValueError(f"capture_rgb_identity_required: {path}")
            for frame in manifest["frames"]:
                image = _inside(path.parent, frame["image"])
                if _sha(image) != frame["image_sha256"]:
                    raise ValueError(f"capture_image_hash_mismatch: {image}")
                register_image(image, library_root=library, source={
                    "path": str(image), "manifest": str(path.resolve()),
                    "image": frame["image"], "session_id": manifest["session_id"],
                    "received_at_utc": frame["received_at_utc"],
                    "camera_file": str(camera_path), "camera_sha256": manifest["camera_sha256"],
                    "camera_serial": camera["serial"], "device_model": camera.get("device_model"),
                    "original_split": manifest.get("split"), "metadata": manifest.get("metadata", {}),
                    "preview_manifest": str(path.parent / "preview_manifest.json")
                        if (path.parent / "preview_manifest.json").exists() else None,
                })
                registered_images.add(image)
                registered += 1
        # A saved RGB whose manifest publication failed must not disappear from training audits.
        for image in sorted(folder.rglob("*")):
            if (not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES
                    or image.resolve() in registered_images
                    or any(image.resolve().is_relative_to(session) for session in opted_out_sessions)):
                continue
            original = image.parent.name == "images"
            raw_check = directory.endswith("d415_preview_upgrade") and image.name.startswith("raw_")
            if original or raw_check:
                register_image(image, library_root=library, date=_date(image), source={
                    "path": str(image.resolve()), "import_kind": "unmanifested_original_rgb",
                })
                registered += 1

    for directory in extra_sources:
        directory = Path(directory).resolve()
        if (not directory.is_dir() or directory == library or library.is_relative_to(directory)
                or directory.is_relative_to(library)):
            raise ValueError("extra_source_must_be_a_photo_directory_outside_library")
        for image in sorted(directory.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                register_image(image, library_root=library, date=_date(image),
                               source={"path": str(image), "import_kind": "explicit_photo_directory"})
                registered += 1

    entries = read_entries(library)
    report = {"schema_version": 1, "library": str(library), "source_visits": registered,
              "unique_images": len(entries), "status_counts": dict(Counter(e["status"] for e in entries)),
              "scope": "Original RGB only; duplicate files share one training identity. Excluded images retain reasons.",
              "independent_validation": False}
    atomic_capture_json(library / "sync_report.json", report)
    return report


def write_review(library, output, weights=None, review_all=False):
    library, output = Path(library).resolve(), Path(output).resolve()
    entries = read_entries(library)
    unavailable = [e for e in entries if e["status"] == "excluded" and not (library / e["image"]).exists()]
    records = [e for e in entries
               if (review_all or e["status"] in {"pending", "conflict"}) and e not in unavailable]
    model = None
    if weights:
        from ultralytics import YOLO
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        model = YOLO(str(weights))
    items = []
    for record in records:
        label_text = ""
        if record.get("label") and (library / record["label"]).exists():
            label_text = (library / record["label"]).read_text(encoding="utf-8")
        if not label_text and model is not None and record["status"] == "pending":
            result = model.predict(str(library / record["image"]), conf=.25, device="cpu", verbose=False)[0]
            label_text = "\n".join("0 " + " ".join(f"{x:.8f}" for x in box)
                                   for box in result.boxes.xywhn.cpu().tolist())
        items.append({"image_sha256": record["image_sha256"],
                      "image_url": (library / record["image"]).as_uri(),
                      "width": record["width"], "height": record["height"],
                      "status": record["status"], "label_text": label_text,
                      "sources": record["sources"], "label_candidates": record.get("label_candidates", [])})
    template = Path(__file__).with_name("port_box_review.html").read_text(encoding="utf-8")
    payload = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("/*__REVIEW_DATA__*/[]", payload), encoding="utf-8")
    return {"review_file": str(output), "images": len(items), "suggestions_are_reviewed": False,
            "excluded_missing_copies": len(unavailable)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "dataset/vision")
    parser.add_argument("--source", type=Path, action="append", default=[], help="Extra original-photo directory")
    parser.add_argument("--review", type=Path, help="Write a local editable box-review HTML")
    parser.add_argument("--all", action="store_true", help="Include ready/excluded images in review")
    suggestions = parser.add_mutually_exclusive_group()
    suggestions.add_argument("--weights", type=Path, help="Explicit suggestion weights; default uses config/vision_model.json")
    suggestions.add_argument("--no-suggestions", action="store_true", help="Draw boxes manually without loading YOLO")
    args = parser.parse_args(argv)
    print(json.dumps(sync_library(ROOT, args.library, args.source), ensure_ascii=False, indent=2))
    if args.review:
        weights = None if args.no_suggestions else (
            args.weights.resolve() if args.weights is not None else current_model_path(verify=True))
        if weights is not None:
            print(f"预标注模型: {weights}")
        print(json.dumps(write_review(args.library, args.review, weights, args.all), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
