"""Prepare an offline, original-pixel nine-hole annotation review page."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from keypoint_annotations import LABELS, normalized_image_key, sha256_file, validate_annotations


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from vision.photo_sources import resolve_photo
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT_DIR / "config/vision_annotation_pilot.json")
    parser.add_argument("--batch-report", type=Path, default=ROOT_DIR / "artifacts/batch_eval_optimized/latest_details.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "artifacts/keypoint_review_pilot")
    parser.add_argument("--annotations", type=Path, help="Existing annotation JSON to embed; never overwritten")
    args = parser.parse_args()
    source, page = prepare(args.selection, args.batch_report, args.output_dir, annotations_path=args.annotations)
    print(f"Annotations: {source.resolve()}")
    print(f"Review page: {page.resolve()}")
    print("Offline review only. Export reviewed_annotations.json after human review; AI drafts are not ground truth.")


if __name__ == "__main__":
    main()
