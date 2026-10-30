"""Check whether port training boxes cover the reviewed visible hole centers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "live" / "python"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from keypoint_annotations import normalized_image_key, validate_annotations  # noqa: E402
from vision.photo_sources import library_label, resolve_photo


def detection_bbox(path: Path, width: int, height: int) -> list[float]:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 5:
        raise ValueError("Expected one five-column port detection label")
    if lines[0][0] != "0":
        raise ValueError("Expected port class id 0")
    values = [float(value) for value in lines[0][1:]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite detection label")
    cx, cy, box_width, box_height = values
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
        raise ValueError("Detection label is outside normalized bounds")
    bounds = [cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2]
    # Six-decimal labels can put an edge very slightly beyond the image.
    if any(value < -1e-6 or value > 1 + 1e-6 for value in bounds):
        raise ValueError("Detection box extends outside the image")
    return [bounds[0] * width, bounds[1] * height, bounds[2] * width, bounds[3] * height]


def check_annotations(annotations: dict, root: Path) -> dict:
    validate_annotations(annotations)
    root = root.resolve()
    rows, skipped = [], []
    for entry in annotations["images"]:
        key = normalized_image_key(entry["image"])
        if entry["review_status"] != "reviewed":
            skipped.append(key)
            continue
        original_path = (root / key).resolve()
        split = entry["split"]
        expected_image = root / "yolo_port" / "images" / split / original_path.name
        migrated = key.startswith("dataset/vision/images/") or (root / "config/legacy_photo_index.json").exists()
        if not migrated and (original_path != expected_image or split not in {"train", "val"}):
            raise ValueError(f"Expected a known library or historical train/val image: {key}")
        label_path = root / "yolo_port" / "labels" / split / (original_path.stem + ".txt")
        visible = {name: point["xy"] for name, point in entry["points"].items()
                   if point["visibility"] == "visible"}
        row = {"image": key, "label_file": label_path.relative_to(root).as_posix(),
               "visible_point_count": len(visible), "bbox_xyxy": None, "outside_points": {}, "errors": []}
        try:
            image_path = resolve_photo(key, root=root)
            image_bytes = image_path.read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != entry["image_sha256"]:
                raise ValueError("Image content differs from reviewed annotations")
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[1::-1] != (entry["width"], entry["height"]):
                raise ValueError("Image dimensions differ from reviewed annotations")
            if migrated:
                label_path = library_label(entry["image_sha256"], root=root)
                row["label_file"] = label_path.relative_to(root).as_posix()
            box = detection_bbox(label_path, entry["width"], entry["height"])
            row["bbox_xyxy"] = box
            row["outside_points"] = {name: xy for name, xy in visible.items()
                                     if not (box[0] <= xy[0] <= box[2] and box[1] <= xy[1] <= box[3])}
        except (OSError, ValueError) as exc:
            row["errors"].append(str(exc))
        row["ok"] = not row["errors"] and not row["outside_points"]
        rows.append(row)
    return {"independent_test_set": False, "reviewed_images": len(rows),
            "skipped_draft_images": skipped, "failed_images": sum(not row["ok"] for row in rows),
            "outside_point_count": sum(len(row["outside_points"]) for row in rows),
            "note": "Visible-point containment checks label consistency; it does not certify a complete object box.",
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    report = check_annotations(annotations, args.root)
    if args.output is not None:
        protected = [args.annotations, Path(__file__)]
        protected.extend(args.root / row["image"] for row in report["rows"])
        protected.extend(args.root / row["label_file"] for row in report["rows"])
        if any(args.output.resolve() == path.resolve()
               or (args.output.exists() and path.exists() and args.output.samefile(path)) for path in protected):
            raise ValueError("Report output cannot overwrite an input or source")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed_images"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
