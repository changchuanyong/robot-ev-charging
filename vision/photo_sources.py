"""Resolve preserved photo identities after the legacy directory is removed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_photo(value, *, root=ROOT, library=None, expected_sha256=None):
    root = Path(root).resolve()
    library = Path(library or root / "dataset/vision").resolve()
    original = (root / value).resolve()
    if not original.is_relative_to(root):
        raise ValueError(f"photo_path_outside_repository: {value}")
    key = original.relative_to(root).as_posix()
    index_path = root / "config/legacy_photo_index.json"
    mapped = None
    if key.startswith("yolo_port/") and index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        mapped = next((item for item in index["images"] if item["legacy_image"] == key), None)
    if mapped:
        if expected_sha256 and mapped["image_sha256"] != expected_sha256:
            raise ValueError(f"legacy_photo_index_hash_mismatch: {key}")
        expected_sha256 = mapped["image_sha256"]
        path = (library / mapped["library_image"]).resolve()
        if not path.is_relative_to(library):
            raise ValueError("legacy_photo_path_outside_library")
    elif original.is_file():
        path = original
    else:
        raise FileNotFoundError(f"原图未找到；请复制本地 dataset/vision 图库: {key}")
    if not path.is_file():
        raise FileNotFoundError(f"本地图库原图缺失，请复制 dataset/vision: {path}")
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"photo_hash_mismatch: {key}")
    return path


def library_label(image_sha256, *, root=ROOT, library=None):
    library = Path(library or Path(root) / "dataset/vision").resolve()
    record = json.loads((library / "records" / f"{image_sha256}.json").read_text(encoding="utf-8"))
    if record["image_sha256"] != image_sha256 or record["status"] != "ready" or not record.get("label"):
        raise ValueError("library_annotation_not_ready")
    path = (library / record["label"]).resolve()
    if not path.is_relative_to(library) or hashlib.sha256(path.read_bytes()).hexdigest() != record["label_sha256"]:
        raise ValueError("library_label_hash_mismatch")
    return path
