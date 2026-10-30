"""Content-addressed RGB library with explicit YOLO annotation readiness."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

import cv2
import numpy as np

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "dataset" / "vision"
_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_HASH = re.compile(r"[0-9a-f]{64}")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("library_path_outside_root")
    return path


@contextmanager
def _lock(root, timeout=10.):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".library.lock"
    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout
    while True:
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(token)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"library_busy_or_stale_lock: {path}")
            time.sleep(.05)
    try:
        yield
    finally:
        # A process must never remove another writer's lock.
        if path.exists() and path.read_text(encoding="utf-8") == token:
            path.unlink()


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 5:
                    raise
                time.sleep(.025 * 2 ** attempt)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_label(label_text: str) -> str:
    """Validate class 0 boxes, including finite values and full image bounds."""
    if not isinstance(label_text, str):
        raise ValueError("YOLO_label_text_required")
    rows = []
    for number, line in enumerate(label_text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"YOLO_label_requires_5_columns: line {number}")
        try:
            category, x, y, width, height = map(float, values)
        except ValueError as exc:
            raise ValueError(f"YOLO_label_not_numeric: line {number}") from exc
        if not all(math.isfinite(value) for value in (category, x, y, width, height)):
            raise ValueError(f"YOLO_label_not_finite: line {number}")
        if category != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"YOLO_label_invalid_class_or_bounds: line {number}")
        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            raise ValueError(f"YOLO_box_outside_image: line {number}")
        rows.append("0 " + " ".join(format(value, ".10g") for value in (x, y, width, height)))
    if len(rows) != len(set(rows)):
        raise ValueError("YOLO_duplicate_box")
    return "\n".join(sorted(rows)) + ("\n" if rows else "")


def _date(value, source):
    value = value or source.get("received_at_utc") or source.get("created_at") or "historical"
    if value == "historical":
        return value
    if not isinstance(value, str):
        raise ValueError("library_date_required_as_ISO_date")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError as exc:
        raise ValueError("library_date_required_as_ISO_date") from exc


def _decode(data):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("library_requires_decodable_uint8_RGB_image")
    return image.shape[1], image.shape[0]


def _write_label(root, entry, label_text, annotation_source, human_reviewed):
    relative = f"labels/{entry['date']}/{entry['image_sha256']}.txt"
    _atomic_bytes(_inside(root, relative), label_text.encode("utf-8"))
    entry.update(label=relative, label_sha256=_hash(label_text.encode("utf-8")),
                 annotation_source=annotation_source, human_reviewed=human_reviewed)


def register_image(image_path, *, library_root, source: dict, date: str | None = None,
                   label_path=None, annotation_source=None, human_reviewed=False) -> dict:
    """Copy a raw image once; unlabelled images remain pending, never negatives."""
    root = Path(library_root).resolve()
    image_path = Path(image_path).resolve()
    if image_path.suffix.lower() not in _EXTENSIONS:
        raise ValueError("unsupported_library_image_extension")
    if not isinstance(source, dict) or type(human_reviewed) is not bool:
        raise ValueError("library_source_object_and_review_boolean_required")
    source = json.loads(json.dumps(source, allow_nan=False))
    source.setdefault("original_image", str(image_path))
    data = image_path.read_bytes()
    digest = _hash(data)
    width, height = _decode(data)
    date = _date(date, source)
    label = None
    if label_path is not None:
        label = normalize_label(Path(label_path).read_text(encoding="utf-8-sig"))
        annotation_source = annotation_source or str(Path(label_path).resolve())
    with _lock(root):
        record_path = root / "records" / f"{digest}.json"
        if record_path.exists():
            entry = json.loads(record_path.read_text(encoding="utf-8"))
            stored_image = _inside(root, entry["image"])
            if not stored_image.exists() and entry["status"] != "excluded":
                raise ValueError("library_existing_image_missing_restore_required")
            if stored_image.exists() and _hash(stored_image.read_bytes()) != digest:
                raise ValueError("library_existing_image_hash_mismatch")
        else:
            relative = f"images/{date}/{digest}{image_path.suffix.lower()}"
            _atomic_bytes(_inside(root, relative), data)
            entry = {"schema_version": 1, "image_sha256": digest, "image": relative,
                     "label": None, "label_sha256": None, "width": width, "height": height,
                     "date": date, "status": "pending", "sources": [], "created_at": _now(),
                     "annotation_source": None, "human_reviewed": False,
                     "label_candidates": [], "annotation_history": []}
        if source not in entry["sources"]:
            entry["sources"].append(source)
        if label is not None:
            candidate_hash = _hash(label.encode("utf-8"))
            known = next((item for item in entry["label_candidates"] if item["label_sha256"] == candidate_hash), None)
            provenance = {"source": source, "annotation_source": annotation_source,
                          "human_reviewed": human_reviewed}
            if known is None:
                entry["label_candidates"].append({"label_text": label, "label_sha256": candidate_hash,
                                                   "sources": [provenance]})
                if entry["status"] == "pending":
                    _write_label(root, entry, label, annotation_source, human_reviewed)
                    entry["status"] = "ready"
                elif entry["status"] != "excluded" and entry["label_sha256"] != candidate_hash:
                    entry["status"] = "conflict"
            elif provenance not in known["sources"]:
                known["sources"].append(provenance)
        entry["updated_at"] = _now()
        _atomic_bytes(record_path, json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        return entry


def read_entries(library_root) -> list[dict]:
    root = Path(library_root).resolve()
    entries = []
    for path in sorted((root / "records").glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("schema_version") != 1 or entry.get("image_sha256") != path.stem or not _HASH.fullmatch(path.stem):
            raise ValueError(f"invalid_library_record: {path}")
        entries.append(entry)
    return entries


def _annotation_input(label_text, annotation_source, human_reviewed, excluded, reason):
    if not isinstance(annotation_source, str) or not annotation_source.strip():
        raise ValueError("annotation_source_required")
    if type(human_reviewed) is not bool or type(excluded) is not bool:
        raise ValueError("annotation_boolean_required")
    if excluded and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("exclusion_reason_required")
    return normalize_label(label_text)


def set_annotation(library_root, image_sha256, label_text, *, annotation_source,
                   human_reviewed=False, excluded=False, reason=None) -> dict:
    """Resolve a pending/conflicting entry; retain prior candidate annotations."""
    label_text = _annotation_input(label_text, annotation_source, human_reviewed, excluded, reason)
    if not isinstance(image_sha256, str) or not _HASH.fullmatch(image_sha256):
        raise ValueError("invalid_image_sha256")
    root = Path(library_root).resolve()
    with _lock(root):
        record_path = root / "records" / f"{image_sha256}.json"
        entry = json.loads(record_path.read_text(encoding="utf-8"))
        stored_image = _inside(root, entry["image"])
        stored_label = _inside(root, entry["label"]) if entry.get("label") else None
        if not excluded and (not stored_image.exists() or
                             (entry["status"] == "excluded" and stored_label is not None and not stored_label.exists())):
            raise ValueError("restore_excluded_or_missing_library_files_before_annotation")
        if stored_image.exists() and _hash(stored_image.read_bytes()) != image_sha256:
            raise ValueError("library_image_hash_mismatch")
        history = {key: entry.get(key) for key in
                   ("status", "label_sha256", "annotation_source", "human_reviewed", "exclusion_reason")}
        history["image_available"] = stored_image.exists()
        history["label_available"] = stored_label is not None and stored_label.exists()
        if history["label_available"]:
            history["label_text"] = stored_label.read_text(encoding="utf-8")
        history["changed_at"] = _now()
        entry["annotation_history"].append(history)
        if excluded:
            # An exclusion is a persistent decision, including when the user removed copies.
            entry.update(annotation_source=annotation_source, human_reviewed=human_reviewed)
        else:
            _write_label(root, entry, label_text, annotation_source, human_reviewed)
            selected_hash = entry["label_sha256"]
            if not any(candidate["label_sha256"] == selected_hash for candidate in entry["label_candidates"]):
                entry["label_candidates"].append({"label_text": label_text, "label_sha256": selected_hash,
                                                   "sources": [{"annotation_source": annotation_source,
                                                                "human_reviewed": human_reviewed}]})
        entry.update(status="excluded" if excluded else "ready",
                     exclusion_reason=reason if excluded else None, updated_at=_now())
        _atomic_bytes(record_path, json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        return entry


def validate_library(library_root) -> list[dict]:
    """Return all trainable records, failing if any sample is unresolved or lost."""
    root = Path(library_root).resolve()
    if not root.is_dir():
        raise ValueError("library_not_ready: no_ready_training_images")
    with _lock(root):
        return _validate_library_locked(root)


def _validate_library_locked(root) -> list[dict]:
    entries = read_entries(root)
    errors, trainable, indexed = [], [], set()
    for entry in entries:
        digest = entry["image_sha256"]
        try:
            path = _inside(root, entry["image"])
            indexed.add(path)
            if entry["status"] == "excluded":
                if not entry.get("exclusion_reason"):
                    raise ValueError("exclusion_reason_required")
                if not path.exists():
                    continue
            elif not path.exists():
                raise ValueError("image_missing_restore_required")
            data = path.read_bytes()
            if _hash(data) != digest or _decode(data) != (entry["width"], entry["height"]):
                raise ValueError("image_hash_or_dimensions_mismatch")
            if entry["status"] == "excluded":
                continue
            if entry["status"] != "ready":
                raise ValueError(f"annotation_{entry['status']}")
            if not entry.get("label"):
                raise ValueError("label_missing")
            raw_label = _inside(root, entry["label"]).read_bytes()
            if _hash(raw_label) != entry["label_sha256"]:
                raise ValueError("label_hash_mismatch_reimport_review_required")
            normalize_label(raw_label.decode("utf-8"))
            trainable.append(entry)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(f"{digest}: {exc}")
    for path in (root / "images").rglob("*"):
        if path.is_file() and path.resolve() not in indexed:
            errors.append(f"unindexed_image: {path.relative_to(root)}")
    if not trainable:
        errors.append("no_ready_training_images")
    if errors:
        raise ValueError("library_not_ready:\n" + "\n".join(errors))
    return trainable


def import_review(library_root, review_path) -> list[dict]:
    payload = json.loads(Path(review_path).read_text(encoding="utf-8-sig"))
    if (payload.get("schema_version") != 1 or payload.get("kind", "yolo_port_library_review") != "yolo_port_library_review"
            or not isinstance(payload.get("annotations"), list)):
        raise ValueError("invalid_library_review_schema")
    known = {entry["image_sha256"] for entry in read_entries(library_root)}
    seen = set()
    for item in payload["annotations"]:
        digest = item.get("image_sha256")
        if digest not in known or digest in seen:
            raise ValueError("review_unknown_or_duplicate_image")
        seen.add(digest)
        _annotation_input(item.get("label_text"), item.get("annotation_source", "user_review_json"),
                          item.get("human_reviewed", True), item.get("excluded", False), item.get("reason"))
    return [set_annotation(library_root, item["image_sha256"], item["label_text"],
                           annotation_source=item.get("annotation_source", "user_review_json"),
                           human_reviewed=item.get("human_reviewed", True),
                           excluded=item.get("excluded", False), reason=item.get("reason"))
            for item in payload["annotations"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description="统一充电口图片库状态与标注导入")
    parser.add_argument("command", choices=["status", "annotate"])
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--review-json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "annotate":
        if args.review_json is None:
            parser.error("annotate 需要 --review-json")
        print(f"已导入 {len(import_review(args.library, args.review_json))} 份标注")
    entries = read_entries(args.library)
    counts = {status: sum(entry["status"] == status for entry in entries)
              for status in ("ready", "pending", "conflict", "excluded")}
    print(json.dumps({"library": str(args.library.resolve()), "total": len(entries), **counts},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
