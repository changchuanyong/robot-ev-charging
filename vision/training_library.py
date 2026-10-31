"""Content-addressed RGB library with explicit YOLO annotation readiness."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
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
DEPTH_DIR = "depth"
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
    legacy = root / ".library.lock"
    path = root / ".library.guard"
    deadline = time.monotonic() + timeout
    if legacy.exists():
        raise RuntimeError(f"legacy_library_lock_requires_manual_check: {legacy}")
    if os.name == "nt":
        import msvcrt
    else:
        import fcntl
    # Keep the guard file: unlinking it could let writers lock different files.
    # The OS releases this lock even when its owning process is terminated.
    with path.open("a+b", buffering=0) as stream:
        acquired = False
        try:
            while True:
                if legacy.exists():
                    raise RuntimeError(f"legacy_library_lock_requires_manual_check: {legacy}")
                stream.seek(0)
                try:
                    if os.name == "nt":
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"library_busy: {path}") from exc
                    time.sleep(.05)
            if legacy.exists():
                raise RuntimeError(f"legacy_library_lock_requires_manual_check: {legacy}")
            yield
        finally:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


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


def _calibration_binding(root, source_file, expected, *, preserve=False):
    """Verified means byte identity, never independent calibration acceptance."""
    binding = {"source_file": source_file, "sha256": expected,
               "file": None, "status": "missing"}
    if expected is None:
        binding["status"] = "unverifiable"
        return binding
    if not isinstance(expected, str) or _HASH.fullmatch(expected) is None:
        raise ValueError("calibration_sha256_invalid")
    if preserve and source_file and Path(source_file).is_file():
        if _hash(Path(source_file).read_bytes()) != expected:
            raise ValueError(f"calibration_sha256_mismatch: {source_file}")
    relative = f"calibrations/{expected}.json"
    stored = _inside(root, relative)
    if stored.is_file():
        path = stored
    elif source_file and Path(source_file).is_file():
        path = Path(source_file)
    else:
        return binding
    data = path.read_bytes()
    if _hash(data) != expected:
        raise ValueError(f"calibration_sha256_mismatch: {path}")
    payload = json.loads(data.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"calibration_object_required: {path}")
    if path != stored and preserve:
        _atomic_bytes(stored, data)
    binding.update(file=relative if stored.is_file() else None,
                   status="verified" if stored.is_file() else "source_only")
    return binding


def _calibration_references(entry):
    sources = entry.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("library_sources_must_be_list")
    seen = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("library_source_must_be_object")
        path, digest = source.get("camera_file"), source.get("camera_sha256")
        if any(value is not None and not isinstance(value, str) for value in (path, digest)):
            raise ValueError("library_camera_reference_must_be_strings")
        if (path or digest) and (path, digest) not in seen:
            seen.add((path, digest))
            yield "rgb", path, digest
    depth = entry.get("depth")
    if depth is not None:
        if not isinstance(depth, dict):
            raise ValueError("library_depth_must_be_object")
        path = depth.get("calibration_file")
        source = depth.get("source")
        if source is not None and not isinstance(source, dict):
            raise ValueError("library_depth_source_must_be_object")
        session = (source or {}).get("session")
        if any(value is not None and not isinstance(value, str) for value in (path, session)):
            raise ValueError("library_depth_reference_must_be_strings")
        if path and not Path(path).is_absolute():
            path = str(_inside(Path(session), path)) if session else None
        yield "depth", path, depth.get("calibration_sha256")


def _preserve_entry_calibrations(root, entry):
    rgb = []
    for kind, path, digest in _calibration_references(entry):
        binding = _calibration_binding(root, path, digest, preserve=True)
        if kind == "rgb":
            rgb.append(binding)
        else:
            entry["depth"]["calibration"] = binding
    if rgb:
        entry["camera_calibrations"] = rgb


def calibration_status(library_root):
    """Read-only audit of library-owned copies, independent of saved status flags."""
    root = Path(library_root).resolve()
    counts = {kind: {state: 0 for state in
                    ("verified", "source_only", "missing", "unverifiable", "invalid")}
              for kind in ("rgb", "depth")}
    items, problems = [], []
    entries = read_entries(root)
    without_rgb = 0
    for entry in entries:
        has_rgb = False
        try:
            for kind, path, digest in _calibration_references(entry):
                has_rgb |= kind == "rgb"
                try:
                    binding = _calibration_binding(root, path, digest)
                except (OSError, ValueError, TypeError) as exc:
                    binding = {"source_file": path, "sha256": digest,
                               "file": None, "status": "invalid", "error": str(exc)}
                    problems.append(f"{entry['image_sha256']} {kind}: {exc}")
                counts[kind][binding["status"]] += 1
                items.append({"image_sha256": entry["image_sha256"], "kind": kind, **binding})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{entry['image_sha256']}: {exc}")
        without_rgb += not has_rgb
    return {"records": len(entries), "rgb": counts["rgb"], "depth": counts["depth"],
            "records_without_rgb_reference": without_rgb,
            "problems": problems, "references": items}


def preserve_calibrations(library_root):
    """Copy recoverable parameters and explicitly mark missing historical evidence."""
    root = Path(library_root).resolve()
    changed = 0
    with _lock(root):
        for entry in read_entries(root):
            previous = copy.deepcopy(entry)
            _preserve_entry_calibrations(root, entry)
            if entry != previous:
                entry["updated_at"] = _now()
                _atomic_bytes(root / "records" / f"{entry['image_sha256']}.json",
                              json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
                changed += 1
    return {"updated_records": changed, **calibration_status(root)}


def _depth_payload(depth) -> dict:
    """Verify a paired raw Z16 file without re-encoding or normalizing it."""
    if not isinstance(depth, dict) or not depth.get("path"):
        raise ValueError("library_depth_path_required")
    path = Path(depth["path"]).resolve()
    data = path.read_bytes()
    file_sha256 = depth.get("file_sha256")
    if not isinstance(file_sha256, str) or _HASH.fullmatch(file_sha256) is None:
        raise ValueError("library_depth_file_sha256_required")
    if _hash(data) != file_sha256:
        raise ValueError("library_depth_file_sha256_mismatch")
    raw = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16 or raw.ndim != 2 or min(raw.shape) < 1:
        raise ValueError("library_depth_requires_uint16_z16")
    raw_pixels_sha256 = depth.get("raw_pixels_sha256")
    if not isinstance(raw_pixels_sha256, str) or _HASH.fullmatch(raw_pixels_sha256) is None:
        raise ValueError("library_depth_pixels_sha256_required")
    if _hash(raw.tobytes()) != raw_pixels_sha256:
        raise ValueError("library_depth_pixels_sha256_mismatch")
    scale = depth.get("depth_scale_m")
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("library_depth_scale_required")
    status, reasons = depth.get("status"), depth.get("reasons")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("library_depth_status_required")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise ValueError("library_depth_reasons_required")
    for key in ("frame", "color_frame"):
        if depth.get(key) is not None and not isinstance(depth[key], dict):
            raise ValueError(f"library_depth_{key}_required")
    calibration_file = depth.get("calibration_file")
    if calibration_file is not None and not isinstance(calibration_file, str):
        raise ValueError("library_depth_calibration_file_required")
    calibration_sha256 = depth.get("calibration_sha256")
    if calibration_sha256 is not None and (
            not isinstance(calibration_sha256, str) or _HASH.fullmatch(calibration_sha256) is None):
        raise ValueError("library_depth_calibration_sha256_required")
    source = depth.get("source")
    if source is not None:
        source = json.loads(json.dumps(source, allow_nan=False))
    return {"data": data, "width": int(raw.shape[1]), "height": int(raw.shape[0]),
            "file_sha256": file_sha256, "raw_pixels_sha256": raw_pixels_sha256,
            "depth_scale_m": float(scale), "frame": depth.get("frame"),
            "color_frame": depth.get("color_frame"), "status": status,
            "reasons": list(reasons), "calibration_file": calibration_file,
            "calibration_sha256": calibration_sha256, "source": source}


def _attach_depth(root, entry, payload) -> None:
    """Bind one verified depth frame; the first frame for an image content wins."""
    existing = entry.get("depth")
    if existing is not None:
        if (isinstance(existing, dict)
                and existing.get("raw_pixels_sha256") == payload["raw_pixels_sha256"]):
            return
        conflicts = entry.setdefault("depth_conflicts", [])
        record = {"source": payload.get("source"), "file_sha256": payload["file_sha256"],
                  "raw_pixels_sha256": payload["raw_pixels_sha256"], "recorded_at": _now()}
        if record not in conflicts:
            conflicts.append(record)
        return
    stored_image = _inside(root, entry["image"])
    if entry.get("status") == "excluded" and not stored_image.exists():
        return
    relative = f"{DEPTH_DIR}/{entry['date']}/{entry['image_sha256']}.png"
    _atomic_bytes(_inside(root, relative), payload["data"])
    entry["depth"] = {
        "schema_version": 1, "file": relative,
        "file_sha256": payload["file_sha256"],
        "raw_pixels_sha256": payload["raw_pixels_sha256"],
        "width": payload["width"], "height": payload["height"],
        "dtype": "uint16", "format": "z16",
        "depth_scale_m": payload["depth_scale_m"],
        "frame": payload["frame"], "color_frame": payload["color_frame"],
        "status": payload["status"], "reasons": payload["reasons"],
        "calibration_file": payload["calibration_file"],
        "calibration_sha256": payload["calibration_sha256"],
        "source": payload["source"], "bound_at": _now(),
    }


def _write_label(root, entry, label_text, annotation_source, human_reviewed):
    relative = f"labels/{entry['date']}/{entry['image_sha256']}.txt"
    _atomic_bytes(_inside(root, relative), label_text.encode("utf-8"))
    entry.update(label=relative, label_sha256=_hash(label_text.encode("utf-8")),
                 annotation_source=annotation_source, human_reviewed=human_reviewed)


def register_image(image_path, *, library_root, source: dict, date: str | None = None,
                   label_path=None, annotation_source=None, human_reviewed=False,
                   depth=None, suggestion=None) -> dict:
    """Copy a raw image once; unlabelled images remain pending, never negatives.

    ``depth`` optionally binds one raw Z16 frame to this image content. Depth is
    evidence, not a training channel: validation of training readiness ignores it.
    """
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
    depth_payload = _depth_payload(depth) if depth is not None else None
    if suggestion is not None:
        suggestion = json.loads(json.dumps(suggestion, allow_nan=False))
        if (not isinstance(suggestion, dict)
                or suggestion.get("source") != "same_frame_detector"
                or suggestion.get("image_sha256") != digest
                or not isinstance(suggestion.get("model_sha256"), str)
                or _HASH.fullmatch(suggestion["model_sha256"]) is None):
            raise ValueError("invalid_same_frame_box_suggestion")
        suggestion["label_text"] = normalize_label(suggestion.get("label_text"))
    label = None
    if label_path is not None:
        label = normalize_label(Path(label_path).read_text(encoding="utf-8-sig"))
        annotation_source = annotation_source or str(Path(label_path).resolve())
    with _lock(root):
        # Validate/copy incoming parameters before registering any new RGB or label.
        _preserve_entry_calibrations(root, {"sources": [source],
                                           **({"depth": depth_payload} if depth_payload else {})})
        record_path = root / "records" / f"{digest}.json"
        if record_path.exists():
            entry = json.loads(record_path.read_text(encoding="utf-8"))
            previous = copy.deepcopy(entry)
            stored_image = _inside(root, entry["image"])
            if not stored_image.exists() and entry["status"] != "excluded":
                raise ValueError("library_existing_image_missing_restore_required")
            if stored_image.exists() and _hash(stored_image.read_bytes()) != digest:
                raise ValueError("library_existing_image_hash_mismatch")
        else:
            previous = None
            relative = f"images/{date}/{digest}{image_path.suffix.lower()}"
            _atomic_bytes(_inside(root, relative), data)
            entry = {"schema_version": 1, "image_sha256": digest, "image": relative,
                     "label": None, "label_sha256": None, "width": width, "height": height,
                     "date": date, "status": "pending", "sources": [], "created_at": _now(),
                     "annotation_source": None, "human_reviewed": False,
                     "label_candidates": [], "annotation_history": []}
        if source not in entry["sources"]:
            entry["sources"].append(source)
        if suggestion is not None and suggestion not in entry.get("box_suggestions", []):
            entry.setdefault("box_suggestions", []).append(suggestion)
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
        if depth_payload is not None:
            _attach_depth(root, entry, depth_payload)
        # Also fills missing metadata when an identical depth frame was already bound.
        _preserve_entry_calibrations(root, entry)
        if (previous is not None
                and {key: value for key, value in entry.items() if key != "updated_at"}
                == {key: value for key, value in previous.items() if key != "updated_at"}):
            return previous
        entry["updated_at"] = _now()
        _atomic_bytes(record_path, json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        return entry


def read_entries(library_root) -> list[dict]:
    root = Path(library_root).resolve()
    entries = []
    for path in sorted((root / "records").glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(entry, dict) or entry.get("schema_version") != 1
                or entry.get("image_sha256") != path.stem or not _HASH.fullmatch(path.stem)):
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
                   human_reviewed=False, excluded=False, reason=None,
                   expected_record_sha256=None) -> dict:
    """Resolve a pending/conflicting entry; retain prior candidate annotations."""
    label_text = _annotation_input(label_text, annotation_source, human_reviewed, excluded, reason)
    if not isinstance(image_sha256, str) or not _HASH.fullmatch(image_sha256):
        raise ValueError("invalid_image_sha256")
    root = Path(library_root).resolve()
    with _lock(root):
        record_path = root / "records" / f"{image_sha256}.json"
        record_bytes = record_path.read_bytes()
        if expected_record_sha256 is not None and _hash(record_bytes) != expected_record_sha256:
            raise ValueError("annotation_record_changed_reload_required")
        entry = json.loads(record_bytes.decode("utf-8"))
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


def depth_status(library_root) -> dict:
    """Audit paired raw depth; training readiness never depends on these files."""
    root = Path(library_root).resolve()
    entries = read_entries(root)
    indexed, problems = set(), []
    with_depth = conflicts = 0
    for entry in entries:
        conflicts += len(entry.get("depth_conflicts", []))
        depth = entry.get("depth")
        if depth is None:
            continue
        with_depth += 1
        digest = entry["image_sha256"]
        try:
            if not isinstance(depth, dict):
                raise ValueError("depth_record_must_be_object")
            path = _inside(root, depth["file"])
            indexed.add(path)
            if path.name != f"{digest}.png":
                raise ValueError("depth_file_name_mismatch")
            if not path.exists():
                raise ValueError("depth_file_missing")
            data = path.read_bytes()
            if _hash(data) != depth.get("file_sha256"):
                raise ValueError("depth_file_sha256_mismatch")
            raw = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
                raise ValueError("depth_not_uint16_z16")
            if (raw.shape[1], raw.shape[0]) != (depth.get("width"), depth.get("height")):
                raise ValueError("depth_dimensions_mismatch")
            if _hash(raw.tobytes()) != depth.get("raw_pixels_sha256"):
                raise ValueError("depth_pixels_sha256_mismatch")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            problems.append(f"{digest}: {exc}")
    depth_dir = root / DEPTH_DIR
    if depth_dir.is_dir():
        for path in depth_dir.rglob("*"):
            if path.is_file() and path.resolve() not in indexed:
                problems.append(f"unindexed_depth: {path.relative_to(root)}")
    return {"records": len(entries), "with_depth": with_depth,
            "conflicts": conflicts, "problems": problems}


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
                           excluded=item.get("excluded", False), reason=item.get("reason"),
                           expected_record_sha256=item.get("expected_record_sha256"))
            for item in payload["annotations"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="统一充电口图片库状态与标注导入")
    parser.add_argument("command", choices=["status", "validate", "annotate", "depth-status",
                                           "calibration-status", "preserve-calibrations"])
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--review-json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        entries = validate_library(args.library)
        print(f"Dataset: {args.library.resolve()}\nReady images: {len(entries)}\nTotal blocking issues: 0")
        return 0
    if args.command in {"calibration-status", "preserve-calibrations"}:
        report = (preserve_calibrations(args.library) if args.command == "preserve-calibrations"
                  else calibration_status(args.library))
        print(json.dumps({"library": str(args.library.resolve()),
                          **{key: value for key, value in report.items() if key != "references"}},
                         ensure_ascii=False, indent=2))
        incomplete = any(report[kind][state] for kind in ("rgb", "depth")
                         for state in ("source_only", "missing", "unverifiable", "invalid"))
        return 1 if report["problems"] or incomplete or report["records_without_rgb_reference"] else 0
    if args.command == "annotate":
        if args.review_json is None:
            parser.error("annotate 需要 --review-json")
        print(f"已导入 {len(import_review(args.library, args.review_json))} 份标注")
    if args.command == "depth-status":
        report = depth_status(args.library)
        print(json.dumps({"library": str(args.library.resolve()), **report},
                         ensure_ascii=False, indent=2))
        return 1 if report["problems"] else 0
    entries = read_entries(args.library)
    counts = {status: sum(entry["status"] == status for entry in entries)
              for status in ("ready", "pending", "conflict", "excluded")}
    print(json.dumps({"library": str(args.library.resolve()), "total": len(entries),
                      "with_depth": sum(entry.get("depth") is not None for entry in entries), **counts},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
