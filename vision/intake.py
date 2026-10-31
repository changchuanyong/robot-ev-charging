"""Bounded background capture, recoverable library intake and same-frame suggestions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .capture_files import atomic_capture_json
from .training_library import DEFAULT_LIBRARY, _lock, register_image

ROOT = Path(__file__).resolve().parents[1]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _inside(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("capture_file_outside_session")
    return path


def _verified(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"capture_file_hash_mismatch: {path}")
    return data


def _saved_sample(session, saved):
    manifest = _json(session / "manifest.json")
    frames = [f for f in manifest["frames"] if f["image"] == saved["image"]]
    if len(frames) != 1 or frames[0]["image_sha256"] != saved["image_sha256"]:
        raise ValueError("capture_frame_not_in_manifest")
    preview_path = session / "preview_manifest.json"
    preview = _json(preview_path) if preview_path.exists() else None
    if preview is None:
        return manifest, frames[0], None
    samples = [s for s in preview["samples"] if s["image"] == saved["image"]]
    if len(samples) != 1 or samples[0]["image_sha256"] != saved["image_sha256"]:
        raise ValueError("preview_frame_incomplete_requires_recovery")
    return manifest, samples[0], preview


def register_saved(session_path, saved, library_root):
    """Import only committed, verified source pixels; predictions stay pending."""
    # Serialize preflight + image registration + split registration across importers.
    # This is separate from the library's record lock and the split registry lock.
    with _lock(Path(library_root).resolve() / ".intake"):
        return _register_saved(session_path, saved, library_root)


def _register_saved(session_path, saved, library_root):
    session = Path(session_path).resolve()
    manifest, saved, preview = _saved_sample(session, saved)
    if manifest.get("metadata", {}).get("training_library_enabled") is False:
        raise ValueError("session_opted_out_of_training_library")
    from train.python.dataset_split import check_capture_split, ensure_capture_split, read_split, group_for, DEFAULT_SPLIT
    check_capture_split(library_root, manifest["batch_id"], manifest["split"], manifest.get("metadata", {}))
    existing = Path(library_root) / "records" / f"{saved['image_sha256']}.json"
    if existing.exists() and manifest["split"] in {"train", "val"}:
        entry, splits = _json(existing), read_split(DEFAULT_SPLIT)
        assigned = splits["groups"].get(group_for(entry, splits), {}).get("split")
        observed = {s.get("original_split") for s in entry["sources"]} - {None, "unknown", ""}
        if assigned:
            observed.add(assigned)
        if observed - {manifest["split"]}:
            raise ValueError("duplicate_image_crosses_dataset_splits")
    image_path = _inside(session, saved["image"])
    if not image_path.is_relative_to(session / "images"):
        raise ValueError("library_requires_original_capture_image")
    _verified(image_path, saved["image_sha256"])
    camera_path = _inside(session, manifest["camera_file"])
    camera = json.loads(_verified(camera_path, manifest["camera_sha256"]).decode("utf-8-sig"))
    if camera.get("stream") != "color" or not camera.get("serial"):
        raise ValueError("capture_color_identity_required")
    depth = saved.get("depth")
    depth_argument = None
    if depth and depth.get("available"):
        depth_argument = {
            **{key: depth.get(key) for key in ("file_sha256", "raw_pixels_sha256", "depth_scale_m",
               "frame", "color_frame", "status", "reasons", "calibration_file", "calibration_sha256")},
            "path": _inside(session, depth["file"]),
            "source": {"session": str(session), "file": depth["file"],
                       "session_id": manifest["session_id"]},
        }
    suggestion = None
    if preview is not None:
        diagnostic = json.loads(_verified(_inside(session, saved["diagnostic"]),
                                         saved["diagnostic_sha256"]).decode("utf-8-sig"))
        if (diagnostic.get("image_sha256") != saved["image_sha256"]
                or diagnostic.get("frame", {}).get("frame_number") != saved["frame_number"]):
            raise ValueError("box_suggestion_frame_mismatch")
        model_sha = (preview.get("assets", {}).get("weights") or {}).get("sha256")
        if model_sha:
            rows = []
            width, height = saved["width"], saved["height"]
            for box in diagnostic.get("frontend", {}).get("detected_boxes", []):
                xyxy = box.get("xyxy", [])
                if len(xyxy) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xyxy):
                    raise ValueError("invalid_saved_detection_box")
                x1, y1, x2, y2 = xyxy
                x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
                y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("empty_saved_detection_box")
                values = ((x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height)
                rows.append("0 " + " ".join(format(v, ".10g") for v in values))
            suggestion = {"source": "same_frame_detector", "model_sha256": model_sha,
                          "image_sha256": saved["image_sha256"], "frame_number": saved["frame_number"],
                          "label_text": "\n".join(dict.fromkeys(rows)) + ("\n" if rows else "")}
    entry = register_image(image_path, library_root=library_root, date=saved["received_at_utc"][:10],
        source={"kind": manifest["kind"], "session_manifest": str(session / "manifest.json"),
                "image": saved["image"], "image_sha256": saved["image_sha256"],
                "received_at_utc": saved["received_at_utc"], "session_id": manifest["session_id"],
                "batch_id": manifest["batch_id"], "original_split": manifest["split"],
                "metadata": manifest.get("metadata", {}), "camera_file": str(camera_path),
                "camera_sha256": manifest["camera_sha256"], "camera_serial": camera["serial"],
                "device_name": camera.get("device_name"), "device_model": camera.get("device_model")},
        depth=depth_argument, suggestion=suggestion)
    ensure_capture_split(library_root, manifest)
    return entry


def retry_session(session_path, library_root, *, attempts=3):
    """Rebuild pending work from the source manifest after interruption, then retry."""
    with _lock(Path(session_path).resolve() / ".intake-retry"):
        return _retry_session(session_path, library_root, attempts=attempts)


def _retry_session(session_path, library_root, *, attempts):
    session, library = Path(session_path).resolve(), Path(library_root).resolve()
    manifest = _json(session / "manifest.json")
    metadata = manifest.get("metadata", {})
    if metadata.get("training_library_enabled") is False:
        return {"imported": 0, "pending_sync": 0, "skipped": True}
    state_path = session / "intake.json"
    if state_path.exists():
        state = _json(state_path)
        if (state.get("schema_version") != 1 or state.get("session_id") != manifest["session_id"]
                or Path(state["library"]).resolve() != library):
            raise ValueError("intake_identity_mismatch")
    else:
        state = {"schema_version": 1, "session_id": manifest["session_id"],
                 "library": str(library), "items": {}}
    for saved in manifest["frames"]:
        state["items"].setdefault(saved["image_sha256"],
                                  {"image": saved["image"], "status": "pending", "attempts": 0})
    atomic_capture_json(state_path, state)
    for saved in manifest["frames"]:
        item = state["items"][saved["image_sha256"]]
        if item["status"] == "imported":
            if (library / "records" / f"{saved['image_sha256']}.json").is_file():
                continue
        for attempt in range(attempts):
            item["attempts"] += 1
            try:
                register_saved(session, saved, library)
                item.update(status="imported", error=None)
            except Exception as exc:
                item.update(status="pending", error=f"{type(exc).__name__}: {exc}")
            atomic_capture_json(state_path, state)
            if item["status"] == "imported":
                break
            if attempt + 1 < attempts:
                time.sleep(.05 * (attempt + 1))
    return {"imported": sum(i["status"] == "imported" for i in state["items"].values()),
            "pending_sync": sum(i["status"] != "imported" for i in state["items"].values()),
            "errors": [i["error"] for i in state["items"].values() if i.get("error")]}


class CaptureIntake:
    """Single writer, at most two queued snapshots; hardware remains on its owner thread."""
    def __init__(self, session, library_root, *, preview=False):
        self.session, self.preview = session, preview
        self.library = Path(library_root).resolve() if library_root is not None else None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture-intake")
        self._lock = threading.Lock()
        self._seen = set()
        self._closed = False
        self._status = dict(queued=0, saved=0, imported=0, pending_sync=0, failed_saves=0,
                            pending_annotation=0, pending_split=0, last_message="", save_ms=[],
                            recovered_previous=0, previous_pending_sync=0, review_file=None,
                            postprocess_error=None, sync_errors=[])
        self._pool.submit(self._recover_previous)

    def poll(self):
        with self._lock:
            return deepcopy(self._status)

    def _update(self, **values):
        with self._lock:
            self._status.update(values)

    def _recover_previous(self):
        if self.library is None:
            return
        recovered = pending = 0
        for manifest_path in sorted(self.session.path.parent.glob("*/manifest.json")):
            if manifest_path.parent == self.session.path:
                continue
            try:
                path = manifest_path.parent / "intake.json"
                if path.exists():
                    state = _json(path)
                    if Path(state["library"]).resolve() != self.library:
                        continue
                    manifest = _json(manifest_path)
                    before = sum(state["items"].get(f["image_sha256"], {}).get("status") != "imported"
                                 or not (self.library / "records" / f"{f['image_sha256']}.json").is_file()
                                 for f in manifest.get("frames", []))
                else:
                    manifest = _json(manifest_path)
                    metadata = manifest.get("metadata", {})
                    if (metadata.get("training_library_enabled") is not True
                            or not metadata.get("training_library_path")
                            or Path(metadata["training_library_path"]).resolve() != self.library):
                        continue
                    before = len(manifest.get("frames", []))
                if before:
                    result = retry_session(path.parent, self.library, attempts=2)
                    recovered += max(0, before-result["pending_sync"])
                    pending += result["pending_sync"]
            except Exception as exc:
                pending += 1
                self._update(last_message=f"旧会话待同步: {exc}")
        self._update(recovered_previous=recovered, previous_pending_sync=pending)

    def submit(self, image, metadata, overlay=None, *, depth_sample=None):
        frame = metadata.get("frame", {}) if self.preview else metadata
        identity = frame.get("frame_number")
        with self._lock:
            if self._closed:
                raise RuntimeError("capture_intake_closed")
            if identity in self._seen:
                self._status["last_message"] = "该帧已保存或正在保存；待入库项退出时会重试"
                return False
            if self._status["queued"] >= 2:
                self._status["last_message"] = "保存队列忙，请稍后再按 F5"
                return False
            self._seen.add(identity)
            self._status["queued"] += 1
            self._status["last_message"] = "已加入保存队列"
        self._pool.submit(self._save, identity, image.copy(), deepcopy(metadata),
                          overlay.copy() if overlay is not None else None, deepcopy(depth_sample))
        return True

    def _save(self, identity, image, metadata, overlay, depth_sample):
        started = time.monotonic()
        saved = None
        try:
            if self.preview:
                saved = self.session.save(image, metadata, overlay, depth_sample=depth_sample)
            else:
                frame = {key: metadata[key] for key in
                         ("frame_number", "device_timestamp_ms", "timestamp_domain", "received_at_utc")}
                saved = self.session.save(image, **frame)
            if saved is not None:
                with self._lock:
                    self._status["saved"] += 1
                self._refresh(attempts=1)
                self._update(last_message="已保存；图库待同步" if self.poll()["pending_sync"] else "保存完成")
            else:
                self._update(last_message="相同图片已保存，已跳过重复")
        except Exception as exc:
            with self._lock:
                if saved is None:
                    self._status["failed_saves"] += 1
                    self._seen.discard(identity)
                else:
                    self._status["pending_sync"] = max(1, self._status["saved"]-self._status["imported"])
                self._status["last_message"] = f"{'保存失败' if saved is None else '原图已保存，待同步'}: {exc}"
        finally:
            with self._lock:
                self._status["queued"] -= 1
                self._status["save_ms"].append((time.monotonic()-started)*1000)

    def _refresh(self, *, attempts):
        if self.library is None:
            return
        result = retry_session(self.session.path, self.library, attempts=attempts)
        manifest = _json(self.session.path / "manifest.json")
        from train.python.dataset_split import read_split, group_for, DEFAULT_SPLIT
        split = read_split(DEFAULT_SPLIT)
        annotations = groups = 0
        for frame in manifest["frames"]:
            record = self.library / "records" / f"{frame['image_sha256']}.json"
            if record.exists():
                entry = _json(record)
                annotations += entry["status"] in {"pending", "conflict"}
                groups += entry["status"] != "excluded" and group_for(entry, split) not in split["groups"]
        self._update(imported=result["imported"], pending_sync=result["pending_sync"],
                     pending_annotation=annotations, pending_split=groups, sync_errors=result.get("errors", []))

    def _finish(self):
        if self.library is None:
            return
        try:
            self._refresh(attempts=2)
            manifest = _json(self.session.path / "manifest.json")
            if self.poll()["imported"]:
                from train.python.prepare_port_library import write_review
                path = self.session.path / "review.html"
                write_review(self.library, path, weights=None, session_id=manifest["session_id"])
                self._update(review_file=str(path))
        except Exception as exc:
            self._update(last_message=f"采集结束，后续处理待完成: {exc}",
                         postprocess_error=f"{type(exc).__name__}: {exc}")
        try:
            with _lock(self.session.path / ".intake-retry"):
                path = self.session.path / "intake.json"
                state = _json(path)
                state["summary"] = self.poll()
                atomic_capture_json(path, state)
        except Exception as exc:
            self._update(postprocess_error=f"intake_summary_not_saved: {exc}")

    def close(self):
        with self._lock:
            if self._closed:
                return deepcopy(self._status)
            self._closed = True
        self._pool.submit(self._finish)
        self._pool.shutdown(wait=True)
        return self.poll()


def main(argv=None):
    parser = argparse.ArgumentParser(description="补同步已保存的采集会话，不启动相机或训练")
    parser.add_argument("--session", type=Path, help="指定会话；省略时检查 captures 和 d415_preview")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    args = parser.parse_args(argv)
    sessions = [args.session] if args.session else [p.parent for folder in ("captures", "d415_preview")
                for p in (ROOT / "artifacts" / folder).glob("*/manifest.json")]
    results = []
    for session in sessions:
        try:
            result = retry_session(session, args.library)
        except Exception as exc:
            result = {"pending_sync": 1, "error": str(exc)}
        results.append({"session": str(session), **result})
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return int(any(r["pending_sync"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
