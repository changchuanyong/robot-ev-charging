"""Verify legacy RGB aliases and preserve original annotation bytes; never delete sources."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.capture_files import atomic_capture_json
from vision.training_library import read_entries


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate(root=ROOT, library=None):
    root = Path(root).resolve()
    library = Path(library or root / "dataset/vision").resolve()
    legacy = root / "yolo_port"
    if not legacy.is_dir():
        raise FileNotFoundError(f"待迁移目录不存在: {legacy}")
    entries = {e["image_sha256"]: e for e in read_entries(library)}
    photos, archives = [], []
    for image in sorted((legacy / "images").rglob("*")):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        digest = sha(image)
        entry = entries.get(digest)
        if entry is None or entry["status"] != "ready":
            raise ValueError(f"legacy_photo_not_ready_in_library: {image}")
        target = (library / entry["image"]).resolve()
        if not target.is_relative_to(library) or not target.is_file() or sha(target) != digest:
            raise ValueError(f"legacy_photo_library_copy_mismatch: {image}")
        photos.append({"legacy_image": image.relative_to(root).as_posix(),
                       "library_image": entry["image"], "image_sha256": digest,
                       "original_split": image.parent.name})
    if not photos:
        raise ValueError("no_legacy_photos")
    for source in sorted(legacy.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in {".json", ".txt", ".yaml", ".yml"}:
            continue
        target = library / "source_annotations" / source.relative_to(root)
        digest = sha(source)
        if target.exists() and sha(target) != digest:
            raise ValueError(f"archived_annotation_already_differs: {target}")
        archives.append({"source": source.relative_to(root).as_posix(),
                         "archive": target.relative_to(library).as_posix(), "sha256": digest})
    # Check every source before copying. No original directory is changed or deleted.
    for item in archives:
        target = library / item["archive"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(root / item["source"], target)
        if sha(target) != item["sha256"]:
            raise ValueError(f"annotation_archive_hash_mismatch: {target}")
    index = {"schema_version": 1, "library": "dataset/vision", "images": photos,
             "source_annotations": archives, "migration_deletes_sources": False}
    path = root / "config/legacy_photo_index.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != index:
        raise ValueError("legacy_photo_index_already_differs")
    atomic_capture_json(path, index)
    report = {"images": len(photos), "unique_images": len({e["image_sha256"] for e in photos}),
              "archived_annotations": len(archives),
              "annotation_extensions": dict(Counter(Path(e["source"]).suffix for e in archives)),
              "index": str(path), "legacy_directory_preserved": True}
    atomic_capture_json(library / "legacy_migration_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(migrate(args.root, args.library), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
