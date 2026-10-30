"""Legacy aliases survive removal without losing photos or original annotations."""
import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid

import cv2
import numpy as np

from train.python.migrate_legacy_photos import migrate
from train.python.prepare_port_library import LATEST_LABELS, sync_library
from train.python.convert_labelme_to_yolo_detect import convert_to_library, shape_to_bbox
from vision.photo_sources import library_label, resolve_photo
from vision.training_library import read_entries, register_image

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LegacyPhotoMigrationTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("mig-" + uuid.uuid4().hex[:10])
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.library = self.root / "dataset/vision"
        self.legacy = self.root / "yolo_port"
        self.alias = "yolo_port/images/train/port.jpg"
        self.image = self.root / self.alias
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(cv2.imencode(".jpg", np.full((32, 48, 3), 82, np.uint8))[1].tobytes())
        self.label = self.legacy / "labels/train/port.txt"
        self.label.parent.mkdir(parents=True)
        self.label.write_text("0 .5 .5 .5 .5\n", encoding="utf-8")
        self.labelme = self.image.with_suffix(".json")
        self.labelme.write_text(json.dumps({"imagePath": "port.jpg", "imageWidth": 48, "imageHeight": 32,
                                           "shapes": [{"label": "port", "shape_type": "rectangle",
                                                       "points": [[10, 8], [38, 24]]}]}), encoding="utf-8")
        (self.legacy / "dataset.yaml").write_text("names: [port]\n", encoding="utf-8")
        self.entry = register_image(self.image, library_root=self.library, source={"path": str(self.image)},
                                    label_path=self.label, annotation_source="corrected_label")

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("Test cleanup outside intended directory")
        shutil.rmtree(self.root)

    def remove_fixture_legacy(self):
        if self.legacy.resolve().parent != self.root.resolve():
            raise ValueError("Legacy fixture outside test root")
        shutil.rmtree(self.legacy)

    def test_archive_preserves_all_annotation_bytes_and_resolves_removed_photo(self):
        sources = {p.relative_to(self.root).as_posix(): p.read_bytes()
                   for p in (self.label, self.labelme, self.legacy / "dataset.yaml")}
        result = migrate(self.root)
        self.assertEqual(result["images"], 1)
        self.assertEqual(result["archived_annotations"], 3)
        self.assertTrue(self.legacy.exists())
        index = json.loads((self.root / "config/legacy_photo_index.json").read_text())
        for row in index["source_annotations"]:
            self.assertEqual((self.library / row["archive"]).read_bytes(), sources[row["source"]])
        self.assertEqual(migrate(self.root), result)
        self.remove_fixture_legacy()
        actual = resolve_photo(self.alias, root=self.root, expected_sha256=self.entry["image_sha256"])
        self.assertEqual(actual, self.library / self.entry["image"])
        self.assertEqual(library_label(self.entry["image_sha256"], root=self.root), self.library / self.entry["label"])
        with self.assertRaisesRegex(ValueError, "index_hash_mismatch"):
            resolve_photo(self.alias, root=self.root, expected_sha256="a" * 64)
        actual.write_bytes(actual.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "photo_hash_mismatch"):
            resolve_photo(self.alias, root=self.root)

    def test_unmigrated_or_missing_library_photo_blocks_deletion_readiness(self):
        copy = self.library / self.entry["image"]
        copy.unlink()
        with self.assertRaisesRegex(ValueError, "library_copy_mismatch"):
            migrate(self.root)
        self.assertTrue(self.image.exists())
        self.assertFalse((self.root / "config/legacy_photo_index.json").exists())

    def test_snapshot_resync_uses_verified_index_after_old_directory_removed(self):
        parent = self.root / Path(LATEST_LABELS).parent
        parent.mkdir(parents=True)
        snapshot_image, snapshot_label = parent / "snapshot.jpg", parent / "snapshot.txt"
        snapshot_image.write_bytes(self.image.read_bytes())
        snapshot_label.write_bytes(self.label.read_bytes())
        (self.root / LATEST_LABELS).write_text(json.dumps({"entries": [{
            "image": "snapshot.jpg", "label": "snapshot.txt", "source_image": self.alias,
            "image_sha256": sha(snapshot_image), "label_sha256": sha(snapshot_label), "split": "train"}]}), encoding="utf-8")
        migrate(self.root)
        self.remove_fixture_legacy()
        result = sync_library(self.root)
        self.assertEqual(result["status_counts"], {"ready": 1})
        self.assertEqual(len(read_entries(self.library)), 1)
        result = sync_library(self.root)
        self.assertEqual(result["unique_images"], 1)
        self.assertFalse(self.legacy.exists())

    def test_archived_labelme_preserves_current_corrected_labels(self):
        migrate(self.root)
        before = (self.library / self.entry["label"]).read_bytes()
        self.remove_fixture_legacy()
        result = convert_to_library(self.library / "source_annotations/yolo_port/images", self.library, root=self.root)
        self.assertEqual(result["existing_annotations_preserved"], 1)
        self.assertEqual(result["converted"], 0)
        self.assertEqual((self.library / self.entry["label"]).read_bytes(), before)
        self.assertFalse(self.legacy.exists())
        self.assertIsNotNone(shape_to_bbox({"label": "port", "points": [[1, 1], [20, 20]]}, 48, 32))

    def test_missing_local_library_reports_fresh_clone_requirement(self):
        migrate(self.root)
        self.remove_fixture_legacy()
        (self.library / self.entry["image"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "dataset/vision"):
            resolve_photo(self.alias, root=self.root)


if __name__ == "__main__":
    unittest.main()
