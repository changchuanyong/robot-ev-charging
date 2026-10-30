from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import threading
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np

from vision.training_library import (import_review, normalize_label, read_entries,
                                     register_image, set_annotation, validate_library)
from vision import training_library


ROOT = Path(__file__).resolve().parents[1]


class TrainingLibraryTests(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT / "artifacts/test-tmp" / ("library-" + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.library = self.scratch / "library"
        self.image = self.scratch / "source.png"
        ok, encoded = cv2.imencode(".png", np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3))
        self.assertTrue(ok)
        self.image.write_bytes(encoded.tobytes())
        self.label = self.scratch / "source.txt"
        self.label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    def cleanup(self):
        if not self.scratch.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("test_directory_outside_workspace_artifacts")
        shutil.rmtree(self.scratch)

    def add(self, *, labelled=True, source=None):
        return register_image(self.image, library_root=self.library,
                              source={"received_at_utc": "2026-10-30T00:00:00Z", **(source or {})},
                              label_path=self.label if labelled else None,
                              annotation_source="existing_test_label")

    def test_duplicate_preserves_aliases_and_exact_source_bytes(self):
        data = self.image.read_bytes()
        first = self.add(source={"manifest": "session_a/manifest.json"})
        self.add(source={"manifest": "session_a/manifest.json"})
        second = self.add(source={"manifest": "session_b/manifest.json"})
        self.assertEqual(first["image_sha256"], second["image_sha256"])
        self.assertEqual(len(second["sources"]), 2)
        self.assertEqual(len(read_entries(self.library)), 1)
        self.assertTrue(second["image"].startswith("images/2026-10-30/"))
        self.assertEqual((self.library / second["image"]).read_bytes(), data)
        self.assertEqual(self.image.read_bytes(), data)
        self.assertEqual(len(validate_library(self.library)), 1)

    def test_pending_is_not_a_background_label(self):
        entry = self.add(labelled=False)
        self.assertEqual(entry["status"], "pending")
        self.assertIsNone(entry["label"])
        with self.assertRaisesRegex(ValueError, "annotation_pending"):
            validate_library(self.library)
        self.label.write_text("", encoding="utf-8")
        entry = self.add()
        self.assertEqual(entry["status"], "ready")
        self.assertEqual((self.library / entry["label"]).read_text(), "")
        self.assertEqual(len(validate_library(self.library)), 1)

    def test_conflict_requires_explicit_resolution_and_does_not_reopen(self):
        entry = self.add()
        original = self.label.read_text(encoding="utf-8")
        self.label.write_text("0 0.4 0.4 0.3 0.3\n", encoding="utf-8")
        entry = self.add()
        self.assertEqual(entry["status"], "conflict")
        self.assertEqual(len(entry["label_candidates"]), 2)
        with self.assertRaisesRegex(ValueError, "annotation_conflict"):
            validate_library(self.library)
        resolved = set_annotation(self.library, entry["image_sha256"], original,
                                  annotation_source="visual_review", human_reviewed=False)
        self.assertEqual(resolved["status"], "ready")
        self.assertEqual(resolved["annotation_history"][0]["status"], "conflict")
        entry = self.add()  # Previously observed conflicting source is reimported.
        self.assertEqual(entry["status"], "ready")
        self.label.write_text("0 0.6 0.6 0.2 0.2\n", encoding="utf-8")
        self.assertEqual(self.add()["status"], "conflict")

    def test_exclusion_requires_reason_and_survives_source_reimport(self):
        entry = self.add()
        with self.assertRaisesRegex(ValueError, "exclusion_reason"):
            set_annotation(self.library, entry["image_sha256"], "", annotation_source="review", excluded=True)
        entry = set_annotation(self.library, entry["image_sha256"], "", annotation_source="review",
                               excluded=True, reason="No charging interface in image")
        self.assertEqual(self.add()["status"], "excluded")
        self.assertEqual(len(read_entries(self.library)), 1)
        with self.assertRaisesRegex(ValueError, "no_ready_training_images"):
            validate_library(self.library)

    def test_deleted_excluded_copies_stay_deleted_after_reimport(self):
        entry = self.add()
        image_copy, label_copy = self.library / entry["image"], self.library / entry["label"]
        image_copy.unlink()
        label_copy.unlink()
        excluded = set_annotation(self.library, entry["image_sha256"], "", annotation_source="user_removal",
                                  human_reviewed=True, excluded=True, reason="User permanently removed this photo")
        self.assertFalse(excluded["annotation_history"][-1]["image_available"])
        self.assertFalse(excluded["annotation_history"][-1]["label_available"])
        reimported = self.add(source={"manifest": "another_source/manifest.json"})
        self.assertEqual(reimported["status"], "excluded")
        self.assertEqual(reimported["exclusion_reason"], "User permanently removed this photo")
        self.assertEqual(len(reimported["sources"]), 2)
        self.assertFalse(image_copy.exists())
        self.assertFalse(label_copy.exists())
        with self.assertRaisesRegex(ValueError, "restore_excluded_or_missing"):
            set_annotation(self.library, entry["image_sha256"], self.label.read_text(), annotation_source="undo")
        self.assertFalse(image_copy.exists())
        self.assertFalse(label_copy.exists())

        # Another ready sample remains trainable while the excluded evidence record persists.
        another = self.scratch / "another.png"
        _, encoded = cv2.imencode(".png", np.full((18, 24, 3), 79, dtype=np.uint8))
        another.write_bytes(encoded.tobytes())
        ready = register_image(another, library_root=self.library, source={}, label_path=self.label)
        self.assertEqual([item["image_sha256"] for item in validate_library(self.library)], [ready["image_sha256"]])

    def test_existing_excluded_image_is_still_checked_and_missing_label_is_not_restored(self):
        entry = self.add()
        image_copy, label_copy = self.library / entry["image"], self.library / entry["label"]
        label_copy.unlink()
        set_annotation(self.library, entry["image_sha256"], "", annotation_source="user_removal",
                       excluded=True, reason="Removed from future training")
        self.assertFalse(label_copy.exists())
        with self.assertRaisesRegex(ValueError, "restore_excluded_or_missing"):
            set_annotation(self.library, entry["image_sha256"], self.label.read_text(), annotation_source="undo")
        image_copy.write_bytes(b"changed excluded image")
        with self.assertRaisesRegex(ValueError, "hash_or_dimensions"):
            validate_library(self.library)

    def test_missing_ready_image_or_label_cannot_be_silently_ignored(self):
        entry = self.add()
        image_copy, label_copy = self.library / entry["image"], self.library / entry["label"]
        image_bytes = image_copy.read_bytes()
        image_copy.unlink()
        with self.assertRaisesRegex(ValueError, "image_missing_restore_required"):
            validate_library(self.library)
        with self.assertRaisesRegex(ValueError, "image_missing_restore_required"):
            self.add()
        self.assertFalse(image_copy.exists())
        image_copy.write_bytes(image_bytes)
        label_copy.unlink()
        with self.assertRaisesRegex(ValueError, "library_not_ready"):
            validate_library(self.library)

    def test_invalid_labels_are_rejected_before_writing(self):
        for label in ("0 nan .5 .2 .2", "0 .5 inf .2 .2", "1 .5 .5 .2 .2",
                      "0 .99 .5 .2 .2", "0 .5 .5 0 .2", "0 .5 .5 .2", "bad"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                normalize_label(label)
        self.label.write_text("0 .5 .5 nan .1", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.add()
        self.assertFalse(self.library.exists())

    def test_corrupt_and_depth_images_rejected(self):
        self.image.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "decodable_uint8_RGB"):
            self.add()
        _, encoded = cv2.imencode(".png", np.ones((10, 10), dtype=np.uint16))
        self.image.write_bytes(encoded.tobytes())
        with self.assertRaisesRegex(ValueError, "decodable_uint8_RGB"):
            self.add()

    def test_image_and_label_tampering_and_unindexed_images_fail(self):
        entry = self.add()
        image_path = self.library / entry["image"]
        original = image_path.read_bytes()
        image_path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash_or_dimensions"):
            validate_library(self.library)
        image_path.write_bytes(original)
        label_path = self.library / entry["label"]
        label = label_path.read_bytes()
        label_path.write_text("0 .4 .4 .1 .1", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "label_hash_mismatch"):
            validate_library(self.library)
        label_path.write_bytes(label)
        extra = image_path.with_name("forgotten.png")
        extra.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "unindexed_image"):
            validate_library(self.library)

    def test_review_import_defaults_to_human_review_and_checks_whole_file_first(self):
        entry = self.add(labelled=False)
        path = self.scratch / "review.json"
        review = {"schema_version": 1, "annotations": [
            {"image_sha256": entry["image_sha256"], "label_text": "0 .5 .5 .4 .4"}]}
        path.write_text(json.dumps(review), encoding="utf-8")
        imported = import_review(self.library, path)
        self.assertTrue(imported[0]["human_reviewed"])
        self.assertEqual(imported[0]["annotation_source"], "user_review_json")
        review["annotations"].append({"image_sha256": "a" * 64, "label_text": ""})
        path.write_text(json.dumps(review), encoding="utf-8")
        history_length = len(read_entries(self.library)[0]["annotation_history"])
        with self.assertRaisesRegex(ValueError, "unknown_or_duplicate"):
            import_review(self.library, path)
        self.assertEqual(len(read_entries(self.library)[0]["annotation_history"]), history_length)
        review["annotations"][1] = review["annotations"][0]
        path.write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown_or_duplicate"):
            import_review(self.library, path)

    def test_concurrent_registration_keeps_all_aliases(self):
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(lambda i: self.add(source={"session_id": str(i)}), range(6)))
        self.assertEqual(len(results), 6)
        self.assertEqual(len(read_entries(self.library)[0]["sources"]), 6)
        self.assertEqual(len(validate_library(self.library)), 1)
        self.assertFalse((self.library / ".library.lock").exists())

    def test_validation_waits_for_registration_record_to_be_published(self):
        self.add()
        second = self.scratch / "second.png"
        _, encoded = cv2.imencode(".png", np.full((18, 24, 3), 91, dtype=np.uint8))
        second.write_bytes(encoded.tobytes())
        publishing, release, validation_started, validation_done = (threading.Event() for _ in range(4))
        atomic_bytes = training_library._atomic_bytes

        def pause_before_record(path, data):
            if path.parent.name == "records":
                publishing.set()
                if not release.wait(5):
                    raise TimeoutError("test_writer_release_timeout")
            return atomic_bytes(path, data)

        def validate():
            validation_started.set()
            try:
                return validate_library(self.library)
            finally:
                validation_done.set()

        with patch("vision.training_library._atomic_bytes", side_effect=pause_before_record), \
                ThreadPoolExecutor(max_workers=2) as executor:
            writer = executor.submit(register_image, second, library_root=self.library,
                                     source={}, label_path=self.label)
            try:
                self.assertTrue(publishing.wait(5))
                reader = executor.submit(validate)
                self.assertTrue(validation_started.wait(5))
                self.assertFalse(validation_done.wait(.15))
            finally:
                release.set()
            writer.result(timeout=5)
            self.assertEqual(len(reader.result(timeout=5)), 2)


if __name__ == "__main__":
    unittest.main()
