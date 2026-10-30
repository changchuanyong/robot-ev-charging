from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
import uuid

import cv2
import numpy as np

from train.python.train_port import CoverageAudit, prepare_training, verify_snapshot, validate_device
from train.python.dataset_split import split_entries, group_for
from vision.training_library import register_image, set_annotation


ROOT = Path(__file__).resolve().parents[1]


class FullPoolTrainingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT / "artifacts/test-tmp" / ("all-train-" + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.library = self.scratch / "library"
        self.weights = self.scratch / "source.pt"
        self.weights.write_bytes(b"snapshot weight bytes; no model training in unit tests")
        self.split_config = self.scratch / "split.json"
        self.split_policy = {"schema_version": 1, "groups": {
            "fixture_train": {"split": "train", "session_ids": ["train_session"]},
            "fixture_val": {"split": "val", "session_ids": ["val_session"]}}}
        self.split_config.write_text(json.dumps(self.split_policy), encoding="utf-8")
        for index in range(7):
            image = self.scratch / f"{index}.png"
            _, encoded = cv2.imencode(".png", np.full((64, 80, 3), index * 30, dtype=np.uint8))
            image.write_bytes(encoded.tobytes())
            label = image.with_suffix(".txt")
            label.write_text("0 .5 .5 .4 .4\n" if index else "", encoding="utf-8")
            register_image(image, library_root=self.library,
                           source={"index": index, "session_id": "train_session" if index < 5 else "val_session"}, label_path=label,
                           annotation_source="synthetic_software_test")

    def cleanup(self):
        if not self.scratch.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("test_directory_outside_workspace_artifacts")
        shutil.rmtree(self.scratch)

    def freeze(self):
        return prepare_training(root=self.scratch, library=self.library, weights=self.weights,
                                project=self.scratch / "runs", synchronize=False, split_config=self.split_config,
                                expected_weights_sha256=hashlib.sha256(self.weights.read_bytes()).hexdigest())

    def test_freeze_is_independent_copy_and_keeps_all_images(self):
        run, manifest = self.freeze()
        self.assertEqual(manifest["image_count"], 7)
        self.assertEqual(manifest["instance_count"], 6)
        self.assertEqual(manifest["split_counts"], {"train": 5, "val": 2})
        self.assertEqual(manifest["evaluation"], "held_out_development")
        self.assertFalse(manifest["independent_validation"])
        yaml = (run / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("train: images/train\nval: images/val", yaml)
        entry = manifest["entries"][0]
        frozen_label = (run / entry["label"]).read_bytes()
        set_annotation(self.library, entry["image_sha256"], "0 .4 .4 .2 .2", annotation_source="updated_review")
        self.weights.write_bytes(b"source changed after snapshot")
        self.assertEqual((run / entry["label"]).read_bytes(), frozen_label)
        self.assertNotEqual((run / "initial.pt").read_bytes(), self.weights.read_bytes())
        next_run, _ = self.freeze()
        self.assertNotEqual(run, next_run)

    def test_pending_sample_blocks_snapshot_instead_of_silently_skipping(self):
        image = self.scratch / "unreviewed.png"
        _, encoded = cv2.imencode(".png", np.full((64, 80, 3), 251, dtype=np.uint8))
        image.write_bytes(encoded.tobytes())
        register_image(image, library_root=self.library, source={})
        with self.assertRaisesRegex(ValueError, "annotation_pending"):
            self.freeze()
        self.assertFalse((self.scratch / "runs").exists())

    def test_snapshot_checks_detect_changes_to_images_labels_weights_and_yaml(self):
        run, manifest = self.freeze()
        result = verify_snapshot(run, manifest)
        self.assertEqual(result["images_checked"], 7)
        self.assertEqual(result["coverage_unit"], "primary_training_image")
        self.assertIn("train/python/train_port.py", manifest["code_sha256"])
        targets = [(manifest["entries"][0]["image"], "image_hash_mismatch"),
                   (manifest["entries"][0]["label"], "label_hash_mismatch"),
                   ("initial.pt", "initial_weights_hash_mismatch"),
                   ("dataset.yaml", "dataset_yaml_hash_mismatch")]
        for relative, error in targets:
            with self.subTest(relative=relative):
                path = run / relative
                before = path.read_bytes()
                path.write_bytes(before + b"changed")
                try:
                    with self.assertRaisesRegex(ValueError, error):
                        verify_snapshot(run, manifest)
                finally:
                    path.write_bytes(before)
        self.assertEqual(verify_snapshot(run, manifest)["status"], "passed")

    def test_actual_ultralytics_loader_and_each_epoch_include_partial_final_batch(self):
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_dataloader, build_yolo_dataset

        run, manifest = self.freeze()
        cfg = get_cfg(overrides={"imgsz": 64, "workers": 0, "batch": 2, "task": "detect",
                                 "mosaic": 1.0, "close_mosaic": 0})
        dataset = build_yolo_dataset(cfg, str(run / "dataset/images/train"), batch=2,
                                     data={"names": {0: "port"}, "nc": 1}, mode="train", stride=32)
        loader = build_dataloader(dataset, batch=2, workers=0, shuffle=True, device="cpu", drop_last=False)
        audit = CoverageAudit(run, manifest)
        self.assertEqual(audit.check_loader(loader)["images"], 5)
        trainer = SimpleNamespace(epoch=0, _coverage_batch_paths=None)
        try:
            for epoch in range(2):
                trainer.epoch = epoch
                audit.start_epoch(trainer)
                sizes = []
                for batch in loader:
                    trainer._coverage_batch_paths = list(batch["im_file"])
                    sizes.append(len(batch["im_file"]))
                    audit.complete_batch(trainer)
                self.assertEqual(sizes, [2, 2, 1])
                audit.complete_epoch(trainer)
            self.assertEqual(audit.completed_epochs, 2)
            coverage = [json.loads(line) for line in (run / "epoch_coverage.jsonl").read_text().splitlines()]
            self.assertTrue(all(item["complete"] and item["unique_images"] == 5 for item in coverage))
            dataset.labels[0]["bboxes"] = np.zeros((0, 4), dtype=np.float32)
            dataset.labels[0]["cls"] = np.zeros((0, 1), dtype=np.float32)
            # Find a labelled entry; the first shuffled file may be the explicit background.
            for label in dataset.labels:
                if len(label["cls"]):
                    label["bboxes"] *= .5
                    break
            with self.assertRaisesRegex(ValueError, "annotation_mismatch"):
                audit.check_loader(loader)
        finally:
            if hasattr(loader, "close"):
                loader.close()

    def test_epoch_missing_or_repeated_images_fails_and_records_evidence(self):
        run, manifest = self.freeze()
        audit = CoverageAudit(run, manifest)
        trainer = SimpleNamespace(epoch=0)
        audit.start_epoch(trainer)
        training = [entry for entry in manifest["entries"] if entry["split"] == "train"]
        trainer._coverage_batch_paths = [str(run / training[0]["image"])] * 2
        audit.complete_batch(trainer)
        with self.assertRaisesRegex(ValueError, "did_not_train_every_image"):
            audit.complete_epoch(trainer)
        evidence = json.loads((run / "epoch_coverage.jsonl").read_text())
        self.assertEqual(len(evidence["missing"]), 4)
        self.assertEqual(list(evidence["repeated"].values()), [2])
        self.assertFalse(evidence["complete"])

    def test_unknown_batch_image_is_not_counted_as_coverage(self):
        run, manifest = self.freeze()
        audit = CoverageAudit(run, manifest)
        trainer = SimpleNamespace(epoch=0, _coverage_batch_paths=[str(self.scratch / "not_in_snapshot.png")])
        with self.assertRaisesRegex(ValueError, "unplanned_images"):
            audit.complete_batch(trainer)
        self.assertEqual(audit.completed_epochs, 0)

    def test_unknown_groups_require_assignment_and_group_members_stay_together(self):
        entries = [{"image_sha256": str(i), "sources": [{"session_id": session}]} for i, session in
                   enumerate(("train_session", "val_session", "new_session", "new_session"))]
        with self.assertRaisesRegex(ValueError, "pending_split"):
            split_entries(entries, self.split_policy)
        self.split_policy["groups"]["session:new_session"] = {"split": "train"}
        assigned = split_entries(entries, self.split_policy)
        self.assertEqual([entry["split"] for entry in assigned], ["train", "val", "train", "train"])
        self.assertEqual(assigned[2]["split_group"], assigned[3]["split_group"])
        self.split_policy["groups"]["conflicting_group"] = {"split": "val", "session_ids": ["train_session"]}
        with self.assertRaisesRegex(ValueError, "multiple_groups"):
            group_for(entries[0], self.split_policy)

    def test_validation_batches_cover_only_held_out_images(self):
        run, manifest = self.freeze()
        audit = CoverageAudit(run, manifest)
        validation = [entry for entry in manifest["entries"] if entry["split"] == "val"]
        labels = []
        for entry in validation:
            rows = np.array([list(map(float, row.split())) for row in
                             (run / entry["label"]).read_text().splitlines()]).reshape(-1, 5)
            labels.append({"im_file": str(run / entry["image"]), "cls": rows[:, :1], "bboxes": rows[:, 1:],
                           "normalized": True, "bbox_format": "xywh"})
        loader = SimpleNamespace(drop_last=False, num_workers=0,
                                 dataset=SimpleNamespace(im_files=[item["im_file"] for item in labels], labels=labels))
        validator = SimpleNamespace(dataloader=loader)
        audit.start_validation(validator)
        for entry in validation:
            validator._coverage_batch_paths = [str(run / entry["image"])]
            audit.complete_validation_batch(validator)
        audit.complete_validation(validator)
        self.assertEqual(audit.validation_passes, 1)
        audit.start_validation(validator)
        with self.assertRaisesRegex(ValueError, "did_not_visit_every"):
            audit.complete_validation(validator)
        training = next(entry for entry in manifest["entries"] if entry["split"] == "train")
        validator._coverage_batch_paths = [str(run / training["image"])]
        with self.assertRaisesRegex(ValueError, "unplanned_images"):
            audit.complete_validation_batch(validator)

    def test_non_generic_weights_rejected_and_cuda_never_silently_falls_back(self):
        with self.assertRaisesRegex(ValueError, "generic_pretrained_weights_hash_required"):
            prepare_training(root=self.scratch, library=self.library, weights=self.weights,
                             synchronize=False, split_config=self.split_config)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
            validate_device("0", fake_torch)
        self.assertEqual(validate_device("cpu", fake_torch)["type"], "cpu")


if __name__ == "__main__":
    unittest.main()
