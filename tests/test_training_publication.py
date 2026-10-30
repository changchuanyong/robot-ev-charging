"""Training completion is the commit point for the shared detector selection."""
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import MagicMock, patch
import uuid

from train.python import train_port
from vision.model_config import current_model_path, resolve_model_config


ROOT = Path(__file__).resolve().parents[1]


class TrainingPublicationTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("publication-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        (self.root / "config").mkdir()
        self.registry = self.root / "config/vision_model.json"
        self.registry.write_text('{"existing_selection":"keep on failure"}', encoding="utf-8")
        self.before = self.registry.read_bytes()
        (self.root / "config/d415_preview.json").write_text(
            json.dumps({"model_registry": "config/vision_model.json", "confidence": .25}), encoding="utf-8")
        self.run = self.root / "runs/candidate"
        self.run.mkdir(parents=True)
        (self.run / "weights").mkdir()
        (self.run / "initial.pt").write_bytes(b"initial fixture")
        self.manifest = {"image_count": 2, "split_counts": {"train": 1, "val": 1}, "entries": [],
                         "initial_weights_sha256": hashlib.sha256(b"initial fixture").hexdigest()}
        for split in ("train", "val"):
            image, label = self.run / f"{split}.png", self.run / f"{split}.txt"
            image.write_bytes(f"{split} immutable fixture bytes".encode())
            label.write_text("0 .5 .5 .4 .4\n", encoding="utf-8")
            self.manifest["entries"].append({
                "image": image.name, "label": label.name, "split": split, "split_group": split,
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest()})
        self.config = {"epochs": 1, "batch": 2, "imgsz": 64, "device": "cpu"}
        self.audit = MagicMock(completed_epochs=1, validation_passes=1)

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("test_cleanup_outside_workspace")
        shutil.rmtree(self.root)

    def run_training(self, effect=None):
        model = MagicMock()

        def completed_training(**_):
            (self.run / "weights/best.pt").write_bytes(b"new trained fixture")
            if effect:
                effect()

        model.train.side_effect = completed_training
        with (patch("ultralytics.YOLO", return_value=model),
              patch.object(train_port, "CoverageAudit", return_value=self.audit),
              patch.object(train_port, "_audited_trainer", return_value=object)):
            return train_port.train_all(self.run, self.manifest, self.config, root=self.root)

    def test_completed_run_publishes_shared_model_and_pins_experiment_config(self):
        report = self.run_training()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(current_model_path(root=self.root, verify=True), self.run / "weights/best.pt")
        pinned = json.loads((self.run / "preview_config.json").read_text(encoding="utf-8"))
        self.assertNotIn("model_registry", pinned)
        self.assertIn("weights_sha256", pinned)
        self.assertEqual(resolve_model_config(pinned, root=self.root, verify=True)["weights"],
                         str(self.run / "weights/best.pt"))
        selected = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertEqual(selected["training_report"], "runs/candidate/training_report.json")

    def test_training_failure_or_interrupt_keeps_previous_selection(self):
        for error, status in ((RuntimeError("training failed"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status):
                def fail():
                    raise error
                with self.assertRaises(type(error)):
                    self.run_training(fail)
                self.assertEqual(self.registry.read_bytes(), self.before)
                report = json.loads((self.run / "training_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], status)

    def test_incomplete_coverage_keeps_previous_selection(self):
        for epochs, validations in ((0, 1), (1, 0)):
            with self.subTest(epochs=epochs, validations=validations):
                self.audit.completed_epochs, self.audit.validation_passes = epochs, validations
                with self.assertRaisesRegex(ValueError, "before_all_requested_epochs|coverage_missing"):
                    self.run_training()
                self.assertEqual(self.registry.read_bytes(), self.before)

    def test_snapshot_tampering_blocks_publication_after_training(self):
        def tamper():
            (self.run / "train.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot_integrity_failed"):
            self.run_training(tamper)
        self.assertEqual(self.registry.read_bytes(), self.before)


if __name__ == "__main__":
    unittest.main()
