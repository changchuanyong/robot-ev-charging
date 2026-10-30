"""Shared detector resolution and completed-run publication, without hardware."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from vision import model_config
from vision.pipeline import PosePipeline


ROOT = Path(__file__).resolve().parents[1]


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "artifacts/test-tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.root = parent / ("model-config-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.cleanup)
        self.weights = self.root / "runs/checked/weights/best.pt"
        self.weights.parent.mkdir(parents=True)
        self.weights.write_bytes(b"checked test weights")
        self.digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.registry = self.root / model_config.CURRENT_MODEL_CONFIG
        self.registry.parent.mkdir()
        self.record = {"schema_version": 1, "weights": "runs/checked/weights/best.pt",
                       "weights_sha256": self.digest, "training_run": "runs/checked"}
        self.registry.write_text(json.dumps(self.record), encoding="utf-8")
        self.report_path = self.weights.parents[1] / "training_report.json"
        self.report = {"status": "completed", "requested_epochs": 40, "completed_epochs": 40,
                       "snapshot_posttrain_check": {"status": "passed"},
                       "best_weights": str(self.weights), "best_weights_sha256": self.digest,
                       "evaluation": "held_out_development", "independent_validation": False}
        self.write_report()

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("test_directory_outside_workspace_artifacts")
        shutil.rmtree(self.root)

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def test_all_current_camera_configs_share_the_same_registry(self):
        expected = model_config.current_model_path()
        for name in ("config/d415_preview.json", "config/realsense_config.json"):
            config = json.loads((ROOT / name).read_text(encoding="utf-8"))
            resolved = model_config.resolve_model_config(config)
            self.assertEqual(Path(resolved["weights"]), expected)
            self.assertNotIn("weights", config)
            self.assertNotIn("model_registry", resolved)
            self.assertEqual(resolved["model_source"]["kind"], "registry")
        self.assertFalse((ROOT / "config/d415_black_preview.json").exists())

    def test_short_preview_command_selects_the_d415_config_before_camera_access(self):
        from vision import preview
        with patch.object(preview, "DiagnosticFrontend", side_effect=RuntimeError("stop before camera")) as frontend:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(preview.main(["--depth"]), 2)
        received = frontend.call_args.args[0]
        self.assertEqual(received["model_registry"], model_config.CURRENT_MODEL_CONFIG)
        self.assertNotIn("serial", received)
        self.assertNotIn("output", received)

    def test_resolved_config_pins_selection_and_does_not_mutate_input(self):
        config = {"model_registry": model_config.CURRENT_MODEL_CONFIG, "target_id": "test"}
        pinned = model_config.resolve_model_config(config, root=self.root, verify=True)
        self.record["weights"] = "runs/later/weights/best.pt"
        self.registry.write_text(json.dumps(self.record), encoding="utf-8")
        self.assertEqual(model_config.resolve_model_config(pinned, root=self.root, verify=True), pinned)
        self.assertNotIn("weights", config)
        self.assertEqual(Path(pinned["weights"]), self.weights)

    def test_explicit_historical_weights_do_not_consult_current_registry(self):
        self.registry.unlink()
        resolved = model_config.resolve_model_config({"weights": "runs/checked/weights/best.pt"},
                                                     root=self.root, verify=True)
        self.assertEqual(Path(resolved["weights"]), self.weights)
        self.assertNotIn("model_source", resolved)

    def test_ambiguous_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not_both"):
            model_config.resolve_model_config({"model_registry": model_config.CURRENT_MODEL_CONFIG,
                                               "weights": str(self.weights)}, root=self.root)

    def test_missing_or_modified_current_model_never_falls_back(self):
        self.weights.write_bytes(b"different contents")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            model_config.current_model_path(root=self.root, verify=True)
        self.weights.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "weights missing"):
            model_config.current_model_path(root=self.root, verify=True)
        self.registry.unlink()
        with self.assertRaises(FileNotFoundError):
            model_config.current_model_path(root=self.root)

    def test_injected_detector_does_not_require_ignored_weight_files(self):
        config = json.loads((ROOT / "config/d415_preview.json").read_text(encoding="utf-8"))
        config.pop("model_registry")
        config["weights"] = str(self.root / "unavailable.pt")
        pipeline = PosePipeline(config, detector=object())
        pipeline._load()
        self.assertIsNotNone(pipeline.model)

    def test_completed_checked_run_is_published_with_relative_paths(self):
        result = model_config.publish_current_model(self.weights, root=self.root, training_report=self.report_path)
        self.assertEqual(result["weights"], "runs/checked/weights/best.pt")
        self.assertEqual(result["weights_sha256"], self.digest)
        self.assertEqual(result["training_report"], "runs/checked/training_report.json")
        self.assertEqual(json.loads(self.registry.read_text(encoding="utf-8")), result)
        self.assertEqual(model_config.current_model_path(root=self.root, verify=True), self.weights)

    def test_failed_incomplete_or_unchecked_training_preserves_current_model(self):
        before = self.registry.read_bytes()
        for changes in ({"status": "failed"}, {"status": "interrupted"}, {"completed_epochs": 39},
                        {"requested_epochs": 0}, {"snapshot_posttrain_check": {"status": "failed"}}):
            with self.subTest(changes=changes):
                changed = {**self.report, **changes}
                self.report_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "completed_checked"):
                    model_config.publish_current_model(self.weights, root=self.root, training_report=self.report_path)
                self.assertEqual(self.registry.read_bytes(), before)

    def test_wrong_weights_or_hash_preserves_current_model(self):
        before = self.registry.read_bytes()
        with self.assertRaisesRegex(ValueError, "must_match"):
            model_config.publish_current_model(self.root / "other.pt", root=self.root, training_report=self.report_path)
        self.report["best_weights_sha256"] = "0" * 64
        self.write_report()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            model_config.publish_current_model(self.weights, root=self.root, training_report=self.report_path)
        self.assertEqual(self.registry.read_bytes(), before)

    def test_failed_atomic_publication_preserves_current_model(self):
        before = self.registry.read_bytes()
        with patch.object(model_config, "atomic_capture_json", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                model_config.publish_current_model(self.weights, root=self.root, training_report=self.report_path)
        self.assertEqual(self.registry.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
