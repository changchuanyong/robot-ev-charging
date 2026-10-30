"""File-based entrypoints use the shared selection unless weights are explicit."""
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live/python"))
import batch_pipeline_eval as batch
import roi


class ModelLoaded(Exception):
    pass


class LegacyModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT / "artifacts/test-tmp" / ("model-entry-" + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.weights = self.scratch / "selected.pt"
        self.weights.write_bytes(b"loader fixture")

    def cleanup(self):
        if not self.scratch.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("test_cleanup_outside_workspace")
        shutil.rmtree(self.scratch)

    def test_roi_resolves_verified_model_when_started(self):
        with (patch.object(roi, "current_model_path", return_value=self.weights) as selected,
              patch.object(roi, "YOLO", side_effect=ModelLoaded) as loader):
            with self.assertRaises(ModelLoaded):
                roi.main()
        selected.assert_called_once_with(verify=True)
        loader.assert_called_once_with(self.weights)

    def test_batch_default_and_explicit_override(self):
        for override in ([], ["--model", str(self.weights)]):
            with self.subTest(override=override):
                argv = ["batch_pipeline_eval.py", "--output-dir", str(self.scratch / "output"), *override]
                with (patch.object(sys, "argv", argv),
                      patch.object(batch, "list_images", return_value=[self.scratch / "image.png"]),
                      patch.object(batch, "current_model_path", return_value=self.weights) as selected,
                      patch.object(batch, "YOLO", side_effect=ModelLoaded) as loader):
                    with self.assertRaises(ModelLoaded):
                        batch.main()
                loader.assert_called_once_with(self.weights)
                if override:
                    selected.assert_not_called()
                else:
                    selected.assert_called_once_with(verify=True)

    def test_missing_current_model_does_not_fall_back(self):
        with (patch.object(sys, "argv", ["batch_pipeline_eval.py"]),
              patch.object(batch, "list_images", return_value=[self.scratch / "image.png"]),
              patch.object(batch, "current_model_path", side_effect=FileNotFoundError("current weights")),
              patch.object(batch, "YOLO") as loader):
            with self.assertRaisesRegex(FileNotFoundError, "current weights"):
                batch.main()
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
