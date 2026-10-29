"""Offline review must preserve edits and bind labels to original image content."""
import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live/python"))

from keypoint_annotations import LABELS, sha256_file
from prepare_keypoint_review import build_annotations, prepare, render_review, require_same_manifest


class KeypointReviewTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("keypoint-review-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        for directory in ("config", "live/python", "images/train", "artifacts"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        camera = self.root / "config/camera_intrinsics.json"
        camera.write_text(json.dumps({"image_width": 200, "image_height": 150}), encoding="utf-8")
        model = self.root / "config/charging_port_model.csv"
        model.write_text("label,x_mm,y_mm,z_mm\n" + "\n".join(f"{name},0,0,0" for name in LABELS), encoding="utf-8")
        source = self.root / "live/python/pnp2.py"
        source.write_text("# test fixture\n", encoding="utf-8")
        self.image_path = self.root / "images/train/a.png"
        image = np.zeros((150, 200, 3), dtype=np.uint8)
        image[40:80, 50:100] = 200
        self.image_path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
        self.batch = {"provenance": {"camera_config_sha256": sha256_file(camera),
                                     "model_points_sha256": sha256_file(model),
                                     "source_sha256": {"pnp2.py": sha256_file(source)}},
                      "rows": [{"image": "images\\train\\a.png", "image_sha256": sha256_file(self.image_path),
                                "split": "train", "roi_bbox_xyxy": [10, 15, 180, 140],
                                "image_points_full": {"S_neg": [50., 60.]}}]}
        self.batch_path = self.root / "artifacts/batch.json"
        self.batch_path.write_text(json.dumps(self.batch), encoding="utf-8")
        self.selection = {"purpose": "development_diagnostic", "independent_test_set": False,
                          "images": [{"image": "images/train/a.png", "selection_reasons": ["诊断"]}]}
        self.selection_path = self.root / "config/selection.json"
        self.selection_path.write_text(json.dumps(self.selection), encoding="utf-8")

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("Test cleanup outside intended directory")
        shutil.rmtree(self.root)

    def build(self):
        return build_annotations(self.selection, self.batch, self.root, self.batch_path)

    def test_initial_annotations_do_not_promote_automatic_points(self):
        annotation = self.build()
        row = annotation["images"][0]
        self.assertEqual(row["auto_points"], {"S_neg": [50., 60.]})
        self.assertEqual(row["review_status"], "draft")
        self.assertTrue(all(p["xy"] is None and p["visibility"] == "unlabeled" for p in row["points"].values()))
        self.assertFalse(annotation["independent_test_set"])

    def test_rebuild_preserves_existing_review_json_exactly(self):
        folder = self.root / "artifacts/review"
        path, page = prepare(self.selection_path, self.batch_path, folder, self.root)
        data = json.loads(path.read_text(encoding="utf-8"))
        row = data["images"][0]
        for point in row["points"].values():
            point.update(visibility="uncertain", source="human")
        row.update(review_status="reviewed", reviewer="reviewer", reviewed_at="2026-10-27T12:00:00+08:00")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=3), encoding="utf-8")
        reviewed_bytes = path.read_bytes()
        prepare(self.selection_path, self.batch_path, folder, self.root)
        self.assertEqual(path.read_bytes(), reviewed_bytes)
        self.assertIn('"review_status": "reviewed"', page.read_text(encoding="utf-8"))

    def test_changed_manifest_refused_without_overwriting_output(self):
        folder = self.root / "artifacts/review"
        path, page = prepare(self.selection_path, self.batch_path, folder, self.root)
        previous_html = page.read_bytes()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["images"][0]["auto_points"]["S_neg"] = [51., 60.]
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            prepare(self.selection_path, self.batch_path, folder, self.root)
        self.assertEqual(page.read_bytes(), previous_html)

    def test_duplicate_content_is_rejected_even_with_different_name(self):
        duplicate = self.root / "images/train/b.png"
        duplicate.write_bytes(self.image_path.read_bytes())
        row = copy.deepcopy(self.batch["rows"][0])
        row["image"] = "images/train/b.png"
        self.batch["rows"].append(row)
        self.selection["images"].append({"image": row["image"], "selection_reasons": []})
        with self.assertRaisesRegex(ValueError, "Duplicate selected image content"):
            self.build()

    def test_point_edits_allowed_but_changed_image_hash_rejected(self):
        original = self.build()
        edited = copy.deepcopy(original)
        edited["images"][0]["points"]["S_neg"].update(xy=[70., 50.], visibility="visible", source="ai_visual")
        require_same_manifest(edited, original)
        edited["images"][0]["image_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            require_same_manifest(edited, original)

    def test_notes_cannot_break_out_of_embedded_json(self):
        data = self.build()
        data["images"][0]["notes"] = '</script><script>alert("bad")</script>'
        html = render_review(data, self.root)
        payload_text = html.split('<script id="payload" type="application/json">', 1)[1].split("</script>", 1)[0]
        self.assertNotIn("<", payload_text)
        payload = json.loads(payload_text)
        self.assertEqual(payload["annotations"]["images"][0]["notes"], data["images"][0]["notes"])
        self.assertTrue(payload["images"][0]["src"].startswith("data:image/png;base64,"))
        self.assertEqual(len(payload["seed_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
