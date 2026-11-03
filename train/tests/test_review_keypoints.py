import copy
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from vision.review.keypoint_annotations import validate_annotations
from vision.review.review_keypoints import KeypointStore, ROOT, _version, create_server
from vision.review.keypoint_suggestions import valid_points


class KeypointReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.library = self.root / "dataset/vision"
        (self.library / "images").mkdir(parents=True)
        (self.library / "records").mkdir()
        for index, status in enumerate(("ready", "pending", "excluded")):
            self.add_image(index, status)
        self.store = KeypointStore(self.library, root=self.root)

    def add_image(self, index, status="ready"):
        _, encoded = cv2.imencode(".png", np.full((48, 64, 3), index * 40, dtype=np.uint8))
        content = encoded.tobytes()
        digest = hashlib.sha256(content).hexdigest()
        image = f"images/{digest}.png"
        (self.library / image).write_bytes(content)
        entry = dict(schema_version=1, image=image, image_sha256=digest, width=64, height=48,
                     status=status, sources=[{"original_split": "train"}])
        (self.library / "records" / f"{digest}.json").write_text(json.dumps(entry), encoding="utf-8")

    def changed_row(self):
        row = copy.deepcopy(self.store.snapshot()["images"][0])
        version = _version(row)
        row["points"]["S_neg"].update(xy=[10.5, 12.25], visibility="visible", source="human")
        return row, version

    def test_all_images_and_draft_restore_without_touching_detection_records(self):
        records = {path: path.read_bytes() for path in (self.library / "records").glob("*.json")}
        self.assertEqual(len(self.store.snapshot()["images"]), 3)
        row, version = self.changed_row()
        self.store.save(row, version)
        restored = KeypointStore(self.library, root=self.root).snapshot()
        self.assertEqual(restored["images"][0], row)
        self.assertTrue(all(path.read_bytes() == content for path, content in records.items()))
        self.add_image(3)
        expanded = KeypointStore(self.library, root=self.root).snapshot()
        self.assertEqual(len(expanded["images"]), 4)
        self.assertEqual(next(item for item in expanded["images"] if item["image_sha256"] == row["image_sha256"]), row)

    def test_stale_tab_cannot_overwrite_saved_work(self):
        row, version = self.changed_row()
        self.store.save(row, version)
        row["notes"] = "stale change"
        with self.assertRaisesRegex(ValueError, "其他页面"):
            self.store.save(row, version)
        self.assertEqual(self.store.snapshot()["images"][0]["notes"], "")

    def test_confirmation_requires_all_nine_statuses(self):
        row, version = self.changed_row()
        row.update(review_status="reviewed", reviewer="test-reviewer", reviewed_at="2026-11-02T03:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "unlabeled point"):
            self.store.save(row, version)
        for point in row["points"].values():
            if point["visibility"] == "unlabeled":
                point.update(visibility="uncertain", source="human")
        self.store.save(row, version)
        validate_annotations(self.store.snapshot())
        self.assertEqual(self.store.snapshot()["images"][0]["review_status"], "reviewed")

    def test_outside_pixels_and_tampered_identity_rejected(self):
        row, version = self.changed_row()
        row["points"]["S_neg"]["xy"] = [64, 20]
        with self.assertRaisesRegex(ValueError, "outside image"):
            self.store.save(row, version)
        row, version = self.changed_row()
        row["image"] = "config/other.json"
        with self.assertRaisesRegex(ValueError, "图片身份"):
            self.store.save(row, version)

    def test_changed_original_cannot_receive_annotations(self):
        row, version = self.changed_row()
        (self.root / row["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "原图内容"):
            self.store.save(row, version)

    def publish_model(self, name):
        (self.root / "config").mkdir(exist_ok=True)
        weights = self.root / (name + ".pt")
        weights.write_bytes(name.encode("utf-8"))
        registry = {"schema_version": 1, "weights": weights.name,
                    "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
        (self.root / "config/vision_model.json").write_text(json.dumps(registry), encoding="utf-8")
        return weights

    def test_published_weights_invalidate_suggestions_without_restart(self):
        row = self.store.snapshot()["images"][0]
        self.publish_model("first")
        generate = Mock(side_effect=lambda entry, image, model: {"points": {}, "weights_sha256": model["weights_sha256"]})
        with patch.object(self.store.suggestion_engine, "suggest", generate):
            first = self.store.suggest(row["image_sha256"])
            self.assertEqual(self.store.suggest(row["image_sha256"]), first)
            self.assertEqual(generate.call_count, 1)
            self.publish_model("second")
            second = self.store.suggest(row["image_sha256"])
        self.assertNotEqual(first["weights_sha256"], second["weights_sha256"])
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(self.store.snapshot()["images"][0]["review_status"], "draft")
        self.assertFalse(self.store.path.exists())

    def test_damaged_weights_do_not_reuse_cached_old_result(self):
        digest = self.store.snapshot()["images"][0]["image_sha256"]
        weights = self.publish_model("first")
        with patch.object(self.store.suggestion_engine, "suggest", return_value={"points": {}}) as generate:
            self.store.suggest(digest)
            weights.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.store.suggest(digest)
            self.assertEqual(generate.call_count, 1)

    def test_ready_image_uses_detector_even_when_old_label_exists(self):
        self.publish_model("first")
        engine = self.store.suggestion_engine
        detector = Mock(return_value=[])
        with patch.object(engine, "_load_detector", return_value=detector):
            result = engine.suggest({"status": "ready", "label": "old_label.txt"}, np.zeros((48, 64, 3), dtype=np.uint8))
        detector.assert_called_once()
        self.assertEqual(result["points"], {})
        self.assertEqual(result["weights_sha256"], engine.current_model()["weights_sha256"])

    def test_detector_reloads_when_published_weights_change(self):
        from train.python.runtime_env import configure_caches

        configure_caches(ROOT)
        self.publish_model("first")
        engine = self.store.suggestion_engine
        with patch("ultralytics.YOLO", side_effect=lambda path: Mock()) as yolo:
            first = engine._load_detector(engine.current_model())
            self.assertIs(engine._load_detector(engine.current_model()), first)
            self.publish_model("second")
            self.assertIsNot(engine._load_detector(engine.current_model()), first)
            self.assertEqual(yolo.call_count, 2)

    def test_suggestion_coordinates_reject_boolean_nan_and_outside_image(self):
        self.assertEqual(valid_points({"PE": [True, 1], "CC2": [float("nan"), 1],
                                       "DC_neg": [-1, 4], "DC_pos": [63.5, 47.5]}, 64, 48),
                         {"DC_pos": [63.5, 47.5]})

    def test_http_page_image_save_and_cross_origin_rejection(self):
        server = create_server(self.library, root=self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.shutdown)
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("GET", "/")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        html = response.read().decode("utf-8")
        payload = json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)[1])
        self.assertEqual(len(payload["images"]), 3)
        self.assertNotIn("__KEYPOINT_PAYLOAD__", html)
        connection.request("GET", payload["images"][0]["src"])
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(response.read().startswith(b"\x89PNG"))
        row, version = self.changed_row()
        body = json.dumps(dict(row=row, record_version=version))
        headers = {"Content-Type": "application/json", "X-Review-Token": payload["token"],
                   "Origin": "https://untrusted.invalid"}
        connection.request("POST", "/api/annotation", body, headers)
        response = connection.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        headers["Origin"] = server.review_url.rstrip("/")
        connection.request("POST", "/api/annotation", body, headers)
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())["record_version"], _version(row))
        self.assertEqual(self.store.snapshot()["images"][0], row)


if __name__ == "__main__":
    unittest.main()
