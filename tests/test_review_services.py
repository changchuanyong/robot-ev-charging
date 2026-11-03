"""Exercise both review APIs against disposable images and annotations."""

import copy
import hashlib
import json
import re
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

import cv2
import numpy as np

from ev_charging.datasets.vision.training_library import register_image
from ev_charging.review import review_keypoints, review_library


class ReviewServicesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.library = self.root / "dataset/vision"
        image = self.root / "image.png"
        ok, content = cv2.imencode(".png", np.full((24, 32, 3), 127, dtype=np.uint8))
        self.assertTrue(ok)
        image.write_bytes(content.tobytes())
        self.digest = hashlib.sha256(image.read_bytes()).hexdigest()
        register_image(
            image, library_root=self.library, source={"original_split": "train"}, date="2026-11-04"
        )
        self.record = self.library / "records" / f"{self.digest}.json"
        self.servers = {}
        self.tokens = {}
        for kind, server in (
            ("boxes", review_library.create_server(self.library)),
            ("keypoints", review_keypoints.create_server(self.library, root=self.root)),
        ):
            thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
            )
            thread.start()
            self.addCleanup(self.close_server, server, thread)
            self.servers[kind] = server
            status, _, page = self.request(kind, "GET", "/")
            self.assertEqual(status, 200)
            self.tokens[kind] = re.search(rb'"token"\s*:\s*"([^"]+)"', page)[1].decode()

    def close_server(self, server, thread):
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        self.assertFalse(thread.is_alive())

    def request(self, kind, method, route="/api/annotation", payload=None, headers=None):
        server = self.servers[kind]
        request_headers = {"Content-Type": "application/json"}
        if method == "POST":
            request_headers.update(Origin=server.review_url.rstrip("/"))
            request_headers["X-Review-Token"] = self.tokens[kind]
        request_headers.update(headers or {})
        body = json.dumps(payload).encode() if payload is not None else None
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(method, route, body=body, headers=request_headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_response_headers_images_and_host_guard(self):
        for kind in self.servers:
            with self.subTest(service=kind):
                status, headers, _ = self.request(kind, "GET", "/")
                self.assertEqual(status, 200)
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
                self.assertEqual(self.request(kind, "GET", "/", headers={"Host": "example.com"})[0], 403)
                status, _, content = self.request(kind, "GET", f"/image/{self.digest}")
                self.assertEqual(status, 200)
                image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(image.shape, (24, 32, 3))
                self.assertTrue(np.all(image == 127))

    def test_write_authorization_does_not_modify_records(self):
        original = self.record.read_bytes()
        for kind in self.servers:
            for headers in (
                {"Origin": "https://example.com"},
                {"X-Review-Token": "invalid"},
                {"Host": "example.com"},
            ):
                with self.subTest(service=kind, headers=headers):
                    self.assertEqual(self.request(kind, "POST", payload={}, headers=headers)[0], 403)
        self.assertEqual(self.record.read_bytes(), original)
        self.assertFalse((self.library / "keypoint_annotations.json").exists())

    def test_invalid_requests_keep_each_services_status_codes(self):
        for kind, expected in (("boxes", 413), ("keypoints", 409)):
            for headers in ({"Content-Length": "131073"}, {"Content-Type": "text/plain"}):
                with self.subTest(service=kind, headers=headers):
                    self.assertEqual(self.request(kind, "POST", payload={}, headers=headers)[0], expected)
            self.assertEqual(self.request(kind, "POST", payload=[])[0], 409)
            self.assertEqual(
                self.request(kind, "POST", payload={}, headers={"Content-Length": "invalid"})[0], 409
            )

    def test_box_draft_confirmation_and_stale_write(self):
        payload = {
            "image_sha256": self.digest,
            "record_version": hashlib.sha256(self.record.read_bytes()).hexdigest(),
            "label_text": "0 0.5 0.5 0.5 0.5\n",
        }
        self.assertEqual(self.request("boxes", "POST", "/api/draft", payload)[0], 200)
        self.assertEqual(json.loads(self.record.read_bytes())["status"], "pending")
        status, _, body = self.request("boxes", "POST", payload=payload)
        self.assertEqual(status, 200, body)
        current = self.record.read_bytes()
        record = json.loads(current)
        self.assertTrue(record["human_reviewed"])
        self.assertEqual(record["status"], "ready")
        self.assertEqual(self.request("boxes", "POST", payload=payload)[0], 409)
        self.assertEqual(self.request("boxes", "POST", "/api/draft", payload)[0], 409)
        self.assertEqual(self.record.read_bytes(), current)

    def test_keypoint_save_preserves_boxes_and_rejects_stale_or_changed_identity(self):
        store = review_keypoints.KeypointStore(self.library, root=self.root)
        row = store.snapshot()["images"][0]
        version = review_keypoints._version(row)
        row["notes"] = "manual draft"
        payload = {"row": row, "record_version": version}
        original_boxes = self.record.read_bytes()
        status, _, body = self.request("keypoints", "POST", payload=payload)
        self.assertEqual(status, 200, body)
        path = self.library / "keypoint_annotations.json"
        saved = path.read_bytes()
        self.assertEqual(json.loads(saved)["images"][0]["notes"], "manual draft")
        self.assertEqual(self.request("keypoints", "POST", payload=payload)[0], 409)
        changed = copy.deepcopy(payload)
        changed["row"]["width"] += 1
        self.assertEqual(self.request("keypoints", "POST", payload=changed)[0], 409)
        self.assertEqual(path.read_bytes(), saved)
        self.assertEqual(self.record.read_bytes(), original_boxes)


if __name__ == "__main__":
    unittest.main()
