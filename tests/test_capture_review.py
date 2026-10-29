"""Capture review binds original RGB frames without inventing labels or old PnP inputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live/python"))
sys.path.insert(0, str(ROOT))

from keypoint_annotations import LABELS, sha256_file, validate_annotations
from prepare_capture_review import build_capture_annotations, prepare
from prepare_keypoint_review import TEMPLATE_PATH
from vision.contracts import intrinsics_signature


class CaptureReviewTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("capture-review-" + uuid.uuid4().hex)
        self.session = self.root / "artifacts/captures/session-one"
        (self.session / "images").mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.camera = {
            "serial": "D415-SYNTHETIC", "device_name": "Intel RealSense D415", "stream": "color",
            "image_width": 160, "image_height": 120,
            "camera_matrix": [[140., 0., 80.], [0., 140., 60.], [0., 0., 1.]],
            "dist_coeffs": [0.] * 5, "distortion_model": "brown_conrady",
            "validated": False, "version": None, "source": "device_color_stream",
            "is_placeholder": False, "is_approximate": False,
        }
        self.camera["intrinsics_sha256"] = intrinsics_signature(self.camera)
        self.camera_path = self.session / "camera.json"
        self.camera_path.write_text(json.dumps(self.camera), encoding="utf-8")
        self.image_path = self.session / "images/frame_000001.png"
        image = np.zeros((120, 160, 3), np.uint8)
        image[35:55, 45:85] = [25, 80, 190]
        self.image_path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
        self.manifest = {
            "schema_version": 1, "kind": "realsense_rgb_capture", "session_id": "session-one",
            "created_at": "2026-10-28T01:20:00Z", "batch_id": "batch-a", "split": "train",
            "notes": "synthetic RGB fixture", "status": "completed", "camera_file": "camera.json",
            "camera_sha256": sha256_file(self.camera_path),
            "frames": [{"image": "images/frame_000001.png", "image_sha256": sha256_file(self.image_path),
                        "width": 160, "height": 120, "frame_number": 11, "device_timestamp_ms": 1234.,
                        "timestamp_domain": "hardware_clock", "received_at_utc": "2026-10-28T01:20:01Z"}],
        }
        self.manifest_path = self.session / "manifest.json"
        self.write_manifest()

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("Test cleanup outside intended directory")
        shutil.rmtree(self.root)

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def build(self):
        return build_capture_annotations(self.manifest_path, self.root)

    def test_preparation_preserves_originals_and_creates_only_unlabeled_points(self):
        inputs = (self.image_path, self.camera_path, self.manifest_path)
        before = {path: path.read_bytes() for path in inputs}
        path, page = prepare(self.manifest_path, root=self.root)
        self.assertEqual(path, self.session / "review/annotations.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 2)
        self.assertFalse(data["independent_test_set"])
        self.assertEqual(data["provenance"]["kind"], "capture_session")
        self.assertNotIn("batch_report_path", data["provenance"])
        self.assertNotIn("model_points_path", data["provenance"])
        self.assertNotIn("pnp_source_sha256", data["provenance"])
        row = data["images"][0]
        self.assertEqual((row["camera_serial"], row["batch_id"], row["session_id"]),
                         ("D415-SYNTHETIC", "batch-a", "session-one"))
        self.assertEqual(row["auto_points"], {})
        self.assertEqual(row["roi_bbox_xyxy"], [])
        self.assertEqual(row["review_status"], "draft")
        self.assertTrue(all(point == {"xy": None, "visibility": "unlabeled", "source": "unlabeled", "note": ""}
                            for point in row["points"].values()))
        payload = json.loads(page.read_text(encoding="utf-8").split(
            '<script id="payload" type="application/json">', 1)[1].split("</script>", 1)[0])
        self.assertEqual(payload["annotations"], data)
        self.assertTrue(payload["images"][0]["src"].startswith("data:image/png;base64,"))
        self.assertEqual(before, {path: path.read_bytes() for path in inputs})

    def test_recording_empty_and_invalid_manifests_never_create_output(self):
        for changed in ({"status": "recording"}, {"status": "unknown"}, {"frames": []},
                        {"camera_file": "../camera.json"}, {"split": "arbitrary"}):
            with self.subTest(changed=changed):
                original = copy.deepcopy(self.manifest)
                self.manifest.update(changed)
                self.write_manifest()
                with self.assertRaises(ValueError):
                    prepare(self.manifest_path, root=self.root)
                self.assertFalse((self.session / "review").exists())
                self.manifest = original
        for status in ("failed", "interrupted"):
            self.manifest["status"] = status
            self.write_manifest()
            self.assertEqual(len(self.build()["images"]), 1)

    def test_frame_and_camera_hash_changes_are_rejected_before_writes(self):
        for path, error in ((self.image_path, "image hash"), (self.camera_path, "camera hash")):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, error):
                    prepare(self.manifest_path, root=self.root)
                self.assertFalse((self.session / "review").exists())
                path.write_bytes(original)

    def test_frame_size_camera_signature_and_placeholder_checks(self):
        self.manifest["frames"][0]["width"] += 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            self.build()
        self.manifest["frames"][0]["width"] -= 1
        for field, value, error in (("is_placeholder", True, "placeholder"),
                                    ("intrinsics_sha256", "0" * 64, "signature changed"),
                                    ("image_width", 161, "camera_resolution_mismatch")):
            with self.subTest(field=field):
                altered = copy.deepcopy(self.camera)
                altered[field] = value
                if field == "image_width":
                    altered["intrinsics_sha256"] = intrinsics_signature(altered)
                self.camera_path.write_text(json.dumps(altered), encoding="utf-8")
                self.manifest["camera_sha256"] = sha256_file(self.camera_path)
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, error):
                    self.build()

    def test_path_escape_and_duplicate_frame_content_are_rejected(self):
        self.manifest["frames"][0]["image"] = "../outside.png"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            self.build()
        self.manifest["frames"][0]["image"] = "images/frame_000001.png"
        duplicate = copy.deepcopy(self.manifest["frames"][0])
        duplicate["image"] = "images/frame_000002.png"
        (self.session / duplicate["image"]).write_bytes(self.image_path.read_bytes())
        self.manifest["frames"].append(duplicate)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            self.build()
        with self.assertRaisesRegex(ValueError, "inside repository"):
            build_capture_annotations(self.manifest_path, self.session / "images")

    def test_regeneration_keeps_reviewed_json_byte_for_byte(self):
        path, page = prepare(self.manifest_path, root=self.root)
        data = json.loads(path.read_text(encoding="utf-8"))
        row = data["images"][0]
        for point in row["points"].values():
            point.update(visibility="uncertain", source="human")
        row["points"]["PE"].update(visibility="visible", xy=[80., 70.])
        row.update(review_status="reviewed", reviewer="reviewer", reviewed_at="2026-10-28T02:00:00Z")
        path.write_text(json.dumps(data, indent=3), encoding="utf-8")
        before = path.read_bytes()
        prepare(self.manifest_path, root=self.root)
        self.assertEqual(path.read_bytes(), before)
        self.assertIn('"review_status": "reviewed"', page.read_text(encoding="utf-8"))
        imported = self.root / "reviewed_annotations.json"
        imported.write_bytes(before)
        prepare(self.manifest_path, root=self.root, annotations_path=imported)
        self.assertEqual(imported.read_bytes(), before)

    def test_changed_manifest_or_bound_annotation_cannot_replace_review(self):
        path, page = prepare(self.manifest_path, root=self.root)
        before = (path.read_bytes(), page.read_bytes())
        self.manifest["notes"] = "changed capture record"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "header/provenance differs"):
            prepare(self.manifest_path, root=self.root)
        self.assertEqual((path.read_bytes(), page.read_bytes()), before)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["images"][0]["camera_serial"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "camera_serial differs"):
            validate_annotations(data)

    def test_new_schema_does_not_bypass_old_batch_provenance(self):
        data = self.build()
        data["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "nonempty string"):
            validate_annotations(data)
        data["schema_version"] = 2
        data["provenance"]["kind"] = "legacy_batch"
        with self.assertRaisesRegex(ValueError, "capture_session"):
            validate_annotations(data)

    def test_capture_session_requires_one_split_but_legacy_batches_may_mix(self):
        data = self.build()
        another = copy.deepcopy(data["images"][0])
        another.update(image="images/frame_000002.png", image_sha256="0" * 64)
        data["images"].append(another)
        validate_annotations(data)
        another["split"] = "val"
        with self.assertRaisesRegex(ValueError, "single split"):
            validate_annotations(data)
        data["schema_version"] = 1
        data["provenance"] = {
            "batch_report_path": "batch.json", "batch_report_sha256": "1" * 64,
            "camera_config_path": "camera.json", "camera_config_sha256": "2" * 64,
            "model_points_path": "model.csv", "model_points_sha256": "3" * 64,
            "pnp_source_sha256": "4" * 64,
        }
        validate_annotations(data)

    @unittest.skipUnless(shutil.which("node"), "Node unavailable for browser import-validator check")
    def test_browser_validator_keeps_capture_bindings_and_accepts_human_edits(self):
        html = TEMPLATE_PATH.read_text(encoding="utf-8")
        validator = html.split("function validate(candidate){", 1)[1].split("function message(", 1)[0]
        declarations = "\n".join(line for line in html.splitlines()
                                 if line.startswith(("const canonical=", "const immutable=", "const editable=")))
        script = "const seed=" + json.dumps(self.build()) + ";\n"
        script += "const labels=seed.labels,visibilityNames={unlabeled:1,visible:1,uncertain:1,occluded:1,out_of_frame:1};\n"
        script += declarations + "\nfunction validate(candidate){" + validator
        script += """
validate(seed);
const edited=structuredClone(seed);
edited.images[0].points.PE={xy:[80,70],visibility:'visible',source:'human',note:''};
validate(edited);
for(const mutation of [d=>d.schema_version=1,d=>d.provenance.manifest_sha256='0'.repeat(64),
    d=>d.images[0].camera_serial='OTHER',d=>d.images[0].image_sha256='0'.repeat(64),
    d=>d.images[0].split='val',d=>d.images[0].review_status='reviewed']){
    const bad=structuredClone(seed);mutation(bad);let rejected=false;
    try{validate(bad)}catch{rejected=true}if(!rejected)throw Error('Changed binding accepted');
}
"""
        result = subprocess.run([shutil.which("node"), "-"], input=script, encoding="utf-8",
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
