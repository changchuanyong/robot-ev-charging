"""Synchronizing all source photos preserves corrections and explicit review gates."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch
import uuid

import cv2
import numpy as np

from vision.training_library import read_entries, set_annotation, validate_library


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_port_library", ROOT / "train/python/prepare_port_library.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareLibraryTests(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT / "artifacts/test-tmp" / ("prep-" + uuid.uuid4().hex[:12])
        self.scratch.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.repo = self.scratch / "repo"
        self.repo.mkdir()
        self.library = self.repo / "dataset/vision"

    def cleanup(self):
        if not self.scratch.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("Test cleanup outside intended directory")
        shutil.rmtree(self.scratch)

    def image(self, relative, value=42):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cv2.imencode(".png", np.full((32, 48, 3), value, np.uint8))[1].tobytes())
        return path

    def json(self, relative, data):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def historical(self, name="train.png", value=42, old_label="0 .5 .5 .2 .2\n"):
        image = self.image(f"yolo_port/images/train/{name}", value)
        label = self.repo / "yolo_port/labels/train" / Path(name).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(old_label, encoding="utf-8")
        return image, label

    def snapshot(self, sources):
        entries = []
        parent = self.repo / Path(prepare.LATEST_LABELS).parent
        for number, (original, text) in enumerate(sources):
            image = parent / f"dataset/images/train/{number}.png"
            label = parent / f"dataset/labels/train/{number}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(original.read_bytes())
            label.write_text(text, encoding="utf-8")
            entries.append({"image": image.relative_to(parent).as_posix(),
                            "label": label.relative_to(parent).as_posix(),
                            "source_image": original.relative_to(self.repo).as_posix(),
                            "image_sha256": sha(image), "label_sha256": sha(label),
                            "split": "train", "annotation_source": "assistant_visual_review"})
        return self.json(prepare.LATEST_LABELS, {"entries": entries})

    def capture(self, value=57):
        relative = "artifacts/d415_preview/session"
        image = self.image(relative + "/images/frame_000001.png", value)
        camera = self.json(relative + "/camera.json", {"serial": "synthetic-test", "stream": "color"})
        manifest = self.json(relative + "/manifest.json", {
            "kind": "realsense_rgb_capture", "session_id": "session", "split": "unknown",
            "camera_file": "camera.json", "camera_sha256": sha(camera),
            "frames": [{"image": "images/frame_000001.png", "image_sha256": sha(image),
                        "received_at_utc": "2026-10-30T09:00:00Z"}]})
        return image, camera, manifest

    def test_latest_corrected_labels_override_old_sources_and_resync_is_idempotent(self):
        original, _ = self.historical()
        self.snapshot([(original, "0 .5 .5 .4 .4\n")])
        first = prepare.sync_library(self.repo)
        entry = read_entries(self.library)[0]
        self.assertEqual(first["unique_images"], 1)
        self.assertEqual(entry["status"], "ready")
        self.assertEqual((self.library / entry["label"]).read_text(), "0 0.5 0.5 0.4 0.4\n")
        sources, candidates = entry["sources"], entry["label_candidates"]
        prepare.sync_library(self.repo)
        again = read_entries(self.library)[0]
        self.assertEqual(again["sources"], sources)
        self.assertEqual(again["label_candidates"], candidates)
        self.assertEqual(len(validate_library(self.library)), 1)

    def test_conflicting_snapshot_labels_need_review_and_resolution_survives_resync(self):
        first, _ = self.historical("one.png")
        second, _ = self.historical("two.png")
        self.snapshot([(first, "0 .5 .5 .4 .4"), (second, "0 .4 .4 .3 .3")])
        prepare.sync_library(self.repo)
        entry = read_entries(self.library)[0]
        self.assertEqual(entry["status"], "conflict")
        with self.assertRaisesRegex(ValueError, "annotation_conflict"):
            validate_library(self.library)
        set_annotation(self.library, entry["image_sha256"], "0 .5 .5 .4 .4", annotation_source="test_review")
        prepare.sync_library(self.repo)
        self.assertEqual(read_entries(self.library)[0]["status"], "ready")

    def test_snapshot_or_original_hash_change_stops_sync(self):
        original, _ = self.historical()
        manifest = self.snapshot([(original, "0 .5 .5 .4 .4")])
        data = json.loads(manifest.read_text())
        label = manifest.parent / data["entries"][0]["label"]
        good_label = label.read_bytes()
        label.write_text("0 .5 .5 .2 .2", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "historical_snapshot_hash_mismatch"):
            prepare.sync_library(self.repo)
        self.assertEqual(read_entries(self.library), [])
        label.write_bytes(good_label)
        original.write_bytes(original.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "original_image_missing_or_changed"):
            prepare.sync_library(self.repo)

    def test_manifest_photos_and_orphans_are_pending_but_overlays_are_not_imported(self):
        self.capture()
        self.image("artifacts/d415_preview/session/overlays/frame_000001.png", 83)
        self.image("artifacts/d415_preview/failed/images/frame_000002.png", 84)
        prepare.sync_library(self.repo)
        entries = read_entries(self.library)
        self.assertEqual(len(entries), 2)
        self.assertEqual({item["status"] for item in entries}, {"pending"})
        self.assertTrue(any(item["date"] == "2026-10-30" for item in entries))
        with self.assertRaisesRegex(ValueError, "annotation_pending"):
            validate_library(self.library)

    def test_capture_camera_and_image_hash_change_stops_sync(self):
        image, camera, _ = self.capture()
        good_camera = camera.read_bytes()
        camera.write_bytes(good_camera + b" ")
        with self.assertRaisesRegex(ValueError, "capture_camera_hash_mismatch"):
            prepare.sync_library(self.repo)
        camera.write_bytes(good_camera)
        image.write_bytes(image.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "capture_image_hash_mismatch"):
            prepare.sync_library(self.repo)

    def test_disabled_library_session_and_its_orphan_images_stay_out(self):
        _, _, manifest = self.capture()
        data = json.loads(manifest.read_text())
        data["metadata"] = {"training_library_enabled": False, "training_library_path": None}
        manifest.write_text(json.dumps(data), encoding="utf-8")
        self.image("artifacts/d415_preview/session/images/frame_000002.png", 63)
        report = prepare.sync_library(self.repo)
        self.assertEqual(report["unique_images"], 0)
        self.assertEqual(read_entries(self.library), [])

    def test_extra_source_records_missing_labels_and_rejects_library_containment(self):
        extra = self.scratch / "new_photos"
        extra.mkdir()
        (extra / "port.png").write_bytes(cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))[1].tobytes())
        prepare.sync_library(self.repo, extra_sources=[extra])
        self.assertEqual(read_entries(self.library)[0]["status"], "pending")
        for forbidden in (self.library, self.library / "images", self.repo):
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(ValueError, "outside_library"):
                prepare.sync_library(self.repo, extra_sources=[forbidden])

    def test_review_all_skips_user_deleted_excluded_copies(self):
        self.capture()
        prepare.sync_library(self.repo)
        entry = read_entries(self.library)[0]
        (self.library / entry["image"]).unlink()
        set_annotation(self.library, entry["image_sha256"], "", annotation_source="user_exclusion",
                       excluded=True, reason="User retained deletion")
        page = self.scratch / "review.html"
        report = prepare.write_review(self.library, page, review_all=True)
        self.assertEqual(report["images"], 0)
        self.assertEqual(report["excluded_missing_copies"], 1)
        self.assertFalse((self.library / entry["image"]).exists())

    def test_review_prelabels_do_not_change_library_labels_or_mark_pending_ready(self):
        self.capture()
        prepare.sync_library(self.repo)
        entry = read_entries(self.library)[0]
        record_path = self.library / "records" / (entry["image_sha256"] + ".json")
        original = record_path.read_bytes()
        fake = MagicMock()
        fake.YOLO.return_value.predict.return_value[0].boxes.xywhn.cpu.return_value.tolist.return_value = [
            [.5, .5, .4, .4]]
        weights = self.scratch / "weights.pt"
        weights.write_bytes(b"mocked_model")
        page = self.scratch / "review.html"
        with patch.dict(sys.modules, {"ultralytics": fake}):
            report = prepare.write_review(self.library, page, weights)
        self.assertFalse(report["suggestions_are_reviewed"])
        self.assertEqual(report["images"], 1)
        self.assertEqual(record_path.read_bytes(), original)
        self.assertEqual(list(self.library.rglob("*.txt")), [])
        self.assertIn("0 0.50000000 0.50000000 0.40000000 0.40000000", page.read_text(encoding="utf-8"))
        self.assertIn("file:///", page.read_text(encoding="utf-8"))

    def test_review_cli_uses_current_model_or_explicit_manual_choice(self):
        weights = self.scratch / "selected.pt"
        for options, expected in (([], weights), (["--no-suggestions"], None),
                                  (["--weights", str(weights)], weights)):
            with self.subTest(options=options):
                with (patch.object(prepare, "sync_library", return_value={}),
                      patch.object(prepare, "current_model_path", return_value=weights) as selected,
                      patch.object(prepare, "write_review", return_value={}) as review):
                    prepare.main(["--library", str(self.library), "--review", str(self.scratch / "review.html"),
                                  *options])
                self.assertEqual(review.call_args.args[2], expected)
                if not options:
                    selected.assert_called_once_with(verify=True)
                else:
                    selected.assert_not_called()

    def test_sync_without_review_does_not_require_model(self):
        with (patch.object(prepare, "sync_library", return_value={}),
              patch.object(prepare, "current_model_path") as selected):
            prepare.main(["--library", str(self.library)])
        selected.assert_not_called()

    @unittest.skipUnless(shutil.which("node"), "Node unavailable for review interaction check")
    def test_review_export_requires_explicit_decision_and_edits_invalidate_acceptance(self):
        self.capture()
        prepare.sync_library(self.repo)
        page = self.scratch / "review.html"
        prepare.write_review(self.library, page)
        script = page.read_text(encoding="utf-8").split("<script>", 1)[1].split("</script>", 1)[0]
        setup = r"""
const assert=require('node:assert/strict'), elements=new Map();
const context=new Proxy({}, {get:()=>()=>{}});
function element(){return {value:'',textContent:'',width:48,height:32,click(){},
getContext(){return context},setPointerCapture(){},getBoundingClientRect(){return {left:0,top:0,width:48,height:32}}};}
const document={getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},createElement:element};
const Image=class{constructor(){this.complete=true;this.naturalWidth=48;}set src(value){this.onload();}};
let exported;
const URL={createObjectURL(blob){exported=blob;return 'blob:test'},revokeObjectURL(){}};
const setTimeout=()=>{};
"""
        checks = r"""
(async()=>{
assert.equal(records.length,1);
elements.get('accept').onclick();assert.equal(Object.keys(decisions).length,0);
elements.get('labels').value='0 .5 .5 .4 .4';elements.get('labels').oninput();
elements.get('accept').onclick();assert.equal(Object.keys(decisions).length,1);
elements.get('labels').value='0 .4 .4 .3 .3';elements.get('labels').oninput();
assert.equal(Object.keys(decisions).length,0);
elements.get('exclude').onclick();elements.get('export').onclick();
const payload=JSON.parse(await exported.text());assert.equal(payload.annotations.length,1);
assert.equal(payload.annotations[0].excluded,true);assert.equal(payload.annotations[0].label_text,'');
assert.equal(payload.annotations[0].image_sha256,records[0].image_sha256);
console.log('review interaction passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-"], input=setup + script + checks,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
