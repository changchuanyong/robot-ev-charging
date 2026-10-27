"""Evaluation gates and paired comparisons must not disguise loss of coverage."""
import copy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live/python"))

from batch_pipeline_eval import classify_result, pose_rejection_reasons, summarize
from compare_pipeline_eval import compare


class BatchEvaluationTests(unittest.TestCase):
    def test_solver_return_does_not_accept_high_residual(self):
        row = {"roi_ok": True, "pnp_ok": True, "pnp_all_in_front": True,
               "edge_density": .05, "candidate_count": 4, "pnp_inlier_count": 0,
               "pnp_reproj_error_px": 135.814719}
        self.assertIn("pnp_point_matching_unstable", classify_result(row))
        self.assertEqual(pose_rejection_reasons(row),
                         ["all_point_mean_exceeds_20px", "fewer_than_four_10px_inliers"])

    def test_ambiguous_low_residual_is_rejected(self):
        row = {"pnp_ok": True, "pnp_all_in_front": True, "pnp_inlier_count": 9,
               "pnp_reproj_error_px": 1., "correspondence_diagnostics": {"ambiguous": True}}
        self.assertEqual(pose_rejection_reasons(row), ["ambiguous_correspondences"])

    def test_rejected_pose_residuals_remain_in_summary(self):
        rows = [{"status": "ok", "pnp_ok": True, "pnp_reproj_error_px": 1.},
                {"status": "fail", "pnp_ok": True, "pnp_reproj_error_px": 101.}]
        summary = summarize(rows)
        self.assertEqual(summary["mean_reproj_error_px"], 51.)
        self.assertEqual(summary["solver_success_count"], 2)

    def test_low_error_pose_still_requires_supported_correspondences(self):
        row = {"pnp_ok": True, "pnp_all_in_front": True, "pnp_inlier_count": 4,
               "pnp_reproj_error_px": 1., "correspondence_diagnostics": {"selection_quality_ok": False}}
        self.assertEqual(pose_rejection_reasons(row), ["correspondence_quality_rejected"])

    def test_comparison_checks_inputs_and_preserves_failed_images(self):
        manifest = {"runs/detect_retrain/weights/best.pt": "weights", "config/camera_intrinsics.json": "camera",
                    "config/charging_port_model.csv": "model", "images/train/a.jpg": "a", "images/val/b.jpg": "b",
                    "artifacts/batch_eval_control/latest_details.json": "baseline"}
        before = {"rows": [{"image": "images\\train\\a.jpg", "status": "warn", "pnp_ok": True,
                            "pnp_reproj_error_px": 20., "candidate_count": 9},
                           {"image": "images/val/b.jpg", "status": "fail", "pnp_ok": False}]}
        after = {"provenance": {"weights_sha256": "weights", "camera_config_sha256": "camera", "model_points_sha256": "model"},
                 "rows": [{"image": "images/train/a.jpg", "image_sha256": "a", "status": "ok", "pnp_ok": True,
                           "pnp_reproj_error_px": 1., "candidate_count": 4},
                          {"image": "images/val/b.jpg", "image_sha256": "b", "status": "fail", "pnp_ok": False}]}
        report = compare(before, after, manifest, baseline_sha256="baseline")
        self.assertEqual(report["groups"]["all"]["total"], 2)
        self.assertEqual(report["groups"]["all"]["paired_without_reduced_point_count_after"]["count"], 0)
        self.assertEqual(report["groups"]["all"]["reduced_point_count_images"], ["images/train/a.jpg"])
        changed = copy.deepcopy(after)
        changed["rows"][0]["image_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Image changed"):
            compare(before, changed, manifest, baseline_sha256="baseline")
        with self.assertRaisesRegex(ValueError, "Baseline report changed"):
            compare(before, after, manifest, baseline_sha256="edited-report")


if __name__ == "__main__":
    unittest.main()
