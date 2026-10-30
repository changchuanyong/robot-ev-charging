"""Candidate tracing preserves measurements and accounts for dropped observations."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live" / "python"))
import latest_candidate_contours as contours


class CandidateTraceTests(unittest.TestCase):
    def extract(self, edges, gray=None, diagnostics=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return contours.filter_candidate_contours(
                edges, np.zeros((*edges.shape, 3), dtype=np.uint8), apply_layout=False,
                diagnostics=diagnostics, gray_source=gray,
            )[0]

    def signature(self, candidates):
        return [(candidate.contour.tobytes(),
                 {key: value for key, value in vars(candidate).items() if key != "contour"})
                for candidate in candidates]

    def grid(self):
        edges = np.zeros((520, 600), dtype=np.uint8)
        for y in range(40, 461, 70):
            for x in range(40, 531, 70):
                cv2.circle(edges, (x, y), 12, 255, 2)
        edges[5, 5] = 255  # A degenerate contour still needs a traceable outcome.
        return edges

    def inner_aperture(self):
        edges = np.zeros((400, 400), dtype=np.uint8)
        gray = np.full_like(edges, 180)
        cv2.circle(edges, (200, 200), 40, 255, 2)
        cv2.ellipse(gray, (202, 200), (22, 15), 15, 0, 360, 40, -1)
        return edges, gray

    def test_trace_does_not_change_candidates_or_measurements(self):
        for edges, gray in ((self.grid(), None), self.inner_aperture()):
            with self.subTest(gray=gray is not None):
                plain = self.extract(edges, gray)
                diagnostics = {}
                traced = self.extract(edges, gray, diagnostics)
                self.assertEqual(self.signature(plain), self.signature(traced))
                json.dumps(diagnostics, allow_nan=False)

    def test_every_contour_is_accounted_for_including_duplicates_and_cap(self):
        diagnostics = {}
        kept = self.extract(self.grid(), diagnostics=diagnostics)
        trace = diagnostics["candidate_trace"]
        outcomes = trace["outcomes"]
        identities = [row["contour_index"] for row in outcomes]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(set(identities), set(range(diagnostics["contour_count"])))
        self.assertEqual(trace["coordinate_frame"], "roi_pixels")
        self.assertEqual(len(kept), 40)
        capped = [row for row in outcomes if row["stage"] == "capped"]
        retained = [row for row in outcomes if row["stage"] == "kept"]
        self.assertEqual(len(capped), 16)
        self.assertEqual([row["contour_index"] for row in retained],
                         [candidate.contour_index for candidate in kept])
        self.assertEqual([row["rank_before_cap"] for row in capped], list(range(41, 57)))
        self.assertTrue(all(row["reason"] == "max_keep" for row in capped))
        self.assertTrue(all(row["fit_valid"] and row["fitted_center"] for row in capped))
        representatives = {row["contour_index"] for row in retained + capped}
        merged = [row for row in outcomes if row["stage"] == "merged"]
        self.assertEqual(len(merged), diagnostics["rejection_counts"]["duplicate"])
        self.assertTrue(all(row["representative_contour_index"] in representatives for row in merged))
        self.assertEqual(sum(row["stage"] == "feature_rejected" for row in outcomes), 1)

    def test_merged_inner_aperture_is_linked_to_final_representative(self):
        edges, gray = self.inner_aperture()
        diagnostics = {}
        kept = self.extract(edges, gray, diagnostics)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].contour_source, "canny")
        outcomes = diagnostics["candidate_trace"]["outcomes"]
        transfers = [row for row in outcomes if row.get("aperture_transferred")]
        self.assertEqual(len(transfers), 1)
        transfer = transfers[0]
        final = next(row for row in outcomes if row["stage"] == "kept")
        self.assertEqual(transfer["stage"], "merged")
        self.assertEqual(transfer["source"], "adaptive_dark")
        self.assertEqual(transfer["representative_contour_index"], final["contour_index"])
        self.assertEqual(transfer["aperture_center"], final["aperture_center"])
        self.assertEqual(transfer["representative_aperture_center_after_merge"], final["aperture_center"])
        self.assertEqual(transfer["representative_aperture_diameter_after_merge"], final["aperture_diameter_px"])
        self.assertEqual(final["measurement_center"], list(kept[0].aperture_center))
        np.testing.assert_allclose(final["aperture_center"], [202, 200], atol=1.0)
        self.assertGreater(np.linalg.norm(np.subtract(final["aperture_center"], final["center"])), 1.0)

    def test_rejected_fits_and_boundaries_keep_positions_and_legacy_records(self):
        edges = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(edges, (5, 200), 30, 255, 2)
        gray = np.full_like(edges, 180)
        cv2.rectangle(gray, (150, 150), (250, 250), 30, -1)
        diagnostics = {}
        self.extract(edges, gray, diagnostics)
        outcomes = diagnostics["candidate_trace"]["outcomes"]
        filtered = [row for row in outcomes if row["stage"] == "filtered"]
        self.assertTrue(any(row["reason"] == "near_border" for row in filtered))
        invalid_dark = [row for row in filtered if row["reason"] == "dark_invalid_fit"]
        self.assertTrue(invalid_dark)
        self.assertTrue(all(row["fit_reason"] and not row["fit_valid"] for row in invalid_dark))
        self.assertTrue(all(len(row["bbox"]) == 4 and len(row["center"]) == 2 for row in filtered))
        for record in diagnostics["rejected_candidates"]:
            self.assertTrue({"reason", "center", "bbox", "area", "ellipse_area", "fit_reason", "source"}
                            <= set(record))
            self.assertTrue(any(row["reason"] == record["reason"] and row["center"] == record["center"]
                                and row["source"] == record["source"] for row in outcomes))
        json.dumps(diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
