"""Exercise the missing-import cases found by static checking."""

import time
import typing
import unittest

from ev_charging.vision.algorithms.latest_candidate_contours import filter_candidate_contours
from ev_charging.vision.contracts import empty_observation


class VisionContractTests(unittest.TestCase):
    def test_invalid_observation_still_has_monotonic_processing_time(self):
        before = time.monotonic()
        observation = empty_observation(None, "port", "camera_starting")
        self.assertFalse(observation["valid"])
        self.assertEqual(observation["reason"], "camera_starting")
        self.assertGreaterEqual(observation["processed_at"], before)
        self.assertLessEqual(observation["processed_at"], time.monotonic())

    def test_contour_return_annotation_can_be_resolved(self):
        self.assertIn("return", typing.get_type_hints(filter_candidate_contours))


if __name__ == "__main__":
    unittest.main()
