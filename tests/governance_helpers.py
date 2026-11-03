"""Deterministic inputs for offline governance regressions."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ev_charging.paths import PROJECT_ROOT
from ev_charging.vision.geometry_reference import load_geometry_reference
from ev_charging.vision.locate.diagnose import DiagnosticFrontend, solve_raw_points

FIXTURE = Path(__file__).parent / "fixtures/governance.json"


def camera(width=32, height=24):
    return {
        "stream": "color",
        "serial": "TEST-OFFLINE",
        "device_name": "Intel RealSense D415",
        "image_width": width,
        "image_height": height,
        "fps": 30,
        "pixel_format": "bgr8",
        "camera_matrix": [[1000, 0, width / 2], [0, 1000, height / 2], [0, 0, 1]],
        "dist_coeffs": [0.0] * 5,
        "distortion_model": "none",
        "validated": False,
        "version": None,
        "is_placeholder": False,
        "is_approximate": False,
    }


class FixedDetector:
    def __init__(self, boxes):
        self.boxes = [
            SimpleNamespace(
                xyxy=np.asarray([box["xyxy"]]),
                cls=np.asarray(box.get("class", 0)),
                conf=np.asarray(box["confidence"]),
            )
            for box in boxes
        ]

    def __call__(self, *args, **kwargs):
        return [SimpleNamespace(boxes=self.boxes)]


def vision_result(case, config):
    image = cv2.imdecode(
        np.frombuffer((PROJECT_ROOT / case["image"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    geometry = load_geometry_reference(PROJECT_ROOT / "config/geometry_nominal.json")
    frontend = DiagnosticFrontend(config, detector=FixedDetector(case["boxes"]))
    cv2.setRNGSeed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        features = frontend.extract(image, camera(image.shape[1], image.shape[0]), geometry)
        pose = solve_raw_points(
            features["named_points_raw_px"], geometry["points_mm"], camera(image.shape[1], image.shape[0])
        )
    return {"features": features, "pose": pose}


def assert_json_close(test, actual, expected):
    if isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected))
        for key in expected:
            assert_json_close(test, actual[key], expected[key])
    elif isinstance(expected, list):
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            assert_json_close(test, left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-7)
    else:
        test.assertEqual(actual, expected)


def baseline():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))
