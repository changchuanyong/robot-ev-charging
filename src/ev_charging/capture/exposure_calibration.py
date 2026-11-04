"""Sweep RGB exposure and gain, then score how well the apertures survive.

The camera measures exposure over the whole frame, so a dark connector on a
bright table keeps only a few grey levels. This tool locks manual exposure,
captures a burst per setting, and ranks the settings by a label-free aperture
score before any images are collected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ev_charging.capture.collect import (
    RGBControls,
    list_devices,
    load_capture_config,
    validate_config,
)
from ev_charging.paths import PROJECT_ROOT

RADII = range(8, 44, 2)
CLIP_LOW_LEVEL = 2
CLIP_HIGH_LEVEL = 253


def _disc_masks(radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extent = int(round(radius * 1.5))
    size = 2 * extent + 1
    yy, xx = np.mgrid[0:size, 0:size]
    distance = np.hypot(xx - extent, yy - extent)
    inside = (distance <= radius).astype(np.float32)
    ring = ((distance > radius * 1.15) & (distance <= radius * 1.45)).astype(np.float32)
    box = (distance <= radius * 1.45).astype(np.float32)
    return inside / inside.sum(), ring / ring.sum(), box / box.sum()


def disc_response(gray: np.ndarray) -> np.ndarray:
    """Scale-normalised ring-versus-disc contrast; larger means a darker disc."""
    base = cv2.GaussianBlur(gray, (5, 5), 1.2).astype(np.float32)
    squared = base * base
    best = np.full(base.shape, -1e9, np.float32)
    for radius in RADII:
        if radius * 3 + 3 >= min(base.shape):
            continue
        inside, ring, box = _disc_masks(radius)
        mean_inside = cv2.filter2D(base, -1, inside)
        mean_ring = cv2.filter2D(base, -1, ring)
        local_square = cv2.filter2D(squared, -1, box)
        local_mean = cv2.filter2D(base, -1, box)
        deviation = np.sqrt(np.maximum(local_square - local_mean * local_mean, 1.0))
        np.maximum(best, (mean_ring - mean_inside) / (deviation + 1.0), out=best)
    return best


def top_peaks(response: np.ndarray, limit: int = 5, min_distance: float = 10.0) -> list[float]:
    """Strongest separated disc responses, so one texture edge cannot dominate."""
    dilated = cv2.dilate(response, np.ones((13, 13), np.uint8))
    ys, xs = np.where((response >= dilated - 1e-9) & (response > 0))
    if not len(ys):
        return []
    order = np.argsort(response[ys, xs])[::-1]
    peaks: list[tuple[float, float, float]] = []
    for index in order:
        x, y = float(xs[index]), float(ys[index])
        if all(math.hypot(x - px, y - py) > min_distance for px, py, _ in peaks):
            peaks.append((x, y, float(response[ys[index], xs[index]])))
        if len(peaks) >= limit:
            break
    return [value for _, _, value in peaks]


def region_of_interest(gray: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    height, width = gray.shape[:2]
    if box is None:
        return gray[int(0.2 * height) : int(0.8 * height), int(0.2 * width) : int(0.8 * width)]
    x1, y1, x2, y2 = box
    left, right = max(0, min(x1, x2)), min(width, max(x1, x2))
    top, bottom = max(0, min(y1, y2)), min(height, max(y1, y2))
    if right - left < 32 or bottom - top < 32:
        raise ValueError("exposure_roi_too_small")
    return gray[top:bottom, left:right]


def score_frame(gray: np.ndarray, box=None) -> dict:
    """Label-free score: strong disc response matters, clipping cancels it."""
    roi = region_of_interest(gray, box)
    values = roi.astype(np.float32)
    clip_low = float((roi <= CLIP_LOW_LEVEL).mean())
    clip_high = float((roi >= CLIP_HIGH_LEVEL).mean())
    median = float(np.median(values))
    p5, p95 = (float(value) for value in np.percentile(values, [5, 95]))
    peaks = top_peaks(disc_response(roi))
    aperture = 0.0
    if peaks:
        # Weak texture maxima must not drag the score to zero when only a few
        # apertures are in frame, so average only the comparable peaks.
        strongest = max(peaks)
        usable = [value for value in peaks if value >= 0.25 * strongest]
        aperture = float(np.mean(usable))
    penalty = max(0.0, 1.0 - 2.0 * clip_low - clip_high)
    return {
        "clip_low": round(clip_low, 4),
        "clip_high": round(clip_high, 4),
        "median": round(median, 1),
        "p5": round(p5, 1),
        "p95": round(p95, 1),
        "spread": round(p95 - p5, 1),
        "aperture_response": round(aperture, 3),
        "peak_count": len(peaks),
        "score": round(aperture * penalty, 3),
    }


def exposure_candidates(limits: dict, count: int = 6) -> list[float]:
    low = max(float(limits["min"]), 1.0)
    high = min(max(float(limits["max"]), low), 20000.0)
    if high <= low:
        return [low]
    values = np.geomspace(low, high, count)
    step = float(limits["step"])
    if step > 0:
        values = [limits["min"] + round((value - limits["min"]) / step) * step for value in values]
    return sorted({float(max(low, min(high, value))) for value in values})


def gain_candidates(limits: dict, count: int = 3) -> list[float]:
    low, high = float(limits["min"]), float(limits["max"])
    if high <= low:
        return [low]
    values = np.linspace(low, low + 0.5 * (high - low), count)
    step = float(limits["step"])
    if step > 0:
        values = [limits["min"] + round((value - limits["min"]) / step) * step for value in values]
    return sorted({float(max(low, min(high, value))) for value in values})


def recommend(rows: list[dict]) -> dict | None:
    usable = [row for row in rows if row.get("status") == "captured"]
    if not usable:
        return None
    return max(usable, key=lambda row: (row["metrics"]["score"], -row["metrics"]["clip_low"]))


def rgb_options(row: dict) -> dict:
    """Exact block to paste into ``profiles.<name>.rgb_options``."""
    return {
        "enable_auto_exposure": 0,
        "exposure": float(row["applied"]["exposure"]),
        "gain": float(row["applied"]["gain"]),
    }


def parse_box(text: str | None) -> tuple[int, int, int, int] | None:
    if text is None:
        return None
    parts = [int(value) for value in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("box_requires_four_integers")
    return tuple(parts)  # type: ignore[return-value]


def run(args) -> dict:
    import pyrealsense2 as rs

    from ev_charging.vision.launch import ensure_camera_privilege

    ensure_camera_privilege(sdk=rs)
    config, _ = load_capture_config(args.config, args.profile)
    config.setdefault("serial", args.serial)
    validate_config(config)
    serial = config["serial"]
    matches = [device for device in list_devices(rs) if device["serial"] == serial]
    if len(matches) != 1:
        raise ValueError("camera_not_found_for_serial")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(
        rs.stream.color,
        int(config["width"]),
        int(config["height"]),
        rs.format.bgr8,
        int(config["fps"]),
    )
    controls = None
    rows: list[dict] = []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    try:
        profile = pipeline.start(rs_config)
        controls = RGBControls(profile.get_device(), rs)
        ranges = controls.snapshot()["ranges"]
        if "exposure" not in ranges:
            raise ValueError("sensor_does_not_expose_manual_exposure")
        exposures = args.exposures or exposure_candidates(ranges["exposure"])
        gains = args.gains or gain_candidates(ranges.get("gain", ranges["exposure"]))
        for _ in range(int(config.get("warmup_frames", 30))):
            pipeline.wait_for_frames(int(config["timeout_ms"]))
        for exposure in exposures:
            for gain in gains:
                entry = {"requested": {"exposure": exposure, "gain": gain}}
                try:
                    applied = controls.apply({"enable_auto_exposure": 0, "exposure": exposure, "gain": gain})
                    entry["applied"] = {name: applied["values"].get(name) for name in ("exposure", "gain")}
                    for _ in range(3):
                        pipeline.wait_for_frames(int(config["timeout_ms"]))
                    frames = []
                    for _ in range(int(args.frames)):
                        packet = pipeline.wait_for_frames(int(config["timeout_ms"]))
                        frame = packet.get_color_frame()
                        frames.append(np.asanyarray(frame.get_data()).copy())
                    gray = np.median(np.stack(frames), axis=0).astype(np.uint8)
                    entry["metrics"] = score_frame(gray, args.box)
                    entry["status"] = "captured"
                    if args.save_frames:
                        name = f"exp{int(round(exposure))}_gain{int(round(gain))}.png"
                        cv2.imwrite(str(output / "frames" / name), gray)
                except (ValueError, RuntimeError) as exc:
                    entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                rows.append(entry)
                print(
                    f"exposure={exposure:8.1f} gain={gain:5.1f} "
                    f"{entry['status']:7s} {json.dumps(entry.get('metrics', {}), ensure_ascii=False)}",
                    flush=True,
                )
    finally:
        if controls is not None:
            try:
                controls.restore()
            except Exception as exc:  # noqa: BLE001 - restoration must not hide the run result
                print(f"restore failed: {exc}", file=sys.stderr)
        pipeline.stop()

    best = recommend(rows)
    report = {
        "kind": "rgb_exposure_calibration",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "serial": serial,
        "device_model": matches[0]["name"],
        "profile": args.profile,
        "frames_per_setting": int(args.frames),
        "roi_box": list(args.box) if args.box else None,
        "metric": "median of the five strongest ring-versus-disc responses, scaled by clipping",
        "settings": rows,
        "recommended": None
        if best is None
        else {"metrics": best["metrics"], "rgb_options": rgb_options(best)},
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sweep RGB exposure/gain and score aperture visibility")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/d415.json")
    parser.add_argument("--profile", default="collect", choices=("collect", "preview"))
    parser.add_argument("--serial", help="Explicit camera serial; otherwise taken from the config")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/exposure_calibration")
    parser.add_argument("--frames", type=int, default=8, help="Frames per setting, median-combined")
    parser.add_argument("--box", type=parse_box, help="ROI x1,y1,x2,y2; default is the centre of the frame")
    parser.add_argument("--exposures", type=float, nargs="*", help="Explicit exposure values")
    parser.add_argument("--gains", type=float, nargs="*", help="Explicit gain values")
    parser.add_argument("--save-frames", action="store_true", help="Write one PNG per setting")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)
    if args.list_devices:
        for device in list_devices():
            print(f"{device['serial']}  {device['name']}")
        return 0
    if args.frames < 1:
        parser.error("--frames must be positive")
    report = run(args)
    best = report["recommended"]
    if best is None:
        print("no usable setting captured", file=sys.stderr)
        return 1
    print("\n推荐 rgb_options:")
    print(json.dumps(best["rgb_options"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
