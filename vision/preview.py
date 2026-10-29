"""D415 visual diagnostics: python -m vision.preview (F5 saves, F6 pauses, Esc exits)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2

from .collect import CaptureColorCamera, validate_config
from .contracts import validate_camera
from .diagnose import DiagnosticFrontend, solve_raw_points
from .geometry_reference import load_geometry_reference
from .pipeline import ROOT
from .preview_render import render_preview
from .preview_session import PreviewSession
from .depth_diagnostic import analyze_depth
from .depth_render import render_depth


WINDOW = "D415 Visual Diagnostics"
DEPTH_WINDOW = "D415 Depth Diagnostics"
SAVE_KEYS = (32, ord("s"), ord("S"), 0x740000, 0xffc2)  # Windows / X11 F5
PAUSE_KEYS = (ord("p"), ord("P"), 0x750000, 0xffc3)  # Windows / X11 F6


def assess_preview(frontend, pose, config):
    """Image quality only; SDK and geometry validation are reported separately."""
    reasons = []
    if frontend.get("reason") not in (None, "ok"):
        reasons.append(frontend["reason"])
    if not pose.get("solved"):
        if pose.get("reason") not in reasons:
            reasons.append(pose.get("reason", "pnp_unsolved"))
        return {"status": "no_pose", "reasons": reasons}
    if frontend.get("ambiguous"):
        reasons.append("ambiguous_correspondences")
    if not frontend.get("association_quality_ok") or frontend.get("selection_quality_ok") is False:
        reasons.append("association_rejected")
    if pose["inlier_count"] < config["min_inliers"]:
        reasons.append("insufficient_inliers")
    if pose["rms_px"] > config["max_reprojection_rms_px"]:
        reasons.append("rms_exceeded")
    if pose["max_px"] > config["max_point_error_px"]:
        reasons.append("point_error_exceeded")
    return {"status": "warning" if reasons else "estimated", "reasons": reasons}


def process_frame(frontend, image, camera, frame, geometry, config, *, depth_sample=None):
    started = time.monotonic()
    validate_camera(camera, image.shape)
    features = frontend.extract(image, camera, geometry)
    pose = solve_raw_points(features["named_points_raw_px"], geometry["points_mm"], camera)
    result = {
        "schema_version": 1, "kind": "d415_visual_diagnostic", "source": "live_diagnostic",
        "authorizes_robot_motion": False, "frame": dict(frame), "camera": dict(camera),
        "geometry": {key: geometry[key] for key in ("version", "status", "validated", "source_sha256")},
        "image_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "frontend": features, "pose": pose, "assessment": assess_preview(features, pose, config),
        "processing_ms": (time.monotonic() - started) * 1000,
    }
    if depth_sample is not None:
        result["schema_version"] = 2
        result["depth"] = analyze_depth(depth_sample, frame, camera, features, pose, geometry)
        result["processing_ms"] = (time.monotonic() - started) * 1000
    return result


def run_preview(camera, frontend, geometry, config, session, *, frames=0, save_first=False):
    """One inference worker; display and save only the matching completed image."""
    processed, paused, pending, displayed = 0, False, None, None
    use_depth = callable(getattr(camera, "read_rgbd", None))
    message = ""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d415-diagnostic")
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1440, 720)
        if use_depth:
            cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DEPTH_WINDOW, 1100, 720)
        while True:
            if use_depth:
                image, metadata, depth_sample = camera.read_rgbd()
            else:
                image, metadata = camera.read()
                depth_sample = None
            stale_color = "repeated_or_reordered_color_frame" in (depth_sample or {}).get("reasons", [])
            if stale_color:
                message = "彩色帧重复或倒序，等待新帧"
            elif message == "彩色帧重复或倒序，等待新帧":
                message = ""
            save_requested = False
            if pending is not None and pending.done() and not paused:
                displayed = pending.result()
                pending = None
                processed += 1
                save_requested = save_first and processed == 1
            if displayed is None:
                displayed = (image.copy(), {
                    "frontend": {}, "pose": {"solved": False, "reason": "processing"},
                    "assessment": {"status": "processing", "reasons": []},
                    "camera": camera.camera, "frame": dict(metadata), "geometry": geometry,
                }, None)
            shown_image, diagnostic, shown_depth = displayed
            overlay = render_preview(shown_image, diagnostic, saved_count=session.saved_count,
                                     paused=paused, message=message)
            cv2.imshow(WINDOW, overlay)
            if use_depth:
                cv2.imshow(DEPTH_WINDOW, render_depth(shown_depth, diagnostic, paused=paused,
                                                     saved_count=session.saved_count, message=message))
            key = cv2.waitKeyEx(1)
            if (key in (27, ord("q"), ord("Q"))
                    or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
                    or (use_depth and cv2.getWindowProperty(DEPTH_WINDOW, cv2.WND_PROP_VISIBLE) < 1)):
                break
            if key in PAUSE_KEYS:
                paused = not paused
            if key in SAVE_KEYS:
                save_requested = True
            if save_requested:
                if processed == 0:
                    message = "请等待首帧分析完成"
                else:
                    save_kwargs = {"depth_sample": shown_depth} if use_depth else {}
                    saved = session.save(shown_image, diagnostic, overlay, **save_kwargs)
                    message = f"已保存 {session.saved_count} 张" if saved is not None else "该帧已保存"
                    print(message, flush=True)
            if frames and processed >= frames:
                break
            if pending is None and not paused and not stale_color:
                snapshot, frame_info = image.copy(), dict(metadata)
                depth_snapshot = deepcopy(depth_sample)

                def analyze(current_image=snapshot, current_frame=frame_info, current_depth=depth_snapshot):
                    options = {"depth_sample": current_depth} if use_depth else {}
                    result = process_frame(frontend, current_image, camera.camera,
                                           current_frame, geometry, config, **options)
                    return current_image, result, current_depth

                pending = pool.submit(analyze)
    finally:
        # No worker accesses hardware. Close the window even if analysis failed.
        for window in (WINDOW, DEPTH_WINDOW) if use_depth else (WINDOW,):
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
        pool.shutdown(wait=True, cancel_futures=True)
    return {"processed_frames": processed, "saved_frames": session.saved_count,
            "last_assessment": displayed[1]["assessment"] if displayed else None,
            "last_depth_status": displayed[1].get("depth", {}).get("status") if displayed else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-config", type=Path, default=ROOT / "config/data_capture_d415.json")
    parser.add_argument("--vision-config", type=Path, default=ROOT / "config/realsense_config.json",
                        help="Detector and PnP settings only; no control output is written")
    parser.add_argument("--geometry", type=Path, default=ROOT / "config/geometry_nominal.json")
    parser.add_argument("--serial", help="Explicit D415 serial; otherwise use capture configuration")
    parser.add_argument("--batch", help="Capture batch; defaults to a unique diagnostic batch")
    parser.add_argument("--specimen")
    parser.add_argument("--printing-batch")
    parser.add_argument("--notes", default="D415 SDK visual diagnostics; unreviewed images")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/d415_preview")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N analyzed frames; 0 keeps window open")
    parser.add_argument("--save-first", action="store_true", help="Save first completed result, including failures")
    parser.add_argument("--depth", action="store_true", help="Add raw depth capture and diagnostic window")
    parser.add_argument("--depth-config", type=Path, default=ROOT / "config/d415_depth_diagnostics.json")
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    session, status, error, code = None, "completed", None, 0
    try:
        capture_config = json.loads(args.capture_config.read_text(encoding="utf-8-sig"))
        if args.serial:
            capture_config["serial"] = args.serial
        validate_config(capture_config)
        if capture_config["device_model"] != "D415":
            raise ValueError("D415 preview requires an explicit D415 capture configuration")
        output = args.output_root.resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError("Preview output must be within the project for annotation review")
        geometry = load_geometry_reference(args.geometry)
        if geometry["status"] == "measured_print" and (
                geometry["specimen_id"] != args.specimen or geometry["printing_batch_id"] != args.printing_batch):
            raise ValueError("Measured geometry requires matching --specimen and --printing-batch")
        config = json.loads(args.vision_config.read_text(encoding="utf-8-sig"))
        config["geometry_reference"] = str(args.geometry.resolve())
        frontend = DiagnosticFrontend(config)
        print("正在加载检测模型和几何…", flush=True)
        frontend._load()
        batch = args.batch or "d415_preview_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if args.depth:
            from .rgbd_capture import CaptureRGBDCamera
            depth_config = json.loads(args.depth_config.read_text(encoding="utf-8-sig"))
            capture = CaptureRGBDCamera(capture_config, depth_config)
        else:
            capture = CaptureColorCamera(capture_config)
        with capture as camera:
            session_options = {"depth_calibration": camera.depth_calibration} if args.depth else {}
            session = PreviewSession(output, camera.camera, batch, pipeline_config=config, geometry=geometry,
                                     metadata={"specimen_id": args.specimen, "printing_batch_id": args.printing_batch,
                                               "geometry_version": geometry["version"], "notes": args.notes},
                                     **session_options)
            print(f"诊断目录: {session.path}\nF5 保存；F6 暂停/继续；Esc 退出（也支持空格/S、P、Q）。", flush=True)
            result = run_preview(camera, frontend, geometry, config, session,
                                 frames=args.frames, save_first=args.save_first)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        status, code = "interrupted", 130
    except Exception as exc:
        status, error, code = "failed", f"{type(exc).__name__}: {exc}", 2
        print(error, flush=True)
    finally:
        if session is not None:
            try:
                session.finish(status, error=error)
                print(f"会话记录: {session.path / 'manifest.json'}", flush=True)
            except Exception as exc:
                code = 2
                print(f"会话结束记录写入失败: {type(exc).__name__}: {exc}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
