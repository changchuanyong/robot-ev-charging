"""D415 visual diagnostics: python -m vision.preview (F5 saves, F6 pauses, Esc exits)."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import cv2

from .collect import (CaptureColorCamera, CaptureRGBDCamera, PreviewCapture, intake_status,
                      load_capture_config, print_intake_summary, validate_config)
from .contracts import validate_camera
from .diagnose import DiagnosticFrontend, solve_raw_points
from .geometry_reference import load_geometry_reference
from .pipeline import ROOT
from .preview_render import render_depth, render_preview
from .capture_session import PreviewSession
from .depth_diagnostic import analyze_depth


WINDOW = "D415 Visual Diagnostics"
DEPTH_WINDOW = "D415 Depth Diagnostics"
SAVE_KEYS = (32, ord("s"), ord("S"), 0x740000, 0xffc2)  # Windows / X11 F5
PAUSE_KEYS = (ord("p"), ord("P"), 0x750000, 0xffc3)  # Windows / X11 F6


class FrameRate:
    """Measure completed events; configured camera fps is never an observation."""
    def __init__(self):
        self.times = deque(maxlen=120)
        self.count = 0
        self.first = None

    def tick(self):
        now = time.monotonic()
        self.times.append(now)
        self.first = now if self.first is None else self.first
        self.count += 1

    def fps(self):
        now = time.monotonic()
        recent = [stamp for stamp in self.times if now - stamp <= 2.]
        elapsed = now - recent[0] if len(recent) > 1 else 0
        return (len(recent) - 1) / elapsed if elapsed > 0 else 0.

    def mean_fps(self):
        elapsed = self.times[-1] - self.first if self.count > 1 else 0
        return (self.count - 1) / elapsed if elapsed > 0 else 0.


def configure_capture(config, *, quality=None, auto_exposure=False, exposure=None, gain=None):
    """Apply explicit preview overrides without changing shared collection JSON."""
    config = deepcopy(config)
    if quality is not None:
        if quality not in {"full-hd", "hd"}:
            raise ValueError("Unsupported preview quality")
        width, height = (1920, 1080) if quality == "full-hd" else (1280, 720)
        config.update(width=width, height=height, fps=30)
    if auto_exposure or exposure is not None or gain is not None:
        if auto_exposure and (exposure is not None or gain is not None):
            raise ValueError("Manual exposure/gain cannot be combined with automatic exposure")
        rgb = config.setdefault("rgb_options", {})
        rgb["enable_auto_exposure"] = 1 if auto_exposure else 0
        if auto_exposure:
            rgb.pop("exposure", None)
            rgb.pop("gain", None)
        if exposure is not None:
            rgb["exposure"] = exposure
        if gain is not None:
            rgb["gain"] = gain
    validate_config(config)
    return config


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


def run_preview(camera, frontend, geometry, config, session, *, frames=0, save_first=False, zoom=False,
                library_root=None):
    """One inference worker; display and save only the matching completed image."""
    from .intake import CaptureIntake

    processed, paused, pending, displayed = 0, False, None, None
    use_depth = callable(getattr(camera, "read_rgbd", None))
    message = ""
    read_rate, analysis_rate, render_rate = FrameRate(), FrameRate(), FrameRate()
    overlay = depth_overlay = None
    rendered_key = depth_key = None
    last_controls = ""
    intake = CaptureIntake(session, library_root, preview=True)
    intake_report = intake.poll()
    last_intake_message = ""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d415-diagnostic")
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1728, 864)
        if use_depth:
            cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DEPTH_WINDOW, 1100, 720)
        while True:
            intake_report = intake.poll()
            if intake_report.get("last_message") and intake_report["last_message"] != last_intake_message:
                message = last_intake_message = intake_report["last_message"]
                print(message, flush=True)
            if use_depth:
                image, metadata, depth_sample = camera.read_rgbd()
            else:
                image, metadata = camera.read()
                depth_sample = None
            read_rate.tick()
            controls = getattr(camera, "control_message", "")
            if controls and controls != last_controls:
                message = last_controls = controls
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
                analysis_rate.tick()
                save_requested = save_first and processed == 1
            if displayed is None:
                displayed = (image.copy(), {
                    "frontend": {}, "pose": {"solved": False, "reason": "processing"},
                    "assessment": {"status": "processing", "reasons": []},
                    "camera": camera.camera, "frame": dict(metadata), "geometry": geometry,
                }, None)
            shown_image, diagnostic, shown_depth = displayed
            capture_stats = (camera.capture_stats() if callable(getattr(camera, "capture_stats", None))
                             else {"capture_fps": read_rate.fps()})
            frame_age = None
            try:
                received = datetime.fromisoformat(diagnostic["frame"]["received_at_utc"])
                frame_age = max(0., (datetime.now(timezone.utc) - received).total_seconds() * 1000)
            except (KeyError, ValueError, TypeError):
                pass
            runtime = {**capture_stats, "analysis_fps": 0. if paused else analysis_rate.fps(),
                       "render_fps": 0. if paused else render_rate.fps(), "frame_age_ms": frame_age,
                       "target_zoom": zoom, "display_image_width": 1280, "intake": intake_report}
            status_line = intake_status(intake_report)
            display_message = status_line + ("；" + message if message else "")
            # Cached panels still process keyboard/window events on every iteration.
            key_now = (processed, paused, display_message, intake_report["saved"], zoom, int(time.monotonic() * 2))
            if key_now != rendered_key:
                overlay = render_preview(shown_image, diagnostic, saved_count=intake_report["saved"],
                                         paused=paused, message=display_message, runtime=runtime, zoom=zoom,
                                         display_width=1280)
                if rendered_key is None or rendered_key[0] != processed:
                    render_rate.tick()
                rendered_key = key_now
            cv2.imshow(WINDOW, overlay)
            if use_depth:
                depth_now = (processed, paused, display_message, intake_report["saved"])
                if depth_now != depth_key:
                    depth_overlay = render_depth(shown_depth, diagnostic, paused=paused,
                                                 saved_count=intake_report["saved"], message=display_message)
                    depth_key = depth_now
                cv2.imshow(DEPTH_WINDOW, depth_overlay)
            key = cv2.waitKeyEx(1)
            if (key in (27, ord("q"), ord("Q"))
                    or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
                    or (use_depth and cv2.getWindowProperty(DEPTH_WINDOW, cv2.WND_PROP_VISIBLE) < 1)):
                break
            if key in PAUSE_KEYS:
                paused = not paused
            if key in (ord("z"), ord("Z")):
                zoom = not zoom
            rgb_action = {ord("a"): ("toggle_auto", None), ord("A"): ("toggle_auto", None),
                          ord("["): ("adjust_exposure", -1), ord("]"): ("adjust_exposure", 1),
                          ord("-"): ("adjust_gain", -1), ord("="): ("adjust_gain", 1),
                          ord("+"): ("adjust_gain", 1)}.get(key)
            if rgb_action:
                if paused:
                    message = "请先按 F6 继续画面，再调曝光查看效果"
                elif callable(getattr(camera, "change_rgb", None)):
                    message = camera.change_rgb(*rgb_action)
                else:
                    message = "当前相机未启用实时曝光控制"
            if key in SAVE_KEYS:
                save_requested = True
            if save_requested:
                if processed == 0:
                    message = "请等待首帧分析完成"
                else:
                    accepted = intake.submit(shown_image, {**diagnostic, "preview_runtime": runtime}, overlay,
                                             depth_sample=shown_depth if use_depth else None)
                    message = "已加入保存队列" if accepted else "队列繁忙或该帧已排队，请稍后重试"
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
        try:
            intake_report = intake.close()
            print_intake_summary(intake_report, session.path, library_root)
        finally:
            # No worker accesses hardware. Close the window even if analysis failed.
            for window in (WINDOW, DEPTH_WINDOW) if use_depth else (WINDOW,):
                try:
                    cv2.destroyWindow(window)
                except cv2.error:
                    pass
            pool.shutdown(wait=True, cancel_futures=True)
    capture_stats = (camera.capture_stats() if callable(getattr(camera, "capture_stats", None))
                     else {"capture_mean_fps": read_rate.mean_fps(), "captured_frames": read_rate.count})
    return {"processed_frames": processed, "saved_frames": intake_report["saved"],
            **capture_stats, "analysis_mean_fps": analysis_rate.mean_fps(),
            "result_render_mean_fps": render_rate.mean_fps(), "consumed_mean_fps": read_rate.mean_fps(),
            **intake_report,
            "last_assessment": displayed[1]["assessment"] if displayed else None,
            "last_depth_status": displayed[1].get("depth", {}).get("status") if displayed else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--capture-config", dest="capture_config", type=Path,
                        default=ROOT / "config/d415.json", help="D415 capture, depth and vision configuration")
    parser.add_argument("--quality", choices=("full-hd", "hd"),
                        help="Override RGB to 1920x1080/30 or 1280x720/30; otherwise use capture config")
    exposure_group = parser.add_mutually_exclusive_group()
    exposure_group.add_argument("--exposure", type=float, help="Manual RGB exposure in SDK option units")
    exposure_group.add_argument("--auto-exposure", action="store_true", help="Enable RGB auto exposure")
    parser.add_argument("--gain", type=float, help="Manual RGB gain in SDK option units")
    parser.add_argument("--zoom", action="store_true", help="Enlarge detected target for display; keep full raw images")
    parser.add_argument("--vision-config", type=Path,
                        help="Detector and PnP settings only; no control output is written")
    parser.add_argument("--geometry", type=Path, default=ROOT / "config/geometry_nominal.json")
    parser.add_argument("--serial", help="Explicit D415 serial; otherwise use capture configuration")
    parser.add_argument("--batch", help="Capture batch; defaults to a unique diagnostic batch")
    parser.add_argument("--split", choices=("train", "val", "unknown"), default="train",
                        help="整批图库用途；默认 train，val 用于开发验证，unknown 留待分组")
    parser.add_argument("--specimen")
    parser.add_argument("--printing-batch")
    parser.add_argument("--notes", default="D415 SDK visual diagnostics; unreviewed images")
    parser.add_argument("--output-root", type=Path, help="Override the preview profile's output_root")
    library_group = parser.add_mutually_exclusive_group()
    library_group.add_argument("--library", type=Path, default=ROOT / "dataset/vision",
                               help="统一 RGB 图库，默认 dataset/vision；新图待标注")
    library_group.add_argument("--no-library", action="store_true", help="仅保存诊断会话，不加入训练图库")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N analyzed frames; 0 keeps window open")
    parser.add_argument("--save-first", action="store_true", help="Save first completed result, including failures")
    parser.add_argument("--depth", action="store_true", help="Add raw depth capture and diagnostic window")
    parser.add_argument("--depth-config", type=Path, help="Override the depth section with a separate configuration")
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    if args.auto_exposure and args.gain is not None:
        parser.error("--gain selects manual exposure; do not combine with --auto-exposure")
    if any(value is not None and not math.isfinite(value) for value in (args.exposure, args.gain)):
        parser.error("exposure and gain must be finite")
    session, status, error, code = None, "completed", None, 0
    try:
        capture_config, options = load_capture_config(args.capture_config, "preview")
        if not options:
            _, options = load_capture_config(ROOT / "config/d415.json", "preview")
        if args.serial:
            capture_config["serial"] = args.serial
        capture_config = configure_capture(capture_config, quality=args.quality,
                                           auto_exposure=args.auto_exposure,
                                           exposure=args.exposure, gain=args.gain)
        if capture_config["device_model"] != "D415":
            raise ValueError("D415 preview requires an explicit D415 capture configuration")
        output = args.output_root or Path(capture_config.get("output_root", "artifacts/d415_preview"))
        output = (output if output.is_absolute() else ROOT / output).resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError("Preview output must be within the project for annotation review")
        library_root = None if args.no_library else (
            args.library if args.library.is_absolute() else ROOT / args.library).resolve()
        batch = args.batch or "d415_preview_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        metadata = {"specimen_id": args.specimen, "printing_batch_id": args.printing_batch,
                    "notes": args.notes, "training_library_enabled": library_root is not None,
                    "training_library_path": str(library_root) if library_root is not None else None,
                    "capture_config": capture_config}
        if library_root is not None:
            from train.python.dataset_split import check_capture_split
            check_capture_split(library_root, batch, args.split, metadata)
        geometry = load_geometry_reference(args.geometry)
        if geometry["status"] == "measured_print" and (
                geometry["specimen_id"] != args.specimen or geometry["printing_batch_id"] != args.printing_batch):
            raise ValueError("Measured geometry requires matching --specimen and --printing-batch")
        config = (json.loads(args.vision_config.read_text(encoding="utf-8-sig"))
                  if args.vision_config else options["vision"])
        config["geometry_reference"] = str(args.geometry.resolve())
        frontend = DiagnosticFrontend(config)
        print("正在加载检测模型和几何…", flush=True)
        frontend._load()
        config = frontend.config
        if args.depth:
            depth_config = (json.loads(args.depth_config.read_text(encoding="utf-8-sig"))
                            if args.depth_config else options["depth"])
            capture = CaptureRGBDCamera(capture_config, depth_config)
        else:
            capture = CaptureColorCamera(capture_config)
        with capture as camera:
            session_options = {"depth_calibration": camera.depth_calibration} if args.depth else {}
            metadata.update(geometry_version=geometry["version"],
                            rgb_settings_at_start=camera.rgb_controls.snapshot()
                            if camera.rgb_controls is not None else None)
            session = PreviewSession(output, camera.camera, batch, pipeline_config=config, geometry=geometry,
                                     metadata=metadata, split=args.split, **session_options)
            print(f"诊断目录: {session.path}\nF5 保存；F6 暂停/继续；Esc 退出（也支持空格/S、P、Q）。", flush=True)
            print(f"彩色实际配置：{camera.camera['image_width']}×{camera.camera['image_height']} / "
                  f"{camera.camera['fps']} fps；Z 局部放大；A 自动曝光；[ ] 曝光；- = 增益。", flush=True)
            with PreviewCapture(camera) as live_camera:
                result = run_preview(live_camera, frontend, geometry, config, session,
                                     frames=args.frames, save_first=args.save_first, zoom=args.zoom,
                                     library_root=library_root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["failed_saves"] or result["pending_sync"] or result.get("postprocess_error"):
            status, error, code = "failed", "保存、入库或后处理未全部完成，见本批采集摘要", 2
        if result.get("previous_pending_sync"):
            code = 2
            print("旧会话仍有待同步项，请检查旧会话 intake.json；本批状态单独记录。", flush=True)
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
