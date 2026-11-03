"""D415 visual diagnostics: python -m ev_charging.capture.preview (F5 saves, F6 pauses, Esc exits)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ev_charging.capture.capture_session import PreviewSession
from ev_charging.capture.collect import (
    CaptureColorCamera,
    CaptureRGBDCamera,
    PreviewCapture,
    intake_status,
    load_capture_config,
    print_intake_summary,
    validate_config,
)
from ev_charging.vision.calibration.depth_diagnostic import analyze_depth
from ev_charging.vision.contracts import validate_camera
from ev_charging.vision.geometry_reference import load_geometry_reference
from ev_charging.vision.locate.diagnose import DiagnosticFrontend, solve_raw_points
from ev_charging.vision.pipeline import ROOT

WINDOW = "D415 Visual Diagnostics"
DEPTH_WINDOW = "D415 Depth Diagnostics"
SAVE_KEYS = (32, ord("s"), ord("S"), 0x740000, 0xFFC2)  # Windows / X11 F5
PAUSE_KEYS = (ord("p"), ord("P"), 0x750000, 0xFFC3)  # Windows / X11 F6


PANEL_WIDTH = 448
LABELS = ("S_neg", "CC2", "S_pos", "CC1", "DC_neg", "DC_pos", "A_neg", "PE", "A_pos")
OBSERVED = (90, 220, 100)
PROJECTED = (220, 90, 240)
CANDIDATE = (110, 190, 210)
BOX = (240, 180, 70)
ROI = (165, 165, 165)
DISPLAY_NEAR_M, DISPLAY_FAR_M = 0.15, 1.5


@lru_cache(maxsize=4)
def _font(size):
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for path in (
        fonts / "msyh.ttc",
        fonts / "simhei.ttf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size), True
    return ImageFont.load_default(size=size), False


def _number(value, digits=2):
    try:
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "--"
    except (TypeError, ValueError, OverflowError):
        return "--"


def _fit_text(draw, value, font, width, *, ellipsis="…"):
    """Fit one display line; SDK exceptions may contain multiple lines."""
    text = " ".join(str(value).split())
    if draw.textlength(text, font=font) <= width:
        return text
    left, right = 0, len(text)
    while left < right:
        middle = (left + right + 1) // 2
        if draw.textlength(text[:middle] + ellipsis, font=font) <= width:
            left = middle
        else:
            right = middle - 1
    return text[:left] + ellipsis


def _point(value):
    try:
        point = np.asarray(value, dtype=float)
        if point.shape == (2,) and np.isfinite(point).all():
            # Bound integers before passing them to OpenCV's C int API.
            return tuple(int(round(x)) for x in np.clip(point, -(2**28), 2**28))
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _inside(point, width, height):
    return point is not None and 0 <= point[0] < width and 0 <= point[1] < height


def _rectangle(image, value, color, thickness):
    try:
        box = np.asarray(value, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
            return
        height, width = image.shape[:2]
        if box[2] < 0 or box[3] < 0 or box[0] >= width or box[1] >= height:
            return
        left = _point(np.maximum(box[:2], [0, 0]))
        right = _point(np.minimum(box[2:], [width - 1, height - 1]))
        cv2.rectangle(image, left, right, color, thickness, cv2.LINE_AA)
    except (TypeError, ValueError, OverflowError):
        return


def _overlay(image, frontend, pose):
    height, width = image.shape[:2]
    for box in frontend.get("detected_boxes", []):
        _rectangle(image, box.get("xyxy"), BOX, 2)
    _rectangle(image, frontend.get("roi_bbox_xyxy"), ROI, 1)
    for value in frontend.get("candidate_points_raw_px", []):
        point = _point(value)
        if _inside(point, width, height):
            cv2.circle(image, point, 3, CANDIDATE, 1, cv2.LINE_AA)
    # Failed solves may contain diagnostic leftovers; never display them as a pose.
    if pose.get("solved"):
        for item in pose.get("per_hole", {}).values():
            observed, projected = _point(item.get("observed_px")), _point(item.get("projected_px"))
            if observed is not None and projected is not None:
                intersects, start, end = cv2.clipLine((0, 0, width, height), observed, projected)
                if intersects:
                    cv2.line(image, start, end, PROJECTED, 1, cv2.LINE_AA)
            if _inside(projected, width, height):
                cv2.drawMarker(image, projected, PROJECTED, cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA)
    for label, value in frontend.get("named_points_raw_px", {}).items():
        point = _point(value)
        if _inside(point, width, height):
            cv2.circle(image, point, 5, OBSERVED, 2, cv2.LINE_AA)
            position = (max(0, min(point[0] + 7, width - 75)), max(14, min(point[1] - 7, height - 3)))
            cv2.putText(
                image, str(label)[:16], position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, OBSERVED, 1, cv2.LINE_AA
            )


def _reason(value, chinese):
    text = str(value)
    translations = {
        "processing": "等待当前帧分析完成",
        "target_not_detected": "未检测到充电口",
        "ambiguous_complete_target_use_target_window": "检测到多个目标，无法确定对象",
        "no_layout_variant": "孔位不足或无法完成孔名对应",
        "fewer_than_four_points": "不足四个对应点，无法求解位姿",
        "association_rejected": "孔名关联未通过质量检查",
        "ambiguous_correspondences": "存在不确定的孔名对应",
        "rms_exceeded": "RMS 重投影残差超出设置值",
        "point_error_exceeded": "单孔最大残差超出设置值",
        "insufficient_inliers": "满足残差条件的孔位数量不足",
        "pnp_unsolved": "PnP 未求解成功",
    }
    key = text.split(": ", 1)[-1]
    return translations.get(key, text) if chinese else text


def _zoom_target(image, frontend):
    """Fit the unique detected target into the image area; no pixel coordinates change."""
    boxes = frontend.get("detected_boxes", [])
    if len(boxes) != 1:
        return image, False
    try:
        box = np.asarray(boxes[0].get("xyxy"), dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            return image, False
        height, width = image.shape[:2]
        left, top = np.floor(np.clip(box[:2], [0, 0], [width, height])).astype(int)
        right, bottom = np.ceil(np.clip(box[2:], [0, 0], [width, height])).astype(int)
        if right - left < 2 or bottom - top < 2:
            return image, False
        crop = image[top:bottom, left:right]
        factor = min(width / crop.shape[1], height / crop.shape[0])
        size = (max(1, round(crop.shape[1] * factor)), max(1, round(crop.shape[0] * factor)))
        enlarged = cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)
        view = np.full_like(image, (29, 25, 22))
        x, y = (width - size[0]) // 2, (height - size[1]) // 2
        view[y : y + size[1], x : x + size[0]] = enlarged
        return view, True
    except (TypeError, ValueError, OverflowError):
        return image, False


def render_preview(
    image,
    diagnostic,
    *,
    saved_count=0,
    paused=False,
    message="",
    runtime=None,
    zoom=False,
    display_width=None,
):
    """Return a BGR image with raw-pixel overlays and a Chinese diagnostic panel."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("preview_requires_uint8_BGR_image")
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("preview_requires_nonempty_image")
    if display_width is not None and (type(display_width) is not int or display_width < 1):
        raise ValueError("positive_integer_display_width_required")
    frame = diagnostic.get("frame", {})
    rgb_settings = frame.get("rgb_settings", {}).get("sensor_options", {}).get("values", {})
    extended = runtime is not None or bool(rgb_settings) or zoom
    view = image.copy()
    frontend, pose = diagnostic.get("frontend", {}), diagnostic.get("pose", {})
    _overlay(view, frontend, pose)
    zoomed = False
    if zoom:
        view, zoomed = _zoom_target(view, frontend)
    if display_width is not None and display_width != width:
        display_height = max(1, round(height * display_width / width))
        view = cv2.resize(
            view,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA if display_width < width else cv2.INTER_LINEAR,
        )
    view_height, view_width = view.shape[:2]
    canvas = np.full(
        (max(view_height, 864 if extended else 720), view_width + PANEL_WIDTH, 3),
        (29, 25, 22),
        dtype=np.uint8,
    )
    canvas[:view_height, :view_width] = view
    # Only the text panel passes through Pillow; raw RGB pixels stay in BGR.
    panel = Image.new("RGB", (PANEL_WIDTH, canvas.shape[0]), (22, 25, 29))
    draw = ImageDraw.Draw(panel)
    font, chinese = _font(17)
    title_font, _ = _font(23)
    x, right = 20, PANEL_WIDTH - 20
    normal, muted = (225, 230, 235), (164, 175, 187)

    def caption(zh, en):
        return zh if chinese else en

    def line(text, y, *, fill=normal, selected_font=font):
        text = _fit_text(draw, text, selected_font, right - x, ellipsis="…" if chinese else "...")
        draw.text((x, y), text, font=selected_font, fill=fill)

    line(caption("D415 实时视觉诊断", "D415 visual diagnostics"), 17, selected_font=title_font)
    camera = diagnostic.get("camera", {})
    line(
        f"{camera.get('serial', '--')}  |  {camera.get('image_width', width)} x "
        f"{camera.get('image_height', height)} @ {camera.get('fps', '--')} fps",
        53,
        fill=muted,
    )
    line(
        caption(
            f"帧 {frame.get('frame_number', '--')}  ·  设备时间 {_number(frame.get('device_timestamp_ms'), 1)} ms",
            f"Frame {frame.get('frame_number', '--')} | Device time {_number(frame.get('device_timestamp_ms'), 1)} ms",
        ),
        79,
        fill=muted,
    )
    assessment = diagnostic.get("assessment", {})
    status = assessment.get("status", "no_pose")
    if not pose.get("solved") and status != "processing":
        status = "no_pose"
    statuses = {
        "estimated": (caption("位姿估计", "Pose estimated"), (132, 198, 235)),
        "warning": (caption("质量告警", "Quality warning"), (246, 189, 87)),
        "no_pose": (caption("暂无位姿", "No pose"), (239, 135, 128)),
        "processing": (caption("正在分析", "Processing"), (132, 198, 235)),
    }
    label, color = statuses.get(status, statuses["no_pose"])
    if paused:
        label += caption("  ·  已暂停", "  |  Paused")
    line(label, 108, fill=color, selected_font=title_font)
    draw.line((x, 147, right, 147), fill=(66, 73, 83))
    named = frontend.get("named_points_raw_px", {})
    line(
        caption(
            f"候选 {len(frontend.get('candidate_points_raw_px', []))}  ·  已命名 {len(named)}/9  ·  耗时 {_number(diagnostic.get('processing_ms'), 0)} ms",
            f"Candidates {len(frontend.get('candidate_points_raw_px', []))} | Named {len(named)}/9 | Analysis {_number(diagnostic.get('processing_ms'), 0)} ms",
        ),
        160,
    )
    metrics = pose if pose.get("solved") else {}
    line(
        f"Mean {_number(metrics.get('mean_px'))}   RMS {_number(metrics.get('rms_px'))}   Max {_number(metrics.get('max_px'))} px",
        186,
    )
    position = metrics.get("tvec_mm", [])
    line(
        "XYZ (mm): " + (" / ".join(_number(v, 1) for v in position) if len(position) == 3 else "--"),
        212,
        fill=muted,
    )
    line(caption("孔名                 单孔残差 / px", "Hole                 Residual / px"), 246, fill=muted)
    for index, name in enumerate(LABELS):
        y = 274 + index * 23
        draw.text((x, y), name, font=font, fill=normal if name in named else muted)
        residual = metrics.get("per_hole", {}).get(name, {}).get("distance_px")
        value = _number(residual) if name in named else caption("未定位", "missing")
        draw.text((x + 234, y), value, font=font, fill=normal if name in named else muted)
    line(caption("绿圈：观测点   紫十字：重投影点", "Green: observed | Magenta: projected"), 490, fill=muted)
    line(
        caption("黄点：候选   蓝框：检测   灰框：ROI", "Yellow: candidate | Blue: box | Gray: ROI"),
        514,
        fill=muted,
    )
    line(
        caption(
            "内参：" + ("已独立验证" if camera.get("validated") else "SDK 参数，未独立验证"),
            "Intrinsics: " + ("validated" if camera.get("validated") else "SDK, not independently checked"),
        ),
        547,
        fill=muted,
    )
    geometry = diagnostic.get("geometry", {})
    line(
        caption("几何：", "Geometry: ")
        + str(geometry.get("version", "--"))
        + (
            caption("（名义/未验收）", " (nominal/unverified)")
            if not geometry.get("validated") or geometry.get("status") != "measured_print"
            else ""
        ),
        571,
        fill=muted,
    )
    reasons = list(assessment.get("reasons", []))
    if not reasons and not pose.get("solved"):
        reasons = [frontend.get("reason") or pose.get("reason") or "not_processed"]
    for index, reason in enumerate(reasons[:2]):
        line(_reason(reason, chinese), 601 + index * 24, fill=color)
    if message:
        line(message, 649, fill=(132, 198, 235))
    elif len(reasons) > 2:
        line(
            caption(
                f"另有 {len(reasons) - 2} 项，见保存的诊断 JSON", f"+{len(reasons) - 2} details in saved JSON"
            ),
            649,
            fill=muted,
        )
    else:
        line(
            message or caption(f"已保存 {saved_count} 帧", f"Saved {saved_count} frames"),
            649,
            fill=(132, 198, 235),
        )
    line(caption("F5 保存   F6 暂停/继续   Esc 退出", "F5 save | F6 pause/resume | Esc quit"), 685)
    if extended:
        stats = runtime or {}
        line(
            caption(
                f"实测 FPS：采集 {_number(stats.get('capture_fps'), 1)} / 分析 {_number(stats.get('analysis_fps'), 1)} / 绘制 {_number(stats.get('render_fps'), 1)}",
                f"Measured FPS: capture {_number(stats.get('capture_fps'), 1)} / analysis {_number(stats.get('analysis_fps'), 1)} / render {_number(stats.get('render_fps'), 1)}",
            ),
            718,
        )
        line(
            caption(
                f"画面年龄：{_number(stats.get('frame_age_ms'), 0)} ms · 配置 FPS 非实测值",
                f"Frame age: {_number(stats.get('frame_age_ms'), 0)} ms | configured FPS differs",
            ),
            742,
            fill=muted,
        )
        auto = rgb_settings.get("enable_auto_exposure")
        mode = caption("自动", "Auto") if auto == 1 else caption("手动", "Manual") if auto == 0 else "--"
        line(
            caption(
                f"RGB 曝光 {mode} {_number(rgb_settings.get('exposure'), 0)} · 增益 {_number(rgb_settings.get('gain'), 0)}（SDK缓存）",
                f"RGB {mode} exposure {_number(rgb_settings.get('exposure'), 0)} / gain {_number(rgb_settings.get('gain'), 0)} (SDK cached)",
            ),
            766,
            fill=muted,
        )
        view_label = (
            caption("局部放大，原图完整保存", "Target zoom; full original saved")
            if zoomed
            else caption("局部放大等待唯一目标", "Zoom awaits one target")
            if zoom
            else caption("完整画面，Z 切换局部放大", "Full image; Z toggles target zoom")
        )
        line(view_label, 790, fill=(132, 198, 235))
        line(caption("A 自动曝光 · [ ] 曝光 · - = 增益", "A auto exposure | [ ] exposure | - = gain"), 814)
        line(
            caption("Z 局部放大 · 调节只影响后续采集帧", "Z target zoom | settings affect later frames"),
            838,
            fill=muted,
        )
    canvas[:, view_width:] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    return canvas


def colorize_depth(raw, scale):
    """Fixed display scale, black invalid pixels, no smoothing or hole filling."""
    depth = raw.astype(np.float64) * scale
    levels = np.clip((depth - DISPLAY_NEAR_M) / (DISPLAY_FAR_M - DISPLAY_NEAR_M), 0, 1)
    result = cv2.applyColorMap(np.rint(255 * (1 - levels)).astype(np.uint8), cv2.COLORMAP_TURBO)
    result[(raw == 0) | ~np.isfinite(depth)] = 0
    return result


def render_depth(packet, diagnostic, *, paused=False, saved_count=0, message=""):
    """A separate panel for the same completed RGB/depth snapshot as the RGB UI."""
    canvas = np.full((720, 1140, 3), (29, 25, 22), dtype=np.uint8)
    depth = diagnostic.get("depth", {})
    raw = packet.get("raw_z16") if packet else None
    scale = (packet or {}).get("calibration", {}).get("depth_scale_m")
    if (
        isinstance(raw, np.ndarray)
        and raw.ndim == 2
        and raw.dtype == np.uint16
        and min(raw.shape) > 0
        and isinstance(scale, (float, int))
        and np.isfinite(scale)
        and scale > 0
    ):
        pixels = colorize_depth(raw, scale)
        roi = depth.get("roi") or {}
        uv = (packet or {}).get("color_pixels")
        box = roi.get("inset_bbox_xyxy")
        if depth.get("status") != "invalid" and box is not None and isinstance(uv, np.ndarray):
            if uv.shape == (*raw.shape, 2):
                mask = (
                    (raw > 0)
                    & np.isfinite(uv).all(axis=2)
                    & (uv[..., 0] >= box[0])
                    & (uv[..., 0] < box[2])
                    & (uv[..., 1] >= box[1])
                    & (uv[..., 1] < box[3])
                )
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(pixels, contours, -1, (255, 255, 255), 1)
        factor = min(640 / raw.shape[1], 480 / raw.shape[0])
        size = (max(1, round(raw.shape[1] * factor)), max(1, round(raw.shape[0] * factor)))
        resized = cv2.resize(pixels, size, interpolation=cv2.INTER_NEAREST)
        canvas[72 : 72 + size[1], 16 : 16 + size[0]] = resized
    legend = np.linspace(DISPLAY_NEAR_M, DISPLAY_FAR_M, 600).reshape(1, -1)
    ramp = np.repeat(np.rint(legend * 10000).astype(np.uint16), 18, axis=0)
    canvas[578:596, 20:620] = colorize_depth(ramp, 0.0001)
    # The native depth image and color ramp never pass through Pillow.
    regions = [(0, 0, 662, 72), (0, 596, 662, 124), (662, 0, 478, 720)]
    panels = [Image.new("RGB", (width, height), (22, 25, 29)) for _, _, width, height in regions]
    draws = [ImageDraw.Draw(panel) for panel in panels]
    font, chinese = _font(17)
    title_font, _ = _font(23)
    normal, muted = (225, 230, 235), (164, 175, 187)

    def text(zh, en, x, y, *, title=False, fill=normal, width=442):
        index = 2 if x >= 662 else 1 if y >= 596 else 0
        draw = draws[index]
        value, chosen = zh if chinese else en, title_font if title else font
        value = _fit_text(draw, value, chosen, width, ellipsis="…" if chinese else "...")
        left, top, _, _ = regions[index]
        draw.text((x - left, y - top), value, font=chosen, fill=fill)

    text(
        "D415 深度辅助诊断" + (" · 已暂停" if paused else ""),
        "D415 depth diagnostics" + (" | Paused" if paused else ""),
        20,
        16,
        title=True,
        width=640,
    )
    text(
        "深度相机视图：原始测量，未对齐、未补洞",
        "Native depth view; no alignment or hole filling",
        20,
        47,
        width=640,
    )
    text(
        "0.15 m（近）                        1.50 m（远）",
        "0.15 m (near)                            1.50 m (far)",
        20,
        602,
        width=640,
    )
    text(
        "黑色：无效深度；超出显示范围只作颜色截断",
        "Black: invalid; display colors clip outside this range",
        20,
        632,
        width=640,
        fill=muted,
    )
    text(
        "白线：检测 ROI 的测量投影，不代表已验证端面",
        "White: detected ROI footprint, not a verified aperture plane",
        20,
        658,
        width=640,
        fill=muted,
    )
    statuses = {
        "invalid": ("深度不可用", "Depth invalid"),
        "scene_only": ("场景深度 · 未确定目标", "Scene only; no target"),
        "roi_insufficient": ("目标深度不足", "Insufficient target depth"),
        "plane_rejected": ("目标平面未通过检查", "Target plane rejected"),
        "candidate_plane": ("候选表面平面 · 仅诊断", "Candidate surface; diagnostic only"),
    }
    status = depth.get("status")
    label = statuses.get(status, ("等待同帧分析", "Waiting for paired analysis"))
    text(*label, 682, 20, title=True, fill=(246, 189, 87))
    frame, color = depth.get("frame") or {}, diagnostic.get("frame", {})
    text(
        f"RGB 帧 {color.get('frame_number', '--')} / 深度帧 {frame.get('frame_number', '--')}",
        f"RGB {color.get('frame_number', '--')} / Depth {frame.get('frame_number', '--')}",
        682,
        65,
    )
    text(
        f"配对时间差：{_number(depth.get('pair_skew_ms'))} ms",
        f"Pair skew: {_number(depth.get('pair_skew_ms'))} ms",
        682,
        93,
    )
    scene = depth.get("scene") or {}
    valid = scene.get("valid_fraction")
    valid_pct = _number(100 * valid, 1) if isinstance(valid, (float, int)) else "--"
    text(f"原生深度有效比例：{valid_pct}%", f"Native valid depth: {valid_pct}%", 682, 132)
    text(
        f"场景深度中位数：{_number(scene.get('median_depth_mm'), 1)} mm",
        f"Scene median depth: {_number(scene.get('median_depth_mm'), 1)} mm",
        682,
        160,
    )
    roi = (depth.get("roi") or {}) if status != "invalid" else {}
    coverage = roi.get("measurement_coverage_fraction")
    coverage_pct = _number(100 * coverage, 1) if isinstance(coverage, (float, int)) else "--"
    text(f"ROI 测量像素覆盖：{coverage_pct}%", f"ROI measured-pixel coverage: {coverage_pct}%", 682, 202)
    text(
        f"ROI 距离（彩色 Z）：{_number(roi.get('median_color_z_mm'), 1)} mm",
        f"ROI median color Z: {_number(roi.get('median_color_z_mm'), 1)} mm",
        682,
        230,
    )
    plane = depth.get("plane", {}) if status == "candidate_plane" else {}
    normal_values = plane.get("normal_color_camera", []) if plane.get("solved") else []
    normal_text = ", ".join(_number(v, 3) for v in normal_values) if len(normal_values) == 3 else "--"
    text("候选法向（彩色相机坐标）：", "Candidate normal (color camera):", 682, 270)
    text(normal_text, normal_text, 682, 298)
    text(
        f"平面 RMS：{_number(plane.get('rms_mm'))} mm",
        f"Plane RMS: {_number(plane.get('rms_mm'))} mm",
        682,
        326,
    )
    compare = (depth.get("pnp_consistency") or {}) if plane.get("solved") else {}
    text(
        f"PnP 法向差：{_number(compare.get('normal_angle_deg'))}°",
        f"PnP normal difference: {_number(compare.get('normal_angle_deg'))} deg",
        682,
        365,
    )
    text(
        f"PnP 参考点到平面：{_number(compare.get('plane_distance_mm'))} mm",
        f"PnP reference point to plane: {_number(compare.get('plane_distance_mm'))} mm",
        682,
        393,
    )
    text(
        "孔口参考面未确认；不同台阶不能混合解释",
        "Aperture reference unverified; do not mix steps",
        682,
        433,
        fill=muted,
    )
    text(
        "拍屏幕只能测到屏幕，不能还原虚拟模型",
        "A screen gives screen depth, not virtual geometry",
        682,
        461,
        fill=muted,
    )
    translations = {
        "target_not_detected": "未检测到目标，暂不估计目标平面",
        "missing_depth": "未收到深度帧",
        "depth_timestamp_domain_mismatch": "彩色与深度时间域不同",
        "depth_pair_skew_exceeded": "彩色与深度时间差过大",
        "repeated_or_reordered_depth_frame": "深度帧重复或倒序",
        "no_valid_depth": "原始深度没有有效测量",
    }
    for index, reason in enumerate(depth.get("reasons", [])[:4]):
        text(translations.get(reason, str(reason)), str(reason), 682, 505 + 27 * index, fill=(239, 160, 145))
    text(
        message or f"已保存 {saved_count} 张",
        message or f"Saved: {saved_count}",
        682,
        607,
        fill=(132, 198, 235),
    )
    text(
        "F5 保存 RGB + 原始深度 | F6 暂停 | Esc 退出",
        "F5 save RGB + raw depth | F6 pause | Esc quit",
        682,
        635,
        width=442,
    )
    text(
        "SDK 参数未独立验证；结果不用于机器人运动",
        "SDK calibration unvalidated; no robot motion",
        682,
        665,
        fill=muted,
    )
    for (left, top, width, height), panel in zip(regions, panels):
        canvas[top : top + height, left : left + width] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    return canvas


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
        recent = [stamp for stamp in self.times if now - stamp <= 2.0]
        elapsed = now - recent[0] if len(recent) > 1 else 0
        return (len(recent) - 1) / elapsed if elapsed > 0 else 0.0

    def mean_fps(self):
        elapsed = self.times[-1] - self.first if self.count > 1 else 0
        return (self.count - 1) / elapsed if elapsed > 0 else 0.0


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
        "schema_version": 1,
        "kind": "d415_visual_diagnostic",
        "source": "live_diagnostic",
        "authorizes_robot_motion": False,
        "frame": dict(frame),
        "camera": dict(camera),
        "geometry": {key: geometry[key] for key in ("version", "status", "validated", "source_sha256")},
        "image_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "frontend": features,
        "pose": pose,
        "assessment": assess_preview(features, pose, config),
        "processing_ms": (time.monotonic() - started) * 1000,
    }
    if depth_sample is not None:
        result["schema_version"] = 2
        result["depth"] = analyze_depth(depth_sample, frame, camera, features, pose, geometry)
        result["processing_ms"] = (time.monotonic() - started) * 1000
    return result


def run_preview(
    camera, frontend, geometry, config, session, *, frames=0, save_first=False, zoom=False, library_root=None
):
    """One inference worker; display and save only the matching completed image."""
    from ev_charging.capture.intake import CaptureIntake

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
                displayed = (
                    image.copy(),
                    {
                        "frontend": {},
                        "pose": {"solved": False, "reason": "processing"},
                        "assessment": {"status": "processing", "reasons": []},
                        "camera": camera.camera,
                        "frame": dict(metadata),
                        "geometry": geometry,
                    },
                    None,
                )
            shown_image, diagnostic, shown_depth = displayed
            capture_stats = (
                camera.capture_stats()
                if callable(getattr(camera, "capture_stats", None))
                else {"capture_fps": read_rate.fps()}
            )
            frame_age = None
            try:
                received = datetime.fromisoformat(diagnostic["frame"]["received_at_utc"])
                frame_age = max(0.0, (datetime.now(timezone.utc) - received).total_seconds() * 1000)
            except (KeyError, ValueError, TypeError):
                pass
            runtime = {
                **capture_stats,
                "analysis_fps": 0.0 if paused else analysis_rate.fps(),
                "render_fps": 0.0 if paused else render_rate.fps(),
                "frame_age_ms": frame_age,
                "target_zoom": zoom,
                "display_image_width": 1280,
                "intake": intake_report,
            }
            status_line = intake_status(intake_report)
            display_message = status_line + ("；" + message if message else "")
            # Cached panels still process keyboard/window events on every iteration.
            key_now = (
                processed,
                paused,
                display_message,
                intake_report["saved"],
                zoom,
                int(time.monotonic() * 2),
            )
            if key_now != rendered_key:
                overlay = render_preview(
                    shown_image,
                    diagnostic,
                    saved_count=intake_report["saved"],
                    paused=paused,
                    message=display_message,
                    runtime=runtime,
                    zoom=zoom,
                    display_width=1280,
                )
                if rendered_key is None or rendered_key[0] != processed:
                    render_rate.tick()
                rendered_key = key_now
            cv2.imshow(WINDOW, overlay)
            if use_depth:
                depth_now = (processed, paused, display_message, intake_report["saved"])
                if depth_now != depth_key:
                    depth_overlay = render_depth(
                        shown_depth,
                        diagnostic,
                        paused=paused,
                        saved_count=intake_report["saved"],
                        message=display_message,
                    )
                    depth_key = depth_now
                cv2.imshow(DEPTH_WINDOW, depth_overlay)
            key = cv2.waitKeyEx(1)
            if (
                key in (27, ord("q"), ord("Q"))
                or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
                or (use_depth and cv2.getWindowProperty(DEPTH_WINDOW, cv2.WND_PROP_VISIBLE) < 1)
            ):
                break
            if key in PAUSE_KEYS:
                paused = not paused
            if key in (ord("z"), ord("Z")):
                zoom = not zoom
            rgb_action = {
                ord("a"): ("toggle_auto", None),
                ord("A"): ("toggle_auto", None),
                ord("["): ("adjust_exposure", -1),
                ord("]"): ("adjust_exposure", 1),
                ord("-"): ("adjust_gain", -1),
                ord("="): ("adjust_gain", 1),
                ord("+"): ("adjust_gain", 1),
            }.get(key)
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
                    accepted = intake.submit(
                        shown_image,
                        {**diagnostic, "preview_runtime": runtime},
                        overlay,
                        depth_sample=shown_depth if use_depth else None,
                    )
                    message = "已加入保存队列" if accepted else "队列繁忙或该帧已排队，请稍后重试"
                    print(message, flush=True)
            if frames and processed >= frames:
                break
            if pending is None and not paused and not stale_color:
                snapshot, frame_info = image.copy(), dict(metadata)
                depth_snapshot = deepcopy(depth_sample)

                def analyze(current_image=snapshot, current_frame=frame_info, current_depth=depth_snapshot):
                    options = {"depth_sample": current_depth} if use_depth else {}
                    result = process_frame(
                        frontend, current_image, camera.camera, current_frame, geometry, config, **options
                    )
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
    capture_stats = (
        camera.capture_stats()
        if callable(getattr(camera, "capture_stats", None))
        else {"capture_mean_fps": read_rate.mean_fps(), "captured_frames": read_rate.count}
    )
    return {
        "processed_frames": processed,
        "saved_frames": intake_report["saved"],
        **capture_stats,
        "analysis_mean_fps": analysis_rate.mean_fps(),
        "result_render_mean_fps": render_rate.mean_fps(),
        "consumed_mean_fps": read_rate.mean_fps(),
        **intake_report,
        "last_assessment": displayed[1]["assessment"] if displayed else None,
        "last_depth_status": displayed[1].get("depth", {}).get("status") if displayed else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "--capture-config",
        dest="capture_config",
        type=Path,
        default=ROOT / "config/d415.json",
        help="D415 capture, depth and vision configuration",
    )
    parser.add_argument(
        "--quality",
        choices=("full-hd", "hd"),
        help="Override RGB to 1920x1080/30 or 1280x720/30; otherwise use capture config",
    )
    exposure_group = parser.add_mutually_exclusive_group()
    exposure_group.add_argument("--exposure", type=float, help="Manual RGB exposure in SDK option units")
    exposure_group.add_argument("--auto-exposure", action="store_true", help="Enable RGB auto exposure")
    parser.add_argument("--gain", type=float, help="Manual RGB gain in SDK option units")
    parser.add_argument(
        "--zoom", action="store_true", help="Enlarge detected target for display; keep full raw images"
    )
    parser.add_argument(
        "--vision-config", type=Path, help="Detector and PnP settings only; no control output is written"
    )
    parser.add_argument("--geometry", type=Path, default=ROOT / "config/geometry_nominal.json")
    parser.add_argument("--serial", help="Explicit D415 serial; otherwise use capture configuration")
    parser.add_argument("--batch", help="Capture batch; defaults to a unique diagnostic batch")
    parser.add_argument(
        "--split",
        choices=("train", "val", "unknown"),
        default="train",
        help="整批图库用途；默认 train，val 用于开发验证，unknown 留待分组",
    )
    parser.add_argument("--specimen")
    parser.add_argument("--printing-batch")
    parser.add_argument("--notes", default="D415 SDK visual diagnostics; unreviewed images")
    parser.add_argument("--output-root", type=Path, help="Override the preview profile's output_root")
    library_group = parser.add_mutually_exclusive_group()
    library_group.add_argument(
        "--library",
        type=Path,
        default=ROOT / "dataset/vision",
        help="统一 RGB 图库，默认 dataset/vision；新图待标注",
    )
    library_group.add_argument("--no-library", action="store_true", help="仅保存诊断会话，不加入训练图库")
    parser.add_argument(
        "--frames", type=int, default=0, help="Stop after N analyzed frames; 0 keeps window open"
    )
    parser.add_argument(
        "--save-first", action="store_true", help="Save first completed result, including failures"
    )
    parser.add_argument("--depth", action="store_true", help="Add raw depth capture and diagnostic window")
    parser.add_argument(
        "--depth-config", type=Path, help="Override the depth section with a separate configuration"
    )
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
        capture_config = configure_capture(
            capture_config,
            quality=args.quality,
            auto_exposure=args.auto_exposure,
            exposure=args.exposure,
            gain=args.gain,
        )
        if capture_config["device_model"] != "D415":
            raise ValueError("D415 preview requires an explicit D415 capture configuration")
        output = args.output_root or Path(capture_config.get("output_root", "artifacts/d415_preview"))
        output = (output if output.is_absolute() else ROOT / output).resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError("Preview output must be within the project for annotation review")
        library_root = (
            None
            if args.no_library
            else (args.library if args.library.is_absolute() else ROOT / args.library).resolve()
        )
        batch = args.batch or "d415_preview_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        metadata = {
            "specimen_id": args.specimen,
            "printing_batch_id": args.printing_batch,
            "notes": args.notes,
            "training_library_enabled": library_root is not None,
            "training_library_path": str(library_root) if library_root is not None else None,
            "capture_config": capture_config,
        }
        if library_root is not None:
            from ev_charging.datasets.vision.dataset_split import check_capture_split

            check_capture_split(library_root, batch, args.split, metadata)
        geometry = load_geometry_reference(args.geometry)
        if geometry["status"] == "measured_print" and (
            geometry["specimen_id"] != args.specimen or geometry["printing_batch_id"] != args.printing_batch
        ):
            raise ValueError("Measured geometry requires matching --specimen and --printing-batch")
        config = (
            json.loads(args.vision_config.read_text(encoding="utf-8-sig"))
            if args.vision_config
            else options["vision"]
        )
        config["geometry_reference"] = str(args.geometry.resolve())
        frontend = DiagnosticFrontend(config)
        print("正在加载检测模型和几何…", flush=True)
        frontend._load()
        config = frontend.config
        if args.depth:
            depth_config = (
                json.loads(args.depth_config.read_text(encoding="utf-8-sig"))
                if args.depth_config
                else options["depth"]
            )
            capture = CaptureRGBDCamera(capture_config, depth_config)
        else:
            capture = CaptureColorCamera(capture_config)
        with capture as camera:
            session_options = {"depth_calibration": camera.depth_calibration} if args.depth else {}
            metadata.update(
                geometry_version=geometry["version"],
                rgb_settings_at_start=camera.rgb_controls.snapshot()
                if camera.rgb_controls is not None
                else None,
            )
            session = PreviewSession(
                output,
                camera.camera,
                batch,
                pipeline_config=config,
                geometry=geometry,
                metadata=metadata,
                split=args.split,
                **session_options,
            )
            print(
                f"诊断目录: {session.path}\nF5 保存；F6 暂停/继续；Esc 退出（也支持空格/S、P、Q）。",
                flush=True,
            )
            print(
                f"彩色实际配置：{camera.camera['image_width']}×{camera.camera['image_height']} / "
                f"{camera.camera['fps']} fps；Z 局部放大；A 自动曝光；[ ] 曝光；- = 增益。",
                flush=True,
            )
            with PreviewCapture(camera) as live_camera:
                result = run_preview(
                    live_camera,
                    frontend,
                    geometry,
                    config,
                    session,
                    frames=args.frames,
                    save_first=args.save_first,
                    zoom=args.zoom,
                    library_root=library_root,
                )
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
