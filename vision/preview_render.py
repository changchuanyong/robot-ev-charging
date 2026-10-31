"""Render RGB and native Z16 diagnostics without modifying captured measurements."""
from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_WIDTH = 448
LABELS = ("S_neg", "CC2", "S_pos", "CC1", "DC_neg", "DC_pos", "A_neg", "PE", "A_pos")
OBSERVED = (90, 220, 100)
PROJECTED = (220, 90, 240)
CANDIDATE = (110, 190, 210)
BOX = (240, 180, 70)
ROI = (165, 165, 165)
DISPLAY_NEAR_M, DISPLAY_FAR_M = .15, 1.5


@lru_cache(maxsize=4)
def _font(size):
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for path in (fonts / "msyh.ttc", fonts / "simhei.ttf",
                 Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
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
            cv2.putText(image, str(label)[:16], position, cv2.FONT_HERSHEY_SIMPLEX,
                        .45, OBSERVED, 1, cv2.LINE_AA)


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
        view[y:y + size[1], x:x + size[0]] = enlarged
        return view, True
    except (TypeError, ValueError, OverflowError):
        return image, False


def render_preview(image, diagnostic, *, saved_count=0, paused=False, message="",
                   runtime=None, zoom=False, display_width=None):
    """Return a BGR image with raw-pixel overlays and a Chinese diagnostic panel."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3):
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
        view = cv2.resize(view, (display_width, display_height),
                          interpolation=cv2.INTER_AREA if display_width < width else cv2.INTER_LINEAR)
    view_height, view_width = view.shape[:2]
    canvas = np.full((max(view_height, 864 if extended else 720), view_width + PANEL_WIDTH, 3),
                     (29, 25, 22), dtype=np.uint8)
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
    line(f"{camera.get('serial', '--')}  |  {camera.get('image_width', width)} x "
         f"{camera.get('image_height', height)} @ {camera.get('fps', '--')} fps", 53, fill=muted)
    line(caption(f"帧 {frame.get('frame_number', '--')}  ·  设备时间 {_number(frame.get('device_timestamp_ms'), 1)} ms",
                 f"Frame {frame.get('frame_number', '--')} | Device time {_number(frame.get('device_timestamp_ms'), 1)} ms"), 79, fill=muted)
    assessment = diagnostic.get("assessment", {})
    status = assessment.get("status", "no_pose")
    if not pose.get("solved") and status != "processing":
        status = "no_pose"
    statuses = {"estimated": (caption("位姿估计", "Pose estimated"), (132, 198, 235)),
                "warning": (caption("质量告警", "Quality warning"), (246, 189, 87)),
                "no_pose": (caption("暂无位姿", "No pose"), (239, 135, 128)),
                "processing": (caption("正在分析", "Processing"), (132, 198, 235))}
    label, color = statuses.get(status, statuses["no_pose"])
    if paused:
        label += caption("  ·  已暂停", "  |  Paused")
    line(label, 108, fill=color, selected_font=title_font)
    draw.line((x, 147, right, 147), fill=(66, 73, 83))
    named = frontend.get("named_points_raw_px", {})
    line(caption(f"候选 {len(frontend.get('candidate_points_raw_px', []))}  ·  已命名 {len(named)}/9  ·  耗时 {_number(diagnostic.get('processing_ms'), 0)} ms",
                 f"Candidates {len(frontend.get('candidate_points_raw_px', []))} | Named {len(named)}/9 | Analysis {_number(diagnostic.get('processing_ms'), 0)} ms"), 160)
    metrics = pose if pose.get("solved") else {}
    line(f"Mean {_number(metrics.get('mean_px'))}   RMS {_number(metrics.get('rms_px'))}   Max {_number(metrics.get('max_px'))} px", 186)
    position = metrics.get("tvec_mm", [])
    line("XYZ (mm): " + (" / ".join(_number(v, 1) for v in position) if len(position) == 3 else "--"), 212, fill=muted)
    line(caption("孔名                 单孔残差 / px", "Hole                 Residual / px"), 246, fill=muted)
    for index, name in enumerate(LABELS):
        y = 274 + index * 23
        draw.text((x, y), name, font=font, fill=normal if name in named else muted)
        residual = metrics.get("per_hole", {}).get(name, {}).get("distance_px")
        value = _number(residual) if name in named else caption("未定位", "missing")
        draw.text((x + 234, y), value, font=font, fill=normal if name in named else muted)
    line(caption("绿圈：观测点   紫十字：重投影点", "Green: observed | Magenta: projected"), 490, fill=muted)
    line(caption("黄点：候选   蓝框：检测   灰框：ROI", "Yellow: candidate | Blue: box | Gray: ROI"), 514, fill=muted)
    line(caption("内参：" + ("已独立验证" if camera.get("validated") else "SDK 参数，未独立验证"),
                 "Intrinsics: " + ("validated" if camera.get("validated") else "SDK, not independently checked")), 547, fill=muted)
    geometry = diagnostic.get("geometry", {})
    line(caption("几何：", "Geometry: ") + str(geometry.get("version", "--")) +
         (caption("（名义/未验收）", " (nominal/unverified)")
          if not geometry.get("validated") or geometry.get("status") != "measured_print" else ""), 571, fill=muted)
    reasons = list(assessment.get("reasons", []))
    if not reasons and not pose.get("solved"):
        reasons = [frontend.get("reason") or pose.get("reason") or "not_processed"]
    for index, reason in enumerate(reasons[:2]):
        line(_reason(reason, chinese), 601 + index * 24, fill=color)
    if message:
        line(message, 649, fill=(132, 198, 235))
    elif len(reasons) > 2:
        line(caption(f"另有 {len(reasons) - 2} 项，见保存的诊断 JSON", f"+{len(reasons) - 2} details in saved JSON"), 649, fill=muted)
    else:
        line(message or caption(f"已保存 {saved_count} 帧", f"Saved {saved_count} frames"), 649, fill=(132, 198, 235))
    line(caption("F5 保存   F6 暂停/继续   Esc 退出", "F5 save | F6 pause/resume | Esc quit"), 685)
    if extended:
        stats = runtime or {}
        line(caption(f"实测 FPS：采集 {_number(stats.get('capture_fps'), 1)} / 分析 {_number(stats.get('analysis_fps'), 1)} / 绘制 {_number(stats.get('render_fps'), 1)}",
                     f"Measured FPS: capture {_number(stats.get('capture_fps'), 1)} / analysis {_number(stats.get('analysis_fps'), 1)} / render {_number(stats.get('render_fps'), 1)}"), 718)
        line(caption(f"画面年龄：{_number(stats.get('frame_age_ms'), 0)} ms · 配置 FPS 非实测值",
                     f"Frame age: {_number(stats.get('frame_age_ms'), 0)} ms | configured FPS differs"), 742, fill=muted)
        auto = rgb_settings.get("enable_auto_exposure")
        mode = caption("自动", "Auto") if auto == 1 else caption("手动", "Manual") if auto == 0 else "--"
        line(caption(f"RGB 曝光 {mode} {_number(rgb_settings.get('exposure'), 0)} · 增益 {_number(rgb_settings.get('gain'), 0)}（SDK缓存）",
                     f"RGB {mode} exposure {_number(rgb_settings.get('exposure'), 0)} / gain {_number(rgb_settings.get('gain'), 0)} (SDK cached)"), 766, fill=muted)
        view_label = (caption("局部放大，原图完整保存", "Target zoom; full original saved") if zoomed else
                      caption("局部放大等待唯一目标", "Zoom awaits one target") if zoom else
                      caption("完整画面，Z 切换局部放大", "Full image; Z toggles target zoom"))
        line(view_label, 790, fill=(132, 198, 235))
        line(caption("A 自动曝光 · [ ] 曝光 · - = 增益", "A auto exposure | [ ] exposure | - = gain"), 814)
        line(caption("Z 局部放大 · 调节只影响后续采集帧", "Z target zoom | settings affect later frames"), 838, fill=muted)
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
    if (isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.dtype == np.uint16
            and min(raw.shape) > 0 and isinstance(scale, (float, int)) and np.isfinite(scale) and scale > 0):
        pixels = colorize_depth(raw, scale)
        roi = depth.get("roi") or {}
        uv = (packet or {}).get("color_pixels")
        box = roi.get("inset_bbox_xyxy")
        if depth.get("status") != "invalid" and box is not None and isinstance(uv, np.ndarray):
            if uv.shape == (*raw.shape, 2):
                mask = ((raw > 0) & np.isfinite(uv).all(axis=2)
                        & (uv[..., 0] >= box[0]) & (uv[..., 0] < box[2])
                        & (uv[..., 1] >= box[1]) & (uv[..., 1] < box[3]))
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pixels, contours, -1, (255, 255, 255), 1)
        factor = min(640 / raw.shape[1], 480 / raw.shape[0])
        size = (max(1, round(raw.shape[1] * factor)), max(1, round(raw.shape[0] * factor)))
        resized = cv2.resize(pixels, size, interpolation=cv2.INTER_NEAREST)
        canvas[72:72 + size[1], 16:16 + size[0]] = resized
    legend = np.linspace(DISPLAY_NEAR_M, DISPLAY_FAR_M, 600).reshape(1, -1)
    ramp = np.repeat(np.rint(legend * 10000).astype(np.uint16), 18, axis=0)
    canvas[578:596, 20:620] = colorize_depth(ramp, .0001)
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

    text("D415 深度辅助诊断" + (" · 已暂停" if paused else ""),
         "D415 depth diagnostics" + (" | Paused" if paused else ""), 20, 16, title=True, width=640)
    text("深度相机视图：原始测量，未对齐、未补洞", "Native depth view; no alignment or hole filling", 20, 47, width=640)
    text("0.15 m（近）                        1.50 m（远）", "0.15 m (near)                            1.50 m (far)", 20, 602, width=640)
    text("黑色：无效深度；超出显示范围只作颜色截断", "Black: invalid; display colors clip outside this range", 20, 632, width=640, fill=muted)
    text("白线：检测 ROI 的测量投影，不代表已验证端面", "White: detected ROI footprint, not a verified aperture plane", 20, 658, width=640, fill=muted)
    statuses = {"invalid": ("深度不可用", "Depth invalid"), "scene_only": ("场景深度 · 未确定目标", "Scene only; no target"),
                "roi_insufficient": ("目标深度不足", "Insufficient target depth"),
                "plane_rejected": ("目标平面未通过检查", "Target plane rejected"),
                "candidate_plane": ("候选表面平面 · 仅诊断", "Candidate surface; diagnostic only")}
    status = depth.get("status")
    label = statuses.get(status, ("等待同帧分析", "Waiting for paired analysis"))
    text(*label, 682, 20, title=True, fill=(246, 189, 87))
    frame, color = depth.get("frame") or {}, diagnostic.get("frame", {})
    text(f"RGB 帧 {color.get('frame_number', '--')} / 深度帧 {frame.get('frame_number', '--')}",
         f"RGB {color.get('frame_number', '--')} / Depth {frame.get('frame_number', '--')}", 682, 65)
    text(f"配对时间差：{_number(depth.get('pair_skew_ms'))} ms", f"Pair skew: {_number(depth.get('pair_skew_ms'))} ms", 682, 93)
    scene = depth.get("scene") or {}
    valid = scene.get("valid_fraction")
    valid_pct = _number(100 * valid, 1) if isinstance(valid, (float, int)) else "--"
    text(f"原生深度有效比例：{valid_pct}%", f"Native valid depth: {valid_pct}%", 682, 132)
    text(f"场景深度中位数：{_number(scene.get('median_depth_mm'), 1)} mm",
         f"Scene median depth: {_number(scene.get('median_depth_mm'), 1)} mm", 682, 160)
    roi = (depth.get("roi") or {}) if status != "invalid" else {}
    coverage = roi.get("measurement_coverage_fraction")
    coverage_pct = _number(100 * coverage, 1) if isinstance(coverage, (float, int)) else "--"
    text(f"ROI 测量像素覆盖：{coverage_pct}%", f"ROI measured-pixel coverage: {coverage_pct}%", 682, 202)
    text(f"ROI 距离（彩色 Z）：{_number(roi.get('median_color_z_mm'), 1)} mm",
         f"ROI median color Z: {_number(roi.get('median_color_z_mm'), 1)} mm", 682, 230)
    plane = depth.get("plane", {}) if status == "candidate_plane" else {}
    normal_values = plane.get("normal_color_camera", []) if plane.get("solved") else []
    normal_text = ", ".join(_number(v, 3) for v in normal_values) if len(normal_values) == 3 else "--"
    text("候选法向（彩色相机坐标）：", "Candidate normal (color camera):", 682, 270)
    text(normal_text, normal_text, 682, 298)
    text(f"平面 RMS：{_number(plane.get('rms_mm'))} mm", f"Plane RMS: {_number(plane.get('rms_mm'))} mm", 682, 326)
    compare = (depth.get("pnp_consistency") or {}) if plane.get("solved") else {}
    text(f"PnP 法向差：{_number(compare.get('normal_angle_deg'))}°",
         f"PnP normal difference: {_number(compare.get('normal_angle_deg'))} deg", 682, 365)
    text(f"PnP 参考点到平面：{_number(compare.get('plane_distance_mm'))} mm",
         f"PnP reference point to plane: {_number(compare.get('plane_distance_mm'))} mm", 682, 393)
    text("孔口参考面未确认；不同台阶不能混合解释", "Aperture reference unverified; do not mix steps", 682, 433, fill=muted)
    text("拍屏幕只能测到屏幕，不能还原虚拟模型", "A screen gives screen depth, not virtual geometry", 682, 461, fill=muted)
    translations = {"target_not_detected": "未检测到目标，暂不估计目标平面",
                    "missing_depth": "未收到深度帧", "depth_timestamp_domain_mismatch": "彩色与深度时间域不同",
                    "depth_pair_skew_exceeded": "彩色与深度时间差过大",
                    "repeated_or_reordered_depth_frame": "深度帧重复或倒序",
                    "no_valid_depth": "原始深度没有有效测量"}
    for index, reason in enumerate(depth.get("reasons", [])[:4]):
        text(translations.get(reason, str(reason)), str(reason), 682, 505 + 27 * index, fill=(239, 160, 145))
    text(message or f"已保存 {saved_count} 张", message or f"Saved: {saved_count}",
         682, 607, fill=(132, 198, 235))
    text("F5 保存 RGB + 原始深度 | F6 暂停 | Esc 退出", "F5 save RGB + raw depth | F6 pause | Esc quit", 682, 635, width=442)
    text("SDK 参数未独立验证；结果不用于机器人运动", "SDK calibration unvalidated; no robot motion", 682, 665, fill=muted)
    for (left, top, width, height), panel in zip(regions, panels):
        canvas[top:top + height, left:left + width] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    return canvas
