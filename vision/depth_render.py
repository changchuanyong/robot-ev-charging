"""Display native Z16 measurements; visualization is never a depth measurement."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .preview_render import _font, _number


DISPLAY_NEAR_M, DISPLAY_FAR_M = .15, 1.5


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
    panel = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(panel)
    font, chinese = _font(17)
    title_font, _ = _font(23)
    normal, muted = (225, 230, 235), (164, 175, 187)

    def text(zh, en, x, y, *, title=False, fill=normal, width=442):
        value, chosen = zh if chinese else en, title_font if title else font
        while value and draw.textlength(value, font=chosen) > width:
            value = value[:-2] + "…"
        draw.text((x, y), value, font=chosen, fill=fill)

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
    return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
