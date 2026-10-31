"""In-memory illumination enhancement and adaptive edge extraction."""
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


# Keep the final image stable for Canny: normalize illumination, avoid excessive local contrast,
# then sharpen only edge regions so flat metal surfaces do not turn into noise.
TARGET_MEAN = 112.0
GAMMA_MIN = 0.75
GAMMA_MAX = 1.45
ILLUMINATION_BLUR_RATIO = 0.085
CLAHE_TILE_GRID_SIZE = (8, 8)
CLAHE_CLIP_MIN = 1.2
CLAHE_CLIP_MAX = 2.4
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 42
BILATERAL_SIGMA_SPACE = 42
SHARPEN_SIGMA = 1.15
SHARPEN_AMOUNT_MIN = 0.20
SHARPEN_AMOUNT_MAX = 0.55
EDGE_SHARPEN_PCT = 72


# 预处理参数
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

EDGE_BILATERAL_D = 5
EDGE_BILATERAL_SIGMA_COLOR = 50
EDGE_BILATERAL_SIGMA_SPACE = 50

# 边缘方法：canny / morph / hybrid
# 默认使用 canny；hybrid 会补齐小孔边缘，但在当前样张上容易把大孔与外圈粘连。
EDGE_METHOD = os.environ.get("VISION_EDGE_METHOD", "canny").strip().lower()

# 自适应 Canny 参数
# 推荐先用这个范围：
# HIGH_PCT 80~90
# LOW_RATIO 0.35~0.55
HIGH_PCT = 90
LOW_RATIO = 0.35

# 形态学梯度用于补充孔位闭合轮廓
MORPH_GRAD_KERNEL = 5
MORPH_GRAD_PCT = 82

# 形态学
CLOSE_KERNEL = 6
CLOSE_ITER = 1

# 结构区域掩膜：抑制桌面、外框边缘，保留中心孔位区域
ENABLE_STRUCTURE_MASK = True
STRUCTURE_MASK_RX_RATIO = 0.48
STRUCTURE_MASK_RY_RATIO = 0.42

# 小连通域去除
MIN_COMPONENT_AREA = 20
# =========================


def image_stats(gray: np.ndarray) -> dict[str, float]:
    p2, p50, p98 = np.percentile(gray, [2, 50, 98])
    return {
        "mean": round(float(np.mean(gray)), 4),
        "std": round(float(np.std(gray)), 4),
        "p2": round(float(p2), 4),
        "p50": round(float(p50), 4),
        "p98": round(float(p98), 4),
        "dynamic_range_2_98": round(float(p98 - p2), 4),
    }


def choose_gamma(gray: np.ndarray) -> float:
    mean_value = max(1.0, float(np.mean(gray)))
    normalized_mean = np.clip(mean_value / 255.0, 1e-3, 0.999)
    normalized_target = np.clip(TARGET_MEAN / 255.0, 1e-3, 0.999)
    gamma = np.log(normalized_target) / np.log(normalized_mean)
    return float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))


def gamma_correction(gray: np.ndarray, gamma: float) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    corrected = np.power(gray_f, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def odd_kernel_from_shape(shape: tuple[int, int]) -> int:
    h, w = shape[:2]
    k = int(round(min(h, w) * ILLUMINATION_BLUR_RATIO))
    k = max(31, k)
    if k % 2 == 0:
        k += 1
    return k


def normalize_illumination(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    kernel = odd_kernel_from_shape(gray.shape)
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    normalized = cv2.divide(gray, background, scale=128)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8), background, kernel


def choose_clahe_clip(gray: np.ndarray) -> float:
    contrast = float(np.std(gray))
    if contrast < 28:
        clip = CLAHE_CLIP_MAX
    elif contrast > 58:
        clip = CLAHE_CLIP_MIN
    else:
        t = (contrast - 28.0) / 30.0
        clip = CLAHE_CLIP_MAX * (1.0 - t) + CLAHE_CLIP_MIN * t
    return float(np.clip(clip, CLAHE_CLIP_MIN, CLAHE_CLIP_MAX))


def limited_unsharp(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_u8 = cv2.convertScaleAbs(grad)

    valid = grad_u8[grad_u8 > 0]
    edge_thresh = int(np.percentile(valid, EDGE_SHARPEN_PCT)) if valid.size >= 50 else 24
    edge_mask = (grad_u8 >= edge_thresh).astype(np.float32)
    edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), sigmaX=1.0)

    contrast = float(np.std(gray))
    amount = np.interp(contrast, [24.0, 64.0], [SHARPEN_AMOUNT_MAX, SHARPEN_AMOUNT_MIN])
    amount = float(np.clip(amount, SHARPEN_AMOUNT_MIN, SHARPEN_AMOUNT_MAX))

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=SHARPEN_SIGMA, sigmaY=SHARPEN_SIGMA)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    blended = gray.astype(np.float32) * (1.0 - edge_mask) + sharp.astype(np.float32) * edge_mask
    return np.clip(blended, 0, 255).astype(np.uint8), grad_u8, amount


def preprocess_roi(roi_bgr: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    gamma = choose_gamma(gray)
    gamma_img = gamma_correction(gray, gamma=gamma)

    normalized, illumination_bg, illumination_kernel = normalize_illumination(gamma_img)

    clahe_clip = choose_clahe_clip(normalized)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=CLAHE_TILE_GRID_SIZE)
    local_contrast = clahe.apply(normalized)

    filtered = cv2.bilateralFilter(
        local_contrast,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )

    final_img, gradient, sharpen_amount = limited_unsharp(filtered)

    metrics = {
        "input": image_stats(gray),
        "output": image_stats(final_img),
        "gamma": round(gamma, 4),
        "illumination_kernel": illumination_kernel,
        "clahe_clip_limit": round(clahe_clip, 4),
        "clahe_tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
        "bilateral": {
            "d": BILATERAL_D,
            "sigma_color": BILATERAL_SIGMA_COLOR,
            "sigma_space": BILATERAL_SIGMA_SPACE,
        },
        "sharpen": {
            "sigma": SHARPEN_SIGMA,
            "amount": round(sharpen_amount, 4),
            "edge_percentile": EDGE_SHARPEN_PCT,
        },
    }

    return {
        "gray": gray,
        "gamma": gamma_img,
        "illumination_background": illumination_bg,
        "illumination_normalized": normalized,
        "clahe": local_contrast,
        "filtered": filtered,
        "gradient": gradient,
        "final": final_img,
        "metrics": metrics,
    }


def remove_small_components(binary_img: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    out = np.zeros_like(binary_img)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255

    return out


def build_structure_mask(shape: tuple[int, int]) -> np.ndarray:
    h_img, w_img = shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    center = (int(w_img / 2), int(h_img / 2))
    axes = (
        max(1, int(STRUCTURE_MASK_RX_RATIO * w_img)),
        max(1, int(STRUCTURE_MASK_RY_RATIO * h_img)),
    )
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def preprocess(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    enhanced = clahe.apply(gray)

    smooth = cv2.bilateralFilter(
        enhanced,
        d=EDGE_BILATERAL_D,
        sigmaColor=EDGE_BILATERAL_SIGMA_COLOR,
        sigmaSpace=EDGE_BILATERAL_SIGMA_SPACE,
    )
    return enhanced, smooth


def compute_adaptive_thresholds(img_gray: np.ndarray) -> tuple[int, int, np.ndarray]:
    """
    用梯度幅值统计来估计 Canny 阈值：
    先对预处理后的灰度图求 Sobel 梯度，
    再按梯度分布百分位数自适应给出 high / low。
    """
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    mag_u8 = cv2.convertScaleAbs(mag)

    valid = mag_u8[mag_u8 > 0]
    if valid.size < 50:
        return 30, 80, mag_u8

    high = int(np.percentile(valid, HIGH_PCT))
    high = max(20, min(255, high))

    low = int(max(0, min(high - 1, LOW_RATIO * high)))
    low = max(5, low)

    return low, high, mag_u8


def morph_gradient_edges(img_gray: np.ndarray) -> tuple[np.ndarray, int]:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_GRAD_KERNEL, MORPH_GRAD_KERNEL),
    )
    grad = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)

    valid = grad[grad > 0]
    if valid.size < 50:
        thresh = 20
    else:
        thresh = int(np.percentile(valid, MORPH_GRAD_PCT))
        thresh = max(12, min(180, thresh))

    _, edges = cv2.threshold(grad, thresh, 255, cv2.THRESH_BINARY)
    return edges, thresh


def adaptive_canny_pipeline(roi_bgr: np.ndarray):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    enhanced, smooth = preprocess(gray)
    low, high, grad_mag = compute_adaptive_thresholds(smooth)

    canny_edges = cv2.Canny(
        smooth,
        threshold1=low,
        threshold2=high,
        L2gradient=True
    )

    method = EDGE_METHOD.lower()
    if method not in {"canny", "morph", "hybrid"}:
        method = "canny"

    morph_thresh = 0
    if method in {"morph", "hybrid"}:
        morph_edges, morph_thresh = morph_gradient_edges(smooth)
    else:
        morph_edges = None

    if method == "morph" and morph_edges is not None:
        edges = morph_edges
    elif method == "hybrid" and morph_edges is not None:
        edges = cv2.bitwise_or(canny_edges, morph_edges)
    else:
        edges = canny_edges

    if ENABLE_STRUCTURE_MASK:
        structure_mask = build_structure_mask(edges.shape)
        edges = cv2.bitwise_and(edges, structure_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITER)

    edges = remove_small_components(edges, MIN_COMPONENT_AREA)

    return gray, enhanced, smooth, grad_mag, edges, low, high, morph_thresh


def build_vis(roi_bgr: np.ndarray, edges: np.ndarray, low: int, high: int, morph_thresh: int) -> np.ndarray:
    vis = roi_bgr.copy()

    # 红色叠加边缘
    vis[edges > 0] = (0, 0, 255)

    cv2.putText(
        vis,
        f"{EDGE_METHOD}  canny={low}/{high} morph={morph_thresh}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        f"HIGH_PCT={HIGH_PCT} LOW_RATIO={LOW_RATIO:.2f} MASK={int(ENABLE_STRUCTURE_MASK)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    return vis
