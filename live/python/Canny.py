from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np


# =========================
# 你先改这里
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
ROI_IMAGE_PATH = ROOT_DIR / "dataset" / "live" / "latest_roi_enhanced.jpg"
OUT_DIR = ROI_IMAGE_PATH.parent

EDGES_PATH = OUT_DIR / "latest_edges.jpg"
VIS_PATH = OUT_DIR / "latest_edges_vis.jpg"

SHOW_WINDOW = (
    os.environ.get("VISION_SHOW_CANNY_WINDOW")
    if os.environ.get("VISION_SHOW_CANNY_WINDOW") is not None
    else os.environ.get("VISION_SHOW_BASE_WINDOWS", "1")
) == "1"
KEY_WINDOWS_ONLY = os.environ.get("VISION_KEY_WINDOWS", "0") == "1"
SLEEP_SHORT = 0.05

# 显示窗口最大尺寸（按比例缩放，不拉伸）
MAX_SHOW_W = 900
MAX_SHOW_H = 700

# 预处理参数
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50

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


def atomic_imwrite(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)

    ok = cv2.imwrite(str(tmp), image)
    if not ok:
        return False

    try:
        if path.exists():
            path.unlink()
        tmp.replace(path)
        return True
    except OSError:
        return False


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
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
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


def show_keep_ratio(win_name: str, img: np.ndarray, max_w: int = MAX_SHOW_W, max_h: int = MAX_SHOW_H) -> None:
    """
    按原图比例显示窗口，避免 imshow 窗口被手动/系统拉伸后看起来变形。
    这里只控制显示，不改变图像实际尺寸。
    """
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    show_w = max(1, int(w * scale))
    show_h = max(1, int(h * scale))

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, show_w, show_h)
    cv2.imshow(win_name, img)


def main():
    print("Adaptive Canny watch started.")
    print(f"Watching: {ROI_IMAGE_PATH}")
    print(f"Edges out: {EDGES_PATH}")
    print(f"Vis out  : {VIS_PATH}")
    print("Press ESC to quit.")

    last_mtime = 0.0
    printed_shape = False

    while True:
        if not ROI_IMAGE_PATH.exists():
            time.sleep(SLEEP_SHORT)
            continue

        try:
            stat = ROI_IMAGE_PATH.stat()
            file_size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            time.sleep(SLEEP_SHORT)
            continue

        if file_size <= 0 or mtime == last_mtime:
            time.sleep(SLEEP_SHORT)
            continue

        frame = cv2.imread(str(ROI_IMAGE_PATH))
        if frame is None or frame.size == 0:
            time.sleep(SLEEP_SHORT)
            continue

        last_mtime = mtime

        gray, enhanced, smooth, grad_mag, edges, low, high, morph_thresh = adaptive_canny_pipeline(frame)
        vis = build_vis(frame, edges, low, high, morph_thresh)

        atomic_imwrite(EDGES_PATH, edges)
        atomic_imwrite(VIS_PATH, vis)

        if not printed_shape:
            print("frame.shape    =", frame.shape)
            print("enhanced.shape =", enhanced.shape)
            print("smooth.shape   =", smooth.shape)
            print("grad_mag.shape =", grad_mag.shape)
            print("edges.shape    =", edges.shape)
            printed_shape = True

        if SHOW_WINDOW:
            if KEY_WINDOWS_ONLY:
                show_keep_ratio("edges_vis", vis)
            else:
                show_keep_ratio("roi", frame)
                show_keep_ratio("enhanced", enhanced)
                show_keep_ratio("smooth", smooth)
                show_keep_ratio("grad_mag", grad_mag)
                show_keep_ratio("edges", edges)
                show_keep_ratio("edges_vis", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
