from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "dataset" / "live" / "latest_roi.jpg"
OUTPUT_PATH = ROOT_DIR / "dataset" / "live" / "latest_roi_enhanced.jpg"
METRICS_PATH = ROOT_DIR / "dataset" / "live" / "latest_roi_enhance_metrics.json"

SAVE_DEBUG = True
DEBUG_DIR = OUTPUT_PATH.parent / "debug_roi_enhance"

PIPELINE_MODE = os.environ.get("VISION_PIPELINE_MODE", "0") == "1"
SHOW_POST_WINDOWS = os.environ.get("VISION_SHOW_POST_WINDOWS", "0") == "1"
KEY_WINDOWS_ONLY = os.environ.get("VISION_KEY_WINDOWS", "0") == "1"
WINDOW_WAIT_MS = max(1, int(os.environ.get("VISION_WINDOW_WAIT_MS", "700")))
SHOW_WINDOW = (not PIPELINE_MODE) or SHOW_POST_WINDOWS
MAX_SHOW_W = 1200
MAX_SHOW_H = 900

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


def atomic_imwrite(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + "_tmp" + path.suffix)

    ok = cv2.imwrite(str(tmp_path), image)
    if not ok:
        return False

    try:
        if path.exists():
            path.unlink()
        tmp_path.replace(path)
        return True
    except OSError:
        return False


def atomic_write_json(path: Path, data: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + "_tmp" + path.suffix)

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if path.exists():
            path.unlink()
        tmp_path.replace(path)
        return True
    except OSError:
        return False


def show_keep_ratio(win_name: str, img: np.ndarray, max_w: int = MAX_SHOW_W, max_h: int = MAX_SHOW_H) -> None:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    show_w = max(1, int(w * scale))
    show_h = max(1, int(h * scale))

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, show_w, show_h)
    cv2.imshow(win_name, img)


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


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Input not found: {INPUT_PATH}")
        return 1

    roi = cv2.imread(str(INPUT_PATH))
    if roi is None or roi.size == 0:
        print("Failed to read ROI image.")
        return 1

    results = preprocess_roi(roi)
    final_img = results["final"]

    if not atomic_imwrite(OUTPUT_PATH, final_img):
        print(f"Failed to save enhanced ROI: {OUTPUT_PATH}")
        return 1

    if not atomic_write_json(METRICS_PATH, results["metrics"]):
        print(f"Failed to save enhance metrics: {METRICS_PATH}")
        return 1

    print("===== ROI Enhance Done =====")
    print(f"Input   : {INPUT_PATH}")
    print(f"Output  : {OUTPUT_PATH}")
    print(f"Metrics : {METRICS_PATH}")
    print(f"Shape   : {roi.shape}")
    print("")
    print("Adaptive params:")
    print(f"  gamma                 = {results['metrics']['gamma']}")
    print(f"  illumination_kernel   = {results['metrics']['illumination_kernel']}")
    print(f"  clahe_clip_limit      = {results['metrics']['clahe_clip_limit']}")
    print(f"  sharpen_amount        = {results['metrics']['sharpen']['amount']}")
    print(f"  input_stats           = {results['metrics']['input']}")
    print(f"  output_stats          = {results['metrics']['output']}")

    if SAVE_DEBUG:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        atomic_imwrite(DEBUG_DIR / "01_gray.jpg", results["gray"])
        atomic_imwrite(DEBUG_DIR / "02_gamma.jpg", results["gamma"])
        atomic_imwrite(DEBUG_DIR / "03_illumination_background.jpg", results["illumination_background"])
        atomic_imwrite(DEBUG_DIR / "04_illumination_normalized.jpg", results["illumination_normalized"])
        atomic_imwrite(DEBUG_DIR / "05_clahe.jpg", results["clahe"])
        atomic_imwrite(DEBUG_DIR / "06_filtered.jpg", results["filtered"])
        atomic_imwrite(DEBUG_DIR / "07_gradient.jpg", results["gradient"])
        atomic_imwrite(DEBUG_DIR / "08_final.jpg", results["final"])
        print(f"Debug saved to: {DEBUG_DIR}")

    if SHOW_WINDOW:
        if KEY_WINDOWS_ONLY:
            show_keep_ratio("enhance_final", results["final"])
        else:
            show_keep_ratio("roi_input", roi)
            show_keep_ratio("01_gray", results["gray"])
            show_keep_ratio("02_gamma", results["gamma"])
            show_keep_ratio("04_illumination_normalized", results["illumination_normalized"])
            show_keep_ratio("05_clahe", results["clahe"])
            show_keep_ratio("06_filtered", results["filtered"])
            show_keep_ratio("07_gradient", results["gradient"])
            show_keep_ratio("08_final", results["final"])
        cv2.waitKey(WINDOW_WAIT_MS)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
