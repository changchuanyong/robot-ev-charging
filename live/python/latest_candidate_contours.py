from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from candidate_types import CandidateContour, LayoutModel
from center_fit import export_fitted_centers, fit_ellipse_on_kept_candidates
from layout_prior import apply_standard_layout_prior, draw_layout_prior, is_duplicate_candidate


ROOT_DIR = Path(__file__).resolve().parents[2]
ROI_IMAGE_PATH = ROOT_DIR / "dataset" / "live" / "latest_roi_enhanced.jpg"
EDGES_IMAGE_PATH = ROOT_DIR / "dataset" / "live" / "latest_edges.jpg"

OUT_DIR = ROI_IMAGE_PATH.parent
CAND_VIS_PATH = OUT_DIR / "latest_candidate_contours.jpg"
CENTER_VIS_PATH = OUT_DIR / "latest_fitted_centers.jpg"
CENTER_JSON_PATH = OUT_DIR / "latest_fitted_centers.json"

SHOW_WINDOW = os.environ.get("VISION_PIPELINE_MODE", "0") != "1"
WAIT_FOR_KEY = os.environ.get("VISION_WAIT_FOR_KEY", "0") == "1"
WINDOW_WAIT_MS = max(1, int(os.environ.get("VISION_WINDOW_WAIT_MS", "700")))
SHOW_RAW_CENTER = True

MIN_CONTOUR_AREA = 60
MAX_CONTOUR_AREA_RATIO = 0.22

MIN_CIRCULARITY = 0.15
MAX_ASPECT_RATIO = 4.00
MIN_FILL_RATIO = 0.05

BORDER_MARGIN_RATIO = 0.03
PRIOR_RX_RATIO = 0.40
PRIOR_RY_RATIO = 0.33
PRIOR_VALUE_MAX = 0.90

DUP_CENTER_DIST = 8
MAX_KEEP = 16

MAX_SHOW_W = 1000
MAX_SHOW_H = 800

PRINT_REJECT_LOG = True


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


def atomic_write_json(path: Path, data) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if path.exists():
            path.unlink()
        tmp.replace(path)
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


def prepare_edges(edges: np.ndarray) -> np.ndarray:
    if len(edges.shape) == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    return binary


def contour_center(cnt: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) > 1e-6:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

    x, y, w, h = cv2.boundingRect(cnt)
    return x + w / 2.0, y + h / 2.0


def is_near_border(bbox: tuple[int, int, int, int], roi_shape: tuple[int, int], margin_ratio: float) -> bool:
    h_img, w_img = roi_shape[:2]
    x, y, w, h = bbox

    mx = int(w_img * margin_ratio)
    my = int(h_img * margin_ratio)

    return (x <= mx) or (y <= my) or (x + w >= w_img - mx) or (y + h >= h_img - my)


def inside_structure_prior(cx: float, cy: float, roi_shape: tuple[int, int]) -> tuple[bool, float]:
    h_img, w_img = roi_shape[:2]
    roi_cx = w_img / 2.0
    roi_cy = h_img / 2.0

    rx = PRIOR_RX_RATIO * w_img
    ry = PRIOR_RY_RATIO * h_img

    dx = cx - roi_cx
    dy = cy - roi_cy

    value = (dx * dx) / (rx * rx + 1e-6) + (dy * dy) / (ry * ry + 1e-6)
    return value <= 1.0, value


def compute_candidate_features(cnt: np.ndarray, roi_shape: tuple[int, int]) -> CandidateContour | None:
    area = float(cv2.contourArea(cnt))
    if area <= 1:
        return None

    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter <= 1:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 1 or h <= 1:
        return None

    cx, cy = contour_center(cnt)

    circularity = 4.0 * math.pi * area / (perimeter * perimeter + 1e-6)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    fill_ratio = area / (w * h + 1e-6)

    inside, prior_value = inside_structure_prior(cx, cy, roi_shape)

    roi_h, roi_w = roi_shape[:2]
    roi_area = roi_h * roi_w
    area_norm = area / (roi_area + 1e-6)

    if area_norm < 0.0015:
        area_score = 0.05
    elif area_norm < 0.004:
        area_score = 0.50
    elif area_norm < 0.035:
        area_score = 1.00
    else:
        area_score = 0.50

    center_score = max(0.0, min(1.0, 1.0 - prior_value))
    circularity_score = max(0.0, min(1.0, circularity))
    aspect_score = max(0.0, min(1.0, 1.0 / aspect_ratio))
    fill_score = max(0.0, min(1.0, fill_ratio))

    score = (
        0.25 * circularity_score +
        0.10 * aspect_score +
        0.10 * fill_score +
        0.30 * center_score +
        0.25 * area_score
    )

    return CandidateContour(
        contour=cnt,
        center=(cx, cy),
        bbox=(x, y, w, h),
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        fill_ratio=fill_ratio,
        prior_value=prior_value,
        inside_prior=inside,
        score=score,
    )


def reject_log(reason: str, feat: CandidateContour) -> None:
    if not PRINT_REJECT_LOG:
        return

    print(
        f"reject={reason:>16s}  "
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"area={feat.area:7.1f}  "
        f"circ={feat.circularity:5.3f}  "
        f"aspect={feat.aspect_ratio:4.2f}  "
        f"fill={feat.fill_ratio:4.2f}  "
        f"prior={feat.prior_value:4.2f}  "
        f"score={feat.score:5.3f}"
    )


def keep_log(tag: str, feat: CandidateContour) -> None:
    print(
        f"{tag:<20s}"
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"area={feat.area:7.1f}  "
        f"circ={feat.circularity:5.3f}  "
        f"aspect={feat.aspect_ratio:4.2f}  "
        f"fill={feat.fill_ratio:4.2f}  "
        f"prior={feat.prior_value:4.2f}  "
        f"score={feat.score:5.3f}"
    )


def filter_candidate_contours(
    edges: np.ndarray,
    roi_bgr: np.ndarray
) -> tuple[list[CandidateContour], list[np.ndarray], LayoutModel | None]:
    h_img, w_img = roi_bgr.shape[:2]
    roi_area = h_img * w_img
    max_area = MAX_CONTOUR_AREA_RATIO * roi_area

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    raw_candidates: list[CandidateContour] = []

    print("===== Stage 1&2: geometry + coarse prior filtering =====")
    for cnt in contours:
        feat = compute_candidate_features(cnt, roi_bgr.shape)
        if feat is None:
            continue

        if feat.area < MIN_CONTOUR_AREA:
            reject_log("area_small", feat)
            continue

        if feat.area > max_area:
            reject_log("area_large", feat)
            continue

        if feat.circularity < MIN_CIRCULARITY:
            reject_log("circularity", feat)
            continue

        if feat.aspect_ratio > MAX_ASPECT_RATIO:
            reject_log("aspect", feat)
            continue

        if feat.fill_ratio < MIN_FILL_RATIO:
            reject_log("fill_ratio", feat)
            continue

        if is_near_border(feat.bbox, roi_bgr.shape, BORDER_MARGIN_RATIO):
            reject_log("near_border", feat)
            continue

        if not feat.inside_prior:
            reject_log("outside_prior", feat)
            continue

        if feat.prior_value > PRIOR_VALUE_MAX:
            reject_log("prior_too_far", feat)
            continue

        raw_candidates.append(feat)
        keep_log("keep_raw", feat)

    raw_candidates.sort(key=lambda c: c.score, reverse=True)

    print("\n===== Stage 3: duplicate removal =====")
    kept: list[CandidateContour] = []
    for cand in raw_candidates:
        duplicated = False
        for old in kept:
            if is_duplicate_candidate(cand, old, DUP_CENTER_DIST):
                print(
                    f"reject=duplicate       "
                    f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                    f"score={cand.score:5.3f}  "
                    f"close_to=({old.center[0]:6.1f},{old.center[1]:6.1f})"
                )
                duplicated = True
                break

        if not duplicated:
            kept.append(cand)
            keep_log("keep_final_dup", cand)

        if len(kept) >= MAX_KEEP:
            break

    kept_layout, layout_model = apply_standard_layout_prior(kept, roi_bgr.shape)
    return kept_layout, contours, layout_model


def build_candidate_vis(
    roi_bgr: np.ndarray,
    all_contours: list[np.ndarray],
    kept_candidates: list[CandidateContour],
    layout_model: LayoutModel | None,
) -> np.ndarray:
    vis = roi_bgr.copy()
    h_img, w_img = roi_bgr.shape[:2]

    cv2.drawContours(vis, all_contours, -1, (160, 160, 160), 1)

    center = (int(w_img / 2), int(h_img / 2))
    axes = (int(PRIOR_RX_RATIO * w_img), int(PRIOR_RY_RATIO * h_img))
    cv2.ellipse(vis, center, axes, 0, 0, 360, (255, 255, 0), 1)
    cv2.circle(vis, center, 3, (255, 255, 0), -1)

    if layout_model is not None:
        draw_layout_prior(vis, layout_model)

    for i, cand in enumerate(kept_candidates):
        if cand.is_main_anchor:
            contour_color = (0, 165, 255)
            center_color = (0, 0, 255)
            box_color = (0, 165, 255)
        elif cand.is_top_backup:
            contour_color = (255, 180, 0)
            center_color = (0, 0, 255)
            box_color = (255, 180, 0)
        else:
            contour_color = (0, 255, 0)
            center_color = (0, 0, 255)
            box_color = (255, 0, 0)

        cv2.drawContours(vis, [cand.contour], -1, contour_color, 2)

        cx = int(round(cand.center[0]))
        cy = int(round(cand.center[1]))
        cv2.circle(vis, (cx, cy), 3, center_color, -1)

        x, y, w, h = cand.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, 1)

        if cand.is_main_anchor:
            text = f"{i}: {cand.layout_name} s={cand.score:.2f}"
        elif cand.is_top_backup:
            text = f"{i}: top_backup d={cand.layout_dist:.2f} s={cand.layout_score:.2f}"
        else:
            text = f"{i}: {cand.layout_name} d={cand.layout_dist:.2f} s={cand.layout_score:.2f}"

        cv2.putText(
            vis,
            text,
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
        )

    return vis


def build_center_vis(
    roi_bgr: np.ndarray,
    fitted_candidates: list[CandidateContour],
) -> np.ndarray:
    vis = roi_bgr.copy()

    for i, cand in enumerate(fitted_candidates):
        if cand.is_main_anchor:
            contour_color = (0, 165, 255)
            box_color = (0, 165, 255)
        elif cand.is_top_backup:
            contour_color = (255, 180, 0)
            box_color = (255, 180, 0)
        else:
            contour_color = (0, 255, 0)
            box_color = (255, 0, 0)

        cv2.drawContours(vis, [cand.contour], -1, contour_color, 2)

        x, y, w, h = cand.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, 1)

        if SHOW_RAW_CENTER:
            raw_cx = int(round(cand.center[0]))
            raw_cy = int(round(cand.center[1]))
            cv2.circle(vis, (raw_cx, raw_cy), 3, (255, 0, 255), -1)

        if cand.fitted_center is not None:
            fit_cx = int(round(cand.fitted_center[0]))
            fit_cy = int(round(cand.fitted_center[1]))
            cv2.circle(vis, (fit_cx, fit_cy), 4, (0, 0, 255), -1)

            if SHOW_RAW_CENTER:
                cv2.line(vis, (raw_cx, raw_cy), (fit_cx, fit_cy), (255, 0, 255), 1)

        if cand.fit_method == "ellipse" and cand.fitted_center is not None and cand.ellipse_axes is not None and cand.ellipse_angle is not None:
            ellipse = (
                (float(cand.fitted_center[0]), float(cand.fitted_center[1])),
                (float(cand.ellipse_axes[0]), float(cand.ellipse_axes[1])),
                float(cand.ellipse_angle),
            )
            cv2.ellipse(vis, ellipse, (0, 255, 255), 2)

        text = f"{i}: {cand.fit_method}"
        cv2.putText(
            vis,
            text,
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    return vis


def main():
    if not ROI_IMAGE_PATH.exists():
        print(f"ROI not found: {ROI_IMAGE_PATH}")
        return

    if not EDGES_IMAGE_PATH.exists():
        print(f"Edges not found: {EDGES_IMAGE_PATH}")
        return

    roi = cv2.imread(str(ROI_IMAGE_PATH))
    edges = cv2.imread(str(EDGES_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    if roi is None or roi.size == 0:
        print("Failed to read ROI image.")
        return

    if edges is None or edges.size == 0:
        print("Failed to read edges image.")
        return

    edges_bin = prepare_edges(edges)

    print("===== Input =====")
    print(f"ROI path        : {ROI_IMAGE_PATH}")
    print(f"Edges path      : {EDGES_IMAGE_PATH}")
    print(f"Candidate vis   : {CAND_VIS_PATH}")
    print(f"Center vis      : {CENTER_VIS_PATH}")
    print(f"Center json     : {CENTER_JSON_PATH}")
    print(f"ROI shape       : {roi.shape}")
    print(f"Edges shape     : {edges_bin.shape}\n")

    candidates, all_contours, layout_model = filter_candidate_contours(edges_bin, roi)

    cand_vis = build_candidate_vis(roi, all_contours, candidates, layout_model)
    atomic_imwrite(CAND_VIS_PATH, cand_vis)

    fitted_candidates = fit_ellipse_on_kept_candidates(candidates)

    center_vis = build_center_vis(roi, fitted_candidates)
    atomic_imwrite(CENTER_VIS_PATH, center_vis)

    center_data = export_fitted_centers(fitted_candidates)
    atomic_write_json(CENTER_JSON_PATH, center_data)

    print("\n===== Final kept candidates =====")
    print(f"Kept candidates: {len(candidates)}")
    for i, c in enumerate(candidates):
        if c.is_main_anchor:
            print(
                f"[{i}] "
                f"center=({c.center[0]:.1f}, {c.center[1]:.1f})  "
                f"area={c.area:.1f}  "
                f"anchor={c.layout_name}  "
                f"raw_score={c.score:.3f}"
            )
        else:
            extra = " top_backup" if c.is_top_backup else ""
            print(
                f"[{i}] "
                f"center=({c.center[0]:.1f}, {c.center[1]:.1f})  "
                f"area={c.area:.1f}  "
                f"layout={c.layout_name}{extra}  "
                f"uv=({c.layout_u:.2f}, {c.layout_v:.2f})  "
                f"dist={c.layout_dist:.2f}/{c.layout_accept_dist:.2f}  "
                f"raw={c.score:.3f}  "
                f"final={c.layout_score:.3f}"
            )

    print("\n===== Final fitted centers =====")
    for i, c in enumerate(fitted_candidates):
        if c.fitted_center is None:
            continue

        print(
            f"[{i}] "
            f"raw_center=({c.center[0]:.1f}, {c.center[1]:.1f})  "
            f"fit_center=({c.fitted_center[0]:.1f}, {c.fitted_center[1]:.1f})  "
            f"method={c.fit_method}"
        )

    print(f"\nSaved candidate vis: {CAND_VIS_PATH}")
    print(f"Saved center vis   : {CENTER_VIS_PATH}")
    print(f"Saved center json  : {CENTER_JSON_PATH}")

    if SHOW_WINDOW:
        show_keep_ratio("roi", roi)
        show_keep_ratio("edges", edges_bin)
        show_keep_ratio("candidate_contours", cand_vis)
        show_keep_ratio("fitted_centers", center_vis)
        if WAIT_FOR_KEY:
            print("Press any key to exit...")
            cv2.waitKey(0)
        else:
            print(f"Windows close automatically after {WINDOW_WAIT_MS} ms.")
            cv2.waitKey(WINDOW_WAIT_MS)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
