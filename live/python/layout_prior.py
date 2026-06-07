from __future__ import annotations

import math

import cv2
import numpy as np

from candidate_types import CandidateContour, LayoutModel


ENABLE_LAYOUT_PRIOR = True

MAIN_PAIR_MIN_DIST_RATIO = 0.18
MAIN_PAIR_MIN_HORIZONTAL = 0.55
MAIN_PAIR_MAX_LEVEL_DIFF_RATIO = 0.45
MAIN_PAIR_MIN_AREA_SIM = 0.18

LAYOUT_SCORE_WEIGHT = 0.55
RAW_SCORE_WEIGHT = 0.45

ENABLE_TOP_BACKUP = True
TOP_BACKUP_MIN_RAW_SCORE = 0.30
TOP_BACKUP_MAX_LAYOUT_DIST = 1.95
TOP_BACKUP_MAX_COUNT = 1
TOP_BACKUP_MAX_ABS_U = 0.70
TOP_BACKUP_MIN_NEG_V = -0.22

ENABLE_LAYOUT_AREA_PRIOR = True
LAYOUT_AREA_PRIOR = {
    "top_mid": (0.0010, 0.0200),
    "center": (0.0010, 0.0600),
    "bottom_left": (0.0010, 0.0700),
    "bottom_right": (0.0010, 0.0700),
    "top_left": (0.0100, 0.0900),
    "top_right": (0.0100, 0.0900),
    "bottom_mid": (0.0300, 0.2400),
}

STANDARD_LAYOUT_ZONES = [
    # name,         u,      v,      tol_u, tol_v, group,   accept_dist
    ("top_left",    -0.30, -0.56,   0.34,  0.30, "top",    1.55),
    ("top_mid",      0.00, -0.54,   0.30,  0.28, "top",    1.48),
    ("top_right",    0.30, -0.56,   0.34,  0.30, "top",    1.55),

    ("center",       0.00, -0.20,   0.18,  0.18, "mid",    1.15),

    ("bottom_left", -0.28,  0.44,   0.24,  0.22, "bottom", 1.22),
    ("bottom_mid",   0.00,  0.62,   0.22,  0.22, "bottom", 1.25),
    ("bottom_right", 0.28,  0.44,   0.24,  0.22, "bottom", 1.22),
]


def is_duplicate_candidate(c1: CandidateContour, c2: CandidateContour, center_dist_thresh: float) -> bool:
    d = math.hypot(c1.center[0] - c2.center[0], c1.center[1] - c2.center[1])
    return d < center_dist_thresh


def layout_reject_log(reason: str, feat: CandidateContour) -> None:
    print(
        f"reject={reason:>16s}  "
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"uv=({feat.layout_u:5.2f},{feat.layout_v:5.2f})  "
        f"layout={feat.layout_name:<12s}  "
        f"group={feat.layout_group:<7s}  "
        f"dist={feat.layout_dist:5.2f}/{feat.layout_accept_dist:4.2f}  "
        f"raw={feat.score:5.3f}  "
        f"final={feat.layout_score:5.3f}"
    )


def layout_keep_log(tag: str, feat: CandidateContour) -> None:
    print(
        f"{tag:<20s}"
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"uv=({feat.layout_u:5.2f},{feat.layout_v:5.2f})  "
        f"layout={feat.layout_name:<12s}  "
        f"group={feat.layout_group:<7s}  "
        f"dist={feat.layout_dist:5.2f}/{feat.layout_accept_dist:4.2f}  "
        f"raw={feat.score:5.3f}  "
        f"final={feat.layout_score:5.3f}"
    )


def find_main_big_hole_pair(candidates: list[CandidateContour], roi_shape: tuple[int, int]) -> tuple[int, int] | None:
    if len(candidates) < 2:
        return None

    h_img, w_img = roi_shape[:2]
    roi_area = h_img * w_img
    min_dim = min(h_img, w_img)

    best_pair = None
    best_score = -1.0

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1 = candidates[i]
            c2 = candidates[j]

            dx = c2.center[0] - c1.center[0]
            dy = c2.center[1] - c1.center[1]
            dist = math.hypot(dx, dy)
            if dist < MAIN_PAIR_MIN_DIST_RATIO * min_dim:
                continue

            horizontal = abs(dx) / (dist + 1e-6)
            if horizontal < MAIN_PAIR_MIN_HORIZONTAL:
                continue

            level_diff_ratio = abs(dy) / (dist + 1e-6)
            if level_diff_ratio > MAIN_PAIR_MAX_LEVEL_DIFF_RATIO:
                continue

            area_sim = min(c1.area, c2.area) / (max(c1.area, c2.area) + 1e-6)
            if area_sim < MAIN_PAIR_MIN_AREA_SIM:
                continue

            area_sum_norm = min(1.0, (c1.area + c2.area) / (0.05 * roi_area + 1e-6))
            center_prior = 1.0 - min(1.0, (c1.prior_value + c2.prior_value) / 2.0)

            pair_score = (
                0.45 * area_sum_norm +
                0.20 * area_sim +
                0.20 * horizontal +
                0.10 * (1.0 - level_diff_ratio) +
                0.05 * center_prior
            )

            if pair_score > best_score:
                best_score = pair_score
                best_pair = (i, j)

    return best_pair


def build_layout_model(left_anchor: CandidateContour, right_anchor: CandidateContour) -> LayoutModel | None:
    p1 = np.array(left_anchor.center, dtype=np.float32)
    p2 = np.array(right_anchor.center, dtype=np.float32)

    vec = p2 - p1
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return None

    ux = vec / dist
    uy = np.array([-ux[1], ux[0]], dtype=np.float32)

    origin = (p1 + p2) / 2.0

    return LayoutModel(
        origin=(float(origin[0]), float(origin[1])),
        ux=(float(ux[0]), float(ux[1])),
        uy=(float(uy[0]), float(uy[1])),
        scale=dist,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
    )


def project_candidate_to_layout(cand: CandidateContour, model: LayoutModel) -> tuple[float, float]:
    p = np.array(cand.center, dtype=np.float32)
    origin = np.array(model.origin, dtype=np.float32)
    ux = np.array(model.ux, dtype=np.float32)
    uy = np.array(model.uy, dtype=np.float32)

    delta = p - origin
    u = float(np.dot(delta, ux) / (model.scale + 1e-6))
    v = float(np.dot(delta, uy) / (model.scale + 1e-6))
    return u, v


def layout_uv_to_xy(u: float, v: float, model: LayoutModel) -> tuple[int, int]:
    origin = np.array(model.origin, dtype=np.float32)
    ux = np.array(model.ux, dtype=np.float32)
    uy = np.array(model.uy, dtype=np.float32)
    p = origin + model.scale * (u * ux + v * uy)
    return int(round(float(p[0]))), int(round(float(p[1])))


def match_candidate_to_layout_zone(cand: CandidateContour, model: LayoutModel) -> CandidateContour:
    u, v = project_candidate_to_layout(cand, model)
    cand.layout_u = u
    cand.layout_v = v

    best_name = ""
    best_group = ""
    best_dist = 999.0
    best_accept = 999.0

    for zone_name, z_u, z_v, tol_u, tol_v, zone_group, accept_dist in STANDARD_LAYOUT_ZONES:
        du = (u - z_u) / (tol_u + 1e-6)
        dv = (v - z_v) / (tol_v + 1e-6)
        d = math.hypot(du, dv)

        if d < best_dist:
            best_dist = d
            best_name = zone_name
            best_group = zone_group
            best_accept = accept_dist

    cand.layout_name = best_name
    cand.layout_group = best_group
    cand.layout_dist = best_dist
    cand.layout_accept_dist = best_accept

    closeness = max(0.0, 1.0 - best_dist / (best_accept + 1e-6))
    cand.layout_score = LAYOUT_SCORE_WEIGHT * closeness + RAW_SCORE_WEIGHT * cand.score
    return cand


def layout_area_prior_ok(cand: CandidateContour, model: LayoutModel) -> bool:
    if not ENABLE_LAYOUT_AREA_PRIOR:
        return True

    if cand.layout_name not in LAYOUT_AREA_PRIOR:
        return True

    min_ratio, max_ratio = LAYOUT_AREA_PRIOR[cand.layout_name]
    area_ratio = cand.area / (model.scale * model.scale + 1e-6)
    return min_ratio <= area_ratio <= max_ratio


def apply_standard_layout_prior(
    candidates: list[CandidateContour],
    roi_shape: tuple[int, int]
) -> tuple[list[CandidateContour], LayoutModel | None]:
    if not ENABLE_LAYOUT_PRIOR:
        return candidates, None

    if len(candidates) < 2:
        return candidates, None

    print("\n===== Stage 4: standard layout prior filtering =====")

    pair_idx = find_main_big_hole_pair(candidates, roi_shape)
    if pair_idx is None:
        print("No valid main-hole pair found. Skip layout prior filtering.")
        return candidates, None

    c1 = candidates[pair_idx[0]]
    c2 = candidates[pair_idx[1]]

    if c1.center[0] <= c2.center[0]:
        left_anchor, right_anchor = c1, c2
    else:
        left_anchor, right_anchor = c2, c1

    model = build_layout_model(left_anchor, right_anchor)
    if model is None:
        print("Failed to build layout model. Skip layout prior filtering.")
        return candidates, None

    left_anchor.is_main_anchor = True
    right_anchor.is_main_anchor = True
    left_anchor.layout_name = "main_left"
    right_anchor.layout_name = "main_right"
    left_anchor.layout_group = "anchor"
    right_anchor.layout_group = "anchor"
    left_anchor.layout_u, left_anchor.layout_v = -0.5, 0.0
    right_anchor.layout_u, right_anchor.layout_v = 0.5, 0.0
    left_anchor.layout_dist = 0.0
    right_anchor.layout_dist = 0.0
    left_anchor.layout_score = left_anchor.score
    right_anchor.layout_score = right_anchor.score
    left_anchor.layout_accept_dist = 0.0
    right_anchor.layout_accept_dist = 0.0

    print(
        f"main_pair left=({left_anchor.center[0]:.1f}, {left_anchor.center[1]:.1f})  "
        f"right=({right_anchor.center[0]:.1f}, {right_anchor.center[1]:.1f})  "
        f"dist={model.scale:.1f}"
    )

    zone_best: dict[str, CandidateContour] = {}
    top_backups: list[CandidateContour] = []

    others = [c for c in candidates if c is not left_anchor and c is not right_anchor]

    for cand in others:
        match_candidate_to_layout_zone(cand, model)

        if cand.layout_dist <= cand.layout_accept_dist:
            if not layout_area_prior_ok(cand, model):
                layout_reject_log("layout_area", cand)
                continue

            old = zone_best.get(cand.layout_name)
            if old is None:
                zone_best[cand.layout_name] = cand
                layout_keep_log("keep_layout_raw", cand)
            else:
                if cand.layout_score > old.layout_score:
                    layout_reject_log("layout_replaced", old)
                    zone_best[cand.layout_name] = cand
                    layout_keep_log("keep_layout_best", cand)
                else:
                    layout_reject_log("layout_weaker", cand)
            continue

        if (
            ENABLE_TOP_BACKUP
            and cand.layout_group == "top"
            and cand.layout_v < TOP_BACKUP_MIN_NEG_V
            and abs(cand.layout_u) < TOP_BACKUP_MAX_ABS_U
            and cand.score >= TOP_BACKUP_MIN_RAW_SCORE
            and cand.layout_dist <= TOP_BACKUP_MAX_LAYOUT_DIST
        ):
            if not layout_area_prior_ok(cand, model):
                layout_reject_log("layout_area", cand)
                continue

            cand.is_top_backup = True
            cand.layout_score = 0.20 + 0.80 * cand.score
            top_backups.append(cand)
            layout_keep_log("keep_top_backup", cand)
            continue

        layout_reject_log("layout_outlier", cand)

    final_kept: list[CandidateContour] = [left_anchor, right_anchor]

    zone_order = [z[0] for z in STANDARD_LAYOUT_ZONES]
    for name in zone_order:
        if name in zone_best:
            final_kept.append(zone_best[name])

    kept_top_count = sum(1 for c in final_kept if c.layout_group == "top")

    if kept_top_count < 2 and len(top_backups) > 0:
        top_backups.sort(key=lambda c: c.layout_score, reverse=True)

        added = 0
        for cand in top_backups:
            too_close = False
            for old in final_kept:
                if is_duplicate_candidate(cand, old, 8):
                    too_close = True
                    break

            if too_close:
                continue

            final_kept.append(cand)
            added += 1
            if added >= TOP_BACKUP_MAX_COUNT:
                break

    def sort_key(c: CandidateContour):
        if c.is_main_anchor:
            return (-10.0, c.center[0])
        return (c.layout_v, c.layout_u)

    final_kept.sort(key=sort_key)

    print("\n===== Layout final kept =====")
    for cand in final_kept:
        if cand.is_main_anchor:
            print(
                f"anchor                center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"layout={cand.layout_name:<12s} raw={cand.score:5.3f}"
            )
        else:
            extra = " backup" if cand.is_top_backup else ""
            print(
                f"layout_keep{extra:<7s}  "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"uv=({cand.layout_u:5.2f},{cand.layout_v:5.2f})  "
                f"layout={cand.layout_name:<12s}  "
                f"group={cand.layout_group:<7s}  "
                f"dist={cand.layout_dist:5.2f}/{cand.layout_accept_dist:4.2f}  "
                f"raw={cand.score:5.3f}  "
                f"final={cand.layout_score:5.3f}"
            )

    return final_kept, model


def draw_layout_prior(vis: np.ndarray, model: LayoutModel) -> None:
    p1 = (int(round(model.left_anchor.center[0])), int(round(model.left_anchor.center[1])))
    p2 = (int(round(model.right_anchor.center[0])), int(round(model.right_anchor.center[1])))
    cv2.line(vis, p1, p2, (0, 255, 255), 1)

    oc = (int(round(model.origin[0])), int(round(model.origin[1])))
    cv2.circle(vis, oc, 3, (255, 255, 0), -1)

    for zone_name, z_u, z_v, tol_u, tol_v, zone_group, accept_dist in STANDARD_LAYOUT_ZONES:
        pt = layout_uv_to_xy(z_u, z_v, model)
        cv2.circle(vis, pt, 4, (255, 0, 255), 1)
        cv2.putText(
            vis,
            zone_name,
            (pt[0] + 4, pt[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 255),
            1,
        )

        pt1 = layout_uv_to_xy(z_u - tol_u, z_v - tol_v, model)
        pt2 = layout_uv_to_xy(z_u + tol_u, z_v + tol_v, model)
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(
            vis,
            (min(x1, x2), min(y1, y2)),
            (max(x1, x2), max(y1, y2)),
            (200, 80, 200),
            1,
        )
