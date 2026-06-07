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
MISSING_ZONE_RECOVERY_ACCEPT_SCALE = 1.00
MISSING_ZONE_RECOVERY_NAMES = {"top_mid", "bottom_mid"}
MAX_MAIN_PAIR_HYPOTHESES = 18
ANCHOR_DUP_DIST = 8
SIDE_ZONE_MIN_ABS_U = 0.14
MID_ZONE_MAX_ABS_U = 0.24
MAIN_PAIR_RELATIVE_SCORE_MIN = 0.78
GLOBAL_ASSIGNMENT_MIN_COVERAGE_GAIN = 3

ENABLE_TOP_BACKUP = True
TOP_BACKUP_MIN_RAW_SCORE = 0.30
TOP_BACKUP_MAX_LAYOUT_DIST = 1.95
TOP_BACKUP_MAX_COUNT = 1
TOP_BACKUP_MAX_ABS_U = 0.70
TOP_BACKUP_MIN_NEG_V = -0.22

ENABLE_LAYOUT_AREA_PRIOR = True
LAYOUT_AREA_PRIOR = {
    "top_mid": (0.0003, 0.0500),
    "center": (0.0003, 0.0800),
    "bottom_left": (0.0003, 0.0900),
    "bottom_right": (0.0003, 0.0900),
    "top_left": (0.0005, 0.1100),
    "top_right": (0.0005, 0.1100),
    "bottom_mid": (0.0003, 0.3600),
}

STANDARD_LAYOUT_ZONES = [
    # name,         u,      v,      tol_u, tol_v, group,   accept_dist
    ("top_left",    -0.30, -0.56,   0.38,  0.32, "top",    1.75),
    ("top_mid",      0.00, -0.54,   0.32,  0.30, "top",    1.62),
    ("top_right",    0.30, -0.56,   0.38,  0.32, "top",    1.75),

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


def score_main_big_hole_pair(
    c1: CandidateContour,
    c2: CandidateContour,
    roi_shape: tuple[int, int],
) -> float | None:
    h_img, w_img = roi_shape[:2]
    roi_area = h_img * w_img
    min_dim = min(h_img, w_img)

    dx = c2.center[0] - c1.center[0]
    dy = c2.center[1] - c1.center[1]
    dist = math.hypot(dx, dy)
    if dist < MAIN_PAIR_MIN_DIST_RATIO * min_dim:
        return None

    horizontal = abs(dx) / (dist + 1e-6)
    if horizontal < MAIN_PAIR_MIN_HORIZONTAL:
        return None

    level_diff_ratio = abs(dy) / (dist + 1e-6)
    if level_diff_ratio > MAIN_PAIR_MAX_LEVEL_DIFF_RATIO:
        return None

    area_sim = min(c1.area, c2.area) / (max(c1.area, c2.area) + 1e-6)
    if area_sim < MAIN_PAIR_MIN_AREA_SIM:
        return None

    area_sum_norm = min(1.0, (c1.area + c2.area) / (0.05 * roi_area + 1e-6))
    center_prior = 1.0 - min(1.0, (c1.prior_value + c2.prior_value) / 2.0)

    return (
        0.45 * area_sum_norm +
        0.20 * area_sim +
        0.20 * horizontal +
        0.10 * (1.0 - level_diff_ratio) +
        0.05 * center_prior
    )


def find_main_big_hole_pair_hypotheses(
    candidates: list[CandidateContour],
    roi_shape: tuple[int, int],
) -> list[tuple[float, int, int]]:
    if len(candidates) < 2:
        return []

    pair_scores: list[tuple[float, int, int]] = []

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            score = score_main_big_hole_pair(candidates[i], candidates[j], roi_shape)
            if score is not None:
                pair_scores.append((score, i, j))

    pair_scores.sort(key=lambda item: item[0], reverse=True)
    return pair_scores[:MAX_MAIN_PAIR_HYPOTHESES]


def find_main_big_hole_pair(candidates: list[CandidateContour], roi_shape: tuple[int, int]) -> tuple[int, int] | None:
    hypotheses = find_main_big_hole_pair_hypotheses(candidates, roi_shape)
    if not hypotheses:
        return None
    _score, i, j = hypotheses[0]
    return i, j


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


def layout_area_prior_ok_for_zone(cand: CandidateContour, model: LayoutModel, zone_name: str) -> bool:
    if not ENABLE_LAYOUT_AREA_PRIOR:
        return True

    if zone_name not in LAYOUT_AREA_PRIOR:
        return True

    min_ratio, max_ratio = LAYOUT_AREA_PRIOR[zone_name]
    area_ratio = cand.area / (model.scale * model.scale + 1e-6)
    return min_ratio <= area_ratio <= max_ratio


def layout_position_prior_ok(zone_name: str, u: float) -> bool:
    if zone_name.endswith("_left"):
        return u <= -SIDE_ZONE_MIN_ABS_U
    if zone_name.endswith("_right"):
        return u >= SIDE_ZONE_MIN_ABS_U
    if zone_name in {"top_mid", "bottom_mid", "center"}:
        return abs(u) <= MID_ZONE_MAX_ABS_U
    return True


def score_candidate_for_zone(
    cand: CandidateContour,
    zone: tuple[str, float, float, float, float, str, float],
    model: LayoutModel,
    accept_scale: float = 1.0,
) -> tuple[bool, float, float, float, float, str]:
    zone_name, z_u, z_v, tol_u, tol_v, zone_group, accept_dist = zone
    u, v = project_candidate_to_layout(cand, model)
    du = (u - z_u) / (tol_u + 1e-6)
    dv = (v - z_v) / (tol_v + 1e-6)
    dist = math.hypot(du, dv)
    scaled_accept = accept_dist * accept_scale
    if dist > scaled_accept:
        return False, u, v, dist, scaled_accept, zone_group
    if not layout_position_prior_ok(zone_name, u):
        return False, u, v, dist, scaled_accept, zone_group

    old_layout = (
        cand.layout_name,
        cand.layout_group,
        cand.layout_u,
        cand.layout_v,
        cand.layout_dist,
        cand.layout_accept_dist,
    )
    cand.layout_name = zone_name
    cand.layout_group = zone_group
    cand.layout_u = u
    cand.layout_v = v
    cand.layout_dist = dist
    cand.layout_accept_dist = scaled_accept
    area_ok = layout_area_prior_ok(cand, model)
    (
        cand.layout_name,
        cand.layout_group,
        cand.layout_u,
        cand.layout_v,
        cand.layout_dist,
        cand.layout_accept_dist,
    ) = old_layout
    return area_ok, u, v, dist, scaled_accept, zone_group


def assign_candidate_to_zone(
    cand: CandidateContour,
    zone_name: str,
    zone_group: str,
    u: float,
    v: float,
    dist: float,
    accept_dist: float,
) -> None:
    cand.layout_name = zone_name
    cand.layout_group = zone_group
    cand.layout_u = u
    cand.layout_v = v
    cand.layout_dist = dist
    cand.layout_accept_dist = accept_dist
    closeness = max(0.0, 1.0 - dist / (accept_dist + 1e-6))
    cand.layout_score = LAYOUT_SCORE_WEIGHT * closeness + RAW_SCORE_WEIGHT * cand.score


def build_global_layout_assignment(
    candidates: list[CandidateContour],
    left_anchor: CandidateContour,
    right_anchor: CandidateContour,
    model: LayoutModel,
) -> tuple[dict[str, tuple[CandidateContour, float, float, float, float, str, float]], float, int]:
    matches: list[tuple[float, str, CandidateContour, float, float, float, float, str]] = []

    for cand in candidates:
        if cand is left_anchor or cand is right_anchor:
            continue
        if is_duplicate_candidate(cand, left_anchor, ANCHOR_DUP_DIST):
            continue
        if is_duplicate_candidate(cand, right_anchor, ANCHOR_DUP_DIST):
            continue

        for zone in STANDARD_LAYOUT_ZONES:
            zone_name, _z_u, _z_v, _tol_u, _tol_v, zone_group, _accept_dist = zone
            area_ok, u, v, dist, accept_dist, _zone_group = score_candidate_for_zone(
                cand,
                zone,
                model,
                accept_scale=1.0,
            )
            if not area_ok:
                continue

            if not layout_area_prior_ok_for_zone(cand, model, zone_name):
                continue

            closeness = max(0.0, 1.0 - dist / (accept_dist + 1e-6))
            score = LAYOUT_SCORE_WEIGHT * closeness + RAW_SCORE_WEIGHT * cand.score

            # Prefer true slot proximity over a high raw contour score when assigning across neighboring zones.
            score += 0.12 * closeness
            matches.append((score, zone_name, cand, u, v, dist, accept_dist, zone_group))

    matches.sort(key=lambda item: item[0], reverse=True)

    assignment: dict[str, tuple[CandidateContour, float, float, float, float, str, float]] = {}
    assigned_candidates: list[CandidateContour] = []

    for score, zone_name, cand, u, v, dist, accept_dist, zone_group in matches:
        if zone_name in assignment:
            continue
        if any(is_duplicate_candidate(cand, old, ANCHOR_DUP_DIST) for old in assigned_candidates):
            continue

        assignment[zone_name] = (cand, u, v, dist, accept_dist, zone_group, score)
        assigned_candidates.append(cand)

    coverage = len(assignment)
    top_count = sum(1 for name in ("top_left", "top_mid", "top_right") if name in assignment)
    bottom_count = sum(1 for name in ("bottom_left", "bottom_mid", "bottom_right") if name in assignment)
    mid_count = 1 if "center" in assignment else 0
    small_mid_bonus = (1 if "top_mid" in assignment else 0) + (1 if "bottom_mid" in assignment else 0)

    assignment_score = sum(item[6] for item in assignment.values())
    coverage_score = 2.4 * coverage
    structure_score = 0.55 * min(top_count, 3) + 0.55 * min(bottom_count, 3) + 0.35 * mid_count
    small_mid_score = 0.45 * small_mid_bonus
    total_score = coverage_score + assignment_score + structure_score + small_mid_score

    return assignment, total_score, coverage


def build_nearest_layout_assignment(
    candidates: list[CandidateContour],
    left_anchor: CandidateContour,
    right_anchor: CandidateContour,
    model: LayoutModel,
) -> tuple[dict[str, tuple[CandidateContour, float, float, float, float, str, float]], float, int]:
    assignment: dict[str, tuple[CandidateContour, float, float, float, float, str, float]] = {}

    for cand in candidates:
        if cand is left_anchor or cand is right_anchor:
            continue
        if is_duplicate_candidate(cand, left_anchor, ANCHOR_DUP_DIST):
            continue
        if is_duplicate_candidate(cand, right_anchor, ANCHOR_DUP_DIST):
            continue

        best_match: tuple[float, str, float, float, float, float, str] | None = None
        for zone in STANDARD_LAYOUT_ZONES:
            zone_name, _z_u, _z_v, _tol_u, _tol_v, zone_group, _accept_dist = zone
            area_ok, u, v, dist, accept_dist, _zone_group = score_candidate_for_zone(cand, zone, model)
            if not area_ok:
                continue

            closeness = max(0.0, 1.0 - dist / (accept_dist + 1e-6))
            score = LAYOUT_SCORE_WEIGHT * closeness + RAW_SCORE_WEIGHT * cand.score
            if best_match is None or dist < best_match[0]:
                best_match = (dist, zone_name, u, v, accept_dist, score, zone_group)

        if best_match is None:
            continue

        dist, zone_name, u, v, accept_dist, score, zone_group = best_match
        old = assignment.get(zone_name)
        if old is None or score > old[6]:
            assignment[zone_name] = (cand, u, v, dist, accept_dist, zone_group, score)

    used: list[CandidateContour] = []
    filtered: dict[str, tuple[CandidateContour, float, float, float, float, str, float]] = {}
    for zone_name in [z[0] for z in STANDARD_LAYOUT_ZONES]:
        if zone_name not in assignment:
            continue
        item = assignment[zone_name]
        cand = item[0]
        if any(is_duplicate_candidate(cand, old, ANCHOR_DUP_DIST) for old in used):
            continue
        filtered[zone_name] = item
        used.append(cand)

    coverage = len(filtered)
    assignment_score = sum(item[6] for item in filtered.values())
    total_score = 2.4 * coverage + assignment_score
    return filtered, total_score, coverage


def apply_standard_layout_prior(
    candidates: list[CandidateContour],
    roi_shape: tuple[int, int]
) -> tuple[list[CandidateContour], LayoutModel | None]:
    if not ENABLE_LAYOUT_PRIOR:
        return candidates, None

    if len(candidates) < 2:
        return candidates, None

    print("\n===== Stage 4: standard layout prior filtering =====")

    pair_hypotheses = find_main_big_hole_pair_hypotheses(candidates, roi_shape)
    if not pair_hypotheses:
        print("No valid main-hole pair found. Skip layout prior filtering.")
        return candidates, None

    best_result: tuple[
        float,
        float,
        int,
        CandidateContour,
        CandidateContour,
        LayoutModel,
        dict[str, tuple[CandidateContour, float, float, float, float, str, float]],
    ] | None = None

    print(f"main_pair_hypotheses={len(pair_hypotheses)}")

    best_pair_score = pair_hypotheses[0][0]

    for pair_score, i, j in pair_hypotheses:
        if pair_score < best_pair_score * MAIN_PAIR_RELATIVE_SCORE_MIN:
            continue

        c1 = candidates[i]
        c2 = candidates[j]

        if c1.center[0] <= c2.center[0]:
            left_anchor, right_anchor = c1, c2
        else:
            left_anchor, right_anchor = c2, c1

        model = build_layout_model(left_anchor, right_anchor)
        if model is None:
            continue

        assignment, assignment_score, coverage = build_global_layout_assignment(
            candidates,
            left_anchor,
            right_anchor,
            model,
        )

        total_score = assignment_score + 1.2 * pair_score
        print(
            f"main_pair_try left=({left_anchor.center[0]:.1f}, {left_anchor.center[1]:.1f})  "
            f"right=({right_anchor.center[0]:.1f}, {right_anchor.center[1]:.1f})  "
            f"dist={model.scale:.1f} coverage={coverage} pair={pair_score:.3f} total={total_score:.3f}"
        )

        if best_result is None or total_score > best_result[0]:
            best_result = (
                total_score,
                pair_score,
                coverage,
                left_anchor,
                right_anchor,
                model,
                assignment,
            )

    if best_result is None:
        print("Failed to build layout model. Skip layout prior filtering.")
        return candidates, None

    total_score, pair_score, coverage, left_anchor, right_anchor, model, assignment = best_result

    _top_pair_score, top_i, top_j = pair_hypotheses[0]
    top_c1 = candidates[top_i]
    top_c2 = candidates[top_j]
    if top_c1.center[0] <= top_c2.center[0]:
        legacy_left, legacy_right = top_c1, top_c2
    else:
        legacy_left, legacy_right = top_c2, top_c1

    legacy_model = build_layout_model(legacy_left, legacy_right)
    if legacy_model is not None:
        legacy_assignment, legacy_score, legacy_coverage = build_nearest_layout_assignment(
            candidates,
            legacy_left,
            legacy_right,
            legacy_model,
        )
        if coverage < legacy_coverage + GLOBAL_ASSIGNMENT_MIN_COVERAGE_GAIN:
            print(
                f"main_pair_fallback nearest coverage={legacy_coverage} global={coverage} "
                f"score={legacy_score:.3f}"
            )
            total_score = legacy_score
            pair_score = _top_pair_score
            coverage = legacy_coverage
            left_anchor = legacy_left
            right_anchor = legacy_right
            model = legacy_model
            assignment = legacy_assignment

    for cand in candidates:
        cand.layout_name = ""
        cand.layout_group = ""
        cand.layout_u = 0.0
        cand.layout_v = 0.0
        cand.layout_dist = 999.0
        cand.layout_score = 0.0
        cand.layout_accept_dist = 999.0
        cand.is_main_anchor = False
        cand.is_top_backup = False

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
        f"dist={model.scale:.1f} coverage={coverage} pair={pair_score:.3f} total={total_score:.3f}"
    )

    final_kept: list[CandidateContour] = [left_anchor, right_anchor]

    zone_order = [z[0] for z in STANDARD_LAYOUT_ZONES]
    for name in zone_order:
        if name not in assignment:
            continue

        cand, u, v, dist, accept_dist, zone_group, score = assignment[name]
        assign_candidate_to_zone(cand, name, zone_group, u, v, dist, accept_dist)
        cand.layout_score = score
        final_kept.append(cand)
        layout_keep_log("keep_layout_global", cand)

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
