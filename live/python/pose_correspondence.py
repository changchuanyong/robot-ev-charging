"""Assign real measured centers to the model under rotation and perspective."""
from __future__ import annotations

import contextlib
import copy
import io

import cv2
import numpy as np

from center_fit import export_fitted_centers, fit_ellipse_on_kept_candidates
from layout_prior import (LAYOUT_TO_MODEL_LABEL, MAIN_PAIR_RELATIVE_SCORE_MIN,
                          find_main_big_hole_pair_hypotheses, normalized_model_points)
import pnp2


def _measurement_center(candidate):
    aperture = getattr(candidate, "aperture_center", None)
    if aperture is not None and np.asarray(aperture).shape == (2,) and np.isfinite(aperture).all():
        return tuple(aperture)
    return candidate.fitted_center


def _orientation_evidence(kept):
    """Require observed PE/signal size asymmetry, independent of image rotation."""
    sizes = {}
    for candidate in kept:
        aperture = getattr(candidate, "aperture_diameter_px", None)
        if aperture is not None and np.isfinite(aperture) and aperture > 0:
            sizes[candidate.layout_name] = float(aperture)
    if not {"bottom_mid", "top_mid"} <= sizes.keys():
        return {"ok": False, "reason": "PE_and_CC2_aperture_boundaries_required", "pe_to_signal_ratio": None}
    signal_sizes = [size for name, size in sizes.items()
                    if name not in {"main_left", "main_right", "bottom_mid"}]
    ratio = sizes["bottom_mid"] / max(signal_sizes)
    return {"ok": bool(ratio >= 1.25),
            "reason": "ok" if ratio >= 1.25 else "PE_signal_size_not_distinct",
            "pe_to_signal_ratio": ratio}


def _minimum_assignment(cost):
    """Rectangular Hungarian assignment; dummy columns represent missing holes."""
    rows, cols = cost.shape
    u, v = np.zeros(rows + 1), np.zeros(cols + 1)
    p, way = np.zeros(cols + 1, int), np.zeros(cols + 1, int)
    for i in range(1, rows + 1):
        p[0], j0 = i, 0
        minimum, used = np.full(cols + 1, np.inf), np.zeros(cols + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            remaining = np.flatnonzero(~used[1:]) + 1
            current = cost[i0 - 1, remaining - 1] - u[i0] - v[remaining]
            better = current < minimum[remaining]
            minimum[remaining[better]] = current[better]
            way[remaining[better]] = j0
            j1 = remaining[np.argmin(minimum[remaining])]
            delta = minimum[j1]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = np.full(rows, -1, int)
    for j in range(1, cols + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result


def _assign(predicted, points, anchor_indices, scale, scores, diameters=None):
    labels = [i for i in range(9) if i not in (4, 5)]
    available = [i for i in range(len(points)) if i not in anchor_indices]
    assignment = {4: anchor_indices[0], 5: anchor_indices[1]}
    if not available:
        return assignment
    distances = np.linalg.norm(predicted[labels, None] - points[available], axis=2)
    cost = (distances / (0.25 * scale)) ** 2 + 0.08 * (1 - scores[available])
    cost[distances > 0.25 * scale] = 1000
    if diameters is not None:
        anchor_diameter = float(np.mean(diameters[list(anchor_indices)]))
        # Signal/auxiliary holes are smaller than DC apertures; PE may be similar.
        ratios = diameters[available] / anchor_diameter
        for row, label in enumerate(labels):
            maximum = 1.25 if label == 7 else 0.85
            cost[row, (ratios < 0.04) | (ratios > maximum)] = 1000
    cost = np.hstack([cost, np.full((len(labels), len(labels)), 1.0)])
    for row, col in enumerate(_minimum_assignment(cost)):
        if col < len(available) and cost[row, col] < 1.0:
            assignment[labels[row]] = available[col]
    return assignment


def _refine(model, points, assignment, predicted, anchors, scale, scores, diameters):
    # Fit every assigned point. No RANSAC removal or fabricated centers.
    for _ in range(2):
        if len(assignment) < 5:
            break
        labels, indices = zip(*sorted(assignment.items()))
        homography, _ = cv2.findHomography(model[list(labels)], points[list(indices)], 0)
        if homography is None or not np.isfinite(homography).all():
            break
        homogeneous = np.column_stack([model, np.ones(len(model))]) @ homography.T
        denominators = homogeneous[:, 2]
        if np.any(np.abs(denominators) < 1e-8) or np.any(denominators * denominators[0] <= 0):
            break
        updated = homogeneous[:, :2] / denominators[:, None]
        if np.max(np.linalg.norm(updated - predicted, axis=1)) > 0.5 * scale:
            break
        proposed = _assign(updated, points, anchors, scale, scores, diameters)
        # Refinement may recover holes, but cannot discard an observed assigned hole.
        if not assignment.keys() <= proposed.keys():
            break
        predicted, assignment = updated, proposed
    return assignment, predicted


def select_correspondences(candidates, roi_shape, object_points, K, dist,
                           offset_xy=(0, 0), point_transform=None,
                           max_mean_error_px=20.0, min_inlier_count=4,
                           min_correspondence_count=6):
    """Return best/diagnostics using only valid fitted input contours.

    ``best.image_points`` stays in raw full-image pixels. An optional transform
    rectifies those pixels solely for PnP; its K/dist must describe that space.
    Hypothesis scores use residuals for every correspondence, including outliers.
    """
    candidates = copy.deepcopy(candidates)
    with contextlib.redirect_stdout(io.StringIO()):
        candidates = fit_ellipse_on_kept_candidates(candidates)
    fitted = [c for c in candidates if getattr(c, "fit_valid", c.fit_method == "ellipse")
              and c.fitted_center is not None and np.isfinite(c.fitted_center).all()]
    # A contour and its nested ring must not become two different model holes.
    unique = []
    for candidate in sorted(fitted, key=lambda c: c.score, reverse=True):
        group_id = getattr(candidate, "hole_group_id", -1)
        if not any((group_id >= 0 and group_id == getattr(old, "hole_group_id", -1))
                   or np.linalg.norm(np.subtract(_measurement_center(candidate), _measurement_center(old))) < 8
                   for old in unique):
            unique.append(candidate)
    fitted = unique
    diagnostics = {"input_candidate_count": len(candidates), "valid_center_count": len(fitted),
                   "hypothesis_count": 0, "ambiguous": False, "score_margin": None}
    if len(fitted) < 2:
        return {"best": None, "hypotheses": [], "diagnostics": diagnostics}
    normalized = normalized_model_points(object_points)
    names = list(LAYOUT_TO_MODEL_LABEL)
    model = np.asarray([normalized[name] for name in names], dtype=np.float64)
    points = np.asarray([_measurement_center(c) for c in fitted], dtype=np.float64)
    scores = np.asarray([c.score for c in fitted])
    diameters = np.asarray([np.sqrt(np.prod(c.ellipse_axes)) for c in fitted])
    pair_candidates = [copy.copy(candidate) for candidate in fitted]
    for candidate in pair_candidates:
        candidate.center = _measurement_center(candidate)
    pairs = find_main_big_hole_pair_hypotheses(pair_candidates, roi_shape)
    if pairs:
        pairs = [pair for pair in pairs if pair[0] >= pairs[0][0] * MAIN_PAIR_RELATIVE_SCORE_MIN]
    proposals = {}
    diagnostics["anchor_pair_count"] = len(pairs)
    for pair_rank, (pair_score, first, second) in enumerate(pairs):
        for anchors in ((first, second), (second, first)):
            origin = (points[anchors[0]] + points[anchors[1]]) / 2
            axis = points[anchors[1]] - points[anchors[0]]
            scale = float(np.linalg.norm(axis))
            perpendicular = np.array([-axis[1], axis[0]])
            for stretch in (0.65, 1.0, 1.35):
                for shear in (-0.25, 0.0, 0.25):
                    predicted = origin + model[:, :1] * axis + model[:, 1:] * (stretch * perpendicular + shear * axis)
                    assignment = _assign(predicted, points, anchors, scale, scores, diameters)
                    assignment, predicted = _refine(model, points, assignment, predicted, anchors, scale, scores, diameters)
                    signature = tuple(sorted(assignment.items()))
                    residual = sum(float(np.linalg.norm(points[j] - predicted[i]) / scale) ** 2
                                   for i, j in signature)
                    geometry_cost = residual + 0.035 * (9 - len(assignment)) + 0.03 * (1 - pair_score)
                    old = proposals.get(signature)
                    if old is None or geometry_cost < old[0]:
                        proposals[signature] = (geometry_cost, scale, pair_rank, residual, pair_score)
    diagnostics["hypothesis_count"] = len(proposals)
    # Keep distinct observed correspondences, not many equivalent initial grids.
    ordered = sorted(proposals.items(), key=lambda item: item[1][0])
    shortlist = dict(ordered[:4])
    for count in range(9, 3, -1):
        group = sorted((item for item in ordered if len(item[0]) == count),
                       key=lambda item: item[1][3] / count + 0.03 * (1 - item[1][4]))
        shortlist.update(group[:2])
    for signature, proposal in ordered:
        if len(shortlist) >= 16:
            break
        shortlist.setdefault(signature, proposal)
    shortlisted = list(shortlist.items())
    results = []
    for signature, (geometry_cost, scale, rank, _residual, pair_score) in shortlisted:
        kept, image_points = [], {}
        for slot, index in signature:
            candidate = copy.copy(fitted[index])
            candidate.layout_name = names[slot]
            candidate.is_main_anchor = slot in (4, 5)
            candidate.layout_u, candidate.layout_v = model[slot]
            candidate.layout_score = candidate.score
            kept.append(candidate)
            image_points[LAYOUT_TO_MODEL_LABEL[names[slot]]] = tuple(points[index] + offset_xy)
        pose, error = None, None
        try:
            solve_points = image_points
            if point_transform is not None:
                labels = list(image_points)
                transformed = np.asarray(point_transform(np.asarray(list(image_points.values()))))
                if transformed.shape != (len(labels), 2) or not np.isfinite(transformed).all():
                    raise ValueError("invalid_point_transform")
                solve_points = dict(zip(labels, transformed))
            pose = pnp2.solve_target_pose(solve_points, object_points, K, dist)
            if not pose["all_in_front"]:
                raise ValueError("nonpositive_camera_depth")
            if not np.isfinite(list(pose["per_point_errors_px"].values())).all():
                raise ValueError("nonfinite_pose_residuals")
        except (ValueError, RuntimeError, cv2.error) as exc:
            error, pose = str(exc), None
        errors = pose["per_point_errors_px"] if pose else {}
        reproj = float(np.mean(list(errors.values()))) if errors else float("inf")
        maximum = max(errors.values(), default=float("inf"))
        count = len(kept)
        # Four/five unlabeled circles can accidentally fit a false small subset.
        # Keep that solve for diagnosis, but require redundant observed support.
        # The detector ROI can include background rings. Retain them in diagnostic
        # hypotheses, but a pose using centers outside the original connector
        # structure prior has no verified target support and cannot be accepted.
        structural_support_ok = all(candidate.inside_prior for candidate in kept)
        orientation_evidence = _orientation_evidence(kept)
        geometric_quality_ok = bool(pose and structural_support_ok and count >= min_correspondence_count
                                    and reproj <= max_mean_error_px
                                    and pose["inlier_count"] >= min_inlier_count)
        quality_ok = geometric_quality_ok and orientation_evidence["ok"]
        # A large false anchor span must not make the same pixel error cheaper.
        score = (20 * count - 2 * reproj - 0.5 * maximum) if pose else -1000 + count
        centers = export_fitted_centers(kept)
        for item, candidate in zip(centers, kept):
            item["measurement_center"] = list(_measurement_center(candidate))
            item["measurement_source"] = "adaptive_dark" if candidate.aperture_center is not None else candidate.contour_source
        indices = dict(signature)
        results.append({"variant_name": f"projective_pair_{rank + 1}", "score": score,
                        "candidates": kept, "centers": centers,
                        "layout_names": {c.layout_name for c in kept},
                        "missing_layouts": sorted(set(names) - {c.layout_name for c in kept}),
                        "pose": pose, "pnp_error": error, "candidate_count": count,
                        "reproj": reproj, "inliers": pose["inlier_count"] if pose else 0,
                        "image_points": image_points, "coverage": count / 9,
                        "all_point_errors_px": errors, "geometry_cost": geometry_cost,
                        "anchor_pair_score": pair_score, "selection_quality_ok": quality_ok,
                        "structural_support_ok": structural_support_ok,
                        "orientation_evidence": orientation_evidence,
                        "geometric_quality_ok": geometric_quality_ok,
                        "anchor_indices": (indices[4], indices[5])})
    results.sort(key=lambda result: (result["selection_quality_ok"], result["score"]), reverse=True)
    best = results[0] if results else None
    comparable = [result for result in results if best and result["selection_quality_ok"] == best["selection_quality_ok"]]
    if len(comparable) > 1:
        margin = comparable[0]["score"] - comparable[1]["score"]
        diagnostics.update(score_margin=float(margin), ambiguous=bool(margin < 2.0))
    if best:
        reverse = [result for result in results if result["geometric_quality_ok"]
                   and result["anchor_indices"] == best["anchor_indices"][::-1]]
        conflict = bool(reverse and (not best["orientation_evidence"]["ok"]
                                    or any(result["orientation_evidence"]["ok"] for result in reverse)))
        if conflict:
            diagnostics.update(ambiguous=True, ambiguity_reason="opposite_directions_without_distinct_aperture_evidence")
    diagnostics["selection_quality_ok"] = bool(best and best["selection_quality_ok"])
    diagnostics["structural_support_ok"] = bool(best and best["structural_support_ok"])
    diagnostics["orientation_evidence"] = best["orientation_evidence"] if best else {"ok": False, "reason": "no_hypothesis"}
    diagnostics["quality_eligible_hypothesis_count"] = sum(result["selection_quality_ok"] for result in results)
    diagnostics["evaluated_hypothesis_count"] = len(results)
    hypotheses = [{key: value for key, value in result.items()
                   if key in ("variant_name", "score", "candidate_count", "coverage", "geometry_cost",
                              "reproj", "inliers", "pnp_error", "all_point_errors_px", "image_points",
                              "anchor_pair_score", "selection_quality_ok", "structural_support_ok",
                              "orientation_evidence")}
                  for result in results]
    for hypothesis in hypotheses:
        if not np.isfinite(hypothesis["reproj"]):
            hypothesis["reproj"] = None
    return {"best": best, "hypotheses": hypotheses, "diagnostics": diagnostics}
