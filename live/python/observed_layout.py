"""Supervised image-layout association, independent of metric PnP geometry."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np

from center_fit import fit_ellipse_on_kept_candidates
from keypoint_annotations import LABELS, validate_annotations
from layout_prior import LAYOUT_TO_MODEL_LABEL


DEFAULT_PRIOR_PATH = Path(__file__).resolve().parents[2] / "config" / "observed_layout_prior.json"


def fit_observed_layout(annotations: dict) -> dict:
    """Fit view templates from reviewed train observations only; never use val."""
    validate_annotations(annotations)
    rows = [row for row in annotations["images"]
            if row["split"] == "train" and row["review_status"] == "reviewed"]
    templates, training = [], []
    for row in rows:
        points = {name: np.asarray(point["xy"], dtype=float) for name, point in row["points"].items()
                  if point["visibility"] == "visible"}
        if not {"DC_neg", "DC_pos", "PE", "CC2"} <= points.keys() or len(points) < 6:
            continue
        axis = points["DC_pos"] - points["DC_neg"]
        scale = float(np.linalg.norm(axis))
        if scale < 20:
            continue
        origin = (points["DC_pos"] + points["DC_neg"]) / 2
        basis = np.array([axis, [-axis[1], axis[0]]]) / scale**2
        uv = {name: (basis @ (point - origin)).tolist() for name, point in points.items()}
        if uv["PE"][1] <= 0.1 or uv["CC2"][1] >= -0.1:
            raise ValueError("Reviewed DC/PE/CC2 orientation inconsistent with the label convention")
        templates.append({"points": uv})
        training.append({"image": row["image"], "sha256": row["image_sha256"]})
    if len(templates) < 3:
        raise ValueError("At least three reviewed train views with DC, PE and CC2 are required")
    medians = {name: np.median([view["points"][name] for view in templates if name in view["points"]], axis=0).tolist()
               for name in LABELS if any(name in view["points"] for view in templates)}
    if set(medians) != set(LABELS):
        raise ValueError("Reviewed train views must collectively observe every hole label")
    result = {"schema_version": 1, "kind": "observed_2d_layout", "labels": list(LABELS),
              "fit_split": "train", "independent_test_set": False,
              "training_images": training, "templates": templates, "fallback_centers": medians,
              "note": "Image-space view templates for association only, not CAD coordinates or calibration."}
    validate_prior(result)
    return result


def validate_prior(prior: dict) -> None:
    if (prior.get("schema_version") != 1 or prior.get("kind") != "observed_2d_layout"
            or prior.get("labels") != list(LABELS) or prior.get("fit_split") != "train"):
        raise ValueError("Invalid observed-layout prior schema")
    if not isinstance(prior.get("templates"), list) or len(prior["templates"]) < 3:
        raise ValueError("Observed-layout prior requires at least three templates")
    for points in [prior.get("fallback_centers", {}), *(view.get("points", {}) for view in prior["templates"])]:
        if not {"DC_neg", "DC_pos", "PE", "CC2"} <= points.keys() or not set(points) <= set(LABELS):
            raise ValueError("Invalid template labels")
        array = np.asarray(list(points.values()), dtype=float)
        if array.shape != (len(points), 2) or not np.isfinite(array).all() or np.max(np.abs(array)) > 3:
            raise ValueError("Invalid normalized template centers")
        if not np.allclose(points["DC_neg"], [-.5, 0], atol=1e-6) or not np.allclose(points["DC_pos"], [.5, 0], atol=1e-6):
            raise ValueError("Template DC anchors must define the normalized frame")
    if set(prior["fallback_centers"]) != set(LABELS):
        raise ValueError("Missing fallback label")


def load_observed_layout(path: Path) -> dict:
    prior = json.loads(path.read_text(encoding="utf-8"))
    validate_prior(prior)
    return prior


load_prior = load_observed_layout


def _center(candidate):
    point = getattr(candidate, "aperture_center", None)
    return np.asarray(point if point is not None else candidate.fitted_center, dtype=float)


def _diameter(candidate):
    diameter = getattr(candidate, "aperture_diameter_px", None)
    if diameter is not None and np.isfinite(diameter) and diameter > 0:
        return float(diameter)
    return float(np.sqrt(np.prod(candidate.ellipse_axes)))


def _prepare(candidates):
    with contextlib.redirect_stdout(io.StringIO()):
        fitted = fit_ellipse_on_kept_candidates(copy.deepcopy(candidates))
    unique = []
    for candidate in sorted(fitted, key=lambda item: item.score, reverse=True):
        if not candidate.fit_valid or candidate.fitted_center is None or not np.isfinite(_center(candidate)).all():
            continue
        group = getattr(candidate, "hole_group_id", -1)
        if any((group >= 0 and group == getattr(other, "hole_group_id", -1))
               or np.linalg.norm(_center(candidate) - _center(other)) < 8 for other in unique):
            continue
        unique.append(candidate)
    return unique


def _anchor_pairs(candidates, points, diameters):
    pairs = []
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            span = float(np.linalg.norm(points[second] - points[first]))
            if span < 30:
                continue
            sizes = diameters[[first, second]]
            if not np.all((sizes / span >= .12) & (sizes / span <= .75)) or min(sizes) / max(sizes) < .55:
                continue
            if not all(getattr(candidates[index], "aperture_diameter_px", None) for index in (first, second)):
                continue
            size = float(np.mean(sizes))
            score = size * (min(sizes) / max(sizes)) * np.mean([candidates[first].score, candidates[second].score])
            pairs.append((score, first, second))
    return sorted(pairs, reverse=True)[:24]


def _assign(predicted, points, diameters, aperture_known, anchors, pe_index, scale, scores):
    # Shared Hungarian implementation is imported lazily to avoid an import cycle
    # when the main correspondence module calls this independent frontend.
    from pose_correspondence import _minimum_assignment
    fixed = {4: anchors[0], 5: anchors[1], 7: pe_index}
    slots = [0, 1, 2, 3, 6, 8]
    available = [index for index in range(len(points)) if index not in fixed.values()]
    if not available:
        return fixed
    distances = np.linalg.norm(predicted[slots, None] - points[available], axis=2) / scale
    cost = (distances / .20)**2 + .04 * (1 - scores[available])
    # Small signal/auxiliary apertures must be visibly smaller than PE.
    # A Canny-only contour can be a tube's outer ring. Its center is usable only
    # very close to the supervised prediction, and its diameter is not an aperture.
    limits = np.where(aperture_known[available], diameters[pe_index] / 1.20, diameters[pe_index] * 2.2)
    invalid_size = (diameters[available] > limits) | (diameters[available] < .04 * scale)
    cost[:, invalid_size] = 1000
    cost[distances > .20] = 1000
    cost[(distances > .08) & ~aperture_known[available][None, :]] = 1000
    cost = np.hstack([cost, np.ones((len(slots), len(slots)))])
    for row, col in enumerate(_minimum_assignment(cost)):
        if col < len(available) and cost[row, col] < 1:
            fixed[slots[row]] = available[col]
    return fixed


def select_observed_correspondences(candidates, roi_shape, prior: dict, offset_xy=(0, 0)) -> dict:
    """Associate observed centers without consulting PnP, filenames or annotations."""
    validate_prior(prior)
    fitted = _prepare(candidates)
    diagnostics = {"method": "observed_2d_layout", "valid_center_count": len(fitted),
                   "hypothesis_count": 0, "selection_quality_ok": False, "ambiguous": False,
                   "structural_support_ok": False, "score_margin": None}
    empty = {"image_points": {}, "candidates": [], "diagnostics": diagnostics, "hypotheses": []}
    if len(fitted) < 4:
        return empty
    points = np.asarray([_center(candidate) for candidate in fitted])
    diameters = np.asarray([_diameter(candidate) for candidate in fitted])
    aperture_known = np.asarray([bool(getattr(candidate, "aperture_diameter_px", None)) for candidate in fitted])
    scores = np.asarray([candidate.score for candidate in fitted])
    templates = [np.asarray([view["points"].get(label, prior["fallback_centers"][label]) for label in LABELS])
                 for view in prior["templates"]]
    proposals = {}
    pairs = _anchor_pairs(fitted, points, diameters)
    diagnostics["anchor_pair_count"] = len(pairs)
    for _, first, second in pairs:
        for anchors in ((first, second), (second, first)):
            origin = (points[first] + points[second]) / 2
            axis = points[anchors[1]] - points[anchors[0]]
            scale = float(np.linalg.norm(axis))
            perp = np.array([-axis[1], axis[0]])
            uv = (points - origin) @ np.array([axis, perp]).T / scale**2
            main_size = float(np.mean(diameters[list(anchors)]))
            pe_choices = [index for index in range(len(points)) if index not in anchors
                          and .15 <= uv[index, 1] <= .85 and abs(uv[index, 0]) <= .65
                          and .25 <= diameters[index] / main_size <= .85
                          and getattr(fitted[index], "aperture_diameter_px", None)]
            for pe_index in pe_choices:
                for template_index, template in enumerate(templates):
                    pe_u, pe_v = template[7]
                    shear = (uv[pe_index, 0] - pe_u) / pe_v
                    stretch = uv[pe_index, 1] / pe_v
                    if not .45 <= stretch <= 1.65 or abs(shear) > .65:
                        continue
                    warped = template.copy()
                    warped[:, 0] += shear * template[:, 1]
                    warped[:, 1] *= stretch
                    predicted = origin + warped[:, :1] * axis + warped[:, 1:] * perp
                    assignment = _assign(predicted, points, diameters, aperture_known, anchors, pe_index, scale, scores)
                    if 1 not in assignment:
                        continue
                    errors = {slot: float(np.linalg.norm(predicted[slot] - points[index]) / scale)
                              for slot, index in assignment.items()}
                    score = len(assignment) - sum((error / .20)**2 for error in errors.values())
                    score -= .15 * (abs(shear) + abs(np.log(stretch)))
                    structural = all(fitted[index].inside_prior for index in assignment.values())
                    # Alternative outer-ring supports are the same hypothesis
                    # when all emitted aperture identities and centers agree.
                    signature = tuple(sorted((slot, index) for slot, index in assignment.items()
                                             if aperture_known[index]))
                    value = {"score": score, "assignment": assignment, "template": template_index,
                             "errors": errors, "structural_support_ok": structural,
                             "anchors": anchors}
                    if (signature not in proposals or (structural, score) >
                            (proposals[signature]["structural_support_ok"], proposals[signature]["score"])):
                        proposals[signature] = value
    diagnostics["hypothesis_count"] = len(proposals)
    if not proposals:
        return empty
    ordered = sorted(proposals.values(), key=lambda item: (item["structural_support_ok"], item["score"]), reverse=True)
    best = ordered[0]
    comparable = [item for item in ordered[1:] if item["structural_support_ok"] == best["structural_support_ok"]]
    margin = best["score"] - comparable[0]["score"] if comparable else None
    association = dict(best["assignment"])
    # Canny-only outer rings may support a layout hypothesis, but they are not
    # verified observations of the inner aperture used by the 3D point model.
    assignment = {slot: index for slot, index in association.items() if aperture_known[index]}
    ambiguous = bool(comparable and margin < .15)
    offset = np.asarray(offset_xy, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("Invalid image offset")
    image_points, kept = {}, []
    layout_names = list(LAYOUT_TO_MODEL_LABEL)
    for slot, index in sorted(assignment.items()):
        candidate = copy.copy(fitted[index])
        candidate.layout_name = layout_names[slot]
        candidate.is_main_anchor = slot in (4, 5)
        image_points[LABELS[slot]] = tuple(points[index] + offset)
        kept.append(candidate)
    aperture_support = len(assignment)
    orientation_ok = bool(1 in assignment and aperture_known[assignment[1]])
    eligible = (len(assignment) >= 6 and aperture_support >= 5 and orientation_ok
                and best["structural_support_ok"] and not ambiguous)
    diagnostics.update(selection_quality_ok=bool(eligible), ambiguous=ambiguous,
                       structural_support_ok=best["structural_support_ok"], score_margin=margin,
                       score=float(best["score"]), template_index=best["template"],
                       aperture_support_count=aperture_support,
                       association_count=len(association),
                       excluded_without_aperture=[LABELS[slot] for slot in association if slot not in assignment],
                       normalized_errors={LABELS[k]: v for k, v in best["errors"].items()},
                       orientation_evidence={"ok": orientation_ok, "reason": "observed_DC_PE_CC2_aperture_sizes"
                                             if orientation_ok else "CC2_aperture_evidence_missing"})
    return {"image_points": image_points, "candidates": kept, "diagnostics": diagnostics,
            "association_points": {LABELS[slot]: tuple(points[index] + offset) for slot, index in association.items()},
            "hypotheses": [{"score": float(item["score"]), "candidate_count": len(item["assignment"]),
                            "structural_support_ok": item["structural_support_ok"]} for item in ordered[:8]]}
