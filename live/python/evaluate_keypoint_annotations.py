"""Compare named observations and automatic points with a fixed PnP backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np

from keypoint_annotations import LABELS, normalized_image_key, sha256_file, validate_annotations, visible_points
import pnp2


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from vision.photo_sources import resolve_photo


def _repo_path(root: Path, value: str) -> Path:
    path = (root / normalized_image_key(value)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves repository: {value}")
    return resolve_photo(value, root=root) if str(value).replace("\\", "/").startswith("yolo_port/") else path


def validate_inputs(annotations: dict, batch: dict, root: Path, batch_report_path: Path) -> dict:
    """Bind annotations to immutable image bytes, report, geometry and solver."""
    validate_annotations(annotations)
    provenance = annotations["provenance"]
    report_path = batch_report_path.resolve()
    if report_path != _repo_path(root, provenance["batch_report_path"]):
        raise ValueError("Batch report path differs from annotation provenance")
    report_bytes = report_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != provenance["batch_report_sha256"]:
        raise ValueError("Batch report hash changed")
    if json.loads(report_bytes) != batch:
        raise ValueError("Loaded batch report differs from bound report bytes")
    batch_provenance = batch.get("provenance", {})
    for path_field, hash_field in (("camera_config_path", "camera_config_sha256"),
                                   ("model_points_path", "model_points_sha256")):
        actual_hash = sha256_file(_repo_path(root, provenance[path_field]))
        if actual_hash != provenance[hash_field] or actual_hash != batch_provenance.get(hash_field):
            raise ValueError(f"Input hash changed: {path_field}")
    source_hash = sha256_file(Path(pnp2.__file__))
    if (source_hash != provenance["pnp_source_sha256"]
            or source_hash != batch_provenance.get("source_sha256", {}).get("pnp2.py")):
        raise ValueError("PnP source hash changed")
    rows = {normalized_image_key(row["image"]): row for row in batch["rows"]}
    if len(rows) != len(batch["rows"]):
        raise ValueError("Batch report contains duplicate image identities")
    camera = json.loads(_repo_path(root, provenance["camera_config_path"]).read_text(encoding="utf-8"))
    camera_size = (camera.get("image_width"), camera.get("image_height"))
    for entry in annotations["images"]:
        key = normalized_image_key(entry["image"])
        if key not in rows:
            raise ValueError(f"Image missing from batch report: {key}")
        row = rows[key]
        image_path = _repo_path(root, key)
        if sha256_file(image_path) != entry["image_sha256"] or row.get("image_sha256") != entry["image_sha256"]:
            raise ValueError(f"Image hash changed: {key}")
        frame = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or (frame.shape[1], frame.shape[0]) != (entry["width"], entry["height"]):
            raise ValueError(f"Image dimensions changed: {key}")
        if camera_size != (entry["width"], entry["height"]):
            raise ValueError(f"Image size does not match fixed camera intrinsics: {key}")
        if entry["auto_points"] != row.get("image_points_full", {}):
            raise ValueError(f"Automatic points changed from report: {key}")
        if not row.get("pnp_ok", False) and entry["auto_points"]:
            raise ValueError(f"Unsolved image unexpectedly supplies selected automatic points: {key}")
        if entry.get("roi_bbox_xyxy") != row.get("roi_bbox_xyxy", []):
            raise ValueError(f"ROI changed from report: {key}")
        if entry["split"] != row.get("split", "unknown"):
            raise ValueError(f"Split changed from report: {key}")
    return rows


def residual_stats(values: list[float]) -> dict:
    return {"count": len(values), "mean_px": statistics.mean(values) if values else None,
            "median_px": statistics.median(values) if values else None,
            "max_px": max(values) if values else None}


def solve_points(points: dict, model: dict, matrix: np.ndarray, distortion: np.ndarray) -> dict:
    result = {"solved": False, "point_count": len(points), "labels": list(points)}
    if len(points) < 4:
        return {**result, "reason": "fewer_than_four_points"}
    try:
        cv2.setRNGSeed(0)
        pose = pnp2.solve_target_pose(points, model, matrix, distortion)
    except (ValueError, RuntimeError, cv2.error) as exc:
        return {**result, "reason": str(exc)}
    projected = cv2.projectPoints(pose["object_points"], pose["rvec"], pose["tvec"], matrix, distortion)[0]
    return {**result, "solved": True, "method": pose["method"],
            "mean_px": float(pose["reproj_error"]), "rms_px": float(pose["reproj_rms_error"]),
            "max_px": float(pose["max_reproj_error"]),
            "inlier_count": int(pose["inlier_count"]), "inlier_threshold_px": float(pose["inlier_threshold_px"]),
            "per_point_errors_px": pose["per_point_errors_px"],
            "projected_points": dict(zip(pose["matched_names"], projected.reshape(-1, 2).tolist())),
            "rvec": pose["rvec"].reshape(3).tolist(), "tvec_model_units": pose["tvec"].reshape(3).tolist()}


def evaluate_image(entry: dict, batch_row: dict, model: dict, matrix: np.ndarray, distortion: np.ndarray) -> dict:
    observed = visible_points(entry)
    automatic = entry["auto_points"]
    common = [name for name in LABELS if name in observed and name in automatic]
    distances = {name: float(np.linalg.norm(np.asarray(observed[name]) - automatic[name])) for name in common}
    return {"image": normalized_image_key(entry["image"]), "split": entry["split"],
            "review_status": entry["review_status"], "reviewer": entry["reviewer"],
            "visible_annotation_count": len(observed), "automatic_point_count": len(automatic),
            "visibility_counts": {state: sum(p["visibility"] == state for p in entry["points"].values())
                                  for state in ("visible", "uncertain", "occluded", "out_of_frame", "unlabeled")},
            "common_labels": common, "missing_automatic_labels": [name for name in observed if name not in automatic],
            "automatic_labels_without_visible_reference": [name for name in automatic if name not in observed],
            "center_deviation_px": distances,
            "original_pose_quality_ok": batch_row.get("pose_quality_ok"),
            "original_pose_rejection_reasons": batch_row.get("pose_rejection_reasons", []),
            "original_report_mean_px": batch_row.get("pnp_reproj_error_px"),
            "annotation_visible": solve_points(observed, model, matrix, distortion),
            "automatic_all": solve_points(automatic, model, matrix, distortion),
            "annotation_common": solve_points({name: observed[name] for name in common}, model, matrix, distortion),
            "automatic_common": solve_points({name: automatic[name] for name in common}, model, matrix, distortion)}


def summarize_group(rows: list[dict], *, reviewed: bool) -> dict:
    summary = {"image_count": len(rows), "human_reviewed": reviewed,
               "independent_test_set": False, "physical_pose_accuracy_verified": False,
               "annotation_visible_point_count": sum(row["visible_annotation_count"] for row in rows),
               "automatic_selected_point_count": sum(row["automatic_point_count"] for row in rows),
               "common_name_point_count": sum(len(row["common_labels"]) for row in rows),
               "missing_automatic_on_visible_count": sum(len(row["missing_automatic_labels"]) for row in rows),
               "center_deviation_px": residual_stats([v for row in rows for v in row["center_deviation_px"].values()]),
               "original_quality_pass_count": sum(row["original_pose_quality_ok"] is True for row in rows),
               "branches": {}}
    for branch in ("annotation_visible", "automatic_all", "annotation_common", "automatic_common"):
        solved = [row[branch] for row in rows if row[branch]["solved"]]
        summary["branches"][branch] = {
            "solved_images": len(solved), "failed_images": [row["image"] for row in rows if not row[branch]["solved"]],
            "image_mean_residual_px": residual_stats([pose["mean_px"] for pose in solved])}
    paired = [row for row in rows if row["annotation_common"]["solved"] and row["automatic_common"]["solved"]]
    summary["paired_same_images_and_point_names"] = {
        "image_count": len(paired), "images": [row["image"] for row in paired],
        "annotation_image_mean_residual_px": residual_stats([row["annotation_common"]["mean_px"] for row in paired]),
        "automatic_image_mean_residual_px": residual_stats([row["automatic_common"]["mean_px"] for row in paired]),
        "automatic_minus_annotation_mean_px": residual_stats(
            [row["automatic_common"]["mean_px"] - row["annotation_common"]["mean_px"] for row in paired])}
    return summary


def evaluate_annotations(annotations: dict, batch_rows: dict, model: dict, matrix: np.ndarray,
                         distortion: np.ndarray, *, include_drafts: bool = False) -> dict:
    """Run only after validate_inputs; drafts never enter the reviewed aggregate."""
    validate_annotations(annotations)
    if list(model) != list(LABELS):
        raise ValueError("Model label order differs from annotation schema")
    groups = {"human_reviewed": []}
    if include_drafts:
        groups["draft_diagnostic"] = []
    pending = []
    for entry in annotations["images"]:
        key = normalized_image_key(entry["image"])
        if entry["review_status"] != "reviewed":
            pending.append(key)
            if not include_drafts:
                continue
        group = "human_reviewed" if entry["review_status"] == "reviewed" else "draft_diagnostic"
        groups[group].append(evaluate_image(entry, batch_rows[key], model, matrix, distortion))
    return {"schema_version": 1, "independent_test_set": False,
            "physical_pose_accuracy_verified": False, "selected_images": len(annotations["images"]),
            "pending_human_review": pending, "drafts_included_for_diagnosis": include_drafts,
            "feature_definition": annotations["feature_definition"],
            "notes": ["Draft visual annotations are not human ground truth; their statistics remain separate.",
                      "All residuals include every observed point in that branch, including outliers.",
                      "Compare paired_same_images_and_point_names to avoid changing the evaluated point set.",
                      "Center deviation includes localization and identity errors; it is not a proven identity error rate.",
                      "Low reprojection residual does not establish correspondence correctness or physical pose accuracy."],
            "groups": {name: summarize_group(rows, reviewed=name == "human_reviewed") for name, rows in groups.items()},
            "rows": {name: rows for name, rows in groups.items()}}


def validate_output_path(output_path: Path, annotations_path: Path, batch_report_path: Path,
                         annotations: dict, root: Path) -> None:
    """A diagnostic report must never replace a bound input or its evaluator."""
    provenance = annotations["provenance"]
    protected = [annotations_path, batch_report_path, Path(pnp2.__file__), Path(__file__),
                 Path(__file__).with_name("keypoint_annotations.py"),
                 _repo_path(root, provenance["camera_config_path"]),
                 _repo_path(root, provenance["model_points_path"])]
    protected.extend(_repo_path(root, entry["image"]) for entry in annotations["images"])
    output = output_path.resolve()
    for source in protected:
        if output == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError(f"Output must not overwrite an input or source: {source}")


def annotations_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--include-drafts", action="store_true", help="Separate unreviewed diagnosis; never human truth")
    args = parser.parse_args(argv)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    batch = json.loads(args.batch_report.read_text(encoding="utf-8"))
    rows = validate_inputs(annotations, batch, args.root, args.batch_report)
    validate_output_path(args.output, args.annotations, args.batch_report, annotations, args.root)
    provenance = annotations["provenance"]
    model = pnp2.load_object_points_from_csv(_repo_path(args.root, provenance["model_points_path"]))
    matrix, distortion, placeholder = pnp2.load_camera_intrinsics(_repo_path(args.root, provenance["camera_config_path"]))
    result = evaluate_annotations(annotations, rows, model, matrix, distortion, include_drafts=args.include_drafts)
    result["provenance"] = {**provenance, "annotations_sha256": sha256_file(args.annotations),
                            "inputs_verified": True, "camera_is_placeholder": placeholder,
                            "evaluator_sha256": sha256_file(Path(__file__)),
                            "schema_validator_sha256": sha256_file(Path(__file__).with_name("keypoint_annotations.py"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected_images": result["selected_images"], "pending_human_review": len(result["pending_human_review"]),
                      "groups": result["groups"]}, ensure_ascii=False, indent=2))


PIXEL_TOLERANCE = 10.0


def _check_output_points(row: dict, annotation: dict) -> None:
    points = row.get("image_points_full", {})
    if not isinstance(points, dict) or not set(points).issubset(LABELS):
        raise ValueError(f"Invalid output labels: {row['image']}")
    for name, xy in points.items():
        if (not isinstance(xy, list) or len(xy) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in xy)
                or not (0 <= xy[0] < annotation["width"] and 0 <= xy[1] < annotation["height"])):
            raise ValueError(f"Invalid original-pixel output: {row['image']}/{name}")
    if not isinstance(row.get("pnp_ok"), bool) or not isinstance(row.get("pose_quality_ok"), bool):
        raise ValueError(f"Missing boolean solver/quality status: {row['image']}")
    if row["pose_quality_ok"] and not row["pnp_ok"]:
        raise ValueError(f"Quality pass without solved pose: {row['image']}")


def validate_comparison_inputs(annotations: dict, before: dict, after: dict,
                               root: Path, before_path: Path, after_path: Path) -> tuple[dict, dict]:
    """The original report binds annotations; the new report may change only vision inputs."""
    old = validate_inputs(annotations, before, root, before_path)
    if json.loads(after_path.read_text(encoding="utf-8")) != after:
        raise ValueError("Loaded after report differs from file")
    provenance = annotations["provenance"]
    new_provenance = after.get("provenance", {})
    for field in ("camera_config_sha256", "model_points_sha256"):
        if new_provenance.get(field) != provenance[field]:
            raise ValueError(f"After fixed input hash changed: {field}")
    if new_provenance.get("source_sha256", {}).get("pnp2.py") != provenance["pnp_source_sha256"]:
        raise ValueError("After PnP source hash changed")
    new = {normalized_image_key(row["image"]): row for row in after["rows"]}
    if len(new) != len(after["rows"]):
        raise ValueError("After report contains duplicate image identities")
    for entry in annotations["images"]:
        key = normalized_image_key(entry["image"])
        if key not in new:
            raise ValueError(f"Annotated image missing from after report: {key}")
        if new[key].get("image_sha256") != entry["image_sha256"]:
            raise ValueError(f"After image hash changed: {key}")
        if new[key].get("split", "unknown") != entry["split"]:
            raise ValueError(f"After split changed: {key}")
        _check_output_points(old[key], entry)
        _check_output_points(new[key], entry)
    return old, new


def error_stats(values: list[float]) -> dict:
    return {"point_count": len(values), "mean_px": statistics.mean(values) if values else None,
            "median_px": statistics.median(values) if values else None,
            "max_px": max(values) if values else None}


def score_image(entry: dict, row: dict) -> dict:
    reference = visible_points(entry)
    output = row.get("image_points_full", {})
    distances = {name: math.dist(xy, output[name]) for name, xy in reference.items() if name in output}
    correct = [name for name, error in distances.items() if error <= PIXEL_TOLERANCE]
    wrong = [name for name, error in distances.items() if error > PIXEL_TOLERANCE]
    missing = [name for name in reference if name not in output]
    unsupported = [name for name in output
                   if entry["points"][name]["visibility"] in ("occluded", "out_of_frame")]
    unscorable = [name for name in output
                 if entry["points"][name]["visibility"] in ("uncertain", "unlabeled")]
    passed = row["pose_quality_ok"]
    return {"output_count": len(output), "correct_labels": correct, "wrong_labels": wrong,
            "missing_labels": missing, "unsupported_nonvisible_labels": unsupported,
            "unscorable_labels": unscorable, "center_errors_px": distances,
            "solver_ok": row["pnp_ok"], "quality_pass": passed,
            "pose_rejection_reasons": row.get("pose_rejection_reasons", []),
            "labeled_failure_gate_pass": bool(passed and (wrong or unsupported)),
            "gate_pass_with_missing_visible": bool(passed and missing),
            "fully_supported_gate_pass": bool(passed and reference and not (wrong or missing or unsupported or unscorable))}


def summarize_version(rows: list[dict], version: str) -> dict:
    data = [row[version] for row in rows]
    reference_count = sum(row["visible_reference_count"] for row in rows)
    correct = sum(len(row["correct_labels"]) for row in data)
    wrong = sum(len(row["wrong_labels"]) for row in data)
    missing = sum(len(row["missing_labels"]) for row in data)
    if correct + wrong + missing != reference_count:
        raise ValueError("Visible reference denominator changed")
    return {
        "visible_reference_count": reference_count,
        "output_point_count": sum(row["output_count"] for row in data),
        "correct_visible_points": correct, "wrong_visible_points": wrong,
        "missing_visible_points": missing,
        "correct_visible_fraction": correct / reference_count if reference_count else None,
        "unsupported_nonvisible_outputs": sum(len(row["unsupported_nonvisible_labels"]) for row in data),
        "unscorable_outputs": sum(len(row["unscorable_labels"]) for row in data),
        "center_error_on_returned_visible_points": error_stats(
            [v for row in data for v in row["center_errors_px"].values()]),
        "solver_success_count": sum(row["solver_ok"] for row in data),
        "quality_pass_count": sum(row["quality_pass"] for row in data),
        "labeled_failure_gate_pass_count": sum(row["labeled_failure_gate_pass"] for row in data),
        "labeled_failure_gate_pass_images": [row["image"] for row in rows if row[version]["labeled_failure_gate_pass"]],
        "gate_pass_with_missing_visible_count": sum(row["gate_pass_with_missing_visible"] for row in data),
        "gate_pass_with_missing_visible_images": [row["image"] for row in rows if row[version]["gate_pass_with_missing_visible"]],
        "fully_supported_gate_pass_count": sum(row["fully_supported_gate_pass"] for row in data),
        "fully_supported_gate_pass_images": [row["image"] for row in rows if row[version]["fully_supported_gate_pass"]],
    }


def compare_labeled(annotations: dict, before_rows: dict, after_rows: dict) -> dict:
    """Compare reviewed labels; call validate_comparison_inputs before file-backed use."""
    validate_annotations(annotations)
    rows, pending = [], []
    for entry in annotations["images"]:
        key = normalized_image_key(entry["image"])
        if entry["review_status"] != "reviewed":
            pending.append(key)
            continue
        for source in (before_rows, after_rows):
            if key not in source:
                raise ValueError(f"Annotated image missing from comparison: {key}")
            _check_output_points(source[key], entry)
        before = score_image(entry, before_rows[key])
        after = score_image(entry, after_rows[key])
        common = [name for name in LABELS if name in before["center_errors_px"] and name in after["center_errors_px"]]
        rows.append({"image": key, "split": entry["split"],
                     "visible_reference_count": len(visible_points(entry)),
                     "before": before, "after": after, "paired_labels": common})
    groups = {}
    for split in ("all", "train", "val", *sorted({row["split"] for row in rows} - {"train", "val"})):
        subset = [row for row in rows if split == "all" or row["split"] == split]
        paired_before = [row["before"]["center_errors_px"][name] for row in subset for name in row["paired_labels"]]
        paired_after = [row["after"]["center_errors_px"][name] for row in subset for name in row["paired_labels"]]
        groups[split] = {"reviewed_images": len(subset),
                         "before": summarize_version(subset, "before"),
                         "after": summarize_version(subset, "after"),
                         "paired_same_visible_labels": {
                             "before": error_stats(paired_before), "after": error_stats(paired_after),
                             "mean_after_minus_before_px": statistics.mean(
                                 [after - before for before, after in zip(paired_before, paired_after)]) if paired_before else None}}
    return {"schema_version": 1, "independent_test_set": False,
            "physical_pose_accuracy_verified": False, "pixel_tolerance_px": PIXEL_TOLERANCE,
            "selected_image_count": len(annotations["images"]), "pending_human_review": pending,
            "notes": ["Every visible reference is counted as correct, wrong, or missing, including unsolved/rejected images.",
                      "The 10 px threshold is a fixed development diagnostic tolerance, not a control safety limit.",
                      "Unsupported nonvisible outputs have no measurable reference coordinate and are counted separately.",
                      "A labeled gate failure means a passed result conflicts with a visible label or uses an explicitly nonvisible hole.",
                      "Fully supported gate passes require all visible references within tolerance and no unsupported/unscorable outputs; this is not physical acceptance.",
                      "Train/val are existing inspected development splits. A val split does not establish an independent test."],
            "groups": groups, "rows": rows}


def markdown_report(report: dict) -> str:
    lines = ["# 人工标注回归对照", "",
             f"固定诊断容差：{report['pixel_tolerance_px']:g} px。不是独立测试集，也不代表物理位姿或插装验收。",
             f"待人工复核图数：{len(report['pending_human_review'])}，不纳入下表。", "",
             "| 分区 | 已复核图 | 可见参考点 | 正确点 前→后 | 错点 前→后 | 漏点 前→后 | 遮挡/出画孔输出 前→后 | 质量通过 前→后 | 标注明确冲突却放行 前→后 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in report["groups"].items():
        a, b = group["before"], group["after"]
        pairs = [f"{a[key]}→{b[key]}" for key in ("correct_visible_points", "wrong_visible_points", "missing_visible_points",
                 "unsupported_nonvisible_outputs", "quality_pass_count", "labeled_failure_gate_pass_count")]
        lines.append(f"| {name} | {group['reviewed_images']} | {a['visible_reference_count']} | " + " | ".join(pairs) + " |")
    all_group = report["groups"]["all"]
    paired = all_group["paired_same_visible_labels"]
    fmt = lambda value: "无可比较点" if value is None else f"{value:.6f} px"
    lines += ["", f"同图同名可见点交集：{paired['before']['point_count']} 点；平均中心偏差 {fmt(paired['before']['mean_px'])} → {fmt(paired['after']['mean_px'])}。",
              "此均值只覆盖双方都输出的点，必须结合正确/错点/漏点总数阅读，不能靠删点证明改善。", "",
              f"返回位姿：{all_group['before']['solver_success_count']} → {all_group['after']['solver_success_count']} 张。",
              f"质量通过但缺少可见孔：{all_group['before']['gate_pass_with_missing_visible_count']} → {all_group['after']['gate_pass_with_missing_visible_count']} 张。",
              f"质量通过且所有可见孔在容差内、无非可见/不确定孔输出：{all_group['before']['fully_supported_gate_pass_count']} → {all_group['after']['fully_supported_gate_pass_count']} 张。",
              "上述完整覆盖口径是诊断指标，不等同于机器人可执行性。", "",
              "改进后仍存在标注明确冲突的门槛放行图："]
    failures = all_group["after"]["labeled_failure_gate_pass_images"]
    lines += [f"- `{name}`" for name in failures] if failures else ["- 无（仍需同时查看覆盖率及未通过图片）。"]
    lines += ["", "逐图错误孔名、缺失孔名、原始门槛原因及输入版本见同名 JSON。",
              "现有 val 为已查看过的开发分区；是否用于二维先验拟合应以模型训练记录为准，不能凭分区名称宣称独立验证。", ""]
    return "\n".join(lines)


def labeled_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".json":
        raise ValueError("Output must use .json; a companion .md is also written")
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    old, new = validate_comparison_inputs(annotations, before, after, args.root, args.before, args.after)
    outputs = (args.output, args.output.with_suffix(".md"))
    for output in outputs:
        validate_output_path(output, args.annotations, args.before, annotations, args.root)
        for protected in (args.after, Path(__file__)):
            if output.resolve() == protected.resolve() or (output.exists() and output.samefile(protected)):
                raise ValueError("Output must not overwrite an after report or evaluator")
    report = compare_labeled(annotations, old, new)
    report["provenance"] = {"annotations_sha256": sha256_file(args.annotations),
                            "before_sha256": sha256_file(args.before), "after_sha256": sha256_file(args.after),
                            "evaluator_sha256": sha256_file(Path(__file__)), "inputs_verified": True,
                            "before": before["provenance"], "after": after["provenance"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))


def image_key(row):
    return row["image"].replace("\\", "/")


def split_name(row):
    parts = image_key(row).split("/")
    return row.get("split", next((name for name in ("train", "val", "test") if name in parts), "unknown"))


def batch_residual_stats(rows):
    values = [row["pnp_reproj_error_px"] for row in rows if row.get("pnp_reproj_error_px") is not None]
    return {"count": len(values), "mean_px": statistics.mean(values) if values else None,
            "median_px": statistics.median(values) if values else None,
            "max_px": max(values) if values else None}


def compare(before, after, manifest, *, baseline_sha256):
    if baseline_sha256 != manifest.get("artifacts/batch_eval_control/latest_details.json"):
        raise ValueError("Baseline report changed")
    old = {image_key(row): row for row in before["rows"]}
    new = {image_key(row): row for row in after["rows"]}
    if len(old) != len(before["rows"]) or len(new) != len(after["rows"]):
        raise ValueError("Duplicate image identities")
    if old.keys() != new.keys():
        raise ValueError("Before/after image sets differ")
    provenance = after["provenance"]
    for name, field in (("runs/detect_retrain/weights/best.pt", "weights_sha256"),
                        ("config/camera_intrinsics.json", "camera_config_sha256"),
                        ("config/charging_port_model.csv", "model_points_sha256")):
        if manifest.get(name) != provenance[field]:
            raise ValueError(f"Input changed: {name}")
    for name, row in new.items():
        if manifest.get(name) != row.get("image_sha256"):
            raise ValueError(f"Image changed: {name}")
    content_groups = {}
    for name, row in new.items():
        content_groups.setdefault(row["image_sha256"], []).append(name)
    groups = {}
    for split in ("all", *sorted({split_name(row) for row in before["rows"]})):
        keys = [key for key in old if split == "all" or split_name(old[key]) == split]
        paired = [key for key in keys if old[key].get("pnp_ok") and new[key].get("pnp_ok")]
        preserved = [key for key in paired if new[key].get("candidate_count", 0) >= old[key].get("candidate_count", 0)]
        groups[split] = {
            "total": len(keys),
            "before_all_solved": batch_residual_stats([old[k] for k in keys]),
            "after_all_solved_including_rejected": batch_residual_stats([new[k] for k in keys]),
            "paired_before": batch_residual_stats([old[k] for k in paired]),
            "paired_after": batch_residual_stats([new[k] for k in paired]),
            "paired_without_reduced_point_count_before": batch_residual_stats([old[k] for k in preserved]),
            "paired_without_reduced_point_count_after": batch_residual_stats([new[k] for k in preserved]),
            "before_legacy_status": {s: sum(old[k]["status"] == s for k in keys) for s in ("ok", "warn", "fail")},
            "after_legacy_status": {s: sum(new[k].get("legacy_status", new[k]["status"]) == s for k in keys) for s in ("ok", "warn", "fail")},
            "after_quality_status": {s: sum(new[k]["status"] == s for k in keys) for s in ("ok", "warn", "fail")},
            "before_full_nine": sum(old[k].get("candidate_count") == 9 for k in keys),
            "after_full_nine": sum(new[k].get("candidate_count") == 9 for k in keys),
            "before_total_matched_points": sum(old[k].get("candidate_count", 0) for k in keys),
            "after_total_matched_points": sum(new[k].get("candidate_count", 0) for k in keys),
            "recovered_solver_images": [k for k in keys if not old[k].get("pnp_ok") and new[k].get("pnp_ok")],
            "lost_solver_images": [k for k in keys if old[k].get("pnp_ok") and not new[k].get("pnp_ok")],
            "reduced_point_count_images": [k for k in keys if new[k].get("candidate_count", 0) < old[k].get("candidate_count", 0)],
        }
    return {"inputs_verified_identical": True, "independent_test_set": False,
            "unique_image_content_count": len(content_groups),
            "duplicate_content_groups": [names for names in content_groups.values() if len(names) > 1],
            "note": "Development comparison. Residuals include all matched points of all returned poses, including quality rejections. Point count is not correspondence ground truth.",
            "groups": groups,
            "per_image": [{"image": key, "before_points": old[key].get("candidate_count", 0),
                           "after_points": new[key].get("candidate_count", 0),
                           "before_mean_px": old[key].get("pnp_reproj_error_px"),
                           "after_mean_px": new[key].get("pnp_reproj_error_px"),
                           "after_rejections": new[key].get("pose_rejection_reasons", [])}
                          for key in old]}


def batch_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    before_bytes = args.before.read_bytes()
    report = compare(json.loads(before_bytes), json.loads(args.after.read_text(encoding="utf-8")),
                     json.loads(args.manifest.read_text(encoding="utf-8")),
                     baseline_sha256=hashlib.sha256(before_bytes).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0].startswith("-") and arguments[0] not in ("-h", "--help"):
        return annotations_main(arguments)
    commands = {"annotations": annotations_main, "labeled": labeled_main, "batch": batch_main}
    if arguments and arguments[0] in commands:
        return commands[arguments[0]](arguments[1:])
    parser = argparse.ArgumentParser(description="Visual development evaluation: annotations, labeled regression or batch comparison")
    parser.add_argument("command", choices=("annotations", "labeled", "batch"))
    parser.parse_args(arguments)


if __name__ == "__main__":
    main()
