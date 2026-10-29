"""Compare vision outputs against reviewed hole labels without hiding missing points."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from evaluate_keypoint_annotations import validate_inputs, validate_output_path
from keypoint_annotations import LABELS, normalized_image_key, sha256_file, validate_annotations, visible_points


ROOT_DIR = Path(__file__).resolve().parents[2]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
