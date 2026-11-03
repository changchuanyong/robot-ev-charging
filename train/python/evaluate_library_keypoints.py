"""Diagnose reviewed library hole centers in original pixels, without fitting to them."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from live.python.keypoint_annotations import validate_annotations
from vision.review.review_keypoints import KeypointStore

SOURCES = ("vision/keypoint_suggestions.py", "live/python/observed_layout.py",
           "live/python/latest_candidate_contours.py", "live/python/center_fit.py", "live/python/enhance.py",
           "vision/review_apertures.py")


def topology_warnings(reference):
    """Flag side-of-DC-axis contradictions; do not relabel user observations."""
    points = {name: np.asarray(point["xy"]) for name, point in reference["points"].items()
              if point["visibility"] == "visible"}
    if not {"DC_neg", "DC_pos", "PE"} <= points.keys():
        return []
    axis = points["DC_pos"] - points["DC_neg"]
    scale2 = float(np.dot(axis, axis))
    if scale2 < 400:
        return []
    origin = (points["DC_pos"] + points["DC_neg"])/2
    normal = np.array([-axis[1], axis[0]]) / scale2
    pe_side = float(np.dot(points["PE"]-origin, normal))
    if abs(pe_side) < .1:
        return []
    warnings = []
    for name in ("S_neg", "CC2", "S_pos", "CC1", "A_neg", "A_pos"):
        if name not in points:
            continue
        side = float(np.dot(points[name]-origin, normal)) * np.sign(pe_side)
        if ((name.startswith("S_") or name.startswith("CC")) and side > .1
                or name.startswith("A_") and side < -.1):
            warnings.append(name)
    return warnings


def point_diagnostic(xy, prediction, trace, tolerance):
    if prediction is not None:
        distance = float(np.linalg.norm(np.asarray(prediction) - xy))
        return {"stage": "matched" if distance <= tolerance else "center_or_identity_error", "error_px": distance}
    boxes = trace.get("detected_boxes", [])
    if len(boxes) != 1:
        return {"stage": "detector_missing" if not boxes else "detector_ambiguous", "error_px": None}
    roi = trace["roi_bbox_xyxy"]
    if not (roi[0] <= xy[0] < roi[2] and roi[1] <= xy[1] < roi[3]):
        return {"stage": "outside_detected_roi", "error_px": None}
    nearby = []
    for item in trace.get("contours", {}).get("candidate_trace", {}).get("outcomes", []):
        center = item.get("measurement_center") or item.get("fitted_center") or item.get("center")
        if center is not None:
            distance = float(np.linalg.norm(np.asarray(center) + roi[:2] - xy))
            if distance <= tolerance:
                nearby.append((distance, item))
    kept = sorted((pair for pair in nearby if pair[1]["stage"] == "kept"), key=lambda pair: pair[0])
    if kept:
        distance, item = kept[0]
        stage = "kept_without_aperture" if item.get("aperture_center") is None else "kept_not_named"
    elif nearby:
        distance, item = min(nearby, key=lambda pair: pair[0])
        stage = "filtered_" + str(item.get("reason") or item["stage"])
    else:
        return {"stage": "no_nearby_candidate", "error_px": None}
    return {"stage": stage, "error_px": None, "nearest_candidate_px": distance,
            "candidate_stage": item["stage"], "candidate_reason": item.get("reason")}


def summarize(rows, tolerance):
    points = [point for row in rows for point in row["points"].values()]
    summary = {"reviewed_images": len(rows), "visible_points": len(points),
               "named_predictions": sum(point["error_px"] is not None for point in points),
               "within_tolerance": sum(point["stage"] == "matched" for point in points),
               "zero_prediction_images": sum(not row["prediction"]["points"] for row in rows),
               "stage_counts": dict(Counter(point["stage"] for point in points)),
               "diagnostic_tolerance_px": tolerance}
    summary["topology_suspect_images"] = [{"image": row["image"], "labels": row.get("topology_warnings", [])}
                                          for row in rows if row.get("topology_warnings")]
    summary["predictions_on_nonvisible_references"] = sum(len(row.get("nonvisible_predictions", [])) for row in rows)
    for source in ("human", "ai_visual"):
        subset = [point for point in points if point["reference_source"] == source]
        errors = [point["error_px"] for point in subset if point["error_px"] is not None]
        summary[source] = {"visible": len(subset), "named_predictions": len(errors),
                           "within_tolerance": sum(point["stage"] == "matched" for point in subset),
                           "mean_error_px": float(np.mean(errors)) if errors else None,
                           "median_error_px": float(np.median(errors)) if errors else None,
                           "max_error_px": max(errors) if errors else None}
    return summary


def compare_reports(before, after):
    if (before["annotation_sha256"] != after["annotation_sha256"]
            or before["model"]["weights_sha256"] != after["model"]["weights_sha256"]
            or before["summary"]["diagnostic_tolerance_px"] != after["summary"]["diagnostic_tolerance_px"]):
        raise ValueError("comparison_requires_same_annotations_weights_and_tolerance")
    def matched(report):
        return {(row["image_sha256"], label) for row in report["rows"]
                for label, point in row["points"].items() if point["stage"] == "matched"}
    old, new = matched(before), matched(after)
    return {"independent_test_set": False, "before": before["summary"], "after": after["summary"],
            "recovered_within_tolerance": [{"image_sha256": digest, "label": label} for digest, label in sorted(new-old)],
            "lost_within_tolerance": [{"image_sha256": digest, "label": label} for digest, label in sorted(old-new)],
            "note": "Same-image development regression, not held-out validation; unchanged automatic references are not independent accuracy evidence."}


def write_comparison_page(before, after, baseline_path, output_dir):
    comparison = compare_reports(before, after)
    stages = {"matched": "5 像素内", "center_or_identity_error": "中心或孔名偏差", "detector_missing": "未检测到充电口",
              "detector_ambiguous": "多个检测框", "outside_detected_roi": "参考点在检测区外",
              "kept_without_aperture": "缺少内孔证据", "kept_not_named": "候选未匹配孔名",
              "no_nearby_candidate": "附近没有候选", "filtered_dark_invalid_fit": "暗区椭圆拟合未通过",
              "filtered_max_keep": "候选数量上限", "filtered_duplicate": "合并去重", "filtered_circularity": "圆度筛选"}
    old_rows = {row["image_sha256"]: (index, row) for index, row in enumerate(before["rows"], 1)}
    cards = []
    for index, row in enumerate(after["rows"], 1):
        old_index, old = old_rows[row["image_sha256"]]
        old_image = Path(os.path.relpath(Path(baseline_path).parent / old["overlay"], output_dir)).as_posix()
        lines = []
        for label, point in row["points"].items():
            distance = "—" if point["error_px"] is None else f"{point['error_px']:.2f}"
            lines.append(f"<tr><td>{label}</td><td>{stages.get(old['points'][label]['stage'],old['points'][label]['stage'])}</td>"
                         f"<td>{stages.get(point['stage'],point['stage'])}</td><td>{distance}</td></tr>")
        warning = ("<p class='warn'>孔名拓扑待检查：" + html.escape(", ".join(row["topology_warnings"])) + "。原标注未改动。</p>"
                   if row.get("topology_warnings") else "")
        if row.get("nonvisible_predictions"):
            warning += "<p class='warn'>算法还在参考标注非可见/不确定的孔名上输出候选：" + html.escape(", ".join(row["nonvisible_predictions"])) + "，需人工检查，未计入恢复数量。</p>"
        cards.append(f"<details><summary>图 {index:02d} · {html.escape(Path(row['image']).name[:12])} · "
                     f"{sum(p['stage']=='matched' for p in old['points'].values())} → {sum(p['stage']=='matched' for p in row['points'].values())} 个孔在 5 像素内</summary>"
                     f"{warning}<div class='pair'><figure><figcaption>修改前</figcaption><img src='{html.escape(old_image)}'></figure>"
                     f"<figure><figcaption>修改后</figcaption><img src='{row['overlay']}'></figure></div>"
                     "<table><tr><th>孔名</th><th>修改前</th><th>修改后</th><th>误差 / px</th></tr>" + ''.join(lines) + "</table></details>")
    a, b = before["summary"], after["summary"]
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>孔位算法改进对比</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui,'Microsoft YaHei';color:#172b3a;background:#f3f6f8}h1{margin-bottom:8px}.note{color:#536778}.warn{color:#9b5200}details,header{background:white;padding:20px;margin-bottom:16px;border-radius:12px}summary{cursor:pointer;font-weight:600}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:12px 0}img{max-width:100%}table{width:100%;border-collapse:collapse}th,td{border-bottom:1px solid #dbe3e9;padding:8px;text-align:left}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>"
            f"<header><h1>孔位算法改进对比</h1><p>{b['reviewed_images']} 张已复核图 · {b['visible_points']} 个可见孔 · 相同参考标注与权重</p>"
            "<table><tr><th>指标</th><th>修改前</th><th>修改后</th></tr>"
            f"<tr><td>同名候选且误差 ≤ {b['diagnostic_tolerance_px']:g} px</td><td>{a['within_tolerance']}</td><td>{b['within_tolerance']}</td></tr>"
            f"<tr><td>人工编辑参考点中恢复的孔</td><td>{a['human']['within_tolerance']}</td><td>{b['human']['within_tolerance']}</td></tr>"
            f"<tr><td>完全无输出的图片</td><td>{a['zero_prediction_images']}</td><td>{b['zero_prediction_images']}</td></tr></table>"
            f"<p>新增恢复 {len(comparison['recovered_within_tolerance'])} 个，原有结果退化 {len(comparison['lost_within_tolerance'])} 个。</p>"
            "<p class='note'>绿色圆圈为已复核参考点，橙色十字为算法候选。此批用于开发调整，不是独立验证集；5 px 是诊断阈值，不是验收标准。原样接受的自动点不能作为独立精度证据。</p></header>"
            + ''.join(cards) + "</html>")
    with (Path(output_dir) / "comparison.html").open("x", encoding="utf-8") as stream:
        stream.write(page)


def write_overlay(image, row, path):
    trace = row["prediction"].get("diagnostics", {})
    roi = trace.get("roi_bbox_xyxy", [0, 0, image.shape[1], image.shape[0]])
    view = image[roi[1]:roi[3], roi[0]:roi[2]].copy()
    for name, point in row["points"].items():
        xy = tuple(np.rint(np.asarray(point["reference_xy"]) - roi[:2]).astype(int))
        cv2.circle(view, xy, 5, (50, 220, 60), 2)
        cv2.putText(view, name, (xy[0]+7, xy[1]-7), cv2.FONT_HERSHEY_SIMPLEX, .4, (50, 220, 60), 1, cv2.LINE_AA)
    for name, point in row["prediction"]["points"].items():
        xy = tuple(np.rint(np.asarray(point) - roi[:2]).astype(int))
        cv2.drawMarker(view, xy, (0, 140, 255), cv2.MARKER_CROSS, 12, 2)
    ok, encoded = cv2.imencode(".png", view)
    if not ok:
        raise ValueError("overlay_encoding_failed")
    path.write_bytes(encoded.tobytes())


def evaluate(annotations_path, output_dir, *, tolerance=5., root=ROOT):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    content = Path(annotations_path).read_bytes()
    annotations = json.loads(content)
    validate_annotations(annotations)
    if annotations["schema_version"] != 3:
        raise ValueError("library_annotation_schema_3_required")
    reviewed = [row for row in annotations["images"] if row["review_status"] == "reviewed"]
    if not reviewed:
        raise ValueError("no_reviewed_images")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive_diagnostic_tolerance_required")
    store = KeypointStore(root / annotations["provenance"]["library_path"], root=root)
    model = store.suggestion_engine.current_model()
    for row in reviewed:
        original = store.by_hash[row["image_sha256"]]
        if any(original[key] != row[key] for key in ("image", "width", "height")):
            raise ValueError("reviewed_image_binding_mismatch")
        store.image(row["image_sha256"])
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "annotations.json").write_bytes(content)
    (output_dir / "overlays").mkdir()
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    rows = []
    for index, reference in enumerate(reviewed, 1):
        digest = reference["image_sha256"]
        image = store.image(digest)
        entry = json.loads((store.library / "records" / (digest + ".json")).read_text(encoding="utf-8"))
        prediction = store.suggestion_engine.suggest(entry, image, model=model, diagnostics=True)
        points = {}
        for name, point in reference["points"].items():
            if point["visibility"] == "visible":
                points[name] = {"reference_xy": point["xy"], "reference_source": point["source"],
                                **point_diagnostic(point["xy"], prediction["points"].get(name),
                                                   prediction.get("diagnostics", {}), tolerance)}
        row = {"image": reference["image"], "image_sha256": digest, "points": points, "prediction": prediction,
               "topology_warnings": topology_warnings(reference),
               "nonvisible_predictions": [name for name in prediction["points"]
                                           if reference["points"][name]["visibility"] != "visible"],
               "overlay": f"overlays/{index:02d}.png"}
        rows.append(row)
        write_overlay(image, row, output_dir / row["overlay"])
        print(f"{index}/{len(reviewed)} {digest[:12]}: {len(prediction['points'])}/{len(points)} named", flush=True)
    report = {"schema_version": 1, "kind": "library_keypoint_diagnostic", "independent_test_set": False,
              "annotation_sha256": hashlib.sha256(content).hexdigest(), "model": model, "code_sha256": hashes,
              "note": "Development diagnostic only. These images may be used for tuning; accepted automatic points are not independent center measurements. The pixel threshold is diagnostic, not an acceptance specification. Nearby candidate stages are attribution hints, not proof of causation.",
              "summary": summarize(rows, tolerance), "rows": rows}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    cards = []
    for index, row in enumerate(rows, 1):
        stats = "; ".join(f"{name}: {point['stage']}" for name, point in row["points"].items())
        if row["topology_warnings"]:
            stats = "孔名拓扑待复核（不自动改标注）：" + ", ".join(row["topology_warnings"]) + "; " + stats
        cards.append(f'<section><h2>{index}. {html.escape(Path(row["image"]).name[:12])}</h2>'
                     f'<img src="{row["overlay"]}"><p>{html.escape(stats)}</p></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>孔位诊断报告</title>'
            '<style>body{font:16px system-ui;margin:32px;background:#f5f7fa;color:#182432}section{background:white;padding:20px;margin:16px 0}img{max-width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>孔位诊断报告</h1><p>绿色圆圈：已复核参考点；橙色十字：算法候选。自动点原样确认部分单列，不能据此证明独立精度。此批用于开发诊断，不是独立验收集。</p>'
            + '<pre>' + html.escape(json.dumps(report["summary"], ensure_ascii=False, indent=2)) + '</pre>' + ''.join(cards))
    (output_dir / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing reports are not overwritten")
    parser.add_argument("--tolerance-px", type=float, default=5.)
    parser.add_argument("--baseline", type=Path, help="Compare against an earlier report.json using identical annotations and weights")
    args = parser.parse_args()
    report = evaluate(args.annotations, args.output_dir, tolerance=args.tolerance_px)
    if args.baseline:
        before = json.loads(args.baseline.read_text(encoding="utf-8"))
        comparison = compare_reports(before, report)
        (args.output_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        write_comparison_page(before, report, args.baseline, args.output_dir)
        print(f"Recovered: {len(comparison['recovered_within_tolerance'])}; lost: {len(comparison['lost_within_tolerance'])}")


if __name__ == "__main__":
    main()
