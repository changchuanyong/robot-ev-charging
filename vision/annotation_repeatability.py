"""Compare two blind annotation passes before freezing a pixel evaluation tolerance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live" / "python"))
from keypoint_annotations import validate_annotations


def compare_passes(first: dict, second: dict, blind_protocol: dict) -> dict:
    for data in (first, second):
        validate_annotations(data)
        if any(image["review_status"] != "reviewed" for image in data["images"]):
            raise ValueError("both_passes_must_be_human_reviewed")
    if first["feature_definition"] != second["feature_definition"]:
        raise ValueError("feature_definition_must_match")
    if (blind_protocol.get("independent_blind_passes") is not True
            or not isinstance(blind_protocol.get("record"), str) or not blind_protocol["record"].strip()):
        raise ValueError("explicit_blind_annotation_protocol_record_required")
    a = {row["image_sha256"]: row for row in first["images"]}
    b = {row["image_sha256"]: row for row in second["images"]}
    if set(a) != set(b):
        raise ValueError("blind_passes_require_identical_image_contents")
    rows, distances, disagreements = [], [], []
    for digest in sorted(a):
        left, right = a[digest], b[digest]
        if (left["width"], left["height"]) != (right["width"], right["height"]):
            raise ValueError("annotation_image_dimensions_mismatch")
        if left["split"] != "train" or right["split"] != "train":
            raise ValueError("repeatability_tolerance_uses_development_train_images_only")
        points = {}
        for name in first["labels"]:
            p, q = left["points"][name], right["points"][name]
            if p["visibility"] != q["visibility"]:
                disagreements.append({"image_sha256": digest, "label": name,
                                      "first": p["visibility"], "second": q["visibility"]})
            if p["visibility"] == q["visibility"] == "visible":
                if p["source"] != "human" or q["source"] != "human":
                    raise ValueError("repeatability_requires_human_point_sources")
                delta = np.asarray(q["xy"])-np.asarray(p["xy"])
                distance = float(np.linalg.norm(delta))
                points[name] = {"delta_xy_px": delta.tolist(), "distance_px": distance}
                distances.append(distance)
        rows.append({"image": left["image"], "image_sha256": digest, "points": points})
    if not distances:
        raise ValueError("no_common_visible_points")
    values = np.asarray(distances)
    return {"schema_version": 1, "kind": "blind_annotation_repeatability", "source": "human_annotations",
            "image_count": len(rows), "planned_image_count": 20, "count_budget_met": len(rows) >= 20,
            "point_count": len(values), "mean_distance_px": float(values.mean()),
            "rms_distance_px": float(np.sqrt(np.mean(values**2))), "median_distance_px": float(np.median(values)),
            "p95_distance_px": float(np.quantile(values, .95)), "max_distance_px": float(values.max()),
            "visibility_disagreements": disagreements, "blind_protocol": blind_protocol,
            "tolerance_frozen": False, "tolerance_px": None, "rows": rows,
            "interpretation": "Inter-pass differences measure repeatability, not absolute accuracy. Review disagreements and freeze a separate tolerance before held-out evaluation."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--blind-record", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    first = json.loads(args.first.read_text(encoding="utf-8-sig"))
    second = json.loads(args.second.read_text(encoding="utf-8-sig"))
    report = compare_passes(first, second, {"independent_blind_passes": True, "record": args.blind_record})
    report["input_sha256"] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.first, args.second)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("image_count", "point_count", "p95_distance_px", "tolerance_frozen")}))


if __name__ == "__main__":
    main()
