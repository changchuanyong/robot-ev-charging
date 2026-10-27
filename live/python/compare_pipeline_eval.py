"""Compare development-image evaluations without hiding failed or rejected poses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def image_key(row):
    return row["image"].replace("\\", "/")


def split_name(row):
    parts = image_key(row).split("/")
    return row.get("split", next((name for name in ("train", "val", "test") if name in parts), "unknown"))


def residual_stats(rows):
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
            "before_all_solved": residual_stats([old[k] for k in keys]),
            "after_all_solved_including_rejected": residual_stats([new[k] for k in keys]),
            "paired_before": residual_stats([old[k] for k in paired]),
            "paired_after": residual_stats([new[k] for k in paired]),
            "paired_without_reduced_point_count_before": residual_stats([old[k] for k in preserved]),
            "paired_without_reduced_point_count_after": residual_stats([new[k] for k in preserved]),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before_bytes = args.before.read_bytes()
    report = compare(json.loads(before_bytes), json.loads(args.after.read_text(encoding="utf-8")),
                     json.loads(args.manifest.read_text(encoding="utf-8")),
                     baseline_sha256=hashlib.sha256(before_bytes).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
