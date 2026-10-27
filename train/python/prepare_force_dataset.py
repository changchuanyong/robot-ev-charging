"""Export force windows from control logs using independently reviewed interval labels.

Labels describe the action verified by bounded-action experiments. They are not
copied from the online GRU decision. Unlabelled intervals stay 'invalid', which
prevents a training window from silently spanning unreviewed contact states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from control.force_dataset import FORCE_UNITS, file_sha256
from control.synergy import ACTIONS


def export_dataset(annotation_path, output_path):
    annotation_path, output_path = Path(annotation_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"use a new dataset path: {output_path}")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotations.get("schema_version") != 1 or not annotations.get("logs"):
        raise ValueError("annotation manifest needs schema_version=1 and nonempty logs")
    result, sources, identifiers = [], [], set()
    for source in annotations["logs"]:
        path = (annotation_path.parent / source["path"]).resolve()
        identifier, batch = source["trajectory_id"], source["batch_id"]
        if not identifier or not batch or identifier in identifiers:
            raise ValueError("unique trajectory IDs and nonempty batch IDs required")
        identifiers.add(identifier)
        intervals = sorted(source["intervals"], key=lambda item: item["start_s"])
        previous_end = -np.inf
        for interval in intervals:
            start, end = float(interval["start_s"]), float(interval["end_s"])
            if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < previous_end:
                raise ValueError("annotation intervals must be finite, positive-length and non-overlapping")
            if interval["label"] not in (*ACTIONS, "ambiguous", "invalid") or not interval.get("basis"):
                raise ValueError("every label needs an allowed class and independent evidence/basis")
            previous_end = end
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifests = [row for row in rows if row.get("kind") == "manifest"]
        if len(manifests) != 1 or manifests[0].get("clock") != "host_monotonic_seconds":
            raise ValueError(f"invalid/versionless source log: {path}")
        cycles = [row for row in rows if row.get("kind") == "cycle"]
        if not cycles:
            raise ValueError("use transformed control-cycle logs; raw sensor-only logs need task-frame calibration first")
        origin = float(cycles[0]["at"])
        emitted = 0
        for row in cycles:
            details = row["decision"].get("details", {})
            if not row["sample"].get("valid") or "filtered_wrench" not in details:
                continue
            wrench = np.asarray(details["filtered_wrench"], dtype=float)
            if wrench.shape != (6,) or not np.isfinite(wrench).all():
                raise ValueError(f"invalid transformed wrench in {path}")
            elapsed = float(row["at"]) - origin
            matched = [item for item in intervals if item["start_s"] <= elapsed < item["end_s"]]
            interval = matched[0] if matched else None
            result.append({"trajectory_id": identifier, "batch_id": batch, "timestamp": float(row["at"]),
                           "wrench": wrench.tolist(), "label": interval["label"] if interval else "invalid",
                           "label_basis": interval["basis"] if interval else "unreviewed",
                           "frame": "task", "units": FORCE_UNITS,
                           "synthetic": bool(manifests[0].get("synthetic", False)),
                           "task_id": row["task_id"], "attempt_id": row["attempt_id"],
                           "source_seq": row["sample"]["seq"]})
            emitted += 1
        if not emitted:
            raise ValueError(f"no usable force samples in {path}")
        sources.append({"path": str(path), "sha256": file_sha256(path), "trajectory_id": identifier,
                        "batch_id": batch, "samples": emitted, "config_sha256": manifests[0]["config_sha256"]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        for row in result:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump({"annotation_sha256": file_sha256(annotation_path), "sources": sources,
                   "data_sha256": file_sha256(output_path), "samples": len(result)}, stream, ensure_ascii=False, indent=2)
    return {"output": str(output_path), "samples": len(result), "source_count": len(sources)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export_dataset(args.annotations, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
