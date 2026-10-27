"""Task-level insertion statistics. Attempts and high-rate samples are not trials."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .replay import load_log


def wilson(successes, count, z=1.959963984540054):
    if not count:
        return None
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def task_metrics(path):
    rows = load_log(path)
    manifest = rows[0]
    cycles = [r for r in rows if r.get("kind") == "cycle"]
    results = [r for r in rows if r.get("kind") == "result"]
    if not cycles:
        raise ValueError(f"{path}: no insertion cycles (collection-only logs are not trials)")
    result = results[-1] if results else {"success": False, "reason": "incomplete_log"}
    loads, intervals, ages, pauses, limited = [], [], [], 0., 0
    for row in cycles:
        d = row["decision"]["details"]
        if d.get("dt") is not None:
            intervals.append(d["dt"])
        ages.append(row["at"] - row["sample"]["timestamp"])
        if d.get("wrench") is not None:
            a = np.asarray(d["wrench"], float)
            if a.shape == (6,) and np.isfinite(a).all():
                loads.append(a)
        if row["decision"]["state"] == "INSERT" and d.get("advance_scale") == 0:
            pauses += d.get("dt", 0.)
        limited += int(d.get("limited", False))
    peaks = None
    if loads:
        a = np.asarray(loads)
        peaks = {"axial_N": float(np.abs(a[:, 2]).max()),
                 "lateral_N": float(np.linalg.norm(a[:, :2], axis=1).max()),
                 "moment_Nm": float(np.linalg.norm(a[:, 3:], axis=1).max())}
    return {"task_id": manifest["task_id"], "method": manifest["config"]["method"],
            "source": manifest["source"], "target_id": manifest["config"]["target_id"],
            "config_sha256": manifest["config_sha256"], "success": result.get("success") is True,
            "reason": result.get("reason"), "duration": cycles[-1]["at"] - cycles[0]["at"],
            "peak_current_wrench_at_tool_in_task_frame": peaks, "pause_seconds": pauses,
            "limited_fraction": limited / len(cycles),
            "period_p50_p95_max": np.percentile(intervals, [50,95,100]).tolist() if intervals else None,
            "sample_age_max": max(ages), "path": str(path)}


def evaluate(paths):
    tasks = [task_metrics(p) for p in paths]
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate task IDs: repeated files/attempts cannot inflate sample size")
    groups = {}
    for t in tasks:
        key = (t["source"], t["target_id"], t["method"])
        groups.setdefault(key, []).append(t)
    summaries = []
    for (source, target, method), group in groups.items():
        count, successes = len(group), sum(t["success"] for t in group)
        durations = [t["duration"] for t in group]
        summaries.append({"source": source, "target_id": target, "method": method,
                          "tasks": count, "successes": successes, "success_rate": successes/count,
                          "wilson_95": wilson(successes,count), "duration_mean": float(np.mean(durations)),
                          "duration_sample_std": float(np.std(durations,ddof=1)) if count>1 else None})
    return {"groups": summaries, "tasks": tasks,
            "note": "Simulation and hardware are separate. E/F comparisons require matched conditions and frozen configurations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args.logs)
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        print(text)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Evaluation rejected: {exc}\n")


if __name__ == "__main__":
    main()
