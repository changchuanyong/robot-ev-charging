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
    if (manifest["config"].get("supervised_probe") is not None
            or manifest.get("experiment", {}).get("calibration_only") is True):
        raise ValueError(f"{path}: supervised probes are calibration-only, excluded from insertion method statistics and the 520 tasks")
    cycles = [r for r in rows if r.get("kind") == "cycle"]
    results = [r for r in rows if r.get("kind") == "result"]
    if (any(r.get("kind") == "device" and r.get("collect_only") for r in rows)
            or (not cycles and any(r.get("kind") == "sample" for r in rows))
            or any(r.get("decision", {}).get("state") == "COLLECT" for r in cycles)):
        raise ValueError(f"{path}: no insertion cycles (collection-only logs are not trials)")
    result = results[-1] if results else {"success": False, "reason": "incomplete_log"}
    if not cycles and not results:
        result = {"success": False, "reason": "incomplete_log_before_feedback"}
    loads, intervals, ages, pauses, limited = [], [], [], 0., 0
    trials, vision_checks, vision_rejects = {}, 0, 0
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
        for name in ("finite_trial", "trial_result"):
            trial = d.get(name) or {}
            if trial.get("trial_id"):
                trials.setdefault(trial["trial_id"], {}).update(trial)
        if row.get("observation") is not None:
            vision_checks += 1
            vision_rejects += int(row["observation"].get("valid") is not True
                                  or row["decision"].get("reason", "").startswith("vision_"))
    peaks = None
    if loads:
        a = np.asarray(loads)
        peaks = {"axial_N": float(np.abs(a[:, 2]).max()),
                 "lateral_N": float(np.linalg.norm(a[:, :2], axis=1).max()),
                 "moment_Nm": float(np.linalg.norm(a[:, 3:], axis=1).max())}
    experiment = manifest.get("experiment", manifest["config"].get("experiment", {}))
    outcomes = {}
    for trial in trials.values():
        outcome = trial.get("outcome") or "unconfirmed"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    duration = result.get("duration")
    if duration is None and cycles:
        duration = cycles[-1]["at"] - cycles[0]["at"]
    return {"task_id": manifest["task_id"], "method": "C" if manifest["config"]["method"] == "E" else manifest["config"]["method"],
            "recorded_method": manifest["config"]["method"],
            "source": manifest["source"], "target_id": manifest["config"]["target_id"],
            "config_sha256": manifest["config_sha256"], "success": result.get("success") is True,
            "rules_version": manifest.get("rules_version", "legacy-unversioned"),
            "source_sha256": manifest.get("source_sha256"), "model_sha256": manifest.get("model_sha256"),
            "protocol_sha256": experiment.get("protocol_sha256"), "condition": experiment.get("condition"),
            "specimen_id": experiment.get("specimen_id"), "printing_batch_id": experiment.get("printing_batch_id"),
            "session_id": experiment.get("session_id"), "pair_id": experiment.get("pair_id"),
            "wear_cycles_before": experiment.get("wear_cycles_before"), "fixture_event": experiment.get("fixture_event"),
            "formal": experiment.get("formal", False),
            "method_config_sha256": experiment.get("method_config_sha256", manifest["config_sha256"]),
            "reason": result.get("reason"), "duration": duration,
            "peak_current_wrench_at_tool_in_task_frame": peaks, "pause_seconds": pauses,
            "limited_fraction": limited / len(cycles) if cycles else None,
            "finite_trials": len(trials), "trial_outcomes": outcomes,
            "trial_effectiveness": outcomes.get("effective", 0) / len(trials) if trials else None,
            "visual_checks": vision_checks, "visual_rejects": vision_rejects,
            "visual_reject_rate": vision_rejects / vision_checks if vision_checks else None,
            "period_p50_p95_max": np.percentile(intervals, [50,95,100]).tolist() if intervals else None,
            "sample_age_max": max(ages) if ages else None, "path": str(path)}


def _summarize(tasks, fields):
    groups = {}
    for task in tasks:
        groups.setdefault(tuple(task[field] for field in fields), []).append(task)
    summaries = []
    for key, group in groups.items():
        count, successes = len(group), sum(t["success"] for t in group)
        durations = [t["duration"] for t in group if t["duration"] is not None]
        trials = sum(t["finite_trials"] for t in group)
        effective = sum(t["trial_outcomes"].get("effective", 0) for t in group)
        checks, rejects = sum(t["visual_checks"] for t in group), sum(t["visual_rejects"] for t in group)
        peaks = [t["peak_current_wrench_at_tool_in_task_frame"] for t in group if t["peak_current_wrench_at_tool_in_task_frame"]]
        reasons = {}
        for task in group:
            reason = task["reason"] or "unspecified"
            reasons[reason] = reasons.get(reason, 0) + 1
        summaries.append({**dict(zip(fields, key)), "tasks": count, "successes": successes,
                          "success_rate": successes / count, "wilson_95": wilson(successes, count),
                          "duration_count": len(durations),
                          "duration_mean": float(np.mean(durations)) if durations else None,
                          "duration_sample_std": float(np.std(durations, ddof=1)) if len(durations) > 1 else None,
                          "peak_wrench_max": {axis: max(p[axis] for p in peaks) for axis in peaks[0]} if peaks else None,
                          "finite_trials": trials, "effective_trials": effective,
                          "trial_effectiveness": effective / trials if trials else None,
                          "visual_checks": checks, "visual_rejects": rejects,
                          "visual_reject_rate": rejects / checks if checks else None, "outcomes": reasons})
    return summaries


def evaluate(paths):
    tasks = [task_metrics(p) for p in paths]
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate task IDs: repeated files/attempts cannot inflate sample size")
    paired = [(t["protocol_sha256"], t["pair_id"], t["method"]) for t in tasks if t["formal"]]
    if len(set(paired)) != len(paired):
        raise ValueError("duplicate formal method/block: E reuses C and retries cannot inflate the planned sample size")
    base = ("source", "target_id", "method", "rules_version", "protocol_sha256", "method_config_sha256", "source_sha256", "model_sha256")
    return {"schema_version": 2, "groups": _summarize(tasks, base + ("condition", "specimen_id")),
            "method_summary": _summarize(tasks, base), "tasks": tasks,
            "note": "Counts are complete task identities, including stops/timeouts/incomplete logs. E aliases C. Source, control rules, protocol and configurations remain separate; printed mechanical insertion only."}


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
