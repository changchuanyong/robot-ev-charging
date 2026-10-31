"""Reviewed trajectory datasets; labels require independent evidence and splits stay disjoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .synergy import ACTIONS

FORCE_UNITS = ["N", "N", "N", "Nm", "Nm", "Nm"]
PARTITION_IDENTITIES = ("printing_batch_id", "specimen_id", "session_id")


def load_records(path):
    path = Path(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("trajectory_id", "batch_id", "timestamp", "wrench", "label", "frame", "units"):
            if field not in row:
                raise ValueError(f"line {line_number}: missing {field}")
        wrench = np.asarray(row["wrench"], dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError(f"line {line_number}: invalid wrench")
        if not np.isfinite(row["timestamp"]):
            raise ValueError(f"line {line_number}: invalid timestamp")
        if row["units"] != FORCE_UNITS or row["frame"] != "task":
            raise ValueError(f"line {line_number}: expected task-frame N/Nm wrench")
        if not str(row["trajectory_id"]) or not str(row["batch_id"]):
            raise ValueError(f"line {line_number}: empty trajectory/batch identity")
        if row["label"] not in ACTIONS and row["label"] not in ("ambiguous", "invalid", "fault"):
            raise ValueError(f"line {line_number}: unknown label {row['label']}")
        if row.get("schema_version") == 2:
            for field in PARTITION_IDENTITIES:
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"line {line_number}: schema 2 requires {field}")
        records.append(row)
    if not records:
        raise ValueError("force dataset is empty")
    return records


def validate_split(records, split):
    """Require all trajectories and complete acquisition batches to be disjoint."""
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("split needs train, validation and test trajectory lists")
    assignment = {}
    for group, identifiers in split.items():
        if not isinstance(identifiers, list) or not identifiers:
            raise ValueError(f"{group}: nonempty trajectory list required")
        for identifier in identifiers:
            if identifier in assignment:
                raise ValueError(f"trajectory leakage: {identifier}")
            assignment[identifier] = group
    trajectories = {r["trajectory_id"] for r in records}
    if set(assignment) != trajectories:
        raise ValueError("split must cover exactly the dataset trajectories")
    batch_groups, trajectory_batches, last_times = {}, {}, {}
    identity_groups, trajectory_identities = {}, {}
    for row in records:
        trajectory, batch = row["trajectory_id"], row["batch_id"]
        group = assignment[trajectory]
        for field in (*PARTITION_IDENTITIES, "task_id"):
            value = row.get(field)
            if row.get("schema_version") == 2 and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"schema 2 requires {field}")
            if value is None:
                continue  # Legacy datasets remain readable, without upgraded provenance claims.
            identity = (field, value)
            if identity in identity_groups and identity_groups[identity] != group:
                raise ValueError(f"{field} leakage: {value}")
            key = (trajectory, field)
            if key in trajectory_identities and trajectory_identities[key] != value:
                raise ValueError(f"trajectory changes {field}: {trajectory}")
            identity_groups[identity] = group
            trajectory_identities[key] = value
        if batch in batch_groups and batch_groups[batch] != group:
            raise ValueError(f"acquisition batch leakage: {batch}")
        if trajectory in trajectory_batches and trajectory_batches[trajectory] != batch:
            raise ValueError(f"trajectory spans batches: {trajectory}")
        if row["timestamp"] <= last_times.get(trajectory, -np.inf):
            raise ValueError(f"non-increasing trajectory time: {trajectory}")
        batch_groups[batch] = group
        trajectory_batches[trajectory] = batch
        last_times[trajectory] = row["timestamp"]
    return assignment


def make_windows(records, split, window_size, sample_period, max_gap, stride=1):
    """Use last-sample action labels, reject windows spanning gaps/invalid labels."""
    if window_size < 2 or stride < 1 or sample_period <= 0 or max_gap < sample_period:
        raise ValueError("invalid force window timing/size")
    assignment = validate_split(records, split)
    trajectories = {}
    for row in records:
        trajectories.setdefault(row["trajectory_id"], []).append(row)
    result = {g: {"x": [], "y": [], "trajectory_id": []} for g in split}
    for identifier, rows in trajectories.items():
        group = result[assignment[identifier]]
        for end in range(window_size, len(rows) + 1, stride):
            window = rows[end - window_size:end]
            gaps = np.diff([r["timestamp"] for r in window])
            # Sampling frequency is part of the model contract, not just the window length.
            if np.any(gaps > max_gap) or np.any(np.abs(gaps - sample_period) > 0.25 * sample_period):
                continue
            if any(r["label"] not in ACTIONS for r in window):
                continue
            group["x"].append([r["wrench"] for r in window])
            group["y"].append(ACTIONS.index(window[-1]["label"]))
            group["trajectory_id"].append(identifier)
    for name, group in result.items():
        if not group["x"]:
            raise ValueError(f"{name}: no valid force windows")
        group["x"] = np.asarray(group["x"], dtype=np.float32)
        group["y"] = np.asarray(group["y"], dtype=np.int64)
    return result


def fit_normalizer(training_windows):
    x = np.asarray(training_windows, dtype=float)
    if x.ndim != 3 or x.shape[2] != 6 or not np.isfinite(x).all():
        raise ValueError("expected finite N x window x 6 training data")
    mean = x.mean(axis=(0, 1))
    scale = x.std(axis=(0, 1))
    # Constant axes remain at zero after centering; never divide by tiny noise.
    scale = np.where(scale > 1e-6, scale, 1.0)
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def classification_metrics(labels, probabilities, confidence=0.7):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), len(ACTIONS)) or not len(labels):
        raise ValueError("invalid classification arrays")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("invalid class probabilities")
    if np.any((labels < 0) | (labels >= len(ACTIONS))) or not 0 <= confidence <= 1:
        raise ValueError("invalid labels or confidence")
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    per_class = {}
    for i, name in enumerate(ACTIONS):
        tp, true_count, predicted_count = matrix[i, i], matrix[i].sum(), matrix[:, i].sum()
        precision = float(tp / predicted_count) if predicted_count else 0.0
        recall = float(tp / true_count) if true_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(true_count)}
    accepted = probabilities.max(axis=1) >= confidence
    return {"count": len(labels), "accuracy": float(np.mean(predictions == labels)),
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "per_class": per_class, "confusion_matrix": matrix.tolist(),
            "classes": list(ACTIONS), "confidence": confidence,
            "acceptance_rate": float(accepted.mean()),
            "accepted_accuracy": float(np.mean(predictions[accepted] == labels[accepted])) if accepted.any() else None}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def in_interval(elapsed, interval):
    return interval["start_s"] - 1e-9 <= elapsed < interval["end_s"] - 1e-9


def trial_results(cycles):
    result = {}
    for row in cycles:
        trial = row["decision"].get("details", {}).get("trial_result") or {}
        if trial.get("trial_id"):
            result[trial["trial_id"]] = trial
    return result


def review_manifest(paths, output_path):
    """Prepare editable labels for pre-action history; suggestions are not truth."""
    output_path = Path(output_path).resolve()
    logs = []
    for path in paths:
        path = Path(path).resolve()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or rows[0].get("kind") != "manifest" or rows[0].get("rules_version") != "finite-trial-v2":
            raise ValueError("review generation needs finite-trial-v2 cycle logs")
        manifest = rows[0]
        cycles = [r for r in rows if r.get("kind") == "cycle"]
        if not cycles:
            raise ValueError("no controller cycles")
        origin = float(cycles[0]["at"])
        baseline_s = float(manifest["config"]["finite_trial"]["baseline_s"])
        learning = manifest["config"].get("learning", {})
        history_s = max(baseline_s, learning.get("window_size", 0) * learning.get("sample_period", 0.))
        intervals = []
        for trial_id, trial in trial_results(cycles).items():
            end = float(trial["started_at"]) - origin
            if end <= 0:
                continue
            intervals.append({"start_s": max(0., end - history_s), "end_s": end,
                              "label": "invalid", "basis": "pending independent review",
                              "reviewed": False, "reviewer": None, "trial_id": trial_id,
                              "suggested_action": trial.get("action"), "observed_outcome": trial.get("outcome")})
        experiment = manifest.get("experiment", {})
        logs.append({"path": str(path), "sha256": file_sha256(path),
                     "trajectory_id": manifest["task_id"], "batch_id": experiment.get("session_id"),
                     **{field: experiment.get(field) for field in PARTITION_IDENTITIES},
                     "intervals": intervals})
    result = {"schema_version": 2, "logs": logs,
              "note": "Review physical evidence before replacing invalid labels. Load decrease alone is insufficient. Intervals end before the tested action. Fault/ambiguous/invalid are excluded from training."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"output": str(output_path), "source_count": len(logs), "reviewed": False}


def validate_trial_label(interval, trials, cycles, origin, config):
    minimum_advance = float(config["finite_trial"]["min_advance_m"])
    label = interval["label"]
    if label in ("invalid", "ambiguous", "fault"):
        return
    if interval.get("reviewed") is not True or not interval.get("reviewer"):
        raise ValueError("schema 2 action labels require explicit independent human review")
    if label == "Hold":
        # Hold means that ordinary advancing was appropriate, never a sensor fault.
        selected = [r for r in cycles if in_interval(r["at"] - origin, interval)]
        depths = [r["decision"].get("details", {}).get("depth") for r in selected]
        if (len(depths) < 2 or any(v is None or not np.isfinite(v) for v in depths)
                or depths[-1] - depths[0] < minimum_advance
                or any(r["decision"].get("state") != "INSERT" or r["decision"].get("stop") for r in selected)
                or interval.get("loads_allowed") is not True):
            raise ValueError("Hold needs reviewed allowed-load forward progress, not an absence of action")
        for row in selected:
            details = row["decision"].get("details", {})
            wrench = np.asarray(details.get("wrench"), float)
            if (wrench.shape != (6,) or not np.isfinite(wrench).all()
                    or np.any(np.abs(wrench) >= np.asarray(config["limits"]["wrench"]) * config["scheduling"]["high"])
                    or details.get("advance_scale", 0) <= 0):
                raise ValueError("Hold requires logged allowed-load forward advancement")
        return
    trial = trials.get(interval.get("trial_id"))
    if (not trial or trial.get("action") != label or trial.get("outcome") != "effective"
            or trial.get("loads_allowed") is not True
            or not np.isfinite(trial.get("measured_advance_m", np.nan))
            or trial["measured_advance_m"] < minimum_advance):
        raise ValueError("direction labels require a matching effective trial with actual permitted advancement")
    if interval["end_s"] > float(trial["started_at"]) - origin + 1e-9:
        raise ValueError("direction training history must end before the action to avoid outcome leakage")
    learning = config.get("learning", {})
    history_s = max(float(config["finite_trial"]["baseline_s"]),
                    learning.get("window_size", 0) * learning.get("sample_period", 0.))
    trial_start = float(trial["started_at"]) - origin
    if abs(interval["end_s"] - trial_start) > 1e-9 or interval["start_s"] < trial_start - history_s - 1e-9:
        raise ValueError("direction labels must cover the bounded history immediately before their trial")
    selected = [r for r in cycles if in_interval(r["at"] - origin, interval)]
    if any((r["decision"].get("details", {}).get("finite_trial") or {}).get("phase") in
           ("TRIAL", "OBSERVE", "CONFIRM") for r in selected):
        raise ValueError("pre-action learning history overlaps an earlier trial or observation")


def export_dataset(annotation_path, output_path):
    annotation_path, output_path = Path(annotation_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"use a new dataset path: {output_path}")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    schema = annotations.get("schema_version")
    if schema not in (1, 2) or not annotations.get("logs"):
        raise ValueError("annotation manifest needs schema_version=1/2 and nonempty logs")
    provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    if provenance_path.exists():
        raise FileExistsError(f"use a new provenance path: {provenance_path}")
    result, sources, identifiers, source_hashes = [], [], set(), set()
    for source in annotations["logs"]:
        path = (annotation_path.parent / source["path"]).resolve()
        identifier, batch = source["trajectory_id"], source["batch_id"]
        if not identifier or not batch or identifier in identifiers:
            raise ValueError("unique trajectory IDs and nonempty batch IDs required")
        identifiers.add(identifier)
        source_hash = file_sha256(path)
        if source_hash in source_hashes:
            raise ValueError("duplicate source log cannot become a second independent trajectory")
        source_hashes.add(source_hash)
        if schema == 2:
            if source.get("sha256") != file_sha256(path):
                raise ValueError("source log hash changed since annotation")
            if any(not isinstance(source.get(field), str) or not source[field].strip() for field in PARTITION_IDENTITIES):
                raise ValueError("schema 2 needs printing_batch_id/specimen_id/session_id")
        intervals = sorted(source["intervals"], key=lambda item: item["start_s"])
        previous_end = -np.inf
        for interval in intervals:
            start, end = float(interval["start_s"]), float(interval["end_s"])
            if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < previous_end:
                raise ValueError("annotation intervals must be finite, positive-length and non-overlapping")
            if interval["label"] not in (*ACTIONS, "ambiguous", "invalid", "fault") or not interval.get("basis"):
                raise ValueError("every label needs an allowed class and independent evidence/basis")
            previous_end = end
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifests = [row for row in rows if row.get("kind") == "manifest"]
        if len(manifests) != 1 or manifests[0].get("clock") != "host_monotonic_seconds":
            raise ValueError(f"invalid/versionless source log: {path}")
        cycles = [row for row in rows if row.get("kind") == "cycle"]
        if not cycles:
            raise ValueError("use transformed control-cycle logs; raw sensor-only logs need task-frame calibration first")
        manifest = manifests[0]
        hardware_probe = manifest.get("source") == "hardware" and (
            manifest.get("config", {}).get("supervised_probe") is not None
            or manifest.get("method_identity") == "PROBE")
        if hardware_probe and any(interval["label"] in ACTIONS for interval in intervals):
            terminal = [row for row in rows if row.get("kind") == "result"]
            if not terminal or terminal[-1].get("stop_feedback_confirmed") is not True:
                raise ValueError("positive hardware probe labels require terminal measured stop confirmation")
        origin = float(cycles[0]["at"])
        if schema == 2:
            if manifests[0].get("rules_version") != "finite-trial-v2":
                raise ValueError("schema 2 labels require finite-trial-v2 source logs")
            minimum_advance = float(manifests[0]["config"]["finite_trial"]["min_advance_m"])
            if not np.isfinite(minimum_advance) or minimum_advance <= 0:
                raise ValueError("finite trial progress threshold must be frozen and positive")
            trials = trial_results(cycles)
            for interval in intervals:
                validate_trial_label(interval, trials, cycles, origin, manifests[0]["config"])
        emitted = 0
        for row in cycles:
            details = row["decision"].get("details", {})
            if not row["sample"].get("valid") or "filtered_wrench" not in details:
                continue
            wrench = np.asarray(details["filtered_wrench"], dtype=float)
            if wrench.shape != (6,) or not np.isfinite(wrench).all():
                raise ValueError(f"invalid transformed wrench in {path}")
            elapsed = float(row["at"]) - origin
            matched = [item for item in intervals if in_interval(elapsed, item)]
            interval = matched[0] if matched else None
            result.append({"trajectory_id": identifier, "batch_id": batch, "timestamp": float(row["at"]),
                           "schema_version": schema,
                           **({field: source[field] for field in PARTITION_IDENTITIES} if schema == 2 else {}),
                           "wrench": wrench.tolist(), "label": interval["label"] if interval else "invalid",
                           "label_basis": interval["basis"] if interval else "unreviewed",
                           "frame": "task", "units": FORCE_UNITS,
                           "synthetic": bool(manifests[0].get("synthetic", False)),
                           "task_id": row["task_id"], "attempt_id": row["attempt_id"],
                           "trial_id": interval.get("trial_id") if interval else None,
                           "condition": manifests[0].get("experiment", {}).get("condition"),
                           "pair_id": manifests[0].get("experiment", {}).get("pair_id"),
                           "rules_version": manifests[0].get("rules_version", "legacy-unversioned"),
                           "config_sha256": manifests[0]["config_sha256"],
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
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": schema, "annotation_sha256": file_sha256(annotation_path), "sources": sources,
                   "data_sha256": file_sha256(output_path), "samples": len(result)}, stream, ensure_ascii=False, indent=2)
    return {"output": str(output_path), "samples": len(result), "source_count": len(sources)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--annotations", type=Path)
    mode.add_argument("--review-logs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    result = (review_manifest(args.review_logs, args.output) if args.review_logs else
              export_dataset(args.annotations, args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
