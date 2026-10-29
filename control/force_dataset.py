"""Trajectory-level force datasets; no window is allowed to cross a split/batch."""
from __future__ import annotations

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
