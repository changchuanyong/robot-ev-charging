"""Append-only task records; a command and its actual feedback are separate."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid

import numpy as np


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def finite_json(value):
    """Invalid sensor numbers remain null in a valid JSON failure record."""
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    return value


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in ("control", "vision", "live/python", "train/python"):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class TaskLog:
    def __init__(self, path, config, source):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite an earlier experiment, even when reusing a command.
        self.stream = self.path.open("x", encoding="utf-8", buffering=1)
        self.task_id = uuid.uuid4().hex
        self.attempt_id = self.task_id + "-1"
        config_text = json.dumps(config, sort_keys=True, default=_json, allow_nan=False)
        try:
            revision = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"],
                                      cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                      text=True, timeout=3, check=False).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            revision = "unknown"
        self.write({"kind": "manifest", "source": source, "synthetic": source == "simulation",
                    "created_utc": time.time(), "config": config,
                    "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                    "git_revision": revision, "source_sha256": source_digest(), "python": platform.python_version(),
                    "clock": "host_monotonic_seconds", "schema_version": 1})

    def write(self, record):
        row = {"task_id": self.task_id, "attempt_id": self.attempt_id, **record}
        self.stream.write(json.dumps(finite_json(row), ensure_ascii=False, default=_json, allow_nan=False) + "\n")

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
