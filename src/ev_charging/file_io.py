"""File fingerprints and atomic JSON writes shared across the project."""

import hashlib
import json
import os
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO


def sha256_file(path: str | Path) -> str:
    """Hash the file's original bytes without decoding or newline conversion."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump_json(stream: TextIO, data: object) -> None:
    json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.flush()
    os.fsync(stream.fileno())


def atomic_json(path: str | Path, data: object) -> None:
    """Replace the complete message, including invalid observations, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            _dump_json(stream, data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


_RETRY_DELAYS_S = (0.025, 0.05, 0.1, 0.2, 0.4)
_RETRYABLE_WINERRORS = {5, 32, 33}


def atomic_capture_json(
    path: str | Path,
    data: object,
    *,
    retry_delays: Sequence[float] = _RETRY_DELAYS_S,
    in_place_fallback: bool = False,
) -> None:
    """Retry temporary Windows sharing conflicts for capture files, outside control loops.

    A reader that does not share delete (antivirus, a watcher, an editor) blocks
    ``os.replace`` until it closes the file, so long-running, non-real-time
    callers that keep rewriting one path pass a longer ``retry_delays`` schedule.
    ``in_place_fallback`` is the last resort: overwrite in place rather than
    aborting the caller, trading atomicity for the survival of a long run.
    """
    for attempt in range(len(retry_delays) + 1):
        try:
            atomic_json(path, data)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in _RETRYABLE_WINERRORS:
                raise
            if attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])
                continue
            if not in_place_fallback:
                raise
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as stream:
                _dump_json(stream, data)
            return


# Long-lived audit files are rewritten for minutes; a scanner or watcher may
# hold them far longer than a control loop can wait.
PATIENT_RETRY_DELAYS_S = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)  # ~12.8 s


def atomic_persist_json(path: str | Path, data: object, *, in_place_fallback: bool = False) -> None:
    """Atomically replace a long-lived audit file, waiting out file scanners.

    Used by training reports, manifests and the model registry, which are
    rewritten for minutes while antivirus or a file watcher may hold them.
    ``in_place_fallback`` trades atomicity for durability of the run; only use it
    for files that are rewritten again soon.
    """
    atomic_capture_json(path, data, retry_delays=PATIENT_RETRY_DELAYS_S, in_place_fallback=in_place_fallback)
