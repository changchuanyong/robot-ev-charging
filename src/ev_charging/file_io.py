"""Atomic JSON writes shared by capture, review and training."""
import json
import os
from pathlib import Path
import time
import uuid


def _dump_json(stream, data) -> None:
    json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.flush()
    os.fsync(stream.fileno())


def atomic_json(path: Path, data: dict) -> None:
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


_RETRY_DELAYS_S = (.025, .05, .1, .2, .4)
_RETRYABLE_WINERRORS = {5, 32, 33}


def atomic_capture_json(path: Path, data: dict, *, retry_delays=_RETRY_DELAYS_S,
                        in_place_fallback: bool = False) -> None:
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
PATIENT_RETRY_DELAYS_S = (.05, .1, .2, .4, .8, 1.6, 3.2, 6.4)  # ~12.8 s


def atomic_persist_json(path: Path, data: dict, *, in_place_fallback: bool = False) -> None:
    """Atomically replace a long-lived audit file, waiting out file scanners.

    Used by training reports, manifests and the model registry, which are
    rewritten for minutes while antivirus or a file watcher may hold them.
    ``in_place_fallback`` trades atomicity for durability of the run; only use it
    for files that are rewritten again soon.
    """
    atomic_capture_json(path, data, retry_delays=PATIENT_RETRY_DELAYS_S,
                        in_place_fallback=in_place_fallback)


