"""Bounded file-sharing retries for capture metadata, outside the control loop."""
from __future__ import annotations

from pathlib import Path
import time

from .contracts import atomic_json


_RETRY_DELAYS_S = (.025, .05, .1, .2, .4)
_RETRYABLE_WINERRORS = {5, 32, 33}


def atomic_capture_json(path: Path, data: dict) -> None:
    """Keep atomic replacement while tolerating short Windows sharing conflicts."""
    for attempt in range(len(_RETRY_DELAYS_S) + 1):
        try:
            atomic_json(path, data)
            return
        except PermissionError as exc:
            if (getattr(exc, "winerror", None) not in _RETRYABLE_WINERRORS
                    or attempt == len(_RETRY_DELAYS_S)):
                raise
            time.sleep(_RETRY_DELAYS_S[attempt])
