"""Keep only the latest owned RGB(D) pair while diagnostics process older frames."""
from __future__ import annotations

from collections import deque
from queue import Empty, Full, Queue
from threading import Condition, Event, Thread
import time


class PreviewCapture:
    """One SDK reader, one pending pair, and bounded sensor-control requests.

    The source camera must return owned arrays, as CaptureColorCamera and
    CaptureRGBDCamera do. Replacing the latest pair never mutates a held pair.
    The caller owns source startup/cleanup; this context owns only its reader.
    """

    def __init__(self, source):
        self.source = source
        self.camera = source.camera
        self.rgb_controls = getattr(source, "rgb_controls", None)
        self._has_depth = callable(getattr(source, "read_rgbd", None))
        self._read = source.read_rgbd if self._has_depth else source.read
        # run_preview detects the callable, preserving its RGB-only contract.
        if not self._has_depth:
            self.read_rgbd = None
        self._condition = Condition()
        self._stop = Event()
        self._commands = Queue(maxsize=8)
        self._thread = None
        self._latest = self._error = None
        self._version = self._consumed = self._count = 0
        self._previous_frame = None
        self._arrivals = deque(maxlen=120)
        self._first_at = self._last_at = None
        self.control_message = ""

    def __enter__(self):
        self._thread = Thread(target=self._run, name="d415-capture", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        try:
            while not self._stop.is_set():
                self._apply_commands()
                sample = self._read()
                now = time.monotonic()
                identity = (sample[1].get("frame_number"), sample[1].get("device_timestamp_ms"))
                with self._condition:
                    self._latest = sample
                    self._version += 1
                    if identity != self._previous_frame:
                        self._previous_frame = identity
                        self._arrivals.append(now)
                        self._first_at = now if self._first_at is None else self._first_at
                        self._last_at = now
                        self._count += 1
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def _next(self):
        timeout = self.source.config.get("timeout_ms", 5000) / 1000 + 1
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._error is not None or self._version > self._consumed
                or self._stop.is_set(), timeout=timeout)
            if self._error is not None:
                raise self._error
            if not ready or self._version <= self._consumed:
                raise RuntimeError("D415 acquisition did not return a new frame")
            self._consumed = self._version
            return self._latest

    def read(self):
        return self._next()[:2]

    def read_rgbd(self):
        return self._next()

    def capture_stats(self):
        with self._condition:
            span = self._arrivals[-1] - self._arrivals[0] if len(self._arrivals) > 1 else 0
            total_span = (self._last_at - self._first_at) if self._count > 1 else 0
            return {"capture_fps": (len(self._arrivals) - 1) / span if span > 0 else 0.,
                    "capture_mean_fps": (self._count - 1) / total_span if total_span > 0 else 0.,
                    "captured_frames": self._count}

    def change_rgb(self, action, direction=None):
        if action not in {"toggle_auto", "adjust_exposure", "adjust_gain"}:
            raise ValueError("Unknown RGB control")
        if self.rgb_controls is None:
            return "当前配置未启用 RGB 曝光控制"
        try:
            self._commands.put_nowait((action, direction))
        except Full:
            return "曝光调整队列已满，请稍后再按"
        self.control_message = "正在调整 RGB 成像参数…"
        return self.control_message

    def _apply_commands(self):
        while True:
            try:
                action, direction = self._commands.get_nowait()
            except Empty:
                return
            try:
                method = getattr(self.rgb_controls, action)
                result = method() if direction is None else method(direction)
                values = result["values"]
                self.control_message = (f"自动曝光 {values.get('enable_auto_exposure', '--')} | "
                                        f"曝光 {values.get('exposure', '--')} | 增益 {values.get('gain', '--')}")
            except (ValueError, RuntimeError) as exc:
                self.control_message = f"RGB 参数调整失败：{exc}"

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.source.config.get("timeout_ms", 5000) / 1000 + 2)
            if self._thread.is_alive():
                raise RuntimeError("D415 acquisition thread did not stop before camera cleanup")
