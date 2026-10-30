"""Session-scoped RGB sensor controls; never change stereo depth exposure."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import sys
import time
from threading import RLock


OPTION_NAMES = ("enable_auto_exposure", "exposure", "gain",
                "enable_auto_white_balance", "auto_exposure_priority")
SWITCHES = {"enable_auto_exposure", "enable_auto_white_balance", "auto_exposure_priority"}


def validate_rgb_options(options):
    if not isinstance(options, dict) or set(options) - set(OPTION_NAMES):
        raise ValueError("rgb_options_requires_known_RGB_option_dictionary")
    for name, value in options.items():
        if (not isinstance(value, (int, float)) or not math.isfinite(value)
                or (isinstance(value, bool) and name not in SWITCHES)
                or (name in SWITCHES and value not in (0, 1))):
            raise ValueError(f"invalid_RGB_option_value: {name}")
    if any(name in options for name in ("exposure", "gain")) and options.get("enable_auto_exposure") == 1:
        raise ValueError("manual_RGB_exposure_or_gain_conflicts_with_auto_exposure")


class RGBControls:
    """Read supported ranges, validate before writing, and restore on close.

    Sensor option reads are cached between user actions to avoid multiple USB
    control transfers for every video frame. Per-frame SDK metadata, when
    available, is recorded separately with its original names and raw values.
    """
    def __init__(self, device, rs):
        self.sensor = device.first_color_sensor()
        self.rs = rs
        self._lock = RLock()
        self._options = {}
        self._ranges = {}
        self._touched = set()
        self._last_changes = {}
        self._windows_exposure = sys.platform == "win32"
        for name in OPTION_NAMES:
            option = getattr(rs.option, name)
            if self.sensor.supports(option):
                limits = self.sensor.get_option_range(option)
                values = [float(getattr(limits, field)) for field in ("min", "max", "step", "default")]
                if (not all(math.isfinite(value) for value in values)
                        or values[0] > values[1] or values[2] < 0):
                    raise ValueError(f"invalid_RGB_option_range: {name}")
                self._options[name] = option
                self._ranges[name] = dict(zip(("min", "max", "step", "default"), values))
                self._ranges[name]["read_only"] = bool(self.sensor.is_option_read_only(option))
        self._original = self.snapshot()["values"]

    def snapshot(self):
        with self._lock:
            values = {name: float(self.sensor.get_option(option)) for name, option in self._options.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("nonfinite_RGB_sensor_option")
            self._snapshot = {"source": "sensor_get_option",
                              "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                              "values": values, "ranges": deepcopy(self._ranges),
                              "last_changes": deepcopy(self._last_changes)}
            return deepcopy(self._snapshot)

    def _validate_value(self, name, value):
        if name not in self._options:
            raise ValueError(f"unsupported_RGB_option: {name}")
        limits = self._ranges[name]
        if limits["read_only"]:
            raise ValueError(f"read_only_RGB_option: {name}")
        if not limits["min"] <= value <= limits["max"]:
            raise ValueError(f"RGB_option_out_of_range: {name} [{limits['min']}, {limits['max']}]")
        step = limits["step"]
        if step > 0 and not math.isclose((value - limits["min"]) / step,
                                         round((value - limits["min"]) / step), abs_tol=1e-6):
            raise ValueError(f"RGB_option_requires_step: {name} {step}")

    def _expected_readback(self, name, value):
        if name == "exposure" and self._windows_exposure and value > 0:
            # librealsense v2.59.1 mf-uvc.cpp from_100micros/to_100micros:
            # Windows uses log2(seconds), even when the reported UVC step is 1.
            exponent = math.log2(value * .0001)
            exponent = math.floor(exponent + .5) if exponent >= 0 else math.ceil(exponent - .5)
            return float(int(10000 * 2. ** exponent))
        return float(value)

    def _write(self, name, value):
        current = float(self.sensor.get_option(self._options[name]))
        if math.isclose(current, value, abs_tol=1e-5):
            self._last_changes[name] = {"requested": float(value), "readback": current, "status": "unchanged"}
            return
        # A failed SDK call can still have changed device state; include it in restoration.
        self._touched.add(name)
        self.sensor.set_option(self._options[name], float(value))
        expected = self._expected_readback(name, value)
        # A short bounded readback window accommodates UVC control propagation.
        for delay in (0., .02, .04, .08, .16, .32):
            if delay:
                time.sleep(delay)
            actual = float(self.sensor.get_option(self._options[name]))
            if math.isfinite(actual) and (math.isclose(actual, value, abs_tol=1e-5)
                                          or math.isclose(actual, expected, abs_tol=1e-5)):
                self._last_changes[name] = {"requested": float(value), "readback": actual,
                                           "status": "applied" if actual == value else "quantized"}
                return
        raise RuntimeError(f"RGB_option_readback_mismatch: {name}, requested={value}, actual={actual}")

    def apply(self, requested):
        validate_rgb_options(requested)
        requested = dict(requested)
        if any(name in requested for name in ("exposure", "gain")):
            requested["enable_auto_exposure"] = 0.
        with self._lock:
            current = self.snapshot()["values"]
            if requested.get("enable_auto_exposure") == 0 and current.get("enable_auto_exposure") == 1:
                # Disabling AE on WMF may reset exposure; keep the prior manual
                # controls unless the caller explicitly requested new values.
                for name in ("exposure", "gain"):
                    if name in current:
                        requested.setdefault(name, current[name])
            for name, value in requested.items():
                self._validate_value(name, value)
            # Disable AE before a manual exposure/gain write, irrespective of dict order.
            ordered = (["enable_auto_exposure"] if requested.get("enable_auto_exposure") == 0 else [])
            ordered += [name for name in requested if name not in ordered]
            self._last_changes = {}
            try:
                for name in ordered:
                    self._write(name, requested[name])
            finally:
                self.snapshot()
            return deepcopy(self._snapshot)

    def toggle_auto(self):
        with self._lock:
            current = self.snapshot()["values"].get("enable_auto_exposure")
            if current is None:
                raise ValueError("unsupported_RGB_option: enable_auto_exposure")
            return self.apply({"enable_auto_exposure": 0 if current else 1})

    def _adjust(self, name, direction):
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError("RGB_adjust_direction_must_be_minus_or_plus_one")
        with self._lock:
            values = self.snapshot()["values"]
            if name not in values:
                raise ValueError(f"unsupported_RGB_option: {name}")
            limits = self._ranges[name]
            # Windows exposure must cross a logarithmic UVC step; gain uses 5%.
            change = (abs(values[name]) * .25 if name == "exposure"
                      else (limits["max"] - limits["min"]) * .05)
            step = limits["step"]
            change = max(step, change)
            target = min(limits["max"], max(limits["min"], values[name] + direction * change))
            if name == "exposure" and self._windows_exposure:
                target = min(limits["max"], max(limits["min"], values[name] * (2. if direction > 0 else .5)))
            if step > 0:
                target = limits["min"] + round((target - limits["min"]) / step) * step
                target = min(limits["max"], max(limits["min"], target))
            return self.apply({name: target})

    def adjust_exposure(self, direction):
        return self._adjust("exposure", direction)

    def adjust_gain(self, direction):
        return self._adjust("gain", direction)

    def frame_settings(self, frame):
        with self._lock:
            cached = deepcopy(self._snapshot)
        values, errors = {}, {}
        for name in ("actual_exposure", "gain_level", "auto_exposure", "white_balance"):
            field = getattr(self.rs.frame_metadata_value, name)
            try:
                if frame.supports_frame_metadata(field):
                    values[name] = int(frame.get_frame_metadata(field))
            except RuntimeError as exc:
                errors[name] = str(exc)
        return {"sensor_options": cached,
                "frame_metadata": {"source": "sdk_frame_metadata", "values": values, "errors": errors},
                "note": "Sensor options are cached reads, not frame exposure. Frame metadata uses raw SDK units."}

    def restore(self):
        with self._lock:
            if not self._touched:
                return
            original, touched = self._original, set(self._touched)
            writes = []
            if touched & {"enable_auto_exposure", "exposure", "gain"}:
                writes.append(("enable_auto_exposure", 0.))
                writes += [(name, original[name]) for name in ("exposure", "gain")
                           if name in original and not self._ranges[name]["read_only"]]
                touched.add("enable_auto_exposure")
            writes += [(name, original[name]) for name in OPTION_NAMES
                       if name in touched and name not in {"enable_auto_exposure", "exposure", "gain"}]
            if "enable_auto_exposure" in touched:
                writes.append(("enable_auto_exposure", original["enable_auto_exposure"]))
            errors = []
            for name, value in writes:
                try:
                    self._write(name, value)
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
            if errors:
                raise RuntimeError("RGB_option_restore_failed: " + "; ".join(errors))
            self._touched.clear()
