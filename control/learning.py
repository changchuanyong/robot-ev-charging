"""Small GRU with frozen training metadata and conservative online action acceptance."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import numpy as np

from .force_dataset import FORCE_UNITS
from .synergy import ACTIONS


def build_model(hidden_size):
    """Import torch only when training/loading a model.

    The candidate uses U(r * h), the reset-before form in the thesis.
    PyTorch nn.GRU uses reset-after and is deliberately not substituted here.
    z selects the candidate state: h' = (1-z)*h + z*candidate (4-8).
    """
    import torch
    from torch import nn

    if not 1 <= int(hidden_size) <= 512:
        raise ValueError("hidden_size must be in [1,512]")

    class ForceGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.reset_x = nn.Linear(6, self.hidden_size)
            self.reset_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.update_x = nn.Linear(6, self.hidden_size)
            self.update_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.candidate_x = nn.Linear(6, self.hidden_size)
            self.candidate_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.classifier = nn.Linear(self.hidden_size, len(ACTIONS))

        def forward(self, window):
            h = window.new_zeros((window.shape[0], self.hidden_size))
            for x in window.unbind(dim=1):
                r = torch.sigmoid(self.reset_x(x) + self.reset_h(h))
                z = torch.sigmoid(self.update_x(x) + self.update_h(h))
                candidate = torch.tanh(self.candidate_x(x) + self.candidate_h(r * h))
                h = (1 - z) * h + z * candidate
            return self.classifier(h)

    return ForceGRU()


def load_checkpoint(path):
    import torch

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"schema_version", "model_kind", "classes", "window_size", "hidden_size",
                "sample_period", "max_gap", "frame", "units", "normalizer", "state_dict",
                "model_version", "trained", "synthetic", "split_manifest", "data_sha256"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("incomplete force model metadata")
    if checkpoint["schema_version"] != 1 or checkpoint["model_kind"] != "thesis_gru":
        raise ValueError("unsupported force model schema")
    if checkpoint["classes"] != list(ACTIONS) or checkpoint["frame"] != "task" or checkpoint["units"] != FORCE_UNITS:
        raise ValueError("force model class/frame/unit mismatch")
    if checkpoint["trained"] is not True or not checkpoint["model_version"]:
        raise ValueError("untrained or unversioned force model")
    if type(checkpoint["synthetic"]) is not bool:
        raise ValueError("force model provenance missing")
    if checkpoint.get("validated_for_hardware") is True:
        approval = checkpoint.get("hardware_acceptance", {})
        if (checkpoint["synthetic"] or approval.get("approved") is not True
                or approval.get("source") != "hardware_acceptance"
                or not approval.get("reviewer") or not approval.get("evidence")
                or not checkpoint.get("acceptance_record_sha256")):
            raise ValueError("hardware validation flag lacks its independent acceptance record")
    normalizer = checkpoint["normalizer"]
    mean, scale = np.asarray(normalizer["mean"], float), np.asarray(normalizer["scale"], float)
    if mean.shape != (6,) or scale.shape != (6,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid force model normalizer")
    if checkpoint["window_size"] < 2 or not 0 < checkpoint["sample_period"] <= checkpoint["max_gap"]:
        raise ValueError("invalid force model timing")
    model = build_model(checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return checkpoint, model


class ForcePolicy:
    """A missing model yields Hold; it is never replaced with random predictions."""
    def __init__(self, config, model_path=None):
        self.config = dict(config)
        self.window_size = int(config.get("window_size", 20))
        self.sample_period = float(config.get("sample_period", 0.01))
        self.max_gap = float(config.get("max_gap", 1.5 * self.sample_period))
        self.max_age = float(config.get("max_age", 0.1))
        self.confidence = float(config.get("confidence", 0.7))
        self.consistent_windows = int(config.get("consistent_windows", 3))
        self.min_hold_s = float(config.get("min_hold_s", 0.05))
        self.max_hold_s = float(config.get("max_hold_s", 2.0))
        if (self.window_size < 2 or self.sample_period <= 0 or self.max_gap < self.sample_period
                or self.max_age <= 0 or not 0 <= self.confidence <= 1
                or self.consistent_windows < 1 or self.min_hold_s < 0 or self.max_hold_s <= 0):
            raise ValueError("invalid force policy configuration")
        self.model_ready = False
        self.model = None
        self.checkpoint = None
        self.last_reason = "model_missing"
        self.last_inference_seconds = 0.0
        self.mean = np.zeros(6)
        self.scale = np.ones(6)
        self.reset()
        model_path = model_path or config.get("model_path")
        if model_path:
            try:
                checkpoint, model = load_checkpoint(model_path)
                if checkpoint["synthetic"] and not config.get("allow_synthetic", False):
                    raise ValueError("synthetic force model is not allowed")
                for key in ("window_size", "sample_period", "max_gap"):
                    if not np.isclose(float(getattr(self, key)), float(checkpoint[key])):
                        raise ValueError(f"force model {key} mismatch")
                if config.get("model_version") and config["model_version"] != checkpoint["model_version"]:
                    raise ValueError("force model version mismatch")
                if config.get("require_hardware_validation", False) and checkpoint.get("validated_for_hardware") is not True:
                    raise ValueError("force model lacks independent hardware acceptance")
                self.checkpoint, self.model = checkpoint, model
                self.mean = np.asarray(checkpoint["normalizer"]["mean"], dtype=float)
                self.scale = np.asarray(checkpoint["normalizer"]["scale"], dtype=float)
                self.model_ready = True
                self.last_reason = "window_incomplete"
            except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                self.last_reason = f"model_unavailable: {exc}"

    def reset(self):
        self.window = deque(maxlen=self.window_size)
        self.context_id = None
        self.last_sample_time = None
        self.last_update_time = None
        self.last_probabilities = None
        self.prediction_time = None
        self._candidate = "Hold"
        self._candidate_count = 0
        self._last_counted_prediction = None
        self.action = "Hold"
        self._action_since = None
        self._hold_since = None

    def _hold(self, timestamp, reason):
        if self.action != "Hold" or self._action_since is None:
            self._action_since = timestamp
        self.action = "Hold"
        if self._hold_since is None:
            self._hold_since = timestamp
        self.last_reason = reason
        return "Hold"

    def update(self, wrench, timestamp, context_id):
        timestamp = float(timestamp)
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all() or not np.isfinite(timestamp):
            self.reset()
            self.last_reason = "invalid_force_window_input"
            return None
        if context_id != self.context_id:
            self.reset()
            self.context_id = context_id
        if self.last_update_time is not None and timestamp <= self.last_update_time:
            self.reset()
            self.context_id = context_id
            self.last_reason = "non_increasing_force_time"
            return None
        self.last_update_time = timestamp
        if self.last_sample_time is not None:
            interval = timestamp - self.last_sample_time
            if interval > self.max_gap:
                self.reset()
                self.context_id = context_id
                self.last_update_time = timestamp
            elif interval < self.sample_period * 0.75:
                return self.last_probabilities
            elif abs(interval - self.sample_period) > self.sample_period * 0.25:
                self.reset()
                self.context_id = context_id
                self.last_update_time = timestamp
        self.last_sample_time = timestamp
        self.window.append(wrench.copy())
        if not self.model_ready:
            return None
        if len(self.window) < self.window_size:
            self.last_reason = "window_incomplete"
            return None
        import torch

        started = time.perf_counter()
        window = (np.asarray(self.window) - self.mean) / self.scale
        try:
            with torch.no_grad():
                logits = self.model(torch.as_tensor(window[None], dtype=torch.float32))
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(float)
            if not np.isfinite(probabilities).all():
                raise ValueError("nonfinite probabilities")
            self.last_probabilities = probabilities
            self.prediction_time = timestamp
            self.last_reason = "prediction_ready"
        except (RuntimeError, ValueError) as exc:
            self.last_probabilities = None
            self.prediction_time = None
            self.last_reason = f"inference_failed: {exc}"
        self.last_inference_seconds = time.perf_counter() - started
        if self.last_inference_seconds > self.max_age:
            self.last_probabilities = None
            self.prediction_time = None
            self.last_reason = "inference_budget_exceeded"
        return self.last_probabilities

    def choose(self, probabilities, allowed, timestamp):
        timestamp = float(timestamp)
        if not self.model_ready or probabilities is None or self.prediction_time is None:
            return self._hold(timestamp, "no_valid_prediction")
        if not np.isfinite(timestamp) or not 0 <= timestamp - self.prediction_time <= self.max_age:
            self._candidate_count = 0
            return self._hold(timestamp, "stale_prediction")
        probabilities = np.asarray(probabilities, dtype=float)
        if (probabilities.shape != (len(ACTIONS),) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
            return self._hold(timestamp, "invalid_probabilities")
        allowed = tuple(a for a in allowed if a in ACTIONS)
        if not allowed:
            return self._hold(timestamp, "empty_allowed_actions")
        candidate = max(allowed, key=lambda a: probabilities[ACTIONS.index(a)])
        if probabilities[ACTIONS.index(candidate)] < self.confidence:
            self._candidate_count = 0
            return self._hold(timestamp, "low_confidence")
        if self.prediction_time != self._last_counted_prediction:
            self._candidate_count = self._candidate_count + 1 if candidate == self._candidate else 1
            self._candidate = candidate
            self._last_counted_prediction = self.prediction_time
        if self._candidate_count < self.consistent_windows:
            return self._hold(timestamp, "temporal_consistency")
        if candidate == "Hold":
            return self._hold(timestamp, "model_hold")
        if self.action != "Hold" and candidate != self.action:
            self._candidate_count = 0
            return self._hold(timestamp, "direction_change_via_hold")
        if self.action == "Hold" and self._action_since is not None and timestamp - self._action_since < self.min_hold_s:
            return self._hold(timestamp, "minimum_hold_time")
        if self.action != candidate:
            self._action_since = timestamp
        self.action = candidate
        self._hold_since = None
        self.last_reason = "accepted"
        return candidate

    def hold_timed_out(self, timestamp):
        return self._hold_since is not None and timestamp - self._hold_since > self.max_hold_s
