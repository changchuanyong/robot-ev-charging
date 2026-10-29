"""Explicit operator-selected calibration probes; never a substitute GRU model."""
from copy import deepcopy

import numpy as np

from .learning import ForcePolicy
from .synergy import ACTIONS

PROBE_VERSION = "supervised-finite-probe-v1"


def prepare_probe(config, action):
    """Return a separate calibration configuration after explicit mode selection."""
    if action not in ACTIONS[:-1]:
        raise ValueError("probe action must be one of the eight non-Hold correction directions")
    if config.get("method") != "C":
        raise ValueError("supervised probes require method C shared current-risk/mechanical constraints")
    experiment = config.get("experiment", {})
    if (experiment.get("formal") or experiment.get("protocol_sha256") or experiment.get("scheduled_task")
            or experiment.get("task_id")):
        raise ValueError("calibration probes cannot consume formal experiment tasks")
    for name in ("session_id", "specimen_id", "printing_batch_id"):
        if not isinstance(experiment.get(name), str) or not experiment[name].strip():
            raise ValueError(f"probe requires experiment.{name} identity")
    marker = {"version": PROBE_VERSION, "action": action, "purpose": "force_direction_calibration",
              "policy_source": "operator_selected_calibration_direction", "model_used": False,
              "selection_values_meaning": "operator_request_weights_not_model_probabilities"}
    if config.get("supervised_probe") not in (None, marker):
        raise ValueError("configured probe identity differs from explicit requested action")
    result = deepcopy(config)
    result["supervised_probe"] = marker
    result["experiment"].update(formal=False, purpose=marker["purpose"],
                                calibration_only=True, method_identity="PROBE")
    result["finite_trial"]["max_trials"] = 1
    result["learning"]["model_path"] = None
    return result


class OperatorProbePolicy(ForcePolicy):
    """One-hot request weights identify an operator choice, not model confidence."""
    source = "operator_selected_calibration_direction"

    def __init__(self, config, action):
        if action not in ACTIONS[:-1]:
            raise ValueError("invalid supervised probe direction")
        super().__init__({**config, "model_path": None})
        self.requested_action = action
        self.model_ready = True  # interface readiness only; source explicitly records no inference

    def reset(self):
        super().reset()
        self.probe_started_at = None

    def update(self, wrench, timestamp, context_id):
        wrench = np.asarray(wrench, float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all() or not np.isfinite(timestamp):
            self.last_reason = "invalid_probe_feedback"
            self.prediction_time = None
            return None
        if context_id != self.context_id or (self.last_update_time is not None and
                not 0 < timestamp - self.last_update_time <= self.max_gap):
            self.reset()
            self.context_id = context_id
        if self.probe_started_at is None:
            self.probe_started_at = float(timestamp)
        self.last_update_time = float(timestamp)
        self.window.append(wrench.copy())
        if len(self.window) < self.window_size or timestamp - self.probe_started_at + 1e-12 < self.window_size * self.sample_period:
            self.prediction_time = None
            self.last_reason = "probe_pretrial_history_incomplete"
            return None
        self.prediction_time = float(timestamp)
        self.last_probabilities = np.zeros(len(ACTIONS))
        self.last_probabilities[ACTIONS.index(self.requested_action)] = 1.
        self.last_reason = self.source
        return self.last_probabilities.copy()
