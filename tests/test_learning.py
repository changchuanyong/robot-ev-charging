import importlib.util
import json
from pathlib import Path
from contextlib import contextmanager
import shutil
import unittest
import uuid

import numpy as np

from control.force_dataset import (FORCE_UNITS, classification_metrics, fit_normalizer,
                                   load_records, make_windows, validate_split)
from control.learning import ForcePolicy, build_model, load_checkpoint
from control.synergy import ACTIONS


def records():
    rows = []
    for group, value in (("train", 1.0), ("validation", 100.0), ("test", 500.0)):
        for step in range(6):
            rows.append({"trajectory_id": group, "batch_id": group, "timestamp": step * 0.01,
                         "wrench": [value] * 6, "label": "+x", "frame": "task", "units": FORCE_UNITS,
                         "synthetic": True})
    return rows


SPLIT = {name: [name] for name in ("train", "validation", "test")}


@contextmanager
def workspace_temp():
    parent = Path(__file__).resolve().parents[1] / "artifacts" / "test_learning"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / uuid.uuid4().hex
    target.mkdir()
    try:
        yield target
    finally:
        if target.resolve().parent != parent.resolve():
            raise RuntimeError("test cleanup escaped its workspace directory")
        shutil.rmtree(target)


class DatasetTests(unittest.TestCase):
    def test_rejects_trajectory_and_batch_leakage(self):
        with self.assertRaisesRegex(ValueError, "trajectory leakage"):
            validate_split(records(), {"train": ["train"], "validation": ["train"], "test": ["test"]})
        rows = records()
        rows[-1]["batch_id"] = "train"
        with self.assertRaisesRegex(ValueError, "batch leakage"):
            validate_split(rows, SPLIT)

    def test_normalizer_uses_only_training_partition(self):
        groups = make_windows(records(), SPLIT, 3, 0.01, 0.015)
        normalizer = fit_normalizer(groups["train"]["x"])
        np.testing.assert_allclose(normalizer["mean"], np.ones(6))
        self.assertEqual(len(groups["train"]["y"]), 4)
        self.assertEqual(set(groups["test"]["trajectory_id"]), {"test"})

    def test_rejects_gap_and_ambiguous_windows(self):
        rows = records()
        for row in rows[:6]:
            row["label"] = "ambiguous"
        with self.assertRaisesRegex(ValueError, "train: no valid"):
            make_windows(rows, SPLIT, 3, 0.01, 0.015)
        rows = records()
        for row in rows[:6]:
            row["timestamp"] *= 5
        with self.assertRaisesRegex(ValueError, "train: no valid"):
            make_windows(rows, SPLIT, 3, 0.01, 0.015)

    def test_load_units_and_metrics(self):
        with workspace_temp() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in records()), encoding="utf-8")
            self.assertEqual(len(load_records(path)), 18)
            row = records()[0]
            row["frame"] = "sensor"
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_records(path)
        p = np.eye(9)[[0, 1, 8]]
        report = classification_metrics([0, 1, 8], p)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["accepted_accuracy"], 1.0)
        self.assertEqual(report["per_class"]["Hold"]["recall"], 1.0)

    def test_demo_is_explicitly_synthetic_and_split_by_batch(self):
        from train.python.create_demo_force_dataset import create_demo

        with workspace_temp() as directory:
            result = create_demo(directory / "fixture", samples=20)
            rows = load_records(result["data"])
            split = json.loads(Path(result["split"]).read_text(encoding="utf-8"))
            validate_split(rows, split)
            self.assertEqual(set(r["label"] for r in rows), set(ACTIONS))
            self.assertTrue(all(r["synthetic"] for r in rows))

    def test_export_uses_reviewed_labels_not_online_action(self):
        from train.python.prepare_force_dataset import export_dataset

        with workspace_temp() as directory:
            log = directory / "control.jsonl"
            rows = [{"kind": "manifest", "clock": "host_monotonic_seconds", "synthetic": True,
                     "config_sha256": "test-config"}]
            for i in range(4):
                rows.append({"kind": "cycle", "at": 1 + i * 0.01, "task_id": "task", "attempt_id": "try1",
                             "sample": {"valid": True, "seq": i},
                             "decision": {"details": {"filtered_wrench": [1] * 6, "action": "-x"}}})
            log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            annotation = directory / "annotations.json"
            annotation.write_text(json.dumps({"schema_version": 1, "logs": [{"path": "control.jsonl",
                "trajectory_id": "t", "batch_id": "b", "intervals": [{"start_s": 0.01, "end_s": 0.025,
                    "label": "+y", "basis": "independently reviewed bounded-action experiment"}]}]}), encoding="utf-8")
            output = directory / "force.jsonl"
            export_dataset(annotation, output)
            result = load_records(output)
            self.assertEqual([r["label"] for r in result], ["invalid", "+y", "+y", "invalid"])
            self.assertTrue(all(r["synthetic"] for r in result))


class PolicyTests(unittest.TestCase):
    def policy(self, **config):
        return ForcePolicy({"window_size": 3, "sample_period": 0.01, "max_gap": 0.015,
                            "consistent_windows": 2, "min_hold_s": 0.02, **config})

    def prediction(self, policy, time, action):
        policy.model_ready = True
        policy.prediction_time = time
        p = np.zeros(9)
        p[ACTIONS.index(action)] = 0.9
        p[ACTIONS.index("Hold")] += 0.1
        return p

    def test_missing_model_and_context_gap_reset(self):
        p = self.policy()
        self.assertFalse(p.model_ready)
        self.assertIsNone(p.update(np.zeros(6), 0, "a"))
        p.update(np.zeros(6), 0.01, "a")
        self.assertEqual(len(p.window), 2)
        p.update(np.zeros(6), 0.03, "a")
        self.assertEqual(len(p.window), 1)
        p.update(np.zeros(6), 0.04, "b")
        self.assertEqual(len(p.window), 1)
        self.assertEqual(p.choose(None, ACTIONS, 0.04), "Hold")

    def test_original_confidence_and_unique_predictions(self):
        p = self.policy()
        probs = self.prediction(p, 0, "+x")
        self.assertEqual(p.choose(probs, ACTIONS, 0), "Hold")
        self.assertEqual(p.choose(probs, ACTIONS, 0.01), "Hold")
        self.assertEqual(p._candidate_count, 1)
        probs = self.prediction(p, 0.02, "+x")
        self.assertEqual(p.choose(probs, ACTIONS, 0.02), "+x")
        self.assertEqual(p.choose(probs, ["-x", "Hold"], 0.03), "Hold")
        self.assertEqual(p.last_reason, "low_confidence")

    def test_reversal_stale_and_hold_timeout(self):
        p = self.policy(consistent_windows=1, min_hold_s=0.02, max_hold_s=0.1)
        self.assertEqual(p.choose(self.prediction(p, 0, "+x"), ACTIONS, 0), "+x")
        self.assertEqual(p.choose(self.prediction(p, 0.01, "-x"), ACTIONS, 0.01), "Hold")
        self.assertEqual(p.last_reason, "direction_change_via_hold")
        self.assertEqual(p.choose(self.prediction(p, 0.02, "-x"), ACTIONS, 0.02), "Hold")
        self.assertEqual(p.choose(self.prediction(p, 0.04, "-x"), ACTIONS, 0.04), "-x")
        self.assertEqual(p.choose(np.eye(9)[1], ACTIONS, 1.0), "Hold")
        self.assertEqual(p.last_reason, "stale_prediction")
        self.assertTrue(p.hold_timed_out(1.11))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is optional for baseline control")
class TorchTests(unittest.TestCase):
    def test_exact_thesis_gate_direction(self):
        import torch

        model = build_model(1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.update_x.bias.fill_(20)
            model.candidate_x.bias.fill_(1)
            model.classifier.weight[0, 0] = 1
            positive = model(torch.zeros((1, 1, 6)))[0, 0]
            self.assertAlmostEqual(float(positive), np.tanh(1), places=6)
            model.update_x.bias.fill_(-20)
            retained = model(torch.zeros((1, 1, 6)))[0, 0]
            self.assertLess(float(retained), 1e-7)

    def test_train_save_reload_and_evaluate(self):
        from train.python.train_force_gru import parser, train
        from train.python.evaluate_force_gru import evaluate
        from argparse import Namespace

        with workspace_temp() as directory:
            directory = Path(directory)
            data, split, output = directory / "data.jsonl", directory / "split.json", directory / "model.pt"
            data.write_text("\n".join(json.dumps(x) for x in records()), encoding="utf-8")
            split.write_text(json.dumps(SPLIT), encoding="utf-8")
            args = parser().parse_args(["--data", str(data), "--split", str(split), "--output", str(output),
                                       "--model-version", "unit-test-synthetic", "--window-size", "3",
                                       "--hidden-size", "4", "--epochs", "2", "--synthetic"])
            report = train(args)
            checkpoint, _ = load_checkpoint(output)
            self.assertTrue(checkpoint["synthetic"])
            self.assertFalse(checkpoint["validated_for_hardware"])
            self.assertEqual(report["test"], "not evaluated during training")
            self.assertFalse(ForcePolicy({"window_size": 3}, output).model_ready)
            policy = ForcePolicy({"window_size": 3, "allow_synthetic": True}, output)
            self.assertTrue(policy.model_ready)
            policy.update(np.ones(6), 0, "test")
            policy.update(np.ones(6), 0.01, "test")
            p = policy.update(np.ones(6), 0.02, "test")
            self.assertEqual(p.shape, (9,))
            result = evaluate(Namespace(model=output, data=data, split=split, partition="test",
                                        batch_size=16, confidence=0.7, output=None))
            self.assertEqual(result["count"], 4)
            self.assertTrue(result["synthetic"])
            from train.python.accept_force_model import accept_model
            with self.assertRaisesRegex(ValueError, "synthetic"):
                accept_model(output, directory / "absent-acceptance.json", directory / "accepted.pt")


if __name__ == "__main__":
    unittest.main()
