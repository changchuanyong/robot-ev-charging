"""Experiment bookkeeping and evidence tests; all data here are synthetic."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from control.experiments import (bind_task, configuration_digest, digest, make_schedule,
                                 protocol_template, validate_protocol)
from control.force_dataset import load_records, make_windows, validate_split
from control.logging import TaskLog
from control.evaluate import evaluate
from tests.test_learning import records, SPLIT, workspace_temp
from train.python.prepare_force_dataset import export_dataset, review_manifest


class ExperimentTests(unittest.TestCase):
    def test_520_unique_complete_tasks_balanced_paired_and_pending(self):
        protocol = protocol_template()
        schedule = make_schedule(protocol)
        self.assertEqual(schedule, make_schedule(protocol))
        tasks = schedule["tasks"]
        self.assertEqual(len(tasks), 520)
        self.assertEqual(len({t["task_id"] for t in tasks}), 520)
        self.assertEqual(set(t["status"] for t in tasks), {"pending"})
        cells = Counter((t["condition"], t["method"], t["specimen_id"]) for t in tasks)
        self.assertEqual(len(cells), 104)
        self.assertEqual(set(cells.values()), {5})
        for start in range(0, 520, 4):
            block = tasks[start:start + 4]
            self.assertEqual({t["method"] for t in block}, {"A", "B", "C", "F"})
            self.assertEqual(len({t["pair_id"] for t in block}), 1)
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_protocol(protocol, frozen=True)

    def test_frozen_task_binding_rejects_config_geometry_and_setup_changes(self):
        with workspace_temp() as directory:
            model = directory / "fake.pt"
            model.write_bytes(b"synthetic test placeholder")
            geometry = directory / "fake_geometry.json"
            geometry.write_text("synthetic binding test", encoding="utf-8")
            protocol = protocol_template()
            config = {"method": "A", "learning": {"model_path": str(model)}}
            protocol.update(status="frozen", protocol_version="fixture-only", geometry_version="fixture-only",
                            pilot_evidence="synthetic unit test", source_sha256="a" * 64,
                            model_sha256=__import__("hashlib").sha256(model.read_bytes()).hexdigest(),
                            offset_amplitudes=[.001] * 4, setup_tolerance=[.0001] * 4,
                            method_config_sha256={m: configuration_digest({**config, "method": m}) for m in "ABCF"})
            for i, specimen in enumerate(protocol["specimens"]):
                specimen["printing_batch_id"] = f"synthetic-print-{i}"
                protocol["geometry_references"][specimen["specimen_id"]] = {
                    "path": str(geometry), "version": "synthetic-geometry",
                    "sha256": __import__("hashlib").sha256(geometry.read_bytes()).hexdigest()}
            schedule = make_schedule(protocol)
            task = next(t for t in schedule["tasks"] if t["method"] == "A")
            setup = {"task_id": task["task_id"], "verified": True, "session_id": "synthetic-session",
                     "fixture_event": "unit-test", "measurement_basis": "synthetic", "wear_cycles_before": 0,
                     "measured_initial_offset": task["planned_initial_offset"], "method": "F"}
            with patch("control.logging.source_digest", return_value="a" * 64):
                bound = bind_task(schedule, task["task_id"], config, setup)
                self.assertEqual(bound["method"], "A")  # setup cannot override a frozen method
                with self.assertRaisesRegex(ValueError, "configuration"):
                    bind_task(schedule, task["task_id"], {**config, "period": .01}, setup)
                bad = deepcopy(schedule)
                bad["protocol"]["geometry_version"] = "changed"
                with self.assertRaisesRegex(ValueError, "hash"):
                    bind_task(bad, task["task_id"], config, setup)
                with self.assertRaisesRegex(ValueError, "deviation"):
                    bind_task(schedule, task["task_id"], config, {**setup, "measured_initial_offset": [1.] * 4})

    def test_specimen_printing_batch_session_leakage_and_fault_exclusion(self):
        for field in ("specimen_id", "printing_batch_id", "session_id"):
            rows = records()
            for row in rows:
                row[field] = row["batch_id"]
            rows[-1][field] = "train"
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_split(rows, SPLIT)
        rows = records()
        for row in rows[:6]:
            row["label"] = "fault"
        with self.assertRaisesRegex(ValueError, "train: no valid"):
            make_windows(rows, SPLIT, 3, .01, .015)

    def test_failed_task_without_feedback_is_retained_and_version_stratified(self):
        with workspace_temp() as directory:
            paths = []
            for i, version in enumerate(("finite-trial-v2", "legacy-v1")):
                path = directory / f"task-{i}.jsonl"
                with TaskLog(path, {"method": "A", "target_id": "synthetic", "rules_version": version}, "simulation") as log:
                    log.write({"kind": "result", "success": False, "reason": "feedback_unavailable"})
                paths.append(path)
            result = evaluate(paths)
            self.assertEqual(len(result["groups"]), 2)
            self.assertEqual(sum(g["tasks"] for g in result["groups"]), 2)
            self.assertTrue(all(g["successes"] == 0 for g in result["groups"]))


class ForceEvidenceTests(unittest.TestCase):
    def test_no_motion_collection_uses_same_wrench_processor_cycle_contract(self):
        import numpy as np
        from control.config import simulation_config
        from control.run import run_device
        from tests.test_device_runner import FakeDevice
        with workspace_temp() as directory:
            config = simulation_config()
            device = FakeDevice()
            clock = [0.]
            def advance(dt):
                clock[0] += dt
            original_read = device.read
            def read():
                sample = original_read()
                sample.metadata = {"simulation": True, "external_compensated": True,
                                   "force_channel": "synthetic_external", "moment_origin": "tool"}
                return sample
            device.read = read
            path = directory / "collection.jsonl"
            with (patch("control.run.preflight", return_value=([], None)),
                  patch("control.run.hardware_config", return_value={}),
                  patch("control.hardware.RealManRobot", return_value=device),
                  patch("control.run.time.monotonic", side_effect=lambda: clock[0]),
                  patch("control.run.time.sleep", side_effect=advance)):
                run_device(config, directory / "unused.json", path, collect_only=True, duration=.01,
                           collection_frame={"version": "synthetic", "validated": True, "T_base_task": np.eye(4).tolist()})
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            cycles = [r for r in rows if r["kind"] == "cycle"]
            self.assertTrue(cycles)
            self.assertTrue(all(r["decision"]["details"]["filtered_wrench"] == [0.] * 6 for r in cycles))
            self.assertNotIn("stop", device.calls)
            self.assertTrue(all(r["decision"]["command"] is None for r in cycles))
            with self.assertRaisesRegex(ValueError, "collection-only"):
                evaluate([path])

    def create_log(self, directory, outcome="effective", group="fixture"):
        path = directory / "trial.jsonl"
        config = {"method": "C", "target_id": "synthetic", "rules_version": "finite-trial-v2",
                  "finite_trial": {"baseline_s": .02, "min_advance_m": .001},
                  "learning": {"window_size": 3, "sample_period": .01},
                  "experiment": {"session_id": group + "-session", "printing_batch_id": group + "-print", "specimen_id": group + "-specimen"}}
        with TaskLog(path, config, "simulation") as log:
            for i in range(8):
                details = {"filtered_wrench": [1.] * 6, "depth": i * .0003}
                if i >= 6:
                    details["trial_result"] = {"trial_id": "1", "started_at": 1.03, "ended_at": 1.06,
                        "action": "+x", "outcome": outcome, "measured_advance_m": .002, "loads_allowed": True}
                log.write({"kind": "cycle", "at": 1. + i * .01, "sample": {"timestamp": 1. + i * .01, "valid": True, "seq": i},
                           "decision": {"state": "INSERT", "details": details}})
        return path

    def test_review_does_not_generate_truth_and_effective_label_uses_preaction_data(self):
        with workspace_temp() as directory:
            path = self.create_log(directory)
            review = directory / "review.json"
            review_manifest([path], review)
            annotation = json.loads(review.read_text(encoding="utf-8"))
            interval = annotation["logs"][0]["intervals"][0]
            self.assertEqual(interval["label"], "invalid")
            self.assertFalse(interval["reviewed"])
            interval.update(label="+x", reviewed=True, reviewer="test-reviewer", basis="synthetic test")
            review.write_text(json.dumps(annotation), encoding="utf-8")
            output = directory / "data.jsonl"
            export_dataset(review, output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            labeled = [r for r in rows if r["label"] == "+x"]
            self.assertTrue(labeled)
            self.assertTrue(all(r["timestamp"] < 1.03 for r in labeled))
            interval["end_s"] = .05
            review.write_text(json.dumps(annotation), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "before the action"):
                export_dataset(review, directory / "bad.jsonl")

    def test_load_release_or_unconfirmed_trial_cannot_become_direction_label(self):
        with workspace_temp() as directory:
            path = self.create_log(directory, "unconfirmed")
            review = directory / "review.json"
            review_manifest([path], review)
            annotation = json.loads(review.read_text(encoding="utf-8"))
            annotation["logs"][0]["intervals"][0].update(label="+x", reviewed=True, reviewer="test-reviewer", basis="load dropped")
            review.write_text(json.dumps(annotation), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "effective trial"):
                export_dataset(review, directory / "bad.jsonl")

    def test_imported_effective_hardware_probe_needs_terminal_stop_feedback(self):
        with workspace_temp() as directory:
            for confirmed in (False, True):
                folder = directory / str(confirmed)
                folder.mkdir()
                path = self.create_log(folder)
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                rows[0]["source"] = "hardware"
                rows[0]["config"]["supervised_probe"] = {"action": "+x"}
                rows.append({"kind": "result", "stop_feedback_confirmed": confirmed})
                path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                review = folder / "review.json"
                review_manifest([path], review)
                # Fault/unreviewed records remain exportable for traceability.
                export_dataset(review, folder / "invalid.jsonl")
                annotation = json.loads(review.read_text(encoding="utf-8"))
                annotation["logs"][0]["intervals"][0].update(label="+x", reviewed=True, reviewer="test", basis="synthetic")
                review.write_text(json.dumps(annotation), encoding="utf-8")
                if confirmed:
                    export_dataset(review, folder / "positive.jsonl")
                else:
                    with self.assertRaisesRegex(ValueError, "terminal measured stop"):
                        export_dataset(review, folder / "positive.jsonl")

    def test_review_export_load_and_windows_with_independent_trajectories(self):
        with workspace_temp() as directory:
            paths = []
            for group in ("train", "validation", "test"):
                folder = directory / group
                folder.mkdir()
                paths.append(self.create_log(folder, group=group))
            review = directory / "review.json"
            review_manifest(paths, review)
            annotation = json.loads(review.read_text(encoding="utf-8"))
            split = {}
            for group, source in zip(("train", "validation", "test"), annotation["logs"]):
                split[group] = [source["trajectory_id"]]
                source["intervals"][0].update(label="+x", reviewed=True, reviewer="unit-test", basis="synthetic progression")
            review.write_text(json.dumps(annotation), encoding="utf-8")
            output = directory / "data.jsonl"
            export_dataset(review, output)
            windows = make_windows(load_records(output), split, 3, .01, .015)
            self.assertTrue(all(len(group["y"]) == 1 for group in windows.values()))

    def test_actual_controller_cycle_logs_feed_preaction_training_windows(self):
        from control.config import simulation_config
        from control.simulate import run_simulation
        from tests.test_state_machine import ControlledPolicy
        with workspace_temp() as directory:
            paths = []
            for group in ("train", "validation", "test"):
                config = simulation_config()
                config["method"] = "C"
                config["learning"].update(consistent_windows=1, min_hold_s=0.)
                config["experiment"] = {"session_id": group, "specimen_id": group, "printing_batch_id": group}
                path = directory / f"{group}.jsonl"
                policy = ControlledPolicy(config["learning"], [1., 0., 0., 0., 0., 0., 0., 0., 0.])
                run_simulation(config, output=path, policy=policy)
                paths.append(path)
            review = directory / "review.json"
            review_manifest(paths, review)
            annotation = json.loads(review.read_text(encoding="utf-8"))
            split = {}
            for group, source in zip(("train", "validation", "test"), annotation["logs"]):
                split[group] = [source["trajectory_id"]]
                trial = next(i for i in source["intervals"] if i["observed_outcome"] == "effective")
                trial.update(label="+x", reviewed=True, reviewer="synthetic-unit-test", basis="software fixture only")
            review.write_text(json.dumps(annotation), encoding="utf-8")
            output = directory / "data.jsonl"
            export_dataset(review, output)
            windows = make_windows(load_records(output), split, config["learning"]["window_size"],
                                   config["learning"]["sample_period"], config["learning"]["max_gap"])
            self.assertTrue(all(len(group["y"]) >= 1 for group in windows.values()))


if __name__ == "__main__":
    unittest.main()
