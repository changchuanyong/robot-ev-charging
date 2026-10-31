"""Train/evaluate force GRU models, create fixtures and record separate human acceptance."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from control.force_dataset import (FORCE_UNITS, classification_metrics, file_sha256,
                                   fit_normalizer, load_records, make_windows)
from control.learning import build_model, load_checkpoint
from control.synergy import ACTIONS, action_vector


def train(args):
    import torch

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if args.output.exists():
        raise FileExistsError(f"use a new versioned output path: {args.output}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    records = load_records(args.data)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    groups = make_windows(records, split, args.window_size, args.sample_period, args.max_gap, args.stride)
    normalizer = fit_normalizer(groups["train"]["x"])
    mean, scale = np.asarray(normalizer["mean"]), np.asarray(normalizer["scale"])
    device = torch.device(args.device)
    model = build_model(args.hidden_size).to(device)
    x = torch.tensor((groups["train"]["x"] - mean) / scale, dtype=torch.float32)
    y = torch.tensor(groups["train"]["y"], dtype=torch.long)
    val_x = torch.tensor((groups["validation"]["x"] - mean) / scale, dtype=torch.float32, device=device)
    val_y = torch.tensor(groups["validation"]["y"], dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=args.batch_size,
                                         shuffle=True, generator=generator, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append((float(loss.detach()), len(batch_y)))
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(val_x), val_y))
        if not np.isfinite(validation_loss):
            raise RuntimeError("nonfinite training/validation loss")
        history.append({"epoch": epoch + 1,
                        "train_loss": sum(l * n for l, n in losses) / len(y),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(val_x), 1).cpu().numpy()
    synthetic = bool(args.synthetic or any(r.get("synthetic", False) for r in records))
    checkpoint = {"schema_version": 1, "model_kind": "thesis_gru", "classes": list(ACTIONS),
                  "window_size": args.window_size, "hidden_size": args.hidden_size,
                  "sample_period": args.sample_period, "max_gap": args.max_gap,
                  "frame": "task", "units": FORCE_UNITS, "normalizer": normalizer,
                  "state_dict": best_state, "model_version": args.model_version,
                  "trained": True, "synthetic": synthetic, "validated_for_hardware": False,
                  "split_manifest": copy.deepcopy(split), "data_sha256": file_sha256(args.data),
                  "seed": args.seed, "gru_equation": "reset_before; h=(1-z)*h_prev+z*candidate"}
    report = {"model_version": args.model_version, "synthetic": synthetic,
              "validated_for_hardware": False, "training_history": history,
              "window_counts": {name: len(group["y"]) for name, group in groups.items()},
              "trajectory_counts": {name: len(ids) for name, ids in split.items()},
              "class_counts_train": {name: int(np.count_nonzero(groups["train"]["y"] == i)) for i, name in enumerate(ACTIONS)},
              "validation": classification_metrics(groups["validation"]["y"], probabilities, args.confidence),
              "test": "not evaluated during training", "split_manifest": split,
              "normalizer": normalizer, "data_sha256": checkpoint["data_sha256"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    report_path = args.output.with_suffix(".training.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report



def evaluate(args):
    import torch

    checkpoint, model = load_checkpoint(args.model)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if split != checkpoint["split_manifest"] or file_sha256(args.data) != checkpoint["data_sha256"]:
        raise ValueError("dataset/split changed since training; use a separately versioned evaluation protocol")
    groups = make_windows(load_records(args.data), split, checkpoint["window_size"],
                          checkpoint["sample_period"], checkpoint["max_gap"])
    group = groups[args.partition]
    mean = np.asarray(checkpoint["normalizer"]["mean"])
    scale = np.asarray(checkpoint["normalizer"]["scale"])
    windows = torch.tensor((group["x"] - mean) / scale, dtype=torch.float32)
    started = time.perf_counter()
    probabilities = []
    with torch.no_grad():
        for batch in windows.split(args.batch_size):
            probabilities.append(torch.softmax(model(batch), 1).numpy())
    elapsed = time.perf_counter() - started
    result = classification_metrics(group["y"], np.concatenate(probabilities), args.confidence)
    result.update({"partition": args.partition, "model_version": checkpoint["model_version"],
                   "model_sha256": file_sha256(args.model), "data_sha256": checkpoint["data_sha256"],
                   "synthetic": checkpoint["synthetic"], "inference_total_seconds": elapsed,
                   "inference_seconds_per_window": elapsed / len(windows),
                   "trajectory_count": len(set(group["trajectory_id"])),
                   "hardware_performance": "not measured"})
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"use a new evaluation output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def accept_model(model_path, record_path, output_path):
    import torch

    model_path, record_path, output_path = Path(model_path), Path(record_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError("accepted model must use a new output file")
    checkpoint, _ = load_checkpoint(model_path)
    if checkpoint["synthetic"]:
        raise ValueError("synthetic fixture weights can never receive hardware acceptance")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    required = ("reviewer", "accepted_at", "procedure", "calibration_version", "wrench_version", "robot_model")
    if (record.get("schema_version") != 1 or record.get("approved") is not True
            or record.get("source") != "hardware_acceptance"
            or any(not isinstance(record.get(k), str) or not record[k].strip() for k in required)):
        raise ValueError("incomplete explicit human acceptance record")
    datetime.fromisoformat(record["accepted_at"].replace("Z", "+00:00"))
    if record["robot_model"] != "RM_75":
        raise ValueError("acceptance must describe the configured RM75 platform")
    if record.get("model_sha256") != file_sha256(model_path) or record.get("model_version") != checkpoint["model_version"]:
        raise ValueError("acceptance record does not identify these frozen weights")
    if not record.get("evidence"):
        raise ValueError("acceptance requires independently recorded hardware evidence")
    for item in record["evidence"]:
        path = (record_path.parent / item["path"]).resolve()
        if item.get("sha256") != file_sha256(path):
            raise ValueError(f"acceptance evidence hash mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifests = [r for r in rows if r.get("kind") == "manifest"]
        if len(manifests) != 1 or manifests[0].get("source") != "hardware" or manifests[0].get("synthetic") is not False:
            raise ValueError("evidence must be a hardware log, not simulation or replay")
        if not any(r.get("kind") in ("cycle", "sample") and r.get("sample", {}).get("valid") for r in rows):
            raise ValueError("acceptance evidence contains no valid hardware samples")
    checkpoint["validated_for_hardware"] = True
    checkpoint["hardware_acceptance"] = record
    checkpoint["acceptance_record_sha256"] = file_sha256(record_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output_path)
    return {"output": str(output_path), "model_sha256": file_sha256(output_path),
            "model_version": checkpoint["model_version"], "reviewer": record["reviewer"],
            "acceptance_record_sha256": checkpoint["acceptance_record_sha256"]}


def create_demo(output_dir, *, sample_period=0.01, samples=60, seed=42):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"use a new demo directory: {output_dir}")
    if not np.isfinite(sample_period) or sample_period <= 0 or samples < 20:
        raise ValueError("positive sampling period and at least 20 samples required")
    rng = np.random.default_rng(seed)
    records, split = [], {name: [] for name in ("train", "validation", "test")}
    for group, repeats in (("train", 3), ("validation", 1), ("test", 1)):
        for index, action in enumerate(ACTIONS):
            for repeat in range(repeats):
                identifier = f"synthetic-{group}-{index}-{repeat}"
                split[group].append(identifier)
                # Distinct signed prototypes verify classification only. The sign
                # convention is synthetic and is not a force-to-correction law.
                prototype = -action_vector(action) * np.array([2, 2, 2, 0.15, 0.15, 0.15])
                prototype[2] = -0.3
                phase = rng.uniform(0, 2 * np.pi)
                bias = rng.normal(0, [0.04, 0.04, 0.02, 0.002, 0.002, 0.002])
                for step in range(samples):
                    value = prototype * (1 + 0.05 * np.sin(phase + step / 8)) + bias
                    value += rng.normal(0, [0.025, 0.025, 0.02, 0.002, 0.002, 0.002])
                    records.append({"trajectory_id": identifier, "batch_id": f"synthetic-{group}-batch",
                                    "timestamp": step * sample_period, "wrench": value.tolist(),
                                    "label": action, "frame": "task", "units": FORCE_UNITS,
                                    "synthetic": True, "label_basis": "synthetic class prototype; not a physical force model"})
    output_dir.mkdir(parents=True)
    with (output_dir / "force.jsonl").open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "SYNTHETIC PIPELINE FIXTURE ONLY. No robot or force sensor data.\n"
        "These classes are arbitrary numerical prototypes, not connector contact mechanics.\n"
        "Training accuracy on this dataset does not measure physical insertion performance.\n", encoding="utf-8")
    return {"synthetic": True, "samples": len(records), "trajectories": {k: len(v) for k, v in split.items()},
            "sample_period": sample_period, "data": str(output_dir / "force.jsonl"),
            "split": str(output_dir / "split.json")}


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="Train frozen weights; hardware acceptance stays false")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-version", required=True)
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--sample-period", type=float, default=0.01)
    p.add_argument("--max-gap", type=float, default=0.015)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--confidence", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--synthetic", action="store_true", help="mark synthetic fixture data; never a real experiment")

    p = commands.add_parser("evaluate", help="Evaluate a frozen validation or test trajectory split")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--partition", choices=("validation", "test"), default="test")
    p.add_argument("--confidence", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("accept", help="Attach an independently completed human hardware-acceptance record")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("demo", help="Create synthetic pipeline fixtures, never physical force data")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-period", type=float, default=0.01)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    return root


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0].startswith("-") and arguments[0] not in {"-h", "--help"}:
        arguments.insert(0, "train")
    p = parser()
    args = p.parse_args(arguments)
    if args.command == "train":
        result = train(args)
        result = {"output": str(args.output), "validation": result["validation"],
                  "synthetic": result["synthetic"], "validated_for_hardware": False}
    elif args.command == "evaluate":
        if args.batch_size < 1:
            p.error("batch-size must be positive")
        result = evaluate(args)
    elif args.command == "accept":
        result = accept_model(args.model, args.record, args.output)
    else:
        result = create_demo(args.output_dir, sample_period=args.sample_period,
                             samples=args.samples, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
