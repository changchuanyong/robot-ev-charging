"""Evaluate frozen force-GRU weights on an untouched trajectory split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from control.force_dataset import classification_metrics, file_sha256, load_records, make_windows
from control.learning import load_checkpoint


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--partition", choices=("validation", "test"), default="test")
    p.add_argument("--confidence", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
