"""Create synthetic class prototypes to exercise training plumbing, not contact physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from control.force_dataset import FORCE_UNITS
from control.synergy import ACTIONS, action_vector


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-period", type=float, default=0.01)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(json.dumps(create_demo(args.output_dir, sample_period=args.sample_period, samples=args.samples, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
