"""Train the thesis reset-before GRU on explicitly split, labelled force trajectories."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from control.force_dataset import (FORCE_UNITS, classification_metrics, file_sha256,
                                   fit_normalizer, load_records, make_windows)
from control.learning import build_model
from control.synergy import ACTIONS


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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
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
    return p


def main():
    args = parser().parse_args()
    result = train(args)
    print(json.dumps({"output": str(args.output), "validation": result["validation"],
                      "synthetic": result["synthetic"], "validated_for_hardware": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
