"""Attach a human hardware-acceptance record to a new version of frozen weights.

This validates the record's identity and evidence hashes; it does not perform an
experiment or establish physical safety. Only an operator's completed acceptance
procedure can supply the approval. Training never enables this flag itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from control.force_dataset import file_sha256
from control.learning import load_checkpoint


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(accept_model(args.model, args.record, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
