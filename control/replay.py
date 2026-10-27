"""Recompute decisions on recorded actual feedback; never import a device driver."""
import argparse
import json
from pathlib import Path

import numpy as np

from .logging import finite_json, source_digest
from .simulate import make_policy
from .state_machine import InsertionController
from .types import RobotSample


def load_log(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or rows[0].get("kind") != "manifest" or rows[0].get("schema_version") != 1:
        raise ValueError("not a versioned control task log")
    ids = {r.get("task_id") for r in rows}
    if len(ids) != 1 or None in ids:
        raise ValueError("a task log must contain exactly one task identity")
    return rows


def replay(path):
    rows = load_log(path)
    config = rows[0]["config"]
    controller = InsertionController(config, make_policy(config))
    mismatches = []
    count = 0
    last = None
    for row in rows:
        if row.get("kind") != "cycle":
            continue
        count += 1
        sample = RobotSample.from_dict(row["sample"])
        last = controller.step(sample, float(row["at"]), row.get("observation"))
        original = row["decision"]
        difference = [k for k in ("state", "stop", "finished", "success", "reason") if last.to_dict()[k] != original[k]]
        old_command = original.get("command")
        if (last.command is None) != (old_command is None):
            difference.append("command_presence")
        elif last.command is not None and not np.allclose(last.command, old_command, atol=1e-9, rtol=0):
            difference.append("command")
        if difference:
            mismatches.append({"cycle": count, "at": row["at"], "fields": difference})
        if last.stop:
            controller.acknowledge_stop()
    if not count:
        raise ValueError("log has no controller cycles to replay")
    return {"hardware_accessed": False, "source": str(path), "cycles": count,
            "source_code_matches": rows[0].get("source_sha256") == source_digest(),
            "decision_mismatches": mismatches, "reproduced": not mismatches,
            "final_state": last.state, "final_reason": last.reason,
            "interpretation": "recorded feedback replay; no claim of alternative closed-loop physical outcome"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = replay(args.input)
        text = json.dumps(finite_json(report), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        print(text)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Replay rejected: {exc}\n")


if __name__ == "__main__":
    main()
