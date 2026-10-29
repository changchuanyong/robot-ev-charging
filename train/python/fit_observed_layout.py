"""Fit image-space correspondence templates from reviewed training annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "live/python"))
from observed_layout import DEFAULT_PRIOR_PATH, fit_observed_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_PRIOR_PATH)
    args = parser.parse_args()
    if args.annotations.resolve() == args.output.resolve():
        raise ValueError("Output must not replace annotations")
    content = args.annotations.read_bytes()
    annotations = json.loads(content)
    prior = fit_observed_layout(annotations)
    prior["annotation_sha256"] = hashlib.sha256(content).hexdigest()
    prior["source_note"] = "Only reviewed train rows contribute to templates; validation rows are excluded."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately refuse to replace a fitted prior silently.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(prior, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"templates": len(prior["templates"]), "fit_split": "train",
                      "training_images": len(prior["training_images"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
