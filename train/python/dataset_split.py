"""Explicit, stable group assignments for the consolidated photo library."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.capture_files import atomic_capture_json
from vision.training_library import read_entries

DEFAULT_SPLIT = ROOT / "config/port_dataset_split.json"


def read_split(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != 1 or not isinstance(config.get("groups"), dict):
        raise ValueError("invalid_split_registry")
    for name, group in config["groups"].items():
        if not name or group.get("split") not in {"train", "val"}:
            raise ValueError(f"invalid_group_split: {name}")
    return config


def group_for(entry, config):
    sources = entry["sources"]
    original_groups = {s["original_group"] for s in sources if s.get("original_group")}
    sessions = {s["session_id"] for s in sources if s.get("session_id")}
    paths = {str(s[key]).replace("\\", "/").lower() for s in sources
             for key in ("path", "original_image", "manifest") if s.get(key)}
    matched = []
    for name, rule in config["groups"].items():
        if (original_groups.intersection(rule.get("original_groups", []))
                or sessions.intersection(rule.get("session_ids", []))
                or any(marker.lower() in path for marker in rule.get("source_path_markers", []) for path in paths)):
            matched.append(name)
    if len(matched) > 1:
        raise ValueError(f"image_matches_multiple_groups: {entry['image_sha256']}: {matched}")
    if matched:
        return matched[0]
    if len(original_groups) == 1:
        return "collection:" + next(iter(original_groups))
    if len(sessions) == 1:
        return "session:" + next(iter(sessions))
    if len(sessions) > 1 or len(original_groups) > 1:
        raise ValueError(f"explicit_merged_group_required: {entry['image_sha256']}")
    originals = sorted(str(s["path"]).replace("\\", "/") for s in sources if s.get("path"))
    if originals:
        return "source_directory:" + originals[0].rsplit("/", 1)[0]
    raise ValueError(f"capture_group_metadata_required: {entry['image_sha256']}")


def split_entries(entries, config):
    assigned, pending = [], Counter()
    seen = set()
    for entry in entries:
        digest = entry["image_sha256"]
        if digest in seen:
            raise ValueError("duplicate_image_in_split_inventory")
        seen.add(digest)
        group = group_for(entry, config)
        if group not in config["groups"]:
            pending[group] += 1
        else:
            assigned.append({**entry, "split_group": group, "split": config["groups"][group]["split"]})
    if pending:
        raise ValueError("pending_split: explicitly assign these groups before training:\n" +
                         "\n".join(f"{group} ({count} images)" for group, count in sorted(pending.items())))
    counts = Counter(entry["split"] for entry in assigned)
    if not counts["train"] or not counts["val"]:
        raise ValueError("nonempty_train_and_val_required")
    return assigned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--library", type=Path, default=ROOT / "dataset/vision")
    parser.add_argument("--list-pending", action="store_true")
    parser.add_argument("--assign-group")
    parser.add_argument("--split", choices=["train", "val"])
    args = parser.parse_args(argv)
    config = read_split(args.config)
    entries = [entry for entry in read_entries(args.library) if entry["status"] == "ready"]
    observed = Counter(group_for(entry, config) for entry in entries)
    if args.assign_group:
        if not args.split:
            parser.error("--assign-group requires --split train|val")
        if args.assign_group not in observed:
            parser.error("group is not present in the ready library")
        existing = config["groups"].get(args.assign_group)
        if existing and existing["split"] != args.split:
            parser.error("existing assignments are fixed; review and version the JSON to change a group")
        config["groups"].setdefault(args.assign_group, {"split": args.split,
                                   "description": "Explicit assignment of a newly observed source group"})
        atomic_capture_json(args.config, config)
    print(json.dumps({"groups": [{"group": group, "images": count,
                                  "split": config["groups"].get(group, {}).get("split", "pending_split")}
                                 for group, count in sorted(observed.items())],
                      "pending_groups": [group for group in sorted(observed) if group not in config["groups"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
