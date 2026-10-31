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

from vision.contracts import atomic_capture_json
from vision.training_library import _lock, read_entries

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
    declared = [s for s in sources if s.get("original_split") in {"train", "val"}]
    batches = {s["batch_id"] for s in declared if s.get("batch_id")}
    specimens = {s.get("metadata", {}).get("specimen_id") for s in declared} - {None, ""}
    printing_batches = {s.get("metadata", {}).get("printing_batch_id") for s in declared} - {None, ""}
    paths = {str(s[key]).replace("\\", "/").lower() for s in sources
             for key in ("path", "original_image", "manifest") if s.get(key)}
    matched = []
    for name, rule in config["groups"].items():
        if (original_groups.intersection(rule.get("original_groups", []))
                or sessions.intersection(rule.get("session_ids", []))
                or batches.intersection(rule.get("batch_ids", []))
                or any(marker.lower() in path for marker in rule.get("source_path_markers", []) for path in paths)):
            matched.append(name)
    # Explicit historical/session assignments remain authoritative. A specimen
    # alias only groups an explicitly assigned source. Unknown historical
    # captures must not acquire a split when another batch names their specimen.
    if not matched:
        matched = [name for name, rule in config["groups"].items()
                   if specimens.intersection(rule.get("specimen_ids", []))
                   or printing_batches.intersection(rule.get("printing_batch_ids", []))]
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


def _capture_ids(batch_id, metadata, session_id=None):
    if not isinstance(batch_id, str) or not batch_id.strip() or batch_id != batch_id.strip():
        raise ValueError("capture_batch_id_required")
    if not isinstance(metadata, dict):
        raise ValueError("capture_metadata_must_be_object")
    result = {"batch_ids": {batch_id}, "session_ids": set(),
              "specimen_ids": set(), "printing_batch_ids": set()}
    for value, key in ((session_id, "session_ids"),
                       (metadata.get("specimen_id"), "specimen_ids"),
                       (metadata.get("printing_batch_id"), "printing_batch_ids")):
        if value is not None:
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"invalid_capture_{key}")
            result[key].add(value)
    return result


def _capture_plan(library_root, batch_id, split, metadata, config, session_id=None):
    identities = _capture_ids(batch_id, metadata, session_id)
    if split not in {"train", "val", "unknown"}:
        raise ValueError("capture_library_split_requires_train_val_or_unknown; test requires --no-library")
    if split == "unknown":
        return {"group": None, "split": "unknown", "assigned": False}, identities
    direct, related, aliases = set(), set(), set()
    for name, rule in config["groups"].items():
        shared = [key for key, values in identities.items() if values.intersection(rule.get(key, []))]
        if shared:
            if rule["split"] != split:
                raise ValueError(f"capture_split_conflict: {shared} already belong to {name} ({rule['split']})")
            related.add(name)
            aliases.add(name)
            if "batch_ids" in shared or "session_ids" in shared:
                direct.add(name)
    for entry in read_entries(library_root):
        matching = []
        for source in entry["sources"]:
            source_ids = {
                "batch_ids": {source.get("batch_id")}, "session_ids": {source.get("session_id")},
                "specimen_ids": {source.get("metadata", {}).get("specimen_id")},
                "printing_batch_ids": {source.get("metadata", {}).get("printing_batch_id")},
            }
            matching.extend(key for key, values in identities.items() if values.intersection(source_ids[key]))
        if not matching:
            continue
        group = group_for(entry, config)
        previous = config["groups"].get(group)
        observed = {source.get("original_split") for source in entry["sources"]} - {None, "unknown", ""}
        if previous:
            observed.add(previous["split"])
            related.add(group)
            if "batch_ids" in matching or "session_ids" in matching:
                direct.add(group)
        if observed - {split}:
            raise ValueError(f"capture_split_conflict: {entry['image_sha256']} has {sorted(observed)}, requested {split}")
    if len(direct) > 1:
        raise ValueError(f"capture_matches_multiple_fixed_groups: {sorted(direct)}")
    if not direct and len(aliases) > 1:
        raise ValueError(f"capture_matches_multiple_identity_groups: {sorted(aliases)}")
    group = (next(iter(direct)) if direct else next(iter(aliases)) if aliases
             else next(iter(related)) if len(related) == 1
             else "capture_batch:" + batch_id)
    existing = config["groups"].get(group)
    if existing and existing["split"] != split:
        raise ValueError(f"capture_split_conflict: {group} already assigned to {existing['split']}")
    return {"group": group, "split": split, "assigned": True}, identities


def check_capture_split(library_root, batch_id, split, metadata, config_path=DEFAULT_SPLIT):
    """Read-only preflight before opening a camera; no group is created yet."""
    result, _ = _capture_plan(library_root, batch_id, split, metadata, read_split(config_path))
    return result


def ensure_capture_split(library_root, manifest, config_path=DEFAULT_SPLIT):
    """Record an explicit whole-capture assignment without moving existing groups."""
    config_path = Path(config_path).resolve()
    if not isinstance(manifest.get("session_id"), str) or not manifest["session_id"].strip():
        raise ValueError("capture_session_id_required")
    # The registry lock is in config/, separate from the image-library lock.
    with _lock(config_path.parent):
        config = read_split(config_path)
        result, identities = _capture_plan(library_root, manifest.get("batch_id"),
                                           manifest.get("split", "unknown"), manifest.get("metadata", {}),
                                           config, manifest["session_id"])
        if not result["assigned"]:
            return {**result, "changed": False}
        group = config["groups"].get(result["group"])
        changed = group is None
        if group is None:
            group = {"split": result["split"], "description": "Explicit whole-capture assignment"}
            config["groups"][result["group"]] = group
        for key, values in identities.items():
            combined = sorted(set(group.get(key, [])) | values)
            if combined and combined != group.get(key):
                group[key] = combined
                changed = True
        if changed:
            atomic_capture_json(config_path, config)
        return {**result, "changed": changed}


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
        with _lock(args.config.resolve().parent):
            config = read_split(args.config)
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
