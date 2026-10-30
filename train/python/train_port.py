from __future__ import annotations

import argparse
from collections import Counter
from copy import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "train_config.json"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision.capture_files import atomic_capture_json
from vision.model_config import publish_current_model
from vision.training_library import normalize_label, read_entries, validate_library
from train.python.dataset_split import read_split, split_entries

GENERIC_PRETRAINED_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(path):
    return os.path.normcase(str(Path(path).resolve()))


def _write_report(run, **values):
    path = run / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report.update(values, updated_at=datetime.now(timezone.utc).isoformat())
    atomic_capture_json(path, report)


def verify_snapshot(run, manifest):
    """Check frozen bytes against their recorded identity without changing the run."""
    run = Path(run).resolve()
    errors = []
    entries = manifest["entries"]
    if len(entries) != manifest["image_count"] or len({item["image_sha256"] for item in entries}) != len(entries):
        raise ValueError("snapshot_manifest_image_inventory_mismatch")
    if "split_counts" in manifest:
        counts = dict(Counter(entry.get("split") for entry in entries))
        if set(counts) != {"train", "val"} or counts != manifest["split_counts"]:
            raise ValueError("snapshot_train_val_inventory_mismatch")
        grouped = {}
        for entry in entries:
            if grouped.setdefault(entry["split_group"], entry["split"]) != entry["split"]:
                raise ValueError("snapshot_group_crosses_train_and_val")

    def check(relative, expected_hash, kind):
        path = (run / relative).resolve()
        if not path.is_relative_to(run):
            errors.append(f"{kind}_path_outside_snapshot: {relative}")
            return
        try:
            if _sha(path) != expected_hash:
                errors.append(f"{kind}_hash_mismatch: {relative}")
        except OSError as exc:
            errors.append(f"{kind}_unreadable: {relative}: {exc}")

    for entry in entries:
        check(entry["image"], entry["image_sha256"], "image")
        check(entry["label"], entry["label_sha256"], "label")
    check(manifest.get("frozen_initial_weights", "initial.pt"), manifest["initial_weights_sha256"], "initial_weights")
    if manifest.get("dataset_yaml_sha256"):
        check("dataset.yaml", manifest["dataset_yaml_sha256"], "dataset_yaml")
    if errors:
        raise ValueError("snapshot_integrity_failed:\n" + "\n".join(errors))
    return {"schema_version": 1, "status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
            "images_checked": len(entries), "labels_checked": len(entries),
            "initial_weights_sha256": manifest["initial_weights_sha256"],
            "dataset_yaml_checked": bool(manifest.get("dataset_yaml_sha256")),
            "coverage_unit": "primary_training_image"}


def prepare_training(*, root=ROOT_DIR, library=None, weights=None, project=None,
                     run_prefix="detect_split", synchronize=True, split_config=None,
                     expected_weights_sha256=GENERIC_PRETRAINED_SHA256):
    """Freeze complete image/label/weight copies before YOLO can skip any sample."""
    root = Path(root).resolve()
    library = Path(library or root / "dataset/vision").resolve()
    weights = Path(weights or root / "yolov8n.pt").resolve()
    project = Path(project or root / "runs").resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"本地初始权重不存在，请用 --weights 指定已有 .pt 文件: {weights}")
    if _sha(weights) != expected_weights_sha256:
        raise ValueError("generic_pretrained_weights_hash_required: project-finetuned weights may contain validation images")
    split_path = Path(split_config or root / "config/port_dataset_split.json").resolve()
    split_policy = read_split(split_path)
    if not run_prefix or Path(run_prefix).name != run_prefix or run_prefix in {".", ".."}:
        raise ValueError("run_name_must_be_a_single_directory_name")
    synchronization = None
    if synchronize:
        from train.python.prepare_port_library import sync_library
        synchronization = sync_library(root=root, library=library)
    entries = split_entries(validate_library(library), split_policy)
    counts = dict(Counter(entry["split"] for entry in entries))
    run = project / (run_prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    _write_report(run, schema_version=1, status="preparing", library=str(library),
                  independent_validation=False, evaluation="held_out_development", split_counts=counts,
                  coverage_unit="primary_training_image",
                  scope="all ready images are assigned exactly once to train or validation; future groups require explicit assignment")
    try:
        dataset = run / "dataset"
        frozen = []
        for entry in entries:
            image_relative = Path(entry["image"])
            label_relative = Path(entry["label"])
            image_target = dataset / "images" / entry["split"] / Path(*image_relative.parts[1:])
            label_target = dataset / "labels" / entry["split"] / Path(*label_relative.parts[1:])
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(library / image_relative, image_target)
            shutil.copyfile(library / label_relative, label_target)
            if _sha(image_target) != entry["image_sha256"] or _sha(label_target) != entry["label_sha256"]:
                raise ValueError("library_changed_while_freezing_training_snapshot")
            label_text = normalize_label(label_target.read_text(encoding="utf-8"))
            frozen.append({**entry, "library_image": entry["image"], "library_label": entry["label"],
                           "image": image_target.relative_to(run).as_posix(),
                           "label": label_target.relative_to(run).as_posix(),
                           "instances": len(label_text.splitlines())})
        initial_hash = _sha(weights)
        shutil.copyfile(weights, run / "initial.pt")
        if _sha(run / "initial.pt") != initial_hash:
            raise ValueError("initial_weights_changed_while_copying")
        # JSON strings are valid YAML scalars, including Windows paths and Unicode.
        yaml = ("# Train and val are disjoint source groups. Historical session independence is unknown.\n"
                f"path: {json.dumps(dataset.as_posix(), ensure_ascii=False)}\n"
                "train: images/train\nval: images/val\nnames:\n  0: port\n")
        (run / "dataset.yaml").write_text(yaml, encoding="utf-8")
        all_entries = read_entries(library)
        manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                    "library": str(library), "image_count": len(frozen),
                    "split_counts": counts, "split_policy": split_policy, "split_config_sha256": _sha(split_path),
                    "initial_weights_kind": "generic_pretrained_coco",
                    "instance_count": sum(entry["instances"] for entry in frozen),
                    "entries": frozen, "initial_weights": str(weights), "initial_weights_sha256": initial_hash,
                    "frozen_initial_weights": "initial.pt", "independent_validation": False,
                    "coverage_unit": "primary_training_image", "dataset_yaml_sha256": _sha(run / "dataset.yaml"),
                    "code_sha256": {name: _sha(ROOT_DIR / name) for name in
                                    ("train/python/train_port.py", "vision/training_library.py",
                                     "train/python/prepare_port_library.py", "train/python/dataset_split.py",
                                     "vision/model_config.py")},
                    "evaluation": "held_out_development", "synchronization": synchronization,
                    "excluded": [{"image_sha256": entry["image_sha256"], "reason": entry["exclusion_reason"]}
                                 for entry in all_entries if entry["status"] == "excluded"]}
        atomic_capture_json(run / "training_manifest.json", manifest)
        _write_report(run, status="prepared", image_count=len(frozen), instance_count=manifest["instance_count"],
                      initial_weights_sha256=initial_hash, manifest_sha256=_sha(run / "training_manifest.json"))
        return run, manifest
    except BaseException as exc:
        _write_report(run, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


class CoverageAudit:
    """Check loaded labels and actual completed training batches for every epoch."""
    def __init__(self, run, manifest):
        self.run = Path(run)
        self.manifest = manifest
        self.by_split = {split: {_key(self.run / entry["image"]): entry for entry in manifest["entries"]
                                 if entry["split"] == split} for split in ("train", "val")}
        self.expected = self.by_split["train"]
        self.validation_seen = Counter()
        self.validation_passes = 0
        self.seen = Counter()
        self.batch_count = 0
        self.completed_epochs = 0

    def check_loader(self, loader, *, mode="train"):
        import numpy as np

        dataset = loader.dataset
        expected = self.by_split[mode]
        loaded = [_key(path) for path in dataset.im_files]
        if len(loaded) != len(expected) or set(loaded) != set(expected):
            raise ValueError(f"{mode}_loader_image_inventory_mismatch")
        if loader.drop_last or loader.num_workers != 0:
            raise ValueError("coverage_requires_drop_last_false_and_workers_zero")
        if mode == "train":
            sampler = loader.sampler
            if (type(sampler).__name__ not in {"RandomSampler", "SequentialSampler"}
                    or getattr(sampler, "replacement", False) or len(sampler) != len(expected)):
                raise ValueError("coverage_requires_full_without_replacement_sampler")
        if len(dataset.labels) != len(expected):
            raise ValueError(f"{mode}_loader_label_inventory_mismatch")
        actual_instances = 0
        for label in dataset.labels:
            entry = expected[_key(label["im_file"])]
            expected_text = (self.run / entry["label"]).read_text(encoding="utf-8")
            expected_rows = np.asarray([list(map(float, line.split())) for line in expected_text.splitlines()],
                                       dtype=float).reshape(-1, 5)
            actual_rows = np.concatenate((np.asarray(label["cls"]).reshape(-1, 1),
                                          np.asarray(label["bboxes"]).reshape(-1, 4)), axis=1)
            if (not label.get("normalized") or label.get("bbox_format") != "xywh"
                    or actual_rows.shape != expected_rows.shape
                    or not np.allclose(actual_rows, expected_rows, rtol=0, atol=1e-6)):
                raise ValueError(f"{mode}_loader_annotation_mismatch: {entry['image_sha256']}")
            actual_instances += len(actual_rows)
        if actual_instances != sum(entry["instances"] for entry in expected.values()):
            raise ValueError(f"{mode}_loader_instance_count_mismatch")
        return {"images": len(loaded), "instances": actual_instances, "drop_last": False, "workers": 0}

    def pretrain(self, trainer):
        if _key(trainer.save_dir) != _key(self.run):
            raise ValueError("trainer_run_directory_changed")
        _write_report(self.run, status="training", snapshot_pretrain_check=verify_snapshot(self.run, self.manifest),
                      train_loader=self.check_loader(trainer.train_loader),
                      validation_loader=self.check_loader(trainer.test_loader, mode="val"))

    def start_validation(self, validator):
        self.check_loader(validator.dataloader, mode="val")
        self.validation_seen.clear()
        validator._coverage_batch_paths = None

    def complete_validation_batch(self, validator):
        paths = getattr(validator, "_coverage_batch_paths", None)
        if not paths:
            raise ValueError("validation_batch_without_image_provenance")
        keys = [_key(path) for path in paths]
        if any(key not in self.by_split["val"] for key in keys):
            raise ValueError("validation_batch_contains_unplanned_images")
        self.validation_seen.update(keys)
        validator._coverage_batch_paths = None

    def complete_validation(self, validator):
        expected = self.by_split["val"]
        missing = sorted(expected[key]["image_sha256"] for key in expected.keys() - self.validation_seen.keys())
        repeated = {expected[key]["image_sha256"]: count for key, count in self.validation_seen.items() if count != 1}
        report = {"pass": self.validation_passes + 1, "after_training_epoch": self.completed_epochs,
                  "validation_images": len(expected), "unique_images": len(self.validation_seen),
                  "complete": not missing and not repeated, "missing": missing, "repeated": repeated,
                  "evaluation": "held_out_development", "independent_validation": False,
                  "seen_image_sha256": sorted(expected[key]["image_sha256"] for key in self.validation_seen)}
        with (self.run / "validation_coverage.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")
        if not report["complete"]:
            raise ValueError("validation_did_not_visit_every_image_once")
        self.validation_passes += 1
        _write_report(self.run, validation_passes=self.validation_passes, last_validation_coverage=report)

    def start_epoch(self, trainer):
        self.seen.clear()
        self.batch_count = 0
        trainer._coverage_batch_paths = None

    def complete_batch(self, trainer):
        paths = getattr(trainer, "_coverage_batch_paths", None)
        if paths is None:
            raise ValueError("completed_batch_without_image_provenance")
        keys = [_key(path) for path in paths]
        if not keys or any(key not in self.expected for key in keys):
            raise ValueError("batch_contains_unplanned_images")
        self.seen.update(keys)
        self.batch_count += 1
        trainer._coverage_batch_paths = None

    def complete_epoch(self, trainer):
        missing = sorted(self.expected[key]["image_sha256"] for key in self.expected.keys() - self.seen.keys())
        repeated = {self.expected[key]["image_sha256"]: count for key, count in self.seen.items() if count != 1}
        report = {"epoch": trainer.epoch + 1, "expected_images": len(self.expected),
                  "coverage_unit": "primary_training_image",
                  "unique_images": len(self.seen), "image_visits": sum(self.seen.values()),
                  "batches": self.batch_count, "missing": missing, "repeated": repeated,
                  "complete": not missing and not repeated,
                  "seen_image_sha256": sorted(self.expected[key]["image_sha256"] for key in self.seen)}
        with (self.run / "epoch_coverage.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _write_report(self.run, last_epoch_coverage=report, completed_epochs=self.completed_epochs + int(report["complete"]))
        if not report["complete"]:
            raise ValueError(f"epoch_{trainer.epoch + 1}_did_not_train_every_image_once")
        self.completed_epochs += 1


def _audited_trainer():
    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

    class AuditedDetectionValidator(DetectionValidator):
        def preprocess(self, batch):
            self._coverage_batch_paths = list(batch["im_file"])
            return super().preprocess(batch)

    class AuditedDetectionTrainer(DetectionTrainer):
        def get_validator(self):
            return AuditedDetectionValidator(self.test_loader, save_dir=self.save_dir,
                                             args=copy(self.args), _callbacks=self.callbacks)

        def preprocess_batch(self, batch):
            self._coverage_batch_paths = list(batch["im_file"])
            return super().preprocess_batch(batch)

    return AuditedDetectionTrainer


def validate_device(device, torch_module=None):
    device = str(device)
    if device == "cpu":
        return {"requested": "cpu", "type": "cpu", "explicit_cpu": True}
    if not device.isdigit():
        raise ValueError("device_must_be_cpu_or_a_single_CUDA_index")
    if torch_module is None:
        import torch as torch_module
    index = int(device)
    if not torch_module.cuda.is_available() or index >= torch_module.cuda.device_count():
        raise RuntimeError("CUDA_device_unavailable: install matching CUDA PyTorch or explicitly use --device cpu; no CPU fallback")
    return {"requested": device, "type": "cuda", "name": torch_module.cuda.get_device_name(index),
            "capability": list(torch_module.cuda.get_device_capability(index)),
            "torch": torch_module.__version__, "cuda_runtime": torch_module.version.cuda}


def train_all(run, manifest, config, *, root=ROOT_DIR):
    from ultralytics import YOLO, __version__

    run = Path(run)
    audit = CoverageAudit(run, manifest)
    parameters = {key: config[key] for key in ("epochs", "imgsz", "batch", "device")}
    device_info = validate_device(config["device"])
    parameters.update(workers=0, optimizer="AdamW", lr0=.0003, cos_lr=True, freeze=0, seed=0,
                      deterministic=True, patience=0, fraction=1.0, compile=False,
                      cache=False, amp=False, plots=True, save=True, val=True)
    _write_report(run, requested_epochs=parameters["epochs"], training_parameters=parameters,
                  coverage_unit="primary_training_image",
                  ultralytics_version=__version__, python_version=sys.version, device=device_info)
    try:
        _write_report(run, snapshot_before_model_load=verify_snapshot(run, manifest))
        model = YOLO(run / "initial.pt")
        for event, callback in (("on_pretrain_routine_end", audit.pretrain),
                                ("on_train_epoch_start", audit.start_epoch),
                                ("on_train_batch_end", audit.complete_batch),
                                ("on_train_epoch_end", audit.complete_epoch),
                                ("on_val_start", audit.start_validation),
                                ("on_val_batch_end", audit.complete_validation_batch),
                                ("on_val_end", audit.complete_validation)):
            model.add_callback(event, callback)
        model.train(trainer=_audited_trainer(), data=str(run / "dataset.yaml"),
                    project=str(run.parent), name=run.name, exist_ok=True, **parameters)
        if audit.completed_epochs != parameters["epochs"]:
            raise ValueError("training_stopped_before_all_requested_epochs")
        if audit.validation_passes < parameters["epochs"]:
            raise ValueError("validation_coverage_missing_for_requested_epochs")
        _write_report(run, snapshot_posttrain_check=verify_snapshot(run, manifest))
        best = run / "weights/best.pt"
        if not best.is_file():
            raise ValueError("training_finished_without_best_weights")
        preview = Path(root) / "config/d415_preview.json"
        preview_config = json.loads(preview.read_text(encoding="utf-8"))
        preview_config.pop("model_registry", None)
        preview_config["description"] = (
            f"YOLOv8n run {run.name}: {manifest['split_counts']['train']} train / "
            f"{manifest['split_counts']['val']} development-val images, {audit.completed_epochs} epochs "
            "from generic COCO pretrained weights. No independent test or hole-pose validation."
        )
        preview_config["weights"] = str(best.resolve())
        preview_config["weights_sha256"] = _sha(best)
        atomic_capture_json(run / "preview_config.json", preview_config)
        _write_report(run, status="completed", completed_epochs=audit.completed_epochs,
                      best_weights=str(best.resolve()), best_weights_sha256=_sha(best),
                      preview_config=str((run / "preview_config.json").resolve()))
        publish_current_model(best, root=root, training_report=run / "training_report.json")
    except BaseException as exc:
        _write_report(run, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}", completed_epochs=audit.completed_epochs)
        raise
    return json.loads((run / "training_report.json").read_text(encoding="utf-8"))


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_train_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"训练配置不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required = ["model_file", "split_config", "project_dir", "run_name", "epochs", "imgsz", "batch",
                "pretrained_weights_sha256", "device"]
    missing = [name for name in required if name not in config]
    if missing:
        raise KeyError(f"训练配置缺少字段: {missing}")

    config["model_file"] = resolve_root_path(config["model_file"])
    config["split_config"] = resolve_root_path(config["split_config"])
    config["project_dir"] = resolve_root_path(config["project_dir"])
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="充电口按固定来源组训练；全库覆盖 train+val，开发验证保留历史会话不明限制")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--split-config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--prepare-only", action="store_true", help="只同步、检查和冻结全量数据，不启动训练")
    args = parser.parse_args(argv)
    config = load_train_config(args.config)
    for key in ("epochs", "batch", "imgsz", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.weights is not None:
        config["model_file"] = args.weights.resolve()
    for key in ("epochs", "batch", "imgsz"):
        if type(config[key]) is not int or config[key] <= 0:
            parser.error(f"{key} 必须为正整数")
    if not Path(config["model_file"]).is_file():
        parser.error(f"需要本地模型文件（不自动下载）: {config['model_file']}")
    device = str(config.get("device", "cpu"))
    if "," in device or device in {"-1", "[-1]"}:
        parser.error("全量覆盖审计仅支持 CPU 或单个设备，请使用 --device cpu 或 --device 0")
    config["device"] = device
    if not args.prepare_only:
        validate_device(device)
    library = args.library.resolve() if args.library else resolve_root_path(config.get("library", "dataset/vision"))
    run, manifest = prepare_training(library=library, weights=config["model_file"],
                                     project=config["project_dir"], run_prefix=config.get("run_name", "detect_split"),
                                     split_config=args.split_config or config["split_config"],
                                     expected_weights_sha256=config["pretrained_weights_sha256"])
    print(f"全量快照：{manifest['image_count']} 张图片，{manifest['instance_count']} 个框。")
    print(f"来源组划分: train={manifest['split_counts']['train']}, val={manifest['split_counts']['val']}；两者不重叠。")
    print("验证图不参与本轮训练；历史照片会话未知，结果定位为开发验证，不保证严格会话独立。")
    print(f"本轮目录: {run}", flush=True)
    if not args.prepare_only:
        report = train_all(run, manifest, config)
        print(f"训练完成；{report['completed_epochs']} 轮全部训练图及验证图覆盖检查通过。")
        print(f"新模型: {report['best_weights']}")
        print(f"预览配置: {report['preview_config']}")
        print("当前模型已更新；下次 vision.preview --depth、D435 定位和批量评估使用此权重。")


if __name__ == "__main__":
    main()
