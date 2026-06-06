from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "train_config.json"


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

    required = ["model_file", "data_file", "project_dir", "run_name", "epochs", "imgsz", "batch"]
    missing = [name for name in required if name not in config]
    if missing:
        raise KeyError(f"训练配置缺少字段: {missing}")

    config["model_file"] = resolve_root_path(config["model_file"])
    config["data_file"] = resolve_root_path(config["data_file"])
    config["project_dir"] = resolve_root_path(config["project_dir"])
    return config


def check_paths(config: dict):
    data_path = Path(config["data_file"])
    if not data_path.exists():
        raise FileNotFoundError(f"没找到 dataset.yaml: {data_path}")

    project_path = Path(config["project_dir"])
    project_path.mkdir(parents=True, exist_ok=True)


def print_env_info():
    import torch

    print("===== Environment Check =====")
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"cuda device {i}: {torch.cuda.get_device_name(i)}")

    has_xpu = hasattr(torch, "xpu")
    xpu_available = has_xpu and torch.xpu.is_available()
    print("xpu available:", xpu_available)
    print("=============================\n")


def main():
    from ultralytics import YOLO

    config = load_train_config(CONFIG_PATH)
    check_paths(config)
    print_env_info()

    print("Loading model...")
    model = YOLO(config["model_file"])
    print("Model loaded.\n")

    train_kwargs = dict(
        data=config["data_file"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        project=config["project_dir"],
        name=config["run_name"],
        exist_ok=True,
    )

    if config.get("device") is not None:
        train_kwargs["device"] = config["device"]

    print("Start training...")
    results = model.train(**train_kwargs)

    print("\n训练完成")
    print(f"结果目录: {config['project_dir']}\\{config['run_name']}")
    print(f"best.pt 通常在: {config['project_dir']}\\{config['run_name']}\\weights\\best.pt")
    print(results)


if __name__ == "__main__":
    main()
