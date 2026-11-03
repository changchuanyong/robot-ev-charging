"""Locate packaged code separately from the project's data and configuration."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def find_project_root() -> Path:
    """Prefer this editable checkout; installed wheels can use EV_CHARGING_ROOT."""
    explicit = os.environ.get("EV_CHARGING_ROOT")
    candidates = (
        [Path(explicit).expanduser().resolve()]
        if explicit
        else [
            *PACKAGE_ROOT.parents,
            Path.cwd(),
            *Path.cwd().parents,
        ]
    )
    for candidate in candidates:
        if (candidate / "config/vision_model.json").is_file() and (candidate / "dataset/vision").is_dir():
            return candidate
    raise RuntimeError(
        "找不到工程数据目录；请从工程目录运行，或将 EV_CHARGING_ROOT 指向包含 config/ 和 dataset/ 的目录。"
    )


PROJECT_ROOT = find_project_root()


def source_path(relative: str) -> Path:
    """Resolve a source manifest entry against the code actually being imported."""
    prefix = "src/ev_charging/"
    if not relative.startswith(prefix):
        raise ValueError(f"Not a package source path: {relative}")
    path = (PACKAGE_ROOT / relative[len(prefix) :]).resolve()
    if not path.is_relative_to(PACKAGE_ROOT):
        raise ValueError(f"Source path escapes package: {relative}")
    return path
