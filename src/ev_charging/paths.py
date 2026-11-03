"""Locate packaged code separately from the project's data and configuration."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def find_project_root(*, require_data: bool = True) -> Path:
    """Locate a known project; explicit validation still requires its data."""
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
        if (
            not require_data
            and candidate.is_dir()
            and (
                explicit
                or ((candidate / "pyproject.toml").is_file() and (candidate / "src/ev_charging").is_dir())
            )
        ):
            return candidate
    raise RuntimeError(
        "找不到工程数据目录；请从工程目录运行，或将 EV_CHARGING_ROOT 指向包含 config/ 和 dataset/ 的目录。"
    )


PROJECT_ROOT = find_project_root(require_data=False)


def source_path(relative: str) -> Path:
    """Resolve a source manifest entry against the code actually being imported."""
    prefix = "src/ev_charging/"
    if not relative.startswith(prefix):
        raise ValueError(f"Not a package source path: {relative}")
    path = (PACKAGE_ROOT / relative[len(prefix) :]).resolve()
    if not path.is_relative_to(PACKAGE_ROOT):
        raise ValueError(f"Source path escapes package: {relative}")
    return path
