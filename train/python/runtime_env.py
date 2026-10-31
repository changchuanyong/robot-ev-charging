"""Keep Ultralytics, Matplotlib and temporary files inside the project.

Ultralytics keeps its settings and its downloaded fonts in the user profile
(``%APPDATA%\\Ultralytics``) and Matplotlib caches fonts in
``%LOCALAPPDATA%\\matplotlib``.  A process whose writes are confined to the
project cannot create either directory, and Ultralytics reuses the existing
read-only directory instead of falling back to a temporary one, so
``check_font`` aborts the whole run before the first batch with
``PermissionError ... .Arial.ttf.<hash>.part``.

Redirecting both caches into ``.cache/`` also removes the network dependency:
``Arial.ttf`` is copied from the installed Windows fonts, so an offline machine
trains without contacting ultralytics.com.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil

CACHE_DIR = ".cache"
FONT_SOURCES = {"Arial.ttf": ("arial.ttf", "Arial.ttf")}


def _seed_fonts(config_dir: Path) -> list[str]:
    """Copy a locally installed font so Ultralytics never has to download one."""
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    seeded = []
    for name, candidates in FONT_SOURCES.items():
        target = config_dir / name
        if target.is_file():
            continue
        source = next((fonts_dir / candidate for candidate in candidates
                       if (fonts_dir / candidate).is_file()), None)
        if source is None:
            continue
        staged = config_dir / f".{name}.part"
        try:
            shutil.copyfile(source, staged)
            os.replace(staged, target)
        except OSError:
            staged.unlink(missing_ok=True)
            continue
        seeded.append(name)
    return seeded


def _require_writable(directory: Path) -> None:
    """Fail early and clearly instead of deep inside a training run."""
    probe = directory / ".write_probe"
    try:
        with probe.open("w", encoding="utf-8"):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"缓存目录不可写: {directory}；训练进程只能写项目目录，请删除同名环境变量覆盖，"
            f"或在项目目录内运行训练") from exc
    probe.unlink(missing_ok=True)


class _SerialPool:
    """Stand-in for ``multiprocessing.pool.ThreadPool`` that runs work inline."""

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def imap(self, func, iterable, chunksize=1):
        return map(func, iterable)


def _named_pipes_available() -> bool:
    """A multiprocessing pool signals through an OS pipe, which some sandboxes refuse."""
    from multiprocessing.connection import Pipe

    try:
        reader, writer = Pipe(duplex=False)
    except OSError:
        return False
    reader.close()
    writer.close()
    return True


def serialise_label_scan() -> bool:
    """Scan labels inline when Ultralytics' thread pool cannot get an OS pipe.

    ``cache_labels`` builds a ``ThreadPool`` once per dataset; creating that pool
    fails with ``PermissionError [WinError 5]`` where named pipes are blocked,
    before the first epoch.  Returns True when the fallback was installed.
    """
    if _named_pipes_available():
        return False
    import ultralytics.data.base
    import ultralytics.data.dataset

    patched = False
    for module in (ultralytics.data.base, ultralytics.data.dataset):
        if callable(getattr(module, "ThreadPool", None)):
            module.ThreadPool = _SerialPool
            patched = True
    return patched


def configure_caches(root: Path | None = None) -> dict[str, object]:
    """Point every third-party cache at ``<root>/.cache`` and describe the layout.

    Must run before ``import ultralytics`` and ``import matplotlib``: both read
    their directories, and Ultralytics downloads its font, at import time.
    Pre-set ``YOLO_CONFIG_DIR`` and ``MPLCONFIGDIR`` are respected, which is the
    escape hatch when a caller really wants the user profile.
    """
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    cache = root / CACHE_DIR
    temporary = cache / "tmp"
    # Ultralytics appends its own "Ultralytics" subdirectory to YOLO_CONFIG_DIR.
    configured = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(cache)))
    matplotlib_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib")))
    ultralytics_dir = configured.expanduser().resolve() / "Ultralytics"
    matplotlib_dir = matplotlib_dir.expanduser().resolve()
    for directory in (temporary, ultralytics_dir, matplotlib_dir):
        directory.mkdir(parents=True, exist_ok=True)
        _require_writable(directory)
    seeded = _seed_fonts(ultralytics_dir)
    # An unwritable OS temp directory makes tempfile fall back to the working
    # directory, which drops matplotlib-* and torchinductor_* caches into the
    # repository root.  Keep that scratch space inside the project instead.
    for name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[name] = str(temporary)
    return {"root": str(root), "ultralytics": str(ultralytics_dir),
            "matplotlib": str(matplotlib_dir), "temporary": str(temporary),
            "seeded_fonts": seeded}
