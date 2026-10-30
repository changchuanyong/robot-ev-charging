from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset/vision"
SPLITS = ("train", "val")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_stems(path: Path, suffixes: tuple[str, ...]) -> set[str]:
    stems: set[str] = set()
    if not path.exists():
        return stems

    for suffix in suffixes:
        stems.update(p.stem for p in path.glob(f"*{suffix}"))
    return stems


def validate_label_file(path: Path) -> list[str]:
    errors: list[str] = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{line_no}: expected 5 columns, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            values = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"{path.name}:{line_no}: non-numeric YOLO label")
            continue

        if class_id < 0:
            errors.append(f"{path.name}:{line_no}: negative class id")

        for value in values:
            if value < 0.0 or value > 1.0:
                errors.append(f"{path.name}:{line_no}: bbox value out of [0, 1]: {value}")

        width = values[2]
        height = values[3]
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path.name}:{line_no}: non-positive width/height")

    return errors


def print_items(title: str, items: set[str]) -> None:
    print(f"{title}: {len(items)}")
    if items:
        print("  " + ", ".join(sorted(items)))


def check_split(split: str) -> int:
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    image_stems = collect_stems(image_dir, IMAGE_EXTS)
    json_stems = collect_stems(image_dir, (".json",))
    label_stems = collect_stems(label_dir, (".txt",))

    missing_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems
    images_without_json = image_stems - json_stems
    json_without_images = json_stems - image_stems

    print(f"\n===== {split} =====")
    print(f"images: {len(image_stems)}")
    print(f"json  : {len(json_stems)}")
    print(f"labels: {len(label_stems)}")
    print_items("missing labels", missing_labels)
    print_items("labels without images", labels_without_images)
    print_items("images without Labelme json", images_without_json)
    print_items("json without images", json_without_images)

    label_errors: list[str] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        label_errors.extend(validate_label_file(label_path))

    print(f"invalid label lines: {len(label_errors)}")
    for error in label_errors:
        print(f"  {error}")

    return sum(
        [
            len(missing_labels),
            len(labels_without_images),
            len(json_without_images),
            len(label_errors),
        ]
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="检查统一图库的图片、标签、哈希和待处理状态")
    parser.add_argument("--library", type=Path, default=DATASET_DIR)
    args = parser.parse_args(argv)
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from vision.training_library import validate_library
    entries = validate_library(args.library)
    print(f"Dataset: {args.library.resolve()}\nReady images: {len(entries)}\nTotal blocking issues: 0")


if __name__ == "__main__":
    main()
