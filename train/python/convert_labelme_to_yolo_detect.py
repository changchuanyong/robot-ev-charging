from __future__ import annotations

import json
from pathlib import Path
import argparse
import hashlib
import sys


# ===== 你只改这里 =====
ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "dataset/vision"
LABEL_MAP = {
    "port": 0,
    "充电口": 0,
    "接口": 0,
}
# ======================


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image(json_path: Path, data: dict, input_dir: Path) -> Path | None:
    # 先用 JSON 里的 imagePath 找
    image_path_str = data.get("imagePath")
    if image_path_str:
        p = Path(image_path_str)
        candidates = [
            json_path.parent / p.name,
            input_dir / p.name,
        ]
        for c in candidates:
            if c.exists():
                return c

    # 再用 json 同名找
    stem = json_path.stem
    for ext in IMG_EXTS:
        c = input_dir / f"{stem}{ext}"
        if c.exists():
            return c

    return None


def shape_to_bbox(shape: dict, img_w: int, img_h: int):
    label = shape.get("label", "").strip()
    if label not in LABEL_MAP:
        return None

    points = shape.get("points", [])
    if len(points) < 2:
        return None

    shape_type = shape.get("shape_type", "polygon")

    if shape_type == "rectangle" and len(points) >= 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

    xmin = max(0.0, min(float(img_w), xmin))
    xmax = max(0.0, min(float(img_w), xmax))
    ymin = max(0.0, min(float(img_h), ymin))
    ymax = max(0.0, min(float(img_h), ymax))

    bw = xmax - xmin
    bh = ymax - ymin
    if bw < 2 or bh < 2:
        return None

    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = bw / img_w
    height = bh / img_h

    class_id = LABEL_MAP[label]
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_to_library(input_dir, library=OUT_DIR, *, root=ROOT_DIR):
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from vision.photo_sources import resolve_photo
    from vision.training_library import normalize_label, read_entries, register_image, set_annotation

    library, root = Path(library).resolve(), Path(root).resolve()
    known = {e["image_sha256"]: e for e in read_entries(library)}
    index_path = root / "config/legacy_photo_index.json"
    aliases = json.loads(index_path.read_text(encoding="utf-8"))["images"] if index_path.exists() else []
    converted, retained, skipped = 0, 0, 0
    for json_path in sorted(Path(input_dir).rglob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        if not data.get("imageWidth") or not data.get("imageHeight"):
            raise ValueError(f"Labelme 缺少图像尺寸: {json_path}")
        image = find_image(json_path, data, json_path.parent)
        if image is None:
            matches = [e for e in aliases if Path(e["legacy_image"]).stem == json_path.stem
                       and Path(e["legacy_image"]).parent.name == json_path.parent.name]
            if len(matches) == 1:
                image = resolve_photo(matches[0]["legacy_image"], root=root, library=library)
        if image is None:
            raise FileNotFoundError(f"Labelme 对应原图缺失: {json_path}")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in known and known[digest]["status"] != "pending":
            retained += 1  # Preserve improved/reviewed labels and exclusions.
            continue
        lines = [line for shape in data.get("shapes", [])
                 if (line := shape_to_bbox(shape, data["imageWidth"], data["imageHeight"]))]
        if not lines:
            skipped += 1  # Missing target shapes never become automatic negatives.
            continue
        label = normalize_label("\n".join(lines))
        record = register_image(image, library_root=library,
                                source={"labelme_file": str(json_path), "import_kind": "labelme"})
        annotation_bytes = json_path.read_bytes()
        archive = library / "source_annotations/labelme" / digest / (hashlib.sha256(annotation_bytes).hexdigest() + ".json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() and archive.read_bytes() != annotation_bytes:
            raise ValueError("labelme_archive_hash_mismatch")
        if not archive.exists():
            archive.write_bytes(annotation_bytes)
        known[digest] = set_annotation(library, record["image_sha256"], label,
                                      annotation_source=f"labelme:{archive.relative_to(library).as_posix()}",
                                      human_reviewed=False)
        converted += 1
    return {"converted": converted, "existing_annotations_preserved": retained, "without_target_shapes": skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description="导入 Labelme 到统一图库；保留已有复核标签")
    parser.add_argument("--library", type=Path, default=OUT_DIR)
    parser.add_argument("--input-dir", type=Path, help="含原图与Labelme JSON的目录；默认使用迁移归档")
    args = parser.parse_args(argv)
    source = args.input_dir or args.library / "source_annotations/yolo_port/images"
    if not source.is_dir():
        parser.error(f"Labelme 输入目录不存在: {source}")
    print(json.dumps(convert_to_library(source, args.library), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
