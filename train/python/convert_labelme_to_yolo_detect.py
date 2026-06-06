from __future__ import annotations

import json
from pathlib import Path


# ===== 你只改这里 =====
ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "yolo_port"
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


def convert_split(split: str):
    input_dir = OUT_DIR / "images" / split
    label_dir = OUT_DIR / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"[提示] {input_dir} 下没有 json 文件，跳过 {split}")
        return 0, 0

    converted = 0
    skipped = 0

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            print(f"[跳过] 缺少 imageWidth/imageHeight: {json_path.name}")
            skipped += 1
            continue

        image_path = find_image(json_path, data, input_dir)
        if image_path is None:
            print(f"[跳过] 找不到对应图片: {json_path.name}")
            skipped += 1
            continue

        yolo_lines = []
        for shape in data.get("shapes", []):
            line = shape_to_bbox(shape, img_w, img_h)
            if line:
                yolo_lines.append(line)

        if not yolo_lines:
            print(f"[跳过] 没有有效标注: {json_path.name}")
            skipped += 1
            continue

        dst_lbl = label_dir / f"{image_path.stem}.txt"
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        converted += 1
        print(f"[完成][{split}] {json_path.name} -> {dst_lbl.name}")

    return converted, skipped


def write_yaml():
    dataset_root = OUT_DIR.relative_to(ROOT_DIR).as_posix()
    yaml_text = f"""path: {dataset_root}
train: images/train
val: images/val
names:
  0: port
"""
    with open(OUT_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)


def main():
    train_converted, train_skipped = convert_split("train")
    val_converted, val_skipped = convert_split("val")
    write_yaml()

    print("\n转换完成")
    print(f"数据集根目录: {OUT_DIR}")
    print(f"train 成功: {train_converted}, 跳过: {train_skipped}")
    print(f"val   成功: {val_converted}, 跳过: {val_skipped}")
    print(f"YAML: {OUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
