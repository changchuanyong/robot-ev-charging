"""Review original-pixel hole centers across the library with local disk saves."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import secrets
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from ev_charging.datasets.vision.training_library import DEFAULT_LIBRARY, _inside, _lock, read_entries
from ev_charging.file_io import atomic_capture_json
from ev_charging.paths import PROJECT_ROOT
from ev_charging.review.http import ReviewRequestHandler, serve_review
from ev_charging.review.keypoint_annotations import LABELS, validate_annotations

ROOT = PROJECT_ROOT
TEMPLATE = Path(__file__).with_name("keypoint_review.html")
EDITABLE = {"points", "review_status", "reviewer", "reviewed_at", "notes"}


def _version(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def build_annotations(library: str | Path, *, root: str | Path = ROOT) -> dict:
    library, root = Path(library).resolve(), Path(root).resolve()
    if not library.is_relative_to(root):
        raise ValueError("孔位图库必须位于项目目录内")
    entries = read_entries(library)
    result = {
        "schema_version": 3,
        "purpose": "development_diagnostic",
        "independent_test_set": False,
        "labels": list(LABELS),
        "feature_definition": "visible inner aperture center; physical reference unverified",
        "provenance": {"kind": "training_library", "library_path": library.relative_to(root).as_posix()},
        "images": [],
    }
    names = {"ready": "已确认", "pending": "待复核", "conflict": "有冲突", "excluded": "已排除"}
    for entry in entries:
        path = _inside(library, entry["image"])
        splits = {source.get("original_split") for source in entry["sources"]} & {"train", "val", "test"}
        result["images"].append(
            {
                "image": path.relative_to(root).as_posix(),
                "image_sha256": entry["image_sha256"],
                "width": entry["width"],
                "height": entry["height"],
                "split": next(iter(splits)) if len(splits) == 1 else "unknown",
                "selection_reasons": ["检测框状态：" + names.get(entry["status"], entry["status"])],
                "review_status": "draft",
                "reviewer": "",
                "reviewed_at": None,
                "notes": "",
                "roi_bbox_xyxy": [],
                "auto_points": {},
                "points": {
                    label: {"xy": None, "visibility": "unlabeled", "source": "unlabeled", "note": ""}
                    for label in LABELS
                },
            }
        )
    if not result["images"]:
        raise ValueError("图库为空，请先采集或导入图片")
    validate_annotations(result)
    return result


class KeypointStore:
    def __init__(self, library=DEFAULT_LIBRARY, *, root=ROOT):
        self.root = Path(root).resolve()
        self.library = Path(library).resolve()
        self.path = self.library / "keypoint_annotations.json"
        self.seed = build_annotations(self.library, root=self.root)
        self.by_hash = {row["image_sha256"]: row for row in self.seed["images"]}
        from ev_charging.review.keypoint_suggestions import SuggestionEngine

        self.suggestion_engine = SuggestionEngine(self.root, self.library)
        self.suggestion_cache = {}

    def suggest(self, digest):
        model = self.suggestion_engine.current_model()
        key = (model["weights"], model["weights_sha256"])
        cached = self.suggestion_cache.get(digest)
        if cached is None or cached[0] != key:
            entry = json.loads((self.library / "records" / f"{digest}.json").read_text(encoding="utf-8"))
            result = self.suggestion_engine.suggest(entry, self.image(digest), model=model)
            self.suggestion_cache[digest] = (key, result)
        return self.suggestion_cache[digest][1]

    def _saved(self):
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        validate_annotations(data)
        if data["provenance"] != self.seed["provenance"]:
            raise ValueError("孔位标注所属图库不一致")
        return {row["image_sha256"]: row for row in data["images"]}

    def _merge(self, original, saved):
        row = copy.deepcopy(original)
        if saved is not None:
            if (saved["width"], saved["height"]) != (row["width"], row["height"]):
                raise ValueError("已保存孔位的图片尺寸与图库不一致")
            row.update({key: copy.deepcopy(saved[key]) for key in EDITABLE})
        return row

    def snapshot(self) -> dict:
        with _lock(self.library):
            saved = self._saved()
            data = copy.deepcopy(self.seed)
            data["images"] = [self._merge(row, saved.get(row["image_sha256"])) for row in data["images"]]
        validate_annotations(data)
        return data

    def image(self, digest: str) -> np.ndarray:
        row = self.by_hash[digest]
        content = _inside(self.root, row["image"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("原图内容已改变，请恢复图库原图")
        frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[1::-1] != (row["width"], row["height"]):
            raise ValueError("原图无法读取或尺寸不符")
        return frame

    def save(self, row: dict, version: str) -> str:
        if not isinstance(row, dict) or not isinstance(version, str):
            raise ValueError("孔位保存请求格式错误")
        digest = row.get("image_sha256")
        if digest not in self.by_hash:
            raise ValueError("图片不属于本次图库")
        original = self.by_hash[digest]

        def bound(item: dict) -> dict:
            return {key: value for key, value in item.items() if key not in EDITABLE}

        if bound(row) != bound(original):
            raise ValueError("不能修改图片身份、尺寸或来源")
        candidate = {**self.seed, "images": [row]}
        validate_annotations(candidate)
        self.image(digest)
        with _lock(self.library):
            saved = self._saved()
            current = self._merge(original, saved.get(digest))
            if _version(current) != version:
                raise ValueError("该图已在其他页面更新，请先导出当前修改，再刷新页面")
            saved[digest] = copy.deepcopy(row)
            document = {**self.seed, "images": list(saved.values())}
            validate_annotations(document)
            atomic_capture_json(self.path, document)
        return _version(row)


def create_server(library: str | Path = DEFAULT_LIBRARY, *, root: str | Path = ROOT) -> HTTPServer:
    store = KeypointStore(library, root=root)
    token = secrets.token_urlsafe(32)

    class Handler(ReviewRequestHandler):
        def do_GET(self):
            if not self.valid_host():
                self.reply(403, {"error": "请使用启动时显示的本机地址"})
                return
            route = urlsplit(self.path).path
            try:
                if route == "/":
                    annotations = store.snapshot()
                    payload = {
                        "annotations": annotations,
                        "api": True,
                        "token": token,
                        "seed_sha256": _version(annotations),
                        "record_versions": {
                            row["image_sha256"]: _version(row) for row in annotations["images"]
                        },
                        "images": [
                            {"image": row["image"], "src": "/image/" + row["image_sha256"]}
                            for row in annotations["images"]
                        ],
                    }
                    page = TEMPLATE.read_text(encoding="utf-8").replace(
                        "__KEYPOINT_PAYLOAD__",
                        json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"),
                    )
                    self.reply(200, page.encode("utf-8"), "text/html; charset=utf-8")
                elif route.startswith("/api/suggestions/") and route[17:] in store.by_hash:
                    self.reply(200, store.suggest(route[17:]))
                elif route.startswith("/image/") and route[7:] in store.by_hash:
                    ok, encoded = cv2.imencode(".png", store.image(route[7:]))
                    if not ok:
                        raise ValueError("原图转换失败")
                    self.reply(200, encoded.tobytes(), "image/png")
                else:
                    self.reply(404, {"error": "未找到复核内容"})
            except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError, cv2.error) as exc:
                self.reply(409, {"error": str(exc)})

        def do_POST(self):
            if not self.valid_write_token(token):
                self.reply(403, {"error": "页面已失效，请重新打开本机复核地址"})
                return
            if urlsplit(self.path).path != "/api/annotation":
                self.reply(404, {"error": "未知操作"})
                return
            try:
                request = self.read_json()
                version = store.save(request.get("row"), request.get("record_version"))
                self.reply(200, {"saved": True, "record_version": version})
            except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
                self.reply(409, {"error": str(exc)})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.review_url = f"http://127.0.0.1:{server.server_port}/"
    server.review_count = len(store.by_hash)
    server.annotation_path = store.path
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="统一图库九孔人工标注，草稿和确认结果直接保存到本机")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    server = create_server(args.library)
    print(
        f"孔位复核 {server.review_count} 张：{server.review_url}\n保存位置：{server.annotation_path}\n"
        "关闭服务请按 Ctrl+C。",
        flush=True,
    )
    serve_review(server, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
