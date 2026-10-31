"""Local box review with explicit confirmation, persisted drafts and safe writeback."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit
import webbrowser

from .contracts import atomic_capture_json
from .training_library import DEFAULT_LIBRARY, _inside, _lock, read_entries, set_annotation


TEMPLATE = Path(__file__).resolve().parents[1] / "train/python/port_box_review.html"
MAX_BODY = 128 * 1024


def suggested_label(entry):
    suggestions = entry.get("box_suggestions", [])
    return suggestions[-1].get("label_text", "") if suggestions else ""


def session_entries(entries, session_id):
    return [entry for entry in entries if any(source.get("session_id") == session_id
                                             for source in entry["sources"])]


def _drafts(root):
    path = root / "review_drafts.json"
    if not path.exists():
        return {"schema_version": 1, "drafts": {}}
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(result, dict) or result.get("schema_version") != 1
            or not isinstance(result.get("drafts"), dict)):
        raise ValueError("invalid_review_drafts")
    return result


def _record(root, digest):
    data = (root / "records" / f"{digest}.json").read_bytes()
    entry = json.loads(data.decode("utf-8"))
    if entry.get("image_sha256") != digest:
        raise ValueError("library_record_identity_mismatch")
    return entry, hashlib.sha256(data).hexdigest()


def _item(root, digest, drafts):
    entry, version = _record(root, digest)
    image = _inside(root, entry["image"])
    label = _inside(root, entry["label"]).read_text(encoding="utf-8") if entry.get("label") else ""
    draft = drafts.get(digest)
    if draft is not None and draft.get("record_version") != version:
        draft = None
    return {"image_sha256": digest, "record_version": version,
            "image_url": f"/image/{digest}", "width": entry["width"], "height": entry["height"],
            "status": entry["status"], "human_reviewed": entry.get("human_reviewed", False),
            "label_text": label or suggested_label(entry), "draft": draft,
            "sources": entry["sources"], "label_candidates": entry.get("label_candidates", []),
            "box_suggestions": entry.get("box_suggestions", []), "image_available": image.is_file()}


def create_server(library=DEFAULT_LIBRARY, session=None):
    """Construct a loopback-only server without starting a thread or browser."""
    root = Path(library).resolve()
    entries = read_entries(root)
    session_id = None
    if session is not None:
        manifest = json.loads((Path(session).resolve() / "manifest.json").read_text(encoding="utf-8-sig"))
        session_id = manifest["session_id"]
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("capture_session_id_required")
        entries = session_entries(entries, session_id)
    else:
        entries = [entry for entry in entries if entry["status"] in {"pending", "conflict"}]
    allowed = tuple(entry["image_sha256"] for entry in entries if _inside(root, entry["image"]).is_file())
    token = secrets.token_urlsafe(32)
    scope = hashlib.sha256((str(root) + ":" + str(session_id)).encode()).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, payload, content_type="application/json; charset=utf-8"):
            data = (json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    if not isinstance(payload, bytes) else payload)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                             "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                             "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get("Host") == self.server.server_address[0] + ":" + str(self.server.server_port)

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, {"error": "请使用启动时显示的本机复核地址。"})
                return
            route = urlsplit(self.path)
            try:
                if route.path == "/":
                    drafts = _drafts(root)["drafts"]
                    items = [_item(root, digest, drafts) for digest in allowed]
                    config = {"api": True, "token": token, "scope_key": scope}
                    page = TEMPLATE.read_text(encoding="utf-8")
                    for marker, value in (("/*__REVIEW_DATA__*/[]", items),
                                          ("/*__REVIEW_CONFIG__*/{}", config)):
                        page = page.replace(marker, json.dumps(value, ensure_ascii=False).replace("<", "\\u003c"))
                    self.reply(200, page.encode("utf-8"), "text/html; charset=utf-8")
                elif route.path.startswith("/image/") and route.path[7:] in allowed:
                    digest = route.path[7:]
                    entry, _ = _record(root, digest)
                    path = _inside(root, entry["image"])
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError("image_hash_mismatch")
                    self.reply(200, data, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                else:
                    self.reply(404, {"error": "未找到本次复核内容。"})
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.reply(409, {"error": f"复核数据已变更或缺失，请检查图库：{exc}"})

        def do_POST(self):
            origin = f"http://{self.server.server_address[0]}:{self.server.server_port}"
            if (not self.valid_host() or self.headers.get("Origin") != origin
                    or not secrets.compare_digest(self.headers.get("X-Review-Token", "").encode(), token.encode())):
                self.reply(403, {"error": "复核页面已失效，请重新打开本机复核地址。"})
                return
            try:
                self.connection.settimeout(10)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY or self.headers.get_content_type() != "application/json":
                    self.reply(413, {"error": "提交内容过大或格式不正确。"})
                    return
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request_object_required")
                route = urlsplit(self.path).path
                if route == "/api/finish":
                    self.reply(200, {"saved": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                digest = payload.get("image_sha256")
                if digest not in allowed:
                    self.reply(404, {"error": "该图片不属于本次复核。"})
                    return
                version, label = payload.get("record_version"), payload.get("label_text")
                if not isinstance(version, str) or not isinstance(label, str) or len(label) > 65536:
                    raise ValueError("invalid_review_payload")
                if route == "/api/draft":
                    with _lock(root):
                        _, actual = _record(root, digest)
                        if actual != version:
                            raise ValueError("annotation_record_changed_reload_required")
                        data = _drafts(root)
                        data["drafts"][digest] = {"record_version": version, "label_text": label}
                        atomic_capture_json(root / "review_drafts.json", data)
                    self.reply(200, {"saved": True})
                elif route == "/api/annotation":
                    excluded = payload.get("excluded", False)
                    if type(excluded) is not bool or (not excluded and not label.strip()):
                        raise ValueError("nonempty_boxes_or_explicit_exclusion_required")
                    set_annotation(root, digest, label, annotation_source="human_local_box_review",
                                   human_reviewed=True, excluded=excluded,
                                   reason="user_review_no_charging_connector" if excluded else None,
                                   expected_record_sha256=version)
                    with _lock(root):
                        data = _drafts(root)
                        if data["drafts"].get(digest, {}).get("record_version") == version:
                            data["drafts"].pop(digest)
                            atomic_capture_json(root / "review_drafts.json", data)
                    self.reply(200, {"saved": True, "record": _item(root, digest, {})})
                else:
                    self.reply(404, {"error": "未知复核操作。"})
            except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
                message = ("该图已被其他操作更新，请刷新页面后重新复核。"
                           if "annotation_record_changed" in str(exc)
                           else f"保存失败，请检查标签后重试：{exc}")
                self.reply(409, {"error": message})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.review_url = f"http://127.0.0.1:{server.server_port}/"
    server.review_count = len(allowed)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="本机复核检测框，确认后直接写回图库；不启动相机或训练。")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--session", type=Path, help="仅复核这次采集会话；默认显示全库待标注图片")
    parser.add_argument("--no-browser", action="store_true", help="只打印地址，不自动打开浏览器")
    args = parser.parse_args(argv)
    server = create_server(args.library, args.session)
    print(f"本次复核 {server.review_count} 张：{server.review_url}\n确认后直接入库；关闭服务请按 Ctrl+C。", flush=True)
    if not args.no_browser:
        webbrowser.open(server.review_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
