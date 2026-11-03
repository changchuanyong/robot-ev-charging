"""Shared transport for the local box and keypoint review services."""

from __future__ import annotations

import json
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

MAX_BODY = 128 * 1024


class RequestBodyError(ValueError):
    """The declared body size or media type is not accepted."""


class ReviewRequestHandler(BaseHTTPRequestHandler):
    """Keep responses and write authorization identical across review pages."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    @property
    def origin(self) -> str:
        return f"http://{self.server.server_address[0]}:{self.server.server_port}"

    def reply(
        self, status: int, payload: object, content_type: str = "application/json; charset=utf-8"
    ) -> None:
        data = (
            payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self) -> bool:
        return self.headers.get("Host") == self.origin.removeprefix("http://")

    def valid_write_token(self, token: str) -> bool:
        return (
            self.valid_host()
            and self.headers.get("Origin") == self.origin
            and secrets.compare_digest(self.headers.get("X-Review-Token", "").encode(), token.encode())
        )

    def read_json(
        self, *, max_body: int = MAX_BODY, object_error: str = "保存请求必须为对象"
    ) -> dict[str, Any]:
        """Read a bounded JSON object; each service retains its own error response."""
        self.connection.settimeout(10)
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= max_body or self.headers.get_content_type() != "application/json":
            raise RequestBodyError("保存内容过大或格式错误")
        payload = json.loads(self.rfile.read(size).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(object_error)
        return payload


def serve_review(server: HTTPServer, *, open_browser: bool) -> None:
    """Serve until interrupted, then release the listening socket."""
    if open_browser:
        webbrowser.open(f"http://{server.server_address[0]}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
