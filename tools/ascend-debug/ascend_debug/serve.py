from __future__ import annotations

import argparse
import functools
import http.server
import pathlib
import sys
import urllib.parse
import webbrowser

from ascend_debug import open_view
from ascend_debug.runner import CommandError


def _request_needs_refresh(path: str) -> bool:
    parsed = urllib.parse.urlparse(path)
    request_path = parsed.path or "/"
    return request_path == "/" or request_path.endswith(".html")


def _make_handler(run_dir: pathlib.Path) -> type[http.server.SimpleHTTPRequestHandler]:
    class RefreshingHandler(http.server.SimpleHTTPRequestHandler):
        def _refresh_views(self) -> bool:
            if not _request_needs_refresh(self.path):
                return True
            try:
                manifest = open_view.load_manifest(run_dir)
                open_view.render_index(run_dir, manifest)
            except CommandError as error:
                self.send_error(500, f"ascend-debug refresh failed: {error}")
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
            if self._refresh_views():
                super().do_GET()

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API.
            if self._refresh_views():
                super().do_HEAD()

    return functools.partial(RefreshingHandler, directory=str(run_dir))  # type: ignore[return-value]


def serve_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = open_view.load_manifest(run_dir)
    index_path = open_view.render_index(run_dir, manifest)
    handler = _make_handler(run_dir)
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    host, port = server.server_address[:2]
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{display_host}:{port}/"
    print(f"ascend-debug.serve.index={index_path}", flush=True)
    print(f"ascend-debug.serve.url={url}", flush=True)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception as error:  # pragma: no cover - depends on host browser setup.
            print(f"ascend-debug.serve.browser=unavailable: {error}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("ascend-debug.serve.stopped=keyboard-interrupt", flush=True)
    finally:
        server.server_close()
    return 0
