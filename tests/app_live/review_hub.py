"""Server unico delle pagine di revisione, raggiungibile dagli altri dispositivi della rete Tailscale.

    python tests/app_live/review_hub.py [cartella_revisioni] [porta]

Ascolta solo sull'indirizzo Tailscale di questo PC (mai su tutte le interfacce): la pagina iniziale elenca le cartelle
review_* dalla più recente; ogni revisione si apre in /r/<cartella>/. I video si servono a intervalli (Range), che
Safari su iPhone richiede per riprodurli; «Pubblica» scrive il .md nella cartella della revisione da cui arriva.
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

DEFAULT_REVIEWS_DIR = r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live"
DEFAULT_PORT = 8790
ALLOWED_MEDIA_ROOTS = [r"E:\ai\ltxstudio", r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch"]
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".json": "application/json", ".jpg": "image/jpeg",
                 ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".mp4": "video/mp4",
                 ".webm": "video/webm", ".md": "text/markdown; charset=utf-8"}
REVIEW_PATH = re.compile(r"^/r/([^/]+)/(.*)$")
READ_BLOCK = 1024 * 1024


def tailscale_ipv4() -> str:
    result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, check=True)
    address = result.stdout.split()[0].strip()
    if not address.startswith("100."):
        raise RuntimeError(f"indirizzo Tailscale inatteso: {address!r}")
    return address


def list_reviews(reviews_dir: str) -> list[tuple[str, float]]:
    entries = []
    for name in os.listdir(reviews_dir):
        index = os.path.join(reviews_dir, name, "index.html")
        if name.startswith("review_") and os.path.isfile(index):
            entries.append((name, os.path.getmtime(index)))
    return sorted(entries, key=lambda entry: entry[1], reverse=True)


def index_page(reviews_dir: str) -> bytes:
    rows = []
    for name, modified in list_reviews(reviews_dir):
        published = [f for f in os.listdir(os.path.join(reviews_dir, name)) if f.startswith("pubblicato_")]
        mark = " · pubblicata" if published else ""
        rows.append(f'<li><a href="/r/{quote(name)}/">{html.escape(name)}</a>'
                    f'<span>{time.strftime("%d/%m %H:%M", time.localtime(modified))}{mark}</span></li>')
    body = f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Revisioni LTX</title>
<style>:root{{--bg:#f6f7f9;--fg:#15171c;--mut:#5b6270;--card:#fff;--bd:#d9dde4;--ac:#2456d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#101216;--fg:#e8eaee;--mut:#9aa3b2;--card:#181b21;--bd:#2b303a;--ac:#8aa8ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;
padding:env(safe-area-inset-top) max(16px,env(safe-area-inset-right)) 32px max(16px,env(safe-area-inset-left))}}
h1{{font-size:1.3rem}}ul{{list-style:none;padding:0;margin:0}}
li{{background:var(--card);border:1px solid var(--bd);border-radius:10px;margin:8px 0;display:flex;flex-direction:column}}
a{{color:var(--ac);padding:12px 14px 2px;min-height:44px;overflow-wrap:anywhere}}a:focus-visible{{outline:3px solid var(--ac)}}
span{{color:var(--mut);font-size:.9rem;padding:0 14px 10px}}</style></head>
<body><h1>Revisioni LTX Studio</h1><ul>{''.join(rows) or '<li><span>nessuna revisione</span></li>'}</ul></body></html>"""
    return body.encode("utf-8")


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    match = re.match(r"bytes=(\d*)-(\d*)$", header.strip())
    if not match or (not match.group(1) and not match.group(2)):
        return None
    if not match.group(1):
        start = max(0, size - int(match.group(2)))
        return start, size - 1
    start = int(match.group(1))
    end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
    return (start, end) if start <= end < size else None


def make_handler(reviews_dir: str):
    reviews_root = os.path.abspath(reviews_dir)
    media_roots = [os.path.abspath(root).lower() for root in ALLOWED_MEDIA_ROOTS + [reviews_root]]

    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_bytes(self, code: int, body: bytes, content_type: str = "text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path: str):
            size = os.path.getsize(path)
            content_type = CONTENT_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
            requested = parse_range(self.headers["Range"], size) if self.headers.get("Range") else None
            start, end = requested if requested else (0, size - 1)
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = handle.read(min(READ_BLOCK, remaining))
                    if not block:
                        break
                    try:
                        self.wfile.write(block)
                    except ConnectionError:      # il browser chiude spesso a metà mentre scorre un video: non è un errore
                        return
                    remaining -= len(block)

        def review_dir_from_referer(self) -> str | None:
            match = REVIEW_PATH.match(urlparse(self.headers.get("Referer", "")).path)
            if not match:
                return None
            folder = os.path.abspath(os.path.join(reviews_root, unquote(match.group(1))))
            return folder if os.path.dirname(folder) == reviews_root and os.path.isdir(folder) else None

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                return self.send_bytes(200, index_page(reviews_root), CONTENT_TYPES[".html"])
            if url.path == "/file":
                path = os.path.abspath(unquote(parse_qs(url.query).get("p", [""])[0]))
                if not any(path.lower().startswith(root) for root in media_roots) or not os.path.isfile(path):
                    return self.send_bytes(404, b"file non disponibile")
                return self.send_file(path)
            match = REVIEW_PATH.match(url.path)
            if not match:
                if url.path.startswith("/r/"):
                    return self.redirect_with_slash(url.path)
                return self.send_bytes(404, b"pagina non trovata")
            folder = os.path.abspath(os.path.join(reviews_root, unquote(match.group(1))))
            path = os.path.abspath(os.path.join(folder, unquote(match.group(2)) or "index.html"))
            if os.path.dirname(folder) != reviews_root or not path.startswith(folder + os.sep) or not os.path.isfile(path):
                return self.send_bytes(404, b"file non trovato")
            self.send_file(path)

        def redirect_with_slash(self, path: str):
            self.send_response(301)
            self.send_header("Location", path + "/")
            self.end_headers()

        def do_POST(self):
            if urlparse(self.path).path != "/publish":
                return self.send_bytes(404, b"pagina non trovata")
            folder = self.review_dir_from_referer()
            if folder is None:
                return self.send_bytes(400, b"revisione di provenienza sconosciuta")
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            path = os.path.join(folder, f"pubblicato_{time.strftime('%Y%m%d_%H%M%S')}.md")
            with open(path, "wb") as handle:
                handle.write(body)
            print(f"pubblicato: {path}", flush=True)
            self.send_bytes(200, json.dumps({"ok": True, "path": path}).encode(), "application/json")

    return ReviewHandler


def main() -> int:
    reviews_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REVIEWS_DIR
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
    address = tailscale_ipv4()
    server = ThreadingHTTPServer((address, port), make_handler(reviews_dir))
    print(f"revisioni su http://{address}:{port}/ (solo rete Tailscale)", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
