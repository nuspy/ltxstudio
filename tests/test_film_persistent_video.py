"""Backend test: routing al worker VIDEO persistente (core/film_studio/gen_service).
Usa un server-stub che imita /system_stats, /submit, /history del ltx2_server — nessuna GPU/modello."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Stub(BaseHTTPRequestHandler):
    OUT = ""                                                 # file che il "server" dichiara come output

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/system_stats":
            self._send({"up": True, "cached_pipelines": 1})
        elif self.path.startswith("/history/"):
            Path(_Stub.OUT).write_bytes(b"\x00\x00")         # simula il video prodotto
            self._send({"status": "success", "output": [_Stub.OUT]})
        else:
            self._send({"status": "unknown"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        self._send({"prompt_id": "abc123"})


def test_routing_al_server_persistente(tmp_path):
    out = str(tmp_path / "shot.mp4")
    _Stub.OUT = str(tmp_path / "server_out.mp4")
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["LTX2_SERVER"] = f"http://127.0.0.1:{port}"

    import importlib
    from core.film_studio import gen_service
    importlib.reload(gen_service)                            # rilegge LTX2_SERVER dall'env

    assert gen_service._ltx2_server_up() is True             # rileva il server
    res = gen_service._submit_persistent({"prompt": "x", "output": out}, out)
    assert res == out and os.path.exists(out)                # output copiato sul path canonico
    srv.shutdown()


def test_server_down_ritorna_false():
    os.environ["LTX2_SERVER"] = "http://127.0.0.1:1"         # porta morta
    import importlib
    from core.film_studio import gen_service
    importlib.reload(gen_service)
    assert gen_service._ltx2_server_up() is False            # → fallback al subprocess in gen_shot_video
