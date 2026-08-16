"""Unit test del downloader unico (core/downloader) — CPU-only, server HTTP locale, zero rete.

Copre: verifica d'integrità (safetensors/gguf/sha), rifiuto no_remote, rename remoto→locale,
espansione deps (deps PRIMA), preflight gated (401 → GatedRepoError con url licenza),
DownloadQueue (filtra presenti/no_remote, scarica dal server finto, callbacks)."""
import http.server
import json
import os
import struct
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import downloader as dl  # noqa: E402
from core import weights_catalog as wc  # noqa: E402


def _mini_safetensors() -> bytes:
    hdr = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    return struct.pack("<Q", len(hdr)) + hdr + b"\x00" * 8


def _serve(payload: bytes):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/file.safetensors"


# ── verify_file ──────────────────────────────────────────────────────────────
def test_verify_safetensors_ok(tmp_path):
    p = tmp_path / "ok.safetensors"
    p.write_bytes(_mini_safetensors())
    dl.verify_file(p)                                    # non solleva


def test_verify_safetensors_troncato(tmp_path):
    p = tmp_path / "ko.safetensors"
    p.write_bytes(_mini_safetensors()[:6])               # header troncato
    with pytest.raises(dl.IntegrityError):
        dl.verify_file(p)


def test_verify_gguf(tmp_path):
    ok = tmp_path / "ok.gguf"; ok.write_bytes(b"GGUF" + b"\x00" * 32)
    dl.verify_file(ok)
    ko = tmp_path / "ko.gguf"; ko.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(dl.IntegrityError):
        dl.verify_file(ko)


def test_verify_sha(tmp_path):
    import hashlib
    p = tmp_path / "f.bin"; p.write_bytes(b"ciao")
    dl.verify_file(p, sha256=hashlib.sha256(b"ciao").hexdigest())
    with pytest.raises(dl.IntegrityError):
        dl.verify_file(p, sha256="00" * 32)


# ── download_entry ───────────────────────────────────────────────────────────
def test_no_remote_rifiutato():
    e = {"id": "x", "no_remote": True, "kind": "file", "filename": "f.safetensors"}
    with pytest.raises(dl.DownloadUnavailable):
        dl.download_entry(e)


def test_download_url_con_rename(tmp_path):
    """kind=file da url diretto: scarica, rinomina al filename LOCALE, verifica struttura."""
    srv, url = _serve(_mini_safetensors())
    try:
        e = {"id": "t", "kind": "file", "filename": "nome_locale.safetensors",
             "url": url, "type": "Upscale", "family": ""}
        out = dl.download_entry(e, base_dir=str(tmp_path))
        assert out.name == "nome_locale.safetensors"
        assert out.parent == tmp_path / "Upscale"
        assert out.stat().st_size == len(_mini_safetensors())
    finally:
        srv.shutdown()


def test_download_corrotto_fallisce(tmp_path):
    srv, url = _serve(b"NON-UN-SAFETENSORS")
    try:
        e = {"id": "t2", "kind": "file", "filename": "x.safetensors",
             "url": url, "type": "Upscale", "family": ""}
        with pytest.raises(dl.IntegrityError):
            dl.download_entry(e, base_dir=str(tmp_path))
    finally:
        srv.shutdown()


# ── deps / gated ─────────────────────────────────────────────────────────────
def test_plan_ids_deps_prima():
    plan = [e["id"] for e in dl.plan_ids(["krea_rt"])]
    assert "wan21_14b_diffusers" in plan and "krea_rt" in plan
    assert plan.index("wan21_14b_diffusers") < plan.index("krea_rt"), "la dep va scaricata PRIMA"


def test_check_gated_401(monkeypatch):
    import requests

    class R:
        status_code = 401

    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    with pytest.raises(dl.GatedRepoError) as ei:
        dl.check_gated("Lightricks/LTX-2.5", None)
    assert "huggingface.co/Lightricks/LTX-2.5" in ei.value.license_url


def test_check_gated_rete_giu_non_blocca(monkeypatch):
    import requests

    def boom(*a, **k):
        raise OSError("rete giù")

    monkeypatch.setattr(requests, "get", boom)
    dl.check_gated("qualsiasi/repo", None)               # best-effort: non solleva


# ── DownloadQueue ────────────────────────────────────────────────────────────
def test_queue_scarica_e_filtra(tmp_path, monkeypatch):
    payload = _mini_safetensors()
    srv, url = _serve(payload)
    fake_catalog = [
        {"id": "a", "kind": "file", "filename": "a.safetensors", "url": url,
         "type": "Upscale", "family": "", "category": "aux", "arch": "t"},
        {"id": "b", "no_remote": True, "kind": "file", "filename": "b.safetensors",
         "type": "Upscale", "family": "", "category": "aux", "arch": "t"},
    ]
    monkeypatch.setattr(wc, "load", lambda: fake_catalog)
    done, errs = [], []
    q = dl.DownloadQueue(base_dir=str(tmp_path),
                         on_done=lambda e, p: done.append((e["id"], p)),
                         on_error=lambda e, ex: errs.append((e["id"], ex)))
    try:
        fresh = q.enqueue(["a", "b"])
        assert fresh == ["a"], f"la no_remote non va accodata: {fresh}"
        t0 = time.monotonic()
        while not done and time.monotonic() - t0 < 15:
            time.sleep(0.1)
        assert done and done[0][0] == "a" and done[0][1].exists()
        assert not errs
        assert q.enqueue(["a"]) == [], "già presente → non riaccodare"
    finally:
        q.stop()
        srv.shutdown()
