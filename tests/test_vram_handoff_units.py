"""Unit test CPU-only del passaggio VRAM (branch feature/vram-handoff-fix).

Copre: esiti per-porta di mem_coordinator._free_ports (freed/busy/absent/timeout), free_for SENZA
il vecchio guard "stessa categoria → skip", clamp dinamico di gpu_mem.wait_free_vram_mb, lock GPU
inter-processo (base/rientranza/stale/abort). Nessuna GPU/torch richiesti.
"""
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import gpu_lock, gpu_mem, mem_coordinator  # noqa: E402


def _serve(handler_cls):
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


class _FreeOk(http.server.BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        type(self).calls.append(self.path)
        body = json.dumps({"busy": False, "freed_pipelines": 1}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                              # silenzia il log http.server
        pass


class _FreeBusy(_FreeOk):
    calls = []

    def do_POST(self):
        type(self).calls.append(self.path)
        body = json.dumps({"busy": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FreeSlow(_FreeOk):
    calls = []

    def do_POST(self):
        time.sleep(3)                                       # oltre il timeout del client
        super().do_POST()


def test_free_ports_outcomes():
    s1, p_ok = _serve(_FreeOk)
    s2, p_busy = _serve(_FreeBusy)
    s3, p_slow = _serve(_FreeSlow)
    with socket.socket() as s:                              # porta chiusa = server assente
        s.bind(("127.0.0.1", 0))
        p_absent = s.getsockname()[1]
    try:
        out = mem_coordinator._free_ports((p_ok, p_busy, p_absent, p_slow), timeout=1.0)
        assert out[p_ok] == "freed", out
        assert out[p_busy] == "busy", out
        # porta chiusa: su alcune macchine Windows (filtri di rete) il refused diventa un timeout —
        # per il prodotto conta solo che NON sia 'freed' (entrambi gli esiti sono non-liberato).
        assert out[p_absent] in ("absent", "timeout"), out
        assert out[p_slow] == "timeout", out
        assert mem_coordinator._freed(out) == [p_ok]
    finally:
        for s in (s1, s2, s3):
            s.shutdown()


def test_free_for_no_guard(monkeypatch):
    """Il vecchio guard "stessa categoria consecutiva → skip" NON deve esistere più: due free_for
    identici di fila devono ENTRAMBI mandare i /free (il guard desincronizzato lasciava FLUX in
    VRAM mentre caricava il 22B)."""
    calls = []
    monkeypatch.setattr(mem_coordinator, "_free_ports",
                        lambda ports, timeout, mode="demote": (calls.append((tuple(ports), mode)),
                                                               {p: "freed" for p in ports})[1])
    mem_coordinator.reset()
    a = mem_coordinator.free_for("video")
    b = mem_coordinator.free_for("video")                   # PRIMA: ritornava [] senza liberare
    assert len(calls) == 2, calls
    assert a == b and a == list(calls[0][0])
    assert calls[0][0] == tuple(mem_coordinator._IMAGE_PORTS)


def test_free_for_llm_demotes(monkeypatch):
    """Categoria 'llm' → demote di TUTTO (mai destroy: gli LLM transienti non devono distruggere
    la residenza del motore che genera subito dopo)."""
    calls = []
    monkeypatch.setattr(mem_coordinator, "_free_ports",
                        lambda ports, timeout, mode="demote": (calls.append((tuple(ports), mode)),
                                                               {p: "freed" for p in ports})[1])
    monkeypatch.setattr(mem_coordinator, "auto_mode", lambda: "demote")
    mem_coordinator.free_for("llm")
    assert calls[0][0] == tuple(mem_coordinator._IMAGE_PORTS + mem_coordinator._VIDEO_PORTS)
    assert calls[0][1] == "demote"
    calls.clear()
    mem_coordinator.free_for("music")                       # effimera → destroy come prima
    assert calls[0][1] == "destroy"


def test_wait_free_vram_clamp(monkeypatch):
    """La soglia è clampata alla VRAM fisica −2GB: su una GPU piccola una richiesta enorme non
    degenera in un'attesa piena (prima: soglia fissa 24GB = sleep 40s garantito su una 12GB)."""
    monkeypatch.setattr(gpu_mem, "total_vram_mb", lambda: 12288)
    monkeypatch.setattr(gpu_mem, "free_vram_mb", lambda: 11000)
    t0 = time.monotonic()
    free = gpu_mem.wait_free_vram_mb(24000, timeout_s=5, poll_s=0.1, tag="unit")
    assert free == 11000
    assert time.monotonic() - t0 < 2, "il clamp non ha evitato l'attesa piena"


def test_wait_free_vram_waits_until_released(monkeypatch):
    """Il poll deve continuare finché la VRAM non torna sopra soglia (rilascio asincrono simulato)."""
    vals = iter([1000, 1000, 30000])
    last = [1000]

    def _free():
        last[0] = next(vals, 30000)
        return last[0]

    monkeypatch.setattr(gpu_mem, "total_vram_mb", lambda: 32768)
    monkeypatch.setattr(gpu_mem, "free_vram_mb", _free)
    free = gpu_mem.wait_free_vram_mb(24000, timeout_s=5, poll_s=0.05, tag="unit")
    assert free == 30000


def test_gpu_lock_reentrant_and_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_lock, "_LOCK_PATH", str(tmp_path / "gpu.lock"))
    gpu_lock._STATE["count"] = 0
    assert gpu_lock.acquire("t:a", timeout_s=1)
    assert gpu_lock.acquire("t:b", timeout_s=0)             # rientrante
    gpu_lock.release()
    assert gpu_lock.holder() is not None                    # ancora tenuto (count 2→1)
    gpu_lock.release()
    assert gpu_lock.holder() is None
    # stale: lock di un pid inesistente → rimosso e riacquisito
    with open(gpu_lock._LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump({"pid": 999999999, "tag": "morto", "ts": 0}, f)
    assert gpu_lock.acquire("t:c", timeout_s=5)
    gpu_lock.release()


def test_gpu_lock_contended_abort(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_lock, "_LOCK_PATH", str(tmp_path / "gpu.lock"))
    gpu_lock._STATE["count"] = 0
    with open(gpu_lock._LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid() + 0, "tag": "occupante", "ts": time.time()}, f)
    # NB pid = NOSTRO pid (vivo) ma count=0 → contesa vera; abort deve uscire subito
    t0 = time.monotonic()
    assert not gpu_lock.acquire("t:x", poll_s=0.1, abort=lambda: True)
    assert time.monotonic() - t0 < 2
    os.remove(gpu_lock._LOCK_PATH)


def test_gpu_lock_cross_process(tmp_path, monkeypatch):
    lock_path = str(tmp_path / "gpu.lock")
    monkeypatch.setattr(gpu_lock, "_LOCK_PATH", lock_path)
    gpu_lock._STATE["count"] = 0
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child = subprocess.Popen([sys.executable, "-c",
                              "import sys,time; sys.path.insert(0, %r);"
                              "from core import gpu_lock; gpu_lock._LOCK_PATH=%r;"
                              "gpu_lock.acquire('child'); time.sleep(2.5)" % (repo, lock_path)])
    time.sleep(1.0)
    assert not gpu_lock.acquire("me", timeout_s=0.5, poll_s=0.1), "doveva essere del figlio"
    child.wait()
    assert gpu_lock.acquire("me", timeout_s=5, poll_s=0.1)  # figlio morto → stale → acquisito
    gpu_lock.release()
