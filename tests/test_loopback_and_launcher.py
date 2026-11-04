"""Blocco dell'avvio del server nativo (03-04/11): porta sorgente scartata dal filtro di rete locale (core/loopback.py) e rilancio
automatico (core/native_launcher.py). Nessuna GPU, nessun modello: solo socket locali e processi `python -c sleep`."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import loopback, native_launcher  # noqa: E402

REAL_IS_STARTUP_STUCK = native_launcher.is_startup_stuck      # i test lo sostituiscono con finte: la funzione vera serve a uno solo


@pytest.fixture()
def installed():
    """Rimedi attivi solo dentro il test (ripristinati dopo)."""
    loopback.uninstall()
    loopback.install()
    yield
    loopback.uninstall()


def test_regola_della_porta_cattiva():
    assert loopback.is_bad_source_port(54600) and loopback.is_bad_source_port(0xC530)      # (porta & 0x0F00) == 0x0500
    assert not loopback.is_bad_source_port(54000) and not loopback.is_bad_source_port(50000)


def test_pick_source_port_mai_cattiva_e_nel_range():
    for _ in range(5000):
        port = loopback.pick_source_port()
        assert 49152 <= port <= 65535 and not loopback.is_bad_source_port(port)


def test_socketpair_funziona_e_scambia_dati(installed):
    server, client = socket.socketpair()                     # e la versione con porta buona e timeout
    try:
        assert not loopback.is_bad_source_port(client.getsockname()[1])
        client.sendall(b"ping")
        assert server.recv(4) == b"ping"
    finally:
        server.close()
        client.close()


def test_socketpair_riprova_se_la_prima_porta_si_perde(monkeypatch):
    """Prima porta cattiva (su questo PC il SYN sparisce, altrove passa comunque), poi una buona: la coppia arriva sempre."""
    real_pick = loopback.pick_source_port
    ports = iter([54600])                                     # poi porte buone vere (alcune sono riservate dal sistema: il tentativo dopo ci pensa)
    monkeypatch.setattr(loopback, "pick_source_port", lambda rng=None: next(ports, None) or real_pick())
    server, client = loopback.socketpair_with_retry(timeout=0.5)
    server.close()
    client.close()


def test_connessione_loopback_parte_da_porta_buona(installed):
    listener = socket.socket()
    listener.bind((loopback.LOOPBACK, 0))
    listener.listen(5)
    seen: list[int] = []
    threading.Thread(target=lambda: seen.append(listener.accept()[1][1]), daemon=True).start()
    conn = socket.create_connection(listener.getsockname(), timeout=3)
    time.sleep(0.2)
    conn.close()
    listener.close()
    assert seen and not loopback.is_bad_source_port(seen[0])


def test_urllib_verso_loopback_usa_la_porta_buona(installed):
    """http.client passa source_address=None in modo posizionale: il rimedio deve scattare anche per urllib (app -> :8189)."""
    import http.server
    import urllib.request
    seen: list[int] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.client_address[1])
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_args):
            pass
    server = http.server.HTTPServer((loopback.LOOPBACK, 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for _ in range(40):
            assert urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=3).read() == b"ok"
    finally:
        server.shutdown()
    assert len(seen) == 40 and not any(loopback.is_bad_source_port(port) for port in seen)


def test_install_idempotente_e_ripristinabile():
    loopback.uninstall()
    original = socket.create_connection
    assert loopback.install() is (sys.platform == "win32")
    assert loopback.install() is False
    loopback.uninstall()
    assert socket.create_connection is original


# ── rilancio automatico ─────────────────────────────────────────────────────────────────────────────────────────────────
class _FakeProcess:
    pid = 0

    def poll(self):
        return None


def _supervision_setup(monkeypatch, up_after_launches: int):
    """Il processo e sempre «bloccato»; is_up diventa vero dopo `up_after_launches` lanci. Ritorna i contatori."""
    state = {"launch": 0, "kill": 0}

    def fake_launch():
        state["launch"] += 1
        native_launcher._server_process = _FakeProcess()
        return True, "avviato"

    def fake_kill(log=print):
        state["kill"] += 1
        native_launcher._server_process = None
        return 2

    monkeypatch.setattr(native_launcher, "launch", fake_launch)
    monkeypatch.setattr(native_launcher, "kill_server_tree", fake_kill)
    monkeypatch.setattr(native_launcher, "is_startup_stuck", lambda root, samples, now: True)
    monkeypatch.setattr(native_launcher, "_tree_cpu_seconds", lambda root: 0.0)
    native_launcher._server_process = _FakeProcess()
    state["launch"] = 1                                        # il primo lancio lo ha fatto il chiamante
    return state, (lambda: state["launch"] > up_after_launches)


def test_rilancia_se_bloccato_e_poi_parte(monkeypatch):
    state, is_up = _supervision_setup(monkeypatch, up_after_launches=1)
    messages: list[str] = []
    assert native_launcher.supervise_startup(is_up, log=messages.append, probe_seconds=0, poll_seconds=0.01, total_timeout=5)
    assert state == {"launch": 2, "kill": 1}
    assert any("RILANCIO 1/2" in m for m in messages)


def test_si_ferma_dopo_due_rilanci(monkeypatch):
    state, is_up = _supervision_setup(monkeypatch, up_after_launches=99)
    messages: list[str] = []
    assert not native_launcher.supervise_startup(is_up, log=messages.append, probe_seconds=0, poll_seconds=0.01, total_timeout=5)
    assert state == {"launch": 3, "kill": 2} and any("non parte dopo 2 riavvii" in m for m in messages)


def test_server_gia_pronto_non_si_tocca(monkeypatch):
    state, _ = _supervision_setup(monkeypatch, up_after_launches=0)
    assert native_launcher.supervise_startup(lambda: True, probe_seconds=0, poll_seconds=0.01, total_timeout=5)
    assert state == {"launch": 1, "kill": 0}


def test_avvio_lento_ma_che_lavora_non_si_chiude(monkeypatch):
    """Passato il tempo di sonda, un processo che consuma CPU (import pesanti) non e «bloccato»."""
    state, _ = _supervision_setup(monkeypatch, up_after_launches=0)
    monkeypatch.setattr(native_launcher, "is_startup_stuck", REAL_IS_STARTUP_STUCK)
    monkeypatch.setattr(native_launcher, "_is_listening", lambda root, port=8189: False)
    cpu = iter(range(0, 10_000, 5))                           # +5 s di CPU a ogni lettura
    monkeypatch.setattr(native_launcher, "_tree_cpu_seconds", lambda root: float(next(cpu)))
    monkeypatch.setattr(native_launcher, "IDLE_WINDOW_S", 0.05)
    assert not native_launcher.supervise_startup(lambda: False, probe_seconds=0, poll_seconds=0.02, total_timeout=0.5)
    assert state["kill"] == 0


def test_kill_server_tree_chiude_solo_il_nostro_albero():
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    ours = subprocess.Popen([sys.executable, "-c", "import subprocess,sys,time; subprocess.Popen(%r); time.sleep(60)" % sleeper])
    other = subprocess.Popen(sleeper)
    try:
        time.sleep(1.0)                                        # il figlio deve esistere
        native_launcher._server_process = ours
        tree_size = len(native_launcher._process_tree(ours))
        assert tree_size >= 2
        assert native_launcher.kill_server_tree() == tree_size
        assert ours.wait(timeout=10) is not None
        assert other.poll() is None                            # un processo estraneo resta vivo
        assert native_launcher._server_process is None
    finally:
        other.kill()
        ours.kill()
