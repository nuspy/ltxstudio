"""Ambiente di test APP-COMPLETA — boota la MainWindow VERA con i meccanismi di memoria ATTIVI e la pilota
come la GUI, verificando lo stato memoria REALE.

Perché: i probe isolati (girare i worker a mano) o la harness offscreen che patchava `_maybe_autostart_native`
NON avevano `mem_coordinator` + `inference_server/component_store.py` attivi → risultati falsati e
potenzialmente distruttivi (VRAM afferrata senza coordinamento). Qui NIENTE patch sulla memoria: parte
l'InferenceServer :8189 (ComponentStore vivo), il mem_coordinator coordina i /free, e i server persistenti
si avviano quando la feature li richiede. Si patcha SOLO i QMessageBox modali (altrimenti offscreen si pianta).

Uso tipico in un caso di test:
    from tests.app_env import harness as H
    w = H.boot_full_app()                      # MainWindow vera + :8189 su
    before = H.snapshot()                      # VRAM/servers/processi PRIMA
    ... pilota i widget e clicca il bottone ...
    after = H.snapshot()                       # confronto stato memoria reale
    H.teardown(w)                              # /free + kill albero worker + verifica VRAM rilasciata
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# porte dei componenti memoria/servizi da osservare (le stesse di mem_coordinator)
PORTS = (8189, 8190, 8191, 8200)

_APP = None


def app():
    from PySide6.QtWidgets import QApplication
    global _APP
    _APP = QApplication.instance() or QApplication(sys.argv[:1])
    return _APP


def _patch_blocking_dialogs() -> None:
    """SOLO i dialog modali bloccanti → non-bloccanti (offscreen altrimenti si pianta). NIENTE patch su
    memoria/autostart/native: quelli DEVONO restare reali. QDialog.exec NON è patchato globalmente — i casi
    che devono pilotare un dialog lo istanziano e lo guidano (come fa il bottone)."""
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.exec = lambda self, *a, **k: QMessageBox.Ok
    QMessageBox.exec_ = lambda self, *a, **k: QMessageBox.Ok


def boot_full_app(*, wait_native: bool = True, native_timeout: float = 240.0, log=print):
    """Costruisce la MainWindow VERA (il suo __init__ avvia l'InferenceServer :8189 via
    `_maybe_autostart_native`) e attende che risponda. NON disattiva ComponentStore/mem_coordinator/autostart.
    Ritorna la finestra. DEV: feature_gate aperto (come l'app dell'owner)."""
    app()
    _patch_blocking_dialogs()
    from ui.main_window import MainWindow
    if hasattr(MainWindow, "_feature_gate"):
        MainWindow._feature_gate = lambda self, *a, **k: True
    log("[app_env] costruisco MainWindow (autostart :8189 REALE)…")
    w = MainWindow()
    w.show()
    if wait_native:
        from core import native_driver
        log("[app_env] attendo InferenceServer :8189…")
        ok = native_driver.ensure_up(timeout=float(native_timeout), log=log)
        log(f"[app_env] InferenceServer :8189 {'UP' if ok else 'NON pronto'}")
    return w


# ─────────────── osservazione stato memoria REALE (mai torch.cuda nel processo app) ───────────────
def vram_free_mb() -> int:
    from core.gpu_mem import free_vram_mb
    return int(free_vram_mb())


def vram_used_mb() -> int:
    """VRAM occupata = totale - libera (via nvidia-smi/NVML, niente contesto CUDA)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return int(out[0]) if out else -1
    except Exception:                                        # noqa: BLE001
        return -1


def server_stats(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2) as r:
            return json.load(r)
    except Exception:                                        # noqa: BLE001
        return None


def servers() -> dict:
    """Mappa porta → stats (o None se giù). Include ComponentStore (:8189) e i server persistenti."""
    return {p: server_stats(p) for p in PORTS}


def python_procs() -> list[dict]:
    """Task manager: processi python con memoria WS (MB). Per rilevare orfani/reload che tengono VRAM/RAM."""
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                           capture_output=True, text=True, timeout=10).stdout
        out = []
        for line in r.strip().splitlines():
            parts = [c.strip('"') for c in line.split('","')]
            if len(parts) >= 5:
                mem_kb = int(parts[4].replace(",", "").replace(" K", "").strip() or 0)
                out.append({"pid": parts[1], "mem_mb": mem_kb // 1024})
        return out
    except Exception:                                        # noqa: BLE001
        return []


def snapshot(label: str = "") -> dict:
    """Stato memoria completo in un colpo: VRAM libera/usata, server attivi (+ loaded), n. processi python."""
    s = servers()
    snap = {
        "label": label,
        "vram_used_mb": vram_used_mb(),
        "vram_free_mb": vram_free_mb(),
        "servers_up": [p for p, v in s.items() if v is not None],
        "server_stats": s,
        "python_procs": len(python_procs()),
    }
    return snap


def print_snapshot(snap: dict, log=print) -> None:
    log(f"[snapshot {snap.get('label','')}] VRAM usata={snap['vram_used_mb']}MB libera={snap['vram_free_mb']}MB "
        f"| server su={snap['servers_up']} | python procs={snap['python_procs']}")


# ─────────────── teardown pulito (niente orfani, il problema "distruttivo") ───────────────
def teardown(w=None, *, free: bool = True, log=print) -> dict:
    """Libera TUTTI i server GPU (mem_coordinator.free_all), chiude la finestra (che killa l'albero dei worker
    via closeEvent, #91), e ritorna lo snapshot finale per verificare che la VRAM sia stata rilasciata."""
    if free:
        try:
            from core import mem_coordinator
            freed = mem_coordinator.free_all()
            log(f"[app_env] free_all → porte liberate: {freed}")
        except Exception as e:                               # noqa: BLE001
            log(f"[app_env] free_all errore: {e}")
    if w is not None:
        try:
            w.close()                                        # closeEvent → kill albero worker (anti-orfano VRAM)
        except Exception as e:                               # noqa: BLE001
            log(f"[app_env] close errore: {e}")
    time.sleep(2)
    return snapshot("teardown")
