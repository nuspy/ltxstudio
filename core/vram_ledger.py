"""Ledger LIVE di ciò che sta in VRAM (e parcheggiato in RAM) — F3 unified-model-manager.

Richiesta utente: un registro dei modelli caricati in VRAM per non perderne mai traccia,
cross-architettura (immagini E video) e cross-venv. "Modello" = tutto ciò che va in VRAM.

Design: SHARDING PER-PID — ogni processo scrive SOLO il proprio file data/vram_ledger/<pid>.json
(atomico: .tmp + os.replace). Niente lock: zero contese di scrittura, zero corruzione. Lo snapshot
fonde tutti i file e SCARTA/cancella quelli di PID morti → un worker che crasha senza unregister
si ripulisce da solo. Liveness su Windows via OpenProcess+GetExitCodeProcess (os.kill(pid,0) è
inaffidabile); i PID riciclati sono mitigati dallo scarto delle voci più vecchie di 24h.

stdlib-only: importabile da OGNI venv (GUI, inference_server, ltx2, lingbot, worker effimeri).
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import sys
import time

_DIR = os.environ.get("LTX_VRAM_LEDGER_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vram_ledger")
_STALE_S = 24 * 3600                                    # voci più vecchie → scartate (PID riciclati)
_own: dict = {}                                         # chiave modello → entry (stato del MIO pid)
_registered_atexit = False


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:                                   # noqa: BLE001 — non-Windows: fallback os.kill
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:                               # noqa: BLE001
            return False


def _own_path() -> str:
    return os.path.join(_DIR, f"{os.getpid()}.json")


def _flush() -> None:
    """Scrive il file del MIO pid (atomico). Vuoto → file rimosso."""
    try:
        os.makedirs(_DIR, exist_ok=True)
        p = _own_path()
        if not _own:
            try:
                os.remove(p)
            except OSError:
                pass
            return
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "venv": os.path.basename(sys.prefix),
                       "ts": int(time.time()), "models": _own}, fh, indent=1)
        os.replace(tmp, p)
    except Exception:                                   # noqa: BLE001 — best-effort, mai fatale
        pass


def register(key: str, mb: float, *, category: str = "", state: str = "vram",
             path: str = "", extra: dict | None = None) -> None:
    """Upsert di un modello caricato da QUESTO processo. state ∈ {"vram", "ram_parked"}."""
    global _registered_atexit
    if not key or mb <= 0:
        return
    _own[key] = {"mb": round(float(mb), 1), "category": category or "", "state": state,
                 "path": path or "", "ts": int(time.time()), **(extra or {})}
    if not _registered_atexit:
        _registered_atexit = True
        atexit.register(clear_own)
    _flush()


def unregister(key: str) -> None:
    if _own.pop(key, None) is not None:
        _flush()


def set_state(key: str, state: str) -> None:
    """Cambio di stato (es. demote: vram → ram_parked) senza ri-dichiarare i metadati."""
    e = _own.get(key)
    if e is not None:
        e["state"] = state
        e["ts"] = int(time.time())
        _flush()


def clear_own() -> None:
    """Svuota le voci di QUESTO processo (fine job / atexit)."""
    _own.clear()
    _flush()


def snapshot() -> list[dict]:
    """Tutte le voci dei processi VIVI: [{pid, venv, key, mb, category, state, path, ts}].
    I file di PID morti o stantii vengono CANCELLATI qui (garbage collection)."""
    out: list[dict] = []
    try:
        names = os.listdir(_DIR)
    except OSError:
        return out
    now = time.time()
    for name in names:
        if not name.endswith(".json"):
            continue
        fp = os.path.join(_DIR, name)
        try:
            pid = int(name[:-5])
        except ValueError:
            continue
        try:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:                               # noqa: BLE001 — replace concorrente: riprova al prossimo giro
            continue
        stale = now - float(data.get("ts", 0)) > _STALE_S
        if stale or not _pid_alive(pid):
            try:
                os.remove(fp)
            except OSError:
                pass
            continue
        for key, e in (data.get("models") or {}).items():
            out.append({"pid": pid, "venv": data.get("venv", ""), "key": key,
                        "mb": float(e.get("mb", 0) or 0), "category": e.get("category", ""),
                        "state": e.get("state", "vram"), "path": e.get("path", ""),
                        "ts": e.get("ts", 0)})
    out.sort(key=lambda d: -d["mb"])
    return out


def total_mb(category: str | None = None, state: str = "vram") -> float:
    return sum(e["mb"] for e in snapshot()
               if (category is None or e["category"] == category) and e["state"] == state)


if __name__ == "__main__":                              # diagnostica CLI
    for e in snapshot():
        print(f"{e['mb']:9.0f} MB  {e['state']:10s} {e['category']:6s} pid={e['pid']:<6} {e['key']}")
    print(f"tot VRAM: {total_mb():.0f} MB — tot RAM parcheggiata: {total_mb(state='ram_parked'):.0f} MB")
