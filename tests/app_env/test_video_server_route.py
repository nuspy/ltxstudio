"""Test del ROUTING video (unificazione motori, commit 1a) — senza GPU, senza generazioni vere.
Valida: arch 'ltx' + server abilitato → path server (submit+poll mockati); LTX_VIDEO_SERVER=0 →
subprocess; arch non-ltx → SEMPRE subprocess; server morto a metà → fallback subprocess;
stop() sul path server fa cancel HTTP e NON kill."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def run() -> int:
    fails = []

    def check(name, cond):
        print(("OK  " if cond else "FAIL") + " " + name, flush=True)
        if not cond:
            fails.append(name)

    from PySide6.QtCore import QCoreApplication
    _app = QCoreApplication.instance() or QCoreApplication([])

    from ui import main_window as MW
    from core import ltx2_service as svc

    # output finto che "esiste" per il ramo success
    _tmp = tempfile.mkdtemp(prefix="vsr_")
    fake_out = os.path.join(_tmp, "fake.mp4")
    Path(fake_out).write_bytes(b"x")

    calls = {"submit": 0, "cancel": 0, "spawn": 0}

    class _FakeSvc:
        @staticmethod
        def server_enabled():
            return os.environ.get("LTX_VIDEO_SERVER", "") != "0"

        @staticmethod
        def ensure_server(wait_s=120):
            return True

        @staticmethod
        def is_up():
            return True

        @staticmethod
        def submit_native(params):
            calls["submit"] += 1
            return "jid123"

        @staticmethod
        def poll_native(jid):
            return {"status": "success", "output": [fake_out],
                    "progress": {"msg": "Denoise", "step": 3, "total": 8}}

        @staticmethod
        def cancel_native(jid):
            calls["cancel"] += 1
            return {"status": "cancelling"}

    # inietta il fake in TUTTI i punti di import (`from core import ltx2_service` usa l'ATTRIBUTO
    # del package se il modulo vero è già importato → va patchato anche quello)
    import core as _core_pkg

    def _inject(mod):
        sys.modules["core.ltx2_service"] = mod  # type: ignore[assignment]
        _core_pkg.ltx2_service = mod

    _inject(_FakeSvc)

    def _fake_popen(*a, **k):
        calls["spawn"] += 1
        raise RuntimeError("subprocess NON deve partire in questo caso")

    results = []

    def _mk(params):
        w = MW._NativeLtx2Worker(dict(params))
        w.done.connect(lambda r: results.append(r))
        w.progress.connect(lambda s: None)
        return w

    try:
        # 1) arch ltx + server abilitato → path SERVER (submit chiamato, nessun subprocess)
        os.environ.pop("LTX_VIDEO_SERVER", None)
        import subprocess as _sp
        _orig_popen = _sp.Popen
        _sp.Popen = _fake_popen
        try:
            results.clear()
            _mk({"arch": "ltx", "prompt": "x", "output": fake_out}).run()
        finally:
            _sp.Popen = _orig_popen
        check("ltx+server → submit_native (no subprocess)",
              calls["submit"] == 1 and calls["spawn"] == 0
              and results and results[0]["status"] == "success" and results[0]["output"] == [fake_out])

        # 2) LTX_VIDEO_SERVER=0 → subprocess (Popen chiamato)
        os.environ["LTX_VIDEO_SERVER"] = "0"
        spawned = {"n": 0}

        class _DeadProc:
            returncode = 1
            stdout = iter(())

            def wait(self):
                return 1

        def _spawn_ok(*a, **k):
            spawned["n"] += 1
            return _DeadProc()

        _sp.Popen = _spawn_ok
        try:
            results.clear()
            _mk({"arch": "ltx", "prompt": "x", "output": os.path.join(_tmp, "no.mp4")}).run()
        finally:
            _sp.Popen = _orig_popen
        check("LTX_VIDEO_SERVER=0 → subprocess", spawned["n"] == 1 and calls["submit"] == 1)

        # 3) arch NON-ltx → sempre subprocess (mai il server)
        os.environ.pop("LTX_VIDEO_SERVER", None)
        spawned["n"] = 0
        _sp.Popen = _spawn_ok
        try:
            results.clear()
            _mk({"arch": "self_forcing", "prompt": "x",
                 "output": os.path.join(_tmp, "no2.mp4")}).run()
        finally:
            _sp.Popen = _orig_popen
        check("arch non-ltx → subprocess", spawned["n"] == 1 and calls["submit"] == 1)

        # 4) server morto a metà (poll unknown = job perso) → fallback subprocess
        class _LostSvc(_FakeSvc):
            @staticmethod
            def poll_native(jid):
                return {"status": "unknown"}

            @staticmethod
            def submit_native(params):
                calls["submit"] += 1
                return "jid_lost"

        _inject(_LostSvc)
        spawned["n"] = 0
        _sp.Popen = _spawn_ok
        try:
            results.clear()
            _mk({"arch": "ltx", "prompt": "x", "output": os.path.join(_tmp, "no3.mp4")}).run()
        finally:
            _sp.Popen = _orig_popen
        check("job perso sul server → fallback subprocess", spawned["n"] == 1 and calls["submit"] == 2)

        # 5) stop() sul path server → cancel HTTP, NON kill
        _inject(_FakeSvc)
        w = _mk({"arch": "ltx", "prompt": "x", "output": fake_out})
        w._server_jid = "jid123"
        killed = {"n": 0}
        _orig_kill = MW._kill_proc_tree
        MW._kill_proc_tree = lambda *_a: killed.__setitem__("n", killed["n"] + 1)
        try:
            w.stop()
        finally:
            MW._kill_proc_tree = _orig_kill
        check("stop() su path server → cancel HTTP, no kill", calls["cancel"] == 1 and killed["n"] == 0)
    finally:
        sys.modules.pop("core.ltx2_service", None)          # ripristina il modulo vero
        import importlib
        _core_pkg.ltx2_service = importlib.import_module("core.ltx2_service")

    print(f"\n{'TUTTO OK' if not fails else 'FALLITI: ' + ', '.join(fails)}")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(run())
