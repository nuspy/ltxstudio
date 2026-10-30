"""Script unico dei test sull'app REALMENTE avviata (task 25, metodo di test del 30/10/2026).

Avvia UNA istanza di app.py (finestra vera + InferenceServer vero) sotto il guardiano commit (`tests/live_guard.py`),
con LTX_TEST_DRIVER=1; poi invia i casi in sequenza, uno per volta, sulla STESSA istanza (nessun riavvio: la memoria
fra job è quella reale). Alla fine chiude l'app e genera la pagina di revisione.

    python tests/app_live/run_suite.py <casi.json> [--fix-loop | --sweep] [--only id1,id2] [--from id] [--dry]
                                       [--min-commit-gb 20] [--max-proc-gb 0] [--timeout 1800] [--out DIR] [--no-review]
                                       [--allow-existing-app] [--keep-app]

 --fix-loop  si ferma al PRIMO caso non riuscito (l'app resta aperta solo se --keep-app): correggi, riavvia e rilancia
             con `--from <id>` (o `--only <id>`) per rifare LO STESSO caso.
 --sweep     annota ogni errore in orch\\app_live\\errori_<data>.md e prosegue (giro «tutte le architetture × modelli»).
 --dry       i casi diventano prove di cablaggio senza GPU (nessun job parte): widget e chiamata al backend sono veri.

Il file dei casi è JSON: {"nome": "...", "cases": [{"id": ..., "fn": "create_image|video|ops|state", ...}, ...]}.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAIN_ROOT = r"E:\ai\ltxstudio"
ORCH = r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live"
PORT = int(os.environ.get("LTX_TEST_DRIVER_PORT", "8765"))


def say(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


class Client:
    def __init__(self):
        self.s = None
        self.f = None

    def connect(self, timeout=240, proc=None):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"l'app/guardiano è terminato (rc={proc.returncode}) prima che il driver rispondesse")
            try:
                self.s = socket.create_connection(("127.0.0.1", PORT), timeout=3)
                self.s.settimeout(None)
                self.f = self.s.makefile("rwb")
                return True
            except OSError:
                time.sleep(2)
        raise TimeoutError("il driver nell'app non risponde")

    def send(self, msg, timeout=None):
        self.s.settimeout(timeout)
        self.f.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        self.f.flush()
        line = self.f.readline()
        if not line:
            raise ConnectionError("connessione chiusa dall'app (crash o guardiano?)")
        return json.loads(line.decode("utf-8"))


def port_busy(port):
    s = socket.socket()
    s.settimeout(1)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("--fix-loop", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--from", dest="from_id", default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--min-commit-gb", type=float, default=20.0)
    ap.add_argument("--max-proc-gb", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800.0, help="tetto per caso (s), se il caso non ne ha uno")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--allow-existing-app", action="store_true")
    ap.add_argument("--keep-app", action="store_true")
    a = ap.parse_args()
    mode = "fix-loop" if a.fix_loop else "sweep"
    spec = json.load(open(a.cases, encoding="utf-8"))
    cases = spec["cases"]
    if a.only:
        ids = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in ids]
    if a.from_id:
        k = next((i for i, c in enumerate(cases) if c["id"] == a.from_id), 0)
        cases = cases[k:]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = a.out or os.path.join(ORCH, f"run_{stamp}")
    os.makedirs(out, exist_ok=True)
    errfile = os.path.join(ORCH, f"errori_{time.strftime('%Y-%m-%d')}.md")
    res_path = os.path.join(out, "results.jsonl")
    say(f"{len(cases)} casi, modalità {mode}{' (DRY)' if a.dry else ''}, uscita {out}")

    # precontrollo: l'app dell'utente (o un altro server) NON deve essere viva sulle stesse porte
    from core import native_driver  # noqa: E402  (ROOT è già in sys.path sotto)
    if not a.allow_existing_app and (port_busy(PORT) or native_driver.is_up()):
        say("ABORT: c'è già un'app/server vivo (porta 8765 o InferenceServer :8189). Chiudila, oppure --allow-existing-app.")
        return 3
    # dati di runtime NON versionati (chiavi incluse): copiati dal principale, mai committati
    for fn in ("user_prefs.json", "lora_overrides.json"):
        src, dst = os.path.join(MAIN_ROOT, "data", fn), os.path.join(ROOT, "data", fn)
        if os.path.exists(src) and fn == "user_prefs.json":
            shutil.copy2(src, dst)
    env = dict(os.environ, LTX_TEST_DRIVER="1", LTX_TEST_DRIVER_PORT=str(PORT), PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    cmd = [sys.executable, os.path.join(ROOT, "tests", "live_guard.py"), "--min-commit-gb", str(a.min_commit_gb),
           "--kill-servers-on-exit", "--out-dir", out]
    if a.max_proc_gb:
        cmd += ["--max-proc-gb", str(a.max_proc_gb)]
    cmd += ["--", sys.executable, "-u", os.path.join(ROOT, "app.py")]
    applog = open(os.path.join(out, "app_stdout.log"), "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=applog, stderr=subprocess.STDOUT)
    cl = Client()
    results, stop = [], False
    try:
        cl.connect(proc=proc)
        say("driver collegato: " + json.dumps(cl.send({"cmd": "ping"})))
        t0 = time.time()
        while time.time() - t0 < 300:                        # InferenceServer pronto (autostart dell'app)
            r = cl.send({"cmd": "ready"}, timeout=60)
            if r.get("server_up"):
                break
            time.sleep(3)
        say(f"app pronta (server_up={r.get('server_up')}) in {time.time() - t0:.0f}s")
        for k, c in enumerate(cases, 1):
            c = dict(c)
            if a.dry:
                c["dry"] = True
            c.setdefault("timeout", a.timeout)
            say(f"[{k}/{len(cases)}] {c['id']} ({c.get('fn')})")
            try:
                r = cl.send({"cmd": "run_case", "case": c}, timeout=float(c["timeout"]) + 180)
            except Exception as e:                           # noqa: BLE001  — app morta/guardiano: il caso è perso
                r = {"id": c["id"], "fn": c.get("fn"), "esito": "crash", "motivo": f"{type(e).__name__}: {e} (app terminata o guardiano commit)"}
                stop = True
            r["case"] = c
            results.append(r)
            with open(res_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            say(f"    -> {r.get('esito')} {r.get('tempo_s', '')}s {str(r.get('motivo', ''))[:200]}")
            if r.get("esito") != "ok":
                with open(errfile, "a", encoding="utf-8") as fh:
                    fh.write(f"## {time.strftime('%Y-%m-%d %H:%M')} {c['id']} ({c.get('fn')}) — {r.get('esito')}\n"
                             f"- caso: `{json.dumps({k: v for k, v in c.items() if k not in ('prompt',)}, ensure_ascii=False)[:700]}`\n"
                             f"- motivo: {str(r.get('motivo'))[:900]}\n"
                             + (f"- handler: {r.get('handler')}\n" if r.get("handler") else "")
                             + (f"- dialoghi GUI: {json.dumps(r.get('dialoghi'), ensure_ascii=False)[:600]}\n" if r.get("dialoghi") else "")
                             + "\n")
                if mode == "fix-loop":
                    stop = True
            if stop:
                break
    finally:
        try:
            if not a.keep_app:
                cl.send({"cmd": "shutdown"}, timeout=30)
        except Exception:                                    # noqa: BLE001
            pass
        t1 = time.time()
        while proc.poll() is None and time.time() - t1 < 60:
            time.sleep(1)
        if proc.poll() is None and not a.keep_app:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        applog.close()
    ok = sum(1 for r in results if r.get("esito") == "ok")
    say(f"FINE: {ok}/{len(results)} ok; risultati in {res_path}")
    json.dump({"nome": spec.get("nome"), "modo": mode, "dry": a.dry, "risultati": len(results), "ok": ok},
              open(os.path.join(out, "sommario.json"), "w", encoding="utf-8"))
    if not a.no_review and results:
        import importlib.util
        _sp = importlib.util.spec_from_file_location("ltx_review", os.path.join(HERE, "review.py"))
        review = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(review)
        idx = review.build(out, spec.get("nome", "giro"))
        say(f"pagina di revisione: {idx}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    raise SystemExit(main())
