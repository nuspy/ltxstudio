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


LAST_EXIT = os.path.join(ORCH, "ultima_uscita_app.txt")


def _wait_clean(min_gap_s=30.0, timeout=120.0):
    """Prima di avviare l'app di test: la precedente deve essere davvero sparita (porta libera, nessun live_guard/app.py di QUESTO worktree) e
    devono essere passati almeno `min_gap_s` dalla sua uscita. Giro 1/11 00:15: la prova a secco era finita alle 00:15:10 e il giro vero e partito
    7 s dopo; l'app nuova non ha mai risposto (nessuna riga di log in 4 minuti) e il giro e morto per TimeoutError senza che nessuno se ne accorgesse."""
    import psutil
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            gap = time.time() - float(open(LAST_EXIT).read().strip())
        except Exception:                                    # noqa: BLE001
            gap = 1e9
        busy = port_busy(PORT)
        left = []
        for pr in psutil.process_iter(["pid", "cmdline"]):
            cl_ = " ".join(pr.info.get("cmdline") or [])
            if ROOT in cl_ and ("live_guard.py" in cl_ or cl_.rstrip().endswith("app.py")) and pr.pid != os.getpid():
                left.append(pr.pid)
        if gap >= min_gap_s and not busy and not left:
            return
        time.sleep(2)
    say("ATTENZIONE: l'app precedente non risulta ancora sparita dopo %.0f s: procedo comunque" % timeout)


def _mark_exit():
    try:
        open(LAST_EXIT, "w").write(str(time.time()))
    except Exception:                                        # noqa: BLE001
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="?", default="")
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
    ap.add_argument("--stop-file", default="", help="se questo file esiste PRIMA di un caso, il giro si ferma li (turno GPU finito): scrive rimasti.json e non lancia altro")
    ap.add_argument("--force-review", action="store_true", help="costruisce la pagina di revisione anche con casi non ok (default: solo se TUTTI i casi del giro sono ok)")
    ap.add_argument("--no-open", action="store_true", help="non aprire la pagina di revisione nel browser a fine giro")
    ap.add_argument("--no-autocheck", action="store_true", help="salta i controlli automatici sui risultati (tests/app_live/autocheck.py)")
    ap.add_argument("--review-ok", action="store_true",
                    help="pagina di revisione con i SOLI casi ok (+ i `baseline`), anche se altri sono rossi: serve ai giri sperimentali, dove i candidati "
                         "che falliscono i controlli automatici non vanno all'utente")
    ap.add_argument("--inventory-out", default="", help="salva l'inventario REALE dei menu dell'app in questo JSON e termina (nessun caso)")
    a = ap.parse_args()
    mode = "fix-loop" if a.fix_loop else "sweep"
    spec = json.load(open(a.cases, encoding="utf-8")) if not a.inventory_out else {"cases": [], "nome": "inventario"}
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
    _wait_clean()
    # dati di runtime NON versionati (chiavi incluse): copiati dal principale, mai committati
    for fn in ("user_prefs.json", "lora_overrides.json"):
        src, dst = os.path.join(MAIN_ROOT, "data", fn), os.path.join(ROOT, "data", fn)
        if os.path.exists(src) and fn == "user_prefs.json":
            shutil.copy2(src, dst)
    env = dict(os.environ, LTX_TEST_DRIVER="1", LTX_TEST_DRIVER_PORT=str(PORT), PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
               LTX_TEST_SNAP_DIR=os.path.join(out, "files"))
    cmd = [sys.executable, os.path.join(ROOT, "tests", "live_guard.py"), "--min-commit-gb", str(a.min_commit_gb),
           "--kill-servers-on-exit", "--out-dir", out]
    if a.max_proc_gb:
        cmd += ["--max-proc-gb", str(a.max_proc_gb)]
    cmd += ["--", sys.executable, "-u", os.path.join(ROOT, "app.py")]
    applog = open(os.path.join(out, "app_stdout.log"), "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=applog, stderr=subprocess.STDOUT)
    cl = Client()
    results, stop, stopped = [], False, False
    ac, ac_hist, ac_run = None, {}, os.path.basename(out)
    if not a.no_autocheck and not a.dry and not a.inventory_out:
        try:                                                 # controlli automatici (a) invariato (b) Integra (c) inpaint (d) video con riferimenti
            sys.path.insert(0, HERE)
            import autocheck as ac                           # noqa: E402
            ac_hist = ac.load_history() or ac.seed_history()
        except Exception as e:                               # noqa: BLE001
            say(f"controlli automatici non disponibili: {e!r}")
            ac = None
    try:
        try:
            cl.connect(timeout=150, proc=proc)
        except TimeoutError:
            # l'app non ha mai avviato il driver (giro 1/11 00:15): si butta giu l'albero, si aspetta che sia pulito e si riprova UNA volta.
            # Lo stack dell'app bloccata e in app_stdout.log (faulthandler, vedi app.py).
            say("il driver non risponde dopo 150 s: riavvio l'app e riprovo una volta (stack dell'app bloccata in app_stdout.log)")
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            time.sleep(5)
            _mark_exit()
            _wait_clean()
            applog.write("\n===== NUOVO TENTATIVO DI AVVIO =====\n")
            applog.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=applog, stderr=subprocess.STDOUT)
            cl = Client()
            cl.connect(timeout=240, proc=proc)
        say("driver collegato: " + json.dumps(cl.send({"cmd": "ping"})))
        t0 = time.time()
        while time.time() - t0 < 900:                        # InferenceServer pronto (autostart dell'app; il 02/11 ci ha messo 306 s)
            r = cl.send({"cmd": "ready"}, timeout=60)
            if r.get("server_up"):
                break
            time.sleep(3)
        say(f"app pronta (server_up={r.get('server_up')}) in {time.time() - t0:.0f}s")
        if a.inventory_out:
            inv = cl.send({"cmd": "inventory"}, timeout=180)
            json.dump(inv.get("inventory", inv), open(a.inventory_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            say("inventario salvato in " + a.inventory_out)
        for k, c in enumerate(cases, 1):
            if a.stop_file and os.path.exists(a.stop_file):           # turno GPU finito: nessun altro caso
                say(f"STOP richiesto ({a.stop_file}): il giro si ferma prima di {c['id']}")
                json.dump({"rimasti": [x["id"] for x in cases[k - 1:]]}, open(os.path.join(out, "rimasti.json"), "w", encoding="utf-8"))
                stopped = True
                break
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
            if ac is not None:
                try:
                    ac.apply(c, r, ac_run, ac_hist)
                    for x in r.get("autocheck") or []:
                        say(f"    controllo {x['nome']}: {x['esito']} {json.dumps(x.get('misure'), ensure_ascii=False)}")
                except Exception as e:                       # noqa: BLE001
                    r["autocheck_errore"] = repr(e)[:300]
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
        _mark_exit()
        if ac is not None:
            try:
                ac.save_history(ac_hist)
            except OSError as errore:
                say(f"storia dei controlli automatici non salvata: {errore!r}")
    try:                                                     # l'app di test scrive le preferenze (es. video_arch) nel worktree: ripristino la copia pulita
        shutil.copy2(os.path.join(MAIN_ROOT, "data", "user_prefs.json"), os.path.join(ROOT, "data", "user_prefs.json"))
        gs = os.path.join(ROOT, "data", "gui_state.json")
        if os.path.exists(gs):
            os.remove(gs)
    except Exception:                                        # noqa: BLE001
        pass
    ok = sum(1 for r in results if r.get("esito") == "ok")
    say(f"FINE: {ok}/{len(results)} ok; risultati in {res_path}")
    json.dump({"nome": spec.get("nome"), "modo": mode, "dry": a.dry, "risultati": len(results), "ok": ok},
              open(os.path.join(out, "sommario.json"), "w", encoding="utf-8"))
    if mode == "sweep" and results and results[-1].get("esito") == "crash" and not a.inventory_out and not stopped:
        ids = [c["id"] for c in spec["cases"]]
        last = results[-1]["id"]
        nxt = ids[ids.index(last) + 1] if last in ids and ids.index(last) + 1 < len(ids) else None
        if nxt:                                              # app morta (crash/guardiano): riavvio l'app e proseguo dal caso dopo
            say(f"app terminata: riavvio e proseguo da {nxt}")
            time.sleep(15)                                   # rilascio memoria/porte
            argv = [sys.executable, os.path.abspath(__file__), a.cases, "--sweep", "--from", nxt, "--out", out,
                    "--min-commit-gb", str(a.min_commit_gb), "--timeout", str(a.timeout)]
            if a.dry:
                argv.append("--dry")
            if a.no_open:
                argv.append("--no-open")
            if a.no_review:
                argv.append("--no-review")                   # prima il riavvio la perdeva e la pagina si apriva a meta giro (03/11)
            if a.stop_file:
                argv += ["--stop-file", a.stop_file]
            if a.force_review:
                argv.append("--force-review")
            if a.max_proc_gb:
                argv += ["--max-proc-gb", str(a.max_proc_gb)]
            return subprocess.run(argv).returncode
    _all = {}
    if os.path.exists(res_path):
        for _l in open(res_path, encoding="utf-8"):
            try:
                _r = json.loads(_l)
                _all[_r["id"]] = _r.get("esito") == "ok"
            except Exception:                                # noqa: BLE001
                pass
    _green = bool(_all) and all(_all.values()) and not stopped
    if not a.no_review and results and not a.inventory_out and (_green or a.force_review or (a.review_ok and ok)):   # la pagina all'utente solo con TUTTO verde (regola 03/11)
        import importlib.util
        _sp = importlib.util.spec_from_file_location("ltx_review", os.path.join(HERE, "review.py"))
        review = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(review)
        idx = review.build(out, spec.get("nome", "giro"), only_ok=bool(a.review_ok))
        say(f"pagina di revisione: {idx}")
        if not a.no_open:
            say("aperta nel browser: " + review.open_in_browser(idx))
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    raise SystemExit(main())
