"""Driver di test DENTRO l'app vera (task 25, metodo di test del 30/10/2026).

Si attiva SOLO con la variabile d'ambiente LTX_TEST_DRIVER=1 (app.py chiama `attach(win)` se esiste questo modulo;
tests/ non viaggia nelle build). L'app parte normalmente (finestra vera, InferenceServer vero, gestione memoria
reale); un socket su 127.0.0.1:LTX_TEST_DRIVER_PORT (default 8765, protocollo JSON a righe) riceve i casi UNO ALLA
VOLTA dallo script `run_suite.py`.

Per ogni caso il driver: imposta i widget VERI, preme il bottone VERO, aspetta il job nella coda reale
(`GenerationQueue`), e restituisce uscite, tempo, picchi VRAM/RAM, errori mostrati in GUI (QMessageBox) e — il punto
chiave — la CORRISPONDENZA: quale worker/endpoint del backend è stato davvero chiamato e con quali parametri
principali, confrontata con quello che dicono i widget. Se differisce il caso FALLISCE anche se esce un file.

`dry: true` nel caso = i widget e la chiamata al backend sono VERI ma il job NON parte (si ferma alla coda e
risponde con un'immagine segnaposto): prova di cablaggio senza GPU.

Nessuna chiamata torch.cuda nel processo app (regola app-no-cuda-context): VRAM via nvidia-smi.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import traceback
import urllib.request

PORT = int(os.environ.get("LTX_TEST_DRIVER_PORT", "8765"))
MEDIA_EXT = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm", ".wav", ".mp3", ".glb", ".obj", ".exr")
_KEEP = ("module", "arch", "task", "model", "model_gguf", "krea2_variant", "krea2_dit", "qwen21_dit", "flux3_dit",
         "ltx_release", "video_arch", "loras", "lora", "prompt", "negative", "seed", "steps", "guidance", "cfg",
         "width", "height", "length", "fps", "strength", "denoise", "init_image", "image_path", "mask", "mask_path",
         "ref_images", "control_image", "control_type", "outpaint", "output", "native_audio", "h3_dit", "h3_profile",
         "engine", "mode", "scale", "images", "audio_path", "klein_te", "video", "references", "msr_lora", "edit_lora", "ltx_release")


def _safe(v, depth=0):
    """Valore JSON-safe e corto (niente blob base64, niente oggetti)."""
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, str):
        return v if len(v) <= 400 else v[:400] + f"…(+{len(v) - 400})"
    if isinstance(v, (list, tuple)):
        return [_safe(x, depth + 1) for x in list(v)[:20]] if depth < 4 else str(v)[:200]
    if isinstance(v, dict):
        return {str(k): _safe(x, depth + 1) for k, x in list(v.items())[:60]} if depth < 4 else str(v)[:200]
    return f"<{type(v).__name__}>"


def main_params(p: dict) -> dict:
    return {k: _safe(p[k]) for k in _KEEP if isinstance(p, dict) and k in p}


def _media_in(obj, out: list, depth=0):
    if depth > 5:
        return
    if isinstance(obj, str):
        if obj.lower().endswith(MEDIA_EXT) and os.path.isfile(obj):
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _media_in(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _media_in(v, out, depth + 1)


class Recorder:
    """Cattura le chiamate al backend e gli errori mostrati in GUI durante un caso."""

    def __init__(self):
        self.calls: list[dict] = []
        self.dialogs: list[dict] = []
        self.finished: list[dict] = []
        self.lock = threading.Lock()
        self.t0 = time.time()

    def reset(self):
        with self.lock:
            self.calls, self.dialogs, self.finished = [], [], []
            self.t0 = time.time()

    def call(self, via, **kw):
        with self.lock:
            self.calls.append({"t": round(time.time() - self.t0, 2), "via": via, **kw})

    def dialog(self, kind, title, text):
        with self.lock:
            self.dialogs.append({"t": round(time.time() - self.t0, 2), "kind": kind, "title": str(title)[:200],
                                 "text": str(text)[:1500]})

    def fin(self, **kw):
        with self.lock:
            self.finished.append({"t": round(time.time() - self.t0, 2), **kw})


class Sampler(threading.Thread):
    """VRAM usata (nvidia-smi), RAM privata dell'albero app (psutil), commit libero."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.vram_peak = 0
        self.vram_base = 0
        self.ram_peak = 0
        self.commit_min = 10 ** 9
        self.active = False

    def _vram(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            return int(out[0])
        except Exception:                                    # noqa: BLE001
            return 0

    def _tree_mb(self):
        try:
            import psutil
            me = psutil.Process(os.getpid())
            tot = 0
            for p in [me] + me.children(recursive=True):
                try:
                    mi = p.memory_info()
                    tot += (getattr(mi, "private", None) or mi.rss) / 2 ** 20
                except Exception:                            # noqa: BLE001
                    pass
            return tot
        except Exception:                                    # noqa: BLE001
            return 0

    def begin(self):
        self.vram_base = self._vram()
        self.vram_peak, self.ram_peak, self.commit_min = self.vram_base, 0, 10 ** 9
        self.active = True

    def end(self) -> dict:
        self.active = False
        return {"vram_base_mb": self.vram_base, "vram_peak_mb": self.vram_peak, "ram_peak_mb": int(self.ram_peak),
                "commit_free_min_mb": int(self.commit_min) if self.commit_min < 10 ** 9 else None}

    def run(self):
        try:
            from core import st_direct
        except Exception:                                    # noqa: BLE001
            st_direct = None
        while not self.stop:
            if self.active:
                self.vram_peak = max(self.vram_peak, self._vram())
                self.ram_peak = max(self.ram_peak, self._tree_mb())
                if st_direct:
                    try:
                        self.commit_min = min(self.commit_min, float(st_direct.commit_avail_mb()))
                    except Exception:                        # noqa: BLE001
                        pass
            time.sleep(1.5)


# ───────────────────────────── widget ──────────────────────────────────────────────────────────────────────
def resolve(w, path: str):
    obj = w
    for part in path.split("."):
        if part == "":
            continue
        if part.endswith("]") and "[" in part:
            name, idx = part[:-1].split("[", 1)
            obj = getattr(obj, name)[int(idx)]
        else:
            obj = getattr(obj, part)
    return obj


def _hidden(cb, i) -> bool:
    """Voce nascosta dal filtro (setRowHidden sulla view del combo): l'utente non puo sceglierla."""
    try:
        return bool(cb.view().isRowHidden(i))
    except Exception:                                        # noqa: BLE001
        return False


def combo_items(cb):
    return [{"i": i, "text": cb.itemText(i), "data": _safe(cb.itemData(i))} for i in range(cb.count()) if not _hidden(cb, i)]


def select_combo(cb, want):
    """Seleziona per indice, per {"data": x}, o per sottostringa del testo / dei dati (senza maiuscole)."""
    if isinstance(want, int):
        cb.setCurrentIndex(want)
        return cb.itemText(want)
    if isinstance(want, dict) and "data" in want:
        i = cb.findData(want["data"])
        if i >= 0:
            cb.setCurrentIndex(i)
            return cb.itemText(i)
    else:
        w = str(want).lower()
        for i in range(cb.count()):
            d = cb.itemData(i)
            if _hidden(cb, i):
                continue
            if w == cb.itemText(i).lower():
                cb.setCurrentIndex(i)
                return cb.itemText(i)
        for i in range(cb.count()):
            d = cb.itemData(i)
            if _hidden(cb, i):
                continue
            if w in cb.itemText(i).lower() or (isinstance(d, str) and w in d.lower()) or \
                    (isinstance(d, dict) and w in json.dumps(d, default=str).lower()):
                cb.setCurrentIndex(i)
                return cb.itemText(i)
    raise LookupError(f"voce assente dal menu: {want!r}; voci: {[cb.itemText(i) for i in range(cb.count())][:40]}")


def set_widget(wd, value):
    from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit, QPlainTextEdit,
                                   QSlider, QTextEdit)
    if isinstance(wd, QComboBox):
        return select_combo(wd, value)
    if isinstance(wd, QAbstractButton):
        wd.setChecked(bool(value))
        return bool(value)
    if isinstance(wd, (QAbstractSpinBox, QSlider)):
        wd.setValue(value)
        return wd.value()
    if isinstance(wd, (QPlainTextEdit, QTextEdit)):
        wd.setPlainText(str(value))
        return str(value)[:80]
    if isinstance(wd, QLineEdit):
        wd.setText(str(value))
        return str(value)[:80]
    raise TypeError(f"widget non gestito: {type(wd).__name__}")


def widget_value(wd):
    from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit, QPlainTextEdit,
                                   QSlider, QTextEdit)
    if isinstance(wd, QComboBox):
        return {"text": wd.currentText(), "data": _safe(wd.currentData())}
    if isinstance(wd, QAbstractButton):
        return wd.isChecked()
    if isinstance(wd, (QAbstractSpinBox, QSlider)):
        return wd.value()
    if isinstance(wd, (QPlainTextEdit, QTextEdit)):
        return wd.toPlainText()
    if isinstance(wd, QLineEdit):
        return wd.text()
    return None


# ───────────────────────────── driver ──────────────────────────────────────────────────────────────────────
class Driver:
    def __init__(self, win):
        from PySide6.QtCore import QObject, Qt, Signal
        self.w = win
        self.rec = Recorder()
        self.sampler = Sampler()
        self.dry = False
        self._dry_png = None
        drv = self

        class _Inv(QObject):
            call = Signal(object)

            def __init__(self):
                super().__init__()
                self.call.connect(self._run, Qt.QueuedConnection)

            def _run(self, job):
                try:
                    job["r"] = job["fn"]()
                except BaseException:                        # noqa: BLE001
                    job["e"] = traceback.format_exc()[-1800:]
                job["ev"].set()

        self._inv = _Inv()
        self._install_hooks()

    # esegue fn nel thread GUI e ne restituisce il valore
    def gui(self, fn, timeout=120):
        job = {"fn": fn, "ev": threading.Event()}
        self._inv.call.emit(job)
        if not job["ev"].wait(timeout):
            raise TimeoutError("il thread GUI non risponde (timeout %ss)" % timeout)
        if "e" in job:
            raise RuntimeError(job["e"])
        return job.get("r")

    # ── hook sul backend e sulla GUI ─────────────────────────────────────────────────────────────────
    def _install_hooks(self):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QMessageBox
        drv, rec = self, self.rec

        def mk(kind, ret):
            def f(*a, **k):
                title = a[1] if len(a) > 1 else k.get("title", "")
                text = a[2] if len(a) > 2 else k.get("text", "")
                rec.dialog(kind, title, text)
                return ret
            return staticmethod(f)
        QMessageBox.warning = mk("warning", QMessageBox.Ok)
        QMessageBox.information = mk("information", QMessageBox.Ok)
        QMessageBox.critical = mk("critical", QMessageBox.Ok)
        QMessageBox.question = mk("question", QMessageBox.Yes)

        def _exec(box, *a, **k):          # QMessageBox personalizzati con exec() bloccherebbero il thread GUI: si registra il testo e si risponde «annulla»
            try:
                rec.dialog("exec", box.windowTitle(), box.text() + (chr(10) + box.informativeText() if box.informativeText() else ""))
            except Exception:             # noqa: BLE001
                pass
            return QMessageBox.Cancel
        QMessageBox.exec = _exec
        QMessageBox.exec_ = _exec

        from ui.job_queue import GenerationQueue
        orig_enqueue = GenerationQueue.enqueue

        def enqueue(q, label, worker, on_done=None, on_progress=None):
            params = getattr(worker, "params", None) or getattr(worker, "cfg", None)      # _VideoEditAIWorker tiene la config in .cfg
            rec.call("queue", label=str(label), worker=type(worker).__name__, module=getattr(worker, "module", None),
                     params=main_params(params) if isinstance(params, dict) else {})
            if drv.dry:
                fake = {"status": "success", "output": [drv._placeholder()], "dry": True}

                def _fin():
                    rec.fin(label=str(label), status="success", dry=True, outputs=fake["output"])
                    if on_done:
                        on_done(fake)
                QTimer.singleShot(300, _fin)
                return -1

            def _wrapped(res):
                outs: list = []
                _media_in(res, outs)
                rec.fin(label=str(label), status=(res or {}).get("status"), error=str((res or {}).get("error", ""))[-1200:],
                        outputs=outs)
                if on_done:
                    on_done(res)
            return orig_enqueue(q, label, worker, on_done=_wrapped, on_progress=on_progress)
        GenerationQueue.enqueue = enqueue

        # dialogo d'anteprima del risultato immagine: si registra il file, niente finestra modale
        w = self.w

        def _show(path, *a, **k):
            rec.fin(label="risultato-immagine", status="success", outputs=[str(path)])
        w._show_inpaint_result = _show

        from core import native_driver
        o_submit = native_driver.submit

        def submit(module, params):
            rec.call("native_driver.submit", module=module, params=main_params(params))
            return o_submit(module, params)
        native_driver.submit = submit
        try:
            from core import img_gen_job
            o_run = img_gen_job.run

            def run(params, *a, **k):
                rec.call("img_gen_job.run", params=main_params(params))
                return o_run(params, *a, **k)
            img_gen_job.run = run
        except Exception:                                    # noqa: BLE001
            pass
        o_open = urllib.request.urlopen

        def urlopen(req, *a, **k):
            try:
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if "127.0.0.1" in url or "localhost" in url:
                    path = url.split("//", 1)[-1].split("/", 1)[-1]
                    if not path.startswith(("history", "system_stats", "preview", "health")):
                        body = None
                        d = getattr(req, "data", None)
                        if d and len(d) < 400000:
                            try:
                                body = main_params(json.loads(d.decode("utf-8")).get("params", json.loads(d.decode("utf-8"))))
                            except Exception:                # noqa: BLE001
                                body = None
                        rec.call("http", url=url.split("//", 1)[-1][:140], body=body)
            except Exception:                                # noqa: BLE001
                pass
            return o_open(req, *a, **k)
        urllib.request.urlopen = urlopen
        o_popen = subprocess.Popen

        class _Popen(o_popen):
            def __init__(s, args, *a, **k):
                try:
                    av = [str(x) for x in (args if isinstance(args, (list, tuple)) else [args])]
                    if any(x.endswith(".py") or x == "-m" for x in av):
                        rec.call("subprocess", argv=[os.path.basename(av[0])] + [x[-90:] for x in av[1:6]])
                except Exception:                            # noqa: BLE001
                    pass
                super().__init__(args, *a, **k)
        subprocess.Popen = _Popen

    def _placeholder(self):
        if self._dry_png and os.path.isfile(self._dry_png):
            return self._dry_png
        from PIL import Image, ImageDraw
        p = os.path.join(os.environ.get("TEMP", "."), "ltx_dry_placeholder.png")
        im = Image.new("RGB", (512, 512), (40, 40, 60))
        ImageDraw.Draw(im).text((150, 240), "DRY RUN (nessun job)", fill=(255, 255, 255))
        im.save(p)
        self._dry_png = p
        return p

    # ── operazioni generiche ─────────────────────────────────────────────────────────────────────────
    def op(self, op: dict) -> dict:
        w = self.w
        if "set" in op:
            wd = resolve(w, op["set"])
            return {"op": f"set {op['set']}", "val": set_widget(wd, op["value"])}
        if "click" in op:
            wd = resolve(w, op["click"])
            wd.click()
            return {"op": f"click {op['click']}"}
        if "call" in op:
            fn = resolve(w, op["call"])
            r = fn(*op.get("args", []))
            return {"op": f"call {op['call']}", "ret": _safe(r)}
        if "menu" in op:
            return {"op": f"menu {op['menu']}", "items": combo_items(resolve(w, op["menu"]))}
        if "lora" in op:                                     # {"lora": "frame1_tab.gen_lora_tab", "piece": "...", "strength": 0.8}
            return {"op": "lora", **self.pick_lora(resolve(w, op["lora"]), op)}
        if "wait" in op:
            time.sleep(float(op["wait"]))
            return {"op": f"wait {op['wait']}"}
        raise ValueError(f"operazione sconosciuta: {op}")

    def pick_lora(self, panel, spec: dict) -> dict:
        panel.refresh()
        piece = str(spec["piece"]).lower()
        hit = [r for r in panel._rows.values()
               if piece in ((r["entry"].filename or "") + " " + (r["entry"].name or "")).lower()]
        if not hit:
            raise LookupError(f"LoRA non nel pannello: {spec['piece']} (righe: {len(panel._rows)})")
        r = hit[0]
        r["cb"].setChecked(True)
        r["sp"].setValue(float(spec.get("strength", 1.0)))
        return {"file": r["entry"].filename, "strength": float(spec.get("strength", 1.0)),
                "triggers": list(getattr(r["entry"], "triggers", []) or [])}

    # ── attesa del job ────────────────────────────────────────────────────────────────────────────────
    def wait_job(self, timeout: float, start_wait: float = 25.0) -> None:
        rec, t0 = self.rec, time.time()
        while time.time() - t0 < start_wait:                 # il click deve generare una chiamata al backend, o un errore in GUI
            if any(c["via"] == "queue" for c in rec.calls) or rec.dialogs:
                break
            time.sleep(0.3)
        if not any(c["via"] == "queue" for c in rec.calls):
            return
        n_q = sum(1 for c in rec.calls if c["via"] == "queue")
        t1 = time.time()
        while time.time() - t1 < timeout:
            fin = [f for f in rec.finished if f.get("label") != "risultato-immagine"]
            if len(fin) >= n_q:
                time.sleep(1.5)
                return
            time.sleep(0.5)
        raise TimeoutError(f"job non terminato entro {timeout:.0f}s")

    # ── casi ──────────────────────────────────────────────────────────────────────────────────────────
    def run_case(self, case: dict) -> dict:
        t0 = time.time()
        self.rec.reset()
        self.dry = bool(case.get("dry"))
        res = {"id": case.get("id"), "fn": case.get("fn"), "dry": self.dry, "esito": "errore", "motivo": ""}
        self.sampler.begin()
        try:
            self._wait_inventory()
            fn = getattr(self, "case_" + str(case.get("fn", "ops")), None)
            if fn is None:
                raise ValueError(f"tipo di caso sconosciuto: {case.get('fn')}")
            res.update(fn(case) or {})
            self._judge(case, res)
        except Exception as e:                               # noqa: BLE001
            res["motivo"] = f"{type(e).__name__}: {e}"[:1800]
            res["traceback"] = traceback.format_exc()[-1500:]
        res.update(self.sampler.end())
        res["tempo_s"] = round(time.time() - t0, 1)
        res["calls"] = self.rec.calls[:12]
        res["dialoghi"] = self.rec.dialogs[:6]
        res["finiti"] = [{k: v for k, v in f.items() if k != "t"} for f in self.rec.finished[:6]]
        try:
            res["gui_log"] = self.gui(lambda: [l[:240] for l in self.w.log.toPlainText().splitlines()[-30:]], 20)
        except Exception:                                    # noqa: BLE001
            pass
        return res

    def _wait_inventory(self, timeout: float = 90.0) -> None:
        """L'inventario dei modelli si carica in modo asincrono dopo l'avvio: il primo caso non deve partire con i menu ancora vuoti
        («(nessun modello trovato)»). Si attende che il menu FLUX e quello video abbiano voci vere."""
        def ready():
            f, vt = self.w.frame1_tab, self.w.video_tab
            t0 = f.gen_model.itemText(0) if f.gen_model.count() else ""
            return bool(f.gen_model.count()) and "nessun modello" not in t0.lower() and vt.model.count() > 1
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.gui(ready, 20):
                    return
            except Exception:                                # noqa: BLE001
                pass
            time.sleep(1.5)

    def _judge(self, case, res):
        """Esito finale: job riuscito + uscita presente + corrispondenza widget↔backend."""
        problems = list(res.get("mismatch", []))
        qcalls = [c for c in self.rec.calls if c["via"] == "queue"]
        if case.get("expect_no_call"):
            ok_call = not qcalls
        else:
            ok_call = bool(qcalls)
        if not ok_call:
            problems.append("nessuna chiamata al backend (bottone senza effetto o errore in GUI)" if not qcalls
                            else "chiamata inattesa al backend")
        fin = [f for f in self.rec.finished if f.get("label") != "risultato-immagine"]
        for f in fin:
            if f.get("status") != "success":
                problems.append(f"job fallito: {f.get('error') or f.get('status')}")
        outs = []
        for f in self.rec.finished:
            outs += f.get("outputs") or []
        outs = [o for i, o in enumerate(outs) if o not in outs[:i] and os.path.isfile(str(o))]
        res["uscite"] = outs
        if qcalls and not outs and not problems and not self.dry:
            problems.append("job finito ma nessun file di uscita")
        errs = [d for d in self.rec.dialogs if d["kind"] in ("critical", "warning", "exec")]
        if errs and not case.get("expect_dialog"):
            problems.append("errore mostrato in GUI: " + errs[0]["text"][:300])
        res["esito"] = "ok" if not problems else "fallito"
        res["problemi"] = problems
        res["motivo"] = "; ".join(problems)[:1800]

    # i casi ----------------------------------------------------------------------------------------------
    def case_ops(self, case):
        out = []
        for o in case.get("ops", []):
            out.append(self.gui(lambda o=o: self.op(o), 60))
        if case.get("wait_job", True) and any("click" in o for o in case.get("ops", [])):
            self.wait_job(float(case.get("timeout", 600)))
        return {"ops_eseguite": out}

    def case_create_image(self, case):
        """Crea → Immagine: arch, modello, LoRA, prompt, formato; bottone VERO «Genera»."""
        w = self.w
        steps = {}

        def setup():
            f, vt = w.frame1_tab, w.video_tab
            from core import offline_inventory as oi
            if case.get("nsfw"):
                f.img_nsfw.setCurrentText("NSFW")
            f.gen_arch.setCurrentText(case["arch"]) if isinstance(case["arch"], str) and \
                f.gen_arch.findText(case["arch"]) >= 0 else select_combo(f.gen_arch, case["arch"])
            if case.get("model"):
                steps["model"] = select_combo(f.gen_model, case["model"])
            if case.get("w"):
                vt.width.setValue(case["w"])
            if case.get("h"):
                vt.height.setValue(case["h"])
            if f.gen_lora_tab is not None and hasattr(f.gen_lora_tab, "_rows"):
                f.gen_lora_tab.refresh()
                for r in f.gen_lora_tab._rows.values():       # parto sempre da LoRA spente
                    r["cb"].setChecked(False)
            steps["loras"] = [self.pick_lora(f.gen_lora_tab, L) for L in case.get("loras", [])]
            f.gen_prompt.setPlainText(case["prompt"])
            if hasattr(f, "gen_neg") and case.get("negative") is not None:
                f.gen_neg.setPlainText(case["negative"])
            f.gen_seed_rand.setChecked(False)
            f.gen_seed.setValue(int(case.get("seed", 1)))
            if case.get("steps"):
                f.gen_steps.setValue(case["steps"])
            if case.get("cfg") is not None:
                f.gen_cfg.setValue(case["cfg"])
            for k, v in (case.get("widgets") or {}).items():
                set_widget(resolve(w, k), v)
            cfg = f.get_gen_config()
            steps["cfg_gui"] = main_params({k: cfg[k] for k in cfg if k != "prompt_raw"})
            steps["widget"] = {"model": widget_value(f.gen_model), "arch": widget_value(f.gen_arch),
                               "prompt": widget_value(f.gen_prompt), "seed": widget_value(f.gen_seed)}
            return cfg
        self.gui(setup, 90)
        self.gui(lambda: w.frame1_tab.gen_btn.click(), 30)
        self.wait_job(float(case.get("timeout", 900)))
        res = {"widget_state": steps}
        self._compare_image(case, steps, res)
        return res

    def _compare_image(self, case, steps, res):
        q = [c for c in self.rec.calls if c["via"] == "queue"]
        mis = []
        if q:
            p = q[0]["params"]
            res["handler"] = f"{q[0]['worker']}:{q[0].get('module')}"
            cfg = steps.get("cfg_gui", {})
            if "seed" in case and p.get("seed") != int(case["seed"]):
                mis.append(f"seed: impostato {case['seed']} ma il backend riceve {p.get('seed')}")
            for k in ("arch", "model", "steps", "seed", "width", "height", "krea2_variant", "qwen21_dit", "flux3_dit"):
                if k in cfg and k in p and cfg[k] != p[k]:
                    mis.append(f"{k}: la GUI dice {cfg[k]!r} ma il backend riceve {p[k]!r}")
            for L in steps.get("loras", []):
                fn = os.path.basename(str(L["file"])).lower()
                got = [x for x in (p.get("loras") or []) if fn in json.dumps(x).lower()]
                if not got:
                    mis.append(f"LoRA {fn} attiva nel pannello ma assente dai parametri del backend")
                else:
                    sv = [v for v in (got[0].values() if isinstance(got[0], dict) else []) if isinstance(v, (int, float))]
                    if sv and abs(float(L["strength"]) - float(sv[-1])) > 1e-6 and float(L["strength"]) not in sv:
                        mis.append(f"forza LoRA {fn}: pannello {L['strength']} ma backend {sv}")
            if not p.get("loras") and steps.get("loras"):
                mis.append("LoRA spuntate ma parametro 'loras' assente nel backend")
            mis += self._extra_loras(steps, p)
            if steps["widget"]["prompt"] and p.get("prompt") and steps["widget"]["prompt"].strip()[:60] not in str(p["prompt"]):
                mis.append("il prompt inviato non contiene quello scritto nel widget")
            mw = case.get("expect_handler")
            if mw and mw not in res["handler"] and mw not in json.dumps(p):
                mis.append(f"handler atteso {mw!r}, chiamato {res['handler']!r}")
            for k, v in (case.get("expect_params") or {}).items():
                if p.get(k) != v:
                    mis.append(f"parametro atteso {k}={v!r}, inviato {p.get(k)!r}")
        res["mismatch"] = mis

    def _extra_loras(self, steps, p) -> list:
        """Corrispondenza INVERSA: ogni LoRA che arriva al backend deve essere una di quelle spuntate nel pannello
        (altrimenti una selezione rimasta da un'altra arch/funzione viene applicata in silenzio)."""
        want = {os.path.basename(str(L["file"])).lower() for L in steps.get("loras", [])}
        out = []
        for x in (p.get("loras") or p.get("lora") or []):
            nm = os.path.basename(str(x.get("name") if isinstance(x, dict) else x).replace("\\", "/")).lower()
            if nm and nm not in want:
                out.append(f"LoRA {nm} arriva al backend ma NON è spuntata nel pannello (selezione rimasta da un'altra arch?)")
        return out

    def case_video(self, case):
        """Crea → Video (LTX / MiniMax H3 / …): bottone VERO «Genera»."""
        w = self.w
        steps = {}

        def setup():
            vt = w.video_tab
            if case.get("nsfw"):
                vt.nsfw.setCurrentText("NSFW")
            i = vt.video_arch.findData(case["arch"])
            if i < 0:
                raise LookupError(f"arch video assente: {case['arch']} ({[vt.video_arch.itemData(k) for k in range(vt.video_arch.count())]})")
            vt.video_arch.setCurrentIndex(i)
            if case.get("model"):
                steps["model"] = select_combo(vt.model, case["model"])
            if hasattr(vt, "lora"):
                vt.lora.refresh()
                for r in vt.lora._rows.values():
                    r["cb"].setChecked(False)
            steps["loras"] = [self.pick_lora(vt.lora, L) for L in case.get("loras", [])]
            vt.prompt.setPlainText(case["prompt"])
            vt.res_preset.setCurrentText("Personalizzato")
            vt.width.setValue(case["w"]); vt.height.setValue(case["h"]); vt.length.setValue(case["secs"])
            if hasattr(vt, "sampler_seed"):                  # Seed fisso: casella + valore sono nel gruppo Sampler/Seed
                vt.sampler_seed.fixed_seed.setChecked(True)
                vt.sampler_seed.seed.setValue(int(case.get("seed", 1)))
            if "audio" in case:
                vt.native_audio.setChecked(bool(case["audio"]))
            if case.get("refs"):                                  # Riferimenti multipli (punto 26): soggetti + SFONDO obbligatorio su LTX
                rp = vt.reference_panel
                rp.enable.setChecked(True)
                rp.clear()                                        # parto sempre dal pannello vuoto (i casi si susseguono sulla stessa istanza)
                for pth in case["refs"]:
                    rp.add_path(pth)
                if case.get("background"):
                    rp.set_background(case["background"])
                steps["refs"] = {"n": len(rp.names()), "bg": getattr(rp, "bg_path", ""), "needs_bg": bool(getattr(rp, "_needs_background", False))}
            for k, v in (case.get("widgets") or {}).items():
                set_widget(resolve(w, k), v)
            vals = vt.get_values()
            steps["valori_gui"] = main_params(vals)
            if hasattr(vt, "effective_video_size"):                # la GUI dichiara l'arrotondamento a 64 (o H3: Tela): si confronta il valore EFFETTIVO
                if case["arch"] in ("minimax_h3", "krea_rt"):
                    steps["valori_gui"].pop("width", None); steps["valori_gui"].pop("height", None)
                    steps["size_note"] = vt.size_effective.text()
                else:
                    ew, eh = vt.effective_video_size(vals["width"], vals["height"])
                    steps["valori_gui"]["width"], steps["valori_gui"]["height"] = ew, eh
                    steps["size_note"] = vt.size_effective.text() if vt.size_effective.isVisible() else ""
                    if (ew, eh) != (vals["width"], vals["height"]) and not vt.size_effective.isVisible():
                        raise AssertionError("la GUI arrotonda le dimensioni ma non lo mostra")
            steps["widget"] = {"model": widget_value(vt.model), "arch": widget_value(vt.video_arch),
                               "prompt": widget_value(vt.prompt)}
        self.gui(setup, 90)
        self.gui(lambda: w.gen_btn.click(), 30)
        self.wait_job(float(case.get("timeout", 1800)))
        res = {"widget_state": steps}
        q = [c for c in self.rec.calls if c["via"] == "queue"]
        mis = []
        if q:
            p = q[0]["params"]
            res["handler"] = f"{q[0]['worker']}:{q[0].get('module')}"
            vals = steps.get("valori_gui", {})
            for k in ("width", "height", "seed", "model", "arch", "video_arch", "native_audio"):
                if k in vals and k in p and vals[k] != p[k]:
                    mis.append(f"{k}: la GUI dice {vals[k]!r} ma il backend riceve {p[k]!r}")
            for L in steps.get("loras", []):
                fn = os.path.basename(str(L["file"])).lower()
                if fn not in json.dumps(p.get("loras") or p.get("lora") or []).lower():
                    mis.append(f"LoRA {fn} attiva nel pannello ma assente dai parametri del backend")
            mis += self._extra_loras(steps, p)
            if case.get("refs"):                                    # punto 26: i riferimenti e lo sfondo devono arrivare al backend
                refs = p.get("references") or []
                want = len(case["refs"]) + (1 if case.get("background") else 0)
                if len(refs) != want:
                    mis.append(f"riferimenti: impostati {want} (soggetti + sfondo) ma il backend ne riceve {len(refs)}")
                if case.get("background") and not any(isinstance(r, dict) and r.get("role") == "background" for r in refs):
                    mis.append("lo SFONDO e impostato nel pannello ma non arriva al backend (role=background assente)")
                if "msr" not in str(p.get("msr_lora", "")).lower():
                    mis.append("nessuna LoRA Multiple-Subject-Reference nei parametri")
            if "seed" in case and p.get("seed") != int(case["seed"]):
                mis.append(f"seed: impostato {case['seed']} ma il backend riceve {p.get('seed')}")
            for k, v in (case.get("expect_params") or {}).items():
                if p.get(k) != v:
                    mis.append(f"parametro atteso {k}={v!r}, inviato {p.get(k)!r}")
        res["mismatch"] = mis
        return res

    # ── Modifica (inpaint / outpaint / Integra = edit per istruzione) ──────────────────────────────────
    def find_button(self, root, key: str):
        """Bottone VERO per chiave i18n (es. f1.inpaint.btn): lo cerca nell'albero dei widget."""
        from PySide6.QtWidgets import QPushButton
        from ui.i18n_manager import tr
        txt = tr(key)
        for b in root.findChildren(QPushButton):
            if b.text() == txt and b.isEnabled():
                return b
        raise LookupError(f"bottone '{key}' ({txt!r}) non trovato")

    def case_edit(self, case):
        """kind: inpaint | outpaint | integrate. Carica la foto nel canvas VERO, disegna la maschera (inpaint) o allarga
        la tela (outpaint), sceglie arch/modello nei menu di Modifica e preme il bottone vero."""
        w = self.w
        steps = {}
        kind = case["kind"]

        def setup():
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QColor, QImage, QPainter
            f = w.frame1_tab
            f.sub.setCurrentIndex(1)                                  # sotto-tab Modifica
            cv = f.canvas
            cv.load_image(case["source"], "canvas")
            if kind == "outpaint":                                    # tela piu larga della foto: bordi vuoti da estendere
                cv.set_aspect(int(cv.aspect_w * float(case.get("grow", 1.4)) // 16 * 16), int(cv.aspect_h))
            elif kind == "inpaint" and case.get("mask", True):
                m = QImage(int(cv.aspect_w), int(cv.aspect_h), QImage.Format_ARGB32)
                m.fill(0)
                pt = QPainter(m)
                pt.setRenderHint(QPainter.Antialiasing)
                pt.setBrush(QColor(255, 0, 0, 255))
                pt.setPen(Qt.NoPen)
                cx, cy, r = cv.aspect_w * 0.5, cv.aspect_h * 0.5, min(cv.aspect_w, cv.aspect_h) * 0.18
                pt.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
                pt.end()
                cv.mask_image = m
                cv._refresh_mask()
            if kind in ("inpaint", "outpaint"):
                if case.get("arch") is not None:
                    steps["arch"] = select_combo(f.inp_arch, case["arch"])
                steps["model"] = select_combo(f.inp_model, case["model"])
                (f.inpaint_prompt if kind == "inpaint" else f.outpaint_prompt).setPlainText(case["prompt"])
                f.inp_seed_rand.setChecked(False)
                f.inp_seed.setValue(int(case.get("seed", 1)))
                f.strength.setValue(int(round(float(case.get("fidelity", 0.0)) * 100)))   # «Fedeltà all'originale»
                steps["fidelity"] = f.strength.value() / 100.0
                if case.get("steps"):
                    f.inp_steps.setValue(int(case["steps"]))
                if case.get("lora"):
                    steps["lora"] = select_combo(f.inp_lora, case["lora"])
            else:
                if case.get("layer"):                                 # Integra fonde una foto incollata sul canvas (+ Img)
                    steps["layer_ok"] = bool(cv.add_layer(case["layer"]))
                steps["engine"] = select_combo(f.integrate_engine, case["engine"])
                if case.get("model"):
                    steps["model"] = select_combo(f.integrate_model, case["model"])
                f.integrate_prompt.setPlainText(case["prompt"])
                f.int_seed_rand.setChecked(False)
                f.int_seed.setValue(int(case.get("seed", 1)))
            for k, v in (case.get("widgets") or {}).items():
                set_widget(resolve(w, k), v)
            cfg = f.get_inpaint_config()
            steps["cfg_gui"] = main_params({k: cfg[k] for k in cfg if k not in ("prompt_raw",)})
            steps["canvas"] = {"mask": bool(cv.has_mask()), "outpaint_area": bool(cv.has_outpaint_area()),
                               "size": [int(cv.aspect_w), int(cv.aspect_h)]}
            key = {"inpaint": "f1.inpaint.btn", "outpaint": "f1.outpaint.btn", "integrate": "f1.integrate.btn"}[kind]
            return self.find_button(f, key)
        btn = self.gui(setup, 90)
        self.gui(lambda: btn.click(), 30)
        self.wait_job(float(case.get("timeout", 900)))
        res = {"widget_state": steps}
        q = [c for c in self.rec.calls if c["via"] == "queue"]
        mis = []
        if q:
            p = q[0]["params"]
            res["handler"] = f"{q[0]['worker']}:{q[0].get('module')}"
            mname = os.path.basename(str((steps.get("cfg_gui") or {}).get("unet") or "")).lower()
            if kind != "integrate" and mname and mname not in json.dumps(p).lower():
                mis.append(f"modello scelto {mname!r} assente dai parametri inviati al backend")
            if "seed" in case and p.get("seed") not in (None, int(case["seed"])):
                mis.append(f"seed: impostato {case['seed']} ma il backend riceve {p.get('seed')}")
            if kind == "inpaint" and "fidelity" in steps and p.get("denoise") is not None:      # punto 13: denoise = 1 − slider
                if abs(float(p["denoise"]) - (1.0 - steps["fidelity"])) > 0.011:
                    mis.append(f"fedeltà {steps['fidelity']:.2f}: il backend dovrebbe ricevere denoise {1 - steps['fidelity']:.2f}, riceve {p['denoise']}")
            if not (p.get("init_image") or p.get("image_path") or p.get("ref_images") or p.get("images")):
                mis.append("nessuna immagine di partenza nei parametri del backend")
            if kind == "inpaint" and case.get("mask", True) and not (p.get("mask") or p.get("mask_path")) \
                    and "kontext" not in json.dumps(p).lower():
                mis.append("inpaint con maschera ma nessuna maschera inviata")
            for k, v in (case.get("expect_params") or {}).items():
                if p.get(k) != v:
                    mis.append(f"parametro atteso {k}={v!r}, inviato {p.get(k)!r}")
        res["mismatch"] = mis
        return res

    def inventory(self) -> dict:
        """Le voci REALI dei menu dell'app (arch x modelli x LoRA): i casi si generano dall'inventario, non da liste a mano."""
        def lor(panel):
            return [{"file": r["entry"].filename, "name": r["entry"].name,
                     "triggers": list(getattr(r["entry"], "triggers", []) or [])} for r in panel._rows.values()]

        def inv():
            w = self.w
            f, vt = w.frame1_tab, w.video_tab
            out = {"create": {}, "video": {}, "edit": {}}
            f.img_nsfw.setCurrentText("NSFW")
            vt.nsfw.setCurrentText("NSFW")
            for i in range(f.gen_arch.count()):
                f.gen_arch.setCurrentIndex(i)
                f.gen_lora_tab.refresh()
                out["create"][f.gen_arch.itemText(i)] = {"models": combo_items(f.gen_model), "loras": lor(f.gen_lora_tab)}
            for i in range(vt.video_arch.count()):
                vt.video_arch.setCurrentIndex(i)
                vt.lora.refresh()
                out["video"][str(vt.video_arch.itemData(i))] = {"label": vt.video_arch.itemText(i),
                                                                "models": combo_items(vt.model), "loras": lor(vt.lora)}
            for i in range(f.inp_arch.count()):
                f.inp_arch.setCurrentIndex(i)
                out["edit"]["inpaint:" + str(f.inp_arch.itemData(i))] = {"label": f.inp_arch.itemText(i), "models": combo_items(f.inp_model)}
            f.inp_arch.setCurrentIndex(0)
            out["edit"]["integrate_engines"] = {}
            for i in range(f.integrate_engine.count()):
                f.integrate_engine.setCurrentIndex(i)
                out["edit"]["integrate_engines"][f.integrate_engine.itemText(i)] = {"models": combo_items(f.integrate_model)}
            return out
        return self.gui(inv, 120)

    def case_compose(self, case):
        """Modifica -> Componi foto multiple: lista di foto, prompt con @image_N, motore/modello, seed/passi, bottone VERO."""
        w = self.w
        steps = {}

        def setup():
            f = w.frame1_tab
            f.sub.setCurrentIndex(1)
            cv = f.canvas
            cv.load_image(case["canvas_image"], "canvas") if case.get("canvas_image") else None
            if case.get("canvas_wh"):
                cv.set_aspect(int(case["canvas_wh"][0]), int(case["canvas_wh"][1]))
            f.compose_list.clear()
            for pth in case["images"]:
                f.compose_list.addItem(pth)
            steps["engine"] = select_combo(f.integrate_engine, case["engine"])
            if case.get("model"):
                steps["model"] = select_combo(f.integrate_model, case["model"])
            f.compose_prompt.setPlainText(case["prompt"])
            f.compose_seed_rand.setChecked(False)
            f.compose_seed.setValue(int(case.get("seed", 1)))
            if case.get("steps"):
                f.compose_steps.setValue(int(case["steps"]))
            steps["canvas"] = [int(cv.aspect_w), int(cv.aspect_h)]
            steps["cfg_gui"] = main_params({k: v for k, v in f.get_compose_config().items() if k != "images"})
            return self.find_button(f, "f1.compose.btn")
        btn = self.gui(setup, 90)
        self.gui(lambda: btn.click(), 30)
        self.wait_job(float(case.get("timeout", 900)))
        res = {"widget_state": steps}
        q = [c for c in self.rec.calls if c["via"] == "queue"]
        mis = []
        if q:
            p = q[0]["params"]
            res["handler"] = f"{q[0]['worker']}:{q[0].get('module')}"
            cw, ch = steps["canvas"]
            if p.get("width") and p.get("height") and (abs(int(p["width"]) - cw) > 16 or abs(int(p["height"]) - ch) > 16):
                mis.append(f"misura: il canvas e {cw}x{ch} ma il backend riceve {p.get('width')}x{p.get('height')}")
            if "seed" in case and p.get("seed") not in (None, int(case["seed"])):
                mis.append(f"seed: impostato {case['seed']} ma il backend riceve {p.get('seed')}")
            refs = p.get("ref_images") or p.get("images") or []
            stitch = (p.get("arch") == "kontext")                  # Kontext: le foto sono affiancate in UN'immagine (image_path)
            if not stitch and len(refs) < min(2, len(case["images"])):
                mis.append(f"riferimenti: attese {len(case['images'])} foto, il backend ne riceve {len(refs)}")
            if "@image" in str(p.get("prompt", "")):
                mis.append("il prompt inviato contiene ancora @image_N (non tradotto nella sintassi del motore)")
            for k, v in (case.get("expect_params") or {}).items():
                if p.get(k) != v:
                    mis.append(f"parametro atteso {k}={v!r}, inviato {p.get(k)!r}")
        res["mismatch"] = mis
        return res

    def case_segment(self, case):
        """Selezione col clic (SAM2) o col lazo (l'AI seleziona l'oggetto contenuto): la maschera deve coprire l'oggetto INTERO."""
        w = self.w
        steps = {}

        def setup():
            from PySide6.QtCore import QPointF
            f = w.frame1_tab
            f.sub.setCurrentIndex(1)
            cv = f.canvas
            cv.load_image(case["image"], "canvas")
            x, y = float(case["point"][0]) * cv.aspect_w, float(case["point"][1]) * cv.aspect_h      # coordinate relative 0..1
            steps["canvas"] = [int(cv.aspect_w), int(cv.aspect_h)]
            if case["kind"] == "point":
                cv.sam_point_at(QPointF(x, y))
            else:                                                     # lazo: poligono attorno al punto
                r = float(case.get("radius", 0.2)) * min(cv.aspect_w, cv.aspect_h)
                cv._lasso_pts = [QPointF(x - r, y - r), QPointF(x + r, y - r), QPointF(x + r, y + r), QPointF(x - r, y + r)]
                cv.lasso_finish()
        self.gui(setup, 60)
        self.wait_job(float(case.get("timeout", 300)))
        res = {"widget_state": steps}
        mis = []
        outs = [f for f in self.rec.finished if f.get("outputs")]
        if self.dry:
            res["mismatch"] = []
            return res
        if outs:
            from PIL import Image
            import numpy as np
            m = np.asarray(Image.open(outs[-1]["outputs"][0]).convert("L")) > 127
            pct = 100.0 * m.mean()
            res["mask_area_pct"] = round(pct, 2)
            if pct < float(case.get("min_area_pct", 3.0)):
                mis.append(f"la maschera copre solo il {pct:.1f}% della foto: sembra una macchia, non l'oggetto intero")
        res["mismatch"] = mis
        return res

    def case_state(self, case):
        """Lettura dei widget indicati (nessuna azione): serve a verificare cosa mostra la GUI."""
        def rd():
            return {p: widget_value(resolve(self.w, p)) if not p.endswith("?menu") else combo_items(resolve(self.w, p[:-5]))
                    for p in case.get("paths", [])}
        self.dry = True
        return {"stato": self.gui(rd, 30), "mismatch": []}

    # ── server ────────────────────────────────────────────────────────────────────────────────────────
    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", PORT))
        srv.listen(1)
        print(f"[test-driver] in ascolto su 127.0.0.1:{PORT}", flush=True)
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        f = conn.makefile("rwb")
        for line in f:
            try:
                msg = json.loads(line.decode("utf-8"))
                cmd = msg.get("cmd")
                if cmd == "ping":
                    out = {"ok": True, "pid": os.getpid(), "root": os.getcwd()}
                elif cmd == "ready":
                    from core import native_driver
                    out = {"ok": True, "server_up": bool(native_driver.is_up()), "jobs_busy": bool(self.gui(lambda: self.w.jobs.is_busy(), 20))}
                elif cmd == "run_case":
                    out = self.run_case(msg["case"])
                elif cmd == "inventory":
                    out = {"ok": True, "inventory": self.inventory()}
                elif cmd == "state":
                    out = self.case_state({"paths": msg.get("paths", [])})
                elif cmd == "shutdown":
                    f.write((json.dumps({"ok": True}) + "\n").encode()); f.flush()
                    self.gui(lambda: (self.w.close(), __import__("PySide6.QtWidgets", fromlist=["QApplication"]).QApplication.quit()), 20)
                    return
                else:
                    out = {"ok": False, "error": f"comando sconosciuto {cmd}"}
            except Exception:                                # noqa: BLE001
                out = {"ok": False, "error": traceback.format_exc()[-1500:]}
            f.write((json.dumps(out, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
            f.flush()

    def start(self):
        self.sampler.start()
        threading.Thread(target=self.serve, daemon=True).start()


def attach(win):
    """Chiamato da app.py (solo con LTX_TEST_DRIVER=1) dopo MainWindow.show()."""
    d = Driver(win)
    d.start()
    win._test_driver = d
    return d
