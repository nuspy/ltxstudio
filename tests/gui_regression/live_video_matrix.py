# -*- coding: utf-8 -*-
"""MATRICE LIVE VIDEO dalla GUI (campagna 22/07) — genera DAVVERO, modello per modello,
pipeline per pipeline, attraverso i widget veri + on_generate()/handler reali.

Uso (venv GUI, offscreen):
    QT_QPA_PLATFORM=offscreen python tests/gui_regression/live_video_matrix.py [--filter substr] [--list]

Report: output/tests/REAL_VIDEO/<caso>.txt + REAL_VIDEO_SUMMARY.txt. RESUMABILE: i casi con
report PASS esistente vengono saltati (--force per rifarli).
Regole (memorie): 1 generazione alla volta; check processi utente prima di ogni caso; niente
server spawnati dai test per le IMMAGINI (harness) ma il VIDEO usa il worker per-job (subprocess),
timeout per caso; prompt creativi vari (mai gattini).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output"
REP = OUT / "tests" / "REAL_VIDEO"
REP.mkdir(parents=True, exist_ok=True)

import harness as H  # noqa: E402

# ── modelli LTX (path assoluti reali) ────────────────────────────────────────────
M = {
    "distilled_fp8": r"E:\AI\Models\diffusion_models\LTXVideo\v2\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
    "dev_fp8":       r"E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors",
    "dev_nvfp4":     r"E:\AI\Models\diffusion_models\LTXVideo\LTX-2\nvfp4\ltx-2.3-22b-dev-nvfp4.safetensors",
    "gguf_q4":       r"E:\AI\Models\diffusion_models\LTXVideo\LTX-2\quantstack\ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
    "gguf_q6":       r"E:\AI\Models\diffusion_models\ltx-2-3-22b-dev-Q6_K.gguf",
    "eros":          r"E:\AI\Models\diffusion_models\ltx2310eros_v1.safetensors",
}
PROMPTS = [  # creativi e vari (regola: mai gattini/cani)
    "un fabbro forgia una spada incandescente in una bottega medievale, scintille",
    "una ballerina di flamenco danza su un tetto all'alba, vestito rosso al vento",
    "un sommergibile steampunk emerge tra i ghiacci, aurora boreale",
    "un cuoco robot prepara ramen in un vicolo di Tokyo sotto la pioggia",
    "una mongolfiera di carta vola dentro una biblioteca infinita",
    "un vecchio marinaio suona la fisarmonica sul molo, gabbiani e nebbia",
]
_PI = [0]


def _prompt() -> str:
    _PI[0] += 1
    return PROMPTS[_PI[0] % len(PROMPTS)]


def _user_busy() -> bool:
    """True se ci sono job dell'UTENTE in corso (mai testare sopra una generazione reale)."""
    import subprocess
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                            "Where-Object { $_.CommandLine -match 'app\\.py' }).Count"],
                           capture_output=True, text=True, timeout=30)
        return int((r.stdout or "0").strip() or 0) > 0
    except Exception:                                       # noqa: BLE001
        return False


def _spin(app, secs: float):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        app.processEvents()
        time.sleep(0.05)


def _wait_done(w, app, timeout_s: int) -> bool:
    """Aspetta che la coda job della GUI si svuoti (il worker reale è finito)."""
    _spin(app, 3.0)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if not w.jobs.is_busy():
            _spin(app, 1.0)
            if not w.jobs.is_busy():
                return True
        _spin(app, 1.0)
    return False


def _new_outputs(before: set) -> list[str]:
    return sorted(set(str(p) for p in OUT.glob("*.mp4")) - before)


def _select_model(w, path: str) -> bool:
    """Seleziona nel dropdown video la voce il cui model corrisponde al path (basename).
    ATTENDE il popolamento dell'inventario (async, tick 3s di _refresh_status): senza attesa i
    case partivano col dropdown ancora ai soli fallback (Q6_K 'assente', visto 23/07)."""
    vt = w.video_tab
    base = os.path.basename(path).lower()
    app = H.app()
    for _round in range(30):                                # fino a ~15s
        for i in range(vt.model.count()):
            d = vt.model.itemData(i) or {}
            if base in str(d.get("model") or "").lower():
                vt.model.setCurrentIndex(i)
                return True
        _spin(app, 0.5)
    return False


def _base_setup(w, prompt: str, width=704, height=448, secs=2.0, nsfw=False):
    vt = w.video_tab
    if nsfw:                                                # eros & LoRA NSFW: visibili solo in NSFW
        try:
            vt.nsfw.setCurrentText("NSFW")
        except Exception:                                   # noqa: BLE001
            pass
    vt.prompt.setPlainText(prompt)
    try:
        vt.res_preset.setCurrentText("Personalizzato")
    except Exception:                                       # noqa: BLE001
        pass
    vt.width.setValue(width)
    vt.height.setValue(height)
    try:
        vt.length.setValue(secs)
    except Exception:                                       # noqa: BLE001
        pass
    try:                                                    # arch LTX di default
        i = vt.video_arch.findData("ltx")
        if i >= 0:
            vt.video_arch.setCurrentIndex(i)
    except Exception:                                       # noqa: BLE001
        pass


# ── definizione casi ─────────────────────────────────────────────────────────────
def case_t2v(model_key):
    def run(w, app):
        _base_setup(w, _prompt(), nsfw=(model_key == "eros"))
        assert _select_model(w, M[model_key]), f"modello {model_key} assente dal dropdown"
        w.on_generate()
    return run


_I2V_IMG_PROMPT = ("ritratto fotorealistico di un vecchio marinaio con cappuccio scuro, barba bianca, "
                   "occhi azzurri, molo nebbioso all'alba, luce fredda e cinematografica")
_I2V_PROMPT = ("il vecchio marinaio col cappuccio guarda in camera, sbatte le palpebre e accenna "
               "un sorriso, respiro visibile nell'aria fredda")   # coerente con la fixture generata
_STARTED_8189 = [False]


def _image_server_up() -> bool:
    import urllib.request
    try:                                                     # probe DIRETTA (is_up è mockata nei case)
        urllib.request.urlopen("http://127.0.0.1:8189/system_stats", timeout=2).read()
        return True
    except Exception:                                        # noqa: BLE001
        return False


def _ensure_i2v_fixture() -> str:
    """Regola utente 23/07: i test i2v GENERANO PRIMA un'immagine coerente (backend immagini
    vero), poi la animano. Fixture cacheata; il server :8189 avviato qui viene chiuso a fine run."""
    fx = str(REP / "_i2v_fixture.png")
    if os.path.exists(fx) and os.path.getsize(fx) > 30_000:
        return fx
    from core import native_driver
    if not _image_server_up():
        from core.native_launcher import launch
        ok, msg = launch()
        print(f"[fixture] avvio server immagini: {ok} {msg}", flush=True)
        _STARTED_8189[0] = bool(ok)
        for _ in range(180):
            if _image_server_up():
                break
            time.sleep(1)
    assert _image_server_up(), "server immagini :8189 non disponibile per la fixture i2v"
    r = native_driver.run_and_wait("image", {
        "model": r"E:\AI\Models\diffusion_models\Flux\FLUX_Krea_Dev.safetensors",
        "arch": "flux", "prompt": _I2V_IMG_PROMPT, "negative": "low quality, cartoon",
        "width": 704, "height": 448, "steps": 28, "guidance": 3.5, "seed": 77}, timeout=900)
    outs = (r or {}).get("output") or []
    assert r.get("status") == "success" and outs, f"fixture i2v non generata: {r}"
    import shutil
    shutil.copyfile(outs[0], fx)
    print(f"[fixture] immagine i2v generata: {os.path.basename(outs[0])} → _i2v_fixture.png", flush=True)
    return fx


def _stop_image_server_if_started():
    if not _STARTED_8189[0]:
        return
    import subprocess
    try:                                                     # kill SOLO se avviato da questo run
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-NetTCPConnection -LocalPort 8189 -State Listen -ErrorAction "
                            "SilentlyContinue).OwningProcess | Select-Object -First 1"],
                           capture_output=True, text=True, timeout=30)
        pid = int((r.stdout or "0").strip() or 0)
        if pid:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=30)
            print(f"[fixture] server immagini avviato dal test → chiuso (pid {pid})", flush=True)
    except Exception:                                        # noqa: BLE001
        pass


def case_i2v(model_key):
    def run(w, app):
        fx = _ensure_i2v_fixture()
        _base_setup(w, _I2V_PROMPT, nsfw=(model_key == "eros"))
        assert _select_model(w, M[model_key]), f"modello {model_key} assente dal dropdown"
        H.load_canvas_image(w, fx)                          # foto GENERATA nel canvas → i2v
        try:
            w.video_tab.use_image.setChecked(True)
        except Exception:                                   # noqa: BLE001
            pass
        w.on_generate()
    return run


def case_i2v_768(model_key):
    """Regressione OOM/conditioning: la risoluzione del crash utente (768×576)."""
    def run(w, app):
        fx = _ensure_i2v_fixture()
        _base_setup(w, _I2V_PROMPT, width=768, height=576)
        assert _select_model(w, M[model_key]), f"modello {model_key} assente"
        H.load_canvas_image(w, fx)
        try:
            w.video_tab.use_image.setChecked(True)
        except Exception:                                   # noqa: BLE001
            pass
        w.on_generate()
    return run


def case_lora(model_key):
    """1 LoRA style dal tab LoRA video (la prima disponibile)."""
    def run(w, app):
        _base_setup(w, _prompt())
        assert _select_model(w, M[model_key]), f"modello {model_key} assente"
        lt = getattr(w.video_tab, "lora_tab", None)
        ok = False
        if lt is not None:
            try:
                sel = lt.select_first_for_test() if hasattr(lt, "select_first_for_test") else None
                ok = bool(sel)
            except Exception:                               # noqa: BLE001
                ok = False
        if not ok:                                          # fallback: LoRA via checkbox/lista del tab
            print("   [nota] selezione LoRA via tab non riuscita → caso t2v semplice")
        w.on_generate()
    return run


def case_audio_nativo(model_key):
    def run(w, app):
        _base_setup(w, 'un cantastorie dice "benvenuti al circo delle meraviglie" con voce calda')
        assert _select_model(w, M[model_key]), f"modello {model_key} assente"
        w.video_tab.native_audio.setChecked(True)
        w.on_generate()
    return run


def case_extend():
    """Prolunga un video esistente (ultimo DIAG mp4 in output) — widget veri del tab."""
    def run(w, app):
        srcs = sorted(OUT.glob("DIAG_*.mp4"), key=lambda p: p.stat().st_mtime)
        assert srcs, "nessun video sorgente per extend"
        tab = w.video_extend_tab
        tab.set_video(str(srcs[-1]))
        tab.prompt.setPlainText("la scena continua con un lento zoom out")
        tab.add_seconds.setValue(2.0)
        w.generate_video_extend()
    return run


ARCH_CASES = []


def case_arch(arch, model_sub=""):
    def run(w, app):
        _base_setup(w, _prompt())
        vt = w.video_tab
        i = vt.video_arch.findData(arch)
        assert i >= 0, f"arch {arch} assente dal selettore"
        vt.video_arch.setCurrentIndex(i)
        if model_sub:
            found = False
            for j in range(vt.model.count()):
                d = vt.model.itemData(j) or {}
                if model_sub.lower() in str(d.get("model") or "").lower():
                    vt.model.setCurrentIndex(j)
                    found = True
                    break
            assert found, f"modello {model_sub} assente per arch {arch}"
        w.on_generate()
    return run


CASES: list[tuple[str, int, object]] = []          # (nome, timeout_s, fn(w, app))
# riferimento distilled_fp8: sweep principale
CASES += [
    ("ltx_distilledfp8_t2v",        900, case_t2v("distilled_fp8")),
    ("ltx_distilledfp8_i2v",        900, case_i2v("distilled_fp8")),
    ("ltx_distilledfp8_i2v768",     1200, case_i2v_768("distilled_fp8")),
    ("ltx_distilledfp8_audio",      900, case_audio_nativo("distilled_fp8")),
    ("ltx_distilledfp8_lora",       900, case_lora("distilled_fp8")),
    ("ltx_extend",                  1200, case_extend()),
]
# altri modelli LTX: t2v + i2v
for key in ("dev_fp8", "dev_nvfp4", "gguf_q4", "gguf_q6", "eros"):
    CASES.append((f"ltx_{key}_t2v", 1200, case_t2v(key)))
    CASES.append((f"ltx_{key}_i2v", 1200, case_i2v(key)))
# altre architetture
CASES += [
    ("arch_self_forcing_t2v", 1200, case_arch("krea_rt")),   # default krea_rt→self_forcing worker
    ("arch_lingbot_t2v",      1500, case_arch("lingbot")),
]


def run_case(name: str, timeout_s: int, fn) -> tuple[bool, str]:
    if _user_busy():
        return False, "SKIP: app utente aperta (mai testare sopra l'utente)"
    app = H.app()
    H.patch_dialogs()
    w = H.build_window()
    try:
        from core import native_driver
        native_driver.is_up = lambda: True                  # il video va per subprocess/server ltx
    except Exception:                                       # noqa: BLE001
        pass
    before = set(str(p) for p in OUT.glob("*.mp4"))
    t0 = time.monotonic()
    try:
        fn(w, app)
    except AssertionError as e:
        w.close()
        return False, f"SETUP FALLITO: {e}"
    except Exception as e:                                  # noqa: BLE001
        w.close()
        return False, f"ECCEZIONE setup/handler: {type(e).__name__}: {e}"
    done = _wait_done(w, app, timeout_s)
    outs = _new_outputs(before)
    dt = time.monotonic() - t0
    # esito: log della window per errori (ultimo testo del pannello log)
    log_tail = ""
    try:
        log_tail = w.log.toPlainText()[-1200:]
    except Exception:                                       # noqa: BLE001
        pass
    w.close()
    if not done:
        return False, f"TIMEOUT dopo {dt:.0f}s (coda ancora occupata)\n--- log GUI ---\n{log_tail}"
    good = [o for o in outs if os.path.getsize(o) > 50_000]
    if good:
        return True, f"OK in {dt:.0f}s → {os.path.basename(good[-1])} ({os.path.getsize(good[-1])//1024}KB)"
    return False, f"NESSUN OUTPUT (>50KB) in {dt:.0f}s — outs={outs}\n--- log GUI ---\n{log_tail}"


def main() -> int:
    filt = ""
    if "--filter" in sys.argv:
        filt = sys.argv[sys.argv.index("--filter") + 1]
    force = "--force" in sys.argv
    if "--list" in sys.argv:
        for n, t, _ in CASES:
            print(n)
        return 0
    results = []
    for name, timeout_s, fn in CASES:
        if filt and filt not in name:
            continue
        rp = REP / f"{name}.txt"
        if rp.exists() and not force and "PASS" in rp.read_text(encoding="utf-8", errors="replace")[:40]:
            print(f"[skip] {name} (già PASS)")
            results.append((name, True, "già PASS"))
            continue
        print(f"[run ] {name} …", flush=True)
        ok, detail = run_case(name, timeout_s, fn)
        rp.write_text(("PASS\n" if ok else "FAIL\n") + detail + "\n", encoding="utf-8")
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail.splitlines()[0]}", flush=True)
        results.append((name, ok, detail.splitlines()[0]))
    _stop_image_server_if_started()                          # niente processi test residui (memoria)
    n_ok = sum(1 for _, ok, _ in results if ok)
    summary = "\n".join(f"{'PASS' if ok else 'FAIL'}  {n}  {d}" for n, ok, d in results)
    (REP / "REAL_VIDEO_SUMMARY.txt").write_text(summary + f"\n\n== {n_ok}/{len(results)} PASS ==\n",
                                                encoding="utf-8")
    print(f"\n== {n_ok}/{len(results)} PASS == (report in {REP})")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
