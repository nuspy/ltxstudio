"""Test LIVE di FLUX.3 Image Klein DALLA GUI (MainWindow vera con l'InferenceServer :8189 reale, widget veri, segnali dei bottoni veri;
generazione reale su GPU). UN test per processo, sotto il guardiano:

  python tests\\live_guard.py --min-commit-gb 20 --max-proc-gb 70 --kill-servers-on-exit -- python tests\\gui_regression\\live_flux3.py L1

Prove (id come in task13_TESTS.md): L1 t2i senza upsampler · L3 determinismo (stesso seed, seed diverso) · L4 t2i con upsampler (Sakura)
· L6 edit singolo (foto del canvas, motore FLUX.3 nel pannello di modifica senza maschera) · L7 multi-riferimento (Componi, 3 foto)
· L8 riquadri di layout · L9 riquadri di edit · L10_2K / L10_4K · L11 «inpaint» (Integra con layer+maschera → composito nostro)
· L13 pesi mancanti (TE ufficiale non scaricato → avviso, nessun job).

Per ogni prova scrive in E:\\ai\\ltxstudio\\output\\ (piatto): DIAG_flux3_<id>.png (+ .log del worker con i tempi per fase) e DIAG_flux3_<id>.json
(secondi, VRAM usata di sistema da nvidia-smi: base e picco). Nessun torch.cuda nel processo della GUI.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.app_env import harness as AH                       # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
MAIN_OUT = Path(r"E:\ai\ltxstudio\output")
CAST = MAIN_OUT / "film_studio" / "project_22" / "cast"
SEED = 7
NEG_TIMEOUT = 1500                                            # s per prova (il guardiano ne ha uno suo)

P_LIGHTHOUSE = ("A tall glass lighthouse standing on a cliff of white salt crystals, dark storm clouds breaking at dawn, "
                "wide-angle photograph, wet salt glittering")
P_SHIP = "a ship inside a cobalt-blue glass bottle"
P_NIGHT = "turn the scene into a rainy night with glowing street lamps and wet reflections on the ground"
P_MULTI = ("The man from the first image and the woman from the second image sit together at a candle-lit table in a tiny Lisbon "
           "tavern at night, the person from the third image stands behind them smiling, warm photographic light")
P_LAYOUT = ("A poster on a windswept coast: a red lighthouse <faro_1> on the right, a small yellow sailboat <nave_1> at the lower "
            "left, and the word \"SALT\" in large white block letters <En_Text_1> across the top, flat illustration style")
P_HAT = "a man wearing a bright green straw hat <hat_1>, same photo otherwise"


class Vram(threading.Thread):
    """Campiona la VRAM usata del sistema (nvidia-smi, nessun contesto CUDA qui) ogni secondo: base e picco."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_ev = threading.Event()
        self.peak = 0
        self.base = AH.vram_used_mb()

    def run(self):
        while not self.stop_ev.is_set():
            self.peak = max(self.peak, AH.vram_used_mb())
            self.stop_ev.wait(1.0)


def _wait(pred, timeout_s: float) -> bool:
    app = AH.app()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.25)
    return False


def hook(w):
    """Cattura l'esito dei percorsi di completamento (generazione / modifica) senza dialog."""
    got = {}
    w._show_inpaint_result = lambda out, *a, **k: got.setdefault("out", out)
    for name in ("_on_native_image_done", "_on_native_inpaint_done"):
        orig = getattr(type(w), name)

        def _done(res, _w=w, _orig=orig):
            got["res"] = res
            if res.get("status") == "success" and res.get("output"):
                got.setdefault("out", res["output"][0])
            try:
                _orig(_w, res)
            except Exception as e:                              # noqa: BLE001
                got["after_error"] = str(e)
        setattr(w, name, _done)
    return got


def finish(label: str, got: dict, t0: float, vr: Vram) -> dict:
    ok = _wait(lambda: "res" in got, NEG_TIMEOUT)
    secs = time.time() - t0
    r = got.get("res") or {}
    info = {"id": label, "ok": False, "seconds": round(secs, 1), "vram_base_mb": vr.base, "vram_peak_mb": vr.peak,
            "status": r.get("status"), "error": str(r.get("error", ""))[-2500:] if r.get("status") != "success" else ""}
    out = got.get("out")
    if r.get("status") == "success" and out and os.path.isfile(out):
        dst = MAIN_OUT / f"DIAG_flux3_{label}.png"
        shutil.copyfile(out, dst)
        info["image"] = str(dst)
        if os.path.isfile(out + ".log"):
            shutil.copyfile(out + ".log", MAIN_OUT / f"DIAG_flux3_{label}.log")
            info["phases"] = [ln for ln in Path(out + ".log").read_text("utf-8", "replace").splitlines()
                              if ln.startswith(("[flux3", "[flux3-pup"))]
        info["ok"] = True
    print(f"[live] {label}: {secs:.0f}s ok={info['ok']} stato={info['status']} base={vr.base}MB picco={vr.peak}MB")
    for ln in info.get("phases", [])[-14:]:
        print("   ", ln[:200])
    if info["error"]:
        print("[live] ERRORE:", info["error"])
    return info


def setup_create(w, *, pup: bool, wh=(1024, 1024), seed=SEED, prompt=P_LIGHTHOUSE, te: str = ""):
    f = w.frame1_tab
    f.sub.setCurrentIndex(0)
    f.img_nsfw.setCurrentText("NSFW")                          # voci 18+ ammesse: compare il combo TE con Krea2 (automatico)
    f.gen_arch.setCurrentText("FLUX.3 Klein")
    f.gen_model.setCurrentIndex(f.gen_model.findData("flux3-dit-fp8r"))
    f._fill_flux3_te()
    f.f3_te.setCurrentIndex(max(0, f.f3_te.findData(te)))
    f.f3_pup.setChecked(pup)
    f.gen_refs.clear()
    f._set_flux3_boxes("f3", [])
    w.video_tab.width.setMaximum(4096); w.video_tab.height.setMaximum(4096)
    w.video_tab.width.setValue(wh[0]); w.video_tab.height.setValue(wh[1])
    f.gen_seed_rand.setChecked(False); f.gen_seed.setValue(seed)
    f.gen_prompt.setPlainText(prompt)
    return f


def run_create(w, vr, label, **kw):
    f = setup_create(w, **kw)
    got = hook(w); t0 = time.time()
    f.gen_btn.click()                                           # bottone vero
    return finish(label, got, t0, vr)


def setup_edit_engine(w, *, pup: bool):
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    f.integrate_engine.setCurrentIndex(4)                       # FLUX.3 Klein
    f._fill_flux3_te()
    f.f3e_te.setCurrentIndex(0)
    f.f3e_pup.setChecked(pup)
    f._set_flux3_boxes("f3e", [])
    f.int_seed_rand.setChecked(False); f.int_seed.setValue(SEED)
    return f


def t_L1(w, vr):
    return run_create(w, vr, "L1", pup=False)


def t_L3(w, vr):
    a = run_create(w, vr, "L3_seed7", pup=False)
    b = run_create(w, vr, "L3_seed8", pup=False, seed=8)
    res = {"id": "L3", "ok": False, "runs": [a, b]}
    if a["ok"] and b["ok"]:
        from PIL import Image, ImageChops
        import numpy as np
        l1 = MAIN_OUT / "DIAG_flux3_L1.png"
        same_l1 = None
        if l1.is_file():
            same_l1 = int(np.abs(np.asarray(Image.open(l1).convert("RGB"), int) - np.asarray(Image.open(a["image"]).convert("RGB"), int)).max())
        diff_seed = int(np.abs(np.asarray(Image.open(a["image"]).convert("RGB"), int) - np.asarray(Image.open(b["image"]).convert("RGB"), int)).max())
        res.update(maxdiff_seed7_vs_L1=same_l1, maxdiff_seed7_vs_seed8=diff_seed, ok=(same_l1 == 0 or same_l1 is None) and diff_seed > 0)
        print(f"[live] L3: seed7 vs L1 maxdiff={same_l1} · seed7 vs seed8 maxdiff={diff_seed}")
    return res


def t_L4(w, vr):
    return run_create(w, vr, "L4", pup=True, prompt=P_SHIP)


def _canvas_photo(w, path):
    f = w.frame1_tab
    f.sub.setCurrentIndex(1)
    f.canvas.load_image(str(path), fit_mode="canvas")
    return f


def t_L6(w, vr):
    f = _canvas_photo(w, CAST / "Dorian.png")
    setup_edit_engine(w, pup=False)
    f.inpaint_prompt.setPlainText(P_NIGHT)
    f.inp_seed_rand.setChecked(False); f.inp_seed.setValue(SEED)
    got = hook(w); t0 = time.time()
    f.canvas.request_inpaint.emit()                              # senza maschera = modifica dell'intera foto col motore scelto
    return finish("L6", got, t0, vr)


def t_L7(w, vr):
    f = w.frame1_tab
    f.sub.setCurrentIndex(1)
    setup_edit_engine(w, pup=False)
    f.compose_list.clear()
    for n in ("Dorian", "Kaira", "Vesna"):
        f.compose_list.addItem(str(CAST / f"{n}.png"))
    f.compose_prompt.setPlainText(P_MULTI)
    got = hook(w); t0 = time.time()
    f.request_compose.emit()
    return finish("L7", got, t0, vr)


def t_L8(w, vr):
    boxes = [{"id": "faro_1", "desc": "a tall red lighthouse with a white lantern room", "bbox": [0.18, 0.62, 0.92, 0.82]},
             {"id": "nave_1", "desc": "a small yellow sailboat", "bbox": [0.62, 0.05, 0.88, 0.38]},
             {"id": "En_Text_1", "desc": "the word \"SALT\" in huge white block capital letters", "bbox": [0.04, 0.10, 0.22, 0.90]}]
    f = setup_create(w, pup=False, prompt=P_LAYOUT)
    f._set_flux3_boxes("f3", boxes)
    got = hook(w); t0 = time.time()
    f.gen_btn.click()
    r = finish("L8", got, t0, vr)
    r["boxes"] = boxes
    return r


def t_L9(w, vr):
    f = _canvas_photo(w, CAST / "Dorian.png")
    setup_edit_engine(w, pup=False)
    f._set_flux3_boxes("f3e", [{"id": "hat_1", "desc": "a bright green straw hat", "mode": "new", "tgt_bbox": [0.0, 0.25, 0.3, 0.75]}])
    f.inpaint_prompt.setPlainText(P_HAT)
    f.inp_seed_rand.setChecked(False); f.inp_seed.setValue(SEED)
    got = hook(w); t0 = time.time()
    f.canvas.request_inpaint.emit()
    r = finish("L9", got, t0, vr)
    f._set_flux3_boxes("f3e", [])
    return r


def _t10(w, vr, side, label):
    return run_create(w, vr, label, pup=False, wh=(side, side),
                      prompt="A vast salt flat at dusk with a single glass pyramid reflecting a violet sky, ultra detailed photograph")


def t_L10_2K(w, vr):
    return _t10(w, vr, 2048, "L10_2K")


def t_L10_4K(w, vr):
    return _t10(w, vr, 4096, "L10_4K")


def t_L11(w, vr):
    """Integra: foto del canvas + un LAYER con maschera → composito e anello di edit → modifica per istruzione + composito nostro."""
    from PIL import Image, ImageDraw
    f = w.frame1_tab
    f.sub.setCurrentIndex(1)
    f.canvas.load_image(str(CAST / "Dorian.png"), fit_mode="canvas")
    layer = MAIN_OUT / "DIAG_flux3_L11_layer.png"
    Image.new("RGB", (400, 400), (200, 30, 30)).save(layer)                 # oggetto da integrare: un quadrato rosso
    msk = MAIN_OUT / "DIAG_flux3_L11_layer_mask.png"
    m = Image.new("L", (400, 400), 0)
    ImageDraw.Draw(m).ellipse((20, 20, 380, 380), fill=255)
    m.save(msk)
    f.canvas.add_layer(str(layer), str(msk))
    setup_edit_engine(w, pup=False)
    f.integrate_prompt.setPlainText("blend the red round object naturally into the scene, matching light and shadows")
    got = hook(w); t0 = time.time()
    f.request_integrate.emit()
    r = finish("L11", got, t0, vr)
    ex = getattr(w, "_inpaint_before", None)
    r["composito_prima"] = ex
    return r


def t_L13(w, vr):
    """TE ufficiale non scaricato: nessun job, avviso con nome e GB."""
    f = setup_create(w, pup=False, te="official")
    seen = {}
    orig = w._missing_weights_notice

    def spy(ids, feature):
        ok = orig(ids, feature)
        seen["ids"], seen["ok"] = list(ids), ok
        return ok
    w._missing_weights_notice = spy
    n0 = len(w._enq_log) if hasattr(w, "_enq_log") else 0
    got = hook(w)
    f.gen_btn.click()
    _wait(lambda: "res" in got, 8)
    ok = seen.get("ok") is False and "flux3-te" in seen.get("ids", []) and "res" not in got
    print(f"[live] L13: ids={seen.get('ids')} notice_ok_false={seen.get('ok')} job_partito={'res' in got}")
    return {"id": "L13", "ok": ok, "ids": seen.get("ids"), "job_started": "res" in got}


TESTS = {k[2:]: v for k, v in globals().items() if k.startswith("t_L")}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TESTS:
        print("uso: live_flux3.py <" + "|".join(TESTS) + ">")
        return 2
    tid = sys.argv[1]
    MAIN_OUT.mkdir(exist_ok=True)
    w = AH.boot_full_app(native_timeout=300)
    snap = AH.snapshot("avvio")
    AH.print_snapshot(snap)
    vr = Vram()
    vr.start()
    info = {}
    try:
        info = TESTS[tid](w, vr)
    finally:
        vr.stop_ev.set()
        end = AH.teardown(w)
        info["vram_after_teardown_mb"] = end.get("vram_used_mb")
        (MAIN_OUT / f"DIAG_flux3_{tid}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), "utf-8")
    print(f"[live] FINE {tid}: ok={info.get('ok')}")
    return 0 if info.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
