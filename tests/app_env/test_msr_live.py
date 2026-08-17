"""MSR LIVE — riferimenti multipli su LTX 2.3, pilotando la GUI VERA.

Sequenza: 3 soggetti + 1 sfondo generati davvero con FLUX → messi nel pannello «Riferimenti
multipli» del tab Video → premuto il bottone Genera → il worker costruisce la pseudo-sequenza e
la dà come video-guida IC-LoRA con la LoRA MSR. Verifica MECCANICA (parte, produce l'mp4, la
guida è coerente col layout atteso); la QUALITÀ — se i soggetti si riconoscono nel video — la
giudica l'utente guardando i file.

  <gui_python> tests/app_env/test_msr_live.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as H  # noqa: E402

OUT = r"E:\ai\ltxstudio\output"
IMG_MODEL = "FLUX_Dev.safetensors"

# soggetti volutamente NETTI e diversi fra loro: se l'identità non passa, si vede subito
SUBJECTS = [
    ("sax", "studio portrait of an elderly bearded jazz saxophonist in a purple velvet suit, "
            "plain white background, sharp focus"),
    ("robot", "a small round copper robot with one big glass eye and brass rivets, "
              "plain white background, product photo"),
    ("parrot", "a scarlet macaw parrot perched, wings folded, plain white background, studio light"),
]
BACKGROUND = ("stage", "empty smoky jazz club stage at night, red curtains, warm lanterns, "
                       "cinematic wide shot")


def _gen_image(name: str, prompt: str, seed: int) -> str:
    """Genera e ritorna il path REALE: il backend immagini ignora `output` e usa il proprio
    schema di nomi (native_flux_<seed>.png), restituendolo in `output[0]`. Copiato poi con un
    nome parlante, così i riferimenti del test si riconoscono in output/.

    RIUSA il file se c'è già: caricare FLUX costa ~28GB di commit e i riferimenti non cambiano
    fra un giro e l'altro. Serve anche a poter iterare sull'MSR quando la macchina è carica
    (il commit esaurito dà access violation 0xC0000005 al load dei safetensors)."""
    import shutil
    dest_esistente = os.path.join(OUT, f"DIAG_msr_ref_{name}.png")
    if os.path.exists(dest_esistente) and os.path.getsize(dest_esistente) > 10000:
        print(f"   [riuso] {os.path.basename(dest_esistente)}", flush=True)
        return dest_esistente
    from core import native_driver
    p = {"arch": "flux", "model": IMG_MODEL, "prompt": prompt, "negative": "text, watermark, blurry",
         "width": 768, "height": 768, "steps": 8, "guidance": 3.5, "seed": seed,
         "output": os.path.join(OUT, f"DIAG_msr_ref_{name}.png")}
    r = native_driver.run_and_wait("image", p, timeout=900)
    src = (r.get("output") or [""])[0]
    if r.get("status") != "success" or not src or not os.path.exists(src):
        print(f"   [gen fallita] status={r.get('status')} error={str(r.get('error'))[:200]}", flush=True)
        return ""
    dest = os.path.join(OUT, f"DIAG_msr_ref_{name}.png")
    shutil.copyfile(src, dest)
    return dest


def _genera(w, etichetta: str):
    """Preme il bottone Genera della GUI e attende l'mp4 nuovo. Ritorna il path o None."""
    import time as _t
    from PySide6.QtWidgets import QApplication
    prima = set(Path(OUT).glob("*.mp4"))
    t0 = _t.time()
    w.on_generate()
    print(f"[gen {etichetta}] accodato, attendo…", flush=True)
    deadline = _t.time() + 1800
    while _t.time() < deadline:
        QApplication.processEvents()
        nuovi = [p for p in Path(OUT).glob("*.mp4") if p not in prima
                 and "_msr_guide" not in p.name and p.stat().st_size > 10000]
        if nuovi:
            out = max(nuovi, key=lambda p: p.stat().st_mtime)
            print(f"[out {etichetta}] {out.name} ({out.stat().st_size / 2**20:.1f} MB "
                  f"in {_t.time() - t0:.0f}s)", flush=True)
            return out
        _t.sleep(1.0)
    print(f"[out {etichetta}] NESSUN VIDEO in {_t.time() - t0:.0f}s", flush=True)
    return None


def main() -> int:
    print("=== MSR LIVE (LTX 2.3, riferimenti multipli) ===", flush=True)
    fails = []

    def check(name, cond, extra=""):
        print(("OK   " if cond else "FAIL ") + name + (f" — {extra}" if extra else ""), flush=True)
        if not cond:
            fails.append(name)

    from core import ltx_release as lr
    msr = lr.get("2.3").msr_lora
    check("LoRA MSR 2.3 su disco", os.path.exists(msr), os.path.basename(msr))
    if not os.path.exists(msr):
        return 2

    w = H.boot_full_app()

    # 1) riferimenti REALI (non rettangoli: serve vedere se l'identità passa)
    refs = []
    for i, (name, prompt) in enumerate(SUBJECTS):
        t0 = time.time()
        p = _gen_image(name, prompt, 100 + i)
        print(f"[ref {name}] {'ok' if p else 'FALLITA'} in {time.time() - t0:.0f}s", flush=True)
        if p:
            refs.append(p)
    bg = _gen_image(BACKGROUND[0], BACKGROUND[1], 200)
    check("3 soggetti + sfondo generati", len(refs) == 3 and bool(bg))
    if len(refs) != 3 or not bg:
        H.teardown(w)
        return 2

    # 2) pannello riferimenti della GUI VERA
    vt = w.video_tab
    from core.model_catalog import video_caps
    vt._apply_reference_limits("ltx", video_caps("ltx"))
    panel = vt.reference_panel
    panel.enable.setChecked(True)
    for p in refs:
        check(f"riferimento accettato: {os.path.basename(p)}", panel.add_path(p))
    panel.set_background(bg)
    check("pannello valido (sfondo presente)", panel.validation_errors() == [],
          str(panel.validation_errors()))
    # i token nel prompt: è così che il modello sa a chi ci si riferisce
    for r in panel._rows:
        vt._insert_prompt_token(r.token)
    vt.prompt.setPlainText(
        "<Picture 1> the saxophonist plays on stage while <Picture 2> the little copper robot "
        "dances beside him and <Picture 3> the scarlet parrot flies over them, smoky jazz club, "
        "warm lanterns, cinematic")
    print(f"[prompt] {vt.prompt.toPlainText()[:110]}…", flush=True)

    # 3) parametri video. RISOLUZIONE NON MINUSCOLA: sotto ~700px LTX-2.3 produce poltiglia a
    #    prescindere dai riferimenti (verificato al 1° giro a 512x320). `length` è in SECONDI:
    #    2s a 24fps = 49 frame, così la guida (41) ci sta dentro senza essere troncata.
    vt.width.setValue(704); vt.height.setValue(448)
    vt.fps.setValue(24); vt.length.setValue(2.0)
    vt.sampler_seed.steps.setValue(8)
    vt.sampler_seed.seed.setValue(7)

    # 4) IL BOTTONE (percorso reale della GUI, non una chiamata al servizio)
    produced = _genera(w, "MSR")
    check("video prodotto", produced is not None)

    # 4b) CONTROLLO senza riferimenti, stessi parametri: se anche questo è poltiglia, il problema
    #     sono le impostazioni e non i riferimenti. Senza questo confronto non si può concludere.
    panel.enable.setChecked(False)
    vt.prompt.setPlainText("an elderly bearded saxophonist plays on a smoky jazz club stage, "
                           "warm lanterns, cinematic")
    control = _genera(w, "CONTROLLO senza riferimenti")
    check("controllo prodotto", control is not None)

    # 5) la pseudo-sequenza di guida deve esistere ed avere il layout atteso
    guide = sorted(Path(OUT).glob("*_msr_guide.mp4"), key=lambda p: p.stat().st_mtime)
    check("pseudo-video di guida creato", bool(guide),
          guide[-1].name if guide else "nessuno")
    if guide:
        from workers import ltx_msr
        lay = ltx_msr.frame_layout(3, 41)
        print(f"[guida] {guide[-1]} — layout atteso: soggetti su "
              f"{[lay.index(s) for s in range(3)]}, sfondo {lay.count(-1)} frame", flush=True)

    H.teardown(w)
    ok = not fails
    print(("=== MSR LIVE OK — ora GUARDA i file e giudica la somiglianza ==="
           if ok else f"=== FAIL: {fails} ==="), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
