"""MSR LIVE 2.5 — riferimenti multipli su LTX 2.5 (slot embedding), pilotando la GUI VERA.

Variante 2.5 di `test_msr_live.py`. Differenze rispetto alla 2.3:
  · niente pseudo-video di guida: i riferimenti entrano come conditioning a frame NEGATIVI
    (Fourier-MLP degli slot dentro la LoRA `LTX-2.5-Licon-MSR-V1`), quindi NON serve lo sfondo;
  · il checkpoint da scegliere nel dropdown è quello 2.5: è la sua selezione a far cambiare
    release al dispatch, e con essa la LoRA MSR.

Resta invariato ciò che conta: i token che il pannello assegna DEVONO comparire nel prompt
(su LTX sono «Image 1», «Image 2»…, non i token angolari di MiniMax H3), altrimenti il modello
non collega testo e riferimento.

  <gui_python> tests/app_env/test_msr_live_25.py
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
# stessi riferimenti del test 2.3: già su disco, e il confronto fra le due release resta diretto
REF_NAMES = ("sax", "robot", "parrot")


def _genera(w, etichetta: str):
    """Preme Genera dalla GUI e attende l'mp4 — stessa logica del test 2.3."""
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
            # il file compare PRIMA che il worker emetta `done`: attendere il thread, altrimenti
            # il job successivo resta appeso (la coda è seriale)
            fine = _t.time() + 180
            while _t.time() < fine:
                QApplication.processEvents()
                wk = getattr(w, "_ltx2_worker", None)
                if wk is None or not wk.isRunning():
                    break
                _t.sleep(0.5)
            QApplication.processEvents()
            return out
        _t.sleep(1.0)
    print(f"[out {etichetta}] NESSUN VIDEO in {_t.time() - t0:.0f}s", flush=True)
    return None


def main() -> int:
    print("=== MSR LIVE (LTX 2.5, slot embedding) ===", flush=True)
    fails = []

    def check(name, cond, extra=""):
        print(("OK   " if cond else "FAIL ") + name + (f" — {extra}" if extra else ""), flush=True)
        if not cond:
            fails.append(name)

    from core import ltx_release as lr
    rel = lr.get("2.5")
    check("LoRA MSR 2.5 su disco", os.path.exists(rel.msr_lora), os.path.basename(rel.msr_lora))
    check("componenti 2.5 complete", not lr.missing_components("2.5"),
          str(lr.missing_components("2.5")))
    if lr.missing_components("2.5"):
        return 2

    refs = [os.path.join(OUT, f"DIAG_msr_ref_{n}.png") for n in REF_NAMES]
    refs = [p for p in refs if os.path.exists(p)]
    check("riferimenti su disco", len(refs) >= 2, f"{len(refs)} trovati")
    if len(refs) < 2:
        print("   (genera prima i riferimenti con test_msr_live.py)", flush=True)
        return 2

    w = H.boot_full_app()
    from PySide6.QtWidgets import QDialog
    QDialog.exec = lambda self, *a, **k: QDialog.Accepted
    QDialog.exec_ = QDialog.exec

    # 1) pannello riferimenti della GUI VERA
    vt = w.video_tab
    from core.model_catalog import video_caps
    vt._apply_reference_limits("ltx", video_caps("ltx"))
    panel = vt.reference_panel
    panel.enable.setChecked(True)
    for p in refs:
        check(f"riferimento accettato: {os.path.basename(p)}", panel.add_path(p))
    # Il pannello pretende lo sfondo per LTX senza distinguere la release («i riferimenti diventano
    # una sequenza e lo sfondo ne riempie i fotogrammi liberi»): è vero per la pseudo-video 2.3,
    # non per lo slot embedding 2.5. Lo si fornisce comunque per superare la validazione — se poi
    # il worker non lo usa lo dirà il log, ed è un vincolo del pannello da rilassare per la 2.5.
    bg = os.path.join(OUT, "DIAG_msr_ref_stage.png")
    if os.path.exists(bg):
        panel.set_background(bg)
    errs = panel.validation_errors()
    print(f"[pannello] errori di validazione: {errs or 'nessuno'}", flush=True)

    # 2) i token VERI assegnati dal pannello, dentro il prompt
    tok = [r.token for r in panel._rows]
    print(f"[token] assegnati dal pannello: {tok}", flush=True)
    vt.prompt.setPlainText(
        f"{tok[0]} the elderly bearded saxophonist in the purple velvet suit plays on stage, "
        f"{tok[1]} the little copper robot dances beside him"
        + (f", {tok[2]} the scarlet parrot flies above them" if len(tok) > 2 else "")
        + ", smoky jazz club, red curtains, warm lanterns, cinematic wide shot")
    print(f"[prompt] {vt.prompt.toPlainText()[:120]}…", flush=True)

    # 3) parametri video (sotto ~700px LTX fa poltiglia a prescindere dai riferimenti)
    vt.width.setValue(704); vt.height.setValue(448)
    vt.fps.setValue(24); vt.length.setValue(2.0)
    vt.sampler_seed.steps.setValue(8)
    vt.sampler_seed.seed.setValue(7)

    # 4) checkpoint 2.5: è la sua scelta a portare il dispatch sulla release 2.5
    _c25 = [i for i in range(vt.model.count())
            if "2.5" in (vt.model.itemText(i) or "")
            and "gguf" not in (vt.model.itemText(i) or "").lower()]
    check("checkpoint 2.5 nel dropdown", bool(_c25),
          vt.model.itemText(_c25[0]) if _c25 else "NESSUNO")
    if not _c25:
        H.teardown(w)
        return 2
    vt.model.setCurrentIndex(_c25[0])
    print(f"[modello] {vt.model.itemText(_c25[0])}", flush=True)

    # 5) IL BOTTONE
    produced = _genera(w, "MSR 2.5")
    check("video prodotto", produced is not None)

    # 6) controllo senza riferimenti, stessi parametri
    panel.enable.setChecked(False)
    vt.prompt.setPlainText("an elderly bearded saxophonist plays on a smoky jazz club stage, "
                           "a small copper robot dances beside him, warm lanterns, cinematic")
    control = _genera(w, "CONTROLLO senza riferimenti")
    check("controllo prodotto", control is not None)

    # 7) la 2.5 NON deve produrre pseudo-video di guida: quello è il meccanismo 2.3
    recenti = [p for p in Path(OUT).glob("*_msr_guide.mp4")
               if time.time() - p.stat().st_mtime < 1800]
    check("nessuna guida pseudo-video (atteso su 2.5)", not recenti,
          str([p.name for p in recenti]))

    H.teardown(w)
    ok = not fails
    print(("=== MSR 2.5 OK — ora GUARDA i file e giudica la somiglianza ==="
           if ok else f"=== FAIL: {fails} ==="), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
