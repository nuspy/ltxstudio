"""Casi GUI (mock) di ERNIE-Image nel tab Immagini: il checkpoint Turbo (convertito da GGUF) compare SOLO sotto
l'architettura Ernie, preset 8 passi / guidance 1.0 del Turbo (il base torna ai default d'arch), e «Migliora prompt»
con l'enhancer ufficiale mancante: avviso con «Apri tab Pesi modelli» / «Continua senza» (nessun download). Lo Spy
intercetta _enqueue: niente GPU.
"""
from __future__ import annotations

from cases_qwen21_image import _run

TURBO = "ernie-image-turbo"


def _turbo_index(f):
    for i in range(f.gen_model.count()):
        if TURBO in f"{f.gen_model.itemText(i)} {f.gen_model.itemData(i) or ''}".lower():
            return i
    return -1


def case_ernie_turbo_only_under_ernie(w, spy):
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("SFW")
    f.gen_arch.setCurrentText("Ernie")
    if _turbo_index(f) < 0:
        return                                                    # il file non è su questo PC: niente da verificare
    for arch in ("FLUX", "Z-Image", "Krea 2", "Qwen-Image 2.1"):
        f.gen_arch.setCurrentText(arch)
        assert _turbo_index(f) < 0, f"ernie-image-turbo compare sotto {arch}"
    f.gen_arch.setCurrentText("Ernie")
    assert _turbo_index(f) >= 0, "ernie-image-turbo sparito dal menu Ernie"


def case_ernie_turbo_steps_guidance(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Ernie")
    i = _turbo_index(f)
    if i < 0:
        return
    f.gen_model.setCurrentIndex(i)
    assert f.gen_steps.value() == 8 and abs(f.gen_cfg.value() - 1.0) < 1e-6, (f.gen_steps.value(), f.gen_cfg.value())
    f.gen_prompt.setPlainText("un faro di vetro soffiato in mezzo a un deserto di sale, luce di mezzogiorno")
    p = _run(w, spy, w.generate_image)
    assert p["arch"] == "ernie" and "turbo" in str(p["model"]).lower(), p
    assert p["steps"] == 8 and abs(p["guidance"] - 1.0) < 1e-6, p
    for j in range(f.gen_model.count()):                          # un altro modello Ernie → default d'arch
        if j != i and f.gen_model.itemText(j) and not f.gen_model.itemText(j).startswith("("):
            f.gen_model.setCurrentIndex(j)
            assert f.gen_steps.value() == 20 and abs(f.gen_cfg.value() - 4.0) < 1e-6, "il base deve tornare a 20/4.0"
            break


def _press(label_key, default, sink):
    """QMessageBox.exec finto: clicca il pulsante con quel testo (nessun dialog reale in test)."""
    from PySide6.QtWidgets import QMessageBox
    from ui.i18n_manager import tr
    want = tr(label_key, default)
    orig = QMessageBox.exec

    def fake(self):
        sink.append(self.text())
        for b in self.buttons():
            if b.text() == want:
                b.click()
                break
        return 0
    QMessageBox.exec = fake
    return lambda: setattr(QMessageBox, "exec", orig)


def case_ernie_pe_missing_continue_or_weights(w, spy):
    from inference_server.backends import ernie_pe
    from core import model_paths as mp
    repo = str(mp.p("ernie-image", r"E:\AI\Models\diffusers\Ernie\ERNIE-Image"))
    if ernie_pe.available(repo):
        return                                                    # enhancer presente: avviso non applicabile
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Ernie")
    if f.gen_model.count() == 0 or f.gen_model.currentText().startswith("("):
        return
    f.gen_enhance.setChecked(True)
    f.gen_prompt.setPlainText("un cavaliere che cucina ramen su un tetto di Tokyo al tramonto")
    msgs: list = []
    undo = _press("ernie.pe.continue", "Continua senza migliorare", msgs)
    try:
        p = _run(w, spy, w.generate_image)
    finally:
        undo()
    assert msgs and "ernie-image-pe" in msgs[0], msgs
    assert p["arch"] == "ernie" and not p.get("enhance"), f"«Continua senza» deve spegnere enhance: {p.get('enhance')}"
    # «Apri tab Pesi modelli»: nessun job, il tab Pesi diventa quello corrente
    n = len([c for c in spy.calls if c["via"] == "enqueue"])
    undo = _press("vedit.ckpt.btn_open_weights_tab", "Apri tab Pesi modelli", msgs)
    try:
        try:
            _run(w, spy, w.generate_image)
        except AssertionError:
            pass                                                  # nessun job accodato = atteso
    finally:
        undo()
    assert len([c for c in spy.calls if c["via"] == "enqueue"]) == n, "il job non doveva partire"
    assert w.tabs.currentWidget() is w.weights_tab, "il tab Pesi modelli non si è aperto"
    f.gen_enhance.setChecked(False)


CASES = [
    ("ernie_turbo_only_under_ernie", "ERNIE-Image Turbo: solo sotto l'arch Ernie, mai sotto FLUX/Z-Image/Krea/Qwen.",
     case_ernie_turbo_only_under_ernie),
    ("ernie_turbo_steps_guidance", "ERNIE Turbo: 8 passi / guidance 1.0 al worker; il base torna a 20/4.0.",
     case_ernie_turbo_steps_guidance),
    ("ernie_pe_missing", "Migliora prompt su Ernie con PE mancante: avviso, «Continua senza» spegne enhance, «Apri Pesi» "
     "non accoda nulla.", case_ernie_pe_missing_continue_or_weights),
]
