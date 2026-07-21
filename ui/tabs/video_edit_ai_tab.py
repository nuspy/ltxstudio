"""Tab «Modifica Video AI» (#24/#59/#60) — editing video nativo con LoRA LTX-2 + tracking SAM2.

Modalità:
- **Edit-anything** (V2V da istruzione): trasforma il video via prompt (add/remove/replace/style),
  LoRA ltx23_edit_anything. ✅ validato.
- **Inpaint oggetto** (sperimentale): seleziona un oggetto a parole → SAM2 lo TRACCIA su tutto il clip
  → rigenera solo quell'area col prompt, LoRA ltx23_inpaint_masked. ⚠️ backend in completamento.
- **Stile** (anime↔real): LoRA ltx23_anime2real / real2anime.

Emette `request_edit` con la config; l'orchestrazione (tracking + worker) è in main_window.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from ui import theme  # noqa: E402
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from ui.i18n_manager import tr

_LORA = "LTX-2/"
MODES = [
    ("edit", "vedit.mode.edit", "LTX-2/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors"),
    ("inpaint", "vedit.mode.inpaint", "LTX-2/ltx23_inpaint_masked_t2v_rank128_v1_10000steps.safetensors"),
    ("anime2real", "vedit.mode.anime2real", "LTX-2/ltx23_anime2real_rank64_v1_4500.safetensors"),
    ("real2anime", "vedit.mode.real2anime", "LTX-2/ltx23_real2anime_rank64_v1_5000.safetensors"),
    # Sostituisci personaggio: IC-LoRA union-control-ref (canny dal video + immagine di riferimento) —
    # rimpiazza il personaggio mantenendo scena/movimento/audio (ricetta SECourses LTX 2.3, pesi già locali).
    ("replace_char", "vedit.mode.replace_char", "LTX-2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
]


class VideoEditAITab(QWidget):
    request_edit = Signal(dict)

    def __init__(self):
        super().__init__()
        self._video = ""
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_pick = QPushButton(tr("vedit.pick", "🎬 Scegli video…")); self.btn_pick.clicked.connect(self._pick)
        self.video_lbl = QLabel(tr("vedit.no_video", "nessun video"))
        top.addWidget(self.btn_pick); top.addWidget(self.video_lbl, 1)
        v.addLayout(top)

        f = QFormLayout()
        self.mode = QComboBox()
        for key, i18n, _lora in MODES:
            self.mode.addItem(tr(i18n, key), key)
        self.mode.currentIndexChanged.connect(self._on_mode)
        f.addRow(tr("vedit.mode_lbl", "Modalità"), self.mode)

        self.obj = QLineEdit(); self.obj.setPlaceholderText(tr("vedit.obj_ph", "oggetto da selezionare (es. car, person) — per Inpaint"))
        f.addRow(tr("vedit.obj_lbl", "Oggetto (a parole)"), self.obj)

        # Sostituisci personaggio: immagine di riferimento = primo frame del video con SOLO il personaggio cambiato.
        rrow = QHBoxLayout()
        self.ref_img = QLineEdit(); self.ref_img.setPlaceholderText(
            tr("vedit.ref.ph", "immagine sostitutiva: = primo frame del video ma con l'altro personaggio"))
        self.ref_btn = QPushButton(tr("vedit.ref.browse", "Sfoglia…")); self.ref_btn.clicked.connect(self._pick_ref)
        rrow.addWidget(self.ref_img, 1); rrow.addWidget(self.ref_btn)
        self.ref_row_lbl = QLabel(tr("vedit.ref_lbl", "Immagine sostitutiva (personaggio)"))
        f.addRow(self.ref_row_lbl, self._wrap_ref(rrow))
        # #102 forza dell'IDENTITÀ del nuovo personaggio (ref) vs movimento/struttura (canny). Prima hardcoded 1.0.
        from PySide6.QtWidgets import QDoubleSpinBox as _DSB0
        self.ref_strength = _DSB0(); self.ref_strength.setRange(0.1, 1.0); self.ref_strength.setSingleStep(0.05)
        self.ref_strength.setValue(0.8)
        self.ref_strength.setToolTip(tr("vedit.ref_strength.tt", "Quanto conta l'immagine di riferimento "
                                        "(identità del nuovo personaggio) rispetto al movimento del video: "
                                        "alto = più fedele alla foto, basso = più libertà."))
        self.ref_strength_lbl = QLabel(tr("vedit.ref_strength_lbl", "Forza identità (riferimento)"))
        f.addRow(self.ref_strength_lbl, self.ref_strength)

        self.prompt = QPlainTextEdit(); self.prompt.setFixedHeight(70)
        self.prompt.setPlaceholderText(tr("vedit.prompt_ph", "cosa generare/cambiare (es. a blue motorcycle / snowy winter)"))
        f.addRow(tr("vedit.prompt_lbl", "Prompt"), self.prompt)
        from PySide6.QtWidgets import QCheckBox as _CBX
        self.vedit_enhance = _CBX(tr("vedit.enhance", "✨ Arricchisci prompt (AI)"))
        self.vedit_enhance.setToolTip(tr("vedit.enhance.tt",
            "Un LLM locale arricchisce il prompt di modifica prima di elaborare. Invisibile, opzionale."))
        f.addRow("", self.vedit_enhance)

        self.strength = QComboBox()
        for s in ("0.6", "0.8", "1.0"):
            self.strength.addItem(s, float(s))
        self.strength.setCurrentText("1.0")
        f.addRow(tr("vedit.strength_lbl", "Forza editing"), self.strength)
        # #59 parametri inpaint (regolabili): CFG (aderenza prompt), guide (ancoraggio all'originale),
        # dilatazione maschera (spazio al nuovo oggetto). Valori equilibrati di default.
        from PySide6.QtWidgets import QDoubleSpinBox as _DSB, QSpinBox as _SB
        self.cfg = _DSB(); self.cfg.setRange(1.0, 10.0); self.cfg.setSingleStep(0.5); self.cfg.setValue(6.0)
        f.addRow(tr("vedit.cfg_lbl", "CFG (aderenza prompt) — inpaint"), self.cfg)
        self.guide = _DSB(); self.guide.setRange(0.1, 1.0); self.guide.setSingleStep(0.1); self.guide.setValue(0.4)
        f.addRow(tr("vedit.guide_lbl", "Forza guide (ancoraggio originale)"), self.guide)
        self.dilate = _SB(); self.dilate.setRange(0, 64); self.dilate.setValue(24)
        self.dilate.setToolTip(tr("vedit.dilate.tt",
            "Quanto ALLARGARE la maschera tracciata (px): più spazio per il nuovo oggetto, bordi meno stretti."))
        f.addRow(tr("vedit.dilate_lbl", "Dilatazione maschera (px)"), self.dilate)
        # blur maschera: era supportato dal worker (mask_blur) ma NON esposto → hardcoded 0. Sfumatura
        # dei BORDI della maschera → transizione morbida tra area rigenerata e video originale.
        self.blur = _DSB(); self.blur.setRange(0.0, 30.0); self.blur.setSingleStep(1.0); self.blur.setValue(6.0)
        self.blur.setToolTip(tr("vedit.blur.tt",
            "Sfumatura dei bordi della maschera (gaussian, px): 0 = bordo netto, 6-12 = transizione morbida."))
        f.addRow(tr("vedit.blur_lbl", "Sfumatura maschera (px)"), self.blur)
        v.addLayout(f)

        self.note = QLabel(""); self.note.setWordWrap(True); v.addWidget(self.note)
        self.btn_go = QPushButton(tr("vedit.go", "✨ Genera modifica")); self.btn_go.clicked.connect(self._go); theme.style_generate(self.btn_go)
        v.addWidget(self.btn_go)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); v.addWidget(self.log, 1)
        self._on_mode()

    def _wrap_ref(self, layout):
        from PySide6.QtWidgets import QWidget as _QW
        w = _QW(); w.setLayout(layout); return w

    def _pick_ref(self):
        from PySide6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.ref.browse", "Sfoglia…"), "",
                                           "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)")
        if p:
            self.ref_img.setText(p)

    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.pick", "Scegli video"), "",
                                           "Video (*.mp4 *.mov *.avi *.mkv *.webm)")
        if p:
            self._video = p
            self.video_lbl.setText(f"🎬 {Path(p).name}")

    def _on_mode(self, *_):
        key = self.mode.currentData()
        is_rc = (key == "replace_char")
        # #102: il campo «oggetto» serve all'inpaint E a replace_char (= personaggio da sostituire, per la
        # maschera SAM2 di precisione — opzionale: vuoto = sostituzione su tutto il frame).
        self.obj.setEnabled(key == "inpaint" or is_rc)
        self.obj.setPlaceholderText(
            tr("vedit.obj_ph_rc", "personaggio da sostituire (es. person) — opz., limita al soggetto") if is_rc
            else tr("vedit.obj_ph", "oggetto da selezionare (es. car, person) — per Inpaint"))
        for w in (self.ref_img, self.ref_btn, self.ref_row_lbl, self.ref_strength, self.ref_strength_lbl):
            w.setEnabled(is_rc)
        if key == "inpaint":
            self.note.setText(tr("vedit.note_inpaint",
                "⚠ Inpaint sperimentale: SAM2 traccia l'oggetto su tutto il clip e ne rigenera l'area."))
        elif key in ("anime2real", "real2anime"):
            self.note.setText(tr("vedit.note_style", "Trasferimento di stile sull'intero video."))
        elif is_rc:
            self.note.setText(tr("vedit.note_replace_char",
                "Sostituisci personaggio: carica un'immagine = primo frame del video ma con SOLO il personaggio "
                "cambiato. Il canny preserva scena/movimento; l'audio è mantenuto. Indica il «personaggio da "
                "sostituire» (a parole) per limitare la sostituzione al soggetto (SAM2, sfondo intatto); "
                "vuoto = tutto il frame. Video lunghi a segmenti."))
        else:
            self.note.setText(tr("vedit.note_edit", "Modifica globale del video da istruzione testuale."))

    def set_video(self, path: str):
        if path:
            self._video = path
            self.video_lbl.setText(f"🎬 {Path(path).name}")

    def _go(self):
        if not self._video:
            self.log.appendPlainText(tr("vedit.err_no_video", "Scegli prima un video.")); return
        key = self.mode.currentData()
        lora = next((l for k, _i, l in MODES if k == key), "")
        cfg = {"video": self._video, "mode": key, "edit_lora": lora,
               "object": self.obj.text().strip(), "prompt": self.prompt.toPlainText().strip(),
               "strength": float(self.strength.currentData()),
               "cfg_scale": float(self.cfg.value()), "guide": float(self.guide.value()),
               "mask_dilate": int(self.dilate.value()), "mask_blur": float(self.blur.value()),
               "enhance": self.vedit_enhance.isChecked(),
               "ref_image": self.ref_img.text().strip(),
               "ref_strength": float(self.ref_strength.value())}   # #102 forza identità del riferimento
        if key == "inpaint" and not cfg["object"]:
            self.log.appendPlainText(tr("vedit.err_no_obj", "Inpaint: indica l'oggetto da selezionare (a parole).")); return
        if key == "replace_char" and not cfg["ref_image"]:
            self.log.appendPlainText(tr("vedit.err_no_ref",
                "Sostituisci personaggio: scegli l'immagine sostitutiva del personaggio.")); return
        if not cfg["prompt"] and key != "replace_char":     # replace_char: il personaggio viene dall'immagine
            self.log.appendPlainText(tr("vedit.err_no_prompt", "Inserisci un prompt.")); return
        self.log.appendPlainText(tr("vedit.queued", "In elaborazione: {m}…").format(m=key))
        self.request_edit.emit(cfg)
