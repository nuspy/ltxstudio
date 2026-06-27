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
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from ui.i18n_manager import tr

_LORA = "LTX-2/"
MODES = [
    ("edit", "vedit.mode.edit", "LTX-2/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors"),
    ("inpaint", "vedit.mode.inpaint", "LTX-2/ltx23_inpaint_masked_t2v_rank128_v1_10000steps.safetensors"),
    ("anime2real", "vedit.mode.anime2real", "LTX-2/ltx23_anime2real_rank64_v1_4500.safetensors"),
    ("real2anime", "vedit.mode.real2anime", "LTX-2/ltx23_real2anime_rank64_v1_5000.safetensors"),
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

        self.prompt = QPlainTextEdit(); self.prompt.setFixedHeight(70)
        self.prompt.setPlaceholderText(tr("vedit.prompt_ph", "cosa generare/cambiare (es. a blue motorcycle / snowy winter)"))
        f.addRow(tr("vedit.prompt_lbl", "Prompt"), self.prompt)

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
        f.addRow(tr("vedit.dilate_lbl", "Dilatazione maschera (px)"), self.dilate)
        v.addLayout(f)

        self.note = QLabel(""); self.note.setWordWrap(True); v.addWidget(self.note)
        self.btn_go = QPushButton(tr("vedit.go", "✨ Genera modifica")); self.btn_go.clicked.connect(self._go)
        v.addWidget(self.btn_go)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); v.addWidget(self.log, 1)
        self._on_mode()

    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.pick", "Scegli video"), "",
                                           "Video (*.mp4 *.mov *.avi *.mkv *.webm)")
        if p:
            self._video = p
            self.video_lbl.setText(f"🎬 {Path(p).name}")

    def _on_mode(self, *_):
        key = self.mode.currentData()
        self.obj.setEnabled(key == "inpaint")
        if key == "inpaint":
            self.note.setText(tr("vedit.note_inpaint",
                "⚠ Inpaint sperimentale: SAM2 traccia l'oggetto su tutto il clip e ne rigenera l'area."))
        elif key in ("anime2real", "real2anime"):
            self.note.setText(tr("vedit.note_style", "Trasferimento di stile sull'intero video."))
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
               "mask_dilate": int(self.dilate.value())}
        if key == "inpaint" and not cfg["object"]:
            self.log.appendPlainText(tr("vedit.err_no_obj", "Inpaint: indica l'oggetto da selezionare (a parole).")); return
        if not cfg["prompt"]:
            self.log.appendPlainText(tr("vedit.err_no_prompt", "Inserisci un prompt.")); return
        self.log.appendPlainText(tr("vedit.queued", "In elaborazione: {m}…").format(m=key))
        self.request_edit.emit(cfg)
