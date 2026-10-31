"""Dialog di avanzamento generazione (video / inpaint / outpaint / immagine).

Mostra: cosa sta succedendo ora (caricamento modello, codifica prompt, sampling, decodifica…),
una barra dell'operazione corrente e una barra del totale APPROSSIMATIVO (pesata: il sampling
pesa molto). Alimentato dagli eventi WebSocket di ComfyUI (executing / progress / preview).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QDialogButtonBox, QSizePolicy,
)

from ui.i18n_manager import tr


def _weight(ct: str) -> int:
    """Peso del nodo per il totale approssimato (il sampling domina i tempi)."""
    c = ct.lower()
    if "sampler" in c or "ksampler" in c:
        return 40
    if "unetloader" in c or "gguf" in c or "checkpointloader" in c:
        return 6
    if "vaedecode" in c:
        return 5
    if "dualcliploader" in c or "cliploader" in c or "samloader" in c:
        return 4
    if "vaeencode" in c:
        return 3
    return 1


def _label(ct: str) -> str:
    c = ct.lower()
    if "sampler" in c or "ksampler" in c:
        return tr("progress.step_sampling")
    if "unetloader" in c or "gguf" in c or "checkpointloader" in c:
        return tr("progress.step_load_model")
    if "dualcliploader" in c or "cliploader" in c:
        return tr("progress.step_load_text_encoder")
    if "vaeloader" in c:
        return tr("progress.step_load_vae")
    if "vaedecode" in c:
        return tr("progress.step_decode")
    if "vaeencode" in c:
        return tr("progress.step_encode")
    if "cliptextencode" in c or "conditioning" in c or "fluxguidance" in c or "promptrelay" in c:
        return tr("progress.step_encode_prompt")
    if "loadimage" in c or "loadmask" in c:
        return tr("progress.step_load_image")
    if "loraloader" in c or "powerlora" in c or "lora" in c:
        return tr("progress.step_apply_lora")
    if "sam" in c or "clipseg" in c or "detector" in c or "segs" in c:
        return tr("progress.step_segmentation")
    if "save" in c or "videocombine" in c or "combine" in c:
        return tr("progress.step_saving")
    if "upscale" in c or "resize" in c:
        return tr("progress.step_upscaling")
    if "padforoutpaint" in c or "growmask" in c or "inpaintmodel" in c:
        return tr("progress.step_prepare_mask")
    return tr("progress.step_processing")


class GenProgressDialog(QDialog):
    def __init__(self, parent=None, on_cancel=None):
        super().__init__(parent)
        self.setWindowTitle(tr("progress.window_title"))
        # Dialog ridimensionabile e grande di default → l'anteprima taesd ha spazio.
        self.setMinimumSize(900, 760)
        self.resize(1100, 900)
        self._on_cancel = on_cancel
        self._weights: dict[str, int] = {}
        self._classes: dict[str, str] = {}
        self._total_w = 1
        self._done_w = 0
        self._counted: set[str] = set()
        self._cur: str | None = None

        v = QVBoxLayout(self)
        self.title_lbl = QLabel("…"); self.title_lbl.setStyleSheet("font-weight:bold;font-size:14px;")
        v.addWidget(self.title_lbl)
        self.status_lbl = QLabel(tr("progress.preparing")); self.status_lbl.setStyleSheet("color:#cdd;")
        v.addWidget(self.status_lbl)

        v.addWidget(QLabel(tr("progress.current_operation")))
        self.cur_bar = QProgressBar(); v.addWidget(self.cur_bar)
        v.addWidget(QLabel(tr("progress.total_progress")))
        self.tot_bar = QProgressBar(); self.tot_bar.setRange(0, 100); v.addWidget(self.tot_bar)
        # Punto 21: quanto e fatto e quanto manca — tempo trascorso e stima del rimanente (da step/secondo reali).
        self.time_lbl = QLabel("")
        self.time_lbl.setStyleSheet("color:#9aa1b0;")
        v.addWidget(self.time_lbl)

        self.preview = QLabel(tr("progress.preview_placeholder"))
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(800, 560)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setScaledContents(False)        # niente stretch, riscala manualmente con KeepAspectRatio
        self.preview.setStyleSheet("background:#111;color:#666;border:1px solid #333;")
        v.addWidget(self.preview, 1)
        # Tieni il pixmap originale per riscalare correttamente al resize del dialog
        self._preview_pix: QPixmap | None = None

        bb = QDialogButtonBox()
        bb.addButton(tr("progress.cancel_generation"), QDialogButtonBox.RejectRole)
        bb.rejected.connect(self._cancel)
        v.addWidget(bb)
        self.setModal(False)

    # ---------- ciclo di vita ----------
    def start(self, prompt: dict, title: str):
        self.title_lbl.setText(title)
        self.status_lbl.setText(tr("progress.sending_to_queue"))
        self._weights = {nid: _weight(n.get("class_type", "")) for nid, n in prompt.items()}
        self._classes = {nid: n.get("class_type", "") for nid, n in prompt.items()}
        self._total_w = max(1, sum(self._weights.values()))
        self._done_w = 0; self._counted = set(); self._cur = None
        self.cur_bar.setRange(0, 0)        # "busy" finché non arriva un progress
        self.tot_bar.setValue(0)
        import time as _t
        self._t_start = _t.time()          # per tempo trascorso / stima (punto 21)
        self._t_first_step = None
        self.time_lbl.setText("")
        self._preview_pix = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText(tr("progress.preview_placeholder"))
        self.show(); self.raise_(); self.activateWindow()

    def on_executing(self, node_id: str):
        # il nodo precedente è terminato → accumula il suo peso
        if self._cur and self._cur not in self._counted:
            self._done_w += self._weights.get(self._cur, 1); self._counted.add(self._cur)
        if not node_id:
            self.status_lbl.setText(tr("progress.finalizing")); self.tot_bar.setValue(99)
            self.cur_bar.setRange(0, 100); self.cur_bar.setValue(100); return
        self._cur = node_id
        self.status_lbl.setText(_label(self._classes.get(node_id, "")))
        self.cur_bar.setRange(0, 0)        # busy
        self._update_total(0.0)

    def on_progress(self, value: int, maximum: int):
        if maximum > 0:
            self.cur_bar.setRange(0, maximum); self.cur_bar.setValue(value)
            ct = self._classes.get(self._cur or "", "")
            extra = tr("progress.step_counter").format(value=value, maximum=maximum) if ("sampler" in ct.lower() or "ksampler" in ct.lower()) else ""
            self.status_lbl.setText(_label(ct) + extra)
            self._update_total(value / maximum)

    def on_preview(self, data: bytes):
        pix = QPixmap()
        if pix.loadFromData(data):
            self._preview_pix = pix
            self.preview.setPixmap(pix.scaled(self.preview.size(),
                                              Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation))

    def on_native_message(self, msg: str):
        """Fase corrente del backend nativo (es. 'Caricamento componenti…', 'Generazione…')."""
        if msg:
            self.status_lbl.setText(msg)
        self._update_time(None)

    def on_native_step(self, step: int, total: int):
        """Progresso per il backend NATIVO (immagini/video): step/total REALI del denoising → barra
        credibile (entrambe le barre = step/total), senza dipendere dai pesi-nodo ComfyUI. Il totale
        avanza monotòno; a step==total resta a 99 finché arriva finish() (decode/salvataggio)."""
        if total <= 0:
            return
        step = max(0, min(int(step), int(total)))
        self.cur_bar.setRange(0, total); self.cur_bar.setValue(step)
        # Totale PESATO sulle fasi (punto 21): caricamento/encoding 0-10 %, sampling 10-92 %, decodifica/salvataggio 92-99 %
        # (prima il totale era la sola percentuale di sampling: a step 1/8 gia il 12 % pur con il caricamento ancora da fare).
        pct = 10 + int(82 * step / total)
        self.tot_bar.setValue(max(self.tot_bar.value(), min(99, pct)))
        self._update_time((step, total))
        self.status_lbl.setText(tr("progress.step_sampling") +
                                tr("progress.step_counter").format(value=step, maximum=total))

    def _update_time(self, st):
        """«Trascorso mm:ss · rimanente ~mm:ss»: il rimanente solo dopo almeno 2 step (velocita reale del sampling)."""
        import time as _t
        t0 = getattr(self, "_t_start", None)
        if t0 is None:
            return
        el = _t.time() - t0
        txt = tr("progress.elapsed", "Trascorso {t}").format(t=self._fmt(el))
        if st:
            step, total = st
            if getattr(self, "_t_first_step", None) is None and step >= 1:
                self._t_first_step = (_t.time(), step)
            fs = getattr(self, "_t_first_step", None)
            if fs and step - fs[1] >= 1:
                rate = (_t.time() - fs[0]) / (step - fs[1])
                txt += "  ·  " + tr("progress.remaining", "rimanente ~{t}").format(t=self._fmt(rate * (total - step) + 3))
        self.time_lbl.setText(txt)

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = int(max(0, sec))
        return f"{sec // 60}:{sec % 60:02d}"

    def resizeEvent(self, event):
        # Mantieni l'anteprima nitida quando l'utente ridimensiona il dialog.
        super().resizeEvent(event)
        if self._preview_pix is not None:
            self.preview.setPixmap(self._preview_pix.scaled(self.preview.size(),
                                                            Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation))

    def _update_total(self, frac: float):
        cw = self._weights.get(self._cur or "", 0)
        pct = int(100 * (self._done_w + cw * frac) / self._total_w)
        self.tot_bar.setValue(max(self.tot_bar.value(), min(99, pct)))   # monotono, 100 solo a fine

    def finish(self):
        self.tot_bar.setValue(100)
        self.cur_bar.setRange(0, 100); self.cur_bar.setValue(100)
        self.hide()

    def _cancel(self):
        if self._on_cancel:
            self._on_cancel()
        self.hide()
