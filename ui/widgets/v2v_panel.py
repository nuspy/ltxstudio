"""Pannello V2V (editing video) per il tab Video di LTX Studio.

Modalità via radio: Normal / Talk (lip-sync) / ReTake (rigenera sezione).
- Sezioni ACCORDION espandibili (freccia ▼/▲).
- ReTake: picker video + "usa ultima generazione" + anteprima + slider START/END (limiti
  bloccati a [3s, durata−3s]) + scelta audio (rigenera sezione / tieni originale).
- Talk: picker video + anteprima + maschera (bocca/volto) + offset + Re-Imagine.
- get_v2v_config() ritorna i valori; validation_errors() elenca cosa manca (per il dialog al Genera).

Indipendente: importabile e istanziabile senza il resto dell'app.
"""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton, QButtonGroup,
    QSlider, QComboBox, QSpinBox, QFileDialog, QFrame, QSizePolicy,
)

from ui.i18n_manager import tr

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
    _HAS_MEDIA = True
except Exception:
    _HAS_MEDIA = False

# contesto minimo richiesto prima/dopo la sezione (vincolo del workflow di riferimento)
EDGE_GUARD_S = 3.0


class CollapsibleSection(QWidget):
    """Sezione con header cliccabile (freccia ▼/▲) che mostra/nasconde il contenuto."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        self._btn = QPushButton(f"▼  {title}")
        self._btn.setCheckable(True); self._btn.setChecked(False)
        self._btn.setStyleSheet("QPushButton{text-align:left;font-weight:bold;padding:6px;}")
        self._btn.clicked.connect(self._toggle)
        self._title = title
        self.content = QWidget(); self.content.setVisible(False)
        v.addWidget(self._btn); v.addWidget(self.content)

    def _toggle(self):
        on = self._btn.isChecked()
        self.content.setVisible(on)
        self._btn.setText(("▲  " if on else "▼  ") + self._title)

    def set_expanded(self, on: bool):
        self._btn.setChecked(on); self._toggle()


class V2VPanel(QWidget):
    mode_changed = Signal(str)   # "normal" | "talk" | "retake"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: str | None = None
        self._duration_s: float = 0.0
        self._last_generation: str | None = None
        self._build()

    # ---------------- UI ----------------
    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(0, 6, 0, 0)

        # --- radio modalità (accanto a "Risoluzione" nel tab Video) ---
        moderow = QHBoxLayout()
        moderow.addWidget(QLabel(tr("v2v.mode_label")))
        self.rb_normal = QRadioButton(tr("v2v.mode_normal"))
        self.rb_talk = QRadioButton(tr("v2v.mode_talk"))
        self.rb_retake = QRadioButton(tr("v2v.mode_retake"))
        self.rb_normal.setChecked(True)
        self._mode_group = QButtonGroup(self)
        for rb in (self.rb_normal, self.rb_talk, self.rb_retake):
            self._mode_group.addButton(rb); moderow.addWidget(rb)
        moderow.addStretch(1)
        root.addLayout(moderow)
        for rb in (self.rb_normal, self.rb_talk, self.rb_retake):
            rb.toggled.connect(self._on_mode)

        # --- sorgente video (condivisa Talk/ReTake) ---
        self._src_wrap = QWidget()
        sv = QVBoxLayout(self._src_wrap); sv.setContentsMargins(0, 0, 0, 0)
        pick = QHBoxLayout()
        self.btn_pick = QPushButton(tr("v2v.pick_video_btn"))
        self.btn_pick.clicked.connect(self._pick_file)
        self.btn_lastgen = QPushButton(tr("v2v.use_last_gen_btn"))
        self.btn_lastgen.clicked.connect(self._use_last)
        pick.addWidget(self.btn_pick); pick.addWidget(self.btn_lastgen); pick.addStretch(1)
        sv.addLayout(pick)
        self.lbl_video = QLabel(tr("v2v.no_video"))
        self.lbl_video.setStyleSheet("color:#888;")
        sv.addWidget(self.lbl_video)
        # anteprima
        if _HAS_MEDIA:
            self._player = QMediaPlayer(self)
            self._audio = QAudioOutput(self); self._player.setAudioOutput(self._audio)
            self._video_w = QVideoWidget(); self._video_w.setMinimumHeight(180)
            self._video_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._player.setVideoOutput(self._video_w)
            self._player.durationChanged.connect(self._on_duration)
            sv.addWidget(self._video_w)
            pc = QHBoxLayout()
            self.btn_play = QPushButton("▶")
            self.btn_play.clicked.connect(self._toggle_play)
            pc.addWidget(self.btn_play); pc.addStretch(1)
            sv.addLayout(pc)
        else:
            sv.addWidget(QLabel(tr("v2v.preview_unavailable")))
        self._src_wrap.setVisible(False)
        root.addWidget(self._src_wrap)

        # --- sezione ReTake (accordion) ---
        self.sec_retake = CollapsibleSection(tr("v2v.retake_section_title"))
        rl = QVBoxLayout(self.sec_retake.content)
        self.lbl_range = QLabel(tr("v2v.range_label"))
        rl.addWidget(self.lbl_range)
        # slider START
        sr = QHBoxLayout(); sr.addWidget(QLabel(tr("v2v.start_label")))
        self.sl_start = QSlider(Qt.Horizontal); self.sl_start.setRange(0, 1000)
        self.lbl_start = QLabel("3.0 s")
        self.sl_start.valueChanged.connect(self._on_start)
        sr.addWidget(self.sl_start, 1); sr.addWidget(self.lbl_start); rl.addLayout(sr)
        # slider END
        er = QHBoxLayout(); er.addWidget(QLabel(tr("v2v.end_label")))
        self.sl_end = QSlider(Qt.Horizontal); self.sl_end.setRange(0, 1000)
        self.lbl_end = QLabel("— s")
        self.sl_end.valueChanged.connect(self._on_end)
        er.addWidget(self.sl_end, 1); er.addWidget(self.lbl_end); rl.addLayout(er)
        self.lbl_guard = QLabel(tr("v2v.guard_constraint").format(guard=f"{EDGE_GUARD_S:.0f}"))
        self.lbl_guard.setStyleSheet("color:#888;"); rl.addWidget(self.lbl_guard)
        # audio
        ar = QHBoxLayout(); ar.addWidget(QLabel(tr("v2v.section_audio_label")))
        self.rb_audio_keep = QRadioButton(tr("v2v.audio_keep"))
        self.rb_audio_regen = QRadioButton(tr("v2v.audio_regen"))
        self.rb_audio_keep.setChecked(True)
        ag = QButtonGroup(self); ag.addButton(self.rb_audio_keep); ag.addButton(self.rb_audio_regen)
        ar.addWidget(self.rb_audio_keep); ar.addWidget(self.rb_audio_regen); ar.addStretch(1)
        rl.addLayout(ar)
        self.sec_retake.setVisible(False)
        root.addWidget(self.sec_retake)

        # --- sezione Talk (accordion) ---
        self.sec_talk = CollapsibleSection(tr("v2v.talk_section_title"))
        tl = QVBoxLayout(self.sec_talk.content)
        tl.addWidget(QLabel(tr("v2v.talk_intro")))
        reg = QHBoxLayout(); reg.addWidget(QLabel(tr("v2v.face_mask_label")))
        self.cmb_region = QComboBox()
        self.cmb_region.addItem(tr("v2v.region_mouth"), "mouth")
        self.cmb_region.addItem(tr("v2v.region_face"), "face")
        reg.addWidget(self.cmb_region, 1); tl.addLayout(reg)
        off = QHBoxLayout(); off.addWidget(QLabel(tr("v2v.mask_offset_label")))
        self.sp_offset = QSpinBox(); self.sp_offset.setRange(0, 128); self.sp_offset.setValue(0)
        off.addWidget(self.sp_offset); off.addStretch(1); tl.addLayout(off)
        self.cb_reimagine = QPushButton(tr("v2v.reimagine_btn"))
        self.cb_reimagine.setCheckable(True)
        tl.addWidget(self.cb_reimagine)
        warn = QLabel(tr("v2v.talk_warning"))
        warn.setWordWrap(True); warn.setStyleSheet("color:#c80;"); tl.addWidget(warn)
        # descrizione dell'albero di decisione audio (in fondo alla sezione Talk)
        tree = QLabel(tr("v2v.audio_tree"))
        tree.setWordWrap(True); tree.setTextFormat(Qt.RichText)
        tree.setStyleSheet("color:#bbb; background:#202020; padding:6px; border-radius:4px;")
        tl.addWidget(tree)
        self.sec_talk.setVisible(False)
        root.addWidget(self.sec_talk)

    # ---------------- logica ----------------
    def current_mode(self) -> str:
        if self.rb_talk.isChecked():
            return "talk"
        if self.rb_retake.isChecked():
            return "retake"
        return "normal"

    def _on_mode(self):
        m = self.current_mode()
        self._src_wrap.setVisible(m in ("talk", "retake"))
        self.sec_retake.setVisible(m == "retake")
        self.sec_talk.setVisible(m == "talk")
        if m == "retake":
            self.sec_retake.set_expanded(True)
        if m == "talk":
            self.sec_talk.set_expanded(True)
        self.mode_changed.emit(m)

    def set_last_generation(self, path: str | None):
        self._last_generation = path
        self.btn_lastgen.setEnabled(bool(path))

    def _use_last(self):
        if self._last_generation:
            self._set_video(self._last_generation)

    def _pick_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, tr("v2v.pick_dialog_title"),
                                            "", tr("v2v.video_filter"))
        if fn:
            self._set_video(fn)

    def _set_video(self, path: str):
        self._video_path = path
        self.lbl_video.setText(f"🎬 {Path(path).name}")
        self.lbl_video.setStyleSheet("")
        self._duration_s = self._probe_duration(path)
        if _HAS_MEDIA:
            self._player.setSource(QUrl.fromLocalFile(path))
        self._reset_sliders()

    @staticmethod
    def _probe_duration(path: str) -> float:
        import subprocess
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    def _on_duration(self, ms: int):
        if ms > 0 and self._duration_s <= 0:
            self._duration_s = ms / 1000.0
            self._reset_sliders()

    def _reset_sliders(self):
        d = self._duration_s
        if d <= 2 * EDGE_GUARD_S:
            # video troppo corto per un ReTake con i 3s di contesto
            self.sl_start.setEnabled(False); self.sl_end.setEnabled(False)
            self.lbl_start.setText("—"); self.lbl_end.setText("—")
            self.lbl_guard.setText(tr("v2v.guard_too_short").format(min=f"{2*EDGE_GUARD_S:.0f}"))
            self.lbl_guard.setStyleSheet("color:#c00;")
            return
        self.sl_start.setEnabled(True); self.sl_end.setEnabled(True)
        self.lbl_guard.setStyleSheet("color:#888;")
        self.lbl_guard.setText(tr("v2v.guard_constraint").format(guard=f"{EDGE_GUARD_S:.0f}"))
        # default: sezione centrale
        self._set_start_s(EDGE_GUARD_S)
        self._set_end_s(min(d - EDGE_GUARD_S, EDGE_GUARD_S + max(1.0, (d - 2 * EDGE_GUARD_S) / 2)))

    def _s_to_slider(self, s: float) -> int:
        d = max(0.001, self._duration_s)
        return int(round(1000 * s / d))

    def _slider_to_s(self, v: int) -> float:
        return self._duration_s * v / 1000.0

    def _set_start_s(self, s: float):
        self.sl_start.blockSignals(True); self.sl_start.setValue(self._s_to_slider(s)); self.sl_start.blockSignals(False)
        self.lbl_start.setText(f"{s:.1f} s")

    def _set_end_s(self, s: float):
        self.sl_end.blockSignals(True); self.sl_end.setValue(self._s_to_slider(s)); self.sl_end.blockSignals(False)
        self.lbl_end.setText(f"{s:.1f} s")

    def _on_start(self, v: int):
        d = self._duration_s
        s = self._slider_to_s(v)
        lo, hi = EDGE_GUARD_S, max(EDGE_GUARD_S, self.v2v_end() - 0.5)
        s = min(max(s, lo), hi)                     # blocco ai limiti
        self._set_start_s(s)
        if _HAS_MEDIA:
            self._player.setPosition(int(s * 1000)); self._player.pause()

    def _on_end(self, v: int):
        d = self._duration_s
        s = self._slider_to_s(v)
        lo, hi = self.v2v_start() + 0.5, max(EDGE_GUARD_S, d - EDGE_GUARD_S)
        s = min(max(s, lo), hi)
        self._set_end_s(s)
        if _HAS_MEDIA:
            self._player.setPosition(int(s * 1000)); self._player.pause()

    def _toggle_play(self):
        if not _HAS_MEDIA:
            return
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause(); self.btn_play.setText("▶")
        else:
            self._player.play(); self.btn_play.setText("⏸")

    def v2v_start(self) -> float:
        return self._slider_to_s(self.sl_start.value())

    def v2v_end(self) -> float:
        return self._slider_to_s(self.sl_end.value())

    # ---------------- API per l'app ----------------
    def get_v2v_config(self) -> dict:
        """Valori V2V da fondere nei `vals` di generazione (vuoto/normal se modalità Normal)."""
        m = self.current_mode()
        cfg = {"v2v_mode": m}
        if m in ("talk", "retake"):
            cfg["v2v_video_path"] = self._video_path        # path locale (l'app lo carica in ComfyUI)
        if m == "retake":
            cfg.update({"v2v_start": round(self.v2v_start(), 2), "v2v_end": round(self.v2v_end(), 2),
                        "v2v_regen_audio": self.rb_audio_regen.isChecked()})
        if m == "talk":
            cfg.update({"talk_region": self.cmb_region.currentData(),
                        "talk_mask_offset": self.sp_offset.value(),
                        "talk_reimagine": self.cb_reimagine.isChecked()})
        return cfg

    def validation_errors(self) -> list[str]:
        """Elenco di cosa manca per generare nella modalità corrente (per il dialog al Genera)."""
        m = self.current_mode()
        errs = []
        if m == "normal":
            return errs
        if not self._video_path:
            # ReTake richiede il video; Talk SENZA video genera dall'immagine del Frame 1 (ok).
            if m == "retake":
                errs.append(tr("v2v.err_no_source_video"))
            return errs
        if self._duration_s <= 0:
            errs.append(tr("v2v.err_no_duration"))
        if m == "retake":
            if self._duration_s and self._duration_s <= 2 * EDGE_GUARD_S:
                errs.append(tr("v2v.err_too_short").format(min=f"{2*EDGE_GUARD_S:.0f}"))
            elif self.v2v_end() - self.v2v_start() < 0.5:
                errs.append(tr("v2v.err_section_too_short"))
        return errs
