"""Dialog di creazione/modifica voce per il TTS / voice cloning.

È l'UNICO punto dell'app dove si allega o registra un campione vocale: la radio
«Clona voce» e il pannello TTS condividono la stessa libreria di voci e questo dialog.

Una voce salvata (`VoicePreset`) porta con sé il proprio campione audio persistito in
`voices/samples/`. Per gli engine di cloning (CosyVoice 2, F5-TTS, …) il campione è
obbligatorio; per Edge TTS (preset Microsoft) serve invece scegliere una «voce base».
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QComboBox,
    QPushButton, QLabel, QPlainTextEdit, QFileDialog, QDialogButtonBox, QMessageBox,
    QWidget,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from core.audio_recorder import AudioRecorder
from core.tts_engine import (
    VoicePreset, available_engines, edge_list_voices, save_voice,
)
from ui.i18n_manager import tr


class VoiceDialog(QDialog):
    """Crea o modifica una voce. Passa `preset` per edit/duplica; `default_engine` per
    pre-selezionare l'engine corrente del pannello TTS."""

    def __init__(self, parent=None, preset: VoicePreset | None = None,
                 default_engine: str = "edge"):
        super().__init__(parent)
        self._preset = preset
        self._is_edit = preset is not None
        self.setWindowTitle(tr("voice.edit_title") if self._is_edit else tr("voice.new_title"))
        self.setMinimumWidth(480)

        self.recorder = AudioRecorder() if AudioRecorder.available() else None
        self._recording = False
        self._sample_path: str | None = preset.sample_path if preset else None
        self._media_player: QMediaPlayer | None = None
        self._audio_output: QAudioOutput | None = None
        # path del campione da copiare alla conferma (None se già gestito/invariato)
        self._sample_to_copy: str | None = None

        self._build(default_engine)
        if preset:
            self._load_preset(preset)
        self._on_engine_changed()

    # ──────────────────────────────────────────────────────────────────
    def _build(self, default_engine: str):
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("voice.name_placeholder"))
        form.addRow(tr("voice.name_label"), self.name_edit)

        # Motore: tutti gli engine; tooltip su chi supporta il cloning
        self.engine_combo = QComboBox()
        for eid, disp, clone, emo, installed in available_engines():
            badge = "✅" if installed else "⬇"
            cap = tr("voice.cap_clonable") if clone else tr("voice.cap_preset_only")
            self.engine_combo.addItem(f"{badge} {disp}  [{cap}]", eid)
        self.engine_combo.setToolTip(tr("voice.engine_tooltip"))
        # default = engine corrente del pannello TTS
        idx = next((i for i in range(self.engine_combo.count())
                    if self.engine_combo.itemData(i) == default_engine), 0)
        self.engine_combo.setCurrentIndex(idx)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        form.addRow(tr("voice.engine_label"), self.engine_combo)

        # Voce base (solo Edge): preset Microsoft IT
        self.base_voice_combo = QComboBox()
        for v in edge_list_voices("it"):
            self.base_voice_combo.addItem(v.name, v.voice_id)
        self.base_voice_row = self._wrap(self.base_voice_combo)
        self.base_voice_label = QLabel(tr("voice.base_voice_label"))
        form.addRow(self.base_voice_label, self.base_voice_row)

        root.addLayout(form)

        # ── Campione vocale (unico meccanismo attach/record) ──
        self.sample_box = QWidget()
        sb = QVBoxLayout(self.sample_box); sb.setContentsMargins(0, 0, 0, 0)
        sb.addWidget(QLabel(tr("voice.sample_header")))
        row = QHBoxLayout()
        self.pick_btn = QPushButton(tr("voice.load_file_btn")); self.pick_btn.clicked.connect(self._pick)
        self.rec_btn = QPushButton(tr("voice.record_btn")); self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._toggle_rec)
        if not self.recorder:
            self.rec_btn.setEnabled(False); self.rec_btn.setText(tr("voice.record_btn_no_mic"))
        self.play_btn = QPushButton(tr("voice.preview_btn")); self.play_btn.clicked.connect(self._preview)
        self.play_btn.setEnabled(bool(self._sample_path))
        row.addWidget(self.pick_btn); row.addWidget(self.rec_btn); row.addWidget(self.play_btn)
        sb.addLayout(row)
        self.sample_lbl = QLabel(self._sample_caption())
        self.sample_lbl.setStyleSheet("color:#9aa1b0;")
        sb.addWidget(self.sample_lbl)
        root.addWidget(self.sample_box)

        # ── Trascrizione del campione ──
        self.transcript_label = QLabel(tr("voice.transcript_label"))
        root.addWidget(self.transcript_label)
        self.transcript_edit = QPlainTextEdit(); self.transcript_edit.setMaximumHeight(60)
        self.transcript_edit.setPlaceholderText(tr("voice.transcript_placeholder"))
        root.addWidget(self.transcript_edit)

        # ── Note ──
        root.addWidget(QLabel(tr("voice.note_label")))
        self.note_edit = QLineEdit()
        root.addWidget(self.note_edit)

        # ── Pulsanti ──
        bb = QDialogButtonBox()
        bb.addButton(tr("voice.save_btn"), QDialogButtonBox.AcceptRole)
        bb.addButton(tr("voice.cancel_btn"), QDialogButtonBox.RejectRole)
        bb.accepted.connect(self._on_accept); bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _wrap(self, w) -> QWidget:
        c = QWidget(); l = QHBoxLayout(c); l.setContentsMargins(0, 0, 0, 0); l.addWidget(w); return c

    def _sample_caption(self) -> str:
        if self._sample_path:
            return tr("voice.sample_caption").format(name=Path(self._sample_path).name)
        return tr("voice.sample_none")

    # ──────────────────────────────────────────────────────────────────
    def _load_preset(self, p: VoicePreset):
        self.name_edit.setText(p.name)
        if p.engine == "edge" and p.voice_id:
            i = self.base_voice_combo.findData(p.voice_id)
            if i >= 0:
                self.base_voice_combo.setCurrentIndex(i)
        self.transcript_edit.setPlainText(p.sample_transcript or "")
        self.note_edit.setText(p.note or "")

    def _current_engine(self) -> str:
        return self.engine_combo.currentData() or "edge"

    def _engine_supports_clone(self) -> bool:
        eid = self._current_engine()
        for e, disp, clone, emo, installed in available_engines():
            if e == eid:
                return bool(clone)
        return False

    def _on_engine_changed(self, *_):
        is_edge = (self._current_engine() == "edge")
        # Edge → mostra voce base, nascondi campione; cloning → viceversa
        self.base_voice_label.setVisible(is_edge)
        self.base_voice_row.setVisible(is_edge)
        self.sample_box.setVisible(not is_edge)
        self.transcript_label.setVisible(not is_edge)
        self.transcript_edit.setVisible(not is_edge)

    # ──────────────────────────────────────────────────────────────────
    def _pick(self):
        fn, _ = QFileDialog.getOpenFileName(self, tr("voice.pick_dialog_title"), "", tr("voice.audio_filter"))
        if fn:
            self._sample_path = fn
            self._sample_to_copy = fn
            self.sample_lbl.setText(self._sample_caption())
            self.play_btn.setEnabled(True)

    def _toggle_rec(self, on: bool):
        if not self.recorder:
            return
        if on:
            self._recording = True; self.rec_btn.setText(tr("voice.stop_btn"))
            self.recorder.start()
            self.sample_lbl.setText(tr("voice.recording_in_progress"))
        else:
            self._recording = False; self.rec_btn.setText(tr("voice.record_btn"))
            # nome unico per non sovrascrivere registrazioni precedenti
            import uuid as _uuid
            path = self.recorder.stop(f"rec_{_uuid.uuid4().hex[:8]}.wav")
            self._sample_path = path
            self._sample_to_copy = path
            self.sample_lbl.setText(self._sample_caption())
            self.play_btn.setEnabled(True)

    def _preview(self):
        if not self._sample_path or not Path(self._sample_path).exists():
            return
        if self._media_player is None:
            self._media_player = QMediaPlayer(self)
            self._audio_output = QAudioOutput(self)
            self._media_player.setAudioOutput(self._audio_output)
        self._media_player.setSource(QUrl.fromLocalFile(self._sample_path))
        self._media_player.play()

    # ──────────────────────────────────────────────────────────────────
    def _on_accept(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, tr("voice.warn_name_title"), tr("voice.warn_name_text")); return
        engine = self._current_engine()
        if engine == "edge":
            voice_id = self.base_voice_combo.currentData()
            if not voice_id:
                QMessageBox.warning(self, tr("voice.warn_base_title"), tr("voice.warn_base_text")); return
        else:
            voice_id = ""
            if not self._sample_path:
                QMessageBox.warning(self, tr("voice.warn_sample_title"),
                    tr("voice.warn_sample_text")); return

        # costruisci/aggiorna il preset
        if self._preset is not None:
            v = self._preset
            v.name = name; v.engine = engine; v.voice_id = voice_id
            v.sample_transcript = self.transcript_edit.toPlainText().strip()
            v.note = self.note_edit.text().strip()
            v.custom = True
        else:
            v = VoicePreset(
                name=name, engine=engine, voice_id=voice_id,
                language="it-IT",
                note=self.note_edit.text().strip(),
                sample_path=self._sample_path,
                sample_transcript=self.transcript_edit.toPlainText().strip(),
                custom=True,
            )
        try:
            # copy_sample_from copia il file in voices/samples/ e aggiorna v.sample_path
            save_voice(v, copy_sample_from=self._sample_to_copy)
        except Exception as e:
            QMessageBox.critical(self, tr("voice.err_save_title"), str(e)); return
        self.saved_voice = v
        self.accept()
