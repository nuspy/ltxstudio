"""Dialog di riproduzione del video generato, con comandi: play/pausa, stop, volume,
indietro (-5s), reinizia, barra di avanzamento (seek), loop e apri cartella."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
)

from ui.i18n_manager import tr


def _fmt(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


class VideoPlayerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("player.window_title"))
        self.resize(740, 760)
        self._path: str | None = None
        self._loop = True

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget(self)
        self.player.setVideoOutput(self.video)

        v = QVBoxLayout(self)
        v.addWidget(self.video, 1)

        # barra di avanzamento + tempi
        srow = QHBoxLayout()
        self.pos_lbl = QLabel("0:00")
        self.seek = QSlider(Qt.Horizontal); self.seek.setRange(0, 0)
        self.dur_lbl = QLabel("0:00")
        srow.addWidget(self.pos_lbl); srow.addWidget(self.seek, 1); srow.addWidget(self.dur_lbl)
        v.addLayout(srow)

        # comandi
        c = QHBoxLayout()
        self.play_btn = QPushButton(tr("player.pause"))
        self.stop_btn = QPushButton(tr("player.stop"))
        self.back_btn = QPushButton(tr("player.back_5s"))
        self.restart_btn = QPushButton(tr("player.restart"))
        self.loop_btn = QPushButton(tr("player.loop")); self.loop_btn.setCheckable(True); self.loop_btn.setChecked(True)
        self.folder_btn = QPushButton(tr("player.open_folder"))
        for b in (self.play_btn, self.stop_btn, self.back_btn, self.restart_btn, self.loop_btn):
            c.addWidget(b)
        c.addStretch(1)
        c.addWidget(QLabel("🔊"))
        self.vol = QSlider(Qt.Horizontal); self.vol.setRange(0, 100); self.vol.setValue(80); self.vol.setMaximumWidth(130)
        c.addWidget(self.vol); c.addWidget(self.folder_btn)
        v.addLayout(c)

        # wiring
        self.play_btn.clicked.connect(self._toggle)
        self.stop_btn.clicked.connect(self.player.stop)
        self.back_btn.clicked.connect(lambda: self.player.setPosition(max(0, self.player.position() - 5000)))
        self.restart_btn.clicked.connect(self._restart)
        self.loop_btn.toggled.connect(lambda on: setattr(self, "_loop", on))
        self.folder_btn.clicked.connect(self._open_folder)
        self.vol.valueChanged.connect(lambda x: self.audio.setVolume(x / 100.0))
        self.audio.setVolume(0.8)
        self.seek.sliderMoved.connect(self.player.setPosition)
        self.player.positionChanged.connect(self._on_pos)
        self.player.durationChanged.connect(self._on_dur)
        self.player.playbackStateChanged.connect(self._on_state)
        self.player.mediaStatusChanged.connect(self._on_media_status)

    def play_file(self, path: str):
        self._path = path
        self.setWindowTitle(tr("player.window_title_file").format(name=Path(path).name))
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()
        self.show(); self.raise_(); self.activateWindow()

    def _toggle(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _restart(self):
        self.player.setPosition(0); self.player.play()

    def _on_pos(self, ms: int):
        if not self.seek.isSliderDown():
            self.seek.setValue(ms)
        self.pos_lbl.setText(_fmt(ms))

    def _on_dur(self, ms: int):
        self.seek.setRange(0, ms); self.dur_lbl.setText(_fmt(ms))

    def _on_state(self, st):
        self.play_btn.setText(tr("player.pause") if st == QMediaPlayer.PlayingState else tr("player.play"))

    def _on_media_status(self, st):
        if self._loop and st == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setPosition(0); self.player.play()

    def _open_folder(self):
        if not self._path:
            return
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer", "/select,", os.path.normpath(self._path)])
            else:
                subprocess.Popen(["xdg-open", str(Path(self._path).parent)])
        except Exception:
            pass

    def closeEvent(self, e):
        self.player.stop()
        super().closeEvent(e)
