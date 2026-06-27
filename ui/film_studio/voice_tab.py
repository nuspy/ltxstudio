"""Tab Voice Studio (spec §8): voce da descrizione + clonazione + binding voce↔attore.
Riusa core/tts_engine e core/film_studio/voice_studio. Zero ComfyUI."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from core.film_studio import db, voice_studio as vs
from ui.i18n_manager import tr


class _DescWorker(QThread):
    done = Signal(dict)

    def __init__(self, desc, name, actor_id):
        super().__init__()
        self._a = (desc, name, actor_id)

    def run(self):
        try:
            self.done.emit(vs.voice_from_description(self._a[0], self._a[1], actor_id=self._a[2]))
        except Exception as e:                              # noqa: BLE001
            self.done.emit({"error": str(e)})


class VoiceTab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(QLabel(tr("fs.voice.list", "Voci")))
        self.lst = QListWidget(); left.addWidget(self.lst, 1)
        b_del = QPushButton(tr("tlora.del", "🗑 Rimuovi")); b_del.clicked.connect(self._del)
        left.addWidget(b_del)
        root.addLayout(left, 1)
        right = QVBoxLayout()
        # voce da descrizione
        d = QGroupBox(tr("fs.voice.described", "Voce da descrizione")); dv = QVBoxLayout(d)
        self.v_name = QLineEdit(); self.v_name.setPlaceholderText(tr("fs.voice.name_ph", "Nome voce (es. Narratore)"))
        dv.addWidget(self.v_name)
        self.v_desc = QPlainTextEdit(); self.v_desc.setFixedHeight(70)
        self.v_desc.setPlaceholderText(tr("fs.voice.desc_ph", "es. uomo anziano, baritono, tono autorevole e stanco, accento del nord"))
        dv.addWidget(self.v_desc)
        self.btn_desc = QPushButton(tr("fs.voice.create_desc", "🗣 Crea voce da descrizione")); self.btn_desc.clicked.connect(self._create_desc)
        dv.addWidget(self.btn_desc)
        right.addWidget(d)
        # clonazione
        c = QGroupBox(tr("fs.voice.cloned", "Voce clonata (da campione)")); cv = QVBoxLayout(c)
        self.c_name = QLineEdit(); self.c_name.setPlaceholderText(tr("fs.voice.name_ph", "Nome voce"))
        cv.addWidget(self.c_name)
        hs = QHBoxLayout()
        self.c_sample = QLineEdit(); self.c_sample.setReadOnly(True)
        b_s = QPushButton(tr("fs.voice.sample", "Campione…")); b_s.clicked.connect(self._pick_sample)
        hs.addWidget(self.c_sample, 1); hs.addWidget(b_s)
        cv.addLayout(hs)
        self.btn_clone = QPushButton(tr("fs.voice.create_clone", "🎙 Registra voce clonata")); self.btn_clone.clicked.connect(self._create_clone)
        cv.addWidget(self.btn_clone)
        right.addWidget(c)
        # binding attore
        b = QGroupBox(tr("fs.voice.binding", "Lega a un attore")); bv = QVBoxLayout(b)
        hb = QHBoxLayout()
        self.actor = QComboBox()
        hb.addWidget(QLabel(tr("fs.voice.actor", "Attore"))); hb.addWidget(self.actor, 1)
        self.btn_bind = QPushButton(tr("fs.voice.bind", "🔗 Lega")); self.btn_bind.clicked.connect(self._bind)
        hb.addWidget(self.btn_bind)
        bv.addLayout(hb)
        right.addWidget(b)
        self.status = QLabel(""); right.addWidget(self.status); right.addStretch(1)
        root.addLayout(right, 2)
        self._refresh()

    def _refresh(self):
        self.lst.clear()
        self._voices = db.query("voice", order="id DESC")
        for v in self._voices:
            tag = f" → attore#{v['actor_id']}" if v.get("actor_id") else ""
            self.lst.addItem(f"[{v['origin']}] {v['name']}{tag}")
        self.actor.clear()
        self._actors = db.find_assets(type_="actor")
        for a in self._actors:
            self.actor.addItem(a["name"], a["id"])

    def _create_desc(self):
        if not self.v_desc.toPlainText().strip():
            QMessageBox.information(self, tr("fs.voice.described", "Voce"), tr("fs.voice.need_desc", "Scrivi una descrizione.")); return
        self.btn_desc.setEnabled(False); self.status.setText(tr("fs.working", "Generazione…"))
        self._worker = _DescWorker(self.v_desc.toPlainText().strip(), self.v_name.text().strip(), 0)
        self._worker.done.connect(self._on_desc); self._worker.start()

    def _on_desc(self, res: dict):
        self.btn_desc.setEnabled(True)
        if res.get("error"):
            self.status.setText(""); QMessageBox.critical(self, tr("fs.voice.described", "Voce"), str(res["error"])[:400]); return
        self.status.setText(tr("fs.voice.created", "Voce creata: {v}").format(v=res.get("edge_voice", "")))
        self._refresh()

    def _pick_sample(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("fs.voice.sample", "Campione"), "", "Audio (*.wav *.mp3 *.flac)")
        if p:
            self.c_sample.setText(p)

    def _create_clone(self):
        if not self.c_sample.text() or not self.c_name.text().strip():
            QMessageBox.information(self, tr("fs.voice.cloned", "Clone"), tr("fs.voice.need_clone", "Nome + campione richiesti.")); return
        try:
            vs.register_clone(self.c_name.text().strip(), self.c_sample.text())
            self._refresh()
            self.status.setText(tr("fs.voice.created", "Voce creata: {v}").format(v=self.c_name.text().strip()))
        except Exception as e:                              # noqa: BLE001
            QMessageBox.critical(self, tr("fs.voice.cloned", "Clone"), str(e)[:400])

    def _bind(self):
        row = self.lst.currentRow()
        if not (0 <= row < len(getattr(self, "_voices", []))) or self.actor.currentData() is None:
            return
        vs.bind(self._voices[row]["id"], self.actor.currentData())
        self._refresh()
        self.status.setText(tr("fs.voice.bound", "Voce legata all'attore."))

    def _del(self):
        row = self.lst.currentRow()
        if 0 <= row < len(getattr(self, "_voices", [])):
            db.delete("voice", self._voices[row]["id"]); self._refresh()
