"""Widget selezione provider LLM (Gemma locale ↔ LM Studio) — da usare PRIMA di generare.

LM Studio: porta modificabile (default 1234), bottone Connetti (auto-launch se non attivo) → popola
la lista modelli da /v1/models; l'utente sceglie il modello. La scelta è persistita (core/llm_service).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from core import llm_service as L
from ui.i18n_manager import tr


class _ConnectWorker(QThread):
    done = Signal(bool, list)

    def __init__(self, port: int):
        super().__init__()
        self.port = port

    def run(self):
        ok = L.ensure_lmstudio(self.port, timeout=45)
        self.done.emit(ok, L.list_models(self.port) if ok else [])


class LlmSettings(QGroupBox):
    """GroupBox riusabile per scegliere il motore LLM del Film Studio."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(tr("fs.llm.title", "Motore LLM (testo)"), parent)
        cfg = L.get_config()
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.provider = QComboBox()
        self.provider.addItem(tr("fs.llm.gemma", "Gemma locale"), "gemma")
        self.provider.addItem(tr("fs.llm.lmstudio", "LM Studio"), "lmstudio")
        self.provider.setCurrentIndex(0 if cfg["provider"] == "gemma" else 1)
        self.provider.currentIndexChanged.connect(self._on_provider)
        row.addWidget(QLabel(tr("fs.llm.provider", "Provider")))
        row.addWidget(self.provider)
        self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(int(cfg["lmstudio_port"]))
        row.addWidget(QLabel(tr("fs.llm.port", "Porta")))
        row.addWidget(self.port)
        self.btn_connect = QPushButton(tr("fs.llm.connect", "Connetti"))
        self.btn_connect.clicked.connect(self._connect)
        row.addWidget(self.btn_connect)
        row.addStretch(1)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.model = QComboBox(); self.model.setMinimumWidth(280)
        if cfg.get("lmstudio_model"):
            self.model.addItem(cfg["lmstudio_model"], cfg["lmstudio_model"])
        self.model.currentIndexChanged.connect(self._on_model)
        row2.addWidget(QLabel(tr("fs.llm.model", "Modello")))
        row2.addWidget(self.model, 1)
        self.status = QLabel("")
        row2.addWidget(self.status)
        v.addLayout(row2)
        self._sync_enabled()

    def _on_provider(self):
        L.set_provider(self.provider.currentData())
        self._sync_enabled()
        self.changed.emit()

    def _sync_enabled(self):
        is_lms = self.provider.currentData() == "lmstudio"
        for w in (self.port, self.btn_connect, self.model):
            w.setEnabled(is_lms)
        self.status.setText("" if is_lms else tr("fs.llm.gemma_ready", "Gemma locale pronta"))

    def _connect(self):
        L.set_lmstudio(port=int(self.port.value()))
        self.btn_connect.setEnabled(False)
        self.status.setText(tr("fs.llm.connecting", "Connessione/avvio LM Studio…"))
        self._w = _ConnectWorker(int(self.port.value()))
        self._w.done.connect(self._on_connected)
        self._w.start()

    def _on_connected(self, ok: bool, models: list):
        self.btn_connect.setEnabled(True)
        if not ok:
            self.status.setText(tr("fs.llm.fail", "LM Studio non raggiungibile"))
            return
        self.model.clear()
        for m in models:
            self.model.addItem(m, m)
        self.status.setText(tr("fs.llm.connected", "Connesso: {n} modelli").format(n=len(models)))
        if models:
            L.set_lmstudio(model=models[0])
        self.changed.emit()

    def _on_model(self):
        if self.model.currentData():
            L.set_lmstudio(model=self.model.currentData())
            self.changed.emit()
