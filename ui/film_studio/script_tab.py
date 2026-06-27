"""Tab Copione (spec §9): 4 input → formato cinematografico via LLM. Riassunto trama + editing +
rigenerazione segmento/totale + versioning."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from core import llm_service as L
from core.film_studio import db
from ui.i18n_manager import tr

_INPUTS = [("full", "fs.script.in.full"), ("idea", "fs.script.in.idea"),
           ("theme", "fs.script.in.theme"), ("interactive", "fs.script.in.interactive")]

_SYS = ("You are a screenwriter. Turn the user's input into a cinematic treatment in the SAME language "
        "as the input. Start with a 2-3 line PLOT SUMMARY, then the structured treatment in scenes. "
        "Respect the requested duration and genres. Keep @Name references intact if present.")


class _LlmWorker(QThread):
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, system: str, user: str):
        super().__init__()
        self.system, self.user = system, user

    def run(self):
        try:
            self.done.emit(L.complete(self.system, self.user, max_tokens=2000))
        except Exception as e:                              # noqa: BLE001
            self.failed.emit(str(e))


class ScriptTab(QWidget):
    script_ready = Signal(int)          # project_id

    def __init__(self, get_project):
        super().__init__()
        self._get_project = get_project           # callable → project_id corrente
        self._worker = None
        v = QVBoxLayout(self)
        box = QGroupBox(tr("fs.script.input", "Input storia"))
        bv = QVBoxLayout(box)
        row = QHBoxLayout()
        self.kind = QComboBox()
        for code, key in _INPUTS:
            self.kind.addItem(tr(key, code), code)
        row.addWidget(QLabel(tr("fs.script.kind", "Tipo input")))
        row.addWidget(self.kind, 1)
        bv.addLayout(row)
        self.inp = QPlainTextEdit(); self.inp.setPlaceholderText(
            tr("fs.script.input_ph", "Incolla la storia, l'idea o il tema… (@Nome per gli asset)"))
        self.inp.setFixedHeight(120)
        bv.addWidget(self.inp)
        from ui.film_studio.asset_dialog import attach as _attach_at
        _attach_at(self.inp)                              # digitando @ si apre l'AssetDialog
        h = QHBoxLayout()
        self.btn_gen = QPushButton(tr("fs.script.gen", "✍️ Genera copione")); self.btn_gen.clicked.connect(self._gen)
        self.btn_regen = QPushButton(tr("fs.script.regen", "↻ Rigenera")); self.btn_regen.clicked.connect(self._regen)
        h.addWidget(self.btn_gen); h.addWidget(self.btn_regen); h.addStretch(1)
        bv.addLayout(h)
        v.addWidget(box)
        v.addWidget(QLabel(tr("fs.script.summary", "Riassunto trama")))
        self.summary = QPlainTextEdit(); self.summary.setFixedHeight(60)
        v.addWidget(self.summary)
        v.addWidget(QLabel(tr("fs.script.text", "Copione (editabile)")))
        self.text = QPlainTextEdit()
        v.addWidget(self.text, 1)
        h2 = QHBoxLayout()
        self.btn_save = QPushButton(tr("fs.save", "💾 Salva")); self.btn_save.clicked.connect(self._save)
        h2.addStretch(1); h2.addWidget(self.btn_save)
        v.addLayout(h2)

    def _build_user(self) -> str:
        pid = self._get_project()
        proj = db.get("project", pid) if pid else None
        meta = ""
        if proj:
            meta = f"\n\nTarget duration: {proj.get('duration_s',60)}s. Genres: {proj.get('genres','')}. " \
                   f"Styles: {proj.get('styles','')}."
        return f"[{self.kind.currentData()}] " + self.inp.toPlainText().strip() + meta

    def _gen(self):
        if not self.inp.toPlainText().strip():
            QMessageBox.information(self, tr("fs.script.gen", "Copione"),
                                   tr("fs.script.need_input", "Inserisci un input.")); return
        if not L.available():
            QMessageBox.warning(self, tr("fs.script.gen", "Copione"),
                               tr("fs.llm.unavailable", "LLM non disponibile (scegli Gemma o connetti LM Studio).")); return
        self.btn_gen.setEnabled(False); self.btn_regen.setEnabled(False)
        self.summary.setPlainText(tr("fs.working", "Generazione…"))
        self._worker = _LlmWorker(_SYS, self._build_user())
        self._worker.done.connect(self._on_done); self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _regen(self):
        self.inp.setPlainText(self.inp.toPlainText() + "\n(varia con un seed diverso, alternativa)")
        self._gen()

    def _on_done(self, txt: str):
        self.btn_gen.setEnabled(True); self.btn_regen.setEnabled(True)
        lines = txt.strip().split("\n\n", 1)
        self.summary.setPlainText(lines[0].strip())
        self.text.setPlainText(lines[1].strip() if len(lines) > 1 else txt.strip())

    def _on_fail(self, e: str):
        self.btn_gen.setEnabled(True); self.btn_regen.setEnabled(True)
        self.summary.setPlainText("")
        QMessageBox.critical(self, tr("fs.script.gen", "Copione"), e[:600])

    def _save(self):
        pid = self._get_project()
        if not pid:
            QMessageBox.information(self, tr("fs.save", "Salva"), tr("fs.no_project", "Crea/seleziona un progetto.")); return
        n = len(db.query("script", "project_id=?", (pid,))) + 1
        db.insert("script", project_id=pid, text=self.text.toPlainText(),
                  summary=self.summary.toPlainText(), version=n)
        self.script_ready.emit(pid)
        QMessageBox.information(self, tr("fs.save", "Salva"), tr("fs.script.saved", "Copione salvato (v{n}).").format(n=n))
