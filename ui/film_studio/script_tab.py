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

# Stadio «Racconto cinematografico»: prosa narrativa PRIMA del breakdown tecnico (voce, tono, atmosfera).
_SYS_NARRATIVE = ("You are a narrative film writer. From the user's idea write a short CINEMATIC PROSE treatment "
                  "(a 'racconto') in the SAME language: evocative prose describing what we SEE and FEEL, the "
                  "tone and atmosphere, NOT a technical scene list. Keep @Name references intact. 1-3 paragraphs.")


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
        # stili (~60) e generi (~23) — spec §9.1 (multi-selezione via menu a tendina con checkbox)
        from core.film_studio import style_catalog as SC
        row2 = QHBoxLayout()
        self.styles = QComboBox(); self.styles.setEditable(True)
        self.styles.addItems([""] + SC.STYLES)
        self.styles.setToolTip(tr("fs.script.styles_tt", "Stile visivo (puoi scriverne più separati da virgola)"))
        self.genres = QComboBox(); self.genres.setEditable(True)
        self.genres.addItems([""] + SC.GENRES)
        self.genres.setToolTip(tr("fs.script.genres_tt", "Genere/i (puoi scriverne più separati da virgola)"))
        row2.addWidget(QLabel(tr("fs.script.styles", "Stile"))); row2.addWidget(self.styles, 1)
        row2.addWidget(QLabel(tr("fs.script.genres", "Genere"))); row2.addWidget(self.genres, 1)
        bv.addLayout(row2)
        self.inp = QPlainTextEdit(); self.inp.setPlaceholderText(
            tr("fs.script.input_ph", "Incolla la storia, l'idea o il tema… (@Nome per gli asset)"))
        self.inp.setFixedHeight(120)
        bv.addWidget(self.inp)
        from ui.film_studio.asset_dialog import attach as _attach_at
        _attach_at(self.inp)                              # digitando @ si apre l'AssetDialog
        h = QHBoxLayout()
        self.btn_table = QPushButton(tr("fs.script.table", "🎲 Tavolo (bivi/beat)"))
        self.btn_table.setToolTip(tr("fs.script.table_tt", "Co-scrittura interattiva: l'LLM propone i beat della "
                                     "storia con bivi alternativi; scegli un ramo per beat e li intreccia nel Racconto."))
        self.btn_table.clicked.connect(self._cowrite_table)
        self.btn_racconto = QPushButton(tr("fs.script.gen_narr", "📖 Genera Racconto")); self.btn_racconto.clicked.connect(self._gen_racconto)
        self.btn_gen = QPushButton(tr("fs.script.gen", "✍️ Genera copione")); self.btn_gen.clicked.connect(self._gen)
        self.btn_regen = QPushButton(tr("fs.script.regen", "↻ Rigenera")); self.btn_regen.clicked.connect(self._regen)
        h.addWidget(self.btn_table); h.addWidget(self.btn_racconto); h.addWidget(self.btn_gen); h.addWidget(self.btn_regen); h.addStretch(1)
        bv.addLayout(h)
        v.addWidget(box)
        v.addWidget(QLabel(tr("fs.script.racconto", "Racconto cinematografico (prosa, editabile)")))
        self.racconto = QPlainTextEdit(); self.racconto.setFixedHeight(90)
        from ui.film_studio.regen_menu import attach_regen_menu as _arm
        _arm(self.racconto); _attach_at(self.racconto)
        v.addWidget(self.racconto)
        v.addWidget(QLabel(tr("fs.script.summary", "Riassunto trama")))
        self.summary = QPlainTextEdit(); self.summary.setFixedHeight(60)
        v.addWidget(self.summary)
        v.addWidget(QLabel(tr("fs.script.text", "Copione (editabile)")))
        self.text = QPlainTextEdit()
        from ui.film_studio.regen_menu import attach_regen_menu
        attach_regen_menu(self.text)                         # tasto destro su selezione → Rigenera segmento
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
        narr = self.racconto.toPlainText().strip() if hasattr(self, "racconto") else ""
        narr = f"\n\nRacconto cinematografico:\n{narr}" if narr and narr != tr("fs.working", "Generazione…") else ""
        return f"[{self.kind.currentData()}] " + self.inp.toPlainText().strip() + narr + meta

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

    def _gen_racconto(self):
        """Stadio Racconto: genera la prosa cinematografica dall'idea (prima del copione tecnico)."""
        if not self.inp.toPlainText().strip():
            QMessageBox.information(self, tr("fs.script.gen_narr", "Racconto"), tr("fs.script.need_input", "Inserisci un input.")); return
        if not L.available():
            QMessageBox.warning(self, tr("fs.script.gen_narr", "Racconto"), tr("fs.llm.unavailable", "LLM non disponibile.")); return
        self.btn_racconto.setEnabled(False)
        self.racconto.setPlainText(tr("fs.working", "Generazione…"))
        self._narr_worker = _LlmWorker(_SYS_NARRATIVE, self._build_user())
        self._narr_worker.done.connect(self._on_racconto); self._narr_worker.failed.connect(self._on_racconto_fail)
        self._narr_worker.start()

    def _on_racconto(self, txt: str):
        self.btn_racconto.setEnabled(True)
        self.racconto.setPlainText(txt.strip())
        pid = self._get_project()
        if pid:                                              # salva subito come Racconto (kind='narrative')
            db.save_script(pid, txt.strip(), summary="racconto", kind="narrative")

    def _on_racconto_fail(self, err: str):
        self.btn_racconto.setEnabled(True)
        QMessageBox.critical(self, tr("fs.script.gen_narr", "Racconto"), str(err)[:600])

    # ---- Co-scrittura interattiva: tavolo a beat/bivi ----
    def _cowrite_table(self):
        """Genera i beat con bivi (worker) e apre il tavolo di scelta."""
        if not self.inp.toPlainText().strip():
            QMessageBox.information(self, tr("fs.script.table", "Tavolo"), tr("fs.script.need_input", "Inserisci un input.")); return
        if not L.available():
            QMessageBox.warning(self, tr("fs.script.table", "Tavolo"), tr("fs.llm.unavailable", "LLM non disponibile.")); return
        self.btn_table.setEnabled(False)
        idea = self.inp.toPlainText().strip()
        styles, genres = self.styles.currentText().strip(), self.genres.currentText().strip()
        from PySide6.QtCore import QThread, Signal

        class _BeatsWorker(QThread):
            done = Signal(list)

            def run(self):
                from core.film_studio import cowriter
                try:
                    self.done.emit(cowriter.propose_beats(idea, styles, genres))
                except Exception:                            # noqa: BLE001
                    self.done.emit([])
        self.summary.setPlainText(tr("fs.working", "Generazione…"))
        self._beats_w = _BeatsWorker(); self._beats_w.done.connect(self._show_table); self._beats_w.start()

    def _show_table(self, beats: list):
        self.btn_table.setEnabled(True)
        self.summary.setPlainText("")
        if not beats:
            QMessageBox.information(self, tr("fs.script.table", "Tavolo"),
                                   tr("fs.script.table_none", "Nessun beat proposto (LLM non disponibile o risposta vuota).")); return
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QScrollArea, QWidget as _W,
                                       QGroupBox, QRadioButton, QButtonGroup, QLabel as _L)
        dlg = QDialog(self); dlg.setWindowTitle(tr("fs.script.table", "🎲 Tavolo di co-scrittura")); dlg.resize(640, 620)
        outer = QVBoxLayout(dlg)
        outer.addWidget(_L(tr("fs.script.table_head", "Scegli un ramo per ogni beat: verranno intrecciati nel Racconto.")))
        area = QScrollArea(); area.setWidgetResizable(True); inner = _W(); iv = QVBoxLayout(inner)
        groups = []
        for bi, b in enumerate(beats):
            gb = QGroupBox(f"{bi + 1}. {b['beat']}"); gv = QVBoxLayout(gb)
            bg = QButtonGroup(dlg)
            for oi, o in enumerate(b["options"]):
                rb = QRadioButton(f"{o['label']} — {o['text']}")
                if oi == 0:
                    rb.setChecked(True)
                bg.addButton(rb, oi); gv.addWidget(rb)
            iv.addWidget(gb); groups.append((b, bg))
        iv.addStretch(1); area.setWidget(inner); outer.addWidget(area, 1)
        bb = QDialogButtonBox(); bb.addButton(tr("fs.script.table_weave", "Intreccia nel Racconto"), QDialogButtonBox.AcceptRole)
        bb.addButton(QDialogButtonBox.Cancel); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        outer.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        chosen = []
        for b, bg in groups:
            oi = bg.checkedId() if bg.checkedId() >= 0 else 0
            o = b["options"][oi] if oi < len(b["options"]) else b["options"][0]
            chosen.append({"beat": b["beat"], "label": o["label"], "text": o["text"]})
        self._assemble_chosen(chosen)

    def _assemble_chosen(self, chosen: list):
        """Intreccia i rami scelti in un racconto (worker) → campo Racconto."""
        self.btn_table.setEnabled(False); self.racconto.setPlainText(tr("fs.working", "Generazione…"))
        idea = self.inp.toPlainText().strip()
        styles, genres = self.styles.currentText().strip(), self.genres.currentText().strip()
        from PySide6.QtCore import QThread, Signal

        class _AsmWorker(QThread):
            done = Signal(str)

            def run(self):
                from core.film_studio import cowriter
                try:
                    self.done.emit(cowriter.assemble(idea, chosen, styles, genres))
                except Exception:                            # noqa: BLE001
                    self.done.emit("")
        self._asm_w = _AsmWorker(); self._asm_w.done.connect(self._on_assembled); self._asm_w.start()

    def _on_assembled(self, txt: str):
        self.btn_table.setEnabled(True)
        if not txt:
            self.racconto.setPlainText("")
            QMessageBox.information(self, tr("fs.script.table", "Tavolo"), tr("fs.script.table_fail", "Racconto non generato.")); return
        self.racconto.setPlainText(txt.strip())
        pid = self._get_project()
        if pid:
            db.save_script(pid, txt.strip(), summary="racconto", kind="narrative")

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
        db.update("project", pid, styles=self.styles.currentText().strip(), genres=self.genres.currentText().strip())
        if self.racconto.toPlainText().strip():              # salva anche il Racconto se presente
            db.save_script(pid, self.racconto.toPlainText().strip(), summary="racconto", kind="narrative")
        rid = db.save_script(pid, self.text.toPlainText(), summary=self.summary.toPlainText(), kind="script")
        n = (db.get("script", rid) or {}).get("version", 1)
        self.script_ready.emit(pid)
        QMessageBox.information(self, tr("fs.save", "Salva"), tr("fs.script.saved", "Copione salvato (v{n}).").format(n=n))
