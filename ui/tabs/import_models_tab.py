"""Tab «Import Models» — interfaccia per gli strumenti del catalogo AI.

Pipeline (E:\\AI\\dictionary\\tools, vedi core/catalog_tooling.py):
  Inbox → Sposta (place_files) → Scansiona (scan) → Arricchisci (Civitai + override)
        → Genera HTML (build_html).  «Ingest completo» esegue tutto in sequenza.

Percorsi configurabili (cartella modelli, workflow, dizionario, inbox, strumenti):
default = installazione attuale. Al salvataggio vengono sincronizzati in
<tools_dir>/paths.json, letto dagli script. L'installer futuro proporrà percorsi in
base a ComfyUI ma lascerà la decisione all'utente.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from threading import Thread

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar, QFileDialog, QSplitter,
    QGroupBox, QScrollArea, QFrame, QMessageBox, QToolButton, QCheckBox,
)

from core import catalog_paths
from core import comfy_compat
from core.catalog_tooling import CatalogTooling


class _DropArea(QLabel):
    """Area drag-and-drop che accetta file e cartelle."""
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__("📂  Trascina qui file o cartelle di modelli da importare")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(64)
        self._base = ("border: 2px dashed {c}; color:#999; font-size:13px; "
                      "background:#1a1a1a; border-radius:6px; padding:10px;")
        self.setStyleSheet(self._base.format(c="#555"))
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(self._base.format(c="#74c0fc"))
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self.setStyleSheet(self._base.format(c="#555"))

    def dropEvent(self, e):
        self.setStyleSheet(self._base.format(c="#555"))
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
        e.acceptProposedAction()


class ImportModelsTab(QWidget):
    """Importa e cataloga modelli AI tramite gli strumenti del dizionario."""

    catalog_updated = Signal()   # emesso dopo enrich/build/ingest → ricarica catalogo LoRA

    _MODEL_FILTER = ("Modelli AI (*.safetensors *.ckpt *.pt *.pth *.gguf *.bin *.pkl);;"
                     "Tutti i file (*.*)")

    def __init__(self):
        super().__init__()
        self._pending: list[str] = []
        self._tooling: CatalogTooling | None = None
        self._path_edits: dict[str, QLineEdit] = {}
        self._build()
        self._load_paths_into_ui()
        self._check_tools()
        self._refresh_compat_status()

    # ============================================================ build
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        split = QSplitter(Qt.Vertical)

        # --- parte alta: controlli scrollabili -------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        host = QWidget()
        v = QVBoxLayout(host)
        v.setSpacing(8)

        intro = QLabel(
            "<b>Import Models</b> — cataloga i tuoi modelli AI. Flusso: metti i file "
            "nell'<b>inbox</b>, poi <b>Sposta</b> (li ordina per tipo/architettura), "
            "<b>Scansiona</b> (estrae i metadati), <b>Arricchisci</b> (dati da Civitai), "
            "<b>Genera HTML</b> (pagina del catalogo). «Ingest completo» fa tutto in sequenza."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#bbb; padding:2px;")
        v.addWidget(intro)

        v.addWidget(self._build_paths_group())
        v.addWidget(self._build_compat_group())
        v.addWidget(self._build_source_group())
        v.addWidget(self._build_ops_group())

        host.setLayout(v)
        scroll.setWidget(host)
        split.addWidget(scroll)

        # --- parte bassa: log + progress ------------------------------------
        log_w = QWidget()
        lv = QVBoxLayout(log_w)
        lv.setContentsMargins(0, 4, 0, 0)
        lv.addWidget(QLabel("Output operazioni:"))
        self.log_txt = QPlainTextEdit()
        self.log_txt.setReadOnly(True)
        self.log_txt.setStyleSheet("font-family: Consolas, monospace; font-size:11px;")
        lv.addWidget(self.log_txt, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)         # indeterminato durante un run
        self.progress.setVisible(False)
        lv.addWidget(self.progress)
        split.addWidget(log_w)

        split.setSizes([520, 240])
        outer.addWidget(split)

    # ---- gruppo configurazione percorsi ----
    def _build_paths_group(self) -> QGroupBox:
        grp = QGroupBox("📁 Percorsi (configurabili)")
        grp.setCheckable(True)
        grp.setChecked(False)           # collassato di default (default = installazione attuale)
        outer = QVBoxLayout(grp)

        note = QLabel(
            "I valori predefiniti corrispondono all'installazione attuale. Modificali solo se hai "
            "spostato ComfyUI, i modelli o il dizionario. Al salvataggio vengono sincronizzati negli "
            "strumenti (paths.json)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:11px;")
        outer.addWidget(note)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        for row, (key, label, desc) in enumerate(catalog_paths.LABELS):
            lab = QLabel(label + ":")
            lab.setToolTip(desc)
            edit = QLineEdit()
            edit.setToolTip(desc)
            edit.setPlaceholderText(catalog_paths.DEFAULTS.get(key, ""))
            self._path_edits[key] = edit
            browse = QToolButton()
            browse.setText("…")
            browse.setToolTip("Sfoglia")
            browse.clicked.connect(lambda _=False, k=key: self._browse_path(k))
            grid.addWidget(lab, row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(browse, row, 2)
        outer.addLayout(grid)

        btns = QHBoxLayout()
        save_btn = QPushButton("💾 Salva percorsi")
        save_btn.clicked.connect(self._save_paths)
        reset_btn = QPushButton("↺ Ripristina default")
        reset_btn.clicked.connect(self._reset_paths)
        suggest_btn = QPushButton("🔍 Suggerisci da ComfyUI")
        suggest_btn.setToolTip("Propone modelli/workflow in base all'installazione ComfyUI rilevata "
                               "(decisione finale tua: nulla viene salvato finché non premi «Salva»).")
        suggest_btn.clicked.connect(self._suggest_paths)
        btns.addWidget(save_btn)
        btns.addWidget(reset_btn)
        btns.addWidget(suggest_btn)
        btns.addStretch(1)
        outer.addLayout(btns)

        self._paths_status = QLabel("")
        self._paths_status.setWordWrap(True)
        self._paths_status.setStyleSheet("color:#a9e34b; font-size:11px;")
        outer.addWidget(self._paths_status)
        return grp

    # ---- gruppo compatibilità ComfyUI (patch LTX-2.3) ----
    def _build_compat_group(self) -> QGroupBox:
        grp = QGroupBox("🧩 Compatibilità ComfyUI (patch LTX-2.3)")
        v = QVBoxLayout(grp)

        note = QLabel(
            "Installa una patch di compatibilità come custom node di ComfyUI: abilita LTX-2.3 "
            "su qualsiasi versione di ComfyUI <b>senza modificare i file core</b> (sopravvive agli "
            "aggiornamenti). Necessaria per la generazione video. Dopo l'installazione, <b>riavvia "
            "ComfyUI</b> per applicarla."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(note)

        self._compat_status = QLabel("")
        self._compat_status.setWordWrap(True)
        self._compat_status.setStyleSheet("font-size:11px; padding:2px;")
        v.addWidget(self._compat_status)

        self.chk_prompt_unlock = QCheckBox(
            "Sblocca il prompt-enhancer LTX-2 (descrive i soggetti senza rifiuti, per contenuti NSFW)")
        self.chk_prompt_unlock.setChecked(True)
        self.chk_prompt_unlock.setToolTip(
            "Inietta una direttiva nel system prompt dell'enhancer LTX-2 così non rifiuta di descrivere "
            "le persone. Se installato, la modifica è effettiva al riavvio di ComfyUI.")
        self.chk_prompt_unlock.toggled.connect(self._on_prompt_unlock_toggled)
        v.addWidget(self.chk_prompt_unlock)

        btns = QHBoxLayout()
        self.btn_compat_install = QPushButton("🧩 Installa / Aggiorna patch")
        self.btn_compat_install.clicked.connect(self._do_install_compat)
        self.btn_compat_uninstall = QPushButton("✖ Disinstalla patch")
        self.btn_compat_uninstall.clicked.connect(self._do_uninstall_compat)
        btns.addWidget(self.btn_compat_install)
        btns.addWidget(self.btn_compat_uninstall)
        btns.addStretch(1)
        v.addLayout(btns)
        return grp

    def _refresh_compat_status(self):
        try:
            st = comfy_compat.status()
        except Exception as e:
            self._compat_status.setText(f"⚠️ Stato non disponibile: {e}")
            return
        if not st["comfy_valid"]:
            self._compat_status.setText(
                f"⚠️ ComfyUI non trovato in: {st['comfy_root']} (manca main.py). "
                "Configura «Cartella ComfyUI» nei Percorsi.")
            self.btn_compat_install.setEnabled(False)
            self.btn_compat_uninstall.setEnabled(False)
            return
        self.btn_compat_install.setEnabled(st["repo_shim_present"])
        self.btn_compat_uninstall.setEnabled(st["installed"])
        if st["installed"]:
            agg = "aggiornata" if st["up_to_date"] else f"DA AGGIORNARE (installata {st['installed_version']} < repo {st['repo_version']})"
            self._compat_status.setText(
                f"✅ Patch installata v{st['installed_version']} ({agg}). "
                f"Prompt-enhancer NSFW: {'ON' if st['prompt_unlock'] else 'OFF'}. "
                "Riavvia ComfyUI per applicare eventuali modifiche.")
            self.chk_prompt_unlock.blockSignals(True)
            self.chk_prompt_unlock.setChecked(st["prompt_unlock"])
            self.chk_prompt_unlock.blockSignals(False)
        else:
            self._compat_status.setText(
                f"⛔ Patch NON installata (repo v{st['repo_version']}). "
                "Premi «Installa» e riavvia ComfyUI.")

    def _do_install_compat(self):
        try:
            comfy_compat.install(prompt_unlock=self.chk_prompt_unlock.isChecked())
            self._log("\n🧩 Patch di compatibilità installata/aggiornata in custom_nodes. "
                      "RIAVVIA ComfyUI per applicarla.")
        except Exception as e:
            QMessageBox.critical(self, "Errore installazione patch", str(e))
            self._log(f"[ERR] install patch: {e}")
        self._refresh_compat_status()

    def _do_uninstall_compat(self):
        if QMessageBox.question(self, "Disinstalla patch",
                                "Rimuovere la patch di compatibilità dai custom_nodes di ComfyUI?\n"
                                "(LTX-2.3 potrebbe non funzionare finché non la reinstalli o "
                                "ripristini le modifiche ai file core.)") != QMessageBox.Yes:
            return
        try:
            comfy_compat.uninstall()
            self._log("\n🧩 Patch disinstallata. Riavvia ComfyUI per applicare.")
        except Exception as e:
            QMessageBox.critical(self, "Errore disinstallazione", str(e))
        self._refresh_compat_status()

    def _on_prompt_unlock_toggled(self, checked: bool):
        try:
            st = comfy_compat.status()
            if st["installed"]:
                comfy_compat.set_prompt_unlock(checked)
                self._log(f"🧩 Prompt-enhancer NSFW {'ON' if checked else 'OFF'} "
                          "(effettivo al riavvio di ComfyUI).")
                self._refresh_compat_status()
        except Exception as e:
            self._log(f"[ERR] prompt_unlock: {e}")

    # ---- gruppo sorgente / staging ----
    def _build_source_group(self) -> QGroupBox:
        grp = QGroupBox("📥 File da importare")
        v = QVBoxLayout(grp)

        # sorgente
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Sorgente operazioni:"))
        self.source_edit = QLineEdit()
        self.source_edit.setToolTip("Cartella su cui agiscono «Analizza» e «Sposta» (e «Ingest»). "
                                    "Di default è l'inbox; puoi puntarla a una cartella di download.")
        srow.addWidget(self.source_edit, 1)
        src_browse = QPushButton("📁 Sfoglia…")
        src_browse.clicked.connect(self._browse_source)
        srow.addWidget(src_browse)
        v.addLayout(srow)

        # drop area
        self.drop = _DropArea()
        self.drop.files_dropped.connect(self._add_paths)
        v.addWidget(self.drop)

        # lista + azioni
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setMaximumHeight(120)
        v.addWidget(self.file_list)

        lb = QHBoxLayout()
        b_files = QPushButton("📂 Aggiungi file…")
        b_files.clicked.connect(self._browse_files)
        b_dir = QPushButton("📁 Aggiungi cartella…")
        b_dir.clicked.connect(self._browse_adddir)
        self.b_to_inbox = QPushButton("📥 Copia nell'inbox")
        self.b_to_inbox.setToolTip("Copia i file elencati nell'inbox configurata, pronti per «Sposta».")
        self.b_to_inbox.clicked.connect(self._copy_to_inbox)
        b_rm = QPushButton("✖ Rimuovi")
        b_rm.clicked.connect(self._remove_selected)
        b_clr = QPushButton("🗑 Pulisci")
        b_clr.clicked.connect(self._clear_list)
        for b in (b_files, b_dir, self.b_to_inbox, b_rm, b_clr):
            lb.addWidget(b)
        lb.addStretch(1)
        v.addLayout(lb)
        return grp

    # ---- gruppo operazioni ----
    def _build_ops_group(self) -> QGroupBox:
        grp = QGroupBox("⚙️ Operazioni")
        v = QVBoxLayout(grp)

        row = QHBoxLayout()
        self.btn_analyze = self._op_btn("🔍 Analizza", "Classifica i file della sorgente per "
                                        "tipo/architettura SENZA spostarli (dry-run).", self._do_analyze)
        self.btn_place = self._op_btn("📦 Sposta", "Sposta i file dalla sorgente nelle cartelle "
                                      "giuste (LoRA/Checkpoints/VAE/… o workflows).", self._do_place)
        self.btn_scan = self._op_btn("🗂 Scansiona", "Aggiorna catalog.json con i metadati di base "
                                     "(architettura, trigger, NSFW). Nessuna rete.", self._do_scan)
        self.btn_enrich = self._op_btn("🌐 Arricchisci", "Lookup Civitai by-hash + riapplica gli "
                                       "override manuali. Richiede internet.", self._do_enrich)
        self.btn_overrides = self._op_btn("✍️ Override", "Riapplica solo gli override manuali "
                                          "(manual_enrichment.json), senza rete.", self._do_overrides)
        self.btn_build = self._op_btn("🖼 Genera HTML", "Rigenera il catalogo HTML statico "
                                      "(index.html + pagine entries/).", self._do_build)
        for b in (self.btn_analyze, self.btn_place, self.btn_scan, self.btn_enrich,
                  self.btn_overrides, self.btn_build):
            row.addWidget(b)
        v.addLayout(row)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color:#333;")
        v.addWidget(sep)

        irow = QHBoxLayout()
        self.btn_ingest = QPushButton("▶  Ingest completo  (Sposta → Scansiona → Arricchisci → HTML)")
        self.btn_ingest.setObjectName("generate")
        self.btn_ingest.setMinimumHeight(40)
        self.btn_ingest.setToolTip("Esegue l'intera pipeline in sequenza sulla sorgente. "
                                   "Al termine il catalogo LoRA dell'app viene ricaricato.")
        self.btn_ingest.clicked.connect(self._do_ingest)
        irow.addWidget(self.btn_ingest)
        v.addLayout(irow)

        self._tools_note = QLabel("")
        self._tools_note.setWordWrap(True)
        self._tools_note.setStyleSheet("color:#888; font-size:11px; padding:2px;")
        v.addWidget(self._tools_note)
        return grp

    @staticmethod
    def _op_btn(label: str, tip: str, slot) -> QPushButton:
        b = QPushButton(label)
        b.setToolTip(tip)
        b.setMinimumHeight(34)
        b.clicked.connect(slot)
        return b

    # ============================================================ percorsi
    def _load_paths_into_ui(self):
        cfg = catalog_paths.load()
        for key, edit in self._path_edits.items():
            edit.setText(cfg.get(key, ""))
        self.source_edit.setText(cfg.get("inbox", ""))

    def _browse_path(self, key: str):
        cur = self._path_edits[key].text().strip() or catalog_paths.DEFAULTS.get(key, "")
        if key == "hashes_jsonl":
            fn, _ = QFileDialog.getOpenFileName(self, "Scegli file hashes.jsonl", cur,
                                                "JSON Lines (*.jsonl);;Tutti i file (*.*)")
            if fn:
                self._path_edits[key].setText(fn)
        else:
            d = QFileDialog.getExistingDirectory(self, "Scegli cartella", cur)
            if d:
                self._path_edits[key].setText(d)

    def _save_paths(self):
        cfg = {k: e.text().strip() for k, e in self._path_edits.items()}
        out = catalog_paths.save(cfg)
        self.source_edit.setText(cfg.get("inbox") or catalog_paths.DEFAULTS["inbox"])
        if out:
            self._paths_status.setText(f"✅ Percorsi salvati e sincronizzati: {out}")
        else:
            self._paths_status.setText(
                "⚠️ Percorsi salvati in LTX Studio, ma la cartella strumenti non esiste: "
                "gli script non sono stati sincronizzati. Verifica «Cartella strumenti».")
        self._check_tools()
        self._refresh_compat_status()

    def _reset_paths(self):
        for key, edit in self._path_edits.items():
            edit.setText(catalog_paths.DEFAULTS.get(key, ""))
        self._paths_status.setText("Ripristinati i default (premi «Salva percorsi» per applicarli).")

    def _suggest_paths(self):
        comfy = catalog_paths.detect_comfy_root()
        if not comfy:
            d = QFileDialog.getExistingDirectory(self, "Seleziona la cartella di ComfyUI (radice)")
            if not d:
                return
            comfy = Path(d)
        sug = catalog_paths.suggest_from_comfy(comfy)
        for key, val in sug.items():
            if key in self._path_edits:
                self._path_edits[key].setText(val)
        self._paths_status.setText(
            f"Suggeriti da ComfyUI ({comfy}). Controlla e premi «Salva percorsi» per applicare. "
            "La decisione resta tua.")

    # ============================================================ sorgente/lista
    def _browse_source(self):
        d = QFileDialog.getExistingDirectory(self, "Cartella sorgente",
                                             self.source_edit.text().strip() or "")
        if d:
            self.source_edit.setText(d)

    def _add_paths(self, paths: list[str]):
        have = {self.file_list.item(i).data(Qt.UserRole) for i in range(self.file_list.count())}
        for p in paths:
            if p in have:
                continue
            it = QListWidgetItem(("📁 " if Path(p).is_dir() else "📄 ") + Path(p).name)
            it.setData(Qt.UserRole, p)
            it.setToolTip(p)
            self.file_list.addItem(it)
            self._pending.append(p)

    def _browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Scegli file modelli",
                                                self._path_edits["models"].text().strip() or "",
                                                self._MODEL_FILTER)
        if paths:
            self._add_paths(paths)

    def _browse_adddir(self):
        d = QFileDialog.getExistingDirectory(self, "Scegli cartella di modelli")
        if d:
            self._add_paths([d])

    def _remove_selected(self):
        for it in self.file_list.selectedItems():
            p = it.data(Qt.UserRole)
            if p in self._pending:
                self._pending.remove(p)
            self.file_list.takeItem(self.file_list.row(it))

    def _clear_list(self):
        self.file_list.clear()
        self._pending.clear()

    def _copy_to_inbox(self):
        if not self._pending:
            self._log("[warn] Nessun file in lista da copiare."); return
        inbox = Path(self._path_edits["inbox"].text().strip() or catalog_paths.DEFAULTS["inbox"])
        items = list(self._pending)
        self._set_busy(True)
        self._log(f"\n📥 Copia di {len(items)} elementi in {inbox} …")

        def _work():
            n = 0
            try:
                inbox.mkdir(parents=True, exist_ok=True)
                for p in items:
                    sp = Path(p)
                    try:
                        if sp.is_dir():
                            for f in sp.iterdir():
                                if f.is_file():
                                    shutil.copy2(f, inbox / f.name); n += 1
                                    self._log(f"  ✓ {f.name}")
                        elif sp.is_file():
                            shutil.copy2(sp, inbox / sp.name); n += 1
                            self._log(f"  ✓ {sp.name}")
                    except Exception as e:
                        self._log(f"  ✗ {sp.name}: {e}")
                self._log(f"✅ Copiati {n} file in {inbox}. La sorgente è impostata sull'inbox.")
            finally:
                self.source_edit.setText(str(inbox))
                self._set_busy(False)

        Thread(target=_work, daemon=True).start()

    # ============================================================ operazioni
    def _make_tooling(self) -> CatalogTooling:
        def _out(line: str):
            self.log_txt.appendPlainText(line)

        def _done(step: str):
            self.log_txt.appendPlainText(f"✅ {step.upper()} completato.")
            self.progress.setVisible(False)
            self._set_busy(False)
            if step in ("enrich", "overrides", "build", "ingest"):
                self.catalog_updated.emit()

        def _err(step: str, err: str):
            self.log_txt.appendPlainText(f"❌ {step.upper()} errore: {err}")
            self.progress.setVisible(False)
            self._set_busy(False)

        return CatalogTooling(on_output=_out, on_done=_done, on_error=_err)

    def _source(self) -> str | None:
        s = self.source_edit.text().strip()
        if not s or not Path(s).exists():
            QMessageBox.warning(self, "Sorgente non valida",
                                "Imposta una cartella sorgente esistente (di default l'inbox).")
            return None
        return s

    def _start(self, op_name: str):
        self._tooling = self._make_tooling()
        self.log_txt.appendPlainText(f"\n🔄 {op_name}…")
        self.progress.setVisible(True)
        self._set_busy(True)
        return self._tooling

    def _do_analyze(self):
        src = self._source()
        if src:
            self._start("Analisi (dry-run)").analyze(src)

    def _do_place(self):
        src = self._source()
        if src:
            self._start("Spostamento file").place(src)

    def _do_scan(self):
        self._start("Scansione catalogo").scan()

    def _do_enrich(self):
        self._start("Arricchimento Civitai").enrich()

    def _do_overrides(self):
        self._start("Override manuali").apply_overrides()

    def _do_build(self):
        self._start("Generazione HTML").build()

    def _do_ingest(self):
        src = self._source()
        if src:
            self._start("Ingest completo").full_ingest(src)

    # ============================================================ stato UI
    def _all_buttons(self):
        return (self.btn_analyze, self.btn_place, self.btn_scan, self.btn_enrich,
                self.btn_overrides, self.btn_build, self.btn_ingest, self.b_to_inbox)

    def _set_busy(self, busy: bool):
        for b in self._all_buttons():
            b.setEnabled(not busy)

    def _check_tools(self):
        td = catalog_paths.tools_dir()
        if not td.exists():
            self._tools_note.setText(
                f"⚠️ Cartella strumenti non trovata: {td}\n"
                "Configura «Cartella strumenti» nei Percorsi qui sopra e premi «Salva».")
            for b in (self.btn_analyze, self.btn_place, self.btn_scan, self.btn_enrich,
                      self.btn_overrides, self.btn_build, self.btn_ingest):
                b.setEnabled(False)
        else:
            self._tools_note.setText(f"✅ Strumenti: {td}")
            for b in (self.btn_analyze, self.btn_place, self.btn_scan, self.btn_enrich,
                      self.btn_overrides, self.btn_build, self.btn_ingest):
                b.setEnabled(True)

    # ---------- API pubblica ----------
    def _log(self, msg: str):
        self.log_txt.appendPlainText(msg)
