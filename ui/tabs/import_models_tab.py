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
from ui.i18n_manager import tr

from core import catalog_paths
from core import comfy_compat
from core.catalog_tooling import CatalogTooling


class _DropArea(QLabel):
    """Area drag-and-drop che accetta file e cartelle."""
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__(tr("imp.drop.lbl"))
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

        intro = QLabel(tr("imp.intro.lbl"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#bbb; padding:2px;")
        v.addWidget(intro)

        v.addWidget(self._build_paths_group())
        v.addWidget(self._build_compat_group())
        v.addWidget(self._build_convert_group())
        v.addWidget(self._build_source_group())
        v.addWidget(self._build_ops_group())

        host.setLayout(v)
        scroll.setWidget(host)
        split.addWidget(scroll)

        # --- parte bassa: log + progress ------------------------------------
        log_w = QWidget()
        lv = QVBoxLayout(log_w)
        lv.setContentsMargins(0, 4, 0, 0)
        lv.addWidget(QLabel(tr("imp.output_lbl")))
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
        grp = QGroupBox(tr("imp.paths.group"))
        grp.setCheckable(True)
        grp.setChecked(False)           # collassato di default (default = installazione attuale)
        outer = QVBoxLayout(grp)

        note = QLabel(tr("imp.paths.note"))
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
            browse.setToolTip(tr("imp.paths.browse.tt"))
            browse.clicked.connect(lambda _=False, k=key: self._browse_path(k))
            grid.addWidget(lab, row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(browse, row, 2)
        outer.addLayout(grid)

        btns = QHBoxLayout()
        save_btn = QPushButton(tr("imp.paths.save.btn"))
        save_btn.clicked.connect(self._save_paths)
        reset_btn = QPushButton(tr("imp.paths.reset.btn"))
        reset_btn.clicked.connect(self._reset_paths)
        suggest_btn = QPushButton(tr("imp.paths.suggest.btn"))
        suggest_btn.setToolTip(tr("imp.paths.suggest.tt"))
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
        grp = QGroupBox(tr("imp.compat.group"))
        v = QVBoxLayout(grp)

        note = QLabel(tr("imp.compat.note"))
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(note)

        self._compat_status = QLabel("")
        self._compat_status.setWordWrap(True)
        self._compat_status.setStyleSheet("font-size:11px; padding:2px;")
        v.addWidget(self._compat_status)

        self.chk_prompt_unlock = QCheckBox(tr("imp.compat.prompt_unlock"))
        self.chk_prompt_unlock.setChecked(True)
        self.chk_prompt_unlock.setToolTip(tr("imp.compat.prompt_unlock.tt"))
        self.chk_prompt_unlock.toggled.connect(self._on_prompt_unlock_toggled)
        v.addWidget(self.chk_prompt_unlock)

        btns = QHBoxLayout()
        self.btn_compat_install = QPushButton(tr("imp.compat.install.btn"))
        self.btn_compat_install.clicked.connect(self._do_install_compat)
        self.btn_compat_uninstall = QPushButton(tr("imp.compat.uninstall.btn"))
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
            self._compat_status.setText(tr("imp.compat.status.unavailable").format(e=e))
            return
        if not st["comfy_valid"]:
            self._compat_status.setText(
                tr("imp.compat.status.not_found").format(comfy_root=st['comfy_root']))
            self.btn_compat_install.setEnabled(False)
            self.btn_compat_uninstall.setEnabled(False)
            return
        self.btn_compat_install.setEnabled(st["repo_shim_present"])
        self.btn_compat_uninstall.setEnabled(st["installed"])
        if st["installed"]:
            agg = (tr("imp.compat.status.uptodate") if st["up_to_date"]
                   else tr("imp.compat.status.outdated").format(
                       installed_version=st['installed_version'], repo_version=st['repo_version']))
            self._compat_status.setText(
                tr("imp.compat.status.installed").format(
                    installed_version=st['installed_version'], agg=agg,
                    nsfw=(tr("imp.compat.on") if st['prompt_unlock'] else tr("imp.compat.off"))))
            self.chk_prompt_unlock.blockSignals(True)
            self.chk_prompt_unlock.setChecked(st["prompt_unlock"])
            self.chk_prompt_unlock.blockSignals(False)
        else:
            self._compat_status.setText(
                tr("imp.compat.status.not_installed").format(repo_version=st['repo_version']))

    def _do_install_compat(self):
        try:
            comfy_compat.install(prompt_unlock=self.chk_prompt_unlock.isChecked())
            self._log("\n🧩 Patch di compatibilità installata/aggiornata in custom_nodes. "
                      "RIAVVIA ComfyUI per applicarla.")
        except Exception as e:
            QMessageBox.critical(self, tr("imp.compat.install.err.title"), str(e))
            self._log(f"[ERR] install patch: {e}")
        self._refresh_compat_status()

    def _do_uninstall_compat(self):
        if QMessageBox.question(self, tr("imp.compat.uninstall.confirm.title"),
                                tr("imp.compat.uninstall.confirm.text")) != QMessageBox.Yes:
            return
        try:
            comfy_compat.uninstall()
            self._log("\n🧩 Patch disinstallata. Riavvia ComfyUI per applicare.")
        except Exception as e:
            QMessageBox.critical(self, tr("imp.compat.uninstall.err.title"), str(e))
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
    def _build_convert_group(self) -> QGroupBox:
        """Conversione modelli ComfyUI → formato nativo (diffusers/HF) — #27. Riusa gli script esistenti."""
        grp = QGroupBox(tr("imp.convert.group", "Conversione modelli (ComfyUI → nativo)"))
        v = QVBoxLayout(grp)
        v.addWidget(QLabel(tr("imp.convert.note",
            "Converte i modelli ComfyUI (fp8/quantizzati) nel formato del backend nativo. Eseguito una tantum.")))
        row = QHBoxLayout()
        b_gemma = QPushButton(tr("imp.convert.gemma", "Gemma-3 → bf16"))
        b_gemma.clicked.connect(lambda: self._run_convert("scripts/convert_gemma_comfy.py", "Gemma-3"))
        b_te = QPushButton(tr("imp.convert.te", "Text encoder → HF"))
        b_te.clicked.connect(lambda: self._run_convert("scripts/convert_te_to_hf.py", "Text encoder"))
        row.addWidget(b_gemma); row.addWidget(b_te); row.addStretch(1)
        v.addLayout(row)
        return grp

    def _run_convert(self, script_rel: str, label: str):
        """Lancia uno script di conversione nel venv inference_server, in un thread, log live."""
        import threading
        import subprocess as _sp
        from pathlib import Path as _P
        root = _P(__file__).resolve().parents[2]
        script = root / script_rel
        py = r"E:\AI\Tools\inference_server_venv\Scripts\python.exe"
        if not script.exists():
            QMessageBox.warning(self, tr("imp.convert.group", "Conversione"),
                                tr("imp.convert.missing", "Script non trovato: {s}").format(s=script)); return
        self._log(f"\n🔄 Conversione {label} ({script_rel})…")

        def _worker():
            try:
                r = _sp.run([py, "-u", str(script)], cwd=str(root), capture_output=True, text=True, timeout=3600)
                out = (r.stdout or "")[-1500:] + (r.stderr or "")[-500:]
                self._log(out + (f"\n✓ {label} convertito." if r.returncode == 0 else f"\n[ERR] exit {r.returncode}"))
            except Exception as e:                          # noqa: BLE001
                self._log(f"[ERR] conversione {label}: {e}")
        threading.Thread(target=_worker, daemon=True).start()

    def _build_source_group(self) -> QGroupBox:
        grp = QGroupBox(tr("imp.files.group"))
        v = QVBoxLayout(grp)

        # sorgente
        srow = QHBoxLayout()
        srow.addWidget(QLabel(tr("imp.source_lbl")))
        self.source_edit = QLineEdit()
        self.source_edit.setToolTip(tr("imp.source.tt"))
        srow.addWidget(self.source_edit, 1)
        src_browse = QPushButton(tr("imp.source.browse.btn"))
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
        b_files = QPushButton(tr("imp.files.add.btn"))
        b_files.clicked.connect(self._browse_files)
        b_dir = QPushButton(tr("imp.files.add_dir.btn"))
        b_dir.clicked.connect(self._browse_adddir)
        self.b_to_inbox = QPushButton(tr("imp.files.inbox.btn"))
        self.b_to_inbox.setToolTip(tr("imp.files.inbox.tt"))
        self.b_to_inbox.clicked.connect(self._copy_to_inbox)
        b_rm = QPushButton(tr("imp.files.remove.btn"))
        b_rm.clicked.connect(self._remove_selected)
        b_clr = QPushButton(tr("imp.files.clear.btn"))
        b_clr.clicked.connect(self._clear_list)
        for b in (b_files, b_dir, self.b_to_inbox, b_rm, b_clr):
            lb.addWidget(b)
        lb.addStretch(1)
        v.addLayout(lb)
        return grp

    # ---- gruppo operazioni ----
    def _build_ops_group(self) -> QGroupBox:
        grp = QGroupBox(tr("imp.ops.group"))
        v = QVBoxLayout(grp)

        row = QHBoxLayout()
        self.btn_analyze = self._op_btn(tr("imp.ops.analyze.btn"), tr("imp.ops.analyze.tt"), self._do_analyze)
        self.btn_place = self._op_btn(tr("imp.ops.place.btn"), tr("imp.ops.place.tt"), self._do_place)
        self.btn_scan = self._op_btn(tr("imp.ops.scan.btn"), tr("imp.ops.scan.tt"), self._do_scan)
        self.btn_enrich = self._op_btn(tr("imp.ops.enrich.btn"), tr("imp.ops.enrich.tt"), self._do_enrich)
        self.btn_overrides = self._op_btn(tr("imp.ops.overrides.btn"), tr("imp.ops.overrides.tt"), self._do_overrides)
        self.btn_build = self._op_btn(tr("imp.ops.build.btn"), tr("imp.ops.build.tt"), self._do_build)
        for b in (self.btn_analyze, self.btn_place, self.btn_scan, self.btn_enrich,
                  self.btn_overrides, self.btn_build):
            row.addWidget(b)
        v.addLayout(row)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color:#333;")
        v.addWidget(sep)

        irow = QHBoxLayout()
        self.btn_ingest = QPushButton(tr("imp.ops.ingest.btn"))
        self.btn_ingest.setObjectName("generate")
        self.btn_ingest.setMinimumHeight(40)
        self.btn_ingest.setToolTip(tr("imp.ingest.tt"))
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
            fn, _ = QFileDialog.getOpenFileName(self, tr("imp.paths.pick_hashes.dlg"), cur,
                                                tr("imp.paths.hashes_filter"))
            if fn:
                self._path_edits[key].setText(fn)
        else:
            d = QFileDialog.getExistingDirectory(self, tr("imp.paths.pick_dir.dlg"), cur)
            if d:
                self._path_edits[key].setText(d)

    def _save_paths(self):
        cfg = {k: e.text().strip() for k, e in self._path_edits.items()}
        out = catalog_paths.save(cfg)
        self.source_edit.setText(cfg.get("inbox") or catalog_paths.DEFAULTS["inbox"])
        if out:
            self._paths_status.setText(tr("imp.paths.saved").format(out=out))
        else:
            self._paths_status.setText(tr("imp.paths.saved_no_tools"))
        self._check_tools()
        self._refresh_compat_status()

    def _reset_paths(self):
        for key, edit in self._path_edits.items():
            edit.setText(catalog_paths.DEFAULTS.get(key, ""))
        self._paths_status.setText(tr("imp.paths.reset.done"))

    def _suggest_paths(self):
        comfy = catalog_paths.detect_comfy_root()
        if not comfy:
            d = QFileDialog.getExistingDirectory(self, tr("imp.paths.pick_comfy.dlg"))
            if not d:
                return
            comfy = Path(d)
        sug = catalog_paths.suggest_from_comfy(comfy)
        for key, val in sug.items():
            if key in self._path_edits:
                self._path_edits[key].setText(val)
        self._paths_status.setText(tr("imp.paths.suggested").format(comfy=comfy))

    # ============================================================ sorgente/lista
    def _browse_source(self):
        d = QFileDialog.getExistingDirectory(self, tr("imp.source.pick.dlg"),
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
        paths, _ = QFileDialog.getOpenFileNames(self, tr("imp.files.pick.dlg"),
                                                self._path_edits["models"].text().strip() or "",
                                                tr("imp.model_filter"))
        if paths:
            self._add_paths(paths)

    def _browse_adddir(self):
        d = QFileDialog.getExistingDirectory(self, tr("imp.files.pick_dir.dlg"))
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
            QMessageBox.warning(self, tr("imp.source.invalid.title"),
                                tr("imp.source.invalid.text"))
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
            self._tools_note.setText(tr("imp.tools.not_found").format(td=td))
            for b in (self.btn_analyze, self.btn_place, self.btn_scan, self.btn_enrich,
                      self.btn_overrides, self.btn_build, self.btn_ingest):
                b.setEnabled(False)
        else:
            self._tools_note.setText(tr("imp.tools.found").format(td=td))
            for b in (self.btn_analyze, self.btn_place, self.btn_scan, self.btn_enrich,
                      self.btn_overrides, self.btn_build, self.btn_ingest):
                b.setEnabled(True)

    # ---------- API pubblica ----------
    def _log(self, msg: str):
        self.log_txt.appendPlainText(msg)
