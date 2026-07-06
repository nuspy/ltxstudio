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

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar, QFileDialog, QSplitter,
    QGroupBox, QScrollArea, QFrame, QMessageBox, QToolButton, QCheckBox, QButtonGroup,
    QStackedWidget, QComboBox, QDialog, QTextEdit, QDialogButtonBox,
)
from ui.i18n_manager import tr

from core import catalog_paths
from core import comfy_compat
from core import model_importer
from core import user_prefs
from core.catalog_tooling import CatalogTooling
from ui.widgets.model_card import ModelCard, badge, flow_grid


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


class _AnalyzeWorker(QThread):
    """Analizza file locali (tipo/arch/nsfw) fuori dal thread GUI. Emette un ImportItem per file."""
    analyzed = Signal(object)                                # ImportItem
    done = Signal()

    def __init__(self, paths: list[str]):
        super().__init__()
        self._paths = paths

    def run(self):
        for p in self._paths:
            try:
                item = model_importer.analyze(p, fetch_civitai=True)
            except Exception as e:                           # noqa: BLE001
                item = model_importer.ImportItem(path=p, note=f"errore analisi: {e}")
            self.analyzed.emit(item)
        self.done.emit()


class _StageWorker(QThread):
    """Copia/scarica gli item ammessi nell'inbox (serializzato — connessione metered)."""
    progress = Signal(int, str)                              # (indice, stato)
    done = Signal(int, int)                                  # (ok, errori)

    def __init__(self, items: list, inbox: str):
        super().__init__()
        self._items = items                                  # list[ImportItem] già gate-filtrati
        self._inbox = Path(inbox)

    def run(self):
        import shutil
        ok = err = 0
        self._inbox.mkdir(parents=True, exist_ok=True)
        for i, it in enumerate(self._items):
            try:
                if it.source == "civitai" and getattr(it, "civitai", None):
                    self.progress.emit(i, tr("imp2.state.downloading", "scarico…"))
                    url = it.civitai.get("download_url")
                    dest = self._inbox / it.name
                    model_importer.civitai_client.download(
                        url, dest, api_key=model_importer.civitai_client.api_key_from_prefs())
                else:
                    self.progress.emit(i, tr("imp2.state.staging", "copio…"))
                    shutil.copy2(it.path, self._inbox / Path(it.path).name)
                self.progress.emit(i, tr("imp2.state.staged", "pronto"))
                ok += 1
            except Exception as e:                           # noqa: BLE001
                self.progress.emit(i, tr("imp2.state.error", "errore: {e}").format(e=e))
                err += 1
        self.done.emit(ok, err)


class _ResolveWorker(QThread):
    """Risolve un URL/ID Civitai (rete) fuori dal thread GUI."""
    resolved = Signal(dict)
    failed = Signal(str)

    def __init__(self, url_or_id: str):
        super().__init__()
        self._ref = url_or_id

    def run(self):
        try:
            self.resolved.emit(model_importer.resolve_civitai(self._ref))
        except Exception as e:                               # noqa: BLE001
            self.failed.emit(str(e))


class _SearchWorker(QThread):
    """Ricerca Civitai (rete) fuori dal thread GUI."""
    results = Signal(list)
    failed = Signal(str)

    def __init__(self, query, types, base_models, nsfw):
        super().__init__()
        self._q, self._types, self._bm, self._nsfw = query, types, base_models, nsfw

    def run(self):
        try:
            data = model_importer.civitai_client.search(
                self._q, types=self._types, base_models=self._bm, nsfw=self._nsfw)
            self.results.emit(data.get("items", []))
        except Exception as e:                               # noqa: BLE001
            self.failed.emit(str(e))


class _ThumbWorker(QThread):
    """Scarica le anteprime (piccole) delle card, in cache, progressivamente."""
    ready = Signal(int, str)

    def __init__(self, jobs: list, cache_dir: str):
        super().__init__()
        self._jobs = jobs                                    # list[(idx, url)]
        self._cache = Path(cache_dir)

    def run(self):
        import hashlib as _h
        import re as _re
        self._cache.mkdir(parents=True, exist_ok=True)
        for idx, url in self._jobs:
            if not url:
                continue
            try:
                small = _re.sub(r"/width=\d+/", "/width=200/", url)
                dest = self._cache / (_h.md5(small.encode()).hexdigest() + ".img")
                if not dest.exists():
                    model_importer.civitai_client.download(small, dest, timeout=30)
                self.ready.emit(idx, str(dest))
            except Exception:                                # noqa: BLE001
                continue


class _LibraryDetailDialog(QDialog):
    """Model card di dettaglio: mostra e MODIFICA i metadati (→ manual_enrichment.json)."""

    def __init__(self, entry: dict, parent=None):
        super().__init__(parent)
        self._entry = entry
        self.changed = False
        self.setWindowTitle(entry.get("filename", "modello"))
        self.setMinimumWidth(440)
        v = QVBoxLayout(self)

        title = QLabel(entry.get("filename", "?"))
        title.setStyleSheet("font-weight:600; font-size:14px;")
        title.setWordWrap(True)
        v.addWidget(title)

        brow = QHBoxLayout()
        for text, kind in ((entry.get("type", ""), "type"), (entry.get("architecture", ""), "arch")):
            if text:
                brow.addWidget(badge(text, kind))
        brow.addStretch(1)
        v.addLayout(brow)

        civ = entry.get("civitai") or {}
        url = civ.get("model_url") or entry.get("civitai_url")
        if url:
            link = QLabel(f'<a href="{url}" style="color:#74c0fc;">{tr("imp2.detail.civitai", "Pagina Civitai")}</a>')
            link.setOpenExternalLinks(True)
            v.addWidget(link)

        v.addWidget(QLabel(tr("imp2.detail.triggers", "Trigger words (separate da virgola):")))
        trigs = entry.get("trigger_words") or civ.get("trigger_words") or []
        self.ed_trig = QLineEdit(", ".join(trigs))
        v.addWidget(self.ed_trig)

        v.addWidget(QLabel(tr("imp2.detail.desc", "Descrizione / note:")))
        self.ed_desc = QTextEdit()
        self.ed_desc.setPlainText(entry.get("user_notes") or entry.get("instructions") or "")
        self.ed_desc.setMaximumHeight(120)
        v.addWidget(self.ed_desc)

        self.cb_nsfw = QCheckBox(tr("imp2.detail.nsfw", "Contenuto NSFW"))
        self.cb_nsfw.setChecked(bool(entry.get("nsfw")))
        v.addWidget(self.cb_nsfw)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _save(self):
        import json
        fn = self._entry.get("filename")
        if not fn:
            self.reject(); return
        p = Path(catalog_paths.load()["dict_root"]) / "data" / "manual_enrichment.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:                                    # noqa: BLE001
            data = {}
        rec = data.get(fn) or {}
        rec["trigger_words"] = [t.strip() for t in self.ed_trig.text().split(",") if t.strip()]
        rec["instructions"] = self.ed_desc.toPlainText().strip()
        rec["nsfw"] = self.cb_nsfw.isChecked()
        data[fn] = rec
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            self.changed = True
        except Exception as e:                               # noqa: BLE001
            QMessageBox.critical(self, tr("imp2.detail.err", "Errore"), str(e))
            return
        self.accept()


class ImportModelsTab(QWidget):
    """Importa e cataloga modelli AI tramite gli strumenti del dizionario."""

    catalog_updated = Signal()   # emesso dopo enrich/build/ingest → ricarica catalogo LoRA

    def __init__(self):
        super().__init__()
        self._pending: list[str] = []
        self._tooling: CatalogTooling | None = None
        self._path_edits: dict[str, QLineEdit] = {}
        self._queue: list = []                               # list[ImportItem] della coda Importa
        self._queue_cards: list = []                         # ModelCard paralleli
        self._workers: list = []                             # QThread vivi (evita GC)
        self._build()
        self._load_paths_into_ui()
        self._check_tools()
        self._refresh_compat_status()

    # ============================================================ build (UX pro-max, segmentata)
    _MODES = [("scopri", "🔍", "imp2.mode.discover", "Scopri"),
              ("importa", "⬇", "imp2.mode.import", "Importa"),
              ("libreria", "🗂", "imp2.mode.library", "Libreria"),
              ("avanzate", "⚙", "imp2.mode.advanced", "Avanzate")]

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # --- barra segmentata ---
        seg = QHBoxLayout()
        seg.setSpacing(4)
        self._seg_group = QButtonGroup(self)
        self._seg_group.setExclusive(True)
        for i, (key, icon, i18n_key, default) in enumerate(self._MODES):
            b = QToolButton()
            b.setText(f"{icon}  {tr(i18n_key, default)}")
            b.setCheckable(True)
            b.setMinimumHeight(34)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                "QToolButton { border:1px solid #333; border-radius:8px; padding:4px 14px; color:#bbb; }"
                "QToolButton:checked { background:#1f2b3d; border-color:#74c0fc; color:#e8f0ff; font-weight:600; }")
            b.clicked.connect(lambda _=False, idx=i: self._select_mode(idx))
            self._seg_group.addButton(b, i)
            seg.addWidget(b)
        seg.addStretch(1)
        outer.addLayout(seg)

        # --- stack delle modalità ---
        self._stack = QStackedWidget()
        self._stack.addWidget(self._page_scopri())           # 0
        self._stack.addWidget(self._page_importa())          # 1
        self._stack.addWidget(self._page_libreria())         # 2
        self._stack.addWidget(self._page_avanzate())         # 3
        outer.addWidget(self._stack, 1)

        # --- log + progress CONDIVISI (sempre in basso) ---
        outer.addWidget(self._build_log())

        self._select_mode(1)                                 # default: Importa

    def _select_mode(self, idx: int):
        self._stack.setCurrentIndex(idx)
        btn = self._seg_group.button(idx)
        if btn and not btn.isChecked():
            btn.setChecked(True)
        if idx == 2 and not getattr(self, "_lib_loaded", True):   # Libreria: carica alla prima apertura
            self._lib_loaded = True
            self._load_library()

    def _build_log(self) -> QWidget:
        log_w = QWidget()
        lv = QVBoxLayout(log_w)
        lv.setContentsMargins(0, 2, 0, 0)
        lv.setSpacing(2)
        lv.addWidget(QLabel(tr("imp.output_lbl")))
        self.log_txt = QPlainTextEdit()
        self.log_txt.setReadOnly(True)
        self.log_txt.setMaximumHeight(150)
        self.log_txt.setStyleSheet("font-family: Consolas, monospace; font-size:11px;")
        lv.addWidget(self.log_txt)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)
        return log_w

    # ---- pagina Avanzate: TUTTO il power-user esistente (nulla si perde) ----
    def _page_avanzate(self) -> QWidget:
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
        v.addStretch(1)
        scroll.setWidget(host)
        return scroll

    # ---- pagina Importa: flusso guidato (URL/ID + drag-drop + coda) ----
    def _page_importa(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)
        intro = QLabel(tr("imp2.import.intro",
            "Incolla un link o ID Civitai, oppure trascina qui file/cartelle. I modelli vengono "
            "riconosciuti, categorizzati e spostati nella cartella giusta, pronti all'uso."))
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#bbb;")
        v.addWidget(intro)

        urow = QHBoxLayout()
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(tr("imp2.url.ph", "https://civitai.com/models/…  oppure  ID"))
        self.url_edit.returnPressed.connect(self._import_url)
        b_url = QPushButton(tr("imp2.url.btn", "Aggiungi da Civitai"))
        b_url.clicked.connect(self._import_url)
        urow.addWidget(self.url_edit, 1)
        urow.addWidget(b_url)
        v.addLayout(urow)

        self.imp_drop = _DropArea()
        self.imp_drop.files_dropped.connect(self._analyze_and_queue)
        v.addWidget(self.imp_drop)
        badd = QHBoxLayout()
        b_files = QPushButton(tr("imp2.addfiles.btn", "Scegli file…"))
        b_files.clicked.connect(self._pick_files_to_queue)
        b_dir = QPushButton(tr("imp2.adddir.btn", "Scegli cartella…"))
        b_dir.clicked.connect(self._pick_dir_to_queue)
        badd.addWidget(b_files)
        badd.addWidget(b_dir)
        badd.addStretch(1)
        v.addLayout(badd)

        v.addWidget(QLabel(tr("imp2.queue.lbl", "Coda di importazione:")))
        self.queue_list = QListWidget()
        self.queue_list.setStyleSheet("QListWidget { background:transparent; border:none; }")
        self.queue_list.setSpacing(3)
        v.addWidget(self.queue_list, 1)

        self._queue_empty = QLabel(tr("imp2.queue.empty", "La coda è vuota — aggiungi modelli sopra."))
        self._queue_empty.setAlignment(Qt.AlignCenter)
        self._queue_empty.setStyleSheet("color:#666; padding:16px;")
        v.addWidget(self._queue_empty)

        arow = QHBoxLayout()
        self.b_import_all = QPushButton(tr("imp2.importall.btn", "⬇  Importa tutto"))
        self.b_import_all.setObjectName("generate")
        self.b_import_all.setMinimumHeight(40)
        self.b_import_all.clicked.connect(self._import_all)
        b_clear_q = QPushButton(tr("imp2.queue.clear", "Svuota coda"))
        b_clear_q.clicked.connect(self._clear_queue)
        arow.addWidget(self.b_import_all, 1)
        arow.addWidget(b_clear_q)
        v.addLayout(arow)
        return w

    # ---- pagina Scopri: ricerca Civitai + griglia card ----
    _SEARCH_TYPES = [("imp2.type.all", "Tutti", None), ("imp2.type.lora", "LoRA", "LORA"),
                     ("imp2.type.ckpt", "Checkpoint", "Checkpoint"), ("imp2.type.vae", "VAE", "VAE"),
                     ("imp2.type.emb", "Embedding", "TextualInversion"),
                     ("imp2.type.cn", "ControlNet", "Controlnet")]
    _SEARCH_BASE = ["", "Flux.1 D", "SDXL 1.0", "Pony", "Illustrious", "SD 1.5", "SD 3.5"]

    def _page_scopri(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(8)
        row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(tr("imp2.search.ph", "Cerca su Civitai…"))
        self.search_edit.returnPressed.connect(self._do_search)
        self.search_type = QComboBox()
        for i18n_key, default, _api in self._SEARCH_TYPES:
            self.search_type.addItem(tr(i18n_key, default))
        self.search_base = QComboBox()
        self.search_base.addItem(tr("imp2.base.all", "Tutte le basi"))
        for bm in self._SEARCH_BASE[1:]:
            self.search_base.addItem(bm)
        self.search_nsfw = QCheckBox(tr("imp2.search.nsfw", "NSFW"))
        b_go = QPushButton(tr("imp2.search.btn", "Cerca"))
        b_go.clicked.connect(self._do_search)
        row.addWidget(self.search_edit, 1)
        row.addWidget(self.search_type)
        row.addWidget(self.search_base)
        row.addWidget(self.search_nsfw)
        row.addWidget(b_go)
        v.addLayout(row)

        self.search_scroll = QScrollArea()
        self.search_scroll.setWidgetResizable(True)
        self.search_scroll.setFrameShape(QScrollArea.NoFrame)
        hint = QLabel(tr("imp2.search.hint", "Cerca modelli e aggiungili alla coda di importazione."))
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#666; padding:30px;")
        self.search_scroll.setWidget(hint)
        v.addWidget(self.search_scroll, 1)
        return w

    def _do_search(self):
        nsfw = self.search_nsfw.isChecked()
        if nsfw and not model_importer.adult_enabled():
            self.search_nsfw.setChecked(False)
            nsfw = False
            QMessageBox.information(self, tr("imp2.gate.title", "Contenuto adulto"),
                tr("imp2.search.nsfw.off", "La ricerca NSFW richiede la funzione contenuti adulti (non attiva)."))
        typ = self._SEARCH_TYPES[self.search_type.currentIndex()][2]
        bi = self.search_base.currentIndex()
        base = [self._SEARCH_BASE[bi]] if bi > 0 else None
        self._log(tr("imp2.searching", "\n🔍 Cerco su Civitai…"))
        w = _SearchWorker(self.search_edit.text().strip(), [typ] if typ else None, base, nsfw)
        w.results.connect(self._populate_search)
        w.failed.connect(lambda e: self._log(tr("imp2.search.err", "[ERR] ricerca: {e}").format(e=e)))
        self._workers.append(w)
        w.start()

    def _populate_search(self, items: list):
        adult = model_importer.adult_enabled()
        cards, thumb_jobs = [], []
        shown = 0
        for model in items:
            it = self._item_from_search(model)
            if it.nsfw and not adult:                        # nascondi NSFW se funzione non attiva
                continue
            card = ModelCard(
                title=model.get("name", it.name),
                subtitle=(it.family if it.family != "Unknown" else ""),
                badges=self._item_badges(it), compact=False,
                actions=[(tr("imp2.add", "＋ Aggiungi"), lambda m=model: self._add_search_result(m))])
            cards.append(card)
            thumb_jobs.append((shown, self._search_preview_url(model)))
            shown += 1
        if not cards:
            empty = QLabel(tr("imp2.search.none", "Nessun risultato."))
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color:#666; padding:30px;")
            self.search_scroll.setWidget(empty)
            return
        self.search_scroll.setWidget(flow_grid(cards, columns=4))
        self._search_cards = cards
        tw = _ThumbWorker(thumb_jobs, str(Path(catalog_paths.APP_ROOT) / "data" / "thumb_cache"))
        tw.ready.connect(lambda i, p: self._search_cards[i].set_thumb(p) if i < len(self._search_cards) else None)
        self._workers.append(tw)
        tw.start()

    @staticmethod
    def _search_preview_url(model: dict) -> str | None:
        for ver in model.get("modelVersions") or []:
            for img in ver.get("images") or []:
                if img.get("url"):
                    return img["url"]
        return None

    def _item_from_search(self, model: dict):
        ver = (model.get("modelVersions") or [{}])[0]
        f = model_importer.civitai_client.pick_primary_file(ver)
        lvl = int(ver.get("nsfwLevel") or 0)
        nsfw = bool(model.get("nsfw")) or bool(lvl & ~model_importer.civitai_client.SFW_MASK)
        family = model_importer._CIVITAI_BASEMODEL_MAP.get(ver.get("baseModel"), "Unknown")
        typ = model_importer._CIVITAI_TYPE_MAP.get(model.get("type"), model.get("type") or "Unknown")
        it = model_importer.ImportItem(
            path="", name=(f or {}).get("name") or (model.get("name", "civitai") + ".safetensors"),
            type=typ, family=family, nsfw=nsfw, nsfw_source="civitai", source="civitai",
            triggers=list(ver.get("trainedWords") or []))
        it.dest_subfolder = model_importer._dest_subfolder(typ, family)
        it.civitai = {"download_url": (f or {}).get("downloadUrl")}
        return it

    def _add_search_result(self, model: dict):
        it = self._item_from_search(model)
        self._add_item_to_queue(it)
        self._log(tr("imp2.added", "＋ Aggiunto alla coda: {n}").format(n=it.name))
        self._select_mode(1)                                 # porta alla coda Importa

    # ---- pagina Libreria: griglia dei modelli catalogati ----
    def _page_libreria(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(8)
        row = QHBoxLayout()
        self.lib_search = QLineEdit()
        self.lib_search.setPlaceholderText(tr("imp2.lib.search", "Filtra la libreria…"))
        self.lib_search.textChanged.connect(self._load_library)
        self.lib_type = QComboBox()
        for i18n_key, default, _api in self._SEARCH_TYPES:
            self.lib_type.addItem(tr(i18n_key, default))
        self.lib_type.currentIndexChanged.connect(self._load_library)
        self.lib_arch = QComboBox()
        self.lib_arch.addItem(tr("imp2.arch.all", "Tutte le arch"))
        self.lib_arch.currentIndexChanged.connect(self._load_library)
        self.lib_nsfw = QCheckBox(tr("imp2.search.nsfw", "NSFW"))
        self.lib_nsfw.toggled.connect(self._load_library)
        b_ref = QPushButton(tr("imp2.lib.refresh", "↻"))
        b_ref.setToolTip(tr("imp2.lib.refresh.tt", "Ricarica la libreria"))
        b_ref.clicked.connect(lambda: self._load_library(force=True))
        row.addWidget(self.lib_search, 1)
        row.addWidget(self.lib_type)
        row.addWidget(self.lib_arch)
        row.addWidget(self.lib_nsfw)
        row.addWidget(b_ref)
        v.addLayout(row)

        self.lib_scroll = QScrollArea()
        self.lib_scroll.setWidgetResizable(True)
        self.lib_scroll.setFrameShape(QScrollArea.NoFrame)
        v.addWidget(self.lib_scroll, 1)
        self._lib_loaded = False
        self.catalog_updated.connect(lambda: self._load_library(force=True))
        return w

    def _library_entries(self) -> list:
        import json
        p = Path(catalog_paths.load()["dict_root"]) / "data" / "catalog.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            return []
        return list((data.get("entries") or {}).values())

    def _load_library(self, force: bool = False):
        entries = self._library_entries()
        # popola il combo arch una volta (dai dati)
        if self.lib_arch.count() <= 1 or force:
            archs = sorted({e.get("architecture", "") for e in entries if e.get("architecture")})
            cur = self.lib_arch.currentText()
            self.lib_arch.blockSignals(True)
            self.lib_arch.clear()
            self.lib_arch.addItem(tr("imp2.arch.all", "Tutte le arch"))
            for a in archs:
                self.lib_arch.addItem(a)
            idx = max(0, self.lib_arch.findText(cur))
            self.lib_arch.setCurrentIndex(idx)
            self.lib_arch.blockSignals(False)
        adult = model_importer.adult_enabled()
        want_nsfw = self.lib_nsfw.isChecked() and adult
        q = self.lib_search.text().strip().lower()
        ftype = self._SEARCH_TYPES[self.lib_type.currentIndex()][1]
        farch = self.lib_arch.currentText() if self.lib_arch.currentIndex() > 0 else None
        cards = []
        for e in entries:
            if e.get("nsfw") and not want_nsfw:              # NSFW visibile solo se adulti + checkbox attiva
                continue
            if self.lib_type.currentIndex() > 0 and (e.get("type") or "") != ftype:
                continue
            if farch and (e.get("architecture") or "") != farch:
                continue
            if q and q not in (e.get("filename", "").lower()):
                continue
            cards.append(self._library_card(e))
        host = flow_grid(cards, columns=4) if cards else QLabel(
            tr("imp2.lib.empty", "Nessun modello in libreria (importane qualcuno)."))
        if isinstance(host, QLabel):
            host.setAlignment(Qt.AlignCenter)
            host.setStyleSheet("color:#666; padding:30px;")
        self.lib_scroll.setWidget(host)

    def _library_card(self, entry: dict) -> ModelCard:
        name = entry.get("filename", "?")
        typ = entry.get("type", "")
        arch = entry.get("architecture", "")
        badges = []
        if typ:
            badges.append((typ, "type"))
        if arch:
            badges.append((arch, "arch"))
        badges.append(("🔞 NSFW", "nsfw") if entry.get("nsfw") else ("SFW", "sfw"))
        fav = user_prefs.is_fav_model(name)
        card = ModelCard(title=name, subtitle=arch, badges=badges, compact=False,
                         actions=[(tr("imp2.lib.details", "Dettagli"),
                                   lambda e=entry: self._open_detail(e))])
        card.clicked.connect(lambda e=entry: self._open_detail(e))
        return card

    def _open_detail(self, entry: dict):
        dlg = _LibraryDetailDialog(entry, self)
        if dlg.exec() == QDialog.Accepted and dlg.changed:
            self._start(tr("imp2.lib.applying", "Applico modifiche")).apply_overrides()

    # legacy: lo stub è stato sostituito dalla libreria reale.

    # ============================================================ coda Importa
    _MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf")

    def _pick_files_to_queue(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("imp.files.pick.dlg"), self._path_edits["models"].text().strip() or "",
            tr("imp.model_filter"))
        if paths:
            self._analyze_and_queue(paths)

    def _pick_dir_to_queue(self):
        d = QFileDialog.getExistingDirectory(self, tr("imp.files.pick_dir.dlg"))
        if d:
            self._analyze_and_queue([d])

    def _analyze_and_queue(self, paths: list[str]):
        """Espande cartelle → analizza i file modello in un thread → li mette in coda con badge."""
        files: list[str] = []
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                files += [str(f) for f in pp.rglob("*") if f.suffix.lower() in self._MODEL_EXTS]
            elif pp.suffix.lower() in self._MODEL_EXTS:
                files.append(str(pp))
        if not files:
            self._log(tr("imp2.nofiles", "[warn] Nessun file modello riconosciuto (.safetensors/.ckpt/…)."))
            return
        self._log(tr("imp2.analyzing", "\n🔎 Analizzo {n} file…").format(n=len(files)))
        w = _AnalyzeWorker(files)
        w.analyzed.connect(self._add_item_to_queue)
        w.done.connect(lambda: self._log(tr("imp2.analyzed", "✓ Analisi completata.")))
        self._workers.append(w)
        w.start()

    def _import_url(self):
        text = self.url_edit.text().strip()
        if not text:
            return
        self._log(tr("imp2.resolving", "\n🔗 Risolvo {u}…").format(u=text))
        w = _ResolveWorker(text)
        w.resolved.connect(self._on_url_resolved)
        w.failed.connect(lambda e: self._log(tr("imp2.resolve.err", "[ERR] {e}").format(e=e)))
        self._workers.append(w)
        w.start()
        self.url_edit.clear()

    def _on_url_resolved(self, info: dict):
        it = model_importer.ImportItem(
            path="", name=info.get("name", "civitai"),
            type=model_importer._CIVITAI_TYPE_MAP.get(info.get("type"), info.get("type") or "Unknown"),
            family=model_importer._CIVITAI_BASEMODEL_MAP.get(info.get("base_model"), "Unknown"),
            triggers=info.get("triggers", []), preview_url=info.get("preview_url"),
            nsfw=bool(info.get("nsfw")), nsfw_source=info.get("nsfw_source", "civitai"), source="civitai")
        it.dest_subfolder = model_importer._dest_subfolder(it.type, it.family)
        it.civitai = {"download_url": info.get("download_url"), "version": info.get("version")}
        self._add_item_to_queue(it)

    def _item_badges(self, item):
        b = []
        if item.type and item.type != "Unknown":
            b.append((item.type, "type"))
        if item.family and item.family != "Unknown":
            b.append((item.family, "arch"))
        b.append(("🔞 NSFW", "nsfw") if item.nsfw else ("SFW", "sfw"))
        return b

    def _add_item_to_queue(self, item):
        self._queue.append(item)
        sub = tr("imp2.dest", "→ {d}").format(d=item.dest_subfolder or "?")
        card = ModelCard(title=item.name, subtitle=sub, badges=self._item_badges(item),
                         compact=True,
                         actions=[(tr("imp2.remove", "✕"), lambda it=item: self._remove_queue_item(it))])
        li = QListWidgetItem(self.queue_list)
        li.setSizeHint(card.sizeHint())
        self.queue_list.setItemWidget(li, card)
        self._queue_cards.append(card)
        self._refresh_queue_empty()

    def _remove_queue_item(self, item):
        if item not in self._queue:
            return
        i = self._queue.index(item)
        self._queue.pop(i)
        self._queue_cards.pop(i)
        self.queue_list.takeItem(i)
        self._refresh_queue_empty()

    def _clear_queue(self):
        self._queue.clear()
        self._queue_cards.clear()
        self.queue_list.clear()
        self._refresh_queue_empty()

    def _refresh_queue_empty(self):
        self._queue_empty.setVisible(not self._queue)

    def _import_all(self):
        if not self._queue:
            self._log(tr("imp2.noqueue", "[warn] Coda vuota."))
            return
        allowed, skipped = [], []
        for it in self._queue:
            (allowed if model_importer.gate_ok(it.nsfw) else skipped).append(it)
        for it in skipped:
            self._log(tr("imp2.skip.nsfw",
                         "⏭  Saltato (contenuto adulto, funzione non attiva): {n}").format(n=it.name))
            self._queue_cards[self._queue.index(it)].set_status(tr("imp2.state.skipped", "saltato"), "nsfw")
        if not allowed:
            QMessageBox.information(self, tr("imp2.gate.title", "Contenuto adulto"),
                tr("imp2.gate.text", "Tutti gli elementi sono NSFW e la funzione contenuti adulti non è attiva."))
            return
        inbox = self._path_edits["inbox"].text().strip() or catalog_paths.DEFAULTS["inbox"]
        self._set_busy(True)
        self.progress.setVisible(True)
        self._log(tr("imp2.staging.start", "\n📥 Preparo {n} modelli…").format(n=len(allowed)))
        w = _StageWorker(allowed, inbox)
        w.progress.connect(lambda i, s: self._on_stage_progress(allowed, i, s))
        w.done.connect(lambda ok, err: self._on_stage_done(inbox, ok, err))
        self._workers.append(w)
        w.start()

    def _on_stage_progress(self, allowed, i, status):
        it = allowed[i]
        if it in self._queue:
            self._queue_cards[self._queue.index(it)].set_status(status)

    def _on_stage_done(self, inbox: str, ok: int, err: int):
        self._log(tr("imp2.staging.done",
                     "✅ Preparati {ok} (errori {err}). Catalogazione in corso…").format(ok=ok, err=err))
        self.source_edit.setText(inbox)
        if ok:
            self._start("Ingest completo").full_ingest(inbox)   # place→scan→enrich→build (+catalog_updated)
        else:
            self._set_busy(False)
            self.progress.setVisible(False)

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
        btns = [self.btn_analyze, self.btn_place, self.btn_scan, self.btn_enrich,
                self.btn_overrides, self.btn_build, self.btn_ingest, self.b_to_inbox]
        if hasattr(self, "b_import_all"):
            btns.append(self.b_import_all)
        return tuple(btns)

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
