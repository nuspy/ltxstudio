"""Modelli — variante «Pro» (switchabile da menu Interfaccia).

Sottoclasse di `ImportModelsTab`: riusa TUTTA la logica (worker, coda, ricerca, libreria, Avanzate) e
sovrascrive SOLO le parti migliorate dalla review UX/UI pro-max. La V1 (Classica) resta intatta.

Migliorie: 1A gating in-regione (upsell al posto del contenuto), 2B regioni nette (URL/ricerca→Scopri,
Importa=solo locale), 3A badge 🔒 sulle modalità gated, 4A log contestuale (solo Importa/Avanzate),
5A thumbnail lazy (bottone «Carica anteprime»), 6A NSFW lock invece del rimbalzo, 7A empty-state coda,
8A gruppi Avanzate collassabili, 9A azioni rapide sulle card di Libreria.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QToolButton,
    QButtonGroup, QStackedWidget, QListWidget, QGroupBox,
)
from ui.i18n_manager import tr

from core import features, model_importer, user_prefs
from ui.tabs.import_models_tab import ImportModelsTab, _DropArea
from ui.widgets.model_card import ModelCard


class ImportModelsTabPro(ImportModelsTab):
    _MODE_CAP = {0: "cap.civitai_import", 3: "cap.import_models"}   # modalità → capability (badge/gate)

    # ---------- gating in-regione (1A) ----------
    def _gate_or(self, cap: str | None, builder):
        """Ritorna il contenuto se la capability è abilitata, altrimenti un UpsellPanel al SUO posto."""
        if not cap or features.is_enabled(cap):
            return builder()
        from ui.widgets.upsell import UpsellPanel
        info = features.manifest().get("features", {}).get(cap, {})
        return UpsellPanel(cap, info.get("i18n_key", ""))

    # ---------- build: segmented con 🔒 (3A) + pagine gated + log contestuale (4A) ----------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        seg = QHBoxLayout(); seg.setSpacing(4)
        self._seg_group = QButtonGroup(self); self._seg_group.setExclusive(True)
        for i, (key, icon, i18n_key, default) in enumerate(self._MODES):
            b = QToolButton(); b.setCheckable(True); b.setMinimumHeight(34); b.setCursor(Qt.PointingHandCursor)
            cap = self._MODE_CAP.get(i)
            locked = bool(cap) and not features.is_enabled(cap)
            b.setText(("🔒 " if locked else "") + f"{icon}  {tr(i18n_key, default)}")
            if locked:
                b.setToolTip(tr("imp2.locked.tt", "Disponibile in un piano superiore"))
            b.setStyleSheet(
                "QToolButton{border:1px solid #333;border-radius:8px;padding:4px 14px;color:#bbb;}"
                "QToolButton:checked{background:#1f2b3d;border-color:#74c0fc;color:#e8f0ff;font-weight:600;}")
            b.clicked.connect(lambda _=False, idx=i: self._select_mode(idx))
            self._seg_group.addButton(b, i); seg.addWidget(b)
        seg.addStretch(1); outer.addLayout(seg)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._gate_or("cap.civitai_import", self._page_scopri))   # 0
        self._stack.addWidget(self._page_importa())                                     # 1
        self._stack.addWidget(self._page_libreria())                                    # 2
        self._stack.addWidget(self._gate_or("cap.import_models", self._page_avanzate))   # 3
        outer.addWidget(self._stack, 1)

        self._log_host = self._build_log()
        outer.addWidget(self._log_host)
        self._select_mode(1)
        self._apply_nsfw_lock()

    def _select_mode(self, idx: int):
        super()._select_mode(idx)
        if hasattr(self, "_log_host"):
            self._log_host.setVisible(idx in (1, 3))          # 4A: log solo dove ci sono operazioni

    # ---------- 2B: URL/ricerca Civitai vivono in Scopri ----------
    def _page_scopri(self) -> QWidget:
        w = super()._page_scopri()
        lay = w.layout()
        urow = QHBoxLayout()
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(tr("imp2.url.ph", "https://civitai.com/models/…  oppure  ID"))
        self.url_edit.returnPressed.connect(self._import_url)
        b_url = QPushButton(tr("imp2.url.btn", "Aggiungi da Civitai"))
        b_url.clicked.connect(self._import_url)
        self._btn_thumbs = QPushButton(tr("imp2pro.thumbs.btn", "🖼 Carica anteprime"))
        self._btn_thumbs.setEnabled(False)
        self._btn_thumbs.clicked.connect(self._load_thumbs)
        urow.addWidget(self.url_edit, 1)
        urow.addWidget(b_url)
        urow.addWidget(self._btn_thumbs)
        lay.insertLayout(0, urow)
        return w

    # ---------- 5A: thumbnail lazy (non scaricare in automatico) ----------
    def _start_thumbs(self, thumb_jobs):
        self._pending_thumbs = thumb_jobs
        if hasattr(self, "_btn_thumbs"):
            self._btn_thumbs.setEnabled(bool(thumb_jobs))

    def _load_thumbs(self):
        jobs = getattr(self, "_pending_thumbs", None)
        if jobs:
            super()._start_thumbs(jobs)
            self._btn_thumbs.setEnabled(False)

    # ---------- 2B + 7A: Importa = solo locale, empty-state ----------
    def _page_importa(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w); v.setSpacing(10)
        intro = QLabel(tr("imp2pro.import.intro",
            "Trascina qui file o cartelle di modelli. Per importare da Civitai usa «🔍 Scopri»."))
        intro.setWordWrap(True); intro.setStyleSheet("color:#bbb;")
        v.addWidget(intro)
        self.imp_drop = _DropArea()
        self.imp_drop.files_dropped.connect(self._analyze_and_queue)
        v.addWidget(self.imp_drop)
        badd = QHBoxLayout()
        b_files = QPushButton(tr("imp2.addfiles.btn", "Scegli file…"))
        b_files.clicked.connect(self._pick_files_to_queue)
        b_dir = QPushButton(tr("imp2.adddir.btn", "Scegli cartella…"))
        b_dir.clicked.connect(self._pick_dir_to_queue)
        badd.addWidget(b_files); badd.addWidget(b_dir); badd.addStretch(1)
        v.addLayout(badd)
        prow = QHBoxLayout()                                          # punto 10 (A: Importa da pagina, B: guida alla chiave API)
        b_page = QPushButton(tr("imp2.page.btn", "Importa da pagina…"))
        b_page.setToolTip(tr("imp2.page.tt", "Hai già scaricato il file: indica la pagina del modello e l'app ricostruisce architettura, trigger, descrizione e immagine."))
        b_page.clicked.connect(self._import_from_page)
        b_key = QPushButton(tr("imp2.key.btn", "Chiave API Civitai (guida)…"))
        b_key.clicked.connect(self._key_wizard)
        prow.addWidget(b_page); prow.addWidget(b_key); prow.addStretch(1)
        v.addLayout(prow)
        v.addWidget(QLabel(tr("imp2.queue.lbl", "Coda di importazione:")))
        self.queue_list = QListWidget()
        self.queue_list.setStyleSheet("QListWidget{background:transparent;border:none;}")
        self.queue_list.setSpacing(3)
        v.addWidget(self.queue_list, 1)
        self._queue_empty = QLabel(tr("imp2pro.queue.empty",
            "⬇  Nessun modello in coda.\nTrascina un file o aggiungine da «Scopri»."))
        self._queue_empty.setAlignment(Qt.AlignCenter)
        self._queue_empty.setStyleSheet("color:#666; padding:24px; border:1px dashed #333; border-radius:8px;")
        v.addWidget(self._queue_empty)
        arow = QHBoxLayout()
        self.b_import_all = QPushButton(tr("imp2.importall.btn", "⬇  Importa tutto"))
        self.b_import_all.setObjectName("generate"); self.b_import_all.setMinimumHeight(40)
        self.b_import_all.clicked.connect(self._import_all)
        b_clear = QPushButton(tr("imp2.queue.clear", "Svuota coda"))
        b_clear.clicked.connect(self._clear_queue)
        arow.addWidget(self.b_import_all, 1); arow.addWidget(b_clear)
        v.addLayout(arow)
        return w

    # ---------- 8A: gruppi Avanzate collassabili ----------
    def _page_avanzate(self) -> QWidget:
        w = super()._page_avanzate()
        host = w.widget() if hasattr(w, "widget") else w
        groups = host.findChildren(QGroupBox) if host is not None else []
        for gb in groups:
            if gb.isCheckable() or not gb.title():
                continue                                     # "Percorsi" è già checkable
            expanded = "oper" in gb.title().lower() or "operaz" in gb.title().lower()
            self._make_collapsible(gb, expanded)
        return w

    @staticmethod
    def _make_collapsible(gb: QGroupBox, expanded: bool):
        gb.setCheckable(True)
        gb.setChecked(expanded)
        inner = [c for c in gb.children() if isinstance(c, QWidget)]
        for c in inner:
            c.setVisible(expanded)
        gb.toggled.connect(lambda on, items=inner: [c.setVisible(on) for c in items])

    # ---------- 6A: NSFW lock invece del rimbalzo ----------
    def _apply_nsfw_lock(self):
        if model_importer.adult_enabled():
            return
        for attr in ("search_nsfw", "lib_nsfw"):
            cb = getattr(self, attr, None)
            if cb is not None:
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.setText("🔒 " + cb.text())
                cb.setToolTip(tr("imp2pro.nsfw.locked",
                    "Contenuti 18+ non attivi in questa edizione."))

    # ---------- 9A: azioni rapide sulle card di Libreria ----------
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
        star = "★" if user_prefs.is_fav_model(name) else "☆"
        card = ModelCard(title=name, subtitle=arch, badges=badges, compact=False, actions=[
            (star, lambda e=entry: self._toggle_fav_lib(e)),
            (tr("imp2pro.openfolder", "📂"), lambda e=entry: self._open_model_folder(e)),
            (tr("imp2.lib.details", "Dettagli"), lambda e=entry: self._open_detail(e)),
        ])
        card.clicked.connect(lambda e=entry: self._open_detail(e))
        return card

    def _toggle_fav_lib(self, entry: dict):
        user_prefs.toggle_fav_model(entry.get("filename", ""))
        self._load_library(force=True)

    def _open_model_folder(self, entry: dict):
        p = entry.get("path") or ""
        folder = os.path.dirname(p) if p else ""
        if folder and os.path.isdir(folder):
            try:
                os.startfile(folder)                          # noqa: S606 (Windows)
            except Exception:                                 # noqa: BLE001
                self._log(tr("imp2pro.openfolder.err", "[warn] Impossibile aprire la cartella."))
        else:
            self._log(tr("imp2pro.nofolder", "[warn] Percorso non disponibile per questo modello."))
