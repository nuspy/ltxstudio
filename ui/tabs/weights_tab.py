"""Tab «Pesi modelli» — wizard download pesi da HuggingFace (locale-first).

- L'utente sceglie la CARTELLA BASE; i pesi sono organizzati per <type>/<family> (Flux, LTX, …)
  e da lì vengono caricati dai moduli.
- Prima di scaricare, può indicare una cartella in cui CERCARE pesi già esistenti (anche in
  sottocartelle) per evitare doppioni.
- Download via scripts/download_weights.py (huggingface_hub), HF token solo per i gated.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QCheckBox, QPushButton,
    QLineEdit, QScrollArea, QPlainTextEdit, QProgressBar, QFormLayout, QFileDialog,
)

from ui.i18n_manager import tr
from core import weights_catalog as wc


class WeightsTab(QWidget):
    # FASE D: modelli arrivati (download/collega/importa) → chi popola i dropdown deve RIFARE
    # l'inventario (prima serviva riavviare l'app: _inventory_set non veniva mai resettato).
    inventory_changed = Signal()

    def __init__(self):
        super().__init__()
        self._proc: QProcess | None = None
        self._checks: dict[str, tuple[QCheckBox, dict]] = {}
        self._found: dict[str, str] = {}                 # id → path trovato dallo scan
        self._show_all = False                           # False = solo i pesi usati dai moduli dell'edizione
        self._total = 0
        self._done = 0

        root = QVBoxLayout(self)
        intro = QLabel(tr("weights.intro",
            "Scarica i pesi UNA VOLTA: poi la generazione è 100% locale/offline. "
            "L'HF token serve solo per i modelli gated (es. FLUX.1-dev)."))
        intro.setWordWrap(True); intro.setStyleSheet("color:#9aa1b0;")
        root.addWidget(intro)

        f = QFormLayout()
        # cartella base
        brow = QHBoxLayout()
        self.base = QLineEdit(wc.get_base_dir())
        self.base.editingFinished.connect(lambda: wc.set_base_dir(self.base.text().strip()) or self._refresh())
        bb = QPushButton(tr("weights.browse", "Sfoglia…")); bb.clicked.connect(self._pick_base)
        brow.addWidget(self.base, 1); brow.addWidget(bb)
        bw = QWidget(); bw.setLayout(brow)
        f.addRow(tr("weights.base", "Cartella base"), bw)
        # cartella in cui cercare pesi esistenti
        srow = QHBoxLayout()
        self.search = QLineEdit(); self.search.setPlaceholderText(tr("weights.scan.hint", "cartella in cui cercare pesi già scaricati (anche sottocartelle)"))
        sb = QPushButton(tr("weights.browse", "Sfoglia…")); sb.clicked.connect(self._pick_search)
        scan = QPushButton(tr("weights.scan.btn", "Cerca esistenti")); scan.clicked.connect(self._scan)
        srow.addWidget(self.search, 1); srow.addWidget(sb); srow.addWidget(scan)
        sw = QWidget(); sw.setLayout(srow)
        f.addRow(tr("weights.scan.dir", "Cerca in"), sw)
        # token
        self.hf = QLineEdit(); self.hf.setEchoMode(QLineEdit.Password)
        self.hf.setPlaceholderText(tr("weights.hf.hint", "necessario solo per i modelli gated"))
        f.addRow(tr("weights.hf", "HF token"), self.hf)
        # chiave API Civitai: Civitai chiede il login per scaricare (anche via API). Resta SOLO in locale
        # (user_prefs) e va solo verso civitai.com; serve per le voci del catalogo con fonte Civitai.
        self.civitai = QLineEdit(); self.civitai.setEchoMode(QLineEdit.Password)
        self.civitai.setPlaceholderText(tr("weights.civitai.hint",
            "civitai.com → Impostazioni account → API Keys (resta solo su questo PC)"))
        self.civitai.setToolTip(tr("weights.civitai.tt",
            "Serve per i modelli con fonte Civitai: senza chiave Civitai rimanda al login e il download fallisce."))
        try:
            from core import user_prefs as _up
            self.civitai.setText(str(_up.get("civitai_api_key", "") or ""))
        except Exception:                                # noqa: BLE001
            pass
        self.civitai.editingFinished.connect(self._save_civitai_key)
        f.addRow(tr("weights.civitai", "Chiave API Civitai"), self.civitai)
        root.addLayout(f)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); self._list = QVBoxLayout(inner); scroll.setWidget(inner)
        root.addWidget(scroll, 1)
        self._build_list()

        h = QHBoxLayout()
        self.filter_txt = QLineEdit()
        self.filter_txt.setPlaceholderText(tr("weights.filter.hint", "filtra per nome/arch/repo…"))
        self.filter_txt.textChanged.connect(lambda *_: self._rebuild_list())
        h.addWidget(self.filter_txt, 1)
        self.chk_all = QCheckBox(tr("weights.show_all", "Mostra tutti i moduli"))
        self.chk_all.setToolTip(tr("weights.show_all.tt", "Disattiva il filtro per edizione/moduli presenti."))
        self.chk_all.toggled.connect(self._on_show_all)
        # weights-unification C: il «Mostra tutti» era un BYPASS del gating d'edizione a portata di
        # click di chiunque → visibile solo in build di sviluppo (curatela/debug).
        try:
            from core import features as _feat
            if _feat.edition() != "dev":
                self.chk_all.setVisible(False)
        except Exception:                                # noqa: BLE001
            pass
        self.btn_refresh = QPushButton(tr("weights.refresh", "Aggiorna stato")); self.btn_refresh.clicked.connect(self._refresh)
        self.btn_missing = QPushButton(tr("weights.sel_missing", "Seleziona mancanti")); self.btn_missing.clicked.connect(self._select_missing)
        self.btn_link = QPushButton(tr("weights.link", "🔗 Collega trovati"))
        self.btn_link.setToolTip(tr("weights.link.tt", "Collega (junction/hardlink, senza admin) i pesi trovati "
                                    "nella cartella di ricerca invece di riscaricarli."))
        self.btn_link.clicked.connect(self._link_found)
        self.btn_import = QPushButton(tr("weights.import", "＋ Importa sconosciuti"))
        self.btn_import.setToolTip(tr("weights.import.tt", "Collega nel catalogo i modelli trovati nella cartella "
                                      "di ricerca ma non elencati (per usarli nell'app)."))
        self.btn_import.clicked.connect(self._import_unknown)
        self.btn_dl = QPushButton(tr("weights.download", "Scarica selezionati")); self.btn_dl.clicked.connect(self._download)
        h.addWidget(self.chk_all); h.addWidget(self.btn_refresh); h.addWidget(self.btn_missing)
        h.addWidget(self.btn_link); h.addWidget(self.btn_import); h.addStretch(1); h.addWidget(self.btn_dl)
        root.addLayout(h)

        self.bar = QProgressBar(); self.status = QLabel("—"); self.status.setStyleSheet("color:#9aa1b0;")
        self.logview = QPlainTextEdit(); self.logview.setReadOnly(True); self.logview.setFixedHeight(150)
        self.logview.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        root.addWidget(self.bar); root.addWidget(self.status); root.addWidget(self.logview)
        self._refresh()

    def _save_civitai_key(self):
        try:
            from core import user_prefs as _up
            _up.set("civitai_api_key", self.civitai.text().strip())
        except Exception:                                # noqa: BLE001
            pass

    def _build_list(self):
        # SCHEMA v2: raggruppa CATEGORIA (immagini/video/musica/voce/3D/ausiliari) → ARCHITETTURA —
        # gli assi tassonomici veri, non più i 19 moduli piatti in ordine JSON. `only_wanted` = solo
        # i pesi delle capability dell'edizione (feature_matrix); «Mostra tutti» esiste SOLO in dev
        # (prima era un bypass del gating a portata di click di qualunque edizione). Filtro testo
        # live su id/label/arch/repo.
        q = (getattr(self, "filter_txt", None) and self.filter_txt.text().strip().lower()) or ""
        groups = wc.by_category(only_wanted=not self._show_all)
        for cat, arches in groups.items():
            cat_label = tr(f"weights.cat.{cat}", wc.CATEGORY_LABELS.get(cat, cat))
            entries_shown = 0
            box = QGroupBox(cat_label); v = QVBoxLayout(box)
            for arch in sorted(arches):
                sub = [e for e in arches[arch]
                       if not q or q in f"{e['id']} {e.get('label', '')} {arch} "
                       f"{e.get('repo_id', '')}".lower()]
                if not sub:
                    continue
                hdr = QLabel(arch)
                hdr.setStyleSheet("font-weight:bold;color:#b7bfd0;margin-top:4px;")
                v.addWidget(hdr)
                for e in sub:
                    cb = QCheckBox()
                    if not wc.downloadable(e):
                        cb.setEnabled(False)             # no_remote: solo Collega/Importa, mai download
                    cb.toggled.connect(lambda on, _id=e["id"]: self._on_check_toggled(_id, on))
                    self._checks[e["id"]] = (cb, e)
                    v.addWidget(cb)
                    entries_shown += 1
            if entries_shown:
                self._list.addWidget(box)
            else:
                box.deleteLater()
        self._list.addStretch(1)

    def _rebuild_list(self):
        """Ricostruisce la lista pesi (dopo il toggle «Mostra tutti»)."""
        while self._list.count():
            it = self._list.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
        self._checks.clear()
        self._build_list()
        self._refresh()

    def _label_for(self, e: dict) -> str:
        base = self.base.text().strip() or None
        gated = " 🔒" if e.get("gated") else ""
        adult = " 🔞" if e.get("sensitivity") == "adult" else ""
        role = " [base]" if wc.install_role(e) == "base" else ""
        fam = f"[{e.get('type','')}/{e.get('family','')}]" if e.get("family") else f"[{e.get('type','')}]"
        if wc.is_present(e, base):
            state = "✓ " + tr("weights.present", "presente")
        elif e["id"] in self._found:
            state = "⇒ " + tr("weights.found", "già presente in: {p}").format(p=self._found[e["id"]])
        elif not wc.downloadable(e):
            state = "∅ " + tr("weights.no_remote", "senza fonte remota: usa Collega/Importa")
        else:
            state = "• " + tr("weights.missing", "mancante")
        return f"{e.get('label', e['id'])} {fam} · ~{e.get('approx_gb','?')}GB{gated}{adult}{role} — {state}"

    def _refresh(self):
        from core import weights_tooltips
        for _id, (cb, e) in self._checks.items():
            cb.setText(self._label_for(e))
            # tooltip ESPLICATIVO (cosa fa / quando preferirlo, i18n) + destinazione — identico
            # negli altri spazi di download (dropdown, dialog primo-uso): weights_tooltips.
            cb.setToolTip(weights_tooltips.tooltip_for(e))
            cb.setChecked(False)

    def _on_check_toggled(self, eid: str, checked: bool):
        """CATENA: spuntare un modello spunta anche le sue dipendenze mancanti (es. LTX 2.5 →
        encoder Gemma + VAE). Il downloader le espande comunque (plan_ids): qui si RENDE VISIBILE
        cosa verrà scaricato, niente sorprese sui GB."""
        if not checked:
            return
        e = (self._checks.get(eid) or (None, None))[1]
        if not e:
            return
        base = self.base.text().strip() or None
        for dep in wc.missing_ids_with_deps(e, base):
            pair = self._checks.get(dep)
            if pair and pair[0].isEnabled() and not pair[0].isChecked():
                pair[0].setChecked(True)

    def _pick_base(self):
        d = QFileDialog.getExistingDirectory(self, tr("weights.base", "Cartella base"), self.base.text().strip())
        if d:
            self.base.setText(d); wc.set_base_dir(d); self._refresh()

    def _pick_search(self):
        d = QFileDialog.getExistingDirectory(self, tr("weights.scan.dir", "Cerca in"), self.search.text().strip())
        if d:
            self.search.setText(d)

    def _scan(self):
        sdir = self.search.text().strip()
        if not sdir:
            self.status.setText(tr("weights.scan.none", "Indica una cartella in cui cercare.")); return
        self.status.setText(tr("weights.scanning", "Ricerca in corso…"))
        base = self.base.text().strip() or None
        self._found.clear()
        n = 0
        for _id, (cb, e) in self._checks.items():
            if wc.is_present(e, base):
                continue
            hit = wc.scan_existing(e, sdir)
            if hit:
                self._found[e["id"]] = hit; n += 1
        self._refresh()
        self.status.setText(tr("weights.scan.done", "Trovati {n} pesi già esistenti.").format(n=n))

    def _on_show_all(self, checked: bool):
        self._show_all = bool(checked)
        self._rebuild_list()

    def _link_found(self):
        """Collega (junction/hardlink, no admin) i pesi trovati dallo scan invece di riscaricarli."""
        if not self._found:
            self.status.setText(tr("weights.link.none", "Nessun peso trovato da collegare (fai prima «Cerca esistenti»).")); return
        from core import linkutil
        base = self.base.text().strip() or None
        # avviso una-tantum: alcuni pesi sono su un ALTRO volume → link impossibile, servirebbe COPIARLI (grande).
        targets = []
        for _id, (cb, e) in self._checks.items():
            src = self._found.get(e["id"])
            if not src:
                continue
            dst = wc.dest_for(e, base)
            if e.get("kind") == "file":
                dst = dst / (e.get("filename") or Path(src).name)
            targets.append((e, src, str(dst)))
        _copy_needed = [t for t in targets if linkutil.would_copy(t[1], t[2])]
        _allow_copy = True
        if _copy_needed:
            from PySide6.QtWidgets import QMessageBox
            _allow_copy = QMessageBox.question(self, tr("weights.link", "Collega"),
                tr("weights.link.crossvol", "{k} pesi sono su un altro disco rispetto alla cartella base: non "
                   "si possono collegare, andrebbero COPIATI (occupano spazio). Copiarli? "
                   "(In alternativa, imposta la cartella base sullo stesso disco dei modelli.)")
                .format(k=len(_copy_needed))) == QMessageBox.Yes
        n = 0
        for e, src, dst in targets:
            r = linkutil.link_or_copy(src, dst, on_copy=(lambda *_a: _allow_copy))
            self.logview.appendPlainText(f"[collega] {e['id']}: {src} → {r}")
            if r in ("hardlink", "junction", "symlink", "copy", "exists"):
                n += 1
        self._found.clear()
        self._refresh()
        self.status.setText(tr("weights.link.done", "Collegati {n} pesi (senza riscaricare).").format(n=n))
        if n:
            self.inventory_changed.emit()

    def _import_unknown(self):
        """Collega nella cartella Models i file-modello trovati nella cartella di ricerca ma NON elencati nel
        catalogo → il detector (model_detect) li rileva e diventano usabili nell'app. Nessuna rete."""
        sdir = self.search.text().strip()
        if not sdir or not Path(sdir).is_dir():
            self.status.setText(tr("weights.scan.none", "Indica una cartella in cui cercare.")); return
        known = set()
        for _id, (cb, e) in self._checks.items():
            for n in e.get("scan_names", []):
                known.add(str(n).lower())
            if e.get("repo_id"):
                known.add(e["repo_id"].split("/")[-1].lower())
        unknown: list[Path] = []
        root = Path(sdir)
        for p in root.rglob("*"):
            try:
                if p.is_file() and p.suffix.lower() in wc.IMG_EXT and p.stat().st_size > 50 * 2**20:
                    stem = p.stem.lower()
                    if not any(k and (k in stem or stem in k) for k in known):
                        unknown.append(p)
            except Exception:                               # noqa: BLE001
                continue
        if not unknown:
            self.status.setText(tr("weights.import.none", "Nessun modello sconosciuto nella cartella.")); return
        from PySide6.QtWidgets import QMessageBox
        names = "\n".join("• " + p.name for p in unknown[:40]) + ("\n…" if len(unknown) > 40 else "")
        if QMessageBox.question(self, tr("weights.import", "Importa sconosciuti"),
                                tr("weights.import.confirm", "Trovati {n} modelli non elencati. Collegarli nella "
                                   "cartella Models (senza copiarli) così l'app li rileva?\n\n{names}")
                                .format(n=len(unknown), names=names)) != QMessageBox.Yes:
            return
        from core import linkutil
        base = self.base.text().strip() or wc.get_base_dir()
        try:
            from core import model_importer as _mi
            _classify = _mi._classify_type_family
        except Exception:                                   # noqa: BLE001
            _classify = None
        n = 0
        for p in unknown:
            try:
                typ, fam = _classify(str(p)) if _classify else ("diffusion_models", "misc")
            except Exception:                               # noqa: BLE001
                typ, fam = "diffusion_models", "misc"
            dst = Path(base) / (typ or "diffusion_models") / (fam or "misc") / p.name
            r = linkutil.link_or_copy(str(p), str(dst))
            self.logview.appendPlainText(f"[importa] {p.name} → {typ}/{fam} · {r}")
            if r in ("hardlink", "junction", "symlink", "copy", "exists"):
                n += 1
        self.status.setText(tr("weights.import.done", "Importati {n} modelli (collegati; il detector li rileverà).").format(n=n))
        if n:
            self.inventory_changed.emit()

    def _select_missing(self):
        base = self.base.text().strip() or None
        for _id, (cb, e) in self._checks.items():
            cb.setChecked(wc.downloadable(e) and not wc.is_present(e, base)
                          and e["id"] not in self._found)

    def _download(self):
        if self._proc is not None:
            return
        ids = [i for i, (cb, _e) in self._checks.items() if cb.isChecked()]
        if not ids:
            self.status.setText(tr("weights.none", "Nessun peso selezionato.")); return
        self._total = len(ids); self._done = 0
        self.bar.setRange(0, self._total); self.bar.setValue(0); self.logview.clear()
        script = str(Path(__file__).resolve().parents[2] / "scripts" / "download_weights.py")
        p = QProcess(self); p.setProcessChannelMode(QProcess.MergedChannels)
        from PySide6.QtCore import QProcessEnvironment
        qenv = QProcessEnvironment.systemEnvironment()
        tok = self.hf.text().strip()
        if tok:
            qenv.insert("HF_TOKEN", tok); qenv.insert("HUGGING_FACE_HUB_TOKEN", tok)
        ckey = self.civitai.text().strip()
        if ckey:
            self._save_civitai_key()
            qenv.insert("CIVITAI_TOKEN", ckey)
        p.setProcessEnvironment(qenv)
        p.readyReadStandardOutput.connect(lambda: self._on_out(p))
        p.finished.connect(self._on_done)
        self._proc = p
        self.btn_dl.setEnabled(False); self.status.setText(tr("weights.downloading", "Scaricamento…"))
        p.start(sys.executable, ["-u", script, "--ids", ",".join(ids), "--base-dir", self.base.text().strip()])

    def _on_out(self, p: QProcess):
        for line in bytes(p.readAllStandardOutput()).decode("utf-8", "replace").splitlines():
            line = line.rstrip()
            if not line:
                continue
            self.logview.appendPlainText(line)
            if "✓" in line and "completato" in line:
                self._done += 1; self.bar.setValue(self._done)
        sb = self.logview.verticalScrollBar(); sb.setValue(sb.maximum())

    def _on_done(self, code, _s):
        self._proc = None; self.btn_dl.setEnabled(True)
        self.status.setText(tr("weights.finished", "Completato (codice {c})").format(c=code))
        self._refresh()
        self.inventory_changed.emit()                    # i dropdown si ripopolano senza riavvio
