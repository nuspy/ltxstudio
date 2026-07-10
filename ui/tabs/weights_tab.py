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

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QCheckBox, QPushButton,
    QLineEdit, QScrollArea, QPlainTextEdit, QProgressBar, QFormLayout, QFileDialog,
)

from ui.i18n_manager import tr
from core import weights_catalog as wc


class WeightsTab(QWidget):
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
        root.addLayout(f)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); self._list = QVBoxLayout(inner); scroll.setWidget(inner)
        root.addWidget(scroll, 1)
        self._build_list()

        h = QHBoxLayout()
        self.chk_all = QCheckBox(tr("weights.show_all", "Mostra tutti i moduli"))
        self.chk_all.setToolTip(tr("weights.show_all.tt", "Disattiva il filtro per edizione/moduli presenti."))
        self.chk_all.toggled.connect(self._on_show_all)
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

    def _build_list(self):
        # only_wanted: mostra SOLO i pesi usati dai moduli PRESENTI nell'edizione (feature_matrix). «Mostra
        # tutti» disattiva il filtro (per curatela/debug).
        groups = wc.by_module(only_wanted=not self._show_all)
        if not groups and not self._show_all:            # edizione minimale / niente moduli → mostra tutto
            groups = wc.by_module()
        for module, entries in groups.items():
            box = QGroupBox(module); v = QVBoxLayout(box)
            for e in entries:
                cb = QCheckBox()
                self._checks[e["id"]] = (cb, e)
                v.addWidget(cb)
            self._list.addWidget(box)
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
        fam = f"[{e.get('type','')}/{e.get('family','')}]"
        if wc.is_present(e, base):
            state = "✓ " + tr("weights.present", "presente")
        elif e["id"] in self._found:
            state = "⇒ " + tr("weights.found", "già presente in: {p}").format(p=self._found[e["id"]])
        else:
            state = "• " + tr("weights.missing", "mancante")
        return f"{e.get('label', e['id'])} {fam} · ~{e.get('approx_gb','?')}GB{gated} — {state}"

    def _refresh(self):
        for _id, (cb, e) in self._checks.items():
            cb.setText(self._label_for(e))
            cb.setToolTip(str(wc.dest_for(e, self.base.text().strip() or None)))
            cb.setChecked(False)

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
        n = 0
        for _id, (cb, e) in self._checks.items():
            src = self._found.get(e["id"])
            if not src:
                continue
            dst = wc.dest_for(e, base)
            if e.get("kind") == "file":
                dst = dst / (e.get("filename") or Path(src).name)
            r = linkutil.link_or_copy(str(src), str(dst))
            self.logview.appendPlainText(f"[collega] {e['id']}: {src} → {r}")
            if r in ("hardlink", "junction", "symlink", "copy", "exists"):
                n += 1
        self._found.clear()
        self._refresh()
        self.status.setText(tr("weights.link.done", "Collegati {n} pesi (senza riscaricare).").format(n=n))

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

    def _select_missing(self):
        base = self.base.text().strip() or None
        for _id, (cb, e) in self._checks.items():
            cb.setChecked(not wc.is_present(e, base) and e["id"] not in self._found)

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
