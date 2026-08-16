# -*- coding: utf-8 -*-
"""Catalog Manager — app STANDALONE per la gestione CRUD di data/weights_catalog.json (schema v2).

Perché separata: il catalogo è il REGISTRO UNICO dei pesi (tab Pesi, installer, primo-uso,
download-on-select). Curarlo a mano nel JSON produceva voci rotte (9 non scaricabili, path col
profilo utente, vocabolario type incoerente). Qui: editor con validazioni + FEATURE MATRIX viva.

Feature matrix (richiesta esplicita):
  - le capability `used_by` si scelgono dalla LISTA REALE della matrix (niente typo);
  - per ogni voce si vede in tempo reale QUALI EDIZIONI la ricevono (cap × cells, con filtro
    adult sulle edizioni che non lo consentono);
  - `install_role: base` marca ciò che l'installer scarica per l'edizione (la stessa fonte
    che core.weights_catalog.base_weights_for_edition dà all'installer — niente più liste
    parallele in install_spec.json).

Uso:  python tools/catalog_manager.py            (qualsiasi python con PySide6)
Env:  LTX_MATRIX_JSON per un path matrix alternativo.

Strumento di SVILUPPO (non fa parte del prodotto distribuito): niente i18n/feature-gating
sull'app stessa; scrive con backup .bak e validazione JSON al salvataggio.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", ""))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from core import weights_catalog as wc  # noqa: E402

CATALOG = REPO / "data" / "weights_catalog.json"
MATRIX_DEFAULT = r"C:\Projects\feature_matrix\projects\ltxstudio\matrix.json"

TYPES = ["diffusion_models", "Checkpoints", "LoRA", "VAE", "vae", "TextEncoders",
         "text_encoders", "latent_upscale_models", "model_patches", "Upscale", "Segmentation",
         "facerestore_models", "TTS", "audio", "diffusers", "SeedVR2", "FlashVSR", "pulid",
         "mesh", "encoder", "segmentation", "depth", "skintoken", "checkpoint", "tts", "lora"]
CANONICAL_TYPES = {"lora": "LoRA", "tts": "TTS", "checkpoint": "Checkpoints",
                   "text_encoder": "TextEncoders", "segmentation": "Segmentation"}
ROLES = ["on_demand", "base"]
SENS = ["", "adult"]
KINDS = ["file", "snapshot"]


# ─────────────────────────── feature matrix (read-only) ───────────────────────────
class Matrix:
    def __init__(self):
        self.path = Path(os.environ.get("LTX_MATRIX_JSON", MATRIX_DEFAULT))
        self.caps: list[str] = []
        self.labels: dict[str, str] = {}
        self.editions: list[dict] = []
        self.cells: dict[str, dict] = {}
        self.ok = False
        try:
            data = json.loads(self.path.read_text("utf-8"))
            self.editions = data.get("editions", [])
            self.cells = data.get("cells", {})
            for f in data.get("features", []):
                if str(f.get("id", "")).startswith("cap."):
                    self.caps.append(f["id"])
                    self.labels[f["id"]] = f.get("label", f["id"])
            self.ok = bool(self.caps)
        except Exception as e:                               # noqa: BLE001
            print(f"[matrix] non caricata ({e}) → fallback MODULE_CAPS", flush=True)
            seen = set()
            for caps in wc.MODULE_CAPS.values():
                for c in caps:
                    if c not in seen:
                        seen.add(c)
                        self.caps.append(c)

    def enabled(self, cap: str, edition: str) -> bool:
        cell = self.cells.get(f"{cap}|{edition}")
        return bool(cell and cell.get("enabled"))

    def editions_for(self, entry: dict) -> list[str]:
        """Edizioni che RICEVONO questo peso: almeno una delle sue cap attiva, e se il peso è
        adult solo le edizioni con allows_adult (stessa logica di weights_catalog.wanted)."""
        caps = wc.caps_for(entry)
        out = []
        for ed in self.editions:
            eid = ed.get("id", "")
            if entry.get("sensitivity") == "adult" and not ed.get("allows_adult"):
                continue
            if not caps or any(self.enabled(c, eid) for c in caps):
                out.append(eid)
        return out


# ─────────────────────────── validazioni ───────────────────────────
def validate(entry: dict, all_ids: set[str], matrix: Matrix, *, is_new: bool) -> list[str]:
    errs = []
    eid = entry.get("id", "").strip()
    if not eid:
        errs.append("id vuoto")
    if is_new and eid in all_ids:
        errs.append(f"id duplicato: {eid}")
    if entry.get("kind") == "file" and not entry.get("filename"):
        errs.append("kind=file richiede filename")
    if not entry.get("no_remote") and not (entry.get("repo_id") or entry.get("url")):
        errs.append("nessuna fonte remota: imposta repo_id/url oppure marca no_remote "
                    "(mai voci 'scaricabili' senza fonte: finivano in KeyError)")
    td = str(entry.get("tool_dest", ""))
    if td and "\\Users\\" in td and not td.startswith("~"):
        errs.append("tool_dest con profilo utente hardcoded: usa ~ (portabilità)")
    t = entry.get("type", "")
    if t in CANONICAL_TYPES:
        errs.append(f"type '{t}' non canonico → usa '{CANONICAL_TYPES[t]}'")
    for c in wc.caps_for(entry):
        if matrix.ok and c not in matrix.caps:
            errs.append(f"capability '{c}' NON esiste nella feature matrix")
    for d in entry.get("deps", []):
        if d not in all_ids and d != eid:
            errs.append(f"dep sconosciuta: {d}")
    if entry.get("category") not in wc.CATEGORIES:
        errs.append(f"category '{entry.get('category')}' non valida {wc.CATEGORIES}")
    return errs


# ─────────────────────────── GUI ───────────────────────────
class Manager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LTX Studio — Catalog Manager (weights_catalog v2)")
        self.resize(1240, 780)
        self.matrix = Matrix()
        self.data = json.loads(CATALOG.read_text("utf-8"))
        self.weights: list[dict] = self.data.get("weights", [])
        self._current: dict | None = None
        self._build()
        self._reload_tree()

    # ---- layout ----
    def _build(self):
        split = QSplitter()
        self.setCentralWidget(split)

        left = QWidget(); lv = QVBoxLayout(left)
        self.search = QLineEdit(placeholderText="cerca id/label/repo…")
        self.search.textChanged.connect(self._reload_tree)
        lv.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["voce", "stato"])
        self.tree.setColumnWidth(0, 340)
        self.tree.itemSelectionChanged.connect(self._on_select)
        lv.addWidget(self.tree, 1)
        row = QHBoxLayout()
        for lbl, fn in (("＋ Nuovo", self._new), ("⧉ Duplica", self._dup), ("🗑 Elimina", self._del)):
            b = QPushButton(lbl); b.clicked.connect(fn); row.addWidget(b)
        lv.addLayout(row)
        split.addWidget(left)

        right = QWidget(); rv = QVBoxLayout(right)
        form_box = QGroupBox("Voce"); form = QFormLayout(form_box)
        self.f_id = QLineEdit(); form.addRow("id", self.f_id)
        self.f_label = QLineEdit(); form.addRow("label", self.f_label)
        self.f_category = QComboBox(); self.f_category.addItems(list(wc.CATEGORIES))
        form.addRow("category", self.f_category)
        self.f_arch = QComboBox(); self.f_arch.setEditable(True)
        form.addRow("arch", self.f_arch)
        self.f_module = QComboBox(); self.f_module.setEditable(True)
        self.f_module.addItems(sorted(wc.MODULE_CAPS))
        form.addRow("module (gruppo GUI/caps)", self.f_module)
        self.f_kind = QComboBox(); self.f_kind.addItems(KINDS); form.addRow("kind", self.f_kind)
        self.f_repo = QLineEdit(placeholderText="owner/repo (HF)"); form.addRow("repo_id", self.f_repo)
        self.f_remote = QLineEdit(placeholderText="path nel repo se ≠ dal nome locale")
        form.addRow("remote_filename", self.f_remote)
        self.f_filename = QLineEdit(); form.addRow("filename (locale)", self.f_filename)
        self.f_url = QLineEdit(); form.addRow("url", self.f_url)
        self.f_type = QComboBox(); self.f_type.setEditable(True)
        self.f_type.addItems(sorted(set(TYPES) - set(CANONICAL_TYPES))); form.addRow("type", self.f_type)
        self.f_family = QLineEdit(placeholderText="vuoto = radice del type"); form.addRow("family", self.f_family)
        self.f_tool_dest = QLineEdit(placeholderText="path fisso (ignora base_dir); ~ ok")
        form.addRow("tool_dest", self.f_tool_dest)
        self.f_gb = QDoubleSpinBox(); self.f_gb.setRange(0, 500); self.f_gb.setDecimals(2)
        form.addRow("approx_gb", self.f_gb)
        self.f_gated = QCheckBox("repo gated (licenza HF + token)"); form.addRow("", self.f_gated)
        self.f_no_remote = QCheckBox("no_remote (solo scan/link, nessun download)")
        form.addRow("", self.f_no_remote)
        self.f_role = QComboBox(); self.f_role.addItems(ROLES); form.addRow("install_role", self.f_role)
        self.f_sens = QComboBox(); self.f_sens.addItems(SENS); form.addRow("sensitivity", self.f_sens)
        self.f_deps = QLineEdit(placeholderText="id separati da virgola"); form.addRow("deps", self.f_deps)
        self.f_scan = QLineEdit(placeholderText="scan_names separati da virgola"); form.addRow("scan_names", self.f_scan)
        self.f_allow = QLineEdit(placeholderText="allow_patterns snapshot, separati da virgola")
        form.addRow("allow_patterns", self.f_allow)
        self.f_note = QPlainTextEdit(); self.f_note.setMaximumHeight(52); form.addRow("note", self.f_note)
        rv.addWidget(form_box)

        caps_box = QGroupBox("Capability (used_by) — dalla feature matrix"
                             + ("" if self.matrix.ok else " [matrix NON caricata: fallback]"))
        cv = QVBoxLayout(caps_box)
        self.caps_list = QListWidget(); self.caps_list.setMaximumHeight(140)
        for c in self.matrix.caps:
            it = QListWidgetItem(f"{c} — {self.matrix.labels.get(c, '')}".rstrip(" —"))
            it.setData(Qt.UserRole, c)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.caps_list.addItem(it)
        self.caps_list.itemChanged.connect(self._refresh_preview)
        cv.addWidget(self.caps_list)
        self.caps_hint = QLabel("(nessuna spunta = caps ereditate dal module)")
        self.caps_hint.setStyleSheet("color:#9aa1b0;")
        cv.addWidget(self.caps_hint)
        rv.addWidget(caps_box)

        self.preview = QLabel(); self.preview.setWordWrap(True)
        self.preview.setStyleSheet("background:#20232b;color:#cdd3e0;padding:6px;border-radius:4px;")
        rv.addWidget(self.preview)

        act = QHBoxLayout()
        b_save = QPushButton("💾 Salva voce"); b_save.clicked.connect(self._save_entry); act.addWidget(b_save)
        b_write = QPushButton("⬇ Scrivi catalogo su disco"); b_write.clicked.connect(self._write); act.addWidget(b_write)
        b_reload = QPushButton("↺ Ricarica da disco"); b_reload.clicked.connect(self._reload_disk); act.addWidget(b_reload)
        rv.addLayout(act)
        split.addWidget(right)
        split.setSizes([460, 780])

        for w in (self.f_sens, self.f_module, self.f_no_remote):
            (w.currentTextChanged if isinstance(w, QComboBox) else w.stateChanged).connect(
                lambda *_: self._refresh_preview())

    # ---- tree ----
    def _reload_tree(self):
        q = self.search.text().lower().strip()
        self.tree.clear()
        arches = sorted({e.get("arch", "generic") for e in self.weights})
        self.f_arch.clear(); self.f_arch.addItems(arches)
        groups: dict[str, dict[str, list[dict]]] = {}
        for e in self.weights:
            if q and q not in json.dumps(e, ensure_ascii=False).lower():
                continue
            groups.setdefault(e.get("category", "aux"), {}).setdefault(
                e.get("arch", "generic"), []).append(e)
        for cat in wc.CATEGORIES:
            if cat not in groups:
                continue
            ci = QTreeWidgetItem([f"{wc.CATEGORY_LABELS.get(cat, cat)} ({sum(len(v) for v in groups[cat].values())})"])
            self.tree.addTopLevelItem(ci)
            for arch in sorted(groups[cat]):
                ai = QTreeWidgetItem([arch]); ci.addChild(ai)
                for e in groups[cat][arch]:
                    st = "∅ no_remote" if e.get("no_remote") else (
                        "✓" if wc.is_present(e) else "⬇")
                    if e.get("sensitivity") == "adult":
                        st += " 🔞"
                    if e.get("install_role") == "base":
                        st += " [base]"
                    it = QTreeWidgetItem([f"{e['id']} — {e.get('label', '')}", st])
                    it.setData(0, Qt.UserRole, e["id"])
                    ai.addChild(it)
            ci.setExpanded(True)

    def _entry(self, eid: str) -> dict | None:
        return next((e for e in self.weights if e["id"] == eid), None)

    def _on_select(self):
        it = self.tree.currentItem()
        eid = it.data(0, Qt.UserRole) if it else None
        if not eid:
            return
        e = self._entry(eid)
        if e:
            self._load_form(e)

    # ---- form ----
    def _load_form(self, e: dict):
        self._current = e
        self.f_id.setText(e.get("id", "")); self.f_id.setReadOnly(True)
        self.f_label.setText(e.get("label", ""))
        self.f_category.setCurrentText(e.get("category", "aux"))
        self.f_arch.setCurrentText(e.get("arch", "generic"))
        self.f_module.setCurrentText(e.get("module", ""))
        self.f_kind.setCurrentText(e.get("kind", "file"))
        self.f_repo.setText(e.get("repo_id", ""))
        self.f_remote.setText(e.get("remote_filename", ""))
        self.f_filename.setText(e.get("filename", ""))
        self.f_url.setText(e.get("url", ""))
        self.f_type.setCurrentText(e.get("type", ""))
        self.f_family.setText(e.get("family", ""))
        self.f_tool_dest.setText(e.get("tool_dest", ""))
        self.f_gb.setValue(float(e.get("approx_gb", 0) or 0))
        self.f_gated.setChecked(bool(e.get("gated")))
        self.f_no_remote.setChecked(bool(e.get("no_remote")))
        self.f_role.setCurrentText(e.get("install_role", "on_demand"))
        self.f_sens.setCurrentText(e.get("sensitivity", ""))
        self.f_deps.setText(",".join(e.get("deps", [])))
        self.f_scan.setText(",".join(e.get("scan_names", [])))
        self.f_allow.setText(",".join(e.get("allow_patterns", [])))
        self.f_note.setPlainText(e.get("note", ""))
        used = set(e.get("used_by", []))
        for i in range(self.caps_list.count()):
            it = self.caps_list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in used else Qt.Unchecked)
        self._refresh_preview()

    def _form_entry(self) -> dict:
        e = dict(self._current or {})
        e["id"] = self.f_id.text().strip()
        e["label"] = self.f_label.text().strip()
        e["category"] = self.f_category.currentText()
        e["arch"] = self.f_arch.currentText().strip() or "generic"
        e["module"] = self.f_module.currentText().strip()
        e["kind"] = self.f_kind.currentText()
        e["type"] = self.f_type.currentText().strip()
        e["family"] = self.f_family.text().strip()
        e["approx_gb"] = round(self.f_gb.value(), 2)
        e["gated"] = self.f_gated.isChecked()
        for key, w in (("repo_id", self.f_repo), ("remote_filename", self.f_remote),
                       ("filename", self.f_filename), ("url", self.f_url),
                       ("tool_dest", self.f_tool_dest)):
            v = w.text().strip()
            if v:
                e[key] = v
            else:
                e.pop(key, None)
        for key, w in (("deps", self.f_deps), ("scan_names", self.f_scan),
                       ("allow_patterns", self.f_allow)):
            vals = [x.strip() for x in w.text().split(",") if x.strip()]
            if vals:
                e[key] = vals
            else:
                e.pop(key, None)
        if self.f_no_remote.isChecked():
            e["no_remote"] = True
        else:
            e.pop("no_remote", None)
        if self.f_role.currentText() != "on_demand":
            e["install_role"] = self.f_role.currentText()
        else:
            e.pop("install_role", None)
        if self.f_sens.currentText():
            e["sensitivity"] = self.f_sens.currentText()
        else:
            e.pop("sensitivity", None)
        note = self.f_note.toPlainText().strip()
        if note:
            e["note"] = note
        else:
            e.pop("note", None)
        used = [self.caps_list.item(i).data(Qt.UserRole) for i in range(self.caps_list.count())
                if self.caps_list.item(i).checkState() == Qt.Checked]
        if used:
            e["used_by"] = used
        else:
            e.pop("used_by", None)
        return e

    def _refresh_preview(self):
        try:
            e = self._form_entry()
        except Exception:                                    # noqa: BLE001
            return
        dest = wc.dest_for(e)
        eds = self.matrix.editions_for(e) if self.matrix.editions else []
        caps = wc.caps_for(e)
        self.preview.setText(
            f"dest → {dest}\n"
            f"caps effettive: {', '.join(caps) or '(nessuna: visibile ovunque)'}\n"
            f"edizioni che ricevono il peso: {', '.join(eds) or '(matrix non caricata)'}"
            + ("\n⚠ install_role=base → l'INSTALLER lo scarica per queste edizioni"
               if e.get("install_role") == "base" else ""))

    # ---- azioni ----
    def _save_entry(self):
        e = self._form_entry()
        is_new = self._current is None or self._current.get("id") != e["id"]
        ids = {x["id"] for x in self.weights if x is not self._current}
        errs = validate(e, ids, self.matrix, is_new=is_new)
        if errs:
            QMessageBox.warning(self, "Validazione", "\n".join("• " + x for x in errs))
            return
        if self._current is not None and self._current in self.weights:
            self.weights[self.weights.index(self._current)] = e
        else:
            self.weights.append(e)
        self._current = e
        self._reload_tree()
        self.statusBar().showMessage(f"voce '{e['id']}' salvata in memoria — ricorda «Scrivi catalogo»", 5000)

    def _new(self):
        self._current = None
        self.f_id.setReadOnly(False)
        for w in (self.f_id, self.f_label, self.f_repo, self.f_remote, self.f_filename,
                  self.f_url, self.f_family, self.f_tool_dest, self.f_deps, self.f_scan,
                  self.f_allow):
            w.clear()
        self.f_note.setPlainText("")
        self.f_gb.setValue(0)
        self.f_gated.setChecked(False)
        self.f_no_remote.setChecked(False)
        self.f_role.setCurrentText("on_demand")
        self.f_sens.setCurrentText("")
        for i in range(self.caps_list.count()):
            self.caps_list.item(i).setCheckState(Qt.Unchecked)

    def _dup(self):
        if self._current:
            e = dict(self._current)
            e["id"] = e["id"] + "-copia"
            self._current = None
            self.f_id.setReadOnly(False)
            self._load_form(e)
            self._current = None                             # resta "nuova" finché non salvi
            self.f_id.setReadOnly(False)

    def _del(self):
        if not self._current:
            return
        eid = self._current.get("id", "")
        dep_of = [x["id"] for x in self.weights if eid in x.get("deps", [])]
        msg = f"Eliminare la voce '{eid}'?"
        if dep_of:
            msg += f"\n⚠ è dipendenza di: {', '.join(dep_of)}"
        if QMessageBox.question(self, "Elimina", msg) == QMessageBox.Yes:
            self.weights.remove(self._current)
            self._current = None
            self._reload_tree()

    def _write(self):
        self.data["weights"] = self.weights
        self.data["schema"] = 2
        shutil.copy2(CATALOG, CATALOG.with_suffix(".json.bak"))
        CATALOG.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", "utf-8")
        json.loads(CATALOG.read_text("utf-8"))               # validazione post-scrittura
        base = wc.base_weights_for_edition(allows_adult=True)
        self.statusBar().showMessage(
            f"catalogo scritto ({len(self.weights)} voci, {len(base)} base per l'installer) — backup .bak", 8000)

    def _reload_disk(self):
        self.data = json.loads(CATALOG.read_text("utf-8"))
        self.weights = self.data.get("weights", [])
        self._current = None
        self._reload_tree()


def main() -> int:
    app = QApplication(sys.argv)
    m = Manager()
    m.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
