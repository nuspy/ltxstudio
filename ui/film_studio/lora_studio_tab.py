"""Tab LoRA Studio (spec §7): libreria LoRA condivisa + scoping (serie/episodio) + binding ad asset +
composizione multi-LoRA (warning soglia). Il TRAINING riusa i tab esistenti (LoRA Video/Foto). Zero ComfyUI.

Le LoRA si scoprono da: DB film_studio (tabella lora) + filesystem output/loras/*.safetensors.
"""
from __future__ import annotations

from pathlib import Path

import json

from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from core.film_studio import db
from ui.i18n_manager import tr

LORAS_DIR = Path(__file__).resolve().parents[2] / "output" / "loras"
MAX_SIMULTANEOUS = 3            # soglia di warning per composizione multi-LoRA (spec §7.5)


class _PipeWorker(QThread):
    """Esegue uno step (o la selezione) dell'orchestratore in background (i training sono lunghi)."""
    logged = Signal(str)
    gated = Signal(int, list)                                # (pipeline_id, foto-prova) → gate di selezione
    finished_state = Signal(str)

    def __init__(self, pid: int, select: list | None = None):
        super().__init__()
        self._pid = pid
        self._select = select                               # None = step; lista = select_photos (gate risolto)

    def run(self):
        from core.film_studio import lora_orchestrator as O
        cb = lambda m: self.logged.emit(str(m))             # noqa: E731
        if self._select is not None:
            st = O.select_photos(self._pid, self._select, on_log=cb)
        else:
            st = O.step(self._pid, on_log=cb)
        if st == O.REVIEW:
            photos = json.loads((db.get("lora_pipeline", self._pid) or {}).get("review_photos") or "[]")
            self.gated.emit(self._pid, photos)
        else:
            self.finished_state.emit(st)


class LoraStudioTab(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        info = QLabel(tr("fs.lorastudio.info",
            "Libreria LoRA condivisa. Il training si fa nei tab «LoRA Video» e «LoRA Foto» dell'app."))
        info.setWordWrap(True); v.addWidget(info)
        h = QHBoxLayout()
        b_ref = QPushButton(tr("fs.mt.refresh", "↻ Aggiorna")); b_ref.clicked.connect(self._refresh)
        h.addWidget(b_ref); h.addStretch(1)
        v.addLayout(h)
        self.lst = QListWidget(); self.lst.currentRowChanged.connect(self._select)
        v.addWidget(self.lst, 1)
        box = QGroupBox(tr("fs.lorastudio.details", "Dettagli LoRA")); bv = QVBoxLayout(box)
        r1 = QHBoxLayout()
        self.trigger = QLineEdit(); self.trigger.setPlaceholderText(tr("fs.lorastudio.trigger", "trigger token"))
        r1.addWidget(QLabel(tr("fs.lorastudio.trigger", "Trigger"))); r1.addWidget(self.trigger, 1)
        self.flavor = QComboBox(); self.flavor.addItems(["image", "video"])
        r1.addWidget(QLabel(tr("fs.lorastudio.flavor", "Tipo"))); r1.addWidget(self.flavor)
        self.scope = QComboBox()
        self.scope.addItem(tr("fs.lorastudio.scope_ep", "Episodio"), "episode")
        self.scope.addItem(tr("fs.lorastudio.scope_se", "Serie"), "series")
        r1.addWidget(QLabel(tr("fs.lorastudio.scope", "Scope"))); r1.addWidget(self.scope)
        bv.addLayout(r1)
        r2 = QHBoxLayout()
        self.asset = QComboBox()
        r2.addWidget(QLabel(tr("fs.lorastudio.asset", "Asset"))); r2.addWidget(self.asset, 1)
        b_save = QPushButton(tr("fs.save", "💾 Salva")); b_save.clicked.connect(self._save)
        r2.addWidget(b_save)
        bv.addLayout(r2)
        v.addWidget(box)
        # ── Pipeline personaggio img→video (orchestratore con gate) ──
        pbox = QGroupBox(tr("fs.lorastudio.pipe", "🎬 Pipeline personaggio (immagine → video)")); pv = QVBoxLayout(pbox)
        pr = QHBoxLayout()
        self.pl_asset = QComboBox()
        pr.addWidget(QLabel(tr("fs.lorastudio.asset", "Asset"))); pr.addWidget(self.pl_asset, 1)
        self.pl_photos = QLineEdit(); self.pl_photos.setPlaceholderText(
            tr("fs.lorastudio.pipe_photos", "cartella foto (vuoto = usa i riferimenti dell'asset)"))
        b_br = QPushButton("…"); b_br.clicked.connect(self._pick_photos)
        pr.addWidget(self.pl_photos, 1); pr.addWidget(b_br)
        self.pl_mode = QComboBox()
        self.pl_mode.addItem(tr("fs.lorastudio.pipe_sup", "Supervisionata (gate)"), "supervised")
        self.pl_mode.addItem(tr("fs.lorastudio.pipe_auto", "Automatica"), "auto")
        pr.addWidget(self.pl_mode)
        self.b_run = QPushButton(tr("fs.lorastudio.pipe_run", "▶ Avvia pipeline")); self.b_run.clicked.connect(self._run_pipeline)
        pr.addWidget(self.b_run)
        pv.addLayout(pr)
        self.pl_log = QPlainTextEdit(); self.pl_log.setReadOnly(True); self.pl_log.setFixedHeight(90)
        pv.addWidget(self.pl_log)
        v.addWidget(pbox)
        self.status = QLabel(""); v.addWidget(self.status)
        self._refresh()

    def _scan(self) -> list[dict]:
        """LoRA da DB (registrate) + file .safetensors orfani in output/loras."""
        rows = db.query("lora", order="id DESC")
        known = {Path(r.get("file", "")).name for r in rows if r.get("file")}
        items = list(rows)
        if LORAS_DIR.exists():
            for f in LORAS_DIR.rglob("*.safetensors"):
                if f.name not in known:
                    items.append({"id": 0, "file": str(f), "trigger": f.stem, "flavor": "image",
                                  "scope": "episode", "asset_id": 0, "_orphan": True})
        return items

    def _refresh(self):
        self.lst.clear()
        self._items = self._scan()
        for it in self._items:
            tag = "·DB" if it.get("id") else "·file"
            self.lst.addItem(f"[{it.get('flavor','?')}/{it.get('scope','?')}] {it.get('trigger') or Path(it['file']).stem} {tag}")
        self.asset.clear()
        self._assets = db.find_assets()
        self.asset.addItem("—", 0)
        for a in self._assets:
            self.asset.addItem(f"[{a['type']}] {a['name']}", a["id"])
        if hasattr(self, "pl_asset"):
            self.pl_asset.clear()
            for a in self._assets:
                self.pl_asset.addItem(f"[{a['type']}] {a['name']}", a["id"])

    def _select(self, row: int):
        if not (0 <= row < len(getattr(self, "_items", []))):
            return
        it = self._items[row]
        self.trigger.setText(it.get("trigger") or "")
        self.flavor.setCurrentText(it.get("flavor") or "image")
        self.scope.setCurrentIndex(0 if it.get("scope") != "series" else 1)
        idx = self.asset.findData(it.get("asset_id") or 0)
        self.asset.setCurrentIndex(max(0, idx))

    def _save(self):
        row = self.lst.currentRow()
        if not (0 <= row < len(getattr(self, "_items", []))):
            return
        it = self._items[row]
        fields = {"trigger": self.trigger.text().strip(), "flavor": self.flavor.currentText(),
                  "scope": self.scope.currentData(), "asset_id": self.asset.currentData() or 0}
        if it.get("id"):
            db.update("lora", it["id"], **fields)
        else:                                               # registra il file orfano nel DB
            db.insert("lora", file=it["file"], version=1, dataset="", **fields)
        self._refresh()
        self.status.setText(tr("fs.lorastudio.saved", "LoRA registrata/aggiornata."))

    # ─────────────── Pipeline personaggio img→video ───────────────
    def _pick_photos(self):
        d = QFileDialog.getExistingDirectory(self, tr("fs.lorastudio.pipe_photos", "Cartella foto"))
        if d:
            self.pl_photos.setText(d)

    def _gather_photos(self, aid: int) -> list[str]:
        import os
        folder = self.pl_photos.text().strip()
        if folder and os.path.isdir(folder):
            return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        return [i["file"] for i in db.query("image", "owner_kind='asset' AND owner_id=?", (aid,)) if i.get("file")]

    def _run_pipeline(self):
        aid = self.pl_asset.currentData()
        if not aid:
            QMessageBox.information(self, tr("fs.lorastudio.pipe", "Pipeline"),
                                    tr("fs.lorastudio.pipe_noasset", "Seleziona un asset.")); return
        photos = self._gather_photos(aid)
        if not photos:
            QMessageBox.information(self, tr("fs.lorastudio.pipe", "Pipeline"),
                                    tr("fs.lorastudio.pipe_nophoto", "Nessuna foto (cartella o riferimenti asset).")); return
        name = next((a["name"] for a in self._assets if a["id"] == aid), f"char{aid}")
        spec = {"trigger": self.trigger.text().strip() or name.lower().replace(" ", "_"),
                "photos": photos, "face_focus": True, "type": "person"}
        from core.film_studio import lora_orchestrator as O
        pid = O.create_pipeline(aid, name, spec, mode=self.pl_mode.currentData())
        self.pl_log.appendPlainText(tr("fs.lorastudio.pipe_start", "Pipeline avviata: {n} foto").format(n=len(photos)))
        self.b_run.setEnabled(False)
        self._start_worker(pid, None)

    def _start_worker(self, pid: int, select):
        self._pw = _PipeWorker(pid, select)
        self._pw.logged.connect(lambda m: self.pl_log.appendPlainText(m))
        self._pw.gated.connect(self._on_gate)
        self._pw.finished_state.connect(self._on_pipe_done)
        self._pw.start()

    def _on_gate(self, pid: int, photos: list):
        """GATE: mostra le foto-prova, l'utente sceglie quelle buone → training video."""
        chosen = self._select_photos_dialog(photos)
        if not chosen:
            self.pl_log.appendPlainText(tr("fs.lorastudio.pipe_cancel", "Gate annullato — pipeline in pausa."))
            self.b_run.setEnabled(True); return
        self.pl_log.appendPlainText(tr("fs.lorastudio.pipe_sel", "{n} foto scelte → training video…").format(n=len(chosen)))
        self._start_worker(pid, chosen)

    def _on_pipe_done(self, state: str):
        self.b_run.setEnabled(True)
        self.pl_log.appendPlainText(tr("fs.lorastudio.pipe_done", "Pipeline: stato «{s}»").format(s=state))
        self._refresh()

    def _select_photos_dialog(self, photos: list) -> list:
        dlg = QDialog(self); dlg.setWindowTitle(tr("fs.lorastudio.pipe_gate", "Scegli le foto buone"))
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(tr("fs.lorastudio.pipe_gate_h", "Seleziona le foto che rappresentano bene il personaggio:")))
        lst = QListWidget(); lst.setIconSize(QSize(96, 96)); lst.setViewMode(QListWidget.IconMode); lst.setResizeMode(QListWidget.Adjust)
        for p in photos:
            it = QListWidgetItem(QIcon(p), Path(p).name); it.setData(Qt.UserRole, p)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable); it.setCheckState(Qt.Checked)
            lst.addItem(it)
        lay.addWidget(lst, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); lay.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return []
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]

    def loras_for_assets(self, asset_ids: list[int], flavor: str = "image") -> list[dict]:
        """Per la composizione: LoRA legate agli asset citati. Avvisa se troppe simultanee."""
        out = []
        for aid in asset_ids:
            for r in db.query("lora", "asset_id=? AND flavor=?", (aid, flavor), order="version"):
                out.append({"file": r["file"], "trigger": r["trigger"], "weight": 0.9})
        if len(out) > MAX_SIMULTANEOUS:
            self.status.setText(tr("fs.lorastudio.too_many",
                "Attenzione: {n} LoRA simultanee (>{m}) — valuta split inquadratura o Percorso Rapido.")
                .format(n=len(out), m=MAX_SIMULTANEOUS))
        return out
