"""Tab LoRA Studio (spec §7): libreria LoRA condivisa + scoping (serie/episodio) + binding ad asset +
composizione multi-LoRA (warning soglia). Il TRAINING riusa i tab esistenti (LoRA Video/Foto). Zero ComfyUI.

Le LoRA si scoprono da: DB film_studio (tabella lora) + filesystem output/loras/*.safetensors.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from core.film_studio import db
from ui.i18n_manager import tr

LORAS_DIR = Path(__file__).resolve().parents[2] / "output" / "loras"
MAX_SIMULTANEOUS = 3            # soglia di warning per composizione multi-LoRA (spec §7.5)


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
