"""Settings dialog — override del modello per ciascun nodo loader del template.

Popola le combo dai file realmente disponibili in ComfyUI (GET /models/{folder});
se ComfyUI è offline usa solo il valore corrente del template. Ritorna overrides
{node_id: {input: value}} che il template_engine applica al build.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QDialogButtonBox, QLabel, QScrollArea, QWidget,
)


class SettingsDialog(QDialog):
    def __init__(self, engine, driver, current_overrides: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Impostazioni — modelli per nodo")
        self.resize(640, 560)
        self.engine = engine
        self.driver = driver
        self._combos: dict[str, tuple[str, QComboBox]] = {}  # node -> (input, combo)
        self._overrides = current_overrides or {}

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Per ogni nodo loader puoi scegliere un modello diverso. "
                             "Le liste vengono dai file disponibili in ComfyUI."))
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); form = QFormLayout(inner)
        # cache liste per folder
        folder_cache: dict[str, list[str]] = {}
        for ln in self.engine.loader_nodes():
            folder = ln["folder"]
            if folder not in folder_cache:
                folder_cache[folder] = self.driver.models_in(folder) if self.driver.is_up() else []
            files = folder_cache[folder]
            combo = QComboBox(); combo.setEditable(True)
            opts = sorted(set(files) | {ln["current"]})
            combo.addItems(opts)
            # valore corrente: override esistente o quello del template
            cur = self._overrides.get(ln["node"], {}).get(ln["input"], ln["current"])
            combo.setCurrentText(cur)
            label = f"{ln['title']} [{ln['node']}] · {folder}"
            form.addRow(label, combo)
            self._combos[ln["node"]] = (ln["input"], combo)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def overrides(self) -> dict:
        out: dict[str, dict] = {}
        for node, (inp, combo) in self._combos.items():
            val = combo.currentText().strip()
            # registra solo se diverso dal valore originale del template
            orig = next((l["current"] for l in self.engine.loader_nodes() if l["node"] == node), None)
            if val and val != orig:
                out[node] = {inp: val}
        return out
