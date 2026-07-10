"""Wizard «Setup modelli» al PRIMO AVVIO — riusa interamente `WeightsTab` (nessuna duplicazione).

Flusso guidato: ① cartella base → ② «Cerca esistenti» (scan ricorsivo) → ③ «Collega trovati» (junction/hardlink,
no download) / «Importa sconosciuti» → ④ «Seleziona mancanti» + «Scarica» (solo i pesi dell'edizione).

Mostrato solo se non c'è ancora una configurazione pesi né modelli presenti. Disattivabile con LTX_NO_SETUP=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from core import weights_catalog as wc
from ui.i18n_manager import tr

_CFG = Path(__file__).resolve().parents[2] / "data" / "weights_config.json"


def should_run() -> bool:
    """Primo avvio = setup mai completato/saltato E nessun modello base già presente."""
    if os.environ.get("LTX_NO_SETUP") == "1":
        return False
    try:
        if json.loads(_CFG.read_text("utf-8")).get("setup_done"):
            return False
    except Exception:                                       # noqa: BLE001
        pass
    try:                                                    # se ci sono già pesi presenti → niente wizard
        base = wc.get_base_dir()
        if any(wc.is_present(e, base) for e in wc.load()):
            return False
    except Exception:                                       # noqa: BLE001
        pass
    return True


def _mark_done() -> None:
    try:
        cfg = {}
        try:
            cfg = json.loads(_CFG.read_text("utf-8"))
        except Exception:                                   # noqa: BLE001
            pass
        cfg["setup_done"] = True
        _CFG.parent.mkdir(parents=True, exist_ok=True)
        _CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", "utf-8")
    except Exception:                                       # noqa: BLE001
        pass


class ModelSetupWizard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("setup.title", "Setup modelli — LTX Studio"))
        self.resize(940, 700)
        v = QVBoxLayout(self)
        head = QLabel(tr("setup.head",
            "Benvenuto! Prima di iniziare, indica dove tieni già i modelli AI: li COLLEGO (senza copiarli né "
            "riscaricarli) e poi scarichi solo i pesi che servono ai moduli della tua edizione.\n"
            "①  Cartella base   →   ②  «Cerca esistenti»   →   ③  «🔗 Collega trovati» / «＋ Importa sconosciuti»"
            "   →   ④  «Seleziona mancanti» + «Scarica»."))
        head.setWordWrap(True)
        head.setStyleSheet("background:#26333f;color:#d6e6f2;padding:8px 10px;border-radius:6px;")
        v.addWidget(head)
        from ui.tabs.weights_tab import WeightsTab
        self.weights = WeightsTab()
        v.addWidget(self.weights, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        skip = QPushButton(tr("setup.skip", "Salta per ora")); skip.clicked.connect(self.reject)
        done = QPushButton(tr("setup.done", "Fatto ✓")); done.clicked.connect(self.accept)
        row.addWidget(skip); row.addWidget(done)
        v.addLayout(row)

    def accept(self):
        _mark_done()
        super().accept()

    def reject(self):
        _mark_done()                                        # anche se saltato: non ripresentarlo ad ogni avvio
        super().reject()


def maybe_show(parent=None) -> bool:
    """Se è il primo avvio, mostra il wizard. Ritorna True se mostrato. Non solleva mai (non deve bloccare l'app)."""
    if not should_run():
        return False
    try:
        ModelSetupWizard(parent).exec()
        return True
    except Exception:                                       # noqa: BLE001
        return False
