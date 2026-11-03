"""Finestra NON modale «Come ottenere SAM 3.1»: stato, passi per chiedere l'accesso a Meta, riassunto della licenza, e il pulsante per tornare a SAM2.

Si apre dal menu «Segmentazione» quando SAM 3.1 non e utilizzabile (non installato, download incompleto, caricatore non pronto) e quando un lavoro
scelto con SAM 3.1 non puo partire. Mai un ripiego silenzioso su SAM2: la finestra dice cosa fare e l'utente decide.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from core import segmentation_engine as SE
from ui.i18n_manager import tr

_open: list = []


def build_text(status: str) -> str:
    """Testo (traducibile) per lo stato: 'missing' | 'partial' | 'loader' | 'ok'."""
    intro = {
        "missing": tr("seg.help.intro_missing", "SAM 3.1 non e installato. Nel frattempo puoi usare SAM2 (e gia pronto e funziona ovunque)."),
        "partial": tr("seg.help.intro_partial", "Il download di SAM 3.1 e incompleto (file mancanti o troncati). Riprendilo dal tab «Pesi modelli» o usa SAM2."),
        "loader": tr("seg.help.intro_loader", "SAM 3.1 e installato, ma il caricatore nativo non e ancora pronto: per ora usa SAM2."),
        "ok": tr("seg.help.intro_ok", "SAM 3.1 e pronto."),
    }.get(status, "")
    steps = tr("seg.help.steps",
               "Come ottenere l'accesso:\n"
               "1. Crea (o usa) un account su huggingface.co ed effettua l'accesso.\n"
               "2. Apri {url}, leggi e accetta la licenza SAM di Meta (compila il modulo).\n"
               "3. Meta deve approvare la richiesta: puo richiedere anche giorni. Finche e in attesa il download risponde 403: significa "
               "«richiesta in attesa di approvazione da Meta», non un errore dell'app.\n"
               "4. Quando e approvata, scarica «SAM 3.1» dal tab «Pesi modelli» (circa 3,5 GB) e scegli SAM 3.1 dal menu «Segmentazione».").format(url=SE.SAM31_URL)
    lic = tr("seg.help.license",
             "Licenza SAM (Meta, 19/11/2025): uso commerciale consentito. Vietati gli usi militari, nucleari, di spionaggio e il reverse engineering. "
             "Il testo completo e nel file LICENSE della cartella del modello e su {url}.").format(url=SE.SAM31_URL)
    return intro + "\n\n" + steps + "\n\n" + lic


def show(parent, status: str | None = None, on_use_sam2=None) -> QDialog:
    st = status
    if st is None:
        s, _bad = SE.sam31_status()
        st = "ok" if (s == "ok" and SE.LOADER_READY) else ("loader" if s == "ok" else s)
    dlg = QDialog(parent)
    dlg.setWindowTitle(tr("seg.help.title", "SAM 3.1 — come ottenerlo"))
    dlg.setModal(False)
    dlg.setAttribute(Qt.WA_DeleteOnClose, True)
    dlg.setMinimumWidth(520)
    lay = QVBoxLayout(dlg)
    lbl = QLabel(build_text(st))
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lay.addWidget(lbl)
    row = QHBoxLayout()
    b_open = QPushButton(tr("seg.help.btn_open", "Apri la pagina su Hugging Face"))
    b_open.setMinimumHeight(44)
    b_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(SE.SAM31_URL)))
    b_sam2 = QPushButton(tr("seg.help.btn_use_sam2", "Usa SAM2 ora"))
    b_sam2.setMinimumHeight(44)

    def _use_sam2():
        SE.set_current(SE.SAM2)
        if on_use_sam2:
            on_use_sam2()
        dlg.close()
    b_sam2.clicked.connect(_use_sam2)
    b_close = QPushButton(tr("seg.help.btn_close", "Chiudi"))
    b_close.setMinimumHeight(44)
    b_close.clicked.connect(dlg.close)
    for b in (b_open, b_sam2, b_close):
        row.addWidget(b)
    lay.addLayout(row)
    dlg.destroyed.connect(lambda *_: _open.remove(dlg) if dlg in _open else None)
    _open.append(dlg)
    dlg.show()
    return dlg
