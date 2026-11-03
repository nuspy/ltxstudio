"""Finestra NON modale «SAM 3.1: cosa fa e quando sceglierlo»: spiega a cosa serve rispetto a SAM2 e, se i pesi mancano, offre il normale download del catalogo.

Niente account, token, approvazioni o controlli periodici: i pesi sono copie NON gated (unsloth/sam3.1, Comfy-Org/sam3.1). Mai un ripiego silenzioso su SAM2:
la finestra dice cosa fare e l'utente decide.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from core import segmentation_engine as SE
from ui.i18n_manager import tr

_open: list = []


def status_text(status: str) -> str:
    return {
        "missing": tr("sam31.help.not_installed", "SAM 3.1 non e installato: scaricalo (circa 1,8 GB, senza account ne approvazioni) oppure usa SAM2, che e gia pronto."),
        "partial": tr("seg.help.intro_partial", "Il download di SAM 3.1 e incompleto (file mancanti o troncati). Riprendilo dal tab «Pesi modelli» o usa SAM2."),
        "loader": tr("seg.help.intro_loader", "SAM 3.1 e installato, ma il caricatore nativo non e ancora pronto: per ora usa SAM2."),
        "ok": tr("seg.help.intro_ok", "SAM 3.1 e pronto."),
    }.get(status, "")


def build_text(status: str) -> str:
    """Testo completo (traducibile): stato, cosa fa SAM 3.1, quando sceglierlo, quando SAM2, licenza."""
    parts = [
        status_text(status),
        tr("sam31.help.what", "SAM 3.1 (Meta) trova gli oggetti da una descrizione scritta («persona», «auto rossa») e ne disegna la maschera, anche per piu oggetti "
                              "insieme. Nei video trova l'oggetto sul primo fotogramma da testo, al posto di GroundingDINO, e SAM2 lo segue nel tempo."),
        tr("sam31.help.choose_sam31", "Scegli SAM 3.1 quando selezioni per nome: selezione da testo, «Trova persone», tracking di un oggetto descritto a parole."),
        tr("sam31.help.choose_sam2", "Scegli SAM2 per clic, lazo e box sulla foto e quando la memoria della scheda e poca: e piu leggero e funziona ovunque."),
        tr("seg.help.license", "Licenza SAM (Meta, 19/11/2025): uso commerciale consentito. Vietati gli usi militari, nucleari, di spionaggio e il reverse engineering. "
                               "Il testo completo e nel file LICENSE della cartella del modello e su {url}.").format(url=SE.SAM31_URL),
    ]
    return "\n\n".join(p for p in parts if p)


def show(parent, status: str | None = None, on_use_sam2=None, on_download=None) -> QDialog:
    """`on_download()` = avvia il normale download dei pesi (dialog di consenso del catalogo); senza, il pulsante non compare."""
    st = status
    if st is None:
        s, _bad = SE.sam31_status()
        st = "ok" if (s == "ok" and SE.LOADER_READY) else ("loader" if s == "ok" else s)
    dlg = QDialog(parent)
    dlg.setWindowTitle(tr("sam31.help.title", "SAM 3.1: cosa fa e quando sceglierlo"))
    dlg.setModal(False)
    dlg.setAttribute(Qt.WA_DeleteOnClose, True)
    dlg.setMinimumWidth(540)
    lay = QVBoxLayout(dlg)
    lbl = QLabel(build_text(st))
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lay.addWidget(lbl)
    dlg.text_label = lbl
    row = QHBoxLayout()
    dlg.btn_download = QPushButton(tr("sam31.help.btn_download", "Scarica SAM 3.1 (circa 1,8 GB)"))
    dlg.btn_download.setMinimumHeight(44)
    dlg.btn_download.setVisible(bool(on_download) and st in ("missing", "partial"))
    if on_download:
        dlg.btn_download.clicked.connect(lambda: (on_download(), dlg.close()))
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
    for b in (dlg.btn_download, b_sam2, b_close):
        row.addWidget(b)
    lay.addLayout(row)
    dlg.destroyed.connect(lambda *_: _open.remove(dlg) if dlg in _open else None)
    _open.append(dlg)
    dlg.show()
    return dlg
