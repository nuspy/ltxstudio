"""Messaggi dell'app NON MODALI, con testo selezionabile e scorrevole (punto 8 dell'utente, 30/10/2026).

`QMessageBox.information/warning/critical` (statici) bloccavano l'interfaccia, tagliavano i testi lunghi (traceback, elenchi)
e non permettevano di copiare. `install()` li sostituisce con una finestra NON modale: titolo, icona, testo in una vista di
sola lettura selezionabile/scorrevole, pulsante «OK» e «Copia». Ritornano subito `QMessageBox.Ok` (i chiamanti usano il
valore solo per proseguire). `question()` resta quello di Qt (serve una risposta).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
                               QStyle, QVBoxLayout)

_open: list = []          # riferimenti: le finestre non modali non devono essere raccolte dal GC
_KIND_ICON = {"information": QStyle.SP_MessageBoxInformation, "warning": QStyle.SP_MessageBoxWarning,
              "critical": QStyle.SP_MessageBoxCritical}


def show_message(parent, title, text, kind: str = "information") -> QDialog:
    dlg = QDialog(parent if (parent is None or hasattr(parent, "isVisible")) else None)
    dlg.setWindowTitle(str(title))
    dlg.setModal(False)
    dlg.setAttribute(Qt.WA_DeleteOnClose, True)
    dlg.setWindowFlag(Qt.WindowStaysOnTopHint, False)
    lay = QVBoxLayout(dlg)
    top = QHBoxLayout()
    ic = QLabel()
    ic.setPixmap(QApplication.style().standardIcon(_KIND_ICON.get(kind, QStyle.SP_MessageBoxInformation)).pixmap(32, 32))
    ic.setAlignment(Qt.AlignTop)
    top.addWidget(ic)
    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setPlainText(str(text))
    view.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
    view.setLineWrapMode(QPlainTextEdit.WidgetWidth)
    view.setMinimumSize(420, 90)
    top.addWidget(view, 1)
    lay.addLayout(top, 1)
    row = QHBoxLayout()
    row.addStretch(1)
    from ui.i18n_manager import tr
    b_copy = QPushButton(tr("msg.copy", "Copia"))
    b_copy.clicked.connect(lambda: QApplication.clipboard().setText(str(text)))
    b_ok = QPushButton("OK")
    b_ok.setDefault(True)
    b_ok.clicked.connect(dlg.close)
    row.addWidget(b_copy)
    row.addWidget(b_ok)
    lay.addLayout(row)
    n = len(str(text))
    dlg.resize(520 if n < 300 else 720, 160 if n < 120 else (260 if n < 600 else 420))
    _open.append(dlg)
    dlg.destroyed.connect(lambda *_: _open.remove(dlg) if dlg in _open else None)
    dlg.show()
    dlg.raise_()
    return dlg


def _make(kind: str):
    def f(parent=None, title="", text="", *a, **k):
        try:
            show_message(parent, title, text, kind)
        except Exception:                                    # noqa: BLE001 — mai bloccare per un problema di UI
            pass
        return QMessageBox.Ok
    return staticmethod(f)


def install() -> None:
    """Sostituisce information/warning/critical statici di QMessageBox. Idempotente."""
    if getattr(QMessageBox, "_ls_nonmodal", False):
        return
    QMessageBox.information = _make("information")
    QMessageBox.warning = _make("warning")
    QMessageBox.critical = _make("critical")
    QMessageBox._ls_nonmodal = True
