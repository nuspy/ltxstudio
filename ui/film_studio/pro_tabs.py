"""Varianti «Pro» dei tab figli del Film Studio — empty-state guidati + rifiniture.

Ogni tab Pro è una sottoclasse del tab Classico che aggiunge, senza toccarne la logica, un **empty-state
guidato**: un overlay centrato sul contenitore principale (tree/lista) che appare quando è vuoto e sparisce
quando si popola, indicando il passo giusto del flusso Copione→Scenografia→Montaggio→Master. ScriptTab (senza
lista) riceve un banner-guida in cima. Le V1 restano intatte.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import QLabel

from ui.i18n_manager import tr
from ui.film_studio.script_tab import ScriptTab
from ui.film_studio.storyboard_tab import StoryboardTab
from ui.film_studio.montage_tab import MontageTab
from ui.film_studio.asset_manager_tab import AssetManagerTab
from ui.film_studio.voice_tab import VoiceTab
from ui.film_studio.lora_studio_tab import LoraStudioTab


class _EmptyOverlay(QLabel):
    """Placeholder guidato sovrapposto a un contenitore (lista/tree), visibile solo quando è vuoto."""

    def __init__(self, target, text: str, count_fn):
        super().__init__(text, target)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("color:#8a90a0; background:rgba(24,24,24,225); border:1px dashed #3a3a3a; "
                           "border-radius:8px; padding:14px; font-size:12px;")
        self._target = target
        self._count = count_fn
        target.installEventFilter(self)
        self._t = QTimer(self)
        self._t.timeout.connect(self._refresh)
        self._t.start(1000)
        self._refresh()

    def eventFilter(self, obj, ev):                          # noqa: N802
        if ev.type() in (QEvent.Resize, QEvent.Show):
            self._place()
        return False

    def _place(self):
        m = 24
        w = max(120, self._target.width() - 2 * m)
        h = 90
        self.setGeometry(m, max(10, self._target.height() // 2 - h // 2), w, h)

    def _refresh(self):
        try:
            empty = self._count() == 0
        except Exception:                                    # noqa: BLE001
            empty = False
        self.setVisible(empty)
        if empty:
            self._place()
            self.raise_()


def _top_banner(owner, text: str):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color:#9aa0b4; padding:6px 10px; background:#181818; border:1px solid #2a2a2a; "
                      "border-radius:6px; font-size:12px;")
    lay = owner.layout()
    if lay is not None:
        lay.insertWidget(0, lab)
    return lab


class ScriptTabPro(ScriptTab):
    def __init__(self, get_project):
        super().__init__(get_project)
        _top_banner(self, tr("fs.pro.script.hint",
            "①  Scrivi un'idea o incolla un copione, poi «✍️ Genera». Usa @ per citare gli asset."))


class StoryboardTabPro(StoryboardTab):
    def __init__(self, get_project):
        super().__init__(get_project)
        self._empty = _EmptyOverlay(self.tree, tr("fs.pro.sb.empty",
            "②  Nessuna scena.\nSalva il copione e premi «Breakdown» per generare scene e inquadrature."),
            self.tree.topLevelItemCount)


class MontageTabPro(MontageTab):
    def __init__(self, get_project):
        super().__init__(get_project)
        self._empty = _EmptyOverlay(self.lst, tr("fs.pro.mt.empty",
            "③  Nessuna clip.\nGenera i video degli shot nella «Scenografia» per popolare la timeline."),
            self.lst.count)


class AssetManagerTabPro(AssetManagerTab):
    def __init__(self):
        super().__init__()
        self._empty = _EmptyOverlay(self.lst, tr("fs.pro.asset.empty",
            "Nessun asset.\nCrea il tuo primo attore, luogo o oggetto con «Nuovo»."), self.lst.count)


class VoiceTabPro(VoiceTab):
    def __init__(self):
        super().__init__()
        self._empty = _EmptyOverlay(self.lst, tr("fs.pro.voice.empty",
            "Nessuna voce.\nAggiungine una: da descrizione, da parametri o clonata."), self.lst.count)


class LoraStudioTabPro(LoraStudioTab):
    def __init__(self):
        super().__init__()
        self._empty = _EmptyOverlay(self.lst, tr("fs.pro.lora.empty",
            "Nessuna LoRA.\nAllenane una dai tab «LoRA» dell'app: comparirà qui."), self.lst.count)
