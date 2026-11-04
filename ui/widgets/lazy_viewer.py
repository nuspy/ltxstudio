"""Contenitore che crea il visore 3D (WebGL/Chromium) solo alla PRIMA apertura del tab.

Perche: i tab Trellis, Rig e Animazione avevano ognuno un visore Chromium creato all'avvio dell'app. All'avvio del 04/11 (turno 43) l'app e crollata con un access violation
dentro `tabs.addTab(anim_tab)`: i processi GPU di piu visori Chromium nati insieme, appena dopo un'altra app che aveva liberato la GPU. Il visore serve solo a chi apre il tab.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget


class LazyViewer(QWidget):
    def __init__(self, make_viewer: Callable[[], QWidget], parent: QWidget | None = None):
        super().__init__(parent)
        self._make_viewer = make_viewer
        self._viewer: QWidget | None = None
        self._pending_model = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = layout

    @property
    def created(self) -> bool:
        return self._viewer is not None

    def load_model(self, glb_path: str) -> None:
        """Se il visore non esiste ancora (tab mai aperto) ricorda il modello e lo carica alla prima apertura."""
        if self._viewer is not None:
            self._viewer.load_model(glb_path)
            return
        self._pending_model = glb_path
        if self.isVisible():
            self._create_viewer()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._viewer is None:
            QTimer.singleShot(0, self._create_viewer)         # dopo che il tab e comparso, non dentro l'evento di apertura

    def _create_viewer(self) -> None:
        if self._viewer is not None or not self.isVisible():
            return
        self._viewer = self._make_viewer()
        self._layout.addWidget(self._viewer)
        if self._pending_model:
            self._viewer.load_model(self._pending_model)
            self._pending_model = ""
