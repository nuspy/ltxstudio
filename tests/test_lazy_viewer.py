"""Il visore 3D (Chromium/WebGL) dei tab Trellis, Rig e Animazione nasce solo alla prima apertura del tab (crash di avvio del 04/11: access violation in addTab con piu visori creati insieme)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class FakeViewer:
    created = 0

    def __new__(cls):
        from PySide6.QtWidgets import QLabel
        FakeViewer.created += 1
        widget = QLabel("visore")
        widget.loaded = []
        widget.load_model = widget.loaded.append
        return widget


def test_viewer_is_created_only_when_the_tab_is_first_shown_and_loads_the_pending_model(qapp):
    from ui.widgets.lazy_viewer import LazyViewer
    FakeViewer.created = 0
    lazy = LazyViewer(FakeViewer)
    lazy.load_model("a.glb")                       # tab mai aperto: ricorda il modello, non crea nulla
    qapp.processEvents()
    assert FakeViewer.created == 0 and not lazy.created
    lazy.show()
    qapp.processEvents()
    qapp.processEvents()
    assert FakeViewer.created == 1 and lazy.created and lazy._viewer.loaded == ["a.glb"]
    lazy.load_model("b.glb")                       # a visore creato il modello va dritto al visore
    lazy.hide()
    lazy.show()
    qapp.processEvents()
    assert FakeViewer.created == 1 and lazy._viewer.loaded == ["a.glb", "b.glb"]


def test_the_three_tabs_use_the_lazy_viewer_and_build_no_chromium_at_startup(qapp, monkeypatch):
    from ui.tabs import trellis_tab
    monkeypatch.setattr(trellis_tab, "make_viewer", FakeViewer)
    FakeViewer.created = 0
    from ui.tabs.anim_tab import AnimTab
    from ui.tabs.rig_tab import RigTab
    from ui.tabs.trellis_tab import TrellisTab
    from ui.widgets.lazy_viewer import LazyViewer
    from ui.tabs import anim_tab, rig_tab
    monkeypatch.setattr(anim_tab, "make_viewer", FakeViewer)
    monkeypatch.setattr(rig_tab, "make_viewer", FakeViewer)
    tabs = [TrellisTab(), RigTab(), AnimTab()]
    qapp.processEvents()
    assert all(isinstance(tab.viewer, LazyViewer) and not tab.viewer.created for tab in tabs) and FakeViewer.created == 0
