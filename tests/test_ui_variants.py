"""Test varianti UI Classica/Pro (switch da menu Interfaccia) — offscreen, nessun backend."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication          # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_variants_are_subclasses():
    from ui.tabs.import_models_tab import ImportModelsTab
    from ui.tabs.import_models_tab_pro import ImportModelsTabPro
    from ui.film_studio.film_studio_window import FilmStudioWindow
    from ui.film_studio.film_studio_window_pro import FilmStudioWindowPro
    assert issubclass(ImportModelsTabPro, ImportModelsTab)
    assert issubclass(FilmStudioWindowPro, FilmStudioWindow)


def test_models_pro_builds_4_modes():
    from ui.tabs.import_models_tab_pro import ImportModelsTabPro
    t = ImportModelsTabPro()
    assert t._stack.count() == 4
    assert hasattr(t, "url_edit")                    # 2B: URL/ID vive in Scopri
    assert t._btn_thumbs.isEnabled() is False        # 5A: thumbnail lazy


def test_swap_models_hot():
    from ui.tabs.advanced_tab import AdvancedTab
    from ui.tabs.import_models_tab import ImportModelsTab
    a = AdvancedTab()
    t1 = ImportModelsTab(); a.attach_models(t1)
    t2 = ImportModelsTab(); a.swap_models(t2)
    assert a._models_host.layout().count() == 1
    assert a._models_host.layout().itemAt(0).widget() is t2


def test_log_contextual():
    from ui.tabs.import_models_tab_pro import ImportModelsTabPro
    t = ImportModelsTabPro()
    t._select_mode(0); assert t._log_host.isHidden() is True     # Scopri → log nascosto
    t._select_mode(1); assert t._log_host.isHidden() is False    # Importa → log visibile
    t._select_mode(2); assert t._log_host.isHidden() is True     # Libreria → nascosto
