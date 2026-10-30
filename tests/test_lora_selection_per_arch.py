"""Punti 3/9 dell'utente (30/10): le LoRA spuntate con un'arch non sono attive con un'altra arch."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_lora_selection_is_scoped_to_arch():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from ui.tabs.lora_tab import LoraTab
    t = LoraTab(allow_nsfw_getter=lambda: True, embedded=True, arch="Krea2")
    rows = list(t._rows.values())
    assert rows, "nessuna LoRA Krea2 nel catalogo di test"
    rows[0]["cb"].setChecked(True)
    assert len(t.selected_loras()) == 1
    t.set_arch("Flux")
    assert t.selected_loras() == [], "la LoRA Krea2 resta attiva con FLUX"
    t.set_arch("Krea2")
    assert len(t.selected_loras()) == 1, "tornando a Krea2 la selezione deve ricomparire"
    app.processEvents()
