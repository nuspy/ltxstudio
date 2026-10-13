"""Combo dei DiT Qwen-Image 2.1 per Cinema Studio e Lucky Director (stessi nomi del tab Immagini).

La scelta è sempre esplicita e salvata nel progetto: nessun DiT viene scelto d'ufficio. Le voci 18+ compaiono solo
se l'edizione le consente; le voci non ancora scaricate hanno il badge ⬇ (il download parte al primo uso).
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from ui.i18n_manager import tr


def make_qwen21_dit_combo() -> QComboBox:
    combo = QComboBox()
    combo.setMinimumWidth(180)
    combo.setToolTip(tr("fs.sb.qwen21_dit.tt", "Modello Qwen-Image 2.1 per i fotogrammi chiave e i ritratti del cast."))
    try:
        from core import weights_catalog as wc
        from ui.tabs.frame1_tab import QWEN21_PROFILES
        adult = True
        try:
            from core import nsfw_policy
            adult = nsfw_policy.adult_allowed()
        except Exception:                                    # noqa: BLE001
            pass
        entries = {e["id"]: e for e in wc.load()}
        for cid, key, fb in QWEN21_PROFILES:
            e = entries.get(cid)
            if not e or (e.get("sensitivity") == "adult" and not adult):
                continue
            ready = wc.ready_with_deps(e)
            combo.addItem(("" if ready else "⬇ ") + tr(key, fb), cid)
    except Exception:                                        # noqa: BLE001 — combo vuoto: il servizio lo segnala
        pass
    return combo
