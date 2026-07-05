"""Applica il feature-gating dell'edizione alla GUI (self-contained, runtime).

Legge il manifest bakeato via `core.features` e:
- rimuove dal QTabWidget principale i TAB disabilitati (locator kind=tab);
- nasconde i widget disabilitati (locator kind=widget, dotted-path);
- mostra una volta l'age-gate 18+ se l'edizione consente contenuti `adult`.
Le capability `capability|api|asset` NON hanno effetto UI qui: le consuma il
backend (`features.is_enabled`/`assert_enabled`) e l'installer (assets).
"""
from __future__ import annotations

import json
from pathlib import Path

from core import features

_AGE_FILE = Path(__file__).resolve().parent.parent / "data" / "age_ack.json"


def _resolve_path(window, dotted: str):
    """Risolve 'frame1_tab.gen_box' come getattr-walk a partire da window."""
    obj = window
    for part in dotted.split("."):
        if not part:
            continue
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def apply(window) -> dict:
    """Applica il gating. Ritorna un piccolo report {removed_tabs, upsell_tabs, hidden}.

    UPSELL (marketing): un tab bloccato ma sbloccabile in un tier superiore NON viene rimosso —
    resta visibile col 🔒 e il suo contenuto è sostituito da un UpsellPanel ("Available on <tier>"
    + link Patreon). Viene rimosso solo se non è sbloccabile in nessun tier (upsell assente)."""
    feats = features.manifest().get("features", {})
    tabs = getattr(window, "tabs", None)
    report = {"edition": features.edition(), "removed_tabs": [], "upsell_tabs": [], "hidden": []}

    for fid, info in feats.items():
        if info.get("enabled", True):
            continue
        loc = info.get("locator", {}) or {}
        kind = loc.get("kind")
        if kind == "tab" and tabs is not None:
            w = getattr(window, loc.get("attr", ""), None)
            if w is not None:
                idx = tabs.indexOf(w)
                if idx >= 0:
                    title = tabs.tabText(idx)
                    tabs.removeTab(idx)
                    if info.get("upsell"):                   # sbloccabile → vetrina al posto del contenuto
                        try:
                            from ui.widgets.upsell import UpsellPanel
                            tabs.insertTab(idx, UpsellPanel(fid, info.get("i18n_key", "")),
                                           "🔒 " + title)
                            report["upsell_tabs"].append(fid)
                        except Exception:
                            report["removed_tabs"].append(fid)
                    else:                                    # non esiste in nessun tier → sparisce
                        report["removed_tabs"].append(fid)
        elif kind == "widget":
            w = _resolve_path(window, loc.get("path", ""))
            if w is not None:
                try:
                    w.setVisible(False)
                    report["hidden"].append(fid)
                except Exception:
                    pass

    # Age-gate 18+ (una volta, persistito) se l'edizione consente contenuti adult.
    if features.allows_adult() and _any_adult_enabled(feats):
        _age_gate(window)
    return report


def _any_adult_enabled(feats: dict) -> bool:
    return any(i.get("enabled") and i.get("sensitivity") == "adult" for i in feats.values())


def _age_gate(window) -> None:
    try:
        if _AGE_FILE.exists() and json.loads(_AGE_FILE.read_text("utf-8")).get("ok"):
            return
    except Exception:
        pass
    try:
        from PySide6.QtWidgets import QMessageBox
        from ui.i18n_manager import tr
        ans = QMessageBox.question(
            window, tr("agegate.title"), tr("agegate.msg"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        ok = ans == QMessageBox.Yes
        _AGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _AGE_FILE.write_text(json.dumps({"ok": ok}), "utf-8")
    except Exception:
        pass
