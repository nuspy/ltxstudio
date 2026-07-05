"""Upsell delle feature bloccate — dati dal manifest bakeato (feature_matrix).

Il codegen del builder aggiunge, a ogni feature DISABILITATA nell'edizione, il blocco
`upsell = {tier, tier_name, price, url}` calcolato come tier MINIMO (per rank) in cui la
feature è abilitata. Qui lo esponiamo alla GUI: invece di nascondere, si mostra
"Available on <tier>" + link Patreon. In DEV (manifest assente) tutto è abilitato →
nessun upsell, GUI invariata (serve per testare).
"""
from __future__ import annotations

from core import features


def info(fid: str) -> dict | None:
    """{tier, tier_name, price, url} se la feature è bloccata e sbloccabile in un tier superiore."""
    feat = features.manifest().get("features", {}).get(fid)
    if not feat or feat.get("enabled", True):
        return None
    up = feat.get("upsell")
    return dict(up) if up else None


def open_url(fid: str) -> bool:
    """Apre il link Patreon del tier minimo per la feature. True se aperto."""
    up = info(fid)
    if not up or not up.get("url"):
        return False
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        return bool(QDesktopServices.openUrl(QUrl(up["url"])))
    except Exception:                                        # noqa: BLE001
        import webbrowser
        return webbrowser.open(up["url"])
