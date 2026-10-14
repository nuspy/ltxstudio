"""Regola «mai il primo elemento al posto della scelta dell'utente» per i combo di modello/checkpoint.

Quando un menu viene ripopolato (matrice dell'edizione, modalità NSFW, file spariti dall'inventario) e il modello che
l'utente aveva scelto non c'è più, il combo NON ricade sul primo elemento: resta su un segnaposto
«— scegli un modello —» e le azioni di generazione si fermano con un messaggio chiaro (`blocked`).

Uso tipico nel codice che ripopola il combo:

    prev = mm.current_key(combo)                     # PRIMA di clear()
    combo.clear(); ...addItem(...)
    mm.settle(combo, prev, candidates)               # DOPO: riseleziona o mette il segnaposto

`candidates` = le chiavi che questo menu PUÒ mostrare (anche quelle filtrate); una scelta precedente fuori da quell'insieme
(es. una voce di un'altra architettura) non è «sparita», quindi non genera il segnaposto. `candidates=None` = «prev
conta come candidata se era una scelta reale» (menu da inventario: il file è sparito).
"""
from __future__ import annotations

PLACEHOLDER = "__placeholder__"
_SCOPE_PROP = "_model_menu_scope"


def _tr(key: str, default: str) -> str:
    try:
        from ui.i18n_manager import tr
        return tr(key, default)
    except Exception:                                         # noqa: BLE001
        return default


def placeholder_text() -> str:
    return _tr("vid.model.placeholder", "— scegli un modello —")


def is_placeholder(combo) -> bool:
    """True se il combo è sul segnaposto (data = stringa PLACEHOLDER o dict con `placeholder`)."""
    if combo is None or not combo.count():
        return False
    d = combo.currentData()
    return d == PLACEHOLDER or (isinstance(d, dict) and bool(d.get("placeholder")))


def default_key(combo, i: int):
    """Chiave di una voce: il suo data se è una stringa, altrimenti il testo senza i suffissi di stato."""
    d = combo.itemData(i)
    if isinstance(d, str) and d:
        return d
    if isinstance(d, dict) and d.get("key"):
        return d["key"]
    return combo.itemText(i).split("  ⛔")[0].split("  ⬇")[0].lstrip("⬇ ").strip()


def current_key(combo, key_of=None):
    """Chiave della selezione corrente (PLACEHOLDER se sul segnaposto, None se il combo è vuoto)."""
    if combo is None or not combo.count():
        return None
    if is_placeholder(combo):
        return PLACEHOLDER
    return (key_of or (lambda i: default_key(combo, i)))(combo.currentIndex())


def settle(combo, prev, candidates=None, *, key_of=None, data=PLACEHOLDER) -> bool:
    """Dopo il ripopolamento: riseleziona `prev` se c'è ancora; se è sparito mette il segnaposto in testa e lo
    seleziona (True). Un segnaposto già attivo resta tale finché il menu ha gli stessi candidati."""
    key_of = key_of or (lambda i: default_key(combo, i))
    if prev != PLACEHOLDER:                                   # segnaposto vecchio non piu' selezionato: via
        for i in reversed(range(combo.count())):
            d = combo.itemData(i)
            if d == PLACEHOLDER or (isinstance(d, dict) and d.get("placeholder")):
                combo.removeItem(i)
    keys = [key_of(i) for i in range(combo.count())]
    if prev is not None and prev != PLACEHOLDER and prev in keys:
        combo.setCurrentIndex(keys.index(prev))
        return False
    if candidates is None:
        real = prev is not None and prev != PLACEHOLDER and not str(prev).startswith("(") and str(prev).strip()
        cand = {prev} if real else set()
    else:
        cand = set(candidates)
    scope = "|".join(sorted(map(str, cand))) if candidates is not None else "*"
    sticky = prev == PLACEHOLDER and combo.property(_SCOPE_PROP) == scope
    if prev in cand or sticky:
        item = data if not isinstance(data, dict) else dict(data)
        combo.insertItem(0, placeholder_text(), item)
        combo.setCurrentIndex(0)
        combo.setProperty(_SCOPE_PROP, scope)
        return True
    return False


def blocked(parent, *combos) -> bool:
    """True (e avviso all'utente) se uno dei combo è sul segnaposto: l'azione di generazione deve fermarsi."""
    if not any(is_placeholder(c) for c in combos if c is not None):
        return False
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.warning(parent, _tr("mw.pick_model_title", "Scegli un modello"),
                        _tr("mw.pick_model_msg", "Il modello scelto non è più disponibile nella tua edizione: "
                            "selezionane uno dal menu Modello."))
    return True
