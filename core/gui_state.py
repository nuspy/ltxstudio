"""Persistenza GENERICA dello stato della GUI (tutti i widget input di tutti i tab).

Scansiona ricorsivamente gli attributi-widget della finestra principale e dei nostri widget
compositi (quelli definiti nel package ``ui.``), salvando/ripristinando i valori dei controlli
input: QComboBox, QLineEdit, QPlainTextEdit/QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
QRadioButton, QSlider.

NESSUNA lista manuale di campi: la chiave è il *path dei nomi di attributo* (es.
``video_tab.fps``), quindi aggiungendo o togliendo un widget questo entra/esce dai settings
salvati AUTOMATICAMENTE. Non serve aggiornare nulla qui quando si aggiungono campi.

File: ``data/gui_state.json``. Scrittura atomica. Tutti gli errori sono silenziati: la
persistenza non deve MAI impedire l'avvio o la chiusura dell'app.

Note:
- I QComboBox sono salvati per *testo* (e indice come fallback): i menu popolati dall'inventario
  live di ComfyUI si riempiono dopo l'avvio, perciò ``restore`` va richiamato di nuovo quando
  l'inventario è pronto (vedi main_window).
- Gli attributi che iniziano con ``_`` (privati) sono ignorati.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "gui_state.json"


# ──────────────────────────────────────────────────────────────────────────────
# lettura/scrittura valore di un singolo widget
# ──────────────────────────────────────────────────────────────────────────────
def _json_safe(v) -> bool:
    """True se la userData è serializzabile in JSON (str/num/bool/None o dict/list di questi)."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, dict):
        return all(isinstance(k, str) and _json_safe(x) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return all(_json_safe(x) for x in v)
    return False


def _data_match(item_data, saved) -> bool:
    """Confronta la userData di un item col valore salvato. Per i dict (es. modello
    {kind,model,key}) basta che combaci la chiave identificativa ('key' o 'model'); altrimenti
    uguaglianza piena. Robusto al round-trip JSON (tuple↔list)."""
    if isinstance(item_data, dict) and isinstance(saved, dict):
        for idkey in ("key", "model", "id"):
            if idkey in item_data and idkey in saved:
                return item_data[idkey] == saved[idkey]
        return item_data == saved
    if isinstance(item_data, (list, tuple)) and isinstance(saved, (list, tuple)):
        return list(item_data) == list(saved)
    return item_data == saved


def _get(w):
    from PySide6.QtWidgets import (
        QComboBox, QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
        QCheckBox, QRadioButton, QSlider,
    )
    if isinstance(w, QComboBox):
        d = {"t": "combo", "text": w.currentText(), "idx": w.currentIndex()}
        # Salva anche l'userData (chiave STABILE) per i combo popolati dall'inventario: il testo
        # ha badge dinamici (🔞/⭐) e l'indice cambia col fav-first → da soli non bastano e
        # possono far ripristinare il modello SBAGLIATO. La userData è la chiave reale del valore.
        cd = w.currentData()
        if cd is not None and _json_safe(cd):   # None = combo senza userData → solo testo/idx
            d["data"] = cd
        return d
    if isinstance(w, QPlainTextEdit):
        return {"t": "text", "v": w.toPlainText()}
    if isinstance(w, QTextEdit):
        return {"t": "text", "v": w.toPlainText()}
    if isinstance(w, QLineEdit):
        return {"t": "line", "v": w.text()}
    if isinstance(w, QDoubleSpinBox):
        return {"t": "float", "v": w.value()}
    if isinstance(w, QSpinBox):
        return {"t": "int", "v": w.value()}
    if isinstance(w, (QCheckBox, QRadioButton)):
        return {"t": "bool", "v": w.isChecked()}
    if isinstance(w, QSlider):
        return {"t": "int", "v": w.value()}
    return None


def _set(w, data):
    try:
        t = data.get("t")
        if t == "combo":
            from PySide6.QtWidgets import QComboBox
            if not isinstance(w, QComboBox):
                return
            # Priorità: 1) match per userData (chiave stabile) 2) per testo 3) per indice.
            chosen = -1
            if data.get("data") is not None:
                saved = data["data"]
                for i in range(w.count()):
                    if _data_match(w.itemData(i), saved):
                        chosen = i
                        break
            if chosen < 0:
                chosen = w.findText(data.get("text", ""))
            if chosen >= 0:
                w.setCurrentIndex(chosen)
            else:
                idx = data.get("idx")
                if isinstance(idx, int) and 0 <= idx < w.count():
                    w.setCurrentIndex(idx)
        elif t == "text":
            w.setPlainText(data.get("v", ""))
        elif t == "line":
            w.setText(data.get("v", ""))
        elif t == "int":
            w.setValue(int(data.get("v", 0)))
        elif t == "float":
            w.setValue(float(data.get("v", 0.0)))
        elif t == "bool":
            w.setChecked(bool(data.get("v", False)))
    except Exception:
        pass


def _is_input(w) -> bool:
    from PySide6.QtWidgets import (
        QComboBox, QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
        QCheckBox, QRadioButton, QSlider,
    )
    return isinstance(w, (QComboBox, QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox,
                          QDoubleSpinBox, QCheckBox, QRadioButton, QSlider))


def _is_our_widget(w) -> bool:
    """True per i widget compositi definiti nel package ``ui.`` (tab e gruppi custom):
    in essi ricorriamo per trovare altri controlli input. I widget Qt nativi non-input
    (layout, label, progress bar, ecc.) vengono ignorati."""
    try:
        from PySide6.QtWidgets import QWidget
        return isinstance(w, QWidget) and type(w).__module__.startswith("ui.")
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# walk ricorsivo
# ──────────────────────────────────────────────────────────────────────────────
def _walk(obj, prefix, state, visited, *, save, loaded=None):
    if id(obj) in visited:
        return
    visited.add(id(obj))
    try:
        items = list(vars(obj).items())
    except TypeError:
        return
    for name, val in items:
        if name.startswith("_"):
            continue
        # I combo di PRECISIONE (gen_precision/precision/...) NON vanno persistiti: un valore restrittivo
        # salvato (es. FP8) nasconderebbe i modelli di altra precisione all'avvio senza che l'utente lo
        # sappia. Restano sempre su "Tutte" all'apertura.
        if name == "precision" or name.endswith("_precision"):
            continue
        key = f"{prefix}.{name}" if prefix else name
        # protocollo custom: widget compositi con stato non standard (es. LoraTab → LoRA selezionate
        # + strength). Hanno gui_get_state()/gui_set_state(); non ricorriamo nei loro figli.
        if hasattr(val, "gui_get_state") and hasattr(val, "gui_set_state"):
            if save:
                try:
                    s = val.gui_get_state()
                    if s is not None:
                        state[key] = {"t": "custom", "v": s}
                except Exception:
                    pass
            elif loaded and isinstance(loaded.get(key), dict) and loaded[key].get("t") == "custom":
                try:
                    val.gui_set_state(loaded[key]["v"])
                except Exception:
                    pass
            continue
        if _is_input(val):
            if save:
                g = _get(val)
                if g is not None:
                    state[key] = g
            elif loaded and key in loaded:
                _set(val, loaded[key])
            continue
        if _is_our_widget(val):
            _walk(val, key, state, visited, save=save, loaded=loaded)


# ──────────────────────────────────────────────────────────────────────────────
# API pubblica
# ──────────────────────────────────────────────────────────────────────────────
def collect(win) -> dict:
    """Snapshot dei valori di tutti i controlli input raggiungibili dalla finestra."""
    state: dict = {}
    _walk(win, "", state, set(), save=True)
    return state


SCHEMA = 2


def _migrate(d: dict) -> dict:
    """Migrazioni UNA TANTUM dello stato salvato. Schema 2 (30/10/2026, punto 13): lo slider di Modifica «Forza» (denoise =
    1 − 0,4·v) diventa «Fedeltà all'originale» (denoise = 1 − v): lo stesso risultato si ottiene con v' = 0,4·v."""
    if int(d.get("_schema", 1) or 1) < 2:
        k = "frame1_tab.strength"
        if isinstance(d.get(k), (int, float)):
            d[k] = int(round(float(d[k]) * 0.4))
        d["_schema"] = 2
    return d


def load() -> dict:
    try:
        if STATE_PATH.exists():
            d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return _migrate(d)
    except Exception:
        pass
    return {}


def store(win) -> None:
    """Salva su disco lo stato corrente della GUI (atomico)."""
    try:
        state = collect(win)
        state["_schema"] = SCHEMA
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, STATE_PATH)
    except Exception:
        pass


def restore(win, state: dict | None = None) -> dict:
    """Applica lo stato salvato ai controlli della finestra. Ritorna lo stato usato
    (così il chiamante può ri-applicarlo quando i menu dinamici si popolano)."""
    if state is None:
        state = load()
    if state:
        _walk(win, "", {}, set(), save=False, loaded=state)
    return state
