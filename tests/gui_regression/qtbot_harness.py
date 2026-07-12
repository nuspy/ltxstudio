"""F2 (#94) — Helper per INTERAZIONI UTENTE REALI sui widget (stile qtbot) per la suite gui_regression.

Complementa `harness.py` (Spy / build_window / patch_dialogs): dove i casi storici impostano i widget
programmaticamente (`setValue`/`setChecked`/`setCurrentText`), questi helper simulano l'UTENTE VERO
(click del mouse, digitazione, tastiera) via `QtTest.QTest`. Funzionano:
  • STANDALONE (run_all / run_case: nessun pytest richiesto — la suite attuale non usa il runner pytest);
  • sotto pytest / pytest-qt (nessuna dipendenza dal fixture `qtbot`; il qtbot può convivere accanto
    per `waitSignal`/`waitUntil` — vedi `test_qtbot_video_poc.py`).

PATTERN (riferimento per i casi F3+):
  • click(btn)                          → QTest.mouseClick al centro. PRIMA verifica enabled+visibile:
                                          un bottone disabilitato/scollegato = AssertionError CHIARA
                                          (scopre wiring rotto, non un no-op silenzioso).
  • type_text(edit, "testo")            → click reale (focus) + Ctrl+A + Canc + QTest.keyClicks.
                                          QLineEdit / QPlainTextEdit / QTextEdit. NIENTE newline
                                          (keyClicks non li supporta) e niente '\\' nei campi con
                                          tag-palette (aprirebbe il popup dei tag).
  • spin_set(spin, v)                   → selectAll + digitazione cifre + Invio (QSpinBox/QDoubleSpinBox;
                                          gestisce suffisso — es. " s" — e separatore decimale del locale).
  • combo_select(combo, index=/text=/data=) → navigazione con le FRECCE sul combo chiuso (emette i
                                          segnali reali currentIndexChanged/activated ad ogni passo);
                                          se l'indice è irraggiungibile in tastiera (item disabilitati
                                          in mezzo) fallback `setCurrentIndex` (segnale comunque reale);
                                          `keyboard_only=True` per pretendere la sola tastiera.
  • set_checked(box, True/False)        → click reale SOLO se lo stato differisce (QCheckBox/QRadioButton),
                                          poi assert sullo stato raggiunto.
  • slider_set(slider, v)               → focus + Home poi PageUp/PageDown + frecce fino al valore esatto
                                          (ogni pressione emette valueChanged reale, come un utente).
  • list_click_item(lw, row)            → scrollToItem + click sul viewport alle coordinate della riga.
  • lora_click_row(lora_tab, "sub")     → click reale sulla CHECKBOX della riga LoRA il cui nome/file
                                          contiene 'sub' (selettore LoRA embedded, `_rows`).
  • wait_until(cond, ...)               → attesa attiva con processEvents (equivalente qtbot.waitUntil).
  • process_events(ms=0)                → drena la event-loop (i segnali sincroni sono consegnati al ritorno).

ROBUSTEZZA: ogni interazione passa da `_assert_interactable()` — widget None / nascosto / disabilitato
→ AssertionError con spiegazione (mai un finto PASS). I widget dentro QScrollArea vengono portati in
vista (best-effort, `ensureWidgetVisible`). NB: il TAB che contiene il widget deve essere quello
corrente (`w.tabs.setCurrentWidget(...)` = l'utente che naviga al tab), altrimenti niente visibilità.

Compatibile con lo Spy: la finestra resta quella di `harness.build_window()`; le azioni qui simulate
(es. click su `gen_btn`) attraversano gli handler REALI e finiscono nelle richieste catturate dallo Spy.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt                                     # noqa: E402
from PySide6.QtTest import QTest                                  # noqa: E402
from PySide6.QtWidgets import (                                   # noqa: E402
    QApplication, QComboBox, QDoubleSpinBox, QListWidget, QScrollArea,
)


# ── event loop ───────────────────────────────────────────────────────────────
def process_events(ms: int = 0):
    """Drena la coda eventi Qt (opz. attende `ms` millisecondi con la loop attiva)."""
    QApplication.processEvents()
    if ms:
        QTest.qWait(ms)
        QApplication.processEvents()


def wait_until(cond, timeout_ms: int = 3000, msg: str = "condizione non raggiunta"):
    """Attesa attiva con event-loop viva (equivalente di qtbot.waitUntil, senza pytest)."""
    t0 = time.monotonic()
    while (time.monotonic() - t0) * 1000.0 < timeout_ms:
        if cond():
            return
        QTest.qWait(20)
    raise AssertionError(f"wait_until: {msg} (timeout {timeout_ms} ms)")


# ── guardie ──────────────────────────────────────────────────────────────────
def _scroll_into_view(w):
    """Best-effort: se il widget sta dentro una QScrollArea, portalo in vista."""
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, QScrollArea):
            try:
                p.ensureWidgetVisible(w)
            except Exception:                                     # noqa: BLE001
                pass
        p = p.parentWidget()


def _name(w) -> str:
    try:
        return w.objectName() or type(w).__name__
    except Exception:                                             # noqa: BLE001
        return repr(w)


def _assert_interactable(w, what: str = "widget"):
    """Un'interazione utente su un widget None/nascosto/disabilitato deve FALLIRE chiaro,
    non degradare in un no-op che poi produce un assert criptico a valle."""
    assert w is not None, f"{what}: widget None (non costruito / attributo errato?)"
    _scroll_into_view(w)
    process_events()
    assert w.isEnabled(), (
        f"{what}: '{_name(w)}' è DISABILITATO — l'interazione utente reale è impossibile "
        f"(verifica gating/toggle/precondizioni del caso)")
    assert w.isVisibleTo(w.window()), (
        f"{what}: '{_name(w)}' NON è visibile nella finestra — porta in primo piano il tab che lo "
        f"contiene (es. w.tabs.setCurrentWidget(...)) prima dell'interazione")


# ── interazioni base ─────────────────────────────────────────────────────────
def click(w, button=Qt.LeftButton, pos=None):
    """Click reale del mouse al centro del widget (bottoni, tool-button, checkbox, radio…)."""
    _assert_interactable(w, "click")
    QTest.mouseClick(w, button, Qt.NoModifier, pos or w.rect().center())
    process_events()


def focus(w):
    """Porta il focus sul widget (come farebbe un click/tab dell'utente)."""
    _assert_interactable(w, "focus")
    w.setFocus(Qt.MouseFocusReason)
    process_events()


def type_text(w, text: str, clear: bool = True):
    """Digitazione reale in QLineEdit / QPlainTextEdit / QTextEdit: click (focus) + Ctrl+A + Canc +
    keyClicks. NB: niente newline; evita '\\' nei campi con tag-palette (aprirebbe il popup)."""
    _assert_interactable(w, "type_text")
    assert "\n" not in text, "type_text: QTest.keyClicks non supporta newline (usa setPlainText per testi multilinea)"
    # click sul viewport per i text-edit (il mouse arriva lì), sul widget per i line-edit
    target = w.viewport() if hasattr(w, "viewport") else w
    QTest.mouseClick(target, Qt.LeftButton, Qt.NoModifier, target.rect().center())
    w.setFocus(Qt.MouseFocusReason)
    process_events()
    if clear:
        QTest.keyClick(w, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(w, Qt.Key_Delete)
    if text:
        QTest.keyClicks(w, text)
    process_events()
    got = w.toPlainText() if hasattr(w, "toPlainText") else w.text()
    assert text in got, f"type_text: il testo digitato non risulta nel campo (atteso {text!r}, campo {got!r})"


def spin_set(spin, value):
    """Imposta uno spinbox DIGITANDO il valore (selectAll + cifre + Invio), come l'utente.
    Gestisce QDoubleSpinBox (decimali/locale) e suffissi (es. ' s')."""
    _assert_interactable(spin, "spin_set")
    QTest.mouseClick(spin, Qt.LeftButton, Qt.NoModifier, spin.rect().center())
    spin.setFocus(Qt.MouseFocusReason)
    process_events()
    spin.selectAll()
    if isinstance(spin, QDoubleSpinBox):
        txt = f"{float(value):.{spin.decimals()}f}".replace(".", spin.locale().decimalPoint())
        tol = 10 ** (-spin.decimals()) / 2.0
    else:
        txt, tol = str(int(value)), 0
    QTest.keyClicks(spin, txt)
    QTest.keyClick(spin, Qt.Key_Enter)          # interpreta il testo → value + editingFinished
    process_events()
    got = spin.value()
    assert abs(float(got) - float(value)) <= tol, \
        f"spin_set: digitato {txt!r} ma il valore è {got} (atteso {value}) su '{_name(spin)}'"


def combo_select(combo: QComboBox, index: int | None = None, text: str | None = None,
                 data=None, keyboard_only: bool = False):
    """Seleziona una voce del combo con la TASTIERA (frecce sul combo chiuso → segnali
    currentIndexChanged/activated reali ad ogni passo). Risoluzione voce per index / testo / itemData.
    Se la voce non è raggiungibile in tastiera (item disabilitati in mezzo) → fallback
    setCurrentIndex (segnale comunque reale) salvo keyboard_only=True. Ritorna currentData()."""
    _assert_interactable(combo, "combo_select")
    if index is None and text is not None:
        index = combo.findText(text)
        assert index >= 0, f"combo_select: voce con testo {text!r} assente in '{_name(combo)}'"
    if index is None and data is not None:
        index = combo.findData(data)
        assert index >= 0, f"combo_select: voce con data {data!r} assente in '{_name(combo)}'"
    assert index is not None and 0 <= index < combo.count(), \
        f"combo_select: indice {index} fuori range (count={combo.count()}) in '{_name(combo)}'"
    m = combo.model()
    it = m.item(index) if hasattr(m, "item") else None
    assert it is None or it.isEnabled(), \
        f"combo_select: la voce {index} ('{combo.itemText(index)}') è DISABILITATA"
    combo.setFocus(Qt.MouseFocusReason)
    process_events()
    guard = 3 * max(combo.count(), 1)
    while combo.currentIndex() != index and guard > 0:
        prev = combo.currentIndex()
        QTest.keyClick(combo, Qt.Key_Down if prev < index else Qt.Key_Up)
        process_events()
        if combo.currentIndex() == prev:        # bloccato (estremo raggiunto / item disabilitati)
            break
        guard -= 1
    if combo.currentIndex() != index:
        assert not keyboard_only, (
            f"combo_select: voce {index} non raggiungibile con la tastiera "
            f"(fermo a {combo.currentIndex()}) in '{_name(combo)}'")
        combo.setCurrentIndex(index)            # fallback: emette comunque currentIndexChanged reale
        process_events()
    assert combo.currentIndex() == index, \
        f"combo_select: selezione fallita (corrente {combo.currentIndex()}, atteso {index})"
    return combo.currentData()


def set_checked(box, checked: bool = True):
    """Porta un QCheckBox/QRadioButton allo stato voluto con un'interazione reale (solo se differisce).

    NB geometria: in un form-layout il checkbox è spesso ALLARGATO oltre il testo → il centro del
    rect cade fuori da `hitButton()` e il click non farebbe toggle. Si clicca quindi sulla zona
    dell'INDICATORE (bordo sinistro, LTR); se il click non basta (stili/RTL), fallback = Spazio
    da tastiera sul widget focalizzato (interazione utente altrettanto reale)."""
    _assert_interactable(box, "set_checked")
    if box.isChecked() == bool(checked):
        return
    from PySide6.QtCore import QPoint
    ind = QPoint(min(10, max(1, box.width() - 1)), box.height() // 2)   # zona indicatore (LTR)
    QTest.mouseClick(box, Qt.LeftButton, Qt.NoModifier, ind)
    process_events()
    if box.isChecked() != bool(checked):                # stile/RTL: il punto non era cliccabile
        box.setFocus(Qt.MouseFocusReason)
        QTest.keyClick(box, Qt.Key_Space)               # toggle da tastiera (utente reale)
        process_events()
    assert box.isChecked() == bool(checked), (
        f"set_checked: né il click sull'indicatore né Spazio hanno portato '{_name(box)}' "
        f"a {checked} (handler che lo reimposta? radio esclusivo?)")


def slider_set(slider, value: int):
    """Porta uno slider al valore esatto con la TASTIERA (Home poi PageUp/PageDown + frecce):
    ogni pressione emette valueChanged reale, come un utente."""
    _assert_interactable(slider, "slider_set")
    mn, mx = slider.minimum(), slider.maximum()
    assert mn <= value <= mx, f"slider_set: {value} fuori range [{mn},{mx}] di '{_name(slider)}'"
    slider.setFocus(Qt.MouseFocusReason)
    process_events()
    QTest.keyClick(slider, Qt.Key_Home)         # → minimo, base deterministica
    process_events()
    page = max(1, slider.pageStep())
    guard = 4 * max(1, (mx - mn))
    while slider.value() != value and guard > 0:
        delta = value - slider.value()
        if delta >= page:
            QTest.keyClick(slider, Qt.Key_PageUp)
        elif delta > 0:
            QTest.keyClick(slider, Qt.Key_Right)
        elif delta <= -page:
            QTest.keyClick(slider, Qt.Key_PageDown)
        else:
            QTest.keyClick(slider, Qt.Key_Left)
        guard -= 1
    process_events()
    assert slider.value() == value, \
        f"slider_set: atteso {value}, ottenuto {slider.value()} su '{_name(slider)}'"


def list_click_item(lw: QListWidget, row):
    """Click reale su una riga di QListWidget (scrollToItem + click sul viewport). `row` = indice o item."""
    _assert_interactable(lw, "list_click_item")
    item = lw.item(row) if isinstance(row, int) else row
    assert item is not None, f"list_click_item: riga {row!r} assente in '{_name(lw)}'"
    lw.scrollToItem(item)
    process_events()
    r = lw.visualItemRect(item)
    assert r.isValid() and not r.isEmpty(), f"list_click_item: riga {row!r} senza rettangolo visibile"
    QTest.mouseClick(lw.viewport(), Qt.LeftButton, Qt.NoModifier, r.center())
    process_events()
    return item


# ── helper specifici dei widget dell'app ─────────────────────────────────────
def lora_click_row(lora_tab, match: str, checked: bool = True):
    """Seleziona/deseleziona con un CLICK REALE la checkbox della riga LoRA il cui nome o file
    contiene `match` (case-insensitive). Ritorna il dict-riga {item, cb, sp, entry, star}."""
    rows = getattr(lora_tab, "_rows", {}) or {}
    low = match.lower()
    key = next((n for n, r in rows.items()
                if low in n.lower() or low in getattr(r.get("entry"), "comfy_name", "").lower()), None)
    assert key, (f"lora_click_row: nessuna riga LoRA contiene {match!r} "
                 f"(righe presenti: {list(rows)[:8]}{'…' if len(rows) > 8 else ''})")
    r = rows[key]
    lora_tab.list.scrollToItem(r["item"])
    process_events()
    set_checked(r["cb"], checked)
    return r
