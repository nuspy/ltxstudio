r"""Palette popup di tag SSML/markup riusabile su qualsiasi QPlainTextEdit / QTextEdit.

Funzionamento:
- L'utente preme ``\`` (configurabile) → si apre un popup floating sopra il cursore
  con l'elenco di tutti i tag disponibili (display + descrizione).
- Scrivendo, l'elenco filtra progressivamente (match substring case-insensitive
  su label/keywords/descrizione).
- Frecce / mouse per scegliere; Enter o doppio click per inserire alla posizione
  cursore; Esc per annullare.
- Tag con il placeholder ``…`` avvolgono la selezione corrente se presente.

L'elenco dei tag è SHARED globale (TAGS_SSML), così la stessa palette funziona
sia nel TTS che nel prompt video — coerenza di markup garantita.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QEvent, QStringListModel
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (
    QListWidget, QListWidgetItem, QLineEdit, QVBoxLayout, QWidget,
    QFrame, QLabel, QSizePolicy,
)

from ui.i18n_manager import tr


# ────────────────────────────────────────────────────────────────────────────
# Elenco MASTER dei tag SSML/markup — ampio, organizzato per categoria.
# Formato: (label, insert_template, keywords, description, category)
#   - `insert_template` con `…` (carattere unicode HORIZONTAL ELLIPSIS U+2026)
#     viene rimpiazzato con la selezione corrente (o lasciato vuoto se nessuna).
# Categorie: pause | rate | volume | pitch | emotion | style | voice | misc
# ────────────────────────────────────────────────────────────────────────────
TAGS_SSML: list[tuple[str, str, list[str], str, str]] = [
    # ─── pause ───
    ("⏸ pausa breve",   "[pause:300ms]",  ["pause", "stop", "break"], "Pausa breve di 300ms", "pause"),
    ("⏸ pausa media",   "[pause:600ms]",  ["pause", "stop", "break"], "Pausa media di 600ms", "pause"),
    ("⏸ pausa lunga",   "[pause:1200ms]", ["pause", "stop", "break"], "Pausa lunga di 1.2s",  "pause"),
    ("⏸ pausa custom",  "[pause:Xms]",    ["pause", "custom"],        "Pausa di X millisecondi (sostituisci X)", "pause"),
    # ─── velocità (rate) ───
    ("🐢 molto lento",  '<prosody rate="x-slow">…</prosody>',  ["slow", "rate", "lento"],   "Molto lento", "rate"),
    ("🐢 lento",        '<prosody rate="slow">…</prosody>',    ["slow", "rate", "lento"],   "Lento",       "rate"),
    ("🚶 normale",      '<prosody rate="medium">…</prosody>',  ["medium", "rate"],          "Velocità normale", "rate"),
    ("🐇 veloce",       '<prosody rate="fast">…</prosody>',    ["fast", "rate", "veloce"],  "Veloce", "rate"),
    ("🐇 molto veloce", '<prosody rate="x-fast">…</prosody>',  ["fast", "rate"],            "Molto veloce", "rate"),
    ("📏 rate %",       '<prosody rate="+25%">…</prosody>',    ["rate", "percent"],         "Velocità custom (+25%, -10%, ...)", "rate"),
    # ─── volume ───
    ("🔈 sussurro",     '<prosody volume="x-soft">…</prosody>', ["whisper", "volume"], "Volume molto basso", "volume"),
    ("🔉 piano",        '<prosody volume="soft">…</prosody>',   ["soft", "volume"],    "Volume basso", "volume"),
    ("🔊 normale",      '<prosody volume="medium">…</prosody>', ["medium", "volume"],  "Volume normale", "volume"),
    ("🔊 forte",        '<prosody volume="loud">…</prosody>',   ["loud", "volume"],    "Volume alto", "volume"),
    ("📢 fortissimo",   '<prosody volume="x-loud">…</prosody>', ["loud", "volume"],    "Volume molto alto", "volume"),
    ("📏 volume %",     '<prosody volume="+30%">…</prosody>',   ["volume", "percent"], "Volume custom (+30%, -50%, ...)", "volume"),
    # ─── tono (pitch) ───
    ("🎵 grave",        '<prosody pitch="x-low">…</prosody>',   ["low", "pitch"],   "Tono molto grave", "pitch"),
    ("🎵 basso",        '<prosody pitch="low">…</prosody>',     ["low", "pitch"],   "Tono basso", "pitch"),
    ("🎵 normale",      '<prosody pitch="medium">…</prosody>',  ["medium", "pitch"], "Tono normale", "pitch"),
    ("🎵 alto",         '<prosody pitch="high">…</prosody>',    ["high", "pitch"],  "Tono alto", "pitch"),
    ("🎵 acuto",        '<prosody pitch="x-high">…</prosody>',  ["high", "pitch"],  "Tono molto acuto", "pitch"),
    ("📏 pitch Hz",     '<prosody pitch="+50Hz">…</prosody>',   ["pitch", "hz"],    "Pitch custom (+50Hz, -30Hz, ...)", "pitch"),
    # ─── emozioni ───
    ("😊 felice",       '<emotion type="happy">…</emotion>',    ["happy", "joy", "felice", "allegro"], "Tono felice", "emotion"),
    ("😢 triste",       '<emotion type="sad">…</emotion>',      ["sad", "triste"], "Tono triste", "emotion"),
    ("😡 arrabbiato",   '<emotion type="angry">…</emotion>',    ["angry", "arrabbiato", "rabbia"], "Tono arrabbiato", "emotion"),
    ("😨 spaventato",   '<emotion type="fear">…</emotion>',     ["fear", "scared", "paura"], "Spaventato", "emotion"),
    ("😍 innamorato",   '<emotion type="love">…</emotion>',     ["love", "amore"], "Tenero / innamorato", "emotion"),
    ("😴 stanco",       '<emotion type="tired">…</emotion>',    ["tired", "stanco"], "Stanco / assonnato", "emotion"),
    ("🤫 sussurrato",   '<emotion type="whisper">…</emotion>',  ["whisper", "soft"], "Sussurrato", "emotion"),
    ("😌 calmo",        '<emotion type="calm">…</emotion>',     ["calm", "calmo"], "Calmo / rilassato", "emotion"),
    ("😏 sarcastico",   '<emotion type="sarcastic">…</emotion>',["sarcastic", "ironia"], "Sarcastico / ironico", "emotion"),
    ("🤩 eccitato",     '<emotion type="excited">…</emotion>',  ["excited", "eccitato"], "Eccitato / entusiasta", "emotion"),
    ("😱 sorpreso",     '<emotion type="surprised">…</emotion>',["surprised", "sorpreso"], "Sorpreso", "emotion"),
    ("😠 disgustato",   '<emotion type="disgust">…</emotion>',  ["disgust", "disgusto"], "Disgustato", "emotion"),
    ("😈 minaccioso",   '<emotion type="threatening">…</emotion>',["threat", "minaccia"], "Minaccioso", "emotion"),
    ("🥰 affettuoso",   '<emotion type="affectionate">…</emotion>',["affection", "tenero"], "Affettuoso", "emotion"),
    # ─── stile vocale ───
    ("📣 grida",        '<voice-style name="shout">…</voice-style>',     ["shout", "grida"], "Urla / grida", "style"),
    ("📰 narrazione",   '<voice-style name="narration">…</voice-style>', ["narration", "narratore"], "Tono narrativo", "style"),
    ("📺 cinegiornale", '<voice-style name="newscast">…</voice-style>',  ["newscast", "news"], "Cinegiornale", "style"),
    ("🎭 teatrale",     '<voice-style name="theatrical">…</voice-style>',["theatrical", "drammatico"], "Teatrale", "style"),
    ("👶 bambino",      '<voice-style name="child">…</voice-style>',     ["child", "bambino"], "Voce da bambino", "style"),
    ("👴 anziano",      '<voice-style name="elderly">…</voice-style>',   ["elderly", "anziano"], "Voce anziana", "style"),
    ("📚 documentario", '<voice-style name="documentary">…</voice-style>',["documentary"], "Documentario", "style"),
    # ─── controllo voce ───
    ("🗣 enfasi forte", '<emphasis level="strong">…</emphasis>',   ["emphasis", "enfasi"], "Enfasi forte", "voice"),
    ("🗣 enfasi media", '<emphasis level="moderate">…</emphasis>', ["emphasis", "enfasi"], "Enfasi moderata", "voice"),
    ("🗣 enfasi ridotta",'<emphasis level="reduced">…</emphasis>', ["emphasis"], "Enfasi ridotta", "voice"),
    ("🔡 letterale",    '<say-as interpret-as="characters">…</say-as>', ["spell", "literal"], "Pronuncia lettera per lettera", "voice"),
    ("🔢 numero",       '<say-as interpret-as="cardinal">…</say-as>', ["number"], "Pronuncia come numero", "voice"),
    ("📅 data",         '<say-as interpret-as="date">…</say-as>', ["date"], "Pronuncia come data", "voice"),
    ("⏰ ora",          '<say-as interpret-as="time">…</say-as>', ["time"], "Pronuncia come ora", "voice"),
    ("💰 valuta",       '<say-as interpret-as="currency">…</say-as>', ["currency", "money"], "Pronuncia come valuta", "voice"),
    # ─── pronuncia / sostituzioni ───
    ("🔤 fonema",       '<phoneme alphabet="ipa" ph="…">testo</phoneme>', ["phoneme", "ipa"], "Pronuncia IPA esplicita", "voice"),
    ("📝 sostituisci",  '<sub alias="testo da leggere">testo originale</sub>', ["sub", "alias"], "Leggi qualcos'altro al posto del testo", "voice"),
    # ─── misc ───
    ("🎙 lingua",       '<lang xml:lang="en-US">…</lang>', ["lang", "language"], "Cambia lingua per un tratto", "misc"),
    ("➡ break",         '<break time="500ms"/>', ["break", "pause"], "Break SSML W3C standard", "pause"),
]


# ────────────────────────────────────────────────────────────────────────────
# Widget popup
# ────────────────────────────────────────────────────────────────────────────

class TagPalettePopup(QFrame):
    """Popup floating con QLineEdit (filtro) + QListWidget (tag matchanti)."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "TagPalettePopup { background:#1d1f23; border:1px solid #555; border-radius:6px; }"
            "QLabel { color:#c8d7e3; padding:2px 6px; font-size:11px; }"
            "QLineEdit { background:#262932; color:#eee; border:1px solid #444; padding:4px; border-radius:3px; }"
            "QListWidget { background:transparent; color:#ddd; border:none; outline:0; }"
            "QListWidget::item { padding:4px 6px; border-radius:3px; }"
            "QListWidget::item:selected { background:#3a4452; color:#fff; }"
        )
        self.setFixedWidth(440)

        v = QVBoxLayout(self); v.setContentsMargins(6, 6, 6, 6); v.setSpacing(4)
        self.title = QLabel(tr("tagpal.title"))
        v.addWidget(self.title)
        self.filter_edit = QLineEdit(); self.filter_edit.setPlaceholderText(tr("tagpal.search_placeholder"))
        self.filter_edit.textChanged.connect(self._refresh)
        v.addWidget(self.filter_edit)
        self.list = QListWidget()
        self.list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list.itemActivated.connect(self._on_pick)        # Enter
        self.list.itemDoubleClicked.connect(self._on_pick)    # double-click
        v.addWidget(self.list)
        hint = QLabel(tr("tagpal.hint"))
        hint.setStyleSheet("color:#777; font-size:10px; padding:4px 2px;")
        v.addWidget(hint)
        self.setFixedHeight(380)

        self._populate(TAGS_SSML)
        self.filter_edit.installEventFilter(self)

    def _populate(self, items: list[tuple[str, str, list[str], str, str]]):
        self.list.clear()
        for label, ins, kws, desc, cat in items:
            it = QListWidgetItem(f"{label}    —    {desc}")
            it.setData(Qt.UserRole, (ins, label))
            self.list.addItem(it)
        if self.list.count() > 0:
            self.list.setCurrentRow(0)

    def _refresh(self, q: str):
        q = (q or "").strip().lower()
        if not q:
            self._populate(TAGS_SSML); return
        out = []
        for entry in TAGS_SSML:
            label, ins, kws, desc, cat = entry
            blob = " ".join([label, desc, cat] + kws).lower()
            if all(tok in blob for tok in q.split()):
                out.append(entry)
        self._populate(out)

    def eventFilter(self, obj, ev):
        if obj is self.filter_edit and ev.type() == QEvent.KeyPress:
            k = ev.key()
            if k in (Qt.Key_Down, Qt.Key_Up, Qt.Key_PageDown, Qt.Key_PageUp):
                # delega frecce al QListWidget
                self.list.keyPressEvent(ev); return True
            if k in (Qt.Key_Return, Qt.Key_Enter):
                cur = self.list.currentItem()
                if cur: self._on_pick(cur)
                return True
            if k == Qt.Key_Escape:
                self.close(); return True
        return super().eventFilter(obj, ev)

    def _on_pick(self, item: QListWidgetItem):
        ins, _label = item.data(Qt.UserRole)
        self._insert_into_editor(ins)
        self.close()

    def _insert_into_editor(self, template: str):
        """Inserisce il template nell'editor. Se contiene `…`, lo sostituisce con la
        selezione corrente (o stringa vuota); poi posiziona il cursore al placeholder."""
        ed = self.editor
        cur: QTextCursor = ed.textCursor()
        sel = cur.selectedText()
        if "…" in template:
            replacement = template.replace("…", sel if sel else "")
        else:
            replacement = template
        # Rimuovi il trigger `\` davanti al cursore se presente (è stato digitato per aprire la palette)
        # NB: il chiamante (install_tag_palette) lo gestisce già; qui inseriamo soltanto.
        cur.insertText(replacement)
        ed.setFocus()

    def open_at_cursor(self):
        """Apre il popup posizionato vicino al cursore di testo dell'editor."""
        ed = self.editor
        rect = ed.cursorRect(ed.textCursor())
        global_pos = ed.viewport().mapToGlobal(rect.bottomRight())
        # se il popup andrebbe sotto lo schermo, mettilo SOPRA il cursore
        screen = self.screen() or ed.screen()
        avail = screen.availableGeometry()
        x = min(global_pos.x(), avail.right() - self.width() - 8)
        y = global_pos.y() + 4
        if y + self.height() > avail.bottom():
            y = avail.bottom() - self.height() - 8
        self.move(x, y)
        self.filter_edit.clear()
        self._populate(TAGS_SSML)
        self.show()
        self.filter_edit.setFocus()


# ────────────────────────────────────────────────────────────────────────────
# Installer riusabile
# ────────────────────────────────────────────────────────────────────────────

class _TriggerFilter(__import__("PySide6.QtCore", fromlist=["QObject"]).QObject):
    """EventFilter Qt che apre la palette quando l'utente digita `trigger_char`."""
    def __init__(self, popup: "TagPalettePopup", trigger_char: str = "\\"):
        super().__init__(popup.editor)
        self._popup = popup
        self._trigger = trigger_char

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress and isinstance(ev, QKeyEvent):
            if ev.text() == self._trigger:
                self._popup.open_at_cursor()
                return True            # NON inserire il trigger nel testo
        return False


def install_tag_palette(editor, trigger_char: str = "\\") -> TagPalettePopup:
    """Installa il filtro evento sull'editor: digitando `trigger_char` apre la palette.
    Ritorna l'istanza del popup (riusabile)."""
    popup = TagPalettePopup(editor)
    editor._tag_palette_popup = popup        # antigarbage-collection
    flt = _TriggerFilter(popup, trigger_char)
    editor._tag_palette_filter = flt          # antigarbage-collection
    editor.installEventFilter(flt)
    return popup
