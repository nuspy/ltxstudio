"""Palette grafica di tag di stile musicale per HeartMuLa e ACE-Step (#punto 3).

Nessuno dei due modelli ha un vocabolario FISSO: entrambi accettano tag liberi separati da virgola
(genere/mood/strumenti/…), in inglese. Questa palette offre una selezione curata a CHIP cliccabili,
raggruppati per categoria, con anteprima live. I chip toggano il tag nel campo; i tag scritti a mano
che non sono nella palette vengono preservati. UX moderna: chip arrotondati che vanno a capo (FlowLayout).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QToolButton,
    QScrollArea, QWidget, QDialogButtonBox, QLineEdit,
)

from ui.i18n_manager import tr
from ui.widgets.flow_layout import FlowLayout

# Categorie → tag (in INGLESE, formato atteso dai modelli). Curati da esempi ACE-Step + convenzioni comuni.
# La chiave i18n titola la categoria; i tag NON si traducono (sono input del modello).
TAG_CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("music.tagcat.genre", "🎼 Genere", [
        "pop", "rock", "electronic", "hip hop", "rap", "jazz", "classical", "orchestral", "cinematic",
        "ambient", "techno", "house", "electro house", "trance", "synthwave", "lo-fi", "funk", "soul",
        "R&B", "metal", "punk", "folk", "country", "blues", "reggae", "disco", "EDM", "trap",
        "drum and bass", "dubstep", "gospel", "world", "latin", "afrobeat", "k-pop"]),
    ("music.tagcat.mood", "🎭 Mood", [
        "happy", "sad", "energetic", "calm", "dark", "uplifting", "melancholic", "epic", "dreamy",
        "aggressive", "romantic", "nostalgic", "mysterious", "playful", "tense", "euphoric",
        "atmospheric", "hopeful", "haunting", "chill", "intense", "sensual"]),
    ("music.tagcat.instrument", "🎹 Strumenti", [
        "piano", "acoustic guitar", "electric guitar", "bass", "drums", "synthesizer", "strings",
        "violin", "cello", "saxophone", "trumpet", "brass", "flute", "organ", "harp", "percussion",
        "808", "pads", "arpeggio", "sub bass", "electronic drums"]),
    ("music.tagcat.vocal", "🎤 Voce", [
        "male vocals", "female vocals", "choir", "harmonies", "spoken word", "whisper", "falsetto",
        "instrumental", "duet", "vocoder"]),
    ("music.tagcat.tempo", "🥁 Ritmo", [
        "slow", "mid-tempo", "fast", "upbeat", "driving", "groovy", "danceable", "syncopated",
        "steady beat", "half-time", "four on the floor"]),
    ("music.tagcat.production", "🎚️ Produzione", [
        "lo-fi", "hi-fi", "vintage", "modern", "warm", "punchy", "reverb-heavy", "distorted",
        "clean", "polished", "raw", "wide stereo", "analog", "tape saturation", "sidechain"]),
]


class _Chip(QToolButton):
    def __init__(self, text: str):
        super().__init__()
        self.setText(text); self.setCheckable(True); self.setCursor(Qt.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setStyleSheet(
            "QToolButton{border:1px solid #3a3f4b;border-radius:12px;padding:4px 12px;"
            "background:#20242e;color:#cdd3df;}"
            "QToolButton:hover{border-color:#6aa;background:#262b37;}"
            "QToolButton:checked{background:#3a6ea5;color:white;border-color:#3a6ea5;font-weight:600;}")


class MusicTagDialog(QDialog):
    """Selettore grafico di tag di stile: pre-spunta i tag già presenti, ritorna la stringa aggiornata."""

    def __init__(self, current: str = "", engine: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("music.tagpicker.title", "🎨 Palette stili musicali") +
                            (f" — {engine}" if engine else ""))
        self.setMinimumSize(680, 560)
        self._chips: dict[str, _Chip] = {}
        # tag correnti (lowercase, senza spazi ai bordi) — quelli NON in palette vanno preservati
        self._present = [t.strip() for t in current.split(",") if t.strip()]
        self._palette_tags = {t.lower() for _, _, tags in TAG_CATEGORIES for t in tags}
        self._extra = [t for t in self._present if t.lower() not in self._palette_tags]
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        hint = QLabel(tr("music.tagpicker.hint",
                         "Clicca i tag per aggiungerli/rimuoverli. I tag scritti a mano restano invariati. "
                         "Suggerimento: metti il GENERE per primo — su HeartMuLa ha peso maggiore."))
        hint.setWordWrap(True); hint.setStyleSheet("color:#8a93a6; padding:2px 2px 6px;")
        v.addWidget(hint)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget(); iv = QVBoxLayout(inner); iv.setSpacing(10)
        present_low = {t.lower() for t in self._present}
        for key, default, tags in TAG_CATEGORIES:
            grp = QGroupBox(tr(key, default))
            fl = FlowLayout(grp, margin=6, spacing=6)
            for t in tags:
                chip = _Chip(t)
                chip.setChecked(t.lower() in present_low)
                chip.toggled.connect(self._update_preview)
                self._chips[t] = chip
                fl.addWidget(chip)
            iv.addWidget(grp)
        iv.addStretch()
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)

        # anteprima live della stringa risultante
        pv = QHBoxLayout()
        pv.addWidget(QLabel(tr("music.tagpicker.preview", "Risultato:")))
        self.preview = QLineEdit(); self.preview.setReadOnly(True)
        pv.addWidget(self.preview, 1)
        v.addLayout(pv)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("music.tagpicker.apply", "Applica"))
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self._update_preview()

    def _selected(self) -> list[str]:
        chosen = [t for t, c in self._chips.items() if c.isChecked()]
        return self._extra + chosen                          # preserva i tag manuali non-palette

    def _update_preview(self):
        self.preview.setText(", ".join(self._selected()))

    def result_tags(self) -> str:
        return ", ".join(self._selected())
