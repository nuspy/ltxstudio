"""Pannello 'Metadata musicali' della tab Musica — BPM, tonalità, metro, lingua vocale (F-A).

ACE-Step (e HeartMuLa) non hanno campi metadata strutturati: il modello consuma questi valori come
semplici TAG TESTUALI dentro il prompt di stile. Qui l'utente li imposta con controlli dedicati e il
widget li COMPONE in coda ai tag di stile (es. "…, 128 bpm, C minor, 4/4, language: it"), come fa la
GUI ufficiale ACE-Step 1.5. Ogni campo ha un valore "(auto)" che NON viene aggiunto (lascia decidere
il modello). Engine-agnostico: funziona sia con ACE-Step sia con HeartMuLa.

Principi UX (ui-ux-pro-max): progressive disclosure (collassato di default), tooltip su ogni campo,
default = auto (non intrusivo).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QFormLayout, QComboBox, QSpinBox

from ui.i18n_manager import tr

_AUTO = "(auto)"

# Tonalità comuni (maggiori/minori) — l'ordine mette prima le più usate.
_KEYS = [_AUTO,
         "C major", "G major", "D major", "A major", "E major", "F major", "Bb major", "Eb major",
         "A minor", "E minor", "B minor", "D minor", "G minor", "C minor", "F# minor", "C# minor"]
# Metri musicali (numeratore usato da ACE-Step: 2/3/4/6).
_TIMESIGS = [_AUTO, "4/4", "3/4", "6/8", "2/4"]
# Lingua vocale (codici ISO usati dal modello). "(auto)" = rilevamento automatico dalle liriche.
_LANGS = [(_AUTO, _AUTO), ("en", "English"), ("it", "Italiano"), ("es", "Español"), ("fr", "Français"),
          ("de", "Deutsch"), ("pt", "Português"), ("ja", "日本語"), ("ko", "한국어"), ("zh", "中文"),
          ("ru", "Русский"), ("ar", "العربية"), ("hi", "हिन्दी")]


class MusicMetadata(QGroupBox):
    """Metadata musicali strutturati. compose(tags)→str li accoda ai tag; get_meta()→dict per il sidecar."""

    def __init__(self):
        super().__init__(tr("music.meta.title", "🎼  Metadata musicali (BPM, tonalità, metro, lingua)"))
        self.setCheckable(True); self.setChecked(False)      # collassato di default
        self._build()

    def _build(self):
        f = QFormLayout(self); f.setLabelAlignment(Qt.AlignRight)
        self.bpm = QSpinBox(); self.bpm.setRange(0, 300); self.bpm.setValue(0)
        self.bpm.setSpecialValueText(tr("music.meta.auto", "(auto)"))   # 0 → mostra "(auto)"
        self.bpm.setSuffix(" BPM")
        self.bpm.setToolTip(tr("music.meta.bpm.tt",
            "Tempo in battiti al minuto. 0 = (auto), il modello decide. Guida: 60-80 lento, 90-120 medio, "
            "120-150 pop/dance/rock, 150-180 molto energico."))
        f.addRow(tr("music.meta.bpm", "BPM"), self.bpm)

        self.key = QComboBox(); self.key.addItems(_KEYS)
        self.key.setToolTip(tr("music.meta.key.tt",
            "Tonalità/scala (es. C major, A minor). Le tonalità minori suonano più cupe/malinconiche, "
            "le maggiori più luminose. (auto) = scelta dal modello."))
        f.addRow(tr("music.meta.key", "Tonalità"), self.key)

        self.timesig = QComboBox(); self.timesig.addItems(_TIMESIGS)
        self.timesig.setToolTip(tr("music.meta.timesig.tt",
            "Metro/tempo in chiave. 4/4 per la maggior parte di pop/rock/hip-hop/electronic, "
            "3/4 per valzer, 6/8 per ballate. (auto) = scelta dal modello."))
        f.addRow(tr("music.meta.timesig", "Metro"), self.timesig)

        self.lang = QComboBox()
        for code, label in _LANGS:
            self.lang.addItem(label, code)
        self.lang.setToolTip(tr("music.meta.lang.tt",
            "Lingua del canto. (auto) = rilevata automaticamente dalle liriche. Impostala se il rilevamento "
            "sbaglia o se le liriche mescolano più lingue."))
        f.addRow(tr("music.meta.lang", "Lingua vocale"), self.lang)

    # ── API ──────────────────────────────────────────────────────────────────
    def compose(self, tags: str) -> str:
        """Accoda i metadata impostati (non-auto) alla stringa di tag di stile, senza duplicati."""
        base = (tags or "").strip().rstrip(",")
        low = base.lower()
        extra: list[str] = []
        if self.bpm.value() > 0 and "bpm" not in low:
            extra.append(f"{self.bpm.value()} bpm")
        key = self.key.currentText()
        if key != _AUTO and key.lower() not in low:
            extra.append(key)
        ts = self.timesig.currentText()
        if ts != _AUTO and ts not in base:
            extra.append(ts)
        lang = self.lang.currentData()
        if lang and lang != _AUTO and "language:" not in low:
            extra.append(f"language: {lang}")
        if not extra:
            return base
        return (base + ", " + ", ".join(extra)).lstrip(", ").strip()

    def get_meta(self) -> dict:
        """Valori impostati per il sidecar (esclude gli auto)."""
        d: dict = {}
        if self.bpm.value() > 0:
            d["bpm"] = self.bpm.value()
        if self.key.currentText() != _AUTO:
            d["keyscale"] = self.key.currentText()
        if self.timesig.currentText() != _AUTO:
            d["timesignature"] = self.timesig.currentText()
        if self.lang.currentData() and self.lang.currentData() != _AUTO:
            d["vocal_language"] = self.lang.currentData()
        return d

    def reset_defaults(self):
        """Reimposta tutti i metadata su (auto)."""
        self.bpm.setValue(0)
        self.key.setCurrentIndex(0); self.timesig.setCurrentIndex(0); self.lang.setCurrentIndex(0)
        self.setChecked(False)
