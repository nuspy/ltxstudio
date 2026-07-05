"""Pannello 'Operazione' ACE-Step — task di editing oltre la generazione (#98 F3).

Task supportati dal pipeline: text2music (default), retake (varianti), repaint (rigenera una sezione),
edit (cambia tag/liriche tenendo la musica), extend (allunga), audio2audio (style-transfer da audio).
Solo ACE-Step. Progressive disclosure: i parametri appaiono in base al task scelto.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox, QLineEdit, QWidget, QLabel,
)

from ui.i18n_manager import tr
from ui.widgets.drop_widgets import DropLineEdit

# (chiave task, etichetta, tooltip)
TASKS = [
    ("text2music", tr("music.task.gen", "Genera (testo→musica)"),
     tr("music.task.gen.tt", "Generazione normale da tag + liriche.")),
    ("retake", tr("music.task.retake", "Retake (varianti)"),
     tr("music.task.retake.tt", "Rigenera lo stesso brano con variazioni. La 'varianza' controlla quanto cambia.")),
    ("repaint", tr("music.task.repaint", "Repaint (rigenera sezione)"),
     tr("music.task.repaint.tt", "Rigenera SOLO un intervallo temporale dell'audio sorgente, preservando il resto.")),
    ("edit", tr("music.task.edit", "Modifica (tag/testo)"),
     tr("music.task.edit.tt", "Cambia i tag di stile o le liriche di un brano esistente PRESERVANDO melodia e voce.")),
    ("extend", tr("music.task.extend", "Estendi (allunga)"),
     tr("music.task.extend.tt", "Aggiunge musica prima/dopo l'audio sorgente.")),
    ("audio2audio", tr("music.task.a2a", "Audio→Audio (stile da audio)"),
     tr("music.task.a2a.tt", "Genera guidato da un audio di RIFERIMENTO (style-transfer). La 'forza' regola quanto seguirlo.")),
]

# Modalità della pipeline ACE-Step 1.5 non ancora disponibili (richiedono la fondazione 1.5 — download).
# Mostrate BLOCCATE (🔒) per esporre la roadmap senza rompere il motore v1 attuale.
_LOCKED_15 = [
    ("extract", tr("music.task.extract", "Extract (isola traccia)"),
     tr("music.task.extract.tt", "Isola una traccia (voce/batteria/basso…) da un mix. Richiede la fondazione ACE-Step 1.5.")),
    ("lego", tr("music.task.lego", "Lego (aggiungi strumento)"),
     tr("music.task.lego.tt", "Sovrappone una singola traccia strumentale a un audio esistente. Richiede ACE-Step 1.5.")),
    ("complete", tr("music.task.complete", "Complete (completa arrangiamento)"),
     tr("music.task.complete.tt", "Riempie gli strumenti mancanti di una traccia parziale. Richiede ACE-Step 1.5.")),
    ("cover", tr("music.task.cover", "Cover (nuovo stile)"),
     tr("music.task.cover.tt", "Ricanta una canzone in un nuovo stile. Richiede ACE-Step 1.5.")),
    ("remix", tr("music.task.remix", "Remix (da audio-guida)"),
     tr("music.task.remix.tt", "Rigenera struttura/stile da un audio-guida. Richiede ACE-Step 1.5.")),
]


class MusicTaskPanel(QGroupBox):
    """Selettore task + parametri condizionali + audio sorgente drag&drop. get_task()→dict per make_job(adv)."""

    def __init__(self):
        super().__init__(tr("music.task.title", "🎛️  Operazione (ACE-Step)"))
        self.setCheckable(True); self.setChecked(False)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        self._content = QWidget(); v = QVBoxLayout(self._content); v.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._content)

        f = QFormLayout(); f.setLabelAlignment(Qt.AlignRight)
        self.task = QComboBox()
        for key, label, tt in TASKS:
            self.task.addItem(label, key)
        for key, label, tt in _LOCKED_15:                    # modalità 1.5 bloccate (roadmap)
            self.task.addItem("🔒 " + label, key)
            it = self.task.model().item(self.task.count() - 1)
            it.setEnabled(False); it.setToolTip(tt)
        self.task.setToolTip(tr("music.task.sel.tt", "Tipo di operazione ACE-Step. Le voci 🔒 richiedono la fondazione 1.5."))
        self.task.currentIndexChanged.connect(self._sync)
        f.addRow(tr("music.task.op", "Operazione"), self.task)

        # audio sorgente (repaint/edit/extend/audio2audio)
        self.src = DropLineEdit(tr("music.task.src.ph", "audio sorgente (trascina o Sfoglia)"))
        self.src.setToolTip(tr("music.task.src.tt", "File audio su cui operare (repaint/edit/extend) o riferimento (audio2audio)."))
        self.row_src = self._row(f, tr("music.task.src", "Audio sorgente"), self.src)

        self.variance = QDoubleSpinBox(); self.variance.setRange(0.0, 1.0); self.variance.setSingleStep(0.05)
        self.variance.setValue(0.5); self.variance.setDecimals(2)
        self.variance.setToolTip(tr("music.task.var.tt", "0 = quasi identico all'originale · 1 = molto diverso."))
        self.row_var = self._row(f, tr("music.task.var", "Varianza"), self.variance)

        self.r_start = QDoubleSpinBox(); self.r_start.setRange(0.0, 600.0); self.r_start.setSuffix(" s")
        self.r_start.setToolTip(tr("music.task.rstart.tt", "Inizio (secondi) della sezione da rigenerare/estendere."))
        self.row_rstart = self._row(f, tr("music.task.rstart", "Inizio sezione"), self.r_start)
        self.r_end = QDoubleSpinBox(); self.r_end.setRange(0.0, 600.0); self.r_end.setSuffix(" s")
        self.r_end.setToolTip(tr("music.task.rend.tt", "Fine (secondi) della sezione da rigenerare/estendere."))
        self.row_rend = self._row(f, tr("music.task.rend", "Fine sezione"), self.r_end)

        self.e_prompt = QLineEdit()
        self.e_prompt.setToolTip(tr("music.task.eprompt.tt", "Nuovi tag di stile per l'edit (vuoto = usa i tag principali)."))
        self.row_eprompt = self._row(f, tr("music.task.eprompt", "Nuovi tag"), self.e_prompt)
        self.e_lyrics = QLineEdit()
        self.e_lyrics.setToolTip(tr("music.task.elyrics.tt", "Nuove liriche per l'edit (vuoto = usa le liriche principali). Modifica piccoli tratti alla volta."))
        self.row_elyrics = self._row(f, tr("music.task.elyrics", "Nuove liriche"), self.e_lyrics)

        self.strength = QDoubleSpinBox(); self.strength.setRange(0.0, 1.0); self.strength.setSingleStep(0.05)
        self.strength.setValue(0.5); self.strength.setDecimals(2)
        self.strength.setToolTip(tr("music.task.str.tt", "Quanto seguire l'audio di riferimento (0=poco, 1=molto)."))
        self.row_str = self._row(f, tr("music.task.str", "Forza riferimento"), self.strength)

        v.addLayout(f)
        self._sync()

    def _row(self, form: QFormLayout, label: str, widget: QWidget):
        """Aggiunge una riga (label+widget) e ritorna un contenitore per mostrarla/nasconderla."""
        lbl = QLabel(label)
        form.addRow(lbl, widget)
        return (lbl, widget)

    def _show(self, row, on: bool):
        row[0].setVisible(on); row[1].setVisible(on)

    def _sync(self):
        t = self.task.currentData()
        self.task.setToolTip(dict((k, tt) for k, _, tt in TASKS).get(t, ""))
        self._show(self.row_src, t in ("repaint", "edit", "extend", "audio2audio"))
        self._show(self.row_var, t == "retake")
        self._show(self.row_rstart, t in ("repaint", "extend"))
        self._show(self.row_rend, t in ("repaint", "extend"))
        self._show(self.row_eprompt, t == "edit")
        self._show(self.row_elyrics, t == "edit")
        self._show(self.row_str, t == "audio2audio")

    def set_engine(self, engine: str):
        self._content.setEnabled(engine == "ACE-Step")

    def set_repaint(self, src_path: str, start: float, end: float):
        """Configura il task su 'repaint' con sorgente + intervallo (usato dal trim del waveform)."""
        idx = self.task.findData("repaint")
        if idx >= 0:
            self.task.setCurrentIndex(idx)
        if src_path:
            self.src.setPath(src_path)
        self.r_start.setValue(float(start)); self.r_end.setValue(float(end))
        self.setChecked(True); self._sync()

    def reset_defaults(self):
        """Reimposta il task a generazione normale + parametri ai default."""
        self.task.setCurrentIndex(0)                         # text2music
        self.src.setPath(""); self.variance.setValue(0.5)
        self.r_start.setValue(0.0); self.r_end.setValue(0.0)
        self.e_prompt.clear(); self.e_lyrics.clear(); self.strength.setValue(0.5)
        self.setChecked(False)

    def get_task(self) -> dict:
        """dict per make_job(adv=...). Vuoto se generazione normale."""
        t = self.task.currentData()
        if t == "text2music":
            return {}
        d = {"task": t}
        if t in ("repaint", "edit", "extend", "audio2audio"):
            src = self.src.path()
            if t == "audio2audio":
                d["ref_audio"] = src; d["ref_strength"] = self.strength.value()
            else:
                d["src_audio"] = src
        if t == "retake":
            d["retake_variance"] = self.variance.value()
        if t in ("repaint", "extend"):
            d["repaint_start"] = self.r_start.value(); d["repaint_end"] = self.r_end.value()
        if t == "edit":
            d["edit_target_prompt"] = self.e_prompt.text().strip()
            d["edit_target_lyrics"] = self.e_lyrics.text().strip()
        return d
