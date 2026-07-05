"""Sezione 'Avanzate' della tab Musica — tutti i parametri fini di ACE-Step (#98 F2).

Principi UX (ui-ux-pro-max):
 - PROGRESSIVE DISCLOSURE: sezione collassabile (QGroupBox checkable), chiusa di default → non travolge.
 - FIELD GROUPING: sotto-gruppi Qualità / Output / Performance con label visibili (non solo placeholder).
 - DISABLED-STATE CHIARO: tutti questi param sono SOLO ACE-Step; con HeartMuLa il contenuto è disabilitato
   (grigio) e una nota spiega il PERCHÉ (non un grigio muto).
 - HELPER TEXT: tooltip dettagliato su OGNI controllo.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QLineEdit, QLabel,
)

from ui.i18n_manager import tr


def _spin(lo, hi, val, step=1.0, dec=2, suffix=""):
    s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    s.setDecimals(dec)
    if suffix:
        s.setSuffix(suffix)
    return s


class MusicAdvanced(QGroupBox):
    """Pannello parametri avanzati ACE-Step. get_adv() → dict per acestep_service.make_job(adv=...)."""

    def __init__(self):
        super().__init__(tr("music.adv.title", "⚙️  Avanzate (ACE-Step)"))
        self.setCheckable(True); self.setChecked(False)      # collassata di default
        self._build()

    def _build(self):
        root = QVBoxLayout(self)

        # Nota che spiega PERCHÉ è disabilitato (mostrata solo con HeartMuLa)
        self.note = QLabel(tr("music.adv.hm_note",
            "ⓘ Questi controlli sono disponibili solo con il motore ACE-Step. "
            "Seleziona ACE-Step per usarli."))
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#e0a458; background:#2a2416; border:1px solid #4a3f1e;"
                                "border-radius:6px; padding:6px 8px;")
        self.note.setVisible(False)
        root.addWidget(self.note)

        # contenuto attivabile/disattivabile in blocco
        self._content = QWidget(); cv = QVBoxLayout(self._content)
        cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(8)
        root.addWidget(self._content)

        # ── Qualità & aderenza ──
        gq = QGroupBox(tr("music.adv.quality", "Qualità & aderenza"))
        fq = QFormLayout(gq); fq.setLabelAlignment(Qt.AlignRight)
        self.steps = QSpinBox(); self.steps.setRange(20, 200); self.steps.setValue(60)
        self.steps.setToolTip(tr("music.adv.steps.tt",
            "Passi di denoising. Più passi = più dettaglio e audio più pulito (ma più lento). "
            "60 è un buon compromesso; sotto ~30 può comparire rumore grigio."))
        fq.addRow(tr("music.adv.steps", "Step di sampling"), self.steps)
        self.scheduler = QComboBox(); self.scheduler.addItems(["euler", "heun", "pingpong"])
        self.scheduler.setToolTip(tr("music.adv.scheduler.tt",
            "Campionatore.\n• pingpong (SDE): migliore coerenza e allineamento di stile/liriche (consigliato per aderenza)\n"
            "• euler: veloce e standard\n• heun: più preciso ma più lento"))
        fq.addRow(tr("music.adv.scheduler", "Scheduler"), self.scheduler)
        self.cfg_type = QComboBox(); self.cfg_type.addItems(["apg", "cfg", "cfg_star"])
        self.cfg_type.setToolTip(tr("music.adv.cfgtype.tt",
            "Algoritmo di guida.\n• apg: consigliato (stabile, ottima aderenza al prompt)\n"
            "• cfg: classico Classifier-Free Guidance\n• cfg_star: variante più recente"))
        fq.addRow(tr("music.adv.cfgtype", "Tipo CFG"), self.cfg_type)
        self.omega = _spin(0.0, 30.0, 10.0, 0.5, 1)
        self.omega.setToolTip(tr("music.adv.omega.tt",
            "Controllo della varianza dello scheduler (default 10). Valori più alti = maggiore consistenza del brano."))
        fq.addRow(tr("music.adv.omega", "Omega scale"), self.omega)
        self.g_interval = _spin(0.0, 1.0, 0.5, 0.05, 2)
        self.g_interval.setToolTip(tr("music.adv.ginterval.tt",
            "Frazione di step in cui la guida è attiva (0.5 = prima metà). Più basso = guida solo all'inizio."))
        fq.addRow(tr("music.adv.ginterval", "Guidance interval"), self.g_interval)
        self.g_decay = _spin(0.0, 5.0, 0.0, 0.1, 2)
        self.g_decay.setToolTip(tr("music.adv.gdecay.tt",
            "Decadimento della guida verso il minimo durante il sampling (0 = costante). "
            ">0 = risultati più naturali verso la fine."))
        fq.addRow(tr("music.adv.gdecay", "Guidance decay"), self.g_decay)
        self.min_g = _spin(0.0, 15.0, 3.0, 0.5, 1)
        self.min_g.setToolTip(tr("music.adv.ming.tt", "Pavimento della guida quando decade (default 3.0)."))
        fq.addRow(tr("music.adv.ming", "Min guidance"), self.min_g)
        cv.addWidget(gq)

        # ── ERG (anti-artefatto) ──
        ge = QGroupBox(tr("music.adv.erg", "ERG — Emphasize & Relax Guidance"))
        fe = QFormLayout(ge); fe.setLabelAlignment(Qt.AlignRight)
        erg_tt = tr("music.adv.erg.tt",
            "ERG mantiene l'aderenza aumentando la diversità/qualità. Consigliato ATTIVO. "
            "Disattiva per risultati più deterministici ma meno vari.")
        self.erg_tag = QCheckBox(tr("music.adv.erg_tag", "Tag")); self.erg_tag.setChecked(True)
        self.erg_lyric = QCheckBox(tr("music.adv.erg_lyric", "Liriche")); self.erg_lyric.setChecked(True)
        self.erg_diff = QCheckBox(tr("music.adv.erg_diff", "Diffusione")); self.erg_diff.setChecked(True)
        for c in (self.erg_tag, self.erg_lyric, self.erg_diff):
            c.setToolTip(erg_tt)
        er = QWidget(); erl = QVBoxLayout(er); erl.setContentsMargins(0, 0, 0, 0)
        erl.addWidget(self.erg_tag); erl.addWidget(self.erg_lyric); erl.addWidget(self.erg_diff)
        fe.addRow(tr("music.adv.erg_lbl", "Attivi"), er)
        cv.addWidget(ge)

        # ── Output & riproducibilità ──
        go = QGroupBox(tr("music.adv.output", "Output & riproducibilità"))
        fo = QFormLayout(go); fo.setLabelAlignment(Qt.AlignRight)
        self.fmt = QComboBox(); self.fmt.addItems(["wav", "mp3", "ogg", "flac"])
        self.fmt.setToolTip(tr("music.adv.format.tt",
            "Formato di output.\n• wav / flac: senza perdita\n• mp3 / ogg: compressi (file più piccoli)"))
        fo.addRow(tr("music.adv.format", "Formato"), self.fmt)
        self.seed = QLineEdit(); self.seed.setPlaceholderText(tr("music.adv.seed.ph", "casuale (es. 42 o 42,100)"))
        self.seed.setToolTip(tr("music.adv.seed.tt",
            "Seme di generazione. Vuoto = casuale ogni volta. Fissarlo (es. 42) rende il risultato "
            "RIPRODUCIBILE. Più semi separati da virgola per il batch."))
        fo.addRow(tr("music.adv.seed", "Seed"), self.seed)
        self.batch = QSpinBox(); self.batch.setRange(1, 8); self.batch.setValue(1)
        self.batch.setToolTip(tr("music.adv.batch.tt",
            "Quante canzoni generare in un colpo solo (varianti multiple dello stesso prompt)."))
        fo.addRow(tr("music.adv.batch", "Batch"), self.batch)
        cv.addWidget(go)

        # ── Performance / VRAM ──
        gp = QGroupBox(tr("music.adv.perf", "Performance / VRAM"))
        fp = QVBoxLayout(gp)
        self.quantized = QCheckBox(tr("music.adv.quantized", "Modello quantizzato (low-VRAM su GPU)"))
        self.quantized.setToolTip(tr("music.adv.quantized.tt",
            "Carica il modello quantizzato: usa MENO VRAM restando su GPU (più veloce del 'Low VRAM' "
            "che sposta i pesi su CPU)."))
        self.overlap = QCheckBox(tr("music.adv.overlap", "Decodifica a sovrapposizione"))
        self.overlap.setToolTip(tr("music.adv.overlap.tt",
            "Decodifica overlapped: meno memoria e più veloce nella fase finale di decodifica audio."))
        self.compile = QCheckBox(tr("music.adv.compile", "torch.compile (più veloce)"))
        self.compile.setToolTip(tr("music.adv.compile.tt",
            "Compila il modello: più veloce dopo il primo avvio (la prima compilazione richiede tempo extra)."))
        for c in (self.quantized, self.overlap, self.compile):
            fp.addWidget(c)
        cv.addWidget(gp)

        # ── LoRA (applica una LoRA allenata in generazione) ──
        from ui.widgets.drop_widgets import DropLineEdit
        gl = QGroupBox(tr("music.adv.lora", "LoRA"))
        fl = QFormLayout(gl); fl.setLabelAlignment(Qt.AlignRight)
        self.lora = DropLineEdit(tr("music.adv.lora.ph", "LoRA .safetensors (opzionale)"),
                                 "LoRA (*.safetensors *.ckpt *.pt)")
        self.lora.setToolTip(tr("music.adv.lora.tt",
            "Applica una LoRA allenata (es. quella prodotta dal trainer qui sotto) alla generazione."))
        fl.addRow(tr("music.adv.lora_file", "File"), self.lora)
        self.lora_w = _spin(0.0, 2.0, 1.0, 0.05, 2)
        self.lora_w.setToolTip(tr("music.adv.lora_w.tt", "Intensità della LoRA (1.0 = piena)."))
        fl.addRow(tr("music.adv.lora_weight", "Peso"), self.lora_w)
        cv.addWidget(gl)

    def set_engine(self, engine: str):
        """Abilita i controlli solo per ACE-Step; con HeartMuLa disabilita + mostra la nota del PERCHÉ."""
        is_ace = (engine == "ACE-Step")
        self._content.setEnabled(is_ace)
        self.note.setVisible(not is_ace)

    def reset_defaults(self):
        """Reimposta tutti i controlli avanzati ai valori ottimali."""
        self.steps.setValue(60)
        self.scheduler.setCurrentText("euler"); self.cfg_type.setCurrentText("apg")
        self.omega.setValue(10.0); self.g_interval.setValue(0.5); self.g_decay.setValue(0.0); self.min_g.setValue(3.0)
        self.erg_tag.setChecked(True); self.erg_lyric.setChecked(True); self.erg_diff.setChecked(True)
        self.fmt.setCurrentText("wav"); self.seed.clear(); self.batch.setValue(1)
        self.quantized.setChecked(False); self.overlap.setChecked(False); self.compile.setChecked(False)
        self.lora.setPath(""); self.lora_w.setValue(1.0)
        self.setChecked(False)                               # richiudi la sezione

    def set_config(self, d: dict):
        """Applica (best-effort) un dict di parametri avanzati ai controlli (per i preset)."""
        d = d or {}
        if "steps" in d:
            self.steps.setValue(int(d["steps"]))
        if "scheduler" in d:
            self.scheduler.setCurrentText(str(d["scheduler"]))
        if "cfg_type" in d:
            self.cfg_type.setCurrentText(str(d["cfg_type"]))
        for key, w in (("omega", self.omega), ("guidance_interval", self.g_interval),
                       ("guidance_interval_decay", self.g_decay), ("min_guidance", self.min_g)):
            if key in d:
                w.setValue(float(d[key]))
        if "format" in d:
            self.fmt.setCurrentText(str(d["format"]))
        if "seeds" in d:
            self.seed.setText(str(d["seeds"]))
        if "batch" in d:
            self.batch.setValue(int(d["batch"]))
        for key, w in (("quantized", self.quantized), ("overlapped_decode", self.overlap),
                       ("torch_compile", self.compile)):
            if key in d:
                w.setChecked(bool(d[key]))
        self.erg_tag.setChecked(not d.get("no_erg_tag", False))
        self.erg_lyric.setChecked(not d.get("no_erg_lyric", False))
        self.erg_diff.setChecked(not d.get("no_erg_diffusion", False))

    def get_adv(self) -> dict:
        """Dizionario opzioni per acestep_service.make_job(adv=...). Solo i valori diversi dal default."""
        adv = {
            "steps": self.steps.value(),
            "scheduler": self.scheduler.currentText(),
            "cfg_type": self.cfg_type.currentText(),
            "omega": self.omega.value(),
            "guidance_interval": self.g_interval.value(),
            "guidance_interval_decay": self.g_decay.value(),
            "min_guidance": self.min_g.value(),
            "format": self.fmt.currentText(),
            "seeds": self.seed.text().strip(),
            "batch": self.batch.value(),
            "quantized": self.quantized.isChecked(),
            "overlapped_decode": self.overlap.isChecked(),
            "torch_compile": self.compile.isChecked(),
        }
        if self.lora.path():                                 # LoRA allenata applicata alla generazione
            adv["lora_path"] = self.lora.path(); adv["lora_weight"] = self.lora_w.value()
        if not self.erg_tag.isChecked():
            adv["no_erg_tag"] = True
        if not self.erg_lyric.isChecked():
            adv["no_erg_lyric"] = True
        if not self.erg_diff.isChecked():
            adv["no_erg_diffusion"] = True
        return adv
