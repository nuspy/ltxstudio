"""Dialog selezione Workflow Predefiniti.

Dropdown con tutti i workflow (35 preset in 9 categorie), dettaglio con:
- Note d'uso + LoRA suggerite
- Parametri tecnici (steps, CFG, risoluzione, FPS, audio)
- Prompt template editabile con avviso placeholder {…}
- Opzioni: applica solo parametri / solo prompt / entrambi
"""
from __future__ import annotations

import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPlainTextEdit,
    QGroupBox, QFormLayout, QPushButton, QDialogButtonBox, QCheckBox, QFrame,
    QLineEdit, QMessageBox,
)

from core.workflow_templates import WORKFLOW_PRESETS
from ui.i18n_manager import tr


class WorkflowDialog(QDialog):
    """Seleziona un workflow predefinito e lo applica alla GUI principale."""

    def __init__(self, apply_callback=None, parent=None):
        super().__init__(parent)
        self.apply_callback = apply_callback
        self.setWindowTitle(tr("wf.window_title"))
        self.setMinimumWidth(1040)          # box dettagli ~3x più larghi (erano illeggibili a 680)
        self.setMinimumHeight(560)
        self._build()
        self._on_workflow_changed(0)

    # ---------- build ----------
    def _build(self):
        v = QVBoxLayout(self)

        # ── Selezione workflow ────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("wf.workflow_label")))
        self.combo = QComboBox()
        self.combo.setMaxVisibleItems(30)
        self.combo.view().setMinimumWidth(520)
        for name in WORKFLOW_PRESETS:
            self.combo.addItem(name)
        self.combo.currentIndexChanged.connect(self._on_workflow_changed)
        row.addWidget(self.combo, 1)
        v.addLayout(row)

        # ── Dettagli workflow ─────────────────────────────────────────────
        detail_grp = QGroupBox(tr("wf.details"))
        df = QFormLayout(detail_grp)
        df.setLabelAlignment(Qt.AlignRight)
        df.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # box dettagli: QLabel multi-riga con bordo, larghi e auto-altezza in base al contenuto.
        def _box(color: str, extra: str = "") -> QLabel:
            lb = QLabel()
            lb.setWordWrap(True)                         # → va a capo + cresce in verticale da solo
            lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lb.setMinimumWidth(740)                      # ~3x rispetto a prima
            lb.setStyleSheet(
                f"color: {color}; {extra} padding: 6px 8px; background: #1b1b1b;"
                "border: 1px solid #333; border-radius: 4px;")
            return lb

        self.note_lbl = _box("#74c0fc")
        df.addRow(tr("wf.notes_row"), self.note_lbl)

        self.params_lbl = _box("#dddddd", "font-family: monospace; font-size: 11px;")
        df.addRow(tr("wf.params_row"), self.params_lbl)

        self.lora_lbl = _box("#a9e34b", "font-size: 11px;")
        df.addRow(tr("wf.lora_suggested_row"), self.lora_lbl)

        v.addWidget(detail_grp)

        # ── Prompt template ───────────────────────────────────────────────
        v.addWidget(QLabel(tr("wf.prompt_template_hint")))
        self.prompt_txt = QPlainTextEdit()
        self.prompt_txt.setMaximumHeight(110)
        self.prompt_txt.textChanged.connect(self._check_placeholders)
        v.addWidget(self.prompt_txt)

        # Avviso placeholder non compilati
        self.placeholder_warn = QLabel()
        self.placeholder_warn.setWordWrap(True)
        self.placeholder_warn.setStyleSheet(
            "color: #ffa94d; font-size: 11px; padding: 2px 4px;"
            "background: #2a1a00; border-radius: 3px;"
        )
        self.placeholder_warn.setVisible(False)
        v.addWidget(self.placeholder_warn)

        # ── Opzioni applicazione ──────────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333; margin: 4px 0;")
        v.addWidget(sep)

        opts_grp = QGroupBox(tr("wf.what_to_apply"))
        ov = QHBoxLayout(opts_grp)

        self.chk_params = QCheckBox(tr("wf.chk_params"))
        self.chk_params.setChecked(True)
        ov.addWidget(self.chk_params)

        self.chk_prompt = QCheckBox(tr("wf.chk_prompt"))
        self.chk_prompt.setChecked(True)
        ov.addWidget(self.chk_prompt)

        v.addWidget(opts_grp)

        # Nota LoRA
        lora_note = QLabel(tr("wf.lora_note"))
        lora_note.setWordWrap(True)
        lora_note.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
        v.addWidget(lora_note)

        # ── Pulsanti ─────────────────────────────────────────────────────
        bb = QDialogButtonBox()
        btn_apply = bb.addButton(tr("wf.btn_apply"), QDialogButtonBox.AcceptRole)
        btn_apply.clicked.connect(self._on_apply)
        bb.addButton(tr("wf.btn_close"), QDialogButtonBox.RejectRole)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    # ---------- slot ----------
    def _on_workflow_changed(self, _idx: int):
        name = self.combo.currentText()
        wf = WORKFLOW_PRESETS.get(name, {})
        if not wf:
            return

        # Note
        self.note_lbl.setText(wf.get("notes", "—"))

        # Parametri tecnici
        res = f"{wf.get('width','?')}×{wf.get('height','?')}"
        sched = wf.get("scheduler", "beta")
        params = (
            f"Steps={wf.get('steps','?')}  ·  CFG={wf.get('cfg','?')}  ·  "
            f"Sampler={wf.get('sampler','?')}  ·  Scheduler={sched}\n"
            f"Risoluzione={res}  ·  FPS={wf.get('fps','?')}  ·  "
            f"Durata={wf.get('length_s','?')}s  ·  "
            f"NSFW={wf.get('nsfw_mode','sfw').upper()}  ·  "
            f"Audio={wf.get('audio_mode','native')}"
        )
        self.params_lbl.setText(params)

        # LoRA suggerite
        loras = wf.get("lora_suggestions", [])
        if loras:
            self.lora_lbl.setText("• " + "\n• ".join(loras))
        else:
            self.lora_lbl.setText(tr("wf.lora_none"))

        # Prompt template
        self.prompt_txt.setPlainText(wf.get("prompt_template", ""))
        # Controlla subito i placeholder
        self._check_placeholders()

    def _check_placeholders(self):
        """Avvisa se il prompt contiene placeholder {…} non compilati."""
        txt = self.prompt_txt.toPlainText()
        placeholders = re.findall(r'\{(\w+)\}', txt)
        if placeholders:
            unique = list(dict.fromkeys(placeholders))   # mantieni ordine, dedup
            self.placeholder_warn.setText(
                tr("wf.placeholder_warn").format(
                    count=len(unique),
                    placeholders=", ".join(f"{{{p}}}" for p in unique),
                )
            )
            self.placeholder_warn.setVisible(True)
        else:
            self.placeholder_warn.setVisible(False)

    def _fill_placeholders(self, tmpl: str) -> str | None:
        """Se il prompt ha placeholder {…}, chiede all'utente di compilarli e li sostituisce.
        Ritorna il prompt compilato, oppure None se l'utente annulla (→ non applicare)."""
        names = list(dict.fromkeys(re.findall(r"\{(\w+)\}", tmpl)))
        if not names:
            return tmpl
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("wf.fill_title"))
        dlg.setMinimumWidth(460)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(tr("wf.fill_hint")))
        form = QFormLayout()
        edits: dict[str, QLineEdit] = {}
        for n in names:
            e = QLineEdit()
            e.setPlaceholderText(n)                       # suggerimento = nome del placeholder
            form.addRow(f"{{{n}}}", e)
            edits[n] = e
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        # "Applica così com'è" = applica col placeholder ancora dentro (terzo pulsante)
        btn_asis = bb.addButton(tr("wf.fill_as_is"), QDialogButtonBox.ActionRole)
        lay.addWidget(bb)
        result = {"mode": None}
        bb.accepted.connect(lambda: (result.update(mode="fill"), dlg.accept()))
        bb.rejected.connect(dlg.reject)
        btn_asis.clicked.connect(lambda: (result.update(mode="asis"), dlg.accept()))
        if dlg.exec() != QDialog.Accepted or result["mode"] is None:
            return None                                   # annullato → non applicare
        if result["mode"] == "asis":
            return tmpl
        out = tmpl
        for n, e in edits.items():
            val = e.text().strip()
            if val:
                out = out.replace(f"{{{n}}}", val)
        return out

    def _on_apply(self):
        if not self.apply_callback:
            self.accept()
            return
        name = self.combo.currentText()
        wf = WORKFLOW_PRESETS.get(name, {})
        fields: dict = {}

        if self.chk_params.isChecked():
            for key in ("steps", "cfg", "sampler", "scheduler", "width", "height",
                        "fps", "length_s", "nsfw_mode", "audio_mode"):
                if key in wf:
                    fields[key] = wf[key]

        if self.chk_prompt.isChecked():
            tmpl = self.prompt_txt.toPlainText().strip()
            if tmpl:
                filled = self._fill_placeholders(tmpl)   # smart: chiede i {placeholder}
                if filled is None:
                    return                                # utente ha annullato → non chiudere
                fields["prompt_positive"] = filled

        if fields:
            self.apply_callback(fields, workflow_name=name)
        self.accept()
