"""Dialog analizzatore impostazioni — mostra errori, avvisi e info con possibilità di auto-fix."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea,
    QWidget, QFrame, QDialogButtonBox,
)

from core.settings_analyzer import SettingsAnalyzer, AnalysisResult, AnalyzerIssue


# Colori per severità
_COLORS = {
    "error":   ("#ff6b6b", "❌"),
    "warning": ("#ffa94d", "⚠️"),
    "info":    ("#74c0fc", "ℹ️"),
}


class _IssueRow(QFrame):
    """Riga che mostra un singolo issue con eventuale pulsante Auto-fix."""

    def __init__(self, issue: AnalyzerIssue, autofix_callback=None, parent=None):
        super().__init__(parent)
        self.issue = issue
        color, icon = _COLORS.get(issue.severity, ("#aaaaaa", "•"))
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"border-left: 4px solid {color}; padding-left: 6px; margin-bottom: 4px;")

        h = QHBoxLayout(self)
        h.setContentsMargins(6, 4, 6, 4)

        lbl = QLabel(f"{icon}  {issue.message}")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {color}; border: none;")
        h.addWidget(lbl, 1)

        if issue.autofix and autofix_callback:
            btn = QPushButton("🔧 Auto-fix")
            btn.setMaximumWidth(90)
            btn.setToolTip("Applica automaticamente la correzione suggerita")
            btn.clicked.connect(lambda: autofix_callback(issue.autofix))
            h.addWidget(btn)


class SettingsAnalyzerDialog(QDialog):
    """Mostra il risultato dell'analisi delle impostazioni con auto-fix opzionale."""

    def __init__(self, result: AnalysisResult, autofix_callback=None, parent=None):
        """
        Args:
            result:           output di SettingsAnalyzer.analyze()
            autofix_callback: callable(fixes: dict) che applica le correzioni alla GUI
            parent:           widget genitore
        """
        super().__init__(parent)
        self.result = result
        self.autofix_callback = autofix_callback
        self.setWindowTitle("🔍 Analizzatore impostazioni")
        self.setMinimumWidth(620)
        self.setMinimumHeight(420)
        self._build()

    def _build(self):
        v = QVBoxLayout(self)

        # Intestazione riepilogo
        errs = len(self.result.errors)
        warns = len(self.result.warnings)
        infos = len(self.result.infos)
        total = len(self.result.issues)

        if total == 0:
            summary = "✅ Nessun problema rilevato — le impostazioni sono valide."
            color = "#51cf66"
        elif errs:
            summary = f"❌ Trovati {errs} errori, {warns} avvisi, {infos} info — correggi gli errori prima di generare."
            color = "#ff6b6b"
        else:
            summary = f"⚠️ Trovati {warns} avvisi, {infos} info — la generazione è possibile ma controllare."
            color = "#ffa94d"

        hdr = QLabel(summary)
        hdr.setWordWrap(True)
        hdr.setStyleSheet(f"color: {color}; font-weight: bold; padding: 6px;")
        v.addWidget(hdr)

        if total == 0:
            v.addWidget(QLabel("Le impostazioni correnti sono compatibili con LTX Studio."))
        else:
            # Scroll area con le righe degli issue
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            container = QWidget()
            cv = QVBoxLayout(container)
            cv.setSpacing(2)

            # Ordine: errori → avvisi → info
            for severity in ("error", "warning", "info"):
                group = [i for i in self.result.issues if i.severity == severity]
                if not group:
                    continue
                _, icon = _COLORS[severity]
                sec_lbl = QLabel(f"{icon}  {'Errori' if severity == 'error' else 'Avvisi' if severity == 'warning' else 'Info'} ({len(group)})")
                sec_lbl.setStyleSheet("font-weight: bold; margin-top: 8px; border: none;")
                cv.addWidget(sec_lbl)
                for issue in group:
                    row = _IssueRow(issue, autofix_callback=self._do_autofix)
                    cv.addWidget(row)

            cv.addStretch()
            scroll.setWidget(container)
            v.addWidget(scroll, 1)

        # Pulsanti Auto-fix tutto + Chiudi
        bb_row = QHBoxLayout()
        if any(i.autofix for i in self.result.issues) and self.autofix_callback:
            btn_fix_all = QPushButton("🔧 Auto-fix tutto")
            btn_fix_all.setToolTip("Applica tutte le correzioni automatiche disponibili")
            btn_fix_all.clicked.connect(self._fix_all)
            bb_row.addWidget(btn_fix_all)
        bb_row.addStretch()
        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.accept)
        bb_row.addWidget(close_btn)
        v.addLayout(bb_row)

    def _do_autofix(self, fixes: dict):
        if self.autofix_callback:
            self.autofix_callback(fixes)
            self._log(f"Auto-fix applicato: {fixes}")

    def _fix_all(self):
        merged: dict = {}
        for issue in self.result.issues:
            if issue.autofix:
                merged.update(issue.autofix)
        if merged and self.autofix_callback:
            self.autofix_callback(merged)
            self._log(f"Auto-fix globale: {merged}")

    def _log(self, msg: str):
        mw = self.parent()
        if mw and hasattr(mw, "_log"):
            mw._log(f"[Analyzer] {msg}")
