"""Finestra di build: scegli edizioni × OS, override major, FIRMA (checkbox), dry-run.

Mostra anche la MATRICE funzioni × edizione (da matrix.json) con etichette localizzate,
così si vede a colpo d'occhio cosa include ogni edizione prima di compilare.

Pilota build/build.py come sottoprocesso, streammando il log. La firma (Azure
Artifact Signing) è OPZIONALE: spunta la checkbox e inserisci il certificato/
credenziali solo per la release reale.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Rendi importabile il pacchetto i18n dell'app anche lanciando questo file da solo.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QSpinBox, QLineEdit, QPushButton, QPlainTextEdit, QLabel, QFormLayout, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt

from ui.i18n_manager import tr

BUILD_PY = str(Path(__file__).resolve().parent / "build.py")
DEFAULT_PROJECT = r"C:\Projects\feature_matrix\projects\ltxstudio\project.json"


class BuildGUI(QWidget):
    def __init__(self, project: str = DEFAULT_PROJECT):
        super().__init__()
        self.project = project
        self.desc = json.loads(Path(project).read_text("utf-8"))
        self.matrix = self._load_matrix()
        self.proc: QProcess | None = None
        self.setWindowTitle(tr("build.title", "LTX Studio — Build"))
        self.resize(900, 720)
        self._build()

    def _load_matrix(self) -> dict:
        """Carica matrix.json (relativo al project) per la vista funzioni × edizione."""
        try:
            mp = self.desc.get("matrix_path", "matrix.json")
            path = (Path(self.project).parent / mp)
            return json.loads(path.read_text("utf-8"))
        except Exception:
            return {}

    def _build(self):
        root = QVBoxLayout(self)

        ed_grp = QGroupBox(tr("build.editions", "Edizioni")); el = QHBoxLayout(ed_grp)
        self.ed_checks = {}
        for e in self.desc["editions"]:
            cb = QCheckBox(f"{e['id']} ({e['public_name']})"); cb.setChecked(True)
            self.ed_checks[e["id"]] = cb; el.addWidget(cb)
        root.addWidget(ed_grp)

        os_grp = QGroupBox(tr("build.os", "Sistemi operativi")); ol = QHBoxLayout(os_grp)
        self.os_checks = {}
        for o in self.desc.get("os_targets", ["windows"]):
            cb = QCheckBox(o); cb.setChecked(o == "windows")
            self.os_checks[o] = cb; ol.addWidget(cb)
        root.addWidget(os_grp)

        # ── Matrice funzioni × edizione (sola lettura) + apertura editor ──────
        edit_row = QHBoxLayout()
        edit_row.addStretch(1)
        self.btn_edit_matrix = QPushButton(tr("build.edit_matrix", "✏️ Modifica matrice / Link tier (upsell)…"))
        self.btn_edit_matrix.setToolTip(tr("build.edit_matrix.tt",
            "Apre l'editor della matrice: abilita/disabilita funzioni per edizione e imposta i link Patreon "
            "per-tier (bottone 'Link tier'). Le modifiche vanno salvate e sono usate al build."))
        self.btn_edit_matrix.clicked.connect(self._open_matrix_editor)
        edit_row.addWidget(self.btn_edit_matrix)
        root.addLayout(edit_row)
        mx = self._matrix_box()
        if mx is not None:
            root.addWidget(mx, 1)

        opt = QFormLayout()
        self.major = QSpinBox(); self.major.setRange(0, 999); self.major.setSpecialValueText("auto")
        opt.addRow(tr("build.major", "Major override (0=auto, minor automatico):"), self.major)
        self.dry = QCheckBox(tr("build.dryrun", "Dry-run (stampa il piano, non compila)")); self.dry.setChecked(True)
        opt.addRow("", self.dry)
        root.addLayout(opt)

        sign_grp = QGroupBox(tr("build.sign.group", "Firma del codice (opzionale)")); sl = QFormLayout(sign_grp)
        self.sign = QCheckBox(tr("build.sign.enable", "Firma con Azure Artifact Signing"))
        sl.addRow("", self.sign)
        self.dlib = QLineEdit(); self.dlib.setPlaceholderText(tr("build.sign.dlib.hint", "percorso Azure.CodeSigning.Dlib.dll"))
        dl_btn = QPushButton("…"); dl_btn.clicked.connect(lambda: self._pick(self.dlib))
        row = QHBoxLayout(); row.addWidget(self.dlib); row.addWidget(dl_btn)
        w = QWidget(); w.setLayout(row); sl.addRow(tr("build.sign.dlib", "Dlib:"), w)
        self.meta = QLineEdit(); self.meta.setPlaceholderText(tr("build.sign.meta.hint", "percorso metadata.json (Endpoint/Account/Profile)"))
        mt_btn = QPushButton("…"); mt_btn.clicked.connect(lambda: self._pick(self.meta))
        row2 = QHBoxLayout(); row2.addWidget(self.meta); row2.addWidget(mt_btn)
        w2 = QWidget(); w2.setLayout(row2); sl.addRow(tr("build.sign.meta", "Metadata:"), w2)
        sl.addRow(QLabel(tr("build.sign.note", "Le credenziali NON vengono salvate nel repo; usate solo per questa build.")))
        root.addWidget(sign_grp)

        self.run_btn = QPushButton(tr("build.compile", "Compila")); self.run_btn.clicked.connect(self._run)
        root.addWidget(self.run_btn)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); root.addWidget(self.log, 1)

    def _matrix_box(self) -> QGroupBox | None:
        feats = self.matrix.get("features") or []
        eds = self.matrix.get("editions") or self.desc.get("editions") or []
        if not feats or not eds:
            return None
        box = QGroupBox(tr("build.matrix", "Matrice funzioni × edizione"))
        v = QVBoxLayout(box)
        table = QTableWidget(len(feats), 1 + len(eds))
        headers = [tr("build.feature", "Funzione")] + [e["public_name"] for e in eds]
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        for r, f in enumerate(feats):
            label = tr(f.get("i18n_key") or ("feat." + f["id"]), f.get("label", f["id"]))
            it0 = QTableWidgetItem(label)
            if f.get("sensitivity") == "adult":
                it0.setText("🔞 " + label)
            table.setItem(r, 0, it0)
            for c, e in enumerate(eds, 1):
                cell = (f.get("cells") or {}).get(e["id"])
                if cell is None:
                    sym = "·"                          # non specificata → regola di default
                else:
                    sym = "✓" if cell.get("enabled") else "✗"
                item = QTableWidgetItem(sym)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 1 + len(eds)):
            table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        v.addWidget(table)
        v.addWidget(QLabel(tr("build.legend", "Legenda: ✓ inclusa · ✗ esclusa · · = regola di default")))
        return box

    def _open_matrix_editor(self):
        """Lancia l'editor della matrice (feature_matrix), da cui si impostano anche i link Patreon per-tier."""
        import subprocess
        import sys
        try:
            subprocess.Popen([sys.executable, "-m", "feature_matrix.editor", DEFAULT_PROJECT],
                             cwd=r"C:\Projects\feature_matrix")
            self.log.appendPlainText("Editor matrice avviato — salva le modifiche prima di ricompilare.")
        except Exception as e:                               # noqa: BLE001
            self.log.appendPlainText(f"Impossibile avviare l'editor matrice: {e}")

    def _pick(self, edit):
        path, _ = QFileDialog.getOpenFileName(self, tr("build.pick", "Scegli file"))
        if path:
            edit.setText(path)

    def _run(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            return
        editions = [e for e, cb in self.ed_checks.items() if cb.isChecked()]
        oses = [o for o, cb in self.os_checks.items() if cb.isChecked()]
        if not editions or not oses:
            self.log.appendPlainText(tr("build.need_selection", "Seleziona almeno un'edizione e un OS.")); return
        args = [BUILD_PY, "--project", self.project]
        for e in editions:
            args += ["--edition", e]
        for o in oses:
            args += ["--os", o]
        if self.major.value() > 0:
            args += ["--major", str(self.major.value())]
        if self.dry.isChecked():
            args.append("--dry-run")
        if self.sign.isChecked():
            args.append("--sign")
        env = os.environ.copy()
        if self.sign.isChecked():
            if self.dlib.text():
                env["AZURE_SIGN_DLIB"] = self.dlib.text()
            if self.meta.text():
                env["AZURE_SIGN_METADATA"] = self.meta.text()
        self.log.clear()
        self.log.appendPlainText("$ python " + " ".join(args) + "\n")
        self.proc = QProcess(self)
        from PySide6.QtCore import QProcessEnvironment
        qenv = QProcessEnvironment()
        for k, v in env.items():
            qenv.insert(k, v)
        self.proc.setProcessEnvironment(qenv)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")))
        self.proc.start(sys.executable, args)


def main():
    project = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROJECT
    app = QApplication([])
    w = BuildGUI(project); w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
