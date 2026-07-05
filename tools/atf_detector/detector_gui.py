"""ATF Detector — app GUI standalone (#107).

Verifica se una FOTO o un VIDEO è stato generato con AI Tools Forge, leggendo la firma di provenienza
invisibile (+ eventuale testo custom). Funziona su qualsiasi fonte: file, screenshot, foto dello schermo.
Drag-and-drop o "Apri…". Nessun originale necessario: rilevamento cieco tramite la chiave.

Esecuzione:
  python tools/atf_detector/detector_gui.py           # dal repo (usa core/stego.py)
Compilazione in .exe autonomo:
  python tools/atf_detector/build_detector.py          # (bundla SOLO core/stego.py + cv2/numpy + PySide6)

L'app dipende solo da core/stego.py (unica fonte della logica/chiave) + numpy + opencv + PySide6.
"""
from __future__ import annotations

import os
import sys

# rende importabile core.stego sia da repo che da bundle PyInstaller
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if hasattr(sys, "_MEIPASS"):                                # dentro il bundle compilato
    sys.path.insert(0, sys._MEIPASS)

from PySide6.QtCore import Qt, QThread, Signal                # noqa: E402
from PySide6.QtGui import QFont                               # noqa: E402
from PySide6.QtWidgets import (                               # noqa: E402
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem,
)

from core import stego                                        # noqa: E402

_ACCENT = "#8e6bff"


class _Worker(QThread):
    done = Signal(str, bool, float, str)                     # path, found, z, text

    def __init__(self, paths):
        super().__init__()
        self._paths = paths

    def run(self):
        for p in self._paths:
            try:
                found, z, _n, text = stego.detect_file(p)
            except Exception:                                # noqa: BLE001
                found, z, text = False, 0.0, None
            self.done.emit(p, bool(found), float(z), text or "")


class Detector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATF Detector — AI Tools Forge")
        self.resize(640, 460)
        self.setAcceptDrops(True)
        v = QVBoxLayout(self)

        title = QLabel("🔎 ATF Detector")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet(f"color:{_ACCENT};")
        v.addWidget(title)
        v.addWidget(QLabel("Trascina qui foto o video (o file screenshot) per verificarne la provenienza."))

        self.drop = QLabel("⬇  Trascina i file qui  ⬇")
        self.drop.setAlignment(Qt.AlignCenter)
        self.drop.setMinimumHeight(90)
        self.drop.setStyleSheet(f"border:2px dashed {_ACCENT}; border-radius:10px; color:#9aa0b4; font-size:15px;")
        v.addWidget(self.drop)

        row = QHBoxLayout()
        b1 = QPushButton("📂 Apri file…"); b1.clicked.connect(self._pick_files)
        b2 = QPushButton("🗂 Scansiona cartella…"); b2.clicked.connect(self._pick_dir)
        row.addWidget(b1); row.addWidget(b2); row.addStretch(1)
        v.addLayout(row)

        self.results = QListWidget()
        v.addWidget(self.results, 1)
        self.status = QLabel("pronto"); self.status.setStyleSheet("color:#8a93a6;")
        v.addWidget(self.status)

    # ---- input ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        self._run([p for p in paths if os.path.isfile(p)])

    def _pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Scegli foto o video", "",
                                                "Media (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.webm *.mkv *.mov)")
        self._run(paths)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Scegli una cartella")
        if not d:
            return
        exts = stego._IMG_EXT | stego._VID_EXT
        paths = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs
                 if os.path.splitext(f)[1].lower() in exts]
        self._run(paths)

    # ---- run ----
    def _run(self, paths):
        if not paths:
            return
        self.results.clear()
        self.status.setText(f"analizzo {len(paths)} file…")
        self._w = _Worker(paths)
        self._w.done.connect(self._on_result)
        self._w.finished.connect(lambda: self.status.setText("fatto."))
        self._w.start()

    def _on_result(self, path, found, z, text):
        name = os.path.basename(path)
        if found:
            msg = f"✅  {name}  —  GENERATO con AI Tools Forge"
            if text:
                msg += f'   ·   testo: "{text}"'
            msg += f"   (affidabilità {z:.1f})"
            color = "#2f9e44"
        else:
            msg = f"❌  {name}  —  nessuna firma rilevata"
            color = "#b0b4c0"
        it = QListWidgetItem(msg)
        it.setForeground(Qt.GlobalColor.green if found else Qt.GlobalColor.gray)
        it.setToolTip(path)
        self.results.addItem(it)


def main():
    app = QApplication(sys.argv)
    w = Detector(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
