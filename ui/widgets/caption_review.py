"""Dialog NON-MODALE di revisione caption per i dataset LoRA (immagini o video) — #review-caption.

Mostra la lista dei media con anteprima + un editor di caption per ognuno. La caption si salva
AUTOMATICAMENTE: all'uscita dal campo (focus-out), alla chiusura della finestra e col pulsante Salva.
Non blocca la finestra sottostante (setModal(False) + show()). UX: ui-ux-pro-max (auto-save, feedback,
anteprima, form-autosave, escape-routes).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QScrollArea, QWidget, QLabel, QPlainTextEdit,
    QPushButton, QDialogButtonBox, QFrame,
)

from ui.i18n_manager import tr

_FFMPEG = r"E:\AI\Tools\ffmpeg-shared\bin\ffmpeg.exe"
_THUMB_DIR = os.path.join(tempfile.gettempdir(), "ltx_capthumbs")
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _video_thumb(path: str) -> str | None:
    """Estrae (una volta, in cache) un frame del video come anteprima."""
    try:
        os.makedirs(_THUMB_DIR, exist_ok=True)
        key = hashlib.md5(path.encode("utf-8")).hexdigest()[:16]
        out = os.path.join(_THUMB_DIR, key + ".png")
        if not os.path.exists(out):
            subprocess.run([_FFMPEG, "-y", "-ss", "0.5", "-i", path, "-frames:v", "1",
                            "-vf", "scale=140:-1", out], capture_output=True, timeout=15)
        return out if os.path.exists(out) else None
    except Exception:                                        # noqa: BLE001
        return None


class _CaptionEdit(QPlainTextEdit):
    """Editor caption che emette 'committed' quando perde il focus (auto-save)."""
    committed = Signal()

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self.committed.emit()


class CaptionReviewDialog(QDialog):
    """items: [{path, caption}]. save_cb(index, path, caption) chiamata ad ogni salvataggio (auto/manuale)."""

    def __init__(self, items: list[dict], save_cb, *, is_video: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("caprev.title", "Revisione caption"))
        self.setModal(False)                                 # NON blocca la finestra sottostante
        self.setMinimumSize(720, 560)
        self._items = items
        self._save_cb = save_cb
        self._is_video = is_video
        self._edits: list[_CaptionEdit] = []
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        top = QLabel(tr("caprev.hint",
            "Rivedi e sovrascrivi le caption. Il salvataggio è automatico all'uscita dal campo, "
            "alla chiusura o con «Salva».").format())
        top.setWordWrap(True); top.setStyleSheet("color:#8a93a6; padding:2px;")
        v.addWidget(top)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); iv = QVBoxLayout(inner); iv.setSpacing(8)
        for i, it in enumerate(self._items):
            iv.addWidget(self._row(i, it))
        iv.addStretch()
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        bb.button(QDialogButtonBox.Save).setText(tr("caprev.save", "Salva"))
        bb.button(QDialogButtonBox.Close).setText(tr("caprev.close", "Chiudi"))
        bb.accepted.connect(self._save_all)                  # Salva (non chiude)
        bb.rejected.connect(self.close)                      # Chiudi (closeEvent salva)
        v.addWidget(bb)

    def _row(self, i: int, it: dict) -> QWidget:
        box = QFrame(); box.setFrameShape(QFrame.StyledPanel)
        h = QHBoxLayout(box); h.setContentsMargins(6, 6, 6, 6)
        # anteprima
        thumb = QLabel(); thumb.setFixedSize(140, 90); thumb.setAlignment(Qt.AlignCenter)
        thumb.setStyleSheet("background:#15161b; border-radius:4px;")
        path = it.get("path", "")
        ext = os.path.splitext(path)[1].lower()
        pm = None
        if ext in _IMG_EXTS:
            pm = QPixmap(path)
        elif self._is_video:
            tp = _video_thumb(path)
            if tp:
                pm = QPixmap(tp)
        if pm and not pm.isNull():
            thumb.setPixmap(pm.scaled(140, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumb.setText("🎬" if self._is_video else "🖼️"); thumb.setStyleSheet(
                thumb.styleSheet() + "font-size:28px; color:#5a6270;")
        h.addWidget(thumb)
        # nome + caption
        col = QVBoxLayout()
        name = QLabel(os.path.basename(path)); name.setStyleSheet("color:#cdd3df; font-weight:600;")
        name.setToolTip(path)
        col.addWidget(name)
        ed = _CaptionEdit(); ed.setPlainText(it.get("caption", "") or ""); ed.setFixedHeight(64)
        ed.setPlaceholderText(tr("caprev.ph", "caption / tag per questo elemento"))
        ed.committed.connect(lambda idx=i: self._save_one(idx))   # auto-save all'uscita dal campo
        self._edits.append(ed)
        col.addWidget(ed)
        h.addLayout(col, 1)
        return box

    def _save_one(self, i: int):
        if 0 <= i < len(self._edits):
            cap = self._edits[i].toPlainText().strip()
            self._items[i]["caption"] = cap
            try:
                self._save_cb(i, self._items[i].get("path", ""), cap)
            except Exception:                                # noqa: BLE001
                pass

    def _save_all(self):
        for i in range(len(self._edits)):
            self._save_one(i)

    def closeEvent(self, e):
        self._save_all()                                     # salva alla chiusura
        super().closeEvent(e)
