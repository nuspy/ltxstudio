"""Navigator Gallery — sfoglia, anteprima e riutilizza le generazioni precedenti.

Layout (QDialog 1200×700):
- Sinistra (240px): lista file con icona 🎬 per video e thumbnail reale per immagini
- Centro (420px):  anteprima ingrandita + prompt completo
- Destra (340px):  metadati in QFormLayout + azioni (Elimina, Riutilizza, Riproduci)

ReuseDialog: seleziona quali parametri ripristinare nella GUI principale.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon, QColor, QPainter, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget, QListWidgetItem,
    QLabel, QFormLayout, QWidget, QPushButton, QPlainTextEdit, QScrollArea,
    QMessageBox, QCheckBox, QDialogButtonBox, QGroupBox,
)

from core.metadata_saver import MetadataSaver
from ui.i18n_manager import tr

OUTPUT_DIR = Path(r"E:\AI\ltxstudio\output")

THUMB_W, THUMB_H = 140, 90


def _video_icon() -> QPixmap:
    """Icona generica 🎬 per i file video (nessuna estrazione frame)."""
    pix = QPixmap(THUMB_W, THUMB_H)
    pix.fill(QColor("#1a1a2e"))
    p = QPainter(pix)
    p.setPen(QColor("#cccccc"))
    font = QFont()
    font.setPointSize(28)
    p.setFont(font)
    p.drawText(pix.rect(), Qt.AlignCenter, "🎬")
    p.end()
    return pix


def _image_thumb(path: Path) -> QPixmap:
    """Carica e ridimensiona la thumbnail di un'immagine."""
    pix = QPixmap(str(path))
    if pix.isNull():
        return _video_icon()
    return pix.scaled(THUMB_W, THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ---------- ReuseDialog ----------

class ReuseDialog(QDialog):
    """Seleziona quali parametri del sidecar ripristinare nella GUI principale."""

    @property
    def _FIELDS(self):
        return [
            ("prompt_positive",  tr("nav.field_prompt_positive")),
            ("prompt_negative",  tr("nav.field_prompt_negative")),
            ("width",            tr("nav.field_width")),
            ("height",           tr("nav.field_height")),
            ("fps",              tr("nav.field_fps")),
            ("length_s",         tr("nav.field_length_s")),
            ("seed",             tr("nav.field_seed")),
            ("steps",            tr("nav.field_steps")),
            ("cfg",              tr("nav.field_cfg")),
            ("sampler",          tr("nav.field_sampler")),
            ("scheduler",        tr("nav.field_scheduler")),
            ("nsfw_mode",        tr("nav.field_nsfw_mode")),
            ("audio_mode",       tr("nav.field_audio_mode")),
            ("model",            tr("nav.field_model")),
            ("loras",            tr("nav.field_loras")),
        ]

    def __init__(self, meta: dict, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.setWindowTitle(tr("nav.reuse_title"))
        self.setMinimumWidth(440)
        self._checks: dict[str, QCheckBox] = {}
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel(tr("nav.reuse_select_params")))

        grp = QGroupBox(tr("nav.reuse_available_params"))
        gl = QVBoxLayout(grp)
        for key, label in self._FIELDS:
            val = self.meta.get(key)
            if val is None or val == "" or val == []:
                continue
            cb = QCheckBox(f"{label}: {self._fmt(val)}")
            cb.setChecked(True)
            self._checks[key] = cb
            gl.addWidget(cb)
        v.addWidget(grp)

        h = QHBoxLayout()
        b_all = QPushButton(tr("nav.select_all"))
        b_all.clicked.connect(lambda: [c.setChecked(True) for c in self._checks.values()])
        b_none = QPushButton(tr("nav.deselect_all"))
        b_none.clicked.connect(lambda: [c.setChecked(False) for c in self._checks.values()])
        h.addWidget(b_all)
        h.addWidget(b_none)
        h.addStretch()
        v.addLayout(h)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("nav.set_params"))
        bb.button(QDialogButtonBox.Cancel).setText(tr("nav.cancel"))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    @staticmethod
    def _fmt(val) -> str:
        if isinstance(val, float):
            return f"{val:.3g}"
        if isinstance(val, list):
            if not val:
                return tr("nav.none_fem")
            names = [d.get("name", str(d)) if isinstance(d, dict) else str(d) for d in val[:3]]
            suffix = f" +{len(val) - 3}" if len(val) > 3 else ""
            return ", ".join(names) + suffix
        return str(val)

    def selected_fields(self) -> dict:
        """Ritorna solo i campi selezionati con il loro valore dal sidecar."""
        return {k: self.meta[k] for k, cb in self._checks.items()
                if cb.isChecked() and k in self.meta}


# ---------- NavigatorDialog ----------

class NavigatorDialog(QDialog):
    """Gallery Navigator: lista generazioni con sidecar, anteprima e riutilizzo."""

    def __init__(self, output_dir: Path = OUTPUT_DIR, apply_callback=None, parent=None):
        """
        Args:
            output_dir:      cartella output da esplorare
            apply_callback:  callable(fields: dict) → chiamato quando l'utente clicca Imposta
            parent:          QWidget genitore
        """
        super().__init__(parent)
        self.output_dir = output_dir
        self.apply_callback = apply_callback
        self._current_meta: dict | None = None
        self._current_path: Path | None = None
        self.setWindowTitle(tr("nav.window_title"))
        self.resize(1200, 700)
        self._build()
        self._load_list()

    # ---------- build ----------
    def _build(self):
        main_v = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)

        # --- Sinistra: lista file ---
        left_w = QWidget()
        lv = QVBoxLayout(left_w)
        lv.setContentsMargins(0, 0, 4, 0)

        hdr_lbl = QLabel(tr("nav.generated_files"))
        hdr_lbl.setStyleSheet("font-weight:bold; padding:4px;")
        lv.addWidget(hdr_lbl)

        self.file_list = QListWidget()
        self.file_list.setIconSize(QSize(THUMB_W, THUMB_H))
        self.file_list.setSpacing(4)
        self.file_list.currentItemChanged.connect(self._on_select)
        lv.addWidget(self.file_list, 1)

        btn_refresh = QPushButton(tr("nav.refresh_list"))
        btn_refresh.clicked.connect(self._load_list)
        lv.addWidget(btn_refresh)

        left_w.setMinimumWidth(200)
        left_w.setMaximumWidth(280)
        split.addWidget(left_w)

        # --- Centro: anteprima + prompt ---
        center_w = QWidget()
        cv = QVBoxLayout(center_w)
        cv.setContentsMargins(4, 0, 4, 0)

        self.preview_lbl = QLabel(tr("nav.select_file_from_list"))
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        self.preview_lbl.setMinimumSize(380, 260)
        self.preview_lbl.setStyleSheet("background:#111; color:#666; border:1px solid #333;")
        cv.addWidget(self.preview_lbl, 2)

        lbl_pos = QLabel(tr("nav.prompt_positive_label"))
        lbl_pos.setStyleSheet("font-weight:bold; margin-top:4px;")
        cv.addWidget(lbl_pos)
        self.prompt_text = QPlainTextEdit()
        self.prompt_text.setReadOnly(True)
        self.prompt_text.setMaximumHeight(110)
        cv.addWidget(self.prompt_text)

        lbl_neg = QLabel(tr("nav.prompt_negative_label"))
        lbl_neg.setStyleSheet("font-weight:bold; margin-top:4px;")
        cv.addWidget(lbl_neg)
        self.neg_text = QPlainTextEdit()
        self.neg_text.setReadOnly(True)
        self.neg_text.setMaximumHeight(55)
        cv.addWidget(self.neg_text)

        split.addWidget(center_w)

        # --- Destra: metadati + azioni ---
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(4, 0, 0, 0)

        meta_hdr = QLabel(tr("nav.generation_metadata"))
        meta_hdr.setStyleSheet("font-weight:bold;")
        rv.addWidget(meta_hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        meta_container = QWidget()
        self.meta_form = QFormLayout(meta_container)
        self.meta_form.setLabelAlignment(Qt.AlignRight)
        scroll.setWidget(meta_container)
        rv.addWidget(scroll, 1)

        act_hdr = QLabel(tr("nav.actions"))
        act_hdr.setStyleSheet("font-weight:bold; margin-top:6px;")
        rv.addWidget(act_hdr)

        self.btn_reuse = QPushButton(tr("nav.reuse_params_btn"))
        self.btn_reuse.setEnabled(False)
        self.btn_reuse.clicked.connect(self._on_reuse)
        rv.addWidget(self.btn_reuse)

        self.btn_play = QPushButton(tr("nav.open_play_btn"))
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._on_play)
        rv.addWidget(self.btn_play)

        self.btn_delete = QPushButton(tr("nav.delete_file_btn"))
        self.btn_delete.setEnabled(False)
        self.btn_delete.setStyleSheet("color: #ff6b6b;")
        self.btn_delete.clicked.connect(self._on_delete)
        rv.addWidget(self.btn_delete)

        right_w.setMinimumWidth(260)
        right_w.setMaximumWidth(380)
        split.addWidget(right_w)

        split.setSizes([240, 580, 380])
        main_v.addWidget(split, 1)

        # Riga pulsante Chiudi
        close_btn = QPushButton(tr("nav.close"))
        close_btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_btn)
        main_v.addLayout(row)

    # ---------- lista ----------
    def _load_list(self):
        self.file_list.clear()
        files = MetadataSaver.list_outputs_with_sidecar(self.output_dir)
        if not files:
            placeholder = QListWidgetItem(tr("nav.no_files_with_metadata"))
            self.file_list.addItem(placeholder)
            return
        for f in files:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, f)
            if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov", ".gif"):
                pix = _video_icon()
            else:
                pix = _image_thumb(f)
            it.setIcon(QIcon(pix))
            dt = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            name_short = f.name if len(f.name) <= 28 else f.name[:25] + "…"
            it.setText(f"{dt:%d/%m %H:%M}\n{name_short}")
            it.setSizeHint(QSize(THUMB_W + 20, THUMB_H + 30))
            self.file_list.addItem(it)

    # ---------- selezione ----------
    def _on_select(self, cur: QListWidgetItem, _prev):
        if cur is None:
            return
        path = cur.data(Qt.UserRole)
        if not isinstance(path, Path):
            return
        self._current_path = path
        meta = MetadataSaver.load_sidecar(path)
        self._current_meta = meta or {}
        self._fill_preview(path, meta)
        self._fill_meta(meta or {})
        self.btn_reuse.setEnabled(bool(meta))
        self.btn_play.setEnabled(True)
        self.btn_delete.setEnabled(True)

    def _fill_preview(self, path: Path, meta: dict | None):
        if path.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov", ".gif"):
            self.preview_lbl.setPixmap(
                _video_icon().scaled(self.preview_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            pix = QPixmap(str(path))
            if pix.isNull():
                self.preview_lbl.setText(tr("nav.preview_unavailable"))
            else:
                self.preview_lbl.setPixmap(
                    pix.scaled(self.preview_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.prompt_text.setPlainText((meta or {}).get("prompt_positive", ""))
        self.neg_text.setPlainText((meta or {}).get("prompt_negative", ""))

    def _fill_meta(self, meta: dict):
        while self.meta_form.rowCount():
            self.meta_form.removeRow(0)
        if not meta:
            self.meta_form.addRow(QLabel(tr("nav.no_metadata_available")))
            return
        _LABELS = [
            ("generated_at",       tr("nav.meta_generated_at")),
            ("generation_type",    tr("nav.meta_type")),
            ("template",           tr("nav.meta_template")),
            ("model",              tr("nav.meta_model")),
            ("width",              tr("nav.meta_width")),
            ("height",             tr("nav.meta_height")),
            ("fps",                tr("nav.meta_fps")),
            ("length_s",           tr("nav.meta_length_s")),
            ("seed",               tr("nav.meta_seed")),
            ("steps",              tr("nav.meta_steps")),
            ("cfg",                tr("nav.meta_cfg")),
            ("sampler",            tr("nav.meta_sampler")),
            ("scheduler",          tr("nav.meta_scheduler")),
            ("nsfw_mode",          tr("nav.meta_nsfw")),
            ("audio_mode",         tr("nav.meta_audio")),
            ("controlnet_enabled", tr("nav.meta_controlnet")),
            ("loras",              tr("nav.meta_loras")),
        ]
        _SKIP = {"ltxstudio_version", "output_file", "prompt_positive", "prompt_negative",
                 "frame1_image", "controlnet_ref_video", "controlnet_type", "model_kind", "prompt_id"}
        for key, lbl_text in _LABELS:
            if key in _SKIP or key not in meta:
                continue
            val = meta[key]
            if val is None or val == "":
                continue
            if isinstance(val, list):
                if not val:
                    val_str = tr("nav.none_fem")
                else:
                    lines = []
                    for d in val:
                        if isinstance(d, dict):
                            lines.append(f"• {d.get('name', '?')} ×{d.get('strength', 1.0):.2f}")
                        else:
                            lines.append(f"• {d}")
                    val_str = "\n".join(lines)
            elif isinstance(val, float):
                val_str = f"{val:.3g}"
            elif isinstance(val, bool):
                val_str = tr("nav.yes") if val else tr("nav.no")
            else:
                val_str = str(val)
            lbl_w = QLabel(val_str)
            lbl_w.setWordWrap(True)
            self.meta_form.addRow(f"{lbl_text}:", lbl_w)

    # ---------- azioni ----------
    def _on_reuse(self):
        if not self._current_meta:
            return
        if not self.apply_callback:
            QMessageBox.information(self, tr("nav.unavailable_title"),
                                    tr("nav.unavailable_msg"))
            return
        dlg = ReuseDialog(self._current_meta, self)
        if dlg.exec():
            fields = dlg.selected_fields()
            if fields:
                self.apply_callback(fields)
                self._log_action(f"Parametri ripristinati: {', '.join(fields.keys())}")
                QMessageBox.information(self, tr("nav.done_title"),
                                        tr("nav.done_msg").format(n=len(fields)))
            else:
                QMessageBox.information(self, tr("nav.nothing_title"), tr("nav.nothing_msg"))

    def _on_play(self):
        if not self._current_path:
            return
        path = self._current_path
        if path.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov"):
            mw = self.parent()
            if mw and hasattr(mw, "video_player"):
                if mw.video_player is None:
                    from ui.video_player_dialog import VideoPlayerDialog
                    mw.video_player = VideoPlayerDialog(mw)
                mw.video_player.play_file(str(path))
                return
        # Fallback: apri con il programma di sistema
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _on_delete(self):
        if not self._current_path:
            return
        path = self._current_path
        resp = QMessageBox.question(
            self, tr("nav.confirm_delete_title"),
            tr("nav.confirm_delete_msg").format(name=path.name),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        try:
            sidecar = MetadataSaver.sidecar_path(path)
            if path.exists():
                path.unlink()
            if sidecar.exists():
                sidecar.unlink()
        except Exception as e:
            QMessageBox.warning(self, tr("nav.delete_error_title"), str(e))
            return
        self._current_path = None
        self._current_meta = None
        self.btn_reuse.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.preview_lbl.setText(tr("nav.file_deleted"))
        self.prompt_text.clear()
        self.neg_text.clear()
        while self.meta_form.rowCount():
            self.meta_form.removeRow(0)
        self._load_list()

    def _log_action(self, msg: str):
        mw = self.parent()
        if mw and hasattr(mw, "_log"):
            mw._log(f"[Navigator] {msg}")
