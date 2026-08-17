"""Pannello «Riferimenti multipli» per il tab Video di LTX Studio.

DIVERSO dallo storyboard: lì ogni immagine è un keyframe ancorato a un ISTANTE del clip
(`ImageConditioningInput(path, frame_idx, strength)`); qui i file sono RIFERIMENTI senza
posizione temporale — guidano identità, stile, movimento e voce.

Il modello li cita nel prompt con token strutturati (`<Picture 1>`, `<Video 1>`, `<Audio 1>`):
per questo ogni riga mostra il proprio token e sa inserirlo nel prompt (segnale
`insert_token`), invece di limitarsi a raccogliere file.

Limiti PER ARCHITETTURA (`set_limits`): MiniMax H3 = 9 immagini + 3 video + 3 audio, max 12
file, clip 2–15 s; LTX-2.5 con la LoRA Multiple-Subject-Reference = 2–5 immagini e basta.

API:
  get_references() -> list[{'path','kind','token','desc'}]   (vuota se disattivo)
  validation_errors() -> list[str]
  set_limits(images=…, videos=…, audios=…, total=…, min_items=…, clip_s=(min,max))
  insert_token (Signal[str]) — il tab lo collega alla casella del prompt

Indipendente: importabile e istanziabile senza il resto dell'app.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from ui.i18n_manager import tr

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# kind → (etichetta token nel prompt, icona)
_KIND_TOKEN = {"image": "Picture", "video": "Video", "audio": "Audio"}
_KIND_ICON = {"image": "🖼", "video": "🎞", "audio": "🎧"}


def kind_of(path: str) -> str | None:
    """Tipo di riferimento dall'estensione; None se non supportata."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def media_duration(path: str) -> float | None:
    """Durata in secondi via ffprobe; None se non determinabile (validazione allora si astiene)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=20)
        return float((r.stdout or "").strip())
    except Exception:                                        # noqa: BLE001
        return None


class _RefRow(QWidget):
    """Un riferimento: icona+token, nome file, descrizione breve, «inserisci nel prompt», rimuovi."""

    def __init__(self, path: str, kind: str, on_remove, on_insert, parent=None):
        super().__init__(parent)
        self.path = path
        self.kind = kind
        self.token = ""                                      # assegnato dal pannello (numerazione per tipo)
        h = QHBoxLayout(self); h.setContentsMargins(0, 2, 0, 2); h.setSpacing(6)
        self.tok_lbl = QLabel("—")
        self.tok_lbl.setMinimumWidth(92)
        self.tok_lbl.setStyleSheet("font-family:Consolas,monospace;color:#7fd1ff;")
        name = QLabel(f"{_KIND_ICON.get(kind, '')} {Path(path).name}")
        name.setMinimumWidth(150)
        name.setToolTip(path)
        name.setStyleSheet("color:#ddd;")
        self.desc = QLineEdit()
        self.desc.setPlaceholderText(tr("vid.ref.desc_ph", "descrizione (es. «la donna con il cappotto rosso»)"))
        self.desc.setToolTip(tr("vid.ref.desc.tt",
                                "Aiuta il modello a capire COSA prendere da questo riferimento. "
                                "Va scritta nel prompt insieme al token."))
        ins = self.ins_btn = QPushButton(tr("vid.ref.insert", "↳ prompt"))
        ins.setToolTip(tr("vid.ref.insert.tt", "Inserisce il token nel prompt, nel punto del cursore."))
        ins.clicked.connect(lambda: on_insert(self.token))
        rm = QPushButton("✕"); rm.setFixedWidth(28)
        rm.setToolTip(tr("vid.ref.remove", "Rimuovi il riferimento"))
        rm.clicked.connect(lambda: on_remove(self))
        h.addWidget(self.tok_lbl); h.addWidget(name, 1); h.addWidget(self.desc, 2)
        h.addWidget(ins); h.addWidget(rm)

    def set_token(self, token: str) -> None:
        self.token = token
        self.tok_lbl.setText(token)

    def value(self) -> dict:
        return {"path": self.path, "kind": self.kind, "token": self.token,
                "desc": self.desc.text().strip()}


class ReferencePanel(QWidget):
    """Lista di riferimenti tipizzati con token. Disattivo di default (checkbox)."""

    insert_token = Signal(str)
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # limiti di default = MiniMax H3 (Ref2VA)
        self._lim = {"image": 9, "video": 3, "audio": 3}
        self._total = 12
        self._min_items = 1
        self._clip_s = (2.0, 15.0)
        self._needs_background = False           # LTX/MSR sì, MiniMax H3 no
        self._rows: list[_RefRow] = []

        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(4)
        self.enable = QCheckBox(tr("vid.ref.enable", "Riferimenti multipli"))
        self.enable.setToolTip(tr("vid.ref.enable.tt",
                                  "Immagini, video e audio usati come RIFERIMENTO (identità, stile, "
                                  "movimento, voce) — non come fotogrammi in un istante preciso."))
        self.enable.toggled.connect(self._on_toggle)
        v.addWidget(self.enable)

        self.body = QWidget(); self.body.setVisible(False)
        bv = QVBoxLayout(self.body); bv.setContentsMargins(0, 0, 0, 0); bv.setSpacing(4)
        self.hint = QLabel()
        self.hint.setWordWrap(True); self.hint.setStyleSheet("color:#999;font-size:11px;")
        bv.addWidget(self.hint)

        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0); self._rows_box.setSpacing(0)
        rows_host = QWidget(); rows_host.setLayout(self._rows_box)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(220); scroll.setWidget(rows_host)
        bv.addWidget(scroll)

        row = QHBoxLayout()
        self.add_btn = QPushButton(tr("vid.ref.add", "＋ Aggiungi riferimento…"))
        self.add_btn.clicked.connect(self._choose_files)
        row.addWidget(self.add_btn)
        row.addStretch(1)
        self.count_lbl = QLabel()
        self.count_lbl.setStyleSheet("color:#8a93a6;font-size:11px;")
        row.addWidget(self.count_lbl)
        bv.addLayout(row)

        # SFONDO: su LTX (MSR pseudo-video) è OBBLIGATORIO — riempie i frame della sequenza di
        # riferimento non occupati dai soggetti. Su MiniMax H3 non serve: lì i riferimenti sono
        # token indipendenti.
        self.bg_row = QWidget()
        bgl = QHBoxLayout(self.bg_row); bgl.setContentsMargins(0, 2, 0, 0); bgl.setSpacing(6)
        self.bg_path: str = ""
        bg_btn = QPushButton(tr("vid.ref.bg_pick", "🖼 Sfondo…"))
        bg_btn.setToolTip(tr("vid.ref.bg.tt",
                             "Immagine di sfondo/scena: obbligatoria su LTX, riempie i fotogrammi "
                             "della sequenza di riferimento non occupati dai soggetti."))
        bg_btn.clicked.connect(self._choose_background)
        self.bg_lbl = QLabel(tr("vid.ref.bg_none", "nessuno (obbligatorio su LTX)"))
        self.bg_lbl.setStyleSheet("color:#c08a5a;")
        bgl.addWidget(bg_btn); bgl.addWidget(self.bg_lbl, 1)
        bv.addWidget(self.bg_row)
        v.addWidget(self.body)
        self._refresh()

    def _choose_background(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("vid.ref.bg_pick", "🖼 Sfondo…"), "",
                                           "Immagini (*" + " *".join(sorted(IMAGE_EXTS)) + ")")
        if f:
            self.set_background(f)

    def set_background(self, path: str) -> None:
        self.bg_path = path or ""
        self.bg_lbl.setText(Path(self.bg_path).name if self.bg_path
                            else tr("vid.ref.bg_none", "nessuno (obbligatorio su LTX)"))
        self.bg_lbl.setStyleSheet("color:#ddd;" if self.bg_path else "color:#c08a5a;")
        self.changed.emit()

    # ── configurazione per architettura ───────────────────────────────────────
    def set_limits(self, *, images: int = 9, videos: int = 3, audios: int = 3,
                   total: int = 12, min_items: int = 1,
                   clip_s: tuple[float, float] = (2.0, 15.0),
                   needs_background: bool = False) -> None:
        """Limiti dell'architettura attiva. `videos=0`/`audios=0` disabilitano quel tipo;
        `needs_background` mostra e richiede lo sfondo (serve alla pseudo-sequenza di LTX)."""
        self._lim = {"image": int(images), "video": int(videos), "audio": int(audios)}
        self._total = int(total)
        self._min_items = int(min_items)
        self._clip_s = clip_s
        self._needs_background = bool(needs_background)
        if hasattr(self, "bg_row"):
            self.bg_row.setVisible(self._needs_background)
        for r in [r for r in self._rows if self._lim.get(r.kind, 0) <= 0]:
            self._remove_row(r)                              # tipi non più ammessi: via
        self._refresh()

    # ── righe ─────────────────────────────────────────────────────────────────
    def _filters(self) -> str:
        parts = []
        if self._lim["image"] > 0:
            parts.append("*" + " *".join(sorted(IMAGE_EXTS)))
        if self._lim["video"] > 0:
            parts.append("*" + " *".join(sorted(VIDEO_EXTS)))
        if self._lim["audio"] > 0:
            parts.append("*" + " *".join(sorted(AUDIO_EXTS)))
        return tr("vid.ref.files", "Riferimenti") + " (" + " ".join(parts) + ")"

    def _choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, tr("vid.ref.add", "＋ Aggiungi riferimento…"),
                                                "", self._filters())
        for f in files:
            self.add_path(f)

    def add_path(self, path: str) -> bool:
        """Aggiunge un file se il tipo è ammesso e i limiti lo consentono. True se aggiunto."""
        kind = kind_of(path)
        if not kind or self._lim.get(kind, 0) <= 0:
            return False
        if len(self._rows) >= self._total or self._count(kind) >= self._lim[kind]:
            return False
        row = _RefRow(path, kind, self._remove_row, self.insert_token.emit)
        self._rows.append(row)
        self._rows_box.addWidget(row)
        self._refresh()
        return True

    def _remove_row(self, row: _RefRow):
        if row in self._rows:
            self._rows.remove(row)
            row.setParent(None)
            row.deleteLater()
            self._refresh()

    def _count(self, kind: str) -> int:
        return sum(1 for r in self._rows if r.kind == kind)

    def _refresh(self):
        """Rinumera i token per tipo (l'ordine è quello di inserimento) e aggiorna contatori/limiti."""
        seq = {"image": 0, "video": 0, "audio": 0}
        for r in self._rows:
            seq[r.kind] += 1
            r.set_token(f"<{_KIND_TOKEN[r.kind]} {seq[r.kind]}>")
        bits = [f"{seq[k]}/{self._lim[k]} {tr('vid.ref.kind.' + k, k)}"
                for k in ("image", "video", "audio") if self._lim[k] > 0]
        self.count_lbl.setText(" · ".join(bits) + f"  ({len(self._rows)}/{self._total})")
        self.add_btn.setEnabled(len(self._rows) < self._total)
        self.hint.setText(tr(
            "vid.ref.hint",
            "Ogni riferimento riceve un token: citalo nel prompt (bottone «↳ prompt») perché il "
            "modello sappia a quale file ti riferisci. I riferimenti NON sono fotogrammi: non "
            "hanno una posizione nel tempo."))
        self.changed.emit()

    def _on_toggle(self, on: bool):
        self.body.setVisible(on)
        self.changed.emit()

    # ── API ───────────────────────────────────────────────────────────────────
    def get_references(self) -> list[dict]:
        """Riferimenti + eventuale sfondo (marcato `role='background'`: il worker lo distingue
        dai soggetti, che non vanno mai ritagliati)."""
        if not self.enable.isChecked():
            return []
        out = [r.value() for r in self._rows]
        if self._needs_background and self.bg_path:
            out.append({"path": self.bg_path, "kind": "image", "token": "", "desc": "",
                        "role": "background"})
        return out

    def validation_errors(self) -> list[str]:
        if not self.enable.isChecked():
            return []
        refs = self.get_references()
        errs: list[str] = []
        if self._needs_background and not self.bg_path:
            errs.append(tr("vid.ref.err_bg",
                           "Serve un'immagine di sfondo: su LTX i riferimenti diventano una "
                           "sequenza e lo sfondo ne riempie i fotogrammi liberi."))
        if len(refs) < self._min_items:
            errs.append(tr("vid.ref.err_min", "Servono almeno {n} riferimenti.").format(n=self._min_items))
        for kind in ("image", "video", "audio"):
            n, lim = self._count(kind), self._lim[kind]
            if n > lim:
                errs.append(tr("vid.ref.err_max", "Troppi riferimenti di tipo {kind}: {n} (massimo {lim}).")
                            .format(kind=tr("vid.ref.kind." + kind, kind), n=n, lim=lim))
        if len(refs) > self._total:
            errs.append(tr("vid.ref.err_total", "Troppi riferimenti in totale: {n} (massimo {lim}).")
                        .format(n=len(refs), lim=self._total))
        lo, hi = self._clip_s
        for r in refs:
            if r["kind"] == "image":
                continue
            d = media_duration(r["path"])                    # None (ffprobe assente) → non blocco
            if d is not None and not (lo <= d <= hi):
                errs.append(tr("vid.ref.err_clip",
                               "{name}: dura {d:.1f}s, fuori dall'intervallo ammesso {lo:.0f}–{hi:.0f}s.")
                            .format(name=Path(r["path"]).name, d=d, lo=lo, hi=hi))
        return errs
