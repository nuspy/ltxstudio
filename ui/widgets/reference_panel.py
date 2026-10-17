"""Pannello «Riferimenti» (immagini, video, audio) condiviso da «Crea immagini» e dal tab Video di LTX Studio.

I file sono RIFERIMENTI senza posizione temporale (identità, stile, movimento, voce), diversi dai keyframe dello
storyboard (`ImageConditioningInput(path, frame_idx, strength)`).

NOMI: ogni file riceve AL CARICAMENTO un nome stabile — `Foto_1`, `Foto_2`… (video `Video_N`, audio `Audio_N`) — e nel
prompt si cita con `@Foto_1` (completamento digitando `@`, vedi `ui/widgets/ref_prompt.py`). Togliere Foto_2 non
rinumera le altre; una nuova immagine prende il primo numero libero (`core/image_refs.NameRegistry`). Alla generazione
`translate()` traduce i token nella sintassi del MODELLO scelto (`set_token_style`: qwen21 → «image 1», h3 →
`<Picture 1>`, ltx → «Image 1»…) e `get_references()` elenca i file nell'ordine in cui vanno passati al worker.
I prompt vecchi con `<Picture N>` si leggono e si traducono ancora; la GUI mostra e inserisce solo `@Foto_N`.

Limiti PER ARCHITETTURA (`set_limits`): MiniMax H3 = 9 immagini + 3 video + 3 audio, max 12 file, clip 2–15 s;
LTX con la LoRA Multiple-Subject-Reference = 2–5 immagini; Qwen-Image 2.1 = 10 immagini.

API:
  get_references() -> list[{'path','kind','token','name','at_token','desc'}]   (vuota se disattivo)
  validation_errors() -> list[str];  prompt_errors(prompt) -> list[str]   (token che citano riferimenti inesistenti)
  names() -> set[(kind,n)];  order() -> {kind: [n…]};  translate(prompt) -> str
  set_limits(images=…, videos=…, audios=…, total=…, min_items=…, clip_s=(min,max))
  insert_token (Signal[str]) — il tab lo collega alla casella del prompt (emette `@Foto_N`)
  request_canvas (Signal) — bottone «Dal canvas» (solo con `canvas_button=True`)

Indipendente: importabile e istanziabile senza il resto dell'app.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from core import image_refs as R
from ui.i18n_manager import tr

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Come il MODELLO cita il riferimento k-esimo (tabella con fonte in core/image_refs.MODEL_SYNTAX):
TOKEN_STYLES = {k: dict(v["fmt"]) for k, v in R.MODEL_SYNTAX.items()}
_KIND_ICON = {"image": "🖼", "video": "🎞", "audio": "🎧"}
_THUMB = 44


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
    """Un riferimento: miniatura, nome `@Foto_N`, file, descrizione breve, «inserisci nel prompt», rimuovi."""

    def __init__(self, path: str, kind: str, n: int, on_remove, on_insert, parent=None):
        super().__init__(parent)
        self.path = path
        self.kind = kind
        self.n = int(n)                                      # numero STABILE (assegnato al caricamento)
        self.token = ""                                      # token del MODELLO (posizionale), lo calcola il pannello
        h = QHBoxLayout(self); h.setContentsMargins(0, 2, 0, 2); h.setSpacing(6)
        self.thumb = QLabel()
        self.thumb.setFixedSize(_THUMB, _THUMB)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("background:#1c1f26;border:1px solid #333;border-radius:4px;")
        if kind == "image" and os.path.isfile(path):
            pm = QPixmap(path)
            if not pm.isNull():
                self.thumb.setPixmap(pm.scaled(QSize(_THUMB - 2, _THUMB - 2), Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))
        else:
            self.thumb.setText(_KIND_ICON.get(kind, ""))
        self.tok_lbl = QLabel(R.token_of(kind, self.n))
        self.tok_lbl.setMinimumWidth(80)
        self.tok_lbl.setStyleSheet("font-family:Consolas,monospace;color:#7fd1ff;font-weight:bold;")
        name = QLabel(Path(path).name)
        name.setMinimumWidth(120)
        name.setToolTip(path)
        name.setStyleSheet("color:#ddd;")
        self.desc = QLineEdit()
        self.desc.setPlaceholderText(tr("vid.ref.desc_ph", "descrizione (es. «la donna con il cappotto rosso»)"))
        self.desc.setToolTip(tr("vid.ref.desc.tt",
                                "Aiuta il modello a capire COSA prendere da questo riferimento. "
                                "Va scritta nel prompt insieme al token."))
        ins = self.ins_btn = QPushButton(tr("vid.ref.insert", "↳ prompt"))
        ins.setToolTip(tr("vid.ref.insert.tt", "Inserisce il token nel prompt, nel punto del cursore."))
        ins.clicked.connect(lambda: on_insert(self.at_token))
        rm = QPushButton("✕"); rm.setFixedWidth(28)
        rm.setToolTip(tr("vid.ref.remove", "Rimuovi il riferimento"))
        rm.clicked.connect(lambda: on_remove(self))
        h.addWidget(self.thumb); h.addWidget(self.tok_lbl); h.addWidget(name, 1); h.addWidget(self.desc, 2)
        h.addWidget(ins); h.addWidget(rm)

    @property
    def name(self) -> str:
        return R.name_of(self.kind, self.n)

    @property
    def at_token(self) -> str:
        return R.token_of(self.kind, self.n)

    def set_token(self, token: str) -> None:
        self.token = token

    def value(self) -> dict:
        return {"path": self.path, "kind": self.kind, "token": self.token, "name": self.name,
                "at_token": self.at_token, "desc": self.desc.text().strip()}


class ReferencePanel(QWidget):
    """Lista di riferimenti tipizzati con nome stabile. Con `checkable=True` (Video) è disattivo di default."""

    insert_token = Signal(str)
    changed = Signal()
    request_canvas = Signal()

    def __init__(self, parent=None, *, checkable: bool = True, canvas_button: bool = False):
        super().__init__(parent)
        # limiti di default = MiniMax H3 (Ref2VA)
        self._lim = {"image": 9, "video": 3, "audio": 3}
        self._total = 12
        self._min_items = 1
        self._clip_s = (2.0, 15.0)
        self._needs_background = False           # LTX/MSR sì, MiniMax H3 no
        self._token_style = "h3"                 # convenzione con cui il modello cita i riferimenti
        self._rows: list[_RefRow] = []
        self._names = R.NameRegistry()
        self._checkable = bool(checkable)
        self.setAcceptDrops(True)

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
        scroll.setMaximumHeight(240); scroll.setWidget(rows_host)
        bv.addWidget(scroll)

        row = QHBoxLayout()
        self.add_btn = QPushButton(tr("vid.ref.add", "＋ Aggiungi riferimento…"))
        self.add_btn.setToolTip(tr("ref.add.tt", "Scegli uno o più file (puoi anche trascinarli qui)."))
        self.add_btn.clicked.connect(self._choose_files)
        row.addWidget(self.add_btn)
        self.canvas_btn = QPushButton(tr("ref.from_canvas", "Dal canvas"))
        self.canvas_btn.setToolTip(tr("ref.from_canvas.tt", "Usa come riferimento l'immagine che hai nel canvas."))
        self.canvas_btn.clicked.connect(self.request_canvas.emit)
        self.canvas_btn.setVisible(bool(canvas_button))
        row.addWidget(self.canvas_btn)
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
        if not self._checkable:                  # «Crea immagini»: sempre attivo, i file caricati si usano
            self.enable.setChecked(True)
            self.enable.setVisible(False)
        self._refresh()

    # ── sfondo ────────────────────────────────────────────────────────────────
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
            self._remove_row(r)                              # tipi non più ammessi dall'arch: via
        # oltre il tetto per tipo (es. H3 9 immagini → LTX 4): scarta le eccedenze, dalla fine
        for kind, lim in self._lim.items():
            for r in [r for r in self._rows if r.kind == kind][lim:]:
                self._remove_row(r)
        self._refresh()

    def set_token_style(self, style: str) -> None:
        """Sintassi con cui il MODELLO cita il riferimento k-esimo (`core/image_refs.MODEL_SYNTAX`): 'h3'
        (`<Picture 1>`), 'ltx' (`Image 1`), 'qwen21'/'krea2_edit' (`image 1`). I nomi mostrati nel prompt restano
        `@Foto_N`: cambia solo la traduzione alla generazione."""
        if style in R.MODEL_SYNTAX and style != self._token_style:
            self._token_style = style
            self._refresh()
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

    def add_path(self, path: str, n: int | None = None) -> bool:
        """Aggiunge un file se il tipo è ammesso e i limiti lo consentono. True se aggiunto. Il nome (`Foto_N`) si
        assegna QUI, al caricamento: il primo numero libero (o `n`, se libero: ripristino dello stato salvato)."""
        kind = kind_of(path)
        if not kind or self._lim.get(kind, 0) <= 0:
            return False
        if len(self._rows) >= self._total or self._count(kind) >= self._lim[kind]:
            return False
        if n is not None and n >= 1 and n not in {r.n for r in self._rows if r.kind == kind}:
            self._names.reserve(kind, n)
        else:
            n = self._names.assign(kind)
        row = _RefRow(path, kind, n, self._remove_row, self.insert_token.emit)
        self._rows.append(row)
        self._rows.sort(key=lambda r: (R.KINDS.index(r.kind), r.n))   # ordine di invio = numero crescente
        self._rows_box.insertWidget(self._rows.index(row), row)
        self._refresh()
        return True

    def clear(self) -> None:
        """Toglie tutti i riferimenti (lo sfondo resta) e azzera la numerazione."""
        for row in list(self._rows):
            self._remove_row(row)

    def _remove_row(self, row: _RefRow):
        if row in self._rows:
            self._rows.remove(row)
            self._names.release(row.kind, row.n)             # niente rinumerazione delle altre
            row.setParent(None)
            row.deleteLater()
            self._refresh()

    def _count(self, kind: str) -> int:
        return sum(1 for r in self._rows if r.kind == kind)

    def _refresh(self):
        """Aggiorna il token del MODELLO (posizionale, per tipo) di ogni riga, i contatori e i limiti."""
        seq = {"image": 0, "video": 0, "audio": 0}
        fmt = R.MODEL_SYNTAX.get(self._token_style, R.MODEL_SYNTAX["h3"])["fmt"]
        for r in self._rows:
            seq[r.kind] += 1
            r.set_token(fmt.get(r.kind, "{n}").format(n=seq[r.kind]))
        bits = [f"{seq[k]}/{self._lim[k]} {tr('vid.ref.kind.' + k, k)}"
                for k in ("image", "video", "audio") if self._lim[k] > 0]
        self.count_lbl.setText(" · ".join(bits) + f"  ({len(self._rows)}/{self._total})")
        self.add_btn.setEnabled(len(self._rows) < self._total)
        self.hint.setText(tr(
            "ref.hint2",
            "Ogni file riceve un nome (@Foto_1, @Foto_2…): citalo nel prompt scrivendo «@» (completamento "
            "automatico) o con «↳ prompt». I nomi non cambiano se ne togli uno. I file caricati ma non citati "
            "vengono comunque passati al modello."))
        self.changed.emit()

    def _on_toggle(self, on: bool):
        self.body.setVisible(on)
        self.changed.emit()

    # ── drag & drop ───────────────────────────────────────────────────────────
    def _drop_paths(self, ev) -> list[str]:
        md = ev.mimeData()
        if not md.hasUrls():
            return []
        out = []
        for u in md.urls():
            p = u.toLocalFile()
            k = kind_of(p) if p else None
            if k and self._lim.get(k, 0) > 0:
                out.append(p)
        return out

    def dragEnterEvent(self, ev):                                   # noqa: N802
        if self._drop_paths(ev):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):                                    # noqa: N802
        self.dragEnterEvent(ev)

    def dropEvent(self, ev):                                        # noqa: N802
        paths = self._drop_paths(ev)
        if not paths:
            ev.ignore()
            return
        if not self.enable.isChecked():
            self.enable.setChecked(True)
        for p in paths:
            self.add_path(p)
        ev.acceptProposedAction()

    # ── nomi, ordine, traduzione ──────────────────────────────────────────────
    def names(self) -> set:
        """{(kind, n)} dei riferimenti esistenti (anche se il pannello è disattivo: servono al completamento)."""
        return {(r.kind, r.n) for r in self._rows}

    def order(self) -> dict:
        """{kind: [n…]} nell'ordine in cui i file vanno passati al worker (numero crescente)."""
        out: dict[str, list[int]] = {}
        for r in self._rows:
            out.setdefault(r.kind, []).append(r.n)
        return out

    def translate(self, prompt: str) -> str:
        """Prompt con `@Foto_N` (e i vecchi `<Picture N>`) tradotti nella sintassi del modello scelto."""
        return R.translate(prompt, self.order(), self._token_style)

    def prompt_errors(self, prompt: str) -> list[str]:
        """Token del prompt che citano un riferimento INESISTENTE (da correggere prima di generare)."""
        if not self.enable.isChecked():
            return []
        bad = R.unknown_tokens(prompt, self.names())
        seen, errs = set(), []
        for t in bad:
            tok = R.token_of(t.kind, t.n)
            if tok not in seen:
                seen.add(tok)
                errs.append(tr("ref.err_unknown", "Il prompt cita {tok}, ma quel riferimento non esiste.")
                            .format(tok=tok))
        return errs

    def uncited(self, prompt: str) -> list[str]:
        """Nomi dei riferimenti caricati e non citati (vengono passati comunque)."""
        return [R.token_of(k, n) for k, n in R.uncited(prompt, self.names())]

    # ── stato (gui_state) ─────────────────────────────────────────────────────
    def gui_get_state(self) -> dict:
        return {"enabled": bool(self.enable.isChecked()) if self._checkable else True,
                "rows": [{"path": r.path, "n": r.n, "desc": r.desc.text()} for r in self._rows],
                "bg": self.bg_path}

    def gui_set_state(self, st) -> None:
        if not isinstance(st, dict):
            return
        self.clear()
        for it in st.get("rows") or []:
            p = (it or {}).get("path") or ""
            if p and os.path.isfile(p) and self.add_path(p, (it or {}).get("n")):
                self._rows[[r.path for r in self._rows].index(p)].desc.setText(str(it.get("desc") or ""))
        if st.get("bg") and os.path.isfile(st["bg"]):
            self.set_background(st["bg"])
        if self._checkable:
            self.enable.setChecked(bool(st.get("enabled")))

    # ── API ───────────────────────────────────────────────────────────────────
    def get_references(self) -> list[dict]:
        """Riferimenti + eventuale sfondo (marcato `role='background'`: il worker lo distingue
        dai soggetti, che non vanno mai ritagliati). Ordine = ordine di invio al worker."""
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
