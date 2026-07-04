"""Tab «Mondo 3D» — genera una scena/ambiente 3D da una foto (o da testo) con WorldGen.

Output: MESH (GLB, esportabile FBX/OBJ via Blender) e/o SPLAT (Gaussian .ply).
Player: visore nativo in-app (QWebEngine + assets/world_player) e player web autonomo
(server http locale + browser), entrambi con navigazione orbit/WASD.

Motore standalone (core/world_service.py): venv isolato + subprocess; i callback (thread
secondario) sono inoltrati alla UI via Qt Signal.
"""
from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt, QObject, Signal, QProcess, QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QPushButton, QLineEdit, QCheckBox,
    QFileDialog, QPlainTextEdit, QProgressBar, QMessageBox,
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings
    _HAS_WEB = True
except Exception:                                       # pragma: no cover
    _HAS_WEB = False

from ui.i18n_manager import tr
from core import world_service as ws


class _Bridge(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(int, bool)        # (returncode, was_mesh)


class WorldTab(QWidget):
    def __init__(self):
        super().__init__()
        self._job: ws.Job | None = None
        self._install_proc: QProcess | None = None
        self._http_proc: QProcess | None = None
        self._pending_splat = False        # per la modalità "entrambi"
        self._cur: ws.WorldParams | None = None
        self._b = _Bridge()
        self._b.log.connect(self._log); self._b.progress.connect(self._on_progress); self._b.done.connect(self._on_done)

        root = QHBoxLayout(self)
        left = QVBoxLayout(); root.addLayout(left, 0)
        left.addWidget(self._input_box())
        left.addWidget(self._actions_box())
        left.addWidget(self._progress_box())
        left.addStretch(1)
        root.addWidget(self._viewer_box(), 1)
        self._refresh_install_state()

    # ── input ───────────────────────────────────────────────────────────────
    def _input_box(self) -> QGroupBox:
        box = QGroupBox(tr("world.input.title", "Scena da foto / testo")); f = QFormLayout(box)
        self.name = QLineEdit("world"); f.addRow(tr("world.name", "Nome"), self.name)
        row = QHBoxLayout()
        self.image = QLineEdit(); self.image.setPlaceholderText(tr("world.image.hint", "foto di partenza (vuoto = solo testo)"))
        b = QPushButton(tr("world.browse", "Sfoglia…")); b.clicked.connect(self._pick_image)
        row.addWidget(self.image, 1); row.addWidget(b)
        w = QWidget(); w.setLayout(row); f.addRow(tr("world.image", "Immagine"), w)
        self.prompt = QLineEdit(); self.prompt.setPlaceholderText(tr("world.prompt.hint", "descrizione/integrazione della scena"))
        f.addRow(tr("world.prompt", "Prompt"), self.prompt)
        self.mode = QComboBox()
        self.mode.addItem(tr("world.mode.mesh", "Mesh (GLB, esportabile)"), "mesh")
        self.mode.addItem(tr("world.mode.splat", "Splat (fotorealistico)"), "splat")
        self.mode.addItem(tr("world.mode.both", "Entrambi"), "both")
        f.addRow(tr("world.mode", "Output"), self.mode)
        self.res = QSpinBox(); self.res.setRange(512, 4096); self.res.setSingleStep(128); self.res.setValue(1600)
        f.addRow(tr("world.res", "Risoluzione"), self.res)
        self.low_vram = QCheckBox(tr("world.lowvram", "Low VRAM")); f.addRow("", self.low_vram)
        self.hf = QLineEdit(); self.hf.setEchoMode(QLineEdit.Password)
        self.hf.setPlaceholderText(tr("world.hf.hint", "HF token (FLUX.1-dev gated)"))
        f.addRow(tr("world.hf", "HF token"), self.hf)
        return box

    def _actions_box(self) -> QGroupBox:
        box = QGroupBox(tr("world.actions", "Azioni")); v = QVBoxLayout(box)
        h1 = QHBoxLayout()
        self.btn_install = QPushButton(tr("world.install", "Installa modulo"))
        self.btn_install.clicked.connect(self._install)
        self.btn_gen = QPushButton(tr("world.gen", "Genera scena")); self.btn_gen.clicked.connect(self._gen)
        self.btn_stop = QPushButton(tr("world.stop", "Stop")); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._stop)
        h1.addWidget(self.btn_install); h1.addStretch(1); h1.addWidget(self.btn_gen); h1.addWidget(self.btn_stop)
        v.addLayout(h1)
        h2 = QHBoxLayout()
        self.btn_fbx = QPushButton(tr("world.export.fbx", "Esporta FBX")); self.btn_fbx.clicked.connect(lambda: self._export("fbx"))
        self.btn_obj = QPushButton(tr("world.export.obj", "Esporta OBJ")); self.btn_obj.clicked.connect(lambda: self._export("obj"))
        self.btn_web = QPushButton(tr("world.web", "Player web")); self.btn_web.clicked.connect(self._open_web_player)
        self.btn_folder = QPushButton(tr("world.folder", "Apri cartella")); self.btn_folder.clicked.connect(self._open_folder)
        for b in (self.btn_fbx, self.btn_obj, self.btn_web, self.btn_folder):
            h2.addWidget(b)
        v.addLayout(h2)
        return box

    def _progress_box(self) -> QGroupBox:
        box = QGroupBox(tr("world.progress", "Progresso")); v = QVBoxLayout(box)
        self.bar = QProgressBar(); self.status = QLabel("—"); self.status.setStyleSheet("color:#9aa1b0;")
        self.logview = QPlainTextEdit(); self.logview.setReadOnly(True); self.logview.setFixedHeight(150)
        self.logview.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        v.addWidget(self.bar); v.addWidget(self.status); v.addWidget(self.logview)
        return box

    def _viewer_box(self) -> QWidget:
        box = QGroupBox(tr("world.viewer", "Visore (orbita / 🚶 WASD)")); v = QVBoxLayout(box)
        if _HAS_WEB:
            self.view = QWebEngineView()
            s = self.view.settings()
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            s.setAttribute(QWebEngineSettings.WebGLEnabled, True)
            v.addWidget(self.view, 1)
        else:
            self.view = None
            lbl = QLabel(tr("world.noweb", "QtWebEngine non disponibile: usa il Player web."))
            lbl.setAlignment(Qt.AlignCenter); lbl.setWordWrap(True); v.addWidget(lbl, 1)
        return box

    # ── slot ──────────────────────────────────────────────────────────────────
    def _pick_image(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("world.image", "Immagine"), "", "Image (*.png *.jpg *.jpeg *.webp)")
        if p:
            self.image.setText(p)

    def _params(self) -> ws.WorldParams:
        return ws.WorldParams(
            name=self.name.text().strip() or "world", image_path=self.image.text().strip(),
            prompt=self.prompt.text().strip(), resolution=self.res.value(),
            low_vram=self.low_vram.isChecked())

    def _refresh_install_state(self):
        ok = ws.is_installed()
        self.btn_install.setEnabled(not ok); self.btn_gen.setEnabled(ok)
        self.btn_install.setText(tr("world.installed", "Modulo installato ✓") if ok else tr("world.install", "Installa modulo"))

    def _gen(self):
        if self._job and self._job.running:
            return
        # #96.5: se l'immagine non è impostata, usa quella del canvas (tab Immagine)
        if not self.image.text().strip():
            getter = getattr(self, "get_canvas_image", None)
            canvas_img = (getter() if callable(getter) else "") or ""
            if canvas_img:
                self.image.setText(canvas_img)
                self._log(f"[mondo] uso l'immagine del canvas: {canvas_img}")
            elif not self.prompt.text().strip():
                QMessageBox.warning(
                    self, tr("world.no_input_title", "Serve un'immagine o un prompt"),
                    tr("world.no_input_msg",
                       "Nessuna immagine impostata e il canvas (tab Immagine) è vuoto.\n\n"
                       "Imposta un'immagine di partenza, oppure carica/genera un'immagine nel canvas, "
                       "oppure scrivi un prompt testuale per generare la scena da testo."))
                return
        p = self._params(); errs = ws.preflight(p)
        if errs:
            QMessageBox.warning(self, tr("world.err", "Impossibile generare"), "\n".join(errs)); return
        self._cur = p
        mode = self.mode.currentData()
        self._pending_splat = (mode == "both")
        first_mesh = (mode in ("mesh", "both"))
        self.logview.clear(); self.bar.setValue(0)
        self._launch(p, mesh=first_mesh)

    def _launch(self, p: ws.WorldParams, mesh: bool):
        job = ws.make_job(p, mesh=mesh, hf_token=self.hf.text().strip() or None,
                          on_log=self._b.log.emit, on_progress=self._b.progress.emit,
                          on_done=lambda rc, m=mesh: self._b.done.emit(rc, m))
        self._job = job
        try:
            job.start()
        except Exception as e:                          # noqa: BLE001
            QMessageBox.critical(self, tr("world.err", "Errore"), str(e)); return
        self.btn_gen.setEnabled(False); self.btn_stop.setEnabled(True)
        self.status.setText(tr("world.generating", "Generazione ({m})…").format(m="mesh" if mesh else "splat"))
        self._log(f"[world] avvio ({'mesh' if mesh else 'splat'})")

    def _stop(self):
        if self._job:
            self._job.stop()
        self._pending_splat = False; self.btn_stop.setEnabled(False)

    def _install(self):
        if self._install_proc is not None:
            return
        if QMessageBox.question(self, tr("world.install", "Installa modulo"),
                                tr("world.install.confirm", "Scarico WorldGen + PyTorch (cu128). I pesi (incl. FLUX.1-dev) al primo run. Procedo?")) != QMessageBox.Yes:
            return
        script = str(Path(__file__).resolve().parents[2] / "scripts" / "install_world.py")
        self.btn_install.setEnabled(False); self._log("[install] avvio…")
        p = QProcess(self); p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self._log(bytes(p.readAllStandardOutput()).decode("utf-8", "replace").rstrip()))
        p.finished.connect(self._install_done)
        self._install_proc = p; p.start(sys.executable, ["-u", script])

    def _install_done(self, code, _s):
        self._log(f"[install] terminato (codice {code})"); self._install_proc = None; self._refresh_install_state()

    def _export(self, fmt: str):
        if not self._cur:
            QMessageBox.information(self, tr("world.export", "Esporta"), tr("world.export.none", "Genera prima una scena mesh.")); return
        glb = self._cur.glb()
        if not glb.exists():
            QMessageBox.information(self, tr("world.export", "Esporta"), tr("world.export.nomesh", "Nessun GLB: genera in modalità Mesh.")); return
        self._log(f"[export] {fmt} via Blender…")
        out, err = ws.export_mesh(glb, fmt)
        if out:
            self._log(f"[export] creato: {out}")
            QMessageBox.information(self, tr("world.export", "Esporta"), tr("world.export.ok", "Creato: {f}").format(f=out))
        else:
            self._log(f"[export] errore: {err}")
            QMessageBox.warning(self, tr("world.export", "Esporta"), err)

    def _load_in_viewer(self, file: Path):
        if not self.view:
            return
        player = ws.PLAYER_DIR / "index.html"
        typ = "splat" if file.suffix.lower() == ".ply" else "glb"
        url = QUrl.fromLocalFile(str(player))
        url.setQuery(f"src={QUrl.fromLocalFile(str(file)).toString()}&type={typ}")
        self.view.load(url)

    def _open_web_player(self):
        if not self._cur:
            QMessageBox.information(self, tr("world.web", "Player web"), tr("world.export.none", "Genera prima una scena.")); return
        proj = self._cur.project_dir()
        # copia il player accanto all'output, così la cartella è autonoma/distribuibile
        import shutil
        try:
            shutil.copy2(ws.PLAYER_DIR / "index.html", proj / "index.html")
        except Exception as e:                          # noqa: BLE001
            self._log(f"[web] copia player fallita: {e}")
        # avvia un server http locale nella cartella (file:// blocca i fetch nel browser)
        proj_str = str(proj)
        if self._http_proc is not None and getattr(self, "_http_dir", None) != proj_str:
            try:                                            # #95: progetto cambiato → riavvia il server sulla nuova cartella
                self._http_proc.kill(); self._http_proc.waitForFinished(2000)
            except Exception:                               # noqa: BLE001
                pass
            self._http_proc = None
        if self._http_proc is None:
            self._http_proc = QProcess(self)
            self._http_proc.start(sys.executable, ["-m", "http.server", "8765", "--directory", proj_str])
            self._http_dir = proj_str
        target = self._cur.glb() if self._cur.glb().exists() else self._cur.splat()
        typ = "splat" if target.suffix.lower() == ".ply" else "glb"
        webbrowser.open(f"http://localhost:8765/index.html?src={target.name}&type={typ}")

    def _open_folder(self):
        ws.WORLD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self._cur.project_dir() if self._cur else ws.WORLD_OUTPUT_DIR))

    def _log(self, line: str):
        if not line:
            return
        self.logview.appendPlainText(line); sb = self.logview.verticalScrollBar(); sb.setValue(sb.maximum())

    def _on_progress(self, cur, total):
        self.bar.setRange(0, total); self.bar.setValue(cur); self.status.setText(f"{cur}/{total}")

    def _on_done(self, code: int, was_mesh: bool):
        if code == 0 and self._pending_splat and was_mesh:
            self._pending_splat = False
            self._log("[world] mesh ok → genero lo splat…")
            self._launch(self._cur, mesh=False); return
        self._job = None; self.btn_gen.setEnabled(ws.is_installed()); self.btn_stop.setEnabled(False)
        if code != 0:
            self.status.setText(tr("world.failed", "Generazione fallita (codice {c})").format(c=code)); return
        self.status.setText(tr("world.done", "Scena pronta ✓"))
        if self._cur:
            tgt = self._cur.glb() if self._cur.glb().exists() else self._cur.splat()
            if tgt.exists():
                self._log(f"[world] output: {tgt}")
                self._load_in_viewer(tgt)
