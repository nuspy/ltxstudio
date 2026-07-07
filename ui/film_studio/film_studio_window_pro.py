"""AI Film Studio — variante «Pro» (switchabile da menu Interfaccia).

Sottoclasse di `FilmStudioWindow`: riusa tutta la logica e i tab e aggiunge i miglioramenti UX/UI pro-max a
livello finestra. La V1 (Classica) resta intatta.

Migliorie:
- FS1: barra progetto **collassabile** (riduce la densità su finestre strette).
- FS2: **banner-flusso** guidato (Copione → Scenografia → Montaggio → Master) sempre visibile.
- FS3: **vista coda** Job Scheduler in basso (in corso / in attesa + barra attività).
- FS4: **gating coerente** — se `cap.filmstudio` è OFF, i tab lasciano il posto a un UpsellPanel (in dev
  fail-open = tutto visibile).
- FS5: **conferma prima del Master** (render lungo).

NB: gli empty-state profondi dei singoli tab e alcune rifiniture per-tab restano possibili estensioni future
(richiederebbero varianti Pro dei tab figli); qui il banner-flusso copre la guida a livello finestra.
"""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QVBoxLayout, QWidget,
)

from core.film_studio import scheduler
from ui.i18n_manager import tr
from ui.film_studio.film_studio_window import FilmStudioWindow


class FilmStudioWindowPro(FilmStudioWindow):
    # ---- tab figli in variante Pro (empty-state guidati + rifiniture) ----
    def _make_script_tab(self):
        from ui.film_studio.pro_tabs import ScriptTabPro
        return ScriptTabPro(self._get_project)

    def _make_asset_tab(self):
        from ui.film_studio.pro_tabs import AssetManagerTabPro
        return AssetManagerTabPro()

    def _make_sb_tab(self):
        from ui.film_studio.pro_tabs import StoryboardTabPro
        return StoryboardTabPro(self._get_project)

    def _make_mt_tab(self):
        from ui.film_studio.pro_tabs import MontageTabPro
        return MontageTabPro(self._get_project)

    def _make_voice_tab(self):
        from ui.film_studio.pro_tabs import VoiceTabPro
        return VoiceTabPro()

    def _make_lora_tab(self):
        from ui.film_studio.pro_tabs import LoraStudioTabPro
        return LoraStudioTabPro()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("fs.title.pro", "🎬 AI Film Studio — Pro ✨"))
        root = self.layout()                                 # [project_bar, LlmSettings, tabs] (Pro: NO banner)
        # RIALLINEAMENTO: il banner-guida al flusso ora sta nella CLASSICA (principianti); la Pro è lean
        # (_flow_banner→None). Prima la guida stava, invertita, nella Pro.
        # FS1 — barra progetto collassabile (project_bar è all'indice 0, niente banner sopra)
        pb = root.itemAt(0).widget()
        grp = QGroupBox(tr("fs.pro.projbar", "⚙ Impostazioni progetto"))
        grp.setCheckable(True); grp.setChecked(True)
        gl = QVBoxLayout(grp); gl.setContentsMargins(6, 4, 6, 4)
        root.removeWidget(pb); pb.setParent(grp); gl.addWidget(pb)
        grp.toggled.connect(pb.setVisible)
        root.insertWidget(1, grp)
        # FS3 — vista coda in fondo
        root.addWidget(self._queue_panel())
        # FS4 — gating coerente
        self._apply_filmstudio_gate()
        # FS5 — conferma prima del Master
        self._wrap_master_confirm()

    # ---------- riallineamento: Pro lean (la guida al flusso sta nella Classica) ----------
    def _flow_banner(self):
        return None                                          # nessun banner-guida in Pro (utenti esperti)

    # ---------- FS3 vista coda scheduler ----------
    def _queue_panel(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(6, 2, 6, 2)
        h.addWidget(QLabel(tr("fs.pro.queue", "🗓 Job:")))
        self._q_lbl = QLabel(tr("fs.pro.queue.idle", "coda vuota"))
        self._q_lbl.setStyleSheet("color:#9aa0b4;")
        self._q_bar = QProgressBar(); self._q_bar.setMaximumWidth(160); self._q_bar.setRange(0, 0)
        self._q_bar.setVisible(False)
        h.addWidget(self._q_lbl); h.addWidget(self._q_bar); h.addStretch(1)
        self._q_timer = QTimer(self)
        self._q_timer.timeout.connect(self._refresh_queue)
        self._q_timer.start(1500)                            # poll GUI-thread (niente callback cross-thread)
        return w

    def _refresh_queue(self):
        try:
            jobs = scheduler.pending()
        except Exception:                                    # noqa: BLE001
            jobs = []
        running = sum(1 for j in jobs if j.get("status") == "running")
        waiting = sum(1 for j in jobs if j.get("status") in ("pending", "retry"))
        self._q_bar.setVisible(bool(running))
        if running or waiting:
            self._q_lbl.setText(tr("fs.pro.queue.busy",
                "{r} in corso · {w} in attesa").format(r=running, w=waiting))
        else:
            self._q_lbl.setText(tr("fs.pro.queue.idle", "coda vuota"))

    # ---------- FS4 gating cap.filmstudio ----------
    def _apply_filmstudio_gate(self):
        from core import features
        if features.is_enabled("cap.filmstudio"):
            return
        from ui.widgets.upsell import UpsellPanel
        info = features.manifest().get("features", {}).get("cap.filmstudio", {})
        self.tabs.setVisible(False)
        self.layout().addWidget(UpsellPanel("cap.filmstudio", info.get("i18n_key", "")))

    # ---------- FS5 conferma Master ----------
    def _wrap_master_confirm(self):
        btn = getattr(self.mt_tab, "btn_master", None)
        if btn is None:
            return
        try:
            btn.clicked.disconnect()
        except Exception:                                    # noqa: BLE001
            pass
        btn.clicked.connect(self._confirm_master)

    def _confirm_master(self):
        ans = QMessageBox.question(
            self, tr("fs.pro.master.confirm.title", "Genera Master"),
            tr("fs.pro.master.confirm", "Avviare il render del Master? L'operazione può richiedere tempo."),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans == QMessageBox.Yes:
            self.mt_tab._master()
