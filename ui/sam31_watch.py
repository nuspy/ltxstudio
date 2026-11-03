"""Controllo dell'accesso a SAM 3.1 in background: ogni 30 minuti (dopo «Ho chiesto l'accesso») una richiesta leggera col token dell'utente; appena Meta
approva parte il download con il DownloadManager esistente (QProcess: progresso nell'icona ⬇, «Annulla», la GUI non si blocca).

La logica (stato, scadenze, esiti) sta in core/sam31_access.py; qui solo Qt: timer, thread del controllo, avvio del download.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from core import sam31_access as AC
from core import segmentation_engine as SE
from ui.i18n_manager import tr

TICK_MS = 60_000                      # il timer batte ogni minuto; il controllo vero parte solo quando AC.due() (30 minuti dall'ultimo)


class _ProbeThread(QThread):
    done = Signal(str, str)

    def __init__(self, probe, parent=None):
        super().__init__(parent)
        self._probe = probe

    def run(self):                                           # noqa: D401  (thread del controllo: solo rete, niente GUI)
        try:
            res, why = self._probe()
        except Exception as e:                               # noqa: BLE001
            res, why = "error", type(e).__name__
        self.done.emit(res, why)


class Sam31Watcher(QObject):
    changed = Signal()                # stato cambiato: bottone, voci di menu, finestra di aiuto
    message = Signal(str)             # messaggio breve per la barra di stato

    def __init__(self, parent=None, *, probe=None, start_download=None, tick_ms: int = TICK_MS):
        super().__init__(parent)
        self._probe = probe or AC.probe
        self._start_download = start_download or self._default_download
        self._thread: _ProbeThread | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(int(tick_ms))
        self._timer.timeout.connect(self._tick)
        self.downloading_job: int | None = None
        try:
            from ui.download_manager import manager
            manager().finished.connect(self._on_download_finished)
            manager().changed.connect(self._on_manager_changed)
        except Exception:                                    # noqa: BLE001
            pass

    def _on_manager_changed(self) -> None:
        try:
            self.changed.emit()
        except RuntimeError:                                 # il watcher e gia stato distrutto (chiusura dell'app)
            pass

    # ── ciclo ──
    def start(self) -> None:
        self._timer.start()
        QTimer.singleShot(5000, self._tick)                  # dopo l'avvio dell'app: un controllo se e gia scaduto

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        if AC.due() and not self.busy():
            self.check_now(auto=True)

    def busy(self) -> bool:
        return bool(self._thread is not None and self._thread.isRunning()) or self.download_running()

    def download_running(self) -> bool:
        try:
            from ui.download_manager import manager
            return any(SE.SAM31_CATALOG_ID in j.ids for j in manager().active())
        except Exception:                                    # noqa: BLE001
            return False

    # ── azioni dell'utente ──
    def confirm_requested(self) -> None:
        AC.confirm_requested()
        self.changed.emit()
        self.check_now(auto=False)                           # primo controllo subito: se l'accesso c'e gia, parte il download

    def snooze(self, seconds: float = AC.SNOOZE_DEFAULT_S) -> None:
        AC.snooze(seconds)
        self.message.emit(tr("sam31.snoozed", "Controllo di SAM 3.1 rimandato di 2 ore."))
        self.changed.emit()

    def cancel(self) -> None:
        AC.cancel_watch()
        self.message.emit(tr("sam31.cancelled", "Controllo automatico di SAM 3.1 annullato."))
        self.changed.emit()

    def refresh(self) -> None:
        self.changed.emit()

    def check_now(self, auto: bool = False) -> None:
        """«Controlla e scarica SAM 3.1»: una richiesta leggera in un thread; se l'accesso c'e, parte il download in background."""
        if self.download_running():
            self.message.emit(tr("sam31.dl_running", "Il download di SAM 3.1 e gia in corso."))
            return
        if self._thread is not None and self._thread.isRunning():
            return
        if not auto:
            self.message.emit(tr("sam31.checking", "Controllo l'accesso a SAM 3.1…"))
        self._thread = _ProbeThread(self._probe, self)
        self._thread.done.connect(self._on_probe)
        self._thread.start()

    def _on_probe(self, res: str, _why: str) -> None:
        AC.record(res)
        self.message.emit(tr(f"sam31.res.{res}", AC.describe(res)))
        if res == "granted":
            self._start_download()
        self.changed.emit()

    def _default_download(self) -> None:
        try:
            from ui.download_manager import manager
            if self.download_running():
                return
            self.downloading_job = manager().start([SE.SAM31_CATALOG_ID], "SAM 3.1")
            self.message.emit(tr("sam31.dl_started", "Accesso approvato: scarico SAM 3.1 in background (circa 3,5 GB). Trovi l'avanzamento e «Annulla» nell'icona ⬇ in alto."))
        except Exception as e:                               # noqa: BLE001
            self.message.emit(f"SAM 3.1: {e}")

    def download_comfy(self) -> None:
        """Alternativa non gated: scarica la copia Comfy-Org (1,75 GB) col normale download del catalogo."""
        try:
            from ui.download_manager import manager
            if any(SE.COMFY_CATALOG_ID in j.ids for j in manager().active()):
                self.message.emit(tr("sam31.dl_running", "Il download di SAM 3.1 e gia in corso."))
                return
            manager().start([SE.COMFY_CATALOG_ID], "SAM 3.1 (Comfy-Org)")
        except Exception as e:                               # noqa: BLE001
            self.message.emit(f"SAM 3.1: {e}")

    def _on_download_finished(self, info: dict) -> None:
        if SE.SAM31_CATALOG_ID in (info.get("ids") or []) or SE.COMFY_CATALOG_ID in (info.get("ids") or []):
            self.downloading_job = None
            self.changed.emit()
