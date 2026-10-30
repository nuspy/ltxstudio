"""Primo-uso UNIFICATO dei pesi (FASE C weights-unification) — mai download silenziosi.

Politica (connessione METERED dell'utente): quando un motore ha bisogno di un peso assente,
o (a) lo dice CHIARO con id di catalogo, dimensione e dove scaricarlo (require → errore
azionabile che i tab mostrano nei loro dialog), o (b) in un contesto GUI lo OFFRE con un
dialog di consenso (offer_download_gui: GB totali dipendenze incluse → QProcess del
downloader unico → progresso). Mai più i 12 motori che scaricavano da soli senza chiedere
(TRELLIS 8GB su C:, TTS in cache HF, GFPGAN 350MB da GitHub…).

stdlib + catalogo; Qt importato SOLO dentro offer_download_gui (usabile dai servizi headless).
"""
from __future__ import annotations

import sys
from pathlib import Path

from core import weights_catalog as wc


class WeightsMissingError(RuntimeError):
    """Pesi assenti. `ids`/`entries`/`total_gb` per chi vuole offrire il download; il messaggio
    è già pronto per i dialog d'errore dei tab (id catalogo + GB + dove scaricare)."""

    def __init__(self, entries: list[dict], feature: str = ""):
        self.entries = entries
        self.ids = [e["id"] for e in entries]
        self.total_gb = round(sum(float(e.get("approx_gb", 0) or 0) for e in entries), 1)
        names = ", ".join(f"{e['id']} (~{e.get('approx_gb', '?')}GB)" for e in entries)
        extra = [e for e in entries if e.get("no_remote")]
        hint = ("Scaricali dal tab «Pesi modelli» (o wizard primo avvio)."
                if not extra else
                "Alcuni non hanno fonte remota (no_remote): vanno collegati/importati "
                "dal tab «Pesi modelli» → Collega/Importa.")
        super().__init__(
            (f"[{feature}] " if feature else "") +
            f"Pesi mancanti (~{self.total_gb}GB): {names}. {hint}")


def missing(ids: list[str]) -> list[dict]:
    """Voci (dipendenze ricorsive INCLUSE, deps prima) non presenti su disco."""
    from core import downloader as dl
    return [e for e in dl.plan_ids(ids) if not wc.is_present(e)]


def require(ids: list[str], feature: str = "") -> None:
    """Solleva WeightsMissingError (chiara e azionabile) se qualcosa manca. Da chiamare nei
    SERVIZI prima di lanciare il worker: l'errore risale al dialog del tab già esistente."""
    miss = missing(ids)
    if miss:
        raise WeightsMissingError(miss, feature)


def offer_download_gui(parent, ids: list[str], feature: str = "") -> bool:
    """Contesto GUI: dialog di CONSENSO (GB totali, deps incluse) → download via QProcess del
    downloader unico con log live → True se tutto presente alla fine. Pattern identico al
    download-on-select dei dropdown (video_tab), qui riusabile da qualunque tab/feature."""
    miss = missing(ids)
    if not miss:
        return True
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QMessageBox, QProgressDialog
    try:
        from ui.i18n_manager import tr
    except Exception:                                    # noqa: BLE001
        def tr(k, d=None):
            return d if d is not None else k
    dl_ids = [e["id"] for e in miss if not e.get("no_remote")]
    nr = [e["id"] for e in miss if e.get("no_remote")]
    gb = round(sum(float(e.get("approx_gb", 0) or 0) for e in miss if not e.get("no_remote")), 1)
    if nr:
        QMessageBox.warning(parent, tr("weights.missing.title", "Pesi mancanti"),
                            tr("weights.missing.no_remote",
                               "Questi pesi non hanno una fonte remota e vanno collegati/importati "
                               "dal tab «Pesi modelli»: {ids}").format(ids=", ".join(nr)))
        if not dl_ids:
            return False
    from core import weights_tooltips
    _lines = "\n".join("• " + weights_tooltips.short_line(e) for e in miss if not e.get("no_remote"))
    msg = tr("weights.missing.offer",
             "{feature}Servono questi pesi (~{gb}GB, dipendenze incluse):\n{ids}\n\nScaricarli ora?"
             ).format(feature=(feature + ": ") if feature else "", gb=gb, ids=_lines)
    if QMessageBox.question(parent, tr("weights.missing.title", "Pesi mancanti"), msg) \
            != QMessageBox.Yes:
        return False
    from core import license_ack as _la
    _acc = _la.confirm_gui(parent, dl_ids)                    # voci non commerciali: conferma di licenza
    if _acc is None:
        return False
    dlg = QProgressDialog(tr("weights.missing.downloading", "Download pesi in corso…"),
                          tr("common.cancel", "Annulla"), 0, 0, parent)
    dlg.setWindowTitle(tr("weights.missing.title", "Pesi mancanti"))
    dlg.setMinimumDuration(0)
    proc = QProcess(parent)
    repo = Path(__file__).resolve().parents[1]
    proc.setWorkingDirectory(str(repo))
    def _mostra_progresso():
        """Ultima riga utile del downloader nell'etichetta. Il chunk letto può essere VUOTO (o solo
        spazi): `splitlines()[-1]` su lista vuota alzava IndexError a ogni tick, riempiendo il log
        di traceback e lasciando il dialog fermo (visto live il 17/08 su un download da 1.2GB)."""
        righe = bytes(proc.readAllStandardOutput()).decode("utf-8", "replace").strip().splitlines()
        if righe:
            dlg.setLabelText(righe[-1][:110])

    proc.readyReadStandardOutput.connect(_mostra_progresso)
    dlg.canceled.connect(proc.kill)
    proc.start(sys.executable, ["-u", str(repo / "scripts" / "download_weights.py"),
                                "--ids", ",".join(dl_ids)] + _la.args_for(_acc))
    while proc.state() != QProcess.NotRunning:
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
    dlg.close()
    return not missing(ids)
