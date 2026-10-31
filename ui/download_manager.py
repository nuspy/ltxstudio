"""Download dei pesi IN BACKGROUND (punto 7 dell'utente, 30/10/2026).

Prima `offer_download_gui` teneva l'app bloccata per tutto il download (loop di processEvents in un dialog). Ora il
download parte in un QProcess (scripts/download_weights.py, lo stesso downloader unico) gestito da `manager()`:
l'interfaccia resta libera, nella barra compare l'icona con MB scaricati/totale, velocita, tempo stimato e «Annulla»,
e a fine download si apre un avviso non modale. `LTX_DOWNLOAD_BLOCKING=1` = comportamento vecchio (test).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

_PROG = re.compile(r"\[weights\] … (?P<id>\S+): (?P<mb>[\d,\.]+)MB \(v=(?P<v>[\d,\.]+)MB/s\)")
_START = re.compile(r"\[weights\] ↓ (?P<id>\S+) — .*?~(?P<gb>[\d\.\?]+)GB")
_DONE = re.compile(r"\[weights\] ✓ (?P<id>\S+) completato")
_FAIL = re.compile(r"\[weights\] ✗ (?P<id>\S+): (?P<why>.*)")


def _f(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


class _Job:
    def __init__(self, jid, ids, feature, proc):
        self.id, self.ids, self.feature, self.proc = jid, list(ids), feature, proc
        self.t0 = time.time()
        self.cur = ""                 # id in corso
        self.totals = {}              # id -> MB totali stimati
        self.done_mb = {}             # id -> MB scaricati (corrente)
        self.finished_ids = []        # id completati
        self.failed = []              # (id, motivo)
        self.speed = 0.0
        self.state = "running"        # running | done | failed | cancelled

    def total_mb(self) -> float:
        return sum(self.totals.get(i, 0.0) for i in self.ids)

    def got_mb(self) -> float:
        return sum(self.totals.get(i, 0.0) if i in self.finished_ids else self.done_mb.get(i, 0.0) for i in self.ids)

    def eta_s(self):
        left = self.total_mb() - self.got_mb()
        return left / self.speed if self.speed > 0.05 and left > 0 else None


class DownloadManager(QObject):
    changed = Signal()
    finished = Signal(dict)

    def __init__(self):
        super().__init__()
        self._jobs: dict = {}
        self._n = 0

    def active(self) -> list:
        return [j for j in self._jobs.values() if j.state == "running"]

    def jobs(self) -> list:
        return list(self._jobs.values())

    def start(self, ids: list, feature: str = "", extra_args=None) -> int:
        self._n += 1
        repo = Path(__file__).resolve().parents[1]
        proc = QProcess(self)
        proc.setWorkingDirectory(str(repo))
        job = _Job(self._n, ids, feature, proc)
        try:                          # totali stimati dal catalogo (MB), finche il downloader non dice di piu
            from core import weights_catalog as wc
            by = {e["id"]: e for e in wc.load()}
            for i in ids:
                job.totals[i] = float(by.get(i, {}).get("approx_gb", 0) or 0) * 1024.0
        except Exception:             # noqa: BLE001
            pass
        proc.readyReadStandardOutput.connect(lambda j=job: self._read(j))
        proc.finished.connect(lambda *_a, j=job: self._end(j))
        self._jobs[job.id] = job
        proc.start(sys.executable, ["-u", str(repo / "scripts" / "download_weights.py"), "--ids", ",".join(ids)] + list(extra_args or []))
        self.changed.emit()
        return job.id

    def cancel(self, jid: int) -> None:
        j = self._jobs.get(jid)
        if j and j.state == "running":
            j.state = "cancelled"
            j.proc.kill()
            self.changed.emit()

    def cancel_all(self) -> None:
        for j in list(self.active()):
            self.cancel(j.id)

    def _read(self, j: _Job) -> None:
        for line in bytes(j.proc.readAllStandardOutput()).decode("utf-8", "replace").splitlines():
            m = _START.search(line)
            if m:
                j.cur = m["id"]
                if m["gb"] != "?":
                    j.totals[m["id"]] = _f(m["gb"]) * 1024.0
            m = _PROG.search(line)
            if m:
                j.cur = m["id"]
                j.done_mb[m["id"]] = _f(m["mb"])
                j.speed = _f(m["v"])
                j.totals[m["id"]] = max(j.totals.get(m["id"], 0.0), _f(m["mb"]))
            m = _DONE.search(line)
            if m:
                j.finished_ids.append(m["id"])
            m = _FAIL.search(line)
            if m:
                j.failed.append((m["id"], m["why"][:200]))
        self.changed.emit()

    def _end(self, j: _Job) -> None:
        self._read(j)
        if j.state == "running":
            ok = (not j.failed) and (len(j.finished_ids) >= len(j.ids) or self._all_present(j.ids))
            j.state = "done" if ok else "failed"
        self.changed.emit()
        self.finished.emit({"id": j.id, "state": j.state, "ids": j.ids, "feature": j.feature, "failed": j.failed,
                            "elapsed_s": time.time() - j.t0})

    @staticmethod
    def _all_present(ids) -> bool:
        try:
            from core import ensure_weights as ew
            return not ew.missing(ids)
        except Exception:             # noqa: BLE001
            return False


_MANAGER = None


def manager() -> DownloadManager:
    global _MANAGER
    if _MANAGER is None:
        from PySide6.QtWidgets import QApplication
        _MANAGER = DownloadManager()
        if QApplication.instance():
            _MANAGER.setParent(QApplication.instance())
    return _MANAGER


# --------------------- interfaccia: elenco dei download in corso ---------------------
def _fmt_eta(s) -> str:
    if s is None:
        return "..."
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else (f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s")


def downloads_dialog(parent=None):
    """Finestra NON modale con i download in corso: MB scaricati/totale, velocita, tempo stimato, Annulla."""
    from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget
    from ui.i18n_manager import tr
    mgr = manager()
    dlg = QDialog(parent)
    dlg.setWindowTitle(tr("dl.title", "Download in corso"))
    dlg.setModal(False)
    dlg.setMinimumWidth(520)
    root = QVBoxLayout(dlg)
    body = QVBoxLayout()
    root.addLayout(body)

    def rebuild(*_):
        while body.count():
            w = body.takeAt(0).widget()
            if w:
                w.deleteLater()
        shown = [j for j in mgr.jobs() if j.state == "running"]
        if not shown:
            body.addWidget(QLabel(tr("dl.none", "Nessun download in corso.")))
        for j in shown:
            box = QWidget()
            v = QVBoxLayout(box)
            tot, got = j.total_mb(), j.got_mb()
            lbl = QLabel(tr("dl.row", "{label}: {done} / {total} MB - {speed} MB/s - ~{eta} rimanenti").format(
                label=(j.feature or j.cur or ", ".join(j.ids))[:70], done=f"{got:,.0f}", total=f"{tot:,.0f}",
                speed=f"{j.speed:.1f}", eta=_fmt_eta(j.eta_s())))
            lbl.setWordWrap(True)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(1000 * got / tot) if tot > 0 else 0)
            row = QHBoxLayout()
            row.addWidget(bar, 1)
            b = QPushButton(tr("dl.cancel", "Annulla download"))
            b.clicked.connect(lambda _=False, jid=j.id: mgr.cancel(jid))
            row.addWidget(b)
            v.addWidget(lbl)
            v.addLayout(row)
            body.addWidget(box)
    mgr.changed.connect(rebuild)
    dlg.finished.connect(lambda *_: mgr.changed.disconnect(rebuild))
    rebuild()
    dlg.show()
    return dlg
