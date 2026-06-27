"""Tab Montaggio/Master (spec §11): timeline dalle clip generate → (A) master ffmpeg + (B) export NLE.
Versione funzionale: raccoglie i video degli shot in ordine, render master e/o export OTIO."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from core.film_studio import db, otio_export
from ui.i18n_manager import tr

_FMT = [("fcpxml", "FCPXML (Filmora/Resolve)"), ("otio", "OpenTimelineIO"),
        ("aaf", "AAF (Premiere/Avid)"), ("edl", "EDL")]
_ASPECT = [("16:9", "16:9"), ("2.39:1", "2.39:1 Cinemascope"), ("9:16", "9:16 Vertical"), ("1:1", "1:1")]


class MontageTab(QWidget):
    def __init__(self, get_project):
        super().__init__()
        self._get_project = get_project
        v = QVBoxLayout(self)
        h = QHBoxLayout()
        self.btn_refresh = QPushButton(tr("fs.mt.refresh", "↻ Aggiorna timeline")); self.btn_refresh.clicked.connect(self._refresh)
        h.addWidget(self.btn_refresh); h.addStretch(1)
        v.addLayout(h)
        self.lst = QListWidget(); v.addWidget(self.lst, 1)
        box = QGroupBox(tr("fs.mt.output", "Output")); bv = QVBoxLayout(box)
        r1 = QHBoxLayout()
        self.aspect = QComboBox()
        for code, lab in _ASPECT:
            self.aspect.addItem(lab, code)
        r1.addWidget(QLabel(tr("fs.mt.aspect", "Aspetto"))); r1.addWidget(self.aspect)
        self.fmt = QComboBox()
        for code, lab in _FMT:
            self.fmt.addItem(lab, code)
        r1.addWidget(QLabel(tr("fs.mt.nle", "Formato NLE"))); r1.addWidget(self.fmt); r1.addStretch(1)
        bv.addLayout(r1)
        r2 = QHBoxLayout()
        self.btn_audio = QPushButton(tr("fs.mt.audio", "🔊 Genera audio (dialoghi+musica)")); self.btn_audio.clicked.connect(self._gen_audio)
        self.btn_master = QPushButton(tr("fs.mt.master", "🎞 Master (A)")); self.btn_master.clicked.connect(self._master)
        self.btn_export = QPushButton(tr("fs.mt.export", "📦 Export NLE (B)")); self.btn_export.clicked.connect(self._export)
        r2.addWidget(self.btn_audio); r2.addWidget(self.btn_master); r2.addWidget(self.btn_export); r2.addStretch(1)
        bv.addLayout(r2)
        self.info = QLabel(""); bv.addWidget(self.info)
        v.addWidget(box)
        self._refresh()

    def _clips(self) -> list[dict]:
        pid = self._get_project()
        if not pid:
            return []
        out = []
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                f = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}.mp4")
                if Path(f).exists():
                    out.append({"file": f, "in_s": 0.0, "out_s": float(sh["duration_s"] or 4),
                                "name": f"S{sc['ord']}.{sh['ord']}"})
        return out

    def _refresh(self):
        self.lst.clear()
        self._cur_clips = self._clips()
        for c in self._cur_clips:
            self.lst.addItem(f"{c['name']}  ({c['out_s']:.0f}s)  {Path(c['file']).name}")
        self.info.setText(tr("fs.mt.count", "{n} clip in timeline").format(n=len(self._cur_clips)))

    def _gen_audio(self):
        """Sintetizza i dialoghi (voce legata all'attore) per ogni shot → tracce audio temporizzate."""
        import json
        from core.film_studio import voice_studio as vs
        pid = self._get_project()
        if not pid:
            return
        self._audio_tracks = []
        t = 0.0
        n = 0
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                dur = float(sh["duration_s"] or 4)
                for di, d in enumerate(json.loads(sh.get("dialogues") or "[]")):
                    actor = (d.get("actor") or "").lstrip("@")
                    text = d.get("text") or ""
                    if not text:
                        continue
                    assets = [a for a in db.find_assets(actor, "actor") if a["name"].lower() == actor.lower()]
                    vrow = vs.voice_for_actor(assets[0]["id"]) if assets else None
                    if not vrow:
                        continue
                    out = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_d{di}.wav")
                    try:
                        vs.synth_dialogue(text, vrow, out)
                        self._audio_tracks.append({"path": out, "start_s": t, "volume": 1.0})
                        n += 1
                    except Exception:                       # noqa: BLE001
                        pass
                t += dur
        self.info.setText(tr("fs.mt.audio_done", "Audio: {n} battute sintetizzate").format(n=n))

    def _master(self):
        clips = [c["file"] for c in (self._cur_clips or [])]
        if not clips:
            QMessageBox.information(self, tr("fs.mt.master", "Master"), tr("fs.mt.no_clips", "Nessuna clip generata.")); return
        pid = self._get_project()
        out = str(db.project_dir(pid) / "master.mp4")
        try:
            otio_export.render_master(clips, out, audio_tracks=getattr(self, "_audio_tracks", None))
            self.info.setText(tr("fs.mt.master_done", "Master: {f}").format(f=out))
        except Exception as e:                              # noqa: BLE001
            QMessageBox.critical(self, tr("fs.mt.master", "Master"), str(e)[:600])

    def _export(self):
        if not self._cur_clips:
            QMessageBox.information(self, tr("fs.mt.export", "Export"), tr("fs.mt.no_clips", "Nessuna clip generata.")); return
        pid = self._get_project()
        out_dir = str(db.project_dir(pid) / "export")
        try:
            p = otio_export.export_nle(self._cur_clips, out_dir, fmt=self.fmt.currentData(),
                                       name=f"project_{pid}")
            self.info.setText(tr("fs.mt.export_done", "Export: {f}").format(f=p))
        except Exception as e:                              # noqa: BLE001
            QMessageBox.critical(self, tr("fs.mt.export", "Export"), str(e)[:600])
