"""Music Video Studio — workflow INTERATTIVO con progetti persistenti (SQLite, core/mv_projects).

Fasi (salvate e riapribili):
  0) CANZONE — genera/rigenera la canzone; tutte le canzoni del progetto in lista, selezione dell'attiva.
  1) RIPRESE/IMMAGINI — elenco riprese editabile (start mm:ss, durata auto, immagine apertura, prompt
     immagine+video), rigenera/sostituisci immagine, riordina, aggiungi (fine/tra due), cancella.
  2) VIDEO — genera i2v per ripresa + elenco video editabile (rigenera/riordina/cancella).
  3) MONTAGGIO — concat nell'ordine + mux audio della canzone attiva.

Tutto persistito in core/mv_projects (mv_project/mv_song/mv_scene). Zero ComfyUI.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QMessageBox, QPlainTextEdit, QPushButton,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from core import music_video_service as mvs, mv_projects as mvp
from ui.i18n_manager import tr


def _mmss(s: float) -> str:
    s = max(0, int(s)); return f"{s // 60:02d}:{s % 60:02d}"


class _Worker(QThread):
    progress = Signal(str)
    done = Signal(dict)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self._fn, self._args, self._kw = fn, args, kw

    def run(self):
        try:
            self.done.emit({"result": self._fn(*self._args, **self._kw)})
        except Exception as e:                              # noqa: BLE001
            self.done.emit({"error": str(e)})


class MusicVideoStudio(QWidget):
    def __init__(self, project_id: int, parent=None):
        super().__init__(parent)
        self.pid = project_id
        self.proj = mvp.get_project(project_id) or {}
        self.setWindowTitle(tr("mvs.title", "🎬 Music Video Studio") + f" — {self.proj.get('name','')}")
        self.resize(1080, 780)
        self.out_dir = str(mvp.project_dir(project_id))
        self.scenes: list[dict] = mvp.list_scenes(project_id)
        self._worker = None

        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._song_tab(), tr("mvs.tab_song", "1 · Canzone"))
        self.tabs.addTab(self._storyboard_tab(), tr("mvs.tab_sb", "2 · Riprese / Immagini"))
        self.tabs.addTab(self._video_tab(), tr("mvs.tab_vid", "3 · Video"))
        root.addWidget(self.tabs, 1)
        self.status = QLabel("")
        root.addWidget(self.status)
        self._refresh_songs()
        self._refresh_sb()
        # riapri alla fase salvata
        ph = self.proj.get("phase", "song")
        self.tabs.setCurrentIndex({"song": 0, "storyboard": 1, "video": 1, "done": 2}.get(ph, 0))

    # ───────────── helpers ─────────────
    def _song_dur(self) -> float:
        a = mvp.active_song(self.pid)
        return mvs._dur(a["file"]) if a and os.path.exists(a["file"]) else float(self.proj.get("song_duration", 120))

    def _export_nle(self):
        if not any(s.get("video_path") for s in self.scenes):
            QMessageBox.information(self, tr("mvs.export_nle", "Export"), tr("mvs.no_vids", "Genera prima i video.")); return
        try:
            p = mvs.export_nle(self.scenes, os.path.join(self.out_dir, "export"),
                               fmt=self.btn_export.currentData(), name=f"mv_{self.pid}")
            self.status.setText(tr("mvs.export_done", "Export NLE: {f}").format(f=p))
        except Exception as e:                              # noqa: BLE001
            QMessageBox.critical(self, tr("mvs.export_nle", "Export"), str(e)[:500])

    def _busy(self, on: bool, msg: str = ""):
        for b in self._all_btns:
            b.setEnabled(not on)
        if msg:
            self.status.setText(msg)

    # ───────────── Fase 0: CANZONE ─────────────
    def _song_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        # parametri canzone (2.4)
        from PySide6.QtWidgets import QDoubleSpinBox as _DS
        ph = QHBoxLayout()
        self.song_temp = _DS(); self.song_temp.setRange(0.1, 2.0); self.song_temp.setSingleStep(0.1)
        self.song_temp.setValue(float(self.proj.get("song_temperature", 1.0) or 1.0))
        self.song_cfg = _DS(); self.song_cfg.setRange(1.0, 10.0); self.song_cfg.setSingleStep(0.5)
        self.song_cfg.setValue(float(self.proj.get("song_cfg", 1.5) or 1.5))
        ph.addWidget(QLabel(tr("mvs.song_temp", "Temperatura"))); ph.addWidget(self.song_temp)
        ph.addWidget(QLabel(tr("mvs.song_cfg", "CFG"))); ph.addWidget(self.song_cfg); ph.addStretch(1)
        v.addLayout(ph)
        v.addWidget(QLabel(tr("mvs.songs", "Canzoni generate per il progetto (seleziona quella attiva)")))
        self.song_list = QListWidget(); self.song_list.currentRowChanged.connect(self._song_select)
        v.addWidget(self.song_list, 1)
        h = QHBoxLayout()
        self.btn_gensong = QPushButton(tr("mvs.gen_song", "🎵 Genera/Rigenera canzone")); self.btn_gensong.clicked.connect(self._gen_song)
        self.btn_play = QPushButton(tr("mvs.play", "▶ Ascolta")); self.btn_play.clicked.connect(self._play_song)
        self.btn_setactive = QPushButton(tr("mvs.set_active", "✓ Usa questa")); self.btn_setactive.clicked.connect(self._set_active)
        self.btn_delsong = QPushButton(tr("tlora.del", "🗑")); self.btn_delsong.clicked.connect(self._del_song)
        for b in (self.btn_gensong, self.btn_play, self.btn_setactive, self.btn_delsong):
            h.addWidget(b)
        h.addStretch(1)
        self.btn_to_sb = QPushButton(tr("mvs.to_sb", "→ Vai alle riprese")); self.btn_to_sb.clicked.connect(self._goto_sb)
        h.addWidget(self.btn_to_sb)
        v.addLayout(h)
        return w

    def _refresh_songs(self):
        self.song_list.clear()
        self._songs = mvp.list_songs(self.pid)
        act = self.proj.get("active_song_id")
        for s in self._songs:
            mark = "✓" if s["id"] == act else "·"
            dur = mvs._dur(s["file"]) if os.path.exists(s["file"]) else 0
            self.song_list.addItem(f"{mark} {Path(s['file']).name}  ({_mmss(dur)})")

    def _song_select(self, row: int):
        pass

    def _gen_song(self):
        self.proj = mvp.get_project(self.pid)
        out = os.path.join(self.out_dir, f"song_{int(time.time())}.mp3")
        self._busy(True, tr("mvs.song_gen", "Generazione canzone…"))
        mvp.update_project(self.pid, song_temperature=float(self.song_temp.value()), song_cfg=float(self.song_cfg.value()))
        self._worker = _Worker(mvs.generate_song, lyrics=self.proj.get("lyrics", ""),
                               tags=self.proj.get("tags", "") or "pop", out_mp3=out,
                               duration_s=float(self.proj.get("song_duration", 120)),
                               temperature=float(self.song_temp.value()), cfg_scale=float(self.song_cfg.value()))
        self._worker.done.connect(self._on_song)
        self._worker.start()

    def _on_song(self, res: dict):
        self._busy(False)
        if res.get("error"):
            QMessageBox.critical(self, tr("mvs.gen_song", "Canzone"), str(res["error"])[:600]); return
        sid = mvp.add_song(self.pid, res["result"])
        mvp.set_active_song(self.pid, sid)
        self.proj = mvp.get_project(self.pid)
        self._refresh_songs()
        self.status.setText(tr("mvs.song_ready", "Canzone pronta e impostata come attiva."))

    def _play_song(self):
        r = self.song_list.currentRow()
        if 0 <= r < len(self._songs):
            try:
                os.startfile(self._songs[r]["file"])      # noqa: S606 (apre nel player di sistema)
            except Exception:                              # noqa: BLE001
                pass

    def _set_active(self):
        r = self.song_list.currentRow()
        if 0 <= r < len(self._songs):
            mvp.set_active_song(self.pid, self._songs[r]["id"])
            self.proj = mvp.get_project(self.pid)
            self._refresh_songs()

    def _del_song(self):
        r = self.song_list.currentRow()
        if 0 <= r < len(self._songs):
            mvp.delete_song(self._songs[r]["id"]); self.proj = mvp.get_project(self.pid); self._refresh_songs()

    def _goto_sb(self):
        if not mvp.active_song(self.pid):
            QMessageBox.information(self, tr("mvs.to_sb", "Riprese"), tr("mvs.need_song", "Genera/seleziona prima una canzone.")); return
        mvp.update_project(self.pid, phase="storyboard")
        self.tabs.setCurrentIndex(1)

    # ───────────── Fase 1: STORYBOARD/IMMAGINI ─────────────
    def _storyboard_tab(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w)
        split = QSplitter(Qt.Horizontal)
        left = QWidget(); lv = QVBoxLayout(left)
        bar = QHBoxLayout()
        self.btn_plan = QPushButton(tr("mvs.plan", "Crea elenco riprese (LLM)")); self.btn_plan.clicked.connect(self._plan)
        self.btn_imgs = QPushButton(tr("mvs.gen_imgs", "Genera immagini")); self.btn_imgs.clicked.connect(self._gen_images)
        bar.addWidget(self.btn_plan); bar.addWidget(self.btn_imgs)
        lv.addLayout(bar)
        self.sb_list = QListWidget(); self.sb_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sb_list.currentRowChanged.connect(self._sb_select)
        lv.addWidget(self.sb_list, 1)
        rb = QHBoxLayout()
        for txt, fn in ((tr("mvs.up", "↑"), self._move_up), (tr("mvs.down", "↓"), self._move_down),
                        (tr("mvs.add", "➕ fine"), self._add_end), (tr("mvs.add_after", "➕ dopo"), self._add_after),
                        (tr("tlora.del", "🗑"), self._del)):
            b = QPushButton(txt); b.clicked.connect(fn); rb.addWidget(b)
        lv.addLayout(rb)
        split.addWidget(left)
        det = QGroupBox(tr("mvs.detail", "Dettaglio ripresa")); dv = QVBoxLayout(det)
        self.prev = QLabel(); self.prev.setFixedHeight(180); self.prev.setAlignment(Qt.AlignCenter)
        self.prev.setStyleSheet("border:1px solid #555;")
        dv.addWidget(self.prev)
        sh = QHBoxLayout()
        sh.addWidget(QLabel(tr("mvs.start", "Inizio (s)")))
        self.start = QDoubleSpinBox(); self.start.setRange(0, 100000); self.start.setDecimals(2)
        self.start.valueChanged.connect(self._start_changed)
        sh.addWidget(self.start)
        self.len_lbl = QLabel("—"); sh.addWidget(QLabel(tr("mvs.length", "Durata:"))); sh.addWidget(self.len_lbl); sh.addStretch(1)
        dv.addLayout(sh)
        dv.addWidget(QLabel(tr("mvs.img_prompt", "Prompt immagine iniziale")))
        self.img_prompt = QPlainTextEdit(); self.img_prompt.setFixedHeight(54); dv.addWidget(self.img_prompt)
        ih = QHBoxLayout()
        b_reimg = QPushButton(tr("mvs.regen_img", "🖼 Rigenera immagine")); b_reimg.clicked.connect(self._regen_image)
        b_repl = QPushButton(tr("mvs.replace_img", "📁 Sostituisci…")); b_repl.clicked.connect(self._replace_image)
        ih.addWidget(b_reimg); ih.addWidget(b_repl); ih.addStretch(1)
        dv.addLayout(ih)
        dv.addWidget(QLabel(tr("mvs.vid_prompt", "Prompt video")))
        self.vid_prompt = QPlainTextEdit(); self.vid_prompt.setFixedHeight(54); dv.addWidget(self.vid_prompt)
        ab = QHBoxLayout()
        b_apply = QPushButton(tr("mvs.apply", "💾 Applica")); b_apply.clicked.connect(self._apply_detail)
        self.btn_to_vid = QPushButton(tr("mvs.to_vid", "→ Vai ai video")); self.btn_to_vid.clicked.connect(lambda: (mvp.update_project(self.pid, phase="video"), self.tabs.setCurrentIndex(2)))
        ab.addWidget(b_apply); ab.addStretch(1); ab.addWidget(self.btn_to_vid)
        dv.addLayout(ab)
        split.addWidget(det); split.setSizes([400, 640])
        h.addWidget(split)
        return w

    # ───────────── Fase 2: VIDEO ─────────────
    def _video_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        from PySide6.QtWidgets import QComboBox as _CB
        bar = QHBoxLayout()
        self.btn_vids = QPushButton(tr("mvs.gen_vids", "Genera video")); self.btn_vids.clicked.connect(self._gen_videos)
        self.transition = _CB()
        for code, lab in (("cut", "mvs.tr_cut"), ("crossfade", "mvs.tr_xfade"), ("fade", "mvs.tr_fade"),
                          ("dissolve", "mvs.tr_dissolve"), ("wipeleft", "mvs.tr_wipe")):
            self.transition.addItem(tr(lab, code), code)
        self.btn_assemble = QPushButton(tr("mvs.assemble", "🎞 Monta + audio")); self.btn_assemble.clicked.connect(self._assemble)
        self.btn_export = _CB()
        for f in ("fcpxml", "otio", "aaf", "edl"):
            self.btn_export.addItem(f.upper(), f)
        b_exp = QPushButton(tr("mvs.export_nle", "📦 Export NLE")); b_exp.clicked.connect(self._export_nle)
        bar.addWidget(self.btn_vids); bar.addStretch(1)
        bar.addWidget(QLabel(tr("mvs.transition", "Transizione"))); bar.addWidget(self.transition)
        bar.addWidget(self.btn_assemble); bar.addWidget(self.btn_export); bar.addWidget(b_exp)
        v.addLayout(bar)
        self.vid_list = QListWidget(); v.addWidget(self.vid_list, 1)
        rb = QHBoxLayout()
        for txt, fn in ((tr("mvs.up", "↑"), self._vmove_up), (tr("mvs.down", "↓"), self._vmove_down),
                        (tr("mvs.regen_vid", "🎬 Rigenera video"), self._regen_video), (tr("tlora.del", "🗑"), self._vdel)):
            b = QPushButton(txt); b.clicked.connect(fn); rb.addWidget(b)
        rb.addStretch(1)
        v.addLayout(rb)
        return w

    @property
    def _all_btns(self):
        names = ["btn_gensong", "btn_play", "btn_setactive", "btn_delsong", "btn_to_sb", "btn_plan",
                 "btn_imgs", "btn_vids", "btn_assemble"]
        return [getattr(self, n) for n in names if hasattr(self, n)]

    # persistenza
    def _save(self):
        mvp.save_scenes(self.pid, self.scenes)

    # ───────────── refresh ─────────────
    def _refresh_sb(self):
        cur = self.sb_list.currentRow()
        self.sb_list.clear()
        if self.scenes:
            mvs.recompute_lengths(self.scenes, self._song_dur())
        for s in self.scenes:
            img = "🖼" if s.get("image_path") and os.path.exists(s["image_path"]) else "·"
            self.sb_list.addItem(f"{img} {_mmss(s['start_s'])}  ({s['length_s']:.1f}s)  {s['video_prompt'][:46]}")
        if 0 <= cur < self.sb_list.count():
            self.sb_list.setCurrentRow(cur)
        self._refresh_vid()

    def _refresh_vid(self):
        from PySide6.QtWidgets import QListWidgetItem
        from PySide6.QtGui import QIcon
        from PySide6.QtCore import QSize
        self.vid_list.setIconSize(QSize(96, 54))
        self.vid_list.clear()
        for s in sorted(self.scenes, key=lambda x: x["order"]):
            has = s.get("video_path") and os.path.exists(s["video_path"])
            mark = "🎥" if has else "·"
            it = QListWidgetItem(f"{mark} {_mmss(s['start_s'])}  ({s['length_s']:.1f}s)  {s['video_prompt'][:46]}")
            if has:                                          # thumbnail = primo frame del video (cache .thumb.jpg)
                thumb = s["video_path"] + ".thumb.jpg"
                if not os.path.exists(thumb):
                    import subprocess
                    subprocess.run(["ffmpeg", "-v", "error", "-i", s["video_path"], "-vframes", "1",
                                    "-vf", "scale=96:-1", thumb, "-y"], check=False)
                if os.path.exists(thumb):
                    it.setIcon(QIcon(thumb))
            self.vid_list.addItem(it)

    def _cur(self):
        r = self.sb_list.currentRow()
        return self.scenes[r] if 0 <= r < len(self.scenes) else None

    def _sb_select(self, row: int):
        s = self._cur()
        if not s:
            return
        self.start.blockSignals(True); self.start.setValue(s["start_s"]); self.start.blockSignals(False)
        self.len_lbl.setText(f"{s['length_s']:.1f}s")
        self.img_prompt.setPlainText(s["img_prompt"]); self.vid_prompt.setPlainText(s["video_prompt"])
        if s.get("image_path") and os.path.exists(s["image_path"]):
            self.prev.setPixmap(QPixmap(s["image_path"]).scaledToHeight(176, Qt.SmoothTransformation))
        else:
            self.prev.setText(tr("mvs.no_img", "(nessuna immagine)"))

    # ───────────── editing storyboard ─────────────
    def _apply_detail(self):
        s = self._cur()
        if not s:
            return
        s["img_prompt"] = self.img_prompt.toPlainText().strip()
        s["video_prompt"] = self.vid_prompt.toPlainText().strip()
        self._save(); self._refresh_sb()

    def _start_changed(self, val):
        s = self._cur()
        if s:
            s["start_s"] = float(val); self._save(); self._refresh_sb()

    def _move_up(self):
        r = self.sb_list.currentRow()
        if r > 0:
            self.scenes[r]["start_s"], self.scenes[r - 1]["start_s"] = self.scenes[r - 1]["start_s"], self.scenes[r]["start_s"]
            self._save(); self._refresh_sb(); self.sb_list.setCurrentRow(r - 1)

    def _move_down(self):
        r = self.sb_list.currentRow()
        if 0 <= r < len(self.scenes) - 1:
            self.scenes[r]["start_s"], self.scenes[r + 1]["start_s"] = self.scenes[r + 1]["start_s"], self.scenes[r]["start_s"]
            self._save(); self._refresh_sb(); self.sb_list.setCurrentRow(r + 1)

    def _new_scene(self, start_s: float) -> dict:
        style = self.proj.get("video_style", "") or "cinematic"
        return {"order": len(self.scenes), "start_s": round(start_s, 2), "length_s": 1.0,
                "video_prompt": style, "img_prompt": style + ", still frame", "image_path": "", "video_path": ""}

    def _add_end(self):
        last = max((s["start_s"] for s in self.scenes), default=0.0)
        self.scenes.append(self._new_scene(min(self._song_dur() - 0.5, last + self.proj.get("scene_seconds", 4))))
        self._save(); self._refresh_sb()

    def _add_after(self):
        s = self._cur()
        if not s:
            return self._add_end()
        nxt = min((x["start_s"] for x in self.scenes if x["start_s"] > s["start_s"]), default=self._song_dur())
        self.scenes.append(self._new_scene((s["start_s"] + nxt) / 2))
        self._save(); self._refresh_sb()

    def _del(self):
        r = self.sb_list.currentRow()
        if 0 <= r < len(self.scenes):
            self.scenes.pop(r); self._save(); self._refresh_sb()

    def _regen_image(self):
        s = self._cur()
        if not s:
            return
        s["img_prompt"] = self.img_prompt.toPlainText().strip()
        out = os.path.join(self.out_dir, f"img_{int(s['start_s']*100)}.png")
        self._run(mvs.gen_scene_image, s, out)

    def _replace_image(self):
        s = self._cur()
        if not s:
            return
        p, _ = QFileDialog.getOpenFileName(self, tr("mvs.replace_img", "Sostituisci"), "", "Immagini (*.png *.jpg *.jpeg *.webp)")
        if p:
            s["image_path"] = p; self._save(); self._refresh_sb(); self._sb_select(self.sb_list.currentRow())

    # ───────────── editing video ─────────────
    def _vmove_up(self):
        r = self.vid_list.currentRow(); ordered = sorted(self.scenes, key=lambda x: x["order"])
        if r > 0:
            ordered[r]["order"], ordered[r - 1]["order"] = ordered[r - 1]["order"], ordered[r]["order"]
            self._save(); self._refresh_vid(); self.vid_list.setCurrentRow(r - 1)

    def _vmove_down(self):
        r = self.vid_list.currentRow(); ordered = sorted(self.scenes, key=lambda x: x["order"])
        if 0 <= r < len(ordered) - 1:
            ordered[r]["order"], ordered[r + 1]["order"] = ordered[r + 1]["order"], ordered[r]["order"]
            self._save(); self._refresh_vid(); self.vid_list.setCurrentRow(r + 1)

    def _vdel(self):
        r = self.vid_list.currentRow(); ordered = sorted(self.scenes, key=lambda x: x["order"])
        if 0 <= r < len(ordered):
            self.scenes.remove(ordered[r]); self._save(); self._refresh_sb()

    def _regen_video(self):
        r = self.vid_list.currentRow(); ordered = sorted(self.scenes, key=lambda x: x["order"])
        if not (0 <= r < len(ordered)):
            return
        s = ordered[r]
        if not s.get("image_path"):
            QMessageBox.information(self, tr("mvs.regen_vid", "Video"), tr("mvs.need_img", "Genera prima l'immagine.")); return
        out = os.path.join(self.out_dir, f"vid_{s['order']}_{int(s['start_s']*100)}.mp4")
        self._run(mvs.gen_scene_video, s, out)

    # ───────────── batch + single (background) ─────────────
    def _plan(self):
        a = mvp.active_song(self.pid)
        if not a:
            QMessageBox.information(self, tr("mvs.plan", "Riprese"), tr("mvs.need_song", "Genera/seleziona prima una canzone.")); return
        self._busy(True, tr("mvs.planning", "Elenco riprese…"))
        self._worker = _Worker(mvs.plan_scenes, lyrics=self.proj.get("lyrics", ""), song_path=a["file"],
                               scene_seconds=float(self.proj.get("scene_seconds", 4)),
                               video_style=self.proj.get("video_style", ""),
                               image_hint=self.proj.get("image_hint", ""))
        self._worker.done.connect(self._on_plan); self._worker.start()

    def _on_plan(self, res: dict):
        self._busy(False)
        if res.get("error"):
            QMessageBox.critical(self, tr("mvs.plan", "Riprese"), str(res["error"])[:600]); return
        self.scenes = res["result"]
        # 2.5 riuso reale dell'immagine fornita come apertura della scena 0
        base_img = self.proj.get("base_image", "")
        if base_img and os.path.exists(base_img) and self.scenes:
            self.scenes[0]["image_path"] = base_img
        self._save(); self._refresh_sb()
        self.status.setText(tr("mvs.planned", "{n} riprese — modifica, poi genera le immagini").format(n=len(self.scenes)))

    def _gen_images(self):
        todo = [s for s in self.scenes if not (s.get("image_path") and os.path.exists(s["image_path"]))]
        if not todo:
            QMessageBox.information(self, tr("mvs.gen_imgs", "Immagini"), tr("mvs.imgs_done", "Tutte le immagini ci sono.")); return
        self._batch(todo, "image")

    def _gen_videos(self):
        todo = [s for s in self.scenes if s.get("image_path") and not (s.get("video_path") and os.path.exists(s["video_path"]))]
        if not todo:
            QMessageBox.information(self, tr("mvs.gen_vids", "Video"), tr("mvs.vids_todo", "Genera prima le immagini.")); return
        self._batch(todo, "video")

    def _batch(self, todo, kind):
        self._queue = list(todo); self._kind = kind
        self._busy(True, f"{kind} 0/{len(todo)}"); self._next_batch()

    def _next_batch(self):
        if not self._queue:
            self._save(); self._busy(False); self._refresh_sb()
            self.status.setText(tr("mvs.batch_done", "Generazione completata.")); return
        s = self._queue.pop(0)
        if self._kind == "image":
            out = os.path.join(self.out_dir, f"img_{int(s['start_s']*100)}.png"); fn, args = mvs.gen_scene_image, (s, out)
        else:
            out = os.path.join(self.out_dir, f"vid_{s['order']}_{int(s['start_s']*100)}.mp4"); fn, args = mvs.gen_scene_video, (s, out)
        self.status.setText(f"{self._kind} @ {_mmss(s['start_s'])}…")
        self._worker = _Worker(fn, *args)
        self._worker.done.connect(lambda r: (self._save(), self._refresh_sb(), self._next_batch()))
        self._worker.start()

    def _run(self, fn, *args):
        self._busy(True, tr("mvs.working", "Generazione…"))
        self._worker = _Worker(fn, *args)
        self._worker.done.connect(self._on_single); self._worker.start()

    def _on_single(self, res: dict):
        self._busy(False)
        if res.get("error"):
            QMessageBox.critical(self, tr("mvs.title", "Studio"), str(res["error"])[:600]); return
        self._save(); self._refresh_sb()
        if self.sb_list.currentRow() >= 0:
            self._sb_select(self.sb_list.currentRow())

    def _assemble(self):
        a = mvp.active_song(self.pid)
        if not a or not any(s.get("video_path") for s in self.scenes):
            QMessageBox.information(self, tr("mvs.assemble", "Monta"), tr("mvs.no_vids", "Servono canzone attiva e video.")); return
        out = os.path.join(self.out_dir, "music_video.mp4")
        self._busy(True, tr("mvs.assembling", "Montaggio…"))
        self._worker = _Worker(mvs.assemble_scenes, self.scenes, a["file"], out,
                               transition=self.transition.currentData())
        def _fin(r):
            self._busy(False)
            if r.get("error"):
                QMessageBox.critical(self, tr("mvs.assemble", "Monta"), str(r["error"])[:600]); return
            mvp.update_project(self.pid, phase="done")
            self.status.setText(tr("mvs.final", "Videomusicale pronto: {f}").format(f=out))
        self._worker.done.connect(_fin); self._worker.start()
