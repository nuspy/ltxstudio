"""Music Video Studio — workflow INTERATTIVO del videomusicale (spec ampliata utente).

Fasi: canzone (o mp3 esistente) → STORYBOARD editabile (elenco riprese: start mm:ss, durata auto,
immagine di apertura, prompt immagine, prompt video) con rigenera/sostituisci immagine, riordina,
aggiungi (in fondo o tra due), cancella → genera immagini → genera video → ELENCO VIDEO editabile
(rigenera col modulo video, riordina, aggiungi, cancella) → montaggio finale + audio. Zero ComfyUI.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
                               QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from core import music_video_service as mvs
from ui.i18n_manager import tr


def _mmss(s: float) -> str:
    s = max(0, int(s)); return f"{s // 60:02d}:{s % 60:02d}"


class _Worker(QThread):
    progress = Signal(str)
    done = Signal(dict)

    def __init__(self, fn, *args):
        super().__init__()
        self._fn, self._args = fn, args

    def run(self):
        try:
            self.done.emit({"result": self._fn(*self._args, progress=self.progress.emit)})
        except TypeError:
            try:
                self.done.emit({"result": self._fn(*self._args)})
            except Exception as e:                          # noqa: BLE001
                self.done.emit({"error": str(e)})
        except Exception as e:                              # noqa: BLE001
            self.done.emit({"error": str(e)})


class MusicVideoStudio(QWidget):
    def __init__(self, *, song_path: str, lyrics: str, image_hint: str = "", scene_seconds: float = 4.0,
                 video_style: str = "", out_dir: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("mvs.title", "🎬 Music Video Studio"))
        self.resize(1050, 740)
        self.song_path = song_path
        self.lyrics = lyrics
        self.image_hint = image_hint
        self.scene_seconds = scene_seconds
        self.video_style = video_style
        self.out_dir = out_dir or str(Path(song_path).parent / "mvstudio")
        os.makedirs(self.out_dir, exist_ok=True)
        self.scenes: list[dict] = []
        self._worker = None
        self.song_dur = mvs._dur(song_path)

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.btn_plan = QPushButton(tr("mvs.plan", "1) Crea elenco riprese (LLM)")); self.btn_plan.clicked.connect(self._plan)
        self.btn_imgs = QPushButton(tr("mvs.gen_imgs", "2) Genera immagini")); self.btn_imgs.clicked.connect(self._gen_images)
        self.btn_vids = QPushButton(tr("mvs.gen_vids", "3) Genera video")); self.btn_vids.clicked.connect(self._gen_videos)
        self.btn_assemble = QPushButton(tr("mvs.assemble", "4) Monta + audio")); self.btn_assemble.clicked.connect(self._assemble)
        for b in (self.btn_plan, self.btn_imgs, self.btn_vids, self.btn_assemble):
            bar.addWidget(b)
        bar.addStretch(1)
        root.addLayout(bar)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._storyboard_tab(), tr("mvs.tab_sb", "Riprese / Immagini"))
        self.tabs.addTab(self._video_tab(), tr("mvs.tab_vid", "Video"))
        root.addWidget(self.tabs, 1)
        self.status = QLabel(f"{tr('mvs.song', 'Canzone')}: {Path(song_path).name} · {_mmss(self.song_dur)}")
        root.addWidget(self.status)

    # ───────────── Fase Storyboard/Immagini ─────────────
    def _storyboard_tab(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w)
        split = QSplitter(Qt.Horizontal)
        # lista scene
        left = QWidget(); lv = QVBoxLayout(left)
        self.sb_list = QListWidget()
        self.sb_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sb_list.currentRowChanged.connect(self._sb_select)
        lv.addWidget(self.sb_list, 1)
        rb = QHBoxLayout()
        for txt, fn in ((tr("mvs.up", "↑"), self._move_up), (tr("mvs.down", "↓"), self._move_down),
                        (tr("mvs.add", "➕ fine"), self._add_end), (tr("mvs.add_after", "➕ dopo"), self._add_after),
                        (tr("tlora.del", "🗑"), self._del)):
            b = QPushButton(txt); b.clicked.connect(fn); rb.addWidget(b)
        lv.addLayout(rb)
        split.addWidget(left)
        # dettaglio scena
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
        self.img_prompt = QPlainTextEdit(); self.img_prompt.setFixedHeight(56); dv.addWidget(self.img_prompt)
        ih = QHBoxLayout()
        b_reimg = QPushButton(tr("mvs.regen_img", "🖼 Rigenera immagine")); b_reimg.clicked.connect(self._regen_image)
        b_repl = QPushButton(tr("mvs.replace_img", "📁 Sostituisci…")); b_repl.clicked.connect(self._replace_image)
        ih.addWidget(b_reimg); ih.addWidget(b_repl); ih.addStretch(1)
        dv.addLayout(ih)
        dv.addWidget(QLabel(tr("mvs.vid_prompt", "Prompt video")))
        self.vid_prompt = QPlainTextEdit(); self.vid_prompt.setFixedHeight(56); dv.addWidget(self.vid_prompt)
        b_apply = QPushButton(tr("mvs.apply", "💾 Applica modifiche")); b_apply.clicked.connect(self._apply_detail)
        dv.addWidget(b_apply)
        split.addWidget(det)
        split.setSizes([380, 620])
        h.addWidget(split)
        return w

    # ───────────── Fase Video ─────────────
    def _video_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        self.vid_list = QListWidget(); v.addWidget(self.vid_list, 1)
        rb = QHBoxLayout()
        for txt, fn in ((tr("mvs.up", "↑"), self._vmove_up), (tr("mvs.down", "↓"), self._vmove_down),
                        (tr("mvs.regen_vid", "🎬 Rigenera video"), self._regen_video),
                        (tr("tlora.del", "🗑"), self._vdel)):
            b = QPushButton(txt); b.clicked.connect(fn); rb.addWidget(b)
        rb.addStretch(1)
        v.addLayout(rb)
        return w

    # ───────────── refresh liste ─────────────
    def _refresh_sb(self):
        cur = self.sb_list.currentRow()
        self.sb_list.clear()
        mvs.recompute_lengths(self.scenes, self.song_dur)
        for s in self.scenes:
            img = "🖼" if s.get("image_path") and os.path.exists(s["image_path"]) else "·"
            self.sb_list.addItem(f"{img} {_mmss(s['start_s'])}  ({s['length_s']:.1f}s)  {s['video_prompt'][:48]}")
        if 0 <= cur < self.sb_list.count():
            self.sb_list.setCurrentRow(cur)
        self._refresh_vid()

    def _refresh_vid(self):
        self.vid_list.clear()
        for s in sorted(self.scenes, key=lambda x: x["order"]):
            mark = "🎥" if s.get("video_path") and os.path.exists(s["video_path"]) else "·"
            self.vid_list.addItem(f"{mark} {_mmss(s['start_s'])}  ({s['length_s']:.1f}s)  {s['video_prompt'][:48]}")

    def _cur(self) -> dict | None:
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
        self._refresh_sb()

    def _start_changed(self, val):
        s = self._cur()
        if s:
            s["start_s"] = float(val); self._refresh_sb()

    def _move_up(self):
        r = self.sb_list.currentRow()
        if r > 0:
            self.scenes[r]["start_s"], self.scenes[r - 1]["start_s"] = self.scenes[r - 1]["start_s"], self.scenes[r]["start_s"]
            self._refresh_sb(); self.sb_list.setCurrentRow(r - 1)

    def _move_down(self):
        r = self.sb_list.currentRow()
        if 0 <= r < len(self.scenes) - 1:
            self.scenes[r]["start_s"], self.scenes[r + 1]["start_s"] = self.scenes[r + 1]["start_s"], self.scenes[r]["start_s"]
            self._refresh_sb(); self.sb_list.setCurrentRow(r + 1)

    def _new_scene(self, start_s: float) -> dict:
        return {"order": len(self.scenes), "start_s": round(start_s, 2), "length_s": 1.0,
                "video_prompt": self.video_style or "cinematic shot", "img_prompt": (self.video_style or "cinematic") + ", still frame",
                "image_path": "", "video_path": ""}

    def _add_end(self):
        last = max((s["start_s"] for s in self.scenes), default=0.0)
        self.scenes.append(self._new_scene(min(self.song_dur - 0.5, last + self.scene_seconds)))
        self._refresh_sb()

    def _add_after(self):
        s = self._cur()
        if not s:
            return self._add_end()
        nxt = min((x["start_s"] for x in self.scenes if x["start_s"] > s["start_s"]), default=self.song_dur)
        self.scenes.append(self._new_scene((s["start_s"] + nxt) / 2))
        self._refresh_sb()

    def _del(self):
        r = self.sb_list.currentRow()
        if 0 <= r < len(self.scenes):
            self.scenes.pop(r); self._refresh_sb()

    def _regen_image(self):
        s = self._cur()
        if not s:
            return
        s["img_prompt"] = self.img_prompt.toPlainText().strip()
        out = os.path.join(self.out_dir, f"img_{int(s['start_s']*100)}.png")
        self._run(mvs.gen_scene_image, s, out, label=tr("mvs.regen_img", "Rigenera immagine"))

    def _replace_image(self):
        s = self._cur()
        if not s:
            return
        p, _ = QFileDialog.getOpenFileName(self, tr("mvs.replace_img", "Sostituisci immagine"), "", "Immagini (*.png *.jpg *.jpeg *.webp)")
        if p:
            s["image_path"] = p; self._refresh_sb(); self._sb_select(self.sb_list.currentRow())

    # ───────────── editing video ─────────────
    def _vmove_up(self):
        r = self.vid_list.currentRow()
        ordered = sorted(self.scenes, key=lambda x: x["order"])
        if r > 0:
            ordered[r]["order"], ordered[r - 1]["order"] = ordered[r - 1]["order"], ordered[r]["order"]
            self._refresh_vid(); self.vid_list.setCurrentRow(r - 1)

    def _vmove_down(self):
        r = self.vid_list.currentRow()
        ordered = sorted(self.scenes, key=lambda x: x["order"])
        if 0 <= r < len(ordered) - 1:
            ordered[r]["order"], ordered[r + 1]["order"] = ordered[r + 1]["order"], ordered[r]["order"]
            self._refresh_vid(); self.vid_list.setCurrentRow(r + 1)

    def _vdel(self):
        r = self.vid_list.currentRow()
        ordered = sorted(self.scenes, key=lambda x: x["order"])
        if 0 <= r < len(ordered):
            self.scenes.remove(ordered[r]); self._refresh_sb()

    def _regen_video(self):
        r = self.vid_list.currentRow()
        ordered = sorted(self.scenes, key=lambda x: x["order"])
        if not (0 <= r < len(ordered)):
            return
        s = ordered[r]
        if not s.get("image_path"):
            QMessageBox.information(self, tr("mvs.regen_vid", "Video"), tr("mvs.need_img", "Genera prima l'immagine.")); return
        out = os.path.join(self.out_dir, f"vid_{s['order']}_{int(s['start_s']*100)}.mp4")
        self._run(mvs.gen_scene_video, s, out, label=tr("mvs.regen_vid", "Rigenera video"))

    # ───────────── azioni globali (background) ─────────────
    def _plan(self):
        self._busy(True, tr("mvs.plan", "Elenco riprese…"))
        self._worker = _Worker(mvs.plan_scenes)
        self._worker._args = ()
        self._worker._fn = lambda progress=None: mvs.plan_scenes(
            lyrics=self.lyrics, song_path=self.song_path, scene_seconds=self.scene_seconds,
            video_style=self.video_style, image_hint=self.image_hint, progress=progress)
        self._worker.progress.connect(lambda m: self.status.setText(m))
        self._worker.done.connect(self._on_plan)
        self._worker.start()

    def _on_plan(self, res: dict):
        self._busy(False)
        if res.get("error"):
            QMessageBox.critical(self, tr("mvs.plan", "Riprese"), str(res["error"])[:600]); return
        self.scenes = res["result"]; self._refresh_sb()
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

    def _batch(self, todo: list, kind: str):
        self._queue = list(todo); self._kind = kind
        self._busy(True, f"0/{len(todo)}")
        self._next_batch()

    def _next_batch(self):
        if not self._queue:
            self._busy(False); self._refresh_sb()
            self.status.setText(tr("mvs.batch_done", "Generazione completata.")); return
        s = self._queue.pop(0)
        if self._kind == "image":
            out = os.path.join(self.out_dir, f"img_{int(s['start_s']*100)}.png")
            fn, args = mvs.gen_scene_image, (s, out)
        else:
            out = os.path.join(self.out_dir, f"vid_{s['order']}_{int(s['start_s']*100)}.mp4")
            fn, args = mvs.gen_scene_video, (s, out)
        self.status.setText(f"{self._kind} @ {_mmss(s['start_s'])}…")
        self._worker = _Worker(fn, *args)
        self._worker.done.connect(lambda r: (self._refresh_sb(), self._next_batch()))
        self._worker.start()

    def _run(self, fn, *args, label=""):
        self._busy(True, label)
        self._worker = _Worker(fn, *args)
        self._worker.done.connect(self._on_single)
        self._worker.start()

    def _on_single(self, res: dict):
        self._busy(False)
        if res.get("error"):
            QMessageBox.critical(self, tr("mvs.title", "Studio"), str(res["error"])[:600]); return
        self._refresh_sb(); self._sb_select(self.sb_list.currentRow())

    def _assemble(self):
        if not any(s.get("video_path") for s in self.scenes):
            QMessageBox.information(self, tr("mvs.assemble", "Monta"), tr("mvs.no_vids", "Genera prima i video.")); return
        out = os.path.join(self.out_dir, "music_video.mp4")
        self._run(mvs.assemble_scenes, self.scenes, self.song_path, out, label=tr("mvs.assemble", "Montaggio…"))
        self._final = out

    def _busy(self, on: bool, msg: str = ""):
        for b in (self.btn_plan, self.btn_imgs, self.btn_vids, self.btn_assemble):
            b.setEnabled(not on)
        if msg:
            self.status.setText(msg)
