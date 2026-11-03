"""Tab «🎭 Scambia» — studio guidato a passi per sostituire i personaggi di un video (1-4 persone, un ruolo ciascuna).

Passi: 1 video -> 2 «Trova persone» (SAM2 + GroundingDINO multi-box; SAM 3.1 quando il caricatore e pronto) con miniatura e nome modificabile ->
3 ruoli (persona del video + foto del nuovo personaggio + nota) -> 4 anteprima della maschera -> 5 genera (una passata H3 per ruolo, in catena,
con ripresa dai clip gia fatti) -> 6 rifiniture (volti e colori: «Migliora qualita -> Video»).

La logica (tracce, ruoli, validazione, passate) e in core/swap_project.py (pura, provata su CPU); qui c'e solo la GUI. I lavori pesanti partono dalla
MainWindow (coda di generazione): il tab emette `request_find` e `request_generate`.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

from core import swap_project as SP
from ui.i18n_manager import tr

_MIN_H = 44                                           # bersagli di tocco >= 44 px


def _btn(text: str, slot=None) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(_MIN_H)
    if slot:
        b.clicked.connect(slot)
    return b


class _RoleRow(QFrame):
    """Un ruolo: persona del video (menu), foto del nuovo personaggio, nota."""

    def __init__(self, n: int, owner: "SwapTab"):
        super().__init__()
        self.owner = owner
        self.n = n
        self.photo = ""
        self.setFrameShape(QFrame.StyledPanel)
        lay = QHBoxLayout(self)
        self.name = QLineEdit(tr("swap.role_name", "Personaggio {n}").format(n=n))
        self.name.setMinimumHeight(_MIN_H)
        self.name.setMaximumWidth(180)
        self.person = QComboBox()
        self.person.setMinimumHeight(_MIN_H)
        self.person.setToolTip(tr("swap.role_person_tt", "Quale persona del video viene sostituita"))
        self.thumb = QLabel()
        self.thumb.setFixedSize(64, 64)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.btn_photo = _btn(tr("swap.photo_btn", "Foto del nuovo personaggio…"), self._pick)
        self.note = QLineEdit()
        self.note.setMinimumHeight(_MIN_H)
        self.note.setPlaceholderText(tr("swap.note_ph", "Nota per il prompt (facoltativa)"))
        for w in (self.name, self.person, self.thumb, self.btn_photo, self.note):
            lay.addWidget(w)
        lay.setStretch(4, 1)

    def _pick(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("swap.photo_btn", "Foto del nuovo personaggio…"), "", "Immagini (*.png *.jpg *.jpeg *.webp)")
        if f:
            self.set_photo(f)

    def set_photo(self, path: str):
        self.photo = path
        pm = QPixmap(path)
        if not pm.isNull():
            self.thumb.setPixmap(pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_people(self, tracks: list):
        cur = self.person.currentData()
        self.person.clear()
        for t in tracks:
            self.person.addItem(t.label, t.id)
        i = self.person.findData(cur)
        self.person.setCurrentIndex(i if i >= 0 else min(self.n - 1, max(0, self.person.count() - 1)))

    def role(self) -> SP.Role:
        tid = self.person.currentData()
        return SP.Role(name=self.name.text().strip() or f"Personaggio {self.n}", track_ids=[int(tid)] if tid is not None else [], photo=self.photo,
                       note=self.note.text().strip())


class SwapTab(QWidget):
    request_find = Signal(dict)              # {video, text, work_dir}
    request_generate = Signal(dict)          # {video, work_dir, passes:[...], h3_profile, short_edge, seed}
    request_open_weights_tab = Signal()
    request_open_quality_tab = Signal()

    def __init__(self):
        super().__init__()
        self.video = ""
        self.work_dir = ""
        self.tracks: list[SP.Track] = []
        self._roles: list[_RoleRow] = []
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)

        # 1 · video
        g1 = QGroupBox(tr("swap.step1", "1 · Il video"))
        l1 = QHBoxLayout(g1)
        self.video_edit = QLineEdit()
        self.video_edit.setMinimumHeight(_MIN_H)
        self.video_edit.setReadOnly(True)
        self.video_edit.setPlaceholderText(tr("swap.video_ph", "Scegli il video in cui sostituire i personaggi"))
        l1.addWidget(self.video_edit, 1)
        l1.addWidget(_btn(tr("swap.video_btn", "Scegli il video…"), self._pick_video))
        lay.addWidget(g1)

        # 2 · trova persone
        g2 = QGroupBox(tr("swap.step2", "2 · Trova le persone"))
        l2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        self.find_text = QLineEdit("person")
        self.find_text.setMinimumHeight(_MIN_H)
        self.find_text.setToolTip(tr("swap.find_text_tt", "Cosa cercare (in inglese): «person» trova le persone."))
        self.max_people = QSpinBox()
        self.max_people.setRange(1, SP.MAX_ROLES)
        self.max_people.setValue(SP.MAX_ROLES)
        self.max_people.setMinimumHeight(_MIN_H)
        self.btn_find = _btn(tr("swap.find_btn", "Trova persone"), self._on_find)
        row.addWidget(self.find_text, 1)
        row.addWidget(QLabel(tr("swap.max_people", "al massimo")))
        row.addWidget(self.max_people)
        row.addWidget(self.btn_find)
        l2.addLayout(row)
        self.people_row = QHBoxLayout()
        pw = QWidget()
        pw.setLayout(self.people_row)
        l2.addWidget(pw)
        self.find_status = QLabel(tr("swap.find_hint", "Il motore di segmentazione si sceglie dal menu «Segmentazione»."))
        self.find_status.setWordWrap(True)
        l2.addWidget(self.find_status)
        lay.addWidget(g2)
        self._name_edits: dict[int, QLineEdit] = {}

        # 3 · ruoli
        g3 = QGroupBox(tr("swap.step3", "3 · Chi diventa chi"))
        self.roles_lay = QVBoxLayout(g3)
        rb = QHBoxLayout()
        self.btn_add_role = _btn(tr("swap.add_role", "+ Aggiungi personaggio"), self._add_role)
        self.btn_del_role = _btn(tr("swap.del_role", "− Togli l'ultimo"), self._del_role)
        rb.addWidget(self.btn_add_role)
        rb.addWidget(self.btn_del_role)
        rb.addStretch(1)
        self.roles_lay.addLayout(rb)
        lay.addWidget(g3)
        self._add_role()

        # 4 · anteprima maschera
        g4 = QGroupBox(tr("swap.step4", "4 · Anteprima della maschera"))
        l4 = QVBoxLayout(g4)
        self.preview = QLabel(tr("swap.preview_ph", "Dopo «Trova persone» vedrai qui le persone che verranno sostituite."))
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(180)
        self.preview.setWordWrap(True)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumHeight(_MIN_H)
        self.slider.valueChanged.connect(self._refresh_preview)
        l4.addWidget(self.preview)
        l4.addWidget(self.slider)
        lay.addWidget(g4)

        # 5 · genera
        g5 = QGroupBox(tr("swap.step5", "5 · Genera"))
        l5 = QVBoxLayout(g5)
        r5 = QHBoxLayout()
        self.profile = QComboBox()
        self.profile.setMinimumHeight(_MIN_H)
        self.profile.setToolTip(tr("swap.profile_tt", "Profilo MiniMax H3 (scelta tua: mai automatica)"))
        self.edge = QComboBox()
        self.edge.setMinimumHeight(_MIN_H)
        for e in (480, 576, 736):
            self.edge.addItem(f"{e}p", e)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(42)
        self.seed.setMinimumHeight(_MIN_H)
        for w in (QLabel(tr("swap.profile", "Profilo H3")), self.profile, QLabel(tr("swap.edge", "Lato corto")), self.edge,
                  QLabel(tr("swap.seed", "Seme")), self.seed):
            r5.addWidget(w)
        r5.addStretch(1)
        l5.addLayout(r5)
        self.btn_go = _btn(tr("swap.go", "Genera (una passata per personaggio)"), self._on_generate)
        l5.addWidget(self.btn_go)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(110)
        l5.addWidget(self.log)
        lay.addWidget(g5)

        # 6 · rifiniture
        g6 = QGroupBox(tr("swap.step6", "6 · Rifiniture"))
        l6 = QHBoxLayout(g6)
        l6.addWidget(QLabel(tr("swap.step6_txt", "Volti e colori si correggono in «Migliora qualità → Video».")), 1)
        l6.addWidget(_btn(tr("swap.open_quality", "Apri Migliora qualità"), lambda: self.request_open_quality_tab.emit()))
        lay.addWidget(g6)
        lay.addStretch(1)
        self.refresh_profiles()

    # ── passi ──────────────────────────────────────────────────────────────────────────────────────
    def refresh_profiles(self):
        from core import video_edit_h3 as vh3
        self.profile.clear()
        for p in vh3.installed_profiles():
            self.profile.addItem(vh3.H3_PROFILE_I18N.get(p, p), p)
        if not self.profile.count():
            self.profile.addItem(tr("swap.no_profile", "nessun profilo H3 installato"), "")

    def set_video(self, path: str):
        self.video = path
        self.video_edit.setText(path)
        self.work_dir = os.path.splitext(path)[0] + "_scambia"
        self.tracks = []
        self._show_people()
        tj = os.path.join(self.work_dir, "tracks.json")
        if os.path.isfile(tj):                              # ricerca gia fatta: la si riusa (niente GPU)
            try:
                self.set_tracks(SP.load_tracks(tj))
                self.find_status.setText(tr("swap.reused", "Persone gia trovate in precedenza: ripresa dalla cartella {d}.").format(d=self.work_dir))
            except Exception:                               # noqa: BLE001
                pass

    def _pick_video(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("swap.video_btn", "Scegli il video…"), "", "Video (*.mp4 *.mov *.mkv *.webm)")
        if f:
            self.set_video(f)

    def _on_find(self):
        if not self.video or not os.path.isfile(self.video):
            self.find_status.setText(tr("swap.err_no_video", "Scegli prima il video."))
            return
        self.find_status.setText(tr("swap.finding", "Cerco le persone… (puo richiedere un minuto)"))
        self.request_find.emit({"video": self.video, "work_dir": self.work_dir, "text": self.find_text.text().strip() or "person",
                               "max_people": int(self.max_people.value())})

    def set_tracks(self, tracks: list):
        self.tracks = list(tracks)
        self._show_people()
        for r in self._roles:
            r.set_people(self.tracks)
        self.slider.setMaximum(max(0, self._frame_count() - 1))
        self._refresh_preview()
        self.find_status.setText(tr("swap.found", "Trovate {n} persone. Dai un nome (facoltativo) e passa ai personaggi.").format(n=len(self.tracks)))

    def _show_people(self):
        while self.people_row.count():
            it = self.people_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._name_edits = {}
        for t in self.tracks:
            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            v = QVBoxLayout(card)
            th = QLabel()
            th.setFixedSize(120, 120)
            th.setAlignment(Qt.AlignCenter)
            if t.thumb and os.path.isfile(t.thumb):
                th.setPixmap(QPixmap(t.thumb).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            ne = QLineEdit(t.label)
            ne.setMinimumHeight(_MIN_H)
            ne.editingFinished.connect(lambda tid=t.id, w=ne: self._rename(tid, w.text()))
            v.addWidget(th)
            v.addWidget(ne)
            self._name_edits[t.id] = ne
            self.people_row.addWidget(card)
        self.people_row.addStretch(1)

    def _rename(self, tid: int, name: str):
        for t in self.tracks:
            if t.id == tid:
                t.label = name.strip() or t.label
        for r in self._roles:
            r.set_people(self.tracks)

    def _add_role(self):
        if len(self._roles) >= SP.MAX_ROLES:
            return
        r = _RoleRow(len(self._roles) + 1, self)
        r.set_people(self.tracks)
        self._roles.append(r)
        self.roles_lay.insertWidget(self.roles_lay.count() - 1, r)
        self.btn_add_role.setEnabled(len(self._roles) < SP.MAX_ROLES)

    def _del_role(self):
        if len(self._roles) <= 1:
            return
        r = self._roles.pop()
        r.deleteLater()
        self.btn_add_role.setEnabled(True)

    def roles(self) -> list[SP.Role]:
        return [r.role() for r in self._roles]

    # API pubblica per i test dal vivo (stessi widget dell'utente, senza finestre di scelta file)
    def set_role_photo(self, i: int, path: str) -> None:
        self._roles[int(i)].set_photo(path)

    def set_role_person(self, i: int, index: int) -> None:
        self._roles[int(i)].person.setCurrentIndex(int(index))

    def set_profile(self, prof: str) -> bool:
        k = self.profile.findData(prof)
        if k >= 0:
            self.profile.setCurrentIndex(k)
        return k >= 0

    # ── anteprima ──────────────────────────────────────────────────────────────────────────────────
    def _frames_dir(self) -> str:
        return os.path.join(self.work_dir, "frames")

    def _frame_count(self) -> int:
        d = self._frames_dir()
        return len([f for f in os.listdir(d) if f.lower().endswith(".png")]) if os.path.isdir(d) else 0

    def _refresh_preview(self, *_):
        d = self._frames_dir()
        if not self.tracks or not os.path.isdir(d):
            return
        import numpy as np
        from PIL import Image
        names = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
        if not names:
            return
        i = min(self.slider.value(), len(names) - 1)
        frame = np.asarray(Image.open(os.path.join(d, names[i])).convert("RGB")).astype(np.float32)
        colors = [(255, 60, 60), (60, 160, 255), (80, 220, 100), (255, 200, 40)]
        chosen = {t for r in self.roles() for t in r.track_ids}
        for t in self.tracks:
            mp = os.path.join(t.mask_dir, names[i])
            if t.id in chosen and os.path.isfile(mp):
                m = np.asarray(Image.open(mp).convert("L").resize((frame.shape[1], frame.shape[0]))) > 127
                c = np.array(colors[(t.id - 1) % 4], np.float32)
                frame[m] = 0.55 * frame[m] + 0.45 * c
        im = Image.fromarray(frame.clip(0, 255).astype("uint8"))
        im.thumbnail((640, 360))
        data = im.tobytes("raw", "RGB")
        from PySide6.QtGui import QImage
        qi = QImage(data, im.width, im.height, im.width * 3, QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qi))

    # ── genera ─────────────────────────────────────────────────────────────────────────────────────
    def build_request(self) -> dict:
        """Richiesta di generazione (o ValueError con il testo per l'utente): ruoli validati, una passata per ruolo."""
        if not self.video:
            raise ValueError(tr("swap.err_no_video", "Scegli prima il video."))
        errs = SP.validate(self.roles(), self.tracks)
        if errs:
            raise ValueError("\n".join(errs))
        prof = self.profile.currentData()
        if not prof:
            raise ValueError(tr("swap.err_profile", "Scegli un profilo H3 di cui hai il DiT Ref2VA installato (tab «Pesi modelli»)."))
        passes = SP.plan_passes(self.roles(), self.tracks, self.work_dir)
        return {"video": self.video, "work_dir": self.work_dir, "h3_profile": prof, "short_edge": int(self.edge.currentData()), "seed": int(self.seed.value()),
                "passes": [{"role": p.role, "mask_dir": p.mask_dir, "photo": p.photo, "note": p.note, "index": p.index} for p in passes]}

    def _on_generate(self):
        try:
            req = self.build_request()
        except ValueError as e:
            self.log.appendPlainText("❌ " + str(e))
            return
        self.log.appendPlainText(tr("swap.start", "[scambia] {n} passata/e: parto…").format(n=len(req["passes"])))
        self.request_generate.emit(req)
