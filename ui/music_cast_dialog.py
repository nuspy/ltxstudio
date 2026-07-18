"""Editor del CAST di un videomusicale — personaggi ricorrenti col volto coerente tra i clip.

Riusa lo stesso apparato del resto dell'app: il cast è la lista `[{name, role, instrument, ref_image,
instrument_image, frequency}]` salvata nel progetto (mv_projects.set_cast), il ritratto di riferimento è
generato col motore immagine nativo (music_video_service.gen_portrait → native_driver), e la coerenza vera
nei clip avviene poi in gen_scene_image(coherence_engine='krea2_edit') usando questi ref. Concettualmente è
l'`asset_manager` di Cinema Studio, in versione compatta per il tab Musica.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget)

from core import music_video_service as mvs, mv_projects as mvp
from ui.i18n_manager import tr

ROLES = [("singer", "Cantante"), ("musician", "Musicista"), ("extra", "Comparsa")]


class _GenWorker(QThread):
    done = Signal(dict)

    def __init__(self, prompt, out, arch, model):
        super().__init__()
        self._a = (prompt, out, arch, model)

    def run(self):
        prompt, out, arch, model = self._a
        try:
            self.done.emit({"result": mvs.gen_portrait(prompt, out, arch=arch, model=model)})
        except Exception as e:                               # noqa: BLE001
            self.done.emit({"error": str(e)})


class MusicCastDialog(QDialog):
    def __init__(self, pid: int, media_cfg: dict | None = None, parent=None):
        super().__init__(parent)
        self.pid = pid
        self.media_cfg = media_cfg or {}
        self.setWindowTitle(tr("mvs.cast.title", "🎭 Cast del videomusicale"))
        self.resize(720, 520)
        self.cast: list[dict] = mvp.get_cast(pid)
        self._worker = None

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(QLabel(tr("mvs.cast.list", "Personaggi ricorrenti")))
        self.lst = QListWidget()
        self.lst.currentRowChanged.connect(self._select)
        left.addWidget(self.lst, 1)
        lb = QHBoxLayout()
        b_add = QPushButton(tr("mvs.cast.add", "➕ Aggiungi")); b_add.clicked.connect(self._add)
        b_del = QPushButton(tr("tlora.del", "🗑")); b_del.clicked.connect(self._del)
        lb.addWidget(b_add); lb.addWidget(b_del)
        left.addLayout(lb)
        root.addLayout(left, 1)

        # dettaglio personaggio
        self.det = QWidget(); dv = QVBoxLayout(self.det)
        self.prev = QLabel(); self.prev.setFixedHeight(200); self.prev.setAlignment(Qt.AlignCenter)
        self.prev.setStyleSheet("border:1px solid #555;")
        dv.addWidget(self.prev)
        dv.addWidget(QLabel(tr("mvs.cast.name", "Nome (@Nome nei prompt)")))
        self.f_name = QLineEdit(); self.f_name.textEdited.connect(self._apply)
        dv.addWidget(self.f_name)
        rr = QHBoxLayout()
        rr.addWidget(QLabel(tr("mvs.cast.role", "Ruolo")))
        self.f_role = QComboBox()
        for key, lbl in ROLES:
            self.f_role.addItem(tr(f"mvs.cast.role.{key}", lbl), key)
        self.f_role.currentIndexChanged.connect(self._apply)
        rr.addWidget(self.f_role, 1)
        dv.addLayout(rr)
        dv.addWidget(QLabel(tr("mvs.cast.instrument", "Strumento (musicisti) — sempre nel prompt")))
        self.f_instr = QLineEdit(); self.f_instr.setPlaceholderText(tr("mvs.cast.instrument_ph",
            "es. chitarra elettrica, violino, batteria")); self.f_instr.textEdited.connect(self._apply)
        dv.addWidget(self.f_instr)
        dv.addWidget(QLabel(tr("mvs.cast.desc", "Aspetto (per generare il ritratto)")))
        self.f_desc = QLineEdit(); self.f_desc.setPlaceholderText(tr("mvs.cast.desc_ph",
            "es. donna ~30 anni, capelli rossi ricci, giacca di pelle")); self.f_desc.textEdited.connect(self._apply)
        dv.addWidget(self.f_desc)
        # riferimenti (identità + strumento)
        r1 = QHBoxLayout()
        self.f_ref = QLineEdit(); self.f_ref.setPlaceholderText(tr("mvs.cast.ref_ph", "Foto identità (volto)"))
        self.f_ref.textEdited.connect(self._apply)
        b_ref = QPushButton("📁"); b_ref.clicked.connect(lambda: self._pick(self.f_ref))
        b_gen = QPushButton(tr("mvs.cast.gen", "✨ Genera")); b_gen.clicked.connect(self._gen_portrait)
        r1.addWidget(self.f_ref, 1); r1.addWidget(b_ref); r1.addWidget(b_gen)
        dv.addLayout(r1)
        r2 = QHBoxLayout()
        self.f_instr_img = QLineEdit(); self.f_instr_img.setPlaceholderText(
            tr("mvs.cast.instr_img_ph", "Foto strumento (opz., 2° riferimento)"))
        self.f_instr_img.textEdited.connect(self._apply)
        b_ii = QPushButton("📁"); b_ii.clicked.connect(lambda: self._pick(self.f_instr_img))
        r2.addWidget(self.f_instr_img, 1); r2.addWidget(b_ii)
        dv.addLayout(r2)
        # frequenza
        fr = QHBoxLayout()
        fr.addWidget(QLabel(tr("mvs.cast.freq", "Frequenza apparizione")))
        self.f_freq = QSlider(Qt.Horizontal); self.f_freq.setRange(0, 100); self.f_freq.setValue(60)
        self.f_freq_lbl = QLabel("60%")
        self.f_freq.valueChanged.connect(lambda v: (self.f_freq_lbl.setText(f"{v}%"), self._apply()))
        fr.addWidget(self.f_freq, 1); fr.addWidget(self.f_freq_lbl)
        dv.addLayout(fr)
        self.status = QLabel(""); dv.addWidget(self.status)
        dv.addStretch(1)
        bb = QHBoxLayout()
        b_ok = QPushButton(tr("common.save", "Salva")); b_ok.clicked.connect(self._save_close)
        b_cancel = QPushButton(tr("common.cancel", "Annulla")); b_cancel.clicked.connect(self.reject)
        bb.addStretch(1); bb.addWidget(b_cancel); bb.addWidget(b_ok)
        dv.addLayout(bb)
        root.addWidget(self.det, 2)

        self._refresh()
        self.det.setEnabled(False)

    # ───────────── model ─────────────
    def _member(self) -> dict | None:
        r = self.lst.currentRow()
        return self.cast[r] if 0 <= r < len(self.cast) else None

    def _refresh(self):
        self.lst.blockSignals(True)
        self.lst.clear()
        for c in self.cast:
            role = dict(ROLES).get(c.get("role", ""), c.get("role", ""))
            instr = f" · {c['instrument']}" if c.get("instrument") else ""
            self.lst.addItem(f"@{c.get('name','?')} — {role}{instr} ({int(c.get('frequency',60))}%)")
        self.lst.blockSignals(False)

    def _select(self, r: int):
        c = self._member()
        self.det.setEnabled(c is not None)
        if not c:
            return
        for w in (self.f_name, self.f_role, self.f_instr, self.f_desc, self.f_ref, self.f_instr_img, self.f_freq):
            w.blockSignals(True)
        self.f_name.setText(c.get("name", ""))
        idx = self.f_role.findData(c.get("role", "singer"))
        self.f_role.setCurrentIndex(idx if idx >= 0 else 0)
        self.f_instr.setText(c.get("instrument", ""))
        self.f_desc.setText(c.get("description", ""))
        self.f_ref.setText(c.get("ref_image", ""))
        self.f_instr_img.setText(c.get("instrument_image", ""))
        self.f_freq.setValue(int(c.get("frequency", 60)))
        self.f_freq_lbl.setText(f"{int(c.get('frequency',60))}%")
        for w in (self.f_name, self.f_role, self.f_instr, self.f_desc, self.f_ref, self.f_instr_img, self.f_freq):
            w.blockSignals(False)
        self._show_prev(c.get("ref_image", ""))

    def _show_prev(self, path: str):
        if path and os.path.exists(path):
            self.prev.setPixmap(QPixmap(path).scaledToHeight(200, Qt.SmoothTransformation))
        else:
            self.prev.setText(tr("mvs.cast.no_ref", "(nessun ritratto)"))

    def _apply(self, *_):
        c = self._member()
        if not c:
            return
        c["name"] = self.f_name.text().strip()
        c["role"] = self.f_role.currentData()
        c["instrument"] = self.f_instr.text().strip()
        c["description"] = self.f_desc.text().strip()
        c["ref_image"] = self.f_ref.text().strip()
        c["instrument_image"] = self.f_instr_img.text().strip()
        c["frequency"] = int(self.f_freq.value())
        r = self.lst.currentRow()
        role = dict(ROLES).get(c.get("role", ""), c.get("role", ""))
        instr = f" · {c['instrument']}" if c.get("instrument") else ""
        if 0 <= r < self.lst.count():
            self.lst.item(r).setText(f"@{c.get('name','?')} — {role}{instr} ({int(c.get('frequency',60))}%)")

    def _add(self):
        self.cast.append({"name": f"Attore{len(self.cast)+1}", "role": "singer", "instrument": "",
                          "description": "", "ref_image": "", "instrument_image": "", "frequency": 60})
        self._refresh(); self.lst.setCurrentRow(len(self.cast) - 1)

    def _del(self):
        r = self.lst.currentRow()
        if 0 <= r < len(self.cast):
            self.cast.pop(r); self._refresh()
            self.det.setEnabled(bool(self.cast))

    def _pick(self, field: QLineEdit):
        p, _ = QFileDialog.getOpenFileName(self, tr("mvs.cast.pick", "Scegli immagine"), "",
                                           "Images (*.png *.jpg *.jpeg *.webp)")
        if p:
            field.setText(p); self._apply()
            if field is self.f_ref:
                self._show_prev(p)

    def _gen_portrait(self):
        c = self._member()
        if not c:
            return
        if not (c.get("name") or "").strip():
            QMessageBox.information(self, tr("mvs.cast.title", "Cast"),
                                    tr("mvs.cast.need_name", "Dai prima un nome al personaggio.")); return
        desc = (c.get("description") or "").strip() or f"a person named {c['name']}"
        instr = (c.get("instrument") or "").strip()
        extra = f", holding a {instr}" if instr else ""
        prompt = (f"cinematic photorealistic headshot portrait of {desc}{extra}, neutral expression, "
                  f"soft studio lighting, plain background, sharp focus")
        out = str(mvp.project_dir(self.pid) / "cast" / f"{c['name']}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        self.status.setText(tr("mvs.cast.gen_run", "Genero il ritratto…"))
        self.setEnabled(False)
        arch = self.media_cfg.get("image_arch", "flux") or "flux"
        model = self.media_cfg.get("image_model", "") or None
        self._worker = _GenWorker(prompt, out, arch, model)
        self._worker.done.connect(self._on_portrait)
        self._worker.start()

    def _on_portrait(self, res: dict):
        self.setEnabled(True)
        if res.get("error"):
            self.status.setText("")
            QMessageBox.critical(self, tr("mvs.cast.title", "Cast"), str(res["error"])[:500]); return
        path = res["result"]
        c = self._member()
        if c:
            c["ref_image"] = path
            self.f_ref.setText(path); self._show_prev(path)
        self.status.setText(tr("mvs.cast.gen_ok", "Ritratto generato."))

    def _save_close(self):
        self.cast = [c for c in self.cast if (c.get("name") or "").strip()]
        mvp.set_cast(self.pid, self.cast)
        self.accept()
