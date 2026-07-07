"""Tab Asset Manager (spec §6): CRUD 4 tipi (actor/prop/location/architecture) + Character/Asset Sheet
via gen_service (multi-ref + PuLID + Kontext), riusa core/lora_coverage per il turnaround. Ricerca LIKE."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QPushButton,
                               QVBoxLayout, QWidget)

from core import lora_coverage
from core.film_studio import db, gen_service
from ui.i18n_manager import tr

_TYPES = [("actor", "tlora.type.person"), ("prop", "tlora.type.object"),
          ("location", "tlora.type.place"), ("architecture", "fs.asset.architecture")]
_COV = {"actor": "person", "prop": "object", "location": "place", "architecture": "place"}


class _SheetWorker(QThread):
    progress = Signal(str)
    done = Signal(dict)

    def __init__(self, asset: dict, photo: str, out_dir: str):
        super().__init__()
        self.asset, self.photo, self.out_dir = asset, photo, out_dir

    def run(self):
        try:
            ctype = _COV.get(self.asset["type"], "person")
            cases = lora_coverage.default_case_ids(ctype, image=True)[:8]
            by = {c: (t, tg) for c, t, tg in lora_coverage.cases_for(ctype)}
            made = []
            for i, cid in enumerate(cases):
                tmpl = by[cid][0].format(subj=self.asset["description"] or self.asset["name"])
                out = str(Path(self.out_dir) / f"{self.asset['name']}_{cid}.png".replace(" ", "_"))
                self.progress.emit(f"{i+1}/{len(cases)}: {cid}")
                gen_service.gen_image(tmpl, out, face_ref=self.photo if ctype == "person" else "",
                                      ref_image="" if ctype == "person" else self.photo, seed=7 + i)
                made.append((cid, out))
            self.done.emit({"images": made})
        except Exception as e:                              # noqa: BLE001
            self.done.emit({"error": str(e)})


class _WardrobeWorker(QThread):
    """Genera il turnaround (6 angoli) del personaggio con un dato abbigliamento."""
    progress = Signal(str)
    done = Signal(dict)

    def __init__(self, char_desc: str, wardrobe_desc: str, face_ref: str, out_dir: str):
        super().__init__()
        self.char_desc, self.ward, self.face, self.out_dir = char_desc, wardrobe_desc, face_ref, out_dir

    def run(self):
        try:
            from core.film_studio.coverage import wardrobe_prompts
            made = []
            prompts = wardrobe_prompts(self.char_desc, self.ward)
            for i, (aid, prompt) in enumerate(prompts):
                out = str(Path(self.out_dir) / f"ward_{aid}.png".replace(" ", "_"))
                self.progress.emit(f"{i+1}/{len(prompts)}: {aid}")
                gen_service.gen_image(prompt, out, face_ref=self.face, seed=7 + i)
                made.append((aid, out))
            self.done.emit({"images": made})
        except Exception as e:                              # noqa: BLE001
            self.done.emit({"error": str(e)})


class AssetManagerTab(QWidget):
    def __init__(self):
        super().__init__()
        self._cur = None
        self._worker = None
        root = QHBoxLayout(self)
        # sinistra: ricerca + lista
        left = QVBoxLayout()
        sr = QHBoxLayout()
        self.search = QLineEdit(); self.search.setPlaceholderText(tr("fs.asset.search", "Cerca (LIKE, AND/OR/NOT)…"))
        self.search.textChanged.connect(self._refresh_list)
        self.ftype = QComboBox(); self.ftype.addItem(tr("fs.asset.all", "Tutti"), "")
        for code, key in _TYPES:
            self.ftype.addItem(tr(key, code), code)
        self.ftype.currentIndexChanged.connect(self._refresh_list)
        sr.addWidget(self.search, 1); sr.addWidget(self.ftype)
        left.addLayout(sr)
        self.lst = QListWidget(); self.lst.currentRowChanged.connect(self._select)
        left.addWidget(self.lst, 1)
        hb = QHBoxLayout()
        b_new = QPushButton(tr("fs.asset.new", "➕ Nuovo")); b_new.clicked.connect(self._new)
        b_del = QPushButton(tr("tlora.del", "🗑")); b_del.clicked.connect(self._del)
        hb.addWidget(b_new); hb.addWidget(b_del); hb.addStretch(1)
        left.addLayout(hb)
        root.addLayout(left, 1)
        # destra: editor
        box = QGroupBox(tr("fs.asset.details", "Dettagli asset"))
        f = QFormLayout(box)
        self.e_type = QComboBox()
        for code, key in _TYPES:
            self.e_type.addItem(tr(key, code), code)
        self.e_name = QLineEdit()
        self.e_desc = QPlainTextEdit(); self.e_desc.setFixedHeight(80)
        self.e_desc.setPlaceholderText(tr("fs.asset.desc_ph", "Descrizione (es. donna 23 anni, capelli neri…) — @ per citare asset"))
        from ui.film_studio.asset_dialog import attach as _attach_at
        _attach_at(self.e_desc)                              # #55 @Reference anche nella descrizione asset
        self.e_tags = QLineEdit(); self.e_tags.setPlaceholderText(tr("fs.asset.tags_ph", "tag, separati, da, virgola"))
        self.e_photo = QLineEdit(); self.e_photo.setReadOnly(True)
        b_ph = QPushButton(tr("tlora.add_photo", "Foto base…")); b_ph.clicked.connect(self._pick_photo)
        ph_row = QHBoxLayout(); ph_row.addWidget(self.e_photo, 1); ph_row.addWidget(b_ph)
        f.addRow(tr("tlora.type", "Tipo"), self.e_type)
        f.addRow(tr("tlora.name", "Nome"), self.e_name)
        f.addRow(tr("fs.asset.desc", "Descrizione"), self.e_desc)
        f.addRow(tr("fs.asset.tags", "Tag"), self.e_tags)
        f.addRow(tr("fs.asset.photo", "Foto base"), ph_row)
        hh = QHBoxLayout()
        b_save = QPushButton(tr("fs.save", "💾 Salva")); b_save.clicked.connect(self._save)
        self.b_sheet = QPushButton(tr("fs.asset.sheet", "🎭 Genera Character Sheet")); self.b_sheet.clicked.connect(self._gen_sheet)
        hh.addWidget(b_save); hh.addStretch(1); hh.addWidget(self.b_sheet)
        f.addRow(hh)
        self.sheet_log = QLabel(""); f.addRow(self.sheet_log)
        # Galleria curatela del set: accetta i buoni ✓ / rigenera SOLO un angolo (non distruttivo)
        from PySide6.QtCore import QSize as _QSize
        from PySide6.QtWidgets import QListWidget as _GLW
        self.gallery = _GLW(); self.gallery.setViewMode(_GLW.IconMode); self.gallery.setIconSize(_QSize(96, 96))
        self.gallery.setResizeMode(_GLW.Adjust); self.gallery.setMaximumHeight(140); self.gallery.setSpacing(6)
        self.gallery.setStyleSheet("QListWidget::item:selected{border:2px solid #4a90d9;}")
        f.addRow(self.gallery)
        self.gallery_empty = QLabel(tr("fs.asset.gallery_empty", "Nessuna immagine — genera il Character Sheet per curarne gli angoli."))
        self.gallery_empty.setStyleSheet("color:#8a93a0; font-style:italic; padding:6px;")
        f.addRow(self.gallery_empty)
        gh = QHBoxLayout()
        self.b_accept = QPushButton(tr("fs.asset.accept", "✓ Accetta/Rimuovi")); self.b_accept.clicked.connect(self._toggle_accept)
        self.b_regen_angle = QPushButton(tr("fs.asset.regen_angle", "↻ Rigenera questo angolo")); self.b_regen_angle.clicked.connect(self._regen_angle)
        self.b_accept.setEnabled(False); self.b_regen_angle.setEnabled(False)
        self.gallery.currentRowChanged.connect(lambda r: (self.b_accept.setEnabled(r >= 0), self.b_regen_angle.setEnabled(r >= 0)))
        gh.addWidget(self.b_accept); gh.addWidget(self.b_regen_angle); gh.addStretch(1)
        f.addRow(gh)
        # Variazioni (età/pettinatura/condizione…) + Abbigliamento (solo attori) — spec §6.1.1/6.1.2
        from PySide6.QtWidgets import QListWidget as _QLW
        f.addRow(QLabel(tr("fs.var.title", "Variazioni / Sub-luoghi")))
        self.var_list = _QLW(); self.var_list.setMaximumHeight(80); f.addRow(self.var_list)
        vh = QHBoxLayout()
        self.var_name = QLineEdit(); self.var_name.setPlaceholderText(tr("fs.var.ph", "es. ferito / sporco / Radura interna"))
        b_var = QPushButton(tr("fs.var.add", "➕ Variazione")); b_var.clicked.connect(self._add_var)
        b_vdel = QPushButton(tr("tlora.del", "🗑")); b_vdel.clicked.connect(self._del_var)
        b_vgen = QPushButton(tr("fs.var.gen", "🎭 Genera set")); b_vgen.clicked.connect(self._gen_var_sheet)
        vh.addWidget(self.var_name, 1); vh.addWidget(b_var); vh.addWidget(b_vgen); vh.addWidget(b_vdel)
        f.addRow(vh)
        self.ward_row = QWidget(); wl = QHBoxLayout(self.ward_row); wl.setContentsMargins(0, 0, 0, 0)
        self.ward_name = QLineEdit(); self.ward_name.setPlaceholderText(tr("fs.ward.ph", "abito (es. abito da sera) — solo attori"))
        b_ward = QPushButton(tr("fs.ward.add", "👗 Abbigliamento")); b_ward.clicked.connect(self._add_wardrobe)
        wl.addWidget(self.ward_name, 1); wl.addWidget(b_ward)
        f.addRow(self.ward_row)
        root.addWidget(box, 2)
        self._refresh_list()

    def _refresh_list(self):
        self.lst.clear()
        items = db.find_assets(self.search.text().strip(), self.ftype.currentData())
        self._items = items
        for a in items:
            self.lst.addItem(f"[{a['type']}] {a['name']}")

    def _refresh_gallery(self):
        """Mostra le immagini dell'asset corrente; accettate = spunta ✓ + sfondo tenue."""
        from PySide6.QtGui import QIcon, QColor
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtWidgets import QListWidgetItem as _Item
        self.gallery.clear()
        imgs = db.query("image", "owner_kind='asset' AND owner_id=?", (self._cur,)) if self._cur else []
        for im in imgs:
            if not im.get("file"):
                continue
            acc = bool(im.get("accepted"))
            it = _Item(QIcon(im["file"]), ("✓ " if acc else "") + (im.get("angle") or "?"))
            it.setData(_Qt.UserRole, im["id"])
            if acc:
                it.setBackground(QColor(30, 70, 40))
            self.gallery.addItem(it)
        has = self.gallery.count() > 0
        self.gallery.setVisible(has); self.gallery_empty.setVisible(not has)

    def _sel_image(self):
        from PySide6.QtCore import Qt as _Qt
        it = self.gallery.currentItem()
        return (it.data(_Qt.UserRole) if it else None), it

    def _toggle_accept(self):
        iid, _ = self._sel_image()
        if not iid:
            return
        im = db.get("image", iid) or {}
        db.update("image", iid, accepted=0 if im.get("accepted") else 1)
        self._refresh_gallery()
        self.sheet_log.setText(tr("fs.asset.accept_done", "Curatela aggiornata."))

    def _regen_angle(self):
        """Rigenera SOLO l'angolo selezionato (mantiene l'angolo), non-distruttivo (snapshot del vecchio)."""
        iid, _ = self._sel_image()
        if not (iid and self._cur):
            return
        im = db.get("image", iid); asset = db.get("asset", self._cur)
        out = im.get("file")
        if not out:
            return
        from core.film_studio import takes
        takes.record_take("asset", self._cur, "sheet", out)   # conserva il tentativo precedente
        from core.film_studio.coverage import turnaround_prompt
        prompt = turnaround_prompt(asset.get("description") or asset["name"], im.get("angle") or "front")
        face = self._face_ref(); seed = int(im.get("seed", 7) or 7)
        self.b_regen_angle.setEnabled(False)
        self.sheet_log.setText(tr("fs.working", "Generazione…"))

        class _AngleWorker(QThread):
            done = Signal(dict)

            def run(self):
                try:
                    gen_service.gen_image(prompt, out, face_ref=face, seed=seed)
                    self.done.emit({"ok": True})
                except Exception as e:                        # noqa: BLE001
                    self.done.emit({"error": str(e)})

        self._angle_worker = _AngleWorker(self)
        self._angle_worker.done.connect(self._on_angle)
        self._angle_worker.start()

    def _face_ref(self) -> str:
        imgs = db.query("image", "owner_kind='asset' AND owner_id=?", (self._cur,)) if self._cur else []
        return next((i["file"] for i in imgs if i.get("file")), self.e_photo.text().strip())

    def _on_angle(self, res: dict):
        self.b_regen_angle.setEnabled(True)
        if res.get("error"):
            self.sheet_log.setText(""); QMessageBox.critical(self, tr("fs.asset.regen_angle", "Rigenera"), str(res["error"])[:500]); return
        self._refresh_gallery()
        self.sheet_log.setText(tr("fs.asset.angle_done", "Angolo rigenerato (il precedente è nella storia take)."))

    def _new(self):
        self._cur = None
        self.e_name.clear(); self.e_desc.clear(); self.e_tags.clear(); self.e_photo.clear()
        if hasattr(self, "gallery"):
            self._refresh_gallery()

    def _select(self, row: int):
        if not (0 <= row < len(getattr(self, "_items", []))):
            return
        a = self._items[row]; self._cur = a["id"]
        self.e_type.setCurrentIndex([c for c, _ in _TYPES].index(a["type"]) if a["type"] in [c for c, _ in _TYPES] else 0)
        self.e_name.setText(a["name"]); self.e_desc.setPlainText(a.get("description") or "")
        self.e_tags.setText(a.get("tags") or "")
        self._refresh_vars()
        self._refresh_gallery()

    def _refresh_vars(self):
        self.var_list.clear()
        if not self._cur:
            return
        self._vars = db.query("asset_variation", "asset_id=?", (self._cur,))
        for v in self._vars:
            wards = db.query("wardrobe", "variation_id=?", (v["id"],))
            wtag = f" · 👗{len(wards)}" if wards else ""
            self.var_list.addItem(f"[{v['kind']}] {v['name']}{wtag}")
        is_actor = db.get("asset", self._cur).get("type") == "actor" if self._cur else False
        self.ward_row.setVisible(is_actor)

    def _add_var(self):
        if not self._cur:
            QMessageBox.information(self, tr("fs.var.title", "Variazione"), tr("fs.asset.save_first", "Salva prima l'asset.")); return
        name = self.var_name.text().strip()
        if not name:
            return
        a = db.get("asset", self._cur)
        kind = "sublocation" if a["type"] in ("location", "architecture") else "variation"
        db.insert("asset_variation", asset_id=self._cur, name=name, kind=kind, description=name)
        self.var_name.clear(); self._refresh_vars()

    def _del_var(self):
        r = self.var_list.currentRow()
        if 0 <= r < len(getattr(self, "_vars", [])):
            vid = self._vars[r]["id"]
            for w in db.query("wardrobe", "variation_id=?", (vid,)):
                db.delete("wardrobe", w["id"])
            db.delete("asset_variation", vid); self._refresh_vars()

    def _gen_var_sheet(self):
        """1.1 Genera il set immagini della variazione selezionata (asset + descrizione variazione)."""
        r = self.var_list.currentRow()
        if not self._cur or not (0 <= r < len(getattr(self, "_vars", []))):
            QMessageBox.information(self, tr("fs.var.gen", "Set"), tr("fs.var.need_sel", "Seleziona una variazione.")); return
        a = dict(db.get("asset", self._cur))
        var = self._vars[r]
        a["description"] = (a.get("description") or a["name"]) + ", " + (var.get("description") or var["name"])
        out_dir = str(db.ROOT / "assets" / f"{a['id']}_{a['name']}_{var['name']}".replace(" ", "_"))
        from pathlib import Path as _P
        _P(out_dir).mkdir(parents=True, exist_ok=True)
        self.b_sheet.setEnabled(False); self.sheet_log.setText(tr("fs.working", "Generazione…"))
        self._var_for = var["id"]
        self._worker = _SheetWorker(a, self.e_photo.text().strip(), out_dir)
        self._worker.progress.connect(lambda s: self.sheet_log.setText(s))
        self._worker.done.connect(self._on_var_sheet)
        self._worker.start()

    def _on_var_sheet(self, res: dict):
        self.b_sheet.setEnabled(True)
        if res.get("error"):
            self.sheet_log.setText(""); QMessageBox.critical(self, tr("fs.var.gen", "Set"), str(res["error"])[:500]); return
        for cid, path in res.get("images", []):
            db.insert("image", owner_kind="variation", owner_id=getattr(self, "_var_for", 0),
                      angle=cid, repr=cid, file=path, seed=7, prompt="", version=1)
        self.sheet_log.setText(tr("fs.asset.sheet_done", "Sheet: {n} immagini").format(n=len(res.get("images", []))))
        self._refresh_vars()

    def _add_wardrobe(self):
        r = self.var_list.currentRow()
        if not (0 <= r < len(getattr(self, "_vars", []))):
            QMessageBox.information(self, tr("fs.ward.add", "Abbigliamento"), tr("fs.ward.need_var", "Seleziona prima una variazione.")); return
        name = self.ward_name.text().strip()
        if not name:
            return
        wid = db.insert("wardrobe", variation_id=self._vars[r]["id"], name=name, description=name, images="")
        self.ward_name.clear(); self._refresh_vars()
        # genera il turnaround 6-angoli del personaggio con questo abbigliamento (prima solo metadato)
        asset = db.get("asset", self._cur) if self._cur else None
        if asset:
            imgs = db.query("image", "owner_kind='asset' AND owner_id=?", (self._cur,))
            face = next((i["file"] for i in imgs if i.get("file")), self.e_photo.text().strip())
            out_dir = str(db.ROOT / "assets" / f"{asset['id']}_{asset['name']}".replace(" ", "_") / f"ward_{wid}")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            self.sheet_log.setText(tr("fs.working", "Generazione…"))
            self._ward_worker = _WardrobeWorker(asset.get("description") or asset["name"], name, face, out_dir)
            self._ward_worker.progress.connect(lambda s: self.sheet_log.setText(s))
            self._ward_worker.done.connect(lambda res, _w=wid: self._on_wardrobe(res, _w))
            self._ward_worker.start()

    def _on_wardrobe(self, res: dict, wid: int):
        if res.get("error"):
            self.sheet_log.setText(""); QMessageBox.critical(self, tr("fs.ward.add", "Abbigliamento"), str(res["error"])[:600]); return
        files = []
        for aid, path in res.get("images", []):
            db.insert("image", owner_kind="wardrobe", owner_id=wid, angle=aid, repr=aid, file=path, seed=7, prompt="", version=1)
            files.append(path)
        db.update("wardrobe", wid, images="|".join(files))
        self._refresh_vars()
        self.sheet_log.setText(tr("fs.ward.done", "Abbigliamento: {n} angoli generati").format(n=len(files)))

    def _pick_photo(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("fs.asset.photo", "Foto base"), "", "Immagini (*.png *.jpg *.jpeg *.webp)")
        if p:
            self.e_photo.setText(p)

    def _save(self):
        if not self.e_name.text().strip():
            QMessageBox.information(self, tr("fs.save", "Salva"), tr("tlora.need_subject", "Serve un nome.")); return
        fields = {"type": self.e_type.currentData(), "name": self.e_name.text().strip(),
                  "description": self.e_desc.toPlainText().strip(), "tags": self.e_tags.text().strip()}
        if self._cur:
            db.update("asset", self._cur, **fields)
        else:
            self._cur = db.insert("asset", scope="project", **fields)
        self._refresh_list()
        QMessageBox.information(self, tr("fs.save", "Salva"), tr("fs.asset.saved", "Asset salvato."))

    def _del(self):
        if self._cur:
            db.delete("asset", self._cur); self._cur = None; self._refresh_list()

    def _gen_sheet(self):
        if not self._cur:
            QMessageBox.information(self, tr("fs.asset.sheet", "Sheet"), tr("fs.asset.save_first", "Salva prima l'asset.")); return
        asset = db.get("asset", self._cur)
        out_dir = str(db.ROOT / "assets" / f"{asset['id']}_{asset['name']}".replace(" ", "_"))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.b_sheet.setEnabled(False); self.sheet_log.setText(tr("fs.working", "Generazione…"))
        self._worker = _SheetWorker(asset, self.e_photo.text().strip(), out_dir)
        self._worker.progress.connect(lambda s: self.sheet_log.setText(s))
        self._worker.done.connect(self._on_sheet)
        self._worker.start()

    def _on_sheet(self, res: dict):
        self.b_sheet.setEnabled(True)
        if res.get("error"):
            self.sheet_log.setText("")
            QMessageBox.critical(self, tr("fs.asset.sheet", "Sheet"), str(res["error"])[:600]); return
        for cid, path in res.get("images", []):
            db.insert("image", owner_kind="asset", owner_id=self._cur, angle=cid, repr=cid,
                      file=path, seed=7, prompt="", version=1)
        self._refresh_gallery()
        self.sheet_log.setText(tr("fs.asset.sheet_done", "Sheet: {n} immagini — curala qui sotto (Accetta/Rigenera angolo).").format(n=len(res.get("images", []))))
