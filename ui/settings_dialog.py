"""Settings dialog — motori di generazione (nativo) + override modello per nodo loader.

Due sezioni:
  1) **Motore di generazione (nativo)** — sceglie COME vengono caricati i modelli:
     - immagini : comfy-native ↔ diffusers   (consumato da core.img_router)
     - video    : comfy-native ↔ ltx2 ufficiale (consumato da core.ltx2_service.generate)
     Persistito in user_prefs (get_engine/set_engine). È il cuore della migrazione nativa.
  2) **Override modelli (template ComfyUI)** — per il path template legacy: per ogni nodo loader
     puoi forzare un file diverso. Popolato dai file disponibili (GET /models/{folder}).
     Ritorna overrides {node_id: {input: value}} che il template_engine applica al build.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QDialogButtonBox, QLabel, QScrollArea,
    QWidget, QGroupBox,
)

from core import user_prefs
from ui.i18n_manager import tr


# valore pref → (etichetta i18n). L'ordine è quello mostrato nella combo.
_IMAGE_ENGINES = [("comfy", "settings.engine.comfy"), ("diffusers", "settings.engine.diffusers")]
_VIDEO_ENGINES = [("comfy", "settings.engine.comfy"), ("lightricks", "settings.engine.lightricks")]


class SettingsDialog(QDialog):
    def __init__(self, engine, driver, current_overrides: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("settings.window_title"))
        self.resize(680, 620)
        self.engine = engine
        self.driver = driver
        self._combos: dict[str, tuple[str, QComboBox]] = {}  # node -> (input, combo)
        self._overrides = current_overrides or {}

        lay = QVBoxLayout(self)

        # ── 1) Motore di generazione (nativo) ─────────────────────────────────
        lay.addWidget(self._build_native_group())

        # ── 2) Privacy / provenienza ──────────────────────────────────────────
        lay.addWidget(self._build_privacy_group())

        # ── 3) Cache degli embedding di prompt ────────────────────────────────
        lay.addWidget(self._build_embcache_group())

        # ── 4) Override modelli (template) ────────────────────────────────────
        lay.addWidget(self._build_overrides_group(), 1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    # ---------- sezione motori nativi ----------
    def _build_native_group(self) -> QGroupBox:
        grp = QGroupBox(tr("settings.native_group"))
        form = QFormLayout(grp)

        hint = QLabel(tr("settings.native_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow(hint)

        # UN SOLO MOTORE → NIENTE DROPDOWN (richiesta utente): senza ComfyUI le opzioni 'comfy'
        # spariscono; se per una categoria resta un solo motore, la riga non viene costruita.
        # Se spariscono entrambe, l'intero gruppo viene nascosto dal chiamante (grp vuoto).
        try:
            from core.ltx2_service import comfy_available as _ca
            _has_comfy = _ca()
        except Exception:                                    # noqa: BLE001
            _has_comfy = True
        img_engines = [e for e in _IMAGE_ENGINES if _has_comfy or e[0] != "comfy"]
        vid_engines = [e for e in _VIDEO_ENGINES if _has_comfy or e[0] != "comfy"]

        if len(img_engines) > 1:
            self.cb_img_engine = QComboBox()
            for val, key in img_engines:
                self.cb_img_engine.addItem(tr(key), val)
            self._select_by_data(self.cb_img_engine, user_prefs.get_engine("image", "comfy"))
            form.addRow(tr("settings.engine_image"), self.cb_img_engine)

        if len(vid_engines) > 1:
            self.cb_vid_engine = QComboBox()
            for val, key in vid_engines:
                self.cb_vid_engine.addItem(tr(key), val)
            self._select_by_data(self.cb_vid_engine, user_prefs.get_engine("video", "comfy"))
            form.addRow(tr("settings.engine_video"), self.cb_vid_engine)

        if len(img_engines) <= 1 and len(vid_engines) <= 1:
            grp.setVisible(False)                            # nessuna scelta da fare → gruppo via
        return grp

    @staticmethod
    def _select_by_data(combo: QComboBox, value: str):
        idx = combo.findData((value or "").lower())
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ---------- sezione override template ----------
    def _build_privacy_group(self) -> QGroupBox:
        from PySide6.QtWidgets import QCheckBox, QLineEdit
        grp = QGroupBox(tr("settings.privacy_group", "Provenienza"))
        form = QFormLayout(grp)
        self.cb_stego = QCheckBox(tr("settings.stego", "Firma invisibile di provenienza (consigliata)"))
        self.cb_stego.setChecked(bool(user_prefs.get("stego_watermark", True)))
        self.cb_stego.setToolTip(tr("settings.stego.tt",
            "Inserisce una firma invisibile e impercettibile in ogni foto/video generato, che permette di "
            "verificare (con lo strumento ATF Detector) che il contenuto è stato creato con questo software. "
            "Non altera visibilmente l'immagine. Attiva di default."))
        form.addRow(self.cb_stego)
        # Testo custom OPZIONALE nella firma invisibile (max 5 caratteri: limite robusto a screenshot).
        self.ed_stego_text = QLineEdit(str(user_prefs.get("stego_text", "") or ""))
        self.ed_stego_text.setMaxLength(5)
        self.ed_stego_text.setPlaceholderText(tr("settings.stego_text.ph", "vuoto = firma senza testo"))
        self.ed_stego_text.setToolTip(tr("settings.stego_text.tt",
            "Testo custom (max 5 caratteri, es. un tuo codice/handle) impresso invisibilmente e leggibile dal "
            "Detector. Vuoto = firma valida senza testo. Limite di 5 caratteri per resistere a screenshot/"
            "compressione; per testo più lungo usare i metadati del file."))
        form.addRow(tr("settings.stego_text", "Testo firma (max 5)"), self.ed_stego_text)
        # Task 21: dati di generazione (prompt, modelli, parametri) dentro i file immagine/video. Default ON.
        self.cb_embed_meta = QCheckBox(tr("settings.embed_meta",
            "Includi i dati di generazione nei file (prompt, modelli, parametri)"))
        self.cb_embed_meta.setChecked(bool(user_prefs.get("embed_gen_metadata", True)))
        self.cb_embed_meta.setToolTip(tr("settings.embed_meta.tt",
            "Scrive dentro ogni immagine e video generato (PNG/JPEG/WebP/MP4) prompt, prompt negativo, modello, "
            "LoRA, encoder, VAE, seed, passi e gli altri parametri, nel formato letto da Civitai e dagli altri "
            "programmi. ATTENZIONE: chi riceve il file vede il prompt e i parametri. Per condividere senza dati "
            "usa \u00abSalva copia senza metadati\u00bb nella Galleria. Non contiene mai percorsi del tuo computer."))
        form.addRow(self.cb_embed_meta)
        return grp

    def _build_embcache_group(self) -> QGroupBox:
        """Cache degli embedding di prompt LTX. Nessun import di torch qui: core.emb_cache è
        stdlib-only all'import proprio perché il processo GUI non deve creare un contesto CUDA."""
        from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QPushButton, QSpinBox
        from core import emb_cache
        grp = QGroupBox(tr("emb.cache.title", "Cache degli embedding di prompt"))
        form = QFormLayout(grp)
        self.cb_embcache = QCheckBox(tr("emb.cache", "Riusa gli embedding dei prompt già calcolati"))
        self.cb_embcache.setChecked(bool(user_prefs.get("emb_cache.enabled", True)))
        self.cb_embcache.setToolTip(tr("emb.cache.tt",
            "Ogni generazione video ricalcola l'encoding del prompt rileggendo 24 GB di text encoder dal "
            "disco (~13 secondi). Con la cache lo stesso prompt viene riusato: un video lungo a 6 spezzoni "
            "lo calcola una volta invece di sei. Il risultato è identico — è lo stesso tensore. Le voci "
            "contengono il testo del prompt in chiaro e stanno in data/emb_cache."))
        form.addRow(self.cb_embcache)
        self.sp_embcache = QSpinBox()
        self.sp_embcache.setRange(1, 200)
        self.sp_embcache.setSuffix(" GB")
        self.sp_embcache.setValue(max(1, int(round(float(user_prefs.get("emb_cache.mb", 4096)) / 1024))))
        self.sp_embcache.setToolTip(tr("emb.cache.budget.tt",
            "Spazio massimo su disco. Ogni voce pesa circa 24 MB, quindi 4 GB tengono ~170 prompt; "
            "oltre il limite si scartano i meno usati di recente."))
        form.addRow(tr("emb.cache.budget", "Spazio massimo"), self.sp_embcache)
        row = QHBoxLayout()
        st = emb_cache.stats()
        self.lbl_embcache = QLabel(tr("emb.cache.status", "{n} voci · {mb} MB")
                                   .format(n=st["n"], mb=f"{st['mb']:.0f}"))
        row.addWidget(self.lbl_embcache, 1)
        btn = QPushButton(tr("emb.cache.clear", "Svuota"))
        btn.setToolTip(tr("emb.cache.clear.tt",
            "Cancella le voci salvate. Utile se hai sostituito un modello mantenendo lo stesso nome file: "
            "in quel caso la cache non può accorgersene da sola."))
        btn.clicked.connect(self._on_clear_embcache)
        row.addWidget(btn)
        w = QWidget(); w.setLayout(row)
        form.addRow(w)
        return grp

    def _on_clear_embcache(self):
        from PySide6.QtWidgets import QMessageBox
        from core import emb_cache
        n = emb_cache.clear()
        st = emb_cache.stats()
        self.lbl_embcache.setText(tr("emb.cache.status", "{n} voci · {mb} MB")
                                  .format(n=st["n"], mb=f"{st['mb']:.0f}"))
        QMessageBox.information(self, tr("emb.cache.title", "Cache degli embedding di prompt"),
                                tr("emb.cache.cleared", "Rimosse {n} voci.").format(n=n))

    def _build_overrides_group(self) -> QGroupBox:
        grp = QGroupBox(tr("settings.overrides_group"))
        gv = QVBoxLayout(grp)
        intro = QLabel(tr("settings.intro")); intro.setWordWrap(True)
        gv.addWidget(intro)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); form = QFormLayout(inner)
        folder_cache: dict[str, list[str]] = {}
        try:
            loaders = self.engine.loader_nodes()
        except Exception:
            loaders = []
        if not loaders:
            form.addRow(QLabel(tr("settings.no_loaders")))
        for ln in loaders:
            folder = ln["folder"]
            if folder not in folder_cache:
                # FASE D: fallback OFFLINE — con ComfyUI giù (funzionamento nativo normale) i combo
                # override restavano VUOTI; ora la lista arriva dal filesystem (offline_inventory).
                if self.driver.is_up():
                    folder_cache[folder] = self.driver.models_in(folder)
                else:
                    try:
                        from core import offline_inventory
                        folder_cache[folder] = offline_inventory.list_models(folder)
                    except Exception:                    # noqa: BLE001
                        folder_cache[folder] = []
            files = folder_cache[folder]
            combo = QComboBox(); combo.setEditable(True)
            opts = sorted(set(files) | {ln["current"]})
            combo.addItems(opts)
            cur = self._overrides.get(ln["node"], {}).get(ln["input"], ln["current"])
            combo.setCurrentText(cur)
            label = f"{ln['title']} [{ln['node']}] · {folder}"
            form.addRow(label, combo)
            self._combos[ln["node"]] = (ln["input"], combo)
        scroll.setWidget(inner)
        gv.addWidget(scroll, 1)
        return grp

    # ---------- accept ----------
    def _on_accept(self):
        """Persiste subito la scelta dei motori nativi, poi chiude (gli override li legge open_settings)."""
        try:
            if hasattr(self, "cb_img_engine"):           # assenti quando c'è un solo motore (riga nascosta)
                user_prefs.set_engine("image", self.cb_img_engine.currentData())
            if hasattr(self, "cb_vid_engine"):
                user_prefs.set_engine("video", self.cb_vid_engine.currentData())
            if hasattr(self, "cb_stego"):
                user_prefs.set("stego_watermark", self.cb_stego.isChecked())
                user_prefs.set("stego_text", self.ed_stego_text.text().strip())
            if hasattr(self, "cb_embed_meta"):
                user_prefs.set("embed_gen_metadata", self.cb_embed_meta.isChecked())
            if hasattr(self, "cb_embcache"):
                user_prefs.set("emb_cache.enabled", self.cb_embcache.isChecked())
                user_prefs.set("emb_cache.mb", int(self.sp_embcache.value()) * 1024)
        except Exception:
            pass
        self.accept()

    def overrides(self) -> dict:
        out: dict[str, dict] = {}
        try:
            loaders = self.engine.loader_nodes()
        except Exception:
            loaders = []
        for node, (inp, combo) in self._combos.items():
            val = combo.currentText().strip()
            orig = next((l["current"] for l in loaders if l["node"] == node), None)
            if val and val != orig:
                out[node] = {inp: val}
        return out
