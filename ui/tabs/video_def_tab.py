"""Tab "Definizione video": restore + super-risoluzione + (opz.) aumento FPS.

Carica in automatico l'ultimo video generato; permette di sceglierne un altro. Motori:
SeedVR2 (consigliato, AI video), ESRGAN per-fotogramma (veloce), FlashVSR (se installato).
Opzione RIFE per aumentare gli FPS. Vedi core/video_enhance_pipeline.py.
Quarto motore: H3 Video Upsampler (MiniMax H3, nativo): upscaler latente + ri-disegno a tile guidato dalla LoRA
h3upscale, vedi core/h3_upsample_service.py.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QCheckBox, QGroupBox, QFileDialog, QScrollArea,
    QSlider, QMessageBox, QApplication,
)
from PySide6.QtCore import Signal, Qt

from core.video_enhance_pipeline import SEEDVR2_MODELS
from core import face_restore_pipeline as facerestore
from ui.i18n_manager import tr

_ENGINE_KEY = {0: "seedvr2", 1: "esrgan", 2: "flashvsr", 3: "h3_upsampler", 4: "flashvsr_plus"}


class VideoDefTab(QWidget):
    request_enhance = Signal()
    request_face_restore = Signal()
    request_exr = Signal()
    request_open_weights_tab = Signal()     # «Apri tab Pesi modelli» (pesi dell'H3 Upsampler mancanti)

    def __init__(self):
        super().__init__()
        self._video: str | None = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QScrollArea.NoFrame)
        host = QWidget(); root = QVBoxLayout(host); sc.setWidget(host); outer.addWidget(sc)

        # sorgente
        srow = QHBoxLayout()
        self.video_lbl = QLabel(tr("def.no_video.lbl"))
        self.video_lbl.setWordWrap(True)
        pick = QPushButton(tr("def.pick.btn")); pick.clicked.connect(self._pick)
        pick.setToolTip(tr("def.pick.tt", "Carica un video da disco per migliorarne la definizione."))
        srow.addWidget(self.video_lbl, 1); srow.addWidget(pick)
        root.addLayout(srow)

        # motore + parametri
        box = QGroupBox(tr("def.restore.group")); f = QFormLayout(box)
        self.engine = QComboBox()
        self.engine.addItems([tr("def.engine.seedvr2"), tr("def.engine.esrgan"), tr("def.engine.flashvsr"),
                              tr("def.engine.h3up", "MiniMax H3 Video Upsampler"),
                              tr("def.engine.fvsrplus", "FlashVSR+ (Turbo / Base)")])
        self.engine.setItemData(4, tr("def.engine.fvsrplus.tt",
            "FlashVSR+: super-risoluzione video in streaming (Wan2.1-1.3B, un solo passo), fino a ×4, "
            "con attenzione ibrida: piu' veloce e con meno memoria di FlashVSR v1.1. Turbo = veloce, "
            "Base = qualita'. Gira anche su schede piccole (la memoria si adatta da sola)."), Qt.ToolTipRole)
        self.engine.setItemData(3, tr("def.engine.h3up.tt",
            "Ingrandisce il video con MiniMax H3: upscaler latente + un ri-disegno a tile guidato dalla LoRA h3upscale "
            "(il clip a metà risoluzione fa da guida). Serve il modello H3 Ref2VA."), Qt.ToolTipRole)
        self.engine.currentIndexChanged.connect(self._on_engine)
        self.engine.setToolTip(tr("def.engine.tt", "Motore di super-risoluzione: SeedVR2 (AI video), ESRGAN (veloce per frame), FlashVSR."))
        f.addRow(tr("def.engine_lbl"), self.engine)
        self.seedvr2_model = QComboBox(); self.seedvr2_model.addItems(list(SEEDVR2_MODELS.keys()))
        self.seedvr2_model.setToolTip(tr("def.seedvr2_model.tt", "Variante del modello SeedVR2 per la super-risoluzione video."))
        f.addRow(tr("def.seedvr2_model_lbl"), self.seedvr2_model)
        self.resolution = QSpinBox(); self.resolution.setRange(256, 8192); self.resolution.setSingleStep(64)
        self.resolution.setValue(1080); self.resolution.setSuffix(tr("def.resolution.suffix"))
        self.resolution.setToolTip(tr("def.resolution.tt", "Risoluzione verticale obiettivo per SeedVR2 (es. 1080 = FullHD)."))
        f.addRow(tr("def.resolution_lbl"), self.resolution)
        self.blocks = QSpinBox(); self.blocks.setRange(0, 36); self.blocks.setValue(0)
        self.blocks.setToolTip(tr("def.blocks.tt"))
        f.addRow(tr("def.blocks_lbl"), self.blocks)
        self.batch = QSpinBox(); self.batch.setRange(1, 121); self.batch.setSingleStep(4); self.batch.setValue(5)
        self.batch.setToolTip(tr("def.batch.tt"))
        f.addRow(tr("def.batch_lbl"), self.batch)
        self.overlap = QSpinBox(); self.overlap.setRange(0, 16); self.overlap.setValue(3)
        self.overlap.setToolTip(tr("def.overlap.tt", "Frame di sovrapposizione temporale tra i chunk SeedVR2: "
                                   "riduce lo scatto ai tagli sui video lunghi (0 = usa il default 3)."))
        f.addRow(tr("def.overlap_lbl", "Overlap temporale"), self.overlap)
        # esrgan
        self.esrgan_model = QComboBox()
        self.esrgan_model.setMaxVisibleItems(20)   # fix: mostra tutti i modelli nel popup
        self.esrgan_model.view().setMinimumWidth(420)
        self.esrgan_model.setToolTip(tr("def.esrgan_model.tt", "Modello ESRGAN per l'upscaling per-fotogramma."))
        f.addRow(tr("def.esrgan_model_lbl"), self.esrgan_model)
        self.scale = QDoubleSpinBox(); self.scale.setRange(1.0, 8.0); self.scale.setSingleStep(0.5); self.scale.setValue(2.0)
        self.scale.setToolTip(tr("def.scale.tt", "Fattore di scala ESRGAN: 2× raddoppia, 4× quadruplica la risoluzione."))
        f.addRow(tr("def.scale_lbl"), self.scale)
        # FlashVSR+: variante (Turbo/Base) e compressione temporale (movimento)
        self.fvp_mode = QComboBox()
        self.fvp_mode.addItem(tr("def.fvp_mode.turbo", "Turbo (veloce)"), "turbo")
        self.fvp_mode.addItem(tr("def.fvp_mode.base", "Base (qualità)"), "base")
        self.fvp_mode.setToolTip(tr("def.fvp_mode.tt",
            "Turbo: attenzione ibrida, il piu' veloce. Base: attenzione completa, qualita' massima."))
        f.addRow(tr("def.fvp_mode_lbl", "Variante FlashVSR+:"), self.fvp_mode)
        self.fvp_motion = QComboBox()
        self.fvp_motion.addItem(tr("def.fvp_motion.normal", "Normale (4 — consigliato)"), 4)
        self.fvp_motion.addItem(tr("def.fvp_motion.fast", "Veloce (2 — piu' lento)"), 2)
        self.fvp_motion.addItem(tr("def.fvp_motion.vfast", "Molto veloce (1 — il piu' lento)"), 1)
        self.fvp_motion.setToolTip(tr("def.fvp_motion.tt",
            "Frame di ingresso per fotogramma latente. 4 va bene per quasi tutti i video; "
            "2 o 1 conservano meglio i dettagli nei movimenti rapidi, a costo di piu' tempo."))
        f.addRow(tr("def.fvp_motion_lbl", "Movimento:"), self.fvp_motion)
        root.addWidget(box)
        self._build_h3up_box(root)

        # RIFE / FPS
        rbox = QGroupBox(tr("def.rife.group")); rf = QFormLayout(rbox)
        self.rife = QCheckBox(tr("def.rife.check")); rf.addRow(self.rife)
        self.rife.setToolTip(tr("def.rife.tt", "Attiva RIFE per aumentare i FPS del video tramite interpolazione."))
        self.rife_mult = QComboBox(); self.rife_mult.addItems(["2×", "3×", "4×"])
        self.rife_mult.setToolTip(tr("def.rife_mult.tt", "Moltiplicatore FPS: 2× raddoppia, 4× quadruplica."))
        rf.addRow(tr("def.rife_mult_lbl"), self.rife_mult)
        self.rife_ckpt = QComboBox(); self.rife_ckpt.addItems(["rife47.pth", "rife48.pth", "rife46.pth"])
        # #95: il RIFE nativo usa Practical-RIFE (modello unico) → la scelta ckpt non è onorata → disabilitata
        self.rife_ckpt.setEnabled(False)
        self.rife_ckpt.setToolTip(tr("def.rife_ckpt.tt",
            "Il RIFE nativo usa Practical-RIFE (modello unico). La selezione checkpoint era per ComfyUI."))
        rf.addRow(tr("def.rife_ckpt_lbl"), self.rife_ckpt)
        root.addWidget(rbox)

        # ── Export HDR/EXR (VFX) — frame OpenEXR float32 per Nuke/Resolve/Blender ──
        ebox = QGroupBox(tr("def.exr.group", "Export HDR / EXR (VFX)")); ef = QFormLayout(ebox)
        self.exr_linear = QCheckBox(tr("def.exr.linear", "Linearizza (de-gamma BT.709 → luce lineare)"))
        self.exr_linear.setChecked(True)
        self.exr_linear.setToolTip(tr("def.exr.linear.tt",
            "ON (standard compositing): converte in luce LINEARE (zscale transfer=linear). "
            "OFF: valori display-referred (gamma). Per un video HDR VERO generalo prima con la "
            "IC-LoRA HDR (LTX-2) nel tab Video, poi esporta da qui."))
        ef.addRow(self.exr_linear)
        self.exr_btn = QPushButton(tr("def.exr.run", "Esporta frame EXR"))
        self.exr_btn.clicked.connect(lambda: self.request_exr.emit())
        self.exr_btn.setToolTip(tr("def.exr.run.tt",
            "Esporta TUTTI i frame del video in OpenEXR float32 (cartella <video>_exr) — pipeline VFX."))
        ef.addRow(self.exr_btn)
        root.addWidget(ebox)

        # catena: dopo la super-risoluzione applica restauro volti (+ RIFE) in UN'unica sequenza
        self.chain_face = QCheckBox(tr("def.chain_face", "🔗 Poi restauro volti (in sequenza)"))
        self.chain_face.setToolTip(tr("def.chain_face.tt", "Dopo la super-risoluzione applica automaticamente il "
                                      "restauro volti e poi (se attivo) il RIFE, senza rilanciare a mano."))
        root.addWidget(self.chain_face)
        self.run_btn = QPushButton(tr("def.run.btn"))
        self.run_btn.setObjectName("generate"); self.run_btn.clicked.connect(self._on_run)
        self.run_btn.setToolTip(tr("def.run.tt", "Avvia il miglioramento definizione video con il motore selezionato."))
        root.addWidget(self.run_btn)
        note = QLabel(tr("def.note.lbl"))
        note.setWordWrap(True); note.setStyleSheet("color:#888;")
        root.addWidget(note)

        # ── Restauro volti (CodeFormer/GFPGAN) ──
        fbox = QGroupBox(tr("def.face.group"))
        ff = QFormLayout(fbox)
        info = QLabel(tr("def.face.info.lbl"))
        info.setWordWrap(True); info.setStyleSheet("color:#9aa1b0;"); info.setTextFormat(Qt.RichText)
        ff.addRow(info)
        self.fr_model = QComboBox(); self.fr_model.addItems(facerestore.FACERESTORE_MODELS)
        self.fr_model.setToolTip(tr("def.face.model.tt", "Modello AI per il restauro volti: CodeFormer (qualità) o GFPGAN (velocità)."))
        ff.addRow(tr("def.face.model_lbl"), self.fr_model)
        self.fr_detection = QComboBox(); self.fr_detection.addItems(facerestore.FACE_DETECTORS)
        self.fr_detection.setToolTip(tr("def.face.detection.tt", "Detector per localizzare i volti nel video prima del restauro."))
        ff.addRow(tr("def.face.detection_lbl"), self.fr_detection)
        frow = QHBoxLayout()
        self.fr_fidelity = QSlider(Qt.Horizontal); self.fr_fidelity.setRange(0, 100); self.fr_fidelity.setValue(70)
        self.fr_fidelity_lbl = QLabel("0.70")
        self.fr_fidelity.valueChanged.connect(lambda v: self.fr_fidelity_lbl.setText(f"{v/100:.2f}"))
        self.fr_fidelity.setToolTip(tr("def.face.fidelity.tt"))
        frow.addWidget(self.fr_fidelity, 1); frow.addWidget(self.fr_fidelity_lbl)
        ff.addRow(tr("def.face.fidelity_lbl"), self._wrap(frow))
        brow = QHBoxLayout()
        self.fr_run_btn = QPushButton(tr("def.face.run.btn"))
        self.fr_run_btn.clicked.connect(self._on_face_restore_clicked)
        self.fr_run_btn.setToolTip(tr("def.face.run.tt", "Avvia il restauro volti sul video caricato."))
        self.fr_install_btn = QPushButton(tr("def.face.install.btn"))
        self.fr_install_btn.clicked.connect(self._install_facerestore)
        brow.addWidget(self.fr_run_btn); brow.addWidget(self.fr_install_btn)
        ff.addRow(self._wrap(brow))
        self.fr_status = QLabel(""); self.fr_status.setStyleSheet("color:#9aa1b0; font-size:11px;")
        self.fr_status.setWordWrap(True)
        ff.addRow(self.fr_status)
        root.addWidget(fbox)
        root.addStretch(1)
        self._on_engine()
        self._refresh_facerestore_status()

    # ---- H3 Video Upsampler ----
    def _build_h3up_box(self, root):
        """Opzioni SOLO del motore H3 Video Upsampler (progressive disclosure: visibile solo con quel motore)."""
        from core import h3_upsample_plan as P
        from core import h3_upsample_spec as U
        self.h3up_box = QGroupBox(tr("def.h3up.group", "Opzioni H3 Video Upsampler"))
        f = QFormLayout(self.h3up_box)
        self.h3up_model = QComboBox()
        self.h3up_model.setToolTip(tr("def.h3up.model.tt",
            "Checkpoint H3 Ref2VA che ri-disegna il video (la LoRA di guida è addestrata su questo). Quelli non "
            "scaricati hanno ⬇: scaricali dal tab «Pesi modelli»."))
        f.addRow(tr("def.h3up.model_lbl", "Modello H3 (Ref2VA)"), self.h3up_model)
        self.h3up_te = QComboBox()
        for tid, key, label in (("mmh3-te-nvfp4", "nvfp4", "Qwen3-VL-32B nvfp4-AWQ (15,7 GB, consigliato)"),
                                ("mmh3-te-fp8", "fp8", "Qwen3-VL-32B fp8 (26,5 GB, RTX 40xx)"),
                                ("mmh3-te-int8", "int8", "Qwen3-VL-32B int8-ConvRot (27 GB, qualità)")):
            self.h3up_te.addItem(tr(f"vid.h3.te.{key}", label), tid)
        self.h3up_te.setToolTip(tr("def.h3up.te.tt",
            "Legge la didascalia della LoRA una volta sola: l'embedding resta in cache e dalla volta dopo il text "
            "encoder non si carica."))
        f.addRow(tr("def.h3up.te_lbl", "Text encoder"), self.h3up_te)
        self.h3up_mp = QDoubleSpinBox()
        self.h3up_mp.setRange(*U.MP_RANGE)
        self.h3up_mp.setDecimals(2)
        self.h3up_mp.setSingleStep(0.05)
        self.h3up_mp.setValue(P.MEGAPIXELS)
        self.h3up_mp.setSuffix(" MP")
        self.h3up_mp.setToolTip(tr("def.h3up.mp.tt",
            "Dimensione in uscita come area, rapporto ESATTO della sorgente (1 MP = 1024×1024): se i lati non sono "
            "multipli di 32 si aggiunge un bordo che a fine lavoro viene ritagliato, mai stirato. Più grande della "
            "sorgente: questa funzione ingrandisce soltanto."))
        f.addRow(tr("def.h3up.mp_lbl", "Megapixel in uscita"), self.h3up_mp)
        self.h3up_guide = QCheckBox(tr("def.h3up.guide.check", "Guida con la LoRA h3upscale (consigliata)"))
        self.h3up_guide.setChecked(True)
        self.h3up_guide.setToolTip(tr("def.h3up.guide.tt",
            "Il ri-disegno guarda il clip a metà risoluzione e ne restaura il dettaglio invece di inventarlo. "
            "Senza, il ri-disegno parte solo dal latente ingrandito."))
        f.addRow(self.h3up_guide)
        self.h3up_denoise = QDoubleSpinBox()
        self.h3up_denoise.setRange(*U.DENOISE_RANGE)
        self.h3up_denoise.setDecimals(3)
        self.h3up_denoise.setSingleStep(0.005)
        self.h3up_denoise.setValue(P.DENOISE_DEFAULT)
        self.h3up_denoise.setToolTip(tr("def.h3up.denoise.tt",
            "Quanto lavora l'unico passo di ri-disegno. 0,1 è il valore del nodo originale (guidato); 0 = solo "
            "ingrandimento, senza caricare né il text encoder né il modello H3."))
        f.addRow(tr("def.h3up.denoise_lbl", "Forza del ri-disegno"), self.h3up_denoise)
        self.h3up_lora = QDoubleSpinBox()
        self.h3up_lora.setRange(-4.0, 4.0)
        self.h3up_lora.setSingleStep(0.05)
        self.h3up_lora.setValue(1.0)
        f.addRow(tr("def.h3up.lora_lbl", "Forza LoRA"), self.h3up_lora)
        self.h3up_tiles = QSpinBox()
        self.h3up_tiles.setRange(0, U.TILE_TOKENS_RANGE[1])      # 0 = automatico
        self.h3up_tiles.setSpecialValueText(tr("def.h3up.tiles.auto", "automatico"))
        self.h3up_tiles.setSingleStep(1024)
        self.h3up_tiles.setValue(0)
        self.h3up_tiles.setToolTip(tr("def.h3up.tiles.tt",
            "Sequenza massima di una forward. Automatico = calcolata dalla VRAM libera dopo aver caricato il modello "
            "(tetto 70.000, il valore della ricetta). Più basso = tile più piccoli e più numerosi: meno VRAM, più "
            "lento, stessa fusione senza cuciture."))
        f.addRow(tr("def.h3up.tiles_lbl", "Budget dei tile (token)"), self.h3up_tiles)
        note = QLabel(tr("def.h3up.note",
            "Upscaler latente + un ri-disegno a tile con MiniMax H3, nativo. Per i video lunghi lavora a segmenti con "
            "dissolvenza nello spazio latente e tile fusi senza cuciture. La colonna sonora resta quella originale. "
            "Licenza MiniMax H3 Community: territori esclusi UE, Regno Unito, USA, Corea del Sud."))
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        f.addRow(note)
        root.addWidget(self.h3up_box)
        self._refilter_h3up_models()

    def _refilter_h3up_models(self):
        """Dropdown modello = profili H3 Ref2VA (scelta dell'utente; i non scaricati hanno ⬇ e il download è nel tab
        Pesi modelli). Nessun profilo scelto al posto suo: l'ordine è quello fisso del menu."""
        from PySide6.QtGui import QColor
        from core import h3_upsample_spec as U
        from core import weights_catalog as wc
        from core import weights_tooltips as wt
        labels = {"nvfp4": tr("vid.h3.prof.nvfp4", "nvfp4 · 12,5 GB (RTX 50xx)"),
                  "fp8": tr("vid.h3.prof.fp8", "fp8 · 21 GB (RTX 40xx)"),
                  "int8": tr("vid.h3.prof.int8", "int8-ConvRot · 21 GB (tutte le GPU)"),
                  "int4": tr("vid.h3.prof.int4", "int4 W4A8 · 12 GB (tutte le GPU)")}
        entries = {e["id"]: e for e in wc.load()}
        prev = self.h3up_model.currentData()
        self.h3up_model.blockSignals(True)
        self.h3up_model.clear()
        for prof in U.PROFILES:
            e = entries.get(U.dit_id(prof))
            if not e or not wc.menu_allowed(e):
                continue
            try:
                present = wc.is_present(e)
            except Exception:                                   # noqa: BLE001
                present = False
            self.h3up_model.addItem(("" if present else "⬇ ") + "MiniMax H3 Ref2VA " + labels[prof], prof)
            i = self.h3up_model.count() - 1
            self.h3up_model.setItemData(i, wt.tooltip_for(e) + "\n" + (tr("vid.present", "presente") if present
                                                                       else tr("vid.missing", "da scaricare")),
                                        Qt.ToolTipRole)
            if not present:
                self.h3up_model.setItemData(i, QColor("#e8b339"), Qt.ForegroundRole)
        j = self.h3up_model.findData(prev) if prev else -1
        if j >= 0:
            self.h3up_model.setCurrentIndex(j)
        self.h3up_model.blockSignals(False)

    def h3up_params(self) -> dict:
        """Params del motore H3 Upsampler (core.h3_upsample_spec.build_params) dai controlli del tab."""
        from core import h3_upsample_spec as U
        return U.build_params({
            "video": self._video, "h3_profile": self.h3up_model.currentData() or "",
            "h3_te": self.h3up_te.currentData() or "", "megapixels": self.h3up_mp.value(),
            "denoise": self.h3up_denoise.value(), "guide": self.h3up_guide.isChecked(),
            "lora_strength": self.h3up_lora.value(), "tile_tokens": self.h3up_tiles.value()})

    def _ask_missing_h3up(self, text: str) -> str:
        """Pesi dell'H3 Upsampler assenti: avviso con «Apri tab Pesi modelli» (nessun download automatico).
        Ritorna 'weights' | 'cancel'. Sovrascrivibile nei test."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(tr("def.h3up.missing.title", "Pesi mancanti"))
        box.setText(tr("def.h3up.missing.msg",
                       "Per l'H3 Video Upsampler mancano questi pesi: {ids}\n\nNon scarico nulla io: scaricali dal "
                       "tab «Pesi modelli».").format(ids=text))
        b_w = box.addButton(tr("vedit.ckpt.btn_open_weights_tab", "Apri tab Pesi modelli"), QMessageBox.ActionRole)
        box.addButton(tr("vedit.ckpt.btn_cancel", "Annulla"), QMessageBox.RejectRole)
        box.exec()
        return "weights" if box.clickedButton() is b_w else "cancel"

    def _preflight_h3up(self) -> bool:
        """True se si può lanciare; altrimenti avvisa (valori errati) o propone il tab Pesi (pesi assenti)."""
        from core import h3_upsample_spec as U
        p = self.h3up_params()
        errs = U.validate(p)
        if errs:
            QMessageBox.warning(self, tr("def.h3up.group", "Opzioni H3 Video Upsampler"), "\n".join(errs))
            return False
        try:
            miss = U.missing_weights(p)
        except Exception as e:                                   # noqa: BLE001
            QMessageBox.warning(self, tr("def.h3up.group", "Opzioni H3 Video Upsampler"), str(e))
            return False
        if miss:
            if self._ask_missing_h3up(U.missing_text(miss)) == "weights":
                self.request_open_weights_tab.emit()
            return False
        return True

    # ---- API ----
    def set_video(self, path: str):
        self._video = path
        self.video_lbl.setText(f"🎬 {Path(path).name}")

    def set_upscale_models(self, names: list[str]):
        self.esrgan_model.clear()
        self.esrgan_model.addItems(names or [tr("def.no_upscale_model")])
        # colori VRAM (header-aware) — path risolto sotto Models\Upscale
        try:
            from ui.widgets.precision_combo import color_existing
            from pathlib import Path as _P
            _root = _P(r"E:\AI\Models\Upscale")
            color_existing(self.esrgan_model,
                           name_to_path=lambda n: str(_root / n) if not _P(n).is_file() else n)
        except Exception:                                    # noqa: BLE001
            pass

    def get_config(self) -> dict:
        return {
            "video": self._video,
            "engine": _ENGINE_KEY[self.engine.currentIndex()],
            "seedvr2_model": SEEDVR2_MODELS[self.seedvr2_model.currentText()],
            "resolution": self.resolution.value(),
            "blocks_to_swap": self.blocks.value(),
            "batch_size": self.batch.value(),
            "esrgan_model": self.esrgan_model.currentText() if not self.esrgan_model.currentText().startswith("(") else "",
            "target_scale": self.scale.value(),
            "temporal_overlap": self.overlap.value(),
            "rife": self.rife.isChecked(),
            "rife_multiplier": self.rife_mult.currentIndex() + 2,
            "rife_ckpt": self.rife_ckpt.currentText(),
            "chain_face": self.chain_face.isChecked(),
            "h3up": self.h3up_params() if _ENGINE_KEY[self.engine.currentIndex()] == "h3_upsampler" else None,
            "fvp": {"mode": self.fvp_mode.currentData() or "turbo",
                    "temporal_compression": int(self.fvp_motion.currentData() or 4)},
        }

    def _wrap(self, layout):
        w = QWidget(); w.setLayout(layout); return w

    def get_face_restore_config(self) -> dict:
        return {
            "video": self._video,
            "model": self.fr_model.currentText(),
            "detection": self.fr_detection.currentText(),
            "fidelity": self.fr_fidelity.value() / 100.0,
        }

    def _refresh_facerestore_status(self):
        try:
            installed = facerestore.node_installed()
        except Exception:
            installed = False
        self.fr_install_btn.setVisible(not installed)
        self.fr_run_btn.setEnabled(installed)
        self.fr_status.setText(tr("def.face.status_ok") if installed else tr("def.face.status_nok"))

    def _on_face_restore_clicked(self):
        if not self._video:
            QMessageBox.warning(self, tr("def.face.no_video.title"), tr("def.face.no_video.msg"))
            return
        self.request_face_restore.emit()

    def _on_run(self):
        """FASE F weights-unification: pesi del motore scelto offerti CON CONSENSO prima del job
        (prima: SeedVR2 'exit N' muto / FlashVSR errore a valle)."""
        _ids = {"seedvr2": ["seedvr2-3b-fp8", "seedvr2-vae"], "flashvsr": ["flashvsr-v11"]}
        key = _ENGINE_KEY.get(self.engine.currentIndex(), "")
        if key == "h3_upsampler":                      # pesi: avviso + «Apri tab Pesi modelli», nessun download
            if self._preflight_h3up():
                self.request_enhance.emit()
            return
        ids = _ids.get(key)
        if key == "flashvsr_plus":                     # i pesi dipendono dalla variante scelta (Turbo / Base)
            from core import flashvsrplus_service as _fvp
            ids = _fvp.weight_ids(self.fvp_mode.currentData() or "turbo")
        if ids:
            try:
                from core import ensure_weights as ew
                if ew.missing(ids) and not ew.offer_download_gui(
                        self, ids, self.engine.currentText()):
                    return
            except Exception:                            # noqa: BLE001 — il preflight del service protegge
                pass
        self.request_enhance.emit()

    def _install_facerestore(self):
        if QMessageBox.question(self, tr("def.face.install.title"),
                                tr("def.face.install.confirm").format(
                                    url=facerestore.FACERESTORE_REPO['url'],
                                    note=facerestore.FACERESTORE_REPO['note'])) != QMessageBox.Yes:
            return
        self.fr_install_btn.setEnabled(False); self.fr_install_btn.setText(tr("def.face.install_update.btn"))
        QApplication.processEvents()
        try:
            ok, msg = facerestore.install_node()
        except Exception as e:
            ok, msg = False, str(e)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Warning)
        box.setWindowTitle(tr("def.face.install.title"))
        box.setText(msg if len(msg) < 400 else msg[:400] + "…"); box.setDetailedText(msg); box.exec()
        self.fr_install_btn.setEnabled(True); self.fr_install_btn.setText(tr("def.face.install.btn"))
        self._refresh_facerestore_status()

    # ---- interno ----
    def _on_engine(self, *_):
        key = _ENGINE_KEY[self.engine.currentIndex()]
        sv = key == "seedvr2"; es = key == "esrgan"; fv = key == "flashvsr"; h3 = key == "h3_upsampler"
        fp = key == "flashvsr_plus"
        self.h3up_box.setVisible(h3)
        self.seedvr2_model.setEnabled(sv); self.blocks.setEnabled(sv); self.batch.setEnabled(sv)
        self.resolution.setEnabled(sv); self.overlap.setEnabled(sv)
        self.esrgan_model.setEnabled(es)
        self.fvp_mode.setEnabled(fp); self.fvp_motion.setEnabled(fp)
        self.scale.setEnabled(es or fv or fp)                      # FlashVSR: gira alla scala scelta (max ×4 nativo — #156)

    def _pick(self):
        fn, _ = QFileDialog.getOpenFileName(self, tr("def.pick.dialog.title"), "", tr("def.pick.dialog.filter"))
        if fn:
            self.set_video(fn)
