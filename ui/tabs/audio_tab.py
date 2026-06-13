"""Tab Audio / Voce — sorgente audio + TTS con voice clone.

Tre sorgenti audio (esclusive):
- nativa: LTX-2.3 genera l'audio del parlato dal prompt.
- utente: file audio caricato manualmente o GENERATO QUI con il TTS.
- clona voce: campione vocale → LTXVReferenceAudio condiziona il timbro;
  l'audio del parlato è comunque generato dal modello (NON è il file caricato).

Sezione TTS — sintesi diretta con Edge TTS (default) + slot per CosyVoice/F5/XTTS.
Voci salvate: preset Microsoft + voci utente custom (filesystem `voices/<slug>.json`).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, QPushButton,
    QLabel, QFileDialog, QGroupBox, QCheckBox, QComboBox, QPlainTextEdit,
    QLineEdit, QInputDialog, QMessageBox, QToolButton, QSizePolicy, QApplication,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from core.audio_recorder import AudioRecorder
from core.tts_engine import (
    VoicePreset, edge_available, edge_list_voices, edge_synthesize, derive_edge_params,
    strip_markers, list_user_voices, save_voice, delete_voice,
    available_engines, synthesize, install_engine, invalidate_engine_cache,
    ENGINE_REPOS, OUTPUT_DIR as TTS_OUT_DIR,
)
from ui.widgets.tag_palette import install_tag_palette
from ui.voice_dialog import VoiceDialog
from ui.i18n_manager import tr


# Marker testuali alti-livello con tooltip leggibile.
_SSML_MARKERS = [
    ("⏸ pausa",   "[pause:500ms]",              "Pausa di 500ms"),
    ("⏸⏸ lunga",  "[pause:1200ms]",             "Pausa lunga di 1.2s"),
    ("🐢 lento",   '<prosody rate="slow">…</prosody>', "Rallenta la parte selezionata"),
    ("🐇 veloce",  '<prosody rate="fast">…</prosody>', "Accelera la parte selezionata"),
    ("🔉 piano",   '<prosody volume="soft">…</prosody>', "Abbassa il volume"),
    ("🔊 forte",   '<prosody volume="loud">…</prosody>', "Alza il volume"),
    ("😊 felice",  '<emotion type="happy">…</emotion>',  "Tono felice"),
    ("😢 triste",  '<emotion type="sad">…</emotion>',    "Tono triste"),
    ("😡 rabbia",  '<emotion type="angry">…</emotion>',  "Tono arrabbiato"),
    ("🤫 sussurro",'<emotion type="whisper">…</emotion>','Sussurro'),
    ("😌 calmo",   '<emotion type="calm">…</emotion>',   "Tono calmo"),
]


class AudioTab(QWidget):
    # emette il path dell'ultimo file TTS generato (per dialog di consenso in on_generate)
    tts_generated = Signal(str)

    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder() if AudioRecorder.available() else None
        self._recording = False
        self._audio_file: str | None = None
        self._tts_last_file: str | None = None       # ultimo TTS generato (per consenso)
        self._media_player: QMediaPlayer | None = None
        self._audio_output: QAudioOutput | None = None
        self._build()

    # ──────────────────────────────────────────────────────────────────
    # UI build
    # ──────────────────────────────────────────────────────────────────
    def _build(self):
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(tr("aud.source.lbl")))
        self.grp = QButtonGroup(self)
        self.rb_native = QRadioButton(tr("aud.rb_native")); self.rb_native.setChecked(True)
        self.rb_user = QRadioButton(tr("aud.rb_user"))
        self.rb_clone = QRadioButton(tr("aud.rb_clone"))
        self.rb_clone.setToolTip(tr("aud.rb_clone.tt"))
        self.rb_mute = QRadioButton(tr("aud.rb_mute"))
        for i, rb in enumerate((self.rb_native, self.rb_user, self.rb_clone, self.rb_mute)):
            self.grp.addButton(rb, i); lay.addWidget(rb)
        self.grp.buttonClicked.connect(self._on_mode)

        # Badge che mostra quale voce salvata è selezionata per il cloning
        self.clone_badge = QLabel("")
        self.clone_badge.setStyleSheet("color:#9aa1b0; padding-left:22px;")
        self.clone_badge.setWordWrap(True)
        lay.addWidget(self.clone_badge)

        self.no_sing = QCheckBox(tr("aud.no_sing"))
        self.no_sing.setChecked(True)
        lay.addWidget(self.no_sing)

        # ── box file audio per la sola modalità «Audio fornito da me» (traccia pronta) ──
        self.file_box = QGroupBox(tr("aud.file.group"))
        fb = QHBoxLayout(self.file_box)
        self.pick_btn = QPushButton(tr("aud.file.pick.btn")); self.pick_btn.clicked.connect(self._pick)
        self.rec_btn = QPushButton(tr("aud.rec.btn")); self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._toggle_rec)
        if not self.recorder:
            self.rec_btn.setEnabled(False); self.rec_btn.setText(tr("aud.rec.no_mic"))
        self.file_lbl = QLabel(tr("aud.file.none"))
        fb.addWidget(self.pick_btn); fb.addWidget(self.rec_btn); fb.addWidget(self.file_lbl, 1)
        lay.addWidget(self.file_box)

        lay.addWidget(self._build_tts_box())

        note = QLabel(tr("aud.note.lbl"))
        note.setWordWrap(True); note.setStyleSheet("color:#888;"); note.setTextFormat(Qt.RichText)
        lay.addWidget(note)
        self._reload_voices()
        self._on_mode()

    def _build_tts_box(self) -> QGroupBox:
        box = QGroupBox(tr("aud.tts.group"))
        v = QVBoxLayout(box)

        # ── riga 0: engine selector + bottone install ──
        r0 = QHBoxLayout()
        r0.addWidget(QLabel(tr("aud.tts.engine_lbl")))
        self.tts_engine = QComboBox(); self.tts_engine.setMinimumWidth(360)
        self._populate_engines()
        self.tts_engine.currentIndexChanged.connect(self._on_engine_changed)
        r0.addWidget(self.tts_engine, 1)
        self.tts_install_btn = QPushButton(tr("aud.tts.install.btn"))
        self.tts_install_btn.setToolTip(tr("aud.tts.install.tt"))
        self.tts_install_btn.clicked.connect(self._install_selected_engine)
        r0.addWidget(self.tts_install_btn)
        v.addLayout(r0)
        self._update_install_button()

        # ── riga 1: voce ──
        r1 = QHBoxLayout()
        r1.addWidget(QLabel(tr("aud.tts.voice_lbl")))
        self.tts_voice = QComboBox(); self.tts_voice.setMinimumWidth(280)
        self.tts_voice.currentIndexChanged.connect(self._update_clone_badge)
        r1.addWidget(self.tts_voice, 1)
        # CRUD voci
        b_new = QPushButton("➕"); b_new.setToolTip(tr("aud.tts.voice.new.tt")); b_new.clicked.connect(self._voice_create)
        b_edit = QPushButton("✏️"); b_edit.setToolTip(tr("aud.tts.voice.edit.tt")); b_edit.clicked.connect(self._voice_edit)
        b_dup = QPushButton("📋"); b_dup.setToolTip(tr("aud.tts.voice.dup.tt")); b_dup.clicked.connect(self._voice_duplicate)
        b_del = QPushButton("🗑️"); b_del.setToolTip(tr("aud.tts.voice.del.tt")); b_del.clicked.connect(self._voice_delete)
        for b in (b_new, b_edit, b_dup, b_del):
            b.setMaximumWidth(36); r1.addWidget(b)
        v.addLayout(r1)

        # ── riga 2: toolbar marker SSML ──
        tb = QHBoxLayout(); tb.setSpacing(2)
        tb.addWidget(QLabel(tr("aud.tts.insert_lbl")))
        for label, marker, tip in _SSML_MARKERS:
            b = QToolButton(); b.setText(label); b.setToolTip(tip)
            b.clicked.connect(lambda _=False, m=marker: self._insert_marker(m))
            tb.addWidget(b)
        tb.addStretch(1)
        v.addLayout(tb)

        # ── riga 3: testo da sintetizzare ──
        v.addWidget(QLabel(tr("aud.tts.text_lbl")))
        self.tts_text = QPlainTextEdit()
        self.tts_text.setPlaceholderText(tr("aud.tts.text.ph"))
        self.tts_text.setMaximumHeight(110)
        v.addWidget(self.tts_text)
        install_tag_palette(self.tts_text)

        # ── riga 4: anteprima testo pulito ──
        self.tts_preview = QLabel(tr("aud.tts.preview.lbl"))
        self.tts_preview.setStyleSheet("color:#999; font-style:italic; padding:2px;")
        self.tts_preview.setWordWrap(True)
        v.addWidget(self.tts_preview)
        self.tts_text.textChanged.connect(self._refresh_preview)

        # ── riga 5: bottoni genera/play + stato ──
        r5 = QHBoxLayout()
        self.tts_gen_btn = QPushButton(tr("aud.tts.gen.btn"))
        self.tts_gen_btn.clicked.connect(self._generate_tts)
        r5.addWidget(self.tts_gen_btn)
        self.tts_play_btn = QPushButton(tr("aud.tts.play.btn"))
        self.tts_play_btn.setEnabled(False)
        self.tts_play_btn.clicked.connect(self._play_tts)
        r5.addWidget(self.tts_play_btn)
        self.tts_use_btn = QPushButton(tr("aud.tts.use.btn"))
        self.tts_use_btn.setEnabled(False)
        self.tts_use_btn.clicked.connect(self._use_tts_as_audio)
        r5.addWidget(self.tts_use_btn)
        r5.addStretch(1)
        v.addLayout(r5)
        self.tts_status = QLabel("")
        self.tts_status.setStyleSheet("color:#cdd;")
        v.addWidget(self.tts_status)

        return box

    # ──────────────────────────────────────────────────────────────────
    # CRUD voci + caricamento
    # ──────────────────────────────────────────────────────────────────
    def _on_engine_changed(self):
        self._reload_voices()
        self._update_install_button()

    def _update_install_button(self):
        """Mostra/nasconde + abilita il bottone Install in base all'engine selezionato."""
        if not hasattr(self, "tts_install_btn"):
            return
        eid = self.tts_engine.currentData()
        if not eid or eid == "edge":
            self.tts_install_btn.setVisible(False); return
        # mostra il bottone se l'engine richiede un custom node non installato
        from core.tts_engine import _engine_installed
        installed = _engine_installed(eid)
        if installed:
            self.tts_install_btn.setVisible(True)
            self.tts_install_btn.setText("🔄 Aggiorna")
            self.tts_install_btn.setToolTip("Esegue git pull + pip install -r requirements.txt")
        else:
            self.tts_install_btn.setVisible(True)
            self.tts_install_btn.setText("📥 Installa")
            self.tts_install_btn.setToolTip(
                ENGINE_REPOS.get(eid, {}).get("note", "Installa il custom node ComfyUI di questo engine."))

    def _install_selected_engine(self):
        """Conferma + git clone + pip install dei requirements + refresh stato."""
        eid = self.tts_engine.currentData()
        if not eid or eid not in ENGINE_REPOS:
            return
        spec = ENGINE_REPOS[eid]
        ans = QMessageBox.question(
            self, f"Installa engine '{eid}'",
            f"Procedere con:\n\n"
            f"  • git clone {spec['url']}\n"
            f"  • pip install -r requirements.txt (nel venv di ComfyUI)\n\n"
            f"{spec['note']}\n\n"
            f"Il modello del motore verrà scaricato al primo uso (potrebbe richiedere alcuni GB).\n"
            f"Continuare?",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        # disabilita UI durante l'install
        self.tts_install_btn.setEnabled(False); self.tts_install_btn.setText("⏳ Installazione…")
        QApplication.processEvents()      # repaint immediato
        try:
            ok, msg = install_engine(eid)
        except Exception as e:
            ok, msg = False, f"Errore inatteso: {e}"
        # mostra log in un dialog scrollabile
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Warning)
        box.setWindowTitle("Installazione " + ("completata" if ok else "fallita"))
        box.setText(msg if len(msg) < 400 else msg[:400] + "…")
        box.setDetailedText(msg)
        box.exec()
        # refresh dropdown engine + bottone
        invalidate_engine_cache()
        cur = self.tts_engine.currentData()
        self._populate_engines()
        # ripristina la selezione
        for i in range(self.tts_engine.count()):
            if self.tts_engine.itemData(i) == cur:
                self.tts_engine.setCurrentIndex(i); break
        self._update_install_button()
        self.tts_install_btn.setEnabled(True)

    def _populate_engines(self):
        """Popola il dropdown dei motori TTS con flag di disponibilità."""
        self.tts_engine.clear()
        first_avail = None
        for eid, disp, clone, emo, installed in available_engines():
            badge = "✅" if installed else "⬇ da installare"
            tags = []
            if clone: tags.append("clone")
            if emo: tags.append("emozioni")
            label = f"{badge}  {disp}   [{', '.join(tags) or 'preset'}]"
            self.tts_engine.addItem(label, eid)
            self.tts_engine.setItemData(self.tts_engine.count() - 1,
                                         "✓ disponibile" if installed else
                                         "Non installato in ComfyUI", Qt.ToolTipRole)
            if installed and first_avail is None:
                first_avail = self.tts_engine.count() - 1
        if first_avail is not None:
            self.tts_engine.setCurrentIndex(first_avail)

    def _reload_voices(self, *_):
        """Ricarica il dropdown voci. Regola: una voce custom è SEMPRE visibile se ha un
        campione (gruppo «Mie voci clonabili») oppure se appartiene all'engine corrente.
        Così una voce appena creata non sparisce mai cambiando engine."""
        self.tts_voice.blockSignals(True)
        self.tts_voice.clear()
        engine_id = (self.tts_engine.currentData() if hasattr(self, "tts_engine") else None) or "edge"
        all_users = list_user_voices()
        # 1) voci clonabili (con campione) — visibili a prescindere dall'engine
        clonable = [v for v in all_users if v.sample_path]
        if clonable:
            self.tts_voice.addItem("── Mie voci (clonabili) ──", None)
            for v in clonable:
                self.tts_voice.addItem(f"⭐ {v.name}  🎤", v)
        # 2) voci custom edge-style senza campione, dell'engine corrente
        edge_custom = [v for v in all_users if not v.sample_path and v.engine == engine_id]
        if edge_custom:
            self.tts_voice.addItem("── Mie voci preset ──", None)
            for v in edge_custom:
                self.tts_voice.addItem(f"⭐ {v.name}", v)
        # 3) preset di sistema dell'engine
        if engine_id == "edge":
            if edge_available():
                self.tts_voice.addItem("── Edge TTS (Microsoft) — Italiano ──", None)
                for v in edge_list_voices("it"):
                    self.tts_voice.addItem(v.name, v)
            else:
                self.tts_voice.addItem("(edge-tts non installato — installa con: pip install edge-tts)", None)
        else:
            if not clonable:
                self.tts_voice.addItem(
                    "(nessuna voce clonabile — crea con ➕ allegando/registrando un campione)", None)
        self.tts_voice.blockSignals(False)
        self._update_clone_badge()

    def _update_clone_badge(self, *_):
        """Aggiorna il badge sotto la radio «Clona voce» con la voce selezionata."""
        if not hasattr(self, "clone_badge"):
            return
        v = self._current_voice()
        if not isinstance(v, VoicePreset):
            self.clone_badge.setText(tr("aud.clone_badge.none"))
            return
        if v.sample_path:
            self.clone_badge.setText(f"↳ voce: <b>{v.name}</b> — 🎤 ha campione (clonabile)")
        else:
            self.clone_badge.setText(f"↳ voce: <b>{v.name}</b> — ⚠ nessun campione (non clonabile, è una voce preset)")
        self.clone_badge.setTextFormat(Qt.RichText)

    def _current_voice(self) -> VoicePreset | None:
        return self.tts_voice.currentData()

    def _select_voice_by_file(self, target):
        """Riseleziona una voce nel dropdown per identità del file json (robusto vs nome)."""
        for i in range(self.tts_voice.count()):
            data = self.tts_voice.itemData(i)
            if isinstance(data, VoicePreset) and target is not None and data.file == target:
                self.tts_voice.setCurrentIndex(i); return True
        return False

    def _voice_create(self):
        dlg = VoiceDialog(self, default_engine=(self.tts_engine.currentData() or "edge"))
        if dlg.exec() and getattr(dlg, "saved_voice", None):
            self._reload_voices()
            self._select_voice_by_file(dlg.saved_voice.file)
            self.tts_status.setText(f"Voce '{dlg.saved_voice.name}' creata.")

    def _voice_edit(self):
        v = self._current_voice()
        if not isinstance(v, VoicePreset):
            return
        if not v.custom:
            QMessageBox.information(self, "Voce preset",
                "Le voci preset di sistema (Edge Microsoft) non sono modificabili. Usa 📋 per duplicarle."); return
        dlg = VoiceDialog(self, preset=v, default_engine=v.engine)
        if dlg.exec() and getattr(dlg, "saved_voice", None):
            self._reload_voices()
            self._select_voice_by_file(dlg.saved_voice.file)

    def _voice_duplicate(self):
        v = self._current_voice()
        if not isinstance(v, VoicePreset):
            return
        # crea una copia (nuovo slug → save_voice ricopia il campione con un nome proprio)
        dup = VoicePreset(name=f"{v.name} (copia)", engine=v.engine, voice_id=v.voice_id,
                          language=v.language, note=v.note,
                          sample_transcript=v.sample_transcript, custom=True)
        # apri il dialog pre-compilato così l'utente può rinominare/cambiare il campione
        dlg = VoiceDialog(self, preset=dup, default_engine=v.engine)
        # pre-carica il campione originale come sorgente da copiare
        if v.sample_path:
            dlg._sample_path = v.sample_path
            dlg._sample_to_copy = v.sample_path
            dlg.sample_lbl.setText(dlg._sample_caption())
            dlg.play_btn.setEnabled(True)
        if dlg.exec() and getattr(dlg, "saved_voice", None):
            self._reload_voices()
            self._select_voice_by_file(dlg.saved_voice.file)

    def _voice_delete(self):
        v = self._current_voice()
        if not isinstance(v, VoicePreset):
            return
        if not v.custom:
            QMessageBox.information(self, "Voce preset", "Le voci preset di sistema non possono essere eliminate."); return
        if QMessageBox.question(self, "Elimina voce", f"Eliminare '{v.name}'?") != QMessageBox.Yes:
            return
        delete_voice(v); self._reload_voices()

    # ──────────────────────────────────────────────────────────────────
    # Toolbar marker SSML
    # ──────────────────────────────────────────────────────────────────
    def _insert_marker(self, marker: str):
        cur = self.tts_text.textCursor()
        if "…" in marker:
            # se c'è selezione, avvolge; altrimenti inserisce con placeholder
            sel = cur.selectedText()
            if sel:
                cur.insertText(marker.replace("…", sel))
            else:
                cur.insertText(marker)
        else:
            cur.insertText(marker)
        self.tts_text.setFocus()

    def _refresh_preview(self):
        clean = strip_markers(self.tts_text.toPlainText())
        self.tts_preview.setText(f"📖 “{clean}”" if clean else "(testo vuoto)")

    # ──────────────────────────────────────────────────────────────────
    # Generazione TTS
    # ──────────────────────────────────────────────────────────────────
    def _generate_tts(self):
        v = self._current_voice()
        if not v:
            QMessageBox.warning(self, "Voce", "Seleziona una voce."); return
        text = self.tts_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Testo", "Scrivi del testo da sintetizzare."); return
        engine_id = self.tts_engine.currentData() or "edge"
        try:
            self.tts_status.setText(f"Sintesi in corso con {engine_id}…")
            out = synthesize(engine_id, v, text)
            self._tts_last_file = str(out)
            self.tts_status.setText(f"✅ Generato: {Path(self._tts_last_file).name}  (motore: {engine_id})")
            self.tts_play_btn.setEnabled(True)
            self.tts_use_btn.setEnabled(True)
            self.tts_generated.emit(self._tts_last_file)
        except Exception as e:
            self.tts_status.setText(f"⛔ Errore: {e}")
            QMessageBox.critical(self, "Errore TTS", str(e))

    def _play_tts(self):
        if not self._tts_last_file:
            return
        if self._media_player is None:
            self._media_player = QMediaPlayer(self)
            self._audio_output = QAudioOutput(self)
            self._media_player.setAudioOutput(self._audio_output)
        self._media_player.setSource(QUrl.fromLocalFile(self._tts_last_file))
        self._media_player.play()

    def _use_tts_as_audio(self):
        """Imposta il TTS appena generato come audio_file e seleziona modalità 'utente'."""
        if not self._tts_last_file:
            return
        self._audio_file = self._tts_last_file
        self.file_lbl.setText(Path(self._tts_last_file).name + " (TTS)")
        self.rb_user.setChecked(True)
        self._on_mode()

    # ──────────────────────────────────────────────────────────────────
    # API esterne (compat con main_window)
    # ──────────────────────────────────────────────────────────────────
    def _on_mode(self, *_):
        is_user = self.rb_user.isChecked()
        is_clone = self.rb_clone.isChecked()
        # file_box ora serve SOLO alla traccia pronta «Audio fornito da me».
        self.file_box.setVisible(is_user)
        self.file_box.setEnabled(is_user)
        self.rec_btn.setEnabled(bool(self.recorder) and is_user)
        # badge clone visibile solo in modalità clone
        self.clone_badge.setVisible(is_clone)
        self.no_sing.setEnabled(not self.rb_mute.isChecked())
        if is_clone:
            self._update_clone_badge()

    def _pick(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Audio", "", "Audio (*.wav *.mp3 *.flac *.m4a)")
        if fn:
            self._audio_file = fn; self.file_lbl.setText(Path(fn).name)

    def _toggle_rec(self, on: bool):
        if not self.recorder: return
        if on:
            self._recording = True; self.rec_btn.setText(tr("aud.rec.stop")); self.recorder.start()
            self.file_lbl.setText(tr("aud.rec.recording"))
        else:
            self._recording = False; self.rec_btn.setText(tr("aud.rec.btn"))
            path = self.recorder.stop("voice_sample.wav")
            self._audio_file = path; self.file_lbl.setText(tr("aud.rec.done"))

    def has_pending_tts(self) -> str | None:
        """Ritorna il path dell'ultimo TTS generato se non ancora applicato come audio_file."""
        if self._tts_last_file and self._audio_file != self._tts_last_file:
            return self._tts_last_file
        return None

    def apply_tts_as_user_audio(self):
        """Chiamato dal dialog di consenso in main_window per applicare il TTS."""
        if self._tts_last_file:
            self._audio_file = self._tts_last_file
            self.rb_user.setChecked(True); self._on_mode()

    def get_audio_config(self) -> dict:
        mode = "native"
        if self.rb_user.isChecked():    mode = "user"
        elif self.rb_clone.isChecked(): mode = "clone"
        elif self.rb_mute.isChecked():  mode = "mute"
        suppress_singing = self.no_sing.isChecked() and mode in ("native", "user", "clone")
        cfg = {"audio_mode": mode,
               "audio_mute": mode == "mute", "suppress_singing": suppress_singing,
               "audio_native": mode == "native", "audio_user": mode == "user",
               "audio_clone": mode == "clone",
               # default: nessuna sorgente concreta (riempiti sotto per modalità)
               "audio_file": None, "voice_sample": None,
               "clone_voice": None, "tts_text": "", "tts_engine": None}
        if mode == "user":
            # traccia pronta caricata/registrata (o TTS applicato con «⤴ Usa come audio»)
            cfg["audio_file"] = self._audio_file
        elif mode == "clone":
            v = self._current_voice() if hasattr(self, "tts_voice") else None
            if isinstance(v, VoicePreset):
                cfg["clone_voice"] = v
                cfg["voice_sample"] = v.sample_path
                cfg["tts_engine"] = self.tts_engine.currentData() or "edge"
            cfg["tts_text"] = self.tts_text.toPlainText().strip() if hasattr(self, "tts_text") else ""
        return cfg


def _pauses_to_dots(text: str) -> str:
    """Edge interpreta '...' come breve pausa; i nostri [pause:Xms] vengono mappati a sequenze
    di puntini proporzionali (rough). Non perfetto come SSML <break>, ma funziona."""
    import re as _re
    def _r(m):
        ms = int(m.group(1) or 500)
        n = max(2, min(8, ms // 200))
        return " " + "." * n + " "
    return _re.sub(r"\[pause[: ]?(\d+)?m?s?\]", _r, text, flags=_re.I)
