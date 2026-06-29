"""Casi di regressione GUI-driven per il tab AUDIO/VOCE (ui/tabs/audio_tab.py).

Particolarità di questo tab (verificata nel codice):
- NON instrada al backend via native_driver/_enqueue: lo Spy non cattura nulla qui.
  Il tab è un FORNITORE DI CONFIG (`get_audio_config()` letto da main_window in on_generate/
  prepare) + un SINTETIZZATORE DIRETTO (`_generate_tts()` chiama `core.tts_engine.synthesize`).
- Quindi i test verificano: (a) no-crash sugli handler della GUI, (b) correttezza del dict
  di config per ogni modalità (native/user/clone/mute), (c) che l'azione "Genera voce" instradi
  il MOTORE SELEZIONATO a `synthesize` — un caso per OGNI motore concreto (così si scopre
  "funziona solo un motore").

I "modelli" del dropdown sono i MOTORI TTS (combo `audio_tab.tts_engine`, itemData = engine id):
  kokoro, handy_clone, edge, cosyvoice2, f5tts, xtts2, fish, vibevoice.

Tutte le chiamate di rete/GPU sono neutralizzate: `synthesize` è patchato a livello di caso,
le liste voci (edge/utente) sono stubbate per evitare la rete a Microsoft.
"""
from __future__ import annotations

import harness as H

from core.tts_engine import VoicePreset, ENGINES_META


# ──────────────────────────────────────────────────────────────────────────────
# helper locali
# ──────────────────────────────────────────────────────────────────────────────
def _fake_voice(sample: str | None = None) -> VoicePreset:
    """Una VoicePreset finta usabile come selezione corrente (clone se sample!=None)."""
    return VoicePreset(name="TestVoice", engine="edge", voice_id="it-IT-ElsaNeural",
                       language="it-IT", custom=True, sample_path=sample)


def _stub_voice_lists():
    """Evita rete/FS: stubba le sorgenti voci usate da _reload_voices nel modulo audio_tab."""
    from ui.tabs import audio_tab as AT
    fake = _fake_voice()
    AT.edge_available = lambda: True
    AT.edge_list_voices = lambda *a, **k: [_fake_voice()]
    AT.list_user_voices = lambda *a, **k: []
    return AT


def _patch_synthesize(AT):
    """Patcha audio_tab.synthesize per catturare (engine, voice, text) senza GPU/rete.
    Ritorna la lista `captured` che riceve i dict delle chiamate."""
    captured: list[dict] = []

    def fake_synthesize(engine, voice, text, **k):
        captured.append({"engine": engine,
                         "voice": getattr(voice, "name", None),
                         "text": text})
        from pathlib import Path
        return Path(H.OUT_DIR / f"_fake_tts_{engine}.mp3")

    AT.synthesize = fake_synthesize
    return captured


def _select_engine(tab, eid: str) -> bool:
    """Seleziona il motore con itemData == eid nel combo tts_engine."""
    c = tab.tts_engine
    for i in range(c.count()):
        if c.itemData(i) == eid:
            c.setCurrentIndex(i)
            return True
    return False


def _goto_audio(w):
    try:
        w.tabs.setCurrentWidget(w.audio_tab)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG per modalità (get_audio_config) — ciò che main_window legge in on_generate
# ──────────────────────────────────────────────────────────────────────────────
def case_config_native(w, spy):
    """Modalità NATIVA: get_audio_config → audio_mode=native, audio_native=True, no sorgenti file."""
    t = w.audio_tab
    t.rb_native.setChecked(True); t._on_mode()
    cfg = t.get_audio_config()
    assert cfg["audio_mode"] == "native", cfg
    assert cfg["audio_native"] is True and cfg["audio_mute"] is False, cfg
    assert cfg["audio_file"] is None and cfg["clone_voice"] is None, cfg


def case_config_user_file(w, spy):
    """Modalità UTENTE: con un file impostato, get_audio_config deve riportarlo in audio_file."""
    t = w.audio_tab
    t.rb_user.setChecked(True); t._on_mode()
    t._audio_file = str(H.OUT_DIR / "_fake_user_track.wav")
    cfg = t.get_audio_config()
    assert cfg["audio_mode"] == "user" and cfg["audio_user"] is True, cfg
    assert cfg["audio_file"] == t._audio_file, f"audio_file non propagato: {cfg}"


def case_config_clone(w, spy):
    """Modalità CLONA: con voce+campione e testo, config deve avere clone_voice, voice_sample,
    tts_engine e tts_text."""
    AT = _stub_voice_lists()
    t = w.audio_tab
    sample = str(H.OUT_DIR / "_fake_sample.wav")
    v = _fake_voice(sample=sample)
    t._current_voice = lambda: v            # forza la selezione voce
    _select_engine(t, "edge")
    t.rb_clone.setChecked(True); t._on_mode()
    t.tts_text.setPlainText("ciao mondo")
    cfg = t.get_audio_config()
    assert cfg["audio_mode"] == "clone" and cfg["audio_clone"] is True, cfg
    assert cfg["clone_voice"] is v, f"clone_voice mancante: {cfg}"
    assert cfg["voice_sample"] == sample, f"voice_sample mancante: {cfg}"
    assert cfg["tts_engine"], f"tts_engine mancante in clone: {cfg}"
    assert cfg["tts_text"] == "ciao mondo", cfg


def case_config_mute(w, spy):
    """Modalità MUTO: audio_mute=True e suppress_singing forzato a False (no audio del tutto)."""
    t = w.audio_tab
    t.rb_mute.setChecked(True); t._on_mode()
    cfg = t.get_audio_config()
    assert cfg["audio_mode"] == "mute" and cfg["audio_mute"] is True, cfg
    assert cfg["suppress_singing"] is False, f"mute non deve sopprimere canto separatamente: {cfg}"


def case_no_sing_toggle(w, spy):
    """Toggle 'no canto': in modalità native, suppress_singing deve seguire la checkbox."""
    t = w.audio_tab
    t.rb_native.setChecked(True); t._on_mode()
    t.no_sing.setChecked(True)
    assert t.get_audio_config()["suppress_singing"] is True, "no_sing ON deve dare suppress_singing=True"
    t.no_sing.setChecked(False)
    assert t.get_audio_config()["suppress_singing"] is False, "no_sing OFF deve dare suppress_singing=False"


# ──────────────────────────────────────────────────────────────────────────────
# Cambio modalità / UI
# ──────────────────────────────────────────────────────────────────────────────
def case_mode_switch_no_crash(w, spy):
    """Cliccare ogni radio (native/user/clone/mute) non deve sollevare eccezioni e deve
    aggiornare la visibilità del box file (solo in 'utente')."""
    _stub_voice_lists()
    t = w.audio_tab
    w.tabs.setCurrentWidget(w.audio_tab)        # i widget di un tab NON attivo non sono "visible"
    for rb in (t.rb_native, t.rb_user, t.rb_clone, t.rb_mute):
        rb.setChecked(True); t._on_mode()
    # ultima = mute → file_box nascosto
    assert t.file_box.isVisibleTo(t) is False
    t.rb_user.setChecked(True); t._on_mode()
    assert t.file_box.isVisibleTo(t) is True, "in modalità utente il box file deve essere visibile"


def case_engine_change_reloads_voices(w, spy):
    """Cambiare motore non deve crashare e deve ricaricare il dropdown voci (no rete)."""
    _stub_voice_lists()
    t = w.audio_tab
    for eid in ("edge", "cosyvoice2", "f5tts"):
        assert _select_engine(t, eid), f"motore {eid} assente nel combo"
        t._on_engine_changed()
    # nessuna eccezione = ok


def case_reload_voices_no_crash(w, spy):
    """_reload_voices con voci utente stubbate (clonabili + edge) non deve crashare."""
    from ui.tabs import audio_tab as AT
    AT.edge_available = lambda: True
    AT.edge_list_voices = lambda *a, **k: [_fake_voice()]
    AT.list_user_voices = lambda *a, **k: [_fake_voice(sample=str(H.OUT_DIR / "s.wav")),
                                           _fake_voice()]
    w.audio_tab._reload_voices()
    assert w.audio_tab.tts_voice.count() > 0, "il dropdown voci è vuoto dopo reload"


# ──────────────────────────────────────────────────────────────────────────────
# Marker SSML + anteprima
# ──────────────────────────────────────────────────────────────────────────────
def case_ssml_marker_insert(w, spy):
    """Inserire un marker [pause:..] aggiunge il testo del marker nell'editor."""
    t = w.audio_tab
    t.tts_text.setPlainText("")
    t._insert_marker("[pause:500ms]")
    assert "[pause:500ms]" in t.tts_text.toPlainText(), "marker non inserito"


def case_ssml_marker_wrap_selection(w, spy):
    """Con selezione, un marker con '…' avvolge il testo selezionato (prosody slow)."""
    t = w.audio_tab
    t.tts_text.setPlainText("parla piano")
    cur = t.tts_text.textCursor()
    cur.select(cur.SelectionType.Document)
    t.tts_text.setTextCursor(cur)
    t._insert_marker('<prosody rate="slow">…</prosody>')
    out = t.tts_text.toPlainText()
    assert "parla piano" in out and "prosody" in out, f"wrap fallito: {out!r}"


def case_preview_strips_markers(w, spy):
    """L'anteprima testo deve mostrare il testo PULITO (senza marker)."""
    t = w.audio_tab
    t.tts_text.setPlainText("Ciao [pause:500ms] mondo")
    t._refresh_preview()
    pv = t.tts_preview.text()
    assert "[pause" not in pv, f"l'anteprima non ha rimosso i marker: {pv!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Generazione TTS (azione principale del tab) — synthesize patchato
# ──────────────────────────────────────────────────────────────────────────────
def case_tts_generate_default(w, spy):
    """'Genera voce': con voce+testo deve chiamare synthesize, salvare _tts_last_file ed
    abilitare i bottoni play/usa-come-audio + emettere tts_generated."""
    AT = _stub_voice_lists()
    cap = _patch_synthesize(AT)
    t = w.audio_tab
    _select_engine(t, "edge")
    t._current_voice = lambda: _fake_voice()
    t.tts_text.setPlainText("buongiorno a tutti")
    emitted: list[str] = []
    t.tts_generated.connect(lambda p: emitted.append(p))
    t._generate_tts()
    assert cap, "synthesize non è stato chiamato"
    assert cap[-1]["text"].strip(), f"testo non passato a synthesize: {cap}"
    assert t._tts_last_file, "_tts_last_file non impostato"
    assert t.tts_play_btn.isEnabled() and t.tts_use_btn.isEnabled(), "bottoni play/usa non abilitati"
    assert emitted, "segnale tts_generated non emesso"


def case_tts_generate_empty_text_no_synth(w, spy):
    """Senza testo, 'Genera voce' deve avvisare e NON chiamare synthesize."""
    AT = _stub_voice_lists()
    cap = _patch_synthesize(AT)
    t = w.audio_tab
    _select_engine(t, "edge")
    t._current_voice = lambda: _fake_voice()
    t.tts_text.setPlainText("")
    t._generate_tts()
    assert not cap, "synthesize non doveva essere chiamato senza testo"


def case_tts_generate_no_voice_no_synth(w, spy):
    """Senza voce selezionata, 'Genera voce' deve avvisare e NON chiamare synthesize."""
    AT = _stub_voice_lists()
    cap = _patch_synthesize(AT)
    t = w.audio_tab
    t._current_voice = lambda: None
    t.tts_text.setPlainText("ciao")
    t._generate_tts()
    assert not cap, "synthesize non doveva essere chiamato senza voce"


def case_tts_synth_error_no_crash(w, spy):
    """Se synthesize solleva, l'handler deve gestirlo (dialog critico) senza propagare."""
    AT = _stub_voice_lists()

    def boom(*a, **k):
        raise RuntimeError("engine non installato")
    AT.synthesize = boom
    t = w.audio_tab
    _select_engine(t, "edge")
    t._current_voice = lambda: _fake_voice()
    t.tts_text.setPlainText("prova")
    t._generate_tts()   # non deve sollevare
    assert t.tts_status.text(), "lo stato dovrebbe riportare l'errore"


# ──────────────────────────────────────────────────────────────────────────────
# Uso del TTS come traccia + consenso (integrazione con main_window)
# ──────────────────────────────────────────────────────────────────────────────
def case_use_tts_as_audio(w, spy):
    """'Usa come audio' deve impostare audio_file=ultimo TTS e passare a modalità utente."""
    AT = _stub_voice_lists()
    _patch_synthesize(AT)
    t = w.audio_tab
    _select_engine(t, "edge")
    t._current_voice = lambda: _fake_voice()
    t.tts_text.setPlainText("ciao")
    t._generate_tts()
    t._use_tts_as_audio()
    assert t.rb_user.isChecked(), "non è passato a modalità utente"
    cfg = t.get_audio_config()
    assert cfg["audio_file"] == t._tts_last_file, f"TTS non applicato come audio_file: {cfg}"


def case_pending_tts_consent_flow(w, spy):
    """Dopo la generazione (senza applicare), has_pending_tts ritorna il path; apply_tts_as_user_audio
    lo applica come traccia utente (flusso di consenso usato in main_window.on_generate)."""
    AT = _stub_voice_lists()
    _patch_synthesize(AT)
    t = w.audio_tab
    _select_engine(t, "edge")
    t._current_voice = lambda: _fake_voice()
    t.rb_native.setChecked(True); t._on_mode()
    t.tts_text.setPlainText("ciao")
    t._generate_tts()
    assert t.has_pending_tts() == t._tts_last_file, "has_pending_tts deve segnalare il TTS non applicato"
    t.apply_tts_as_user_audio()
    assert t.rb_user.isChecked() and t.has_pending_tts() is None, "apply non ha applicato il TTS"


# ──────────────────────────────────────────────────────────────────────────────
# CRUD voci (no-crash: i dialog sono auto-annullati dall'harness)
# ──────────────────────────────────────────────────────────────────────────────
def case_voice_create_dialog_cancelled(w, spy):
    """Aprire 'nuova voce' col dialog annullato (harness) non deve crashare né creare voci."""
    _stub_voice_lists()
    w.audio_tab._voice_create()   # QDialog.exec patchato → 0 (annulla)


def case_voice_edit_preset_blocked(w, spy):
    """Modificare un PRESET (custom=False) deve avvisare e non aprire l'editor (no-crash)."""
    _stub_voice_lists()
    t = w.audio_tab
    t._current_voice = lambda: VoicePreset(name="Preset", engine="edge",
                                           voice_id="it-IT-ElsaNeural", custom=False)
    t._voice_edit()   # deve solo mostrare info, no crash


def case_voice_delete_preset_blocked(w, spy):
    """Eliminare un PRESET (custom=False) deve essere bloccato (no-crash)."""
    _stub_voice_lists()
    t = w.audio_tab
    t._current_voice = lambda: VoicePreset(name="Preset", engine="edge",
                                           voice_id="it-IT-ElsaNeural", custom=False)
    t._voice_delete()   # info + return, no crash


# ──────────────────────────────────────────────────────────────────────────────
# Install motore (no-crash) — install_engine patchato
# ──────────────────────────────────────────────────────────────────────────────
def case_install_button_visibility(w, spy):
    """Il bottone Install è nascosto per 'edge' e visibile per i motori che richiedono nodi."""
    _stub_voice_lists()
    t = w.audio_tab
    w.tabs.setCurrentWidget(w.audio_tab)
    _select_engine(t, "edge"); t._update_install_button()
    assert t.tts_install_btn.isVisibleTo(t) is False, "Install non deve comparire per edge"
    _select_engine(t, "cosyvoice2"); t._update_install_button()
    assert t.tts_install_btn.isVisibleTo(t) is True, "Install deve comparire per cosyvoice2"


def case_install_engine_no_crash(w, spy):
    """Avviare l'install di un motore (install_engine patchato) non deve crashare."""
    from ui.tabs import audio_tab as AT
    _stub_voice_lists()
    AT.install_engine = lambda eid, **k: (True, "ok (stub)")
    AT.invalidate_engine_cache = lambda: None
    t = w.audio_tab
    _select_engine(t, "f5tts")
    t._install_selected_engine()   # QMessageBox patchati → Yes, dialog risultato non-bloccante


# ──────────────────────────────────────────────────────────────────────────────
# PER MOTORE (modello): la GUI deve instradare OGNI motore selezionato a synthesize
# (così si scopre "funziona solo un motore TTS").
# ──────────────────────────────────────────────────────────────────────────────
def _make_engine_case(eid: str):
    def _case(w, spy):
        AT = _stub_voice_lists()
        cap = _patch_synthesize(AT)
        t = w.audio_tab
        assert _select_engine(t, eid), f"motore {eid} assente nel combo tts_engine"
        # per i motori di cloning passo anche un campione (innocuo: synthesize è patchato)
        t._current_voice = lambda: _fake_voice(sample=str(H.OUT_DIR / "s.wav"))
        t.tts_text.setPlainText(f"prova del motore {eid}")
        t._generate_tts()
        assert cap, f"synthesize non chiamato per il motore {eid}"
        assert cap[-1]["engine"] == eid, \
            f"la GUI ha instradato il motore sbagliato: atteso {eid}, ottenuto {cap[-1]['engine']}"
    return _case


_ENGINE_CASES = [
    (f"tts_generate_engine_{eid}",
     f"Genera voce con motore '{eid}' ({disp}): la GUI instrada {eid} a synthesize.",
     _make_engine_case(eid))
    for eid, disp, *_ in ENGINES_META
]


# ──────────────────────────────────────────────────────────────────────────────
# Registro casi
# ──────────────────────────────────────────────────────────────────────────────
CASES = [
    ("audio_config_native", "Modalità nativa: get_audio_config → audio_native, niente sorgenti file.",
     case_config_native),
    ("audio_config_user_file", "Modalità utente: il file impostato finisce in audio_file.",
     case_config_user_file),
    ("audio_config_clone", "Modalità clona: clone_voice + voice_sample + tts_engine + tts_text presenti.",
     case_config_clone),
    ("audio_config_mute", "Modalità muto: audio_mute=True, suppress_singing=False.",
     case_config_mute),
    ("audio_no_sing_toggle", "Toggle no-canto: suppress_singing segue la checkbox.",
     case_no_sing_toggle),
    ("audio_mode_switch_no_crash", "Cambio radio modalità: no-crash + box file solo in 'utente'.",
     case_mode_switch_no_crash),
    ("audio_engine_change_reloads", "Cambio motore TTS: ricarica voci senza crash (no rete).",
     case_engine_change_reloads_voices),
    ("audio_reload_voices_no_crash", "_reload_voices con voci stubbate: dropdown popolato, no-crash.",
     case_reload_voices_no_crash),
    ("audio_ssml_marker_insert", "Marker SSML: [pause:500ms] inserito nell'editor.",
     case_ssml_marker_insert),
    ("audio_ssml_marker_wrap", "Marker con '…': avvolge la selezione (prosody).",
     case_ssml_marker_wrap_selection),
    ("audio_preview_strips_markers", "Anteprima: mostra testo pulito senza marker.",
     case_preview_strips_markers),
    ("audio_tts_generate_default", "Genera voce: chiama synthesize, abilita play/usa, emette tts_generated.",
     case_tts_generate_default),
    ("audio_tts_generate_empty_text", "Genera voce senza testo: avvisa, NON chiama synthesize.",
     case_tts_generate_empty_text_no_synth),
    ("audio_tts_generate_no_voice", "Genera voce senza voce: avvisa, NON chiama synthesize.",
     case_tts_generate_no_voice_no_synth),
    ("audio_tts_synth_error_no_crash", "Errore synthesize: gestito con dialog, no crash, stato aggiornato.",
     case_tts_synth_error_no_crash),
    ("audio_use_tts_as_audio", "Usa come audio: imposta audio_file=TTS e passa a modalità utente.",
     case_use_tts_as_audio),
    ("audio_pending_tts_consent", "Consenso TTS: has_pending_tts + apply_tts_as_user_audio.",
     case_pending_tts_consent_flow),
    ("audio_voice_create_cancelled", "Nuova voce (dialog annullato): no-crash, nessuna voce creata.",
     case_voice_create_dialog_cancelled),
    ("audio_voice_edit_preset_blocked", "Modifica preset di sistema: bloccata con avviso, no-crash.",
     case_voice_edit_preset_blocked),
    ("audio_voice_delete_preset_blocked", "Elimina preset di sistema: bloccata, no-crash.",
     case_voice_delete_preset_blocked),
    ("audio_install_button_visibility", "Bottone Install: nascosto per edge, visibile per cosyvoice2.",
     case_install_button_visibility),
    ("audio_install_engine_no_crash", "Install motore (stub): flusso completo senza crash.",
     case_install_engine_no_crash),
] + _ENGINE_CASES
