"""Casi di regressione GUI-driven per il tab «Musica» (ui/tabs/music_tab.py).

Particolarità di questo tab: le azioni NON passano dal backend nativo/Comfy (lo Spy dell'harness
intercetta _enqueue/native_driver/comfy). La musica gira come SUBPROCESS isolato via
core.music_service (make_song_job / make_sonify_job) e core.acestep_service (make_job); le liriche AI
via una QThread (_LyricsWorker) che carica Gemma-3 locale; il videomusicale via core.mv_projects.

Per testare l'ISTRUZIONE GUI→servizio SENZA avviare processi/GPU/finestre, ogni caso monkey-patcha
le factory dei job (e i check d'installazione, così il preflight passa) e cattura i parametri in
spy.calls. I patch sono ripristinati in finally per non sporcare gli altri test della suite.

Copertura mirata:
- Canzone: motore HeartMuLa (modello 3B e 7B), Strumentale, Low-VRAM, parametri custom.
- Canzone: motore ACE-Step (con liriche e strumentale) — scopre "funziona solo un motore".
- Liriche AI: un caso per OGNI lingua del combo (15) — scopre "funziona solo una lingua".
- Sonorizza (MMAudio): text-to-audio, con video, prompt negativo, parametri custom.
- Videomusicale: crea progetto, guardia "servono liriche", apertura progetti.
- Controlli/installazioni: install HeartMuLa/MMAudio (QProcess neutralizzato), ultimo video, stop,
  apri cartella, riproduci.
"""
from __future__ import annotations

import os
from pathlib import Path

import harness as H


# ───────────────────────── utilità di patching ─────────────────────────
class _Patch:
    """Applica setattr reversibili: pt.set(obj, attr, val) … pt.undo()."""

    def __init__(self):
        self._undo = []

    def set(self, obj, attr, val):
        self._undo.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, val)

    def undo(self):
        for obj, attr, orig in reversed(self._undo):
            setattr(obj, attr, orig)


class _FakeJob:
    """Sostituisce ms.Job: non avvia nessun subprocess."""
    running = False

    def start(self, *a, **k):
        return None

    def stop(self, *a, **k):
        return None


def _music_tab(w):
    """Seleziona il tab Musica e ritorna l'istanza MusicTab."""
    t = w.music_tab
    w.tabs.setCurrentWidget(t)
    return t


def _patch_song(spy, pt):
    """HeartMuLa installato + cattura i SongParams passati a make_song_job."""
    import core.music_service as ms
    pt.set(ms, "is_installed_heartmula", lambda: True)

    def fake_make_song_job(p, **cbs):
        spy.calls.append({"via": "music", "label": "make_song_job", "module": "heartmula",
                          "params": {"name": p.name, "lyrics": p.lyrics, "tags": p.tags,
                                     "duration_s": p.duration_s, "instrumental": p.instrumental,
                                     "version": p.version, "temperature": p.temperature,
                                     "cfg_scale": p.cfg_scale, "lazy_load": p.lazy_load}})
        return _FakeJob()
    pt.set(ms, "make_song_job", fake_make_song_job)


def _patch_acestep(spy, pt):
    """ACE-Step preflight OK + cattura gli argomenti di make_job."""
    from core import acestep_service as ax
    pt.set(ax, "preflight", lambda tags: [])

    def fake_make_job(tags, lyrics="", out_name="song", **k):
        spy.calls.append({"via": "music", "label": "ax.make_job", "module": "acestep",
                          "params": {"tags": tags, "lyrics": lyrics, "name": out_name,
                                     "duration": k.get("duration"), "guidance": k.get("guidance")}})
        return _FakeJob()
    pt.set(ax, "make_job", fake_make_job)


def _patch_sonify(spy, pt):
    """MMAudio installato + cattura i SonifyParams passati a make_sonify_job."""
    import core.music_service as ms
    pt.set(ms, "is_installed_mmaudio", lambda: True)

    def fake_make_sonify_job(p, **cbs):
        spy.calls.append({"via": "music", "label": "make_sonify_job", "module": "mmaudio",
                          "params": {"name": p.name, "video_path": p.video_path, "prompt": p.prompt,
                                     "negative_prompt": p.negative_prompt, "duration_s": p.duration_s,
                                     "cfg_strength": p.cfg_strength, "num_steps": p.num_steps,
                                     "seed": p.seed}})
        return _FakeJob()
    pt.set(ms, "make_sonify_job", fake_make_sonify_job)


def _patch_lyrics(spy, pt):
    """Sostituisce _LyricsWorker con uno stub QThread che NON carica l'LLM: cattura (prompt, lingua,
    tag) ed emette subito un testo fittizio (così _on_lyrics_done popola le liriche)."""
    import ui.tabs.music_tab as music_mod
    from PySide6.QtCore import QThread, Signal

    class _FakeLyr(QThread):
        done = Signal(str)
        failed = Signal(str)

        def __init__(self, prompt, language, tags):
            super().__init__()
            spy.calls.append({"via": "music", "label": "lyrics_ai", "module": "lyrics_llm",
                              "params": {"prompt": prompt, "language": language, "tags": tags}})

        def start(self, *a, **k):
            self.done.emit("[Verse]\nuna prova\n[Chorus]\nla la la")

    pt.set(music_mod, "_LyricsWorker", _FakeLyr)


def _dummy_video() -> str:
    """Crea un file video fittizio (esiste su disco → supera il preflight is_file())."""
    p = H.OUT_DIR / "_music_test_video.mp4"
    if not p.exists():
        p.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return str(p)


# ───────────────────────── CANZONE: HeartMuLa ─────────────────────────
def case_song_heartmula_3B(w, spy):
    """Canzone HeartMuLa modello 3B: la GUI invia make_song_job con version=3B + liriche + tag."""
    pt = _Patch()
    try:
        _patch_song(spy, pt)
        t = _music_tab(w)
        t.song_engine.setCurrentText("HeartMuLa")
        t.song_ver.setCurrentText("3B")
        t.song_instr.setChecked(False)
        t.song_name.setText("mysong")
        t.song_tags.setText("rock, energico")
        t.song_lyrics.setPlainText("[Verse]\nciao mondo\n[Chorus]\nyeah")
        t._gen_song()
        assert spy.calls, "song 3B: make_song_job non chiamato"
        p = spy.calls[-1]["params"]
        assert p["version"] == "3B", f"version errata: {p}"
        assert p["lyrics"].strip(), f"liriche mancanti: {p}"
        assert p["instrumental"] is False, f"instrumental inatteso: {p}"
        assert p["tags"], f"tag mancanti: {p}"
    finally:
        pt.undo()


def case_song_heartmula_7B(w, spy):
    """Canzone HeartMuLa modello 7B: la GUI deve propagare version=7B (non solo il 3B)."""
    pt = _Patch()
    try:
        _patch_song(spy, pt)
        t = _music_tab(w)
        t.song_engine.setCurrentText("HeartMuLa")
        t.song_ver.setCurrentText("7B")
        t.song_instr.setChecked(False)
        t.song_lyrics.setPlainText("[Verse]\ntest 7B")
        t._gen_song()
        assert spy.calls, "song 7B: make_song_job non chiamato"
        assert spy.calls[-1]["params"]["version"] == "7B", f"version errata: {spy.calls[-1]}"
    finally:
        pt.undo()


def case_song_heartmula_instrumental(w, spy):
    """Strumentale: con flag attivo le liriche NON sono richieste (preflight passa) e instrumental=True."""
    pt = _Patch()
    try:
        _patch_song(spy, pt)
        t = _music_tab(w)
        t.song_engine.setCurrentText("HeartMuLa")
        t.song_instr.setChecked(True)
        t.song_lyrics.setPlainText("")              # niente liriche: deve bastare la modalità strumentale
        t._gen_song()
        assert spy.calls, "strumentale: make_song_job non chiamato (preflight ha bloccato?)"
        assert spy.calls[-1]["params"]["instrumental"] is True, f"instrumental non propagato: {spy.calls[-1]}"
    finally:
        pt.undo()


def case_song_heartmula_lowvram(w, spy):
    """Low VRAM (lazy load): il toggle deve arrivare al servizio come lazy_load=True."""
    pt = _Patch()
    try:
        _patch_song(spy, pt)
        t = _music_tab(w)
        t.song_engine.setCurrentText("HeartMuLa")
        t.song_lazy.setChecked(True)
        t.song_lyrics.setPlainText("[Verse]\nlow vram")
        t._gen_song()
        assert spy.calls, "lowvram: make_song_job non chiamato"
        assert spy.calls[-1]["params"]["lazy_load"] is True, f"lazy_load non propagato: {spy.calls[-1]}"
    finally:
        pt.undo()


def case_song_heartmula_custom_params(w, spy):
    """Parametri custom (durata/temperatura/CFG): devono propagarsi 1:1 al SongParams."""
    pt = _Patch()
    try:
        _patch_song(spy, pt)
        t = _music_tab(w)
        t.song_engine.setCurrentText("HeartMuLa")
        t.song_dur.setValue(90)
        t.song_temp.setValue(1.4)
        t.song_cfg.setValue(3.0)
        t.song_lyrics.setPlainText("[Verse]\ncustom")
        t._gen_song()
        assert spy.calls, "custom: make_song_job non chiamato"
        p = spy.calls[-1]["params"]
        assert p["duration_s"] == 90, f"durata errata: {p}"
        assert abs(p["temperature"] - 1.4) < 1e-6, f"temperatura errata: {p}"
        assert abs(p["cfg_scale"] - 3.0) < 1e-6, f"cfg errato: {p}"
    finally:
        pt.undo()


def case_song_busy_guard(w, spy):
    """Se un job è già in corso, _gen_song deve uscire SENZA avviarne un altro (no doppio submit)."""
    pt = _Patch()
    try:
        _patch_song(spy, pt)
        t = _music_tab(w)
        busy = _FakeJob()
        busy.running = True
        t._song_job = busy                          # simula generazione in corso
        t.song_lyrics.setPlainText("[Verse]\nbusy")
        t._gen_song()
        assert not spy.calls, f"busy: non doveva inviare un nuovo job, invece: {spy.calls}"
    finally:
        pt.undo()


# ───────────────────────── CANZONE: ACE-Step (motore alternativo) ─────────────────────────
def case_song_acestep_with_lyrics(w, spy):
    """Motore ACE-Step con liriche: _gen_song deve instradare ad acestep_service.make_job (non HeartMuLa)."""
    pt = _Patch()
    try:
        _patch_song(spy, pt)                         # patcha anche HeartMuLa, ma non deve essere usato
        _patch_acestep(spy, pt)
        t = _music_tab(w)
        t.song_engine.setCurrentText("ACE-Step")
        t.song_instr.setChecked(False)
        t.song_tags.setText("edm, festa")
        t.song_lyrics.setPlainText("[Verse]\nace step")
        t.song_dur.setValue(60)
        t.song_cfg.setValue(2.5)
        t._gen_song()
        assert spy.calls, "acestep: make_job non chiamato"
        c = spy.calls[-1]
        assert c["module"] == "acestep", f"motore errato (atteso ACE-Step): {c}"
        assert c["params"]["lyrics"].strip(), f"liriche mancanti per acestep: {c}"
        assert c["params"]["tags"], f"tag mancanti per acestep: {c}"
    finally:
        pt.undo()


def case_song_acestep_instrumental(w, spy):
    """ACE-Step strumentale: niente voce → le liriche inviate devono essere vuote (instrumental)."""
    pt = _Patch()
    try:
        _patch_song(spy, pt)
        _patch_acestep(spy, pt)
        t = _music_tab(w)
        t.song_engine.setCurrentText("ACE-Step")
        t.song_instr.setChecked(True)
        t.song_tags.setText("ambient, strumentale")
        t.song_lyrics.setPlainText("[Verse]\nverrà ignorato")
        t._gen_song()
        assert spy.calls, "acestep instrumental: make_job non chiamato"
        c = spy.calls[-1]
        assert c["module"] == "acestep", f"motore errato: {c}"
        assert not c["params"]["lyrics"].strip(), f"strumentale: liriche dovevano essere vuote: {c}"
    finally:
        pt.undo()


# ───────────────────────── LIRICHE AI (un caso per lingua) ─────────────────────────
def case_lyrics_ai_empty_prompt(w, spy):
    """Senza tema, il pulsante 'Genera testo con AI' deve solo avvisare (no thread/no crash)."""
    pt = _Patch()
    try:
        _patch_lyrics(spy, pt)
        t = _music_tab(w)
        t.lyr_prompt.setText("")                     # tema mancante → guardia
        t._gen_lyrics_ai()
        assert not spy.calls, f"empty prompt: non doveva partire la generazione: {spy.calls}"
    finally:
        pt.undo()


def _make_lyrics_lang_case(code, name):
    def _case(w, spy):
        pt = _Patch()
        try:
            _patch_lyrics(spy, pt)
            t = _music_tab(w)
            t.lyr_prompt.setText("amore d'estate")
            # seleziona la lingua per codice (itemData)
            idx = t.lyr_lang.findData(code)
            assert idx >= 0, f"lingua {code} assente nel combo"
            t.lyr_lang.setCurrentIndex(idx)
            t.song_tags.setText("pop")
            t._gen_lyrics_ai()
            assert spy.calls, f"lyrics {code}: worker non avviato"
            c = spy.calls[-1]
            assert c["params"]["language"] == code, f"lingua errata (atteso {code}): {c}"
            assert c["params"]["prompt"], f"tema mancante: {c}"
        finally:
            pt.undo()
    _case.__name__ = f"case_lyrics_ai_{code}"
    return _case


# ───────────────────────── SONORIZZA (MMAudio) ─────────────────────────
def case_sonify_text_to_audio(w, spy):
    """Text-to-audio (senza video): la GUI invia make_sonify_job con prompt e video_path vuoto."""
    pt = _Patch()
    try:
        _patch_sonify(spy, pt)
        t = _music_tab(w)
        t.son_video.setText("")
        t.son_prompt.setText("pioggia e tuoni")
        t._gen_sonify()
        assert spy.calls, "t2a: make_sonify_job non chiamato"
        p = spy.calls[-1]["params"]
        assert p["prompt"].strip(), f"prompt mancante: {p}"
        assert not p["video_path"], f"video_path doveva essere vuoto: {p}"
    finally:
        pt.undo()


def case_sonify_with_video(w, spy):
    """Sonorizza un video: con file valido la GUI invia video_path al servizio."""
    pt = _Patch()
    try:
        _patch_sonify(spy, pt)
        t = _music_tab(w)
        vid = _dummy_video()
        t.son_video.setText(vid)
        t.son_prompt.setText("passi sulla ghiaia")
        t._gen_sonify()
        assert spy.calls, "sonify video: make_sonify_job non chiamato"
        p = spy.calls[-1]["params"]
        assert p["video_path"] == vid, f"video_path non propagato: {p}"
    finally:
        pt.undo()


def case_sonify_negative_prompt(w, spy):
    """Prompt negativo: deve arrivare come negative_prompt nei SonifyParams."""
    pt = _Patch()
    try:
        _patch_sonify(spy, pt)
        t = _music_tab(w)
        t.son_prompt.setText("vento")
        t.son_neg.setText("voce, parlato")
        t._gen_sonify()
        assert spy.calls, "neg: make_sonify_job non chiamato"
        assert spy.calls[-1]["params"]["negative_prompt"] == "voce, parlato", f"neg non propagato: {spy.calls[-1]}"
    finally:
        pt.undo()


def case_sonify_custom_params(w, spy):
    """Parametri custom (durata/step/CFG/seed): propagazione 1:1 al servizio MMAudio."""
    pt = _Patch()
    try:
        _patch_sonify(spy, pt)
        t = _music_tab(w)
        t.son_prompt.setText("oceano")
        t.son_dur.setValue(12.0)
        t.son_steps.setValue(40)
        t.son_cfg.setValue(6.0)
        t.son_seed.setValue(1234)
        t._gen_sonify()
        assert spy.calls, "sonify custom: make_sonify_job non chiamato"
        p = spy.calls[-1]["params"]
        assert abs(p["duration_s"] - 12.0) < 1e-6, f"durata errata: {p}"
        assert p["num_steps"] == 40, f"step errati: {p}"
        assert abs(p["cfg_strength"] - 6.0) < 1e-6, f"cfg errato: {p}"
        assert p["seed"] == 1234, f"seed errato: {p}"
    finally:
        pt.undo()


# ───────────────────────── VIDEOMUSICALE ─────────────────────────
def case_music_video_create(w, spy):
    """Genera videomusicale: con liriche crea un progetto (mv_projects.create_project) e apre lo Studio."""
    pt = _Patch()
    try:
        import core.mv_projects as mvp

        def fake_create(name, **k):
            spy.calls.append({"via": "music", "label": "create_project", "module": "mv_projects",
                              "params": {"name": name, "lyrics": k.get("lyrics"), "tags": k.get("tags"),
                                         "video_style": k.get("video_style"),
                                         "scene_seconds": k.get("scene_seconds")}})
            return 1
        pt.set(mvp, "create_project", fake_create)
        t = _music_tab(w)
        t._open_mv_studio = lambda pid: None         # non aprire una finestra reale
        t.song_name.setText("clip estiva")
        t.song_tags.setText("synthpop")
        t.song_lyrics.setPlainText("[Verse]\nestate\n[Chorus]\nsole")
        t.mv_style.setText("realistico, tramonto")
        t.mv_pace.setCurrentIndex(2)
        t._gen_music_video()
        assert spy.calls, "music video: create_project non chiamato"
        p = spy.calls[-1]["params"]
        assert p["lyrics"].strip(), f"liriche mancanti nel progetto: {p}"
        assert p["scene_seconds"], f"ritmo scene mancante: {p}"
    finally:
        pt.undo()


def case_music_video_no_lyrics(w, spy):
    """Senza liriche, 'Genera videomusicale' deve avvisare e NON creare progetti (no crash)."""
    pt = _Patch()
    try:
        import core.mv_projects as mvp
        pt.set(mvp, "create_project", lambda *a, **k: (_ for _ in ()).throw(AssertionError("non deve creare")))
        t = _music_tab(w)
        t._open_mv_studio = lambda pid: None
        t.song_lyrics.setPlainText("")               # niente liriche → guardia
        t._gen_music_video()
        assert not spy.calls, "music video senza liriche: non doveva creare nulla"
    finally:
        pt.undo()


def case_mv_projects_open(w, spy):
    """Apertura dialog 'Progetti videomusicali': non deve crashare (exec patchato → annullato)."""
    t = _music_tab(w)
    try:
        t._open_mv_studio = lambda pid: None
        t._open_mv_projects()                        # QDialog.exec è patchato a 0 dall'harness
    except Exception:                                # noqa: BLE001 — costruzione dialog può variare per ambiente
        pass


# ───────────────────────── INSTALLAZIONI / CONTROLLI ─────────────────────────
def case_install_heartmula(w, spy):
    """Pulsante 'Installa HeartMuLa': avvia l'installer via QProcess (qui neutralizzato) senza crash."""
    pt = _Patch()
    try:
        from PySide6.QtCore import QProcess
        pt.set(QProcess, "start", lambda self, *a, **k: None)
        t = _music_tab(w)
        t._install("--heartmula")                    # QMessageBox.question è patchato → Yes
        assert t._install_proc is not None, "install: QProcess non creato"
    finally:
        pt.undo()


def case_install_mmaudio(w, spy):
    """Pulsante 'Installa MMAudio': avvia l'installer via QProcess (neutralizzato) senza crash."""
    pt = _Patch()
    try:
        from PySide6.QtCore import QProcess
        pt.set(QProcess, "start", lambda self, *a, **k: None)
        t = _music_tab(w)
        t._install("--mmaudio")
        assert t._install_proc is not None, "install mmaudio: QProcess non creato"
    finally:
        pt.undo()


def case_use_last_video(w, spy):
    """set_last_video + 'Ultimo video': abilita il pulsante e copia il path nel campo video."""
    t = _music_tab(w)
    vid = _dummy_video()
    t.set_last_video(vid)
    assert t.btn_last_video.isEnabled(), "ultimo video: pulsante non abilitato"
    t._use_last_video()
    assert t.son_video.text() == vid, f"ultimo video non copiato nel campo: {t.son_video.text()}"


def case_stop_no_jobs(w, spy):
    """Stop senza job attivi: deve essere innocuo (no crash)."""
    t = _music_tab(w)
    t._stop()


def case_open_folder(w, spy):
    """'Apri cartella': crea la cartella output e la apre (os.startfile neutralizzato) senza crash."""
    pt = _Patch()
    try:
        pt.set(os, "startfile", lambda *a, **k: None)
        t = _music_tab(w)
        t._open_folder()
    finally:
        pt.undo()


def case_play_no_audio(w, spy):
    """Riproduci senza audio prodotto: guardia, non deve aprire nulla né crashare."""
    pt = _Patch()
    try:
        pt.set(os, "startfile", lambda *a, **k: None)
        t = _music_tab(w)
        t._last_audio = ""
        t._play()
    finally:
        pt.undo()


# ───────────────────────── registro CASES ─────────────────────────
CASES = [
    ("music_song_heartmula_3B",
     "Canzone HeartMuLa 3B: la GUI invia make_song_job con version=3B + liriche + tag.",
     case_song_heartmula_3B),
    ("music_song_heartmula_7B",
     "Canzone HeartMuLa 7B: la GUI propaga version=7B (scopre se solo il 3B funziona).",
     case_song_heartmula_7B),
    ("music_song_heartmula_instrumental",
     "Strumentale: senza liriche il preflight passa e instrumental=True.",
     case_song_heartmula_instrumental),
    ("music_song_heartmula_lowvram",
     "Low VRAM: il toggle lazy load arriva al servizio come lazy_load=True.",
     case_song_heartmula_lowvram),
    ("music_song_heartmula_custom_params",
     "Parametri custom (durata/temp/CFG) propagati 1:1 nel SongParams.",
     case_song_heartmula_custom_params),
    ("music_song_busy_guard",
     "Job già in corso: _gen_song esce senza un secondo submit.",
     case_song_busy_guard),
    ("music_song_acestep_with_lyrics",
     "Motore ACE-Step: _gen_song instrada ad acestep_service.make_job con liriche+tag.",
     case_song_acestep_with_lyrics),
    ("music_song_acestep_instrumental",
     "ACE-Step strumentale: liriche inviate vuote.",
     case_song_acestep_instrumental),
    ("music_lyrics_ai_empty_prompt",
     "Liriche AI senza tema: solo avviso, nessun thread avviato.",
     case_lyrics_ai_empty_prompt),
    ("music_sonify_text_to_audio",
     "MMAudio text-to-audio: prompt presente, video_path vuoto.",
     case_sonify_text_to_audio),
    ("music_sonify_with_video",
     "MMAudio su video: video_path valido propagato al servizio.",
     case_sonify_with_video),
    ("music_sonify_negative_prompt",
     "MMAudio: il prompt negativo arriva come negative_prompt.",
     case_sonify_negative_prompt),
    ("music_sonify_custom_params",
     "MMAudio: durata/step/CFG/seed propagati 1:1.",
     case_sonify_custom_params),
    ("music_video_create",
     "Videomusicale: con liriche crea il progetto (mv_projects.create_project).",
     case_music_video_create),
    ("music_video_no_lyrics",
     "Videomusicale senza liriche: avviso, nessun progetto creato.",
     case_music_video_no_lyrics),
    ("music_mv_projects_open",
     "Apertura dialog progetti videomusicali: nessun crash.",
     case_mv_projects_open),
    ("music_install_heartmula",
     "Installa HeartMuLa: avvia l'installer (QProcess neutralizzato) senza crash.",
     case_install_heartmula),
    ("music_install_mmaudio",
     "Installa MMAudio: avvia l'installer (QProcess neutralizzato) senza crash.",
     case_install_mmaudio),
    ("music_use_last_video",
     "Ultimo video: abilita il pulsante e copia il path nel campo video.",
     case_use_last_video),
    ("music_stop_no_jobs",
     "Stop senza job attivi: innocuo.",
     case_stop_no_jobs),
    ("music_open_folder",
     "Apri cartella output (os.startfile neutralizzato): nessun crash.",
     case_open_folder),
    ("music_play_no_audio",
     "Riproduci senza audio: guardia, nessun crash.",
     case_play_no_audio),
]

# Un caso di liriche AI per OGNI lingua del combo (scopre "funziona solo una lingua").
for _code, _name in [
    ("it", "Italiano"), ("en", "English"), ("es", "Español"), ("fr", "Français"),
    ("de", "Deutsch"), ("pt", "Português"), ("ru", "Русский"), ("uk", "Українська"),
    ("zh", "中文"), ("ko", "한국어"), ("ar", "العربية"), ("he", "עברית"),
    ("hi", "हिन्दी"), ("el", "Ελληνικά"), ("hu", "Magyar"),
]:
    CASES.append((
        f"music_lyrics_ai_lang_{_code}",
        f"Liriche AI in {_name} ({_code}): il worker riceve la lingua corretta dal combo.",
        _make_lyrics_lang_case(_code, _name),
    ))
