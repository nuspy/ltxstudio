"""Casi di regressione GUI-driven per il tab «LoRA Video (LTX-2.3)» (w.lora_video_tab).

Il tab NON passa dal backend nativo (native_driver/_enqueue): le azioni «Genera dataset» e
«Genera + Allena LoRA» costruiscono i soggetti e lanciano un QThread `_LoraWorker` che chiama
direttamente `core.lora_video_service.make_lora`/`generate_dataset`. Per intercettare la richiesta
GUI→backend SENZA avviare il thread (niente GPU/dataset reale) si patcha `_LoraWorker.start` con una
funzione che cattura gli argomenti (subjects, name, out_dir, opts, full) in spy.calls.

Copertura:
- azioni: genera dataset (full=False), genera+allena (full=True), installa trainer;
- validazione precondizioni: nessun soggetto / soggetto senza media → warning, nessun avvio;
- ogni TIPO di soggetto del combo `ed_type` (person/place/object/concept) — un caso per tipo, così
  si scopre se «funziona solo person»;
- opzioni/toggle del riquadro training: rank, steps, durata clip, PuLID, Low VRAM, nome LoRA;
- editor soggetto: focus volto, situazioni custom, angoli foto (combo photo_angle), media reale,
  checkbox di copertura, cambio tipo che ricostruisce la copertura, add/remove soggetto, multi-soggetto.
"""
from __future__ import annotations

import harness as H

from core import lora_coverage
from ui.tabs import lora_video_tab as LV


# ───────────────────────── utility ─────────────────────────
def _capture_worker(spy):
    """Patcha _LoraWorker.start per catturare la richiesta SENZA avviare il thread (no GPU)."""
    def fake_start(self):
        subjects, name, out_dir, opts, full = self._a
        params = {"name": name, "out_dir": out_dir, "full": full,
                  "n_subjects": len(subjects), "subjects": list(subjects)}
        params.update(dict(opts or {}))
        spy.calls.append({"via": "lora_worker", "label": "start",
                          "module": "lora_video_service", "params": params})
    LV._LoraWorker.start = fake_start


def _last_worker(spy):
    for c in reversed(spy.calls):
        if c.get("via") == "lora_worker":
            return c
    return None


def _tab(w):
    w.tabs.setCurrentWidget(w.lora_video_tab)
    return w.lora_video_tab


def _add_subject(tab, name="Martina", stype="person", with_photo=True,
                 angle="front", real_media=False):
    """Aggiunge un soggetto via la GUI e imposta tipo/nome/foto direttamente nel modello dati."""
    tab._add_subject()                                   # crea blank + seleziona
    idx = [c for c, _ in LV._TYPE_LABELS].index(stype)
    tab.ed_type.setCurrentIndex(idx)                     # → _on_type_change (se cambia)
    tab.ed_name.setText(name)                            # textChanged → _save_cur
    cur = tab._cur
    if with_photo:
        img = H.make_test_image(str(H.OUT_DIR / "_lora_video_face.png"))
        tab._subjects[cur]["photos"].append({"path": img, "angle": angle})
    if real_media:
        vid = str(H.OUT_DIR / "_lora_video_real.mp4")
        tab._subjects[cur]["real_media"].append(vid)
    tab._refresh_editor()
    tab._save_cur()
    return tab


# ───────────────────────── validazione precondizioni ─────────────────────────
def case_validation_no_subject(w, spy):
    """Senza soggetti l'azione deve mostrare un warning e NON avviare il worker."""
    _capture_worker(spy)
    tab = _tab(w)
    tab._start(True)
    assert _last_worker(spy) is None, "avvio worker senza alcun soggetto (manca la validazione)"


def case_validation_no_media(w, spy):
    """Soggetto con nome ma senza foto/video reale → warning, nessun avvio del worker."""
    _capture_worker(spy)
    tab = _add_subject(w.lora_video_tab and _tab(w), name="Pippo", with_photo=False)
    tab._start(True)
    assert _last_worker(spy) is None, "avvio worker con soggetto privo di media (manca la validazione)"


# ───────────────────────── azioni: dataset / training ─────────────────────────
def case_dataset_only_person(w, spy):
    """«Genera dataset» (full=False): la GUI deve avviare il worker con full=False e ≥1 soggetto."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina", stype="person")
    tab._start(False)
    c = _last_worker(spy)
    assert c, "genera-dataset: nessuna richiesta catturata"
    assert c["params"]["full"] is False, f"atteso full=False: {c['params']}"
    assert c["params"]["n_subjects"] >= 1, f"nessun soggetto inviato: {c['params']}"


def case_full_train_person(w, spy):
    """«Genera + Allena LoRA» (full=True) per tipo PERSON: worker avviato con full=True."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina", stype="person")
    tab._start(True)
    c = _last_worker(spy)
    assert c, "full-train person: nessuna richiesta catturata"
    assert c["params"]["full"] is True, f"atteso full=True: {c['params']}"
    s = c["params"]["subjects"][0]
    assert s.type == "person", f"tipo soggetto errato: {s.type}"


def case_full_train_place(w, spy):
    """Full-train per tipo PLACE: il combo ed_type deve produrre un soggetto type='place'."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Colosseo", stype="place")
    tab._start(True)
    c = _last_worker(spy)
    assert c, "full-train place: nessuna richiesta catturata"
    assert c["params"]["subjects"][0].type == "place", "tipo place non propagato al backend"


def case_full_train_object(w, spy):
    """Full-train per tipo OBJECT."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Vaso", stype="object")
    tab._start(True)
    c = _last_worker(spy)
    assert c, "full-train object: nessuna richiesta catturata"
    assert c["params"]["subjects"][0].type == "object", "tipo object non propagato al backend"


def case_full_train_concept(w, spy):
    """Full-train per tipo CONCEPT (nessun caso di copertura predefinito): deve comunque avviarsi."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Stile_acquerello", stype="concept")
    tab._start(True)
    c = _last_worker(spy)
    assert c, "full-train concept: nessuna richiesta catturata"
    assert c["params"]["subjects"][0].type == "concept", "tipo concept non propagato al backend"


# ───────────────────────── opzioni / toggle training ─────────────────────────
def case_opts_rank_steps_clip(w, spy):
    """rank/steps/durata-clip dei widget devono finire negli opts inviati al worker."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina")
    tab.lora_rank.setValue(64)
    tab.lora_steps.setValue(4000)
    tab.lora_clip.setValue(5.0)
    tab._start(True)
    p = _last_worker(spy)["params"]
    assert p.get("rank") == 64, f"rank non propagato: {p}"
    assert p.get("steps") == 4000, f"steps non propagato: {p}"
    assert abs(p.get("clip_seconds", 0) - 5.0) < 1e-6, f"clip_seconds non propagato: {p}"


def case_toggle_pulid_off(w, spy):
    """Disattivare PuLID deve inviare use_pulid=False."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina")
    tab.lora_pulid.setChecked(False)
    tab._start(True)
    p = _last_worker(spy)["params"]
    assert p.get("use_pulid") is False, f"use_pulid non propagato: {p}"


def case_toggle_lowvram_off(w, spy):
    """Disattivare Low VRAM deve inviare low_vram=False."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina")
    tab.lora_lowvram.setChecked(False)
    tab._start(True)
    p = _last_worker(spy)["params"]
    assert p.get("low_vram") is False, f"low_vram non propagato: {p}"


def case_lora_name_passed(w, spy):
    """Il nome LoRA del campo deve essere passato al worker (non 'my_lora' fisso)."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina")
    tab.lora_name.setText("martina_v1")
    tab._start(True)
    p = _last_worker(spy)["params"]
    assert p.get("name") == "martina_v1", f"nome LoRA non propagato: {p}"


def case_face_focus_off(w, spy):
    """Disattivare 'focus volto' deve riflettersi in Subject.face_focus=False."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina")
    tab.ed_face.setChecked(False)                        # toggled → _save_cur
    tab._start(True)
    s = _last_worker(spy)["params"]["subjects"][0]
    assert s.face_focus is False, f"face_focus non propagato: {s.face_focus}"


def case_customs_situations(w, spy):
    """Le situazioni custom (una per riga) devono entrare in Subject.customs."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina")
    tab.ed_customs.setPlainText("corre sulla spiaggia\nballa felice")   # textChanged → _save_cur
    tab._save_cur()
    tab._start(True)
    s = _last_worker(spy)["params"]["subjects"][0]
    assert "corre sulla spiaggia" in s.customs and "balla felice" in s.customs, \
        f"customs non propagate: {s.customs}"


# ───────────────────────── editor: angoli foto / media reale ─────────────────────────
def case_photo_angles_all(w, spy):
    """Combo angolo foto: tutti gli angoli (_ANGLES) devono poter essere associati alle foto."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina", with_photo=False)
    img = H.make_test_image(str(H.OUT_DIR / "_lora_video_face.png"))
    for ang in LV._ANGLES:
        tab._subjects[tab._cur]["photos"].append({"path": img, "angle": ang})
    tab._refresh_editor(); tab._save_cur()
    tab._start(False)
    s = _last_worker(spy)["params"]["subjects"][0]
    angles = {p["angle"] for p in s.all_photos()}
    assert set(LV._ANGLES) <= angles, f"angoli mancanti tra le foto: {angles}"


def case_real_media_only(w, spy):
    """Soggetto con SOLO materiale reale (niente foto) deve superare la validazione e avviarsi."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina", with_photo=False, real_media=True)
    tab._start(True)
    c = _last_worker(spy)
    assert c, "soggetto con solo media reale: nessun avvio (validazione errata)"
    assert c["params"]["subjects"][0].real_media, "real_media non propagato al backend"


# ───────────────────────── installa trainer ─────────────────────────
def case_install_trainer(w, spy):
    """«Installa trainer» deve lanciare lo script di install (subprocess.Popen) senza crashare."""
    import subprocess
    tab = _tab(w)
    orig = subprocess.Popen
    seen = []
    subprocess.Popen = lambda *a, **k: seen.append(a) or None
    try:
        tab._install()
    finally:
        subprocess.Popen = orig
    assert seen, "install: subprocess.Popen non invocato (nessun launch del trainer)"
    assert any("install_ltx_trainer" in str(x) for x in seen[0][0]), \
        f"install: script atteso non trovato nel comando: {seen}"


# ───────────────────────── copertura / soggetti (solo UI) ─────────────────────────
def case_coverage_checkbox_toggle(w, spy):
    """Disattivare una checkbox di copertura deve rimuovere il caso da Subject.case_ids."""
    _capture_worker(spy)
    tab = _add_subject(_tab(w), name="Martina", stype="person")
    assert tab._cov_checks, "nessuna checkbox di copertura costruita per person"
    first_cid = next(iter(tab._cov_checks))
    tab._cov_checks[first_cid].setChecked(False)         # toggled → _save_cur
    tab._save_cur()
    tab._start(True)
    s = _last_worker(spy)["params"]["subjects"][0]
    assert first_cid not in s.case_ids, f"caso {first_cid} ancora attivo dopo deselezione: {s.case_ids}"


def case_type_change_rebuilds_coverage(w, spy):
    """Cambiare tipo soggetto deve ricostruire le checkbox di copertura coerenti col nuovo tipo."""
    tab = _add_subject(_tab(w), name="X", stype="person")
    # passa a 'place' → coverage deve combaciare con cases_for('place')
    tab.ed_type.setCurrentIndex([c for c, _ in LV._TYPE_LABELS].index("place"))
    expected = {cid for cid, _t, _tags in lora_coverage.cases_for("place")}
    assert set(tab._cov_checks) == expected, \
        f"copertura non ricostruita per 'place': {set(tab._cov_checks)} != {expected}"


def case_add_remove_subject(w, spy):
    """Aggiunta e rimozione di soggetti dalla lista non deve crashare e deve aggiornare il modello."""
    tab = _tab(w)
    _add_subject(tab, name="A", stype="person")
    _add_subject(tab, name="B", stype="object")
    assert len(tab._subjects) == 2, f"attesi 2 soggetti: {len(tab._subjects)}"
    tab.subj_list.setCurrentRow(0)
    tab._del_subject()
    assert len(tab._subjects) == 1, f"rimozione soggetto fallita: {len(tab._subjects)}"


def case_multi_subject_train(w, spy):
    """Più soggetti insieme: il worker deve ricevere TUTTI i soggetti con nome+media."""
    _capture_worker(spy)
    tab = _tab(w)
    _add_subject(tab, name="Martina", stype="person")
    _add_subject(tab, name="Vaso", stype="object")
    tab._start(True)
    c = _last_worker(spy)
    assert c and c["params"]["n_subjects"] == 2, \
        f"attesi 2 soggetti inviati: {c['params'] if c else None}"


CASES = [
    ("lora_video_validation_no_subject",
     "Senza soggetti: warning e nessun avvio del worker di training.", case_validation_no_subject),
    ("lora_video_validation_no_media",
     "Soggetto senza foto/video reale: warning e nessun avvio.", case_validation_no_media),
    ("lora_video_dataset_only_person",
     "Genera dataset (full=False) con un soggetto person: worker avviato.", case_dataset_only_person),
    ("lora_video_full_train_person",
     "Genera+Allena (full=True) tipo person: worker avviato con full=True.", case_full_train_person),
    ("lora_video_full_train_place",
     "Genera+Allena tipo place: il combo tipo propaga type='place'.", case_full_train_place),
    ("lora_video_full_train_object",
     "Genera+Allena tipo object: il combo tipo propaga type='object'.", case_full_train_object),
    ("lora_video_full_train_concept",
     "Genera+Allena tipo concept (no copertura predefinita): si avvia comunque.", case_full_train_concept),
    ("lora_video_opts_rank_steps_clip",
     "rank/steps/durata-clip propagati negli opts del worker.", case_opts_rank_steps_clip),
    ("lora_video_toggle_pulid_off",
     "PuLID off → use_pulid=False negli opts.", case_toggle_pulid_off),
    ("lora_video_toggle_lowvram_off",
     "Low VRAM off → low_vram=False negli opts.", case_toggle_lowvram_off),
    ("lora_video_lora_name_passed",
     "Nome LoRA del campo passato al worker (non fisso).", case_lora_name_passed),
    ("lora_video_face_focus_off",
     "Focus volto off → Subject.face_focus=False.", case_face_focus_off),
    ("lora_video_customs_situations",
     "Situazioni custom (righe) → Subject.customs.", case_customs_situations),
    ("lora_video_photo_angles_all",
     "Combo angolo foto: tutti gli angoli associabili alle foto del soggetto.", case_photo_angles_all),
    ("lora_video_real_media_only",
     "Solo materiale reale (no foto): supera la validazione e si avvia.", case_real_media_only),
    ("lora_video_install_trainer",
     "Installa trainer: lancia lo script install_ltx_trainer senza crash.", case_install_trainer),
    ("lora_video_coverage_checkbox_toggle",
     "Deselezionare una checkbox di copertura rimuove il caso da case_ids.", case_coverage_checkbox_toggle),
    ("lora_video_type_change_rebuilds_coverage",
     "Cambio tipo ricostruisce le checkbox di copertura coerenti.", case_type_change_rebuilds_coverage),
    ("lora_video_add_remove_subject",
     "Aggiunta/rimozione soggetti aggiorna il modello senza crash.", case_add_remove_subject),
    ("lora_video_multi_subject_train",
     "Più soggetti: il worker riceve tutti i soggetti.", case_multi_subject_train),
]
