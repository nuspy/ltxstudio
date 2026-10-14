"""Casi di regressione GUI-driven per il tab «Modifica Video AI» (video_edit_ai_tab).

Pattern: vedi tests/gui_regression/cases.py + harness.py. Ogni caso imposta i campi del tab,
invoca l'AZIONE (VideoEditAITab._go → segnale request_edit → MainWindow._gen_video_edit_ai →
self._enqueue, INTERCETTATO dallo Spy), e verifica no-crash + richiesta corretta.

NOTA sul backend di questo tab: _gen_video_edit_ai NON usa native_driver né passa params al worker
via attributi `module`/`params`; crea un _VideoEditAIWorker(cfg) (QThread, subprocess) e lo accoda
con self._enqueue(). Quindi lo Spy cattura la chiamata enqueue (params={} perché il worker non espone
.params), e la CONFIG reale dell'azione è in `w._edit_ai_worker.cfg`. Le assert verificano entrambi:
  - spy.calls            → l'azione ha effettivamente accodato un lavoro (request partita)
  - w._edit_ai_worker.cfg → mode/edit_lora/prompt/strength/parametri inpaint corretti

Casi di VALIDAZIONE (no video / inpaint senza oggetto / prompt vuoto): l'azione deve BLOCCARE e
NON emettere request_edit → si verifica l'ASSENZA di spy.calls.

I dropdown "modello" di questo tab sono le MODALITÀ (mode combo): 4 voci concrete, ognuna con una
edit_lora diversa → un caso per OGNI modalità (così si scopre se "funziona solo una modalità").
"""
from __future__ import annotations

import os

import harness as H


# ── helper locali ────────────────────────────────────────────────────────────────
def _tab(w):
    return w.video_edit_ai_tab


def _ack_dev(v: bool):
    """«Non ripetere in questa sessione» dell'avviso sul checkpoint dev (attributo di classe, solo sessione)."""
    from ui.tabs.video_edit_ai_tab import VideoEditAITab
    VideoEditAITab._dev_ack_session = bool(v)


def _set_video(w, name: str = "_vedit_test_input.mp4") -> str:
    """Precondizione: imposta un video sorgente (file dummy: il worker NON viene eseguito)."""
    path = str(H.OUT_DIR / name)
    if not os.path.exists(path):
        try:
            with open(path, "wb") as fh:
                fh.write(b"\x00\x00\x00\x18ftypmp42")    # header fittizio, basta come path valido
        except Exception:                                # noqa: BLE001
            pass
    _tab(w).set_video(path)
    _ack_dev(True)                                   # casi storici: niente dialog modale sul checkpoint dev
    return path


def _select_mode(w, key: str) -> str:
    """Seleziona la modalità per chiave (famiglia + modalità, UI a due livelli); ritorna il testo visibile."""
    t = _tab(w)
    t.select_mode(key)
    assert t.mode.currentData() == key, f"select_mode({key}) non ha selezionato la modalità: {t.mode.currentData()}"
    return t.mode.currentText()


def _cfg_of(w) -> dict:
    """Config dell'ultima azione accodata (worker creato da _gen_video_edit_ai)."""
    wk = getattr(w, "_edit_ai_worker", None)
    assert wk is not None, "nessun _VideoEditAIWorker creato: l'azione non è arrivata a _gen_video_edit_ai"
    return dict(getattr(wk, "cfg", {}) or {})


# ── un caso per OGNI modalità/LoRA (i 4 'modelli' concreti del tab) ────────────────
_MODE_LORA = {
    "edit": "ltx23_edit_anything",
    "inpaint": "ltx23_inpaint_masked",
    "anime2real": "ltx23_anime2real",
    "real2anime": "ltx23_real2anime",
}


def _make_mode_case(key: str, lora_token: str):
    def _case(w, spy):
        w.tabs.setCurrentWidget(_tab(w))
        _set_video(w)
        _select_mode(w, key)
        if key == "inpaint":
            _tab(w).obj.setText("car")
        _tab(w).prompt.setPlainText("a blue motorcycle in the snow")
        _tab(w)._go()
        assert spy.calls, f"modalità {key}: nessuna richiesta accodata (_go non ha chiamato _enqueue)"
        cfg = _cfg_of(w)
        assert cfg.get("mode") == key, f"mode errato: {cfg.get('mode')} (atteso {key})"
        assert lora_token in (cfg.get("edit_lora") or ""), \
            f"edit_lora errata per {key}: {cfg.get('edit_lora')} (atteso contenga {lora_token})"
        assert cfg.get("prompt"), f"prompt mancante in cfg: {cfg}"
        assert cfg.get("video"), f"video mancante in cfg: {cfg}"
    return _case


# ── opzioni: forza editing (combo strength) ────────────────────────────────────────
def _make_strength_case(value: str):
    def _case(w, spy):
        w.tabs.setCurrentWidget(_tab(w))
        _set_video(w)
        _select_mode(w, "edit")
        _tab(w).strength.setCurrentText(value)
        _tab(w).prompt.setPlainText("turn the sky purple")
        _tab(w)._go()
        assert spy.calls, f"strength={value}: nessuna richiesta accodata"
        cfg = _cfg_of(w)
        assert abs(float(cfg.get("strength", -1)) - float(value)) < 1e-6, \
            f"strength non propagata: cfg={cfg.get('strength')} atteso {value}"
    return _case


# ── opzioni inpaint: cfg_scale / guide / mask_dilate dalla GUI ─────────────────────
def case_inpaint_params_propagated(w, spy):
    """Inpaint: i parametri regolabili (CFG, forza guide, dilatazione maschera) devono finire in cfg."""
    w.tabs.setCurrentWidget(_tab(w))
    _set_video(w)
    _select_mode(w, "inpaint")
    _tab(w).obj.setText("person")
    _tab(w).prompt.setPlainText("a robot")
    _tab(w).cfg.setValue(8.5)
    _tab(w).guide.setValue(0.7)
    _tab(w).dilate.setValue(40)
    _tab(w)._go()
    assert spy.calls, "inpaint params: nessuna richiesta accodata"
    cfg = _cfg_of(w)
    assert cfg.get("object") == "person", f"object non propagato: {cfg.get('object')}"
    assert abs(float(cfg.get("cfg_scale", -1)) - 8.5) < 1e-6, f"cfg_scale errato: {cfg.get('cfg_scale')}"
    assert abs(float(cfg.get("guide", -1)) - 0.7) < 1e-6, f"guide errato: {cfg.get('guide')}"
    assert int(cfg.get("mask_dilate", -1)) == 40, f"mask_dilate errato: {cfg.get('mask_dilate')}"


# ── UI: il campo Oggetto è abilitato SOLO in modalità inpaint ──────────────────────
def case_object_field_enabled_only_inpaint(w, spy):
    """Switch di modalità: obj abilitato solo per inpaint (logica _on_mode), no-crash."""
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _select_mode(w, "inpaint")
    assert t.obj.isEnabled(), "obj dovrebbe essere abilitato in modalità inpaint"
    _select_mode(w, "edit")
    assert not t.obj.isEnabled(), "obj NON dovrebbe essere abilitato fuori da inpaint"
    _select_mode(w, "anime2real")
    assert not t.obj.isEnabled(), "obj NON dovrebbe essere abilitato in anime2real"


# ── UI: set_video aggiorna stato/label ─────────────────────────────────────────────
def case_set_video_updates_state(w, spy):
    """set_video imposta il path interno e l'etichetta; nessuna richiesta backend (solo UI)."""
    t = _tab(w)
    p = _set_video(w)
    assert t._video == p, f"_video non impostato: {t._video}"
    assert os.path.basename(p) in t.video_lbl.text(), f"label video non aggiornata: {t.video_lbl.text()}"
    assert not spy.calls, "set_video non deve accodare nulla"


# ── VALIDAZIONE: nessun video → l'azione deve BLOCCARE (no enqueue) ────────────────
def case_no_video_blocks(w, spy):
    """Senza video l'azione deve fermarsi (log errore) e NON emettere request_edit."""
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    t._video = ""                                        # garantisce assenza video
    _select_mode(w, "edit")
    t.prompt.setPlainText("make it night")
    t._go()
    assert not spy.calls, f"senza video NON deve partire alcuna richiesta: {spy.calls}"


# ── VALIDAZIONE: inpaint senza oggetto → blocca ────────────────────────────────────
def case_inpaint_no_object_blocks(w, spy):
    """Inpaint senza oggetto (a parole) deve bloccare e non emettere request_edit."""
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _set_video(w)
    _select_mode(w, "inpaint")
    t.obj.setText("")
    t.prompt.setPlainText("a red ball")
    t._go()
    assert not spy.calls, f"inpaint senza oggetto NON deve accodare: {spy.calls}"


# ── VALIDAZIONE: prompt vuoto → blocca ─────────────────────────────────────────────
def case_no_prompt_blocks(w, spy):
    """Prompt vuoto deve bloccare l'azione (in tutte le modalità) e non emettere request_edit."""
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _set_video(w)
    _select_mode(w, "edit")
    t.prompt.setPlainText("   ")                         # solo spazi → strip() vuoto
    t._go()
    assert not spy.calls, f"prompt vuoto NON deve accodare: {spy.calls}"


# ── default: strength di default = 1.0, parametri inpaint di default coerenti ───────
def case_defaults_edit(w, spy):
    """Default modalità edit: strength=1.0, control_video impostato (via worker.cfg.strength)."""
    w.tabs.setCurrentWidget(_tab(w))
    _set_video(w)
    _select_mode(w, "edit")
    _tab(w).prompt.setPlainText("add rain")
    _tab(w)._go()
    assert spy.calls, "default edit: nessuna richiesta accodata"
    cfg = _cfg_of(w)
    assert abs(float(cfg.get("strength", -1)) - 1.0) < 1e-6, f"strength default atteso 1.0: {cfg.get('strength')}"


# ── #102 REPLACE CHAR (LIVE): video reale + ref personaggio + SAM2 → composite ─────
def case_replace_char_sam2(w, spy):
    """#102: replace_char con maschera SAM2. Video sorgente REALE + ref_image personaggio B +
    object='person' → _VideoEditAIWorker (SAM2 tracking → canny → iclora union-control-ref →
    _composite_char_mask). In run_real il worker viene ESEGUITO (generazione reale)."""
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _set_video(w)                                        # _vedit_test_input.mp4 ora è un video REALE
    _select_mode(w, "replace_char")
    t.ref_img.setText(str(H.OUT_DIR / "_portraitB.png"))  # personaggio sostitutivo (ritratto reale)
    t.obj.setText("person")                              # SAM2 limita la sostituzione al soggetto
    t.ref_strength.setValue(0.8)
    t.prompt.setPlainText("")                            # replace_char: il personaggio viene dall'immagine
    t._go()
    assert spy.calls, "replace_char: nessuna richiesta accodata"
    cfg = _cfg_of(w)
    assert cfg.get("mode") == "replace_char", f"mode errato: {cfg.get('mode')}"
    assert cfg.get("ref_image"), "ref_image mancante in cfg"
    assert cfg.get("object") == "person", f"object non propagato: {cfg.get('object')}"
    assert abs(float(cfg.get("ref_strength", -1)) - 0.8) < 1e-6, f"ref_strength errato: {cfg.get('ref_strength')}"


def case_replace_char_fullframe(w, spy):
    """#102: replace_char SENZA object → sostituzione su tutto il frame (nessun composite SAM2)."""
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _set_video(w)
    _select_mode(w, "replace_char")
    t.ref_img.setText(str(H.OUT_DIR / "_portraitB.png"))
    t.obj.setText("")                                    # niente SAM2 → tutto il frame
    t.ref_strength.setValue(0.9)
    t.prompt.setPlainText("")
    t._go()
    assert spy.calls, "replace_char full-frame: nessuna richiesta accodata"
    cfg = _cfg_of(w)
    assert cfg.get("mode") == "replace_char"
    assert not (cfg.get("object") or ""), f"object doveva essere vuoto: {cfg.get('object')}"


CASES = (
    [("vedit_replace_char_sam2",
      "#102 LIVE: replace_char con SAM2 (video reale + ref + object) → composite personaggio.",
      case_replace_char_sam2),
     ("vedit_replace_char_fullframe",
      "#102 LIVE: replace_char senza object → sostituzione full-frame.",
      case_replace_char_fullframe)]
    + [(f"vedit_mode_{k}", f"Modalità {k}: _go accoda l'edit con edit_lora '{tok}', prompt e video.",
      _make_mode_case(k, tok)) for k, tok in _MODE_LORA.items()]
    + [(f"vedit_strength_{v.replace('.', '')}",
        f"Forza editing={v}: il valore del combo è propagato in cfg['strength'].",
        _make_strength_case(v)) for v in ("0.6", "0.8", "1.0")]
    + [
        ("vedit_inpaint_params",
         "Inpaint: CFG/guide/dilatazione maschera dalla GUI finiscono in cfg.",
         case_inpaint_params_propagated),
        ("vedit_object_enabled_only_inpaint",
         "UI: campo Oggetto abilitato solo in modalità inpaint (logica _on_mode).",
         case_object_field_enabled_only_inpaint),
        ("vedit_set_video_state",
         "UI: set_video aggiorna _video e l'etichetta, senza accodare nulla.",
         case_set_video_updates_state),
        ("vedit_no_video_blocks",
         "Validazione: senza video l'azione si ferma e NON emette request_edit.",
         case_no_video_blocks),
        ("vedit_inpaint_no_object_blocks",
         "Validazione: inpaint senza oggetto si ferma e NON emette request_edit.",
         case_inpaint_no_object_blocks),
        ("vedit_no_prompt_blocks",
         "Validazione: prompt vuoto si ferma e NON emette request_edit.",
         case_no_prompt_blocks),
        ("vedit_defaults_edit",
         "Default edit: strength=1.0 propagata in cfg.",
         case_defaults_edit),
    ]
)
