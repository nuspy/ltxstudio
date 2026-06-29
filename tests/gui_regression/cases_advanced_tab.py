"""Casi di regressione GUI-driven per il tab "Avanzate" (ui/tabs/advanced_tab.py).

Copre:
  • SamplerSeedGroup (embeddato in video_tab come `w.video_tab.sampler_seed`):
      - ogni SAMPLER concreto      → get_values()["sampler"] coerente
      - ogni SCHEDULER concreto    → get_values()["scheduler"] coerente
      - ogni VIDEO_ENCODER         → text_encoder presente/assente (auto = niente forzatura)
      - preset qualità Fast/Pro/HQ → compilano steps+cfg attesi
      - toggle prompt_enhance / fixed_seed
      - shift base/max → ltxv_base_shift / ltxv_max_shift
      - default sani (32 step, CFG 1.0, sampler/scheduler di riferimento)
      - REGRESSIONE INTEGRAZIONE: steps/cfg/seed scelti ARRIVANO al backend nativo
        (on_generate → _generate_video_native: steps/guidance/seed). Scopre "funziona solo il default".
  • AdvancedTab (hub `w.advanced_tab`):
      - segnali open_workflow / open_analyzer / open_navigator → handler MainWindow senza crash
      - analizzatore legge davvero i valori sampler scelti
      - long_video_config() riflette i toggle/valori
      - auto-extend: durata > soglia attiva w._auto_extend in on_generate
      - switch dei sotto-tab (Strumenti / Modelli)

Backend nativo (data/backend_config.json video=native) → on_generate instrada a
_generate_video_native, che mette in coda un worker con .params catturato dallo Spy.
"""
from __future__ import annotations

import harness as H

# Liste/opzioni REALI dal modulo del tab (così i casi seguono il codice, non copie).
from ui.tabs.advanced_tab import SAMPLERS, SCHEDULERS, VIDEO_ENCODERS, SamplerSeedGroup


# ───────────────────────── helper ─────────────────────────
def _ss(w):
    """Il gruppo Sampler/Seed definito in advanced_tab e montato nel tab Video."""
    return w.video_tab.sampler_seed


def _arm_video(w, prompt="a dog running on the beach"):
    """Precondizione minima per far partire on_generate (t2v): prompt + tab attivo."""
    w.video_tab.prompt.setPlainText(prompt)
    w.tabs.setCurrentWidget(w.video_tab)


def _last_params(spy):
    assert spy.calls, "nessuna richiesta inviata al backend"
    return spy.calls[-1]["params"]


# ───────────────────────── SAMPLER: un caso per OGNI sampler concreto ─────────────────────────
def _make_sampler_case(name):
    def case(w, spy):
        ss = _ss(w)
        i = ss.sampler.findText(name)
        assert i >= 0, f"sampler '{name}' assente nel combo: {[ss.sampler.itemText(k) for k in range(ss.sampler.count())]}"
        ss.sampler.setCurrentIndex(i)
        v = ss.get_values()
        assert v.get("sampler") == name, f"get_values sampler={v.get('sampler')!r} != {name!r}"
    return case


# ───────────────────────── SCHEDULER: un caso per OGNI scheduler concreto ─────────────────────────
def _make_scheduler_case(name):
    def case(w, spy):
        ss = _ss(w)
        i = ss.scheduler.findText(name)
        assert i >= 0, f"scheduler '{name}' assente nel combo: {[ss.scheduler.itemText(k) for k in range(ss.scheduler.count())]}"
        ss.scheduler.setCurrentIndex(i)
        v = ss.get_values()
        assert v.get("scheduler") == name, f"get_values scheduler={v.get('scheduler')!r} != {name!r}"
    return case


# ───────────────────────── ENCODER: un caso per OGNI encoder (auto + concreti) ─────────────────────────
def _make_encoder_case(idx, fn):
    def case(w, spy):
        ss = _ss(w)
        assert idx < ss.encoder.count(), f"indice encoder {idx} fuori range ({ss.encoder.count()})"
        ss.encoder.setCurrentIndex(idx)
        v = ss.get_values()
        if fn is None:
            # 'Auto' → NON deve forzare l'encoder (lo sceglie il preset SFW/NSFW)
            assert "text_encoder" not in v, f"encoder 'auto' non deve forzare text_encoder: {v.get('text_encoder')}"
        else:
            assert v.get("text_encoder") == fn, f"text_encoder={v.get('text_encoder')!r} != {fn!r}"
    return case


# ───────────────────────── PRESET QUALITÀ: Fast / Pro / HQ ─────────────────────────
def _make_quality_case(idx, steps, cfg):
    def case(w, spy):
        ss = _ss(w)
        ss.quality_preset.setCurrentIndex(idx)
        assert ss.steps.value() == steps, f"preset non ha impostato steps={steps} (è {ss.steps.value()})"
        assert abs(ss.cfg.value() - cfg) < 0.01, f"preset non ha impostato cfg={cfg} (è {ss.cfg.value()})"
    return case


# ───────────────────────── DEFAULT sani del gruppo ─────────────────────────
def case_sampler_defaults(w, spy):
    """All'avvio: 32 step, CFG 1.0, sampler/scheduler di riferimento LTX-2.3."""
    v = _ss(w).get_values()
    assert v["steps"] == 32, f"steps default atteso 32, trovato {v['steps']}"
    assert abs(v["cfg"] - 1.0) < 0.01, f"cfg default atteso 1.0, trovato {v['cfg']}"
    assert v["sampler"] == "euler_ancestral_cfg_pp", f"sampler default inatteso: {v['sampler']}"
    assert v["scheduler"] == "LTXVScheduler (shift)", f"scheduler default inatteso: {v['scheduler']}"
    assert "ltxv_base_shift" in v and "ltxv_max_shift" in v, f"shift mancanti nei get_values: {v}"


# ───────────────────────── TOGGLE prompt_enhance ─────────────────────────
def case_prompt_enhance_toggle(w, spy):
    """Attivando prompt_enhance, get_values lo riporta True; OFF di default → False."""
    ss = _ss(w)
    assert ss.get_values().get("prompt_enhance") is False, "prompt_enhance dovrebbe essere OFF di default"
    ss.prompt_enhance.setChecked(True)
    assert ss.get_values().get("prompt_enhance") is True, "prompt_enhance ON non riflesso in get_values"


# ───────────────────────── TOGGLE fixed_seed ─────────────────────────
def case_fixed_seed_off_no_seed(w, spy):
    """Seed NON fisso (default): get_values NON deve includere 'seed' (sarà randomizzato)."""
    ss = _ss(w)
    ss.fixed_seed.setChecked(False)
    v = ss.get_values()
    assert "seed" not in v, f"con seed non-fisso get_values non deve forzare seed: {v.get('seed')}"
    assert ss.wants_fixed_seed() is False


def case_fixed_seed_on_emits_seed(w, spy):
    """Seed fisso: get_values include il seed impostato."""
    ss = _ss(w)
    ss.fixed_seed.setChecked(True)
    ss.seed.setValue(424242)
    v = ss.get_values()
    assert v.get("seed") == 424242, f"seed fisso non riportato: {v.get('seed')}"
    assert ss.wants_fixed_seed() is True


# ───────────────────────── SHIFT base/max ─────────────────────────
def case_shift_values(w, spy):
    """base_shift/max_shift di LTXVScheduler finiscono in ltxv_base_shift/ltxv_max_shift."""
    ss = _ss(w)
    ss.base_shift.setValue(1.50)
    ss.max_shift.setValue(3.00)
    v = ss.get_values()
    assert abs(v["ltxv_base_shift"] - 1.50) < 0.01, f"base_shift non propagato: {v['ltxv_base_shift']}"
    assert abs(v["ltxv_max_shift"] - 3.00) < 0.01, f"max_shift non propagato: {v['ltxv_max_shift']}"


# ───── INTEGRAZIONE: i valori del gruppo ARRIVANO al backend nativo (on_generate) ─────
def case_steps_reach_backend(w, spy):
    """REGRESSIONE: lo step scelto nel gruppo avanzato deve arrivare al worker nativo."""
    ss = _ss(w)
    ss.steps.setValue(17)
    _arm_video(w)
    w.on_generate()
    p = _last_params(spy)
    assert p.get("steps") == 17, f"steps scelto (17) non arrivato al backend: {p.get('steps')}"


def case_cfg_reach_backend(w, spy):
    """REGRESSIONE: il CFG scelto diventa 'guidance' nel worker nativo LTX-2."""
    ss = _ss(w)
    ss.cfg.setValue(4.5)
    _arm_video(w)
    w.on_generate()
    p = _last_params(spy)
    assert abs(float(p.get("guidance", -1)) - 4.5) < 0.01, f"cfg→guidance non arrivato: {p.get('guidance')}"


def case_fixed_seed_reach_backend(w, spy):
    """REGRESSIONE: con seed fisso il valore arriva identico al worker nativo."""
    ss = _ss(w)
    ss.fixed_seed.setChecked(True)
    ss.seed.setValue(12345)
    _arm_video(w)
    w.on_generate()
    p = _last_params(spy)
    assert p.get("seed") == 12345, f"seed fisso non arrivato al backend: {p.get('seed')}"


# ───────────────────────── HUB: segnali → handler MainWindow ─────────────────────────
def case_open_workflow_signal(w, spy):
    """Il pulsante Workflow emette open_workflow → MainWindow._open_workflow (dialog) senza crash."""
    w.tabs.setCurrentWidget(w.advanced_tab)
    w.advanced_tab.open_workflow.emit()


def case_open_analyzer_signal(w, spy):
    """Il pulsante Analizzatore emette open_analyzer → MainWindow._open_analyzer senza crash."""
    w.tabs.setCurrentWidget(w.advanced_tab)
    w.advanced_tab.open_analyzer.emit()


def case_open_navigator_signal(w, spy):
    """Il pulsante Navigator emette open_navigator → MainWindow._open_navigator senza crash."""
    w.tabs.setCurrentWidget(w.advanced_tab)
    w.advanced_tab.open_navigator.emit()


def case_analyzer_reads_sampler(w, spy):
    """L'analizzatore deve leggere i valori sampler scelti (no crash con sampler non-default)."""
    ss = _ss(w)
    i = ss.sampler.findText("euler")
    if i >= 0:
        ss.sampler.setCurrentIndex(i)
    j = ss.scheduler.findText("karras")
    if j >= 0:
        ss.scheduler.setCurrentIndex(j)
    ss.steps.setValue(24)
    w.advanced_tab.open_analyzer.emit()


# ───────────────────────── VIDEO LUNGHI: long_video_config ─────────────────────────
def case_long_video_config(w, spy):
    """long_video_config() riflette esattamente i toggle/valori impostati nel tab."""
    a = w.advanced_tab
    a.lv_auto.setChecked(False)
    a.lv_resume.setChecked(False)
    a.lv_threshold.setValue(15.0)
    a.lv_seg.setValue(8.0)
    a.lv_overlap.setValue(12)
    a.lv_context.setValue(24)
    a.lv_ctx_strength.setValue(0.30)
    a.lv_ref_anchor.setValue(0.40)
    a.lv_anchor_every.setValue(5)
    cfg = a.long_video_config()
    assert cfg["auto"] is False and cfg["resume"] is False, f"toggle non riflessi: {cfg}"
    assert cfg["threshold"] == 15.0 and cfg["seg_seconds"] == 8.0, f"durate non riflesse: {cfg}"
    assert cfg["overlap"] == 12 and cfg["context"] == 24, f"overlap/context non riflessi: {cfg}"
    assert abs(cfg["context_strength"] - 0.30) < 0.01, f"context_strength non riflesso: {cfg}"
    assert abs(cfg["ref_anchor_strength"] - 0.40) < 0.01, f"ref_anchor non riflesso: {cfg}"
    assert cfg["anchor_every"] == 5, f"anchor_every non riflesso: {cfg}"


def case_long_video_autoextend_triggers(w, spy):
    """Durata > soglia + auto ON → on_generate attiva w._auto_extend (estensione automatica)."""
    a = w.advanced_tab
    a.lv_auto.setChecked(True)
    a.lv_threshold.setValue(10.0)
    try:
        w.video_tab.length.setValue(60.0)
    except Exception:
        pass
    _arm_video(w, "a very long continuous scene")
    w.on_generate()
    assert getattr(w, "_auto_extend", None) is not None, \
        "durata 60s > soglia 10s: _auto_extend doveva essere attivato"
    assert float(w._auto_extend.get("target", 0)) > 10.0, f"target auto-extend errato: {w._auto_extend}"


def case_long_video_autoextend_off(w, spy):
    """Auto OFF → nessun auto-extend anche con durata alta (rispetta la scelta utente)."""
    a = w.advanced_tab
    a.lv_auto.setChecked(False)
    try:
        w.video_tab.length.setValue(60.0)
    except Exception:
        pass
    _arm_video(w)
    w.on_generate()
    assert getattr(w, "_auto_extend", None) is None, "auto OFF non deve attivare _auto_extend"


# ───────────────────────── SOTTO-TAB: switch Strumenti/Modelli ─────────────────────────
def case_subtabs_switch(w, spy):
    """Switch di tutti i sotto-tab di Avanzate (Strumenti, Modelli) senza crash."""
    w.tabs.setCurrentWidget(w.advanced_tab)
    sub = w.advanced_tab.sub
    assert sub.count() >= 2, f"attesi almeno 2 sotto-tab (Strumenti/Modelli), trovati {sub.count()}"
    for i in range(sub.count()):
        sub.setCurrentIndex(i)


# ───────────────────────── assemblaggio CASES ─────────────────────────
CASES: list = []

# Un caso per OGNI sampler concreto (scopre "funziona solo un sampler").
for _name in SAMPLERS:
    CASES.append((
        f"advanced_sampler_{_name}",
        f"Sampler '{_name}': selezionato nel gruppo avanzato, get_values lo riporta corretto.",
        _make_sampler_case(_name),
    ))

# Un caso per OGNI scheduler concreto.
for _name in SCHEDULERS:
    _safe = _name.replace(" ", "_").replace("(", "").replace(")", "")
    CASES.append((
        f"advanced_scheduler_{_safe}",
        f"Scheduler '{_name}': selezionato, get_values lo riporta corretto.",
        _make_scheduler_case(_name),
    ))

# Un caso per OGNI encoder testo del video (auto + concreti).
for _i, (_label_key, _fn) in enumerate(VIDEO_ENCODERS):
    _tag = (_fn or "auto").replace(".safetensors", "")
    CASES.append((
        f"advanced_encoder_{_tag}",
        f"Encoder '{_label_key}': get_values forza text_encoder={_fn!r} (auto = nessuna forzatura).",
        _make_encoder_case(_i, _fn),
    ))

# Preset qualità Fast/Pro/HQ (saltando 'Personalizzato' = no-op).
for _idx, (_key, _steps, _cfg) in enumerate(SamplerSeedGroup._QUALITY_PRESETS):
    if _steps is None:
        continue
    CASES.append((
        f"advanced_quality_preset_{_key.split('.')[-1]}",
        f"Preset qualità '{_key}': imposta steps={_steps}, cfg={_cfg}.",
        _make_quality_case(_idx, _steps, _cfg),
    ))

CASES += [
    ("advanced_sampler_defaults",
     "Default gruppo sampler: 32 step, CFG 1.0, sampler/scheduler di riferimento + shift presenti.",
     case_sampler_defaults),
    ("advanced_prompt_enhance_toggle",
     "Toggle prompt_enhance: OFF di default, ON riflesso in get_values.",
     case_prompt_enhance_toggle),
    ("advanced_fixed_seed_off",
     "Seed non-fisso: get_values non forza 'seed' (verrà randomizzato).",
     case_fixed_seed_off_no_seed),
    ("advanced_fixed_seed_on",
     "Seed fisso: get_values riporta il seed impostato.",
     case_fixed_seed_on_emits_seed),
    ("advanced_shift_values",
     "Shift base/max → ltxv_base_shift / ltxv_max_shift in get_values.",
     case_shift_values),
    ("advanced_steps_reach_backend",
     "Integrazione: lo step scelto arriva al worker nativo (on_generate).",
     case_steps_reach_backend),
    ("advanced_cfg_reach_backend",
     "Integrazione: il CFG scelto diventa 'guidance' nel worker nativo.",
     case_cfg_reach_backend),
    ("advanced_fixed_seed_reach_backend",
     "Integrazione: seed fisso arriva identico al worker nativo.",
     case_fixed_seed_reach_backend),
    ("advanced_open_workflow",
     "Hub: open_workflow → MainWindow._open_workflow (WorkflowDialog) senza crash.",
     case_open_workflow_signal),
    ("advanced_open_analyzer",
     "Hub: open_analyzer → MainWindow._open_analyzer (SettingsAnalyzerDialog) senza crash.",
     case_open_analyzer_signal),
    ("advanced_open_navigator",
     "Hub: open_navigator → MainWindow._open_navigator (NavigatorDialog) senza crash.",
     case_open_navigator_signal),
    ("advanced_analyzer_reads_sampler",
     "Analizzatore: legge sampler/scheduler/steps scelti senza crash.",
     case_analyzer_reads_sampler),
    ("advanced_long_video_config",
     "Video lunghi: long_video_config() riflette toggle e valori impostati.",
     case_long_video_config),
    ("advanced_long_video_autoextend_on",
     "Video lunghi: durata > soglia + auto ON attiva w._auto_extend in on_generate.",
     case_long_video_autoextend_triggers),
    ("advanced_long_video_autoextend_off",
     "Video lunghi: auto OFF non attiva l'estensione automatica.",
     case_long_video_autoextend_off),
    ("advanced_subtabs_switch",
     "Sotto-tab Avanzate (Strumenti/Modelli): switch senza crash.",
     case_subtabs_switch),
]
