"""Casi di regressione GUI-driven per il tab «Allena LoRA» (ui/tabs/train_tab.py → w.train_tab).

Particolarità: il training NON passa dal backend nativo/Comfy (lo Spy non lo intercetta): la GUI
costruisce una config ai-toolkit e lancia un SUBPROCESS isolato (core.training_service.TrainingRun.start
→ subprocess.Popen) oppure un QProcess (install / caption). Per testare l'ISTRUZIONE senza GPU/IO:
  • si patcha `ts.TrainingRun.start` per CATTURARE i params/config (build_config) senza spawnare nulla;
  • si patcha `ts.is_installed` (preflight richiede l'ambiente installato);
  • per install/caption si sostituisce `QProcess` nel namespace del tab con un fake che registra start().

Obiettivo: scoprire regressioni reali — es. "il training funziona solo per UN modello" (arch/name_or_path
sbagliati nel config), toggle che non finiscono nella config, preflight che non blocca, comandi
install/caption malformati. Ogni caso è no-crash + assert mirati sulla config/comando catturati.
"""
from __future__ import annotations

import os

import harness as H


# ── helper di setup ──────────────────────────────────────────────────────────
def _prep_dataset(w) -> str:
    """Crea una cartella dataset valida (1 immagine + 1 caption) e la imposta nel tab."""
    ds = os.path.join(str(H.OUT_DIR), "_train_ds")
    os.makedirs(ds, exist_ok=True)
    H.make_test_image(os.path.join(ds, "img1.png"), 768, 512)
    try:
        with open(os.path.join(ds, "img1.txt"), "w", encoding="utf-8") as fh:
            fh.write("a test subject")
    except Exception:                                          # noqa: BLE001
        pass
    w.train_tab.ds_path.setText(ds)
    return ds


def _select_model(w, key: str) -> str:
    """Seleziona nel combo il modello base con itemData==key; asserisce che esista."""
    combo = w.train_tab.model
    for i in range(combo.count()):
        if combo.itemData(i) == key:
            combo.setCurrentIndex(i)
            return combo.itemText(i)
    raise AssertionError(f"modello base '{key}' assente dal combo (BASE_MODELS cambiato?)")


def _proc(config: dict) -> dict:
    """Ritorna il blocco process[0] della config ai-toolkit."""
    return config["config"]["process"][0]


def _start_capture(w, setup=None) -> dict:
    """Imposta dataset valido + (opz.) setup, patcha is_installed/TrainingRun.start e invoca _start().
    Ritorna {'params', 'config'} catturati (vuoto se preflight/blocco impedisce l'avvio)."""
    from core import training_service as ts
    captured: dict = {}
    orig_inst = ts.is_installed
    orig_start = ts.TrainingRun.start

    def fake_start(self):
        captured["params"] = self.params
        captured["config"] = ts.build_config(self.params)
        return "config.json"

    ts.is_installed = lambda: True
    ts.TrainingRun.start = fake_start
    try:
        _prep_dataset(w)
        if setup:
            setup(w)
        w.train_tab._start()
    finally:
        ts.is_installed = orig_inst
        ts.TrainingRun.start = orig_start
    return captured


def _install_fake_qprocess():
    """Sostituisce QProcess nel modulo del tab con un fake che registra start() senza spawnare.
    Ritorna (modulo_tab, QProcess_originale) per il ripristino."""
    import ui.tabs.train_tab as tt

    class _Sig:
        def connect(self, *a, **k):
            pass

    class _FakeProc:
        MergedChannels = 0

        def __init__(self, *a, **k):
            self.readyReadStandardOutput = _Sig()
            self.finished = _Sig()
            self.started_with = None

        def setProcessChannelMode(self, *a, **k):
            pass

        def start(self, prog, args=None):
            self.started_with = (prog, list(args) if args is not None else args)

    orig = tt.QProcess
    tt.QProcess = _FakeProc
    return tt, orig


# ── TRAINING: un caso per OGNI modello base (scopre "funziona solo un modello") ──
def _make_model_case(key: str):
    def case(w, spy):
        from core import training_service as ts
        label = _select_model(w, key)
        captured = _start_capture(w, setup=lambda ww: _select_model(ww, key))
        assert captured.get("params"), \
            f"training NON avviato per {label} ({key}): preflight/dialog ha bloccato l'avvio?"
        p = captured["params"]
        assert p.base_model == key, f"base_model errato: atteso {key}, ottenuto {p.base_model}"
        m = _proc(captured["config"])["model"]
        spec = ts.BASE_MODELS[key]
        assert m.get("name_or_path") == spec.default_path, \
            f"name_or_path errato per {key}: {m.get('name_or_path')} (atteso {spec.default_path})"
        if spec.is_flux:                                       # FLUX.1: is_flux, niente arch
            assert m.get("is_flux") is True, f"FLUX.1 senza is_flux: {m}"
            assert "arch" not in m, f"FLUX.1 non deve avere arch: {m}"
        else:                                                 # FLUX.2: arch obbligatoria
            assert m.get("arch"), f"arch mancante per {key}: {m}"
            assert m["arch"] == spec.arch, f"arch errata per {key}: {m['arch']} (atteso {spec.arch})"
    return case


# ── TRAINING: toggle/parametri devono finire nella config ────────────────────
def case_quantize_off(w, spy):
    """Disattivare 'Quantizza' deve impostare model.quantize=False nella config."""
    captured = _start_capture(w, setup=lambda ww: ww.train_tab.quantize.setChecked(False))
    assert captured.get("config"), "training non avviato"
    assert _proc(captured["config"])["model"]["quantize"] is False, "quantize non propagato (off)"


def case_low_vram_on(w, spy):
    """Attivare 'Low VRAM' deve impostare model.low_vram=True nella config."""
    captured = _start_capture(w, setup=lambda ww: ww.train_tab.low_vram.setChecked(True))
    assert captured.get("config"), "training non avviato"
    assert _proc(captured["config"])["model"].get("low_vram") is True, "low_vram non propagato"


def case_train_text_encoder_on(w, spy):
    """Attivare 'Allena anche il text encoder' → train.train_text_encoder=True."""
    captured = _start_capture(w, setup=lambda ww: ww.train_tab.train_te.setChecked(True))
    assert captured.get("config"), "training non avviato"
    assert _proc(captured["config"])["train"]["train_text_encoder"] is True, "train_te non propagato"


def case_single_resolution(w, spy):
    """Lasciare attiva una sola risoluzione (512) → datasets.resolution == [512]."""
    def setup(ww):
        ww.train_tab.res[768].setChecked(False)
        ww.train_tab.res[1024].setChecked(False)
        ww.train_tab.res[512].setChecked(True)
    captured = _start_capture(w, setup=setup)
    assert captured.get("config"), "training non avviato"
    assert _proc(captured["config"])["datasets"][0]["resolution"] == [512], "risoluzione non propagata"


def case_numeric_params(w, spy):
    """rank/alpha/steps/lr/batch/seed impostati in UI devono finire identici nella config."""
    def setup(ww):
        t = ww.train_tab
        t.rank.setValue(32); t.alpha.setValue(64); t.steps.setValue(1500)
        t.batch.setValue(2); t.seed.setValue(123); t.lr.setCurrentText("2e-4")
    captured = _start_capture(w, setup=setup)
    cfg = captured.get("config")
    assert cfg, "training non avviato"
    pr = _proc(cfg)
    assert pr["network"]["linear"] == 32, f"rank errato: {pr['network']}"
    assert pr["network"]["linear_alpha"] == 64, f"alpha errato: {pr['network']}"
    assert pr["train"]["steps"] == 1500, f"steps errato: {pr['train']['steps']}"
    assert pr["train"]["batch_size"] == 2, f"batch errato: {pr['train']['batch_size']}"
    assert abs(pr["train"]["lr"] - 2e-4) < 1e-9, f"lr errato: {pr['train']['lr']}"
    assert pr["sample"]["seed"] == 123, f"seed errato: {pr['sample']['seed']}"


def case_trigger_word_propagation(w, spy):
    """Trigger word: deve comparire come trigger_word, nel nome run e prepended nei sample prompt."""
    captured = _start_capture(w, setup=lambda ww: ww.train_tab.trigger.setText("ohwx"))
    p = captured.get("params")
    cfg = captured.get("config")
    assert p and cfg, "training non avviato"
    assert p.trigger_word == "ohwx", f"trigger_word non raccolto: {p.trigger_word}"
    assert p.name.startswith("ohwx_"), f"il nome run non include il trigger: {p.name}"
    pr = _proc(cfg)
    assert pr["trigger_word"] == "ohwx", f"trigger_word non in config: {pr['trigger_word']}"
    assert all(pp.startswith("ohwx,") for pp in pr["sample"]["prompts"]), \
        f"trigger non anteposto ai sample prompt: {pr['sample']['prompts']}"


def case_custom_sample_prompts(w, spy):
    """Prompt di anteprima personalizzati (uno per riga) devono finire in sample.prompts."""
    def setup(ww):
        ww.train_tab.prompts.setPlainText("a custom prompt one\na custom prompt two")
    captured = _start_capture(w, setup=setup)
    cfg = captured.get("config")
    assert cfg, "training non avviato"
    prompts = _proc(cfg)["sample"]["prompts"]
    assert prompts == ["a custom prompt one", "a custom prompt two"], f"sample prompt errati: {prompts}"


# ── PREFLIGHT: deve BLOCCARE l'avvio nelle condizioni invalide ───────────────
def case_preflight_blocks_missing_dataset(w, spy):
    """Senza dataset valido il preflight deve impedire l'avvio (nessun TrainingRun, _run resta None)."""
    from core import training_service as ts
    captured: dict = {}
    orig_inst, orig_start = ts.is_installed, ts.TrainingRun.start
    ts.is_installed = lambda: True
    ts.TrainingRun.start = lambda self: captured.setdefault("started", True) or "x"
    try:
        w.train_tab.ds_path.setText("Z:/__no_such_dataset_dir__")
        w.train_tab._start()
    finally:
        ts.is_installed, ts.TrainingRun.start = orig_inst, orig_start
    assert "started" not in captured, "training avviato nonostante dataset mancante (preflight non blocca)"
    assert w.train_tab._run is None, "_run non è None dopo preflight fallito"


def case_preflight_blocks_not_installed(w, spy):
    """Con modulo NON installato il preflight deve bloccare anche se il dataset è valido."""
    from core import training_service as ts
    captured: dict = {}
    orig_inst, orig_start = ts.is_installed, ts.TrainingRun.start
    ts.is_installed = lambda: False
    ts.TrainingRun.start = lambda self: captured.setdefault("started", True) or "x"
    try:
        _prep_dataset(w)
        w.train_tab._start()
    finally:
        ts.is_installed, ts.TrainingRun.start = orig_inst, orig_start
    assert "started" not in captured, "training avviato senza modulo installato (preflight non blocca)"


# ── INSTALL / CAPTION: comando del subprocess corretto (QProcess fake) ───────
def case_install_module_command(w, spy):
    """'Installa modulo training' deve lanciare il python corrente su scripts/install_training.py."""
    import sys
    tt, orig = _install_fake_qprocess()
    try:
        w.train_tab._install_module()
    finally:
        tt.QProcess = orig
    proc = w.train_tab._install_proc
    assert proc is not None and proc.started_with, "install: QProcess non avviato"
    prog, args = proc.started_with
    assert prog == sys.executable, f"install non usa il python corrente: {prog}"
    assert any("install_training" in str(a) for a in (args or [])), f"script install mancante: {args}"


def case_caption_command(w, spy):
    """Auto-caption (Florence-2): con dataset valido + trigger + overwrite, il comando QProcess deve
    includere --dir <ds>, --trigger ohwx, --overwrite ed usare il python del venv di training."""
    from core import training_service as ts
    orig_inst = ts.is_installed
    ts.is_installed = lambda: True
    tt, orig = _install_fake_qprocess()
    try:
        _prep_dataset(w)
        w.train_tab.trigger.setText("ohwx")
        w.train_tab.cap_overwrite.setChecked(True)
        w.train_tab._caption_dataset()
    finally:
        ts.is_installed = orig_inst
        tt.QProcess = orig
    proc = w.train_tab._caption_proc
    assert proc is not None and proc.started_with, "caption: QProcess non avviato"
    prog, args = proc.started_with
    args = args or []
    # Router multi-engine: joycaption→venv15 (default se pronto), florence2→venv training.
    # Il python deve appartenere a UNO dei motori validi (il vecchio assert pretendeva solo
    # il venv training → falso FAIL col design attuale).
    from core import caption_service as cs
    _valid = {str(cs.VENV15_PY), str(ts.VENV_PYTHON)}
    assert str(prog) in _valid, f"caption non usa un python dei motori validi: {prog}"
    assert "--dir" in args, f"--dir mancante: {args}"
    assert "--trigger" in args and "ohwx" in args, f"--trigger mancante: {args}"
    assert "--overwrite" in args, f"--overwrite mancante: {args}"
    assert any("caption_dataset" in str(a) for a in args), f"script caption mancante: {args}"


def case_caption_blocked_not_installed(w, spy):
    """Caption senza ALCUN motore disponibile: deve avvisare e NON avviare alcun processo.
    Col router multi-engine va spento ANCHE joycaption (spegnere solo florence2 lasciava
    partire joycaption — comportamento corretto del prodotto, falso FAIL del caso)."""
    from core import training_service as ts
    from core import caption_service as cs
    orig_inst = ts.is_installed
    orig_joy = cs.joycaption_ready
    ts.is_installed = lambda: False
    cs.joycaption_ready = lambda: False
    tt, orig = _install_fake_qprocess()
    try:
        _prep_dataset(w)
        w.train_tab._caption_dataset()
    finally:
        ts.is_installed = orig_inst
        cs.joycaption_ready = orig_joy
        tt.QProcess = orig
    assert w.train_tab._caption_proc is None, "caption avviato senza alcun motore disponibile"


def case_caption_blocked_bad_dataset(w, spy):
    """Caption con cartella dataset invalida: deve avvisare e NON avviare alcun processo."""
    from core import training_service as ts
    orig_inst = ts.is_installed
    ts.is_installed = lambda: True
    tt, orig = _install_fake_qprocess()
    try:
        w.train_tab.ds_path.setText("Z:/__no_such_dataset_dir__")
        w.train_tab._caption_dataset()
    finally:
        ts.is_installed = orig_inst
        tt.QProcess = orig
    assert w.train_tab._caption_proc is None, "caption avviato con dataset invalido"


# ── UI-only: nessun crash + comportamenti deterministici ─────────────────────
def case_model_switch_updates_path(w, spy):
    """Cambiare modello base deve aggiornare 'Repo HF / cartella' col default_path dello spec;
    selezionando FLUX.2 [dev] deve auto-attivare Low VRAM."""
    from core import training_service as ts
    combo = w.train_tab.model
    for i in range(combo.count()):
        combo.setCurrentIndex(i)
        key = combo.itemData(i)
        spec = ts.BASE_MODELS[key]
        assert w.train_tab.model_path.text() == spec.default_path, \
            f"model_path non aggiornato per {key}: {w.train_tab.model_path.text()}"
    _select_model(w, "flux2-dev")
    assert w.train_tab.low_vram.isChecked(), "FLUX.2 [dev] non auto-attiva Low VRAM"


def case_update_ds_info(w, spy):
    """_update_ds_info: cartella valida mostra conteggio (no crash); cartella invalida → messaggio."""
    _prep_dataset(w)
    w.train_tab._update_ds_info()
    txt_ok = w.train_tab.ds_info.text()
    assert txt_ok and "—" != txt_ok, f"info dataset non aggiornata: {txt_ok}"
    w.train_tab.ds_path.setText("Z:/__no_such_dataset_dir__")
    w.train_tab._update_ds_info()                              # solo no-crash


def case_browse_dataset_cancel(w, spy):
    """Sfoglia dataset annullato (QFileDialog patchato → '') non deve cambiare il path né crashare."""
    w.train_tab.ds_path.setText("keep-me")
    w.train_tab._browse_dataset()
    assert w.train_tab.ds_path.text() == "keep-me", "browse annullato ha sovrascritto il path"


def case_collect_params_resolution_fallback(w, spy):
    """Se nessuna risoluzione è selezionata, _collect_params deve ripiegare su [1024]."""
    _prep_dataset(w)
    for cb in w.train_tab.res.values():
        cb.setChecked(False)
    p = w.train_tab._collect_params()
    assert p.resolutions == [1024], f"fallback risoluzione errato: {p.resolutions}"


def case_collect_params_bad_lr_fallback(w, spy):
    """Learning rate non numerico in _collect_params deve ripiegare su 1e-4 (no crash)."""
    _prep_dataset(w)
    w.train_tab.lr.setCurrentText("not-a-number")
    p = w.train_tab._collect_params()
    assert abs(p.lr - 1e-4) < 1e-9, f"fallback lr errato: {p.lr}"


def case_stop_without_run(w, spy):
    """Stop senza training attivo: no-crash."""
    w.train_tab._stop()


def case_progress_loss_done_slots(w, spy):
    """Slot di progresso/loss/fine non devono crashare anche senza run attivo."""
    t = w.train_tab
    t._on_progress(50, 100)
    t._on_loss(0.1234)
    t._on_done(0)                                             # fine ok (find_final_lora → None branch)
    t._on_done(1)                                             # fine con errore
    t._append_log("hello")
    t._append_log("")


def case_tab_switch_no_crash(w, spy):
    """Portare la GUI sul tab train non deve crashare."""
    w.tabs.setCurrentWidget(w.train_tab)


# ── registrazione casi ───────────────────────────────────────────────────────
_MODEL_KEYS = ["flux1-dev", "flux2-klein-4b", "flux2-klein-9b", "flux2-dev"]

CASES = [
    *[(f"train_start_model_{k.replace('-', '_')}",
       f"Avvio training con modello base '{k}': la config ai-toolkit ha name_or_path/arch corretti.",
       _make_model_case(k)) for k in _MODEL_KEYS],
    ("train_quantize_off", "Toggle Quantizza OFF → model.quantize=False nella config.",
     case_quantize_off),
    ("train_low_vram_on", "Toggle Low VRAM ON → model.low_vram=True nella config.",
     case_low_vram_on),
    ("train_text_encoder_on", "Toggle Allena text encoder ON → train.train_text_encoder=True.",
     case_train_text_encoder_on),
    ("train_single_resolution", "Una sola risoluzione (512) → datasets.resolution=[512].",
     case_single_resolution),
    ("train_numeric_params", "rank/alpha/steps/lr/batch/seed propagati identici nella config.",
     case_numeric_params),
    ("train_trigger_word", "Trigger word: in trigger_word, nel nome run e prepended nei sample prompt.",
     case_trigger_word_propagation),
    ("train_custom_sample_prompts", "Prompt di anteprima personalizzati → sample.prompts.",
     case_custom_sample_prompts),
    ("train_preflight_no_dataset", "Preflight blocca l'avvio senza dataset valido (_run resta None).",
     case_preflight_blocks_missing_dataset),
    ("train_preflight_not_installed", "Preflight blocca l'avvio se il modulo non è installato.",
     case_preflight_blocks_not_installed),
    ("train_install_command", "Installa modulo: QProcess su python corrente + scripts/install_training.py.",
     case_install_module_command),
    ("train_caption_command", "Auto-caption: comando con --dir/--trigger/--overwrite sul python del venv.",
     case_caption_command),
    ("train_caption_blocked_not_installed", "Caption bloccata senza modulo installato (nessun processo).",
     case_caption_blocked_not_installed),
    ("train_caption_blocked_bad_dataset", "Caption bloccata con dataset invalido (nessun processo).",
     case_caption_blocked_bad_dataset),
    ("train_model_switch_updates_path", "Cambio modello aggiorna model_path; FLUX.2 dev auto-attiva Low VRAM.",
     case_model_switch_updates_path),
    ("train_update_ds_info", "Info dataset: conteggio su cartella valida, no-crash su invalida.",
     case_update_ds_info),
    ("train_browse_cancel", "Sfoglia dataset annullato non cambia il path.",
     case_browse_dataset_cancel),
    ("train_resolution_fallback", "Nessuna risoluzione selezionata → fallback [1024].",
     case_collect_params_resolution_fallback),
    ("train_bad_lr_fallback", "Learning rate non numerico → fallback 1e-4.",
     case_collect_params_bad_lr_fallback),
    ("train_stop_without_run", "Stop senza run attivo: no-crash.",
     case_stop_without_run),
    ("train_progress_loss_done_slots", "Slot progresso/loss/done: no-crash.",
     case_progress_loss_done_slots),
    ("train_tab_switch", "Switch al tab train: no-crash.",
     case_tab_switch_no_crash),
]
