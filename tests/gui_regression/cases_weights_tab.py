"""Casi di regressione GUI-driven per il tab «Pesi modelli» (WeightsTab).

Particolarità di questo tab: NON instrada le richieste verso _enqueue/native_driver (lo Spy
dell'harness), ma lancia lo script scripts/download_weights.py tramite QProcess. Per intercettare
l'AZIONE "scarica selezionati" SENZA scaricare nulla, qui si patcha QProcess.start (e
setProcessEnvironment) catturando il comando costruito (programma + argomenti + env), poi si
verifica che gli argomenti (--ids, --base-dir) e l'HF token (per i gated) siano corretti.

Copre:
  • apertura tab / refresh senza crash
  • "Scarica selezionati" senza selezione → nessun processo, status "nessun peso"
  • costruzione comando download (eseguibile python + script + --ids + --base-dir)
  • UN caso per OGNI modello del catalogo (seleziona SOLO quello → --ids deve contenere solo il suo
    id): così si scopre se "funziona solo un modello"
  • HF token iniettato nell'ambiente per i modelli gated
  • "Seleziona mancanti" → download coerente con i check
  • refresh azzera le selezioni
  • scan pesi esistenti: cartella vuota (messaggio) e cartella con un peso che combacia (_found)
  • browse cartella base/scan (dialog patchati → no crash)

Tutti i casi sono no-rete/no-GPU: la patch su QProcess impedisce qualsiasi spawn reale.
"""
from __future__ import annotations

import contextlib
import os
import sys

import harness as H
from core import weights_catalog as wc

from PySide6.QtCore import QProcess


# ── helper: cattura il comando QProcess senza eseguirlo ──────────────────────────
@contextlib.contextmanager
def _capture_qprocess():
    """Patcha QProcess.start/setProcessEnvironment per catturare il comando di download
    SENZA avviarlo. Ritorna {'starts': [(program, [args...]), ...], 'envs': [QProcessEnvironment,...]}.
    Ripristina i metodi originali all'uscita (no leak verso altri test)."""
    cap = {"starts": [], "envs": []}
    orig_start = QProcess.start
    orig_env = QProcess.setProcessEnvironment

    def fake_start(self, program=None, arguments=None, *a, **k):
        cap["starts"].append((program, list(arguments) if arguments else []))
        # NON chiama l'originale: nessun processo reale

    def fake_env(self, env):
        cap["envs"].append(env)
        return orig_env(self, env)

    QProcess.start = fake_start
    QProcess.setProcessEnvironment = fake_env
    try:
        yield cap
    finally:
        QProcess.start = orig_start
        QProcess.setProcessEnvironment = orig_env


def _open(w):
    """Porta in primo piano il tab pesi e ritorna l'istanza WeightsTab."""
    wt = w.weights_tab
    try:
        w.tabs.setCurrentWidget(wt)
    except Exception:                                       # noqa: BLE001
        pass
    return wt


def _deselect_all(wt):
    for _id, (cb, _e) in wt._checks.items():
        cb.setChecked(False)


def _args_ids(args: list[str]) -> list[str]:
    assert "--ids" in args, f"argomento --ids assente: {args}"
    return args[args.index("--ids") + 1].split(",")


# ───────────────────────── apertura / refresh ─────────────────────────
def case_open_no_crash(w, spy):
    """Apertura del tab + refresh: nessun crash; ogni voce del catalogo ha un checkbox."""
    wt = _open(w)
    wt._refresh()
    assert wt._checks, "nessun checkbox costruito dal catalogo"
    # ogni id del catalogo deve avere un checkbox con etichetta non vuota
    for _id, (cb, _e) in wt._checks.items():
        assert cb.text(), f"checkbox senza etichetta per {_id}"


def case_refresh_clears_selection(w, spy):
    """_refresh() deve azzerare le selezioni (tutti i checkbox deselezionati)."""
    wt = _open(w)
    first_id = next(iter(wt._checks))
    wt._checks[first_id][0].setChecked(True)
    wt._refresh()
    assert not any(cb.isChecked() for cb, _e in wt._checks.values()), "refresh non ha azzerato i check"


# ───────────────────────── download: guardie ─────────────────────────
def case_download_no_selection(w, spy):
    """«Scarica selezionati» senza nulla selezionato: NON deve avviare processi e deve impostare
    uno status (nessun peso selezionato)."""
    wt = _open(w)
    _deselect_all(wt)
    with _capture_qprocess() as cap:
        wt._download()
    assert not cap["starts"], f"download avviato senza selezione: {cap['starts']}"
    assert wt.status.text() and wt.status.text() != "—", "nessuno status mostrato per selezione vuota"


def case_download_command_args(w, spy):
    """Con un peso selezionato il comando deve essere: python -u <download_weights.py> --ids <id>
    --base-dir <base>."""
    wt = _open(w)
    _deselect_all(wt)
    first_id = next(iter(wt._checks))
    wt._checks[first_id][0].setChecked(True)
    with _capture_qprocess() as cap:
        wt._download()
    assert cap["starts"], "download non avviato con un peso selezionato"
    prog, args = cap["starts"][-1]
    assert prog == sys.executable, f"eseguibile inatteso: {prog}"
    assert any("download_weights.py" in a for a in args), f"script download assente: {args}"
    assert first_id in _args_ids(args), f"--ids non contiene {first_id}: {args}"
    assert "--base-dir" in args, f"--base-dir assente: {args}"
    # la base-dir passata deve coincidere col campo GUI
    assert args[args.index("--base-dir") + 1] == wt.base.text().strip(), "base-dir non coincide col campo"


def case_double_download_guarded(w, spy):
    """Se un download è già in corso (_proc != None) una seconda richiesta non avvia un secondo
    processo."""
    wt = _open(w)
    _deselect_all(wt)
    first_id = next(iter(wt._checks))
    wt._checks[first_id][0].setChecked(True)
    with _capture_qprocess() as cap:
        wt._download()                      # avvia (fake) → _proc resta impostato
        wt._download()                      # deve essere ignorato
    assert len(cap["starts"]) == 1, f"doppio download non protetto: {cap['starts']}"


# ───────────────────────── seleziona mancanti ─────────────────────────
def case_select_missing_then_download(w, spy):
    """«Seleziona mancanti» deve spuntare solo i pesi non presenti; il download successivo deve
    inviare ESATTAMENTE quegli id (o nessun processo se è tutto presente)."""
    wt = _open(w)
    wt._select_missing()
    checked = [i for i, (cb, _e) in wt._checks.items() if cb.isChecked()]
    with _capture_qprocess() as cap:
        wt._download()
    if checked:
        assert cap["starts"], "mancanti selezionati ma download non avviato"
        _prog, args = cap["starts"][-1]
        assert set(_args_ids(args)) == set(checked), f"ids inviati {_args_ids(args)} != selezionati {checked}"
    else:
        assert not cap["starts"], "nessun mancante ma download avviato"


# ───────────────────────── HF token per i gated ─────────────────────────
def case_hf_token_in_env(w, spy):
    """Per un peso gated, l'HF token digitato deve finire nell'ambiente del processo
    (HF_TOKEN + HUGGING_FACE_HUB_TOKEN)."""
    wt = _open(w)
    gated = next((i for i, (_cb, e) in wt._checks.items() if e.get("gated")), None)
    if gated is None:
        return                              # catalogo senza gated: niente da verificare
    _deselect_all(wt)
    wt._checks[gated][0].setChecked(True)
    wt.hf.setText("hf_test_token_123")
    with _capture_qprocess() as cap:
        wt._download()
    assert cap["envs"], "setProcessEnvironment non chiamato"
    env = cap["envs"][-1]
    assert env.contains("HF_TOKEN") and env.value("HF_TOKEN") == "hf_test_token_123", "HF_TOKEN non iniettato"
    assert env.contains("HUGGING_FACE_HUB_TOKEN"), "HUGGING_FACE_HUB_TOKEN non iniettato"


def case_no_token_no_hf_env(w, spy):
    """Senza token digitato, l'ambiente non deve forzare HF_TOKEN vuoto (non si aggiunge la chiave)."""
    wt = _open(w)
    _deselect_all(wt)
    first_id = next(iter(wt._checks))
    wt._checks[first_id][0].setChecked(True)
    wt.hf.setText("")
    with _capture_qprocess() as cap:
        wt._download()
    assert cap["envs"], "setProcessEnvironment non chiamato"
    # il token non deve essere stato inserito esplicitamente dalla GUI (campo vuoto → branch saltato)
    # (non si può distinguere dall'env di sistema, quindi si verifica solo no-crash + processo avviato)
    assert cap["starts"], "download non avviato"


# ───────────────────────── scan pesi esistenti ─────────────────────────
def case_scan_empty_dir(w, spy):
    """Scan senza cartella indicata: status di avviso, nessun crash, _found vuoto."""
    wt = _open(w)
    wt.search.setText("")
    wt._scan()
    assert wt.status.text() and wt.status.text() != "—", "nessuno status per scan senza cartella"
    assert not wt._found, "scan senza cartella non dovrebbe trovare nulla"


def case_scan_finds_existing(w, spy):
    """Scan in una cartella che CONTIENE un peso che combacia (scan_names) → l'id finisce in _found."""
    wt = _open(w)
    # usa un peso non-gated del catalogo come bersaglio
    target = next((e for _i, (_cb, e) in wt._checks.items()
                   if e.get("scan_names") and not wc.is_present(e, wt.base.text().strip() or None)), None)
    if target is None:
        return                              # tutti già presenti: niente da scoprire
    name = target["scan_names"][0]
    sdir = H.OUT_DIR / "_weights_scan_test"
    folder = sdir / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors").write_bytes(b"\x00")     # file fittizio dentro la cartella
    wt.search.setText(str(sdir))
    wt._scan()
    assert target["id"] in wt._found, f"scan non ha trovato {target['id']} in {sdir} (found={wt._found})"


# ───────────────────────── browse (dialog patchati) ─────────────────────────
def case_browse_base_no_crash(w, spy):
    """«Sfoglia…» cartella base: il QFileDialog è patchato (ritorna '') → nessun crash, base invariata."""
    wt = _open(w)
    before = wt.base.text()
    wt._pick_base()
    assert wt.base.text() == before, "base modificata pur con dialog annullato"


def case_browse_search_no_crash(w, spy):
    """«Sfoglia…» cartella di ricerca: dialog patchato → nessun crash."""
    wt = _open(w)
    wt._pick_search()                       # ritorna '' → nessuna modifica, no crash


# ───────────────────────── un caso per OGNI modello del catalogo ─────────────────────────
def _make_model_case(entry: dict):
    eid = entry["id"]

    def case(w, spy):
        wt = _open(w)
        _deselect_all(wt)
        assert eid in wt._checks, f"modello {eid} assente dalla lista del tab"
        wt._checks[eid][0].setChecked(True)
        with _capture_qprocess() as cap:
            wt._download()
        assert cap["starts"], f"download non avviato per {eid}"
        _prog, args = cap["starts"][-1]
        ids = _args_ids(args)
        assert ids == [eid], f"selezionato solo {eid} ma --ids = {ids}"
        assert "--base-dir" in args, f"--base-dir assente per {eid}: {args}"

    return case


CASES = [
    ("weights_open_no_crash",
     "Apertura tab Pesi + refresh: nessun crash, ogni voce del catalogo ha un checkbox etichettato.",
     case_open_no_crash),
    ("weights_refresh_clears_selection",
     "refresh azzera le selezioni dei checkbox.",
     case_refresh_clears_selection),
    ("weights_download_no_selection",
     "Scarica selezionati senza selezione: nessun processo + status 'nessun peso'.",
     case_download_no_selection),
    ("weights_download_command_args",
     "Download: comando = python -u download_weights.py --ids <id> --base-dir <base> (coerente con la GUI).",
     case_download_command_args),
    ("weights_double_download_guarded",
     "Un download già in corso non ne avvia un secondo (guardia _proc).",
     case_double_download_guarded),
    ("weights_select_missing_then_download",
     "Seleziona mancanti → il download invia esattamente gli id mancanti (o nessuno se tutto presente).",
     case_select_missing_then_download),
    ("weights_hf_token_in_env",
     "Peso gated: l'HF token digitato finisce in env (HF_TOKEN + HUGGING_FACE_HUB_TOKEN).",
     case_hf_token_in_env),
    ("weights_no_token_no_hf_env",
     "Senza token: download avviato comunque, nessun crash (branch token saltato).",
     case_no_token_no_hf_env),
    ("weights_scan_empty_dir",
     "Scan senza cartella: status di avviso, _found vuoto, no crash.",
     case_scan_empty_dir),
    ("weights_scan_finds_existing",
     "Scan in cartella con peso combaciante (scan_names): l'id finisce in _found.",
     case_scan_finds_existing),
    ("weights_browse_base_no_crash",
     "Sfoglia cartella base (dialog annullato): nessun crash, base invariata.",
     case_browse_base_no_crash),
    ("weights_browse_search_no_crash",
     "Sfoglia cartella di ricerca (dialog annullato): nessun crash.",
     case_browse_search_no_crash),
]

# un caso di download per OGNI modello concreto del catalogo (scopre "funziona solo un modello")
for _e in wc.load():
    CASES.append((
        f"weights_download_model_{_e['id'].replace('-', '_')}",
        f"Seleziona SOLO il modello «{_e.get('label', _e['id'])}» ({_e['id']}) → --ids deve contenere solo quel id.",
        _make_model_case(_e),
    ))
