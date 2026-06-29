"""Casi di regressione GUI-driven per il tab «LoRA Foto (FLUX.2)» (ui/tabs/lora_image_tab.py).

Particolarità del tab: l'azione di training NON passa da _enqueue/native_driver (che lo Spy
dell'harness intercetta), ma avvia direttamente un QThread `_LoraImageWorker` che chiama
`core.lora_image_service`. Per testare l'ISTRUZIONE senza GPU/disco/rete sostituiamo il worker
con un finto che:
  - registra (subjects, name, out_dir, opts, full) dentro spy.calls (riusiamo la lista dello Spy),
  - NON avvia alcun thread (start/isRunning no-op).

Copertura:
  - Azioni: «Genera dataset» (_start False), «Genera + Allena» (_start True), «Installa ai-toolkit».
  - Combo «Modello base»: UN caso per OGNI modello concreto (flux2-dev / klein-4b / klein-9b /
    flux1-dev) → scopre il caso "funziona solo un modello".
  - Combo «Risoluzione»: 768 / 1024 / 1280 (propagazione width/height).
  - Spinbox rank/steps (propagazione opzioni).
  - Editor soggetto: tipi person/place/object/concept, focus-volto, situazioni custom.
  - Guard di precondizione: nessun soggetto / nessuna foto-media → NESSUNA richiesta (validazione).
"""
from __future__ import annotations

import harness as H

_BASES = [("flux2-dev", "FLUX.2 [dev]"), ("flux2-klein-4b", "FLUX.2 Klein 4B"),
          ("flux2-klein-9b", "FLUX.2 Klein 9B"), ("flux1-dev", "FLUX.1 [dev]")]
_RES = ["768", "1024", "1280"]
_TYPES = ["person", "place", "object", "concept"]


# ───────────────────────── fake worker (cattura senza eseguire) ─────────────────────────
class _FakeSignal:
    def connect(self, *a, **k):
        return None

    def emit(self, *a, **k):
        return None


class _FakeWorker:
    """Sostituisce _LoraImageWorker: registra i parametri e non avvia thread."""
    _spy = None

    def __init__(self, subjects, name, out_dir, opts, full):
        self.progress = _FakeSignal()
        self.finished_lora = _FakeSignal()
        if _FakeWorker._spy is not None:
            _FakeWorker._spy.calls.append({
                "via": "lora_worker",
                "label": "full" if full else "dataset",
                "module": "lora_image_service",
                "params": {
                    "name": name,
                    "out_dir": out_dir,
                    "full": full,
                    "n_subjects": len(subjects),
                    "subject_names": [s.name for s in subjects],
                    "subject_types": [s.type for s in subjects],
                    "face_focus": [s.face_focus for s in subjects],
                    "n_customs": [len(s.customs) for s in subjects],
                    "n_photos": [len(s.all_photos()) for s in subjects],
                    **(opts or {}),
                },
            })

    def isRunning(self):
        return False

    def start(self):
        return None


def _install_worker_spy(spy):
    """Patcha il riferimento globale a _LoraImageWorker nel modulo del tab."""
    import ui.tabs.lora_image_tab as lit
    _FakeWorker._spy = spy
    lit._LoraImageWorker = _FakeWorker


# ───────────────────────── helpers ─────────────────────────
def _set_base(tab, code):
    cb = tab.base_model
    for i in range(cb.count()):
        if cb.itemData(i) == code:
            cb.setCurrentIndex(i)
            return cb.itemText(i)
    raise AssertionError(f"modello base '{code}' non presente nel combo")


def _set_type(tab, stype):
    cb = tab.ed_type
    for i in range(cb.count()):
        if cb.itemData(i) == stype:
            cb.setCurrentIndex(i)
            return
    raise AssertionError(f"tipo soggetto '{stype}' non presente nel combo")


def _prep(w, spy, with_media: bool = True, stype: str | None = None):
    """Crea un soggetto valido (nome + foto) e installa il worker-spy. Ritorna il tab."""
    tab = w.lora_image_tab
    _install_worker_spy(spy)
    w.tabs.setCurrentWidget(tab)
    tab._add_subject()
    tab.ed_name.setText("Martina")          # textChanged → _save_cur (imposta name=trigger)
    if stype:
        _set_type(tab, stype)               # currentIndexChanged → _on_type_change
    if with_media:
        path = H.make_test_image(str(H.OUT_DIR / "_lora_img_ref.png"))
        tab._subjects[0]["photos"].append({"path": path, "angle": "front"})
    tab._save_cur()
    return tab


# ───────────────────────── azioni base ─────────────────────────
def case_dataset_only(w, spy):
    """«Genera dataset» (_start False): invia subjects+name al servizio con full=False."""
    tab = _prep(w, spy)
    tab._start(False)
    assert spy.calls, "genera-dataset: nessuna richiesta al worker"
    p = spy.calls[-1]["params"]
    assert p.get("full") is False, f"atteso full=False: {p}"
    assert p.get("n_subjects") == 1 and "Martina" in p.get("subject_names", []), f"soggetto non passato: {p}"
    assert p.get("n_photos") == [1], f"foto non passate al dataset: {p}"


def case_train_full_default(w, spy):
    """«Genera + Allena» (_start True): invia opzioni complete (base/rank/steps/width/height)."""
    tab = _prep(w, spy)
    tab._start(True)
    assert spy.calls, "genera+allena: nessuna richiesta al worker"
    p = spy.calls[-1]["params"]
    assert p.get("full") is True, f"atteso full=True: {p}"
    assert p.get("base"), f"modello base mancante: {p}"
    assert p.get("rank") and p.get("steps"), f"rank/steps mancanti: {p}"
    assert p.get("width") and p.get("height"), f"width/height mancanti: {p}"


def case_install_button(w, spy):
    """«Installa ai-toolkit»: deve lanciare il processo installer senza crashare (Popen mockato)."""
    import subprocess
    tab = w.lora_image_tab
    w.tabs.setCurrentWidget(tab)
    launched = []
    orig = subprocess.Popen
    subprocess.Popen = lambda *a, **k: launched.append(a)
    try:
        tab._install()
    finally:
        subprocess.Popen = orig
    assert launched, "install: nessun processo installer avviato"


# ───────────────────────── opzioni: rank/steps ─────────────────────────
def case_rank_steps_propagation(w, spy):
    """Rank/Steps non-default devono arrivare al servizio invariati."""
    tab = _prep(w, spy)
    tab.lora_rank.setValue(64)
    tab.lora_steps.setValue(4000)
    tab._start(True)
    p = spy.calls[-1]["params"]
    assert p.get("rank") == 64 and p.get("steps") == 4000, f"rank/steps non propagati: {p}"


# ───────────────────────── editor soggetto: focus-volto / custom ─────────────────────────
def case_face_focus_off(w, spy):
    """Disattivando «Focus sul volto» il soggetto deve propagare face_focus=False."""
    tab = _prep(w, spy)
    tab.ed_face.setChecked(False)           # toggled → _save_cur
    tab._save_cur()
    tab._start(False)
    p = spy.calls[-1]["params"]
    assert p.get("face_focus") == [False], f"face_focus non propagato: {p}"


def case_custom_situations(w, spy):
    """Le situazioni custom (una per riga) devono arrivare come lista al soggetto."""
    tab = _prep(w, spy)
    tab.ed_customs.setPlainText("corre sulla spiaggia\nballa felice\nbeve un caffe")
    tab._save_cur()
    tab._start(False)
    p = spy.calls[-1]["params"]
    assert p.get("n_customs") == [3], f"situazioni custom non propagate: {p}"


# ───────────────────────── guard di precondizione ─────────────────────────
def case_guard_no_subject(w, spy):
    """Nessun soggetto: la GUI deve avvisare e NON inviare nulla al servizio."""
    tab = w.lora_image_tab
    _install_worker_spy(spy)
    w.tabs.setCurrentWidget(tab)
    tab._start(True)
    assert not spy.calls, f"avviato training senza soggetti: {spy.calls}"


def case_guard_no_media(w, spy):
    """Soggetto con nome ma SENZA foto/media: deve avvisare e NON inviare nulla."""
    tab = _prep(w, spy, with_media=False)
    tab._start(True)
    assert not spy.calls, f"avviato training senza foto/media: {spy.calls}"


# ───────────────────────── generatori parametrici (modelli/risoluzioni/tipi) ─────────────────────────
def _make_model_case(code):
    def case(w, spy):
        tab = _prep(w, spy)
        _set_base(tab, code)
        tab._start(True)
        assert spy.calls, f"modello {code}: nessuna richiesta inviata"
        p = spy.calls[-1]["params"]
        assert p.get("base") == code, f"modello base errato: atteso {code}, ottenuto {p.get('base')}"
        assert p.get("full") is True, f"atteso full=True: {p}"
    return case


def _make_res_case(res):
    def case(w, spy):
        tab = _prep(w, spy)
        tab.img_res.setCurrentText(res)
        tab._start(True)
        p = spy.calls[-1]["params"]
        assert p.get("width") == int(res) and p.get("height") == int(res), \
            f"risoluzione {res} non propagata in width/height: {p}"
    return case


def _make_type_case(stype):
    def case(w, spy):
        tab = _prep(w, spy, stype=stype)
        tab._start(False)
        assert spy.calls, f"tipo {stype}: nessuna richiesta inviata"
        p = spy.calls[-1]["params"]
        assert stype in p.get("subject_types", []), f"tipo soggetto non propagato: atteso {stype} in {p}"
        assert p.get("full") is False, f"atteso full=False (dataset): {p}"
    return case


# ───────────────────────── registro casi ─────────────────────────
CASES = [
    ("lora_image_dataset_only",
     "«Genera dataset»: invia subjects+nome al servizio con full=False (no training).",
     case_dataset_only),
    ("lora_image_train_full_default",
     "«Genera + Allena»: invia base+rank+steps+width+height con full=True.",
     case_train_full_default),
    ("lora_image_install_button",
     "«Installa ai-toolkit»: avvia il processo installer senza crashare.",
     case_install_button),
    ("lora_image_rank_steps_propagation",
     "Rank/Steps personalizzati arrivano al servizio invariati.",
     case_rank_steps_propagation),
    ("lora_image_face_focus_off",
     "Disattivando il focus-volto il soggetto propaga face_focus=False.",
     case_face_focus_off),
    ("lora_image_custom_situations",
     "Le situazioni custom (3 righe) arrivano come lista al soggetto.",
     case_custom_situations),
    ("lora_image_guard_no_subject",
     "Nessun soggetto: avvisa e NON invia richieste al servizio.",
     case_guard_no_subject),
    ("lora_image_guard_no_media",
     "Soggetto senza foto/media: avvisa e NON invia richieste al servizio.",
     case_guard_no_media),
]

# un caso per OGNI modello base concreto (scopre "funziona solo un modello")
for _code, _label in _BASES:
    CASES.append((f"lora_image_model_{_code.replace('-', '_')}",
                  f"Training con modello base '{_label}' ({_code}): base propagato correttamente.",
                  _make_model_case(_code)))

# un caso per OGNI risoluzione esposta
for _r in _RES:
    CASES.append((f"lora_image_res_{_r}",
                  f"Risoluzione {_r}: width/height propagati = {_r}.",
                  _make_res_case(_r)))

# un caso per OGNI tipo di soggetto (copertura casi cambia per tipo)
for _t in _TYPES:
    CASES.append((f"lora_image_subject_type_{_t}",
                  f"Soggetto di tipo '{_t}': dataset generato col tipo corretto.",
                  _make_type_case(_t)))
