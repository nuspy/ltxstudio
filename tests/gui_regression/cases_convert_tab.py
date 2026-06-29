"""Casi di regressione GUI-driven per il tab "Convertitore 3D" (ui/tabs/convert_tab.py).

Questo tab NON usa ComfyUI/native_driver: la richiesta GUI→backend passa per Blender headless
tramite MainWindow._start_blender("convert.py", args, ...). Lo Spy dell'harness (che intercetta
_enqueue/ComfyDriver/native_driver) NON cattura quindi nulla per questo tab. Per testare l'ISTRUZIONE
che la GUI manda al backend, ogni caso:
  1) patcha core.blender_service.blender_available → True (Blender "presente", niente winget),
  2) patcha w._start_blender per CATTURARE (script, args) SENZA lanciare il sottoprocesso Blender,
  3) imposta input/formato/opzioni nel tab,
  4) invoca l'azione reale del tab (w.convert_tab._on_run()) → emette request_convert → run_convert,
  5) verifica che l'istruzione catturata sia coerente (script=convert.py, input, format, opzioni,
     output con estensione giusta).

Copertura:
  - guardia "manca input" (nessuna richiesta),
  - run base (formato di default),
  - UN caso per OGNI formato di export concreto (scopre "funziona solo glb/…"),
  - opzioni: scale, embed_textures (toggle), animations (toggle),
  - API set_input_path / set_format,
  - ramo "Blender assente" → _offer_blender_install (nessun job lanciato).

Importabile senza side-effect (i setup incerti sono dentro i casi).
"""
from __future__ import annotations

from pathlib import Path

import harness as H

from ui.tabs.convert_tab import EXPORT_FORMATS


# ───────────────────────── helper ─────────────────────────
def _dummy_input(name: str = "_convert_test_input.glb") -> str:
    """Crea un file fittizio da usare come input 3D (mai realmente letto: Blender è intercettato)."""
    p = H.OUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(b"glTF\x00\x00\x00\x00")   # contenuto irrilevante: niente Blender nei test
    return str(p)


def _capture_blender(w):
    """Patcha blender_available→True e w._start_blender per catturare l'istruzione Blender.

    Ritorna la lista `captured` di dict {script, args, timeout}. NON viene lanciato alcun
    sottoprocesso Blender (né QThread)."""
    import core.blender_service as bs
    bs.blender_available = lambda: True

    captured: list[dict] = []

    def fake_start(script, args, on_ok, on_err, timeout: int = 900):
        captured.append({"script": script, "args": dict(args or {}), "timeout": timeout})

    w._start_blender = fake_start
    return captured


def _prep(w, input_path: str | None = None):
    """Porta il tab convert in primo piano, imposta un input valido, attiva la cattura Blender."""
    w.tabs.setCurrentWidget(w.convert_tab)
    captured = _capture_blender(w)
    w.convert_tab.set_input_path(input_path or _dummy_input())
    return captured


# ───────────────────────── guardia: input mancante ─────────────────────────
def case_no_input_guard(w, spy):
    """Senza file di input, _on_run NON deve inviare alcun job Blender (solo log a video)."""
    w.tabs.setCurrentWidget(w.convert_tab)
    captured = _capture_blender(w)
    w.convert_tab.set_input_path("")          # nessun input
    w.convert_tab._on_run()
    assert not captured, f"convert senza input non doveva lanciare Blender: {captured}"


# ───────────────────────── run base (formato di default) ─────────────────────────
def case_run_default(w, spy):
    """Run con input valido + formato di default: deve lanciare convert.py con input/format/output."""
    captured = _prep(w)
    w.convert_tab._on_run()
    assert captured, "convert: nessun job Blender inviato da _on_run()"
    c = captured[-1]
    assert c["script"] == "convert.py", f"script atteso convert.py, ricevuto {c['script']}"
    a = c["args"]
    assert a.get("input"), f"input mancante nei params: {a}"
    assert a.get("format"), f"format mancante nei params: {a}"
    assert a.get("output", "").lower().endswith("." + a["format"].lower()), \
        f"output non coerente col formato: {a.get('output')} (fmt={a.get('format')})"
    # le opzioni devono propagarsi
    opts = a.get("options", {})
    assert "scale" in opts and "embed_textures" in opts and "animations" in opts, \
        f"opzioni mancanti: {opts}"


# ───────────────────────── opzioni ─────────────────────────
def case_scale_option(w, spy):
    """Lo scale impostato nello spinbox deve arrivare al backend in options.scale."""
    captured = _prep(w)
    w.convert_tab.scale.setValue(2.5)
    w.convert_tab._on_run()
    assert captured, "convert(scale): nessun job inviato"
    assert abs(captured[-1]["args"]["options"]["scale"] - 2.5) < 1e-6, \
        f"scale non propagato: {captured[-1]['args']['options']}"


def case_embed_textures_off(w, spy):
    """Disattivando 'incorpora texture', options.embed_textures deve essere False."""
    captured = _prep(w)
    w.convert_tab.embed.setChecked(False)
    w.convert_tab._on_run()
    assert captured, "convert(embed): nessun job inviato"
    assert captured[-1]["args"]["options"]["embed_textures"] is False, \
        f"embed_textures non disattivato: {captured[-1]['args']['options']}"


def case_animations_off(w, spy):
    """Disattivando 'animazioni', options.animations deve essere False."""
    captured = _prep(w)
    w.convert_tab.animations.setChecked(False)
    w.convert_tab._on_run()
    assert captured, "convert(anim): nessun job inviato"
    assert captured[-1]["args"]["options"]["animations"] is False, \
        f"animations non disattivato: {captured[-1]['args']['options']}"


# ───────────────────────── API set_input_path / set_format ─────────────────────────
def case_set_input_path_api(w, spy):
    """set_input_path deve aggiornare la riga path e finire nella config inviata al backend."""
    path = _dummy_input("_convert_api_input.fbx")
    captured = _prep(w, input_path=path)
    assert w.convert_tab.path_edit.text() == path, "path_edit non aggiornato da set_input_path"
    w.convert_tab._on_run()
    assert captured and captured[-1]["args"].get("input") == path, \
        f"input dell'API non propagato: {captured}"


def case_set_format_api(w, spy):
    """set_format('obj') deve selezionare la voce combo giusta e produrre format='obj'."""
    captured = _prep(w)
    w.convert_tab.set_format("obj")
    assert w.convert_tab.fmt.currentData() == "obj", \
        f"set_format non ha selezionato obj: {w.convert_tab.fmt.currentData()}"
    w.convert_tab._on_run()
    assert captured and captured[-1]["args"].get("format") == "obj", \
        f"format obj non propagato: {captured}"


# ───────────────────────── ramo Blender assente ─────────────────────────
def case_blender_unavailable_offers_install(w, spy):
    """Se Blender non è disponibile, run_convert deve offrire l'installazione e NON lanciare job."""
    w.tabs.setCurrentWidget(w.convert_tab)
    import core.blender_service as bs
    bs.blender_available = lambda: False
    captured: list[dict] = []
    w._start_blender = lambda *a, **k: captured.append(a)
    offered = {"n": 0}
    w._offer_blender_install = lambda *a, **k: offered.__setitem__("n", offered["n"] + 1) or False
    w.convert_tab.set_input_path(_dummy_input())
    w.convert_tab._on_run()
    assert offered["n"] == 1, "con Blender assente doveva offrire l'installazione"
    assert not captured, f"con Blender assente non doveva lanciare job: {captured}"


# ───────────────────────── un caso per OGNI formato di export ─────────────────────────
def _make_format_case(code: str, label: str):
    def case(w, spy):
        captured = _prep(w)
        w.convert_tab.set_format(code)
        assert w.convert_tab.fmt.currentData() == code, \
            f"formato {code} non selezionabile nel combo (label atteso: {label})"
        w.convert_tab._on_run()
        assert captured, f"convert({code}): nessun job inviato"
        a = captured[-1]["args"]
        assert a.get("format") == code, f"format errato per {code}: {a.get('format')}"
        assert a.get("output", "").lower().endswith("." + code.lower()), \
            f"output non coerente per {code}: {a.get('output')}"
    case.__name__ = f"case_format_{code}"
    return case


_FORMAT_CASES = [
    (f"convert_format_{code}",
     f"Export formato '{code}' ({label}): la GUI invia convert.py con format={code} e output .{code}.",
     _make_format_case(code, label))
    for code, label in EXPORT_FORMATS
]


CASES = [
    ("convert_no_input_guard",
     "Senza input: _on_run non lancia alcun job Blender (solo log).",
     case_no_input_guard),
    ("convert_run_default",
     "Run base: la GUI invia convert.py con input + format + output coerente + opzioni complete.",
     case_run_default),
    ("convert_scale_option",
     "Opzione scale: lo spinbox arriva al backend in options.scale.",
     case_scale_option),
    ("convert_embed_textures_off",
     "Toggle 'incorpora texture' OFF → options.embed_textures=False.",
     case_embed_textures_off),
    ("convert_animations_off",
     "Toggle 'animazioni' OFF → options.animations=False.",
     case_animations_off),
    ("convert_set_input_path_api",
     "API set_input_path: aggiorna la GUI e propaga l'input al backend.",
     case_set_input_path_api),
    ("convert_set_format_api",
     "API set_format('obj'): seleziona il combo e propaga format=obj.",
     case_set_format_api),
    ("convert_blender_unavailable_offers_install",
     "Blender assente: offre l'installazione e non lancia job.",
     case_blender_unavailable_offers_install),
    *_FORMAT_CASES,
]
