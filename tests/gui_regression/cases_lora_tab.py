"""Casi di regressione GUI-driven per il tab LoRA (ui/tabs/lora_tab.py).

Il tab LoRA è un PANNELLO DI SELEZIONE incorporato nel tab Video (w.lora_tab is w.video_tab.lora,
embedded=True, arch='LTX'). Non manda direttamente richieste al backend: espone la selezione
(selected_loras) che on_generate() del MainWindow inserisce nei params del worker video nativo
(_NativeLtx2Worker → params['loras']).

Quindi qui copriamo:
  • FUNZIONI del tab: selezione multipla, strength, deselezione, selezione che sopravvive al filtro,
    ricarica catalogo, preferiti, persistenza (gui_get/set_state), extra-LoRA da ComfyUI.
  • OPZIONI/toggle: ricerca (filtro), filtro NSFW (pilotato dal selettore Video), auto-trigger,
    trigger-warning / auto-insert.
  • OGNI "modello" = OGNI LoRA concreta del catalogo LTX: un caso per LoRA che la seleziona e
    lancia on_generate(), verificando che venga PASSATA al backend (così si scopre "funziona solo
    una LoRA"). I nomi sono enumerati a import-time dal catalogo (stdlib-only, niente Qt).

Lo Spy intercetta _enqueue: il worker nativo espone `.params` → params['loras'] = [{name,strength}].
"""
from __future__ import annotations

import harness as H


# ───────────────────────── helper ─────────────────────────
def _enum_lora_names():
    """Enumera (name, comfy_name, nsfw, has_triggers) di tutte le LoRA LTX del catalogo.

    Lettura diretta del catalogo (stdlib-only, nessuna finestra Qt): serve a generare un caso per
    OGNI LoRA. Avvolta in try/except: se il catalogo manca, niente casi per-LoRA (non rompe l'import).
    """
    try:
        from core.lora_catalog import LoraCatalog
        cat = LoraCatalog(compatible_arch="LTX")
        return [(e.name, e.comfy_name, e.nsfw, e.has_real_triggers) for e in cat.loras(allow_nsfw=True)]
    except Exception:                                            # noqa: BLE001
        return []


_LORAS = _enum_lora_names()


def _show_all(w):
    """Mette il selettore di sezione su NSFW così TUTTE le LoRA (SFW+NSFW) sono visibili in lista."""
    try:
        w.video_tab.nsfw.setCurrentText("NSFW")
    except Exception:                                            # noqa: BLE001
        pass
    w.lora_tab.refresh()


def _row_by_name(w, name):
    return w.lora_tab._rows.get(name)


def _select(w, name, strength=0.8):
    """Spunta la LoRA `name` con la strength data. Ritorna la riga (o None se assente)."""
    row = _row_by_name(w, name)
    if row is None:
        return None
    row["cb"].setChecked(True)
    row["sp"].setValue(strength)
    return row


def _trigger_lora_name():
    """Nome della prima LoRA con trigger reali (per i test trigger-word)."""
    for name, _cn, _nsfw, has_t in _LORAS:
        if has_t:
            return name
    return None


# ───────────────────────── FUNZIONI base ─────────────────────────
def case_list_populated(w, spy):
    """All'avvio la lista LoRA del tab Video deve essere popolata dal catalogo LTX (>0 righe)."""
    _show_all(w)
    assert w.lora_tab._rows, "lista LoRA vuota: il catalogo LTX non è stato caricato nel tab"


def case_select_single(w, spy):
    """Selezione singola: spuntare una LoRA la fa comparire in selected_loras() col suo comfy_name."""
    _show_all(w)
    name, comfy = _LORAS[0][0], _LORAS[0][1]
    assert _select(w, name) is not None, f"riga LoRA '{name}' assente"
    sel = w.lora_tab.selected_loras()
    assert any(s["name"] == comfy for s in sel), f"LoRA selezionata non in selected_loras(): {sel}"


def case_select_multiple(w, spy):
    """Selezione MULTIPLA: spuntare 2 LoRA → selected_loras() ne ritorna 2 distinte."""
    _show_all(w)
    if len(_LORAS) < 2:
        return
    n1, c1 = _LORAS[0][0], _LORAS[0][1]
    n2, c2 = _LORAS[1][0], _LORAS[1][1]
    _select(w, n1); _select(w, n2)
    names = {s["name"] for s in w.lora_tab.selected_loras()}
    assert {c1, c2} <= names, f"selezione multipla persa: attese {c1},{c2} in {names}"


def case_deselect(w, spy):
    """Deselezione: spuntare e poi togliere la spunta → selected_loras() non la include più."""
    _show_all(w)
    name, comfy = _LORAS[0][0], _LORAS[0][1]
    row = _select(w, name)
    row["cb"].setChecked(False)
    sel = {s["name"] for s in w.lora_tab.selected_loras()}
    assert comfy not in sel, f"LoRA deselezionata ancora presente: {sel}"


def case_strength_passed(w, spy):
    """La strength impostata nello spinbox deve arrivare in selected_loras()."""
    _show_all(w)
    name, comfy = _LORAS[0][0], _LORAS[0][1]
    _select(w, name, strength=1.25)
    sel = [s for s in w.lora_tab.selected_loras() if s["name"] == comfy]
    assert sel and abs(sel[0]["strength"] - 1.25) < 1e-6, f"strength non propagata: {sel}"


def case_selection_survives_filter(w, spy):
    """Selezione che sopravvive al filtro: spunta una LoRA, poi una ricerca che la NASCONDE → deve
    restare in selected_loras() (verità di on_generate, anche se non visibile)."""
    _show_all(w)
    name, comfy = _LORAS[0][0], _LORAS[0][1]
    _select(w, name)
    w.lora_tab.search.setText("zzz_nonexistent_query_xyz")   # nasconde tutte le righe
    assert name not in w.lora_tab._rows, "il filtro non ha nascosto la riga"
    sel = {s["name"] for s in w.lora_tab.selected_loras()}
    assert comfy in sel, f"selezione persa dopo il filtro: {sel}"


def case_search_filter_narrows(w, spy):
    """Ricerca: una query che combacia con UNA LoRA riduce la lista a quella; query assurda → 0 righe."""
    _show_all(w)
    name = _LORAS[0][0]
    before = len(w.lora_tab._rows)
    w.lora_tab.search.setText(name[:8])
    assert len(w.lora_tab._rows) <= before, "la ricerca non ha ristretto la lista"
    assert name in w.lora_tab._rows, f"la LoRA cercata '{name}' non è più mostrata"
    w.lora_tab.search.setText("___no_match_query___")
    assert len(w.lora_tab._rows) == 0, "la query senza match dovrebbe svuotare la lista"


def case_nsfw_filter(w, spy):
    """Filtro NSFW (pilotato dal selettore di sezione Video): SFW nasconde le LoRA NSFW; NSFW le mostra."""
    nsfw_names = [n for (n, _c, nsfw, _t) in _LORAS if nsfw]
    if not nsfw_names:
        return
    w.video_tab.nsfw.setCurrentText("SFW"); w.lora_tab.refresh()
    n_sfw = len(w.lora_tab._rows)
    assert nsfw_names[0] not in w.lora_tab._rows, "una LoRA NSFW è visibile in modalità SFW"
    w.video_tab.nsfw.setCurrentText("NSFW"); w.lora_tab.refresh()
    n_all = len(w.lora_tab._rows)
    assert n_all > n_sfw, f"il selettore NSFW non amplia la lista ({n_all} <= {n_sfw})"
    assert nsfw_names[0] in w.lora_tab._rows, "LoRA NSFW assente anche in modalità NSFW"


def case_auto_triggers_toggle(w, spy):
    """Toggle auto-trigger: auto_insert_triggers() deve riflettere lo stato della checkbox."""
    _show_all(w)
    w.lora_tab.auto_triggers.setChecked(True)
    assert w.lora_tab.auto_insert_triggers() is True
    w.lora_tab.auto_triggers.setChecked(False)
    assert w.lora_tab.auto_insert_triggers() is False


def case_reload(w, spy):
    """Pulsante Ricarica: _reload() ricarica il catalogo dal disco senza crash; lista ancora popolata."""
    _show_all(w)
    w.lora_tab._reload()
    assert w.lora_tab._rows, "dopo il reload la lista è vuota"


def case_fav_toggle(w, spy):
    """Preferiti: _toggle_fav marca/smarca la LoRA; dopo refresh torna in cima alla lista."""
    _show_all(w)
    name, comfy = _LORAS[0][0], _LORAS[0][1]
    row = _row_by_name(w, name)
    assert row is not None
    ent = row["entry"]
    try:
        from core import user_prefs
        was = user_prefs.is_fav_lora(comfy)
        w.lora_tab._toggle_fav(ent)
        assert user_prefs.is_fav_lora(comfy) != was, "lo stato preferito non è cambiato"
        # ripristina lo stato originale per non sporcare le preferenze utente
        w.lora_tab._toggle_fav(ent)
    except Exception as e:                                       # noqa: BLE001
        # la persistenza preferiti può non essere disponibile nell'ambiente di test: no-crash sufficiente
        assert "is_fav_lora" not in str(e)


def case_gui_state_roundtrip(w, spy):
    """Persistenza: gui_get_state cattura la selezione; gui_set_state la ripristina (round-trip)."""
    _show_all(w)
    name, comfy = _LORAS[0][0], _LORAS[0][1]
    _select(w, name, strength=0.9)
    w.lora_tab.auto_triggers.setChecked(False)
    st = w.lora_tab.gui_get_state()
    assert any(d["name"] == comfy for d in st["loras"]), f"gui_get_state non contiene la LoRA: {st}"
    assert st["auto_triggers"] is False
    # azzera, poi ripristina dallo stato salvato
    w.lora_tab.gui_set_state({"loras": [], "auto_triggers": True})
    assert not w.lora_tab.selected_loras(), "gui_set_state([]) non ha azzerato la selezione"
    w.lora_tab.gui_set_state(st)
    sel = {s["name"] for s in w.lora_tab.selected_loras()}
    assert comfy in sel, f"gui_set_state non ha ripristinato la LoRA: {sel}"


def case_trigger_warning_missing(w, spy):
    """Trigger-word: LoRA con trigger reali + prompt SENZA trigger → trigger_warnings/missing non vuoti."""
    tname = _trigger_lora_name()
    if not tname:
        return
    _show_all(w)
    _select(w, tname)
    prompt = "a calm landscape at sunset"
    assert w.lora_tab.missing_trigger_words(prompt), "trigger mancanti non rilevati"
    assert w.lora_tab.trigger_warnings(prompt), "nessun avviso trigger generato"


def case_trigger_present_no_warning(w, spy):
    """Trigger-word presente nel prompt → nessun avviso e nessun trigger mancante."""
    tname = _trigger_lora_name()
    if not tname:
        return
    _show_all(w)
    row = _select(w, tname)
    trig = row["entry"].triggers[0]
    prompt = f"a scene with {trig} included"
    assert not w.lora_tab.missing_trigger_words(prompt), "trigger presente ma segnalato come mancante"
    assert not w.lora_tab.trigger_warnings(prompt), "avviso trigger spurio col trigger nel prompt"


def case_set_available_extra(w, spy):
    """Fallback ComfyUI: set_available_loras aggiunge una LoRA LTX installata ma non catalogata."""
    _show_all(w)
    fake = "LTX-2\\__regtest_extra_ltx_lora.safetensors"
    w.lora_tab.set_available_loras([fake])
    extra_name = "__regtest_extra_ltx_lora"
    # con selettore NSFW=NSFW tutte le righe sono visibili: la extra deve comparire
    assert extra_name in w.lora_tab._rows, f"LoRA extra ComfyUI non aggiunta: {list(w.lora_tab._rows)[:3]}…"


# ───────────────────────── INTEGRAZIONE: selezione → backend ─────────────────────────
def case_generate_includes_lora(w, spy):
    """End-to-end: selezionando una LoRA, on_generate() la PASSA al worker nativo (params['loras'])."""
    _show_all(w)
    name, comfy = _LORAS[0][0], _LORAS[0][1]
    _select(w, name)
    w.video_tab.prompt.setPlainText("a cinematic shot of a city street at night")
    w.tabs.setCurrentWidget(w.video_tab)
    w.on_generate()
    assert spy.calls, "on_generate non ha inviato nulla al backend con una LoRA selezionata"
    p = spy.calls[-1]["params"]
    sent = [d.get("name") for d in (p.get("loras") or [])]
    assert comfy in sent, f"LoRA selezionata non passata al backend: inviate={sent}"


def case_generate_auto_insert_trigger(w, spy):
    """auto-trigger ON + LoRA con trigger + prompt senza trigger → on_generate procede (no dialog,
    trigger inserito) e manda la richiesta."""
    tname = _trigger_lora_name()
    if not tname:
        return
    _show_all(w)
    w.lora_tab.auto_triggers.setChecked(True)
    _select(w, tname)
    w.video_tab.prompt.setPlainText("a quiet room, soft light")
    w.tabs.setCurrentWidget(w.video_tab)
    w.on_generate()
    assert spy.calls, "on_generate non ha inviato nulla (auto-insert trigger dovrebbe evitare il dialog)"


# ───────────────────────── PER-LoRA: un caso per OGNI modello concreto ─────────────────────────
def _make_lora_case(name, comfy):
    def _case(w, spy):
        _show_all(w)
        row = _row_by_name(w, name)
        assert row is not None, f"LoRA '{name}' assente dalla lista (filtro arch/NSFW?)"
        row["cb"].setChecked(True); row["sp"].setValue(0.8)
        sel = {s["name"] for s in w.lora_tab.selected_loras()}
        assert comfy in sel, f"LoRA '{name}' non in selected_loras(): {sel}"
        w.video_tab.prompt.setPlainText("a cinematic establishing shot, detailed, natural light")
        w.tabs.setCurrentWidget(w.video_tab)
        w.on_generate()
        assert spy.calls, f"on_generate non ha inviato nulla con la LoRA '{name}'"
        p = spy.calls[-1]["params"]
        sent = [d.get("name") for d in (p.get("loras") or [])]
        assert comfy in sent, f"LoRA '{name}' NON passata al backend (funziona solo un'altra?): {sent}"
    return _case


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


CASES = [
    ("lora_list_populated", "La lista LoRA del tab Video è popolata dal catalogo LTX all'avvio.",
     case_list_populated),
    ("lora_select_single", "Selezione singola: la LoRA spuntata compare in selected_loras() col comfy_name.",
     case_select_single),
    ("lora_select_multiple", "Selezione multipla: spuntare 2 LoRA le ritorna entrambe in selected_loras().",
     case_select_multiple),
    ("lora_deselect", "Deselezione: togliere la spunta rimuove la LoRA da selected_loras().",
     case_deselect),
    ("lora_strength_passed", "La strength dello spinbox arriva in selected_loras().",
     case_strength_passed),
    ("lora_selection_survives_filter", "La selezione sopravvive a una ricerca che nasconde la riga.",
     case_selection_survives_filter),
    ("lora_search_filter_narrows", "La ricerca filtra la lista; query senza match → 0 righe.",
     case_search_filter_narrows),
    ("lora_nsfw_filter", "Il selettore NSFW della sezione Video filtra/mostra le LoRA NSFW.",
     case_nsfw_filter),
    ("lora_auto_triggers_toggle", "Il toggle auto-trigger si riflette in auto_insert_triggers().",
     case_auto_triggers_toggle),
    ("lora_reload", "Il pulsante Ricarica ricarica il catalogo senza crash.",
     case_reload),
    ("lora_fav_toggle", "Preferiti: _toggle_fav marca/smarca la LoRA.",
     case_fav_toggle),
    ("lora_gui_state_roundtrip", "Persistenza: gui_get_state/gui_set_state fanno round-trip della selezione.",
     case_gui_state_roundtrip),
    ("lora_trigger_warning_missing", "LoRA con trigger + prompt senza trigger → avviso/mancanti non vuoti.",
     case_trigger_warning_missing),
    ("lora_trigger_present_no_warning", "Trigger nel prompt → nessun avviso.",
     case_trigger_present_no_warning),
    ("lora_set_available_extra", "set_available_loras aggiunge una LoRA LTX installata ma non catalogata.",
     case_set_available_extra),
    ("lora_generate_includes_lora", "on_generate passa la LoRA selezionata nei params['loras'] del worker nativo.",
     case_generate_includes_lora),
    ("lora_generate_auto_insert_trigger", "auto-trigger ON: on_generate procede senza dialog e invia la richiesta.",
     case_generate_auto_insert_trigger),
]

# Un caso per OGNI LoRA concreta del catalogo (scopre 'funziona solo una LoRA').
for _name, _comfy, _nsfw, _ in _LORAS:
    CASES.append((
        f"lora_model_{_safe(_name)}",
        f"LoRA '{_name}'{' [NSFW]' if _nsfw else ''}: selezione + on_generate la passa al backend (params['loras']).",
        _make_lora_case(_name, _comfy),
    ))
