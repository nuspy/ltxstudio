"""Tabella UNICA azione -> motori del tab «Modifica Video AI» (e dello studio «Scambia»).

Requisito dell'utente (02/11): per ogni azione (modifica globale, oggetti, sostituisci personaggio, cambia sfondo...) un dropdown
«Motore» con SOLO i motori che fanno quell'azione; sceglie l'utente. La GUI legge da qui; i test verificano che il motore scelto nel
dropdown sia quello davvero chiamato nel backend (`engine_params` -> chiavi `engine`/`engine_kind` nei params del worker).

Modulo PURO (nessun Qt, nessun torch). Regole:
- il motore di oggi per ciascuna azione e il DEFAULT; poi si ricorda l'ultima scelta dell'utente (user_prefs, chiave `vedit_engine.<azione>`);
- un motore senza pesi installati resta nel menu, marcato «non installato» (la GUI offre «Apri tab Pesi»): MAI download automatico,
  MAI cambio di motore/checkpoint automatico (se il motore scelto non e' installato si rifiuta con messaggio chiaro);
- `weights`: lista di GRUPPI; un gruppo e soddisfatto se ALMENO UNA delle sue voci e presente. Voce = id di catalogo oppure
  «file:<percorso relativo a E:\\AI\\Models>» per i pesi che nel catalogo non hanno un id (es. le LoRA edit-anything).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ── azioni (ordine = ordine nei menu) ─────────────────────────────────────────────────────────
GLOBAL_EDIT = "global_edit"            # cambia tutto il video a parole (stile, luce, ...)
OBJECT_EDIT = "object_edit"            # cambia / rimuovi / aggiungi un oggetto (maschera SAM2 dell'oggetto)
REPLACE_CHARACTER = "replace_character"  # sostituisci il personaggio con quello di una foto
CHANGE_BACKGROUND = "change_background"  # tieni la persona (e il suo movimento), rigenera lo sfondo
ACTIONS: tuple[str, ...] = (GLOBAL_EDIT, OBJECT_EDIT, REPLACE_CHARACTER, CHANGE_BACKGROUND)
ACTION_I18N: dict[str, str] = {a: f"vedit.action.{a}" for a in ACTIONS}

# modalita' del tab (core/video_edit_modes.MODE_SPECS) -> azione
MODE_ACTION: dict[str, str] = {"edit": GLOBAL_EDIT, "inpaint": OBJECT_EDIT, "replace_char": REPLACE_CHARACTER,
                               "change_background": CHANGE_BACKGROUND}

# tipi di motore = quale codice del backend viene chiamato
KIND_LTX_EDIT = "ltx_edit_anything"     # LTX-2.3 + IC-LoRA edit-anything (global)
KIND_LTX_INPAINT = "ltx_inpaint"        # LTX-2.3 + LoRA inpaint masked t2v/r2v (maschera SAM2) — worker «vinpaint»
KIND_LTX_UNION = "ltx_union_canny"      # LTX-2.3 + IC-LoRA union-control-ref0.5 su canny — replace_char storico
KIND_H3_EDIT = "h3_edit"                # MiniMax H3: latente di edit con maschera (workers/minimax_h3_infer.py, modo «edit»)


@dataclass(frozen=True)
class Engine:
    id: str                                         # id stabile (salvato nelle preferenze e nei params del worker)
    kind: str                                       # KIND_*
    label_key: str                                  # etichetta i18n del motore (vedit.engine.<id>)
    label: str                                      # testo di ripiego (italiano)
    actions: tuple[str, ...]                        # azioni che questo motore fa
    weights: tuple[tuple[str, ...], ...] = ()       # gruppi «almeno uno» di voci (id catalogo | file:<rel>)
    release: str = "2.3"                            # LTX: release del checkpoint richiesta; H3: "h3"
    needs_object: bool = False                      # serve il campo «oggetto» (SAM2)
    needs_ref_image: bool = False                   # serve una foto (personaggio / luogo)
    ref_optional: bool = False                      # la foto e opzionale (es. foto del luogo per il cambio sfondo)
    params: dict = field(default_factory=dict)      # parametri propri del motore (default)
    note_key: str = ""

    def supports(self, action: str) -> bool:
        return action in self.actions


_H3_DIT = ("mmh3-ref2va-int8", "mmh3-ref2va-fp8", "mmh3-ref2va-nvfp4", "mmh3-ref2va-int4",
           "mmh3-dit-pinkcherry18", "mmh3-dit-10erosmax-b6", "mmh3-dit-dasiwa-v4")
_H3_TE = ("mmh3-te-nvfp4", "mmh3-te-fp8", "mmh3-te-int8", "mmh3-te-heretic-nvfp4")

ENGINES: dict[str, Engine] = {e.id: e for e in (
    Engine(id="ltx_edit_anything", kind=KIND_LTX_EDIT, label_key="vedit.engine.ltx_edit_anything",
           label="LTX 2.3 · Edit anything (IC-LoRA)", actions=(GLOBAL_EDIT,),
           weights=(("file:LoRA/LTX-2/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors",),)),
    Engine(id="ltx_inpaint", kind=KIND_LTX_INPAINT, label_key="vedit.engine.ltx_inpaint",
           label="LTX 2.3 · Inpaint con maschera (SAM2)", actions=(OBJECT_EDIT,), needs_object=True,
           weights=(("file:LoRA/LTX-2/ltx23_inpaint_masked_t2v_rank128_v1_10000steps.safetensors",),),
           params={"guide": 0.4, "cfg_scale": 6.0, "mask_dilate": 24}),
    Engine(id="ltx_union_canny", kind=KIND_LTX_UNION, label_key="vedit.engine.ltx_union_canny",
           label="LTX 2.3 · Union-control su canny (foto al frame 0)", actions=(REPLACE_CHARACTER,),
           needs_ref_image=True, weights=(("ltx-iclora-union",),)),
    Engine(id="h3_edit", kind=KIND_H3_EDIT, label_key="vedit.engine.h3_edit",
           label="MiniMax H3 · Edit con maschera (video lunghi a clip)",
           actions=(REPLACE_CHARACTER, CHANGE_BACKGROUND), release="h3",   # OBJECT_EDIT con H3: fase 3
           needs_object=True,
           ref_optional=True,
           weights=(_H3_DIT, _H3_TE, ("mmh3-video-vae",), ("mmh3-audio-vae",), ("mmh3-configs",)),
           params={"clip_s": 6.0, "overlap_frames": 39, "grow_px": 16, "feather_px": 2, "headroom": 0.10,
                   "pose_hold_frames": 2, "person_grow_px": 40, "steps": 6, "video_shift": 12.0, "audio_shift": 2.0},
           note_key="vedit.engine.h3_edit.note"),
)}

# motore di OGGI per ciascuna azione (il default: mai cambiato in silenzio)
DEFAULT_ENGINE: dict[str, str] = {GLOBAL_EDIT: "ltx_edit_anything", OBJECT_EDIT: "ltx_inpaint",
                                  REPLACE_CHARACTER: "ltx_union_canny", CHANGE_BACKGROUND: "h3_edit"}


# ── interrogazioni ────────────────────────────────────────────────────────────────────────────
def engines_for(action: str) -> list[Engine]:
    """SOLO i motori che fanno `action`, nell'ordine della tabella (default per primo)."""
    out = [e for e in ENGINES.values() if e.supports(action)]
    d = DEFAULT_ENGINE.get(action)
    out.sort(key=lambda e: (e.id != d,))
    return out


def action_of_mode(mode_key: str) -> str | None:
    return MODE_ACTION.get(mode_key)


def default_engine(action: str) -> str:
    return DEFAULT_ENGINE.get(action) or (engines_for(action)[0].id if engines_for(action) else "")


def remembered(action: str) -> str:
    """Ultima scelta dell'utente per `action` (se ancora valida), altrimenti il default."""
    try:
        from core import user_prefs
        v = str(user_prefs.get(f"vedit_engine.{action}", "") or "")
    except Exception:                                        # noqa: BLE001
        v = ""
    if v and v in ENGINES and ENGINES[v].supports(action):
        return v
    return default_engine(action)


def remember(action: str, engine_id: str) -> None:
    if engine_id in ENGINES and ENGINES[engine_id].supports(action):
        from core import user_prefs
        user_prefs.set(f"vedit_engine.{action}", engine_id)


def _present(item: str) -> bool:
    if item.startswith("file:"):
        from core import model_paths as mp
        return os.path.isfile(os.path.join(str(mp.models_root()), item[5:].replace("/", os.sep)))
    from core import weights_catalog as wc
    e = {x["id"]: x for x in wc.load()}.get(item)
    return bool(e) and wc.is_present(e)


def weight_status(engine_id: str, present=None) -> list[tuple[tuple[str, ...], bool]]:
    """[(gruppo, soddisfatto)] — `present` iniettabile nei test."""
    present = present or _present
    e = ENGINES[engine_id]
    return [(g, any(present(i) for i in g)) for g in e.weights]


def is_installed(engine_id: str, present=None) -> bool:
    return all(ok for _g, ok in weight_status(engine_id, present))


def missing_groups(engine_id: str, present=None) -> list[tuple[str, ...]]:
    return [g for g, ok in weight_status(engine_id, present) if not ok]


def missing_message(engine_id: str, present=None) -> str:
    """Messaggio d'errore chiaro (mai download): cosa manca e dove prenderlo."""
    miss = missing_groups(engine_id, present)
    if not miss:
        return ""
    e = ENGINES[engine_id]
    parts = [" oppure ".join(i[5:] if i.startswith("file:") else i for i in g) for g in miss]
    return (f"Il motore «{e.label}» non e' installato: mancano {'; '.join(parts)}. "
            "Scaricali dal tab «Pesi modelli» (nessun download automatico).")


def engine_params(action: str, engine_id: str) -> dict:
    """Chiavi che la GUI aggiunge ai params del job: il worker/dispatcher sceglie il backend da QUI (corrispondenza widget -> backend)."""
    if engine_id not in ENGINES or not ENGINES[engine_id].supports(action):
        raise ValueError(f"Il motore «{engine_id}» non fa l'azione «{action}».")
    e = ENGINES[engine_id]
    return {"engine": e.id, "engine_kind": e.kind, "engine_action": action, **{f"engine_{k}": v for k, v in e.params.items()}}


def dropdown_items(action: str, present=None) -> list[dict]:
    """Voci del dropdown «Motore»: [{id, label_key, label, installed, default}] — i motori NON installati restano visibili."""
    d = default_engine(action)
    return [{"id": e.id, "label_key": e.label_key, "label": e.label, "installed": is_installed(e.id, present),
             "default": e.id == d, "note_key": e.note_key} for e in engines_for(action)]
