"""Riempie i valori-widget dei nodi convertiti (ui2api.py) usando object_info di ComfyUI,
mappando i `_widgets_values` posizionali ai NOMI degli input. Gestisce il caso
`control_after_generate` (seed/noise_seed → un valore widget extra da saltare).
Poi VALIDA la struttura: class_type esistenti, input richiesti presenti, link non dangling.

Uso: python scripts/ui2api_fill.py <api_in.json> <api_out.json>
"""
import json, sys, urllib.request, os
from pathlib import Path

# object_info: prima prova la cache locale (.oi4.json), poi ComfyUI live.
_CACHE = Path(__file__).resolve().parent.parent / ".oi4.json"
if _CACHE.exists():
    OI = json.load(open(_CACHE, encoding="utf-8"))
else:
    OI = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=30))

# tipi che NON sono widget (sono solo link): gli altri scalari sono widget
LINK_TYPES = {"MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK", "SIGMAS",
              "SAMPLER", "GUIDER", "NOISE", "AUDIO", "CONTROL_NET", "UPSCALE_MODEL", "*",
              "FACE_ANALYSIS", "FACEPARSING_MODEL", "FACEPARSING_RESULT", "BBOX_DETECTOR",
              "SEGS", "SAM_MODEL", "GROUNDING_DINO_MODEL", "INSIGHTFACE", "PHOTOMAKER",
              "WEBCAM", "FLOATS", "INTS", "STRINGS"}


def widget_input_order(class_type):
    """Lista (name, is_widget, has_control) degli input di un nodo, in ordine, da object_info."""
    spec = OI.get(class_type)
    if not spec:
        return None
    inp = spec.get("input", {})
    order = []
    for grp in ("required", "optional"):
        for name, decl in inp.get(grp, {}).items():
            t = decl[0] if isinstance(decl, list) else decl
            opts = decl[1] if isinstance(decl, list) and len(decl) > 1 and isinstance(decl[1], dict) else {}
            is_list = isinstance(t, list)               # COMBO
            is_widget = is_list or (isinstance(t, str) and t not in LINK_TYPES)
            # control_after_generate: INT/COMBO seed-like con questa opzione → +1 valore widget
            has_ctrl = bool(opts.get("control_after_generate")) or name in ("seed", "noise_seed")
            order.append((name, is_widget, has_ctrl, t))
    return order


def fill(api):
    warnings = []
    for nid, node in api.items():
        ct = node["class_type"]
        order = widget_input_order(ct)
        wv = node.pop("_widgets_values", None)
        if order is None:
            warnings.append(f"[{nid}] class_type SCONOSCIUTO: {ct}")
            continue
        if not isinstance(wv, list):
            continue
        # Regola ComfyUI: gli input CONVERTITI A LINK sono RIMOSSI da widgets_values. Quindi
        # consumiamo un valore SOLO per i widget non-link, in ordine object_info.
        i = 0
        for name, is_widget, has_ctrl, t in order:
            if not is_widget or name in node["inputs"]:
                continue
            if i < len(wv):
                node["inputs"][name] = wv[i]
                i += 1
                if has_ctrl and i < len(wv) and isinstance(wv[i], str) and wv[i] in (
                        "fixed", "increment", "decrement", "randomize"):
                    i += 1
    return api, warnings


def validate(api):
    errs = []
    present = set(api)
    for nid, node in api.items():
        ct = node["class_type"]
        spec = OI.get(ct)
        if not spec:
            errs.append(f"[{nid}] class_type inesistente in ComfyUI: {ct}")
            continue
        req = spec.get("input", {}).get("required", {})
        for name in req:
            if name not in node["inputs"]:
                errs.append(f"[{nid}] {ct}: input richiesto MANCANTE '{name}'")
        for name, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)):
                if str(v[0]) not in present:
                    errs.append(f"[{nid}] {ct}.{name}: link a nodo INESISTENTE {v[0]}")
    return errs


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    api = json.load(open(src, encoding="utf-8"))
    api, warns = fill(api)
    errs = validate(api)
    Path(dst).write_text(json.dumps(api, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {dst} ({len(api)} nodi)")
    for w in warns:
        print("  WARN", w)
    print(f"=== VALIDAZIONE: {len(errs)} errori ===")
    for e in errs[:40]:
        print("  ", e)
