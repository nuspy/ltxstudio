"""Convertitore workflow ComfyUI UI-format → API-format (prompt).

Risolve i nodi virtuali rgthree/easy-use che NON esistono in API:
- SetNode/GetNode (passano valori per NOME): GetNode → sorgente del SetNode omonimo.
- Reroute / Reroute (rgthree): passthrough → sorgente a monte.
- PrimitiveNode: passthrough del widget al consumer.
Produce {id: {class_type, inputs:{name: [src_id,slot] | widget_value}, _meta:{title}}}.

Uso: python scripts/ui2api.py <input_ui.json> <output_api.json>
"""
import json, sys
from pathlib import Path


import re as _re
VIRTUAL = {"Reroute", "Reroute (rgthree)", "PrimitiveNode", "Note", "MarkdownNote",
           "GetNode", "SetNode", "easy getNode", "easy setNode",
           "Fast Groups Bypasser (rgthree)", "Fast Groups Muter (rgthree)",
           "Fast Bypasser (rgthree)", "Anything Everywhere", "Anything Everywhere3",
           "Prompt Enhancer"}


def _is_subgraph(t):
    return bool(t) and bool(_re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-", str(t)))


def convert(wf: dict) -> dict:
    nodes = {n["id"]: n for n in wf["nodes"]}
    # link id -> (from_node, from_slot, to_node, to_slot)
    links = {l[0]: (l[1], l[2], l[3], l[4]) for l in wf.get("links", [])}

    # mappa SetNode: nome -> link di input (la sorgente che il SetNode memorizza)
    set_by_name = {}
    for nid, n in nodes.items():
        t = n.get("type")
        if t in ("SetNode", "easy setNode"):
            name = None
            wv = n.get("widgets_values")
            if isinstance(wv, list) and wv:
                name = wv[0]
            for ip in n.get("inputs", []):
                if ip.get("link") is not None:
                    set_by_name[name] = ip["link"]
                    break

    def node_title(n):
        return (n.get("title") or n.get("type") or "")

    def resolve_link(link_id, _depth=0):
        """Ritorna [src_node_id, src_slot] saltando i nodi virtuali. None se irrisolvibile."""
        if link_id is None or link_id not in links or _depth > 64:
            return None
        fn, fs, _tn, _ts = links[link_id]
        node = nodes.get(fn)
        if node is None:
            return None
        t = node.get("type")
        if t in ("GetNode", "easy getNode"):
            wv = node.get("widgets_values")
            name = wv[0] if isinstance(wv, list) and wv else None
            return resolve_link(set_by_name.get(name), _depth + 1)
        if t in ("Reroute", "Reroute (rgthree)"):
            inp = node.get("inputs", [])
            return resolve_link(inp[0].get("link") if inp else None, _depth + 1)
        if t == "PrimitiveNode":
            # PrimitiveNode propaga il proprio widget; lo trattiamo come valore costante
            # ritornando un marcatore speciale gestito dal chiamante (qui: nessuna sorgente).
            return None
        return [fn, fs]

    def primitive_value(link_id, _depth=0):
        """Se la catena risale a un PrimitiveNode, ritorna (True, value); altrimenti (False, None)."""
        if link_id is None or link_id not in links or _depth > 64:
            return (False, None)
        fn, fs, _tn, _ts = links[link_id]
        node = nodes.get(fn)
        if node is None:
            return (False, None)
        t = node.get("type")
        if t in ("GetNode", "easy getNode"):
            wv = node.get("widgets_values"); name = wv[0] if isinstance(wv, list) and wv else None
            return primitive_value(set_by_name.get(name), _depth + 1)
        if t in ("Reroute", "Reroute (rgthree)"):
            inp = node.get("inputs", [])
            return primitive_value(inp[0].get("link") if inp else None, _depth + 1)
        if t == "PrimitiveNode":
            wv = node.get("widgets_values")
            return (True, wv[0] if isinstance(wv, list) and wv else None)
        return (False, None)

    # ordine dei widget per ogni nodo: i nomi degli input "widget" mancano nel UI-format,
    # quindi usiamo l'object_info? No: in UI-format i widgets_values sono in ordine. Per l'API
    # servono i NOMI. Senza object_info non possiamo mappare i widget→nomi in generale.
    # Strategia: copiamo i widgets_values nell'API sotto chiavi posizionali "_w0,_w1,..." e
    # NON li usiamo come input nominali (verranno settati a mano dai binding dopo la conversione,
    # oppure il nodo li riprende dai default). MA per i nodi che hanno SOLO widget (es. PrimitiveFloat)
    # questo è un problema. Per questo motivo la conversione qui è "strutturale" (link risolti) e i
    # valori widget vanno riassegnati con object_info nel passo successivo (ui2api_fill.py).
    api = {}
    for nid, n in nodes.items():
        t = n.get("type")
        if t in VIRTUAL or _is_subgraph(t):       # subgraph (es. Prompt Enhancer) → bypassato
            continue
        inputs = {}
        for ip in n.get("inputs", []):
            nm = ip.get("name")
            ln = ip.get("link")
            isprim, val = primitive_value(ln)
            if isprim:
                inputs[nm] = val
                continue
            src = resolve_link(ln)
            if src is not None:
                inputs[nm] = src
        api[str(nid)] = {
            "class_type": t,
            "inputs": inputs,
            "_meta": {"title": node_title(n)},
            "_widgets_values": n.get("widgets_values"),   # temporaneo, per il fill dei widget
        }
    return api


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    wf = json.load(open(src, encoding="utf-8"))
    api = convert(wf)
    Path(dst).write_text(json.dumps(api, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"convertiti {len(api)} nodi → {dst}")
