"""Riquadri (bounding box) e caption di FLUX.3 Image Klein — modulo PURO (stdlib, nessun Qt/torch), testabile con pytest.

Formato verificato sul codice ufficiale BFL (src/flux3/bbox.py + pup.py, Apache-2.0; README HF/Comfy; docs.bfl.ml):
  · un riquadro è [y_min, x_min, y_max, x_max] con origine in alto a sinistra; l'app e l'API usano 0–1, il modello
    vuole la griglia 0–1000 (×1000, arrotondato);
  · in coda al caption va un array JSON di righe; il caption cita ogni riquadro come `<id>` (es. `<balloon_1>`);
  · layout (t2i):  {"id","bbox","desc"};  edit con riferimenti: {"id","from":"ref_image_N"|"target","src_bbox",
    "tgt_bbox","desc"}: NUOVO = from «target» solo tgt_bbox · SPOSTA = src≠tgt · ANCORA (resta) = src=tgt ·
    RIMUOVI = tgt_bbox null. «I riquadri fissano posizione e scala, non sono una maschera rigida»;
  · con i riquadri l'upsampler lavora in modalità `t2i_bbox`/`i2i_bbox`: riceve il prompt + i riquadri BLOCCATI e
    la sua risposta viene riconciliata (`reconcile`) perché i riquadri dell'utente restano com'erano.
Questo è il NOSTRO codice (riscritto dalla specifica, non copiato). `from` usa l'indice ZERO-based dei riferimenti
(`ref_image_0`…), l'app invece nomina le foto `@Foto_N` (core/image_refs): la conversione è `ref_name(k)`.
"""
from __future__ import annotations

import json
import math
import re

from core import flux3_spec as S

ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def ordinal(k: int) -> str:
    """«first»…«tenth» per il k-esimo riferimento (1-based)."""
    return ORDINALS[k - 1] if 1 <= k <= len(ORDINALS) else f"{k}th"


def ref_name(k: int) -> str:
    """Nome che il modello dà al k-esimo riferimento (k 1-based, come l'ordine di invio): ref_image_{k-1}."""
    return f"ref_image_{int(k) - 1}"


# ── validazione e conversione dei riquadri dell'utente (0–1) ─────────────────────────────────────────────
def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _box01(b):
    """[y1,x1,y2,x2] in 0–1 valido → lista di float, altrimenti None."""
    if not isinstance(b, (list, tuple)) or len(b) != 4 or not all(_finite(v) for v in b):
        return None
    y1, x1, y2, x2 = (float(v) for v in b)
    if not (0.0 <= y1 < y2 <= 1.0 and 0.0 <= x1 < x2 <= 1.0):
        return None
    return [y1, x1, y2, x2]


def validate(boxes, n_refs: int = 0) -> list[str]:
    """Errori leggibili sui riquadri (lista vuota = ok). n_refs > 0 = modalità edit (campi from/src/tgt)."""
    errs: list[str] = []
    if not boxes:
        return errs
    if len(boxes) > S.MAX_BOXES:
        errs.append(f"al massimo {S.MAX_BOXES} riquadri")
    seen: set[str] = set()
    for i, b in enumerate(boxes):
        if not isinstance(b, dict):
            errs.append(f"riquadro {i + 1}: formato non valido")
            continue
        rid = str(b.get("id") or "")
        if not ID_RE.fullmatch(rid):
            errs.append(f"riquadro {i + 1}: id «{rid}» non valido (lettera iniziale, poi lettere/cifre/_)")
        elif rid in seen:
            errs.append(f"riquadro {i + 1}: id «{rid}» duplicato")
        seen.add(rid)
        mode = str(b.get("mode") or ("new" if n_refs else "layout"))
        if not n_refs:
            if _box01(b.get("bbox")) is None:
                errs.append(f"riquadro {rid or i + 1}: coordinate non valide (servono 4 valori 0–1, max > min)")
            if not str(b.get("desc") or "").strip():
                errs.append(f"riquadro {rid or i + 1}: manca la descrizione")
            continue
        if mode not in S.BOX_MODES:
            errs.append(f"riquadro {rid or i + 1}: modalità «{mode}» sconosciuta ({', '.join(S.BOX_MODES)})")
            continue
        if mode != "remove" and not str(b.get("desc") or "").strip():
            errs.append(f"riquadro {rid or i + 1}: manca la descrizione")
        if mode == "new":
            if _box01(b.get("tgt_bbox")) is None:
                errs.append(f"riquadro {rid or i + 1}: serve la zona di destinazione")
        else:
            k = b.get("ref")
            if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n_refs:
                errs.append(f"riquadro {rid or i + 1}: riferimento di origine non valido (1–{n_refs})")
            if _box01(b.get("src_bbox")) is None:
                errs.append(f"riquadro {rid or i + 1}: serve la zona di origine")
            if mode in ("move", "anchor") and _box01(b.get("tgt_bbox")) is None:
                errs.append(f"riquadro {rid or i + 1}: serve la zona di destinazione")
            if mode == "move" and _box01(b.get("src_bbox")) and b.get("src_bbox") == b.get("tgt_bbox"):
                errs.append(f"riquadro {rid or i + 1}: «sposta» con origine = destinazione è «ancora»")
    return errs


def _grid(box):
    if not box:
        return None
    return [round(max(0.0, min(1.0, float(v))) * S.BOX_SCALE) for v in box]


def to_rows(boxes) -> list[dict]:
    """Riquadri dell'utente (0–1) → righe per l'upsampler/DiT (griglia 0–1000). `validate` va chiamata prima."""
    rows = []
    for i, b in enumerate(boxes or []):
        rid = str(b.get("id") or f"obj_{i + 1}")
        desc = str(b.get("desc") or "")
        if b.get("bbox"):                                     # layout t2i
            rows.append({"id": rid, "bbox": _grid(b["bbox"]), "desc": desc})
            continue
        mode = str(b.get("mode") or "new")
        if mode == "new":
            rows.append({"id": rid, "from": "target", "src_bbox": None, "tgt_bbox": _grid(b.get("tgt_bbox")),
                         "desc": desc})
        elif mode == "remove":
            rows.append({"id": rid, "from": ref_name(b["ref"]), "src_bbox": _grid(b.get("src_bbox")),
                         "tgt_bbox": None, "desc": desc})
        else:                                                 # move / anchor
            tgt = b.get("src_bbox") if mode == "anchor" and not b.get("tgt_bbox") else b.get("tgt_bbox")
            rows.append({"id": rid, "from": ref_name(b["ref"]), "src_bbox": _grid(b.get("src_bbox")),
                         "tgt_bbox": _grid(tgt), "desc": desc})
    return rows


# ── upsampler: piano della richiesta ─────────────────────────────────────────────────────────────────────
def upsampling_input(prompt: str, rows: list[dict]) -> str:
    return (f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n"
            "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at the "
            "top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
            f"provenance value, null, and coordinate exactly:\n{json.dumps(rows, ensure_ascii=False)}")


def plan(prompt: str, rows: list[dict] | None, n_images: int) -> tuple[str, str, list[dict] | None]:
    """(chiave del system prompt, testo utente, righe bloccate): t2i | i2i, oppure *_bbox con i riquadri."""
    task = "i2i" if n_images else "t2i"
    if not rows:
        return task, prompt, None
    return f"{task}_bbox", upsampling_input(prompt, rows), rows


I2I_SCAFFOLD = {
    False: ("Reference images follow in order (the first image, the second image, ...).\n"
            "User request: {instruction}\nOutput the caption."),
    True: ("Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
           "User editing instruction: {instruction}\n"
           "Decide the edit, imagine the target image, and output the annotation record."),
}


def user_message(prompt: str, rows: list[dict] | None, n_images: int) -> tuple[str, str]:
    """(chiave system prompt, testo del turno utente) — con riferimenti il testo è avvolto nello scaffold i2i."""
    key, text, locked = plan(prompt, rows, n_images)
    if n_images:
        text = I2I_SCAFFOLD[locked is not None].format(instruction=text)
    return key, text


# ── lettura di una risposta dell'upsampler ───────────────────────────────────────────────────────────────
def split(text: str) -> tuple[str, list | None]:
    """(caption, righe): l'ultima «[» da cui il resto è una lista JSON. Non una regex: le desc possono avere parentesi."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


def valid_box(box):
    """Riquadro sulla griglia 0–1000 valido → lista, altrimenti None. max <= 1 = coordinate 0–1, non griglia."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf") for v in box):
        return None
    y1, x1, y2, x2 = box
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > S.BOX_SCALE or x2 > S.BOX_SCALE or max(box) <= 1:
        return None
    return list(box)


def _row_ok(r) -> bool:
    return (isinstance(r, dict) and isinstance(r.get("id"), str) and ID_RE.fullmatch(r["id"]) is not None
            and any(valid_box(r.get(k)) for k in ("bbox", "src_bbox", "tgt_bbox")))


def detect(prompt: str):
    """(prompt, righe) se il prompt porta già i riquadri sulla griglia 0–1000 («<caption> [righe]»), altrimenti None."""
    text, rows = split(prompt)
    if not rows or not all(_row_ok(r) for r in rows):
        return None
    m = re.fullmatch(r"ORIGINAL PROMPT:\n(.*?)\n\nLOCKED BOUNDING-BOX ANNOTATIONS .*", text, re.S)
    return (m.group(1) if m else text), rows


def is_finished(prompt: str) -> bool:
    """Risposta già completa dell'upsampler: ogni riquadro è citato come <id> nel caption → va al DiT com'è."""
    found = detect(prompt)
    return found is not None and all(f"<{r['id']}>" in found[0] for r in found[1])


def iou(a: list, b: list) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _valid_added(row, caption: str):
    """Riga aggiunta dall'upsampler: tenuta solo se ben formata e citata come <id> nel caption."""
    if not isinstance(row, dict):
        return None
    rid, desc = row.get("id"), row.get("desc")
    if (not isinstance(rid, str) or not ID_RE.fullmatch(rid) or f"<{rid}>" not in caption
            or not isinstance(desc, str) or not desc.strip()):
        return None
    box = valid_box(row.get("bbox"))
    if box is not None:
        return {"id": rid, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = valid_box(src) if src is not None else None
    tgt_box = valid_box(tgt) if tgt is not None else None
    if (not isinstance(origin, str) or not re.fullmatch(r"ref_image_\d+|target", origin)
            or (src is not None and src_box is None) or (tgt is not None and tgt_box is None)
            or (src_box is None and tgt_box is None) or (origin == "target") == (src_box is not None)):
        return None
    return {"id": rid, "from": origin, "src_bbox": src_box, "tgt_bbox": tgt_box, "desc": desc.strip()}


def reconcile(text: str, locked: list[dict]) -> str:
    """Risposta dell'upsampler con i riquadri dell'utente rimessi com'erano:
    1. una riga che ripianifica un riquadro bloccato con un id suo (IoU ≥ 0,9) prende l'id bloccato, nelle righe e
       nelle citazioni <id> del caption;
    2. le righe bloccate restano verbatim (adottano la desc dell'upsampler se c'è); seguono le sue righe valide."""
    caption, rows = split(text)
    rows = rows or []
    locked_ids = {r["id"] for r in locked}
    row_ids = {r.get("id") for r in rows if isinstance(r, dict)}
    renamed: dict[str, str] = {}
    for lock in locked:
        lock_box = lock.get("bbox") or lock.get("tgt_bbox") or lock.get("src_bbox")
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") in renamed or row.get("id") in locked_ids:
                continue
            box = valid_box(row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox"))
            if box is not None and iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")
    by_id = {r.get("id"): r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)}
    out = []
    for lock in locked:
        d = by_id.get(lock["id"], {}).get("desc")
        out.append({**lock, "desc": d.strip() if isinstance(d, str) and d.strip() else lock.get("desc", "")})
    for row in rows:
        added = _valid_added(row, caption)
        if added is not None and added["id"] not in {r["id"] for r in out}:
            out.append(added)
    return f"{caption} {json.dumps(out, ensure_ascii=False)}"


def caption_without_upsampler(prompt: str, rows: list[dict] | None) -> str:
    """Senza upsampler il prompt dell'utente va al DiT con la tabella dei riquadri in coda (formato del DiT)."""
    if not rows:
        return prompt
    return f"{prompt.rstrip()} {json.dumps(rows, ensure_ascii=False)}"


# ── riferimenti: dimensioni (stesse regole del codice ufficiale) ─────────────────────────────────────────
def _ceil16(v: int) -> int:
    return -(-v // S.GRID) * S.GRID


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(dimensione adattata, canvas) del riferimento SINGOLO: canvas = adattata arrotondata per eccesso a 16, ≤ 4 MiPixel,
    lati 64–4096."""
    scale = min(1.0, math.sqrt(S.FIRST_REF_MAX_PIXELS / (width * height)), S.MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(S.MAX_SIDE, max(S.MIN_SIDE, _ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= S.FIRST_REF_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def multi_reference_size(width: int, height: int) -> tuple[int, int]:
    """Dimensione di ogni riferimento di un edit MULTIPLO: ≤ 1 MP (lati arrotondati per difetto), ritaglio centrato sul
    multiplo di 16. Ritorna (w, h) dell'immagine ridimensionata prima del ritaglio e la forma finale è (w//16·16, h//16·16)."""
    if width * height > S.MULTI_REF_MAX_PIXELS:
        s = math.sqrt(S.MULTI_REF_MAX_PIXELS / (width * height))
        width, height = int(width * s), int(height * s)
    return width, height


def pup_reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    """Dimensione di un riferimento passato all'UPSAMPLER (multipli di 32, budget di pixel secondo n_refs)."""
    if max(width, height) / min(width, height) > S.PUP_MAX_ASPECT:
        raise ValueError(f"rapporto {width}x{height} oltre {S.PUP_MAX_ASPECT}:1")
    m = S.PUP_REF_MULTIPLE
    floor = lambda v: max(m, int(math.floor(v / m)) * m)
    cap = S.PUP_MAX_PIXELS[max(1, min(n_refs, S.MAX_REFS))]
    w, h = max(m, round(width / m) * m), max(m, round(height / m) * m)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == m:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    return w, h
