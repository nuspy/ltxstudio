"""Studio «🎭 Scambia» — modello dati e pianificazione, modulo PURO (niente Qt, niente torch, niente GPU): testabile su CPU.

Flusso a passi (PROPOSTA_openjutsu §B): 1 video -> 2 «Trova persone» (tracce con maschere per frame, miniatura, nome) -> 3 ruoli (1-4 personaggi:
ogni ruolo = una o piu tracce + la foto del nuovo personaggio + una nota) -> 4 anteprima della maschera -> 5 genera (UNA PASSATA H3 PER RUOLO, in
catena: l'uscita di una passata e il video sorgente della successiva, le maschere restano quelle del video originale perche la durata e i frame non
cambiano) -> 6 rifiniture (volti/colori in «Migliora qualita -> Video»).

La ricerca delle persone la fa il backend (SAM2 + GroundingDINO multi-box oggi; SAM 3.1 quando il caricatore e pronto): qui si orchestra con funzioni
INIETTATE (`detect_boxes`, `track_box`), cosi la logica si prova senza GPU.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_ROLES = 4
IOU_DUP = 0.6


class SwapError(RuntimeError):
    pass


@dataclass
class Track:
    id: int
    box: list[float]                  # riquadro sul frame in cui e stato trovato [x0, y0, x1, y1]
    score: float = 0.0
    label: str = ""                   # nome dato dal VLM o dall'utente («uomo con occhiali», «Persona 2»…)
    mask_dir: str = ""                # cartella delle maschere per frame (PNG 0/255), stessi numeri dei frame del video
    thumb: str = ""                   # miniatura (ritaglio sul frame in cui la persona e piu grande)
    frames: int = 0
    area: float = 0.0                 # area media della maschera (quota del frame): ordina «principale» prima


@dataclass
class Role:
    name: str                         # «Personaggio 1»…
    track_ids: list[int]
    photo: str = ""                   # foto del nuovo personaggio (obbligatoria)
    note: str = ""                    # nota libera per il prompt (opzionale)


@dataclass
class Pass:
    role: str
    mask_dir: str
    photo: str
    note: str
    index: int = 0


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b[:4]
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def dedupe_boxes(boxes: list[list[float]], thresh: float = IOU_DUP, limit: int = MAX_ROLES) -> list[list[float]]:
    """Tiene i riquadri a punteggio piu alto; scarta quelli che si sovrappongono di piu di `thresh` a uno gia tenuto (la stessa persona rilevata due volte)."""
    keep: list[list[float]] = []
    for b in sorted(boxes, key=lambda x: -(x[4] if len(x) > 4 else 0.0)):
        if all(iou(b, k) < thresh for k in keep):
            keep.append(list(b))
        if len(keep) >= limit:
            break
    return keep


def find_people(video: str, work_dir: str, *, detect_boxes, track_box, make_thumb=None, label_of=None, log=None, max_people: int = MAX_ROLES) -> list[Track]:
    """Trova le persone: `detect_boxes(video)` -> [[x0,y0,x1,y1,score], ...] sul frame di riferimento; ogni riquadro (senza doppioni) viene tracciato con
    `track_box(video, box, mask_dir)` -> {'frames': n, 'area': quota media} e le maschere finiscono in work_dir/track_<k>/masks. Le tracce escono
    ordinate dalla piu grande (la «principale»). `label_of(track)` (VLM) e `make_thumb(track)` sono opzionali; senza, «Persona k»."""
    say = log or (lambda m: None)
    boxes = dedupe_boxes(list(detect_boxes(video)), limit=max_people)
    if not boxes:
        raise SwapError("Nessuna persona trovata nel video. Prova un altro video o scrivi cosa cercare.")
    tracks: list[Track] = []
    for k, b in enumerate(boxes, 1):
        md = os.path.join(work_dir, f"track_{k}", "masks")
        say(f"[scambia] traccio la persona {k}/{len(boxes)}…")
        info = track_box(video, b, md) or {}
        tracks.append(Track(id=k, box=[float(x) for x in b[:4]], score=float(b[4]) if len(b) > 4 else 0.0, mask_dir=md,
                            frames=int(info.get("frames", 0)), area=float(info.get("area", 0.0))))
    tracks.sort(key=lambda t: -t.area)
    for new_id, t in enumerate(tracks, 1):                 # ids ricalcolati dopo l'ordinamento: «Persona 1» = la piu grande
        t.id = new_id
    for t in tracks:
        t.label = (label_of(t) if label_of else "") or f"Persona {t.id}"
        if make_thumb:
            t.thumb = make_thumb(t) or ""
    return tracks


def save_tracks(tracks: list[Track], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump([asdict(t) for t in tracks], open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def load_tracks(path: str) -> list[Track]:
    return [Track(**d) for d in json.load(open(path, encoding="utf-8"))]


def validate(roles: list[Role], tracks: list[Track]) -> list[str]:
    """Errori (testo per l'utente) prima di generare. Lista vuota = si puo partire."""
    errs: list[str] = []
    ids = {t.id for t in tracks}
    if not roles:
        return ["Assegna almeno un personaggio: scegli una persona del video e la foto del nuovo personaggio."]
    if len(roles) > MAX_ROLES:
        errs.append(f"Al massimo {MAX_ROLES} personaggi per volta.")
    used: dict[int, str] = {}
    for r in roles:
        if not r.track_ids:
            errs.append(f"{r.name}: scegli quale persona del video sostituisce.")
        for tid in r.track_ids:
            if tid not in ids:
                errs.append(f"{r.name}: la persona {tid} non esiste.")
            elif tid in used:
                errs.append(f"La persona {tid} e assegnata sia a «{used[tid]}» sia a «{r.name}».")
            else:
                used[tid] = r.name
        if not r.photo:
            errs.append(f"{r.name}: manca la foto del nuovo personaggio.")
        elif not os.path.isfile(r.photo):
            errs.append(f"{r.name}: la foto non esiste ({r.photo}).")
    return errs


def merge_track_masks(mask_dirs: list[str], out_dir: str) -> str:
    """Unione delle maschere di piu tracce (stesso ruolo) frame per frame, in `out_dir`. Con una traccia sola ritorna la sua cartella (nessuna copia)."""
    if len(mask_dirs) == 1:
        return mask_dirs[0]
    import numpy as np
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    names = sorted({f for d in mask_dirs for f in os.listdir(d) if f.lower().endswith(".png")})
    for n in names:
        acc = None
        for d in mask_dirs:
            p = os.path.join(d, n)
            if not os.path.isfile(p):
                continue
            m = np.asarray(Image.open(p).convert("L")) > 127
            acc = m if acc is None else (acc | m)
        if acc is not None:
            Image.fromarray((acc * 255).astype("uint8")).save(os.path.join(out_dir, n))
    return out_dir


def plan_passes(roles: list[Role], tracks: list[Track], work_dir: str) -> list[Pass]:
    """Una passata per ruolo, nell'ordine dei ruoli. Errore se `validate` trova problemi."""
    errs = validate(roles, tracks)
    if errs:
        raise SwapError("\n".join(errs))
    by_id = {t.id: t for t in tracks}
    out: list[Pass] = []
    for i, r in enumerate(roles, 1):
        md = merge_track_masks([by_id[t].mask_dir for t in r.track_ids], os.path.join(work_dir, f"role_{i}_masks"))
        out.append(Pass(role=r.name, mask_dir=md, photo=r.photo, note=r.note, index=i))
    return out


def source_for_pass(index: int, video: str, outputs: list[str]) -> str:
    """Video sorgente della passata `index` (0-based): la prima parte dal video originale, le altre dall'uscita della passata prima (catena)."""
    return video if index == 0 else outputs[index - 1]
