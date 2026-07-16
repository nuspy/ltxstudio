"""Lucky Director — genera un FILM COMPLETO in automatico da un solo prompt (o file), senza interazione.

Crea un normale progetto Cinema Studio (run_mode='auto') → copione (LLM) → breakdown scene/riprese →
keyframe → video → audio → musica → master, con formato/risoluzione/durata e transizioni a due livelli
scelti nella maschera. Tutto sotto il cofano: riusa gli stessi moduli di Cinema Studio
(breakdown, gen_service, auto_run, montage_render). Il progetto risultante è riapribile e modificabile.

`make_film(spec, on_progress, cancel)` è headless (nessun Qt); la maschera UI vive in
ui/film_studio/lucky_director_dialog.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from core import llm_service as L
from core.film_studio import db, breakdown, auto_run

_SCRIPT_SYS = (
    "You are a screenwriter. Turn the user's idea into a concise, shootable SCREENPLAY. "
    "Write scene headings, action lines and dialogue. Introduce recurring characters and refer to them "
    "consistently with @Name tags (e.g. @Maria) so they stay visually consistent across scenes. "
    "Keep it focused and paced for the requested total duration. Output ONLY the screenplay text."
)

# formati (aspect) e risoluzioni (lato corto) esposti dalla maschera — riusano le convenzioni del montaggio.
ASPECTS = ["16:9", "9:16", "2.39:1", "3:2", "2:3", "1:1", "4:3", "3:4"]
RESOLUTIONS = [("360p", 360), ("480p", 480), ("720p", 720), ("1080p", 1080)]
_ASPECT_R = {"16:9": 16 / 9, "9:16": 9 / 16, "2.39:1": 2.39, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4,
             "3:2": 3 / 2, "2:3": 2 / 3}   # 3:2 (foto classica orizzontale) + 2:3 (verticale)


def resolution_wh(short_side: int, aspect: str) -> tuple[int, int]:
    """(w,h) dalla risoluzione (lato corto in px) e dall'aspetto, arrotondati a multipli di 32 (richiesto da LTX)."""
    r = _ASPECT_R.get(aspect, 16 / 9)
    if r >= 1:                                               # orizzontale/quadrato: il lato corto è l'altezza
        h, w = short_side, round(short_side * r)
    else:                                                   # verticale: il lato corto è la larghezza
        w, h = short_side, round(short_side / r)
    q = lambda x: max(32, int(round(x / 32)) * 32)          # noqa: E731
    return q(w), q(h)


# ─────────────── lettura sorgente (prompt o file) ───────────────
def read_source_file(path: str) -> str:
    """Estrae il testo da .txt/.md/.fountain (diretto), .pdf/.docx (se una libreria è disponibile).
    Solleva con messaggio chiaro se il formato non è leggibile."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".txt", ".md", ".fountain", ".fdx", ""):
        return p.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        for mod in ("pypdf", "PyPDF2"):
            try:
                m = __import__(mod)
                reader = m.PdfReader(str(p))
                return "\n".join((pg.extract_text() or "") for pg in reader.pages)
            except Exception:                                # noqa: BLE001
                continue
        try:
            import pdfplumber
            with pdfplumber.open(str(p)) as pdf:
                return "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        except Exception:                                    # noqa: BLE001
            raise RuntimeError("Per leggere i PDF installa 'pypdf' o 'pdfplumber'.")
    if ext == ".docx":
        try:
            import docx
            return "\n".join(par.text for par in docx.Document(str(p)).paragraphs)
        except Exception:                                    # noqa: BLE001
            raise RuntimeError("Per leggere i DOCX installa 'python-docx'.")
    # fallback: prova come testo
    return p.read_text(encoding="utf-8", errors="ignore")


def source_text(spec: dict) -> str:
    """Testo di partenza dalla spec: prompt diretto oppure contenuto del file caricato."""
    fp = (spec.get("file") or "").strip()
    if fp:
        return read_source_file(fp).strip()
    return (spec.get("prompt") or "").strip()


# ─────────────── pipeline completa ───────────────
def make_film(spec: dict, *, on_progress=None, cancel=None) -> dict:
    """Genera l'intero film dalla spec. spec keys:
        prompt|file, name?, duration_s, aspect, resolution (short side px), consistency (fast|lora|mixed),
        scene_trans, shot_trans, scene_dur, shot_dur, fps?, make_music?
    on_progress(phase, index, total, prompt, pct); cancel() -> bool. Ritorna {'project_id','master'}."""
    def emit(phase, idx, tot, prompt, pct):
        if on_progress:
            try:
                on_progress(phase, idx, tot, prompt, pct)
            except Exception:                                # noqa: BLE001
                pass

    text = source_text(spec)
    if not text:
        raise RuntimeError("Nessun prompt o file da cui partire")
    duration_s = float(spec.get("duration_s") or 60.0)
    aspect = spec.get("aspect") or "16:9"
    short = int(spec.get("resolution") or 480)
    width, height = resolution_wh(short, aspect)
    consistency = spec.get("consistency") or "fast"
    name = (spec.get("name") or "").strip() or f"Lucky Director {time.strftime('%Y-%m-%d %H:%M', time.localtime())}"

    # progetto (auto, riapribile in Cinema Studio)
    cfg = {"img_arch": "flux", "img_model": "", "vid_gguf": "",
           "vid_arch": spec.get("vid_arch", "ltx"),          # 2ª architettura video (LTX/LingBot)
           "forbidden": "", "required": "",
           "style_bible": {}, "lucky_director": True}
    pid = db.insert("project", name=name, kind="film", duration_s=duration_s, genres="", styles="",
                    path_mode=consistency, run_mode="auto", config=json.dumps(cfg, ensure_ascii=False),
                    updated=time.time())

    # FASE 0: copione
    emit("script", 1, 1, text[:120], 0.01)
    if cancel and cancel():
        raise RuntimeError("Operazione annullata")
    script = L.complete(_SCRIPT_SYS, f"Target total duration ~{int(duration_s)}s.\n\nIDEA:\n{text}",
                        max_tokens=2000)
    db.save_script(pid, script, kind="script")

    # FASE 1: breakdown scene/riprese + distribuzione durata
    emit("breakdown", 1, 1, "", 0.03)
    if cancel and cancel():
        raise RuntimeError("Operazione annullata")
    breakdown.build(pid)
    breakdown.normalize_durations(pid, duration_s)

    # FASI 2-6: keyframe → video → audio → musica → master (orchestratore condiviso con Cinema Studio)
    res = auto_run.run(pid, aspect=aspect, scene_trans=spec.get("scene_trans", ""),
                       shot_trans=spec.get("shot_trans", ""),
                       scene_dur=float(spec.get("scene_dur", 0.8)), shot_dur=float(spec.get("shot_dur", 0.5)),
                       fps=float(spec.get("fps", 25.0)), make_music=bool(spec.get("make_music", True)),
                       width=width, height=height, on_progress=on_progress, cancel=cancel)
    return {"project_id": pid, "master": res.get("master", "")}
