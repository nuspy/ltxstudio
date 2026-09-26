"""Avatar parlante con MiniMax H3 (2° motore del tab Avatar e di /avatar/generate).

H3 genera voce e video INSIEME: niente TTS a parte, le battute vanno nel prompt (<d>[Lingua] …</d>).
  · FL2VA  = foto come PRIMO FOTOGRAMMA + battute → la persona della foto parla (voce inventata dal modello);
  · Ref2VA = foto come RIFERIMENTO + registrazione della voce come riferimento di TIMBRO (<Audio 1>) → parla
             con quella voce. Serve il checkpoint Ref2VA (scelta di modalità, non di modello).
Durata 5–15 s (griglia 17n+5 a 24 fps). Stesso worker del tab Video (workers/minimax_h3_infer.py).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from core import minimax_h3_spec as S

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output" / "avatar"


def _out(prefix: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    return str(OUT / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4")


def build_params(cfg: dict) -> dict:
    """cfg (GUI/API, chiavi piatte) → params del worker H3.
    cfg: image_path, prompt, lines=[{speaker, language, text}], voice_ref_path, duration_s, h3_profile, h3_te,
    h3_turbo, h3_canvas, soundscape, music, seed, output."""
    img = cfg.get("image_path") or ""
    voice = cfg.get("voice_ref_path") or ""
    p = {
        "arch": "minimax_h3", "prompt": cfg.get("prompt") or "",
        "h3_profile": cfg.get("h3_profile") or "nvfp4", "h3_te": cfg.get("h3_te") or "",
        "h3_turbo": str(cfg.get("h3_turbo") or ""), "h3_canvas": str(cfg.get("h3_canvas") or "768"),
        "h3_dialogue": [d for d in (cfg.get("lines") or []) if (d.get("text") or "").strip()],
        "h3_soundscape": cfg.get("soundscape") or "", "h3_music": cfg.get("music") or "",
        "duration_s": float(cfg.get("duration_s") or 8.0),
        "seed": int(cfg["seed"]) if int(cfg.get("seed", -1)) >= 0 else int(time.time()) % 2**31,
        "generate_audio": True, "output": cfg.get("output") or _out("avatar_h3"),
    }
    if voice:
        # timbro della voce → Ref2VA: la foto diventa il riferimento del soggetto (<Picture 1>)
        refs = []
        if img:
            refs.append({"path": img, "kind": "image", "token": "<Picture 1>", "desc": "",
                         "role": "the person who speaks"})
        refs.append({"path": voice, "kind": "audio", "token": "<Audio 1>", "desc": "",
                     "role": "voice timbre of the speaker"})
        p["references"] = refs
        if img and "<Picture 1>" not in p["prompt"]:
            p["prompt"] = f"The person in <Picture 1> talks to the camera, with the voice of <Audio 1>. {p['prompt']}".strip()
    elif img:
        p["image_path"] = img
    if not p["h3_te"]:
        p.pop("h3_te")
    p["num_frames"] = S.snap_frames(p["duration_s"])
    return p


def preflight(p: dict) -> list[str]:
    from core import model_paths as mp
    from core import weights_catalog as wc
    errs = []
    try:
        ids = S.weight_ids(p)
    except ValueError as e:
        return [str(e)]
    miss = [i for i in dict.fromkeys(ids.values())
            if not (mp.entry(i) and wc.is_present(mp.entry(i), str(mp.models_root())))]
    if miss:
        errs.append("Pesi MiniMax H3 mancanti: " + ", ".join(miss) + " — scaricali dal tab «Pesi modelli».")
    if not (p.get("h3_dialogue") or p.get("prompt")):
        errs.append("Serve almeno una battuta o un prompt.")
    from core import h3_prompt
    for d in p.get("h3_dialogue") or []:
        try:
            h3_prompt._line_sentence(d, 1)
        except ValueError as e:
            errs.append(str(e))
    for r in p.get("references") or []:
        if not os.path.isfile(r.get("path") or ""):
            errs.append(f"File di riferimento inesistente: {r.get('path')}")
    if p.get("image_path") and not os.path.isfile(p["image_path"]):
        errs.append(f"Immagine iniziale inesistente: {p['image_path']}")
    if p.get("references") and not any(r.get("kind") != "audio" for r in p["references"]):
        errs.append("Con la voce di riferimento serve anche una foto (H3 non accetta solo audio).")
    return errs


def make_job(params: dict, **cbs):
    """Job subprocess (non avviato) nel venv diffusers_next, come gli altri motori isolati."""
    from core.ltx_video_job import MINIMAX_H3_PY, MINIMAX_H3_WORKER
    from core.music_service import Job
    p = dict(params)
    p["parent_pid"] = os.getpid()
    OUT.mkdir(parents=True, exist_ok=True)
    pj = OUT / f"_params_h3_{int(time.time() * 1000)}.json"
    pj.write_text(json.dumps(p, ensure_ascii=False, indent=1), "utf-8")
    env = {"PYTHONPATH": None, "PYTHONHOME": None, "PYTHONSTARTUP": None, "PYTHONIOENCODING": "utf-8"}
    job = Job([MINIMAX_H3_PY, "-u", MINIMAX_H3_WORKER, str(pj)], REPO, env=env, **cbs)
    job.output = p["output"]
    job.params_file = str(pj)
    return job


def run(req: dict, *, log=None) -> dict:
    """Richiesta /avatar/generate con engine=minimax_h3 → {'video', 'image', 'audio', 'intermediates'}."""
    from core.avatar import image_step as IS
    v = dict(req.get("video") or {})
    a = dict(req.get("audio") or {})
    h3 = dict(v.get("h3") or {})
    img_req = IS.ImageRequest.from_dict(req.get("image") or {"mode": "none"})
    image = IS.make_image(img_req, str(OUT / f"start_{uuid.uuid4().hex[:8]}.png"), log=log)
    from core import lang_codes as LC
    lines = [dict(d) for d in (h3.get("dialogue_lines") or [])]
    if not lines and a.get("text"):
        lines = [{"speaker": "", "language": a.get("language") or "", "text": a["text"]}]
    for d in lines:                                  # lingua del testo: della battuta, se no audio.language
        d["language"] = LC.h3_name(d.get("language") or a.get("language") or "") or (d.get("language") or "")
    cfg = {"image_path": image, "prompt": v.get("prompt", ""), "lines": lines,
           "voice_ref_path": a.get("voice_reference_path") or "",
           "duration_s": h3.get("duration_s") or v.get("max_duration_s") or 8.0,
           "h3_profile": v.get("precision_profile") if v.get("precision_profile") in S.PROFILES else "nvfp4",
           "h3_te": h3.get("text_encoder") or "", "h3_turbo": h3.get("turbo_steps") or "",
           "soundscape": h3.get("soundscape", ""), "music": h3.get("music_desc", ""),
           "seed": v.get("seed", -1)}
    p = build_params(cfg)
    errs = preflight(p)
    if errs:
        raise RuntimeError("; ".join(errs))
    lines_out: list[str] = []

    def _l(s):
        lines_out.append(s)
        if log:
            log(s)
    job = make_job(p, on_log=_l)
    rc = job.run()
    if rc != 0 or not os.path.isfile(p["output"]):
        err = next((x for x in reversed(lines_out) if x.startswith("ERRORE")), "") or "\n".join(lines_out[-10:])
        raise RuntimeError(err or f"MiniMax H3 terminato con codice {rc}")
    return {"video": p["output"], "image": image, "audio": "", "intermediates": [x for x in (image,) if x],
            "warnings": [], "languages": {(d.get("speaker") or "S1"): [d["language"]] for d in p["h3_dialogue"]}}
