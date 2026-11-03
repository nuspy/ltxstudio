"""«Modifica Video AI» con il motore MiniMax H3 (edit con maschera): costruzione dei params del worker — modulo PURO (senza Qt/torch).

Condiviso da GUI (ui/main_window.py::_VideoEditAIWorker) e, in futuro, REST API. Il worker H3 e `workers/minimax_h3_infer.py`
(ramo `h3_edit` -> `workers/h3_edit_run.py`), nel venv di H3 (core.ltx_video_job.MINIMAX_H3_PY).
"""
from __future__ import annotations

import os

from core import video_edit_engines as E

# profili H3 per l'edit: l'edit usa SEMPRE i DiT Ref2VA (foto + video di riferimento + audio); l'utente sceglie il profilo, mai in automatico
H3_PROFILES: tuple[str, ...] = ("int8", "fp8", "nvfp4", "int4")
H3_PROFILE_I18N = {"int8": "INT8 ConvRot (21 GB)", "fp8": "FP8 (21 GB)", "nvfp4": "NVFP4 (12,5 GB, RTX 50xx)", "int4": "INT4 W4A8 (12 GB)"}
DEFAULT_TURBO = "v4"                          # LoRA turbo larryvrh v4 (ha la variante Ref2VA, 8 passi, shift video 12)
DEFAULT_SHORT_EDGE = 480                      # lato corto della tela (Openjutsu: 480p in 6-13 min per 6 s su 16 GB; default 736p)
SHORT_EDGES = (480, 576, 736)


class H3EditError(RuntimeError):
    pass


def installed_profiles(present=None) -> list[str]:
    """Profili con il DiT Ref2VA su disco (il resto dei pesi lo controlla `video_edit_engines.missing_message`)."""
    present = present or E._present
    ids = {"int8": "mmh3-ref2va-int8", "fp8": "mmh3-ref2va-fp8", "nvfp4": "mmh3-ref2va-nvfp4", "int4": "mmh3-ref2va-int4"}
    return [p for p in H3_PROFILES if present(ids[p])]


def track_dir_of(video: str) -> str:
    return os.path.splitext(video)[0] + "_track"


def build_params(cfg: dict, out: str, *, action: str | None = None, present=None) -> dict:
    """`params` per workers/minimax_h3_infer.py (senza il tracking SAM2, che lo fa il chiamante e riempie `mask_dir`).

    cfg: video, mode (change_background|replace_char), object (cosa tenere / chi sostituire), prompt (il nuovo luogo / note), ref_image
    (foto del luogo, opzionale, o del nuovo personaggio, obbligatoria), h3_profile, short_edge, seed, engine."""
    mode = cfg.get("mode")
    act = action or E.action_of_mode(mode or "")
    if act not in (E.CHANGE_BACKGROUND, E.REPLACE_CHARACTER):
        raise H3EditError(f"Il motore H3 non fa l'azione «{act}».")
    eng = cfg.get("engine") or "h3_edit"
    msg = E.missing_message(eng, present)
    if msg:
        raise H3EditError(msg)
    prof = str(cfg.get("h3_profile") or "")
    if prof not in installed_profiles(present):
        raise H3EditError("Scegli il profilo H3 (int8/fp8/nvfp4/int4) di cui hai il DiT Ref2VA installato: nessuna scelta automatica.")
    video = str(cfg.get("video") or "")
    if not os.path.isfile(video):
        raise H3EditError(f"Video sorgente non trovato: {video}")
    keep_text = str(cfg.get("object") or "").strip()
    if not keep_text:
        raise H3EditError("Indica la persona (a parole, es. «person»): serve a SAM2 per la maschera." if act == E.CHANGE_BACKGROUND
                          else "Indica il personaggio da sostituire (a parole, es. «person»).")
    pic = str(cfg.get("ref_image") or "").strip()
    if act == E.REPLACE_CHARACTER and (not pic or not os.path.isfile(pic)):
        raise H3EditError("Sostituisci personaggio: scegli la foto del nuovo personaggio.")
    if act == E.CHANGE_BACKGROUND and not pic and not str(cfg.get("prompt") or "").strip():
        raise H3EditError("Cambia sfondo: descrivi il nuovo luogo a parole o scegli la foto del luogo.")
    ep = E.ENGINES[eng].params
    e = {"action": act, "video": video, "mask_dir": os.path.join(track_dir_of(video), "masks"),
         "short_edge": int(cfg.get("short_edge") or DEFAULT_SHORT_EDGE), "keep_text": keep_text,
         "prompt_text": str(cfg.get("prompt") or "").strip(), "picture": pic if pic and os.path.isfile(pic) else "",
         "picture_desc": str(cfg.get("picture_desc") or ""), "keep_audio": bool(cfg.get("keep_audio", True)),
         "clip_s": float(ep["clip_s"]), "overlap": int(ep["overlap_frames"]), "grow_px": int(ep["grow_px"]),
         "feather_px": int(ep["feather_px"]), "headroom": float(ep["headroom"]), "hold_frames": int(ep["pose_hold_frames"]),
         "person_grow_px": int(ep["person_grow_px"]), "replace_expand": float(ep.get("replace_expand", 0.0))}
    return {"arch": "minimax_h3", "h3_edit": e, "h3_profile": prof, "h3_turbo": DEFAULT_TURBO, "h3_video_shift": float(ep["video_shift"]),
            "seed": int(cfg.get("seed") or 42), "output": out, **E.engine_params(act, eng)}
