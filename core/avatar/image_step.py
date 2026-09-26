"""Immagine iniziale dell'avatar parlante — i quattro modi dell'API /avatar/generate:

  A · text             → immagine generata dal prompt (motore esplicito, default Qwen-Image 2.1)
  B · image            → immagine già pronta (path)
  C · character        → personaggio da UN riferimento (foto o asset «@Nome» di Film Studio) con modifiche
                          (pettinatura, abiti, sfondo, espressione, posa, luce) e identità preservata
  D · character_multi  → come C ma con più riferimenti (≤10 con Qwen-Image 2.1, ≤2 con Krea 2 Edit)

Il motore immagine è SEMPRE esplicito (regola: niente scelta automatica del checkpoint): se i suoi pesi
mancano, l'errore lo dice con gli ID di catalogo da scaricare. Il ritratto viene chiesto «adatto a parlare»
(volto visibile, bocca libera, frontale), perché è ciò che serve al lip-sync.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

TALKING_HINT = ("The person faces the camera with the mouth clearly visible and relaxed, natural expression, "
                "head and shoulders framing, sharp focus on the face.")
ENGINES = ("qwen_image21", "krea2_edit", "krea2", "flux", "flux2", "zimage")
MAX_REFS = {"qwen_image21": 10, "krea2_edit": 2}


@dataclass
class ImageRequest:
    mode: str = "image"                              # text | image | character | character_multi | none
    prompt: str = ""
    negative: str = ""
    engine: str = "qwen_image21"
    image_path: str = ""
    ref_images: list[str] = field(default_factory=list)
    asset: str = ""                                  # «@Nome» o «Nome» di Film Studio
    changes: dict = field(default_factory=dict)      # hair, outfit, background, expression, pose, lighting, extra
    keep_background: bool = True
    aspect: str = ""                                 # formato 2K (Qwen) — es. "9:16"
    width: int = 0
    height: int = 0
    seed: int = -1
    steps: int = 0
    profile: str = "fp8"                             # profilo pesi Qwen-Image 2.1: nvfp4 | fp8 | int8
    turbo: str = ""                                  # Qwen: "" | "4" | "6"
    enhance_prompt: bool = False
    model: str = ""                                  # FLUX/Z-Image: file checkpoint esplicito

    @classmethod
    def from_dict(cls, d: dict | None) -> "ImageRequest":
        d = dict(d or {})
        ch = d.pop("character", None) or {}
        r = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if ch:
            r.ref_images = list(ch.get("ref_images") or r.ref_images)
            r.asset = ch.get("asset", r.asset) or ""
            r.changes = dict(ch.get("changes") or r.changes)
            r.keep_background = bool(ch.get("keep_background", r.keep_background))
        return r


def build_instruction(changes: dict, keep_background: bool, n_refs: int, base: str = "") -> str:
    """Istruzione di editing che PRESERVA l'identità e applica solo le modifiche chieste."""
    who = "the person in image 1" if n_refs <= 1 else "the same person shown in the reference images"
    parts = [f"Create a new portrait photo of {who}. Keep the identity exactly the same: face, facial features, "
             f"age, skin tone and body type."]
    labels = {"hair": "Hairstyle", "outfit": "Clothing", "expression": "Expression", "pose": "Pose",
              "lighting": "Lighting"}
    for k, lab in labels.items():
        v = str((changes or {}).get(k) or "").strip()
        if v:
            parts.append(f"{lab}: {v}.")
    bg = str((changes or {}).get("background") or "").strip()
    if bg:
        parts.append(f"Background: {bg}.")
    elif keep_background:
        parts.append("Keep the original background unchanged.")
    extra = str((changes or {}).get("extra") or "").strip()
    if extra:
        parts.append(extra)
    if base:
        parts.append(base)
    parts.append(TALKING_HINT)
    return " ".join(parts)


def resolve_asset(name: str) -> list[str]:
    """Immagini di riferimento di un personaggio di Film Studio («@Nome» o «Nome»)."""
    from core.film_studio import gen_service
    tag = name if name.startswith("@") else f"@{name}"
    return [p for p in gen_service.resolve_refs(tag).get("ref_images", []) if p and os.path.isfile(p)]


def _run_image(params: dict, log=None) -> str:
    from core import native_driver
    if log:
        log(f"[avatar] immagine: {params.get('arch')} · {len(params.get('ref_images') or params.get('images') or [])} rif.")
    res = native_driver.run_and_wait("image", params, timeout=3600)
    out = (res.get("output") or [None])[0]
    if res.get("status") != "success" or not out or not os.path.isfile(out):
        raise RuntimeError(f"generazione immagine fallita: {str(res.get('error', ''))[:800]}")
    return out


def _check_weights(r: ImageRequest):
    if r.engine != "qwen_image21":
        return
    from core import ensure_weights as ew
    from core.qwen21_spec import PROFILES, TURBO
    if r.profile not in PROFILES:
        raise ValueError(f"profilo Qwen-Image 2.1 sconosciuto: {r.profile} (nvfp4 | fp8 | int8)")
    ids = [PROFILES[r.profile][0]] + ([TURBO[r.turbo][0]] if r.turbo in TURBO else [])
    miss = [e["id"] for e in ew.missing(ids)]
    if miss:
        ok = [p for p, (d, _t) in PROFILES.items() if not ew.missing([d])]
        raise RuntimeError(f"pesi Qwen-Image 2.1 ({r.profile}) mancanti: {', '.join(miss)}. "
                           f"Profili installati: {', '.join(ok) or 'nessuno'}.")


def make_image(req: ImageRequest | dict, output: str, *, log=None) -> str:
    """Esegue il modo richiesto e ritorna il path dell'immagine iniziale ('' per mode='none')."""
    r = req if isinstance(req, ImageRequest) else ImageRequest.from_dict(req)
    if r.mode == "none":
        return ""
    if r.mode == "image":
        if not (r.image_path and os.path.isfile(r.image_path)):
            raise ValueError(f"immagine inesistente: {r.image_path}")
        return r.image_path
    if r.engine not in ENGINES:
        raise ValueError(f"motore immagine sconosciuto: {r.engine} ({', '.join(ENGINES)})")
    seed = r.seed if r.seed is not None and r.seed >= 0 else random.randint(0, 2**31 - 1)
    params: dict = {"arch": r.engine, "negative": r.negative, "seed": seed, "output": output}
    if r.width and r.height:
        params.update(width=int(r.width), height=int(r.height))
    if r.steps:
        params["steps"] = int(r.steps)
    if r.engine == "qwen_image21":
        params.update(qwen21_profile=r.profile, qwen21_turbo=r.turbo, aspect_2k=r.aspect)
    if r.model:
        params["model"] = r.model
    if r.mode == "text":
        prompt = r.prompt.strip()
        if not prompt:
            raise ValueError("modo 'text': serve image.prompt")
        if r.enhance_prompt:
            from core import prompt_enhance
            prompt = prompt_enhance.enhance(prompt, kind="image") or prompt
        _check_weights(r)
        params["prompt"] = f"{prompt} {TALKING_HINT}"
        return _run_image(params, log)
    if r.mode in ("character", "character_multi"):
        refs = [p for p in (r.ref_images or []) if p and os.path.isfile(p)]
        if r.asset:
            refs = resolve_asset(r.asset) + refs
        if not refs:
            raise ValueError("modo personaggio: servono character.ref_images o character.asset")
        if r.mode == "character":
            refs = refs[:1]
        lim = MAX_REFS.get(r.engine, 1)
        if len(refs) > lim:
            if log:
                log(f"[avatar] {r.engine} accetta al massimo {lim} riferimenti: uso i primi {lim}")
            refs = refs[:lim]
        if r.engine not in MAX_REFS:
            raise ValueError(f"il motore {r.engine} non fa editing con riferimenti: usa qwen_image21 o krea2_edit")
        _check_weights(r)
        params["prompt"] = build_instruction(r.changes, r.keep_background, len(refs), r.prompt.strip())
        if r.engine == "qwen_image21":
            params["ref_images"] = refs
        else:                                            # Krea 2 Edit: 1 = single, 2 = multi (scena, soggetto)
            params.update({"krea2_variant": "turbo", "grounding_px": 1024, "guidance": 0.0,
                           "steps": params.get("steps", 8)})
            if len(refs) == 1:
                params["image_path"] = refs[0]
            else:
                params["images"] = refs
        return _run_image(params, log)
    raise ValueError(f"modo immagine sconosciuto: {r.mode}")
