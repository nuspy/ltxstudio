"""H3 Video Upsampler: costanti e pesi (stdlib, nessun torch: lo usano GUI, API e worker).

Funzione = upscale video; architettura = MiniMax H3 (nel catalogo i pesi sono arch=minimax_h3, use=upscale_video).
Il ri-disegno è fatto dal DiT *Ref2VA* di H3 (la LoRA di guida è addestrata su quello): il checkpoint lo sceglie
l'utente con il profilo (nvfp4 / int4 / int8 / fp8) — nessuna scelta automatica.
"""
from __future__ import annotations

from core import minimax_h3_spec as S

ENGINE = "h3_upsampler"                  # valore `engine` di /video/enhance e chiave del motore nel tab Definizione video
ARCH = "minimax_h3"
FUNCTION = "upscale_video"
CAP = "cap.video_h3_upsampler"           # riga della feature matrix

UPSCALER_ID = "mmh3-latent-upscaler-v1"  # LBH-123-AI/Minimax_h3_latent_Upscaler (Apache-2.0)
GUIDE_LORA_ID = "mmh3-lora-h3upscale-x2" # sandpies/ComfyUI-H3-Video-Upsampler lora/h3upscale_v2_x2.safetensors

PROFILES = ("nvfp4", "int4", "int8", "fp8")          # ordine di presentazione nel menu (non una scelta)
MP_RANGE = (0.1, 16.0)
TILE_TOKENS_RANGE = (8192, 1048576)
DENOISE_RANGE = (0.0, 1.0)

LICENSE_NOTE = ("Licenza MiniMax H3 Community: territori esclusi UE, Regno Unito, USA, Corea del Sud; gli output non "
                "si usano per addestrare altri modelli.")


def dit_id(profile: str) -> str:
    """ID del DiT Ref2VA del profilo (la LoRA di guida richiede il Ref2VA, non il FL2VA)."""
    if profile not in S.PROFILES:
        raise ValueError(f"profilo MiniMax H3 sconosciuto: {profile!r} ({' | '.join(S.PROFILES)})")
    return S.PROFILES[profile][1]


def weight_ids(p: dict) -> dict:
    """ID di catalogo richiesti da una richiesta. Profilo ESPLICITO (niente ripiego): senza `h3_profile` errore."""
    prof = str(p.get("h3_profile") or "").lower()
    if not prof:
        raise ValueError("H3 Video Upsampler: scegli il modello H3 (profilo nvfp4/int4/int8/fp8)")
    te = p.get("h3_te") or S.PROFILES[prof if prof in S.PROFILES else "nvfp4"][2]
    if te not in S.TE_IDS:
        raise ValueError(f"text encoder MiniMax H3 sconosciuto: {te}")
    ids = {"dit": dit_id(prof), "te": te, "configs": "mmh3-configs", "vae": "mmh3-video-vae",
           "audio_vae": "mmh3-audio-vae", "upscaler": UPSCALER_ID}
    if p.get("guide", True):
        ids["guide_lora"] = GUIDE_LORA_ID
    return ids


def build_params(cfg: dict) -> dict:
    """cfg della GUI/API → params del worker. Nessun default di comportamento nascosto: i valori mancanti prendono
    quelli del nodo H3 Video Upsampler studiato (megapixel 2.0, denoise 0.1, forza LoRA 1.0, budget 70.000)."""
    from core import h3_upsample_plan as P
    p = {
        "video": cfg.get("video") or "", "output": cfg.get("output") or "",
        "h3_profile": str(cfg.get("h3_profile") or ""), "h3_te": str(cfg.get("h3_te") or ""),
        "megapixels": float(cfg.get("megapixels") or P.MEGAPIXELS),
        "denoise": float(cfg["denoise"]) if cfg.get("denoise") is not None else P.DENOISE_DEFAULT,
        "guide": bool(cfg.get("guide", True)),
        "lora_strength": float(cfg.get("lora_strength") or 1.0),
        "tile_tokens": int(cfg.get("tile_tokens") or P.TOKEN_BUDGET),
    }
    if not p["h3_te"]:
        p.pop("h3_te")
    return p


def validate(p: dict) -> list[str]:
    """Errori di valore (senza toccare disco né modelli)."""
    errs = []
    if not p.get("video"):
        errs.append("Scegli un video.")
    try:
        weight_ids(p)
    except ValueError as e:
        errs.append(str(e))
    lo, hi = MP_RANGE
    if not lo <= float(p.get("megapixels", 0)) <= hi:
        errs.append(f"megapixel fuori intervallo ({lo:g}–{hi:g}).")
    lo, hi = DENOISE_RANGE
    if not lo <= float(p.get("denoise", -1)) <= hi:
        errs.append("denoise fuori intervallo (0–1).")
    lo, hi = TILE_TOKENS_RANGE
    if not lo <= int(p.get("tile_tokens", 0)) <= hi:
        errs.append(f"budget dei tile fuori intervallo ({lo}–{hi}).")
    return errs


def missing_weights(p: dict) -> list[dict]:
    """Voci di catalogo (dipendenze incluse, deps prima) richieste e NON presenti su disco. Nessun download."""
    from core import ensure_weights as ew
    return ew.missing(list(weight_ids(p).values()))


def missing_text(entries: list[dict]) -> str:
    gb = round(sum(float(e.get("approx_gb", 0) or 0) for e in entries), 1)
    return ", ".join(f"{e['id']} (~{e.get('approx_gb', '?')} GB)" for e in entries) + f" — totale ~{gb} GB"
