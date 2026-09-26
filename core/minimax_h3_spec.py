"""MiniMax H3: costanti condivise da GUI (tab Video / Avatar), API e worker.

Tutto viene dalla documentazione ufficiale (diffusers modular MiniMax-H3) e dalle schede lightx2v Turbo:
  · 24 fps fissi, durata 5–15 s, fotogrammi agganciati a 17·n + 5 (griglia del VAE video);
  · tela: lato corto 768, area ≤ 768×1344, lati multipli di 32 (960×544 ≈ 2,3× più veloce);
  · guidance distillata: niente negative prompt né CFG, una passata per step;
  · `num_inference_steps` del pipeline conta i punti della griglia sigma (0 finale compreso) = NFE + 1;
  · Turbo lightx2v: 4 NFE (shift video 12) o 8 NFE (shift video 6, consigliato dallo Studio), audio shift 3.
"""
from __future__ import annotations

import math

FPS = 24
MIN_S, MAX_S = 5.0, 15.0
FRAME_GRID = 17                       # fotogrammi per chunk del VAE video
DEFAULT_NFE = 49                      # default del pipeline: 50 punti di griglia
VIDEO_SHIFT, AUDIO_SHIFT = 12.0, 3.0

# profilo → (DiT FL2VA, DiT Ref2VA, TE consigliato). Nessuna scelta automatica: la GUI propone, l'utente conferma.
PROFILES = {
    "nvfp4": ("mmh3-fl2va-nvfp4", "mmh3-ref2va-nvfp4", "mmh3-te-nvfp4"),
    "fp8": ("mmh3-fl2va-fp8", "mmh3-ref2va-fp8", "mmh3-te-fp8"),
    "int8": ("mmh3-fl2va-int8", "mmh3-ref2va-int8", "mmh3-te-int8"),
}
TE_IDS = ("mmh3-te-nvfp4", "mmh3-te-fp8", "mmh3-te-int8", "mmh3-te-heretic-nvfp4")
TE_ADULT = "mmh3-te-heretic-nvfp4"
COMMON_IDS = ("mmh3-configs", "mmh3-video-vae", "mmh3-audio-vae")

# turbo: chiave GUI → {modo: id LoRA}, NFE, shift video
TURBO = {
    "4": ({"fl2va": "mmh3-turbo-lora-fl2v", "ref2va": "mmh3-turbo-lora-ref2v"}, 4, 12.0),
    "8": ({"fl2va": "mmh3-turbo-lora-fl2v-8"}, 8, 6.0),
}

# formati: rapporto → (larghezza, altezza) del rapporto; la tela esatta la calcola il pipeline
ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "21:9": (21, 9), "4:3": (4, 3), "3:4": (3, 4), "1:1": (1, 1)}
CANVAS = {"768": (768, 768 * 1344), "544": (544, 544 * 960)}     # lato corto, area massima

MAX_REFS = {"image": 9, "video": 3, "audio": 3, "total": 12}
DIALOGUE_LANGS = ("Italian", "English", "Chinese", "Spanish", "French", "German", "Portuguese",
                  "Japanese", "Korean", "Russian", "Arabic")


def mode_for(params: dict) -> str:
    return "ref2va" if params.get("references") else "fl2va"


def snap_frames(seconds: float) -> int:
    """Durata richiesta → num_frames valido (17·n + 5 per eccesso) dentro 5–15 s."""
    s = min(MAX_S, max(MIN_S, float(seconds)))
    n = max(0, math.ceil((round(s * FPS) - 5) / FRAME_GRID))
    frames = FRAME_GRID * n + 5
    while frames / FPS > MAX_S + 1e-6:
        frames -= FRAME_GRID
    while frames / FPS < MIN_S - 1e-6:
        frames += FRAME_GRID
    return frames


def frames_seconds(frames: int) -> float:
    return frames / FPS


def dialogue_tag(lang: str, text: str, speaker: str = "") -> str:
    """Battuta nel formato della guida ufficiale: <d>[Lingua] testo</d> (speaker opzionale davanti)."""
    body = f"<d>[{lang}] {text.strip()}</d>"
    return f"{speaker.strip()}: {body}" if speaker.strip() else body


def weight_ids(params: dict) -> dict:
    """ID di catalogo richiesti da una richiesta (profilo/TE/turbo espliciti; nessun ripiego)."""
    mode = mode_for(params)
    prof = str(params.get("h3_profile") or "nvfp4").lower()
    if prof not in PROFILES:
        raise ValueError(f"profilo MiniMax H3 sconosciuto: {prof} ({' | '.join(PROFILES)})")
    fl, ref, te = PROFILES[prof]
    ids = {"dit": ref if mode == "ref2va" else fl, "te": params.get("h3_te") or te,
           "configs": "mmh3-configs", "vae": "mmh3-video-vae", "audio_vae": "mmh3-audio-vae"}
    if ids["te"] not in TE_IDS:
        raise ValueError(f"text encoder MiniMax H3 sconosciuto: {ids['te']}")
    t = str(params.get("h3_turbo") or "")
    if t:
        if t not in TURBO or mode not in TURBO[t][0]:
            raise ValueError(f"LoRA turbo {t} passi non disponibile per {mode}")
        ids["turbo"] = TURBO[t][0][mode]
    return ids
