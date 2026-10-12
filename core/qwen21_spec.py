"""Costanti di Qwen-Image 2.1 condivise fra GUI, API e worker (stdlib-only, nessun effetto all'import)."""
from __future__ import annotations

# profilo di peso → (DiT, text encoder di default) — voci di catalogo (API/avatar: nomi brevi storici)
PROFILES = {
    "nvfp4": ("qwen21-dit-nvfp4", "qwen21-te-nvfp4"),
    "fp8": ("qwen21-dit-fp8", "qwen21-te-fp8"),
    "int8": ("qwen21-dit-int8", "qwen21-te-fp8"),
}
# DiT di catalogo → metadati. Il text encoder è lo stesso Qwen3-VL-8B per tutti (Turbo compreso: tensori
# identici a quelli del base), quindi qui si sceglie solo la quantizzazione che fa coppia col DiT.
#   turbo: checkpoint distillato a 8 passi con la sua schedule (TURBO_SIGMAS), CFG 1, niente LoRA turbo
#   adult: checkpoint senza censura → visibile solo in modalità NSFW (e in catalogo sensitivity=adult)
#   steps/cfg: impostazioni consigliate dall'autore (sovrascrivono i default all'arrivo nella GUI)
DITS = {
    "qwen21-dit-nvfp4": {"te": "qwen21-te-nvfp4"},
    "qwen21-dit-fp8": {"te": "qwen21-te-fp8"},
    "qwen21-dit-int8": {"te": "qwen21-te-fp8"},
    "qwen21-turbo-dit-nvfp4": {"te": "qwen21-te-nvfp4", "turbo": True},
    "qwen21-turbo-dit-int8": {"te": "qwen21-te-fp8", "turbo": True},
    "qwen21-uc-dit-nvfp4": {"te": "qwen21-te-nvfp4", "adult": True},
    "qwen21-uc-dit-int8": {"te": "qwen21-te-fp8", "adult": True},
    "qwen21-noctq-v4-int8": {"te": "qwen21-te-fp8", "adult": True, "steps": 25, "cfg": 3.0},
    "qwen21-noctq-anime-int8": {"te": "qwen21-te-fp8", "adult": True, "steps": 25, "cfg": 3.0},
}
TE_VARIANTS = {"heretic": "qwen21-te-heretic-nvfp4"}
TURBO = {"4": ("qwen21-turbo-lora-4", 4), "6": ("qwen21-turbo-lora-6", 6)}
# schedule ufficiale di Qwen-Image-2.1-Turbo (model_index.json «sample_sigmas»; lo 0 finale lo aggiunge lo scheduler)
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
TURBO_STEPS = len(TURBO_SIGMAS)
# formati 2K nativi (README ufficiale)
ASPECTS_2K = {"1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696),
              "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752)}
MAX_REFS = 10
RGBA_TEMPLATE = ("This is an RGBA image with transparency. {p} The image has alpha channel and the "
                 "background is transparent.")
MASK_HINT = ("Only change the area highlighted in red in image 1 and keep everything else exactly the same. "
             "Remove the red highlight in the result.")
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 1.0
DEFAULT_STRENGTH = 0.6          # img2img: quota di rumore (0 = immagine invariata, 1 = generazione da zero)


def dit_for(params: dict) -> str:
    """DiT di catalogo scelto: `qwen21_dit` esplicito, altrimenti dal profilo breve (API/avatar)."""
    dit = str(params.get("qwen21_dit") or "")
    if dit in DITS:
        return dit
    prof = str(params.get("qwen21_profile") or "fp8").lower()
    if prof not in PROFILES:
        raise ValueError(f"Profilo Qwen-Image 2.1 sconosciuto: {prof} ({' | '.join(PROFILES)})")
    return PROFILES[prof][0]


def is_turbo(dit_id: str) -> bool:
    return bool(DITS.get(dit_id, {}).get("turbo"))


def is_adult(dit_id: str) -> bool:
    return bool(DITS.get(dit_id, {}).get("adult"))
