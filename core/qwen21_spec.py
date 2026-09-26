"""Costanti di Qwen-Image 2.1 condivise fra GUI, API e worker (stdlib-only, nessun effetto all'import)."""
from __future__ import annotations

# profilo di peso → (DiT, text encoder di default) — voci di catalogo
PROFILES = {
    "nvfp4": ("qwen21-dit-nvfp4", "qwen21-te-nvfp4"),
    "fp8": ("qwen21-dit-fp8", "qwen21-te-fp8"),
    "int8": ("qwen21-dit-int8", "qwen21-te-fp8"),
}
TE_VARIANTS = {"heretic": "qwen21-te-heretic-nvfp4"}
TURBO = {"4": ("qwen21-turbo-lora-4", 4), "6": ("qwen21-turbo-lora-6", 6)}
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
