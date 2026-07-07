"""Selezione modelli CONDIVISA (Cinema Studio, Music Video Studio, …): elenca i checkpoint immagine e i GGUF
video locali per popolare i dropdown, senza hardcode. L'arch dal nome è un'euristica — il backend la RIVALIDA
dai canali del checkpoint (arch-guard in image_flux), quindi è tollerante agli errori."""
from __future__ import annotations

import glob
import os

_DM = r"E:\AI\Models\diffusion_models"
_CK = r"E:\AI\Models\Checkpoints"
_SKIP = ("vae", "clip", "t5", "text_encoder", "lora", "upscal", "controlnet", "ipadapter", "pulid")


def arch_of(name: str) -> str:
    """Famiglia d'architettura dedotta dal nome file (flux/flux2/zimage/ernie)."""
    nl = name.lower()
    if "flux2" in nl or "klein" in nl:
        return "flux2"
    if "z-image" in nl or "zimage" in nl or "z_image" in nl:
        return "zimage"
    if "ernie" in nl:
        return "ernie"
    return "flux"


def list_image_models() -> list[tuple[str, str, str]]:
    """[(label, path, arch)] dei checkpoint IMMAGINE locali (esclude vae/clip/lora/… e file piccoli)."""
    out, seen = [], set()
    for root in (_DM, _CK):
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True):
            k = os.path.normpath(p).lower()
            if k in seen:
                continue
            seen.add(k)
            n = os.path.basename(p)
            if any(s in n.lower() for s in _SKIP):
                continue
            try:
                if os.path.getsize(p) < 500 * 1024 * 1024:   # < 500MB → non è un checkpoint pesante
                    continue
            except OSError:
                continue
            out.append((n, p, arch_of(n)))
    out.sort(key=lambda t: t[0].lower())
    return out


def list_video_gguf() -> list[tuple[str, str]]:
    """[(label, path)] dei GGUF video (LTX-2) locali."""
    out, seen = [], set()
    for p in glob.glob(os.path.join(_DM, "**", "*.gguf"), recursive=True):
        k = os.path.normpath(p).lower()
        if k in seen:
            continue
        seen.add(k)
        out.append((os.path.basename(p), p))
    out.sort(key=lambda t: t[0].lower())
    return out
