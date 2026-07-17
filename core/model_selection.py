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
    if "krea-2" in nl or "krea2" in nl:
        return "krea2"
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
    """VUOTA di proposito: i GGUF VIDEO (LTX-2) sono stati rimossi dal prodotto (13/07).

    Motivo: il loader dequantizza in Python A OGNI forward (1660 Linear × step) → ~500 s/step
    contro i ~4 s/step dell'fp8, cioè oltre un'ora per clip; e il Q6_K (16.6GB) residente andava
    in OOM alle risoluzioni normali (704x1280), producendo cartelle di .json e zero video.
    Non c'era alcuno scenario in cui convenisse. Il video usa SEMPRE il checkpoint fp8.

    NB: i GGUF delle IMMAGINI (FLUX Fill per l'inpaint) restano: quelli funzionano e sono usati.
    La funzione resta (firma invariata) per non rompere i chiamanti: semplicemente non offre nulla.
    """
    return []
