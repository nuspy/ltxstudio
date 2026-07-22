"""Auto-selezione unificata modelli/LoRA/settings (parità col vecchio ComfyUI).

Dato (task, SFW/NSFW, preferenza) sceglie il MIGLIOR modello disponibile + i settings ottimali
(steps/cfg da ModelCatalog.RECOMMENDED) + le LoRA compatibili (stessa architettura + contenuto).
Riusa core/model_catalog.py (arch_of/is_nsfw/recommended_*) e core/offline_inventory.list_models.

API:
  auto_config(task, nsfw=, prefer=) -> dict | None   # config pronta (model/arch/settings/loras)
  best_model(task, nsfw=, prefer=) -> str | None
  settings_for(model) -> dict
  compatible_loras(model, nsfw=) -> list[str]
"""
from __future__ import annotations

from core.model_catalog import ModelCatalog
from core import offline_inventory as inv

# task → sottostringhe di architettura compatibili (match su arch_of o sul nome file)
TASK_ARCH = {
    "image":   ["flux", "krea2", "zimage", "sd3", "sdxl", "lumina", "ernie"],
    "kontext": ["kontext", "flux2"],
    "inpaint": ["inpaint", "fill", "flux2"],
    "video":   ["ltx", "lingbot", "krea_rt", "wan", "hunyuan"],
}
# preferenza di qualità per famiglia (più alto = preferito); euristica, sovrascrivibile
ARCH_PREF = {"flux2": 6, "flux": 5, "krea2": 5, "kontext": 5, "zimage": 4, "sd3": 3, "lumina": 3,
             "sdxl": 2, "ernie": 2, "ltx": 6, "lingbot": 6, "krea_rt": 5, "wan": 4, "hunyuan": 3}

_cat = ModelCatalog()


def _arch(name: str) -> str:
    return (_cat.arch_of(name) or "").lower()


def _pref(name: str) -> int:
    a = _arch(name) + " " + name.lower()
    return max((p for k, p in ARCH_PREF.items() if k in a), default=0)


def _candidates(task: str) -> list[str]:
    keys = TASK_ARCH.get(task, [task])
    out: list[str] = []
    for cat in ("diffusion_models", "checkpoints"):
        try:
            for m in inv.list_models(cat):
                hay = _arch(m) + " " + m.lower()
                if any(k in hay for k in keys):
                    out.append(m)
        except Exception:                              # noqa: BLE001
            pass
    return out


def best_model(task: str = "image", nsfw: bool = False, prefer: str | None = None) -> str | None:
    _cat.reload_if_changed()
    cands = _candidates(task)
    if not cands:
        return None
    pool = [m for m in cands if _cat.is_nsfw(m) == nsfw] or cands   # match contenuto se possibile
    def score(m):
        fav = 1 if (prefer and prefer.lower() in m.lower()) else 0
        return (fav, _pref(m))
    return max(pool, key=score)


def settings_for(model: str) -> dict:
    return {
        "steps": _cat.recommended_steps(model) or 20,
        "cfg": _cat.recommended_cfg(model) or 3.5,
        "note": _cat.recommended_note(model) or "",
    }


def compatible_loras(model: str, nsfw: bool = False) -> list[str]:
    """LoRA con la STESSA architettura del modello (o arch ignota) e contenuto coerente."""
    arch = _arch(model)
    fam = next((k for k in ARCH_PREF if k in arch), "")
    out = []
    try:
        for lo in inv.list_models("loras"):
            la = _arch(lo)
            same_fam = (fam and fam in la) or (not la)      # stessa famiglia o arch ignota
            if same_fam and (nsfw or not _cat.is_nsfw(lo)):  # in SFW escludi le LoRA NSFW
                out.append(lo)
    except Exception:                                  # noqa: BLE001
        pass
    return out


def auto_config(task: str = "image", nsfw: bool = False, prefer: str | None = None) -> dict | None:
    m = best_model(task, nsfw, prefer)
    if not m:
        return None
    cfg = {"model": m, "arch": _cat.arch_of(m), "nsfw": _cat.is_nsfw(m)}
    cfg.update(settings_for(m))
    cfg["loras_compatible"] = compatible_loras(m, nsfw)
    return cfg
