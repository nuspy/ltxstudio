"""Orchestrazione import modelli/LoRA (#17) — categorizza, decide SFW/NSFW, applica il gate.

RIUSA la logica canonica del catalogo (E:\\AI\\dictionary\\tools): `classify_safetensors` (tipo+famiglia) e
`family_to_subfolder` (cartella di destinazione). Il rilevamento arch nativo `core/model_detect` è il fallback
istantaneo. La cascata SFW/NSFW: Civitai (autoritativo) → euristica nome/tag → classificatore preview →
default NSFW (conservativo). Il GATE blocca gli item NSFW quando `cap.adult_mode` è OFF (skip, non-bloccante).

Lo SPOSTAMENTO reale (place→scan→enrich→build) resta a `CatalogTooling` (chiamato dalla GUI su un source-dir):
qui prepariamo la decisione per-item (preview + gate) PRIMA dello staging.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from core import catalog_paths, civitai_client, nsfw_classifier

# hint testuali per l'euristica NSFW (nome file / tag / cartella)
_NSFW_HINTS = (
    "nsfw", "porn", "hentai", "nude", "nudity", "naked", "explicit", "erotic", "sex",
    "boob", "breast", "nipple", "pussy", "penis", "genital", "lewd", "ecchi", "r18",
    "xxx", "18+", "onlyfans", "cum", "bdsm", "fetish",
)
_TOOLS_ADDED = False


def _add_tools() -> None:
    global _TOOLS_ADDED
    if _TOOLS_ADDED:
        return
    td = str(catalog_paths.tools_dir())
    if Path(td).exists() and td not in sys.path:
        sys.path.insert(0, td)
    _TOOLS_ADDED = True


def _place_files():
    _add_tools()
    try:
        import place_files                                   # type: ignore
        return place_files
    except Exception:                                        # noqa: BLE001
        return None


@dataclass
class ImportItem:
    path: str
    name: str = ""
    type: str = "Unknown"                                    # LoRA/Checkpoint/VAE/TextEncoder/…
    family: str = "Unknown"                                  # Flux/SDXL/LTX/Wan21/…
    dest_subfolder: str | None = None                        # es. "LoRA/Flux"
    sha256: str = ""
    triggers: list = field(default_factory=list)
    preview_url: str | None = None
    nsfw: bool = False
    nsfw_source: str = ""                                    # civitai/heuristic/classifier/default
    civitai: dict | None = None
    source: str = "local"                                    # local/civitai
    note: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = Path(self.path).name


# ---------------- classificazione tipo/arch ----------------
def _classify_type_family(path: str) -> tuple[str, str]:
    """(type, family) canonici via place_files.classify_safetensors; fallback su model_detect nativo."""
    p = Path(path)
    pf = _place_files()
    if pf is not None:
        try:
            if p.suffix.lower() in (".safetensors", ".ckpt", ".pt"):
                typ, fam, _reason = pf.classify_safetensors(p, p.stat().st_size)
            else:
                typ, fam = pf.classify_other(p)
            return typ, fam
        except Exception:                                    # noqa: BLE001
            pass
    # fallback nativo: solo arch (il tipo resta grezzo)
    try:
        from core import model_detect
        det = model_detect.detect_one(path)
        fam = {"flux1": "Flux", "flux2": "Flux", "sdxl": "SDXL", "sd15": "SD1.5",
               "ltxv": "LTX", "wan21": "Wan21", "wan22": "Wan22", "zimage": "Z-Image",
               "ernie": "Ernie"}.get(det.get("family", ""), "Unknown")
        return "Checkpoint", fam
    except Exception:                                        # noqa: BLE001
        return "Unknown", "Unknown"


def _dest_subfolder(typ: str, family: str) -> str | None:
    pf = _place_files()
    if pf is not None:
        try:
            return pf.family_to_subfolder(typ, family)
        except Exception:                                    # noqa: BLE001
            pass
    return f"{typ}/{family}"


# ---------------- SFW / NSFW ----------------
def _heuristic_nsfw(name: str, tags: list | None) -> bool:
    blob = (name or "").lower() + " " + " ".join(str(t).lower() for t in (tags or []))
    return any(h in blob for h in _NSFW_HINTS)


def classify_nsfw(civitai_meta: dict | None, name: str, tags: list | None,
                  preview_path: str | None = None, allow_classifier_download: bool = False) -> tuple[bool, str]:
    """Cascata: Civitai (autoritativo) → euristica → classificatore → default NSFW. (nsfw, fonte)."""
    if civitai_meta:                                          # 1) Civitai è autoritativo
        model = civitai_meta.get("model") or {}
        lvl = int(civitai_meta.get("nsfwLevel") or 0)
        above_sfw = lvl & ~civitai_client.SFW_MASK           # bit R/X/XXX/Blocked
        return bool(model.get("nsfw") or above_sfw), "civitai"
    if _heuristic_nsfw(name, tags):                          # 2) euristica nome/tag
        return True, "heuristic"
    if preview_path:                                         # 3) classificatore sulla preview
        c = nsfw_classifier.is_nsfw(preview_path, allow_classifier_download)
        if c is not None:
            return c, "classifier"
    return True, "default"                                   # 4) incerto → NSFW (conservativo)


# ---------------- analisi item ----------------
def analyze(path: str, fetch_civitai: bool = True, api_key: str | None = None,
            preview_path: str | None = None, allow_classifier_download: bool = False) -> ImportItem:
    """Analizza un file locale → ImportItem (tipo, arch, dest, sha, triggers, nsfw). Rete solo se
    fetch_civitai=True (by-hash), altrimenti offline."""
    item = ImportItem(path=path)
    item.type, item.family = _classify_type_family(path)
    item.dest_subfolder = _dest_subfolder(item.type, item.family)
    meta = None
    if fetch_civitai and Path(path).suffix.lower() in (".safetensors", ".ckpt", ".pt"):
        try:
            item.sha256 = civitai_client.sha256_file(path)
            meta = civitai_client.by_hash(item.sha256, api_key)
        except Exception:                                    # noqa: BLE001
            meta = None
    if meta:
        item.civitai = meta
        item.triggers = list(meta.get("trainedWords") or [])
        imgs = meta.get("images") or (meta.get("model") or {}).get("images") or []
        if imgs:
            item.preview_url = imgs[0].get("url")
        bm = meta.get("baseModel") or ""
        item.family = _CIVITAI_BASEMODEL_MAP.get(bm, item.family)
        mtype = (meta.get("model") or {}).get("type")
        if mtype:
            item.type = _CIVITAI_TYPE_MAP.get(mtype, item.type)
        item.dest_subfolder = _dest_subfolder(item.type, item.family)
        item.source = "civitai-matched"
    tags = (meta.get("model") or {}).get("tags") if meta else None
    item.nsfw, item.nsfw_source = classify_nsfw(meta, item.name, tags, preview_path,
                                                allow_classifier_download)
    return item


# ---------------- Civitai URL/ID ----------------
def resolve_civitai(url_or_id: str, api_key: str | None = None) -> dict:
    """Risolve un URL/ID Civitai → {version, file, name, preview_url, triggers, base_model, type, nsfw,
    download_url}. Solleva su input non valido / non trovato."""
    api_key = api_key or civitai_client.api_key_from_prefs()
    ref = civitai_client.parse_url(url_or_id)
    if ref.get("version_id"):
        ver = civitai_client.version(ref["version_id"], api_key)
    else:
        mdl = civitai_client.model(ref["model_id"], api_key)
        vers = mdl.get("modelVersions") or []
        if not vers:
            raise ValueError("Nessuna versione trovata per questo modello Civitai.")
        ver = vers[0]
        ver.setdefault("model", {"name": mdl.get("name"), "type": mdl.get("type"),
                                 "nsfw": mdl.get("nsfw"), "tags": mdl.get("tags")})
    f = civitai_client.pick_primary_file(ver)
    imgs = ver.get("images") or (ver.get("model") or {}).get("images") or []
    nsfw, src = classify_nsfw(ver, (f or {}).get("name", ""), (ver.get("model") or {}).get("tags"))
    return {
        "version": ver, "file": f,
        "name": (f or {}).get("name") or f"civitai_{ver.get('id')}.safetensors",
        "download_url": (f or {}).get("downloadUrl"),
        "preview_url": imgs[0].get("url") if imgs else None,
        "triggers": list(ver.get("trainedWords") or []),
        "base_model": ver.get("baseModel"),
        "type": (ver.get("model") or {}).get("type"),
        "nsfw": nsfw, "nsfw_source": src,
    }


# ---------------- gate NSFW ----------------
def adult_enabled() -> bool:
    """True se il contenuto adulto è abilitato (feature_matrix cap.adult_mode; fail-open in dev)."""
    try:
        from core import features
        return bool(features.is_enabled("cap.adult_mode"))
    except Exception:                                        # noqa: BLE001
        return True


def gate_ok(item_nsfw: bool) -> bool:
    """False se l'item è NSFW e la feature adult è OFF (va saltato con log)."""
    return (not item_nsfw) or adult_enabled()


# baseModel Civitai → famiglia app (allineata a family_to_subfolder)
_CIVITAI_BASEMODEL_MAP = {
    "Flux.1 D": "Flux", "Flux.1 S": "Flux", "Flux.1 Kontext": "Flux", "Flux.2": "Flux",
    "SDXL 1.0": "SDXL", "SDXL Turbo": "SDXL", "Pony": "SDXL", "Illustrious": "SDXL",
    "SD 1.5": "SD1.5", "SD 1.4": "SD1.5", "SD 2.1": "SD1.5",
    "SD 3": "SD3", "SD 3.5": "SD3",
    "LTXV": "LTX", "Wan Video 14B": "Wan21", "Wan Video 1.3B": "Wan21", "Wan Video 2.2": "Wan22",
    "Hunyuan Video": "Hunyuan", "Hunyuan 1": "Hunyuan",
}
# type Civitai → tipo catalogo
_CIVITAI_TYPE_MAP = {
    "LORA": "LoRA", "LoCon": "LoRA", "DoRA": "LoRA", "Checkpoint": "Checkpoint",
    "TextualInversion": "Embedding", "VAE": "VAE", "Controlnet": "ControlNet",
    "Upscaler": "Upscaler",
}
