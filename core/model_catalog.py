"""Lookup leggero del catalogo per i MODELLI (checkpoint/diffusion_models): architettura e
NSFW. Usato per filtrare modelli e LoRA per architettura + contenuto in tutte le sezioni.

- architettura: dal campo 'architecture' del catalog.json (affidabile, per basename).
- nsfw: flag civitai (sparso) OPPURE euristica sul nome (eros, nsfw, hardcore, …).
Ricarica dinamica su mtime. stdlib-only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CATALOG_PATH = Path(r"E:\AI\dictionary\data\catalog.json")

# Default GUI per ARCHITETTURA (sampler/scheduler/steps/cfg/clip_type) — fonte UNICA (review FASE G):
# prima vivevano hardcoded in frame1_tab._ARCH_DEFAULTS. Allineati ai workflow ufficiali Comfy-Org.
ARCH_DEFAULTS = {
    "FLUX":    {"sampler": "euler", "scheduler": "simple", "steps": 20, "cfg": 3.5, "clip_type": "flux"},
    "Z-Image": {"sampler": "res_multistep", "scheduler": "simple", "steps": 8, "cfg": 1.0, "clip_type": "lumina2"},
    "Ernie":   {"sampler": "euler", "scheduler": "simple", "steps": 20, "cfg": 4.0, "clip_type": "flux2"},
}

# Capability per ARCHITETTURA VIDEO — usata dalla GUI per abilitare/disabilitare i controlli quando
# l'utente cambia architettura (video_tab._on_arch_changed). LingBot NON supporta audio/a2v, controlnet,
# v2v/retake, storyboard a keyframe latenti; SUPPORTA invece il refiner (2 stadi) e l'extend a chunk.
# Vedi [[native-video-engine]] / piano LingBot.
ARCH_VIDEO_CAPS = {
    "ltx": {
        "t2v": True, "i2v": True, "ti2v": True, "a2v": True, "audio": True,
        "controlnet": True, "v2v": True, "retake": True, "storyboard": True,
        "extend": True, "refiner": False, "lipsync": True, "keyframes": True,
    },
    "lingbot": {
        "t2v": True, "i2v": True, "ti2v": True, "a2v": False, "audio": False,
        "controlnet": False, "v2v": False, "retake": False, "storyboard": False,
        "extend": True, "refiner": True, "lipsync": False, "keyframes": False,
    },
    # Krea Realtime: autoregressivo → t2v + v2v + extend/long NATIVI (num_blocks); niente audio/controlnet/
    # refiner/storyboard-latente.
    "krea_rt": {
        "t2v": True, "i2v": True, "ti2v": True, "a2v": False, "audio": False,
        "controlnet": False, "v2v": True, "retake": False, "storyboard": False,
        "extend": True, "refiner": False, "lipsync": False, "keyframes": False,
    },
}

# Capability per ARCHITETTURA IMMAGINE (analogo per le immagini). Krea 2 = solo t2i (niente inpaint/kontext/
# controlnet nel primo giro). Le arch FLUX/… coprono tutto (default).
ARCH_IMAGE_CAPS = {
    # Krea 2: t2i + editing per istruzione (single/multi, "kontext"-like) via Krea2EditPipeline.
    # inpaint mascherato/controlnet/pulid non ancora supportati.
    "krea2": {"t2i": True, "inpaint": False, "kontext": True, "edit": True, "controlnet": False, "pulid": False},
}


def video_caps(arch: str) -> dict:
    """Capability dell'architettura video (default = LTX se sconosciuta)."""
    return ARCH_VIDEO_CAPS.get((arch or "ltx").lower(), ARCH_VIDEO_CAPS["ltx"])


def image_caps(arch: str) -> dict:
    """Capability dell'architettura immagine. Default = tutto abilitato (FLUX/Z-Image/Ernie)."""
    _full = {"t2i": True, "inpaint": True, "kontext": True, "edit": True, "controlnet": True, "pulid": True}
    return ARCH_IMAGE_CAPS.get((arch or "").lower(), _full)

# Valori consigliati hardcoded per i modelli più comuni (con nota "supposed" se non da Civitai)
_RECOMMENDED: dict[str, dict] = {
    "ltx-2.3-22b-distilled": {"steps": 8, "cfg": 0.9, "note": "distilled"},
    "ltx-2.3-22b": {"steps": 30, "cfg": 3.0, "note": ""},
    # eros: finetune NSFW del 22B. A 8 step il SINGOLO SamplerCustomAdvanced (audio+video
    # congiunti in LTXVConcatAVLatent) resta sotto-denoised → video rumoroso E audio nativo
    # come ronzio/parlato incomprensibile. Serve la ricetta piena del 22B (≈32 step + LTXVScheduler,
    # CFG 1.0 con NAG), come per il dev/base: solo così l'audio nativo diventa intelligibile.
    "ltx2310eros": {"steps": 32, "cfg": 1.0, "note": "32 step + LTXVScheduler (audio+video puliti)"},
    "q4_k_m": {"steps": 6, "cfg": 0.9, "note": "gguf-distilled"},
    "q4_k_s": {"steps": 6, "cfg": 0.9, "note": "gguf-distilled"},
    "q4_k": {"steps": 6, "cfg": 0.9, "note": "gguf-distilled"},
    "flux2": {"steps": 20, "cfg": 3.5, "note": "supposed"},
    "flux": {"steps": 20, "cfg": 3.5, "note": "supposed"},
    # LingBot-Video (default ufficiali dal README: 40 step base, guidance 3, refiner 8 step)
    "lingbot": {"steps": 40, "cfg": 3.0, "note": "lingbot base"},
    "lingbot-dense": {"steps": 40, "cfg": 3.0, "note": "lingbot dense 1.3b"},
    "lingbot-moe": {"steps": 40, "cfg": 3.0, "note": "lingbot moe 30b + refiner 8 step"},
    # Krea 2 (immagini): Turbo distillato (8 step, cfg 0, mu 1.15) / Raw (52 step, cfg 3.5)
    "krea-2-turbo": {"steps": 8, "cfg": 0.0, "note": "krea2 turbo distilled (mu 1.15)"},
    "krea2-turbo": {"steps": 8, "cfg": 0.0, "note": "krea2 turbo distilled"},
    "krea-2-raw": {"steps": 52, "cfg": 3.5, "note": "krea2 raw (qualità/LoRA)"},
    "krea2-raw": {"steps": 52, "cfg": 3.5, "note": "krea2 raw"},
    # Krea Realtime (video): distillato ~4 step, autoregressivo
    "krea-realtime": {"steps": 4, "cfg": 1.0, "note": "krea realtime autoregressivo"},
    "krea_rt": {"steps": 4, "cfg": 1.0, "note": "krea realtime 14b"},
}


def _basename(path_or_name: str) -> str:
    return os.path.basename((path_or_name or "").replace("\\", "/")).lower()


def _lvl(x) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


class ModelCatalog:
    def __init__(self, path: Path = CATALOG_PATH):
        self.path = Path(path)
        self._mtime = 0.0
        self._arch: dict[str, str] = {}    # basename -> architettura
        self._nsfw: dict[str, bool] = {}   # basename -> nsfw (da catalogo)
        self.load()

    def load(self) -> None:
        self._arch = {}; self._nsfw = {}
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        entries = data.get("entries", data)
        for raw in (entries.values() if isinstance(entries, dict) else entries):
            if not isinstance(raw, dict) or raw.get("missing"):
                continue                       # salta le voci stale (file spostati/assenti)
            bn = _basename(raw.get("filename") or raw.get("path", ""))
            if not bn:
                continue
            arch = raw.get("architecture") or ""
            if arch and arch not in ("Unknown", "—") or bn not in self._arch:
                self._arch[bn] = arch          # preferisci un'architettura nota
            civ = raw.get("civitai") or {}
            sm = raw.get("swarmui_metadata") or {}
            nsfw = bool(
                raw.get("nsfw") is True        # campo scritto da scan.py
                or civ.get("nsfw") is True
                or _lvl(civ.get("nsfw_level") or civ.get("nsfwLevel")) >= 4
                or _lvl(raw.get("nsfw_level")) >= 4
                or _lvl(sm.get("preview_nsfw_level")) >= 4
            )
            self._nsfw[bn] = self._nsfw.get(bn, False) or nsfw   # OR: basta una voce NSFW
        try:
            self._mtime = self.path.stat().st_mtime
        except OSError:
            pass

    def reload_if_changed(self) -> None:
        try:
            if self.path.stat().st_mtime != self._mtime:
                self.load()
        except OSError:
            pass

    def arch_of(self, name: str) -> str:
        """Architettura del modello (es. 'Flux', 'Z-Image', 'Ernie', 'LTX'). '' se ignota."""
        return self._arch.get(_basename(name), "")

    def is_nsfw(self, name: str) -> bool:
        """True se il modello è NSFW SECONDO IL CATALOGO (flag top-level 'nsfw' o civitai).
        Nessuna euristica: i modelli senza dato nel catalogo NON sono considerati NSFW."""
        return bool(self._nsfw.get(_basename(name), False))

    def recommended_steps(self, name: str) -> "int | None":
        """Steps consigliati per il modello, o None se sconosciuto."""
        n = (name or "").lower()
        for key, vals in _RECOMMENDED.items():
            if key in n:
                return vals["steps"]
        return None

    def recommended_cfg(self, name: str) -> "float | None":
        """CFG consigliato per il modello, o None se sconosciuto."""
        n = (name or "").lower()
        for key, vals in _RECOMMENDED.items():
            if key in n:
                return vals["cfg"]
        return None

    def recommended_note(self, name: str) -> str:
        """Nota sul modello (es. 'distilled', 'gguf-distilled', 'supposed')."""
        n = (name or "").lower()
        for key, vals in _RECOMMENDED.items():
            if key in n:
                return vals.get("note", "")
        return ""
