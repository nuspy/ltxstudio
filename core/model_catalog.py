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
