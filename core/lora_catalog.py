"""Lettore dinamico del catalogo LoRA per LTX Studio.

Fonte dati: E:\\AI\\dictionary\\data\\catalog.json (prodotto dalla skill ai-catalog).
- Caricamento dinamico: ricarica sempre la versione più recente del file (mtime).
- Filtro: espone SOLO le LoRA compatibili con LTX-2 (architecture == "LTX", type == "LoRA").
- Trigger words: ripulite da placeholder non-trigger ("non necessaria", "nessuna", ...).
- Preview: URL Civitai (può essere .mp4) + livello NSFW per filtrare le thumbnail.

Modulo stdlib-only: nessuna dipendenza da ComfyUI/Qt, quindi testabile in isolamento.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(r"E:\AI\dictionary\data\catalog.json")

# Stringhe che NON sono vere trigger word (definizioni/placeholder messi nel catalogo).
# Confronto case-insensitive, normalizzato. Vedi requisito: evitare "non necessaria" ecc.
_TRIGGER_BLOCKLIST = {
    "", "none", "nessuna", "nessuno", "n/a", "na", "-", "—",
    "non necessaria", "non necessario", "non richiesta", "non richiesto",
    "not required", "not needed", "no trigger", "no triggers",
    "nessun trigger", "nessuna trigger", "nessuna parola trigger",
    "trigger non necessaria", "non serve",
}

_NSFW_HINTS = {
    "nsfw", "xxx", "adult", "nude", "nudity", "naked", "erotic", "porn",
    "sex", "sexual", "deepthroat", "penile", "penis", "pussy", "anal",
    "blowjob", "breast", "boob",
}


def _clean_triggers(raw: Any) -> list[str]:
    """Restituisce solo trigger word reali, scartando placeholder/definizioni."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        s = t.strip()
        norm = re.sub(r"\s+", " ", s).strip(" .!?,;:").lower()
        if norm in _TRIGGER_BLOCKLIST:
            continue
        if len(norm) < 2:  # singoli caratteri/segni
            continue
        out.append(s)
    # dedup preservando l'ordine
    seen: set[str] = set()
    uniq = []
    for t in out:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


@dataclass
class LoraEntry:
    sha256: str
    filename: str
    path: str
    architecture: str
    triggers: list[str] = field(default_factory=list)
    description: str = ""
    nsfw: bool = False
    preview_url: str | None = None
    preview_is_video: bool = False
    preview_nsfw_level: int = 0
    tags: list[str] = field(default_factory=list)
    downloads: int = 0
    civitai_url: str | None = None

    @property
    def name(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def comfy_name(self) -> str:
        """Nome lora come atteso da ComfyUI (relativo alla cartella loras), es. 'LTX-2\\file.safetensors'."""
        pl = self.path.replace("/", "\\")
        for marker in ("\\LoRA\\", "\\loras\\", "\\Lora\\"):
            i = pl.lower().find(marker.lower())
            if i != -1:
                return pl[i + len(marker):]
        return self.filename

    @property
    def has_real_triggers(self) -> bool:
        return len(self.triggers) > 0


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_nsfw(*parts: Any) -> bool:
    # FASE D weights-unification: euristica UNIFICATA (core/nsfw_policy) — la lista locale
    # _NSFW_HINTS resta come fallback se il modulo non è importabile.
    try:
        from core import nsfw_policy
        return nsfw_policy.looks_nsfw(*parts)
    except Exception:                                    # noqa: BLE001
        text = " ".join(str(p or "") for p in parts).lower()
        return any(k in text for k in _NSFW_HINTS)


def _entry_from_raw(raw: dict) -> LoraEntry:
    civ = raw.get("civitai") or {}
    previews = civ.get("preview_images") or []
    purl = None
    pvid = False
    plevel = 0
    if previews:
        first = previews[0] or {}
        purl = first.get("url")
        plevel = int(first.get("nsfw_level") or 0)
        pvid = bool(purl and purl.lower().split("?")[0].endswith((".mp4", ".webm")))

    # trigger: prima dal campo top-level, poi dal civitai
    triggers = _clean_triggers(raw.get("trigger_words"))
    if not triggers:
        triggers = _clean_triggers(civ.get("trigger_words"))

    nsfw_level = max(
        int(civ.get("nsfw_level") or civ.get("nsfwLevel") or 0),
        int(raw.get("nsfw_level") or 0),
        plevel,
    )
    return LoraEntry(
        sha256=raw.get("sha256") or raw.get("id") or "",
        filename=raw.get("filename") or os.path.basename(raw.get("path", "")),
        path=raw.get("path", ""),
        architecture=raw.get("architecture", ""),
        triggers=triggers,
        description=_strip_html(civ.get("description") or raw.get("instructions") or ""),
        nsfw=bool(raw.get("nsfw") or civ.get("nsfw") or nsfw_level >= 4
                  or _looks_nsfw(raw.get("filename"), raw.get("path"), civ.get("tags"))),
        preview_url=purl,
        preview_is_video=pvid,
        preview_nsfw_level=plevel,
        tags=list(civ.get("tags") or []),
        downloads=int(civ.get("downloads") or 0),
        civitai_url=civ.get("model_url"),
    )


class LoraCatalog:
    """Carica e filtra il catalogo. Ricarica dinamicamente quando il file cambia."""

    def __init__(self, path: Path = CATALOG_PATH, compatible_arch: str = "LTX"):
        self.path = Path(path)
        self.compatible_arch = compatible_arch
        self._mtime: float = 0.0
        self._entries: list[LoraEntry] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Catalogo non trovato: {self.path}")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        raw_entries = data.get("entries", data)  # supporta {entries:{}} o dict piatto
        if isinstance(raw_entries, dict):
            raw_iter = raw_entries.values()
        else:
            raw_iter = raw_entries
        out: list[LoraEntry] = []
        for raw in raw_iter:
            if not isinstance(raw, dict):
                continue
            if raw.get("type") != "LoRA":
                continue
            if (raw.get("architecture") or "").upper() != self.compatible_arch.upper():
                continue
            if raw.get("missing"):
                continue
            out.append(_entry_from_raw(raw))
        out.sort(key=lambda e: (e.preview_url is None, -e.downloads, e.name.lower()))
        self._entries = out
        self._mtime = self.path.stat().st_mtime

    def reload_if_changed(self) -> bool:
        """Ricarica se il file su disco è cambiato. True se ricaricato."""
        try:
            if self.path.stat().st_mtime != self._mtime:
                self.load()
                return True
        except OSError:
            pass
        return False

    def loras(self, allow_nsfw: bool = True) -> list[LoraEntry]:
        if allow_nsfw:
            return list(self._entries)
        return [e for e in self._entries if not e.nsfw]

    def preview_url_for(self, entry: LoraEntry, allow_nsfw: bool) -> str | None:
        """URL preview rispettando il filtro NSFW thumbnail (livello <=2 se SFW)."""
        if entry.preview_url is None:
            return None
        if not allow_nsfw and entry.preview_nsfw_level > 2:
            return None
        return entry.preview_url

    def missing_triggers_in_prompt(self, entry: LoraEntry, prompt: str) -> list[str]:
        """Se la LoRA ha trigger reali e NESSUNO è nel prompt, restituisce la lista
        dei trigger (per avvisare l'utente). Lista vuota = ok / nessun trigger."""
        if not entry.has_real_triggers:
            return []
        p = prompt.lower()
        if any(t.lower() in p for t in entry.triggers):
            return []
        return list(entry.triggers)


if __name__ == "__main__":
    cat = LoraCatalog()
    loras = cat.loras()
    print(f"LoRA LTX compatibili: {len(loras)}")
    print(f"  con preview: {sum(1 for e in loras if e.preview_url)}")
    print(f"  con trigger reali: {sum(1 for e in loras if e.has_real_triggers)}")
    print(f"  NSFW: {sum(1 for e in loras if e.nsfw)}")
    print("\nPrime 8:")
    for e in loras[:8]:
        trig = ", ".join(e.triggers) if e.triggers else "(nessun trigger reale)"
        pv = "video" if e.preview_is_video else ("img" if e.preview_url else "—")
        print(f"  - {e.name[:48]:48} | preview={pv:5} | nsfw={e.nsfw!s:5} | trig: {trig[:40]}")
    # test trigger check
    print("\nTest trigger-check (prompt vuoto):")
    for e in loras:
        if e.has_real_triggers:
            miss = cat.missing_triggers_in_prompt(e, "a cat walking")
            print(f"  {e.name[:40]:40} -> mancanti: {miss}")
            break
