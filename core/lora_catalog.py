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
    display_name: str = ""          # nome mostrato se il file ha un nome generico (es. diffusion_pytorch_model)

    @property
    def name(self) -> str:
        return self.display_name or os.path.splitext(self.filename)[0]

    @property
    def comfy_name(self) -> str:
        """Nome lora come atteso da ComfyUI (relativo alla cartella loras), es. 'LTX-2\\file.safetensors'."""
        pl = self.path.replace("/", "\\")
        for marker in ("\\LoRA\\", "\\loras\\", "\\Lora\\"):
            i = pl.lower().find(marker.lower())
            if i != -1:
                return pl[i + len(marker):]
        # file fuori dalla cartella LoRA (es. Checkpoints\DreamO\...): il solo nome non si risolve → path assoluto
        # (i motori FLUX/Krea2/Qwen21 accettano path assoluti)
        if os.path.isabs(self.path) and os.path.isfile(self.path):
            return self.path
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
                  or _looks_nsfw(raw.get("filename"), raw.get("path"), civ.get("tags"))
                  or (raw.get("filename") or "").lower() in _overrides()["adult"]),
        preview_url=purl,
        preview_is_video=pvid,
        preview_nsfw_level=plevel,
        tags=list(civ.get("tags") or []),
        downloads=int(civ.get("downloads") or 0),
        civitai_url=civ.get("model_url"),
    )


# Architetture che lo scanner del dizionario non distingue ancora: arch richiesta → (arch del dizionario,
# regola sul percorso/nome). Es. le LoRA di Qwen-Image 2.1 risultano «Qwen» come quelle di Qwen-Image 1.
_ARCH_ALIASES = {
    "QWEN21": ("QWEN", re.compile(r"[\\/]qwen21[\\/]|qwen[\s\-_]?image[\s\-_]?2[\.\-_]?1|qwen[\s\-_]?2[\.\-_]1",
                                  re.IGNORECASE)),
}


def _arch_matches(raw: dict, wanted: str) -> bool:
    if wanted.upper() in ("WAN21", "WAN22"):
        return False                       # le LoRA Wan vengono dalla scansione cartelle (_folder_loras)
    ovr = _overrides()["arch"].get((raw.get("filename") or "").lower())
    if ovr:
        return ovr.upper() == wanted.upper()
    arch = (raw.get("architecture") or "").upper()
    if arch == wanted.upper():
        return True
    alias = _ARCH_ALIASES.get(wanted.upper())
    return bool(alias and arch == alias[0]
                and alias[1].search(f"{raw.get('path', '')} {raw.get('filename', '')}"))


# arch del pannello LoRA → family del registro pesi: il pannello video dice «LTX» per 2.3 e 2.5
_REGISTRY_FAMILIES = {"LTX": ("LTX", "LTX-2", "LTX-2.5")}


# IC-LoRA (guidate da un video di controllo) e file ausiliari: NON sono LoRA «normali» da applicare a un t2v/i2v
# (senza la guida producono rumore). Si usano dal tab «Modifica Video AI» / ControlNet, non dal pannello LoRA.
# Restano visibili le LoRA camera-control / CameraControls (LoRA normali t2v/i2v).
# `inpaint_masked`: LoRA dell'inpaint video (t2v/r2v, task 8) guidate dal video+maschera SAM2 di «Modifica Video AI»
_IC_LORA_MARKERS = ("ic-lora", "iclora", "ic_lora", "scene-emb", "inpaint_masked")


def is_ic_lora_file(*names: str) -> bool:
    """True se nome/percorso indica una IC-LoRA (ltx-*ic-lora*, Cameraman, ReStyle, ...) o il file scene-emb HDR."""
    return any(m in (n or "").lower() for n in names for m in _IC_LORA_MARKERS)


def _registry_loras(arch: str) -> list[LoraEntry]:
    """LoRA del registro pesi (data/weights_catalog.json) già su disco per questa famiglia: compaiono anche
    prima che lo scanner del dizionario le abbia viste. 18+ = `sensitivity: adult`. Le LoRA di distillazione
    (tag `distill_lora`) restano fuori: hanno il loro selettore «Velocità»."""
    try:
        from core import model_paths as mp
        from core import weights_catalog as wc
        root = str(mp.models_root())
        out = []
        fams = {f.upper() for f in _REGISTRY_FAMILIES.get(arch.upper(), (arch,))}
        for e in wc.load():
            if e.get("type") != "LoRA" or (e.get("family") or "").upper() not in fams:
                continue
            # fuori dal pannello LoRA: distillazione («Velocità») e LoRA di FUNZIONE (upscale_video = guida h3upscale,
            # che ha il suo posto nel tab Definizione video e non è una LoRA di stile per generare)
            if {"distill_lora", "upscale_video"} & set(e.get("use") or []) or not wc.is_present(e, root):
                continue
            path = str(Path(wc.dest_for(e, root)) / e["filename"])
            out.append(LoraEntry(sha256=e.get("sha256") or e["id"], filename=e["filename"], path=path,
                                 architecture=arch, description=e.get("note") or "",
                                 triggers=_clean_triggers(e.get("triggers")),    # catalogo: anche più d'una
                                 civitai_url=e.get("page"),
                                 nsfw=e.get("sensitivity") == "adult" or _looks_nsfw(e["filename"])))
        return out
    except Exception:                                    # noqa: BLE001 — registro assente: solo il dizionario
        return []


# ── correzioni locali al dizionario (data/lora_overrides.json) ──────────────────────────────────────────
_OVR_PATH = Path(__file__).resolve().parents[1] / "data" / "lora_overrides.json"
_OVR_CACHE: tuple[float, dict] = (-1.0, {})


def _overrides() -> dict:
    """adult (nomi file 18+), arch (nome file → arch pannello), extra (file fuori cartella LoRA). Cache per mtime."""
    global _OVR_CACHE
    try:
        mt = _OVR_PATH.stat().st_mtime
        if mt != _OVR_CACHE[0]:
            d = json.loads(_OVR_PATH.read_text(encoding="utf-8"))
            _OVR_CACHE = (mt, {"adult": {str(n).lower() for n in d.get("adult", [])},
                               "arch": {str(k).lower(): v for k, v in (d.get("arch") or {}).items()},
                               "extra": list(d.get("extra") or [])})
        return _OVR_CACHE[1]
    except (OSError, ValueError):
        return {"adult": set(), "arch": {}, "extra": []}


# Pannelli le cui LoRA NON si fidano solo del dizionario (scanner esterno, spesso indietro sui file spostati):
# arch → sottocartelle di <radice>\LoRA da scandire. Wan: «Wan21» = MultiTalk/Wan 2.1, «Wan22» = Wan 2.2
# (le LoRA «Wan2_2-…» stanno in LoRA\Wan ma sono 2.2: le distingue il nome, vedi _wan_arch_of).
_FOLDERS = {"FLUX": ("Flux",), "FLUX3": ("Flux3", "Flux"), "WAN21": ("Wan",), "WAN22": ("Wan", "Wan2.2")}
_LORA_EXT = (".safetensors", ".pt", ".ckpt")
_WAN22_NAME = re.compile(r"wan2[_.\-]?2|high[_-]?noise|low[_-]?noise", re.IGNORECASE)


def _wan_arch_of(path: str, filename: str) -> str:
    ovr = _overrides()["arch"].get(filename.lower())
    if ovr:
        return ovr
    if f"{os.sep}wan2.2{os.sep}" in path.lower() or _WAN22_NAME.search(filename):
        return "Wan22"
    return "Wan21"


_FLUX3_LORA_KEYS = re.compile(r"(^|\.)(content_)?mode_blocks\.(image|image_cond)\.\d+\.|(^|\.)txt_mode_blocks\.\d+\.")
_LORA_FAMILY_CACHE: dict = {}


def lora_family_from_keys(path: str) -> str:
    """Famiglia di una LoRA dalle CHIAVI dell'header (mai dai metadati, REGOLA 14/10): «flux3» se ha i blocchi per
    modalità di FLUX.3 Image Klein (nativo `content_mode_blocks`/`txt_mode_blocks` o diffusers `mode_blocks`), altrimenti
    "" (non dice nulla: le altre famiglie le decidono dizionario/cartella). Cache per (path, mtime, size)."""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    sig = (path, st.st_mtime_ns, st.st_size)
    if sig in _LORA_FAMILY_CACHE:
        return _LORA_FAMILY_CACHE[sig]
    fam = ""
    try:
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if 0 < n < 100_000_000:
                keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
                if any(_FLUX3_LORA_KEYS.search(k) for k in keys):
                    fam = "flux3"
    except Exception:                                    # noqa: BLE001 — non è un safetensors leggibile
        fam = ""
    if len(_LORA_FAMILY_CACHE) > 4096:
        _LORA_FAMILY_CACHE.clear()
    _LORA_FAMILY_CACHE[sig] = fam
    return fam


def _is_adult(filename: str, *extra: Any) -> bool:
    return filename.lower() in _overrides()["adult"] or _looks_nsfw(filename, *extra)


def _models_root() -> Path:
    try:
        from core import model_paths as mp
        return Path(mp.models_root())
    except Exception:                                    # noqa: BLE001
        return Path(r"E:\AI\Models")


def _folder_loras(arch: str, seen: set[str], meta: dict[str, dict]) -> list[LoraEntry]:
    """LoRA presenti su disco nelle cartelle dell'arch ma assenti dal dizionario (o con arch sbagliata lì).
    `meta` = voci del dizionario per nome file (trigger/descrizione/preview quando esistono)."""
    folders = _FOLDERS.get(arch.upper())
    if not folders:
        return []
    out: list[LoraEntry] = []
    for sub in folders:
        d = _models_root() / "LoRA" / sub
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.suffix.lower() not in _LORA_EXT:
                continue
            if arch.upper().startswith("WAN") and _wan_arch_of(str(f), f.name).upper() != arch.upper():
                continue
            key = os.path.normcase(str(f))
            if key in seen:
                continue
            raw = meta.get(f.name.lower())
            if raw:
                e = _entry_from_raw(dict(raw, path=str(f), architecture=arch))
            else:
                e = LoraEntry(sha256=f"disk:{key}", filename=f.name, path=str(f), architecture=arch)
            e.nsfw = e.nsfw or _is_adult(f.name, str(f))
            out.append(e)
            seen.add(key)
    return out


def _extra_loras(arch: str, seen: set[str]) -> list[LoraEntry]:
    """File fuori dalle cartelle LoRA che valgono come LoRA di questo pannello (es. FLUX.1-Turbo-Alpha, finito
    in Checkpoints\\DreamO): vale il primo percorso candidato esistente."""
    root = _models_root()
    out: list[LoraEntry] = []
    for x in _overrides()["extra"]:
        if str(x.get("arch", "")).upper() != arch.upper():
            continue
        f = next((root / c for c in x.get("paths", []) if (root / c).is_file()), None)
        if f is None or os.path.normcase(str(f)) in seen:
            continue
        seen.add(os.path.normcase(str(f)))
        out.append(LoraEntry(sha256=f"extra:{x.get('id', f.name)}", filename=f.name, path=str(f), architecture=arch,
                             display_name=x.get("display_name", ""), description=x.get("description", ""),
                             triggers=_clean_triggers(x.get("triggers")),
                             nsfw=bool(x.get("nsfw")) or _is_adult(f.name)))
    return out


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
        meta: dict[str, dict] = {}           # voci LoRA del dizionario per nome file (metadati per la scansione cartelle)
        for raw in raw_iter:
            if not isinstance(raw, dict):
                continue
            if raw.get("type") != "LoRA":
                continue
            if raw.get("missing"):
                continue
            # il dizionario può restare indietro sui file spostati/rinominati: voce fantasma = file inesistente
            if not os.path.isfile(raw.get("path", "")):
                continue
            meta.setdefault((raw.get("filename") or "").lower(), raw)
            if not _arch_matches(raw, self.compatible_arch):
                continue
            out.append(_entry_from_raw(raw))
        seen = {os.path.normcase(e.path) for e in out}
        out += [e for e in _registry_loras(self.compatible_arch) if os.path.normcase(e.path) not in seen]
        seen = {os.path.normcase(e.path) for e in out}
        out += _folder_loras(self.compatible_arch, seen, meta)
        out += _extra_loras(self.compatible_arch, seen)
        out = [e for e in out if not is_ic_lora_file(e.filename, e.path)]    # le IC-LoRA non sono LoRA normali
        if self.compatible_arch.upper() in ("FLUX", "FLUX3"):
            # FLUX.3 ≠ FLUX.1: una LoRA FLUX.3 (chiavi a blocchi per modalità) compare SOLO sotto «FLUX.3 Klein», mai fra
            # le LoRA FLUX.1/2 anche se sta nella stessa cartella o il dizionario la chiama «FLUX»
            want3 = self.compatible_arch.upper() == "FLUX3"
            out = [e for e in out if (lora_family_from_keys(e.path) == "flux3") == want3]
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
