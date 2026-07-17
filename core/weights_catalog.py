"""Catalogo pesi del wizard: lettura, destinazione type/family, presenza, scan pre-download.

Puro Python (no torch/hf). La destinazione è derivata dalla CARTELLA BASE scelta dall'utente:
  <base_dir>/<type>/<family>/<repo_basename>      (snapshot)
  <base_dir>/<type>/<family>/<filename>           (file)
I pesi tool-specifici hanno `tool_dest` (percorso fisso del modulo, ignora base_dir).
"""
from __future__ import annotations

import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "data" / "weights_catalog.json"
CONFIG = Path(__file__).resolve().parents[1] / "data" / "weights_config.json"
DEFAULT_BASE = r"E:\AI\Models"

IMG_EXT = (".safetensors", ".ckpt", ".pth", ".pt", ".bin", ".gguf", ".onnx")


def load() -> list[dict]:
    try:
        return json.loads(CATALOG.read_text("utf-8")).get("weights", [])
    except Exception:                                   # noqa: BLE001
        return []


# ── cartella base configurabile ──────────────────────────────────────────────
def get_base_dir() -> str:
    try:
        return json.loads(CONFIG.read_text("utf-8")).get("base_dir", DEFAULT_BASE)
    except Exception:                                   # noqa: BLE001
        return DEFAULT_BASE


def set_base_dir(path: str) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    try:
        cfg = json.loads(CONFIG.read_text("utf-8"))
    except Exception:                                   # noqa: BLE001
        pass
    cfg["base_dir"] = path
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", "utf-8")


# ── destinazione coerente type/family ────────────────────────────────────────
def repo_basename(entry: dict) -> str:
    return entry.get("repo_id", "").split("/")[-1]


def dest_for(entry: dict, base: str | None = None) -> Path:
    if entry.get("tool_dest"):
        return Path(entry["tool_dest"])
    base = base or get_base_dir()
    d = Path(base) / entry.get("type", "misc") / entry.get("family", "misc")
    if entry.get("kind") == "file":
        return d
    return d / repo_basename(entry)                     # snapshot → sottocartella col nome repo


def is_present(entry: dict, base: str | None = None) -> bool:
    d = dest_for(entry, base)
    if entry.get("kind") == "file":
        fn = entry.get("filename", "")
        return (d / fn).exists() if fn else False
    return d.is_dir() and any(d.iterdir())


def missing_ids_with_deps(entry: dict, base: str | None = None) -> list[str]:
    """id (entry + `deps` ricorsive) NON ancora presenti, in ordine di download (deps prima).
    Serve al download-on-select: es. krea_rt richiede anche la base wan → li scarica insieme."""
    by_id = {e["id"]: e for e in load()}
    out: list[str] = []
    seen: set[str] = set()

    def visit(eid: str) -> None:
        if eid in seen:
            return
        seen.add(eid)
        e = by_id.get(eid)
        if not e:
            return
        for dep in e.get("deps", []):               # dipendenze prima (base Wan prima del modello)
            visit(dep)
        if not is_present(e, base) and eid not in out:
            out.append(eid)

    visit(entry.get("id", ""))
    return out


def ready_with_deps(entry: dict, base: str | None = None) -> bool:
    """True se l'entry E tutte le sue dipendenze sono presenti."""
    return not missing_ids_with_deps(entry, base)


# ── scan pre-download: cerca se esiste già in una directory dell'utente ───────
def scan_existing(entry: dict, search_dir: str, max_depth: int = 6) -> str | None:
    """Cerca ricorsivamente in `search_dir` una cartella/file che corrisponda agli scan_names
    del peso. Ritorna il primo path trovato, altrimenti None."""
    root = Path(search_dir)
    if not root.is_dir():
        return None
    names = [n.lower() for n in entry.get("scan_names", []) if n]
    if not names:
        names = [repo_basename(entry).lower()]
    base_len = len(root.parts)
    for p in root.rglob("*"):
        try:
            if len(p.parts) - base_len > max_depth:
                continue
            pl = p.name.lower()
            if p.is_dir() and any(n in pl for n in names) and any(p.iterdir()):
                return str(p)
            if p.is_file() and p.suffix.lower() in IMG_EXT and any(n in pl for n in names):
                return str(p)
        except Exception:                               # noqa: BLE001
            continue
    return None


# ── mappa modulo → capability (feature_matrix) ───────────────────────────────
# Un modello è mostrato/scaricato solo se una delle sue capability è ATTIVA nell'edizione (core/features).
# La chiave è il campo `module` del catalogo; un'entry può sovrascrivere con `used_by` esplicito.
MODULE_CAPS: dict[str, list[str]] = {
    "Immagini": ["cap.gen_images"],
    "Inpaint": ["cap.inpaint"],
    "Editing": ["cap.inpaint"],
    "Video": ["cap.video"],
    "Modello 3D (TRELLIS)": ["cap.threed"],
    "Rig 3D": ["cap.threed"],
    "Mondo 3D (WorldGen)": ["cap.threed"],
    "Musica (HeartMuLa)": ["cap.music_song", "cap.music_video"],
    "Musica (ACE-Step)": ["cap.music_song"],
    "Musica (MMAudio)": ["cap.music_sonify"],
    "Voce (clone)": ["cap.audio_tts_clone"],
    "Upscale": ["cap.upscale_image"],
    "FaceRestore": ["cap.restore"],
    "Face": ["cap.gen_images"],
    "Segmentazione": ["cap.sam"],
    "ControlNet": ["cap.controlnet"],
    "TextEncoder": ["cap.gen_images", "cap.video"],
    "VAE": ["cap.gen_images", "cap.video"],
    "LoRA": ["cap.gen_images", "cap.video"],
}

# Probe di PRESENZA dei moduli opzionali a venv/tool separato: se il tool non è installato, i suoi pesi non
# vanno proposti. I moduli "core" (immagini/video/inpaint) sono sempre presenti → non elencati qui = presente.
_TOOL_PROBES: dict[str, list[str]] = {
    "Modello 3D (TRELLIS)": [r"E:\AI\Tools\TRELLIS2"],
    "Rig 3D": [r"E:\AI\Comfy_Nexus.bak\ComfyUI\custom_nodes\ComfyUI-SkinToken"],
    "Mondo 3D (WorldGen)": [r"E:\AI\Tools\WorldGen"],
    "Musica (HeartMuLa)": [r"E:\AI\Tools\heartlib"],
    "Musica (ACE-Step)": [r"E:\AI\Tools\ACE-Step", r"E:\AI\Tools\acestep"],
    "Voce (clone)": [r"E:\AI\Tools\CosyVoice", r"E:\AI\Tools\F5-TTS"],
}


def caps_for(entry: dict) -> list[str]:
    """Capability che USANO questo peso (per il gating d'edizione). `used_by` esplicito o derivato dal modulo."""
    if entry.get("used_by"):
        return list(entry["used_by"])
    return MODULE_CAPS.get(entry.get("module", ""), [])


def module_present(module: str) -> bool:
    """True se il modulo è disponibile su questa macchina (core = sempre; tool-venv = solo se installato)."""
    probes = _TOOL_PROBES.get(module)
    if not probes:
        return True                                         # modulo core / sempre presente
    from pathlib import Path as _P
    return any(_P(p).exists() for p in probes)


def _cap_enabled(cap: str) -> bool:
    try:
        from core import features
        return bool(features.is_enabled(cap))
    except Exception:                                       # noqa: BLE001
        return True                                         # dev/fail-open: tutto attivo


def wanted(entry: dict, *, check_edition: bool = True, check_present: bool = True,
           allows_adult: bool | None = None) -> bool:
    """Il peso serve a QUESTA installazione? (edizione: una sua cap attiva) AND (modulo presente) AND
    (sensibilità: adult solo se l'edizione lo consente)."""
    if entry.get("sensitivity") == "adult":
        if allows_adult is None:
            try:
                from core import features
                allows_adult = bool(features.allows_adult())
            except Exception:                               # noqa: BLE001
                allows_adult = True
        if not allows_adult:
            return False
    if check_present and not module_present(entry.get("module", "")):
        return False
    if check_edition:
        caps = caps_for(entry)
        if caps and not any(_cap_enabled(c) for c in caps):
            return False
    return True


def by_module(*, only_wanted: bool = False, **kw) -> dict[str, list[dict]]:
    """Raggruppa i pesi per modulo. `only_wanted=True` → solo i pesi usati dai moduli presenti nell'edizione."""
    groups: dict[str, list[dict]] = {}
    for e in load():
        if only_wanted and not wanted(e, **kw):
            continue
        groups.setdefault(e.get("module", "Altro"), []).append(e)
    return groups
