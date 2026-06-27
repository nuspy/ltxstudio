"""Motore TTS per LTX Studio: sintesi vocale + storage voci salvate.

Backend di default: Microsoft Edge TTS (`pip install edge-tts`). Free, no API key,
multilingua nativa (IT incluso), supporta SSML per pause/velocità/tono/volume.
Le voci sono "preset" di Microsoft (~400 disponibili); per voice cloning vero
serve installare un custom node ComfyUI (CosyVoice 2 / F5-TTS) — questo modulo
espone uno strato uniforme per quando saranno disponibili.

Storage voci salvate: `E:\\AI\\ltxstudio\\voices\\<slug>.json`.
- nome (display), engine (`edge` / `cosyvoice` / `f5`), voice_id (edge: nome
  voce MS; clone: percorso al sample WAV), lingua, note.
Le voci di sistema (preset Edge) sono caricate dinamicamente; quelle utente
custom sono sovrapposte sopra.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
SAMPLES_DIR = VOICES_DIR / "samples"     # campioni vocali persistiti (associati alle voci)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "tts"
# CosyVoice/F5/XTTS: il tokenizer vocale (Whisper) ha un tetto FISSO di 1500 frame = 30s.
# Un campione più lungo crasha (ONNX "right operand cannot broadcast"). Tronchiamo a un valore
# sotto soglia con margine. Per il zero-shot cloning bastano comunque 3-15s di parlato pulito.
MAX_CLONE_SAMPLE_S = 28.0
# Attesa massima per la sintesi via ComfyUI. Generosa: il PRIMO avvio scarica i pesi del
# modello (CosyVoice-300M ~1.5GB da modelscope) + carica gli ONNX. Le run successive sono rapide.
TTS_SYNTH_TIMEOUT_S = 1200.0


def _probe_audio_duration(path: str | Path) -> float | None:
    """Durata in secondi via ffprobe; None se non determinabile."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=20)
        return float((r.stdout or "").strip())
    except Exception:
        return None


def prepare_clone_sample(sample_path: str | Path, max_seconds: float = MAX_CLONE_SAMPLE_S) -> str:
    """Restituisce un campione pronto per il cloning, garantendo durata ≤ max_seconds.
    Se il campione è già abbastanza corto lo ritorna invariato; altrimenti ne tronca i
    primi `max_seconds` secondi (16 kHz mono WAV) in una cache, evitando il crash del
    tokenizer Whisper di CosyVoice/F5/XTTS (limite 1500 frame = 30 s)."""
    src = Path(sample_path)
    dur = _probe_audio_duration(src)
    if dur is not None and dur <= max_seconds + 0.05:
        return str(src)
    # tronca + normalizza a 16k mono (ciò che gli engine usano internamente)
    cache = OUTPUT_DIR / "clone_cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{src.stem}_{int(max_seconds)}s16k.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-t", f"{max_seconds:.2f}", "-i", str(src),
             "-ac", "1", "-ar", "16000", str(out)],
            capture_output=True, text=True, timeout=60, check=True)
        return str(out)
    except Exception:
        return str(src)   # se ffmpeg fallisce, prova comunque con l'originale


# ────────────────────────────────────────────────────────────────────────────
# Engine registry — rilevamento dinamico backend disponibili
# ────────────────────────────────────────────────────────────────────────────
# Ogni entry: (id, display_name, supports_cloning, supports_emotions, notes)
# La disponibilità reale è risolta a runtime dalla funzione `available_engines()`.
ENGINES_META = [
    ("kokoro",     "Kokoro (Handy-API) — IT nativo, REST",        False, False, "Handy-API :8720, nessun ComfyUI, voci IT (if_sara/im_nicola)"),
    ("handy_clone", "Voice clone (Handy-API) — Chatterbox/ZONOS2", True,  False, "Handy-API :8720, cloning zero-shot nativo, nessun ComfyUI"),
    ("edge",       "Edge TTS — Microsoft (preset, no cloning)",   False, True,  "Apache compatibile, IT nativo, ~400 voci preset"),
    ("cosyvoice2", "CosyVoice 2 — venv nativo / Handy",            True,  True,  "venv standalone o Handy-API (no ComfyUI)"),
    ("f5tts",      "F5-TTS — SOTA qualità (EN nativo, IT FT)",     True,  False, "MIT, in IT solo via fine-tune community"),
    ("xtts2",      "XTTS-v2 — Coqui (CC-BY-NC)",                   True,  True,  "Licenza non commerciale, 17 lingue native"),
    ("fish",       "Fish Speech v1.5 — qualità + Apache 2.0",      True,  True,  "Apache 2.0, codec compatto"),
    ("vibevoice",  "VibeVoice — Microsoft (multi-speaker)",        True,  True,  "MIT, ottimo per dialoghi lunghi"),
]


def available_engines() -> list[tuple[str, str, bool, bool, bool]]:
    """Ritorna [(id, display, supports_cloning, supports_emotions, is_installed), ...].
    `is_installed` è True quando il backend Python o il nodo ComfyUI è rilevato."""
    out = []
    for eid, disp, clone, emo, _ in ENGINES_META:
        out.append((eid, disp, clone, emo, _engine_installed(eid)))
    return out


def _engine_installed(eid: str) -> bool:
    if eid in ("kokoro", "handy_clone"):                    # nativi via Handy-API (REST :8720)
        try:
            from core import handy_api
            return handy_api.available()
        except Exception:                                  # noqa: BLE001
            return False
    if eid == "edge":
        return edge_available()
    if eid == "cosyvoice2":                                 # nativo: venv standalone OPPURE Handy
        try:
            from core import cosyvoice_service
            if cosyvoice_service.is_installed():
                return True
        except Exception:                                  # noqa: BLE001
            pass
        try:
            from core import handy_api
            return handy_api.available()
        except Exception:                                  # noqa: BLE001
            return _comfy_node_present(("CosyVoiceNode", "CosyVoice2Run", "CosyVoice2_Run", "CosyVoiceRun"))
    if eid == "f5tts":
        return _comfy_node_present(("F5TTSCreate", "F5TTSAudio", "F5TTS_Run"))
    if eid == "xtts2":
        return _comfy_node_present(("XTTSGenerate", "XTTS_v2", "XTTS"))
    if eid == "fish":
        return _comfy_node_present(("FishSpeechRun", "FishSpeech", "FishSpeech_v15"))
    if eid == "vibevoice":
        return _comfy_node_present(("VibeVoiceGenerate", "VibeVoice", "VibeVoiceRun"))
    return False


_COMFY_NODES_CACHE: set[str] | None = None


def _comfy_node_present(candidates: tuple[str, ...]) -> bool:
    """Verifica se uno dei nomi `candidates` esiste tra i NODE_CLASS_MAPPINGS di ComfyUI.
    Cache statica: la prima chiamata ispeziona custom_nodes/. Veloce + safe in headless."""
    global _COMFY_NODES_CACHE
    if _COMFY_NODES_CACHE is None:
        _COMFY_NODES_CACHE = _scan_comfy_nodes()
    return any(c in _COMFY_NODES_CACHE for c in candidates)


def _scan_comfy_nodes() -> set[str]:
    """Best-effort: cerca import statements + NODE_CLASS_MAPPINGS nei custom_nodes."""
    out: set[str] = set()
    try:
        from pathlib import Path as _P
        candidates = [
            _P(r"E:\AI\Comfy_Nexus\ComfyUI\custom_nodes"),
            _P(r"E:\AI\ComfyUI\ComfyUI\custom_nodes"),
        ]
        for root in candidates:
            if not root.exists():
                continue
            import re as _re
            for py in root.rglob("__init__.py"):
                try:
                    txt = py.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for m in _re.finditer(r"NODE_CLASS_MAPPINGS\s*=\s*\{([^}]+)\}", txt, flags=_re.S):
                    body = m.group(1)
                    out.update(_re.findall(r"['\"]([A-Za-z0-9_+\- ]+)['\"]\s*:", body))
            break  # primo root valido basta
    except Exception:
        pass
    return out


def invalidate_engine_cache() -> None:
    """Forza un nuovo scan dei custom_nodes al prossimo `_engine_installed()`."""
    global _COMFY_NODES_CACHE
    _COMFY_NODES_CACHE = None


# ────────────────────────────────────────────────────────────────────────────
# Installazione automatica dei custom node TTS
# ────────────────────────────────────────────────────────────────────────────
# Per ogni engine: repo git da clonare in custom_nodes/, eventuale requirements.txt.
# La cartella di destinazione viene desunta dal nome del repo.
ENGINE_REPOS: dict[str, dict] = {
    "cosyvoice2": {
        "url":  "https://github.com/AIFSH/CosyVoice-ComfyUI.git",
        "name": "CosyVoice-ComfyUI",
        "note": "CosyVoice 2 (AIFSH). Scarica il modello a runtime (~3 GB) al primo uso.",
    },
    "f5tts": {
        "url":  "https://github.com/niknah/ComfyUI-F5-TTS.git",
        "name": "ComfyUI-F5-TTS",
        "note": "F5-TTS (niknah). Modelli da HuggingFace ~1 GB al primo uso.",
    },
    "xtts2": {
        "url":  "https://github.com/AIFSH/ComfyUI-XTTS.git",
        "name": "ComfyUI-XTTS",
        "note": "XTTS-v2 (AIFSH). Modello ~2 GB al primo uso. Licenza CC-BY-NC.",
    },
    "fish": {
        "url":  "https://github.com/AIFSH/ComfyUI-FishSpeech.git",
        "name": "ComfyUI-FishSpeech",
        "note": "Fish Speech v1.5 (AIFSH). Modello ~1.5 GB al primo uso.",
    },
    "vibevoice": {
        "url":  "https://github.com/Enemyx-net/VibeVoice-ComfyUI.git",
        "name": "VibeVoice-ComfyUI",
        "note": "VibeVoice (Microsoft via Enemyx-net). Modello ~3 GB al primo uso.",
    },
}


def custom_nodes_root() -> Path | None:
    """Ritorna il path della cartella custom_nodes di ComfyUI (None se non trovata)."""
    candidates = [
        Path(r"E:\AI\Comfy_Nexus\ComfyUI\custom_nodes"),
        Path(r"E:\AI\ComfyUI\ComfyUI\custom_nodes"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def venv_python() -> Path | None:
    """Python del venv di ComfyUI (per pip install dei requirements del custom node)."""
    candidates = [
        Path(r"E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe"),
        Path(r"E:\AI\ComfyUI\ComfyUI\venv\Scripts\python.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def install_engine(engine_id: str, *, log=print) -> tuple[bool, str]:
    """Installa il custom node ComfyUI di un engine TTS.

    Step:
    1. Verifica che `git` sia disponibile
    2. Trova `custom_nodes/` di ComfyUI
    3. `git clone` del repo (skip se già presente → tenta `git pull`)
    4. Installa `requirements.txt` se presente nel python del venv di ComfyUI
    5. Invalida la cache dei NODE_CLASS_MAPPINGS

    Ritorna (ok, message). `message` contiene log multilinea adatto al dialog.
    """
    import shutil, subprocess
    if engine_id not in ENGINE_REPOS:
        return False, f"Nessun installer noto per l'engine '{engine_id}'."
    if shutil.which("git") is None:
        return False, "❌ Git non trovato nel PATH. Installa Git da https://git-scm.com/ e riprova."
    root = custom_nodes_root()
    if root is None:
        return False, "❌ Cartella ComfyUI/custom_nodes non trovata."
    spec = ENGINE_REPOS[engine_id]
    target = root / spec["name"]
    log_lines: list[str] = [f"📦 Installazione {engine_id} → {target}"]
    try:
        if target.exists():
            log_lines.append("• Repo già presente. Tento `git pull`…")
            res = subprocess.run(["git", "pull"], cwd=str(target),
                                  capture_output=True, text=True, timeout=120)
            log_lines.append(res.stdout.strip() or res.stderr.strip() or "(nessun output)")
        else:
            log_lines.append(f"• git clone {spec['url']}")
            res = subprocess.run(["git", "clone", spec["url"], str(target)],
                                  capture_output=True, text=True, timeout=600)
            if res.returncode != 0:
                return False, "\n".join(log_lines + [f"❌ git clone fallito:\n{res.stderr.strip()}"])
            log_lines.append(res.stdout.strip() or "✓ clonato")
        # requirements.txt
        req = target / "requirements.txt"
        py = venv_python()
        if req.exists() and py:
            log_lines.append(f"• pip install -r requirements.txt nel venv di ComfyUI")
            res = subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)],
                                  capture_output=True, text=True, timeout=900)
            tail = (res.stdout + res.stderr).strip().splitlines()[-6:]
            log_lines.extend(tail)
            if res.returncode != 0:
                return False, "\n".join(log_lines + ["❌ pip install fallito (vedi sopra)."])
        elif req.exists() and not py:
            log_lines.append("⚠ requirements.txt trovato ma il venv di ComfyUI non è stato individuato.")
            log_lines.append("   Installa manualmente: pip install -r " + str(req))
        invalidate_engine_cache()
        log_lines.append("")
        log_lines.append("✅ Installazione completata. Riavvia ComfyUI per caricare il nuovo nodo.")
        return True, "\n".join(log_lines)
    except subprocess.TimeoutExpired:
        return False, "\n".join(log_lines + ["❌ Timeout: il comando ha impiegato troppo. Riprova."])
    except Exception as e:
        return False, "\n".join(log_lines + [f"❌ Errore inatteso: {e}"])


def _slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_").lower()
    return s or "voice"


@dataclass
class VoicePreset:
    """Una voce selezionabile per il TTS."""
    name: str                       # nome utente / display
    engine: str = "edge"            # edge / cosyvoice / f5 / xtts
    voice_id: str = ""              # edge: voice short name; clone: path al sample
    language: str = "it-IT"
    note: str = ""
    sample_path: str | None = None  # per cloning: percorso al sample WAV
    sample_transcript: str = ""     # per cloning: trascrizione del sample (migliora la qualità)
    custom: bool = True             # False per preset di sistema (non eliminabili)
    file: Path | None = field(default=None, repr=False)

    def slug(self) -> str:
        return _slug(self.name)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "engine": self.engine, "voice_id": self.voice_id,
            "language": self.language, "note": self.note,
            "sample_path": self.sample_path, "sample_transcript": self.sample_transcript,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, d: dict, file: Path | None = None) -> "VoicePreset":
        return cls(name=d.get("name", "?"), engine=d.get("engine", "edge"),
                   voice_id=d.get("voice_id", ""), language=d.get("language", "it-IT"),
                   note=d.get("note", ""), sample_path=d.get("sample_path"),
                   sample_transcript=d.get("sample_transcript", ""),
                   custom=bool(d.get("custom", True)), file=file)


# ────────────────────────────────────────────────────────────────────────────
# CRUD voci utente — filesystem-based
# ────────────────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_user_voices() -> list[VoicePreset]:
    ensure_dirs()
    out: list[VoicePreset] = []
    for f in sorted(VOICES_DIR.glob("*.json")):
        try:
            out.append(VoicePreset.from_dict(json.loads(f.read_text(encoding="utf-8")), file=f))
        except Exception:
            continue
    return out


def copy_sample_into_managed(src: str, slug: str) -> str:
    """Copia il campione audio in SAMPLES_DIR/<slug>_<8hex>.<ext> (nome unico per-voce, così
    sopravvive a rinomina/duplica) e ritorna il path gestito. Usa shutil.copy2."""
    ensure_dirs()
    ext = Path(src).suffix.lower() or ".wav"
    dst = SAMPLES_DIR / f"{slug}_{uuid.uuid4().hex[:8]}{ext}"
    shutil.copy2(src, dst)
    return str(dst)


def _is_managed_sample(path: str | None) -> bool:
    if not path:
        return False
    try:
        return Path(path).resolve().parent == SAMPLES_DIR.resolve()
    except Exception:
        return False


def save_voice(v: VoicePreset, *, copy_sample_from: str | None = None) -> Path:
    """Salva la voce su disco. Se `copy_sample_from` è un file fuori da SAMPLES_DIR, lo copia
    nella cartella gestita e aggiorna `v.sample_path` (persistenza del campione). Idempotente
    se il sample è già gestito."""
    ensure_dirs()
    if copy_sample_from and not _is_managed_sample(copy_sample_from):
        if Path(copy_sample_from).exists():
            v.sample_path = copy_sample_into_managed(copy_sample_from, v.slug())
    target = v.file or (VOICES_DIR / f"{v.slug()}.json")
    target.write_text(json.dumps(v.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    v.file = target
    return target


def delete_voice(v: VoicePreset) -> bool:
    ok = False
    # rimuovi anche il campione gestito (mai file fuori da SAMPLES_DIR)
    if _is_managed_sample(v.sample_path):
        try:
            Path(v.sample_path).unlink(missing_ok=True)
        except Exception:
            pass
    if v.file and v.file.exists():
        v.file.unlink()
        ok = True
    return ok


# ────────────────────────────────────────────────────────────────────────────
# Edge TTS — backend di default
# ────────────────────────────────────────────────────────────────────────────

_EDGE_AVAILABLE: bool | None = None


def edge_available() -> bool:
    global _EDGE_AVAILABLE
    if _EDGE_AVAILABLE is None:
        try:
            import edge_tts  # noqa: F401
            _EDGE_AVAILABLE = True
        except Exception:
            _EDGE_AVAILABLE = False
    return _EDGE_AVAILABLE


def edge_list_voices(language_prefix: str = "it") -> list[VoicePreset]:
    """Elenca le voci Edge TTS disponibili (preset Microsoft). Filtrate per lingua."""
    if not edge_available():
        return []
    import edge_tts
    try:
        voices = asyncio.run(edge_tts.list_voices())
    except RuntimeError:
        # event loop già attivo (es. dentro Qt) → run in thread
        import threading
        result: list = []
        def _runner():
            result.append(asyncio.new_event_loop().run_until_complete(edge_tts.list_voices()))
        t = threading.Thread(target=_runner); t.start(); t.join()
        voices = result[0]
    out: list[VoicePreset] = []
    for v in voices:
        loc = v.get("Locale", "")
        if language_prefix and not loc.lower().startswith(language_prefix.lower()):
            continue
        short = v.get("ShortName", "")
        gender = v.get("Gender", "?")
        # nome leggibile: "Elsa (it-IT, Female)"
        loc_short = loc
        display = f"{short.split('-')[-1].replace('Neural','')}  ({loc_short}, {gender})"
        out.append(VoicePreset(name=display, engine="edge", voice_id=short,
                               language=loc, custom=False))
    return out


def edge_synthesize(text: str, voice_id: str, output_wav: Path, *,
                    rate: str = "+0%", volume: str = "+0%", pitch: str = "+0Hz") -> Path:
    """Sintetizza testo → file MP3 (edge-tts native) o WAV (via ffmpeg).
    rate/volume/pitch sono modificatori globali Edge (es. '+25%', '-10%')."""
    if not edge_available():
        raise RuntimeError("edge-tts non installato: pip install edge-tts")
    import edge_tts
    ensure_dirs()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    # Edge produce MP3 nativo; salviamo come MP3 e poi siamo compatibili con VHS_LoadAudioUpload.
    target_mp3 = output_wav.with_suffix(".mp3")

    async def _run():
        comm = edge_tts.Communicate(text, voice_id, rate=rate, volume=volume, pitch=pitch)
        await comm.save(str(target_mp3))

    try:
        asyncio.run(_run())
    except RuntimeError:
        import threading
        ex: list = []
        def _runner():
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_run())
            except Exception as e:
                ex.append(e)
        t = threading.Thread(target=_runner); t.start(); t.join()
        if ex:
            raise ex[0]
    return target_mp3


# ────────────────────────────────────────────────────────────────────────────
# Markup → parametri Edge (estensione semplice del prompt)
# ────────────────────────────────────────────────────────────────────────────

# Marker testuali "alti livelli" → traduzione naive a tag SSML Edge / Prosody.
# Edge TTS supporta SSML completo, ma per semplicità qui applichiamo modificatori
# globali (rate/pitch) e rimuoviamo i marker dal testo prima dell'invio.
_EMOTION_PRESETS = {
    "happy":    {"rate": "+5%",  "pitch": "+30Hz", "volume": "+0%"},
    "sad":      {"rate": "-15%", "pitch": "-30Hz", "volume": "-10%"},
    "angry":    {"rate": "+10%", "pitch": "+10Hz", "volume": "+20%"},
    "whisper":  {"rate": "-10%", "pitch": "-20Hz", "volume": "-50%"},
    "calm":     {"rate": "-5%",  "pitch": "-10Hz", "volume": "-5%"},
}


def strip_markers(text: str) -> str:
    """Rimuove i marker [pause:Xms] / <emotion>...</emotion> / <prosody …>…</prosody>
    e restituisce il testo "leggibile" (per anteprima)."""
    text = re.sub(r"\[pause[: ]?\d*m?s?\]", " ", text, flags=re.I)
    text = re.sub(r"<\s*emotion[^>]*>", "", text)
    text = re.sub(r"</\s*emotion\s*>", "", text)
    text = re.sub(r"<\s*prosody[^>]*>", "", text)
    text = re.sub(r"</\s*prosody\s*>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def derive_edge_params(text: str) -> dict[str, str]:
    """Heuristica: se il testo apre con <emotion type="X">, applica il preset al rate/pitch."""
    m = re.search(r"<\s*emotion\s+type\s*=\s*['\"]?([a-z]+)['\"]?\s*>", text, flags=re.I)
    if not m:
        return {"rate": "+0%", "volume": "+0%", "pitch": "+0Hz"}
    return _EMOTION_PRESETS.get(m.group(1).lower(), {"rate": "+0%", "volume": "+0%", "pitch": "+0Hz"})


# ────────────────────────────────────────────────────────────────────────────
# Dispatcher multi-engine
# ────────────────────────────────────────────────────────────────────────────

def synthesize(engine: str, voice: VoicePreset, text: str, *,
               output_dir: Path | None = None) -> Path:
    """Punto d'ingresso unico per la sintesi. Sceglie il backend in base a `engine`
    (id stringa: edge / cosyvoice2 / f5tts / xtts2 / fish / vibevoice).

    Per gli engine "remoti" (cosyvoice2/f5tts/xtts2/fish/vibevoice) la sintesi avviene
    via REST a ComfyUI inviando un workflow ad-hoc col custom node corrispondente.
    Se il nodo non è installato, viene sollevato RuntimeError con istruzioni.
    """
    ensure_dirs()
    output_dir = output_dir or OUTPUT_DIR
    import datetime as _dt
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"tts_{engine}_{voice.slug()}_{stamp}.mp3"
    clean = strip_markers(_pauses_to_dots_internal(text))
    # NATIVI via Handy-API (zero ComfyUI)
    if engine in ("kokoro", "handy_clone"):
        from core import handy_api
        wav = str(out.with_suffix(".wav"))
        if engine == "handy_clone":                        # voce clonata per nome (registrata in Handy)
            r = handy_api.tts(clean, voice=voice.name, out_wav=wav, model="chatterbox-multilingual")
        else:
            r = handy_api.tts(clean, voice=(voice.voice_id or "if_sara"),
                              lang=(voice.language or "it-IT")[:2], out_wav=wav)
        if r:
            return Path(r)
        raise RuntimeError("Handy-API non disponibile: avvia/installa Handy-API (:8720).")
    if engine == "edge":
        params = derive_edge_params(text)
        return edge_synthesize(clean, voice.voice_id, out, **params)
    if engine == "cosyvoice2":                              # nativo: venv standalone, poi Handy, poi (legacy) comfy
        try:
            from core import cosyvoice_service
            if cosyvoice_service.is_installed() and voice.sample_path:
                job = cosyvoice_service.make_job(clean, voice.sample_path,
                                                 prompt_text=voice.sample_transcript, out_name=voice.slug())
                job.run() if hasattr(job, "run") else None
                f = find_latest_audio() if "find_latest_audio" in globals() else None
                if f:
                    return Path(f)
        except Exception:                                  # noqa: BLE001
            pass
        from core import handy_api
        if handy_api.available():
            wav = str(out.with_suffix(".wav"))
            r = handy_api.tts(clean, voice=voice.name, out_wav=wav, model="chatterbox-multilingual")
            if r:
                return Path(r)
        return _synthesize_comfy(engine, voice, text, out)  # ultimo fallback (legacy ComfyUI)
    if engine in ("f5tts", "xtts2", "fish", "vibevoice"):  # cloning: reindirizza a Handy (nativo)
        from core import handy_api
        if handy_api.available():
            wav = str(out.with_suffix(".wav"))
            r = handy_api.tts(clean, voice=voice.name, out_wav=wav, model="chatterbox-multilingual")
            if r:
                return Path(r)
        return _synthesize_comfy(engine, voice, text, out)  # fallback legacy se Handy assente
    raise RuntimeError(f"Engine TTS sconosciuto: {engine}")


def _pauses_to_dots_internal(text: str) -> str:
    """Edge: [pause:Xms] → sequenza di puntini (proxy per <break>)."""
    def _r(m):
        ms = int(m.group(1) or 500)
        n = max(2, min(8, ms // 200))
        return " " + "." * n + " "
    return re.sub(r"\[pause[: ]?(\d+)?m?s?\]", _r, text, flags=re.I)


def _synthesize_comfy(engine: str, voice: VoicePreset, text: str, output: Path) -> Path:
    """Sintesi via custom node ComfyUI. Richiede il nodo installato + ComfyUI online."""
    if not _engine_installed(engine):
        node_names = {
            "cosyvoice2": "ComfyUI-CosyVoice2 (kijai/CosyVoice2)",
            "f5tts":      "ComfyUI-F5-TTS (niknah/ComfyUI-F5-TTS)",
            "xtts2":      "ComfyUI-XTTS o ComfyUI-Coqui-XTTS",
            "fish":       "ComfyUI-FishSpeech",
            "vibevoice":  "ComfyUI-VibeVoice (Enemyx-net/VibeVoice)",
        }[engine]
        raise RuntimeError(
            f"Il motore '{engine}' richiede il custom node ComfyUI:\n"
            f"  {node_names}\n"
            f"Clonalo in ComfyUI\\custom_nodes\\ e riavvia ComfyUI, poi ricarica i motori."
        )
    if not voice.sample_path or not Path(voice.sample_path).exists():
        raise RuntimeError(
            f"La voce '{voice.name}' non ha un sample audio associato.\n"
            "Per il voice cloning serve un campione vocale (5-15s di parlato pulito).\n"
            "Usa il bottone «✏️ Modifica voce» per allegare un file e una trascrizione."
        )
    # CosyVoice zero-shot (3s极速复刻) richiede la TRASCRIZIONE del campione (prompt_text):
    # senza, il nodo asserisce "prompt文本为空" (prompt text vuoto). Diamo un errore chiaro.
    if engine in ("cosyvoice2", "f5tts", "fish") and not (voice.sample_transcript or "").strip():
        raise RuntimeError(
            f"Il motore '{engine}' (clone zero-shot) richiede la TRASCRIZIONE del campione "
            f"(cosa dice esattamente l'audio di '{voice.name}').\n"
            "Aprila con «✏️ Modifica voce» e compila il campo «Trascrizione campione», poi riprova."
        )
    # Costruzione workflow specifico per engine. Endpoint singolo: POST /prompt.
    workflow = _build_tts_workflow(engine, voice, text, output)
    # Esegue tramite ComfyDriver (API reale: ComfyDriver() + download_view, niente singleton)
    from core.comfy_driver import ComfyDriver
    drv = ComfyDriver()
    if not drv.is_up():
        raise RuntimeError(f"ComfyUI non è in esecuzione. Avvialo prima di usare il TTS '{engine}'.")
    # Upload del sample audio nel folder input di ComfyUI (upload_image accetta anche audio).
    # Tronca a ≤30s: oltre, il tokenizer Whisper di CosyVoice/F5/XTTS crasha (limite 1500 frame).
    sample_for_upload = prepare_clone_sample(voice.sample_path)
    sample_name = drv.upload_image(sample_for_upload).get("name", Path(sample_for_upload).name)
    # Rimpiazza il placeholder del sample nel workflow
    for n in workflow.values():
        ins = n.get("inputs", {})
        for k, v in list(ins.items()):
            if v == "__SAMPLE_AUDIO__":
                ins[k] = sample_name
    pid = drv.submit(workflow)
    # Attesa risultato. `history[pid]` compare SOLO a esecuzione conclusa (successo o errore):
    # mentre ComfyUI lavora resta assente. Quindi NON è un timeout di inattività — il primo
    # avvio di CosyVoice/F5/XTTS scarica i pesi (CosyVoice-300M ~1.5GB da modelscope) e carica
    # gli ONNX: può richiedere parecchi minuti. Attesa generosa + uscita immediata su errore.
    import time
    deadline = time.monotonic() + TTS_SYNTH_TIMEOUT_S
    while time.monotonic() < deadline:
        hist = drv.history(pid)
        entry = hist.get(pid) if hist else None
        if entry:
            status = entry.get("status", {}) or {}
            # errore reale del nodo → esci subito col messaggio, non aspettare la deadline
            if status.get("status_str") == "error" or status.get("completed") is False:
                msgs = status.get("messages") or []
                detail = ""
                for m in msgs:
                    if isinstance(m, list) and len(m) == 2 and "error" in str(m[0]).lower():
                        detail = str(m[1])
                        break
                raise RuntimeError(f"TTS {engine}: errore in ComfyUI durante la sintesi. {detail}".strip())
            for nout in (entry.get("outputs", {}) or {}).values():
                for audio in nout.get("audio", []) or []:
                    fn = audio.get("filename", "")
                    if not fn:
                        continue
                    data = drv.download_view(fn, audio.get("subfolder", ""),
                                             audio.get("type", "output"))
                    final = output.with_suffix(Path(fn).suffix)
                    final.write_bytes(data)
                    return final
            # concluso ma senza audio → inutile attendere oltre
            raise RuntimeError(f"TTS {engine}: esecuzione conclusa ma nessun output audio prodotto.")
        time.sleep(2)
    # timeout: il job è quasi certamente piantato (es. deadlock del nodo CosyVoice su mix CPU/GPU).
    # Interrompi su ComfyUI così non resta appeso a occupare VRAM.
    drv.interrupt()
    raise RuntimeError(
        f"TTS {engine}: nessun output entro {int(TTS_SYNTH_TIMEOUT_S)}s — sintesi interrotta. "
        "Il nodo si è probabilmente piantato (deadlock noto di CosyVoice-ComfyUI su alcune GPU). "
        "Usa «Audio fornito da me» con una traccia pronta, oppure un altro engine di clone (F5-TTS/XTTS).")


def _build_tts_workflow(engine: str, voice: VoicePreset, text: str, output: Path) -> dict:
    """Workflow API ComfyUI per ogni engine. Struttura minima a 4-5 nodi."""
    # Trascrizione del sample (richiesta da CosyVoice/F5/XTTS per migliorare il clone)
    transcript = voice.sample_transcript or ""
    if engine == "cosyvoice2":
        # CosyVoice-ComfyUI (AIFSH): TextNode produce un TEXT custom da una STRING.
        # CosyVoiceNode con inference_mode "3s极速复刻" = zero-shot voice cloning (3 sec).
        return {
            "1": {"class_type": "LoadAudio", "inputs": {"audio": "__SAMPLE_AUDIO__"}},
            "2": {"class_type": "TextNode", "inputs": {"text": text}},
            "3": {"class_type": "TextNode", "inputs": {"text": transcript}},
            "4": {"class_type": "CosyVoiceNode",
                  "inputs": {
                      "tts_text": ["2", 0],
                      "prompt_text": ["3", 0],
                      "prompt_wav": ["1", 0],
                      "inference_mode": "3s极速复刻",      # zero-shot cloning
                      "sft_dropdown": "中文女",
                      "speed": 1.0, "seed": 42,
                  }},
            "5": {"class_type": "SaveAudio",
                  "inputs": {"audio": ["4", 0], "filename_prefix": output.stem}},
        }
    if engine == "f5tts":
        return {
            "1": {"class_type": "LoadAudio", "inputs": {"audio": "__SAMPLE_AUDIO__"}},
            "2": {"class_type": "F5TTSAudio",
                  "inputs": {"sample_audio": ["1", 0], "sample_text": transcript,
                             "speech": text, "seed": 0}},
            "3": {"class_type": "SaveAudio",
                  "inputs": {"audio": ["2", 0], "filename_prefix": output.stem}},
        }
    if engine == "xtts2":
        return {
            "1": {"class_type": "LoadAudio", "inputs": {"audio": "__SAMPLE_AUDIO__"}},
            "2": {"class_type": "XTTSGenerate",
                  "inputs": {"text": text, "reference_audio": ["1", 0],
                             "language": (voice.language or "it").split("-")[0]}},
            "3": {"class_type": "SaveAudio",
                  "inputs": {"audio": ["2", 0], "filename_prefix": output.stem}},
        }
    if engine == "fish":
        return {
            "1": {"class_type": "LoadAudio", "inputs": {"audio": "__SAMPLE_AUDIO__"}},
            "2": {"class_type": "FishSpeechRun",
                  "inputs": {"text": text, "reference_audio": ["1", 0],
                             "reference_text": transcript, "seed": 0}},
            "3": {"class_type": "SaveAudio",
                  "inputs": {"audio": ["2", 0], "filename_prefix": output.stem}},
        }
    if engine == "vibevoice":
        return {
            "1": {"class_type": "LoadAudio", "inputs": {"audio": "__SAMPLE_AUDIO__"}},
            "2": {"class_type": "VibeVoiceGenerate",
                  "inputs": {"script": text, "speaker_1_voice": ["1", 0]}},
            "3": {"class_type": "SaveAudio",
                  "inputs": {"audio": ["2", 0], "filename_prefix": output.stem}},
        }
    raise RuntimeError(f"workflow non definito per engine '{engine}'")
