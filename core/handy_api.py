"""Client Handy-API — TTS + STT + voice cloning via REST (porta 8720).

Handy-API (C:\\Projects\\Handy-API, https://github.com/nuspy/Handy-API) è un'app Tauri con REST API
interna su :8720. Questo client la rileva in automatico e, se assente, la individua/installa, così
l'app è distribuibile su sistemi diversi.

Endpoint usati: GET /health, POST /transcribe (STT multipart→{text}), POST /tts/test (one-shot→audio_b64),
POST /voices/clone, GET /voices/cloned, GET /tts/models, GET /models/available.

API:
  is_up(port=) -> bool
  ensure(port=) -> bool                       # rileva, prova ad avviare l'app, attende /health
  transcribe(audio_path, port=) -> str        # STT → testo
  tts(text, voice=, lang=, speed=, out_wav=, port=) -> str   # TTS → wav
  clone_voice(name, audio_path, port=) -> dict
  list_tts_models(port=) / list_cloned(port=)
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

DEFAULT_PORT = 8720
HANDY_DIR = Path(r"C:\Projects\Handy-API")
HANDY_REPO = "https://github.com/nuspy/Handy-API"


def _base(port: int | None = None) -> str:
    return f"http://localhost:{port or DEFAULT_PORT}"


def is_up(port: int | None = None) -> bool:
    try:
        with urllib.request.urlopen(_base(port) + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:                                       # noqa: BLE001
        return False


# ─────────── auto-detect / auto-install ───────────
def _find_handy_exe() -> str | None:
    """Trova l'eseguibile Handy installato (Tauri) o lo script di avvio nel repo."""
    cands = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Handy" / "Handy.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Handy" / "Handy.exe",
        HANDY_DIR / "src-tauri" / "target" / "release" / "handy.exe",
        HANDY_DIR / "src-tauri" / "target" / "release" / "Handy.exe",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    exe = shutil.which("handy")
    return exe


def ensure(port: int | None = None, timeout: float = 30) -> bool:
    """Se Handy non risponde, prova ad avviarlo (eseguibile trovato). Ritorna True se /health risponde."""
    if is_up(port):
        return True
    exe = _find_handy_exe()
    if exe:
        try:
            subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:                                   # noqa: BLE001
            pass
        t0 = time.time()
        while time.time() - t0 < timeout:
            if is_up(port):
                return True
            time.sleep(2)
    return is_up(port)


def install_instructions() -> str:
    """Istruzioni d'installazione se Handy non è presente (repo non clonato)."""
    if HANDY_DIR.exists():
        return (f"Handy-API trovato in {HANDY_DIR} ma non in esecuzione. Avvialo (build Tauri) "
                f"oppure: cd {HANDY_DIR} && cargo tauri dev")
    return (f"Handy-API non installato. Clona ed esegui:\n  git clone {HANDY_REPO} {HANDY_DIR}\n"
            f"  cd {HANDY_DIR} && (vedi BUILD.md)")


def clone_repo(dest: Path = HANDY_DIR) -> bool:
    """Clona il repo Handy-API se assente (codice; la build resta all'utente)."""
    if dest.exists():
        return True
    try:
        subprocess.run(["git", "clone", HANDY_REPO, str(dest)], check=True)
        return True
    except Exception:                                       # noqa: BLE001
        return False


# ─────────── REST helpers ───────────
def _post_json(path: str, payload: dict, port: int | None = None, timeout: int = 600) -> dict:
    req = urllib.request.Request(_base(port) + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get_json(path: str, port: int | None = None, timeout: int = 15) -> dict:
    with urllib.request.urlopen(_base(port) + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ─────────── STT ───────────
def transcribe(audio_path: str, port: int | None = None, timeout: int = 600) -> str:
    """STT: audio → testo via POST /transcribe (multipart). Ritorna '' se non disponibile."""
    if not ensure(port):
        return ""
    import mimetypes
    import uuid
    boundary = "----handy" + uuid.uuid4().hex
    fn = os.path.basename(audio_path)
    ctype = mimetypes.guess_type(fn)[0] or "application/octet-stream"
    with open(audio_path, "rb") as fh:
        data = fh.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fn}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(_base(port) + "/transcribe", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()).get("text", "")
    except Exception:                                       # noqa: BLE001
        return ""


# ─────────── TTS ───────────
def tts(text: str, *, voice: str = "if_sara", lang: str = "it", speed: float = 1.0,
        out_wav: str = "", model: str = "", port: int | None = None) -> str:
    """TTS one-shot via POST /tts/test → salva WAV (decodifica audio_b64). Ritorna il path o ''."""
    if not ensure(port):
        return ""
    payload = {"text": text, "voice": voice, "lang": lang, "speed": speed}
    if model:
        payload["model"] = model
    res = {}
    for attempt in range(3):
        try:
            res = _post_json("/tts/test", payload, port=port)
            break
        except (ConnectionError, TimeoutError, OSError) as e:
            # Handy chiude a volte la connessione a metà (WinError 10054, visto 27/09 fra due battute di un
            # dialogo): si riprova dopo aver riverificato che il servizio sia su, invece di far fallire la voce
            if attempt == 2:
                raise
            print(f"[handy] TTS interrotto ({e}): nuovo tentativo {attempt + 2}/3", flush=True)
            time.sleep(2 * (attempt + 1))
            if not ensure(port):
                raise
    b64 = res.get("audio_b64")
    if not b64 or not out_wav:
        return ""
    try:                                                     # b64 corrotto / disco pieno / permessi → "" (no crash)
        os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
        open(out_wav, "wb").write(base64.b64decode(b64))
    except Exception as e:                                   # noqa: BLE001
        print(f"[handy] scrittura TTS fallita: {e}", flush=True)
        return ""
    return out_wav


def clone_voice(name: str, audio_path: str, port: int | None = None) -> dict:
    if not ensure(port):
        return {}
    try:                                                     # file assente/illeggibile → {} (no FileNotFoundError)
        b64 = base64.b64encode(open(audio_path, "rb").read()).decode()
    except Exception as e:                                   # noqa: BLE001
        print(f"[handy] lettura campione voce fallita ({audio_path}): {e}", flush=True)
        return {}
    fmt = os.path.splitext(audio_path)[1].lstrip(".").lower() or "wav"
    return _post_json("/voices/clone", {"name": name, "audio_b64": b64, "format": fmt}, port=port)


def list_tts_models(port: int | None = None) -> dict:
    try:
        return _get_json("/tts/models", port=port)
    except Exception:                                       # noqa: BLE001
        return {}


def list_cloned(port: int | None = None) -> dict:
    try:
        return _get_json("/voices/cloned", port=port)
    except Exception:                                       # noqa: BLE001
        return {}


def available() -> bool:
    """True se Handy-API è raggiungibile (per scegliere il backend voce/STT)."""
    return is_up()
