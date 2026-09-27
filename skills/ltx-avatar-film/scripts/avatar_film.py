#!/usr/bin/env python3
"""Client dello skill «ltx-avatar-film»: invia una richiesta a POST /avatar/generate di LTX Studio, segue il
job e stampa il PATH del video creato. Solo libreria standard (gira con qualsiasi Python ≥ 3.9).

  python avatar_film.py --request richiesta.json          # invia, attende, stampa  VIDEO: <path>
  python avatar_film.py --request richiesta.json --dry-run # controlla la richiesta (file, marcatori) senza inviare
  python avatar_film.py --options                         # motori, voci, lingue e profili installati
  python avatar_film.py --markup "Ciao [pause:500ms] [whisper]segreto[/whisper]"   # anteprima dei marcatori

Cosa fa in più di una semplice POST:
  · avvia l'API se è spenta (python -m api.autostart nella cartella di LTX Studio) e attende /health;
  · carica con /files/upload i file locali che stanno FUORI dalle cartelle dell'API (foto, audio, riferimenti)
    e sostituisce i path nella richiesta;
  · controlla i marcatori espressivi (tag chiusi, valori nei limiti) prima di spendere minuti di GPU;
  · fa polling di /jobs/{id} mostrando avanzamento e messaggi, e alla fine stampa:
        VIDEO: <path locale>      DOWNLOAD: <url>      (+ IMAGE:/AUDIO: se return_intermediates)
Variabili: LTX_API_URL (default http://127.0.0.1:8200), LTX_STUDIO_ROOT (default: la cartella che contiene
questo skill oppure E:\\AI\\ltxstudio), LTX_API_TOKEN (solo se l'API è esposta in rete), LTX_API_PYTHON.
Uscita: 0 = video pronto, 1 = errore del job, 2 = richiesta non valida, 3 = API non raggiungibile.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

API = os.environ.get("LTX_API_URL", "http://127.0.0.1:8200").rstrip("/")
TOKEN = os.environ.get("LTX_API_TOKEN", "")
HERE = Path(__file__).resolve()


def _root() -> Path:
    env = os.environ.get("LTX_STUDIO_ROOT")
    if env:
        return Path(env)
    for p in HERE.parents:                               # skill dentro al repo (skills/ltx-avatar-film/…)
        if (p / "api" / "autostart.py").is_file():
            return p
    return Path(r"E:\AI\ltxstudio")


# ── HTTP ─────────────────────────────────────────────────────────────────────────────────────
def _req(method: str, path: str, body: dict | None = None, timeout: float = 60.0):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method)
    if data is not None:
        r.add_header("Content-Type", "application/json")
    if TOKEN:
        r.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code} su {path}: {detail[:1500]}") from None


def _upload(path: str) -> str:
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
               f"Content-Type: {ctype}\r\n\r\n").encode() + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    r = urllib.request.Request(API + "/files/upload", data=payload, method="POST")
    r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if TOKEN:
        r.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.loads(resp.read())["path"]


def ensure_api(wait: float = 120.0) -> bool:
    try:
        _req("GET", "/health", timeout=3)
        return True
    except Exception:                                    # noqa: BLE001
        pass
    root = _root()
    py = os.environ.get("LTX_API_PYTHON") or sys.executable
    print(f"[avatar-film] API spenta: la avvio ({root})", flush=True)
    # l'autostart figlio ha un suo timeout e può stampare FAILED mentre l'API sta ancora partendo: lo stato vero
    # lo decide il polling di /health qui sotto, quindi il suo output non serve
    subprocess.Popen([py, "-m", "api.autostart"], cwd=str(root),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < wait:
        try:
            _req("GET", "/health", timeout=3)
            return True
        except Exception:                                # noqa: BLE001
            time.sleep(2)
    return False


# ── lingue (core/lang_codes del repo, se lo skill gira dentro LTX Studio) ─────────────────────
def _load_lang_codes():
    try:
        root = str(_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        from core import lang_codes
        return lang_codes
    except Exception:                                    # noqa: BLE001 — skill installato fuori dal repo
        return None


_LC = _load_lang_codes()


def check_languages(req: dict) -> tuple[list[str], list[str]]:
    """(errori, avvisi) sulla lingua del TESTO di speaker e battute rispetto alla lingua della VOCE."""
    errs, warns = [], []
    a = req.get("audio") or {}
    engine = req.get("engine", "longcat")
    if engine == "longcat" and a.get("mode") in ("tts", "dialogue"):
        default = a.get("language") or ""
        for sp in a.get("speakers") or [{}]:
            name, lang = sp.get("name", "A"), sp.get("language") or default
            voice, eng = sp.get("voice_id") or "", sp.get("engine", "edge")
            cloned = bool((sp.get("clone") or {}).get("sample_path"))
            if not lang and (sp.get("voice_description") or cloned or not voice):
                errs.append(f"speaker «{name}»: manca language (lingua del TESTO, es. it-IT / en-US): con "
                            "voice_description, clone o senza voce non si ricava")
                continue
            if not lang:
                warns.append(f"speaker «{name}»: language non indicata → verrà usata quella della voce {voice!r}")
                continue
            if _LC:
                w = _LC.check(eng, lang, voice_id=voice, cloned=cloned, allow_accent=bool(sp.get("allow_accent")))
                if w:
                    warns.append(f"speaker «{name}»: {w}")
                for m in re.finditer(r"\[lang:([^\]]+)\]", a.get("text") or "", re.I):
                    w = _LC.check(eng, m.group(1), voice_id=voice, cloned=cloned,
                                  allow_accent=bool(sp.get("allow_accent")))
                    if w and f"[lang:{m.group(1)}]" not in " ".join(warns):
                        warns.append(f"speaker «{name}» [lang:{m.group(1)}]: {w}")
    if engine == "minimax_h3":
        lines = ((req.get("video") or {}).get("h3") or {}).get("dialogue_lines") or []
        if not lines and a.get("text"):
            lines = [{"language": a.get("language") or "", "text": a["text"]}]
        for i, d in enumerate(lines):
            lang = d.get("language") or a.get("language") or ""
            if not lang:
                errs.append(f"video.h3.dialogue_lines[{i}]: manca language (lingua del testo della battuta)")
            elif _LC and not _LC.h3_name(lang):
                errs.append(f"video.h3.dialogue_lines[{i}]: MiniMax H3 non parla {lang!r} (lingue: "
                            + ", ".join(_LC.LANGS[k][1] for k in _LC.H3_LANGS) + ")")
            for m in re.finditer(r"\[lang:([^\]]+)\]", d.get("text") or "", re.I):
                if _LC and not _LC.h3_name(m.group(1)):
                    errs.append(f"video.h3.dialogue_lines[{i}]: [lang:{m.group(1)}] non supportata da MiniMax H3")
    return errs, warns


# ── controlli della richiesta ─────────────────────────────────────────────────────────────────
_TAG = re.compile(r"\[(/?)(pause|breath|emotion|speed|accel|pitch|volume|loud|soft|whisper|emph|lang)(?::([^\]]*))?\]", re.I)
_EMOTIONS = {"neutral", "happy", "sad", "angry", "whisper", "calm", "excited", "fearful", "surprised", "tender",
             "serious"}
_OPEN_CLOSE = {"emotion", "speed", "accel", "pitch", "volume", "loud", "soft", "whisper", "emph", "lang"}


def check_markup(text: str) -> list[str]:
    """Errori nei marcatori espressivi (tag non chiusi, valori fuori limite, emozioni sconosciute)."""
    errs, stack = [], []
    for m in _TAG.finditer(text or ""):
        close, tag, arg = m.group(1), m.group(2).lower(), (m.group(3) or "").strip()
        if close:
            if not stack or stack[-1] != tag:
                errs.append(f"[/{tag}] senza [{tag}] aperto (o chiusura nell'ordine sbagliato)")
            else:
                stack.pop()
            continue
        if tag in _OPEN_CLOSE:
            stack.append(tag)
        try:
            if tag == "emotion" and arg.lower() not in _EMOTIONS:
                errs.append(f"emozione sconosciuta: {arg!r} (valide: {', '.join(sorted(_EMOTIONS))})")
            elif tag == "speed" and not 0.5 <= float(arg) <= 2.0:
                errs.append(f"[speed:{arg}] fuori da 0.5–2.0")
            elif tag == "accel":
                a, b = (float(x) for x in arg.split("-"))
                if not (0.5 <= a <= 2.0 and 0.5 <= b <= 2.0):
                    errs.append(f"[accel:{arg}] fuori da 0.5–2.0")
            elif tag == "pitch" and not -12 <= float(arg) <= 12:
                errs.append(f"[pitch:{arg}] fuori da −12…+12 semitoni")
            elif tag == "volume" and not -30 <= float(arg) <= 12:
                errs.append(f"[volume:{arg}] fuori da −30…+12 dB")
            elif tag == "lang" and _LC and not _LC.base(arg):
                errs.append(f"[lang:{arg}] lingua non riconosciuta (usa it, en, es, fr, de, pt, zh, ja… o it-IT)")
            elif tag == "pause" and arg and not re.fullmatch(r"\d+(\.\d+)?(ms|s)?", arg.lower()):
                errs.append(f"[pause:{arg}] non valido (es. 500ms, 1.5s)")
        except ValueError:
            errs.append(f"valore non numerico in [{tag}:{arg}]")
    errs += [f"[{t}] aperto e mai chiuso con [/{t}]" for t in stack]
    return errs


def _texts(req: dict) -> list[tuple[str, str]]:
    out = []
    a = req.get("audio") or {}
    if a.get("text"):
        out.append(("audio.text", a["text"]))
    for i, d in enumerate(((req.get("video") or {}).get("h3") or {}).get("dialogue_lines") or []):
        out.append((f"video.h3.dialogue_lines[{i}].text", d.get("text", "")))
    return out


_PATH_KEYS = ("image_path", "audio_path", "audio2_path", "voice_reference_path", "sample_path", "path")


def _localize(obj, api_roots: tuple[str, ...], dry: bool, errs: list[str]):
    """Carica i file locali fuori dalle cartelle dell'API e sostituisce i path (ricorsivo)."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k in _PATH_KEYS and isinstance(v, str) and v:
                if not os.path.isfile(v):
                    errs.append(f"file inesistente: {k}={v}")
                elif not any(os.path.abspath(v).lower().startswith(r.lower()) for r in api_roots) and not dry:
                    obj[k] = _upload(v)
                    print(f"[avatar-film] caricato {os.path.basename(v)} → {obj[k]}", flush=True)
            elif k == "ref_images" and isinstance(v, list):
                new = []
                for x in v:
                    if isinstance(x, str) and os.path.isfile(x) and not dry and \
                            not any(os.path.abspath(x).lower().startswith(r.lower()) for r in api_roots):
                        new.append(_upload(x))
                    else:
                        if isinstance(x, str) and not x.startswith("@") and not os.path.isfile(x):
                            errs.append(f"riferimento inesistente: {x}")
                        new.append(x)
                obj[k] = new
            else:
                _localize(v, api_roots, dry, errs)
    elif isinstance(obj, list):
        for x in obj:
            _localize(x, api_roots, dry, errs)


def validate(req: dict) -> list[str]:
    errs = []
    engine = req.get("engine", "longcat")
    if engine not in ("longcat", "minimax_h3"):
        errs.append(f"engine sconosciuto: {engine} (longcat | minimax_h3)")
    for where, t in _texts(req):
        errs += [f"{where}: {e}" for e in check_markup(t)]
    a = req.get("audio") or {}
    if engine == "longcat" and a.get("mode", "file") == "file" and not a.get("audio_path"):
        errs.append("LongCat con audio.mode=file richiede audio.audio_path (oppure usa mode=tts/dialogue)")
    if engine == "longcat" and a.get("mode") in ("tts", "dialogue") and not a.get("text"):
        errs.append("audio.text vuoto: scrivi il testo da far dire")
    if engine == "minimax_h3":
        h3 = (req.get("video") or {}).get("h3") or {}
        d = float(h3.get("duration_s") or 8)
        if not 5 <= d <= 15:
            errs.append(f"MiniMax H3: durata {d}s fuori da 5–15 s")
        if not h3.get("dialogue_lines") and not a.get("text") and not (req.get("video") or {}).get("prompt"):
            errs.append("MiniMax H3: servono battute (video.h3.dialogue_lines) o un prompt")
    im = req.get("image") or {}
    if im.get("mode") == "text" and not im.get("prompt"):
        errs.append("image.mode=text richiede image.prompt")
    if im.get("mode") == "image" and not im.get("image_path"):
        errs.append("image.mode=image richiede image.image_path")
    if im.get("mode") in ("character", "character_multi"):
        ch = im.get("character") or {}
        if not (ch.get("ref_images") or ch.get("asset")):
            errs.append("image.character richiede ref_images o asset «@Nome»")
    return errs


# ── main ─────────────────────────────────────────────────────────────────────────────────────
def run(req: dict, poll: float = 5.0, timeout_s: float = 4 * 3600) -> int:
    job = _req("POST", "/avatar/generate", req, timeout=120)
    jid = job["job_id"]
    print(f"[avatar-film] job {jid} accodato", flush=True)
    t0, last = time.time(), ""
    while time.time() - t0 < timeout_s:
        st = _req("GET", f"/jobs/{jid}", timeout=30)
        msg = f"{st.get('status')} {int(100 * float(st.get('progress') or 0))}% {st.get('message') or ''}".strip()
        if st.get("status") == "pending" and st.get("queue_position"):
            msg += f" (in coda, posizione {st['queue_position']})"
        if msg != last:
            print(f"[avatar-film] {msg}", flush=True)
            last = msg
        if st.get("status") == "success":
            for w in st.get("warnings") or []:
                print(f"AVVISO: {w}")
            if (st.get("info") or {}).get("languages"):
                print(f"LINGUE: {json.dumps(st['info']['languages'], ensure_ascii=False)}")
            files = st.get("files") or []
            if not files:
                print("ERRORE: job riuscito ma senza file")
                return 1
            video = next((f for f in files if f["path"].lower().endswith((".mp4", ".mov", ".webm"))), files[0])
            print(f"VIDEO: {video['path']}")
            print(f"DOWNLOAD: {API}{video['download_url'] if video['download_url'].startswith('/') else '/' + video['download_url']}"
                  if not video["download_url"].startswith("http") else f"DOWNLOAD: {video['download_url']}")
            for f in files:
                if f is video:
                    continue
                kind = "IMAGE" if f["path"].lower().endswith((".png", ".jpg", ".jpeg", ".webp")) else "AUDIO"
                print(f"{kind}: {f['path']}")
            return 0
        if st.get("status") == "error":
            print(f"ERRORE: {st.get('error') or st.get('message')}")
            return 1
        time.sleep(poll)
    print(f"ERRORE: timeout dopo {timeout_s / 60:.0f} min (il job {jid} può essere ancora in corso)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--request", help="file JSON con la richiesta per /avatar/generate (o '-' = stdin)")
    ap.add_argument("--dry-run", action="store_true", help="valida senza inviare")
    ap.add_argument("--options", action="store_true", help="stampa GET /avatar/options")
    ap.add_argument("--markup", help="anteprima/controllo di un testo con marcatori espressivi")
    a = ap.parse_args()
    if a.markup is not None:
        errs = check_markup(a.markup)
        clean = re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s+", " ", _TAG.sub(" ", a.markup))).strip()
        print(f"TESTO PULITO: {clean}")
        print("OK" if not errs else "ERRORI:\n  " + "\n  ".join(errs))
        return 0 if not errs else 2
    if not ensure_api():
        print(f"ERRORE: API non raggiungibile su {API}")
        return 3
    if a.options:
        print(json.dumps(_req("GET", "/avatar/options"), ensure_ascii=False, indent=1))
        return 0
    if not a.request:
        ap.print_help()
        return 2
    raw = sys.stdin.read() if a.request == "-" else Path(a.request).read_text("utf-8")
    req = json.loads(raw)
    errs = validate(req)
    lerrs, lwarns = check_languages(req)
    errs += lerrs
    for w in lwarns:
        print(f"AVVISO LINGUA: {w}")
    root = _root()
    api_roots = (str((root / "output").resolve()), str((root / "output" / "api_uploads").resolve()))
    _localize(req, api_roots, a.dry_run, errs)
    if errs:
        print("RICHIESTA NON VALIDA:\n  " + "\n  ".join(errs))
        return 2
    if a.dry_run:
        print("OK (dry-run): richiesta valida\n" + json.dumps(req, ensure_ascii=False, indent=1))
        return 0
    return run(req)


if __name__ == "__main__":
    sys.exit(main())
