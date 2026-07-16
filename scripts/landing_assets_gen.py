#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestratore RIPRENDIBILE per gli asset della landing TaDaMoo.

Legge marketing/tadamoo_landing/ASSETS_MANIFEST.json (unica fonte dei prompt) e genera
ogni pezzo in ordine manifest usando SOLO le API interne validate:

  - immagini  → server nativo :8189 via core.native_driver.run_and_wait('image', params)
                (arch flux = t2i, kontext = edit, inpaint = FLUX Fill; outpaint = inpaint
                 su tela allargata; maschere generate programmaticamente con PIL)
  - video     → worker diretto E:\\AI\\Tools\\ltx2_venv\\Scripts\\python.exe workers\\ltx2_infer.py
                (t2v / i2v / 60s con chunk_frames / triplette canny via
                 core.controlnet_preprocess.preprocess_video)
  - musica    → core.music_service (HeartMuLa make_song_job, MMAudio make_sonify_job)
                e core.acestep_service.make_job — job.run() bloccante
  - voce      → core.tts_engine.synthesize con una voce esistente di voices\\

Gli slot 3D/GLB, mondo, Cinema Studio, videomusicale (e i pezzi la cui API non è nella
mappa validata: PuLID, face-swap, compose, storyboard, lip-sync, ReTake, SeedVR2, LoRA,
screenshot UI) sono marcati 'manual' nello stato, con motivo.

Stato:    marketing/tadamoo_landing/assets/_gen_state.json  (riprende da dove era)
Output:   marketing/tadamoo_landing/assets/<id>.<ext>
          + copia piatta E:\\AI\\ltxstudio\\output\\LANDING_<id>.<ext>

Uso:
  python scripts/landing_assets_gen.py --dry-run        # stampa il piano, non genera
  python scripts/landing_assets_gen.py                  # genera tutto (riprendibile)
  python scripts/landing_assets_gen.py --only <id>      # rigenera un pezzo (o uno slot)

Nessuna eccezione è fatale: un pezzo fallito → stato 'failed' con motivo, si continua.
Niente parallelismo (GPU singola). Prima di ogni job GPU: POST :8189/free (best-effort).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LANDING = REPO / "marketing" / "tadamoo_landing"
MANIFEST_FILE = LANDING / "ASSETS_MANIFEST.json"
ASSETS = LANDING / "assets"
WORK = ASSETS / "_work"                      # file intermedi (png lossless, params json, guide)
STATE_FILE = ASSETS / "_gen_state.json"
FLAT_OUT = Path(r"E:\AI\ltxstudio\output")   # copia piatta per ispezione utente

LTX_PY = Path(r"E:\AI\Tools\ltx2_venv\Scripts\python.exe")
LTX_WORKER = REPO / "workers" / "ltx2_infer.py"
FLUX_MODEL = r"Flux\FLUX_Dev.safetensors"          # t2i + edit kontext
FILL_MODEL = r"Flux\FLUX_DEV_Fill.safetensors"     # inpaint/outpaint
FREE_URL = "http://127.0.0.1:8189/free"
FPS = 24                                            # il worker LTX-2 lavora a 24fps (manifest dice 25: vedi report)
VIDEO_STEPS = 8                                     # default auto-GGUF distilled Q4

# ── testi fissi (derivati dai prompt del manifest) ───────────────────────────
TTS_TEXT = ("[warm] Signore e signori... [pause] allacciate la fantasia. "
            "[excited] Benvenuti a bordo del razzo TaDaMoo! [pause] "
            "Prossima fermata: [whisper] la Luna. [laugh]")
CLONE_TEXT = ("Capitano ai motori: rotta verso il Mare della Tranquillità. "
              "E stavolta il dolce lo portiamo noi.")

SONG_TAGS = ("italian disco pop, summer, female vocals, 118 bpm, funk guitar, handclaps, "
             "analog synth bass, catchy chorus, golden hour, road trip")
SONG_LYRICS = """[verse]
Finestrini giù sulla strada della costa
il sale nei capelli, il motore che ci sposta
la radio canta piano, il sole va giù
un'estate intera dentro un ritornello blu

[pre-chorus]
Tienimi la mano, conta fino a tre
il molo sta finendo ma io credo a te

[chorus]
Stasera si vola, si vola fino alla luna
finestrini giù, si accende la fortuna
balliamo sull'asfalto che profuma di sale
stasera si vola e nessuno può fermare

[verse]
Fari come lucciole sulla litoranea
una bandana rossa, la risata è spontanea
scriviamo i nostri nomi sopra il lungomare
domani chi si sveglia, stanotte si può sognare

[chorus]
Stasera si vola, si vola fino alla luna
finestrini giù, si accende la fortuna
balliamo sull'asfalto che profuma di sale
stasera si vola, si vola oltre il finale

[outro]
onde e risate, si vola fino alla luna
"""
ACESTEP_TAGS = ("instrumental, surf rock, 140 bpm, twangy electric guitar, spring reverb, "
                "driving drums, floor toms, organ stabs, energetic, triumphant bridge, "
                "beach bonfire at midnight")

# ── piano per-pezzo: SOLO API della mappa validata; il resto è 'manual' ──────
# kind: t2i | kontext | mask | inpaint | outpaint | t2v | i2v | canny_guide |
#       control_t2v | heartmula | acestep | sonify | tts | clone_sample | clone_tts | manual
_M_SLOT3D = "slot 3D/GLB/mondo: da fare a parte (indicazione esplicita utente)"
_M_CINEMA = "Cinema Studio / videomusicale: da fare a parte (indicazione esplicita utente)"
SPECS: dict[str, dict] = {
    # hero
    "hero-bg-01":          {"kind": "t2v", "w": 960, "h": 544, "sec": 20, "web": True},
    # immagini
    "img-t2i-01":          {"kind": "t2i", "w": 2048, "h": 1152},
    "img-edit-01-src":     {"kind": "t2i", "w": 1344, "h": 1024},
    "img-edit-01-out":     {"kind": "kontext", "src": "img-edit-01-src", "w": 1344, "h": 1024},
    "img-inpaint-01-src":  {"kind": "t2i", "w": 1536, "h": 1536},
    # blob morbido sull'area della torta (centro tavolo, leggermente in basso)
    "img-inpaint-01-mask": {"kind": "mask", "src": "img-inpaint-01-src",
                            "region": (0.50, 0.62, 0.24, 0.20)},
    "img-inpaint-01-out":  {"kind": "inpaint", "src": "img-inpaint-01-src",
                            "mask": "img-inpaint-01-mask", "w": 1536, "h": 1536},
    "img-outpaint-01-src": {"kind": "t2i", "w": 1024, "h": 1024},
    "img-outpaint-01-out": {"kind": "outpaint", "src": "img-outpaint-01-src", "w": 2048, "h": 1024},
    "img-compose-01-a":    {"kind": "t2i", "w": 1024, "h": 1024},
    "img-compose-01-b":    {"kind": "t2i", "w": 1024, "h": 1024},
    "img-compose-01-out":  {"kind": "manual",
                            "note": "compose multi-reference non nella mappa API validata — fare dalla GUI (Componi)"},
    "img-pulid-01-face":   {"kind": "t2i", "w": 1024, "h": 1024},
    "img-pulid-01-out":    {"kind": "manual",
                            "note": "PuLID (riferimento volto) non nella mappa API validata — fare dalla GUI"},
    "img-faceswap-01-src": {"kind": "t2i", "w": 1344, "h": 1024},
    "img-faceswap-01-out": {"kind": "manual",
                            "note": "face-swap non nella mappa API validata — fare dalla GUI"},
    # video
    "vid-t2v-01":          {"kind": "t2v", "w": 1280, "h": 704, "sec": 20},
    "vid-i2v-01-src":      {"kind": "t2i", "w": 1280, "h": 704},
    "vid-i2v-01-out":      {"kind": "i2v", "src": "vid-i2v-01-src", "w": 1280, "h": 704, "sec": 15},
    "vid-storyboard-01":   {"kind": "manual",
                            "note": "storyboard multi-keyframe non nella mappa API validata — fare dal pannello Storyboard"},
    "vid-long-01":         {"kind": "t2v", "w": 1280, "h": 704, "sec": 60, "chunk": 241},
    "vid-controlnet-01-src":   {"kind": "t2v", "w": 1280, "h": 704, "sec": 15},
    "vid-controlnet-01-guide": {"kind": "canny_guide", "src": "vid-controlnet-01-src"},
    "vid-controlnet-01-out":   {"kind": "control_t2v", "guide": "vid-controlnet-01-guide",
                                "w": 1280, "h": 704, "sec": 15},
    "vid-replace-01-src":  {"kind": "t2v", "w": 1280, "h": 704, "sec": 15},
    "vid-replace-01-ref":  {"kind": "t2i", "w": 1024, "h": 1024},
    # cambia-soggetto: control_video (guida canny estratta dal src, in _work) + prompt nuovo soggetto
    "vid-replace-01-out":  {"kind": "control_t2v", "guide_from": "vid-replace-01-src",
                            "w": 1280, "h": 704, "sec": 15},
    "vid-lipsync-01":      {"kind": "manual",
                            "note": "lip-sync a2v (foto+audio) non nella mappa API validata — fare dalla GUI"},
    "vid-retake-01-src":   {"kind": "t2v", "w": 1280, "h": 704, "sec": 15},
    "vid-retake-01-out":   {"kind": "manual",
                            "note": "ReTake non nella mappa API validata — fare dal pannello ReTake"},
    "vid-upscale-01-src":  {"kind": "t2v", "w": 1280, "h": 704, "sec": 15,
                            "degrade": (480, 360)},   # degradato di proposito (VHS 360p)
    "vid-upscale-01-out":  {"kind": "manual",
                            "note": "upscale SeedVR2 4K non nella mappa API validata — fare da Migliora qualità"},
    # musica
    "mus-song-01":         {"kind": "heartmula", "sec": 90, "tags": SONG_TAGS, "lyrics": SONG_LYRICS},
    "mus-acestep-01":      {"kind": "acestep", "sec": 60, "tags": ACESTEP_TAGS},
    "mus-sonify-01-src":   {"kind": "t2v", "w": 1280, "h": 704, "sec": 15, "silent": True},
    "mus-sonify-01-out":   {"kind": "sonify", "src": "mus-sonify-01-src", "sec": 15},
    "mus-musicvideo-01":   {"kind": "manual", "note": _M_CINEMA},
    # voce
    "voice-tts-01":        {"kind": "tts", "text": TTS_TEXT},
    "voice-clone-01-sample": {"kind": "clone_sample",
                              "note": "usa il campione della voce esistente in voices\\ (manifest chiede registrazione reale)"},
    "voice-clone-01-out":  {"kind": "clone_tts", "text": CLONE_TEXT},
    # cinema
    "cinema-film-01":      {"kind": "manual", "note": _M_CINEMA},
    "cinema-lucky-01":     {"kind": "manual", "note": _M_CINEMA + " (screen-capture)"},
    # 3D / mondi
    "d3-mesh-01-src":      {"kind": "manual", "note": _M_SLOT3D},
    "d3-mesh-01-out":      {"kind": "manual", "note": _M_SLOT3D},
    "d3-rig-01":           {"kind": "manual", "note": _M_SLOT3D},
    "d3-mocap-01-src":     {"kind": "manual", "note": _M_SLOT3D + " (video persona reale)"},
    "d3-mocap-01-skel":    {"kind": "manual", "note": _M_SLOT3D},
    "d3-mocap-01-out":     {"kind": "manual", "note": _M_SLOT3D},
    "world-01-src":        {"kind": "manual", "note": _M_SLOT3D},
    "world-01-out":        {"kind": "manual", "note": _M_SLOT3D},
    # LoRA / UI
    "lora-photo-01-src":   {"kind": "manual",
                            "note": "richiede PuLID + dataset coerente, non nella mappa API validata"},
    "lora-photo-01-out":   {"kind": "manual", "note": "richiede training LoRA — da fare a parte"},
    "lora-video-01":       {"kind": "manual", "note": "richiede training LoRA video — da fare a parte"},
    "models-ui-01":        {"kind": "manual", "note": "screenshot reale della UI — da fare a mano"},
}

# ═════════════════════════════ util ══════════════════════════════
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stable_seed(pid: str) -> int:
    """Seed deterministico derivato dall'id del pezzo (riproducibilità)."""
    return int.from_bytes(hashlib.sha256(pid.encode("utf-8")).digest()[:4], "big") % (2**31 - 1)


def frames_for(sec: int) -> int:
    """num_frames = durata*fps arrotondato a 8n+1."""
    return (sec * FPS // 8) * 8 + 1


def ensure_dirs() -> None:
    for d in (ASSETS, WORK, FLAT_OUT):
        d.mkdir(parents=True, exist_ok=True)


def copy_flat(path: Path) -> Path:
    """Copia il file anche nella cartella output piatta come LANDING_<nome>."""
    FLAT_OUT.mkdir(parents=True, exist_ok=True)
    dst = FLAT_OUT / f"LANDING_{path.name}"
    if path.resolve() != dst.resolve():
        shutil.copy2(path, dst)
    return dst


SERVER_PY = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")
SERVER_LOG = REPO / "output" / "server8189.log"
SERVER_ERR = REPO / "output" / "server8189.err"


def server_up(timeout: float = 2.0) -> bool:
    """Il server risponde? (la root dà 404: basta che risponda HTTP)."""
    try:
        urllib.request.urlopen("http://127.0.0.1:8189/", timeout=timeout)
        return True
    except urllib.error.HTTPError:                          # 404 = vivo
        return True
    except Exception:                                       # noqa: BLE001 — connection refused = morto
        return False


def stop_native_server() -> None:
    """SPEGNE il server immagini prima dei job VIDEO. Non basta /free: il processo server, dopo aver
    caricato/scaricato più modelli, destabilizza il worker LTX che parte subito dopo → access violation
    (exit 3221225477 = 0xC0000005). Misurato: hero (server fresco) OK; vid-t2v-01 dopo 8 modelli → crash.
    Single-GPU: un pesante alla volta. Il server viene riacceso da solo al prossimo job immagine."""
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "Where-Object { $_.CommandLine -match 'inference_server.server' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                   check=False, capture_output=True)
    _LAST_IMG_PIPE["key"] = ("<ignoto>", "<ignoto>")        # server spento → stato ignoto
    time.sleep(4)                                           # lascia rilasciare la VRAM
    log("  server :8189 spento (GPU libera per il worker LTX)")


def restart_native_server() -> None:
    """Riavvia il server immagini. NECESSARIO al cambio di pipeline: il server ACCUMULA stato e dopo aver
    caricato FLUX_Dev (t2i/kontext) il load di FLUX-Fill lo fa CRASHARE SECCO (processo morto, nessun
    traceback) → da lì ogni job va in connection-refused. /free NON basta (misurato: /free ok → crash lo
    stesso). Con processo FRESCO lo stesso inpaint passa in 75s. Costa ~30s, ma è l'unico modo affidabile."""
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "Where-Object { $_.CommandLine -match 'inference_server.server' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                   check=False, capture_output=True)
    time.sleep(3)
    env = dict(os.environ, NATIVE_ATTENTION="auto", NATIVE_FBCACHE="1", NATIVE_FLUX1_FP8="1")
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([str(SERVER_PY), "-u", "-m", "inference_server.server"], cwd=str(REPO), env=env,
                     stdout=open(SERVER_LOG, "ab"), stderr=open(SERVER_ERR, "ab"))
    for _ in range(90):                                     # ~3 min di grazia per l'avvio
        if server_up():
            log("  server :8189 riavviato e pronto")
            return
        time.sleep(2)
    raise RuntimeError("il server :8189 non è tornato su dopo il riavvio")


def ffmpeg_exe() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                       # noqa: BLE001
        raise RuntimeError("ffmpeg non trovato (né in PATH né imageio-ffmpeg)")


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def to_mp3(src: Path, dst: Path) -> None:
    run_ffmpeg(["-i", str(src), "-c:a", "libmp3lame", "-b:a", "320k", str(dst)])


# ═════════════════════════ stato ══════════════════════════
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            log(f"ATTENZIONE: {STATE_FILE.name} corrotto, riparto da zero")
    return {}


def save_state(state: dict) -> None:
    ensure_dirs()
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ═════════════════════════ piano dal manifest ══════════════════════════
def piece_prompt(piece: dict) -> str:
    for k, v in piece.items():
        if k.startswith("prompt"):
            return v
    return ""


def build_plan(manifest: dict) -> tuple[list[dict], list[str]]:
    """Piano ordinato (ordine manifest) + lista di discrepanze trovate."""
    plan: list[dict] = []
    problems: list[str] = []
    seen: set[str] = set()
    for slot in manifest.get("slots", []):
        sid = slot.get("id", "?")
        for piece in slot.get("pezzi", []):
            pid = piece.get("id", "?")
            if pid in seen:
                problems.append(f"id DOPPIO nel manifest: {pid}")
                continue
            seen.add(pid)
            spec = SPECS.get(pid)
            if spec is None:
                problems.append(f"pezzo senza handler nello script: {pid} (slot {sid})")
                spec = {"kind": "manual", "note": "nessun handler previsto nello script"}
            if not piece_prompt(piece) and spec["kind"] != "manual":
                problems.append(f"pezzo senza campo prompt*: {pid}")
            plan.append({"pid": pid, "slot": sid, "tipo": slot.get("tipo", "?"),
                         "spec": spec, "piece": piece})
    for pid in SPECS:
        if pid not in seen:
            problems.append(f"handler nello script per un pezzo NON presente nel manifest: {pid}")
    return plan, problems


# ═════════════════════════ backend immagini (:8189) ══════════════════════════
# (arch, model) dell'ultimo job immagine. All'avvio è IGNOTO, non None: il server può essere già vivo con
# una pipeline residua di una sessione precedente (es. FLUX-Fill), e mandargli un job di arch diversa lo fa
# crashare → il primo job immagine deve SEMPRE riavviarlo per partire da uno stato certo.
_LAST_IMG_PIPE: dict = {"key": ("<ignoto>", "<ignoto>")}


def gen_image_native(params: dict, timeout: float = 3600) -> Path:
    """Job immagine sul server :8189. NB: il server TIENE in VRAM la pipeline del job precedente e,
    cambiando arch/checkpoint (flux t2i ↔ kontext edit ↔ inpaint FLUX-Fill), ne carica una nuova SENZA
    scaricare la vecchia → le ACCUMULA fino a saturare i 32GB e ogni job successivo muore con
    "OOM: tried to allocate 160MiB, 10GB free" (e lascia il modello mezzo su CPU → cascata di fallimenti).
    Liberiamo esplicitamente PRIMA di ogni cambio di pipeline; a parità di arch+modello NON liberiamo
    (niente reload inutile tra t2i consecutivi)."""
    from core import native_driver
    key = (params.get("arch"), params.get("model"))
    if _LAST_IMG_PIPE["key"] is not None and key != _LAST_IMG_PIPE["key"]:
        log(f"  cambio pipeline {_LAST_IMG_PIPE['key']} → {key}: riavvio il server")
        restart_native_server()
    elif not server_up():                                   # crashato per conto suo → risuscitalo
        log("  server :8189 non risponde → riavvio")
        restart_native_server()
    _LAST_IMG_PIPE["key"] = key
    entry = native_driver.run_and_wait("image", params, timeout=timeout)
    if entry.get("status") != "success" or not entry.get("output"):
        raise RuntimeError(entry.get("error") or f"job immagine fallito (status={entry.get('status')})")
    return Path(entry["output"][0])


def save_image_asset(src_img: Path, pid: str) -> list[Path]:
    """PNG lossless in _work (input per i pezzi a valle) + WebP q85 in assets + copia piatta."""
    from PIL import Image
    ensure_dirs()
    im = Image.open(src_img)
    im.load()
    png = WORK / f"{pid}.png"
    im.save(png)
    webp = ASSETS / f"{pid}.webp"
    im.convert("RGB").save(webp, "WEBP", quality=85)
    return [webp, copy_flat(webp)]


def _need_work_png(src_pid: str) -> Path:
    p = WORK / f"{src_pid}.png"
    if not p.exists():
        raise RuntimeError(f"dipendenza non pronta: {src_pid} ({p})")
    return p


def _need_asset(src_pid: str, ext: str) -> Path:
    p = ASSETS / f"{src_pid}.{ext}"
    if not p.exists():
        raise RuntimeError(f"dipendenza non pronta: {src_pid} ({p})")
    return p


def do_t2i(pid: str, spec: dict, prompt: str) -> list[Path]:
    out = gen_image_native({"arch": "flux", "model": FLUX_MODEL, "prompt": prompt,
                            "width": spec["w"], "height": spec["h"], "seed": stable_seed(pid)})
    return save_image_asset(out, pid)


def do_kontext(pid: str, spec: dict, prompt: str) -> list[Path]:
    src = _need_work_png(spec["src"])
    out = gen_image_native({"arch": "kontext", "model": FLUX_MODEL, "prompt": prompt,
                            "width": spec["w"], "height": spec["h"], "seed": stable_seed(pid),
                            "image_path": str(src)})
    return save_image_asset(out, pid)


def do_mask(pid: str, spec: dict, _prompt: str) -> list[Path]:
    """Maschera programmatica: blob bianco morbido su nero nella regione indicata (PIL)."""
    from PIL import Image, ImageDraw, ImageFilter
    ensure_dirs()
    src = _need_work_png(spec["src"])
    w, h = Image.open(src).size
    cx, cy, rx, ry = spec["region"]
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    d.ellipse([w * (cx - rx), h * (cy - ry), w * (cx + rx), h * (cy + ry)], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(radius=max(4, int(w * 0.02))))
    m.save(WORK / f"{pid}.png")                            # input reale per l'inpaint
    webp = ASSETS / f"{pid}.webp"
    m.convert("RGB").save(webp, "WEBP", quality=85)
    return [webp, copy_flat(webp)]


def do_inpaint(pid: str, spec: dict, prompt: str) -> list[Path]:
    src = _need_work_png(spec["src"])
    mask = _need_work_png(spec["mask"])
    out = gen_image_native({"arch": "inpaint", "model": FILL_MODEL, "prompt": prompt,
                            "width": spec["w"], "height": spec["h"], "seed": stable_seed(pid),
                            "image_path": str(src), "mask_path": str(mask)})
    return save_image_asset(out, pid)


def do_outpaint(pid: str, spec: dict, prompt: str) -> list[Path]:
    """Outpaint = inpaint su tela allargata: sorgente centrata, bianco (rigenera) ai lati."""
    from PIL import Image, ImageDraw, ImageFilter
    ensure_dirs()
    src = Image.open(_need_work_png(spec["src"])).convert("RGB")
    W, H = spec["w"], spec["h"]
    sw = int(src.width * H / src.height)
    src = src.resize((sw, H))
    x0 = (W - sw) // 2
    canvas = Image.new("RGB", (W, H), (127, 127, 127))
    canvas.paste(src, (x0, 0))
    canvas_p = WORK / f"{pid}.canvas.png"
    canvas.save(canvas_p)
    feather = max(16, W // 32)                             # la maschera morde i bordi della foto
    m = Image.new("L", (W, H), 255)
    ImageDraw.Draw(m).rectangle([x0 + feather, 0, x0 + sw - feather, H], fill=0)
    m = m.filter(ImageFilter.GaussianBlur(radius=feather // 2))
    mask_p = WORK / f"{pid}.mask.png"
    m.save(mask_p)
    out = gen_image_native({"arch": "inpaint", "model": FILL_MODEL, "prompt": prompt,
                            "width": W, "height": H, "seed": stable_seed(pid),
                            "image_path": str(canvas_p), "mask_path": str(mask_p)})
    return save_image_asset(out, pid)


# ═════════════════════════ backend video (worker LTX-2 diretto) ══════════════════════════
def run_ltx_worker(params: dict, tag: str) -> None:
    ensure_dirs()
    pj = WORK / f"{tag}.params.json"
    pj.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    log(f"  worker LTX: {params.get('width')}x{params.get('height')} "
        f"{params.get('num_frames')}f steps={params.get('steps')} seed={params.get('seed')}")
    proc = subprocess.Popen([str(LTX_PY), "-u", str(LTX_WORKER), str(pj)],
                            cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(f"  [{tag}] {line}")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"worker LTX exit code {rc}")
    outp = Path(params["output"])
    if not outp.exists() or outp.stat().st_size < 1024:
        raise RuntimeError(f"worker LTX ok ma output mancante/vuoto: {outp}")


def _base_video_params(pid: str, spec: dict, prompt: str, out: Path) -> dict:
    p = {"prompt": prompt, "width": spec["w"], "height": spec["h"],
         "num_frames": frames_for(spec["sec"]), "fps": FPS, "steps": VIDEO_STEPS,
         "seed": stable_seed(pid), "output": str(out)}
    if spec.get("chunk"):
        p["chunk_frames"] = spec["chunk"]                  # 60s = catena autoregressiva
    return p


def do_t2v(pid: str, spec: dict, prompt: str) -> list[Path]:
    ensure_dirs()
    stop_native_server()
    final = ASSETS / f"{pid}.mp4"
    post = spec.get("silent") or spec.get("degrade") or spec.get("web")
    raw = (WORK / f"{pid}.raw.mp4") if post else final
    run_ltx_worker(_base_video_params(pid, spec, prompt, raw), pid)
    if spec.get("web"):                                    # sfondo hero: ricompressione leggera per il web
        run_ffmpeg(["-i", str(raw), "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(final)])
    elif spec.get("silent"):                               # clip muta (per la demo MMAudio)
        run_ffmpeg(["-i", str(raw), "-c:v", "copy", "-an", str(final)])
    elif spec.get("degrade"):                              # degradato di proposito (vid-upscale src)
        dw, dh = spec["degrade"]
        run_ffmpeg(["-i", str(raw), "-vf", f"scale={dw}:{dh}:flags=bicubic",
                    "-c:v", "libx264", "-b:v", "300k", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "64k", str(final)])
    return [final, copy_flat(final)]


def do_i2v(pid: str, spec: dict, prompt: str) -> list[Path]:
    ensure_dirs()
    src = _need_work_png(spec["src"])
    stop_native_server()
    final = ASSETS / f"{pid}.mp4"
    params = _base_video_params(pid, spec, prompt, final)
    params["image_path"] = str(src)
    run_ltx_worker(params, pid)
    return [final, copy_flat(final)]


def do_canny_guide(pid: str, spec: dict, _prompt: str) -> list[Path]:
    """Estrae la guida canny dal video sorgente (pezzo pubblicato della tripletta)."""
    from core import controlnet_preprocess
    ensure_dirs()
    src = _need_asset(spec["src"], "mp4")
    final = ASSETS / f"{pid}.mp4"
    controlnet_preprocess.preprocess_video(str(src), "canny", str(final))
    return [final, copy_flat(final)]


def do_control_t2v(pid: str, spec: dict, prompt: str) -> list[Path]:
    """Video finale guidato canny (tripletta ControlNet / cambia-soggetto)."""
    ensure_dirs()
    if spec.get("guide"):                                  # guida già pubblicata come pezzo
        guide = _need_asset(spec["guide"], "mp4")
    else:                                                  # cambia-soggetto: guida in _work dal src
        from core import controlnet_preprocess
        src = _need_asset(spec["guide_from"], "mp4")
        guide = WORK / f"{pid}.guide.mp4"
        if not guide.exists():
            controlnet_preprocess.preprocess_video(str(src), "canny", str(guide))
    stop_native_server()
    final = ASSETS / f"{pid}.mp4"
    params = _base_video_params(pid, spec, prompt, final)
    params.update({"control_video": str(guide), "control_type": "canny",
                   "control_strength": 0.8})
    run_ltx_worker(params, pid)
    return [final, copy_flat(final)]


# ═════════════════════════ backend audio ══════════════════════════
def _run_music_job(job, tag: str) -> None:
    rc = job.run()                                         # bloccante; i log passano da on_log
    if rc != 0:
        raise RuntimeError(f"job {tag} exit code {rc}")


def do_heartmula(pid: str, spec: dict, _prompt: str) -> list[Path]:
    from core import music_service
    ensure_dirs()
    p = music_service.SongParams(name=f"LANDING_{pid}", lyrics=spec["lyrics"],
                                 tags=spec["tags"], duration_s=spec["sec"])
    errs = music_service.preflight_song(p)
    if errs:
        raise RuntimeError("; ".join(errs))
    stop_native_server()
    job = music_service.make_song_job(p, on_log=lambda s: log(f"  [heartmula] {s}"))
    _run_music_job(job, pid)
    mp3 = p.out_mp3()                                      # già E:\AI\ltxstudio\output\LANDING_<pid>.mp3
    if not mp3.exists():
        raise RuntimeError(f"HeartMuLa ok ma output mancante: {mp3}")
    asset = ASSETS / f"{pid}.mp3"
    shutil.copy2(mp3, asset)
    return [asset, mp3]


def do_acestep(pid: str, spec: dict, _prompt: str) -> list[Path]:
    from core import acestep_service
    ensure_dirs()
    errs = acestep_service.preflight(spec["tags"])
    if errs:
        raise RuntimeError("; ".join(errs))
    stop_native_server()
    name = f"LANDING_{pid}"
    job = acestep_service.make_job(spec["tags"], "", name, duration=float(spec["sec"]),
                                   on_log=lambda s: log(f"  [acestep] {s}"))
    _run_music_job(job, pid)
    wav = Path(acestep_service.OUT_DIR) / f"{name}.wav"
    if not wav.exists():
        raise RuntimeError(f"ACE-Step ok ma output mancante: {wav}")
    asset = ASSETS / f"{pid}.mp3"
    to_mp3(wav, asset)
    return [asset, copy_flat(asset)]


def do_sonify(pid: str, spec: dict, _prompt: str) -> list[Path]:
    """MMAudio: sonorizza la clip muta (audio dedotto dai frame, prompt vuoto)."""
    from core import music_service
    ensure_dirs()
    src = _need_asset(spec["src"], "mp4")
    stop_native_server()
    t0 = time.time()
    p = music_service.SonifyParams(name=f"LANDING_{pid}", video_path=str(src),
                                   prompt="", duration_s=float(spec["sec"]),
                                   seed=stable_seed(pid))
    job = music_service.make_sonify_job(p, on_log=lambda s: log(f"  [mmaudio] {s}"))
    _run_music_job(job, pid)
    # MMAudio scrive <video>.flac + <video>.mp4 in MUSIC_OUTPUT_DIR: prendi l'mp4 più recente
    out_dir = Path(music_service.MUSIC_OUTPUT_DIR)
    vids = [f for f in out_dir.glob("*.mp4") if f.stat().st_mtime >= t0]
    if not vids:
        raise RuntimeError(f"MMAudio ok ma nessun mp4 nuovo in {out_dir}")
    newest = max(vids, key=lambda f: f.stat().st_mtime)
    asset = ASSETS / f"{pid}.mp4"
    shutil.copy2(newest, asset)
    return [asset, copy_flat(asset)]


# ═════════════════════════ backend voce ══════════════════════════
_ENGINE_MAP = {"cosyvoice": "cosyvoice2", "f5": "f5tts", "xtts": "xtts2"}


def _pick_voice():
    from core import tts_engine
    voices = tts_engine.list_user_voices()
    if not voices:
        raise RuntimeError("nessuna voce in voices\\ — crearne una dalla GUI (tab Voce)")
    for v in voices:                                       # preferisci una voce con campione (clone)
        if v.sample_path and Path(v.sample_path).exists():
            return v
    return voices[0]


def _synthesize(pid: str, text: str, *, clone: bool = False) -> list[Path]:
    """Sintesi via Handy-API (:8720), l'unico stack voce REALMENTE installato qui: l'unica voce utente
    ('Pluto') è registrata su cosyvoice2, che NON è installato → il motore della voce non si può usare.
      - narrazione (clone=False) → kokoro (TTS nativo, nessun campione richiesto)
      - clone (clone=True)      → clone VERO: registriamo il campione della voce in Handy (chatterbox)
                                  e sintetizziamo con quella. Niente surrogati: la demo di clonazione
                                  dev'essere una clonazione vera."""
    from core import handy_api, tts_engine
    ensure_dirs()
    v = _pick_voice()
    if not handy_api.is_up():
        raise RuntimeError("Handy-API (:8720) non attivo: serve per le voci (cosyvoice2 non installato)")
    if clone:
        if not (v.sample_path and Path(v.sample_path).exists()):
            raise RuntimeError(f"la voce '{v.name}' non ha campione: impossibile clonarla davvero")
        # clone_voice manda il campione in BASE64 dentro un JSON (+33% di peso): col wav PCM originale
        # (34s, 44.1kHz, 2.9MB → ~3.9MB di payload) Handy risponde "HTTP 413 Payload Too Large".
        # Serve un campione COMPRESSO, non solo più corto: mp3 20s mono 16kHz 64k = 157KB (b64 210KB) → ok.
        ref = WORK / f"{pid}.ref.mp3"
        if not ref.exists():
            run_ffmpeg(["-i", str(v.sample_path), "-t", "20", "-ac", "1", "-ar", "16000",
                        "-b:a", "64k", str(ref)])
        log(f"  clone REALE della voce {v.name} — campione {ref.name} ({ref.stat().st_size / 1048576:.2f} MB)")
        handy_api.clone_voice(v.name, str(ref))             # registra il clone in Handy
        engine = "handy_clone"
    else:
        engine = "kokoro"
    log(f"  voce: {v.name} (engine {engine})")
    stop_native_server()
    wav = tts_engine.synthesize(engine, v, text)
    asset = ASSETS / f"{pid}.mp3"
    to_mp3(Path(wav), asset)
    return [asset, copy_flat(asset)]


def do_tts(pid: str, spec: dict, _prompt: str) -> list[Path]:
    return _synthesize(pid, spec["text"])


def do_clone_sample(pid: str, spec: dict, _prompt: str) -> list[Path]:
    """Pubblica il campione reale della voce clonata (già presente in voices\\samples)."""
    ensure_dirs()
    v = _pick_voice()
    if not (v.sample_path and Path(v.sample_path).exists()):
        raise RuntimeError(f"la voce '{v.name}' non ha un campione audio")
    asset = ASSETS / f"{pid}.mp3"
    to_mp3(Path(v.sample_path), asset)
    return [asset, copy_flat(asset)]


def do_clone_tts(pid: str, spec: dict, _prompt: str) -> list[Path]:
    return _synthesize(pid, spec["text"], clone=True)


HANDLERS = {
    "t2i": do_t2i, "kontext": do_kontext, "mask": do_mask, "inpaint": do_inpaint,
    "outpaint": do_outpaint, "t2v": do_t2v, "i2v": do_i2v, "canny_guide": do_canny_guide,
    "control_t2v": do_control_t2v, "heartmula": do_heartmula, "acestep": do_acestep,
    "sonify": do_sonify, "tts": do_tts, "clone_sample": do_clone_sample,
    "clone_tts": do_clone_tts,
}


# ═════════════════════════ dry-run / esecuzione ══════════════════════════
def describe(item: dict) -> str:
    spec = item["spec"]
    k = spec["kind"]
    if k == "manual":
        return "MANUAL"
    bits = [k]
    if "w" in spec:
        bits.append(f"{spec['w']}x{spec['h']}")
    if "sec" in spec:
        if k in ("t2v", "i2v", "control_t2v"):             # i frame hanno senso solo per i video
            extra = f"+chunk{spec['chunk']}" if spec.get("chunk") else ""
            bits.append(f"{spec['sec']}s={frames_for(spec['sec'])}f{extra}")
        else:
            bits.append(f"{spec['sec']}s")
    if spec.get("silent"):
        bits.append("muto")
    if spec.get("degrade"):
        bits.append(f"degrade→{spec['degrade'][0]}x{spec['degrade'][1]}")
    for dep in ("src", "mask", "guide", "guide_from"):
        if spec.get(dep):
            bits.append(f"{dep}={spec[dep]}")
    bits.append(f"seed={stable_seed(item['pid'])}")
    return " ".join(bits)


def print_plan(plan: list[dict], state: dict, problems: list[str]) -> None:
    counts: dict[str, int] = {}
    print(f"\nPIANO — {len(plan)} pezzi (ordine manifest)\n" + "─" * 100)
    cur_slot = None
    for item in plan:
        if item["slot"] != cur_slot:
            cur_slot = item["slot"]
            print(f"\n▪ slot {cur_slot} [{item['tipo']}]")
        k = item["spec"]["kind"]
        counts[k] = counts.get(k, 0) + 1
        st = state.get(item["pid"], {}).get("status", "-")
        note = item["spec"].get("note", "")
        print(f"    {item['pid']:<26} {describe(item):<70} stato:{st}" + (f"  ({note})" if note else ""))
    print("\nRIEPILOGO per tipo:")
    for k in sorted(counts):
        print(f"  {k:<14} {counts[k]}")
    n_manual = counts.get("manual", 0)
    print(f"  → generabili: {len(plan) - n_manual}, manual: {n_manual}")
    if problems:
        print("\nDISCREPANZE MANIFEST/SCRIPT:")
        for p in problems:
            print(f"  ! {p}")
    else:
        print("\nNessuna discrepanza manifest/script.")
    print()


def run_piece(item: dict, state: dict) -> None:
    pid, spec = item["pid"], item["spec"]
    prompt = piece_prompt(item["piece"])
    t0 = time.time()
    if spec["kind"] == "manual":
        state[pid] = {"status": "manual", "note": spec.get("note", ""), "ts": time.strftime("%F %T")}
        log(f"⏭  {pid}: MANUAL — {spec.get('note', '')}")
        save_state(state)
        return
    log(f"▶  {pid}: {describe(item)}")
    try:
        files = HANDLERS[spec["kind"]](pid, spec, prompt)
        dt = time.time() - t0
        state[pid] = {"status": "done", "elapsed_s": round(dt, 1),
                      "files": [str(f) for f in files],
                      "note": spec.get("note", ""), "ts": time.strftime("%F %T")}
        log(f"✔  {pid}: DONE in {dt:.0f}s → {files[0]}")
    except KeyboardInterrupt:
        state[pid] = {"status": "failed", "error": "interrotto dall'utente (Ctrl+C)",
                      "ts": time.strftime("%F %T")}
        save_state(state)
        raise
    except Exception as e:                                 # noqa: BLE001 — mai fatale, si continua
        dt = time.time() - t0
        state[pid] = {"status": "failed", "error": str(e), "elapsed_s": round(dt, 1),
                      "ts": time.strftime("%F %T")}
        log(f"✘  {pid}: FAILED in {dt:.0f}s — {e}")
    save_state(state)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generatore riprendibile asset landing TaDaMoo")
    ap.add_argument("--dry-run", action="store_true", help="stampa il piano senza generare")
    ap.add_argument("--only", metavar="ID", help="rigenera SOLO questo pezzo (o tutti i pezzi dello slot)")
    args = ap.parse_args()

    if not MANIFEST_FILE.exists():
        print(f"Manifest non trovato: {MANIFEST_FILE}")
        return 2
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    plan, problems = build_plan(manifest)
    state = load_state()

    if args.only:
        sel = [i for i in plan if i["pid"] == args.only or i["slot"] == args.only]
        if not sel:
            print(f"--only {args.only}: nessun pezzo/slot con questo id nel manifest")
            return 2
        plan = sel

    if args.dry_run:
        print_plan(plan, state, problems)
        return 0

    for p in problems:
        log(f"DISCREPANZA: {p}")

    done = failed = manual = skipped = 0
    t_start = time.time()
    for item in plan:
        prev = state.get(item["pid"], {}).get("status")
        if not args.only and prev in ("done", "manual"):   # riprendibile: salta ciò che è già fatto
            skipped += 1
            continue
        run_piece(item, state)
        st = state[item["pid"]]["status"]
        done += st == "done"
        failed += st == "failed"
        manual += st == "manual"
    log(f"FINITO in {(time.time() - t_start) / 60:.1f} min — "
        f"done:{done} failed:{failed} manual:{manual} già-fatti-saltati:{skipped}")
    if failed:
        log(f"pezzi falliti: vedi {STATE_FILE} (rilanciando lo script vengono ritentati)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
