"""Generazione musicale per LTX Studio (motori STANDALONE, non ComfyUI).

Due motori, ciascuno nel proprio venv isolato (come il modulo training), pilotati via subprocess:

- **HeartMuLa** (heartlib) — canzoni complete con voce+liriche+tag, oppure strumentale.
  CLI: examples/run_music_generation.py --model_path ./ckpt --version 3B
       --lyrics F.txt --tags F.txt --save_path out.mp3 --max_audio_length_ms N
       --temperature --cfg_scale --topk --lazy_load
  (liriche e tag sono FILE; output MP3)

- **MMAudio** — sonorizzazione: da un VIDEO genera l'audio sincronizzato (foley/colonna sonora);
  senza video fa text-to-audio.
  CLI: demo.py --video V --prompt P --negative_prompt N --duration S --cfg_strength
       --num_steps --seed --output DIR   (output: <video>.flac + <video>.mp4)

I pesi (HeartMuLa+HeartCodec, MMAudio) si scaricano da HuggingFace (vedi scripts/install_music.py).
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Percorsi (layout E:\AI\Tools come il modulo training) ────────────────────
TOOLS = Path(r"E:\AI\Tools")
HEARTLIB_DIR = TOOLS / "heartlib"
HEARTLIB_VENV_PY = HEARTLIB_DIR / "venv" / "Scripts" / "python.exe"
HEARTLIB_CKPT = HEARTLIB_DIR / "ckpt"
HEARTLIB_RUN = HEARTLIB_DIR / "examples" / "run_music_generation.py"
# Wrapper NOSTRO: monkey-patcha torchaudio.save → soundfile prima di lanciare HeartMuLa
# (torchaudio>=2.9 richiederebbe torchcodec per salvare → la canzone falliva DOPO la generazione).
HEARTLIB_WRAPPER = Path(__file__).resolve().parent.parent / "workers" / "heartmula_run.py"

MMAUDIO_DIR = TOOLS / "MMAudio"
MMAUDIO_VENV_PY = MMAUDIO_DIR / "venv" / "Scripts" / "python.exe"
MMAUDIO_DEMO = MMAUDIO_DIR / "demo.py"

MUSIC_OUTPUT_DIR = Path(r"E:\AI\ltxstudio\output")     # #97: musica/sonify in output\ (root), non in sottocartella

# Default per i prompt di stile (HeartMuLa usa tag separati da virgola)
DEFAULT_TAGS = "pop, upbeat, female vocals, synth, catchy"
INSTRUMENTAL_TAG = "instrumental"

# Progresso: tqdm/loop → "37/100" o percentuali; fine HeartMuLa → "saved to"
_RE_STEP = re.compile(r"(\d+)\s*/\s*(\d+)")
_RE_PCT = re.compile(r"(\d{1,3})\s*%")


def is_installed_heartmula() -> bool:
    return HEARTLIB_RUN.exists() and HEARTLIB_VENV_PY.exists() and HEARTLIB_CKPT.exists()


def is_installed_mmaudio() -> bool:
    return MMAUDIO_DEMO.exists() and MMAUDIO_VENV_PY.exists()


# ───────────────────────── parametri ─────────────────────────
@dataclass
class SongParams:
    """Canzone/strumentale con HeartMuLa."""
    name: str = "song"
    lyrics: str = ""                       # con sezioni [Verse]/[Chorus]/…
    tags: str = DEFAULT_TAGS               # stile, separati da virgola
    duration_s: int = 120                  # durata massima
    instrumental: bool = False
    version: str = "3B"                    # "3B" | "7B"
    temperature: float = 1.0
    cfg_scale: float = 1.5
    topk: int = 50
    lazy_load: bool = False                # offload per VRAM ridotta

    def out_mp3(self) -> Path:
        return MUSIC_OUTPUT_DIR / f"{_safe(self.name)}.mp3"


@dataclass
class SonifyParams:
    """Sonorizzazione video (o text-to-audio) con MMAudio."""
    name: str = "sfx"
    video_path: str = ""                   # vuoto ⇒ text-to-audio
    prompt: str = ""
    negative_prompt: str = ""
    duration_s: float = 8.0
    cfg_strength: float = 4.5
    num_steps: int = 25
    seed: int = 42


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "song").strip()) or "song"


# ───────────────────────── job subprocess generico ─────────────────────────
class Job:
    """Esegue un comando in subprocess, inoltra log e progresso via callback.

    Callbacks: on_log(str), on_progress(cur,total), on_done(returncode).
    Il parsing del progresso è euristico (step "x/y" o percentuali "n%").
    """

    def __init__(self, cmd: list[str], cwd: Path, *, env: dict | None = None,
                 on_log=None, on_progress=None, on_done=None):
        self.cmd = [str(c) for c in cmd]
        self.cwd = cwd
        self.env = env
        self.on_log = on_log
        self.on_progress = on_progress
        self.on_done = on_done
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        if sys.platform == "win32" and "expandable_segments" in env.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)          # #96/2: non supportato su Windows → evita il UserWarning
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            self.cmd, cwd=str(self.cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, bufsize=0, creationflags=flags)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def run(self) -> int:
        """Esecuzione BLOCCANTE: avvia e attende la fine (per chiamate sincrone, es. voice-clone tts_engine).
        Ritorna il returncode."""
        self.start()
        if self._thread is not None:
            self._thread.join()
        return self._proc.returncode if self._proc is not None else -1

    def _emit(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if self.on_log:
            self.on_log(line)
        if self.on_progress:
            m = _RE_STEP.search(line)
            if m and int(m.group(2)) > 0:
                self.on_progress(int(m.group(1)), int(m.group(2)))
            else:
                mp = _RE_PCT.search(line)
                if mp and int(mp.group(1)) <= 100:
                    self.on_progress(int(mp.group(1)), 100)

    def _pump(self) -> None:
        buf = ""
        stream = self._proc.stdout
        while True:
            chunk = stream.read(256)
            if not chunk:
                break
            buf += chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk
            buf = buf.replace("\r", "\n")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                self._emit(line)
        if buf:
            self._emit(buf)
        rc = self._proc.wait()
        if self.on_done:
            self.on_done(rc)

    def stop(self) -> None:
        if not self._proc or self._proc.poll() is not None:
            return
        try:
            self._proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
        except Exception:                                  # noqa: BLE001
            try:
                self._proc.terminate()
            except Exception:                              # noqa: BLE001
                pass

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ───────────────────────── HeartMuLa: canzone/strumentale ─────────────────────────
def write_song_inputs(p: SongParams) -> tuple[Path, Path]:
    """Scrive i file lyrics.txt e tags.txt richiesti dalla CLI; ritorna i due path."""
    MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = MUSIC_OUTPUT_DIR / _safe(p.name)
    lyr = base.with_suffix(".lyrics.txt")
    tag = base.with_suffix(".tags.txt")
    lyrics = "" if p.instrumental else (p.lyrics or "")
    tags = p.tags or DEFAULT_TAGS
    if p.instrumental and INSTRUMENTAL_TAG not in tags.lower():
        tags = f"{INSTRUMENTAL_TAG}, {tags}"
    lyr.write_text(lyrics, encoding="utf-8")
    tag.write_text(tags, encoding="utf-8")
    return lyr, tag


def make_song_job(p: SongParams, **cbs) -> Job:
    lyr, tag = write_song_inputs(p)
    cmd = [HEARTLIB_VENV_PY, "-u", str(HEARTLIB_WRAPPER),
           "--model_path", HEARTLIB_CKPT, "--version", p.version,
           "--lyrics", lyr, "--tags", tag, "--save_path", p.out_mp3(),
           "--max_audio_length_ms", int(max(5, p.duration_s) * 1000),
           "--temperature", p.temperature, "--cfg_scale", p.cfg_scale, "--topk", p.topk,
           "--lazy_load", "true" if p.lazy_load else "false"]
    return Job(cmd, HEARTLIB_DIR, **cbs)


# ───────────────────────── MMAudio: sonorizzazione ─────────────────────────
def make_sonify_job(p: SonifyParams, **cbs) -> Job:
    MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [MMAUDIO_VENV_PY, "-u", MMAUDIO_DEMO,
           "--prompt", p.prompt or "", "--negative_prompt", p.negative_prompt or "",
           "--duration", float(p.duration_s), "--cfg_strength", float(p.cfg_strength),
           "--num_steps", int(p.num_steps), "--seed", int(p.seed),
           "--output", MUSIC_OUTPUT_DIR]
    if p.video_path:
        cmd += ["--video", p.video_path]
    return Job(cmd, MMAUDIO_DIR, **cbs)


def find_latest_audio(after: float = 0.0) -> Path | None:
    """Ultimo file audio prodotto in MUSIC_OUTPUT_DIR (per aprirlo a fine job)."""
    if not MUSIC_OUTPUT_DIR.exists():
        return None
    cands = [f for f in MUSIC_OUTPUT_DIR.iterdir()
             if f.suffix.lower() in (".mp3", ".flac", ".wav") and f.stat().st_mtime >= after]
    return max(cands, key=lambda f: f.stat().st_mtime) if cands else None


def preflight_song(p: SongParams) -> list[str]:
    errs = []
    if not is_installed_heartmula():
        errs.append("HeartMuLa non installato: usa il pulsante d'installazione.")
    if not p.instrumental and not (p.lyrics or "").strip():
        errs.append("Inserisci le liriche, oppure attiva la modalità strumentale.")
    return errs


def preflight_sonify(p: SonifyParams) -> list[str]:
    errs = []
    if not is_installed_mmaudio():
        errs.append("MMAudio non installato: usa il pulsante d'installazione.")
    if p.video_path and not Path(p.video_path).is_file():
        errs.append(f"Video inesistente: {p.video_path}")
    if not p.video_path and not (p.prompt or "").strip():
        errs.append("Indica un video da sonorizzare, oppure un prompt (text-to-audio).")
    return errs
