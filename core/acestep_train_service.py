"""Servizio training LoRA ACE-Step: prepara il dataset (name.ext + _prompt.txt + _lyrics.txt) e lancia
il wrapper scripts/acestep_train.py (convert2hf → trainer). HeartMuLa NON ha training (nessuno script).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from core.acestep_service import AS_VENV_PY, AS_DIR, Job, _safe

ROOT = Path(r"E:\AI\ltxstudio")
TRAIN_SCRIPT = ROOT / "scripts" / "acestep_train.py"
WORK_ROOT = Path(r"E:\AI\Models\LoRA\ACEStep")        # dataset + LoRA prodotte
ACE_DIR = Path(r"E:\AI\Tools\ace-step\ACE-Step")
DEFAULT_LORA_CONFIG = ACE_DIR / "config" / "zh_rap_lora_config.json"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class TrainParams:
    audio_paths: list[str] = field(default_factory=list)
    exp_name: str = "music_lora"
    global_tags: str = ""                 # tag applicati ai file SENZA sibling _prompt.txt
    global_lyrics: str = ""               # liriche applicate ai file SENZA sibling _lyrics.txt
    learning_rate: float = 1e-4
    max_steps: int = 2000
    precision: str = "bf16-mixed"
    batch: int = 1
    repeat_count: int = 2000
    lora_config: str = str(DEFAULT_LORA_CONFIG)


def is_installed() -> bool:
    return TRAIN_SCRIPT.exists() and AS_VENV_PY.exists() and (ACE_DIR / "trainer.py").exists()


def preflight(p: TrainParams) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("ACE-Step o il suo venv non sono installati (serve trainer.py).")
    audios = [a for a in p.audio_paths if Path(a).suffix.lower() in AUDIO_EXTS and Path(a).is_file()]
    if len(audios) < 1:
        errs.append("Aggiungi almeno un file audio al dataset (trascina file o cartelle).")
    if not (p.exp_name or "").strip():
        errs.append("Dai un nome alla LoRA (exp_name).")
    return errs


def _sidecar_text(audio: Path, suffix: str) -> str | None:
    """Ritorna il contenuto del file gemello <name>_prompt.txt / <name>_lyrics.txt se esiste."""
    cand = audio.with_name(audio.stem + suffix)
    if cand.exists():
        try:
            return cand.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def build_dataset(p: TrainParams) -> Path:
    """Prepara la cartella data/ nel formato richiesto: per ogni audio → copia + _prompt.txt + _lyrics.txt."""
    exp = _safe(p.exp_name)
    data_dir = WORK_ROOT / exp / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for src in p.audio_paths:
        sp = Path(src)
        if sp.suffix.lower() not in AUDIO_EXTS or not sp.is_file():
            continue
        stem = _safe(sp.stem)
        dst = data_dir / f"{stem}{sp.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(sp, dst)
        tags = _sidecar_text(sp, "_prompt.txt") or p.global_tags or ""
        lyr = _sidecar_text(sp, "_lyrics.txt") or p.global_lyrics or ""
        (data_dir / f"{stem}_prompt.txt").write_text(tags, encoding="utf-8")
        (data_dir / f"{stem}_lyrics.txt").write_text(lyr, encoding="utf-8")
    return data_dir


def make_job(p: TrainParams, **cbs) -> Job:
    exp = _safe(p.exp_name)
    data_dir = build_dataset(p)
    base = WORK_ROOT / exp
    dataset_path = base / "hf_dataset"
    logger_dir = base / "logs"
    cmd = [AS_VENV_PY, "-u", str(TRAIN_SCRIPT),
           "--data-dir", str(data_dir), "--dataset-path", str(dataset_path),
           "--repeat", int(p.repeat_count), "--exp-name", exp,
           "--lr", float(p.learning_rate), "--max-steps", int(p.max_steps),
           "--precision", p.precision, "--batch", int(p.batch),
           "--lora-config", p.lora_config, "--logger-dir", str(logger_dir)]
    return Job(cmd, AS_DIR, **cbs)
