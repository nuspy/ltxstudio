"""LoRA LingBot-Video — training SPERIMENTALE (target = architettura LingBot).

⚠️ SPERIMENTALE / NON VALIDATO: LingBot non ha una ricetta LoRA ufficiale. Il transformer è un DiT
diffusers standard → il training LoRA è tecnicamente possibile via diffusers/peft, ma la ricetta
(flow-matching loss, VAE video, shift/scheduler) va confermata live. Questo modulo:
  - RIUSA il generatore di dataset di `core/lora_video_service` (Subject/plan_clips/coverage/PuLID/
    face-crop/recaption) generando i clip-sorgente col MOTORE LingBot (source_arch='lingbot').
  - lancia un trainer LoRA dedicato `workers/lingbot_train.py` nel venv LingBot.

Formato di output selezionabile nel wizard: LTX-2.3 (usa core.lora_video_service.make_lora) oppure
LingBot (questo modulo). Vedi [[lora-auto-dataset]] e il piano LingBot.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from core import lora_video_service as lvs

REPO = Path(__file__).resolve().parent.parent
LINGBOT_PY = r"E:\AI\Tools\lingbot_venv\Scripts\python.exe"
TRAIN_WORKER = str(REPO / "workers" / "lingbot_train.py")


def trainer_installed() -> bool:
    """True se il venv LingBot esiste (il training gira lì)."""
    return os.path.exists(LINGBOT_PY)


def _resolve_base_model() -> str:
    """Cartella del modello LingBot da fine-tunare (Dense per il training: più leggero)."""
    try:
        from core import weights_catalog as wc
        ent = {e["id"]: e for e in wc.load()}
        for eid in ("lingbot_dense", "lingbot_dense_fp8"):
            e = ent.get(eid)
            if e and wc.is_present(e):
                return str(wc.dest_for(e))
    except Exception:                                        # noqa: BLE001
        pass
    return ""


def train(dataset_dir: str, out_lora: str, *, trigger: str = "", rank: int = 16, steps: int = 1500,
          progress=None) -> str:
    """Training LoRA SPERIMENTALE sul transformer LingBot (subprocess nel venv LingBot)."""
    def log(m):
        if progress:
            progress(m)
    base = _resolve_base_model()
    if not base:
        raise RuntimeError("Modello base LingBot non trovato: scarica 'LingBot-Video Dense 1.3b' dal wizard Pesi.")
    dj = lvs.build_dataset_json(dataset_dir)
    log(f"[lingbot-lora] training SPERIMENTALE (rank={rank}, steps={steps}) su {base}")
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [LINGBOT_PY, "-u", TRAIN_WORKER, "--model", base, "--dataset", dj,
           "--out", out_lora, "--rank", str(rank), "--steps", str(steps), "--trigger", trigger]
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
    proc.wait()
    if not os.path.exists(out_lora):
        raise RuntimeError(f"LoRA LingBot non prodotta (exit {proc.returncode}). Training sperimentale — vedi log.")
    return out_lora


def make_lora(subjects: list, name: str, out_dir: str, *, rank: int = 16, steps: int = 1500,
              clip_seconds: float = 3.0, use_pulid: bool = True,
              caption_engine: str = "procedural", caption_mode: str = "descriptive",
              progress=None) -> dict:
    """Pipeline COMPLETA LoRA LingBot: dataset (clip generati DA LingBot) → train LingBot.
    Ritorna {dataset, lora}. SPERIMENTALE."""
    W, H = 768, 512
    work = os.path.join(out_dir, name)
    ds = os.path.join(work, "dataset")
    # dataset generato col MOTORE LingBot (source_arch='lingbot') riusando tutta la logica esistente
    lvs.generate_dataset(subjects, ds, width=W, height=H, clip_seconds=clip_seconds, use_pulid=use_pulid,
                         caption_engine=caption_engine, caption_mode=caption_mode,
                         source_arch="lingbot", progress=progress)
    trig = ", ".join(sorted({s.trig() for s in subjects}))
    lora = os.path.join(work, name + "_lingbot.safetensors")
    train(ds, lora, trigger=trig, rank=rank, steps=steps, progress=progress)
    return {"dataset": ds, "lora": lora}
