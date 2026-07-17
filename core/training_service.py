"""Addestramento LoRA per FLUX.1 e FLUX.2 tramite ai-toolkit (ostris).

Architettura (coerente con gli altri moduli "esterni" dell'app, es. workers/mocap_worker):
- ai-toolkit è installato in un venv ISOLATO (vedi scripts/install_training.py), NON nel venv
  di ComfyUI: ha dipendenze diverse (torch 2.9.1 cu128, optimum-quanto, transformers recenti).
- Il training si avvia come SUBPROCESS: `<venv>/python run.py <config.json>`.
- Il config è generato in JSON (ai-toolkit accetta .json/.jsonc/.yaml/.yml) → nessuna dipendenza
  da PyYAML lato app.
- Il progresso viene parsato dallo stdout/stderr del subprocess (tqdm usa '\r', quindi si legge
  a chunk e si splitta su '\r'/'\n').

NB sui pesi: ai-toolkit carica i modelli in formato HuggingFace/transformers (NON i .safetensors
di ComfyUI). I modelli gated (FLUX.1-dev, FLUX.2-dev, Mistral) richiedono un HF token (env HF_TOKEN).

Selezione del modello base (chiave `model.arch` di ai-toolkit):
- FLUX.1 dev          → is_flux: true            (TE: T5+CLIP, scaricati con il modello)
- FLUX.2 dev          → arch: "flux2"            (TE: Mistral-Small-3.1-24B)
- FLUX.2 Klein 4B     → arch: "flux2_klein_4b"   (TE: Qwen3-4B)
- FLUX.2 Klein 9B     → arch: "flux2_klein_9b"   (TE: Qwen3-8B)
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

# ── Percorsi (allineati al layout E:\AI\… dell'utente) ───────────────────────
AI_TOOLKIT_DIR = Path(r"E:\AI\Tools\ai-toolkit")
VENV_PYTHON = AI_TOOLKIT_DIR / "venv" / "Scripts" / "python.exe"   # Windows
RUN_PY = AI_TOOLKIT_DIR / "run.py"
TRAINING_DATA_DIR = Path(r"E:\AI\TrainingData")                    # progetti dataset+output
LORA_OUTPUT_DIR = Path(r"E:\AI\Models\LoRA\Flux")                  # destinazione finale (catalogo)

# ── Target supportati: etichetta UI → parametri model di ai-toolkit ──────────
# `default_path` è il default per name_or_path (repo HF o cartella locale), editabile in UI.
@dataclass(frozen=True)
class BaseModelSpec:
    key: str
    label: str
    default_path: str          # repo HF o cartella locale
    arch: str | None           # None ⇒ FLUX.1 (usa is_flux)
    is_flux: bool
    note: str                  # avviso pratico (download/VRAM)


BASE_MODELS: dict[str, BaseModelSpec] = {
    "flux1-dev": BaseModelSpec(
        "flux1-dev", "FLUX.1 [dev]", "black-forest-labs/FLUX.1-dev", None, True,
        "~24GB (gated: serve HF token). Ottimo sulla 5090."),
    "flux2-klein-4b": BaseModelSpec(
        "flux2-klein-4b", "FLUX.2 Klein 4B", "ostris/Flux.2-Klein-4B", "flux2_klein_4b", False,
        "Leggero (transformer 4B + Qwen3-4B). Il FLUX.2 più pratico."),
    "flux2-klein-9b": BaseModelSpec(
        "flux2-klein-9b", "FLUX.2 Klein 9B", "ostris/Flux.2-Klein-9B", "flux2_klein_9b", False,
        "Medio (transformer 9B + Qwen3-8B)."),
    "flux2-dev": BaseModelSpec(
        "flux2-dev", "FLUX.2 [dev]", "black-forest-labs/FLUX.2-dev", "flux2", False,
        "PESANTE: ~64GB transformer + ~48GB Mistral. Richiede quantize+low_vram."),
    # Krea 2: ai-toolkit (ostris) supporta l'arch krea2 (trainer ufficiale). Si allena sul RAW
    # (qualità piena) e la LoRA risultante si applica anche sul Turbo distillato. TE Qwen3-VL pesante
    # → quantize_te + low_vram consigliati su 32GB.
    "krea2": BaseModelSpec(
        "krea2", "Krea 2 [Raw]", "krea/Krea-2-Raw", "krea2", False,
        "~26GB transformer + Qwen3-VL. Allena sul Raw; la LoRA vale anche sul Turbo. quantize+low_vram su 32GB."),
}

DEFAULT_SAMPLE_PROMPTS = [
    "portrait photo, natural light, detailed skin texture",
    "full body shot, standing, studio background",
    "close-up, cinematic lighting, shallow depth of field",
]

# tqdm/ai-toolkit: "  5%|▌ | 100/2000 [01:23<25:00,  1.27it/s]" e righe con "loss: 0.1234"
_RE_STEP = re.compile(r"(\d+)\s*/\s*(\d+)")
_RE_LOSS = re.compile(r"loss[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


@dataclass
class TrainParams:
    """Parametri di training raccolti dalla UI."""
    name: str                                  # nome run/LoRA (es. "mia_lora_v1")
    dataset_path: str                          # cartella immagini + .txt caption
    base_model: str = "flux1-dev"              # chiave di BASE_MODELS
    model_path: str = ""                        # override name_or_path (repo HF o cartella locale)
    rank: int = 16                             # network linear (dim LoRA)
    alpha: int = 16                            # network linear_alpha
    lr: float = 1e-4
    steps: int = 2000
    resolutions: list[int] = field(default_factory=lambda: [512, 768, 1024])
    batch_size: int = 1
    save_every: int = 250
    sample_every: int = 250
    quantize: bool = True                      # quantizza il transformer (risparmio VRAM)
    low_vram: bool = False                     # offload su CPU (necessario per FLUX.2 dev su 32GB)
    train_text_encoder: bool = False
    trigger_word: str = ""                     # token attivatore (prepend ai sample)
    sample_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_SAMPLE_PROMPTS))
    seed: int = 42

    def spec(self) -> BaseModelSpec:
        return BASE_MODELS.get(self.base_model, BASE_MODELS["flux1-dev"])

    def project_dir(self) -> Path:
        return TRAINING_DATA_DIR / self.name


def build_config(p: TrainParams) -> dict:
    """Costruisce il dict di config per ai-toolkit (serializzabile in JSON)."""
    spec = p.spec()
    name_or_path = (p.model_path or "").strip() or spec.default_path

    # Sezione model: FLUX.1 usa is_flux, FLUX.2 usa arch.
    model: dict = {"name_or_path": name_or_path, "quantize": bool(p.quantize)}
    if spec.is_flux:
        model["is_flux"] = True
    if spec.arch:
        model["arch"] = spec.arch
        model["quantize_te"] = bool(p.quantize)     # quantizza anche il TE (Mistral/Qwen pesanti)
    if p.low_vram:
        model["low_vram"] = True

    # Prompt di sample: prepend del trigger word se presente.
    tw = (p.trigger_word or "").strip()
    prompts = [f"{tw}, {s}" if tw else s for s in (p.sample_prompts or DEFAULT_SAMPLE_PROMPTS)]

    return {
        "job": "extension",
        "config": {
            "name": p.name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": str(p.project_dir() / "output"),
                "device": "cuda:0",
                "trigger_word": tw or None,
                "network": {"type": "lora", "linear": int(p.rank), "linear_alpha": int(p.alpha)},
                "save": {"dtype": "float16", "save_every": int(p.save_every),
                         "max_step_saves_to_keep": 4, "push_to_hub": False},
                "datasets": [{
                    "folder_path": p.dataset_path,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [int(r) for r in p.resolutions],
                }],
                "train": {
                    "batch_size": int(p.batch_size),
                    "steps": int(p.steps),
                    "gradient_accumulation_steps": 1,
                    "train_unet": True,
                    "train_text_encoder": bool(p.train_text_encoder),
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "lr": float(p.lr),
                    "dtype": "bf16",
                },
                "model": model,
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": int(p.sample_every),
                    "width": 1024, "height": 1024,
                    "prompts": prompts,
                    "neg": "",
                    "seed": int(p.seed),
                    "walk_seed": True,
                    "guidance_scale": 4,
                    "sample_steps": 20,
                },
            }],
        },
        "meta": {"name": p.name, "version": "1.0"},
    }


def write_config(p: TrainParams) -> Path:
    """Scrive il config JSON nel project dir e ritorna il path."""
    proj = p.project_dir()
    proj.mkdir(parents=True, exist_ok=True)
    cfg_path = proj / "config.json"
    cfg_path.write_text(json.dumps(build_config(p), indent=2), encoding="utf-8")
    return cfg_path


def is_installed() -> bool:
    """True se ai-toolkit + venv sono pronti all'uso."""
    return RUN_PY.exists() and VENV_PYTHON.exists()


def preflight(p: TrainParams) -> list[str]:
    """Controlli rapidi PRIMA di lanciare: ritorna lista di problemi (vuota = ok)."""
    errs: list[str] = []
    if not p.name.strip():
        errs.append("Nome run mancante.")
    ds = Path(p.dataset_path or "")
    if not ds.is_dir():
        errs.append(f"Cartella dataset inesistente: {p.dataset_path}")
    else:
        imgs = [f for f in ds.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        if not imgs:
            errs.append("La cartella dataset non contiene immagini (.jpg/.png/.webp).")
    if not is_installed():
        errs.append("ai-toolkit non installato: esegui prima l'installazione del modulo training.")
    spec = p.spec()
    if not (p.model_path or "").strip() and spec.key.startswith("flux2") and not spec.default_path:
        errs.append(f"Specifica il path/repo del modello {spec.label}.")
    return errs


class TrainingRun:
    """Gestisce un singolo processo di training: avvio, parsing progresso, stop.

    Callbacks (tutti opzionali):
      on_log(str)              → riga grezza di log
      on_progress(step, total) → avanzamento step
      on_loss(float)           → ultima loss vista
      on_done(returncode)      → processo terminato
    """

    def __init__(self, params: TrainParams, *, on_log=None, on_progress=None,
                 on_loss=None, on_done=None, hf_token: str | None = None):
        self.params = params
        self.on_log = on_log
        self.on_progress = on_progress
        self.on_loss = on_loss
        self.on_done = on_done
        self.hf_token = hf_token
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = False
        self.last_step = 0
        self.total_steps = int(params.steps)

    def start(self) -> Path:
        """Scrive il config e avvia il subprocess (non bloccante). Ritorna il path config."""
        cfg_path = write_config(self.params)
        env = os.environ.copy()
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
        env.setdefault("PYTHONUNBUFFERED", "1")          # output immediato per il parsing
        env.setdefault("PYTHONIOENCODING", "utf-8")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            [str(VENV_PYTHON), "-u", str(RUN_PY), str(cfg_path)],
            cwd=str(AI_TOOLKIT_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, bufsize=0, creationflags=creationflags,
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return cfg_path

    def _emit_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if self.on_log:
            self.on_log(line)
        m = _RE_STEP.search(line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            if total > 0 and step <= total:                # evita falsi positivi (es. date)
                self.last_step, self.total_steps = step, total
                if self.on_progress:
                    self.on_progress(step, total)
        ml = _RE_LOSS.search(line)
        if ml and self.on_loss:
            try:
                self.on_loss(float(ml.group(1)))
            except ValueError:
                pass

    def _pump(self) -> None:
        """Legge stdout a chunk (tqdm usa '\\r') e spezza in righe logiche."""
        buf = ""
        assert self._proc and self._proc.stdout
        stream = self._proc.stdout
        while True:
            chunk = stream.read(256)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
            buf += text
            # normalizza i ritorni carrello in newline per ottenere righe complete
            buf = buf.replace("\r", "\n")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                self._emit_line(line)
        if buf:
            self._emit_line(buf)
        rc = self._proc.wait()
        if self.on_done:
            self.on_done(rc)

    def stop(self) -> None:
        """Interrompe il training (SIGINT → terminate)."""
        self._stop = True
        if not self._proc or self._proc.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._proc.send_signal(signal.SIGINT)
        except Exception:                                  # noqa: BLE001
            try:
                self._proc.terminate()
            except Exception:                              # noqa: BLE001
                pass

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


def find_final_lora(p: TrainParams) -> Path | None:
    """Trova il .safetensors finale prodotto dal training (per copiarlo nel catalogo)."""
    out = p.project_dir() / "output"
    if not out.exists():
        return None
    cands = sorted(out.rglob("*.safetensors"), key=lambda f: f.stat().st_mtime, reverse=True)
    # scarta gli optimizer/embedding; tieni i LoRA (nome contiene il run name o step)
    for c in cands:
        if "optimizer" not in c.name.lower():
            return c
    return cands[0] if cands else None
