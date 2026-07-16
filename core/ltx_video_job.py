"""Punto UNICO per generare una clip col worker LTX-2 nativo.

Perché esiste: la stessa sequenza — costruisci i params, scrivi il .json, lancia `ltx2_infer.py`,
controlla l'mp4 — era COPIATA in 5 posti (music_video_service ×2, lora_video_service ×2,
film_studio/gen_service). Ogni copia aveva le sue scelte, e così il fix #158 era arrivato solo in
alcune: il videoclip continuava a forzare il GGUF (dequant per-forward ≈ 500 s/step ⇒ oltre un'ora a
clip, e OOM col Q6_K residente) e produceva cartelle piene di .json senza un solo video.

Qui la policy sta in UN POSTO solo:
  • NIENTE GGUF di default. In auto il worker sceglie: fp8 residente se entra in VRAM (~4 s/step),
    GGUF+cpu come fallback per le GPU piccole. Il GGUF si usa solo se l'utente lo sceglie.
  • NIENTE offload forzato: `_auto_offload` del worker decide 'none' (residente, veloce) o 'cpu'.
  • VRAM liberata prima del job (il FLUX dei keyframe resta residente in sticky-mode e farebbe
    sbordare il transformer video da 22B in shared-VRAM).
  • L'assenza dell'mp4 è un ERRORE, con l'output del worker allegato (niente fallimenti muti).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
LTX2_WORKER = str(REPO / "workers" / "ltx2_infer.py")
# 2ª architettura video: LingBot-Video (venv + worker dedicati). Selezione via params["arch"].
# venv311 = Python 3.11 + torch 2.11 cu130 + FlashAttention-2 (sm120): la ricetta che genera CORRETTAMENTE
# (FA + caption JSON ricca + --batch_cfg). Il venv cp312 senza FA dava garbage. Vedi [[lingbot-video-2nd-arch]].
LINGBOT_PY = r"E:\AI\Tools\lingbot_venv311\Scripts\python.exe"
LINGBOT_WORKER = str(REPO / "workers" / "lingbot_infer.py")

# quale (python, worker) usare per ogni architettura
_ARCH_RUNNER = {
    "ltx": (LTX2_PY, LTX2_WORKER),
    "lingbot": (LINGBOT_PY, LINGBOT_WORKER),
}

NEG = "worst quality, inconsistent motion, blurry, jittery, distorted"


class VideoJobError(RuntimeError):
    """La clip non è stata prodotta. Porta con sé l'output del worker."""


def free_video_vram(timeout: int = 20, min_free_mb: int = 24000, wait_s: int = 40) -> int:
    """Sfratta i modelli delle altre categorie E ASPETTA che la VRAM sia DAVVERO libera.

    Il /free è ASINCRONO: torna subito, ma il rilascio della VRAM avviene dopo. Se il worker da 22B
    parte in quel buco, CUDA alloca in SHARED memory (WDDM) — e lì RESTA, anche quando la VRAM si
    libera: GPU al 60% ma dedicata 9.7GB e shared 30GB, cioè il modello che gira nella RAM di
    sistema, 10-20× più lento. La clip non finisce più.
    Perciò dopo il /free si aspetta il rilascio EFFETTIVO, misurato con nvidia-smi (mai torch.cuda
    nel processo app: creerebbe un contesto CUDA e farebbe crashare i worker isolati).
    """
    try:
        from core import mem_coordinator as _mc
        _mc.free_for("video", timeout=timeout)
    except Exception:                                        # noqa: BLE001 — best-effort, mai fatale
        pass
    free_mb = 0
    try:
        from core import gpu_mem
        for _ in range(wait_s):
            free_mb = gpu_mem.free_vram_mb()
            if free_mb >= min_free_mb:
                break
            time.sleep(1)
        print(f"[video] VRAM libera pre-worker: {free_mb} MB "
              f"({'ok' if free_mb >= min_free_mb else 'ATTENZIONE: sotto la soglia → rischio shared'})",
              flush=True)
    except Exception:                                        # noqa: BLE001
        pass
    return free_mb


def build_params(
    *,
    output: str,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int = 25,
    image_path: str = "",
    keyframes: list | None = None,
    negative: str = NEG,
    steps: int = 8,
    guidance: float = 3.0,
    seed: int = 7,
    model: str | None = None,
    model_gguf: str | None = None,
    arch: str = "ltx",
    extra: dict | None = None,
) -> dict:
    """Params del worker con la policy #158 applicata una volta per tutte.

    arch = "ltx" (default) o "lingbot" (2ª architettura). Il campo viaggia nei params e in run_clip
    seleziona il (python, worker) giusto. La policy GGUF/offload qui sotto è LTX-specifica: per LingBot
    si passano solo model/variant + extra e la memoria la gestisce il suo worker.

    model_gguf (o un `model` che finisce in .gguf) = scelta ESPLICITA dell'utente: solo allora si usa
    il GGUF, e in quel caso l'offload 'cpu' è d'obbligo (il transformer resta quantizzato residente).
    Altrimenti non si passa né gguf né offload: decide il worker.
    """
    arch = (arch or "ltx").lower()
    params: dict = {
        "prompt": prompt, "negative": negative,
        "width": int(width), "height": int(height),
        "num_frames": int(num_frames), "fps": int(fps),
        "steps": int(steps), "guidance": float(guidance), "seed": int(seed),
        "output": output, "arch": arch,
    }
    if keyframes:
        params["keyframes"] = keyframes
    elif image_path:
        params["image_path"] = image_path

    if arch == "lingbot":
        # LingBot: nessuna policy GGUF LTX. model = path cartella diffusers (o vuoto → il worker
        # risolve dal catalogo). La variante (dense/moe) e run_refiner arrivano via `extra`.
        if model:
            params["model"] = model
    else:
        gguf = model_gguf or (model if str(model or "").lower().endswith(".gguf") else None)
        if gguf:                                             # GGUF solo se scelto esplicitamente
            params["gguf_transformer"] = gguf
            params["offload"] = "cpu"
        elif model:                                          # checkpoint non-GGUF esplicito (fp8/bf16)
            params["model"] = model

    if extra:
        params.update(extra)                                 # ref_strength, loras, chunk_frames, audio, lingbot_variant…
    return params


def run_clip(params: dict, *, free_vram: bool = True, timeout: int | None = None) -> str:
    """Scrive il job, lancia il worker, verifica l'mp4. Ritorna il path; solleva se non c'è."""
    out = params["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if free_vram:
        free_video_vram()

    jf = out + ".json"
    with open(jf, "w", encoding="utf-8") as fh:
        json.dump(params, fh)

    arch = (params.get("arch") or "ltx").lower()
    py, worker = _ARCH_RUNNER.get(arch, _ARCH_RUNNER["ltx"])
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    r = subprocess.run([py, "-u", worker, jf], env=env,
                       capture_output=True, text=True, timeout=timeout)
    if not os.path.exists(out):
        tail = ((r.stdout or "")[-500:] + (r.stderr or "")[-500:]).strip()
        raise VideoJobError(f"clip non generata ({os.path.basename(out)}): {tail}")
    return out


def generate(*, free_vram: bool = True, timeout: int | None = None, **kw) -> str:
    """Scorciatoia: build_params(**kw) + run_clip()."""
    return run_clip(build_params(**kw), free_vram=free_vram, timeout=timeout)
