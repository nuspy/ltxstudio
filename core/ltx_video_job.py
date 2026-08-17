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
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
LTX2_WORKER = str(REPO / "workers" / "ltx2_infer.py")
# 2ª architettura video: LingBot-Video (venv + worker dedicati). Selezione via params["arch"].
# venv311 = Python 3.11 + torch 2.11 cu130 + FlashAttention-2 (sm120): la ricetta che genera CORRETTAMENTE
# (FA + caption JSON ricca + --batch_cfg). Il venv cp312 senza FA dava garbage. Vedi [[lingbot-video-2nd-arch]].
LINGBOT_PY = r"E:\AI\Tools\lingbot_venv311\Scripts\python.exe"
LINGBOT_WORKER = str(REPO / "workers" / "lingbot_infer.py")

# 3ª architettura video: Krea Realtime 14B (autoregressivo, venv dedicato).
KREA_RT_PY = r"E:\AI\Tools\krea_rt_venv\Scripts\python.exe"
KREA_RT_WORKER = str(REPO / "workers" / "krea_rt_infer.py")

# variante REALTIME di krea_rt: Self-Forcing 1.3B (autoregressivo leggero, ~realtime sulla 5090). Riusa il
# venv krea_rt_venv; il worker gira il runner del repo Self-Forcing come libreria. Vedi [[krea-arches]].
SELF_FORCING_WORKER = str(REPO / "workers" / "self_forcing_infer.py")

# 4ª architettura video: MiniMax H3 (omni-modale 33.1B: video + audio stereo nella STESSA passata,
# riferimenti multipli nativi). Venv dedicato: serve diffusers con le classi MiniMaxH3* + transformers
# con Qwen3-VL. Vedi workers/minimax_h3_infer.py e scripts/install_minimax_h3.py.
MINIMAX_H3_PY = r"E:\AI\Tools\minimax_h3_venv\Scripts\python.exe"
MINIMAX_H3_WORKER = str(REPO / "workers" / "minimax_h3_infer.py")

# "Krea Realtime" nell'app = motore **Self-Forcing 1.3B** (l'unico VALIDATO: realtime ~2s di calcolo per 1s
# di video, decode streaming-su-disco + sfratto KV-cache → durata illimitata a memoria costante, VRAM sotto
# saturazione — confermato da log storici + mp4 reali). Il transformer **14B** (`krea_rt_infer`) è rimasto
# uno scaffold pesante (fp8 layerwise + flex_attention, ~1000× più lento, mai validato) → resta raggiungibile
# SOLO come opt-in di debug con env `KREA_RT_USE_14B=1`. Mettendo il redirect DENTRO `_ARCH_RUNNER`, tutti i
# punti di dispatch (run_clip qui sotto + `_NativeLtx2Worker` in main_window, che legge questo dict) prendono
# il motore giusto da un unico posto. Vedi [[krea-arches]].
_KREA_RT_WORKER_EFF = KREA_RT_WORKER if os.environ.get("KREA_RT_USE_14B") == "1" else SELF_FORCING_WORKER

# quale (python, worker) usare per ogni architettura
_ARCH_RUNNER = {
    "ltx": (LTX2_PY, LTX2_WORKER),
    "lingbot": (LINGBOT_PY, LINGBOT_WORKER),
    "krea_rt": (KREA_RT_PY, _KREA_RT_WORKER_EFF),        # → Self-Forcing 1.3B (default) | 14B se KREA_RT_USE_14B=1
    "self_forcing": (KREA_RT_PY, SELF_FORCING_WORKER),
    "minimax_h3": (MINIMAX_H3_PY, MINIMAX_H3_WORKER),
}


def runner_for(params: dict) -> tuple[str, str]:
    """(python, worker) per un job. Per l'arch `ltx` la scelta la fa la RELEASE (2.3/2.5), non l'arch:
    le due release hanno venv incompatibili (2.5 richiede transformers>=5.8 per Gemma 4, 2.3 gira su
    4.x), quindi il runner arriva da `core.ltx_release`. Le altre arch restano su `_ARCH_RUNNER`.
    Punto UNICO: lo usano sia `run_clip` qui sotto sia `_NativeLtx2Worker` in main_window."""
    arch = (params.get("arch") or "ltx").lower()
    if arch == "ltx":
        from core import ltx_release
        rel = params.get("ltx_release") or ltx_release.DEFAULT_RELEASE
        return ltx_release.runner(rel)
    return _ARCH_RUNNER.get(arch, _ARCH_RUNNER["ltx"])

NEG = "worst quality, inconsistent motion, blurry, jittery, distorted"


class VideoJobError(RuntimeError):
    """La clip non è stata prodotta. Porta con sé l'output del worker."""


def free_video_vram(timeout: int = 20, min_free_mb: int | None = None, wait_s: int = 40,
                    *, model: str = "", gguf: str = "", arch: str = "ltx") -> int:
    """Sfratta i modelli delle altre categorie E ASPETTA che la VRAM sia DAVVERO libera.

    Il /free è ASINCRONO: torna subito, ma il rilascio della VRAM avviene dopo. Se il worker da 22B
    parte in quel buco, CUDA alloca in SHARED memory (WDDM) — e lì RESTA, anche quando la VRAM si
    libera: GPU al 60% ma dedicata 9.7GB e shared 30GB, cioè il modello che gira nella RAM di
    sistema, 10-20× più lento. La clip non finisce più.
    Perciò dopo il /free si aspetta il rilascio EFFETTIVO, misurato con nvidia-smi (mai torch.cuda
    nel processo app: creerebbe un contesto CUDA e farebbe crashare i worker isolati).

    `min_free_mb=None` (default) → soglia DINAMICA: il fabbisogno vero del modello (`need_mb`
    dalla cache misurata/header) + headroom, mai oltre la VRAM fisica della scheda — su una GPU
    piccola una soglia fissa non si raggiunge mai e l'attesa degenera in uno sleep pieno.
    """
    # F5 unified-model-manager: chiave modello VERA + need_mb dalla cache misurata/header —
    # prima ModelKey era degenere ("video:-") e need_mb=0: il riuso video non veniva mai
    # riconosciuto e lo sfratto size-aware non scattava.
    # NB: _m/_need FUORI dal try del registro: servono anche alla soglia dinamica qui sotto.
    _m = gguf or model
    _need = 0
    try:
        from core import vram_estimate
        if gguf and os.path.exists(gguf):
            _need = int(os.path.getsize(gguf) / 2**20) + 9000         # GGUF ≈ dim file + working video
        elif model and os.path.exists(model):
            _need, _ = vram_estimate.estimate_mb(model, fp8_storage=True, working_mb=9000)
    except Exception:                                        # noqa: BLE001
        pass
    try:
        from core import gpu_registry                        # registro unico: libera le altre categorie (immagine)
        gpu_registry.before_job("video", arch=arch or "video", model=_m, need_mb=_need, log=print)
    except Exception:                                        # noqa: BLE001 — best-effort, mai fatale
        try:
            from core import mem_coordinator as _mc
            _mc.free_for("video", timeout=timeout)
        except Exception:                                    # noqa: BLE001
            pass
    free_mb = 0
    try:
        from core import gpu_mem
        if min_free_mb is None:                              # soglia dinamica: fabbisogno vero + headroom
            min_free_mb = int(_need) + 2000 if _need else 24000
        total_mb = gpu_mem.total_vram_mb()
        if total_mb:                                         # clamp alla scheda: soglia mai irraggiungibile
            min_free_mb = min(min_free_mb, max(total_mb - 2048, 1024))
        for _ in range(wait_s):
            free_mb = gpu_mem.free_vram_mb()
            if free_mb >= min_free_mb:
                break
            time.sleep(1)
        print(f"[video] VRAM libera pre-worker: {free_mb} MB (soglia {min_free_mb}) "
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
    ltx_release: str | None = None,
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

    if arch != "ltx":
        # Architetture diffusers (lingbot, krea_rt): nessuna policy GGUF LTX. model = path cartella
        # diffusers (o vuoto → il worker risolve dal catalogo). Variante/refiner/kv-cache via `extra`.
        if model:
            params["model"] = model
    else:
        gguf = model_gguf or (model if str(model or "").lower().endswith(".gguf") else None)
        if gguf:                                             # GGUF solo se scelto esplicitamente
            params["gguf_transformer"] = gguf
            params["offload"] = "cpu"
        elif model:                                          # checkpoint non-GGUF esplicito (fp8/bf16)
            params["model"] = model
        # RELEASE LTX (2.3 | 2.5): dedotta dal checkpoint se non dichiarata. Se è dichiarata ma il
        # checkpoint appartiene all'altra release, si ERRORE: i due set non sono intercambiabili
        # (2.5 è uno split pack con Gemma 4, 2.3 un monolite con Gemma 3) e sceglierne una al posto
        # dell'utente violerebbe la regola «mai sostituire il modello senza consenso».
        from core import ltx_release as _rel
        _ck = gguf or model or ""
        _det = _rel.detect_release(_ck) if _ck else None
        if ltx_release and _det and _det != str(ltx_release):
            raise VideoJobError(
                f"Il checkpoint '{os.path.basename(str(_ck))}' è di LTX {_det}, ma la release "
                f"selezionata è LTX {ltx_release}. Scegli un checkpoint {ltx_release} oppure "
                f"cambia release (nessuna sostituzione automatica).")
        params["ltx_release"] = str(ltx_release or _det or _rel.DEFAULT_RELEASE)
        # Cache degli embedding di prompt (core/emb_cache): la preferenza viaggia nei PARAMS, non in
        # env, perché il server persistente non erediterebbe una variabile impostata dopo il suo avvio.
        # Qui, punto unico, la ereditano GUI, API, Film Studio e videoclip senza doverla ripetere.
        try:
            from core import user_prefs as _up
            params["emb_cache"] = bool(_up.get("emb_cache.enabled", True))
        except Exception:                                    # noqa: BLE001 — prefs assenti: cache attiva
            params["emb_cache"] = True

    if extra:
        params.update(extra)                                 # ref_strength, loras, chunk_frames, audio, lingbot_variant…
    return params


def run_clip(params: dict, *, free_vram: bool = True, timeout: int | None = None) -> str:
    """Scrive il job, lancia il worker, verifica l'mp4. Ritorna il path; solleva se non c'è."""
    out = params["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if free_vram:
        free_video_vram(model=str(params.get("model", "") or ""),
                        gguf=str(params.get("gguf_transformer", "") or ""),
                        arch=(params.get("arch") or "ltx").lower())

    jf = out + ".json"
    with open(jf, "w", encoding="utf-8") as fh:
        json.dump(params, fh)

    py, worker = runner_for(params)                          # per 'ltx' decide la RELEASE, non l'arch
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    r = subprocess.run([py, "-u", worker, jf], env=env,
                       capture_output=True, text=True, timeout=timeout)
    # Lo stdout del worker viene scartato quando il job riesce: le diagnosi della cache embedding
    # (hit/miss e secondi risparmiati) andrebbero perse proprio nei casi che ne fanno di più —
    # Film Studio e videoclip, che passano tutti da qui. Ripubblico solo quelle righe.
    for _ln in (r.stdout or "").splitlines():
        if _ln.startswith(("[embcache]", "[emb-probe]")):
            print(_ln, flush=True)
    if not os.path.exists(out):
        tail = ((r.stdout or "")[-500:] + (r.stderr or "")[-500:]).strip()
        raise VideoJobError(f"clip non generata ({os.path.basename(out)}): {tail}")
    return out


def generate(*, free_vram: bool = True, timeout: int | None = None, **kw) -> str:
    """Scorciatoia: build_params(**kw) + run_clip()."""
    return run_clip(build_params(**kw), free_vram=free_vram, timeout=timeout)
