"""Handler immagini del backend nativo (diffusers): text-to-image, editing (Kontext), inpaint.

Dispatch per `arch`/`task`. Dove la pipeline diffusers supporta `from_single_file` si riusano
i pesi LOCALI ComfyUI (nessun download); altrimenti serve un repo diffusers (`from_pretrained`).

| arch     | pipeline             | single_file | input        |
|----------|----------------------|-------------|--------------|
| flux     | FluxPipeline         | sì          | —            |
| flux2    | Flux2Pipeline        | NO (repo)   | —            |
| zimage   | ZImagePipeline       | sì          | —            |
| ernie    | ErnieImagePipeline   | NO (repo)   | —            |
| kontext  | FluxKontextPipeline  | sì          | image        |
| inpaint  | FluxFillPipeline     | sì          | image + mask |
"""
from __future__ import annotations

import os
from pathlib import Path

# (nome classe diffusers, supporta from_single_file, richiede immagine input)
SPEC = {
    "flux":    ("FluxPipeline",        True,  False),
    "flux2":   ("Flux2Pipeline",       False, False),
    "zimage":  ("ZImagePipeline",      True,  False),
    "ernie":   ("ErnieImagePipeline",  False, False),
    "kontext": ("FluxKontextPipeline", True,  True),
    "inpaint": ("FluxFillPipeline",    True,  True),
    # #14 — FLUX.2 per editing/inpaint/outpaint/kontext (assemblati via _assemble_flux2; assemblaggio
    # locale = supports_sf irrilevante, gestito da arch.startswith('flux2') in _load). I nomi delle
    # classi diffusers vanno VALIDATI su GPU (diffusers evolve): qui le best-guess attuali. L'outpaint
    # riusa flux2_inpaint con maschera INVERTITA (core.mask_utils.invert_mask).
    # FLUX.2 editing per istruzione (no maschera) = Klein (Flux2KontextPipeline NON esiste in diffusers
    # 0.38; l'editing FLUX.2 è la famiglia Klein). API da validare su GPU.
    "flux2_kontext": ("Flux2KleinPipeline",        False, True),
    "flux2_inpaint": ("Flux2KleinInpaintPipeline", False, True),
}
# Stima VRAM realistica (tutto-su-GPU) per decidere GPU vs offload. FLUX.1 dev/Fill/Kontext bf16
# pesano ~24GB il solo transformer: con TE(T5)+VAE il totale sfora i 32GB → DEVE andare in
# enable_model_cpu_offload, altrimenti spilla nella shared-RAM di Windows (lentissimo, vedi outpaint).
VRAM_MB = {"flux": 24000, "flux2": 24000, "zimage": 12000, "ernie": 16000,
           "kontext": 26000, "inpaint": 26000, "flux2_kontext": 24000, "flux2_inpaint": 24000}
# archi che richiedono una maschera (inpaint-family)
_MASK_ARCHS = ("inpaint", "flux2_inpaint")

# Text encoder convertiti (pesi locali + config/tokenizer) e VAE locale, per i file solo-transformer
TE_DIR = r"E:\AI\Models\diffusers\_te"
AE_FLUX = r"E:\AI\Models\VAE\Flux\ae.safetensors"
# Ernie: repo diffusers (config/vae/tokenizer/scheduler) + transformer e text-encoder come single-file locali
ERNIE_REPO = r"E:\AI\Models\diffusers\Ernie\ERNIE-Image"
ERNIE_TRANSFORMER = r"E:\AI\Models\diffusion_models\Ernie\ernie-image.safetensors"
ERNIE_TE = r"E:\AI\Models\TextEncoders\Ernie\ministral-3-3b.safetensors"
FLUX2_DIR = r"E:\AI\Models\diffusers\flux2"          # TE Mistral HF + tokenizer + config (gated, scaricati)
FLUX2_VAE = r"E:\AI\Models\VAE\Flux\flux2-vae.safetensors"
# FLUX.2 KLEIN (edit/inpaint): encoder Qwen3 (NON Mistral). 4B ufficiale → hidden 2560 (jad 7680),
# miraclein NSFW 8B → hidden 4096 (jad 12288). Il dir giusto è scelto dal joint_attention_dim del
# transformer (vedi _assemble_flux2_klein). VAE = lo stesso flux2-vae.
KLEIN_QWEN_DIRS = {7680: os.path.join(TE_DIR, "qwen3_4b"), 12288: os.path.join(TE_DIR, "qwen3_8b")}


def _place(pipe, vram_mb: int, resident_mb: int | None = None) -> str:
    """Posizionamento VRAM a TRE livelli (carica→usa→libera i componenti; l'inutilizzato esce dalla VRAM):

    1. **gpu**        : tutto residente, se free ≥ vram_mb·margine + floor (nessun offload, il più veloce).
    2. **offload**    : `enable_model_cpu_offload` — un COMPONENTE alla volta su GPU (tokenizer→T5→transformer
                        →VAE, ognuno caricato/usato/scaricato). Tiene però il transformer INTERO su GPU nel denoise.
    3. **sequential** : `enable_sequential_cpu_offload` — granularità per-MODULO (un blocco alla volta su GPU,
                        caricato/usato/liberato). Picco VRAM minimo → entra SEMPRE, niente spill nella shared-RAM
                        (più lento). Si attiva quando nemmeno il transformer intero + attivazioni ci sta.

    `resident_mb` = footprint del componente più grande tenuto residente sotto model_cpu_offload (≈ transformer).
    NB Windows: forzare GPU con VRAM stretta innesca il *system memory fallback* del driver (RAM come VRAM,
    lentissimo). Override: NATIVE_FORCE_OFFLOAD=1|seq, NATIVE_GPU_MARGIN (1.35), NATIVE_GPU_FLOOR_MB,
    NATIVE_ACT_HEADROOM_MB (9000)."""
    import os
    import torch
    _forced = os.environ.get("NATIVE_FORCE_OFFLOAD", "")
    if _forced in ("seq", "sequential"):
        try:
            pipe.enable_sequential_cpu_offload(); return "sequential"
        except Exception:                               # noqa: BLE001
            pipe.enable_model_cpu_offload(); return "offload"
    if _forced == "1" or not torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        return "offload"
    free_mb = torch.cuda.mem_get_info()[0] / 1024**2
    # margine 1.10 (era 1.35, troppo prudente): un modello che ENTRA (o quasi) resta SU GPU — molto più veloce
    # che offload su RAM. Lo sforo delle attivazioni va in shared-VRAM (poco, tollerabile) invece di streammare
    # i pesi dalla RAM ad ogni run. Override NATIVE_GPU_MARGIN. Solo se NON ci sta proprio → model_cpu_offload.
    margin = float(os.environ.get("NATIVE_GPU_MARGIN", "1.12"))
    floor_mb = float(os.environ.get("NATIVE_GPU_FLOOR_MB", "1200"))  # lascia margine al sistema
    _need = vram_mb * margin + floor_mb
    print(f"[place] free={free_mb:.0f}MB est={vram_mb}MB margin={margin} need={_need:.0f}MB "
          f"→ {'gpu' if free_mb >= _need else 'offload'}", flush=True)
    if free_mb >= _need:
        pipe.to("cuda")                                 # tutto residente su GPU; la cache ModelManager lo riusa
        return "gpu"
    # Se il FULL-STACK non entra tutto-su-GPU → model_cpu_offload: il transformer resta su GPU per TUTTO il
    # denoise loop (veloce), mentre T5/VAE vengono onload/offloadati intorno (encode/decode una volta ciascuno).
    # NB: `pipe.to("cuda")` metterebbe ANCHE il T5 (9.5GB) sulla GPU → su un transformer da 22GB su 32GB sfora
    # SEMPRE in shared-RAM (era il "base lentissimo": 31.5GB su 32GB → spill delle attivazioni). model_cpu_offload
    # tiene su GPU solo ~22GB (transformer) + attivazioni → ENTRA e va veloce (~step nativi). L'onload del
    # transformer è UNA volta per generazione (non per-step). NB niente sequential auto (segfault fp8 su Windows).
    _ = resident_mb  # (non più usato per forzare gpu: to('cuda') includerebbe il T5 → spill garantito su 32GB)
    pipe.enable_model_cpu_offload()
    return "offload"


def _ensure_placement(pipe, mm=None, self_key: str | None = None) -> str:
    """Placement per-JOB (coda seriale → un solo pipe attivo alla volta). Con i componenti CONDIVISI
    (ComponentStore) lo stato device/hook è GLOBALE: un'altra pipeline può aver attaccato hook di offload
    al T5 condiviso o averlo demotato su CPU. A inizio job: se l'epoch del proprio ultimo placement è
    ancora quello dello store → no-op (caso comune, stesso pipe riusato). Altrimenti: strip di TUTTI gli
    hook accelerate sui propri moduli (anche se attaccati da un'ALTRA shell — mai due catene di offload
    vive), make_room per il fabbisogno mancante (i componenti PROPRI sono refcountati → protetti), e
    _place con la VRAM di ADESSO. Ping-pong gpu↔offload ≈ 1-2s di PCIe vs 30-90s di reload da SSD."""
    from inference_server.component_store import STORE
    if getattr(pipe, "_ls_no_rebind", False):            # flux2 blockswap: pesi su CPU by design, MAI ri-piazzare
        return getattr(pipe, "_ls_place_mode", "blockswap")
    if getattr(pipe, "_ls_place_epoch", -1) == STORE.placement_epoch:
        return getattr(pipe, "_ls_place_mode", "gpu")
    try:
        pipe.remove_all_hooks()                          # solo _hf_hook accelerate; i _diffusers_hook
    except Exception:                                    # noqa: BLE001   (layerwise fp8) sopravvivono
        pass
    import torch
    keys = getattr(pipe, "_ls_component_keys", ()) or ()
    est = int(getattr(pipe, "_ls_est_mb", VRAM_MB.get("flux", 24000)))
    est_missing = max(est - int(STORE.resident_of(keys)), 2000)   # spazio dei residenti già pagato
    if mm is not None:
        try:                                             # sfratta le ALTRE shell/orfani per far posto
            total_mb = torch.cuda.mem_get_info()[1] / 1024**2 if torch.cuda.is_available() else 0
            mm.make_room(min(est_missing * 1.12, max(total_mb - 2000, 0)),
                         keep={self_key} if self_key else set())
        except Exception:                                # noqa: BLE001
            pass
    mode = _place(pipe, est_missing)
    for k in keys:
        STORE.note_device(k, "cuda" if mode == "gpu" else "cpu")
    pipe._ls_place_mode = mode
    pipe._ls_place_epoch = STORE.bump_epoch()            # invalida l'epoch delle ALTRE pipe → loro rebind
    return mode


def _place_flux2_blockswap(pipe) -> str:
    """Placement memory-bounded per FLUX.2 (~32B): block-swap manuale (no accelerate, che segfaulta coi
    buffer float8). I Linear quantizzati (Fp8Linear/Nvfp4Linear) tengono i pesi su CPU e li spostano su
    GPU 1 layer alla volta nel forward; i moduli leggeri restano su GPU. VAE intero su GPU (piccolo).
    Forza pipe._execution_device='cuda' perché coi pesi fp8 su CPU il device della pipe sarebbe ambiguo."""
    import torch
    from inference_server.backends.flux2_transformer import enable_blockswap
    dev = torch.device("cuda")
    enable_blockswap(pipe.transformer, dev)
    te = getattr(pipe, "text_encoder", None)
    if te is not None:
        try:
            enable_blockswap(te, dev)
        except Exception:                                   # noqa: BLE001
            te.to(dev)
    try:
        pipe.vae.to(dev)
    except Exception:                                       # noqa: BLE001
        pass
    # coi pesi fp8 su CPU, DiffusionPipeline.device (→ _execution_device) può risolvere 'cpu' e rompere la
    # creazione di latenti/embed. Forziamo cuda sulla classe della pipe (l'app esegue FLUX.2 solo su GPU).
    try:
        pipe.__class__._execution_device = property(lambda self: dev)
    except Exception:                                       # noqa: BLE001
        pass
    return "blockswap"


def _set_attention(pipe, arch: str = "") -> str:
    """Backend di attention configurabile con FALLBACK (per la distribuzione su GPU diverse).
    NATIVE_ATTENTION: auto|sage|sage3|sdpa. 'auto' prova Sage e ricade su SDPA (sempre disponibile).
    Z-Image usa attention_mask → SageAttention non lo supporta ('attn_mask is not supported for sage
    attention') → forziamo SDPA per arch 'zimage'."""
    import os
    if arch == "zimage":
        return "sdpa"
    # Gli arch IMAGE-CONDITIONED (inpaint/Fill, kontext, edit, Klein) sono sensibili all'errore INT8 di
    # SageAttention → RUMORE visibile sul blending della regione mascherata. Usa SDPA full-precision
    # (reference-correct). Sage resta per il t2i puro e il video. Override esplicito: NATIVE_ATTENTION=sage.
    if (os.environ.get("NATIVE_ATTENTION") or "").lower() not in ("sage", "sage3"):
        if arch in ("inpaint", "kontext") or str(arch).startswith("flux2_"):
            return "sdpa"
    want = (os.environ.get("NATIVE_ATTENTION") or "auto").lower()
    if want in ("auto", "sage", "sage3"):
        try:
            pipe.transformer.set_attention_backend("sage")
            return "sage"
        except Exception:                               # noqa: BLE001
            if want in ("sage", "sage3"):
                print("[image] Sage non disponibile su questa GPU → SDPA", flush=True)
    return "sdpa"                                       # default torch (sempre presente)


def _apply_fbcache(pipe) -> bool:
    """First Block Cache (come WaveSpeed di ComfyUI): salta i blocchi DiT quando il residuo cambia poco →
    ~2x più veloce ma PERDITA DI QUALITÀ visibile. DEFAULT OFF (qualità prima, parità con ComfyUI): la
    generazione nativa saltava blocchi solo sul t2i → immagini più povere di ComfyUI. Opt-in velocità:
    NATIVE_FBCACHE=1 (soglia NATIVE_FBCACHE_THRESHOLD, default 0.12: più alto = più veloce, meno fedele)."""
    import os
    if os.environ.get("NATIVE_FBCACHE", "0") != "1":
        return False
    try:
        from diffusers import FirstBlockCacheConfig            # cache NATIVO diffusers (compatibile)
        thr = float(os.environ.get("NATIVE_FBCACHE_THRESHOLD", "0.12"))
        pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=thr))
        return True
    except Exception as e:                              # noqa: BLE001
        print(f"[image] FBCache non applicato: {e}", flush=True)
        return False


def _maybe_compile(pipe) -> bool:
    """torch.compile del transformer (+~1.5x). Warm-up lungo, ma il server è persistente.
    Env NATIVE_COMPILE=0|1 (default 0: opt-in, può fallire su alcuni checkpoint fp8)."""
    import os
    import torch
    if os.environ.get("NATIVE_COMPILE", "0") != "1":
        return False
    try:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False)
        return True
    except Exception as e:                              # noqa: BLE001
        print(f"[image] compile non applicato: {e}", flush=True)
        return False


def _flux_in_channels(transformer_path: str) -> int | None:
    """Canali in ingresso del transformer FLUX (x_embedder/img_in) dal solo header safetensors.
    64 = FLUX standard; 384 = FLUX.1 Fill (inpaint/outpaint). None se non-safetensors/gguf o chiave assente."""
    try:
        from core.model_detect import read_header, _shape
        tensors, _ = read_header(transformer_path)
        shp = _shape(tensors, "x_embedder.weight", "img_in.weight")
        if shp and len(shp) >= 2:
            return int(shp[1])
    except Exception:                                        # noqa: BLE001
        pass
    return None


def _load_flux_tr(transformer_path: str, arch: str = "flux"):
    """Carica il SOLO transformer FLUX.1 da single file (config locale per canali reali, GGUF-aware).
    Estratto da _assemble_flux: riusato dal ComponentStore (chiave per-file) e dalla FASE C (fp8)."""
    import os
    import torch
    from diffusers import FluxTransformer2DModel
    bf16 = torch.bfloat16
    _in_ch = _flux_in_channels(transformer_path)
    _is_fill = (_in_ch == 384) or (_in_ch is None and arch == "inpaint")
    _cfg_name = "flux_fill_transformer" if _is_fill else "flux_transformer"
    _tr_cfg = os.path.join(TE_DIR, _cfg_name)
    # GGUF (llama.cpp Qn_K): transformer quantizzato tenuto tale in VRAM, dequant per-layer on-the-fly.
    _gguf_kw = {}
    if str(transformer_path).lower().endswith(".gguf"):
        from diffusers import GGUFQuantizationConfig
        _gguf_kw = {"quantization_config": GGUFQuantizationConfig(compute_dtype=bf16)}
    if os.path.isdir(_tr_cfg):
        tr = FluxTransformer2DModel.from_single_file(
            transformer_path, config=_tr_cfg, torch_dtype=bf16, local_files_only=True, **_gguf_kw)
    else:
        tr = FluxTransformer2DModel.from_single_file(transformer_path, torch_dtype=bf16, **_gguf_kw)
    return tr, _cfg_name


def _assemble_flux(Pipe, transformer_path: str, arch: str = "flux"):
    """Assembla una pipeline FLUX-family da: transformer LOCALE (single_file) + VAE LOCALE
    (ae.safetensors) + text encoder convertiti (CLIP-L, T5-xxl). Per i file 'solo transformer'
    (diffusion_models) che non contengono i text encoder.

    COMPONENT STORE: T5/CLIP/VAE (identici per TUTTI i FLUX.1) e il transformer (per-file) vengono presi
    dallo store con refcount → mai ricaricati finché vivono lì; la pipeline è una "shell" leggera che li
    referenzia (pipe._ls_component_keys). Rollback dei refcount se l'assemblaggio fallisce a metà."""
    import os
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from inference_server.component_store import STORE
    bf16 = torch.bfloat16
    _in_ch = _flux_in_channels(transformer_path)
    if arch == "inpaint" and _in_ch == 64:
        print("[flux] Modello FLUX STANDARD (64 canali) selezionato per l'inpaint: per il Fill nativo di "
              "qualità usa il modello 'FLUX.1 Fill' (384 canali). Procedo col config standard.")
    _cfg_name = "flux_fill_transformer" if (_in_ch == 384 or (_in_ch is None and arch == "inpaint")) \
        else "flux_transformer"
    tr_key = f"flux_tr:{os.path.normpath(transformer_path).lower()}:{_cfg_name}"
    got: list[str] = []                                  # rollback anti refcount-leak
    try:
        tr = STORE.acquire(tr_key, lambda: _load_flux_tr(transformer_path, arch)[0]); got.append(tr_key)
        vae = STORE.acquire("vae_flux", lambda: AutoencoderKL.from_single_file(
            AE_FLUX, config=os.path.join(TE_DIR, "flux_vae"), torch_dtype=bf16)); got.append("vae_flux")
        te = STORE.acquire("clip_l", lambda: CLIPTextModel.from_pretrained(
            os.path.join(TE_DIR, "clip_l"), torch_dtype=bf16)); got.append("clip_l")
        te2 = STORE.acquire("t5xxl", lambda: T5EncoderModel.from_pretrained(
            os.path.join(TE_DIR, "t5xxl"), torch_dtype=bf16)); got.append("t5xxl")
    except Exception:
        for k in got:
            STORE.release(k)
        raise
    # tokenizer/scheduler: leggeri, ricreati ogni volta — MAI nello store
    tok = CLIPTokenizer.from_pretrained(os.path.join(TE_DIR, "clip_l"))
    tok2 = T5TokenizerFast.from_pretrained(os.path.join(TE_DIR, "t5xxl"))
    sched = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
        base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)
    pipe = Pipe(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok,
                text_encoder_2=te2, tokenizer_2=tok2, scheduler=sched)
    pipe._ls_component_keys = (tr_key, "vae_flux", "clip_l", "t5xxl")
    pipe._ls_shared = True
    return pipe


def _normalize_flux_prefix(path: str) -> str:
    """Normalizza FLUX.1 single-file non standard per from_single_file:
    - prefisso `model.diffusion_model.` (export SwarmUI) → strip (altrimenti KeyError single_blocks.N);
    - dtype **fp8 E5M2** (es. fuxCapacity) → upcast a bf16 (il converter diffusers gestisce E4M3 ma non
      E5M2 → errore di mappatura chiavi).
    Scrive UNA VOLTA una copia normalizzata (cache su disco). Ritorna il path normalizzato (o l'originale)."""
    import os
    import json
    import struct
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        keys = [k for k in hdr if k != "__metadata__"]
        has_prefix = bool(keys) and all(k.startswith("model.diffusion_model.") for k in keys)
        has_e5m2 = any(hdr[k].get("dtype") == "F8_E5M2" for k in keys)
        if not has_prefix and not has_e5m2:
            return path                                       # già standard → niente da fare
    except Exception:                                         # noqa: BLE001
        return path
    cache = path + ".normalized.safetensors"
    if os.path.isfile(cache) and os.path.getsize(cache) > 1024:
        return cache
    import torch
    from safetensors.torch import load_file, save_file
    sd = load_file(path)
    pref = "model.diffusion_model."
    out = {}
    for k, v in sd.items():
        nk = k[len(pref):] if k.startswith(pref) else k
        out[nk] = v.to(torch.bfloat16) if v.dtype == torch.float8_e5m2 else v
    tmp = cache + ".tmp"
    save_file(out, tmp)
    os.replace(tmp, cache)
    import logging
    logging.info("[flux] normalizzato (prefix=%s, e5m2=%s) → %s", has_prefix, has_e5m2, os.path.basename(cache))
    return cache


def _assemble_zimage(transformer_path: str):
    """Assembla ZImagePipeline da: transformer LOCALE + VAE FLUX LOCALE + Qwen3 convertito."""
    import os
    import torch
    from diffusers import (ZImageTransformer2DModel, AutoencoderKL,
                           FlowMatchEulerDiscreteScheduler, ZImagePipeline)
    from transformers import AutoModel, AutoTokenizer
    bf16 = torch.bfloat16
    tr = ZImageTransformer2DModel.from_single_file(transformer_path, torch_dtype=bf16)
    vae = AutoencoderKL.from_single_file(AE_FLUX, config=os.path.join(TE_DIR, "flux_vae"), torch_dtype=bf16)
    te = AutoModel.from_pretrained(os.path.join(TE_DIR, "qwen3_4b"), torch_dtype=bf16)
    tok = AutoTokenizer.from_pretrained(os.path.join(TE_DIR, "qwen3_4b"))
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    return ZImagePipeline(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _assemble_flux2_klein(Pipe, transformer_path: str):
    """Assembla una pipeline FLUX.2 KLEIN (edit/inpaint/kontext) — variante DISTILLATA, encoder Qwen3
    (NON Mistral), transformer fp8 puro col prefisso `model.diffusion_model.`. Config transformer
    DEDOTTA dai dim del file → 4B (jad 7680, qwen3_4b) o miraclein 8B (jad 12288, qwen3_8b). VAE = flux2-vae.
    Block-swap memory-bounded come Dev. Niente ComfyUI."""
    import json
    import torch
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from inference_server.backends.flux2_transformer import (klein_config_from_header,
                                                             load_flux2_klein_transformer)
    bf16 = torch.bfloat16
    cfg = klein_config_from_header(transformer_path)
    jad = cfg["joint_attention_dim"]
    qdir = KLEIN_QWEN_DIRS.get(jad)
    # l'encoder può essere single-file (model.safetensors) O SHARDATO (model.safetensors.index.json +
    # model-0000X-of-0000N.safetensors) — Qwen3-8B è shardato. from_pretrained gestisce entrambi.
    _te_present = bool(qdir) and (os.path.exists(os.path.join(qdir, "model.safetensors"))
                                  or os.path.exists(os.path.join(qdir, "model.safetensors.index.json")))
    if not _te_present:
        need = {7680: "Qwen3-4B", 12288: "Qwen3-8B"}.get(jad, f"Qwen3 (jad={jad})")
        raise RuntimeError(
            f"FLUX.2 Klein richiede l'encoder {need} in '{qdir}'. Manca su disco: scaricalo "
            f"col wizard pesi (è l'unico file mancante per Klein nativo).")
    tr = load_flux2_klein_transformer(transformer_path, cfg)
    vae = AutoencoderKLFlux2.from_config(json.load(open(os.path.join(FLUX2_DIR, "vae", "config.json"))))
    vae.load_state_dict(load_file(FLUX2_VAE), assign=True)
    vae = vae.to(bf16)
    # Qwen3 (Qwen3ForCausalLM): la pipeline Klein chiama text_encoder(..., output_hidden_states=True) e
    # concatena i layer [9,18,27]. Caricato in bf16 dai pesi locali (no repo runtime, no ComfyUI).
    te = AutoModelForCausalLM.from_pretrained(qdir, torch_dtype=bf16)
    tok = AutoTokenizer.from_pretrained(qdir)
    try:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(FLUX2_DIR, "scheduler"))
    except Exception:                                   # noqa: BLE001
        sched = FlowMatchEulerDiscreteScheduler()
    return Pipe(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _assemble_flux2(Pipe, transformer_path: str):
    """Assembla una pipeline FLUX.2 (t2i/inpaint/kontext) da: transformer LOCALE (single_file) +
    VAE FLUX.2 LOCALE + TE Mistral HF + tokenizer (scaricati gated, solo questi). Pipe è la
    classe diffusers concreta (Flux2Pipeline, Flux2KleinInpaintPipeline, …)."""
    import os
    import torch
    # Klein è un'architettura DISTILLATA con encoder Qwen3 (≠ Dev/Mistral): instradata a parte.
    if "Klein" in getattr(Pipe, "__name__", ""):
        return _assemble_flux2_klein(Pipe, transformer_path)
    from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoModel, AutoTokenizer
    bf16 = torch.bfloat16
    import json
    from safetensors.torch import load_file
    # Transformer FLUX.2: loader fp8-aware memory-bounded (applica le weight_scale che from_single_file
    # ignora → qualità corretta; streaming+assign-and-free → niente picco RAM ~50GB). Fallback a
    # from_single_file solo se il config transformer non è presente.
    tr_cfg_dir = os.path.join(FLUX2_DIR, "transformer")
    if os.path.exists(os.path.join(tr_cfg_dir, "config.json")):
        from inference_server.backends.flux2_transformer import load_flux2_transformer
        tr = load_flux2_transformer(transformer_path, tr_cfg_dir)
    else:
        tr = Flux2Transformer2DModel.from_single_file(transformer_path, torch_dtype=bf16)
    # AutoencoderKLFlux2 NON supporta from_single_file → from_config + load_state_dict (chiavi 1:1)
    vae = AutoencoderKLFlux2.from_config(json.load(open(os.path.join(FLUX2_DIR, "vae", "config.json"))))
    vae.load_state_dict(load_file(FLUX2_VAE), assign=True)
    vae = vae.to(bf16)
    # Text encoder FLUX.2 = Mistral3 nvfp4 (~12GB, file LOCALE) — dequant→bf16 (no repo 48GB, no ComfyUI)
    te_dir = os.path.join(FLUX2_DIR, "text_encoder")
    if os.path.exists(os.path.join(te_dir, "model.safetensors")):
        te = AutoModel.from_pretrained(te_dir, torch_dtype=bf16)      # se già convertito su disco
    else:
        from inference_server.backends.flux2_te import load_flux2_text_encoder
        te = load_flux2_text_encoder(te_dir)                          # dequant nvfp4 dal file locale
    # FLUX.2 vuole il PROCESSOR (PixtralProcessor), NON un tokenizer puro: Flux2Pipeline.encode_prompt usa
    # tokenizer.apply_chat_template col formato multimodale + system message. Con AutoTokenizer
    # (LlamaTokenizerFast) il chat template/i token speciali sono diversi → embedding sbagliati → RUMORE.
    from transformers import AutoProcessor
    try:
        tok = AutoProcessor.from_pretrained(os.path.join(FLUX2_DIR, "tokenizer"))
    except Exception:                                   # noqa: BLE001
        tok = AutoTokenizer.from_pretrained(os.path.join(FLUX2_DIR, "tokenizer"))
    try:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(FLUX2_DIR, "scheduler"))
    except Exception:                                   # noqa: BLE001
        sched = FlowMatchEulerDiscreteScheduler()
    return Pipe(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _assemble_ernie():
    """Assembla ErnieImagePipeline (Baidu ERNIE-Image) da pesi LOCALI, con shim per transformers <5
    (il repo diffusers vuole transformers 5: TokenizersBackend / Ministral3 / model_type 'ministral3').
    transformer (single-file, chiavi 1:1) + text-encoder Mistral3 (single-file, remap model.→language_model.)
    + vae/tokenizer/scheduler dal repo. NIENTE ComfyUI."""
    import json
    import torch
    import transformers
    from transformers import (Mistral3Model, Mistral3Config, MinistralForCausalLM, PreTrainedTokenizerFast)
    from safetensors.torch import load_file
    from diffusers import (ErnieImageTransformer2DModel, AutoencoderKLFlux2,
                           FlowMatchEulerDiscreteScheduler, ErnieImagePipeline)
    bf16 = torch.bfloat16
    if not hasattr(transformers, "Ministral3ForCausalLM"):
        transformers.Ministral3ForCausalLM = MinistralForCausalLM
    if not hasattr(transformers, "TokenizersBackend"):
        transformers.TokenizersBackend = PreTrainedTokenizerFast
    # transformer: config dal repo + pesi dal single-file (chiavi identiche, 409/409)
    tr = ErnieImageTransformer2DModel.from_config(json.load(open(os.path.join(ERNIE_REPO, "transformer", "config.json"))))
    tr.load_state_dict(load_file(ERNIE_TRANSFORMER), assign=True)
    tr = tr.to(bf16)
    # text encoder Mistral3: 'ministral3' (sconosciuto a transformers 4) → 'ministral' + sliding_window=full
    tcfg = json.load(open(os.path.join(ERNIE_REPO, "text_encoder", "config.json")))
    tc = tcfg.get("text_config", {})
    if tc.get("model_type") == "ministral3":
        tc["model_type"] = "ministral"
    if tc.get("sliding_window") is None:
        tc["sliding_window"] = tc.get("max_position_embeddings", 262144)
    te = Mistral3Model(Mistral3Config(**tcfg))
    raw = load_file(ERNIE_TE)
    rem = {("language_model." + k[6:]) if k.startswith("model.") else k: v
           for k, v in raw.items() if not k.startswith("tekken_model")}
    te.load_state_dict(rem, strict=False)
    te = te.to(bf16)
    vae = AutoencoderKLFlux2.from_pretrained(os.path.join(ERNIE_REPO, "vae"), torch_dtype=bf16)
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ERNIE_REPO, "tokenizer", "tokenizer.json"))
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(ERNIE_REPO, "scheduler"))
    return ErnieImagePipeline(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _auto_settings(arch: str, model: str) -> tuple[int, float, bool]:
    """Step + guidance OTTIMALI AUTOMATICI per (arch, modello), per generazione/inpaint/outpaint/integra.
    Evita il RUMORE da sotto-denoising (FLUX.1 Fill vuole ~40 step, non 20) e il 'piatto' dei distillati
    (hyper/turbo/lightning: pochi step, guidance bassa). La guidance fine dell'inpaint resta pilotata da
    InpaintPipeline.effective_guidance (retention-aware); qui il default per chi non la fornisce."""
    name = str(model).lower()
    hyper = any(k in name for k in ("hyp", "turbo", "lightning", "lcm", "distill", "schnell", "lightn"))
    steps = {"inpaint": 40, "kontext": 28, "flux": 28, "flux2": 28, "flux2_inpaint": 28,
             "flux2_kontext": 28, "zimage": 8, "ernie": 28}.get(arch, 28)
    guid = {"inpaint": 30.0, "kontext": 2.5, "flux": 3.5, "flux2": 4.0, "flux2_inpaint": 4.0,
            "flux2_kontext": 4.0, "zimage": 3.5, "ernie": 4.5}.get(arch, 3.5)
    if hyper:
        steps, guid = 8, min(guid, 3.5)
    return steps, guid, hyper


def _arch_of(params: dict) -> str:
    a = (params.get("task") or params.get("arch") or "flux").lower()
    if a == "flux" and any(t in str(params.get("model", "")).lower()
                           for t in ("flux2", "flux_2", "flux 2", "flux.2")):
        return "flux2"                                   # FLUX_2_Dev/FLUX.2 → loader flux2 (config+TE diversi)
    return a if a in SPEC else "flux"


_SIBLING_ARCHS = ("flux", "kontext", "inpaint")          # arch FLUX.1 che condividono gli stessi componenti


def _sibling_pipe(mm, model_raw: str, arch: str):
    """Se in cache c'è una pipeline SORELLA (stesso file modello, arch diverso della famiglia FLUX.1,
    shell condivisa) → ritornala: la nuova pipeline si costruisce con from_pipe (ZERO load da disco)."""
    if mm is None:
        return None
    for a in _SIBLING_ARCHS:
        if a == arch:
            continue
        e = mm._cache.get(f"{a}:{model_raw}")
        if e is not None and getattr(e.obj, "_ls_shared", False):
            return e.obj
    return None


def _load(model_path: str, arch: str, mm=None):
    """Loader per il ModelManager: ritorna (pipe, vram_mb)."""
    import torch
    import diffusers
    _model_raw = model_path                              # chiave cache originale (per il lookup sorelle)
    # Risolvi il path RELATIVO (es. "Flux\\FLUX_Krea_Dev.safetensors" dall'inventario filesystem) →
    # assoluto sotto diffusion_models. Senza, Path.is_file() è False e si cade in from_pretrained che
    # interpreta il nome come repo_id HF ("Repo id must use alphanumeric..."). Tollera spazi↔underscore.
    if not Path(model_path).is_file():
        dm = Path(ERNIE_TRANSFORMER).parent.parent              # ...\Models\diffusion_models
        # cerca in diffusion_models E in Checkpoints (i checkpoint completi stanno lì, es. flux1-schnell-fp8)
        bases = [dm, dm.parent / "Checkpoints", dm.parent / "checkpoints", dm.parent]
        _name = Path(model_path).name
        cands = []
        for b in bases:
            cands += [b / model_path, (b / model_path).with_name(_name.replace(" ", "_")),
                      b / _name, b / "Flux" / _name]
        for cand in cands:
            if cand.is_file():
                model_path = str(cand)
                break
    # AUTO-DEQUANT: se è un checkpoint quantizzato ComfyUI (comfy_quant), convertilo in bf16 (cache).
    try:
        from core.dequant_comfy import ensure_dequantized
        model_path = ensure_dequantized(model_path)
    except Exception:                                       # noqa: BLE001
        pass
    # GUARD FAMIGLIA: se l'arch (dal detector/GUI) contraddice i CANALI REALI del checkpoint, correggila —
    # altrimenti un FLUX.1 finisce nell'assembler FLUX.2 (KeyError 'double_blocks…') e viceversa.
    # x_embedder/img_in: 64|384 = FLUX.1 (base|Fill) · 128 = FLUX.2.
    if Path(model_path).is_file() and (arch in ("flux", "kontext", "inpaint") or str(arch).startswith("flux2")):
        _ich = _flux_in_channels(model_path)
        if str(arch).startswith("flux2") and _ich in (64, 384):
            _new = {"flux2": "flux", "flux2_inpaint": "inpaint", "flux2_kontext": "kontext"}.get(arch, "flux")
            print(f"[flux] arch '{arch}' richiesto ma il checkpoint è FLUX.1 ({_ich} canali) → uso '{_new}'.")
            arch = _new
        elif arch in ("flux", "kontext", "inpaint") and _ich == 128:
            _new = {"flux": "flux2", "inpaint": "flux2_inpaint", "kontext": "flux2_kontext"}.get(arch, "flux2")
            print(f"[flux] arch '{arch}' richiesto ma il checkpoint è FLUX.2 (128 canali) → uso '{_new}'.")
            arch = _new

    # FLUX.1 (flux/kontext/inpaint) in formato BFL/SwarmUI col prefisso model.diffusion_model. →
    # normalizza (strip prefisso, cache) altrimenti from_single_file fallisce. flux2/Klein gestiscono da sé.
    if arch in ("flux", "kontext", "inpaint") and Path(model_path).is_file():
        try:
            model_path = _normalize_flux_prefix(model_path)
        except Exception:                                  # noqa: BLE001
            pass
    cls_name, supports_sf, _ = SPEC[arch]
    # INPAINT: la pipeline dipende dai CANALI del modello. FluxFillPipeline vuole un transformer a 384 canali
    # (FLUX.1 Fill); un modello FLUX STANDARD (64) darebbe "mat1 4624x384 · mat2 64x3072". Per i modelli a 64
    # canali usa FluxInpaintPipeline (inpaint base mascherato, con strength) → funziona con qualsiasi FLUX.1.
    if arch == "inpaint" and Path(model_path).is_file() and _flux_in_channels(model_path) == 64:
        cls_name = "FluxInpaintPipeline"
        print("[flux] modello FLUX standard (64 canali) per inpaint → FluxInpaintPipeline (base). "
              "Per il Fill dedicato scegli FLUX.1 Fill (384 canali).", flush=True)
    Pipe = getattr(diffusers, cls_name, None)
    if Pipe is None:
        # es. Flux2KontextPipeline non esiste in diffusers 0.38 → messaggio chiaro + suggerimento
        alt = "kontext" if arch == "flux2_kontext" else "flux2_inpaint" if arch.startswith("flux2") else ""
        raise RuntimeError(
            f"La classe diffusers '{cls_name}' per arch '{arch}' non esiste in diffusers "
            f"{diffusers.__version__}." + (f" Usa '{alt}' per questa funzione." if alt else
            " Aggiorna diffusers o scegli un'altra pipeline."))
    is_file = Path(model_path).is_file()
    # RIUSO SORELLA (from_pipe): stesso FILE modello già caricato con un altro arch FLUX.1 (es. t2i→kontext)
    # → costruisci la nuova pipeline dagli STESSI componenti (zero load da disco, ~1s). Solo 64ch: il Fill
    # (384ch) ha un transformer incompatibile e non arriva mai qui con lo stesso file.
    if (arch in _SIBLING_ARCHS and is_file and _flux_in_channels(model_path) == 64):
        _src = _sibling_pipe(mm, _model_raw, arch)
        if _src is not None:
            try:
                from inference_server.component_store import STORE as _ST
                try:
                    _src.remove_all_hooks()               # from_pipe NON gestisce gli hook → sanifica PRIMA
                except Exception:                         # noqa: BLE001
                    pass
                _ST.bump_epoch()                          # stato toccato → tutte le pipe faranno rebind
                pipe = Pipe.from_pipe(_src)
                pipe._ls_component_keys = _src._ls_component_keys
                for _ck in pipe._ls_component_keys:
                    _ST.retain(_ck)                       # la nuova shell referenzia gli stessi componenti
                pipe._ls_shared = True
                pipe._ls_est_mb = getattr(_src, "_ls_est_mb", VRAM_MB.get(arch, 24000))
                _set_attention(pipe, arch)
                print(f"[flux] from_pipe: riuso componenti della sorella per '{arch}' (zero load)", flush=True)
                return pipe, 50
            except Exception as _e:                       # noqa: BLE001 — fallback: load normale
                print(f"[flux] from_pipe fallito ({_e}) → load completo", flush=True)
    if arch == "ernie":
        pipe = _assemble_ernie()                            # assemblaggio locale (shim transformers <5)
    elif arch.startswith("flux2") and is_file:
        # FLUX.2: transformer locale + VAE locale + TE Mistral HF (assemblaggio, anche inpaint/kontext)
        pipe = _assemble_flux2(Pipe, model_path)
    elif is_file and not supports_sf:
        raise RuntimeError(
            f"{cls_name} non supporta from_single_file: per '{arch}' serve un repo diffusers "
            f"(scaricalo col wizard pesi), non il file singolo {Path(model_path).name}.")
    elif is_file and str(model_path).lower().endswith(".gguf"):
        # GGUF = SOLO transformer quantizzato (no TE/VAE nel file) → assembla direttamente coi
        # componenti locali (il transformer è caricato con GGUFQuantizationConfig dentro _assemble_*).
        if arch in ("flux", "kontext", "inpaint"):
            pipe = _assemble_flux(Pipe, model_path, arch=arch)
        elif arch == "zimage":
            pipe = _assemble_zimage(model_path)
        else:
            raise RuntimeError(f"GGUF non supportato per arch '{arch}' (solo flux/kontext/inpaint/zimage).")
    elif is_file:
        try:
            pipe = Pipe.from_single_file(model_path, torch_dtype=torch.bfloat16)
        except Exception:                               # noqa: BLE001
            # file 'solo transformer' (manca TE/VAE nel checkpoint) → assembla coi componenti locali
            if arch in ("flux", "kontext", "inpaint"):
                pipe = _assemble_flux(Pipe, model_path, arch=arch)
            elif arch == "zimage":
                pipe = _assemble_zimage(model_path)
            else:
                raise
    else:
        pipe = Pipe.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    # VAE TILING: il decode a tessere lascia GIUNZIONI RETTANGOLARI visibili ("mosaico di rettangoli"),
    # particolarmente evidenti su inpaint/editing. Le nostre immagini sono ≤1536 (cap canvas) e il decode
    # del latente FLUX entra comodamente nei 32GB SENZA tiling → di default DISATTIVATO (niente seam).
    # Riattivabile per immagini enormi con NATIVE_VAE_TILING=1.
    if os.environ.get("NATIVE_VAE_TILING") == "1":
        try:
            pipe.vae.enable_tiling()
        except Exception:                               # noqa: BLE001
            pass
    else:
        try:
            pipe.vae.disable_tiling()
        except Exception:                               # noqa: BLE001
            pass
    attn = _set_attention(pipe, arch)
    # FBCache rompe gli arch image-conditioned (Kontext/inpaint): la pipeline non entra nel context
    # manager dell'hook → "No context is set". Lo applichiamo solo ai t2i puri.
    fb = _apply_fbcache(pipe) if not SPEC.get(arch, ("", False, False))[2] else False
    comp = _maybe_compile(pipe)
    # Stima footprint REALE dalla dimensione del file pesi (robusto per fp16/fp8/gguf, ogni arch):
    # tutto-su-GPU ≈ pesi transformer + TE/VAE/working. Se sovrastima la VRAM, _place sceglie
    # enable_model_cpu_offload invece di spillare nella shared-RAM (vedi outpaint lento). Vale per
    # generazione/inpaint/outpaint/integra/kontext con QUALSIASI modello, non solo i preset.
    try:
        sz_mb = os.path.getsize(model_path) / 1024**2 if os.path.isfile(model_path) else 0
    except Exception:                                       # noqa: BLE001
        sz_mb = 0
    # Dim EFFETTIVA in memoria (≠ file): fp8 → diffusers UPCASTA a bf16 (2× il file: getphat 11GB→22GB);
    # GGUF → resta QUANTIZZATO (dequant per-layer on-the-fly) ≈ dim file. Senza correzioni _place sbaglia
    # il placement (fp8 sottostimato → 'gpu' che sfora; GGUF gonfiato dal floor → 'offload' inutile).
    _base = os.path.basename(model_path).lower()
    _is_gguf = _base.endswith(".gguf")
    _eff_mb = sz_mb * 2 if (sz_mb and "fp8" in _base and not _is_gguf) else sz_mb
    overhead = 4000 if arch == "zimage" else 11000          # T5/CLIP/Mistral + VAE + working
    if _is_gguf and sz_mb:
        est_mb = int(_eff_mb + overhead)                    # NIENTE floor VRAM_MB: il Q4/Q5 è davvero piccolo
    else:
        est_mb = max(VRAM_MB.get(arch, 16000), int(_eff_mb + overhead)) if sz_mb else VRAM_MB.get(arch, 16000)
    # FLUX.2 (transformer ~32B fp8 = 33GB + TE Mistral): NON entra tutto-su-GPU nei 32GB e nemmeno in
    # enable_model_cpu_offload (offload a livello-modello = transformer intero su GPU). Serve offload
    # SEQUENZIALE (granularità per-modulo: un layer alla volta su GPU). Memory-bounded e imposta
    # correttamente pipe._execution_device=cuda. Override: NATIVE_FLUX2_PLACE=offload|sequential.
    if arch.startswith("flux2"):
        # FLUX.2 (transformer ~32B fp8 = 33GB + TE Mistral nvfp4): NON entra tutto-su-GPU nei 32GB.
        #  - 'blockswap' (DEFAULT): i pesi fp8/nvfp4 restano su CPU, spostati 1 layer alla volta nel forward
        #    (Fp8Linear/Nvfp4Linear), i moduli leggeri (norm/embed/modulation/bias) stabilmente su GPU.
        #    Picco VRAM ≈ moduli leggeri + 1 layer pesante → entra nei 32GB ed è veloce (no spill shared-RAM).
        #  - 'offload'    : enable_model_cpu_offload (su Windows spilla in shared-RAM → praticamente bloccato).
        #  - 'sequential' : enable_sequential_cpu_offload (Segmentation fault coi buffer float8).
        # Override: NATIVE_FLUX2_PLACE=blockswap|offload|sequential.
        mode = (os.environ.get("NATIVE_FLUX2_PLACE") or "blockswap").lower()
        if mode == "blockswap":
            where = _place_flux2_blockswap(pipe)
        elif mode == "sequential":
            try:
                pipe.enable_sequential_cpu_offload(); where = "sequential"
            except Exception as _e:                         # noqa: BLE001
                print(f"[image] flux2 sequential offload fallito ({_e}) → blockswap", flush=True)
                where = _place_flux2_blockswap(pipe)
        else:
            pipe.enable_model_cpu_offload(); where = "offload"
        # blockswap/offload flux2: pesi fp8 su CPU by design → MAI ri-piazzare (rebind lo salta)
        pipe._ls_no_rebind = True
        pipe._ls_place_mode = where
    else:
        # PLACEMENT RINVIATO a inizio job (_ensure_placement in run()): con componenti CONDIVISI lo stato
        # device/hook è globale e può cambiare tra un job e l'altro (altra shell in offload, demote su CPU)
        # → la decisione gpu|offload va presa con la VRAM di ADESSO, non a load-time. _load carica su CPU.
        pipe._ls_est_mb = est_mb
        where = "deferred"
    print(f"[image] {arch}: {where} (est {est_mb}MB, file {int(sz_mb)}MB), attention={attn}, fbcache={fb}, compile={comp}", flush=True)
    # Le shell condivise pesano ~50MB proprie (tokenizer/scheduler): i GB veri sono nel ComponentStore
    # (contati lì, una volta sola per componente). Le monolitiche mantengono la stima piena.
    return pipe, (50 if getattr(pipe, "_ls_shared", False) else VRAM_MB.get(arch, 16000))


def register(mm, model: str, arch: str) -> str:
    key = f"{arch}:{model}"
    mm.register(key, lambda: _load(model, arch, mm))
    return key


# ── Preview live per-step (latent2rgb) ───────────────────────────────────────────────────────────
# Fattori latent→RGB per FLUX (16 canali), come la preview di ComfyUI: proiezione lineare LEGGERA (no
# VAE-decode → nessun costo/rischio OOM mid-generazione). Danno un'anteprima cromatica dell'immagine che
# sta emergendo, aggiornata ad ogni step. Vedi task #88.
_FLUX_L2RGB = [
    [-0.0346, 0.0244, 0.0681], [0.0034, 0.0210, 0.0687], [0.0275, -0.0668, -0.0433],
    [-0.0174, 0.0160, 0.0617], [0.0859, 0.0721, 0.0329], [0.0004, 0.0383, 0.0115],
    [0.0405, 0.0861, 0.0915], [-0.0236, -0.0185, -0.0259], [-0.0245, 0.0250, 0.1180],
    [0.1008, 0.0755, -0.0421], [-0.0515, 0.0201, 0.0011], [0.0428, -0.0012, -0.0036],
    [0.0817, 0.0765, 0.0749], [-0.1264, -0.0522, -0.1103], [-0.0280, -0.0881, -0.0499],
    [-0.1262, -0.0982, -0.0778]]
_FLUX_L2RGB_BIAS = [-0.0329, -0.0718, -0.0851]


_PREVIEW_WARNED = False    # logga UNA sola volta la causa del mancato decode anteprima (no spam per-step)
_PREVIEW_CB_LOGGED = False  # logga UNA volta che il callback preview è attaccato (+ forma latente)


def _latent_to_pil(latent, arch):
    """Anteprima veloce del latente → PIL, senza VAE. FLUX: latent2rgb con i fattori noti; altri arch:
    normalizzazione min-max dei primi 3 canali (anteprima cromatica approssimativa). Best-effort."""
    import torch
    from PIL import Image
    x = latent[0].detach().float().cpu()                     # (C,h,w)
    c = x.shape[0]
    if arch in ("flux", "kontext", "inpaint") and c == 16:
        f = torch.tensor(_FLUX_L2RGB, dtype=torch.float32)
        b = torch.tensor(_FLUX_L2RGB_BIAS, dtype=torch.float32)
        rgb = torch.einsum("chw,cr->hwr", x, f) + b          # (h,w,3)
    else:
        rgb = x[:3].permute(1, 2, 0) if c >= 3 else x[:1].permute(1, 2, 0).repeat(1, 1, 3)
    lo, hi = rgb.amin(), rgb.amax()
    rgb = (rgb - lo) / (hi - lo + 1e-6)                       # min-max → [0,1] (sempre visibile)
    arr = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(arr, "RGB")


def _make_preview_cb(arch, width, height, steps):
    """Callback diffusers callback_on_step_end: pubblica un'anteprima del latente ogni ~1/8 degli step
    (e all'ultimo) sul preview_bus. Best-effort: qualsiasi errore NON deve rompere la generazione."""
    import io
    import torch
    every = max(1, int(steps) // 8)

    def _cb(pp, step, timestep, cbk):
        from inference_server import preview_bus
        global _PREVIEW_CB_LOGGED
        if not _PREVIEW_CB_LOGGED:                            # log UNA volta: callback attaccato + forma latente
            _PREVIEW_CB_LOGGED = True
            try:
                _lat = cbk.get("latents")
                _sh = tuple(_lat.shape) if _lat is not None else None
                print(f"[preview] callback attivo (arch={arch}, latents={_sh}, every={every})", flush=True)
            except Exception:                                # noqa: BLE001
                pass
        # progresso ad OGNI step (leggerissimo) → barra credibile
        try:
            preview_bus.set_progress(step + 1, int(steps))
        except Exception:                                    # noqa: BLE001
            pass
        try:
            # anteprima (decode latent2rgb) ogni ~1/8 degli step e all'ultimo
            if step % every != 0 and step != int(steps) - 1:
                return cbk
            lat = cbk.get("latents")
            if lat is None:
                return cbk
            with torch.no_grad():
                l = lat
                # latente FLUX-family PACKED (B, tokens, 64) → unpack a (B,16,h,w). Vale anche per Fill/Klein.
                if l.dim() == 3 and hasattr(pp, "_unpack_latents"):
                    vsf = getattr(pp, "vae_scale_factor", 8)
                    l = pp._unpack_latents(lat, height, width, vsf)
                img = _latent_to_pil(l, arch)
                img.thumbnail((384, 384))
                buf = io.BytesIO(); img.save(buf, "PNG")
                preview_bus.set_preview(buf.getvalue(), step + 1, int(steps))
        except Exception as _pe:                             # noqa: BLE001
            global _PREVIEW_WARNED
            if not _PREVIEW_WARNED:                           # logga la CAUSA una sola volta (no spam)
                _PREVIEW_WARNED = True
                print(f"[preview] anteprima per-step non disponibile (arch={arch}): "
                      f"{type(_pe).__name__}: {_pe}", flush=True)
        return cbk

    return _cb


def run(params: dict, mm, output_dir: str) -> dict:
    import torch
    from PIL import Image
    from inference_server import preview_bus
    arch = _arch_of(params)
    model = params["model"]
    key = register(mm, model, arch)
    preview_bus.set_message("Caricamento modello…")         # fase visibile nella finestra di anteprima
    # need_mb=None: lo sfratto pre-load distruggerebbe/demoterebbe anche i COMPONENTI che questa pipeline
    # sta per riusare (refcount ancora 0 prima dell'acquire). _load carica su CPU; lo spazio si libera in
    # _ensure_placement (make_room con i componenti PROPRI ormai refcountati → protetti).
    pipe = mm.ensure(key)
    where = _ensure_placement(pipe, mm, key)                # placement per-JOB (gpu|offload con VRAM attuale)
    _set_attention(pipe, arch)                              # per-job: il transformer CONDIVISO può avere il
    print(f"[image] job {arch}: placement={where}", flush=True)  # processor dell'arch precedente (sage↔sdpa)
    mm.apply_loras(pipe, params.get("loras"))
    preview_bus.set_message("Generazione…")

    # Face adapter PuLID (identità del volto da una foto) — FLUX/kontext
    _pulid_h = None
    if params.get("pulid_face") and arch in ("flux", "kontext"):
        try:
            from core import pulid_adapter
            _faces = params["pulid_face"]                     # str singola, lista, o "a|b|c" (multi-foto → media)
            if isinstance(_faces, str) and "|" in _faces:
                _faces = [f for f in _faces.split("|") if f.strip()]
            _pl = pulid_adapter.load(pipe.transformer)
            _emb = pulid_adapter.extract_id(_pl, _faces)
            _pulid_h = pulid_adapter.attach(pipe.transformer, _pl, _emb,
                                            float(params.get("pulid_strength", 0.8)),
                                            start_step=int(params.get("pulid_start_step", 0)))
            _nf = len(_faces) if isinstance(_faces, list) else 1
            print(f"[pulid] attivo ({len(_pulid_h)} hook, {_nf} foto, strength={params.get('pulid_strength', 0.8)}, "
                  f"start_step={params.get('pulid_start_step', 0)})", flush=True)
        except Exception as _e:                          # noqa: BLE001
            print(f"[pulid] disattivato (errore): {_e}", flush=True)
            _pulid_h = None

    g = torch.Generator(device="cuda").manual_seed(int(params.get("seed", 0)))
    # SETTINGS: il valore ESPLICITO (dal GUI) va SEMPRE rispettato — niente override lato pipeline.
    # _auto_settings serve SOLO come default quando il parametro NON è fornito (es. API/mobile senza GUI).
    # L'ottimizzazione dei parametri pre-compila i WIDGET del GUI (vedi optimal_settings), non forza qui.
    _ast, _agd, _hyp = _auto_settings(arch, model)
    _steps = int(params.get("steps") or 0) or _ast
    _guid = float(params["guidance"]) if params.get("guidance") is not None else _agd
    kw = dict(prompt=params.get("prompt", ""),
              num_inference_steps=_steps,
              guidance_scale=_guid,
              generator=g)
    neg = params.get("negative")
    # PuLID true CFG: migliora l'aderenza al prompt / bilancia identità (solo se la pipeline lo supporta).
    _tcfg = float(params.get("pulid_true_cfg", 1.0) or 1.0)
    if _tcfg > 1.0 and params.get("pulid_face"):
        try:
            import inspect
            if "true_cfg_scale" in inspect.signature(pipe.__call__).parameters:
                kw["true_cfg_scale"] = _tcfg
                kw["negative_prompt"] = neg or "bad quality, worst quality, low quality, blurry"
        except Exception:                                    # noqa: BLE001
            pass
    # immagine/maschera per kontext e inpaint
    _, _, needs_image = SPEC[arch]
    if needs_image:
        img_p = params.get("image_path")
        if not img_p or not Path(img_p).exists():
            return {"status": "error", "error": f"'{arch}' richiede 'image_path' valido."}
        kw["image"] = Image.open(img_p).convert("RGB")
        # PROPORZIONI: senza width/height espliciti, FluxFill/Kontext default a 1024² (quadrato) →
        # la foto viene distorta. Deriviamo W/H dalle dimensioni REALI dell'immagine (aspect-correct),
        # multipli di 16 (richiesti da Flux: vae 8 × patch 2), lato lungo cap 1536 per la VRAM.
        _orig_wh = kw["image"].size                          # dimensioni ESATTE originali (per il restore finale)
        _iw, _ih = _orig_wh
        _cap = int(params.get("max_side", 1536))
        _s = min(1.0, _cap / max(_iw, _ih))
        _W = max(256, round(_iw * _s / 16) * 16)
        _H = max(256, round(_ih * _s / 16) * 16)
        kw["width"], kw["height"] = _W, _H
        if (_W, _H) != (_iw, _ih):                           # allinea l'immagine alla griglia /16 + cap
            kw["image"] = kw["image"].resize((_W, _H), Image.LANCZOS)
        if arch in _MASK_ARCHS:
            mk = params.get("mask_path")
            if not mk or not Path(mk).exists():
                return {"status": "error", "error": f"'{arch}' richiede 'mask_path' valido."}
            # OUTPAINT = inpaint della regione COMPLEMENTARE → maschera invertita (riusa mask_utils)
            if params.get("outpaint"):
                try:
                    from core import mask_utils
                    inv = str(Path(output_dir) / "_outpaint_mask.png")
                    os.makedirs(output_dir, exist_ok=True)
                    mk = mask_utils.invert_mask(mk, inv)
                except Exception as _e:                      # noqa: BLE001
                    print(f"[inpaint] invert_mask fallito ({_e}); uso maschera originale", flush=True)
            _mimg = Image.open(mk).convert("RGB")
            if _mimg.size != (_W, _H):                       # maschera allineata a width/height
                _mimg = _mimg.resize((_W, _H), Image.NEAREST)
            kw["mask_image"] = _mimg
    else:
        kw["width"] = int(params.get("width", 1024))
        kw["height"] = int(params.get("height", 1024))
    # negative_prompt: passalo SOLO se la pipeline lo espone davvero (introspezione firma) → bulletproof
    # contro "unexpected keyword argument 'negative_prompt'" su FluxFill/Klein/zimage (guidance-distilled).
    if neg:
        try:
            import inspect
            if "negative_prompt" in inspect.signature(pipe.__call__).parameters:
                kw["negative_prompt"] = neg
        except Exception:                                   # noqa: BLE001
            pass
    # FluxInpaintPipeline (inpaint BASE, modelli 64ch, SEGUE il prompt) usa 'strength' = quanto rigenerare la
    # regione mascherata (FluxFillPipeline invece fa denoise pieno, niente strength). Default alto = sostituzione
    # decisa. NB: il FLUX base ha guidance EMBEDDED BASSA (~3.5) come nel t2i — a 30 si BRUCIA (≠ Fill dev che
    # vuole 30). Clampo la guidance al range base.
    if type(pipe).__name__ == "FluxInpaintPipeline":
        kw["strength"] = float(params.get("denoise") or 0.9)
        kw["guidance_scale"] = min(float(kw.get("guidance_scale", 3.5)), 5.0)

    # Preview live per-step (task #88): anteprima del latente ad ogni ~1/8 degli step sul preview_bus,
    # che la GUI polla via GET /preview. Best-effort: se la pipeline non supporta callback_on_step_end
    # (es. alcune pipeline custom), si riprova senza.
    try:
        kw["callback_on_step_end"] = _make_preview_cb(arch, int(kw.get("width", 1024)),
                                                      int(kw.get("height", 1024)), kw["num_inference_steps"])
        kw["callback_on_step_end_tensor_inputs"] = ["latents"]
    except Exception:                                       # noqa: BLE001
        pass
    try:
        out = pipe(**kw)
    except TypeError as _te:                                # pipeline senza callback_on_step_end
        if "callback" in str(_te):
            kw.pop("callback_on_step_end", None); kw.pop("callback_on_step_end_tensor_inputs", None)
            out = pipe(**kw)
        else:
            raise
    finally:
        if _pulid_h:
            from core import pulid_adapter
            pulid_adapter.detach(_pulid_h)
    preview_bus.set_message("Decodifica e salvataggio…")
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_{arch}_{params.get('seed', 0)}.png")
    res_img = out.images[0]
    # genero a dimensioni /16 (richiesto da Flux), ma ripristino le dimensioni ESATTE dell'input → output
    # combacia 1:1 con la foto/bbox (niente disallineamento in load_inpaint_result, aspetto identico).
    if needs_image and res_img.size != _orig_wh:
        res_img = res_img.resize(_orig_wh, Image.LANCZOS)
    # INPAINT/OUTPAINT: FluxFill/inpaint rigenera TUTTA l'immagine (VAE encode→decode dell'intera foto) → le
    # zone FUORI dalla maschera prendono artefatti del VAE ("quadrettatura"/macchie sulle parti intatte).
    # Ricompone i PIXEL ORIGINALI dove la maschera è nera (feather sui bordi) → solo la regione mascherata è
    # rigenerata, il resto resta identico all'input. Env NATIVE_INPAINT_KEEP=0 per disattivare (rigenera tutto).
    _mask_pil = kw.get("mask_image")
    if arch in _MASK_ARCHS and _mask_pil is not None and params.get("image_path") \
            and os.environ.get("NATIVE_INPAINT_KEEP", "1") == "1":
        try:
            from PIL import ImageFilter
            _orig = Image.open(params["image_path"]).convert("RGB").resize(res_img.size, Image.LANCZOS)
            _feather = max(2, min(8, int(round(min(res_img.size) / 256))))   # bordo morbido proporzionale
            _m = _mask_pil.convert("L").resize(res_img.size, Image.NEAREST)
            # OUTPAINT: dilata la regione generata prima di sfumare, così la sfumatura NON tocca il bordo NERO
            # della tela (niente linea scura). INPAINT: NIENTE dilatazione → l'originale è preservato fino al
            # bordo della maschera (anello di modifica sottile, meno "degradazione" attorno all'edit).
            if params.get("outpaint"):
                _m = _m.filter(ImageFilter.MaxFilter(_feather * 2 + 1))
            _m = _m.filter(ImageFilter.GaussianBlur(_feather))
            res_img = Image.composite(res_img, _orig, _m)            # bianco=generato · nero=pixel originali
        except Exception as _e:                                      # noqa: BLE001
            print(f"[inpaint] ricomposizione originale fuori-maschera fallita ({_e})", flush=True)
    # DITHER anti-BANDING: i gradienti morbidi (pelle, cielo) a 8-bit fanno "scalini" di posterizzazione,
    # peggio coi modelli quantizzati (Q5/fp8). Un rumore ±1.7 livelli IMPERCETTIBILE li rompe. Env NATIVE_DITHER=0.
    if os.environ.get("NATIVE_DITHER", "1") == "1":
        try:
            import numpy as _np
            _a = _np.asarray(res_img).astype(_np.float32) + _np.random.uniform(-1.7, 1.7, (res_img.height, res_img.width, 1))
            res_img = Image.fromarray(_np.clip(_a, 0, 255).astype("uint8"))
        except Exception:                                            # noqa: BLE001
            pass
    res_img.save(path)
    try:                                                     # #107 firma invisibile di provenienza (default ON, tutte le edizioni)
        from core import stego
        if stego.enabled():
            stego.sign_file(path)
    except Exception:                                        # noqa: BLE001
        pass
    try:                                                     # #106 watermark obbligatorio (gated feature_matrix)
        from core import watermark
        path = watermark.apply_to_file(path)                 # forced → fail-closed (rimuove se non applicabile)
    except RuntimeError:
        return {"status": "error", "error": "watermark obbligatorio non applicabile"}
    except Exception:                                        # noqa: BLE001
        pass
    return {"status": "success", "output": [path]}
