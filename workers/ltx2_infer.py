"""Worker inference LTX-2.3 — pacchetto UFFICIALE Lightricks (github.com/Lightricks/LTX-2).

Eseguito nel venv dedicato `E:\\AI\\Tools\\ltx2_venv` (torch ~2.7) via subprocess dal backend.
ZERO ComfyUI, ZERO diffusers: usa ltx-core + ltx-pipelines col checkpoint 2.3 LOCALE.
Legge i parametri da un file JSON (argv[1]), genera video+audio, salva mp4.

Pesi (tutti già locali):
- checkpoint  ltx-2.3-22b-dev-fp8.safetensors
- upsampler   ltx-2.3-spatial-upscaler-x2-1.1.safetensors
- gemma       cartella HF gemma3
"""
import json
import os
import sys

import torch

# Il block-streaming (offload CPU) di LTX salva tensori per autograd → incompatibile con
# torch.inference_mode() usato di default dalle pipeline. Lo rimpiazziamo con no_grad.
torch.inference_mode = torch.no_grad
# Inferenza pura: disabilito l'autograd GLOBALMENTE. Senza, il conditioning (Gemma/encoder) costruisce
# un grafo trattenuto (~37GB di tensori live non-modulo) → con la via GGUF il transformer non entra.
torch.set_grad_enabled(False)

# Compat torch 2.9.1+cu130: torch._scaled_mm (path quant fp8_scaled_mm di ltx_core) ora ESIGE le
# scale in float32 ("scales should be float and singletons"); ltx_core le passa in bf16 →
# "RuntimeError: Invalid scaling configuration". Patchiamo qui (nostro worker, NON il pacchetto):
# castiamo a float32 le scale prima della chiamata.
_orig_scaled_mm = torch._scaled_mm
def _scaled_mm_f32(*args, **kw):                              # noqa: ANN001,ANN201
    if kw.get("scale_a") is not None and kw["scale_a"].dtype != torch.float32:
        kw["scale_a"] = kw["scale_a"].float()
    if kw.get("scale_b") is not None and kw["scale_b"].dtype != torch.float32:
        kw["scale_b"] = kw["scale_b"].float()
    return _orig_scaled_mm(*args, **kw)
torch._scaled_mm = _scaled_mm_f32

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage   # audio→video (lip-sync canto), nativo
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode

# Fix meta-tensor: il meta-device loading di ltx-core + l'ottimizzazione SDPA-mask di transformers
# fa `.item()` su un tensore 'meta' (crash su qualsiasi torch recente). Disabilitando lo skip,
# la causal mask viene costruita normalmente (un filo più lavoro, nessun crash).
try:
    import transformers.masking_utils as _mu
    _mu._ignore_causal_mask_sdpa = lambda *a, **k: False
except Exception:  # noqa: BLE001
    pass


def _materialize_meta(model, device):
    """ltx-core costruisce su 'meta' e carica i pesi con assign=True, ma i BUFFER calcolati
    (rotary inv_freq, embed_scale) non sono nel checkpoint Gemma di ComfyUI → restano su meta.
    Qui li ricreiamo sul device reale (inv_freq via rope_init_fn)."""
    import torch
    cfg = getattr(model, "config", None)
    txt = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    hidden = getattr(txt, "hidden_size", None)
    for mod in model.modules():
        for bn, buf in list(getattr(mod, "_buffers", {}).items()):
            if buf is None or not getattr(buf, "is_meta", False):
                continue
            if bn in ("inv_freq", "original_inv_freq") and hasattr(mod, "rope_init_fn") and hasattr(mod, "config"):
                inv, scaling = mod.rope_init_fn(mod.config, device)
                mod.register_buffer("inv_freq", inv.to(device=device), persistent=False)
                if hasattr(mod, "attention_scaling"):
                    mod.attention_scaling = scaling
                if hasattr(mod, "original_inv_freq"):
                    mod.original_inv_freq = mod.inv_freq
            elif bn == "embed_scale" and hidden:
                mod.register_buffer("embed_scale", torch.tensor(float(hidden) ** 0.5, device=device,
                                                                dtype=torch.float32), persistent=False)
            else:
                mod.register_buffer(bn, torch.zeros(buf.shape, dtype=buf.dtype, device=device), persistent=False)


def _apply_blockwise_offload(model, device):
    """Offload per-blocco STILE ComfyUI (replica DynamicVram): il transformer LTX resta su CPU,
    ogni blocco va su GPU solo durante il suo forward e torna su CPU subito dopo → ~poche GB VRAM
    invece di 27GB. Le parti non-block (patchify/embeddings/timestep/norm/proj) restano su GPU.
    Cerca i transformer_blocks anche dentro wrapper (X0Model)."""
    import torch
    best = None
    best_n = 0
    holder = None
    stacks = []                                          # TUTTI gli stack (video + audio + Gemma)
    for mod in model.modules():
        for attr in ("transformer_blocks", "layers"):
            tb = getattr(mod, attr, None)
            if isinstance(tb, torch.nn.ModuleList) and len(tb) >= 8:
                stacks.append((mod, tb))
    if not stacks:
        model.to(device)                                # modello piccolo (VAE/...): tutto su GPU
        return 0
    # Modello grande (transformer 27GB / Gemma 22GB): NON spostare tutto su GPU (OOM). Solo le
    # parti NON-layer (embed/patchify/timestep/norm/proj) su GPU; i layer di TUTTI gli stack
    # restano su CPU e vanno su GPU 1 alla volta via wrapping di .forward.
    holders = [m for m, _ in stacks]
    blocks = [blk for _, tb in stacks for blk in tb]
    _layer_ids = {id(t) for blk in blocks for t in list(blk.parameters()) + list(blk.buffers())}
    for p in model.parameters():
        if id(p) not in _layer_ids:
            p.data = p.data.to(device)
    for b in model.buffers():
        if id(b) not in _layer_ids:
            b.data = b.data.to(device)
    # Wrappiamo direttamente .forward di ogni layer (NON forward_hook: transformers chiama i
    # decoder layer via .forward() diretto / GradientCheckpointingLayer, bypassando gli hook).
    # Il layer va su GPU per il suo forward e torna su CPU subito dopo → 1 layer alla volta su GPU.
    for blk in blocks:
        blk.to("cpu")

        def _make_fwd(b, orig):
            def _fwd(*a, **k):
                b.to(device)
                try:
                    return orig(*a, **k)
                finally:
                    b.to("cpu")
                    torch.cuda.empty_cache()           # libera subito i pesi del layer (no accumulo)
            return _fwd

        blk.forward = _make_fwd(blk, blk.forward)

    # Il pacchetto fa .to(device) rimettendo TUTTI i layer su GPU prima del forward: qui li
    # rispingiamo su CPU all'inizio del forward del modulo proprietario, poi il wrapping di
    # .forward li carica 1 alla volta on-demand. (forward_hook NON basta: i layer sono chiamati
    # via .forward() diretto.)
    def _force_layers_cpu(mod, args):
        for blk in blocks:
            blk.to("cpu")
        torch.cuda.empty_cache()

    for _h in holders:
        _h.register_forward_pre_hook(_force_layers_cpu)
    return len(blocks)


def _patch_block_offload():
    """Monkey-patch DiffusionStage._build_transformer: dopo la build (X0Model su GPU), applica
    l'offload per-blocco → solo le parti non-block + 1 blocco alla volta restano su GPU."""
    import os as _os
    if _os.environ.get("LTX_BLOCK_OFFLOAD") != "1":
        return
    try:
        import torch
        import ltx_pipelines.utils.blocks as _blk
        _orig_bt = _blk.DiffusionStage._build_transformer

        def _patched_bt(self, **kwargs):
            # _build_transformer fa `X0Model(build).to(target)`: il `.to(target)` rimette i 27GB su
            # GPU. Lo neutralizziamo (no-op) SOLO qui; l'offload (parti non-layer su GPU via
            # Tensor.to, layer su CPU) è già applicato dentro build() via _patched_build. Tensor.to
            # NON è intaccato (patchiamo solo Module.to), quindi le parti non-layer restano su GPU.
            _real_to = torch.nn.Module.to
            torch.nn.Module.to = lambda s, *a, **k: s
            try:
                x0 = _orig_bt(self, **kwargs)
            finally:
                torch.nn.Module.to = _real_to
            print(f"[bt] transformer pronto (offload attivo), VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
            return x0

        _blk.DiffusionStage._build_transformer = _patched_bt
        print("[block-offload] patch _build_transformer attivo", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[block-offload] patch fallito: {_e}", flush=True)


_patch_block_offload()


def _patch_fe_diag():
    """DIAGNOSTICO: logga la shape degli hidden-states concatenati nel feature_extractor."""
    if os.environ.get("LTX_FE_DIAG") != "1":
        return
    try:
        import ltx_core.text_encoders.gemma.feature_extractor as _fe
        _orig = _fe.norm_and_concat_per_token_rms

        def _logged(encoded, mask, *a, **k):
            print(f"[fe-diag] encoded.shape={tuple(encoded.shape)} dtype={encoded.dtype} "
                  f"GB={encoded.numel()*encoded.element_size()/1e9:.1f}", flush=True)
            return _orig(encoded, mask, *a, **k)

        _fe.norm_and_concat_per_token_rms = _logged
    except Exception as _e:  # noqa: BLE001
        print(f"[fe-diag] patch fallito: {_e}", flush=True)


_patch_fe_diag()


def _patch_cpu_prompt_encoder():
    """FIX VRAM: esegue l'encode del Gemma (12B, 22GB) su CPU (0 VRAM pesi; gli hidden-states sono
    solo ~0.4GB). Poi sposta gli hidden-states su GPU per l'embeddings_processor. Così durante il
    denoising la GPU ha solo il transformer. Encode su CPU ~30-60s ma sblocca i 32GB."""
    if os.environ.get("LTX_CPU_ENCODE") != "1":
        return
    try:
        import gc
        import torch
        import ltx_pipelines.utils.blocks as _blk

        def _cpu_call(self, prompts, *, enhance_first_prompt=False,
                      enhance_prompt_image=None, enhance_prompt_seed=42):
            te = self._text_encoder_builder.build(device=torch.device("cpu"), dtype=self._dtype).eval()
            raw = te.encode(list(prompts))
            del te
            gc.collect()
            torch.cuda.empty_cache()

            def _to_gpu(x):
                if torch.is_tensor(x):
                    return x.to("cuda")
                if isinstance(x, (list, tuple)):
                    return type(x)(_to_gpu(e) for e in x)
                return x

            raw_gpu = [(_to_gpu(hs), _to_gpu(mask)) for hs, mask in raw]
            with _blk.gpu_model(self._build_embeddings_processor()) as ep:
                result = [ep.process_hidden_states(hs, mask) for hs, mask in raw_gpu]
            return result

        _blk.PromptEncoder.__call__ = _cpu_call
        print("[cpu-encode] PromptEncoder patchato: Gemma encode su CPU", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[cpu-encode] patch fallito: {_e}", flush=True)


_patch_cpu_prompt_encoder()


try:
    import ltx_core.loader.single_gpu_model_builder as _sgmb
    _orig_build = _sgmb.SingleGPUModelBuilder.build

    def _patched_build(self, device=None, dtype=None, **kw):
        import torch
        dev = torch.device("cuda") if device is None else device
        _offl = os.environ.get("LTX_BLOCK_OFFLOAD") == "1"
        load_dev = torch.device("cpu") if _offl else dev
        m = _orig_build(self, device=load_dev, dtype=dtype, **kw)
        metas = [n for n, b in m.named_buffers() if getattr(b, "is_meta", False)]
        metap = [n for n, p in m.named_parameters() if getattr(p, "is_meta", False)]
        print(f"[meta-fix] build {type(m).__name__}: {len(metas)} buffer meta, {len(metap)} param meta", flush=True)
        print("[meta-fix] ATTESI(primi):", metap[:6], flush=True)
        print("[meta-fix] ATTESI(text):", [n for n in metap if 'language_model' in n][:5], flush=True)
        print("[meta-fix] ATTESI(altro):", [n for n in metap if 'vision' not in n and 'language_model' not in n][:6], flush=True)
        try:
            _materialize_meta(m, load_dev)
            left = [n for n, b in m.named_buffers() if getattr(b, "is_meta", False)]
            print(f"[meta-fix] dopo fix: {len(left)} buffer meta rimasti {left[:3]}", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[meta-fix] warn: {_e}", flush=True)
        if os.environ.get("LTX_BLOCK_OFFLOAD") == "1":
            try:
                n = _apply_blockwise_offload(m, dev)
                if n:
                    print(f"[block-offload] {type(m).__name__}: {n} layer su CPU on-demand", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[block-offload] warn: {_e}", flush=True)
        return m

    _sgmb.SingleGPUModelBuilder.build = _patched_build
except Exception:  # noqa: BLE001
    pass

CHECKPOINT = r"E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors"
UPSAMPLER = r"E:\AI\Models\latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA = r"E:\AI\Models\TextEncoders\gemma-3-ltx"              # Gemma-3 12B UFFICIALE Lightricks (non-gated, formato ltx-core)
LORA_DIR = r"E:\AI\Models\LoRA"
# IC-LoRA di controllo (canny/depth/pose) presenti su disco — #24 ControlNet nativo, nessun download
IC_LORA = {
    "canny": r"E:\AI\Models\LoRA\LTX-2\ltx-2-19b-ic-lora-canny-control.safetensors",
    "depth": r"E:\AI\Models\LoRA\LTX-2\ltx-2-19b-ic-lora-depth-control.safetensors",
    "pose": r"E:\AI\Models\LoRA\LTX-2\ltx-2-19b-ic-lora-pose-control.safetensors",
}


_PIPE_CACHE: dict = {}


def _resolve_ckpt(model: str) -> str:
    ckpt = model or CHECKPOINT
    if not os.path.isabs(ckpt):
        for base in (r"E:\AI\Models\diffusion_models\LTX", r"E:\AI\Models\Checkpoints\LTX"):
            cand = os.path.join(base, ckpt)
            if os.path.exists(cand):
                return cand
    return ckpt


def _get_pipe(ckpt: str, qmode: str, offload_str: str, loras: list, stages: str = "two", ptype: str = "ti2v"):
    """Istanzia (e CACHEA) la pipeline. Nel server persistente la cache evita il ri-load da disco.
    ptype: 'ti2v' (testo/immagine→video) oppure 'a2v' (audio→video, lip-sync/canto)."""
    lkey = tuple((lo.path, lo.strength) for lo in loras)
    key = (ckpt, qmode, offload_str, lkey, stages, ptype)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]
    quant = None
    if qmode == "fp8_scaled_mm":
        from ltx_core.quantization import fp8_scaled_mm
        quant = fp8_scaled_mm.build_policy(ckpt)
    elif qmode == "fp8_cast":
        from ltx_core.quantization import fp8_cast
        quant = fp8_cast.build_policy(ckpt)
    offload = {"cpu": OffloadMode.CPU, "disk": OffloadMode.DISK,
               "none": OffloadMode.NONE}.get(offload_str, OffloadMode.NONE)
    # ComfyUI-like: fp8 NATIVO (scaled_mm, niente upcast a bf16) + offload per-blocco insieme.
    # Il pacchetto blocca questa combo con un assert (streaming solo con fp8_cast) che però
    # confronta solo il fuse_rule (serve solo per fondere LoRA, qui assenti). Allineandolo
    # rendiamo l'assert un no-op → transformer streamato a blocchi + matmul fp8 = ~poche GB VRAM.
    if qmode == "fp8_scaled_mm" and offload != OffloadMode.NONE and quant is not None:
        import ltx_pipelines.utils.blocks as _blocks
        _blocks.fp8_cast_fuse_rule = quant.fuse_rule
    if ptype == "iclora":                                    # #24 ControlNet IC-LoRA (canny/depth/pose)
        from ltx_pipelines.ic_lora import ICLoraPipeline
        pipe = ICLoraPipeline(
            distilled_checkpoint_path=ckpt, spatial_upsampler_path=UPSAMPLER,
            gemma_root=GEMMA, loras=loras, quantization=quant, offload_mode=offload)
    else:
        Klass = A2VidPipelineTwoStage if ptype == "a2v" else TI2VidTwoStagesPipeline
        pipe = Klass(
            checkpoint_path=ckpt, distilled_lora=[], spatial_upsampler_path=UPSAMPLER,
            gemma_root=GEMMA, loras=loras, quantization=quant, offload_mode=offload)
    _PIPE_CACHE.clear()                                       # 1 pipeline alla volta (VRAM/RAM)
    _PIPE_CACHE[key] = pipe
    return pipe


def _auto_gguf() -> str | None:
    """Trova il miglior GGUF transformer LTX disponibile (Q4 distilled di default). Permette al 22B di
    entrare nei 32GB col dequant on-the-fly (come ComfyUI). Restituisce None se nessun .gguf."""
    import glob
    bases = [r"E:\AI\Models\diffusion_models", r"E:\AI\Models\Checkpoints"]
    pats = ["**/*distilled*Q4_K_M*.gguf", "**/*Q4_K_M*.gguf", "**/ltx*Q6_K*.gguf", "**/ltx*Q4*.gguf"]
    for b in bases:
        for pat in pats:
            hits = glob.glob(os.path.join(b, pat), recursive=True)
            if hits:
                return hits[0]
    return None


def _run_chunked(p: dict, chunk: int) -> str:
    """Genera un video lungo a CHUNK in-process (pipe in cache → carica una volta sola), con
    continuità: l'ultimo frame di un chunk diventa l'immagine di conditioning del successivo;
    l'audio è affettato per chunk (audio_start_time/max_duration). Concat finale via ffmpeg.
    Evita l'OOM/sborzo dello stage_2 a lunghezza piena tenendo ogni chunk corto."""
    import subprocess
    fps = float(p.get("fps", 25.0))
    total = int(p["num_frames"])
    seg_dur = chunk / fps
    prev_img = p.get("image_path")
    base = os.path.splitext(p["output"])[0]
    parts: list[str] = []
    n = (total + chunk - 1) // chunk
    for i in range(n):
        cf = min(chunk, total - i * chunk)
        if cf < 9:
            break
        cp = dict(p)
        cp.pop("chunk_frames", None)
        cp["num_frames"] = cf
        cp["image_path"] = prev_img
        if p.get("audio_path"):
            cp["audio_start_time"] = round(i * seg_dur, 3)
            cp["audio_max_duration"] = round(seg_dur, 3)
        cp["output"] = f"{base}_chunk{i}.mp4"
        print(f"[chunk] {i + 1}/{n} ({cf} frame, audio_start={cp.get('audio_start_time', 0)})", flush=True)
        try:
            run_generation(cp)                               # pipe in cache dal 2° chunk → veloce
        except Exception as _e:                              # es. ultimo chunk con audio più corto dei frame
            print(f"[chunk] {i} saltato ({type(_e).__name__}: {_e})", flush=True)
            break
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        last = f"{base}_chunk{i}_last.png"
        subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.12", "-i", cp["output"],
                        "-update", "1", "-q:v", "2", last, "-y"], check=False)
        if os.path.exists(last):
            prev_img = last                                  # continuità visiva
        parts.append(cp["output"])
    lst = f"{base}_list.txt"
    with open(lst, "w", encoding="utf-8") as fh:
        for pt in parts:
            fh.write(f"file '{pt}'\n")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", p["output"], "-y"], check=False)
    print(f"[chunk] concat → {p['output']} ({len(parts)} chunk)", flush=True)
    return p["output"]


def run_generation(p: dict) -> str:
    """Genera un video dai params. Riusabile da worker one-shot e da server persistente."""
    import time
    # Video lungo → chunking in-process (memory-bounded, niente sborzo stage_2). Attiva con
    # chunk_frames>0 (o auto sotto). Ricorsione sicura: i chunk hanno num_frames<=chunk.
    _chunk = int(p.get("chunk_frames", 0))
    if _chunk and int(p.get("num_frames", 0)) > _chunk:
        return _run_chunked(p, _chunk)
    # Auto-GGUF: se non specificato, usa il GGUF Q4 (dequant on-the-fly) così il 22B entra nei 32GB.
    # Disattivabile passando esplicitamente gguf_transformer="".
    if "gguf_transformer" not in p:
        g = _auto_gguf()
        if g:
            p["gguf_transformer"] = g
    # Con l'offload attivo (Gemma CPU + transformer per-blocco) il picco è ~31GB su 32: lasciamo
    # che PyTorch usi tutta la VRAM + sysmem-fallback per il piccolo surplus (lento ma genera).
    # (Il memory_fraction era solo per la diagnosi VRAM, ora rimosso.)
    if torch.cuda.is_available() and os.environ.get("LTX_VRAM_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["LTX_VRAM_FRACTION"]))
    loras = []
    for lo in p.get("loras", []):
        path = lo["name"]
        if not os.path.isabs(path):
            path = os.path.join(LORA_DIR, path)
        loras.append(LoraPathStrengthAndSDOps(path, float(lo.get("strength", 1.0)),
                                              LTXV_LORA_COMFY_RENAMING_MAP))
    ckpt = _resolve_ckpt(p.get("model"))
    qmode = p.get("quant") or ("fp8_scaled_mm" if "fp8" in os.path.basename(ckpt).lower() else "fp8_cast")
    t0 = time.time()
    # #24 ControlNet IC-LoRA: se c'è un control_video e il peso IC-LoRA del tipo è su disco → pipeline iclora
    _ic = IC_LORA.get(p.get("control_type", "")) if p.get("control_video") else None
    if _ic and os.path.exists(_ic):
        ptype = "iclora"
        loras = list(loras) + [LoraPathStrengthAndSDOps(_ic, float(p.get("control_strength", 0.8)),
                                                        LTXV_LORA_COMFY_RENAMING_MAP)]
    else:
        ptype = "a2v" if p.get("audio_path") else "ti2v"      # audio dato → lip-sync/canto NATIVO
    pipe = _get_pipe(ckpt, qmode, p.get("offload", "cpu"), loras, ptype=ptype)
    print(f"[load] {time.time() - t0:.0f}s (model={os.path.basename(ckpt)} quant={qmode} pipe={ptype})", flush=True)
    if torch.cuda.is_available():
        print(f"[vram] post-load: {torch.cuda.memory_allocated()/1e9:.1f}GB alloc / "
              f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved", flush=True)

    # TRANSFORMER da GGUF (~14GB) per entrare nei 32GB come ComfyUI: gli altri componenti (Gemma/vae/
    # vocoder) restano sul checkpoint fp8 con offload; il transformer si carica quantizzato dal GGUF e
    # dequantizza per-layer al volo. config dal fp8 (stessa arch).
    if p.get("gguf_transformer"):
        try:
            from workers import gguf_native as _GG
        except ModuleNotFoundError:
            import gguf_native as _GG          # eseguito come script: workers/ è già nel sys.path
        for _st in (getattr(pipe, "stage_1", None), getattr(pipe, "stage_2", None)):
            if _st is not None:
                _GG.patch_stage_to_gguf(_st, p["gguf_transformer"], ckpt)
        print(f"[gguf] transformer da {os.path.basename(p['gguf_transformer'])} (dequant on-the-fly)", flush=True)

    vg = MultiModalGuiderParams(cfg_scale=float(p.get("guidance", 3.0)),
                                stg_scale=float(p.get("stg", 1.0)),
                                rescale_scale=0.7, modality_scale=3.0, stg_blocks=[29])
    ag = MultiModalGuiderParams(cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                                modality_scale=3.0, stg_blocks=[29])
    nf = int(p.get("num_frames", 121))
    fps = float(p.get("fps", 25.0))
    tg = time.time()
    if ptype == "a2v":                                       # AUDIO→VIDEO nativo: foto + canzone → canto
        imgs = [ImageConditioningInput(p["image_path"], 0, float(p.get("ref_strength", 1.0)), 33)] if p.get("image_path") else []
        video, audio = pipe(
            prompt=p.get("prompt", ""), negative_prompt=p.get("negative", "worst quality, blurry"),
            seed=int(p.get("seed", 42)), height=int(p.get("height", 512)), width=int(p.get("width", 768)),
            num_frames=nf, frame_rate=fps, num_inference_steps=int(p.get("steps", 30)),
            video_guider_params=vg, images=imgs, audio_path=p["audio_path"],
            audio_start_time=float(p.get("audio_start_time", 0.0)),
            audio_max_duration=p.get("audio_max_duration"),
            tiling_config=TilingConfig.default(), enhance_prompt=bool(p.get("enhance", False)))
    elif ptype == "iclora":                                  # #24 ControlNet IC-LoRA (video-guida canny/depth/pose)
        cv = p["control_video"]; cstr = float(p.get("control_strength", 0.8))
        print(f"[controlnet] IC-LoRA '{p.get('control_type')}' guida={cv} strength={cstr}", flush=True)
        imgs = [ImageConditioningInput(p["image_path"], 0, float(p.get("ref_strength", 1.0)), 33)] if p.get("image_path") else []
        video, audio = pipe(
            prompt=p.get("prompt", ""), negative_prompt=p.get("negative", "worst quality, blurry"),
            seed=int(p.get("seed", 42)), height=int(p.get("height", 512)), width=int(p.get("width", 768)),
            num_frames=nf, frame_rate=fps, num_inference_steps=int(p.get("steps", 40)),
            video_guider_params=vg, images=imgs, video_conditioning=[(cv, cstr)],
            tiling_config=TilingConfig.default())
    else:
        images = []
        if p.get("keyframes"):                               # multi-keyframe: foto a frame diversi (rotazione/angoli coerenti)
            for kf in p["keyframes"]:
                fr = int(kf.get("frame", 0))
                fr = max(0, min(nf - 1, fr))
                images.append(ImageConditioningInput(kf["path"], fr, float(kf.get("strength", 1.0)), 33))
        elif p.get("image_path"):                            # i2v: condiziona sul primo frame
            images = [ImageConditioningInput(p["image_path"], 0, 1.0, 33)]
        video, audio = pipe(
            prompt=p.get("prompt", ""), negative_prompt=p.get("negative", "worst quality, blurry"),
            seed=int(p.get("seed", 42)), height=int(p.get("height", 512)), width=int(p.get("width", 768)),
            num_frames=nf, frame_rate=fps, num_inference_steps=int(p.get("steps", 40)),
            video_guider_params=vg, audio_guider_params=ag, images=images,
            tiling_config=TilingConfig.default(), enhance_prompt=bool(p.get("enhance", False)))
    print(f"[gen] {time.time() - tg:.0f}s ({nf} frames @ {fps}fps = {nf / fps:.1f}s video, {ptype})", flush=True)
    out = p["output"]
    encode_video(video=video, fps=fps, audio=audio, output_path=out,
                 video_chunks_number=get_video_chunks_number(nf, TilingConfig.default()))
    return out


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    print("OK:" + run_generation(p), flush=True)


if __name__ == "__main__":
    main()
