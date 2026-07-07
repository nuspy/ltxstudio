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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root repo (import workers.*)

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
        # PARAMETRI su meta: alcuni checkpoint (es. distilled "transformer_only") non contengono i pesi
        # del feature_extractor audio/video (video_aggregate_embed/audio_aggregate_embed, usati solo per
        # i2v/a2v) → restano su meta e il forward esplode ("Tensor on device meta"). Li materializziamo a
        # zero sul device reale: per il t2v sono inutilizzati (contributo nullo), per i2v/a2v verrebbero
        # comunque sovrascritti dai pesi del checkpoint se presenti.
        for pn, p in list(getattr(mod, "_parameters", {}).items()):
            if p is None or not getattr(p, "is_meta", False):
                continue
            # dtype di CALCOLO (bf16): un meta-param float32 (default) andrebbe in conflitto col forward
            # bf16 → "mat1 and mat2 dtype BFloat16 != Float". Se il param era già non-float32 lo rispettiamo.
            _pd = p.dtype if p.dtype not in (torch.float32, torch.float64) else torch.bfloat16
            mod._parameters[pn] = torch.nn.Parameter(
                torch.zeros(p.shape, dtype=_pd, device=device), requires_grad=False)


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


def _patch_sage_attention():
    """ACCELERAZIONE: instrada l'attention di ltx-core (SDPAAttention chiama
    F.scaled_dot_product_attention) su **SageAttention** (attention quantizzata fp8, come ComfyUI) →
    forward del transformer molto più veloce (taglia i ~90s/step del 22B). Fallback sicuro alla SDPA
    originale per i casi non supportati (mask presente, head_dim≠64/128, dtype non half). Env LTX_SAGE=0 disattiva."""
    if os.environ.get("LTX_SAGE", "1") != "1":
        return
    try:
        import torch
        import torch.nn.functional as F
        from sageattention import sageattn
        _orig_sdpa = F.scaled_dot_product_attention

        def _sdpa_sage(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
            try:
                if (attn_mask is None and dropout_p == 0.0 and q.dtype in (torch.float16, torch.bfloat16)
                        and q.dim() == 4 and q.shape[-1] in (64, 128)):
                    return sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)
            except Exception:                                # noqa: BLE001 — qualunque incompatibilità → SDPA
                pass
            return _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, **kw)

        F.scaled_dot_product_attention = _sdpa_sage
        print("[sage] SageAttention attiva (SDPA→sageattn, fp8) — fallback SDPA per mask/shape non supportate", flush=True)
    except Exception as _e:                                  # noqa: BLE001
        print(f"[sage] non attivata: {_e}", flush=True)


_patch_sage_attention()


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
    if os.path.isabs(ckpt) and os.path.exists(ckpt):
        return ckpt
    if not os.path.isabs(ckpt):
        for base in (r"E:\AI\Models\diffusion_models\LTX", r"E:\AI\Models\Checkpoints\LTX",
                     r"E:\AI\Models\diffusion_models", r"E:\AI\Models\Checkpoints"):
            cand = os.path.join(base, ckpt)
            if os.path.exists(cand):
                return cand
        import glob
        hits = glob.glob(os.path.join(r"E:\AI\Models\diffusion_models", "**", os.path.basename(ckpt)),
                         recursive=True)
        if hits:
            return hits[0]
    # NON trovato: invece di restituire un path fantasma (→ FileNotFoundError su OGNI chunk = 0 output),
    # ripiega sul checkpoint di default e AVVISA. Il modello scelto era inesistente/spostato.
    print(f"[ckpt] '{ckpt}' non trovato → uso il default {os.path.basename(CHECKPOINT)}", flush=True)
    return CHECKPOINT


def _resolve_gguf(g: str) -> str:
    """Path ASSOLUTO del GGUF transformer (la GUI passa relativi → np.memmap fallisce). Cerca in
    diffusion_models/Checkpoints; se il file non esiste, ripiega sull'auto-GGUF esistente (mai un path
    fantasma che crasha). Ritorna "" se davvero non c'è (→ il worker userà il checkpoint fp8 base)."""
    if not g:
        return g
    if os.path.isabs(g) and os.path.exists(g):
        return g
    import glob
    for base in (r"E:\AI\Models\diffusion_models", r"E:\AI\Models\Checkpoints"):
        cand = os.path.join(base, g)
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(r"E:\AI\Models\diffusion_models", "**", os.path.basename(g)), recursive=True)
    if hits:
        return hits[0]
    ag = _auto_gguf()
    if ag:
        print(f"[gguf] '{g}' non trovato → auto {os.path.basename(ag)}", flush=True)
        return ag
    print(f"[gguf] '{g}' non trovato e nessun GGUF disponibile → uso il checkpoint fp8 base", flush=True)
    return ""


def _auto_offload(p: dict, ckpt: str) -> str:
    """Modalità offload sulla VRAM LIBERA, stimando dal transformer EFFETTIVO:
      - GGUF esplicito → resta QUANTIZZATO e residente (~dim file, es. Q3 10GB / Q4 14GB), NON il fp8 base.
      - altrimenti il checkpoint fp8 (fp8_scaled_mm, niente upcast).
    'none' = TUTTO residente su GPU (transformer + gemma/vae) = massima velocità e ZERO offload → si sfrutta
    la VRAM libera invece di offloadare a vuoto (richiesta utente: coi 15GB liberi del Q3 tieni gli encoder in
    GPU). 'cpu' solo se non c'entra il full-stack. Rispetta p['offload'] esplicito."""
    if p.get("offload"):
        return p["offload"]
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        _gguf = p.get("gguf_transformer")
        if _gguf and os.path.exists(_gguf):
            tr_gb = os.path.getsize(_gguf) / 1e9            # GGUF: resta quantizzato → footprint ≈ dim file
            _how = f"gguf {os.path.basename(_gguf)[:24]}"
        else:
            try:
                from core import vram_estimate as _ve
                _mb, _src = _ve.estimate_mb(ckpt, fp8_storage=True, working_mb=0)
                tr_gb = (_mb / 1024.0) if _mb else (os.path.getsize(ckpt) / 1e9 if os.path.exists(ckpt) else 22.0)
                _how = _src if _mb else "file"
            except Exception:                                # noqa: BLE001
                tr_gb = os.path.getsize(ckpt) / 1e9 if os.path.exists(ckpt) else 22.0
                _how = "file"
        _ENC_GB = 10.0                                       # gemma text-encoder (~9) + vae + vocoder
        _WORK_GB = 8.0                                       # attivazioni temporali (picco denoise/decode)
        need = tr_gb + _ENC_GB + _WORK_GB                    # full-stack residente
        mode = "none" if free_gb >= need else "cpu"
        print(f"[auto-offload] free={free_gb:.0f}GB tr={tr_gb:.0f}({_how})+enc{_ENC_GB:.0f}+work{_WORK_GB:.0f}"
              f"=need{need:.0f} → {mode}", flush=True)
        return mode
    except Exception:                                        # noqa: BLE001
        return "cpu"


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
    if ptype == "vinpaint":                                  # #59 latent-inpaint VIDEO (sottoclasse ic_lora)
        from workers.ltx_inpaint_pipeline import InpaintPipeline
        pipe = InpaintPipeline(
            distilled_checkpoint_path=ckpt, spatial_upsampler_path=UPSAMPLER,
            gemma_root=GEMMA, loras=loras, quantization=quant, offload_mode=offload)
    elif ptype == "iclora":                                  # #24 ControlNet IC-LoRA (canny/depth/pose)
        from ltx_pipelines.ic_lora import ICLoraPipeline
        pipe = ICLoraPipeline(
            distilled_checkpoint_path=ckpt, spatial_upsampler_path=UPSAMPLER,
            gemma_root=GEMMA, loras=loras, quantization=quant, offload_mode=offload)
    elif ptype == "retake":                                  # #24 ReTake: rigenera sezione [start,end] di un video
        from ltx_pipelines.retake import RetakePipeline
        pipe = RetakePipeline(
            checkpoint_path=ckpt, gemma_root=GEMMA, loras=loras,
            quantization=quant, offload_mode=offload, distilled=True)
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
                print(f"[auto-gguf] selezionato: {hits[0]}", flush=True)
                return hits[0]
    print("[auto-gguf] nessun .gguf LTX trovato → checkpoint pieno", flush=True)
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
        # #29 durata-illimitata MULTI-PROMPT: prompt diverso per finestra temporale (prompts=[{start_s,prompt}])
        if p.get("prompts"):
            t_chunk = i * seg_dur
            active = p["prompts"][0].get("prompt", cp.get("prompt", ""))
            for sp in p["prompts"]:
                if float(sp.get("start_s", 0)) <= t_chunk:
                    active = sp.get("prompt", active)
            cp["prompt"] = active
            cp.pop("prompts", None)
        if p.get("audio_path"):
            cp["audio_start_time"] = round(i * seg_dur, 3)
            cp["audio_max_duration"] = round(seg_dur, 3)
        cp["output"] = f"{base}_chunk{i}.mp4"
        print(f"[chunk] {i + 1}/{n} ({cf} frame, audio_start={cp.get('audio_start_time', 0)})", flush=True)
        try:
            run_generation(cp)                               # pipe in cache dal 2° chunk → veloce
        except Exception as _e:                              # es. ultimo chunk con audio più corto dei frame
            print(f"[chunk] {i} saltato ({type(_e).__name__}: {_e})", flush=True)
            if i == 0:                                       # il PRIMO chunk fallito = nessun output: NON
                raise                                        # inghiottire, propaga il vero errore all'utente
            break                                            # un chunk finale corto è tollerabile (tieni i precedenti)
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


def run_t2a(p: dict) -> str:
    """#29 text→audio NATIVO (T2AOneStagePipeline, audio-only). Genera audio dal prompt → wav. ≤10s test."""
    import time
    from ltx_pipelines.t2a_one_stage import T2AOneStagePipeline
    from ltx_pipelines.utils.media_io import encode_audio
    ckpt = _resolve_ckpt(p.get("model", CHECKPOINT))
    # Il block-streaming (offload CPU) della T2AOneStagePipeline supporta SOLO bf16/fp8_cast
    # (fp8_scaled_mm non è ammesso) → default a fp8_cast, che entra nei 32GB col cast on-the-fly.
    qmode = p.get("quant", "fp8_cast")
    quant = None
    if qmode == "fp8_scaled_mm":
        from ltx_core.quantization import fp8_scaled_mm
        quant = fp8_scaled_mm.build_policy(ckpt)
    elif qmode == "fp8_cast":
        from ltx_core.quantization import fp8_cast
        quant = fp8_cast.build_policy(ckpt)
    pipe = T2AOneStagePipeline(checkpoint_path=ckpt, gemma_root=GEMMA, loras=[], quantization=quant,
                               offload_mode=OffloadMode.CPU)
    fps = float(p.get("fps", 25.0))
    nf = int(p.get("num_frames", int(float(p.get("seconds", 5.0)) * fps) // 8 * 8 + 1))
    ag = MultiModalGuiderParams(cfg_scale=float(p.get("guidance", 7.0)), stg_scale=1.0,
                                rescale_scale=0.7, modality_scale=3.0, stg_blocks=[29])
    t0 = time.time()
    audio = pipe(prompt=p.get("prompt", ""), negative_prompt=p.get("negative", ""),
                 seed=int(p.get("seed", 42)), num_frames=nf, frame_rate=fps, audio_guider_params=ag,
                 num_inference_steps=int(p.get("steps", 30)))
    out = p["output"]
    encode_audio(audio=audio, output_path=out)
    print(f"[t2a] {time.time()-t0:.0f}s → {out}", flush=True)
    return out


def _ensure_stereo(audio_path: str) -> str:
    """L'audio VAE LTX-2 richiede 2 canali (conv_in groups=1, 2ch). Le registrazioni voce sono spesso
    MONO → conv2d fallisce ('expected 2 channels, got 1'). Convertiamo a stereo via ffmpeg se serve;
    ritorna il path (eventualmente convertito in temp). Idempotente: se già stereo, ritorna l'originale."""
    import subprocess
    import tempfile
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                                "-show_entries", "stream=channels", "-of", "csv=p=0", audio_path],
                               capture_output=True, text=True, timeout=30)
        ch = (probe.stdout or "").strip()
        if ch == "2":
            return audio_path
    except Exception:                                          # noqa: BLE001
        pass                                                  # in dubbio: forziamo la conversione sotto
    out = os.path.join(tempfile.gettempdir(), "ltx_a2v_stereo.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio_path, "-ac", "2", out],
                   capture_output=True, timeout=120)
    return out if os.path.exists(out) else audio_path


def _prep_retake_source(video_path: str, width, height) -> str:
    """#24 ReTake OOM-fix: la pipeline retake encoda il sorgente alla sua risoluzione NATIVA (usata come
    output_shape). Un sorgente grande (es. 704x512+) fa saturare la VRAM nell'image_conditioner (encoder
    su GPU insieme al transformer) → OOM su 32GB anche con offload. Qui ridimensioniamo il sorgente al
    TARGET richiesto (multipli di 32) se è più grande; se il target non è dato, cap del lato lungo a 768.
    Idempotente: se il sorgente è già ≤ target, ritorna l'originale."""
    import subprocess
    import tempfile
    def _r32(x):
        return max(32, int(round(float(x) / 32.0)) * 32)
    try:
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
                            capture_output=True, text=True, timeout=30)
        sw, sh = (int(v) for v in (pr.stdout or "").strip().split(",")[:2])
    except Exception:                                          # noqa: BLE001
        return video_path                                      # non probeable → lascia decidere alla pipeline
    if width and height:
        tw, th = _r32(width), _r32(height)
    else:                                                      # nessun target → cap lato lungo a 768
        cap = 768
        if max(sw, sh) <= cap:
            return video_path
        scale = cap / float(max(sw, sh))
        tw, th = _r32(sw * scale), _r32(sh * scale)
    if sw <= tw and sh <= th:
        return video_path                                      # già abbastanza piccolo
    out = os.path.join(tempfile.gettempdir(), "ltx_retake_src.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_path, "-vf", f"scale={tw}:{th}",
                    "-an", out], capture_output=True, timeout=180)
    if os.path.exists(out):
        print(f"[retake] sorgente {sw}x{sh} → {tw}x{th} (anti-OOM image_conditioner)", flush=True)
        return out
    return video_path


def _stage(msg: str) -> None:
    """Scrive la FASE corrente della pipeline su core.preview_file → il dialog anteprima la mostra (richiesta
    utente: sapere cosa sta facendo la GPU). Anche su stdout per i log. Best-effort (mai rompe la gen)."""
    print(f"[stage] {msg}", flush=True)
    try:
        from core import preview_file as _pf
        _pf.write_message(msg)
    except Exception:                                          # noqa: BLE001
        pass


def _install_step_progress() -> None:
    """Intercetta il tqdm del sampler LTX (ltx_pipelines.utils.samplers) per emettere lo STEP del denoise su
    core.preview_file → barra 'step n/N' nel dialog anteprima. Best-effort, una volta per processo."""
    try:
        import ltx_pipelines.utils.samplers as _S
        if getattr(_S, "_ltx_tqdm_patched", False):
            return
        from tqdm import tqdm as _BT
        from core import preview_file as _pf

        class _PTqdm(_BT):
            def update(self, n=1):
                r = super().update(n)
                try:
                    _pf.write_progress(int(self.n), int(self.total or self.n))
                except Exception:                              # noqa: BLE001
                    pass
                return r
        _S.tqdm = _PTqdm
        _S._ltx_tqdm_patched = True
        print("[stage] progresso per-step attivo (denoise → dialog anteprima)", flush=True)
    except Exception as _e:                                    # noqa: BLE001
        print(f"[stage] progresso per-step non attivo: {_e}", flush=True)


def run_generation(p: dict) -> str:
    """Genera un video dai params. Riusabile da worker one-shot e da server persistente."""
    import time
    _install_step_progress()
    if p.get("t2a"):                                          # #29 text→audio (no video)
        return run_t2a(p)
    # RISOLUZIONE multipla di 64 (le pipeline two-stage LTX-2 lo richiedono: altrimenti il chunk viene
    # SALTATO con "Resolution HxW is not divisible by 64" → 0 frame → nessun output). Arrotondiamo al 64
    # più vicino (min 64) qui, choke point per ogni chiamante (GUI/API), invece di far fallire il chunk.
    def _r64(x):
        return max(64, int(round(float(x) / 64.0)) * 64)
    _ow, _oh = int(p.get("width", 768)), int(p.get("height", 512))
    p["width"], p["height"] = _r64(_ow), _r64(_oh)
    if (p["width"], p["height"]) != (_ow, _oh):
        print(f"[nativo] risoluzione {_ow}x{_oh} → {p['width']}x{p['height']} (multipli di 64 richiesti da LTX-2)", flush=True)
    # Video lungo → chunking in-process (memory-bounded, niente sborzo stage_2). Attiva con
    # chunk_frames>0 (o auto sotto). Ricorsione sicura: i chunk hanno num_frames<=chunk.
    _chunk = int(p.get("chunk_frames", 0))
    if _chunk and int(p.get("num_frames", 0)) > _chunk:
        return _run_chunked(p, _chunk)
    # Un GGUF è transformer-only (niente vae/gemma): se arriva come 'model', spostalo su gguf_transformer
    # (il checkpoint base di default fornisce i componenti mancanti).
    if str(p.get("model", "")).lower().endswith(".gguf") and "gguf_transformer" not in p:
        p["gguf_transformer"] = p.pop("model")
    # Auto-GGUF: SOLO se l'utente NON ha scelto un modello (auto). Con un modello ESPLICITO (es. il 22B fp8)
    # lo RISPETTIAMO e lo facciamo entrare con l'offload dinamico per-blocco (come ComfyUI) — niente più
    # sostituzione a forza con un GGUF più piccolo (era un downgrade non richiesto della qualità, e nel caso
    # limite puntava a un file inesistente → 0 output). Forzabile con gguf_transformer esplicito.
    if "gguf_transformer" not in p and not p.get("model"):
        g = _auto_gguf()
        if g:
            print("[auto-gguf] nessun modello scelto → GGUF Q4 di default", flush=True)
            p["gguf_transformer"] = g
            p["_gguf_auto"] = True                          # AUTO (non scelto): scartabile se il fp8 entra su GPU
    # RISOLVI il GGUF a path ASSOLUTO: la GUI passa un path RELATIVO (es. "LTXVideo\\LTX-2\\…Q4.gguf") e
    # np.memmap NON lo trova rispetto alla cwd → "cannot mmap" / FileNotFoundError su ogni chunk. Resolve in
    # diffusion_models; se manca del tutto, ripiega sull'auto-GGUF esistente.
    if p.get("gguf_transformer"):
        p["gguf_transformer"] = _resolve_gguf(p["gguf_transformer"])
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
    # #59/#60 Edit/Inpaint VIDEO: LoRA editing (ltx23_edit_anything / ltx23_inpaint_masked) + guide video
    # (sorgente per edit; mask-embedded magenta per inpaint). Riusa la pipeline iclora (video_conditioning).
    _edit = p.get("edit_lora")
    if _edit and not os.path.isabs(_edit):
        _edit = os.path.join(LORA_DIR, _edit)
    if p.get("inpaint_masks_dir") and p.get("control_video") and _edit and os.path.exists(_edit):
        ptype = "vinpaint"                                   # #59 inpaint: guide=video orig + attention_mask + CFG
        loras = list(loras) + [LoraPathStrengthAndSDOps(_edit, float(p.get("control_strength", 1.0)),
                                                        LTXV_LORA_COMFY_RENAMING_MAP)]
    elif p.get("retake") and p.get("video_path"):            # #24 ReTake: rigenera sezione di un video
        ptype = "retake"
    elif _edit and os.path.exists(_edit) and p.get("control_video"):
        ptype = "iclora"
        loras = list(loras) + [LoraPathStrengthAndSDOps(_edit, float(p.get("control_strength", 1.0)),
                                                        LTXV_LORA_COMFY_RENAMING_MAP)]
    elif _ic and os.path.exists(_ic):
        ptype = "iclora"
        loras = list(loras) + [LoraPathStrengthAndSDOps(_ic, float(p.get("control_strength", 0.8)),
                                                        LTXV_LORA_COMFY_RENAMING_MAP)]
    else:
        ptype = "a2v" if p.get("audio_path") else "ti2v"      # audio dato → lip-sync/canto NATIVO
    _off = _auto_offload(p, ckpt)
    if _off == "none" and p.get("_gguf_auto"):
        # SOLO se il GGUF era AUTO (nessuna scelta): con 'none' il fp8 nativo entra su GPU → più veloce del
        # dequant GGUF. Un GGUF ESPLICITO (scelto dall'utente) NON si scarta: resta residente quantizzato
        # (patch_stage_to_gguf forza OffloadMode.NONE, per-layer dequant) e gli encoder stanno pure in GPU.
        if p.pop("gguf_transformer", None):
            print("[auto-offload] none → fp8 nativo su GPU (GGUF auto disattivato per velocità)", flush=True)
    _stage(f"Caricamento modello ({os.path.basename(ckpt)[:28]}, offload={_off})…")
    pipe = _get_pipe(ckpt, qmode, _off, loras, ptype=ptype)
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
        for _st in (getattr(pipe, "stage_1", None), getattr(pipe, "stage_2", None),
                    getattr(pipe, "stage", None)):           # retake usa 'stage' singolo
            if _st is not None:
                _GG.patch_stage_to_gguf(_st, p["gguf_transformer"], ckpt)
        print(f"[gguf] transformer da {os.path.basename(p['gguf_transformer'])} (dequant on-the-fly)", flush=True)

    # #104 TeaCache: cache training-free del velocity-model (salta forward su step quasi-invariati). thresh>0
    # attiva; conviene sulle modalità a molti step (HQ/Pro). Soglia da calibrare live. Subprocess per-job → no disable.
    _tc_thresh = float(p.get("teacache_thresh", 0) or 0)
    if _tc_thresh > 0:
        try:
            from workers import teacache as _TC
        except ModuleNotFoundError:
            import teacache as _TC
        # PATCH DI CLASSE (fix calibrazione 07/07): il transformer è costruito LAZY dentro la run →
        # il vecchio patch sugli stage non lo trovava mai ("attivo su 0 stage" = zero skip reali).
        _ok = _TC.enable_class(_tc_thresh)
        print(f"[teacache] patch classe X0Model {'ATTIVO' if _ok is not None else 'NON attivo'} "
              f"thresh={_tc_thresh}", flush=True)

    vg = MultiModalGuiderParams(cfg_scale=float(p.get("guidance", 3.0)),
                                stg_scale=float(p.get("stg", 1.0)),
                                rescale_scale=0.7, modality_scale=3.0, stg_blocks=[29])
    ag = MultiModalGuiderParams(cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                                modality_scale=float(p.get("audio_scale", 3.0)), stg_blocks=[29])
    nf = int(p.get("num_frames", 121))
    fps = float(p.get("fps", 25.0))
    tg = time.time()
    _stage(f"Generazione video — {int(p.get('steps', 30))} step (encode prompt + denoise)…")
    if ptype == "vinpaint":                                  # #59 inpaint: guide=video orig + attention_mask + CFG
        import glob as _glob
        from workers.ltx_inpaint_cond import load_mask_tensor
        mks = sorted(_glob.glob(os.path.join(p["inpaint_masks_dir"], "*.png")))
        nf = min(nf, len(mks))
        masks_t = load_mask_tensor(mks[:nf], int(p.get("height", 320)), int(p.get("width", 512)),
                                   dilate=int(p.get("mask_dilate", 0)), blur=float(p.get("mask_blur", 0)))
        video = pipe.inpaint(
            prompt=p.get("prompt", ""), negative_prompt=p.get("negative", "worst quality, blurry"),
            seed=int(p.get("seed", 42)), height=int(p.get("height", 320)), width=int(p.get("width", 512)),
            num_frames=nf, frame_rate=fps, attention_mask=masks_t,
            video_conditioning=[(p["control_video"], float(p.get("control_strength", 0.5)))],
            cfg_scale=float(p.get("cfg_scale", 5.0)), num_steps=int(p.get("num_steps", 30)))
        audio = None
    elif ptype == "a2v":                                     # AUDIO→VIDEO nativo: foto + canzone → canto
        imgs = [ImageConditioningInput(p["image_path"], 0, float(p.get("ref_strength", 1.0)), 33)] if p.get("image_path") else []
        video, audio = pipe(
            prompt=p.get("prompt", ""), negative_prompt=p.get("negative", "worst quality, blurry"),
            seed=int(p.get("seed", 42)), height=int(p.get("height", 512)), width=int(p.get("width", 768)),
            num_frames=nf, frame_rate=fps, num_inference_steps=int(p.get("steps", 30)),
            video_guider_params=vg, images=imgs, audio_path=_ensure_stereo(p["audio_path"]),
            audio_start_time=float(p.get("audio_start_time", 0.0)),
            audio_max_duration=p.get("audio_max_duration"),
            tiling_config=TilingConfig.default(), enhance_prompt=bool(p.get("enhance", False)))
    elif ptype == "retake":                                  # #24 ReTake: rigenera [start,end] di un video sorgente
        print(f"[retake] {p['video_path']} sezione [{p.get('start_time',0)},{p.get('end_time',0)}]s", flush=True)
        _rt_src = _prep_retake_source(p["video_path"], p.get("width"), p.get("height"))
        # #24 anti-OOM 32GB: il retake carica un encoder extra su GPU (image_conditioner) col transformer
        # già in memoria → picco al limite. Deframmenta prima (expandable_segments + empty_cache) per dare
        # spazio contiguo all'encoder ed evitare l'OOM flaky al bordo dei 32GB.
        import gc as _gc
        _gc.collect(); torch.cuda.empty_cache()
        video, audio = pipe(
            video_path=_rt_src, prompt=p.get("prompt", ""),
            start_time=float(p.get("start_time", 0.0)), end_time=float(p.get("end_time", 0.0)),
            seed=int(p.get("seed", 42)), negative_prompt=p.get("negative", "worst quality, blurry"),
            num_inference_steps=int(p.get("steps", 40)), video_guider_params=vg,
            regenerate_video=bool(p.get("regenerate_video", True)),
            regenerate_audio=bool(p.get("regenerate_audio", True)),
            tiling_config=TilingConfig.default())
    elif ptype == "iclora":                                  # #24 ControlNet IC-LoRA (video-guida canny/depth/pose)
        cv = p["control_video"]; cstr = float(p.get("control_strength", 0.8))
        print(f"[controlnet] IC-LoRA '{p.get('control_type')}' guida={cv} strength={cstr}", flush=True)
        imgs = [ImageConditioningInput(p["image_path"], 0, float(p.get("ref_strength", 1.0)), 33)] if p.get("image_path") else []
        # #59 INPAINT (meccanismo ufficiale InOutpainting): conditioning_attention_mask = INVERSO della
        # maschera oggetto → dove mask oggetto=1 l'attention=0 (la IC-LoRA non vincola → genera dal prompt),
        # fuori attention=1 (preserva il guide=video originale). NIENTE noise injection.
        cam = None
        if p.get("attention_mask_dir"):
            import glob as _g2
            from workers.ltx_inpaint_cond import load_mask_tensor
            mks = sorted(_g2.glob(os.path.join(p["attention_mask_dir"], "*.png")))[:nf]
            mt = load_mask_tensor(mks, int(p.get("height", 512)), int(p.get("width", 768)))  # (T,H,W) 0..1
            cam = (1.0 - mt)[None, None].to("cuda")          # (1,1,T,H,W) inverso = inpaint
        # ICLoraPipeline.__call__: prompt/seed/height/width/num_frames/frame_rate/images/video_conditioning
        # (+ enhance_prompt/tiling/conditioning_attention_strength/conditioning_attention_mask). NO negative/steps.
        video, audio = pipe(
            prompt=p.get("prompt", ""), seed=int(p.get("seed", 42)),
            height=int(p.get("height", 512)), width=int(p.get("width", 768)),
            num_frames=nf, frame_rate=fps, images=imgs, video_conditioning=[(cv, cstr)],
            enhance_prompt=bool(p.get("enhance", False)), tiling_config=TilingConfig.default(),
            conditioning_attention_strength=float(p.get("conditioning_attention", 1.0)),
            conditioning_attention_mask=cam)
    else:
        images = []
        if p.get("keyframes"):                               # multi-keyframe: foto a frame diversi (rotazione/angoli coerenti)
            for kf in p["keyframes"]:
                fr = int(kf.get("frame", 0))
                fr = max(0, min(nf - 1, fr))
                images.append(ImageConditioningInput(kf["path"], fr, float(kf.get("strength", 1.0)), 33))
        elif p.get("image_path"):                            # i2v: condiziona sul primo frame
            # Forza immagine (slider GUI ref_strength): 1.0 = guida piena ("addguide"); valori più bassi =
            # riferimento "soft/inplace" (l'immagine condiziona meno → il modello reinterpreta di più).
            _ref_str = float(p.get("ref_strength", 1.0))
            if p.get("i2v_method") == "inplace":             # inplace-soft: riferimento sempre morbido
                _ref_str = min(_ref_str, 0.6)
            images = [ImageConditioningInput(p["image_path"], 0, _ref_str, 33)]
        video, audio = pipe(
            prompt=p.get("prompt", ""), negative_prompt=p.get("negative", "worst quality, blurry"),
            seed=int(p.get("seed", 42)), height=int(p.get("height", 512)), width=int(p.get("width", 768)),
            num_frames=nf, frame_rate=fps, num_inference_steps=int(p.get("steps", 40)),
            video_guider_params=vg, audio_guider_params=ag, images=images,
            tiling_config=TilingConfig.default(), enhance_prompt=bool(p.get("enhance", False)))
    print(f"[gen] {time.time() - tg:.0f}s ({nf} frames @ {fps}fps = {nf / fps:.1f}s video, {ptype})", flush=True)
    out = p["output"]
    _stage("Decodifica VAE e salvataggio video…")
    encode_video(video=video, fps=fps, audio=audio, output_path=out,
                 video_chunks_number=get_video_chunks_number(nf, TilingConfig.default()))
    try:                                                     # #107 firma invisibile di provenienza (default ON)
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception as _se:                                 # noqa: BLE001
        print(f"[stego] {_se}", flush=True)
    try:                                                     # #106 watermark obbligatorio (gated feature_matrix)
        from core import watermark
        out = watermark.apply_to_file(out)
    except Exception as _we:                                 # noqa: BLE001
        print(f"[watermark] {_we}", flush=True)
    return out


def _free_image_server() -> None:
    """Libera la VRAM del server immagini (:8189) PRIMA di calcolare l'offload video: senza questo, un
    modello immagine cachato (es. FLUX 22GB) resta in VRAM mentre il video ne chiede 25-33 → _auto_offload
    vede 10GB liberi e sceglie 'cpu' (lento) o peggio sborda in shared-RAM/OOM. Single-GPU: un pesante alla
    volta; il ModelManager del server ricaricherà il suo modello alla prossima richiesta immagine."""
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8189/free", data=b"{}", headers={"Content-Type": "application/json"}),
            timeout=20).read()
        print("[worker] VRAM server immagini liberata (/free)", flush=True)
    except Exception:                                        # noqa: BLE001  (server giù = niente da liberare)
        pass


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    _free_image_server()
    print("OK:" + run_generation(p), flush=True)


if __name__ == "__main__":
    main()
