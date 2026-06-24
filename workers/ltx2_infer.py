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

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
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
    for mod in model.modules():                          # stack di layer più grande (transformer/Gemma)
        for attr in ("transformer_blocks", "layers"):
            tb = getattr(mod, attr, None)
            if isinstance(tb, torch.nn.ModuleList) and len(tb) > best_n:
                best, best_n, holder = tb, len(tb), mod
    if best is None or best_n < 8:
        return 0
    model.to(device)                                    # parti non-layer (embed/norm/proj) su GPU
    blocks = list(best)
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

    holder.register_forward_pre_hook(_force_layers_cpu)
    return best_n


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
            print(f"[bt] _build_transformer chiamato, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
            x0 = _orig_bt(self, **kwargs)
            try:
                n = _apply_blockwise_offload(x0, torch.device("cuda"))
                print(f"[block-offload] transformer: {n} blocchi su CPU on-demand", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[block-offload] warn: {_e}", flush=True)
            return x0

        _blk.DiffusionStage._build_transformer = _patched_bt
        print("[block-offload] patch _build_transformer attivo", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[block-offload] patch fallito: {_e}", flush=True)


_patch_block_offload()


try:
    import ltx_core.loader.single_gpu_model_builder as _sgmb
    _orig_build = _sgmb.SingleGPUModelBuilder.build

    def _patched_build(self, device=None, dtype=None, **kw):
        import torch
        dev = torch.device("cuda") if device is None else device
        m = _orig_build(self, device=device, dtype=dtype, **kw)
        metas = [n for n, b in m.named_buffers() if getattr(b, "is_meta", False)]
        metap = [n for n, p in m.named_parameters() if getattr(p, "is_meta", False)]
        print(f"[meta-fix] build {type(m).__name__}: {len(metas)} buffer meta, {len(metap)} param meta", flush=True)
        print("[meta-fix] ATTESI(primi):", metap[:6], flush=True)
        print("[meta-fix] ATTESI(text):", [n for n in metap if 'language_model' in n][:5], flush=True)
        print("[meta-fix] ATTESI(altro):", [n for n in metap if 'vision' not in n and 'language_model' not in n][:6], flush=True)
        try:
            _materialize_meta(m, dev)
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


_PIPE_CACHE: dict = {}


def _resolve_ckpt(model: str) -> str:
    ckpt = model or CHECKPOINT
    if not os.path.isabs(ckpt):
        for base in (r"E:\AI\Models\diffusion_models\LTX", r"E:\AI\Models\Checkpoints\LTX"):
            cand = os.path.join(base, ckpt)
            if os.path.exists(cand):
                return cand
    return ckpt


def _get_pipe(ckpt: str, qmode: str, offload_str: str, loras: list, stages: str = "two"):
    """Istanzia (e CACHEA) la pipeline. Nel server persistente la cache evita il ri-load da disco."""
    lkey = tuple((lo.path, lo.strength) for lo in loras)
    key = (ckpt, qmode, offload_str, lkey, stages)
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
    pipe = TI2VidTwoStagesPipeline(
        checkpoint_path=ckpt, distilled_lora=[], spatial_upsampler_path=UPSAMPLER,
        gemma_root=GEMMA, loras=loras, quantization=quant, offload_mode=offload)
    _PIPE_CACHE.clear()                                       # 1 pipeline alla volta (VRAM/RAM)
    _PIPE_CACHE[key] = pipe
    return pipe


def run_generation(p: dict) -> str:
    """Genera un video dai params. Riusabile da worker one-shot e da server persistente."""
    import time
    # Niente sysmem-fallback Windows (che gonfia a 80-90GB mascherando l'OOM reale):
    # limita PyTorch alla VRAM fisica → numeri veri + OOM pulito, come fa ComfyUI.
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.98)
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
    pipe = _get_pipe(ckpt, qmode, p.get("offload", "cpu"), loras)
    print(f"[load] {time.time() - t0:.0f}s (model={os.path.basename(ckpt)} quant={qmode})", flush=True)
    if torch.cuda.is_available():
        print(f"[vram] post-load: {torch.cuda.memory_allocated()/1e9:.1f}GB alloc / "
              f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved", flush=True)

    vg = MultiModalGuiderParams(cfg_scale=float(p.get("guidance", 3.0)),
                                stg_scale=float(p.get("stg", 1.0)),
                                rescale_scale=0.7, modality_scale=3.0, stg_blocks=[29])
    ag = MultiModalGuiderParams(cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                                modality_scale=3.0, stg_blocks=[29])
    images = []
    if p.get("image_path"):                                  # i2v: condiziona sul primo frame
        images = [ImageConditioningInput(p["image_path"], 0, 1.0, 33)]
    nf = int(p.get("num_frames", 121))
    fps = float(p.get("fps", 25.0))
    tg = time.time()
    video, audio = pipe(
        prompt=p.get("prompt", ""), negative_prompt=p.get("negative", "worst quality, blurry"),
        seed=int(p.get("seed", 42)), height=int(p.get("height", 512)), width=int(p.get("width", 768)),
        num_frames=nf, frame_rate=fps, num_inference_steps=int(p.get("steps", 40)),
        video_guider_params=vg, audio_guider_params=ag, images=images,
        tiling_config=TilingConfig.default(), enhance_prompt=bool(p.get("enhance", False)))
    print(f"[gen] {time.time() - tg:.0f}s ({nf} frames @ {fps}fps = {nf / fps:.1f}s video)", flush=True)
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
