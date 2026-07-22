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
from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig


def _tiling_config(p: dict) -> TilingConfig:
    """TilingConfig per il decode VAE. Il decode LTX è GIÀ streaming (tile temporali → chunk → encode_video
    scrive su disco a flusso). `decode_lowmem` (toggle GUI) usa tile più piccoli: meno VRAM di picco per film
    lunghi / alta risoluzione (più lento, ma niente OOM). Default = preset ufficiale (768px / 80 frame)."""
    if p.get("decode_lowmem"):
        return TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=32, tile_overlap_in_frames=8),
        )
    return TilingConfig.default()
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage   # audio→video (lip-sync canto), nativo
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video as _encode_video_torchcodec
from ltx_pipelines.utils.types import OffloadMode


def _torchcodec_ok() -> bool:
    """torchcodec riesce a caricare le sue DLL FFmpeg?

    Se NO, `encode_video` — che è la funzione che SCRIVE l'mp4 — muore con SEGFAULT (exit 139):
    il video viene generato benissimo (denoise completo) e poi il processo esplode NELL'ATTO DI
    SALVARE, senza traceback e senza file. È il motivo per cui il videoclip riempiva la cartella di
    .json e non produceva un solo video, e per cui i test 'a occhio' sembravano crash casuali.
    """
    try:
        from torchcodec.encoders import VideoEncoder  # noqa: F401
        return True
    except Exception as _e:                                  # noqa: BLE001
        print(f"[encode] torchcodec NON caricabile ({type(_e).__name__}) → salvo con PyAV", flush=True)
        return False


def _encode_video_pyav(video, fps: int, audio, output_path: str, **_kw) -> None:
    """Salvataggio del video con PyAV (fallback quando torchcodec non carica le FFmpeg).

    Stessa semantica di encode_video: accetta un tensore [F,H,W,C] o un iteratore di chunk.
    """
    import av
    import numpy as np

    chunks = [video] if torch.is_tensor(video) else list(video)
    if not chunks:
        raise RuntimeError("encode: nessun frame da salvare")
    first = chunks[0]
    h, w = int(first.shape[-3]), int(first.shape[-2])

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "19", "preset": "veryfast"}
    for chunk in chunks:
        arr = chunk.detach().to("cpu").float().clamp(0, 1).mul(255).to(torch.uint8).numpy()
        for frame in arr:                                    # [H,W,C] uint8
            container.mux(stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame),
                                                                   format="rgb24")))
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()

    if audio is not None:                                    # audio: muxato con ffmpeg (PyAV già scritto il video)
        try:
            import shutil
            import subprocess
            import tempfile
            wav = tempfile.mktemp(suffix=".wav")
            import soundfile as sf
            # L'oggetto Audio di ltx_pipelines espone `waveform` (C,N), NON `data`: chiedere .data sollevava
            # AttributeError, l'except qui sotto lo declassava a una riga di log e OGNI video a2v (canto/talk)
            # usciva MUTO — proprio la funzione il cui unico scopo è il lip-sync sull'audio. Il ramo torchcodec
            # (che usa .waveform) non lo nascondeva, ma su Windows torchcodec non carica le sue FFmpeg → si
            # passa SEMPRE di qui.
            _w = getattr(audio, "waveform", None)
            if _w is None:
                _w = audio.data                              # compat con eventuali altri tipi audio
            arr = _w.detach().cpu().float().numpy()
            if arr.ndim == 3:                                # (B,C,N) → (C,N)
                arr = arr[0]
            sf.write(wav, arr.T, int(audio.sampling_rate))   # soundfile vuole (N,C)
            tmp = output_path + ".mux.mp4"
            ff = shutil.which("ffmpeg") or "ffmpeg"
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", output_path, "-i", wav,
                            "-c:v", "copy", "-c:a", "aac", "-shortest", tmp], check=True)
            os.replace(tmp, output_path)
        except Exception as _ae:                             # noqa: BLE001 — meglio il video muto che nessun video
            print(f"[encode] !! AUDIO NON MUXATO ({type(_ae).__name__}: {_ae}) → video MUTO", flush=True)


def encode_video(video, fps, audio, output_path, video_chunks_number=None, **kw):
    """Scrive l'mp4: torchcodec se le sue DLL caricano, altrimenti PyAV."""
    if _torchcodec_ok():
        return _encode_video_torchcodec(video=video, fps=fps, audio=audio, output_path=output_path,
                                        video_chunks_number=video_chunks_number, **kw)
    return _encode_video_pyav(video, fps, audio, output_path)

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
    originale per i casi non supportati (mask presente, head_dim≠64/128, dtype non half).

    ATTIVAZIONE: preferenza GUI (core.attention_backend), non più env. Default = flash/SDPA (esatto, più
    veloce su Blackwell) → Sage OFF salvo scelta esplicita 'sage'/'auto-su-GPU-vecchia'. Un solo switch."""
    try:
        from core import attention_backend
        _use_sage = attention_backend.ltx_use_sage()
    except Exception:                                        # noqa: BLE001 — fallback: env legacy
        _use_sage = os.environ.get("LTX_SAGE", "1") == "1"
    if not _use_sage:
        print("[attn] SageAttention OFF (preferenza GUI: flash/SDPA esatta)", flush=True)
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


def _patch_no_mmap_loader():
    """FIX 0xC0000005 / 'invalid python storage' (22/07): su Windows il percorso safetensors→torch
    (safe_open + UntypedStorage.from_file) mappa il checkpoint copy-on-write e impegna ~2× la sua
    dimensione di COMMIT (RAM+pagefile) → crash quando il commit scarseggia (misurato: soglia
    ≈ commit disponibile / 2; il file è integro). Lettura a I/O DIRETTO via core/st_direct
    (seek+readinto per-tensore): commit di picco = un tensore, non 2× il file.
    Opt-out: LTX_MMAP=1 ripristina il loader mmap originale."""
    if os.environ.get("LTX_MMAP") == "1":
        return
    try:
        from core import st_direct as _std
        import ltx_core.loader.sft_loader as _sft
        from ltx_core.loader.primitives import StateDict as _SD

        _DT = _std._dtypes()

        def _load(self, path, sd_ops=None, device=None):
            sd, size, dtypes = {}, 0, set()
            device = device or torch.device("cpu")
            for shard in (path if isinstance(path, list) else [path]):
                hdr, base = _std.read_header(shard)
                with open(shard, "rb") as fh:
                    for name, info in hdr.items():
                        if name == "__metadata__":
                            continue
                        expected = name if sd_ops is None else sd_ops.apply_to_key(name)
                        if expected is None:
                            continue
                        value = _std.read_tensor(fh, base, info, _DT).to(device=device)
                        pairs = ((expected, value),)
                        if sd_ops is not None:
                            pairs = sd_ops.apply_to_key_value(expected, value)
                        for k, v in pairs:
                            size += v.nbytes
                            dtypes.add(v.dtype)
                            sd[k] = v
            return _SD(sd=sd, device=device, size=size, dtype=dtypes)

        def _metadata(self, path):
            meta = _std.metadata(path)
            return json.loads(meta["config"]) if "config" in meta else {}

        _sft.SafetensorsStateDictLoader.load = _load
        _sft.SafetensorsModelStateDictLoader.metadata = _metadata
        # anche i chiamanti DIRETTI di safe_open (block_streaming.builder: scan chiavi + streaming
        # per-blocco con offload=cpu) — senza, il ramo block-swap moriva di os error 1455.
        if _std.patch_safe_open():
            print("[no-mmap] anche safetensors.safe_open → shim I/O diretto", flush=True)
        print("[no-mmap] loader safetensors a I/O diretto (commit-safe col pagefile piccolo)", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[no-mmap] patch fallito: {_e}", flush=True)


_patch_no_mmap_loader()


# ── PARCHEGGIO RAM (unified-model-manager): VRAM→RAM invece di eliminare ─────────────────────
# Richiesta utente: un modello scaricato che potrà servire di nuovo (stessa pipeline, modello
# comune tra pipeline, o pipeline successiva ignota) NON va distrutto ma parcheggiato in RAM
# e ricaricato da lì; quando la pipeline successiva è nota e non lo usa → rimosso dalla RAM.
try:
    from ltx_core.loader.registry import DummyRegistry as _DummyReg
except Exception:                                       # noqa: BLE001
    _DummyReg = object


class _CpuParkRegistry(_DummyReg):
    """Registry ltx-core che parcheggia in RAM i pesi letti da disco.

    Eredita da DummyRegistry DI PROPOSITO: `apply_loras` fonde in-place (rebind del dict) solo
    quando isinstance(registry, DummyRegistry) — con un registry "vero" farebbe un clone completo
    del transformer (2× VRAM sul load device = OOM sui 32GB). I tensori parcheggiati non vengono
    mai mutati: get() ritorna una SHALLOW-copy del dict (la fusione LoRA rebinda gli slot della
    copia, mai i tensori) e il .to(device/dtype) a valle crea copie nuove.

    Commit-aware: parcheggia solo se il commit (RAM+pagefile) resta sopra il floor — su questa
    macchina il limite vero è il commit, non la RAM fisica (vedi core/st_direct).
    Env: LTX_RAM_PARK=0 disattiva; LTX_RAM_PARK_MB budget (default 48GB);
         LTX_COMMIT_FLOOR_MB floor di commit da non intaccare (default 20GB)."""

    def __init__(self):
        self._sd: dict = {}                             # id → StateDict (tensori su CPU)
        self._info: dict = {}                           # id → {paths, mb, used}

    @staticmethod
    def _gen_id(paths, sd_ops) -> str:
        import hashlib
        parts = [os.path.abspath(str(p)) for p in paths]
        if sd_ops is not None:
            parts.append(getattr(sd_ops, "name", str(sd_ops)))
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    @staticmethod
    def _enabled() -> bool:
        try:                                            # commit 2: politica unificata (LS_RAM_CACHE=0
            from core import mem_budget                 # o legacy LTX_RAM_PARK=0 → off)
            return mem_budget.enabled()
        except Exception:                               # noqa: BLE001
            return os.environ.get("LTX_RAM_PARK", "1") != "0"

    def _parked_mb(self) -> float:
        return sum(i["mb"] for i in self._info.values())

    def get(self, paths, sd_ops):
        if not self._enabled():
            return None
        import time as _tm
        sid = self._gen_id(paths, sd_ops)
        e = self._sd.get(sid)
        if e is None:
            return None
        self._info[sid]["used"] = _tm.monotonic()
        print(f"[park] riuso da RAM: {os.path.basename(str(paths[0]))} "
              f"({self._info[sid]['mb']:.0f}MB, no disco)", flush=True)
        from ltx_core.loader.primitives import StateDict as _SD
        return _SD(sd=dict(e.sd), device=e.device, size=e.size, dtype=e.dtype)

    def add(self, paths, sd_ops, state_dict):
        if not self._enabled():
            return
        import time as _tm
        sid = self._gen_id(paths, sd_ops)
        if sid in self._sd:
            return
        mb = sum(v.nbytes for v in state_dict.sd.values()) / 2**20
        from core import mem_budget                     # commit 2: budget/floor/predicato unificati
        budget = mem_budget.ram_cache_mb()
        while self._info and self._parked_mb() + mb > budget:  # LRU nel budget
            self._drop(min(self._info, key=lambda k: self._info[k]["used"]))
        if not mem_budget.fits(mb, self._parked_mb()):
            print(f"[park] SKIP {os.path.basename(str(paths[0]))} ({mb:.0f}MB): "
                  "commit/RAM/budget insufficiente → niente parcheggio", flush=True)
            return
        cpu_sd = {k: (v if v.device.type == "cpu" else v.to("cpu")) for k, v in state_dict.sd.items()}
        from ltx_core.loader.primitives import StateDict as _SD
        self._sd[sid] = _SD(sd=cpu_sd, device=torch.device("cpu"),
                            size=state_dict.size, dtype=state_dict.dtype)
        self._info[sid] = {"paths": [os.path.abspath(str(p)) for p in paths], "mb": mb,
                           "used": _tm.monotonic()}
        print(f"[park] parcheggiato in RAM: {os.path.basename(str(paths[0]))} "
              f"({mb:.0f}MB, totale {self._parked_mb():.0f}MB)", flush=True)
        try:                                            # F3: ledger live (parcheggio = ram_parked)
            from core import vram_ledger
            vram_ledger.register(f"park:{os.path.basename(str(paths[0]))}", mb,
                                 category="video", state="ram_parked", path=str(paths[0]))
        except Exception:                               # noqa: BLE001
            pass

    def pop(self, paths, sd_ops):
        sid = self._gen_id(paths, sd_ops)
        self._info.pop(sid, None)
        return self._sd.pop(sid, None)

    def _drop(self, sid: str) -> None:
        i = self._info.pop(sid, None)
        self._sd.pop(sid, None)
        if i:
            print(f"[park] rimosso dalla RAM: {os.path.basename(i['paths'][0])} ({i['mb']:.0f}MB)",
                  flush=True)
            try:                                        # F3: ledger live
                from core import vram_ledger
                vram_ledger.unregister(f"park:{os.path.basename(i['paths'][0])}")
            except Exception:                           # noqa: BLE001
                pass

    def drop_except(self, keep_paths) -> float:
        """Regola di drop (pipeline del job ora NOTA): i pesi parcheggiati che il job non userà
        vengono rimossi dalla RAM. `keep_paths` = file O directory (prefix-match) del job."""
        keep = [os.path.abspath(str(p)) for p in keep_paths if p]
        gone = 0.0
        for sid in list(self._info):
            used = any(pp == k or pp.startswith(k + os.sep)
                       for pp in self._info[sid]["paths"] for k in keep)
            if not used:
                gone += self._info[sid]["mb"]
                self._drop(sid)
        return gone

    def clear(self) -> None:
        self._sd.clear()
        self._info.clear()

    def stats(self) -> list:
        return [{"path": i["paths"][0], "mb": round(i["mb"], 1)} for i in self._info.values()]


_PARK = _CpuParkRegistry()

CHECKPOINT = r"E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors"
UPSAMPLER = r"E:\AI\Models\latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA = r"E:\AI\Models\TextEncoders\gemma-3-ltx"              # Gemma-3 12B UFFICIALE Lightricks (non-gated, formato ltx-core)
LORA_DIR = r"E:\AI\Models\LoRA"
# LoRA di DISTILLAZIONE ufficiale (Lightricks/LTX-2.3, 7.6GB): OBBLIGATORIA per usare i checkpoint DEV
# (non-distilled) con la pipeline two-stage — senza, il dev esce "dematerializzato" (nuvole di voxel).
DISTILLED_LORA = r"E:\AI\Models\LoRA\LTX-2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
# IC-LoRA di controllo (canny/depth/pose) presenti su disco — #24 ControlNet nativo, nessun download
IC_LORA = {
    "canny": r"E:\AI\Models\LoRA\LTX-2\ltx-2-19b-ic-lora-canny-control.safetensors",
    "depth": r"E:\AI\Models\LoRA\LTX-2\ltx-2-19b-ic-lora-depth-control.safetensors",
    "pose": r"E:\AI\Models\LoRA\LTX-2\ltx-2-19b-ic-lora-pose-control.safetensors",
}


_PIPE_CACHE: dict = {}


def _nvfp4_guard(ckpt: str) -> None:
    """I checkpoint NVFP4 hanno scale di quantizzazione A BLOCCHI (es. (128,256)): la catena
    prequant di ltx-core supporta SOLO scale scalari per-tensore (fp8) e non esiste un percorso
    FP4 scaled-mm nello stack. Senza guardia il load moriva a metà col criptico "Unsupported
    scale shape (128, 256) for transformer_blocks…" (visto live 22/07). Qui: errore CHIARO e
    IMMEDIATO, prima di leggere 12GB. Check doppio: nome file + header (scale non-scalari)."""
    _name = os.path.basename(ckpt)
    _blocky = "nvfp4" in _name.lower()
    if not _blocky:
        try:
            from core import st_direct
            hdr, _ = st_direct.read_header(ckpt)
            _blocky = any(k.endswith("_scale") and len(v.get("shape") or []) > 0
                          for k, v in hdr.items() if k != "__metadata__")
        except Exception:                                   # noqa: BLE001 — header illeggibile: ci pensa il loader
            return
    if _blocky:
        raise RuntimeError(
            f"Checkpoint NVFP4 non supportato dal motore nativo: '{_name}' usa scale di "
            "quantizzazione a blocchi (FP4) che il loader attuale non sa caricare. "
            "Seleziona la versione fp8 (es. ltx-2.3-22b-dev-fp8.safetensors).")


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
            # GGUF → SEMPRE 'cpu' (percorso TESTATO): gli encoder (gemma/vae) su CPU, il transformer resta
            # comunque RESIDENTE e quantizzato (patch_stage_to_gguf forza OffloadMode.NONE → NON viene
            # streamato → non è questo a rallentare; la lentezza del GGUF è il dequant per-layer). offload
            # 'none' col GGUF dà latenti DEGENERI (video VERDE): mai usarlo. Vale per Q3/Q4/Q6 e ogni modello.
            print(f"[auto-offload] GGUF {os.path.basename(_gguf)[:24]} → cpu "
                  f"(transformer residente, encoder su CPU — percorso testato)", flush=True)
            return "cpu"
        # F6 unified-model-manager: peso del SOLO transformer dal DB canonico (misurato → header,
        # via transformer_mb che ESCLUDE gemma/vae/vocoder già alla fonte). Prima si usava il bundle
        # intero (37GB) corretto a mano con `file_gb-10`: need=55 su 32GB ⇒ sempre 'cpu' =
        # block-streaming = pinned memory ⇒ SEGFAULT col pagefile 16GB (13/07). L'euristica resta
        # come ULTIMO fallback se header e cache sono entrambi illeggibili.
        _file_gb = os.path.getsize(ckpt) / 1e9 if os.path.exists(ckpt) else 29.0
        try:
            from core import vram_estimate as _ve
            _tr_mb = _ve.transformer_mb(ckpt, fp8_storage=True)
            if _tr_mb > 0:
                tr_gb, _how = _tr_mb / 1024.0, "transformer_mb"
            else:                                             # header illeggibile → euristica storica
                tr_gb, _how = max(12.0, _file_gb - 10.0), "file-10"
        except Exception:                                     # noqa: BLE001
            tr_gb, _how = max(12.0, _file_gb - 10.0), "file-10"
        _ENC_GB = 0.0                                         # gemma/vae vanno su CPU comunque
        _WORK_GB = 8.0                                        # attivazioni temporali (picco denoise/decode)
        # M4 — budget attivazioni DINAMICO (env LTX_DYNAMIC_WORKBUDGET=1): scala col volume del clip
        # (width×height×frames) invece del fisso 8GB. Su clip piccoli tiene tutto residente più spesso; su clip
        # grandi alza il margine → meno spill invisibile in shared-VRAM. Default OFF (comportamento invariato).
        if os.environ.get("LTX_DYNAMIC_WORKBUDGET") == "1":
            try:
                _w = int(p.get("width", 768) or 768); _h = int(p.get("height", 512) or 512)
                _f = int(p.get("num_frames", p.get("frames", 97)) or 97)
                _ref = 768 * 512 * 97
                _WORK_GB = max(6.0, min(16.0, 8.0 * (_w * _h * _f) / _ref))
                print(f"[auto-offload] work-budget dinamico (M4) {_w}x{_h}x{_f} → {_WORK_GB:.1f}GB", flush=True)
            except Exception:                                # noqa: BLE001
                _WORK_GB = 8.0
        need = tr_gb + _ENC_GB + _WORK_GB                     # full-stack residente (solo fp8, no GGUF)
        mode = "none" if free_gb >= need else "cpu"
        # F6: planner combinatorio (plan_placement) come decisore DEFAULT (scelta utente 22/07):
        # transformer residente se entra (gli encoder girano comunque per-stage), componenti dal DB
        # canonico. Kill-switch: LTX_PLAN_PLACEMENT=0 → confronto diretto storico qui sopra.
        if os.environ.get("LTX_PLAN_PLACEMENT", "1") != "0":
            try:
                from core import vram_calibration as _vc
                _pl = _vc.plan_placement(int(tr_gb * 1024), "ltx2", int(free_gb * 1024),
                                         int(_WORK_GB * 1024))
                mode = "none" if "transformer" in _pl.get("resident", []) else "cpu"
                print(f"[plan] resident={_pl.get('resident')} offload={_pl.get('offload')} "
                      f"→ {mode}", flush=True)
            except Exception as _pe:                          # noqa: BLE001
                print(f"[plan] plan_placement non disponibile ({_pe}) → euristica diretta", flush=True)
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
    def _build(Klass, **kw):
        """Istanzia la pipeline passando il registry di parcheggio RAM (_PARK); se la classe non
        accetta `registry` (firma diversa), fallback senza — il job gira, solo senza parcheggio."""
        try:
            return Klass(registry=_PARK, **kw)
        except TypeError:
            return Klass(**kw)

    if ptype == "vinpaint":                                  # #59 latent-inpaint VIDEO (sottoclasse ic_lora)
        from workers.ltx_inpaint_pipeline import InpaintPipeline
        pipe = _build(InpaintPipeline,
                      distilled_checkpoint_path=ckpt, spatial_upsampler_path=UPSAMPLER,
                      gemma_root=GEMMA, loras=loras, quantization=quant, offload_mode=offload)
    elif ptype == "iclora":                                  # #24 ControlNet IC-LoRA (canny/depth/pose)
        from ltx_pipelines.ic_lora import ICLoraPipeline
        pipe = _build(ICLoraPipeline,
                      distilled_checkpoint_path=ckpt, spatial_upsampler_path=UPSAMPLER,
                      gemma_root=GEMMA, loras=loras, quantization=quant, offload_mode=offload)
    elif ptype == "retake":                                  # #24 ReTake: rigenera sezione [start,end] di un video
        from ltx_pipelines.retake import RetakePipeline
        pipe = _build(RetakePipeline,
                      checkpoint_path=ckpt, gemma_root=GEMMA, loras=loras,
                      quantization=quant, offload_mode=offload, distilled=True)
    else:
        Klass = A2VidPipelineTwoStage if ptype == "a2v" else TI2VidTwoStagesPipeline
        # DEV (non-distilled) sulla two-stage: SENZA la LoRA di distillazione lo stage distillato
        # sotto-denoisa → figure "dematerializzate" (nuvole di voxel). Se il checkpoint non è un
        # distilled e la LoRA ufficiale è su disco, la applichiamo; altrimenti warning esplicito.
        _dlora = []
        if "distilled" not in os.path.basename(ckpt).lower():
            if os.path.exists(DISTILLED_LORA):
                _dlora = [LoraPathStrengthAndSDOps(DISTILLED_LORA, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
                print(f"[dev] checkpoint non-distilled → applico la distilled-LoRA ufficiale "
                      f"({os.path.basename(DISTILLED_LORA)})", flush=True)
            else:
                print("[warn] checkpoint DEV senza LoRA di distillazione su disco: la two-stage "
                      "sotto-denoisa (video 'dematerializzato'). Scarica Lightricks/LTX-2.3 "
                      f"ltx-2.3-22b-distilled-lora-384-1.1.safetensors in {os.path.dirname(DISTILLED_LORA)}",
                      flush=True)
        pipe = _build(Klass,
                      checkpoint_path=ckpt, distilled_lora=_dlora, spatial_upsampler_path=UPSAMPLER,
                      gemma_root=GEMMA, loras=loras, quantization=quant, offload_mode=offload)
    # M5 (versione sicura) — LRU di pipeline configurabile (env LTX_VIDEO_PIPE_LRU, default 1 = comportamento
    # storico: 1 pipeline alla volta). Chi ha VRAM di scorta può alzarlo a 2 per tenere ti2v+a2v residenti ed
    # evitare il reload di gemma/VAE alternando tipo di generazione (talk↔normale). Il refcount PROFONDO dei
    # componenti condivisi (gemma/VAE tra pipeline diverse) richiede di iniettarli nei costruttori del package
    # esterno ltx_pipelines → va fatto con ispezione + test live (rimandato).
    _lru = max(1, int(os.environ.get("LTX_VIDEO_PIPE_LRU", "1") or 1))
    _PIPE_CACHE[key] = pipe
    while len(_PIPE_CACHE) > _lru:                            # evinci i più vecchi (dict = ordine d'inserimento)
        _oldk = next(iter(_PIPE_CACHE))
        if _oldk == key:
            break
        _old = _PIPE_CACHE.pop(_oldk)
        try:
            del _old
        except Exception:                                    # noqa: BLE001
            pass
    import gc as _gc
    _gc.collect()
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception:                                        # noqa: BLE001
        pass
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


def _blend_images(a: str, b: str, alpha: float, out: str) -> bool:
    """Miscela due immagini (a=ultimo frame, b=foto originale) al peso `alpha` → conditioning anti-drift.
    Ritorna True se scritta. Best-effort."""
    try:
        from PIL import Image
        ia = Image.open(a).convert("RGB")
        ib = Image.open(b).convert("RGB").resize(ia.size)
        Image.blend(ia, ib, max(0.0, min(1.0, float(alpha)))).save(out)
        return os.path.exists(out)
    except Exception:                                        # noqa: BLE001
        return False


def _composite_char_mask(out: str, char_mask_dir: str, original_frames_dir: str, *,
                         fps: float, mask_dilate: int = 12, mask_blur: float = 6.0) -> bool:
    """#102 Character Replacement — composite deterministico PER-FRAME.

    Dopo la generazione union-control (iclora) l'INTERO frame è rigenerato; qui ricomponiamo tenendo il
    frame ORIGINALE FUORI dalla maschera del personaggio (dilatata + feather) e il frame GENERATO DENTRO →
    solo il personaggio è sostituito, lo sfondo resta preservato al pixel. Il canny segue posa/movimento →
    il personaggio generato è già allineato → composite pulito.

    In-place: riscrive `out` (video), ri-muxando l'audio originale se presente. Best-effort: False → tiene
    l'output non composito (nessun peggioramento). Riusa `load_mask_tensor` (dilate+blur) per coerenza col
    ramo inpaint. NB: eseguito PRIMA di stego/watermark così la firma copre il risultato finale."""
    import glob as _g
    import subprocess as _sp
    import tempfile as _tf
    try:
        import numpy as _np
        from PIL import Image as _Im
    except Exception as _e:                                   # noqa: BLE001
        print(f"[replace_char] composite saltato (PIL/numpy assenti): {_e}", flush=True)
        return False

    masks = sorted(_g.glob(os.path.join(char_mask_dir, "*.png")))
    origs = sorted(_g.glob(os.path.join(original_frames_dir, "*.png")))
    if not masks or not origs:
        print(f"[replace_char] composite saltato: masks={len(masks)} origs={len(origs)}", flush=True)
        return False

    work = _tf.mkdtemp(prefix="rc_comp_")
    gen_dir = os.path.join(work, "gen")
    comp_dir = os.path.join(work, "comp")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    try:
        # 1) estrai i frame GENERATI dal video di output
        r = _sp.run(["ffmpeg", "-v", "error", "-i", out, "-start_number", "0",
                     os.path.join(gen_dir, "g_%06d.png")], capture_output=True, text=True, timeout=1800)
        gens = sorted(_g.glob(os.path.join(gen_dir, "*.png")))
        if r.returncode != 0 or not gens:
            print(f"[replace_char] composite: estrazione frame fallita ({r.stderr[:200]})", flush=True)
            return False

        n = min(len(gens), len(masks), len(origs))
        # dimensioni target = frame generato (l'output può differire dall'originale per padding/scale)
        gw, gh = _Im.open(gens[0]).size
        # 2) composite per-frame: fuori maschera = originale, dentro = generato (bordo feathered)
        from PIL import ImageFilter as _F
        for i in range(n):
            g = _Im.open(gens[i]).convert("RGB")
            o = _Im.open(origs[i]).convert("RGB").resize((gw, gh))
            m = _Im.open(masks[i]).convert("L").resize((gw, gh))
            if mask_dilate > 0:
                k = mask_dilate if mask_dilate % 2 == 1 else mask_dilate + 1
                m = m.filter(_F.MaxFilter(k))
            if mask_blur > 0:
                m = m.filter(_F.GaussianBlur(mask_blur))
            ga = _np.asarray(g, _np.float32)
            oa = _np.asarray(o, _np.float32)
            ma = (_np.asarray(m, _np.float32) / 255.0)[..., None]     # (H,W,1) in [0,1]
            comp = oa * (1.0 - ma) + ga * ma                          # dentro=gen, fuori=orig
            _Im.fromarray(_np.clip(comp, 0, 255).astype(_np.uint8)).save(
                os.path.join(comp_dir, f"c_{i:06d}.png"))

        # 3) riassembla + ri-mux audio dall'output originale (se presente)
        tmp_v = os.path.join(work, "composited.mp4")
        r2 = _sp.run(["ffmpeg", "-v", "error", "-framerate", str(fps), "-start_number", "0",
                      "-i", os.path.join(comp_dir, "c_%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-crf", "16", tmp_v, "-y"], capture_output=True, text=True, timeout=1800)
        if r2.returncode != 0 or not os.path.exists(tmp_v):
            print(f"[replace_char] composite: riassemblaggio fallito ({r2.stderr[:200]})", flush=True)
            return False

        has_audio = bool(_sp.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                                  "-show_entries", "stream=index", "-of", "csv=p=0", out],
                                 capture_output=True, text=True, timeout=30).stdout.strip())
        final = os.path.join(work, "final.mp4")
        if has_audio:
            r3 = _sp.run(["ffmpeg", "-v", "error", "-i", tmp_v, "-i", out, "-map", "0:v", "-map", "1:a",
                          "-c:v", "copy", "-c:a", "aac", "-shortest", final, "-y"],
                         capture_output=True, text=True, timeout=600)
            src = final if (r3.returncode == 0 and os.path.exists(final)) else tmp_v
        else:
            src = tmp_v

        import shutil as _sh
        _sh.copyfile(src, out)
        print(f"[replace_char] composite OK: {n} frame ricomposti (audio={'sì' if has_audio else 'no'})", flush=True)
        return True
    except Exception as _e:                                   # noqa: BLE001
        print(f"[replace_char] composite errore: {_e}", flush=True)
        return False
    finally:
        import shutil as _sh
        try:
            _sh.rmtree(work, ignore_errors=True)
        except Exception:                                    # noqa: BLE001
            pass


def _probe_frames(path: str) -> int:
    """Numero di frame video reali (per il RESUME: un chunk valido ha ~cf frame). 0 se illeggibile."""
    try:
        import json as _j
        import subprocess
        r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames", "-of", "json", path],
                           capture_output=True, text=True, timeout=60)
        return int(_j.loads(r.stdout)["streams"][0]["nb_read_frames"])
    except Exception:                                        # noqa: BLE001
        return 0


def _xfade_concat(parts: list, out: str, fps: float, overlap_s: float) -> bool:
    """Concat con CROSSFADE video progressivo tra clip (join morbidi sui video lunghi). Usato SOLO per clip
    SENZA audio (l'a2v resta concat netto per non desincronizzare il lip-sync). Fallback False → concat netto."""
    try:
        import json as _j
        import subprocess
        durs = []
        for pt in parts:
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", pt],
                               capture_output=True, text=True, timeout=30)
            durs.append(float(_j.loads(r.stdout)["format"]["duration"]))
        ov = max(0.05, min(float(overlap_s), min(durs) * 0.5))
        inputs = []
        for pt in parts:
            inputs += ["-i", pt]
        vf = ""
        prev = "0:v"
        offset = durs[0] - ov
        for k in range(1, len(parts)):
            outv = f"v{k}"
            vf += f"[{prev}][{k}:v]xfade=transition=fade:duration={ov:.3f}:offset={offset:.3f}[{outv}];"
            prev = outv
            offset += durs[k] - ov
        vf = vf.rstrip(";")
        cmd = ["ffmpeg", "-v", "error", *inputs, "-filter_complex", vf, "-map", f"[{prev}]",
               "-r", str(fps), out, "-y"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0
    except Exception:                                        # noqa: BLE001
        return False


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
    orig_img = prev_img                                       # #39 anti-drift: la foto ORIGINALE per il re-anchor
    _anchor_every = int(p.get("anchor_every", 0) or 0)        # ogni N chunk riparte dalla foto (azzera la deriva)
    _ref_anchor = max(0.0, min(1.0, float(p.get("ref_anchor_strength", 0.0) or 0.0)))  # blend foto→conditioning
    _resume = bool(p.get("resume", False))                    # #38 riusa i chunk già prodotti (ripresa dopo crash)
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
        # #39 ANTI-DRIFT: frena il collasso autoregressivo re-ancorando alla foto originale.
        if i > 0 and orig_img and os.path.exists(str(orig_img)):
            if _anchor_every and (i % _anchor_every == 0):    # hard reset: riparte dalla foto (piccolo salto)
                cp["image_path"] = orig_img
                print(f"[chunk] {i}: re-anchor HARD alla foto originale", flush=True)
            elif _ref_anchor > 0 and prev_img and os.path.exists(str(prev_img)):
                _bl = f"{base}_chunk{i}_anchor.png"           # blend morbido ultimo-frame↔foto come conditioning
                if _blend_images(prev_img, orig_img, _ref_anchor, _bl):
                    cp["image_path"] = _bl
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
        # #38 RESUME: se il chunk esiste già ed è valido (n. frame ~ atteso) → riusalo (ripresa dopo crash).
        if _resume and os.path.exists(cp["output"]) and _probe_frames(cp["output"]) >= cf - 2:
            print(f"[chunk] {i} già valido → riuso (resume)", flush=True)
            last = f"{base}_chunk{i}_last.png"
            subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.12", "-i", cp["output"],
                            "-update", "1", "-q:v", "2", last, "-y"], check=False)
            if os.path.exists(last):
                prev_img = last                              # continuità dal chunk riusato
            parts.append(cp["output"])
            continue
        print(f"[chunk] {i + 1}/{n} ({cf} frame, audio_start={cp.get('audio_start_time', 0)})", flush=True)
        try:
            run_generation(cp)                               # pipe in cache dal 2° chunk → veloce
        except GenerationCancelled:                          # annullato dall'utente: MAI retry, propaga
            raise
        except Exception as _e:                              # es. ultimo chunk con audio più corto dei frame
            print(f"[chunk] {i} fallito ({type(_e).__name__}: {_e}) → 1 RETRY dopo pulizia VRAM", flush=True)
            import gc as _gc2
            _gc2.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                run_generation(cp)                           # RETRY: recupera la durata piena (era troncata muta)
            except GenerationCancelled:                      # annullato: niente video ridotto "muto"
                raise
            except Exception as _e2:                         # noqa: BLE001
                print(f"[chunk] {i} fallito anche al retry ({type(_e2).__name__}: {_e2})", flush=True)
                if i == 0:                                   # il PRIMO chunk fallito = nessun output: NON
                    raise                                    # inghiottire, propaga il vero errore all'utente
                break                                        # chunk successivo perso → durata ridotta (avvisata sotto)
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
    # CROSSFADE (join morbidi) solo se richiesto (env) e SENZA audio: l'a2v (talk/canto) resta concat netto
    # per non desincronizzare il lip-sync. Fallback automatico al concat netto se il filtergraph fallisce.
    _ov = int(p.get("overlap_frames", 0) or 0)
    _xf = (os.environ.get("LTX_CHUNK_XFADE") == "1" and _ov > 0 and len(parts) > 1
           and not p.get("audio_path")
           and _xfade_concat(parts, p["output"], fps, _ov / max(fps, 1)))
    if not _xf:
        lst = f"{base}_list.txt"
        with open(lst, "w", encoding="utf-8") as fh:
            for pt in parts:
                fh.write(f"file '{pt}'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", p["output"], "-y"], check=False)
    # DURATA REALE vs richiesta: se qualche chunk è fallito, la durata è RIDOTTA. NON silenzioso (l'utente
    # vedeva "10s→5s" senza spiegazione): avvisa con i secondi effettivi prodotti. Il primo chunk (i==0) fa
    # già raise; qui gestiamo i chunk SUCCESSIVI persi.
    _fps = float(p.get("fps", 25.0))
    _frames_done = sum(min(chunk, total - k * chunk) for k in range(len(parts)))
    if len(parts) < n:
        _sr, _st = _frames_done / max(_fps, 1), total / max(_fps, 1)
        _stage(f"⚠ Video RIDOTTO: {len(parts)}/{n} segmenti → {_sr:.1f}s invece di {_st:.1f}s "
               f"(un segmento è fallito — vedi log sopra).")
        print(f"[chunk] ATTENZIONE: durata {_sr:.1f}s invece di {_st:.1f}s ({len(parts)}/{n} chunk riusciti)",
              flush=True)
    print(f"[chunk] concat → {p['output']} ({len(parts)}/{n} chunk, ~{_frames_done / max(_fps,1):.1f}s)", flush=True)
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


def _has_audio_track(path: str) -> bool:
    """Il file ha davvero una traccia audio? (ffprobe). Usato per accorgersi che un a2v è uscito muto."""
    import subprocess
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=codec_type", "-of", "csv=p=0", path],
                           capture_output=True, text=True, timeout=30)
        return "audio" in (r.stdout or "")
    except Exception:                                        # noqa: BLE001
        return True                                          # in dubbio: non toccare il file


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


class GenerationCancelled(Exception):
    """Job annullato dall'utente (server /cancel o stop() GUI via server). Cancel COOPERATIVO:
    controllato tra gli step del denoise (tqdm patch) e ai confini di fase (_stage) — NON è
    istantaneo durante il decode VAE (scatta alla fase successiva)."""


import threading as _threading                                 # noqa: E402
_CANCEL = _threading.Event()


def request_cancel() -> None:
    """Chiede l'annullamento del job in corso (chiamato dal server :8190 su POST /cancel)."""
    _CANCEL.set()


def clear_cancel() -> None:
    _CANCEL.clear()


def _check_cancel() -> None:
    if _CANCEL.is_set():
        raise GenerationCancelled("annullato dall'utente")


def _stage(msg: str) -> None:
    """Scrive la FASE corrente della pipeline su core.preview_file → il dialog anteprima la mostra (richiesta
    utente: sapere cosa sta facendo la GPU). Anche su stdout per i log. Best-effort (mai rompe la gen).
    È anche un punto di check del cancel cooperativo (abort ai confini di fase)."""
    _check_cancel()
    print(f"[stage] {msg}", flush=True)
    try:
        from core import preview_file as _pf
        _pf.write_message(msg)
    except Exception:                                          # noqa: BLE001
        pass


def _install_step_progress() -> None:
    """Intercetta il tqdm del sampler LTX (ltx_pipelines.utils.samplers) per emettere lo STEP del denoise su
    core.preview_file → barra 'step n/N' nel dialog anteprima. Best-effort, una volta per processo.
    Qui vive anche il check del cancel cooperativo per-step."""
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
                _check_cancel()                                # abort tra gli step del denoise
                return r
        _S.tqdm = _PTqdm
        _S._ltx_tqdm_patched = True
        print("[stage] progresso per-step attivo (denoise → dialog anteprima)", flush=True)
    except Exception as _e:                                    # noqa: BLE001
        print(f"[stage] progresso per-step non attivo: {_e}", flush=True)


def run_generation(p: dict) -> str:
    """Genera un video dai params. Riusabile da worker one-shot e da server persistente.
    NB cancel: il clear del flag è responsabilità del CHIAMANTE top-level (main() / ltx2_server)
    — qui no, perché i video lunghi rientrano ricorsivamente per-chunk e un clear annullerebbe
    un cancel arrivato tra un chunk e l'altro."""
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
    # ── GGUF VIDEO: RIMOSSO DAL PRODOTTO (13/07) ─────────────────────────────────────────────
    # Il loader GGUF dequantizza in Python A OGNI forward (1660 Linear × step) → ~500 s/step contro
    # i ~4 s/step dell'fp8: oltre un'ORA per clip. E il Q6_K (16.6GB) residente andava in OOM alle
    # risoluzioni normali (704x1280x161f) → cartelle piene di .json e ZERO mp4 (misurato: 13 job,
    # 0 video). Non esisteva uno scenario in cui convenisse. Il video usa SEMPRE il checkpoint fp8.
    # Qui non si "ignora in silenzio": un GGUF che arriva viene SCARTATO con un avviso esplicito,
    # perché i progetti/gui_state vecchi possono ancora contenerne uno selezionato.
    # (I GGUF delle IMMAGINI — FLUX Fill per l'inpaint — non c'entrano e restano.)
    _stale = p.pop("gguf_transformer", None)
    if str(p.get("model", "")).lower().endswith(".gguf"):
        _stale = _stale or p.pop("model")
        p.pop("model", None)
    if _stale:
        print(f"[gguf] IGNORATO '{os.path.basename(str(_stale))}': i GGUF video sono stati rimossi "
              f"(≈500 s/step e OOM). Uso il checkpoint fp8.", flush=True)
        p.pop("offload", None)                              # l'offload 'cpu' era imposto DAL GGUF
    # Con l'offload attivo (Gemma CPU + transformer per-blocco) il picco è ~31GB su 32: lasciamo
    # che PyTorch usi tutta la VRAM + sysmem-fallback per il piccolo surplus (lento ma genera).
    # (Il memory_fraction era solo per la diagnosi VRAM, ora rimosso.)
    if torch.cuda.is_available() and os.environ.get("LTX_VRAM_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["LTX_VRAM_FRACTION"]))
    loras = []
    for lo in p.get("loras", []):
        path = lo.get("name") or lo.get("path")             # robusto: alcuni chiamanti usano "path"
        if not os.path.isabs(path):
            path = os.path.join(LORA_DIR, path)
        loras.append(LoraPathStrengthAndSDOps(path, float(lo.get("strength", 1.0)),
                                              LTXV_LORA_COMFY_RENAMING_MAP))
    ckpt = _resolve_ckpt(p.get("model"))
    _nvfp4_guard(ckpt)                                       # NVFP4 non caricabile: errore chiaro SUBITO
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
        # #102 GUARDIA: control_video presente ma NESSUN IC-LoRA risolto → il ControlNet/replace_char verrebbe
        # ignorato IN SILENZIO (degrado a ti2v). Il GUI fa già hard-error sul peso mancante; qui almeno lo urlo.
        if p.get("control_video"):
            print(f"[warn] control_video presente ma IC-LoRA non risolto (edit_lora={p.get('edit_lora')!r}, "
                  f"control_type={p.get('control_type')!r}) → guida IGNORATA, degrado a {ptype}. "
                  "Scarica il peso IC-LoRA.", flush=True)
    _off = _auto_offload(p, ckpt)                            # GGUF→cpu sempre; fp8→none se c'entra tutto in GPU
    # M-park: la pipeline del job è ora NOTA → i pesi parcheggiati in RAM che questo job NON userà
    # vengono rimossi (regola di drop richiesta); quelli comuni (Gemma/VAE/upsampler/ckpt) restano.
    try:
        _keep = [ckpt, GEMMA, UPSAMPLER, DISTILLED_LORA, *IC_LORA.values(),
                 *[lo.path for lo in loras]]
        if _edit:
            _keep.append(_edit)
        _gone = _PARK.drop_except(_keep)
        if _gone:
            print(f"[park] drop pipeline-aware: liberati {_gone:.0f}MB di RAM", flush=True)
    except Exception as _pe:                                 # noqa: BLE001
        print(f"[park] drop fallito (ignoro): {_pe}", flush=True)
    _stage(f"Caricamento modello ({os.path.basename(ckpt)[:28]}, offload={_off})…")
    pipe = _get_pipe(ckpt, qmode, _off, loras, ptype=ptype)
    print(f"[load] {time.time() - t0:.0f}s (model={os.path.basename(ckpt)} quant={qmode} pipe={ptype})", flush=True)
    if torch.cuda.is_available():
        print(f"[vram] post-load: {torch.cuda.memory_allocated()/1e9:.1f}GB alloc / "
              f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved", flush=True)
    try:                                                     # F3: ledger live — il job video dichiara il suo modello
        from core import vram_ledger as _vl
        from core import vram_estimate as _vest
        _est_mb, _ = _vest.estimate_mb(ckpt, fp8_storage=True, working_mb=0)
        _vl.register(f"video:{os.path.basename(ckpt)}", _est_mb or os.path.getsize(ckpt) / 2**20,
                     category="video", state="vram", path=ckpt)
    except Exception:                                        # noqa: BLE001
        pass

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
    _tc = _tiling_config(p)                                   # decode VAE a chunk (streaming); tile piccoli se lowmem
    tg = time.time()
    _stage(f"Generazione video — {int(p.get('steps', 30))} step (encode prompt + denoise)…")
    if ptype == "vinpaint":                                  # #59 inpaint: guide=video orig + attention_mask + CFG
        import glob as _glob
        from workers.ltx_inpaint_cond import load_mask_tensor
        mks = sorted(_glob.glob(os.path.join(p["inpaint_masks_dir"], "*.png")))
        nf = min(nf, len(mks))
        masks_t = load_mask_tensor(mks[:nf], int(p.get("height", 320)), int(p.get("width", 512)),
                                   dilate=int(p.get("mask_dilate", 24)), blur=float(p.get("mask_blur", 6.0)))
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
            tiling_config=_tc, enhance_prompt=bool(p.get("enhance", False)))
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
            tiling_config=_tc)
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
            enhance_prompt=bool(p.get("enhance", False)), tiling_config=_tc,
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
            tiling_config=_tc, enhance_prompt=bool(p.get("enhance", False)))
    print(f"[gen] {time.time() - tg:.0f}s ({nf} frames @ {fps}fps = {nf / fps:.1f}s video, {ptype})", flush=True)
    out = p["output"]
    _stage("Decodifica VAE e salvataggio video…")
    # LTX-2 restituisce SEMPRE una traccia audio anche per i2v/ti2v/iclora: senza condizionamento audio è
    # RUMORE non condizionato ("rumore grigio"). La scartiamo (video muto) tranne quando l'audio è davvero
    # richiesto — a2v (canto/parlato) o audio_path fornito. Così i video generati/editati NON hanno più rumore.
    # native_audio: l'utente CHIEDE l'audio nativo (parlato/suoni dal prompt, es. «says "hello world"»). LTX-2.3
    # genera audio+video insieme (ti2v con audio_guider) → tenerlo dà voce + lip-sync. Senza il flag si scarta
    # come prima (evita il rumore non-condizionato sui prompt non-parlato).
    _audio_requested = (ptype == "a2v") or bool(p.get("audio_path")) or bool(p.get("native_audio"))
    if not _audio_requested and audio is not None:
        print("[audio] nessun audio richiesto → scarto la traccia LTX-2 (rumore non condizionato)", flush=True)
        audio = None
    elif bool(p.get("native_audio")) and audio is not None:
        print("[audio] native_audio ON → TENGO la traccia LTX-2 (parlato/suoni dal prompt + lip-sync)", flush=True)
    # GUARDIA anti-NaN: pesi corrotti (es. GGUF con download incompleto) producono frame NaN che il
    # VAE/encoder trasforma in VIDEO VERDE UNIFORME, consegnato come "successo". Meglio un errore chiaro.
    def _nan_guard(chunks):
        for _c in chunks:
            if torch.is_tensor(_c) and (torch.isnan(_c).any() or torch.isinf(_c).any()):
                raise RuntimeError(
                    "Frame NaN/inf dal denoise: pesi del modello corrotti o incompatibili "
                    "(es. GGUF con download incompleto). Verifica/riscarica il checkpoint selezionato.")
            yield _c
    if torch.is_tensor(video):
        video = next(_nan_guard([video]))
    else:
        video = _nan_guard(video)
    encode_video(video=video, fps=fps, audio=audio, output_path=out,
                 video_chunks_number=get_video_chunks_number(nf, _tc))
    # RETE DI SICUREZZA: se l'audio era RICHIESTO (a2v: canto/lip-sync) ma il file è uscito muto — encoder
    # in fallback, mux fallito, traccia vuota — rimuxa dal file audio ORIGINALE, sulla stessa finestra usata
    # per condizionare la generazione (audio_start_time + durata del chunk) → resta in sync anche nei video
    # lunghi. Un lip-sync senza audio non è un output valido: meglio ricostruirlo che consegnarlo muto.
    if _audio_requested and p.get("audio_path") and os.path.exists(out) and not _has_audio_track(out):
        import subprocess                                     # non è importato a livello modulo
        _a0 = float(p.get("audio_start_time", 0.0) or 0.0)
        _dur = nf / float(fps)
        _mux = out + ".fixaudio.mp4"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out, "-ss", f"{_a0:.3f}",
                            "-t", f"{_dur:.3f}", "-i", _ensure_stereo(p["audio_path"]),
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                            "-shortest", _mux], capture_output=True)
        if r.returncode == 0 and os.path.exists(_mux) and os.path.getsize(_mux) > 0:
            os.replace(_mux, out)
            print(f"[audio] video uscito muto → audio rimuxato dal file sorgente "
                  f"(offset {_a0:.2f}s, durata {_dur:.2f}s)", flush=True)
        else:
            print(f"[audio] !! rimux di sicurezza FALLITO: {r.stderr[-300:] if r.stderr else '?'}", flush=True)
    # #102 Character Replacement: se il ramo iclora ha una maschera personaggio (SAM2 tracking), ricomponi
    # PER-FRAME — solo il personaggio è sostituito, lo sfondo resta pixel-perfect dall'originale. Deterministico
    # e sicuro senza test; l'alternativa attention_mask (cam) resta env-gated (REPLACE_CHAR_ATTN_MASK, sopra).
    if ptype == "iclora" and p.get("char_mask_dir") and p.get("original_frames_dir"):
        _stage("Ricomposizione personaggio (maschera SAM2)…")
        _composite_char_mask(out, p["char_mask_dir"], p["original_frames_dir"], fps=fps,
                             mask_dilate=int(p.get("mask_dilate", 12)),
                             mask_blur=float(p.get("mask_blur", 6.0)))
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
    try:                                                     # F3: fine job → la VRAM del modello è rilasciata
        from core import vram_ledger as _vl
        _vl.unregister(f"video:{os.path.basename(ckpt)}")
    except Exception:                                        # noqa: BLE001
        pass
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
    clear_cancel()                                           # top-level: nessun cancel residuo
    _free_image_server()
    print("OK:" + run_generation(p), flush=True)


if __name__ == "__main__":
    main()
