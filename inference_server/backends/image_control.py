"""ControlNet e Redux nativi per FLUX.1 e Z-Image (modulo job «image_control»).

Modalità (`params["mode"]`):

| mode             | pipeline diffusers          | pesi extra                                                       |
|------------------|-----------------------------|------------------------------------------------------------------|
| flux_cn          | FluxControlNetPipeline      | ControlNet Union Pro 2.0 (Shakker, 4.3 GB): canny/depth/pose/gray |
| flux_cn_inpaint  | FluxControlNetPipeline      | ControlNet Inpainting Beta (AlimamaCreative, 4.3 GB)              |
| flux_redux       | FluxPipeline (embeds)       | Redux 0.13 GB + SigLIP so400m-384 (3.5 GB file, solo la torre visiva)|
| zimage_cn        | ZImageControlNetPipeline    | Z-Image Turbo Fun ControlNet Union 2.1 8 step (6.7 GB)            |

La pipeline BASE (FLUX.1 / Z-Image) è quella di `image_flux` (stessa `register`/`mm.ensure`, stessa chiave di residenza:
crea → ControlNet sullo stesso file NON ricarica il modello); qui se ne deriva una pipeline ControlNet che CONDIVIDE i
componenti (nessuna copia di pesi). Solo il ControlNet (e per Redux la torre SigLIP) si carica a parte e si scarica a fine job.

Nessun ComfyUI: i pesi dei ControlNet sono in formato diffusers (FLUX) o VideoX-Fun (Z-Image, mappato dal loader di diffusers).
Licenze: ControlNet FLUX e Redux = FLUX.1 [dev] non commerciale; Z-Image ControlNet = Apache-2.0.
"""
from __future__ import annotations

import os
from pathlib import Path

from core.model_paths import p as _cat_p                    # stdlib-only, mai solleva

CN_FLUX_UNION = _cat_p("flux-cn-union-pro2", r"E:\AI\Models\ControlNet\Flux\shakker_union_pro_v2.safetensors")
CN_FLUX_INPAINT = _cat_p("flux-cn-inpaint-alimama", r"E:\AI\Models\ControlNet\Flux\alimama_flux_inpainting.safetensors")
CN_ZIMAGE = _cat_p("zimage-cn-union21", r"E:\AI\Models\ControlNet\Z-Image\Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors")
REDUX_FILE = _cat_p("flux-redux", r"E:\AI\Models\Checkpoints\Flux\FLUX_DEV_Redux.safetensors")
SIGLIP_FILE = _cat_p("siglip-so400m-384", r"E:\AI\Models\VisionEncoders\model.safetensors")

MODES = ("flux_cn", "flux_cn_inpaint", "flux_redux", "zimage_cn")
_BASE_ARCH = {"flux_cn": "flux", "flux_cn_inpaint": "flux", "flux_redux": "flux", "zimage_cn": "zimage"}
_EXTRA_MB = {"flux_cn": 4400, "flux_cn_inpaint": 4400, "flux_redux": 1200, "zimage_cn": 6800}
_ALIMAMA_EXTRA_CH = 4                                        # maschera impacchettata 2x2: 1 canale · 4


def _log(m: str) -> None:
    print(f"[image-control] {m}", flush=True)


def _need(path: str, what: str, gb: float) -> None:
    if not os.path.isfile(path):
        raise RuntimeError(f"Manca {what} ({Path(path).name}, ~{gb:g} GB): scaricalo dal tab «Pesi modelli».")


# ── caricamento dei ControlNet ───────────────────────────────────────────────────────────────────────────
def load_flux_controlnet(path: str, dtype=None):
    """FluxControlNetModel dal file (struttura dedotta dalle chiavi: blocchi doppi/singoli, guidance, mode embed).
    Con 68 canali in `controlnet_x_embedder` (64 latenti + 4 maschera) è l'Inpainting di AlimamaCreative."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import FluxControlNetModel
    from safetensors.torch import load_file
    dtype = dtype or torch.bfloat16
    sd = load_file(path)
    n_dbl = len({k.split(".")[1] for k in sd if k.startswith("transformer_blocks.")})
    n_sgl = len({k.split(".")[1] for k in sd if k.startswith("single_transformer_blocks.")})
    guid = "time_text_embed.guidance_embedder.linear_1.weight" in sd
    n_mode = sd["controlnet_mode_embedder.weight"].shape[0] if "controlnet_mode_embedder.weight" in sd else None
    extra = int(sd["controlnet_x_embedder.weight"].shape[1]) - int(sd["x_embedder.weight"].shape[1])
    with init_empty_weights():
        cn = FluxControlNetModel(num_layers=n_dbl, num_single_layers=n_sgl, guidance_embeds=guid, num_mode=n_mode)
        if extra:
            cn.controlnet_x_embedder = torch.nn.Linear(int(sd["x_embedder.weight"].shape[1]) + extra,
                                                       cn.controlnet_x_embedder.out_features)
            cn.__class__ = _alimama_class()
    cn.load_state_dict(sd, strict=True, assign=True)
    del sd
    cn.extra_channels = extra
    return cn.to(dtype).eval()


def _alimama_class():
    from diffusers import FluxControlNetModel

    class AlimamaFluxControlNet(FluxControlNetModel):
        """FluxControlNetModel con 4 canali extra di condizione (maschera impacchettata) accodati al `controlnet_cond`
        dei latenti dell'immagine mascherata. La maschera la imposta il chiamante in `self.extra_cond` (B, seq, 4)."""
        extra_cond = None

        def forward(self, hidden_states, controlnet_cond, *args, **kwargs):
            import torch
            if self.extra_cond is None:
                raise RuntimeError("AlimamaFluxControlNet: maschera mancante (extra_cond)")
            ex = self.extra_cond.to(device=controlnet_cond.device, dtype=controlnet_cond.dtype)
            if ex.shape[0] != controlnet_cond.shape[0]:
                ex = ex.expand(controlnet_cond.shape[0], -1, -1)
            return super().forward(hidden_states, torch.cat([controlnet_cond, ex], dim=-1), *args, **kwargs)

    return AlimamaFluxControlNet


ZIMAGE_CN_PLACES = lambda n: [2 * i for i in range(n)]       # noqa: E731  (VideoX-Fun: un blocco di controllo ogni 2)


def load_zimage_controlnet(path: str, dtype=None):
    """ZImageControlNetModel (Union 2.x) dal file VideoX-Fun: config dedotta dalle chiavi."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import ZImageControlNetModel
    from safetensors.torch import load_file
    dtype = dtype or torch.bfloat16
    sd = load_file(path)
    n_ctl = len({k.split(".")[1] for k in sd if k.startswith("control_layers.")})
    n_ref = len({k.split(".")[1] for k in sd if k.startswith("control_noise_refiner.")})
    emb = next(k for k in sd if k.startswith("control_all_x_embedder.") and k.endswith(".weight"))
    in_dim = int(sd[emb].shape[1]) // 4                       # patch 2·2 · f_patch 1
    dim = int(sd[emb].shape[0])
    with_refiner = n_ref > 0 and "control_noise_refiner.0.before_proj.weight" in sd
    cfg = dict(control_layers_places=ZIMAGE_CN_PLACES(n_ctl),
               control_refiner_layers_places=list(range(n_ref)) if with_refiner else None,
               control_in_dim=in_dim,
               add_control_noise_refiner="control_noise_refiner" if with_refiner else "control_layers",
               dim=dim, n_refiner_layers=max(n_ref, 2), n_heads=dim // 128, n_kv_heads=dim // 128)
    with init_empty_weights():
        cn = ZImageControlNetModel(**cfg)
    if not with_refiner:                                      # 2.0: i blocchi del refiner non fanno parte del file
        sd = {k: v for k, v in sd.items() if not k.startswith("control_noise_refiner.")}
    missing, unexpected = cn.load_state_dict(sd, strict=False, assign=True)
    # i rami condivisi col transformer (t_embedder, noise_refiner...) sono None finche' non c'e' from_transformer
    missing = [k for k in missing if not k.startswith(("t_embedder", "all_x_embedder", "cap_embedder", "rope_embedder",
                                                         "noise_refiner", "context_refiner", "x_pad", "cap_pad"))]
    if missing or unexpected:
        raise RuntimeError(f"ControlNet Z-Image: chiavi mancanti {missing[:5]} / inattese {list(unexpected)[:5]}")
    del sd
    return cn.to(dtype).eval()


# ── utilità comuni ───────────────────────────────────────────────────────────────────────────────────────
def _derive(pipe, cls, **extra):
    """Pipeline `cls` che CONDIVIDE i componenti di `pipe` (nessuna copia di pesi) più i moduli in `extra`."""
    import inspect
    names = [n for n in inspect.signature(cls.__init__).parameters if n != "self"]
    comps = {n: getattr(pipe, n, None) for n in names if n not in extra}
    return cls(**comps, **extra)


def _pack_latents(x):
    """Come FluxPipeline._pack_latents: (B,C,H,W) → (B, H/2·W/2, C·4)."""
    b, c, h, w = x.shape
    return x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def alimama_inputs(image, mask, width: int, height: int):
    """(immagine di controllo PIL con l'area da rifare NERA, maschera impacchettata (1, seq, 4)): la convenzione di
    AlimamaCreative (`prepare_image_with_mask`): maschera >0.5 = da rigenerare, resa nera nell'immagine; al ControlNet va
    1 − maschera alla risoluzione dei latenti (H/8), poi impacchettata 2x2 (1 canale → 4)."""
    import numpy as np
    import torch
    from PIL import Image
    img = image.convert("RGB").resize((width, height), Image.LANCZOS)
    m = mask.convert("L").resize((width, height), Image.NEAREST)
    mb = np.asarray(m, dtype=np.uint8) > 127
    arr = np.asarray(img).copy()
    arr[mb] = 0
    lat = m.resize((width // 8, height // 8), Image.NEAREST)
    keep = 1.0 - (torch.from_numpy((np.asarray(lat, dtype=np.uint8) > 127).astype("float32"))[None, None])
    return Image.fromarray(arr), _pack_latents(keep)


def _size_from_image(img, params) -> tuple[int, int]:
    """W/H multipli di 16, lato lungo ≤ max_side (come image_flux): per l'inpaint segue la foto."""
    cap = int(params.get("max_side", 1536))
    iw, ih = img.size
    s = min(1.0, cap / max(iw, ih))
    return max(256, round(iw * s / 16) * 16), max(256, round(ih * s / 16) * 16)


# ── Redux ────────────────────────────────────────────────────────────────────────────────────────────────
def load_siglip_vision(path: str = "", dtype=None):
    """Torre visiva di SigLIP so400m-patch14-384 dal file completo (testo+visione) di google/siglip-so400m-patch14-384."""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import SiglipVisionConfig, SiglipVisionModel
    path = path or SIGLIP_FILE
    _need(path, "l'encoder visivo SigLIP so400m-384 per Redux", 3.5)
    dtype = dtype or torch.bfloat16
    cfg = SiglipVisionConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16,
                             image_size=384, patch_size=14, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6)
    with init_empty_weights():
        m = SiglipVisionModel(cfg)
    sd = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.startswith("vision_model."):
                sd[k] = f.get_tensor(k)
    m.load_state_dict(sd, strict=True, assign=True)
    return m.to(dtype).eval()


def load_redux_embedder(path: str = "", dtype=None):
    import torch
    from accelerate import init_empty_weights
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
    from safetensors.torch import load_file
    path = path or REDUX_FILE
    _need(path, "FLUX Redux", 0.13)
    sd = load_file(path)
    with init_empty_weights():
        enc = ReduxImageEncoder(redux_dim=int(sd["redux_up.weight"].shape[1]),
                                txt_in_features=int(sd["redux_down.weight"].shape[0]))
    enc.load_state_dict(sd, strict=True, assign=True)
    return enc.to(dtype or torch.bfloat16).eval()


def siglip_preprocess(img, dtype, device):
    """SiglipImageProcessor di Redux: ridimensiona a 384×384 (bicubico, senza mantenere il rapporto), /255, (x−.5)/.5."""
    import numpy as np
    import torch
    from PIL import Image
    a = np.asarray(img.convert("RGB").resize((384, 384), Image.BICUBIC), dtype="float32") / 255.0
    t = torch.from_numpy((a - 0.5) / 0.5).permute(2, 0, 1)[None]
    return t.to(device=device, dtype=dtype)


def redux_embeds(vision, embedder, images, device, strength: float = 1.0):
    """Immagini di riferimento → token di stile (1, 729·N, 4096): SigLIP (last_hidden_state) → Redux; le N immagini
    si concatenano nella sequenza."""
    import torch
    outs = []
    with torch.no_grad():
        for im in images:
            x = siglip_preprocess(im, next(vision.parameters()).dtype, device)
            h = vision(pixel_values=x).last_hidden_state
            outs.append(embedder(h).image_embeds * float(strength))
    return torch.cat(outs, dim=1)


def redux_prompt_embeds(pipe, image_embeds, prompt: str, device):
    """Token di testo + token di stile. Prompt vuoto → testo a zero (come FluxPriorReduxPipeline senza encoder di testo);
    con un prompt → T5/CLIP veri della pipeline FLUX (come il nodo StyleModelApply di ComfyUI)."""
    import torch
    if (prompt or "").strip():
        te, pooled, _ids = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, device=device, num_images_per_prompt=1,
                                              max_sequence_length=512)
    else:
        te = torch.zeros((1, 512, 4096), device=device, dtype=image_embeds.dtype)
        pooled = torch.zeros((1, 768), device=device, dtype=image_embeds.dtype)
    return torch.cat([te.to(image_embeds.dtype), image_embeds], dim=1), pooled.to(image_embeds.dtype)


# ── placement del ControlNet accanto alla pipeline base ──────────────────────────────────────────────────
def _attach_cn(cn, where: str):
    """Mette il ControlNet dove lavora la base: gpu/sticky → su cuda; offload → hook accelerate (sale al primo forward)."""
    import torch
    if not torch.cuda.is_available():
        return None
    if where in ("gpu", "sticky"):
        cn.to("cuda")
        return None
    from accelerate import cpu_offload_with_hook
    _m, hook = cpu_offload_with_hook(cn, "cuda")
    return hook


def _detach_cn(cn, hook) -> None:
    try:
        if hook is not None:
            hook.offload()
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(cn, recurse=True)
        cn.to("cpu")
    except Exception:                                        # noqa: BLE001
        pass


# ── esecuzione ───────────────────────────────────────────────────────────────────────────────────────────
def _common_kw(params, arch, W, H, defaults):
    import torch
    steps = int(params.get("steps") or 0) or defaults[0]
    guid = float(params["guidance"]) if params.get("guidance") is not None else defaults[1]
    _gd = params.get("gen_device") or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=_gd).manual_seed(int(params.get("seed", 0)))
    kw = dict(prompt=params.get("prompt", ""), num_inference_steps=steps, guidance_scale=guid, generator=g,
              height=H, width=W)
    return kw


def build_job(pipe, params: dict, cn=None, extra: dict | None = None):
    """Costruisce (pipeline derivata, kwargs di chiamata, immagine di riferimento per la ricomposizione|None).
    Separato da `run` così i test CPU lo provano con pipeline minuscole, senza GPU."""
    from PIL import Image
    mode = params["mode"]
    arch = _BASE_ARCH[mode]
    from core import control_maps as cm
    model = str(params.get("model") or "")
    if mode == "flux_redux":
        raise ValueError("flux_redux non passa da build_job")
    from inference_server.backends import image_flux as F
    _ast, _agd, _h = F._auto_settings(arch, model)
    keep_for = None
    if mode == "flux_cn_inpaint":
        from diffusers import FluxControlNetPipeline
        img = Image.open(params["image_path"]).convert("RGB")
        mk = Image.open(params["mask_path"]).convert("L")
        W, H = _size_from_image(img, params)
        ctl, packed = alimama_inputs(img, mk, W, H)
        cn.extra_cond = packed
        cpipe = _derive(pipe, FluxControlNetPipeline, controlnet=cn)
        kw = _common_kw(params, arch, W, H, (28, 3.5))
        kw.update(control_image=ctl, controlnet_conditioning_scale=float(params.get("control_scale", 0.9)),
                  control_guidance_start=float(params.get("control_start", 0.0)),
                  control_guidance_end=float(params.get("control_end", 1.0)))
        if float(params.get("true_cfg", 1.0)) > 1.0:
            kw.update(true_cfg_scale=float(params["true_cfg"]), negative_prompt=params.get("negative") or "")
        keep_for = (img, mk)                          # risoluzione ORIGINALE: il risultato torna alla misura della foto
        return cpipe, kw, keep_for
    W = int(params.get("width") or 1024)
    H = int(params.get("height") or 1024)
    kind = params.get("control_preprocess") or "canny"
    if not params.get("control_image") or not os.path.isfile(params["control_image"]):
        raise ValueError("ControlNet: manca la foto guida (control_image)")
    ctl = cm.make_control_map(params["control_image"], kind, W, H, device=0 if _cuda() else -1)
    if mode == "flux_cn":
        from diffusers import FluxControlNetPipeline
        cpipe = _derive(pipe, FluxControlNetPipeline, controlnet=cn)
        sc, ge = cm.FLUX_UNION_DEFAULTS.get(kind, (0.7, 0.8))
        kw = _common_kw(params, arch, W, H, (30, 3.5))
        kw.update(control_image=ctl,
                  controlnet_conditioning_scale=float(params.get("control_scale") or sc),
                  control_guidance_start=float(params.get("control_start", 0.0)),
                  control_guidance_end=float(params.get("control_end") or ge))
        if float(params.get("true_cfg", 1.0)) > 1.0:
            kw.update(true_cfg_scale=float(params["true_cfg"]), negative_prompt=params.get("negative") or "")
        return cpipe, kw, None
    # zimage_cn
    from diffusers import ZImageControlNetPipeline
    cpipe = _derive(pipe, ZImageControlNetPipeline, controlnet=cn)
    kw = _common_kw(params, arch, W, H, (_ast, _agd))
    kw.update(control_image=ctl, controlnet_conditioning_scale=float(params.get("control_scale") or 0.75))
    if params.get("negative") and kw["guidance_scale"] > 0:
        kw["negative_prompt"] = params["negative"]
    return cpipe, kw, None


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:                                        # noqa: BLE001
        return False


def _finish(img, params, output_dir, mode, keep_for):
    """Ricomposizione fuori maschera (inpaint), salvataggio, firma stego e watermark come image_flux."""
    from PIL import Image, ImageFilter
    if keep_for is not None and os.environ.get("NATIVE_INPAINT_KEEP", "1") == "1":
        try:
            orig, mk = keep_for
            from inference_server.backends import image_flux as F
            import numpy as np
            res = img.resize(orig.size, Image.LANCZOS)
            featherpx = int(params.get("feather") or 0) or max(4, min(14, int(round(min(res.size) / 128))))
            mbool = np.asarray(mk, dtype=np.uint8) > 127
            if os.environ.get("NATIVE_INPAINT_COLORMATCH", "1") == "1":
                res = F._seam_tone_match(res, orig, mbool, featherpx, gain=True, tau_mult=6.0, clamp=32)
            m = mk.filter(ImageFilter.GaussianBlur(featherpx))
            img = Image.composite(res, orig, m)
        except Exception as e:                               # noqa: BLE001
            _log(f"ricomposizione fuori-maschera fallita ({e})")
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_{mode}_{params.get('seed', 0)}.png")
    img.save(path)
    try:
        from core import stego
        if stego.enabled():
            stego.sign_file(path)
    except Exception:                                        # noqa: BLE001
        pass
    try:
        from core import watermark
        path = watermark.apply_to_file(path)
    except RuntimeError:
        return {"status": "error", "error": "watermark obbligatorio non applicabile"}
    except Exception:                                        # noqa: BLE001
        pass
    return {"status": "success", "output": [path]}


def run(params: dict, mm, output_dir: str) -> dict:
    import gc
    import torch
    from PIL import Image
    from inference_server import preview_bus
    from inference_server.backends import image_flux as F
    mode = params.get("mode")
    if mode not in MODES:
        return {"status": "error", "error": f"image_control: modalità sconosciuta «{mode}» (valide: {', '.join(MODES)})"}
    F._step_reset(params.get("_t_button"))
    arch = _BASE_ARCH[mode]
    model = params.get("model") or F._default_model(arch)
    # i file dei ControlNet DEVONO esserci: errore chiaro prima di caricare i GB del modello base
    if mode == "flux_cn":
        _need(CN_FLUX_UNION, "il ControlNet FLUX Union Pro 2.0", 4.3)
    elif mode == "flux_cn_inpaint":
        _need(CN_FLUX_INPAINT, "il ControlNet FLUX Inpainting (Alimama)", 4.3)
    elif mode == "zimage_cn":
        _need(CN_ZIMAGE, "il ControlNet Z-Image Union 2.1", 6.7)
    else:
        _need(REDUX_FILE, "FLUX Redux", 0.13)
        _need(SIGLIP_FILE, "l'encoder visivo SigLIP so400m-384", 3.5)
    key = F.register(mm, model, arch)
    ident = F._weights_ident(arch, F._resolve_model_path(str(model)), "")
    preview_bus.set_message("Caricamento modello…")
    if os.environ.get("NATIVE_AUTOEVICT", "1") == "1" and key not in mm._cache:
        try:
            from inference_server.component_store import STORE
            if mm.evict_others(key, keep_ident=ident):
                STORE.evict_unused(torch.cuda.mem_get_info()[1] if torch.cuda.is_available() else 0)
                mm.flush()
                gc.collect()
        except Exception as e:                               # noqa: BLE001
            _log(f"pre-evict saltato ({e})")
    pipe = mm.ensure(key)
    est0 = int(getattr(pipe, "_ls_est_mb", F.VRAM_MB.get(arch, 24000)))
    pipe._ls_est_mb = est0 + _EXTRA_MB[mode]                 # il ControlNet/SigLIP occupa VRAM accanto alla base
    try:
        where = F._ensure_placement(pipe, mm, key)
    finally:
        pipe._ls_est_mb = est0
    F._set_attention(pipe, arch)
    _log(f"{mode}: base {arch} placement={where}")
    mm.apply_loras(pipe, params.get("loras"))
    cn = hook = vision = embedder = None
    try:
        preview_bus.set_message("Carico il ControlNet…" if mode != "flux_redux" else "Carico Redux…")
        if mode == "flux_redux":
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            imgs = [Image.open(p).convert("RGB") for p in (params.get("ref_images") or [params.get("image_path")])
                    if p and os.path.isfile(p)]
            if not imgs:
                return {"status": "error", "error": "Redux: manca l'immagine di riferimento"}
            vision = load_siglip_vision().to(dev)
            embedder = load_redux_embedder().to(dev)
            emb = redux_embeds(vision, embedder, imgs, dev, float(params.get("redux_strength", 1.0)))
            vision.to("cpu"); embedder.to("cpu"); del vision, embedder
            vision = embedder = None
            pe, pooled = redux_prompt_embeds(pipe, emb, params.get("prompt", ""), dev)
            W, H = int(params.get("width") or 1024), int(params.get("height") or 1024)
            _ast, _agd, _h = F._auto_settings("flux", str(model))
            kw = _common_kw(params, "flux", W, H, (_ast, _agd))
            kw.pop("prompt", None)
            kw.update(prompt_embeds=pe, pooled_prompt_embeds=pooled)
            cpipe, keep_for = pipe, None
        else:
            if mode == "zimage_cn":
                cn = load_zimage_controlnet(CN_ZIMAGE)
            else:
                cn = load_flux_controlnet(CN_FLUX_INPAINT if mode == "flux_cn_inpaint" else CN_FLUX_UNION)
            hook = _attach_cn(cn, where)
            cpipe, kw, keep_for = build_job(pipe, params, cn)
        try:
            kw["callback_on_step_end"] = F._make_preview_cb(arch, int(kw["width"]), int(kw["height"]),
                                                            kw["num_inference_steps"])
            kw["callback_on_step_end_tensor_inputs"] = ["latents"]
        except Exception:                                    # noqa: BLE001
            pass
        preview_bus.set_message("Generazione…")
        out = cpipe(**kw)
        preview_bus.set_message("Decodifica e salvataggio…")
        return _finish(out.images[0], params, output_dir, mode, keep_for)
    finally:
        if cn is not None:
            _detach_cn(cn, hook)
            del cn
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:                                    # noqa: BLE001
            pass
