"""LTX Studio — REST API (FastAPI).

Espone TUTTE le funzioni della GUI: generazione immagine, image-to-image/edit (Kontext),
inpaint/outpaint (con maschera), recolor, upscale immagine, segmentazione (SAM/CLIPSeg),
generazione video (con prompt/parametri/audio/controlnet/i2v), definizione video (SeedVR2/
ESRGAN/RIFE), prolunga video, upload audio (per voce/clone), elenco modelli, download file.

Ogni input-file può essere inviato come UPLOAD (multipart) **oppure** come PATH locale del
server (campo `*_path`). C'è un endpoint /download per scaricare un file dal path del server.

Avvio:  python -m uvicorn server.api:app --host 0.0.0.0 --port 8200
Swagger: http://localhost:8200/docs    OpenAPI: http://localhost:8200/openapi.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from core.gen_pipeline import GenPipeline
from core.kontext_pipeline import KontextPipeline
from core.inpaint_pipeline import InpaintPipeline
from core.upscale_pipeline import UpscalePipeline
from core.segment_pipeline import SegmentPipeline
from core.video_enhance_pipeline import VideoEnhancePipeline, SEEDVR2_MODELS
from core.extend_pipeline import ExtendPipeline
from core.template_engine import TemplateEngine
from core.model_catalog import ModelCatalog
from core.lora_catalog import LoraCatalog
from core.gen_pipeline import is_flux2
from server import runner

_mcat = ModelCatalog()
_FLUX_KW = ("flux", "getphat", "fuxcapacity", "krea")
_ZIMG_KW = ("z-image", "z_image", "zimage")
_ERNIE_KW = ("ernie",)

app = FastAPI(
    title="LTX Studio API",
    version="1.0.0",
    description="REST API per generazione/editing immagini e video (ComfyUI + LTX-2.3 / FLUX). "
                "Ogni file può essere caricato (multipart) o indicato come path locale del server "
                "(`*_path`). Output scaricabili via /download.",
)

_gen = GenPipeline()
_kontext = KontextPipeline()
_inpaint = InpaintPipeline()
_upscale = UpscalePipeline()
_segment = SegmentPipeline()
_enhance = VideoEnhancePipeline()
_extend = ExtendPipeline()


def _result(r: dict) -> dict:
    """Aggiunge URL di download agli output."""
    def dl(p): return f"/download?path={p}"
    r["download"] = [dl(f) for f in r.get("files", [])]
    return r


def _img_input(upload: UploadFile | None, path: str | None, field: str) -> str:
    """Risolve un input immagine (upload o path) e lo carica in ComfyUI → nome comfy."""
    data = upload.file.read() if upload else None
    local = runner.resolve_local(data, upload.filename if upload else None, path)
    if not local:
        raise HTTPException(400, f"Manca '{field}' (carica un file o indica '{field}_path').")
    return runner.to_comfy_image(local), local


def _err(e: Exception):
    raise HTTPException(500, f"{type(e).__name__}: {e}")


# ───────────────────────── utilità ─────────────────────────
@app.get("/health", tags=["util"])
def health():
    return {"comfyui_up": runner.driver().is_up()}


@app.get("/models", tags=["util"])
def models(folder: str = Query("diffusion_models", description="Cartella ComfyUI (diffusion_models, loras, vae, upscale_models…)")):
    try:
        return {"folder": folder, "models": runner.driver().models_in(folder)}
    except Exception as e:
        _err(e)


@app.get("/models/seedvr2", tags=["util"])
def models_seedvr2():
    return {"seedvr2": SEEDVR2_MODELS}


@app.get("/models/image", tags=["util"])
def models_image(allow_nsfw: bool = Query(True, description="Includi modelli NSFW")):
    """Modelli per la GENERAZIONE IMMAGINE, raggruppati per architettura, con flag NSFW."""
    try:
        _mcat.reload_if_changed()
    except Exception:
        pass
    out = {"flux": [], "flux2": [], "z-image": [], "ernie": []}
    for m in runner.driver().models_in("diffusion_models"):
        ml = m.lower(); nsfw = _mcat.is_nsfw(m)
        if not allow_nsfw and nsfw:
            continue
        item = {"model": m, "nsfw": nsfw}
        if "fill" in ml or "kontext" in ml:
            continue                                  # specializzati (inpaint/edit), non txt2img
        if any(k in ml for k in _FLUX_KW):
            (out["flux2"] if is_flux2(m) else out["flux"]).append(item)
        elif any(k in ml for k in _ZIMG_KW):
            out["z-image"].append(item)
        elif any(k in ml for k in _ERNIE_KW):
            out["ernie"].append(item)
    out["note"] = "La GUI usa solo FLUX.2 per la generazione; via API puoi scegliere qualsiasi arch."
    return out


@app.get("/models/video", tags=["util"])
def models_video():
    """Modelli per la GENERAZIONE VIDEO (LTX-2.3). NSFW → modello eros (auto con nsfw_mode=nsfw)."""
    ltx = [m for m in runner.driver().models_in("diffusion_models") if "ltx" in m.lower()]
    return {
        "standard": [m for m in ltx if "eros" not in m.lower()],
        "eros_nsfw": [m for m in ltx if "eros" in m.lower()],
        "all_ltx": ltx,
        "note": "Per il video usa /video/generate con nsfw_mode sfw|nsfw: la pipeline seleziona "
                "automaticamente il modello (standard vs eros).",
    }


@app.get("/models/loras", tags=["util"])
def models_loras(
    arch: str = Query("LTX", description="Architettura: LTX (video) | Flux (immagini)"),
    allow_nsfw: bool = Query(True),
):
    """Elenco LoRA compatibili con descrizione e trigger word (cosa fanno / come attivarle)."""
    try:
        cat = LoraCatalog(compatible_arch=arch)
        loras = [{"name": e.name, "comfy_name": e.comfy_name, "triggers": e.triggers,
                  "description": (e.description or "")[:300], "nsfw": e.nsfw}
                 for e in cat.loras(allow_nsfw=allow_nsfw)]
        return {"arch": arch, "count": len(loras), "loras": loras}
    except Exception as e:
        _err(e)


@app.get("/models/upscale", tags=["util"])
def models_upscale():
    """Modelli di upscale: per-fotogramma immagine/video (ESRGAN) + SeedVR2 (video)."""
    return {"image_and_frame": runner.driver().models_in("upscale_models"), "video_seedvr2": SEEDVR2_MODELS}


@app.post("/files/upload", tags=["util"])
def files_upload(file: UploadFile = File(..., description="File da salvare sul server")):
    """Salva un file sul server e ritorna il suo path locale (riutilizzabile via `*_path`)."""
    local = runner.resolve_local(file.file.read(), file.filename, None)
    return {"path": local, "download": f"/download?path={local}"}


@app.get("/download", tags=["util"])
def download(path: str = Query(..., description="Path locale del file sul server")):
    """Scarica un file dato il suo path locale sul server."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"File non trovato: {path}")
    return FileResponse(str(p), filename=p.name)


# ───────────────────────── immagini ─────────────────────────
@app.post("/image/generate", tags=["image"])
def image_generate(
    prompt: str = Form(..., description="Prompt testuale"),
    arch: str = Form("flux", description="flux | z-image | ernie"),
    model: str = Form(..., description="Nome file modello (es. Flux\\nvfp4\\flux2-dev-nvfp4-mixed.safetensors)"),
    negative: str = Form(""), width: int = Form(1024), height: int = Form(1024),
    steps: int = Form(20), guidance: float = Form(3.5), seed: int = Form(0),
    loras: str = Form("[]", description='JSON: [{"name":"...","strength":1.0}]'),
):
    """Generazione immagine txt2img (FLUX/FLUX.2/Z-Image/Ernie)."""
    try:
        p = _gen.build(arch, model, prompt, width=width, height=height, seed=seed,
                       steps=steps, guidance=guidance, negative=negative, loras=json.loads(loras))
        return _result(runner.run_graph(p))
    except Exception as e:
        _err(e)


@app.post("/image/edit", tags=["image"])
def image_edit(
    prompt: str = Form("", description="Come integrare/modificare (vuoto = automatico)"),
    image: Optional[UploadFile] = File(None), image_path: str = Form(None),
    engine: str = Form("kontext", description="kontext (FLUX.1) | flux2"),
    model: str = Form(None, description="Modello FLUX.2 se engine=flux2"),
    negative: str = Form(""), steps: int = Form(20), seed: int = Form(0),
    aux_lora: str = Form(None, description="LoRA ausiliaria (es. MagicQuill\\puzzle_lora.safetensors)"),
):
    """Image-to-image / integrazione (FLUX Kontext o FLUX.2): l'immagine è il riferimento,
    rigenerata/armonizzata secondo il prompt (non incolla)."""
    try:
        cn, _ = _img_input(image, image_path, "image")
        p = _kontext.integrate(cn, prompt, engine=engine, model=model, negative=negative,
                               steps=steps, seed=seed, aux_lora=aux_lora)
        return _result(runner.run_graph(p))
    except Exception as e:
        _err(e)


@app.post("/image/inpaint", tags=["image"])
def image_inpaint(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None), image_path: str = Form(None),
    mask: Optional[UploadFile] = File(None), mask_path: str = Form(None),
    model: str = Form(None, description="Checkpoint inpaint (FLUX Fill); vuoto = default"),
    steps: int = Form(20), denoise: float = Form(1.0), guidance: float = Form(30.0),
    seed: int = Form(0), negative: str = Form(""),
):
    """Inpainting: rigenera l'area mascherata (invia immagine + maschera)."""
    try:
        cn, _ = _img_input(image, image_path, "image")
        mdata = mask.file.read() if mask else None
        mlocal = runner.resolve_local(mdata, mask.filename if mask else None, mask_path)
        if not mlocal:
            raise HTTPException(400, "Manca 'mask' (file o mask_path).")
        runner.upload_comfy_mask(mlocal, cn)
        p = _inpaint.build(cn, prompt, mode="inpaint", mask_file=cn, denoise=denoise, steps=steps,
                           guidance=guidance, seed=seed, negative=negative, checkpoint=model)
        return _result(runner.run_graph(p))
    except Exception as e:
        _err(e)


@app.post("/image/recolor", tags=["image"])
def image_recolor(
    target: str = Form(..., description="Colore (es. 'rosso acceso' o #cc0000)"),
    subject: str = Form("", description="Cosa ricolorare (es. 'la maglietta')"),
    image: Optional[UploadFile] = File(None), image_path: str = Form(None),
    engine: str = Form("kontext"), model: str = Form(None), steps: int = Form(20), seed: int = Form(0),
):
    """Cambia colore (Kontext prompt-based): ricolora mantenendo texture e luce."""
    try:
        cn, _ = _img_input(image, image_path, "image")
        p = _kontext.recolor(cn, target, subject=subject, engine=engine, model=model, steps=steps, seed=seed)
        return _result(runner.run_graph(p))
    except Exception as e:
        _err(e)


@app.post("/image/upscale", tags=["image"])
def image_upscale(
    image: Optional[UploadFile] = File(None), image_path: str = Form(None),
    model: str = Form(..., description="Modello upscale (es. 4x-UltraSharpV2.safetensors)"),
    scale: float = Form(4.0), refine: bool = Form(False),
    refine_model: str = Form(None, description="Modello FLUX per rifinitura AI"),
    refine_steps: int = Form(12), refine_denoise: float = Form(0.3),
    prompt: str = Form(""), seed: int = Form(0),
):
    """Aumenta definizione immagine (modello AI) + opzionale rifinitura FLUX."""
    try:
        cn, _ = _img_input(image, image_path, "image")
        if refine and not refine_model:
            raise HTTPException(400, "refine=True richiede refine_model (FLUX).")
        p = _upscale.build(cn, model, target_scale=scale, refine=refine, refine_model=refine_model,
                           refine_steps=refine_steps, refine_denoise=refine_denoise,
                           refine_prompt=prompt, seed=seed)
        return _result(runner.run_graph(p))
    except Exception as e:
        _err(e)


@app.post("/image/segment", tags=["image"])
def image_segment(
    image: Optional[UploadFile] = File(None), image_path: str = Form(None),
    text: str = Form(..., description="Oggetto da selezionare (CLIPSeg+SAM)"),
    sam_model: str = Form(None), threshold: float = Form(0.4),
):
    """Selezione oggetto via testo (CLIPSeg + SAM) → maschera."""
    try:
        cn, _ = _img_input(image, image_path, "image")
        p = _segment.build_text(cn, text, sam_model=sam_model, clipseg_threshold=threshold)
        return _result(runner.run_graph(p))
    except Exception as e:
        _err(e)


# ───────────────────────── video ─────────────────────────
def _build_video(params: dict, image_comfy: str | None) -> dict:
    eng = TemplateEngine()
    try:
        eng.set_object_info(runner.driver().object_info())
    except Exception:
        pass
    eng.use_template("superset")
    vals = {k: v for k, v in params.items() if v is not None}
    if image_comfy:
        vals["ref_image_file"] = image_comfy
    loras = json.loads(params.get("loras") or "[]")
    prompt = eng.build_prompt(vals, loras=loras)
    disable = ["prompt_enhancer", "prompt_relay"]
    if not params.get("controlnet_enabled"):
        disable.append("controlnet")
    if params.get("audio_mode") != "clone":
        disable.append("voice")
    if params.get("audio_mode") != "user":
        disable.append("audio_user")
    disable += [f"keyframe{i}" for i in range(8, 1, -1)]
    prompt, _ = eng.disable_modules(prompt, disable)
    return prompt


@app.post("/video/generate", tags=["video"])
def video_generate(
    prompt: str = Form(...), negative: str = Form(""),
    width: int = Form(704), height: int = Form(1280), fps: int = Form(24),
    length_s: float = Form(8.0), steps: int = Form(8), cfg: float = Form(0.9),
    sampler: str = Form("euler_ancestral_cfg_pp"), seed: int = Form(0),
    nsfw_mode: str = Form("sfw", description="sfw | nsfw"),
    audio_mode: str = Form("native", description="native | user | clone | mute"),
    hr_refine: bool = Form(False, description="Rifinitura HR dopo upscale"),
    ref_strength: float = Form(0.8, description="Forza immagine i2v (0..1)"),
    image: Optional[UploadFile] = File(None, description="Immagine di partenza (i2v)"),
    image_path: str = Form(None),
    audio: Optional[UploadFile] = File(None, description="Audio per modalità user/clone"),
    audio_path: str = Form(None),
    control_video: Optional[UploadFile] = File(None, description="ControlNet: video di movimenti (pose/depth/canny)"),
    control_video_path: str = Form(None),
    controlnet_type: int = Form(0, description="0=pose, 1=depth, 2=canny (in base allo switch 5564)"),
    loras: str = Form("[]"),
):
    """Generazione VIDEO LTX-2.3 (t2v o i2v) con audio, parametri, HR e ControlNet (video-guida).
    Per i2v invia 'image'; per ControlNet invia 'control_video' (movimenti)."""
    try:
        image_comfy = None
        if image or image_path:
            image_comfy, _ = _img_input(image, image_path, "image")
        params = dict(prompt_positive=prompt, prompt_negative=negative, width=width, height=height,
                      fps=fps, length_s=length_s, steps=steps, cfg=cfg, sampler=sampler, seed=seed,
                      nsfw_mode=nsfw_mode, audio_mode=audio_mode, hr_refine=hr_refine,
                      ref_strength=ref_strength, loras=loras)
        if audio_mode in ("user", "clone") and (audio or audio_path):
            adata = audio.file.read() if audio else None
            alocal = runner.resolve_local(adata, audio.filename if audio else None, audio_path)
            acomfy = runner.to_comfy_image(alocal)
            params["user_audio_file" if audio_mode == "user" else "voice_sample_file"] = acomfy
        if control_video or control_video_path:
            cvdata = control_video.file.read() if control_video else None
            cvlocal = runner.resolve_local(cvdata, control_video.filename if control_video else None, control_video_path)
            params["control_video"] = runner.to_comfy_video(cvlocal)
            params["controlnet_type"] = int(controlnet_type)
            params["controlnet_enabled"] = True
        prompt_graph = _build_video(params, image_comfy)
        return _result(runner.run_graph(prompt_graph, timeout=2400))
    except Exception as e:
        _err(e)


@app.post("/video/enhance", tags=["video"])
def video_enhance(
    video: Optional[UploadFile] = File(None), video_path: str = Form(None),
    engine: str = Form("seedvr2", description="seedvr2 | esrgan | flashvsr"),
    seedvr2_model: str = Form("seedvr2_ema_7b_fp16.safetensors"),
    resolution: int = Form(1080), blocks_to_swap: int = Form(0), batch_size: int = Form(5),
    esrgan_model: str = Form("4x-UltraSharpV2.safetensors"), target_scale: float = Form(2.0),
    fps: int = Form(24), rife: bool = Form(False), rife_multiplier: int = Form(2),
    rife_ckpt: str = Form("rife47.pth"), seed: int = Form(0),
):
    """Definizione video: restore + super-risoluzione (SeedVR2/ESRGAN) + opz. RIFE (FPS)."""
    try:
        vdata = video.file.read() if video else None
        vlocal = runner.resolve_local(vdata, video.filename if video else None, video_path)
        if not vlocal:
            raise HTTPException(400, "Manca 'video' (file o video_path).")
        name = runner.to_comfy_video(vlocal)
        p = _enhance.build(name, engine=engine, seedvr2_model=seedvr2_model, resolution=resolution,
                           blocks_to_swap=blocks_to_swap, batch_size=batch_size, esrgan_model=esrgan_model,
                           target_scale=target_scale, fps=fps, rife=rife, rife_multiplier=rife_multiplier,
                           rife_ckpt=rife_ckpt, seed=seed)
        return _result(runner.run_graph(p, timeout=3600))
    except Exception as e:
        _err(e)


@app.post("/video/extend", tags=["video"])
def video_extend(
    prompt: str = Form(..., description="Come prosegue la scena"),
    video: Optional[UploadFile] = File(None), video_path: str = Form(None),
    width: int = Form(768), height: int = Form(768), fps: int = Form(24),
    add_seconds: float = Form(5.0), segments: int = Form(1), overlap_frames: int = Form(8),
    context_frames: int = Form(24), steps: int = Form(8), seed: int = Form(0), negative: str = Form(""),
):
    """Prolunga video (LTX-2.3 nativo, multi-segmento)."""
    try:
        vdata = video.file.read() if video else None
        vlocal = runner.resolve_local(vdata, video.filename if video else None, video_path)
        if not vlocal:
            raise HTTPException(400, "Manca 'video' (file o video_path).")
        name = runner.to_comfy_video(vlocal)
        p = _extend.build(name, prompt, width=width, height=height, fps=fps, add_seconds=add_seconds,
                          segments=segments, overlap_frames=overlap_frames, context_frames=context_frames,
                          steps=steps, seed=seed, negative=negative)
        return _result(runner.run_graph(p, timeout=3600))
    except Exception as e:
        _err(e)


@app.post("/audio/upload", tags=["audio"])
def audio_upload(file: UploadFile = File(..., description="Audio pre-registrato (per voce/clone)")):
    """Carica un audio pre-registrato; ritorna il path locale per riusarlo in /video/generate."""
    local = runner.resolve_local(file.file.read(), file.filename, None)
    return {"path": local, "download": f"/download?path={local}"}


@app.get("/", tags=["util"])
def root():
    return JSONResponse({"name": "LTX Studio API", "docs": "/docs", "openapi": "/openapi.json"})
