"""Endpoint IMMAGINI — esecuzione NATIVA via InferenceServer (:8189), zero ComfyUI.

generate (t2i) · edit/kontext · recolor · inpaint · outpaint · upscale/restauro · segment · faceswap · face-id (PuLID).
Qwen-Image 2.1 (`arch: qwen_image21`, nessun file-modello: il DiT è una voce di catalogo `qwen21_dit`) vale per
generate (anche img2img con `init_image`+`strength`), edit (≤10 riferimenti), recolor, inpaint e outpaint.
Ogni POST crea un job async → GET /jobs/{id} per stato + link di download.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api import gate
from api.jobs import MANAGER
from api.handlers._common import run_image, parse_loras, out_path


def _submit(kind: str, fn):
    return {"job_id": MANAGER.submit(kind, fn), "status": "pending"}


# ─────────────── modelli richiesta ───────────────
class Qwen21Opts(BaseModel):
    """Campi propri di Qwen-Image 2.1 (ignorati dalle altre architetture)."""
    qwen21_dit: str | None = Field(None, description="DiT di catalogo (GET /image/qwen21/options), "
                                                     "default qwen21-dit-fp8")
    qwen21_te: str | None = Field(None, description="'' | heretic (encoder 18+)")
    qwen21_turbo: str | None = Field(None, description="'' | 4 | 6 (LoRA turbo, solo DiT non Turbo)")
    transparent: bool = False
    aspect_2k: str | None = Field(None, description="1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 16:9 | 9:16")
    ref_images: list[str] | None = Field(None, description="≤10 riferimenti, citati come «image N» nel prompt")


class GenerateReq(Qwen21Opts):
    prompt: str
    model: str | None = Field(None, description="nome file modello (es. flux1-dev.safetensors); "
                                                "non serve con qwen_image21")
    arch: str = Field("flux", description="flux | flux2 | zimage | ernie | qwen_image21")
    init_image: str | None = Field(None, description="img2img (qwen_image21): foto di partenza")
    strength: float | None = Field(None, description="img2img: 0.05-1.0 (default 0.6)")
    negative: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance: float = 3.5
    seed: int = 0
    loras: list | None = None
    pulid_face: str | None = Field(None, description="path foto identità (PuLID, solo flux)")
    pulid_strength: float = 0.8


class EditReq(Qwen21Opts):
    prompt: str
    image_path: str | None = None                              # single (o usa `images`)
    images: list[str] | None = None                            # multi (krea2_edit max 2, qwen_image21 max 10)
    model: str | None = None                                   # non richiesto per krea2_edit/qwen_image21
    arch: str = Field("kontext", description="kontext | flux2_kontext | krea2_edit | qwen_image21")
    negative: str | None = None
    steps: int = 20
    guidance: float = 2.5
    seed: int = 0
    max_side: int = 1536
    grounding_px: int = 768                                     # solo krea2_edit (512-1536, edit↔identità)
    krea2_variant: str = "turbo"                               # solo krea2_edit: turbo | raw


class RecolorReq(Qwen21Opts):
    image_path: str
    target: str = Field(..., description="colore/aspetto desiderato, es. 'capelli rossi'")
    subject: str = ""
    model: str | None = None                                   # non serve con qwen_image21
    arch: str = Field("kontext", description="kontext | qwen_image21")
    steps: int = 20
    guidance: float = 2.5
    seed: int = 0


class InpaintReq(Qwen21Opts):
    prompt: str
    image_path: str
    mask_path: str
    model: str | None = None                                   # non serve con qwen_image21
    arch: str = Field("inpaint", description="inpaint | flux2_inpaint | qwen_image21")
    negative: str | None = None
    steps: int = 20
    guidance: float = 30.0
    seed: int = 0
    strength: float | None = Field(None, description="qwen_image21: denoise nell'area (default 1.0)")


class OutpaintReq(InpaintReq):
    pass


class UpscaleReq(BaseModel):
    image_path: str
    model: str = Field(..., description="upscaler, es. 4x-UltraSharpV2.safetensors")
    target_scale: float = 4.0
    seed: int = 0


class SegmentReq(BaseModel):
    image_path: str
    text: str = Field(..., description="oggetto da segmentare (es. 'person')")
    threshold: float = 0.3
    use_sam: bool = True


class FaceSwapReq(BaseModel):
    target_path: str = Field(..., description="immagine su cui mettere il volto")
    source_face_path: str = Field(..., description="foto del volto sorgente")


def _need_model(p: dict) -> None:
    if p.get("arch") != "qwen_image21" and not p.get("model"):
        raise HTTPException(422, f"'model' obbligatorio con arch={p.get('arch')}")


def _qwen21(p: dict, req: BaseModel | None = None) -> dict:
    """Valida e completa un job Qwen-Image 2.1: gate d'edizione (anche 18+), DiT di catalogo, passi/CFG
    consigliati per il DiT quando il client lascia i default dell'endpoint (pensati per FLUX)."""
    if req is not None:
        for k in ("steps", "guidance"):
            f = type(req).model_fields.get(k)
            if f is not None and p.get(k) == f.default:
                p.pop(k, None)
    from core.qwen21_spec import DITS, TE_VARIANTS, TURBO_STEPS, DEFAULT_STEPS, DEFAULT_GUIDANCE, MAX_REFS
    gate.require("cap.image_qwen21")
    dit = p.setdefault("qwen21_dit", "qwen21-dit-fp8")
    if dit not in DITS:
        raise HTTPException(422, f"qwen21_dit sconosciuto: {dit} ({', '.join(DITS)})")
    meta = DITS[dit]
    if meta.get("adult") or p.get("qwen21_te") in TE_VARIANTS:
        gate.require("cap.adult_mode")
    if p.get("steps") is None:
        p["steps"] = TURBO_STEPS if meta.get("turbo") else int(meta.get("steps", DEFAULT_STEPS))
    if meta.get("turbo") or p.get("guidance") is None:
        p["guidance"] = 1.0 if meta.get("turbo") else float(meta.get("cfg", DEFAULT_GUIDANCE))
    if len(p.get("ref_images") or []) > MAX_REFS:
        raise HTTPException(422, f"al massimo {MAX_REFS} riferimenti")
    p.pop("model", None)
    return p


# ─────────────── registrazione ───────────────
def register(app: FastAPI):

    @app.get("/image/qwen21/options", tags=["immagini"])
    def image_qwen21_options():
        """DiT Qwen-Image 2.1 disponibili (le voci 18+ solo nelle edizioni che le consentono) e se sono già
        scaricati; formati 2K nativi."""
        from core import ensure_weights as ew
        from core.qwen21_spec import DITS, ASPECTS_2K
        adult = gate.is_on("cap.adult_mode")
        return {"dits": [{"id": d, "turbo": bool(m.get("turbo")), "adult": bool(m.get("adult")),
                          "installed": not ew.missing([d])}
                         for d, m in DITS.items() if adult or not m.get("adult")],
                "aspects_2k": {k: list(v) for k, v in ASPECTS_2K.items()}}

    @app.post("/image/generate", tags=["immagini"])
    def image_generate(req: GenerateReq):
        gate.require("cap.gen_images")                         # rispetta l'edizione come la GUI (403 se off)
        p = req.model_dump(exclude_none=True)
        _need_model(p)
        if p["arch"] == "qwen_image21":
            _qwen21(p, req)
        p["loras"] = parse_loras(p.get("loras"))
        return _submit("image.generate", lambda job: run_image(p))

    @app.post("/image/edit", tags=["immagini"])
    def image_edit(req: EditReq):
        gate.require("cap.krea2_edit" if req.arch == "krea2_edit" else "cap.gen_images")
        p = req.model_dump(exclude_none=True)
        if p["arch"] == "qwen_image21":                           # immagini → riferimenti «image N»
            refs = ([p.pop("image_path")] if p.get("image_path") else []) + list(p.pop("images", None) or [])
            p["ref_images"] = refs + list(p.get("ref_images") or [])
            if not p["ref_images"]:
                raise HTTPException(422, "serve almeno un'immagine (image_path, images o ref_images)")
            for k in ("grounding_px", "krea2_variant", "max_side"):
                p.pop(k, None)
            _qwen21(p, req)
        return _submit("image.edit", lambda job: run_image(p))

    @app.post("/image/recolor", tags=["immagini"])
    def image_recolor(req: RecolorReq):
        gate.require("cap.inpaint")
        subj = (req.subject + " ") if req.subject else ""
        if req.arch == "qwen_image21":
            p = {**req.model_dump(exclude_none=True, include=set(Qwen21Opts.model_fields)),
                 "arch": "qwen_image21", "ref_images": [req.image_path], "seed": req.seed,
                 "prompt": f"Recolor {subj or 'the main subject '}in image 1 to {req.target}. Keep the original "
                           f"texture, material, lighting and shadows; change only the color."}
            _qwen21(p)
            return _submit("image.recolor", lambda job: run_image(p))
        p = {"arch": req.arch, "model": req.model, "image_path": req.image_path,
             "prompt": f"recolor {subj}to {req.target}, keep everything else identical",
             "steps": req.steps, "guidance": req.guidance, "seed": req.seed}
        _need_model(p)
        return _submit("image.recolor", lambda job: run_image(p))

    def _qwen21_inpaint(p: dict, req: BaseModel) -> dict:
        p["init_image"] = p.pop("image_path")                     # foto da ritoccare + maschera (bianco = rifai)
        return _qwen21(p, req)

    @app.post("/image/inpaint", tags=["immagini"])
    def image_inpaint(req: InpaintReq):
        gate.require("cap.inpaint")
        p = req.model_dump(exclude_none=True)
        _need_model(p)
        if p["arch"] == "qwen_image21":
            _qwen21_inpaint(p, req)
        return _submit("image.inpaint", lambda job: run_image(p))

    @app.post("/image/outpaint", tags=["immagini"])
    def image_outpaint(req: OutpaintReq):
        gate.require("cap.inpaint")
        p = req.model_dump(exclude_none=True)
        _need_model(p)
        if p["arch"] == "qwen_image21":                           # l'immagine arriva già allargata, maschera = bordo
            _qwen21_inpaint(p, req)
        else:
            p["outpaint"] = True
        return _submit("image.outpaint", lambda job: run_image(p))

    @app.post("/image/upscale", tags=["immagini"])
    def image_upscale(req: UpscaleReq):
        gate.require("cap.upscale_image")
        p = req.model_dump(exclude_none=True); p["_module"] = "upscale"
        return _submit("image.upscale", lambda job: run_image(p))

    @app.post("/image/segment", tags=["immagini"])
    def image_segment(req: SegmentReq):
        gate.require("cap.sam")
        p = req.model_dump(exclude_none=True); p["_module"] = "segment"
        return _submit("image.segment", lambda job: run_image(p))

    @app.post("/image/faceswap", tags=["immagini"])
    def image_faceswap(req: FaceSwapReq):
        gate.require("cap.gen_images")
        def fn(job):
            from core import faceswap
            if not faceswap.available():
                raise RuntimeError("Face-swap non disponibile (inswapper_128.onnx / insightface mancanti).")
            out = out_path("faceswap", "png")
            return [faceswap.swap_image(req.target_path, req.source_face_path, out)]
        return _submit("image.faceswap", fn)
