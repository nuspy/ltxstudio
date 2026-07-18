"""Endpoint IMMAGINI — esecuzione NATIVA via InferenceServer (:8189), zero ComfyUI.

generate (t2i) · edit/kontext · recolor · inpaint · outpaint · upscale/restauro · segment · faceswap · face-id (PuLID).
Ogni POST crea un job async → GET /jobs/{id} per stato + link di download.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api import gate
from api.jobs import MANAGER
from api.handlers._common import run_image, parse_loras, out_path


def _submit(kind: str, fn):
    return {"job_id": MANAGER.submit(kind, fn), "status": "pending"}


# ─────────────── modelli richiesta ───────────────
class GenerateReq(BaseModel):
    prompt: str
    model: str = Field(..., description="nome file modello (es. flux1-dev.safetensors)")
    arch: str = Field("flux", description="flux | flux2 | zimage | ernie")
    negative: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance: float = 3.5
    seed: int = 0
    loras: list | None = None
    pulid_face: str | None = Field(None, description="path foto identità (PuLID, solo flux)")
    pulid_strength: float = 0.8


class EditReq(BaseModel):
    prompt: str
    image_path: str | None = None                              # single (o usa `images`)
    images: list[str] | None = None                            # multi (krea2_edit: scena + soggetto, max 2)
    model: str | None = None                                   # non richiesto per krea2_edit (snapshot)
    arch: str = Field("kontext", description="kontext | flux2_kontext | krea2_edit")
    negative: str | None = None
    steps: int = 20
    guidance: float = 2.5
    seed: int = 0
    max_side: int = 1536
    grounding_px: int = 768                                     # solo krea2_edit (512-1536, edit↔identità)
    krea2_variant: str = "turbo"                               # solo krea2_edit: turbo | raw


class RecolorReq(BaseModel):
    image_path: str
    target: str = Field(..., description="colore/aspetto desiderato, es. 'capelli rossi'")
    subject: str = ""
    model: str
    arch: str = "kontext"
    steps: int = 20
    guidance: float = 2.5
    seed: int = 0


class InpaintReq(BaseModel):
    prompt: str
    image_path: str
    mask_path: str
    model: str
    arch: str = Field("inpaint", description="inpaint | flux2_inpaint")
    negative: str | None = None
    steps: int = 20
    guidance: float = 30.0
    seed: int = 0


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


# ─────────────── registrazione ───────────────
def register(app: FastAPI):

    @app.post("/image/generate", tags=["immagini"])
    def image_generate(req: GenerateReq):
        gate.require("cap.gen_images")                         # rispetta l'edizione come la GUI (403 se off)
        p = req.model_dump(exclude_none=True)
        p["loras"] = parse_loras(p.get("loras"))
        return _submit("image.generate", lambda job: run_image(p))

    @app.post("/image/edit", tags=["immagini"])
    def image_edit(req: EditReq):
        gate.require("cap.krea2_edit" if req.arch == "krea2_edit" else "cap.gen_images")
        p = req.model_dump(exclude_none=True)
        return _submit("image.edit", lambda job: run_image(p))

    @app.post("/image/recolor", tags=["immagini"])
    def image_recolor(req: RecolorReq):
        gate.require("cap.inpaint")
        subj = (req.subject + " ") if req.subject else ""
        p = {"arch": req.arch, "model": req.model, "image_path": req.image_path,
             "prompt": f"recolor {subj}to {req.target}, keep everything else identical",
             "steps": req.steps, "guidance": req.guidance, "seed": req.seed}
        return _submit("image.recolor", lambda job: run_image(p))

    @app.post("/image/inpaint", tags=["immagini"])
    def image_inpaint(req: InpaintReq):
        gate.require("cap.inpaint")
        p = req.model_dump(exclude_none=True)
        return _submit("image.inpaint", lambda job: run_image(p))

    @app.post("/image/outpaint", tags=["immagini"])
    def image_outpaint(req: OutpaintReq):
        gate.require("cap.inpaint")
        p = req.model_dump(exclude_none=True)
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
