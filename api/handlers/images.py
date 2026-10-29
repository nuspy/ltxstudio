"""Endpoint IMMAGINI — esecuzione NATIVA via InferenceServer (:8189), zero ComfyUI.

generate (t2i) · edit/kontext · recolor · inpaint · outpaint · upscale/restauro · segment · faceswap · face-id (PuLID) ·
controlnet (FLUX.1 / Z-Image) · redux · magicquill (task 9).
Qwen-Image 2.1 (`arch: qwen_image21`, nessun file-modello: il DiT è una voce di catalogo `qwen21_dit`) vale per
generate (anche img2img con `init_image`+`strength`), edit (≤10 riferimenti), recolor, inpaint e outpaint.
FLUX.3 Image Klein (`arch: flux3`, oppure `model: "flux3"`; nessun file-modello, DiT di catalogo `flux3_dit`): generate (t2i, layout
con `boxes`), edit (≤10 riferimenti `ref_images`/`images`, riquadri di edit) e inpaint (modifica per istruzione + composito NOSTRO con
la maschera binarizzata: FLUX.3 non ha un inpaint nativo). Pesi mancanti → 409 (nome e GB), nessun download.
Ogni POST crea un job async → GET /jobs/{id} per stato + link di download.
"""
from __future__ import annotations

import os

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
    qwen21_te: str | None = Field(None, description="'' (abbinato al DiT) | nvfp4 | fp8 | heretic (18+)")
    qwen21_turbo: str | None = Field(None, description="'' | 4 | 6 (LoRA turbo, solo DiT non Turbo)")
    transparent: bool = False
    aspect_2k: str | None = Field(None, description="1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 16:9 | 9:16")
    ref_images: list[str] | None = Field(None, description="≤10 riferimenti, citati come «image N» nel prompt")
    qwen21_pe: str | None = Field(None, description="arricchitore ufficiale: auto | t2i | i2i")
    qwen21_pe_think: bool = Field(False, description="arricchitore in modalità approfondita (~3 min invece di ~40 s)")


class Flux3Opts(BaseModel):
    """Campi propri di FLUX.3 Image Klein (ignorati dalle altre architetture)."""
    flux3_dit: str | None = Field(None, description="DiT di catalogo (default flux3-dit-fp8r)")
    flux3_pup: bool = Field(True, description="upsampler di prompt (BFL: raccomandato SEMPRE)")
    flux3_te: str | None = Field(None, description="text encoder: '' (automatico: Krea2 uncensored se su disco, 18+) | krea2 | official")
    boxes: list[dict] | None = Field(None, description="riquadri, coordinate 0-1 [y_min,x_min,y_max,x_max]: layout "
        "{id,desc,bbox}; con riferimenti {id,desc,mode:new|move|anchor|remove,ref(1-based),src_bbox,tgt_bbox}")


class GenerateReq(Flux3Opts, Qwen21Opts):
    prompt: str
    model: str | None = Field(None, description="nome file modello (es. flux1-dev.safetensors); "
                                                "non serve con qwen_image21")
    arch: str = Field("flux", description="flux | flux2 | zimage | ernie | qwen_image21 | flux3")
    init_image: str | None = Field(None, description="img2img (qwen_image21): foto di partenza")
    control_image: str | None = Field(None, description="ControlNet (qwen_image21): foto guida della struttura")
    control_preprocess: str | None = Field(None, description="canny | lineart | depth | gray | none")
    control_scale: float | None = Field(None, description="forza del controllo 0.1-1.0 (default 1.0)")
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


class EditReq(Flux3Opts, Qwen21Opts):
    prompt: str
    image_path: str | None = None                              # single (o usa `images`)
    images: list[str] | None = None                            # multi (krea2_edit max 2, qwen_image21 max 10)
    model: str | None = None                                   # non richiesto per krea2_edit/qwen_image21
    arch: str = Field("kontext", description="kontext | flux2_kontext | krea2_edit | qwen_image21 | flux3")
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


class InpaintReq(Flux3Opts, Qwen21Opts):
    prompt: str
    image_path: str
    mask_path: str
    model: str | None = None                                   # non serve con qwen_image21
    arch: str = Field("inpaint", description="inpaint | flux2_inpaint | qwen_image21 | flux3 (composito nostro)")
    negative: str | None = None
    steps: int = 20
    guidance: float = 30.0
    seed: int = 0
    strength: float | None = Field(None, description="qwen_image21: denoise nell'area (default 1.0)")
    controlnet_inpaint: bool = Field(False, description="FLUX.1 base: usa il ControlNet Inpainting (Alimama); arch resta 'inpaint'")
    control_scale: float = Field(0.9, description="forza del ControlNet Inpainting (0.6-1.0)")


class OutpaintReq(InpaintReq):
    pass


class ControlNetReq(BaseModel):
    """ControlNet di struttura: FLUX.1 (Union Pro 2.0) o Z-Image (Union 2.1). Il file guida è una foto (si estrae la mappa)
    o già una mappa (`control_preprocess=none`)."""
    prompt: str
    model: str | None = Field(None, description="checkpoint FLUX.1 (arch=flux) o Z-Image (arch=zimage), come in /image/generate")
    arch: str = Field("flux", description="flux | zimage")
    control_image: str = Field(..., description="foto guida (o mappa già pronta con control_preprocess=none)")
    control_preprocess: str = Field("canny", description="canny | depth | pose | gray | none")
    control_scale: float | None = Field(None, description="forza; default dagli autori: canny 0.7, depth 0.8, pose 0.9, gray 0.9 (zimage 0.75)")
    control_end: float | None = Field(None, description="FLUX: fino a che punto agisce il controllo (0.2-1.0; default 0.8, posa 0.65)")
    negative: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int | None = Field(None, description="default del modello (FLUX 30, Z-Image Turbo 8)")
    guidance: float | None = Field(None, description="default del modello (FLUX 3.5, Z-Image Turbo 0.0)")
    seed: int = 0
    loras: list | None = None


class ReduxReq(BaseModel):
    """Redux (FLUX.1): variazioni di una o più immagini di riferimento; il prompt è facoltativo."""
    model: str
    ref_images: list[str] = Field(..., min_length=1, description="immagini di riferimento (i token si concatenano)")
    prompt: str = ""
    redux_strength: float = 1.0
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    guidance: float | None = None
    seed: int = 0
    loras: list | None = None


class MagicQuillReq(BaseModel):
    """MagicQuill V2 su FLUX.1 Kontext. Maschere: bianco = area (local/removal/color) o tratti (edge)."""
    mode: str = Field(..., description="local | removal | edge | color | foreground")
    model: str = Field(..., description="checkpoint FLUX.1 Kontext")
    image_path: str
    mask_path: str | None = None
    edge_path: str | None = Field(None, description="edge: tratti da AGGIUNGERE")
    erase_path: str | None = Field(None, description="edge: bordi da TOGLIERE")
    color_path: str | None = Field(None, description="color: immagine con le zone ricolorate")
    color_rgb: list[int] | None = Field(None, description="color: [r,g,b] applicato all'area mask_path")
    prepared: bool = Field(False, description="foreground: image_path ha già l'oggetto e l'alone bianco (come fa la GUI)")
    prompt: str = ""
    steps: int = 20
    guidance: float = 3.5
    seed: int = 0
    grow: int | None = None
    strength_local: float | None = None
    strength_edge: float | None = None
    strength_color: float | None = None
    fix_perspective: bool = False


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
    if p.get("arch") not in ("qwen_image21", "flux3") and not p.get("model"):
        raise HTTPException(422, f"'model' obbligatorio con arch={p.get('arch')}")


def _qwen21(p: dict, req: BaseModel | None = None) -> dict:
    """Valida e completa un job Qwen-Image 2.1: gate d'edizione (anche 18+), DiT di catalogo, passi/CFG
    consigliati per il DiT quando il client lascia i default dell'endpoint (pensati per FLUX)."""
    if req is not None:
        for k in ("steps", "guidance"):
            f = type(req).model_fields.get(k)
            if f is not None and p.get(k) == f.default:
                p.pop(k, None)
    from core.qwen21_spec import DITS, ADULT_TE, TE_VARIANTS, TURBO_STEPS, DEFAULT_STEPS, DEFAULT_GUIDANCE, MAX_REFS
    gate.require("cap.image_qwen21")
    dit = p.setdefault("qwen21_dit", "qwen21-dit-fp8")
    if dit not in DITS:
        raise HTTPException(422, f"qwen21_dit sconosciuto: {dit} ({', '.join(DITS)})")
    meta = DITS[dit]
    if p.get("qwen21_te") and p["qwen21_te"] not in TE_VARIANTS:
        raise HTTPException(422, f"qwen21_te sconosciuto: {p['qwen21_te']} ({', '.join(TE_VARIANTS)})")
    if meta.get("adult") or p.get("qwen21_te") in ADULT_TE:
        gate.require("cap.adult_mode")
    if p.get("qwen21_pe"):
        gate.require("cap.qwen21_prompt_enhancer")
    if p.get("control_image"):
        gate.require("cap.qwen21_controlnet")
    if p.get("steps") is None:
        p["steps"] = TURBO_STEPS if meta.get("turbo") else int(meta.get("steps", DEFAULT_STEPS))
    if meta.get("turbo") or p.get("guidance") is None:
        p["guidance"] = 1.0 if meta.get("turbo") else float(meta.get("cfg", DEFAULT_GUIDANCE))
    if len(p.get("ref_images") or []) > MAX_REFS:
        raise HTTPException(422, f"al massimo {MAX_REFS} riferimenti")
    _foto_tokens(p)
    p.pop("model", None)
    return p


def _flux3_arch(p: dict) -> None:
    """`model: "flux3"` vale come `arch: "flux3"` (nessun file-modello: lo risolve il catalogo)."""
    if str(p.get("model") or "").lower() == "flux3":
        p["arch"] = "flux3"


def _flux3(p: dict, req: BaseModel | None = None) -> dict:
    """Valida e completa un job FLUX.3 Klein: gate d'edizione, riquadri (formato di core.flux3_caption), ≤10 riferimenti,
    token @Foto_N → «the first image»…, pesi su disco (409 altrimenti). Distillato: niente guidance né negative."""
    from core import flux3_caption as C3
    from core import flux3_spec as S3
    gate.require("cap.image_flux3")
    p.setdefault("flux3_dit", S3.DEFAULT_DIT)
    if p["flux3_dit"] not in S3.DITS:
        raise HTTPException(422, f"flux3_dit sconosciuto: {p['flux3_dit']} ({', '.join(S3.DITS)})")
    if p.get("flux3_pup", True):
        gate.require("cap.flux3_upsampler")
    if p.get("flux3_te") not in (None, *S3.TE_CHOICES):
        raise HTTPException(422, f"flux3_te sconosciuto: {p['flux3_te']} ({' | '.join(repr(c) for c in S3.TE_CHOICES)})")
    te_id = S3.te_id(p.get("flux3_te") or "", _present)
    if te_id == S3.TE_KREA2:                                      # encoder uncensored: solo dove le voci 18+ sono ammesse
        gate.require("cap.adult_mode")
    p["flux3_te"] = p.get("flux3_te") or ""
    refs = [r for r in (p.get("ref_images") or []) if r]
    if len(refs) > S3.MAX_REFS:
        raise HTTPException(422, f"al massimo {S3.MAX_REFS} riferimenti")
    n_refs = len(refs) + (1 if p.get("init_image") else 0)
    if p.get("boxes"):
        gate.require("cap.flux3_boxes")
        errs = C3.validate(p["boxes"], n_refs)
        if errs:
            raise HTTPException(422, "riquadri non validi: " + "; ".join(errs))
    if req is not None:
        f = type(req).model_fields.get("steps")
        if f is not None and p.get("steps") == f.default:
            p.pop("steps", None)                                  # il default dell'endpoint è pensato per FLUX.1
    p["steps"] = max(S3.MIN_STEPS, min(S3.MAX_STEPS, int(p.get("steps") or S3.DEFAULT_STEPS)))
    for k in ("guidance", "negative", "model", "qwen21_dit", "qwen21_te", "qwen21_turbo", "qwen21_pe", "qwen21_pe_think",
              "aspect_2k", "transparent", "control_image", "control_preprocess", "control_scale", "strength"):
        p.pop(k, None)
    _foto_tokens(p, "flux3")
    _weights_409(_flux3_ids(p["flux3_dit"], p.get("flux3_pup", True), p["flux3_te"]), "FLUX.3 Klein")
    return p


def _present(entry_id: str) -> bool:
    from core import model_paths as mp
    from core import weights_catalog as wc
    e = mp.entry(entry_id)
    return bool(e) and wc.is_present(e, str(mp.models_root()))


def _flux3_ids(dit: str, pup: bool, te: str = "") -> list[str]:
    from core import flux3_spec as S3
    te_id = S3.te_id(te, _present)
    return ([dit, te_id, S3.VAE] + ([S3.TE_CONFIGS] if te_id != S3.TE else [])
            + ([S3.PUP_DEFAULT, "flux3-pup-configs"] if pup else []))


def _foto_tokens(p: dict, style: str = "qwen21") -> None:
    """Riferimenti nominati come nella GUI: l'elemento k-esimo di `ref_images` è `@Foto_k`. Il prompt può citarlo con
    `@Foto_N` (tradotto in «image N», la sintassi di Qwen-Image 2.1); un token che cita una foto inesistente è 422,
    mai ignorato in silenzio. Le immagini passate ma non citate vanno comunque al modello. «image N» scritto a mano
    resta valido."""
    from core import image_refs as R
    refs = [r for r in (p.get("ref_images") or []) if r]
    prompt = p.get("prompt") or ""
    order = {"image": list(range(1, len(refs) + 1))}
    bad = R.unknown_tokens(prompt, {("image", n) for n in order["image"]})
    if bad:
        names = ", ".join(sorted({R.token_of(t.kind, t.n) for t in bad}))
        raise HTTPException(422, f"il prompt cita {names}, ma ref_images ne contiene {len(refs)} "
                                 f"(@Foto_1…@Foto_{len(refs)})" if refs else
                                 f"il prompt cita {names} ma non ci sono immagini in ref_images")
    p["prompt"] = R.translate(prompt, order, style)


def _weights_409(ids: list[str], feature: str) -> None:
    """Pesi assenti → 409 con nome e GB (nessun download dall'API: li scarica l'utente dal tab Pesi)."""
    from core import ensure_weights as ew
    try:
        ew.require(ids, feature)
    except ew.WeightsMissingError as e:
        raise HTTPException(409, str(e))


def _flux1_or_422(model: str | None, what: str) -> str:
    """Il modello deve esistere ed essere FLUX.1 (header, core.arch_filter): FLUX.2/Krea/Qwen non hanno questa funzione."""
    if not model:
        raise HTTPException(422, f"'model' obbligatorio per {what}")
    from core import arch_filter as af
    c = af.classify(model)
    if c["source"] != "none" and c["family"] != "flux1":
        raise HTTPException(422, f"{what} vale solo per FLUX.1: «{model}» è {c['family']}")
    return model


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

    @app.get("/image/flux3/options", tags=["immagini"])
    def image_flux3_options():
        """FLUX.3 Image Klein: DiT disponibili e se i pesi sono su disco (DiT, text encoder, VAE, upsampler); limiti."""
        from core import ensure_weights as ew
        from core import flux3_spec as S3
        return {"dits": [{"id": d, "steps": m["steps"], "installed": not ew.missing(_flux3_ids(d, False))}
                         for d, m in S3.DITS.items()],
                "upsampler": {"id": S3.PUP_DEFAULT, "installed": not ew.missing([S3.PUP_DEFAULT, "flux3-pup-configs"])},
                "max_refs": S3.MAX_REFS, "side": [S3.MIN_SIDE, S3.MAX_SIDE], "box_modes": list(S3.BOX_MODES),
                "inpaint_native": False}

    @app.post("/image/generate", tags=["immagini"])
    def image_generate(req: GenerateReq):
        gate.require("cap.gen_images")                         # rispetta l'edizione come la GUI (403 se off)
        p = req.model_dump(exclude_none=True)
        _flux3_arch(p)
        _need_model(p)
        if p["arch"] == "flux3":
            _flux3(p, req)
        elif p["arch"] == "qwen_image21":
            _qwen21(p, req)
        p["loras"] = parse_loras(p.get("loras"))
        return _submit("image.generate", lambda job: run_image(p))

    @app.post("/image/edit", tags=["immagini"])
    def image_edit(req: EditReq):
        gate.require("cap.krea2_edit" if req.arch == "krea2_edit" else "cap.gen_images")
        p = req.model_dump(exclude_none=True)
        _flux3_arch(p)
        if p["arch"] == "flux3":                                  # immagini → riferimenti «the first image»…
            refs = ([p.pop("image_path")] if p.get("image_path") else []) + list(p.pop("images", None) or [])
            p["ref_images"] = refs + list(p.get("ref_images") or [])
            if not p["ref_images"]:
                raise HTTPException(422, "serve almeno un'immagine (image_path, images o ref_images)")
            for k in ("grounding_px", "krea2_variant", "max_side"):
                p.pop(k, None)
            _flux3(p, req)
        elif p["arch"] == "qwen_image21":                         # immagini → riferimenti «image N»
            refs = ([p.pop("image_path")] if p.get("image_path") else []) + list(p.pop("images", None) or [])
            p["ref_images"] = refs + list(p.get("ref_images") or [])
            if not p["ref_images"]:
                raise HTTPException(422, "serve almeno un'immagine (image_path, images o ref_images)")
            for k in ("grounding_px", "krea2_variant", "max_side"):
                p.pop(k, None)
            _qwen21(p, req)
        return _submit("image.edit", lambda job: run_image(p))

    @app.post("/image/controlnet", tags=["immagini"])
    def image_controlnet(req: ControlNetReq):
        """Generazione guidata dalla STRUTTURA di una foto (bordi/profondità/posa/luminosità): FLUX.1 o Z-Image."""
        gate.require("cap.image_controlnet")
        if req.arch not in ("flux", "zimage"):
            raise HTTPException(422, "arch: flux | zimage (Qwen-Image 2.1 ha il suo ControlNet in /image/generate)")
        from core.control_maps import KINDS
        if req.control_preprocess not in KINDS:
            raise HTTPException(422, f"control_preprocess: {', '.join(KINDS)}")
        if not os.path.isfile(req.control_image):
            raise HTTPException(422, f"control_image inesistente: {req.control_image}")
        if req.arch == "flux":
            _flux1_or_422(req.model, "il ControlNet FLUX.1")
        elif not req.model:
            raise HTTPException(422, "'model' obbligatorio con arch=zimage")
        mode = "flux_cn" if req.arch == "flux" else "zimage_cn"
        ids = ["flux-cn-union-pro2" if req.arch == "flux" else "zimage-cn-union21"]
        ids += {"depth": ["depth-anything-v2"], "pose": ["openpose-body"]}.get(req.control_preprocess, [])
        _weights_409(ids, "ControlNet")
        p = req.model_dump(exclude_none=True)
        p.update({"_module": "image_control", "mode": mode})
        p["loras"] = parse_loras(p.get("loras"))
        return _submit("image.controlnet", lambda job: run_image(p))

    @app.post("/image/redux", tags=["immagini"])
    def image_redux(req: ReduxReq):
        """Variazioni da immagine (FLUX.1 Redux)."""
        gate.require("cap.image_redux")
        _flux1_or_422(req.model, "Redux")
        missing = [x for x in req.ref_images if not os.path.isfile(x)]
        if missing:
            raise HTTPException(422, f"ref_images inesistenti: {missing}")
        _weights_409(["flux-redux", "siglip-so400m-384"], "Redux")
        p = req.model_dump(exclude_none=True)
        p.update({"_module": "image_control", "mode": "flux_redux", "arch": "flux"})
        p["loras"] = parse_loras(p.get("loras"))
        return _submit("image.redux", lambda job: run_image(p))

    @app.post("/image/magicquill", tags=["immagini"])
    def image_magicquill(req: MagicQuillReq):
        """MagicQuill V2 (FLUX.1 Kontext + LoRA di controllo): modifica locale, rimozione, bordi, colore, integrazione."""
        gate.require("cap.image_magicquill")
        if req.mode not in ("local", "removal", "edge", "color", "foreground"):
            raise HTTPException(422, "mode: local | removal | edge | color | foreground")
        from core import arch_filter as af
        if not af.is_kontext(req.model):
            raise HTTPException(422, f"MagicQuill richiede un checkpoint FLUX.1 Kontext: «{req.model}»")
        from core import optional_components as oc
        if not oc.magicquill_installed():                         # componente opzionale (licenza non commerciale)
            raise HTTPException(409, oc.NOT_INSTALLED_MSG)
        _weights_409([f"mq2-{'puzzle' if req.mode == 'foreground' else req.mode}-lora"], "MagicQuill V2")
        p = req.model_dump(exclude_none=True)
        p.update({"_module": "image_mq", "arch": "kontext"})
        return _submit("image.magicquill", lambda job: run_image(p))

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

    def _cn_inpaint(p: dict, outpaint: bool) -> dict:
        """Inpaint/outpaint col ControlNet Inpainting (Alimama) su un FLUX.1 BASE (header: inpaint_kind == base)."""
        gate.require("cap.image_cn_inpaint")
        if p.get("arch") != "inpaint":
            raise HTTPException(422, "controlnet_inpaint vale solo con arch=inpaint (FLUX.1 base)")
        from core import arch_filter as af
        if af.inpaint_kind(p.get("model") or "") != "base":
            raise HTTPException(422, f"controlnet_inpaint richiede un FLUX.1 BASE (non Fill/Klein): «{p.get('model')}»")
        _weights_409(["flux-cn-inpaint-alimama"], "ControlNet Inpainting (Alimama)")
        q = {"_module": "image_control", "mode": "flux_cn_inpaint", "arch": "flux", "model": p["model"],
             "prompt": p["prompt"], "image_path": p["image_path"], "mask_path": p["mask_path"],
             "steps": p.get("steps", 28), "guidance": min(float(p.get("guidance", 3.5)), 6.0), "seed": p.get("seed", 0),
             "control_scale": p.get("control_scale", 0.9), "outpaint": outpaint}
        if p.get("negative"):
            q["negative"] = p["negative"]
        return q

    @app.post("/image/inpaint", tags=["immagini"])
    def image_inpaint(req: InpaintReq):
        gate.require("cap.inpaint")
        p = req.model_dump(exclude_none=True)
        _flux3_arch(p)
        _need_model(p)
        if p["arch"] == "flux3":
            p["init_image"] = p.pop("image_path")                 # modifica per istruzione + composito con maschera binarizzata
            _flux3(p, req)
        elif p["arch"] == "qwen_image21":
            _qwen21_inpaint(p, req)
        elif p.pop("controlnet_inpaint", False):
            p = _cn_inpaint(p, False)
        else:
            p.pop("control_scale", None)
        return _submit("image.inpaint", lambda job: run_image(p))

    @app.post("/image/outpaint", tags=["immagini"])
    def image_outpaint(req: OutpaintReq):
        gate.require("cap.inpaint")
        p = req.model_dump(exclude_none=True)
        _flux3_arch(p)
        _need_model(p)
        if p["arch"] == "flux3":
            raise HTTPException(422, "FLUX.3 Klein non ha outpaint: usa /image/edit con le immagini o /image/inpaint (composito)")
        if p["arch"] == "qwen_image21":                           # l'immagine arriva già allargata, maschera = bordo
            _qwen21_inpaint(p, req)
            p["outpaint"] = True                                  # il worker lo usa per riempimento/tono (task 14)
        elif p.pop("controlnet_inpaint", False):
            p = _cn_inpaint(p, True)
        else:
            p.pop("control_scale", None)
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
