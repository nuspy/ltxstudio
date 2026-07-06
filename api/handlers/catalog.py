"""Endpoint CATALOGO/FEATURE per il client — listing modelli, componenti AUTO, LoRA, e stato feature-matrix.

- GET /features   → stato attivazione/disattivazione (identico al desktop) + upsell per capability.
- GET /models     → modelli disponibili per task + LoRA + modello consigliato (auto).
- GET /components → componenti AUTO per un modello scelto (arch/vae/text-encoder/steps/cfg/LoRA compatibili).
- GET /loras      → LoRA (tutte o compatibili con un modello).
Read-only e sincroni (nessun job). Riusa core/offline_inventory, core/auto_select, core/model_detect.
"""
from __future__ import annotations

from fastapi import FastAPI, Query

from api import gate
from core import auto_select
from core import model_detect
from core import offline_inventory as inv
from core.offline_inventory import MODELS_ROOT

# task app → categoria inventario
_TASK_CAT = {"image": "diffusion_models", "video": "diffusion_models",
             "upscale": "upscale_models", "restore": "facerestore"}


def _resolve(model: str, *cats: str) -> str | None:
    """Path assoluto di un modello (nome relativo) cercandolo nelle sottocartelle delle categorie."""
    leaf = model.replace("/", "\\").split("\\")[-1]
    for cat in cats:
        for sub in inv._CATS.get(cat, ([], ()))[0]:
            base = MODELS_ROOT / sub
            direct = base / model
            if direct.exists():
                return str(direct)
            if base.exists():
                hit = list(base.rglob(leaf))
                if hit:
                    return str(hit[0])
    return None


def register(app: FastAPI):

    @app.get("/features", tags=["catalogo"])
    def features_state():
        """Stato del feature matrix (STESSE attivazioni della GUI desktop) — l'app gate le funzioni con questo."""
        return gate.snapshot()

    @app.get("/models", tags=["catalogo"])
    def models(task: str = Query("image", description="image | video | upscale | restore")):
        cat = _TASK_CAT.get(task, "diffusion_models")
        mods = list(inv.list_models(cat))
        if task in ("image", "video"):
            mods += inv.list_models("checkpoints")
        auto = None
        try:
            auto = auto_select.auto_config("video" if task == "video" else "image")
        except Exception:                                    # noqa: BLE001
            pass
        return {"task": task, "models": sorted(set(mods)),
                "loras": inv.list_models("loras"), "recommended": auto}

    @app.get("/components", tags=["catalogo"])
    def components(model: str = Query(..., description="nome file del modello scelto"),
                   task: str = Query("image")):
        """Componenti AUTO per il modello scelto — cuore della 'selezione automatica'."""
        path = _resolve(model, _TASK_CAT.get(task, "diffusion_models"), "checkpoints")
        comp = {}
        if path:
            try:
                comp = model_detect.components_for_path(path) or {}
            except Exception:                                # noqa: BLE001
                comp = {}
        try:
            settings = auto_select.settings_for(model)
        except Exception:                                    # noqa: BLE001
            settings = {}
        return {
            "model": model, "path": path,
            "arch": comp.get("family"), "variant": comp.get("variant"),
            "vae": comp.get("vae"), "text_encoders": comp.get("text_encoders", []),
            "steps": settings.get("steps"), "cfg": settings.get("cfg"),
            "loras_compatible": auto_select.compatible_loras(model),
        }

    @app.get("/loras", tags=["catalogo"])
    def loras(model: str | None = Query(None, description="se dato, solo le LoRA compatibili")):
        if model:
            return {"loras": auto_select.compatible_loras(model)}
        return {"loras": inv.list_models("loras")}
