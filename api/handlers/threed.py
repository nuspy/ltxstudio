"""Endpoint 3D / RIG / SKINNING / MONDO / CONVERSIONE — servizi nativi (venv isolati + Blender).

image_to_3d (TRELLIS) · rig (scheletro+skinning, SkinToken) · world (scena 3D, WorldGen) · convert (GLB→FBX/OBJ).
NB: mocap (live webcam) e retarget/animazione Blender restano nella GUI (interattivi) — non esposti qui.
"""
from __future__ import annotations

import glob
import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api import gate
from api.jobs import MANAGER


def _submit(kind, fn):
    return {"job_id": MANAGER.submit(kind, fn), "status": "pending"}


def _find(dir_path: str, exts: tuple[str, ...]) -> list[str]:
    out = []
    for e in exts:
        out += glob.glob(os.path.join(dir_path, f"**/*{e}"), recursive=True)
    return sorted(out, key=os.path.getmtime, reverse=True)


class ImageTo3DReq(BaseModel):
    image_path: str
    name: str = "model"
    seed: int = 1
    ss_steps: int = 12
    slat_steps: int = 12
    simplify: float = 0.95
    texture_size: int = 1024
    hf_token: str | None = None


class RigReq(BaseModel):
    mesh_path: str
    name: str = "rig"
    template: str = Field("Mixamo", description="template scheletro")
    file_format: str = Field("glb", description="glb | fbx")


class WorldReq(BaseModel):
    name: str = "world"
    image_path: str = Field("", description="foto di partenza (vuoto = testo→scena)")
    prompt: str = ""
    resolution: int = 1600
    low_vram: bool = False
    mesh: bool = Field(True, description="True = mesh GLB, False = splat PLY")
    hf_token: str | None = None


class ConvertReq(BaseModel):
    glb_path: str
    fmt: str = Field("fbx", description="fbx | obj")


def register(app: FastAPI):

    @app.post("/threed/image_to_3d", tags=["3d"])
    def image_to_3d(req: ImageTo3DReq):
        gate.require("cap.threed")
        def fn(job):
            from core import trellis_service as ts
            errs = ts.preflight(req.image_path)
            if errs:
                raise RuntimeError("; ".join(errs) + " — installa con scripts/install_trellis.py")
            j = ts.make_job(req.image_path, req.name, seed=req.seed, ss_steps=req.ss_steps,
                            slat_steps=req.slat_steps, simplify=req.simplify,
                            texture_size=req.texture_size, hf_token=req.hf_token, on_log=job.log)
            j.run()
            d = str(ts.OUT_DIR / ts._safe(req.name))
            return _find(d, (".glb",))
        return _submit("threed.image_to_3d", fn)

    @app.post("/threed/rig", tags=["3d"])
    def rig(req: RigReq):
        gate.require("cap.threed")
        def fn(job):
            from core import rig_service as rs
            _errs = rs.preflight(req.mesh_path)             # preflight richiede il mesh_path
            if _errs:
                raise RuntimeError("; ".join(_errs))
            j = rs.make_job(req.mesh_path, req.name, template=req.template,
                            file_format=req.file_format, on_log=job.log)
            j.run()
            base = os.path.join(str(getattr(rs, "OUT_DIR", "output/3d")), req.name)
            return _find(base, (f".{req.file_format}", ".glb", ".fbx"))
        return _submit("threed.rig", fn)

    @app.post("/threed/world", tags=["3d"])
    def world(req: WorldReq):
        gate.require("cap.threed")
        def fn(job):
            from core import world_service as ws
            p = ws.WorldParams(name=req.name, image_path=req.image_path, prompt=req.prompt,
                               resolution=req.resolution, low_vram=req.low_vram)
            _errs = ws.preflight(p)                          # una sola chiamata (era invocata 2×)
            if _errs:
                raise RuntimeError("; ".join(_errs))
            j = ws.make_job(p, mesh=req.mesh, hf_token=req.hf_token, on_log=job.log)
            j.run()
            base = os.path.join(str(getattr(ws, "OUT_ROOT", "output/worlds")), req.name)
            return _find(base, (".glb", ".ply"))
        return _submit("threed.world", fn)

    @app.post("/threed/convert", tags=["3d"])
    def convert(req: ConvertReq):
        gate.require("cap.threed")
        def fn(job):
            from pathlib import Path
            from core import world_service as ws
            out, msg = ws.export_mesh(Path(req.glb_path), fmt=req.fmt)
            if not out:
                raise RuntimeError(msg or "conversione fallita")
            return [str(out)]
        return _submit("threed.convert", fn)
