"""Utility condivise eseguite DENTRO Blender (usano `import bpy`).

Importate da convert.py / mesh_ops.py / retarget.py aggiungendo questa cartella
a sys.path. Forniscono: parsing args JSON dopo `--`, emissione del risultato
marcato `@@RESULT@@{...}`, pulizia scena, e import/export universale per
estensione/formato (glTF/GLB, FBX, OBJ, DAE, STL, PLY, USD/USDZ, ABC, BVH, X3D,
DXF). Tutto difensivo: prova l'operatore "nuovo" (Blender 4.x) e ricade su
quello legacy; segnala con chiarezza i formati non supportati dalla build.
"""
from __future__ import annotations

import json
import sys

import bpy  # type: ignore  # disponibile solo dentro Blender

RESULT_MARKER = "@@RESULT@@"

# Estensione di input -> lista di operatori d'import da tentare in ordine.
# Ogni voce è (funzione, kwargs_extra). filepath viene aggiunto a runtime.
IMPORT_OPS = {
    ".glb":  [("import_scene", "gltf", {})],
    ".gltf": [("import_scene", "gltf", {})],
    ".fbx":  [("import_scene", "fbx", {})],
    ".obj":  [("wm", "obj_import", {}), ("import_scene", "obj", {})],
    ".dae":  [("wm", "collada_import", {})],
    ".stl":  [("wm", "stl_import", {}), ("import_mesh", "stl", {})],
    ".ply":  [("wm", "ply_import", {}), ("import_mesh", "ply", {})],
    ".usd":  [("wm", "usd_import", {})],
    ".usda": [("wm", "usd_import", {})],
    ".usdc": [("wm", "usd_import", {})],
    ".usdz": [("wm", "usd_import", {})],
    ".abc":  [("wm", "alembic_import", {})],
    ".bvh":  [("import_anim", "bvh", {})],
    ".x3d":  [("import_scene", "x3d", {})],
    ".dxf":  [("import_scene", "dxf", {})],
}

# Formato target -> (estensione, lista operatori d'export da tentare).
EXPORT_OPS = {
    "glb":  (".glb",  [("export_scene", "gltf", {"export_format": "GLB"})]),
    "gltf": (".gltf", [("export_scene", "gltf", {"export_format": "GLTF_SEPARATE"})]),
    "fbx":  (".fbx",  [("export_scene", "fbx", {})]),
    "obj":  (".obj",  [("wm", "obj_export", {}), ("export_scene", "obj", {})]),
    "dae":  (".dae",  [("wm", "collada_export", {})]),
    "stl":  (".stl",  [("wm", "stl_export", {}), ("export_mesh", "stl", {})]),
    "ply":  (".ply",  [("wm", "ply_export", {}), ("export_mesh", "ply", {})]),
    "usd":  (".usd",  [("wm", "usd_export", {})]),
    "usda": (".usda", [("wm", "usd_export", {})]),
    "usdc": (".usdc", [("wm", "usd_export", {})]),
    "usdz": (".usdz", [("wm", "usd_export", {})]),
    "abc":  (".abc",  [("wm", "alembic_export", {})]),
    "bvh":  (".bvh",  [("export_anim", "bvh", {})]),
    "x3d":  (".x3d",  [("export_scene", "x3d", {})]),
    "dxf":  (".dxf",  [("export_scene", "dxf", {})]),
}

# Add-on da abilitare al volo per certi formati (non sempre attivi di default).
_ADDONS_FOR_FORMAT = {
    "dxf": ["io_import_dxf", "io_export_dxf"],
}
_ADDONS_FOR_EXT = {
    ".dxf": ["io_import_dxf"],
    ".usd": ["io_scene_usd"], ".usda": ["io_scene_usd"],
    ".usdc": ["io_scene_usd"], ".usdz": ["io_scene_usd"],
}


def parse_args() -> dict:
    """Args JSON passati dopo `--` sulla command line di Blender."""
    argv = sys.argv
    if "--" in argv:
        payload = argv[argv.index("--") + 1:]
        if payload:
            return json.loads(payload[0])
    return {}


def emit_result(ok: bool, **extra) -> None:
    """Stampa la riga di risultato che blender_service.py parsa."""
    out = {"ok": ok}
    out.update(extra)
    print(RESULT_MARKER + json.dumps(out), flush=True)


def _enable_addons(names) -> None:
    import addon_utils  # type: ignore
    for name in names:
        try:
            addon_utils.enable(name, default_set=False, persistent=False)
        except Exception:
            pass


def clear_scene() -> None:
    """Scena vuota (rimuove ogni oggetto e datablock orfano)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _call_op(group: str, name: str, **kwargs):
    op_group = getattr(bpy.ops, group)
    op = getattr(op_group, name)
    return op(**kwargs)


def import_file(path: str) -> None:
    """Importa `path` deducendo l'operatore dall'estensione."""
    import os
    ext = os.path.splitext(path)[1].lower()
    ops = IMPORT_OPS.get(ext)
    if not ops:
        raise RuntimeError(f"Formato di import non supportato: {ext}")
    _enable_addons(_ADDONS_FOR_EXT.get(ext, []))
    last_err = None
    for group, name, extra in ops:
        try:
            _call_op(group, name, filepath=path, **extra)
            return
        except Exception as e:  # noqa: BLE001 - prova il prossimo operatore
            last_err = e
    raise RuntimeError(f"Import fallito ({ext}): {last_err}")


def export_scene(path: str, fmt: str, options: dict | None = None) -> str:
    """Esporta l'intera scena in `path` nel formato `fmt`. Ritorna il path finale."""
    import os
    options = options or {}
    fmt = fmt.lower().lstrip(".")
    spec = EXPORT_OPS.get(fmt)
    if not spec:
        raise RuntimeError(f"Formato di export non supportato: {fmt}")
    ext, ops = spec
    if not path.lower().endswith(ext):
        path = os.path.splitext(path)[0] + ext
    _enable_addons(_ADDONS_FOR_FORMAT.get(fmt, []))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    last_err = None
    for group, name, extra in ops:
        try:
            kwargs = dict(extra)
            kwargs["filepath"] = path
            _apply_export_options(group, name, kwargs, options)
            _call_op(group, name, **kwargs)
            return path
        except Exception as e:  # noqa: BLE001 - prova il prossimo operatore
            last_err = e
    raise RuntimeError(f"Export fallito ({fmt}): {last_err}")


def _apply_export_options(group: str, name: str, kwargs: dict, options: dict) -> None:
    """Aggiunge opzioni comuni (scala, animazioni) solo se l'operatore le supporta."""
    op = getattr(getattr(bpy.ops, group), name)
    try:
        props = op.get_rna_type().properties.keys()
    except Exception:
        props = []
    scale = options.get("scale")
    if scale and "global_scale" in props:
        kwargs["global_scale"] = float(scale)
    # FBX: include animazioni e tutto il contenuto
    if name == "fbx":
        if "bake_anim" in props:
            kwargs["bake_anim"] = bool(options.get("animations", True))
        if "path_mode" in props:
            kwargs["path_mode"] = "COPY"
        if "embed_textures" in props:
            kwargs["embed_textures"] = bool(options.get("embed_textures", True))
    # glTF: embed o separati
    if name == "gltf":
        if "export_animations" in props:
            kwargs["export_animations"] = bool(options.get("animations", True))


def has_armature() -> bool:
    """True se nella scena esiste almeno un'armatura (scheletro)."""
    return any(o.type == "ARMATURE" for o in bpy.data.objects)


def armature_objects():
    return [o for o in bpy.data.objects if o.type == "ARMATURE"]
