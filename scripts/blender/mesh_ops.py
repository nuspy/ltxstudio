"""Operazioni mesh sul modello 3D (eseguite dentro Blender headless).

Operazioni (campo "op"):
  - "list":          elenca gli oggetti (nome, tipo, facce, vertici, dimensioni)
  - "separate_loose": separa ogni mesh per parti sciolte (connected components)
  - "delete":        cancella gli oggetti indicati in "names"
  - "join":          unisce tutte le mesh in un solo oggetto

Args JSON (dopo `--`):
  {"op": "...", "input": "<file>", "output": "<file opz>", "format": "<opz>",
   "names": ["..."]}
Le operazioni che modificano la geometria esportano in "output" (formato dedotto
dall'estensione di output o dal campo "format"). "list" non esporta.
Emette `@@RESULT@@{"ok": true, "objects": [...], "output": "<path>"}`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bl_common as B  # noqa: E402
import bpy  # type: ignore  # noqa: E402


def _list_objects() -> list:
    out = []
    for o in bpy.data.objects:
        info = {"name": o.name, "type": o.type}
        if o.type == "MESH" and o.data:
            info["faces"] = len(o.data.polygons)
            info["verts"] = len(o.data.vertices)
            info["dims"] = [round(float(d), 4) for d in o.dimensions]
        out.append(info)
    return out


def _deselect_all() -> None:
    for o in bpy.data.objects:
        o.select_set(False)


def _separate_loose() -> None:
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    for obj in meshes:
        _deselect_all()
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.separate(type="LOOSE")
        finally:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass


def _delete(names: list) -> None:
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def _join() -> None:
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if len(meshes) < 2:
        return
    _deselect_all()
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()


def main() -> None:
    try:
        args = B.parse_args()
        op = args["op"]
        inp = args["input"]
        if not os.path.isfile(inp):
            raise RuntimeError(f"File di input inesistente: {inp}")
        B.clear_scene()
        B.import_file(inp)

        if op == "list":
            B.emit_result(True, objects=_list_objects())
            return

        if op == "separate_loose":
            _separate_loose()
        elif op == "delete":
            _delete(args.get("names", []))
        elif op == "join":
            _join()
        else:
            raise RuntimeError(f"Operazione sconosciuta: {op}")

        # Esporta il risultato modificato.
        out = args.get("output") or inp
        fmt = args.get("format") or os.path.splitext(out)[1].lstrip(".") or "glb"
        final = B.export_scene(out, fmt, args.get("options", {}))
        B.emit_result(True, output=final, objects=_list_objects())
    except Exception as e:  # noqa: BLE001
        B.emit_result(False, error=str(e))


main()
