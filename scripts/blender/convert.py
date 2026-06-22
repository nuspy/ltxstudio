"""Convertitore di formato 3D (eseguito dentro Blender headless).

Uso (via core/blender_service.run_blender_job):
  blender --background --python convert.py -- '{"input": "...", "output": "...",
                                               "format": "fbx", "options": {...}}'
Importa l'input (formato dedotto dall'estensione), esporta nel formato target.
Emette `@@RESULT@@{"ok": true, "output": "<path finale>"}`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bl_common as B  # noqa: E402


def main() -> None:
    try:
        args = B.parse_args()
        inp = args["input"]
        out = args["output"]
        fmt = args["format"]
        options = args.get("options", {})
        if not os.path.isfile(inp):
            raise RuntimeError(f"File di input inesistente: {inp}")
        B.clear_scene()
        B.import_file(inp)
        final = B.export_scene(out, fmt, options)
        B.emit_result(True, output=final)
    except Exception as e:  # noqa: BLE001
        B.emit_result(False, error=str(e))


main()
