"""Generazione headless di una scena 3D con WorldGen (senza il server viser interattivo).

Usa l'API di WorldGen e salva l'output:
  - mesh mode  → <output_dir>/mesh.glb   (GLB, apribile in Blender/engine/model-viewer)
  - splat mode → <output_dir>/splat.ply  (3D Gaussian Splatting)

Eseguito nel venv di WorldGen dalla tab «Mondo 3D».

Uso:
  python worldgen_generate.py --output_dir DIR [--image IMG] [--prompt P]
                              [--mesh] [--resolution 1600] [--low_vram] [--use_sharp]
"""
from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image", default=None, help="immagine di partenza (i2s); assente = t2s")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--mesh", action="store_true", help="mesh mode (GLB); default = splat (.ply)")
    ap.add_argument("--resolution", type=int, default=1600)
    ap.add_argument("--low_vram", action="store_true")
    ap.add_argument("--use_sharp", action="store_true")
    ap.add_argument("--parent-pid", type=int, default=0)          # watchdog: se l'app muore, ci autodistruggiamo
    args = ap.parse_args()

    try:                                                          # anti-orfano (vedi scripts/_watchdog.py)
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from _watchdog import start_watchdog
        start_watchdog(args.parent_pid)
    except Exception:                                             # noqa: BLE001
        pass

    # LIBERA la VRAM dei server GPU (immagini :8189, api :8200) prima di caricare i pesi 3D — evita l'OOM
    # "crash all'avvio" quando il ModelManager tiene i modelli residenti dopo una generazione immagine.
    import urllib.request as _u
    for _port in (8189, 8200, 8190):                          # M1/M2: libera anche il server video
        try:
            _u.urlopen(_u.Request(f"http://127.0.0.1:{_port}/free", data=b"{}",
                                  headers={"Content-Type": "application/json"}), timeout=15).read()
        except Exception:
            pass

    os.makedirs(args.output_dir, exist_ok=True)
    import torch
    from worldgen import WorldGen

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "i2s" if args.image else "t2s"
    print(f"[worldgen] mode={mode} mesh={args.mesh} res={args.resolution} device={device}", flush=True)

    wg = WorldGen(mode=mode, use_sharp=args.use_sharp, inpaint_bg=False,
                  resolution=args.resolution, device=device, low_vram=args.low_vram)

    image = None
    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")

    if image is not None:
        scene = wg.generate_world(args.prompt, image, return_mesh=args.mesh)
    else:
        scene = wg.generate_world(args.prompt, return_mesh=args.mesh)

    if args.mesh:
        import open3d as o3d
        out = os.path.join(args.output_dir, "mesh.glb")
        o3d.io.write_triangle_mesh(out, scene)
    else:
        out = os.path.join(args.output_dir, "splat.ply")
        scene.save(out)

    print(f"[worldgen] saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
