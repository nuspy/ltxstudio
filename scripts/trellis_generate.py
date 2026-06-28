"""Runner TRELLIS.2 nativo (immagine→mesh GLB) — codice MIT (ComfyUI-TRELLIS2) via shim, niente ComfyUI.

Chiamato da core/trellis_service.py. Stampa `OK:<glb>` a fine.
"""
import argparse
import os
import sys
from pathlib import Path

_TR = r"E:\AI\Tools\TRELLIS2"
sys.path.insert(0, str(Path(__file__).resolve().parent))   # trellis_comfy_shim
sys.path.insert(0, _TR)                                     # trellis_nodes (codice MIT copiato)
_FB = r"E:\AI\Tools\ffmpeg-shared\bin"
if os.path.isdir(_FB):
    os.environ["PATH"] = _FB + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_FB)
    except (AttributeError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ss_steps", type=int, default=12)
    ap.add_argument("--slat_steps", type=int, default=12)
    ap.add_argument("--simplify", type=float, default=0.95)
    ap.add_argument("--texture_size", type=int, default=1024)
    ap.add_argument("--resolution", default="1024_cascade",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--bg", default="black")
    a = ap.parse_args()

    import numpy as np
    import torch
    import trimesh
    from PIL import Image

    from trellis_comfy_shim import install_shim
    install_shim(_TR, "cuda" if torch.cuda.is_available() else "cpu")
    from trellis_nodes import stages   # codice MIT

    im = Image.open(a.image).convert("RGB")
    arr = np.asarray(im, np.float32) / 255.0
    image = torch.from_numpy(arr)[None]                       # [1,H,W,3] formato IMAGE
    mask = torch.ones((1, arr.shape[0], arr.shape[1]), dtype=torch.float32)

    model_config = {"resolution": a.resolution, "dtype": torch.bfloat16, "attn_backend": "auto"}
    include_1024 = a.resolution in ("1024", "1024_cascade", "1536_cascade")

    conditioning, _prep = stages.run_conditioning(model_config, image, mask,
                                                  include_1024=include_1024, background_color=a.bg)
    verts, faces, _slat, _subs = stages.run_shape_generation(conditioning, seed=a.seed)

    out = os.path.join(a.output_dir, "model.glb")
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    v = verts.detach().cpu().numpy() if hasattr(verts, "detach") else np.asarray(verts)
    f = faces.detach().cpu().numpy() if hasattr(faces, "detach") else np.asarray(faces)
    trimesh.Trimesh(vertices=v, faces=f).export(out)
    print("OK:" + out, flush=True)


if __name__ == "__main__":
    main()
