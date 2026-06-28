"""Runner TRELLIS.2 nativo (immagine→mesh GLB) — repo UFFICIALE microsoft/TRELLIS.2 (MIT), ZERO ComfyUI.

API ufficiale: Trellis2ImageTo3DPipeline.run(image) → mesh; o_voxel.postprocess.to_glb → GLB con PBR.
Chiamato da core/trellis_service.py. Stampa `OK:<glb>` a fine.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO = r"E:\AI\Tools\TRELLIS2\TRELLIS2_official"
if os.path.isdir(_REPO):
    sys.path.insert(0, _REPO)
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
    ap.add_argument("--texture_size", type=int, default=2048)
    ap.add_argument("--decimation", type=int, default=1000000)   # target facce
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    a = ap.parse_args()

    from PIL import Image
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(a.model)
    pipeline.cuda()

    image = Image.open(a.image).convert("RGB")
    torch.manual_seed(a.seed)
    mesh = pipeline.run(image)[0]
    mesh.simplify(16777216)                                       # limite nvdiffrast

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=a.decimation, texture_size=a.texture_size,
        remesh=True, remesh_band=1, remesh_project=0, verbose=False)

    out = os.path.join(a.output_dir, "model.glb")
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    glb.export(out, extension_webp=True)
    print("OK:" + out, flush=True)


if __name__ == "__main__":
    main()
