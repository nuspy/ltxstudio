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
    # Default SENSATI per velocità: decimation 1M + texture 2048 + remesh = postprocess lentissimo.
    # 150k facce e texture 1024 danno un GLB ottimo ed esportano in pochi secondi (override da GUI).
    ap.add_argument("--texture_size", type=int, default=1024)
    ap.add_argument("--decimation", type=int, default=150000)     # target facce (era 1_000_000)
    ap.add_argument("--simplify_cap", type=int, default=2000000)  # cap pre-to_glb (era 16.7M = nessuno)
    # Preferisci i pesi LOCALI (scaricati via scripts/download_trellis2_curl.py) → niente cache HF/ridownload.
    _local = r"E:\AI\Tools\TRELLIS2\models\TRELLIS.2-4B"
    _default_model = _local if os.path.isdir(_local) and os.path.exists(os.path.join(_local, "pipeline.json")) \
        else "microsoft/TRELLIS.2-4B"
    ap.add_argument("--model", default=_default_model)
    a = ap.parse_args()

    # LIBERA la VRAM dei server GPU (immagini :8189, api :8200) PRIMA di caricare TRELLIS-4B: col
    # ModelManager che ora tiene i modelli residenti, una generazione 3D dopo un'immagine trovava la VRAM
    # piena → OOM al load = "crash all'avvio". stdlib-only (il venv TRELLIS non ha `core`).
    import urllib.request as _u
    for _port in (8189, 8200):
        try:
            _u.urlopen(_u.Request(f"http://127.0.0.1:{_port}/free", data=b"{}",
                                  headers={"Content-Type": "application/json"}), timeout=15).read()
            print(f"[trellis] VRAM server :{_port} liberata", flush=True)
        except Exception:                                    # noqa: BLE001 (server giù = niente da liberare)
            pass

    import time
    from PIL import Image
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(a.model)
    pipeline.cuda()
    print(f"[trellis] load {time.time()-t0:.0f}s", flush=True)

    image = Image.open(a.image).convert("RGB")
    torch.manual_seed(a.seed)
    t1 = time.time()
    mesh = pipeline.run(image)[0]
    print(f"[trellis] diffusion {time.time()-t1:.0f}s", flush=True)
    t2 = time.time()
    mesh.simplify(int(a.simplify_cap))                            # cap facce prima del postprocess
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(a.decimation), texture_size=int(a.texture_size),
        remesh=True, remesh_band=1, remesh_project=0, verbose=False)
    print(f"[trellis] to_glb (dec={a.decimation},tex={a.texture_size}) {time.time()-t2:.0f}s", flush=True)

    out = os.path.join(a.output_dir, "model.glb")
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    glb.export(out, extension_webp=True)
    print("OK:" + out, flush=True)


if __name__ == "__main__":
    main()
