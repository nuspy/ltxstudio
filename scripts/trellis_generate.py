"""Generazione 3D headless con TRELLIS (microsoft/TRELLIS) — eseguito nel venv E:\\AI\\Tools\\TRELLIS2.

immagine → mesh GLB (con texture PBR). Nessun ComfyUI: usa direttamente l'API ufficiale TRELLIS.
Chiamato da core/trellis_service.py via subprocess. Stampa `OK:<glb>` a fine.

API ufficiale (github.com/microsoft/TRELLIS):
  TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large").run(image)
  postprocessing_utils.to_glb(gaussian, mesh) -> trimesh -> .export(glb)
"""
import argparse
import os
import sys

# il package `trellis` (repo microsoft/TRELLIS) è usato via sys.path (non pip-installato)
_REPO = r"E:\AI\Tools\TRELLIS2\TRELLIS"
if os.path.isdir(_REPO):
    sys.path.insert(0, _REPO)
# DLL FFmpeg shared (imageio/torchcodec)
_FB = r"E:\AI\Tools\ffmpeg-shared\bin"
if os.path.isdir(_FB):
    os.environ["PATH"] = _FB + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_FB)
    except (AttributeError, OSError):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ss_steps", type=int, default=12)
    ap.add_argument("--slat_steps", type=int, default=12)
    ap.add_argument("--simplify", type=float, default=0.95)
    ap.add_argument("--texture_size", type=int, default=1024)
    a = ap.parse_args()

    # backend attention/sparse-conv (TRELLIS): flash-attn + spconv se disponibili
    os.environ.setdefault("ATTN_BACKEND", "flash-attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()

    image = Image.open(a.image).convert("RGB")
    outputs = pipe.run(
        image,
        seed=a.seed,
        sparse_structure_sampler_params={"steps": a.ss_steps, "cfg_strength": 7.5},
        slat_sampler_params={"steps": a.slat_steps, "cfg_strength": 3.0},
    )
    out = os.path.join(a.output_dir, "model.glb")
    try:
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0], outputs["mesh"][0],
            simplify=a.simplify, texture_size=a.texture_size)
        glb.export(out)
    except Exception as e:  # noqa: BLE001
        # texture PBR richiede nvdiffrast (build MSVC su Windows). Fallback: shape-mesh senza texture.
        print(f"[trellis] texture non disponibile ({type(e).__name__}: {str(e)[:80]}) → export shape-mesh", flush=True)
        import trimesh
        m = outputs["mesh"][0]
        verts = m.vertices.detach().cpu().numpy() if hasattr(m.vertices, "detach") else m.vertices
        faces = m.faces.detach().cpu().numpy() if hasattr(m.faces, "detach") else m.faces
        trimesh.Trimesh(vertices=verts, faces=faces).export(out)
    print("OK:" + out, flush=True)


if __name__ == "__main__":
    main()
