"""Rig NATIVO via SkinToken come LIBRERIA (no ComfyUI server).

Riusa il node ComfyUI-SkinToken (validato) caricandolo come package Python e chiamando direttamente
`_run_skin_token` (inferenza scheletro+skinning) + `_export_asset` (export armatura via Blender headless).
Gira nel venv ComfyUI (Comfy_Nexus.bak) che ha torch+deps+le dipendenze skintokens; Blender 5.1 per l'export.

Uso: python rig_native_skintoken.py --mesh IN.glb --output OUT.glb [--template Mixamo|Keep|UE5]
                                    [--format glb|fbx] [--device cuda]
"""
from __future__ import annotations
import argparse, os, sys, importlib.util, time, traceback
from pathlib import Path

COMFY = os.environ.get("SKINTOKEN_COMFY_ROOT", r"E:\AI\Comfy_Nexus.bak\ComfyUI")
SKDIR = os.path.join(COMFY, "custom_nodes", "ComfyUI-SkinToken")


def _load_node():
    """Carica ComfyUI-SkinToken come package (gestendo il trattino) → relative import .vendor ok."""
    sys.path.insert(0, COMFY)                                  # comfy.model_management, folder_paths
    sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
    pkg = "ComfyUI_SkinToken_lib"
    spec = importlib.util.spec_from_file_location(
        pkg, os.path.join(SKDIR, "__init__.py"), submodule_search_locations=[SKDIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module(pkg + ".sktn_nodes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--template", default="Mixamo")           # Mixamo | Keep | UE5
    ap.add_argument("--format", default="glb")                # glb | fbx
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--group_per_vertex", type=int, default=4)
    a = ap.parse_args()

    sktn = _load_node()
    print("[rig] sktn_nodes importato", flush=True)
    import trimesh as Trimesh
    mesh = Trimesh.load(a.mesh, force="mesh")
    print(f"[rig] mesh in: {len(mesh.vertices)} vert", flush=True)

    ckpt = sktn.DEFAULT_TOKENRIG_CKPT
    sktn._ensure_required_models(ckpt)                        # scarica grpo/VAE/Qwen al primo uso
    print("[rig] modelli pronti, inferenza skin-token…", flush=True)
    t0 = time.time()
    asset = sktn._run_skin_token(
        mesh=mesh, ckpt_name=ckpt, device_name=a.device,
        top_k=a.top_k, top_p=a.top_p, temperature=a.temperature,
        repetition_penalty=a.repetition_penalty, num_beams=a.num_beams,
        use_postprocess=True, skeleton_template=a.template)
    print(f"[rig] inferenza {time.time()-t0:.0f}s → export Blender…", flush=True)

    Path(os.path.dirname(a.output)).mkdir(parents=True, exist_ok=True)
    backend = sktn._export_asset(
        asset=asset, output_path=Path(a.output), use_transfer=True,
        group_per_vertex=a.group_per_vertex, bottom_center_origin=False, smooth_angle=55.0)
    sz = os.path.getsize(a.output) if os.path.exists(a.output) else 0
    print(f"OK:{a.output} backend={backend} size={sz}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
