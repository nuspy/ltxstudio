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

# TIMEOUT DI SICUREZZA sul Blender headless: sktn._run_headless_blender fa subprocess.run SENZA timeout → se
# l'export si impianta (mesh troppo densa) resta appeso all'INFINITO (visto: 24 min). Monkeypatchiamo
# subprocess.run per iniettare un timeout di default (kill del figlio allo scadere → errore pulito, niente hang).
import subprocess as _sp
_orig_run = _sp.run
def _run_with_timeout(*_a, **_kw):
    _kw.setdefault("timeout", int(os.environ.get("RIG_BLENDER_TIMEOUT", "420") or 420))
    return _orig_run(*_a, **_kw)
_sp.run = _run_with_timeout

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
    ap.add_argument("--parent-pid", type=int, default=0)          # watchdog: se l'app muore, ci autodistruggiamo
    a = ap.parse_args()

    try:                                                          # anti-orfano (vedi scripts/_watchdog.py)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _watchdog import start_watchdog
        start_watchdog(a.parent_pid)
    except Exception:                                             # noqa: BLE001
        pass

    # LIBERA la VRAM dei server GPU (immagini :8189, api :8200) prima di caricare i pesi 3D — evita l'OOM
    # "crash all'avvio" quando il ModelManager tiene i modelli residenti dopo una generazione immagine.
    import urllib.request as _u
    for _port in (8189, 8190, 8191, 8192):                    # tutti i server GPU (8200 non ha /free)
        try:                                                  # categoria EFFIMERA → destroy ESPLICITO
            _u.urlopen(_u.Request(f"http://127.0.0.1:{_port}/free", data=b'{"mode": "destroy"}',
                                  headers={"Content-Type": "application/json"}), timeout=15).read()
        except Exception:
            pass

    sktn = _load_node()
    print("[rig] sktn_nodes importato", flush=True)
    import trimesh as Trimesh
    mesh = Trimesh.load(a.mesh, force="mesh")
    print(f"[rig] mesh in: {len(mesh.vertices)} vert / {len(mesh.faces)} facce", flush=True)
    # DECIMAZIONE anti-hang: una mesh molto densa (es. 467k facce da TRELLIS) fa IMPANTANARE l'export Blender
    # (remove_doubles/gltf) → il rig restava appeso all'infinito. La portiamo a ~100k facce (qualità del rig
    # invariata, export in pochi secondi). Env RIG_MAX_FACES (0 = disattiva).
    _maxf = int(os.environ.get("RIG_MAX_FACES", "120000") or 0)
    if _maxf and len(mesh.faces) > _maxf:
        try:
            _tr = max(0.05, min(0.95, 1.0 - (100000.0 / len(mesh.faces))))
            _dec = mesh.simplify_quadric_decimation(_tr)
            if _dec is not None and len(_dec.faces) > 100:
                print(f"[rig] mesh decimata {len(mesh.faces)}→{len(_dec.faces)} facce (evita l'hang dell'export)",
                      flush=True)
                mesh = _dec
        except Exception as _de:                              # noqa: BLE001
            print(f"[rig] decimazione saltata ({_de}) — export potenzialmente lento", flush=True)

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
