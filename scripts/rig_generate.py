"""Rigging headless SkinTokens/TokenRig — eseguito nel venv E:\\AI\\Tools\\SkinTokens.

mesh (GLB/OBJ) → GLB rigged (scheletro + skin weights). Chiamato da core/rig_service.py.
Stampa `OK:<glb>`. Sostituisce il custom node ComfyUI-SkinToken (zero ComfyUI runtime).

NB: l'API esatta di SkinTokens va allineata al repo al primo install (la firma di
predict/auto-rig varia tra versioni). La struttura (carica → predici rig → esporta) è qui.
"""
import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_dir", default=os.environ.get("SKINTOKENS_MODEL",
                                                          r"E:\AI\Models\3D\SkinTokens"))
    a = ap.parse_args()

    # API repo VAST-AI/SkinTokens (TokenRig): carica il modello, predice scheletro + skin weights,
    # applica e salva il rig. (firma esatta da verificare contro la versione installata)
    from skin_tokens.inference import auto_rig  # type: ignore  # noqa: F401

    glb = auto_rig(mesh_path=a.mesh, model_dir=a.model_dir, output_path=a.output)
    print("OK:" + (glb or a.output), flush=True)


if __name__ == "__main__":
    main()
