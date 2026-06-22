"""Pipeline per rigging automatico — SkinTokens / TokenRig (VAST-AI).

Da una mesh 3D (.glb/.obj) genera automaticamente **scheletro + pesi di skinning**
in un'unica sequenza autoregressiva (skin token), esportando un GLB/FBX rigged.
Successore di UniRig: +98-133% accuratezza skinning, +17-22% bone prediction.

Catena di nodi ComfyUI:
    Trellis2LoadMesh (path .glb → TRIMESH)
      → SkinTokenRigTrimesh (scheletro + skinning + export GLB/FBX)

Wrapper: custom node `ComfyUI-SkinToken` (Rizzlord), modello VAST-AI/SkinTokens.
NB: su Python 3.12 l'export passa SEMPRE da Blender headless → richiede Blender 4.0+
installato (SKINTOKEN_BLENDER_BIN o su PATH). Pesi auto-scaricati da HuggingFace
(~4GB VRAM). Backbone Qwen3-0.6B.
"""
from __future__ import annotations

# Checkpoint TokenRig di default (vedi sktn_nodes.DEFAULT_TOKENRIG_CKPT)
DEFAULT_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"

# Etichette template scheletro (label esposte dal nodo, non le chiavi interne)
SKELETON_TEMPLATES = ["Keep model names", "Mixamo", "Unreal Engine 5"]
EXPORT_FORMATS = ["glb", "fbx"]

DEFAULT_PREFIX = "3D/ltxstudio_rig"


class RigPipeline:
    """Costruisce il prompt-API ComfyUI per il rigging SkinTokens."""

    def build(
        self,
        mesh_path: str,
        *,
        ckpt_name: str = DEFAULT_CKPT,
        file_format: str = "glb",
        skeleton_template: str = "Keep model names",
        use_transfer: bool = False,
        use_postprocess: bool = False,
        group_per_vertex: int = 4,
        bottom_center_origin: bool = False,
        smooth_angle: float = 55.0,
        # ── campionamento autoregressivo ──────────────────────────────────
        top_k: int = 5,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 2.0,
        num_beams: int = 10,
        filename_prefix: str = DEFAULT_PREFIX,
    ) -> dict:
        """Ritorna il dizionario prompt-API per ComfyUI.

        Args:
            mesh_path:    path ASSOLUTO al file mesh (.glb/.obj) sul server ComfyUI.
            file_format:  'glb' (anteprima nel visore) | 'fbx' (Mixamo/engine).
            skeleton_template: rinomina le ossa — 'Mixamo' per retarget animazioni
                          Mixamo, 'Unreal Engine 5' per il Mannequin UE5.
            use_transfer: trasferisce il rig predetto sulla mesh ad alto dettaglio
                          originale (consigliato per mesh dettagliate).
            use_postprocess: post-processing voxel-based dei pesi di skinning.
            group_per_vertex: max influenze ossee per vertice (deformazioni più
                          morbide a costo maggiore).
            bottom_center_origin: origine scheletro ai piedi (workflow game engine).
            num_beams / temperature: più beam + temperatura bassa = scheletro più
                          deterministico.
        """
        if file_format not in EXPORT_FORMATS:
            file_format = "glb"
        if skeleton_template not in SKELETON_TEMPLATES:
            skeleton_template = "Keep model names"

        p: dict = {}

        # 1. Carica la mesh come TRIMESH (accetta path assoluto)
        p["1"] = {"class_type": "Trellis2LoadMesh",
                  "inputs": {"mesh_path": mesh_path}}

        # 2. Rigging SkinTokens: scheletro + skinning + export
        p["2"] = {"class_type": "SkinTokenRigTrimesh",
                  "inputs": {
                      "trimesh": ["1", 0],
                      "ckpt_name": ckpt_name,
                      "device": "auto",
                      "save_file": True,
                      "filename_prefix": filename_prefix,
                      "file_format": file_format,
                      "use_transfer": bool(use_transfer),
                      "use_postprocess": bool(use_postprocess),
                      "group_per_vertex": int(group_per_vertex),
                      "bottom_center_origin": bool(bottom_center_origin),
                      "smooth_angle": float(smooth_angle),
                      "skeleton_template": skeleton_template,
                      "top_k": int(top_k),
                      "top_p": float(top_p),
                      "temperature": float(temperature),
                      "repetition_penalty": float(repetition_penalty),
                      "num_beams": int(num_beams),
                  }}

        # 3. Nodo terminale di output: SkinTokenRigTrimesh non è un OUTPUT_NODE,
        #    quindi ComfyUI non eseguirebbe il grafo ("prompt_no_outputs").
        #    Preview3D (output node) riceve il path del file rigged e rende valido il grafo.
        p["3"] = {"class_type": "Preview3D",
                  "inputs": {"model_file": ["2", 1]}}   # ["2",1] = rigged_path (STRING)
        return p
