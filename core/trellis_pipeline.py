"""Pipeline per generazione 3D da immagine — TRELLIS.2 (Microsoft).

Da una SINGOLA immagine genera una mesh 3D con materiali PBR (base color,
roughness, metallic, opacity), esportata in GLB/OBJ/PLY/STL.

Catena di nodi ComfyUI (custom node `ComfyUI-TRELLIS2`, repo PozzettiAndrea):
    LoadImage
      → Trellis2RemoveBackground        (rimozione sfondo BiRefNet → mask)   [opzionale]
    LoadTrellis2Models                   (config: risoluzione/precisione/attn)
      → Trellis2GetConditioning          (DinoV3 features dall'immagine+mask)
      → Trellis2ImageToShape             (struttura sparsa → mesh + shape_slat)
    [ramo TEXTURE]
      → Trellis2ShapeToTexturedMesh      (voxelgrid PBR)
      → Trellis2Simplify                 (decimazione mesh)
      → Trellis2UVUnwrap                 (atlas UV)
      → Trellis2RasterizePBR             (bake texture PBR su UV)
      → Trellis2ExportTrimesh            (GLB)
    [ramo GEOMETRIA]
      → Trellis2Simplify
      → Trellis2ExportTrimesh

Output: <ComfyUI>/output/<prefix>_*.glb  (poi recuperato dal main_window).

NB: i pesi (~25GB, microsoft/TRELLIS.2-4B) si scaricano automaticamente al primo
run dentro ComfyUI. Servono ~24GB di VRAM consigliati (12GB minimo @512).
"""
from __future__ import annotations

# Modalità risoluzione esposte dal loader (vedi nodes_loader.RESOLUTION_MODES)
RESOLUTION_MODES = ["512", "1024", "1024_cascade", "1536_cascade"]
PRECISION_MODES = ["auto", "bf16", "fp16", "fp32"]
ATTN_BACKENDS = ["auto", "flash_attn", "xformers", "sdpa", "sageattn"]
EXPORT_FORMATS = ["glb", "obj", "ply", "stl", "3mf", "dae"]
BACKGROUND_COLORS = ["black", "gray", "white"]

DEFAULT_PREFIX = "ltxstudio_trellis"


class TrellisPipeline:
    """Costruisce il prompt-API ComfyUI per TRELLIS.2 (immagine → mesh 3D)."""

    def build(
        self,
        image_name: str,
        *,
        # ── modello ───────────────────────────────────────────────────────
        resolution: str = "1024_cascade",
        precision: str = "auto",
        attn_backend: str = "auto",
        # ── preprocess ────────────────────────────────────────────────────
        remove_background: bool = True,
        background_color: str = "black",
        # ── geometria (Trellis2ImageToShape) ──────────────────────────────
        seed: int = 0,
        ss_guidance_strength: float = 6.5,
        ss_guidance_rescale: float = 0.05,
        ss_sampling_steps: int = 12,
        shape_guidance_strength: float = 6.5,
        shape_guidance_rescale: float = 0.05,
        shape_sampling_steps: int = 12,
        max_tokens: int = 49152,
        # ── mesh (Trellis2Simplify / UVUnwrap) ────────────────────────────
        target_face_count: int = 500000,
        fill_holes: bool = True,
        # ── texture (ramo PBR) ────────────────────────────────────────────
        texture: bool = True,
        tex_guidance_strength: float = 3.0,
        tex_guidance_rescale: float = 0.20,
        tex_sampling_steps: int = 12,
        texture_size: int = 2048,
        # ── export ────────────────────────────────────────────────────────
        file_format: str = "glb",
        filename_prefix: str = DEFAULT_PREFIX,
    ) -> dict:
        """Ritorna il dizionario prompt-API per ComfyUI.

        Args:
            image_name:   nome file immagine caricata nella input di ComfyUI.
            resolution:   '512' | '1024' | '1024_cascade' | '1536_cascade'.
            remove_background: se True, BiRefNet separa il soggetto (consigliato
                          per immagini generate su sfondo).
            seed:         seed (riproducibilità geometria + texture).
            ss_/shape_guidance_strength: aderenza della struttura/forma all'immagine.
            max_tokens:   budget token della struttura sparsa = densità geometria
                          (16384–262144). Più alto = più dettaglio + VRAM/tempo.
            target_face_count: decimazione mesh finale.
            texture:      se True genera + cuoce i materiali PBR; altrimenti
                          esporta la sola geometria (mesh grigia).
            texture_size: lato della texture PBR bake (512–16384).
            file_format:  formato export (glb consigliato, materiali inclusi).
        """
        if resolution not in RESOLUTION_MODES:
            resolution = "1024_cascade"
        if file_format not in EXPORT_FORMATS:
            file_format = "glb"
        if background_color not in BACKGROUND_COLORS:
            background_color = "black"

        p: dict = {}

        # 1. Carica immagine
        p["1"] = {"class_type": "LoadImage",
                  "inputs": {"image": image_name}}

        # 2. Rimozione sfondo (BiRefNet) → image + mask, oppure usa alpha di LoadImage
        if remove_background:
            p["2"] = {"class_type": "Trellis2RemoveBackground",
                      "inputs": {"image": ["1", 0], "low_vram": True}}
            cond_image = ["2", 0]
            cond_mask = ["2", 1]
        else:
            cond_image = ["1", 0]
            cond_mask = ["1", 1]   # MASK dall'alpha dell'immagine (tutto-FG se opaca)

        # 3. Config modello TRELLIS.2 (scarica i pesi al primo run)
        p["3"] = {"class_type": "LoadTrellis2Models",
                  "inputs": {"resolution": resolution,
                             "precision": precision,
                             "attn_backend": attn_backend}}

        # 4. Conditioning DinoV3 (preprocess + crop + feature)
        p["4"] = {"class_type": "Trellis2GetConditioning",
                  "inputs": {"model_config": ["3", 0],
                             "image": cond_image,
                             "mask": cond_mask,
                             "background_color": background_color}}

        # 5. Immagine → struttura/forma 3D (mesh grezza + shape_slat + subs)
        p["5"] = {"class_type": "Trellis2ImageToShape",
                  "inputs": {"model_config": ["3", 0],
                             "conditioning": ["4", 0],
                             "seed": int(seed),
                             "ss_guidance_strength": float(ss_guidance_strength),
                             "ss_guidance_rescale": float(ss_guidance_rescale),
                             "ss_sampling_steps": int(ss_sampling_steps),
                             "shape_guidance_strength": float(shape_guidance_strength),
                             "shape_guidance_rescale": float(shape_guidance_rescale),
                             "shape_sampling_steps": int(shape_sampling_steps),
                             "max_tokens": int(max_tokens)}}

        if texture:
            # 6. Forma → voxelgrid PBR (texture latente)
            p["6"] = {"class_type": "Trellis2ShapeToTexturedMesh",
                      "inputs": {"model_config": ["3", 0],
                                 "conditioning": ["4", 0],
                                 "shape_slat": ["5", 1],
                                 "subs": ["5", 2],
                                 "seed": int(seed),
                                 "tex_guidance_strength": float(tex_guidance_strength),
                                 "tex_guidance_rescale": float(tex_guidance_rescale),
                                 "tex_sampling_steps": int(tex_sampling_steps)}}
            # 7. Decimazione mesh
            p["7"] = {"class_type": "Trellis2Simplify",
                      "inputs": {"trimesh": ["5", 0],
                                 "target_face_count": int(target_face_count),
                                 "fill_holes": bool(fill_holes),
                                 "fill_holes_perimeter": 0.03,
                                 "remesh": False,
                                 "remesh_band": 1.0}}
            # 8. UV unwrap (necessario per il bake texture)
            p["8"] = {"class_type": "Trellis2UVUnwrap",
                      "inputs": {"trimesh": ["7", 0],
                                 "chart_cone_angle": 90.0,
                                 "chart_refine_iterations": 0,
                                 "chart_global_iterations": 1,
                                 "chart_smooth_strength": 1}}
            # 9. Bake PBR su atlas UV
            p["9"] = {"class_type": "Trellis2RasterizePBR",
                      "inputs": {"trimesh": ["8", 0],
                                 "voxelgrid": ["6", 0],
                                 "original_mesh": ["5", 0],
                                 "texture_size": int(texture_size)}}
            export_src = ["9", 0]
        else:
            # 7. Sola geometria → decimazione → export
            p["7"] = {"class_type": "Trellis2Simplify",
                      "inputs": {"trimesh": ["5", 0],
                                 "target_face_count": int(target_face_count),
                                 "fill_holes": bool(fill_holes),
                                 "fill_holes_perimeter": 0.03,
                                 "remesh": False,
                                 "remesh_band": 1.0}}
            export_src = ["7", 0]

        # 10. Export su file (scrive in <ComfyUI>/output/<prefix>_*.<fmt>)
        p["10"] = {"class_type": "Trellis2ExportTrimesh",
                   "inputs": {"trimesh": export_src,
                              "filename_prefix": filename_prefix,
                              "file_format": file_format}}
        return p
