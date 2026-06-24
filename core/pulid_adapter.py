"""Face adapter PuLID-FLUX nativo: imprime l'identità di un volto (da foto) sull'immagine FLUX.

PuLID (Pure and Lightning ID customization): vincola SOLO il volto, lascia posa/scena/abito al
prompt (a differenza di InstantID che fissa anche la posa). Maturo su FLUX, leggero, alta somiglianza.

Componenti (lib già installate: facexlib, insightface, timm, onnxruntime-gpu):
- Face encoder: InsightFace **antelopev2** (det + ArcFace) → crop+embedding identità; **EVA-CLIP** → feature.
  (antelopev2 + eva-clip vengono scaricati automaticamente da insightface/facexlib al primo uso.)
- Adapter: `E:\\AI\\Models\\pulid\\pulid_flux_v0.9.1.safetensors` (cross-attention ID, ~1GB, in download).

Integrazione: in `inference_server/backends/image_flux.py`, per arch flux/kontext, se i params hanno
`pulid_face` (path foto) + `pulid_strength`, si applica l'adapter al transformer FLUX prima del denoising.
GUI: gruppo "Adattatore volto (PuLID)" in `ui/tabs/frame1_tab.py` (gen_pulid_enable/face/strength) + i18n 15 lingue.

Riferimento implementazione: repo ToTheBeginning/PuLID `pulid/pipeline_flux.py` (estrazione id_embedding via
insightface+eva_clip, poi `model.pulid_ca` cross-attention con scala = pulid_strength). Va adattato a
diffusers `FluxTransformer2DModel` agganciando i blocchi attention (hook) con l'id embedding.
"""
from __future__ import annotations

import os

PULID_WEIGHTS = r"E:\AI\Models\pulid\pulid_flux_v0.9.1.safetensors"


def available() -> bool:
    """True se i pesi PuLID e le lib sono presenti."""
    if not os.path.exists(PULID_WEIGHTS):
        return False
    try:
        import insightface  # noqa: F401
        import facexlib  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# TODO (al completamento del download dei pesi):
#  1. _get_face_encoder(): insightface.app.FaceAnalysis(name="antelopev2") + EVA-CLIP (facexlib).
#  2. extract_id(face_image_path) -> id_embedding (ArcFace + eva-clip features).
#  3. load_pulid_adapter(transformer): carica PULID_WEIGHTS, inserisce i moduli pulid_ca nel FLUX transformer.
#  4. apply(pipe, face_image_path, strength): estrae l'embedding e configura l'adapter con la scala.
#  Poi cablaggio in image_flux.run (param pulid_face/pulid_strength) + main_window (legge gen_pulid_*).
