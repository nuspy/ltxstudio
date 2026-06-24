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


# STRUTTURA PESI (verificata): 312 chiavi = 20 moduli `pulid_ca.0..19`, ognuno norm1/norm2/to_q/to_kv/
# to_out (cross-attention ID). Si agganciano ai blocchi del FluxTransformer: dopo l'attention del
# blocco i, `hidden = hidden + strength * pulid_ca[i](id_embedding, hidden)`.
#
# CODICE DI RIFERIMENTO (clonato): E:\AI\Tools\PuLID
#  - pulid/encoders.py + encoders_transformer.py : IDEncoder/IDFormer (id_embedding finale)
#  - pulid/pipeline_flux.py : get_id_embedding() (insightface antelopev2 + eva_clip) + come aggancia pulid_ca
#  - pulid/attention_processor.py : PuLIDAttnProcessor (la cross-attention)
#  - eva_clip/ : EVA-CLIP incluso nel repo (no download)
#  NB il repo usa il SUO flux (cartella flux/), NON diffusers → l'integrazione va PORTATA al
#  diffusers FluxTransformer2DModel (i nostri image_flux usano diffusers).
#
# TODO core (tutto pronto: lib+pesi+codice+eva_clip):
#  1. extract_id(face_path): sys.path PuLID → from pulid.pipeline_flux import ... ; FaceAnalysis(antelopev2)
#     + eva_clip → id_embedding (tensor). antelopev2 auto-download da insightface al 1° uso.
#  2. load_pulid_ca(): carica PULID_WEIGHTS → 20 moduli pulid_ca (nn.Module), su device/dtype del transformer.
#  3. attach(transformer, id_embedding, strength): forward_hook sui transformer_blocks (double) del
#     FluxTransformer2DModel diffusers → applica pulid_ca[i] all'output, scalato da strength.
#  4. detach(): rimuove gli hook (per generazioni senza PuLID).
#  Cablaggio: image_flux._load/run legge params pulid_face/pulid_strength (arch flux/kontext) →
#  extract_id + load_pulid_ca + attach prima del denoising, detach dopo. main_window.on_generate
#  legge gen_pulid_enable/face/strength da frame1_tab e li mette nei params.
