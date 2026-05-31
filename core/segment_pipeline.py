"""Segmentazione oggetti via ComfyUI. ID numerici.

Detector supportati (selezionabili da GUI):

- "grounding_dino" — RACCOMANDATO. Usa comfyui_segment_anything (GroundingDINO + SAM-HQ
  in un solo nodo `GroundingDinoSAMSegment`). Robusto su prompt arbitrari, non richiede
  Impact-Pack CLIPSeg. SAM-HQ ritorna direttamente MASK.

- "clipseg" — legacy. CLIPSegDetectorProvider (richiede ComfyUI-CLIPSeg di biegert) →
  BboxDetectorSEGS → SAMDetectorCombined (Impact Pack). Mantenuto per retro-compatibilità.

- REGIONE (click o box dell'utente, "lazzo magnetico"):
    LoadImageMask(box/punto) → MaskToSEGS → SAMDetectorCombined(image) → MASK

Tutti i rami salvano la MASK come PNG (MaskToImage→SaveImage), che l'app riscarica e
carica nel layer maschera del canvas.
"""
from __future__ import annotations


# Modelli hardcoded dei loader di comfyui_segment_anything (vedi node.py)
GROUNDING_DINO_DEFAULT = "GroundingDINO_SwinB (938MB)"
SAM_HQ_DEFAULT = "sam_hq_vit_l (1.25GB)"


class SegmentPipeline:
    def __init__(self, sam_model: str | None = None):
        self.sam_model = sam_model      # nome file in models/sams (es. 'sam_vit_l_0b3195.pth')

    # ---------- selezione da testo (dispatcher) ----------
    def build_text(self, image_file: str, text: str, *, detector: str = "grounding_dino",
                   sam_model: str | None = None, threshold: float = 0.3,
                   blur: float = 7.0, dilation: int = 4,
                   detection_hint: str = "center-1",
                   grounding_dino_model: str = GROUNDING_DINO_DEFAULT,
                   sam_hq_model: str = SAM_HQ_DEFAULT) -> dict:
        d = (detector or "grounding_dino").lower()
        if d == "grounding_dino_only":
            return self._build_text_grounding(image_file, text, threshold=threshold,
                                              grounding_dino_model=grounding_dino_model,
                                              sam_hq_model=sam_hq_model, use_sam=False)
        if d == "clipseg":
            # legacy CLIPSeg — solo se richiesto ESPLICITAMENTE (richiede il nodo
            # CLIPSegDetectorProvider di biegert, NON installato di default).
            return self._build_text_clipseg(image_file, text, sam_model=sam_model,
                                            clipseg_threshold=threshold, blur=blur,
                                            dilation=dilation, detection_hint=detection_hint)
        # Tutto il resto (grounding_dino, dino, e anche "sam_only": SAM da solo NON
        # sa selezionare da testo, serve un detector) → GroundingDINO + SAM-HQ.
        return self._build_text_grounding(image_file, text, threshold=threshold,
                                          grounding_dino_model=grounding_dino_model,
                                          sam_hq_model=sam_hq_model, use_sam=True)

    # ---------- ramo GroundingDINO + SAM-HQ (preferito) ----------
    def _build_text_grounding(self, image_file: str, text: str, *,
                              threshold: float = 0.5,
                              grounding_dino_model: str = GROUNDING_DINO_DEFAULT,
                              sam_hq_model: str = SAM_HQ_DEFAULT,
                              use_sam: bool = True) -> dict:
        """GroundingDinoSAMSegment: GroundingDINO trova le bbox dal testo, SAM-HQ rifinisce
        il contorno. `threshold` è il box_threshold di GroundingDINO: valori bassi (<0.3)
        producono molte bbox che, unite da SAM, coprono ~tutta l'immagine → alziamo il
        default a 0.5. `use_sam=False` resta come hook (la pipeline comfyui_segment_anything
        espone solo il nodo combinato, quindi attualmente equivale a use_sam=True)."""
        p: dict = {}
        p["1"] = {"class_type": "LoadImage", "inputs": {"image": image_file}}
        p["2"] = {"class_type": "SAMModelLoader (segment anything)",
                  "inputs": {"model_name": sam_hq_model}}
        p["3"] = {"class_type": "GroundingDinoModelLoader (segment anything)",
                  "inputs": {"model_name": grounding_dino_model}}
        p["4"] = {"class_type": "GroundingDinoSAMSegment (segment anything)",
                  "inputs": {"sam_model": ["2", 0], "grounding_dino_model": ["3", 0],
                             "image": ["1", 0], "prompt": text or "object",
                             "threshold": float(threshold)}}
        # output (IMAGE, MASK) → indice 1 è la MASK
        self._save_mask(p, ["4", 1])
        return p

    # ---------- ramo CLIPSeg + SAM (legacy) ----------
    def _build_text_clipseg(self, image_file: str, text: str, *, sam_model: str | None = None,
                            clipseg_threshold: float = 0.4, blur: float = 7.0, dilation: int = 4,
                            detection_hint: str = "center-1") -> dict:
        sam = sam_model or self.sam_model
        p: dict = {}
        p["1"] = {"class_type": "LoadImage", "inputs": {"image": image_file}}
        p["2"] = {"class_type": "CLIPSegDetectorProvider",
                  "inputs": {"text": text or "object", "blur": float(blur),
                             "threshold": float(clipseg_threshold), "dilation_factor": int(dilation)}}
        p["3"] = {"class_type": "BboxDetectorSEGS",
                  "inputs": {"bbox_detector": ["2", 0], "image": ["1", 0], "threshold": 0.5,
                             "dilation": 10, "crop_factor": 3.0, "drop_size": 10, "labels": "all"}}
        mask_ref = self._segs_to_mask(p, segs=["3", 0], image=["1", 0], sam=sam,
                                      detection_hint=detection_hint)
        self._save_mask(p, mask_ref)
        return p

    # ---------- selezione da regione (punto/box) ----------
    def build_region(self, image_file: str, region_mask_file: str, *,
                     sam_model: str | None = None, detection_hint: str = "center-1",
                     threshold: float = 0.50) -> dict:
        """region_mask_file: PNG bianco-su-nero col punto (cerchietto) o il riquadro disegnato.
        threshold: soglia maschera SAM (0=tutto l'oggetto … 1=solo il nucleo). Più BASSA = selezione
        più ampia. Era hardcoded a 0.93 (→ solo una macchia uniforme); ora viene dalla UI."""
        sam = sam_model or self.sam_model
        if not sam:
            raise RuntimeError("Selezione oggetto a click/box richiede un modello SAM in models/sams.")
        if "hq" in sam.lower():
            # SAMLoader di Impact-Pack ricava il tipo dal nome ('vit_l' → costruisce un SAM
            # vanilla) e poi carica i pesi: con un checkpoint SAM-HQ il state_dict NON combacia
            # (l'HQ ha layer extra) → errore. Il puntatore deve usare un SAM vanilla.
            raise RuntimeError(
                f"Il modello «{sam}» è un SAM-HQ: non è compatibile con la selezione a "
                "click/box (SAMLoader di Impact-Pack carica solo SAM vanilla). "
                "Scegli un SAM vanilla (es. sam_vit_b / sam_vit_l) nel menu «Modello SAM», "
                "oppure usa la selezione «da testo» (GroundingDINO usa SAM-HQ automaticamente).")
        p: dict = {}
        p["1"] = {"class_type": "LoadImage", "inputs": {"image": image_file}}
        p["8"] = {"class_type": "LoadImageMask",
                  "inputs": {"image": region_mask_file, "channel": "red"}}
        # crop_factor più ampio: dal click (punto piccolo) SAM deve vedere abbastanza intorno
        # per segmentare l'INTERO oggetto, non solo i pixel sotto il cursore.
        p["9"] = {"class_type": "MaskToSEGS",
                  "inputs": {"mask": ["8", 0], "combined": True, "crop_factor": 5.0,
                             "bbox_fill": False, "drop_size": 10, "contour_fill": False}}
        mask_ref = self._segs_to_mask(p, segs=["9", 0], image=["1", 0], sam=sam,
                                      detection_hint=detection_hint, threshold=threshold)
        self._save_mask(p, mask_ref)
        return p

    # ---------- comuni ----------
    def _segs_to_mask(self, p: dict, segs, image, sam: str | None, detection_hint: str,
                      threshold: float = 0.50):
        if sam:
            p["4"] = {"class_type": "SAMLoader",
                      "inputs": {"model_name": sam, "device_mode": "AUTO"}}
            p["5"] = {"class_type": "SAMDetectorCombined",
                      "inputs": {"sam_model": ["4", 0], "segs": segs, "image": image,
                                 "detection_hint": detection_hint, "dilation": 0,
                                 "threshold": float(threshold), "bbox_expansion": 0,
                                 "mask_hint_threshold": 0.7, "mask_hint_use_negative": "False"}}
            return ["5", 0]
        # fallback senza SAM: maschera grezza dalle SEGS
        p["5"] = {"class_type": "SegsToCombinedMask", "inputs": {"segs": segs}}
        return ["5", 0]

    def _save_mask(self, p: dict, mask_ref):
        p["6"] = {"class_type": "MaskToImage", "inputs": {"mask": mask_ref}}
        p["7"] = {"class_type": "SaveImage",
                  "inputs": {"images": ["6", 0], "filename_prefix": "ltxstudio_segmask"}}
