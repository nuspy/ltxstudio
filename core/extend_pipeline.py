"""Long video / Prolunga video: estensione LTX-2.3 AUTOREGRESSIVA e memory-bounded.

Metodo (ispirato al workflow di riferimento "LTX-2.3 Long Video" e migliorato con il context
latent IAMCCS): il video viene esteso a CHUNK. Per ogni segmento:
  1. si prende la CODA (ultimi N frame) del video accumulato finora (ImageBatchExtendWithOverlap
     → start_images, senza calcolo indici);
  2. la si inietta come contesto latente (IAMCCS_LTX2_ContextLatent) in un latent vuoto;
  3. si genera il nuovo spezzone (IAMCCS_LTX2_ExtendSampler) con overlap temporale;
  4. si DECODIFICA subito il segmento (VAEDecode) → VRAM bounded per segmento;
  5. si concatena al video accumulato con blending dell'overlap (ImageBatchExtendWithOverlap).

Poiché ogni segmento viene decodificato e poi si lavora in spazio IMMAGINE (non si tiene un
unico latent gigante in VRAM), la lunghezza è di fatto ILLIMITATA: bastano più segmenti.

Componenti LTX-2.3 condivisi col golden Pluto per coerenza visiva (stesso modello/encoder/VAE),
+ NAG e LTXVScheduler come nel resto dell'app (vedi quality upgrade).
"""
from __future__ import annotations

LTX = {
    # -1.1: il file SENZA "-1.1" non esiste più (solo .metadata.json) → UNETLoader "not in list".
    # Resta fp8 distilled, coerente con la distilled-lora-1.1 sotto.
    "unet": "LTXVideo\\v2\\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
    "distilled_lora": "LTX-2\\ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
    "video_vae": "LTX23_video_vae_bf16.safetensors",
    "clip1": "gemma_3_12B_it.safetensors",
    "clip2": "ltx-2.3_text_projection_bf16.safetensors",
}

# Limite di sicurezza: oltre questo numero di segmenti avvisiamo (tempo/disco), non blocchiamo.
SOFT_MAX_SEGMENTS = 60

# Regola di segmentazione (richiesta utente):
# - durata da aggiungere <= SINGLE_PASS_MAX_S → UN UNICO spezzone (niente giunzioni → niente
#   segmento nero da stitch);
# - oltre → spezzoni da SPLIT_SEG_S con sovrapposizione (overlap_frames) per la continuità.
SINGLE_PASS_MAX_S = 20.0
SPLIT_SEG_S = 18.0


def to_8np1(n: int) -> int:
    """LTX richiede length = 8n+1."""
    n = int(round(n))
    return max(9, ((n - 1) // 8) * 8 + 1)


class ExtendPipeline:
    def __init__(self):
        self.unet = LTX["unet"]

    def ensure_unet(self, available: list[str] | None):
        """Corregge self.unet se il default non è tra gli UNET installati (es. file rinominato):
        preferisce un distilled fp8 LTX-2.3 presente, così l'extend non fallisce la validazione."""
        avail = list(available or [])
        if not avail or self.unet in avail:
            return
        def score(n: str) -> int:
            nl = n.lower()
            s = 0
            if "ltx" not in nl or "2.3" not in nl and "2-3" not in nl: return -1
            if "distilled" in nl: s += 4
            if "fp8" in nl: s += 2
            if "transformer_only" in nl: s += 2
            if "1.1" in nl: s += 1
            if "eros" in nl or "gguf" in nl: s -= 3
            return s
        best = max(avail, key=score, default=None)
        if best is not None and score(best) >= 0:
            self.unet = best

    def segment_count(self, add_seconds: float, seg_seconds: float) -> int:
        """Numero di segmenti per coprire `add_seconds` a chunk di `seg_seconds`."""
        seg_seconds = max(1.0, float(seg_seconds))
        import math
        return max(1, math.ceil(max(1.0, add_seconds) / seg_seconds))

    def plan_segments(self, add_seconds: float,
                      split_seg_s: float | None = None) -> tuple[int, float]:
        """Pianifica (n_segmenti, secondi_per_segmento) secondo la regola:
        - `add_seconds` <= SINGLE_PASS_MAX_S (20s): 1 spezzone della durata piena (niente giunzioni);
        - oltre: spezzoni da `split_seg_s` (default 18s) CON sovrapposizione (overlap_frames).
        """
        import math
        add = max(1.0, float(add_seconds))
        seg = float(split_seg_s or SPLIT_SEG_S)
        seg = max(2.0, min(seg, SINGLE_PASS_MAX_S))
        if add <= SINGLE_PASS_MAX_S:
            return 1, add
        return max(2, math.ceil(add / seg)), seg

    def build(self, video_name: str, prompt: str, *, width: int = 768, height: int = 768,
              fps: int = 24, add_seconds: float = 5.0, seg_seconds: float = 4.0,
              segments: int | None = None,
              overlap_frames: int = 8, overlap_strength: float = 0.5,
              context_frames: int = 16, steps: int = 8, cfg: float = 1.0,
              seed: int = 0, negative: str = "", nag: bool = True,
              out_format: str = "video/h264-mp4",
              unet: str | None = None, sampler: str = "euler_ancestral_cfg_pp",
              base_shift: float = 0.95, max_shift: float = 2.05,
              clip1: str | None = None, use_distilled_lora: bool | None = None) -> dict:
        # I parametri sopra EREDITANO dalla passata base (stesso modello/sampler/step/cfg/encoder/
        # shift) → l'estensione continua nello STESSO spazio visivo: senza questo, base=dev-fp8/30step
        # ed extend=distilled/8step davano incoerenza e RUMORE. La continuità IAMCCS regge solo se il
        # modello che genera è lo stesso che ha prodotto i frame di contesto.
        unet = unet or self.unet
        clip1 = clip1 or LTX["clip1"]
        # numero di segmenti: esplicito o calcolato dalla durata target (illimitato)
        n_seg = int(segments) if segments else self.segment_count(add_seconds, seg_seconds)
        n_seg = max(1, n_seg)
        per_seg = to_8np1(max(1.0, seg_seconds) * fps)            # frame nuovi per segmento
        context_lat = max(1, round(context_frames / 8))           # ctx in LATENT frames (8x pixel)
        overlap_stitch = max(1, context_lat * 8)                  # pixel frame riprodotti dal ctx

        p: dict = {}
        # ── modello LTX-2.3 (ereditato dalla base) ──
        p["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
        model_base = ["1", 0]
        # distilled-LoRA SOLO sui modelli distilled (sul dev-fp8 la base NON la usa → niente LoRA,
        # altrimenti si rompe la coerenza). None = auto in base al nome del modello.
        apply_lora = ("distilled" in unet.lower()) if use_distilled_lora is None else bool(use_distilled_lora)
        if apply_lora:
            p["2"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": ["1", 0], "lora_name": LTX["distilled_lora"], "strength_model": 1.0}}
            model_base = ["2", 0]
        p["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": LTX["video_vae"]}}
        vae = ["3", 0]
        p["4"] = {"class_type": "DualCLIPLoader",
                  "inputs": {"clip_name1": clip1, "clip_name2": LTX["clip2"], "type": "ltxv"}}
        p["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt or "", "clip": ["4", 0]}}
        p["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative or "", "clip": ["4", 0]}}
        # NAG: avvolge il model usando il negativo come guida (qualità)
        guider_model = model_base
        if nag:
            p["nag"] = {"class_type": "LTX2_NAG",
                        "inputs": {"model": model_base, "nag_scale": 11.0, "nag_alpha": 0.25,
                                   "nag_tau": 2.5, "nag_cond_video": ["6", 0],
                                   "nag_cond_audio": ["6", 0], "inplace": True}}
            guider_model = ["nag", 0]
        p["7"] = {"class_type": "CFGGuider",
                  "inputs": {"model": guider_model, "positive": ["5", 0], "negative": ["6", 0], "cfg": float(cfg)}}
        p["8"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
        p["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
        p["10"] = {"class_type": "LTXVScheduler",
                   "inputs": {"steps": int(steps), "max_shift": float(max_shift), "base_shift": float(base_shift),
                              "stretch": True, "terminal": 0.1}}
        # ── video di input (fotogrammi) ──
        p["11"] = {"class_type": "VHS_LoadVideo",
                   "inputs": {"video": video_name, "force_rate": 0, "custom_width": 0,
                              "custom_height": 0, "frame_load_cap": 0, "skip_first_frames": 0,
                              "select_every_nth": 1, "format": "AnimateDiff"}}
        acc = ["11", 0]   # video accumulato (parte dai frame in input)

        # ── catena di segmenti (autoregressiva, memory-bounded) ──
        for i in range(n_seg):
            sid = f"s{i}"
            # coda del video accumulato → contesto (start_images = source[-overlap:])
            p[f"{sid}_tail"] = {"class_type": "ImageBatchExtendWithOverlap",
                                "inputs": {"source_images": acc, "overlap": overlap_stitch,
                                           "overlap_side": "source", "overlap_mode": "linear_blend"}}
            tail = [f"{sid}_tail", 1]      # slot 1 = start_images (ultimi overlap_stitch frame)
            # latent vuoto per il nuovo segmento (nuovo + contesto)
            p[f"{sid}_empty"] = {"class_type": "EmptyLTXVLatentVideo",
                                 "inputs": {"width": int(width), "height": int(height),
                                            "length": to_8np1(per_seg + overlap_stitch), "batch_size": 1}}
            # inietta il contesto latente dalla coda
            p[f"{sid}_ctx"] = {"class_type": "IAMCCS_LTX2_ContextLatent",
                               "inputs": {"previous_video": tail, "vae": vae,
                                          "latent": [f"{sid}_empty", 0], "enable": True,
                                          "context_latent_frames": int(context_lat),
                                          "exclude_last_frame": False,
                                          "context_strength": float(overlap_strength)}}
            # genera il nuovo spezzone
            p[f"{sid}_ext"] = {"class_type": "IAMCCS_LTX2_ExtendSampler",
                               "inputs": {"model": guider_model, "vae": vae, "noise": ["8", 0],
                                          "sampler": ["9", 0], "sigmas": ["10", 0], "guider": ["7", 0],
                                          "latents": [f"{sid}_ctx", 0], "num_new_frames": int(per_seg),
                                          "temporal_overlap": int(overlap_frames),
                                          "temporal_overlap_cond_strength": float(overlap_strength)}}
            # decodifica subito (VRAM bounded)
            p[f"{sid}_dec"] = {"class_type": "VAEDecode",
                               "inputs": {"samples": [f"{sid}_ext", 0], "vae": vae}}
            # concatena al video accumulato con blending dell'overlap
            p[f"{sid}_stitch"] = {"class_type": "ImageBatchExtendWithOverlap",
                                  "inputs": {"source_images": acc, "new_images": [f"{sid}_dec", 0],
                                             "overlap": overlap_stitch, "overlap_side": "source",
                                             "overlap_mode": "linear_blend"}}
            acc = [f"{sid}_stitch", 2]     # slot 2 = extended_images (video più lungo)

        # ── salva il video finale (illimitato) ──
        p["out"] = {"class_type": "VHS_VideoCombine",
                    "inputs": {"images": acc, "frame_rate": int(fps), "loop_count": 0,
                               "filename_prefix": "ltxstudio_longvideo", "format": out_format,
                               "pingpong": False, "save_output": True}}
        return p
