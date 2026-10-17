"""Voci di catalogo per i pesi GIÀ SUL DISCO usati dall'app o da integrare (pulizia modelli del 14/10/2026).

Ogni voce punta al file dove sta davvero (family = sottocartella reale), così `weights_catalog.is_present`
lo vede senza spostare nulla. Fonte remota messa solo se verificata su HF (nome+byte); altrimenti `no_remote`
(file locale importato dall'utente: vale per scan/link/filtro 18+, mai proposto al download).
Le voci 18+ portano sens="adult": i filtri SFW/NSFW per nome file (core.weights_catalog.adult_names) le
nascondono fuori dalla modalità NSFW anche quando il nome non ha parole chiave (getphat, bigLoveZ).
"""

# prefissi/id aggiornati anche se già nel json (REFRESH di catalog_v2)
REFRESH_IDS = (
    "zimage-biglove-zt1", "zimage-turbo-bf16-single", "zimage-turbo-nsfw-2602",
    "flux1-fuxcapacity", "flux1-getphat", "flux1-dev-nunchaku-fp4", "flux1-kontext-nunchaku-fp4",
    "flux2-dev-nvfp4", "ltx23-dev-gguf-q6k", "ltx23-dev-nvfp4", "ltx23-gguf-q4-distilled",
    "te-gemma3-lyrics", "te-flux1-clip-l-diffusers", "te-flux1-t5xxl-diffusers",
)

# voci ritirate da questo giro (sostituite da una versione migliore)
REMOVED_IDS = (
    # FLUX.2 dev fp8 (35 GB, non entra in VRAM) → sostituita da flux2-dev-nvfp4 (22.8 GB, kernel FP4)
    "flux2-dev-fp8mixed",
)


def entries(N):
    return [
        # ───────── Z-Image Turbo ─────────
        N("zimage-biglove-zt1-fp8", "Immagini", "Big Love Z · ZT1 (Z-Image Turbo, 18+, fp8, 5.73GB)",
          category="image", arch="zimage", type="diffusion_models", family="Z-Image",
          filename="bigLoveZ_zt1_pruned_fp8.safetensors", gb=5.73, sens="adult", no_remote=True,
          note="Civitai Big Love Z ZT1, fp8 E4M3 (chiavi cap_embedder = Z-Image). Import locale: il formato fp8 è già "
               "quello che _assemble_zimage carica. Diversa dalla Zora1 bf16 (zimage-biglove-zora1)."),
        N("zimage-turbo-bf16-single", "Immagini", "Z-Image Turbo bf16 (single-file, 11.46GB)",
          category="image", arch="zimage", repo="Comfy-Org/z_image_turbo",
          remote="split_files/diffusion_models/z_image_turbo_bf16.safetensors",
          type="diffusion_models", family="Z-Image", filename="Z_Image_BF16.safetensors", gb=11.46,
          note="Variante single-file di zimage-turbo (caricabile senza lo snapshot diffusers)."),
        N("zimage-turbo-nsfw-2602", "Immagini", "Z-Image Turbo NSFW 2602 (18+, bf16, 11.46GB)",
          category="image", arch="zimage", type="diffusion_models", family="Z-Image",
          filename="zimageTurboNSFWBy_2602NSFWBF16.safetensors", gb=11.46, sens="adult", no_remote=True,
          scan=["zimageTurboNSFWBy_2602NSFWBF16.safetensors"],
          note="Finetune NSFW di Z-Image Turbo (import locale)."),

        # ───────── FLUX.1 ─────────
        N("flux1-fuxcapacity-nsfw-fp8", "Immagini", "fuxCapacity NSFW Porn Flux 2.0 (FLUX.1 dev, 18+, fp8, 11.08GB)",
          category="image", arch="flux1", type="diffusion_models", family="Flux",
          filename="fuxCapacityNSFWPornFlux_20_(SwarmUI).safetensors", gb=11.08, sens="adult", no_remote=True,
          deps=["flux-vae-ae", "t5xxl-fp16|t5xxl-enconly-fp8"],
          scan=["fuxCapacityNSFWPornFlux_20_(SwarmUI).safetensors",
                "fuxCapacityNSFWPornFlux_20_(SwarmUI).safetensors.normalized.safetensors"],
          note="Checkpoint FLUX.1 dev NSFW (import locale). Il .normalized accanto è la cache generata da image_flux."),
        N("flux1-getphat-v23-softcore-fp8", "Immagini", "getphat Flux Reality v23 Softcore (FLUX.1 dev, 18+, fp8, 11.08GB)",
          category="image", arch="flux1", type="diffusion_models", family="Flux",
          filename="getphat Flux Reality v23 Softcore FP8.safetensors", gb=11.08, sens="adult", no_remote=True,
          deps=["flux-vae-ae", "t5xxl-fp16|t5xxl-enconly-fp8"],
          note="Checkpoint FLUX.1 dev NSFW fp8 (780 tensori F8_E4M3). Nome senza parole chiave: il flag 18+ viene da qui."),
        N("flux1-dev-nunchaku-fp4", "Immagini", "FLUX.1 dev nunchaku SVDQ fp4 r32 (Blackwell, 6.56GB)",
          category="image", arch="flux1", repo="nunchaku-ai/nunchaku-flux.1-dev",
          type="diffusion_models", family="Flux/nunchaku", filename="svdq-fp4_r32-flux.1-dev.safetensors", gb=6.56,
          deps=["flux-vae-ae", "t5xxl-fp16|t5xxl-enconly-fp8"],
          note="Caricato dal motore nunchaku (image_flux._load_nunchaku_flux_tr): richiede la wheel nunchaku e una RTX 50xx."),
        N("flux1-kontext-nunchaku-fp4", "Immagini", "FLUX.1 Kontext dev nunchaku SVDQ fp4 r32 (Blackwell, 6.56GB)",
          category="image", arch="flux1", repo="nunchaku-ai/nunchaku-flux.1-kontext-dev",
          type="diffusion_models", family="Flux/nunchaku", filename="svdq-fp4_r32-flux.1-kontext-dev.safetensors", gb=6.56,
          deps=["flux-vae-ae", "t5xxl-fp16|t5xxl-enconly-fp8"],
          note="Come flux1-dev-nunchaku-fp4, per l'editing Kontext."),

        # ───────── FLUX.2 dev: NVFP4 (versione migliore, sostituisce l'fp8 da 35 GB) ─────────
        N("flux2-dev-nvfp4", "Immagini", "FLUX.2 dev NVFP4 mixed (22.8GB, kernel FP4 Blackwell)",
          category="image", arch="flux2", repo="black-forest-labs/FLUX.2-dev-NVFP4",
          type="diffusion_models", family="Flux/nvfp4", filename="flux2-dev-nvfp4-mixed.safetensors", gb=21.21,
          deps=["flux2-vae", "mistral-te-fp8-single"],
          note="Ufficiale BFL (byte identici: 22772036568). Entra nei 32 GB; sostituisce flux2-dev-fp8mixed (35 GB)."),

        # ───────── LTX-2.3 ─────────
        N("ltx23-dev-gguf-q6k", "Video", "LTX-2.3 22B dev GGUF Q6_K (16.55GB)",
          category="video", arch="ltx23", repo="unsloth/LTX-2.3-GGUF", remote="ltx-2.3-22b-dev-Q6_K.gguf",
          type="diffusion_models", family="", filename="ltx-2-3-22b-dev-Q6_K.gguf", gb=16.55,
          deps=["ltx23-gemma-te", "ltx23-distilled-lora-384", "ltx23-spatial-upscaler"],
          note="Caricato da workers/gguf_native.py (dequant GPU). Il file locale differisce di 24 byte dal repo "
               "unsloth (stesso modello, header riscritto): niente sha256."),
        N("ltx23-dev-nvfp4", "Video", "LTX-2.3 22B dev NVFP4 (20.21GB)",
          category="video", arch="ltx23", repo="Lightricks/LTX-2.3-nvfp4",
          type="diffusion_models", family="LTXVideo/LTX-2/nvfp4", filename="ltx-2.3-22b-dev-nvfp4.safetensors", gb=20.21,
          deps=["ltx23-gemma-te", "ltx23-distilled-lora-384", "ltx23-spatial-upscaler"],
          note="Ufficiale Lightricks (byte identici: 21701367310). Variante di precisione del dev: da validare dal vivo "
               "prima di preferirla al dev-fp8."),
        # distilled Q4_K_M: la versione unsloth UD-Q4_K_M (16.4GB, quant dinamica) rimpiazza la quantstack Q4_K_M
        N("ltx23-gguf-q4-distilled", "Video", "LTX-2.3 distilled GGUF UD-Q4_K_M (unsloth, GPU piccole)",
          category="video", arch="ltx23", repo="unsloth/LTX-2.3-GGUF",
          remote="distilled-1.1/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf",
          type="diffusion_models", family="LTXVideo/LTX-2/unsloth/distilled-1.1",
          filename="ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf", gb=15.26,
          deps=["ltx23-gemma-te", "ltx23-spatial-upscaler"],
          note="Versione migliore della QuantStack Q4_K_M (quant dinamica UD). Trovata dall'auto-calibrazione VRAM via glob "
               "'*distilled*Q4_K_M.gguf' (core/ltx2_service.py)."),

        # ───────── text encoder usati dal codice ─────────
        N("te-gemma3-lyrics", "TextEncoder", "Gemma-3 12B IT (testi canzoni, cartella gemma3, 12.8GB)",
          category="aux", arch="gemma3", kind="snapshot", type="TextEncoders", family="gemma3", gb=12.77, no_remote=True,
          used_by=["cap.music_song"],
          note="Letto da core/lyrics_llm.py (GEMMA_DIRS); fallback al TE LTX gemma-3-ltx. Import locale."),
        N("te-flux1-clip-l-diffusers", "TextEncoder", "CLIP-L FLUX.1 (formato diffusers, cartella _te/clip_l, 0.23GB)",
          category="image", arch="flux1", kind="snapshot", type="diffusers", family="_te",
          tool_dest=r"E:\AI\Models\diffusers\_te\clip_l", gb=0.23, no_remote=True,
          note="Letto da image_flux (TE_DIR). Non cancellare."),
        N("te-flux1-t5xxl-diffusers", "TextEncoder", "T5-XXL FLUX.1 (formato diffusers, cartella _te/t5xxl, 9.1GB)",
          category="image", arch="flux1", kind="snapshot", type="diffusers", family="_te",
          tool_dest=r"E:\AI\Models\diffusers\_te\t5xxl", gb=9.12, no_remote=True,
          note="Letto da image_flux (TE_DIR). Doppione di TextEncoders/t5xxl_fp16: dedup SOLO con hardlink, mai cancellare."),
    ]
