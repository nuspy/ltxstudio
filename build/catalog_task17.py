"""Voci di catalogo del task 17 (29/10/2026): integrazioni dei modelli di PULIZIA_MODELLI §3 + versioni nuove della lista utente.

1. Pesi GIA' sul disco usati dall'app ma senza voce: 12 upscaler (spandrel), DiffPitcher, LoRA identità Krea 2 Edit,
   nunchaku Z-Image (bug noto della wheel stabile: non proposto al download).
2. Versioni Civitai uscite dopo il 14/10 (Public): SickRealism v4.0 fp8, NSFW Unlocked V2.1, Perky Breasts v1.1,
   Round Breasts v1.5, BFS H3 Character Swap v1.1. Le versioni vecchie RESTANO (decide l'utente).
Trigger e sha256 letti dall'API Civitai il 29/10/2026.
"""

C = "https://civitai.com/api/download/models/"
P = "https://civitai.red/models/"

REFRESH_IDS: tuple = ()

# (id, file, architettura, GB, nota). Tutti caricabili da inference_server/backends/image_upscale.py (spandrel 0.4.2, provato su CPU).
UPSCALERS = (
    ("upscale-2x-text2hd-realplksr", "2x_Text2HD_v.1-RealPLKSR.pth", "RealPLKSR 2x", 0.03, "2x leggero per testo/documenti"),
    ("upscale-4x-animesharp", "4x-AnimeSharp.safetensors", "ESRGAN 4x", 0.06, "anime/illustrazioni; complementare a UltraSharp v2"),
    ("upscale-4x-nmkd-siax", "4x_NMKD-Siax_200k.pth", "ESRGAN 4x", 0.06, "generico, dettaglio naturale"),
    ("upscale-4x-realisticrescaler", "4x_RealisticRescaler_100000_G.pth", "ESRGAN 4x", 0.12, "foto realistiche"),
    ("upscale-4x-faceupdat", "4xFaceUpDAT.pth", "DAT 4x", 0.14, "specializzato sui volti (super-risoluzione, non restauro)"),
    ("upscale-4x-nomos2-dat2", "4xNomos2_hq_dat2.pth", "DAT 4x", 0.13, "famiglia Nomos2, foto di qualità"),
    ("upscale-4x-nomos2-drct-l", "4xNomos2_hq_drct-l.safetensors", "DRCT 4x", 0.23, "famiglia Nomos2, variante DRCT-L"),
    ("upscale-4x-nomos8k-dat", "4xNomos8kDAT.pth", "DAT 4x", 0.29, "dataset Nomos8k, architettura DAT (diverso da nomos8k-schat-l)"),
    ("upscale-4x-realwebphoto-drct-l", "4xRealWebPhoto_v4_drct-l.pth", "DRCT 4x", 0.23, "foto da web compresse"),
    ("upscale-hat-l-srx4", "HAT-L_SRx4_ImageNet-pretrain.safetensors", "HAT 4x", 0.15, "transformer HAT-L (ImageNet), architettura non ESRGAN"),
    ("upscale-realesrgan-x2plus", "RealESRGAN_x2plus.pth", "ESRGAN 2x", 0.06, "Real-ESRGAN 2x"),
    ("upscale-realesrgan-x4plus-st", "RealESRGAN_x4plus.safetensors", "ESRGAN 4x", 0.06, "Real-ESRGAN 4x (safetensors, preferibile al .pth gemello)"),
)


def entries(N):
    out = []
    for i, fn, arch, gb, note in UPSCALERS:
        out.append(N(i, "Upscale", f"{fn.rsplit('.', 1)[0]} ({arch}, {gb}GB)", category="aux", arch="esrgan",
                     filename=fn, type="Upscale", gb=gb, no_remote=True,
                     note=f"{note}. File locale (import utente): la cartella Upscale è scandita per intero e caricata con spandrel."))

    out += [
        # ───────── DiffPitcher (correzione intonazione, tab Musica): pesi in E:\AI\Models\DiffPitcher ─────────
        N("diffpitcher-world", "Musica (ACE-Step)", "DiffPitcher · WORLD fixed 40 (correzione intonazione, 0.47GB)",
          category="music", arch="diffpitcher", filename="Diff-Pitcher_world_fixed_40.safetensors", type="DiffPitcher",
          gb=0.468, no_remote=True, note="Modo template e spartito (scripts/diffpitcher_run.py). Import locale."),
        N("diffpitcher-bigvgan", "Musica (ACE-Step)", "DiffPitcher · BigVGAN 24kHz 100 band (vocoder, 0.45GB)",
          category="music", arch="diffpitcher", filename="Diff-Pitcher_bigvgan_24khz_100band.safetensors", type="DiffPitcher",
          gb=0.45, no_remote=True, note="Vocoder del runner DiffPitcher. Import locale."),
        N("diffpitcher-pitchformer", "Musica (ACE-Step)", "DiffPitcher · PitchFormer (modo spartito MIDI, 0.10GB)",
          category="music", arch="diffpitcher", filename="Diff-Pitcher_transformer_pitch_360.safetensors", type="DiffPitcher",
          gb=0.102, no_remote=True,
          note="Serve solo al modo «Da spartito MIDI» (config Pitchformer.yaml accanto). Import locale."),
        N("diffpitcher-config", "Musica (ACE-Step)", "DiffPitcher · config.json + Pitchformer.yaml (configurazione)",
          category="music", arch="diffpitcher", filename="config.json", type="DiffPitcher", gb=0.001, no_remote=True,
          scan=["config.json"], note="Accanto: Pitchformer.yaml (modo spartito). Import locale."),

        # ───────── LoRA identità di Krea 2 Edit: applicata IN AUTOMATICO (workers/krea2_infer._find_identity_lora) ─────────
        N("krea2-identity-edit-lora", "Immagini", "Krea 2 Edit · LoRA identità v1.2 (automatica, 1.83GB)",
          category="image", arch="krea2", filename="krea2_identity_edit_v1_2.safetensors", type="LoRA", family="",
          gb=1.83, no_remote=True, used_by=["cap.gen_images"],
          note="Applicata da sola a peso 1.0 in ogni editing Krea 2 per conservare l'identità del soggetto "
               "(glob krea2_identity_edit*.safetensors sotto LoRA). Sta alla radice di LoRA e NON compare nel pannello LoRA."),

        # ───────── nunchaku Z-Image: il codice c'è, la wheel stabile Windows no ─────────
        N("zimage-turbo-nunchaku-fp4", "Immagini", "Z-Image Turbo nunchaku SVDQ fp4 r128 (4.2GB, BUG NOTO wheel stabile)",
          category="image", arch="zimage", type="diffusion_models", family="Z-Image/nunchaku",
          filename="svdq-fp4_r128-z-image-turbo.safetensors", gb=4.2, no_remote=True, used_by=["cap.gen_images"],
          note="Motore nativo in image_flux (NunchakuZImageTransformer2DModel) ma su wheel nunchaku STABILE per Windows "
               "fallisce (assert rotary_emb, docs/FUTURE_CHECKS.md #3, docs/MODEL_REVIEW_2026-07.md): da ricontrollare "
               "con la wheel nightly. Non proposto al download; dà errore chiaro se scelto."),

        # ───────── versioni Civitai nuove (Public dopo il 14/10); le vecchie restano ─────────
        N("krea2-sickrealism-v4", "Immagini", "Krea2 SickRealism · v4.0 fp8 (18+, 12.84GB)",
          category="image", arch="krea2", url=C + "3402173?fileId=3292944",
          filename="krea2Sickrealism_v40_fp8-native.safetensors", type="diffusion_models", family="Krea2", gb=12.84,
          page=P + "2970908?modelVersionId=3402173", sens="adult", deps=["krea2_turbo"],
          sha256="2f02249be11dd93eef9f40b68f9665867bd73d22187e424581b5943b01255c52",
          convert="krea2_dit", remote="krea2Sickrealism_v40_3292944.safetensors",
          note="Civitai 2970908 · Krea 2 · v4.0 (Public dal 25/10/2026; Early Access finito). Sostituisce la v2.0 "
               "(krea2-sickrealism-v2, che RESTA: la cancella solo l'utente). La v3.0 (17/10) non è installata."),
        N("mmh3-lora-unlocked-v21", "Video", "LoRA H3 NSFW Unlocked V2.1 (18+, 0.298GB)",
          category="video", arch="minimax_h3", url=C + "3413468?fileId=3305979",
          filename="Minimax_H3_Unlocked_V2.1-native.safetensors", type="LoRA", family="MiniMax-H3", gb=0.298,
          page=P + "2925727?modelVersionId=3413468", sens="adult", used_by=["cap.video_minimax_h3"],
          sha256="a5fcea727e52394427547dec8d1217e508298c58a79fdc95e647ccdd9ff86ac6",
          convert="mmh3_lora", remote="Minimax_H3_Unlocked_V2.1.safetensors",
          triggers=["nude", "topless", "bare breasts", "naked", "completely unclothed"],
          note="Civitai 2925727 · MiniMax H3 · V2.1 (18/10/2026), sostituisce la V2 (mmh3-lora-unlocked-v2, che RESTA). "
               "Parole d'attivazione dalla pagina del modello: core (trigger sopra); tag anatomici: perky nipples, areolas, "
               "exposed cleavage, bare buttocks, toned abdomen, subtle skin sheen; movimento: caressing bare breasts, intimate "
               "kissing, slipping off robe, disrobing, breathless sigh, skin-on-skin contact, arching back. Peso 0.3-1."),
        N("krea2-lora-civ-perky-breasts-v1-1", "Immagini", "PERKY BREASTS · v1.1 (18+, 0.235GB)",
          category="image", arch="krea2", url=C + "3418708?fileId=3312224",
          filename="Perkybreasts55880_1000step.safetensors", type="LoRA", family="Krea2", gb=0.235,
          page=P + "2997219?modelVersionId=3418708", sens="adult",
          sha256="8fd77de7c89241840874043422cc72b2fc492768e524aede1bc306f747e9dbb6",
          triggers=["perky breasts", "puffy nipples"],
          note="Civitai 2997219 · Krea 2 · v1.1 (20/10/2026): trigger DIVERSI dalla v1.0 (perkybreasts). La v1.0 RESTA."),
        N("mmh3-lora-civ-nsfw-posing-touching-round-breasts-h3-v1-5", "Video",
          "NSFW | Posing + Touching + Round Breasts | H3 LoRA · v1.5 (18+, 0.31GB)",
          category="video", arch="minimax_h3", url=C + "3406313?fileId=3297736",
          filename="H3-nsfw-round-breasts-posing-V1.5-native.safetensors", type="LoRA", family="MiniMax-H3", gb=0.31,
          page=P + "2985832?modelVersionId=3406313", sens="adult", used_by=["cap.video_minimax_h3"],
          sha256="1eb33417114cfbfb1c88e553c6cc015f3f444a6326a39b8855918f9f2db788f8",
          convert="mmh3_lora", remote="H3-nsfw-round-breasts-posing-V1.5.safetensors",
          note="Civitai 2985832 · MiniMax H3 · v1.5 (16/10/2026). Nessun token d'attivazione (l'autore: prompt naturale). "
               "La v1 (mmh3-lora-civ-nsfw-posing-touching-round-breasts-h3-lo) RESTA."),
        N("mmh3-lora-civ-bfs-best-face-swap-h3-character-swap-v1-1", "Video",
          "BFS - Best Face Swap · H3 Character Swap v1.1 (18+, 0.081GB)",
          category="video", arch="minimax_h3", url=C + "3412620?fileId=3305013",
          filename="minimax_h3_character_swap_v1.1_ref2va_fro-native.safetensors", type="LoRA", family="MiniMax-H3",
          gb=0.081, page=P + "2027766?modelVersionId=3412620", sens="adult", used_by=["cap.video_minimax_h3"],
          sha256="ab21167bed71db99ac09527ad1f26f5f3517687ebf2e1f1bd5e91cc0bdf9b9c3",
          convert="mmh3_lora", remote="minimax_h3_character_swap_v1.1_ref2va_fro.safetensors",
          triggers=["body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, "
                    "face, head, hair or fur, skin, body shape, proportions and clothing define the character in the target "
                    "video. The guide video provides the scene and the performance: its background, environment, lighting, "
                    "camera, framing, body pose, hand and arm movement, head rotation, gaze direction, lip movement and "
                    "facial expression, frame by frame; only the main subject of the guide video is replaced: its face, hair, "
                    "body and clothing become <Subject 1>.  summary: [video generation + reference] Replace the guide subject "
                    "with <Subject 1>, keeping the scene, camera and the complete motion and performance of the guide video.  "
                    "detailed_description: [Shot 1] <Subject 1> takes the exact place of the guide subject and moves with the "
                    "same timing, body pose, gestures, head motion, gaze, speech and facial expressions, while keeping the "
                    "complete identity from <Picture 1>: facial structure, hair or fur, skin, body shape and clothing stay "
                    "stable in every frame. The background, lighting, colors and camera of the guide video are preserved. "
                    "Prevent identity drift, clothing changes, texture crawling and flicker."],
          note="Civitai 2027766 · MiniMax H3 · v1.1 (18/10/2026). La v1.0 (mmh3-lora-civ-bfs-best-face-swap-h3-character-"
               "swap-v1) RESTA."),
    ]
    return out


USE = {i: (["upscale_image"], []) for i, *_ in UPSCALERS}
USE.update({
    "diffpitcher-world": (["music"], []),
    "diffpitcher-bigvgan": (["music"], []),
    "diffpitcher-pitchformer": (["music"], []),
    "diffpitcher-config": (["music"], []),
    "krea2-identity-edit-lora": (["lora_style"], []),
    "zimage-turbo-nunchaku-fp4": (["ckpt_image", "nvfp4"], []),
    "krea2-sickrealism-v4": (["ckpt_image", "uncensored"], ["krea2_turbo"]),
    "mmh3-lora-unlocked-v21": (["lora_style", "uncensored"], []),
    "krea2-lora-civ-perky-breasts-v1-1": (["lora_style", "uncensored"], []),
    "mmh3-lora-civ-nsfw-posing-touching-round-breasts-h3-v1-5": (["lora_style", "uncensored"], []),
    "mmh3-lora-civ-bfs-best-face-swap-h3-character-swap-v1-1": (["lora_style", "uncensored"], []),
})
