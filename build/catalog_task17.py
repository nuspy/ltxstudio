"""Voci di catalogo del task 17 (29/10/2026): integrazioni dei modelli di PULIZIA_MODELLI §3 + versioni nuove della lista utente.

1. Pesi GIA' sul disco usati dall'app ma senza voce: 12 upscaler (spandrel), DiffPitcher, LoRA identità Krea 2 Edit,
   nunchaku Z-Image (bug noto della wheel stabile: non proposto al download).
2. Versioni Civitai uscite dopo il 14/10 (Public): SickRealism v4.0 fp8, NSFW Unlocked V2.1, Perky Breasts v1.1,
   Round Breasts v1.5, BFS H3 Character Swap v1.1. Le versioni vecchie RESTANO (decide l'utente).
Trigger e sha256 letti dall'API Civitai il 29/10/2026.
"""

import os

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


DP_PAGE = "https://github.com/haidog-yaqub/DiffPitcher"
UPSCALER_PAGES = {'upscale-4x-animesharp': 'https://openmodeldb.info/models/4x-AnimeSharp', 'upscale-4x-nmkd-siax': 'https://openmodeldb.info/models/4x-NMKD-Siax-CX', 'upscale-4x-realisticrescaler': 'https://openmodeldb.info/models/4x-RealisticRescaler', 'upscale-4x-faceupdat': 'https://openmodeldb.info/models/4x-FaceUpDAT', 'upscale-4x-nomos2-dat2': 'https://openmodeldb.info/models/4x-Nomos2-hq-dat2', 'upscale-4x-nomos2-drct-l': 'https://openmodeldb.info/models/4x-Nomos2-hq-drct-l', 'upscale-4x-nomos8k-dat': 'https://openmodeldb.info/models/4x-Nomos8kDAT', 'upscale-4x-realwebphoto-drct-l': 'https://openmodeldb.info/models/4x-RealWebPhoto-v4-drct-l', 'upscale-hat-l-srx4': 'https://github.com/XPixelGroup/HAT', 'upscale-realesrgan-x2plus': 'https://github.com/xinntao/Real-ESRGAN', 'upscale-realesrgan-x4plus-st': 'https://github.com/xinntao/Real-ESRGAN'}


# (id, file, arch, GB, 18+, trigger, pagina) — LoRA §3 di PULIZIA_MODELLI gia' su disco; generato da orch/task17_audit/c5_gen_loras.py
LOCAL_LORAS = (
    ('lora-flux-almuflux', 'almuFLUX.safetensors', 'flux1', 0.613, False, ['almu'], ''),
    ('lora-flux-aphoflux', 'aphoFLUX.safetensors', 'flux1', 0.613, False, ['apho'], ''),
    ('lora-flux-bllefluxreality-softcore', 'blleFLUXReality Softcore.safetensors', 'flux1', 0.613, True, ['blle'], ''),
    ('lora-flux-dvelflux', 'dvelFLUX.safetensors', 'flux1', 0.613, False, ['dvel'], ''),
    ('lora-flux-est3tic-lingerie-v1-flux-dev-swarmui', 'est3tic_lingerie_v1_FLUX-DEV_(SwarmUI).safetensors', 'flux1', 0.04, False, ['est3tic_lingerie'], ''),
    ('lora-flux-hourglassv32-flux', 'hourglassv32_FLUX.safetensors', 'flux1', 0.306, True, ['navel', 'cleavage', 'panties', 'nipples'], 'https://civitai.red/models/129130?modelVersionId=1668530'),
    ('lora-flux-jpavfluxreality', 'jpavFLUXReality.safetensors', 'flux1', 0.613, False, ['jpav'], ''),
    ('lora-flux-lmilflux', 'lmilFLUX.safetensors', 'flux1', 0.613, False, ['lmil'], ''),
    ('lora-flux-mawiflux', 'mawiFLUX.safetensors', 'flux1', 0.613, False, ['mawi'], ''),
    ('lora-flux-micro-thong-panties-v2', 'micro-thong-panties-V2.safetensors', 'flux1', 0.04, False, ['micro_thong'], ''),
    ('lora-flux-mnraflux', 'mnraFLUX.safetensors', 'flux1', 0.613, False, ['mnra'], ''),
    ('lora-flux-mohrflux', 'mohrFLUX.safetensors', 'flux1', 0.613, False, ['mohr'], ''),
    ('lora-flux-moody-000019', 'Moody-000019.safetensors', 'flux1', 0.307, False, ['moody'], ''),
    ('lora-flux-mysticxxx-v3', 'MysticXXX-V3.safetensors', 'flux1', 0.156, True, [], ''),
    ('lora-flux-oiledskin-fluxdev-swarmui', 'OiledSkin_FluxDev_(SwarmUI).safetensors', 'flux1', 0.344, True, ['OiledSkin'], ''),
    ('lora-flux-pov-fist', 'PoV_Fist.safetensors', 'flux1', 0.306, True, ['POVF1ST'], ''),
    ('lora-flux-realunderboob-fl', 'RealUnderboob_FL.safetensors', 'flux1', 0.172, True, ['RealUnderboob'], 'https://civitai.red/models/1466115?modelVersionId=1658153'),
    ('lora-flux-riccflux', 'riccFLUX.safetensors', 'flux1', 0.613, False, ['ricc'], ''),
    ('lora-flux-sexy-festival-goers', 'Sexy_Festival_Goers.safetensors', 'flux1', 0.613, True, ['sxfestgoers'], 'https://civitai.red/models/1474673?modelVersionId=1668026'),
    ('lora-flux-sgmkfluxreality', 'sgmkFLUXReality.safetensors', 'flux1', 0.613, True, ['sgmk'], ''),
    ('lora-flux-steepslopetiddies-v1-2-rank16-bf16', 'SteepSlopeTiddies_v1-2_rank16_bf16.safetensors', 'flux1', 0.067, True, ['SteepSlopeTiddies'], ''),
    ('lora-flux-tan-lines-01', 'tan-lines-01.safetensors', 'flux1', 0.538, True, ['tan lines on skin'], 'https://civitai.red/models/1517501?modelVersionId=1716933'),
    ('lora-flux-underland-alice-000012', 'underland-alice-000012.safetensors', 'flux1', 0.04, True, ['UnderlandAlice'], 'https://civitai.red/models/1498122?modelVersionId=1694726'),
    ('lora-flux-vmorflux', 'vmorFLUX.safetensors', 'flux1', 0.613, True, ['vmor'], ''),
    ('lora-flux-yanjfluxreality-hardcore', 'yanjFLUXReality Hardcore.safetensors', 'flux1', 0.613, True, ['yanj'], ''),
    ('lora-flux-yanjfluxreality-softcore', 'yanjFLUXReality Softcore.safetensors', 'flux1', 0.613, True, ['yanj'], ''),
    ('lora-flux-yschfluxreality-softcore-v11', 'yschFLUXReality Softcore v11.safetensors', 'flux1', 0.613, True, ['ysch'], ''),
    ('lora-wan-deepthroat-epoch-80-loras', 'deepthroat_epoch_80_(LoRAs).safetensors', 'wan21', 0.307, True, ['blowjob', 'deepthroat'], 'https://civitai.red/models/1340403?modelVersionId=1513684'),
    ('lora-wan-fingering-i2v-e248-loras', 'fingering_i2v_e248_(LoRAs).safetensors', 'wan21', 0.359, True, ['She inserts two fingers into her pussy. She masturbates by sliding her fingers in and out of her pussy.'], 'https://civitai.red/models/1444759?modelVersionId=1694292'),
    ('lora-wan-grinding-pussyjob-e130-loras', 'grinding_pussyjob_e130_(LoRAs).safetensors', 'wan21', 0.307, True, [], 'https://civitai.red/models/1476907?modelVersionId=1681144'),
    ('lora-wan-t2v-pov-bj-14b-loras', 'T2V-POV_BJ-14B_(LoRAs).safetensors', 'wan21', 0.307, True, [], ''),
    ('lora-wan-detailz-wan', 'detailz-wan.safetensors', 'wan21', 0.307, True, ['detailz'], 'https://civitai.red/models/1385506?modelVersionId=1565668'),
    ('lora-wan-masturbation-cumshot-wani2v480p-v1', 'masturbation_cumshot_wanI2V480p_v1.safetensors', 'wan21', 0.307, True, [], 'https://civitai.red/models/1368557?modelVersionId=1546156'),
    ('lora-wan-nipplesuckwan-e40', 'nippleSuckWan_e40.safetensors', 'wan21', 0.307, True, ['nipple lick', 'nipple licking', 'nipple suck', 'nipple sucking'], ''),
    ('lora-wan-phantom-wan-14b-fusionx-lora', 'Phantom_Wan_14B_FusionX_LoRA.safetensors', 'wan21', 0.317, False, [], 'https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX'),
    ('lora-wan-wan-facefuck-epoch110-fast-speed-full-insertion-', 'WAN-facefuck-epoch110 fast speed full insertion man has sex with womans mouth.safetensors', 'wan21', 0.359, True, ['facefuck fast speed full insertion', 'man has sex with womans mouth'], ''),
    ('lora-wan-wan-nsfw-e14-fixed', 'wan-nsfw-e14-fixed.safetensors', 'wan21', 0.614, True, ['nsfwsks'], 'https://civitai.red/models/1307155?modelVersionId=1475095'),
    ('lora-wan-wan-t2v-i2v-anal-combined-e50', 'wan-t2v-i2v-anal-combined-e50.safetensors', 'wan21', 0.307, True, ['She moves up and down on top of him as his penis penetrates her asshole.'], 'https://civitai.red/models/1426284?modelVersionId=1612131'),
    ('lora-wan-wan-cumshot-i2v', 'wan_cumshot_i2v.safetensors', 'wan21', 0.359, True, [], ''),
    ('lora-wan-wan-dr34mj0b', 'WAN_dr34mj0b.safetensors', 'wan21', 0.18, True, ['bl0wj0b', 'd0ub13_bj'], ''),
    ('lora-wan-fusionx-facenaturalizer', 'FusionX_FaceNaturalizer.safetensors', 'wan21', 0.614, False, [], 'https://civitai.red/models/1755105?modelVersionId=1986305'),
)


def entries(N):
    out = []
    for i, fn, arch, gb, note in UPSCALERS:
        out.append(N(i, "Upscale", f"{fn.rsplit('.', 1)[0]} ({arch}, {gb}GB)", category="aux", arch="esrgan",
                     filename=fn, type="Upscale", gb=gb, no_remote=True, page=UPSCALER_PAGES.get(i),
                     note=f"{note}. File locale (import utente): la cartella Upscale è scandita per intero e caricata con spandrel."))

    for i, fn, arch, gb, adult, trig, page in LOCAL_LORAS:
        folder = "Flux" if arch == "flux1" else "Wan"
        out.append(N(i, "Immagini" if arch == "flux1" else "Video", f"{os.path.splitext(fn)[0]} ({'FLUX.1' if arch == 'flux1' else 'Wan 2.1'}{', 18+' if adult else ''}, {gb}GB)",
                     category="image" if arch == "flux1" else "video", arch=arch, filename=fn, type="LoRA", family=folder,
                     gb=gb, no_remote=True, sens="adult" if adult else None, triggers=trig or None, page=page or None,
                     note="LoRA locale (PULIZIA_MODELLI §3, integrata task 17): pannello LoRA " + ("FLUX" if arch == "flux1" else "Wan 2.1")
                          + "; architettura dalle chiavi dell'header. " + ("Pagina verificata (stesso nome file/dimensione)." if page else
                          "Pagina non trovata: file importato, origine non verificabile (hash assente su Civitai).")))

    out += [
        # ───────── DiffPitcher (correzione intonazione, tab Musica): pesi in E:\AI\Models\DiffPitcher ─────────
        N("diffpitcher-world", "Musica (ACE-Step)", "DiffPitcher · WORLD fixed 40 (correzione intonazione, 0.47GB)",
          category="music", arch="diffpitcher", filename="Diff-Pitcher_world_fixed_40.safetensors", type="DiffPitcher",
          gb=0.468, no_remote=True, page=DP_PAGE, note="Modo template e spartito (scripts/diffpitcher_run.py). Import locale."),
        N("diffpitcher-bigvgan", "Musica (ACE-Step)", "DiffPitcher · BigVGAN 24kHz 100 band (vocoder, 0.45GB)",
          category="music", arch="diffpitcher", filename="Diff-Pitcher_bigvgan_24khz_100band.safetensors", type="DiffPitcher",
          gb=0.45, no_remote=True, page=DP_PAGE, note="Vocoder del runner DiffPitcher. Import locale."),
        N("diffpitcher-pitchformer", "Musica (ACE-Step)", "DiffPitcher · PitchFormer (modo spartito MIDI, 0.10GB)",
          category="music", arch="diffpitcher", filename="Diff-Pitcher_transformer_pitch_360.safetensors", type="DiffPitcher",
          gb=0.102, no_remote=True, page=DP_PAGE,
          note="Serve solo al modo «Da spartito MIDI» (config Pitchformer.yaml accanto). Import locale."),
        N("diffpitcher-config", "Musica (ACE-Step)", "DiffPitcher · config.json + Pitchformer.yaml (configurazione)",
          category="music", arch="diffpitcher", filename="config.json", type="DiffPitcher", gb=0.001, no_remote=True, page=DP_PAGE,
          scan=["config.json"], note="Accanto: Pitchformer.yaml (modo spartito). Import locale."),

        # ───────── LoRA identità di Krea 2 Edit: applicata IN AUTOMATICO (workers/krea2_infer._find_identity_lora) ─────────
        N("krea2-identity-edit-lora", "Immagini", "Krea 2 Edit · LoRA identità v1.2 (automatica, 1.83GB)",
          category="image", arch="krea2", filename="krea2_identity_edit_v1_2.safetensors", type="LoRA", family="",
          gb=1.83, no_remote=True, used_by=["cap.gen_images"],
          page="https://huggingface.co/conradlocke/krea2-identity-edit",
          note="Applicata da sola a peso 1.0 in ogni editing Krea 2 per conservare l'identità del soggetto "
               "(glob krea2_identity_edit*.safetensors sotto LoRA). Sta alla radice di LoRA e NON compare nel pannello LoRA."),

        # ───────── nunchaku Z-Image: il codice c'è, la wheel stabile Windows no ─────────
        N("zimage-turbo-nunchaku-fp4", "Immagini", "Z-Image Turbo nunchaku SVDQ fp4 r128 (4.2GB, BUG NOTO wheel stabile)",
          category="image", arch="zimage", type="diffusion_models", family="Z-Image/nunchaku",
          filename="svdq-fp4_r128-z-image-turbo.safetensors", gb=4.2, no_remote=True, used_by=["cap.gen_images"],
          page="https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo",
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
USE.update({r[0]: (["lora_style", "uncensored"] if r[4] else ["lora_style"], []) for r in LOCAL_LORAS})
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
