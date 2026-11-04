"""MiniMax H3: costanti condivise da GUI (tab Video / Avatar), API e worker.

Tutto viene dalla documentazione ufficiale (diffusers modular MiniMax-H3) e dalle schede lightx2v Turbo:
  · 24 fps fissi, durata 5–15 s, fotogrammi agganciati a 17·n + 5 (griglia del VAE video);
  · tela: lato corto 768, area ≤ 768×1344, lati multipli di 32 (960×544 ≈ 2,3× più veloce);
  · guidance distillata: niente negative prompt né CFG, una passata per step;
  · `num_inference_steps` del pipeline conta i punti della griglia sigma (0 finale compreso) = NFE + 1;
  · Turbo lightx2v: 4 NFE (shift video 12) o 8 NFE (shift video 6, consigliato dallo Studio), audio shift 3.
"""
from __future__ import annotations

import math

FPS = 24
MIN_S, MAX_S = 5.0, 15.0
FRAME_GRID = 17                       # fotogrammi per chunk del VAE video
DEFAULT_NFE = 49                      # default del pipeline: 50 punti di griglia
VIDEO_SHIFT, AUDIO_SHIFT = 12.0, 3.0

# profilo → (DiT FL2VA, DiT Ref2VA, TE consigliato). Nessuna scelta automatica: la GUI propone, l'utente conferma.
PROFILES = {
    "nvfp4": ("mmh3-fl2va-nvfp4", "mmh3-ref2va-nvfp4", "mmh3-te-nvfp4"),
    "fp8": ("mmh3-fl2va-fp8", "mmh3-ref2va-fp8", "mmh3-te-fp8"),
    "int8": ("mmh3-fl2va-int8", "mmh3-ref2va-int8", "mmh3-te-int8"),
    # W4A8 comfy-kitchen (pesi int4 a codebook + GEMM int8): metà memoria dell'int8, kernel int8 su ogni GPU
    "int4": ("mmh3-fl2va-int4", "mmh3-ref2va-int4", "mmh3-te-nvfp4"),
}
# DiT PERSONALIZZATI (finetune 18+, scelti esplicitamente): un solo file «hybrid» vale per FL2VA e Ref2VA (stesse
# chiavi del DiT pruned ufficiale). turbo_fused = turbo già nei pesi: niente LoRA turbo, passi e shift propri.
CUSTOM_DITS = {
    "mmh3-dit-pinkcherry18": {"label": "PinkCherry H3 v1.8", "modes": ("fl2va", "ref2va")},
    # beta7 fix1 (int8 «HQ», bordi BF16): turbo gia fuso, 8 passi, shift video 12 / audio 3. L'autore la consiglia con attenzione sparsa video (VSA) e sparsita 90 %.
    "mmh3-dit-10erosmax-b7": {"label": "10Eros-Max beta7 Turbo", "modes": ("fl2va", "ref2va"),
                              "turbo_fused": True, "nfe": 8, "video_shift": 12.0, "hint_key": "vedit.h3.model.b7hint"},
    # base Ref2VA (adaln_t_table e norme identici al pruned Ref2VA ufficiale), «Hybrid» per l'autore: int4 W4A8
    "mmh3-dit-dasiwa-v4": {"label": "DaSiWa Hybrid v4 (int4)", "modes": ("fl2va", "ref2va")},
    # nvfp4 MISTA di rockerBOO (Ref2VA pruned: qkv int8-ConvRot, bordi e out_proj in BF16, resto nvfp4; layer config a parte): risposta al sospetto che i nostri nvfp4
    # (Abiray, stessa dimensione esatta di coolthor/wsjLL: tutti i 200 layer in nvfp4) siano fatti peggio. Serve per il confronto a parita di seme/prompt/encoder.
    "mmh3-dit-rockerboo-nvfp4": {"label": "MiniMax-H3 Ref2VA nvfp4 + int8 ConvRot (rockerBOO, bordi BF16)", "modes": ("ref2va",), "official": True},   # quantizzazione del modello ufficiale, non un finetune
}
TE_IDS = ("mmh3-te-nvfp4", "mmh3-te-fp8", "mmh3-te-int8", "mmh3-te-heretic-nvfp4")
TE_ADULT = "mmh3-te-heretic-nvfp4"
COMMON_IDS = ("mmh3-configs", "mmh3-video-vae", "mmh3-audio-vae")

# turbo: chiave GUI → {modo: id LoRA}, NFE, shift video
TURBO = {
    "4": ({"fl2va": "mmh3-turbo-lora-fl2v", "ref2va": "mmh3-turbo-lora-ref2v"}, 4, 12.0),
    "8": ({"fl2va": "mmh3-turbo-lora-fl2v-8"}, 8, 6.0),
    # larryvrh v4 step600 EMA (con cui è addestrato Veda): sampler a griglia uniforme, shift video 12 / audio 3
    "v4": ({"fl2va": "mmh3-turbo-v4-fl2v", "ref2va": "mmh3-turbo-v4-ref2v"}, 8, 12.0),
    # merge DARE-TIES di DMAD 4 passi «full critic» e larryvrh v4 (Civitai 2837571): 6-8 passi, shift 12/3
    "dmad": ({"fl2va": "mmh3-lora-civ-minimax-h3-turbo-loras-dmad-larryvrhv4-m"}, 8, 12.0),
    # lightx2v Minimax-h3-Turbo in formato DIFFUSERS (1,38 GB, senza le coppie AdaLN della versione ComfyUI/Comfy-Org): ref2v 4 passi v1.0 e 8 passi v1.1, 768p.
    # Passi e shift video come le altre turbo lightx2v (4 passi shift 12, 8 passi shift 6): da confermare dalla resa nel giro dal vivo.
    # DMAD 4 passi «full» (Civitai 3382801, rank 39): la LoRA dell'originale Openjutsu (workflow object/simple: 6 passi, shift video 12, forza 1)
    "dmad4": ({"ref2va": "mmh3-turbo-dmad4full-ref2v"}, 6, 12.0),
    "r4v10": ({"ref2va": "mmh3-turbo-ref2v-4-v10"}, 4, 12.0),
    "r8v11": ({"ref2va": "mmh3-turbo-ref2v-8-v11"}, 8, 6.0),
}

# formati: rapporto → (larghezza, altezza) del rapporto; la tela esatta la calcola il pipeline
ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "21:9": (21, 9), "4:3": (4, 3), "3:4": (3, 4), "1:1": (1, 1)}
CANVAS = {"768": (768, 768 * 1344), "544": (544, 544 * 960)}     # lato corto, area massima

MAX_REFS = {"image": 9, "video": 3, "audio": 3, "total": 12}

# Campionatore proprio di alcune LoRA turbo (chiave di TURBO -> {"method", "knots"}): `renoise` = passo stocastico con griglia fissa di nodi (scala 1000), come gli adattatori
# Pika H3 Flex a 6 passi (workers/h3_renoise_scheduler.py). Le turbo senza voce qui usano l'Euler uniforme di sempre.
TURBO_SAMPLER: dict = {}

# Lato lungo del video di MOVIMENTO (<Video 1>) negli edit: come l'originale Openjutsu (object/simple.api.json, nodo «Motion reference video (~512 px)»,
# ImageScaleToMaxDimension 512). Il riferimento porta posa e tempi, non i dettagli: a 736p (1344x736) a tela piena ha ~6,7 volte i pixel (token del testo,
# codifica VAE e righe di riferimento nel DiT). 0 = tela del bersaglio, come prima del 04/11.
REF_VIDEO_MAX_SIDE = 512


def reference_video_canvas(height: int, width: int, max_side: int = REF_VIDEO_MAX_SIDE, multiple: int = 32) -> dict:
    """Tela del video di riferimento di un edit: {short_edge, max_pixels, height, width}. `short_edge` e `max_pixels` sono i valori da dare a
    `canvas_short_edge` / `canvas_max_pixels` del pipeline (resolve_canvas_size ricava la tela dal rapporto del video, arrotondata a `multiple`);
    `height`/`width` la tela che ne risulta. `max_side` <= 0 o non piu piccolo della tela: la tela del bersaglio."""
    if max_side <= 0 or max_side >= max(height, width):
        return {"short_edge": min(height, width), "max_pixels": height * width, "height": height, "width": width}
    short_edge = max(multiple, round(min(height, width) * max_side / max(height, width)))
    ratio = width / height
    float_width, float_height = (short_edge * ratio, float(short_edge)) if ratio >= 1.0 else (float(short_edge), short_edge / ratio)
    return {"short_edge": short_edge, "max_pixels": 4 * max_side * max_side, "height": max(multiple, round(float_height / multiple) * multiple),
            "width": max(multiple, round(float_width / multiple) * multiple)}
DIALOGUE_LANGS = ("Italian", "English", "Chinese", "Spanish", "French", "German", "Portuguese",
                  "Japanese", "Korean", "Russian", "Arabic")


# Predittore Veda (attenzione sparsa appresa, workers/veda_h3): addestrato su Ref2VA + Turbo 8 passi, clip ~5 s,
# geometrie 16:9 / 9:16 / 4:3 / 1:1; fuori da lì il piano a tile è approssimato. Budget come la release: 32 tile.
VEDA_R2VA = "mmh3-veda-r2va"
# LightVAE (lightx2v): VAE video potato, stessa interfaccia latente. "dec" = decoder Lite (decode ~3×, metà VRAM),
# "encdec" = anche l'encoder Lite (prima/ultima immagine e riferimenti codificati più in fretta)
LIGHTVAE = {"dec": ("mmh3-lightvae-dec",), "encdec": ("mmh3-lightvae-dec", "mmh3-lightvae-enc")}
VEDA_BUDGET = "32"


def mode_for(params: dict) -> str:
    # la guida di latenti allineata (workers/h3_guide.py) vive nei DiT Ref2VA anche senza riferimenti nativi: le LoRA di guida sono addestrate su quei pesi
    return "ref2va" if (params.get("references") or params.get("h3_guide") or params.get("_h3_guide_latents") is not None) else "fl2va"


def snap_frames(seconds: float) -> int:
    """Durata richiesta → num_frames valido (17·n + 5 per eccesso) dentro 5–15 s."""
    s = min(MAX_S, max(MIN_S, float(seconds)))
    n = max(0, math.ceil((round(s * FPS) - 5) / FRAME_GRID))
    frames = FRAME_GRID * n + 5
    while frames / FPS > MAX_S + 1e-6:
        frames -= FRAME_GRID
    while frames / FPS < MIN_S - 1e-6:
        frames += FRAME_GRID
    return frames


def frames_seconds(frames: int) -> float:
    return frames / FPS


def dialogue_tag(lang: str, text: str, speaker: str = "") -> str:
    """Battuta nel formato della guida ufficiale: <d>[Lingua] testo</d> (speaker opzionale davanti)."""
    body = f"<d>[{lang}] {text.strip()}</d>"
    return f"{speaker.strip()}: {body}" if speaker.strip() else body


def weight_ids(params: dict) -> dict:
    """ID di catalogo richiesti da una richiesta (profilo/TE/turbo espliciti; nessun ripiego)."""
    mode = mode_for(params)
    prof = str(params.get("h3_profile") or "nvfp4").lower()
    if prof not in PROFILES:
        raise ValueError(f"profilo MiniMax H3 sconosciuto: {prof} ({' | '.join(PROFILES)})")
    fl, ref, te = PROFILES[prof]
    ids = {"dit": ref if mode == "ref2va" else fl, "te": params.get("h3_te") or te,
           "configs": "mmh3-configs", "vae": "mmh3-video-vae", "audio_vae": "mmh3-audio-vae"}
    custom = str(params.get("h3_dit") or "")
    if custom:                                    # DiT personalizzato al posto di quello del profilo
        meta = CUSTOM_DITS.get(custom)
        if meta is None:
            raise ValueError(f"DiT MiniMax H3 sconosciuto: {custom} ({' | '.join(CUSTOM_DITS)})")
        if mode not in meta["modes"]:
            raise ValueError(f"{meta['label']}: modalità {mode} non supportata")
        ids["dit"] = custom
    if ids["te"] not in TE_IDS:
        raise ValueError(f"text encoder MiniMax H3 sconosciuto: {ids['te']}")
    t = str(params.get("h3_turbo") or "")
    if CUSTOM_DITS.get(custom, {}).get("turbo_fused"):
        t = ""                                    # turbo già nei pesi: la LoRA turbo lo raddoppierebbe
    if t:
        if t not in TURBO or mode not in TURBO[t][0]:
            raise ValueError(f"LoRA turbo {t} passi non disponibile per {mode}")
        ids["turbo"] = TURBO[t][0][mode]
    if params.get("h3_veda"):                     # attenzione sparsa Veda: predittore addestrato per Ref2VA
        if mode != "ref2va":
            raise ValueError("Attenzione sparsa Veda: il predittore disponibile è per Ref2VA (servono riferimenti)")
        ids["veda"] = VEDA_R2VA
    lv = str(params.get("h3_lightvae") or "")
    if lv:
        if lv not in LIGHTVAE:
            raise ValueError(f"LightVAE sconosciuto: {lv} ({' | '.join(LIGHTVAE)})")
        for k, cid in zip(("vae_dec_lite", "vae_enc_lite"), LIGHTVAE[lv]):
            ids[k] = cid
    return ids
