"""Registro delle RELEASE LTX (2.3 / 2.5) — fonte UNICA di verità per componenti e capability.

Prima d'ora la "versione" LTX non esisteva come concetto: c'era una sola arch `"ltx"` e i path 2.3
erano hardcoded in ~15 punti (worker, servizi, GUI, template). Qui la release diventa una dimensione
di prima classe, ORTOGONALE all'architettura: l'arch resta `"ltx"`, la release è `"2.3"` o `"2.5"`.

Perché serve un registro e non un secondo `_ARCH_RUNNER`:
- 2.5 è uno **split pack** (un safetensors per componente: transformer, text encoder, VAE video,
  VAE audio+vocoder, upscaler, duration head), mentre 2.3 è un checkpoint monolitico che impacchetta
  transformer + VAE + text projection. I due set NON sono intercambiabili.
- 2.5 richiede `transformers>=5.8,<5.15` (Gemma 4) mentre 2.3 gira su transformers 4.x → **venv
  diversi**, quindi ogni release porta il proprio `python`/`worker`.
- Le capability differiscono per *disponibilità dei pesi*, non per capacità del modello: 2.5 ha oggi
  una sola IC-LoRA ufficiale contro le ~16 della 2.3, nessun checkpoint NSFW e nessun GGUF.

stdlib-only e SENZA torch: questo modulo è importato anche dal processo GUI, che non deve mai creare
un contesto CUDA (vedi [[app-no-cuda-context]]).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_MODELS = Path(r"E:\AI\Models")
_TOOLS = Path(r"E:\AI\Tools")
_REPO = Path(__file__).resolve().parents[1]

# FASE G weights-unification: i path dei componenti vengono dal REGISTRO UNICO (weights_catalog
# v2, per id) con i letterali storici come fallback — una sola fonte di verità su dove vive un
# peso; `AI_MODELS_DIR`/«Cartella base» valgono anche qui. `model_paths.p` non solleva mai.
from core.model_paths import p as _P                       # stdlib-only (nessun torch)

# Stato di una capability. "cross" = tecnicamente possibile solo riusando un adapter dell'altra
# release: il DiT ha la stessa topologia (48 layer, dim 4096, 128 canali latenti) e Lightricks
# dichiara che «la grande maggioranza delle LoRA/IC-LoRA 2.3 gira su 2.5 senza modifiche», ma il
# README del repo ufficiale è più prudente («a LoRA only works with the model it was trained on»).
# Due fonti ufficiali in disaccordo → non blocchiamo, avvisiamo. La GUI mostra il badge di warning.
CAP_YES, CAP_NO, CAP_CROSS = "yes", "no", "cross"


@dataclass(frozen=True)
class LtxRelease:
    """Tutto ciò che distingue una release LTX. I path sono ASSOLUTI e possono non esistere:
    `missing_components()` dice cosa manca senza mai sostituire nulla in automatico."""

    id: str                                   # "2.3" | "2.5"
    label: str                                # etichetta GUI
    python: str                               # interprete del venv dedicato
    worker: str                               # worker di inferenza (release-aware)
    text_encoder: str                         # cartella (2.3, formato ltx-core) o file (2.5, split)
    video_vae: str                            # "" per 2.3: è dentro il checkpoint monolitico
    audio_vae: str                            # "" per 2.3 (idem); in 2.5 contiene ANCHE il vocoder
    upsampler_spatial: str
    upsampler_temporal: str                   # "" se la release non ce l'ha
    distilled_lora: str                       # obbligatoria coi checkpoint dev, inutile coi distilled
    msr_lora: str                             # Multiple-Subject-Reference (riferimenti multipli): file
    #                                           DIVERSO per release, addestrato sulla sua architettura
    duration_head: str                        # "" se assente
    diffusion_decoder: str                    # "" se assente (2.5: decoder DiT alternativo al Conv VAE)
    aux_bundle: str                           # bundle componenti ausiliarie (solo 2.3)
    rgb_factors: str                          # chiave dei fattori latent2rgb per la preview
    ic_lora: dict[str, str]
    caps: dict[str, str]
    ckpt_hints: tuple[str, ...]               # sottostringhe che identificano un checkpoint della release
    sigma_driven: bool                        # True = guidato da schedule di sigma, non da num_inference_steps
    te_overhead_gb: float                     # VRAM del text encoder + VAE, per il semaforo dei dropdown
    min_vram_gb: float                        # minimo realistico col profilo più leggero
    notes: str = ""
    required: tuple[str, ...] = field(default_factory=tuple)   # attributi che DEVONO esistere su disco


_LORA23 = _MODELS / "LoRA" / "LTX-2"

# ---------------------------------------------------------------------------------------------
# LTX-2.3 — checkpoint monolitico, Gemma 3, ecosistema completo (16 IC-LoRA, NSFW, GGUF).
# ---------------------------------------------------------------------------------------------
_R23 = LtxRelease(
    id="2.3",
    label="LTX 2.3",
    python=str(_TOOLS / "ltx2_venv" / "Scripts" / "python.exe"),
    worker=str(_REPO / "workers" / "ltx2_infer.py"),
    text_encoder=_P("ltx23-gemma-te", str(_MODELS / "TextEncoders" / "gemma-3-ltx")),
    video_vae="",                                     # dentro il checkpoint
    audio_vae="",                                     # dentro il checkpoint
    upsampler_spatial=_P("ltx23-spatial-upscaler",
                         str(_MODELS / "latent_upscale_models" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")),
    upsampler_temporal="",
    distilled_lora=_P("ltx23-distilled-lora-384",
                      str(_LORA23 / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")),
    msr_lora=_P("ltx23-lora-multisubject", str(_LORA23 / "LTX-2.3-Licon-MSR-V2.safetensors")),
    duration_head="",
    diffusion_decoder="",
    aux_bundle="_ltx23_aux_components.safetensors",   # risolto accanto al checkpoint
    rgb_factors="ltx23",
    ic_lora={
        "canny": _P("ltx-iclora-canny", str(_LORA23 / "ltx-2-19b-ic-lora-canny-control.safetensors")),
        "depth": _P("ltx-iclora-depth", str(_LORA23 / "ltx-2-19b-ic-lora-depth-control.safetensors")),
        "pose": _P("ltx-iclora-pose", str(_LORA23 / "ltx-2-19b-ic-lora-pose-control.safetensors")),
        "detailer": _P("ltx-iclora-detailer", str(_LORA23 / "ltx-2-19b-ic-lora-detailer.safetensors")),
        "outpaint": _P("ltx-iclora-outpaint", str(_LORA23 / "ltx-2.3-22b-ic-lora-outpaint.safetensors")),
        "refocus": _P("ltx-iclora-refocus", str(_LORA23 / "ltx-2.3-22b-ic-lora-refocus.safetensors")),
        "uncompress": _P("ltx-iclora-uncompress", str(_LORA23 / "ltx-2.3-22b-ic-lora-uncompress.safetensors")),
        "deinterlace": _P("ltx-iclora-deinterlace", str(_LORA23 / "ltx-2.3-22b-ic-lora-deinterlace.safetensors")),
        "motiondeblur": _P("ltx-iclora-motiondeblur", str(_LORA23 / "ltx-2.3-22b-ic-lora-motiondeblur.safetensors")),
        "hdr": _P("ltx-iclora-hdr", str(_LORA23 / "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors")),
        "lipdub": _P("ltx-iclora-lipdub", str(_LORA23 / "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors")),
        "restyle": _P("ltx-iclora-restyle", str(_LORA23 / "852654_LTX2.3-22B_ReStyle_IC-LoRA_8000_v0.1.safetensors")),
        "motion-track": _P("ltx-iclora-motion-track", str(_LORA23 / "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors")),
        "union": _P("ltx-iclora-union", str(_LORA23 / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")),
        # Cameraman (Civitai 2525197): IC-LoRA con video di RIFERIMENTO del movimento camera (v1, 10500 step)
        "cameraman": _P("ltx-iclora-cameraman", str(_LORA23 / "LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors")),
    },
    caps={
        "t2v": CAP_YES, "i2v": CAP_YES, "ti2v": CAP_YES, "a2v": CAP_YES, "audio": CAP_YES,
        "controlnet": CAP_YES, "v2v": CAP_YES, "retake": CAP_YES, "storyboard": CAP_YES,
        "extend": CAP_YES, "lipsync": CAP_YES, "keyframes": CAP_YES,
        "gguf": CAP_YES, "nsfw_ckpt": CAP_YES, "lora": CAP_YES,
        "multiref": CAP_YES,                  # MSR V2 LiconStudio (dal 1/06, la più provata)
        "multishot": CAP_NO, "duration_head": CAP_NO,
        "temporal_upscale": CAP_NO, "diffusion_decoder": CAP_NO, "hdr_master": CAP_NO,
    },
    ckpt_hints=("ltx-2.3", "ltx2.3", "ltx23", "ltx-2-3", "eros", "sulphur"),
    sigma_driven=False,
    te_overhead_gb=14.0,
    min_vram_gb=12.0,                                 # GGUF Q4_K_M (13,2 GB su disco)
    notes="Ecosistema completo: 16 IC-LoRA, checkpoint NSFW, GGUF. Inquadratura singola.",
    required=("text_encoder",),
)

# ---------------------------------------------------------------------------------------------
# LTX-2.5 — split pack, Gemma 4, multishot nativo. Ecosistema ancora quasi vuoto.
# ---------------------------------------------------------------------------------------------
_D25 = _MODELS / "diffusion_models"
_VAE25 = _MODELS / "vae"
_TE25 = _MODELS / "text_encoders"

_R25 = LtxRelease(
    id="2.5",
    label="LTX 2.5",
    # transformers>=5.8,<5.15 (Gemma 4) → NON può condividere il venv della 2.3, che gira su 4.x.
    python=str(_TOOLS / "lingbot_venv" / "Scripts" / "python.exe"),
    worker=str(_REPO / "workers" / "ltx2_infer.py"),
    # Text encoder bf16 UFFICIALE (24,5 GiB). Qui c'era il nvfp4 locale da 6,9 GiB, scelto
    # assumendo che il layout weight/weight_scale/weight_scale_2 fosse leggibile come per
    # flux2_transformer.Nvfp4Linear: NON lo è. Quel loader vale per il transformer; il TE Gemma
    # passa da `transformers` (ltx_core.text_encoders.gemma), che non ha alcun supporto alla
    # quantizzazione — costruisce il modello a piena larghezza e ci copia dentro i pesi. Coi pesi
    # fp4 impacchettati a metà larghezza il caricamento muore su «size mismatch [3840,1920] vs
    # [3840,3840]», ed è ciò che ha impedito alla 2.5 di generare. Le varianti compresse
    # (int8-convrot 15,4 GiB, GGUF Q5_K_M 9,5 GiB) esistono solo per la strada ComfyUI.
    text_encoder=_P("ltx25-te-bf16", str(_TE25 / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")),
    video_vae=_P("ltx25-video-vae-conv", str(_VAE25 / "ltx-2.5-video-vae-conv-bf16.safetensors")),
    audio_vae=_P("ltx25-audio-vae", str(_VAE25 / "ltx-2.5-audio-vae-bf16.safetensors")),
    # Lo spatial upscaler 2.5 esiste, ma il README rimanda ancora a quello 2.3 — che abbiamo già
    # su disco. `resolve()` prova prima il 2.5 e ricade sul 2.3 (stessa famiglia, non è una
    # sostituzione di modello: è un componente ausiliario dichiarato compatibile da Lightricks).
    upsampler_spatial=_P("ltx25-spatial-upscaler",
                         str(_MODELS / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")),
    upsampler_temporal=_P("ltx25-temporal-upscaler",
                          str(_MODELS / "latent_upscale_models" / "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors")),
    distilled_lora=_P("ltx25-distilled-lora-450",
                      str(_MODELS / "LoRA" / "LTX-2.5" / "ltx-2.5-22b-distilled-lora-450-bf16.safetensors")),
    msr_lora=_P("ltx25-lora-multisubject",
                str(_MODELS / "LoRA" / "LTX-2.5" / "LTX-2.5-Licon-MSR-V1.safetensors")),
    duration_head=_P("ltx25-duration-head", str(_MODELS / "model_patches" / "ltx-2.5-duration-head-bf16.safetensors")),
    diffusion_decoder=_P("ltx25-diffusion-decoder", str(_VAE25 / "ltx-2.5-video-vae-bf16.safetensors")),
    aux_bundle="",
    rgb_factors="ltx25",
    # Unico adapter ufficiale 2.5 al 11/08/2026. Tutto il resto del controllo (canny/depth/pose/
    # union/HDR/DubIt/outpaint/…) esiste solo per 2.3 → capability "cross".
    ic_lora={
        "pixel-upscaler": _P("ltx25-iclora-pixel-upscaler",
                             str(_MODELS / "LoRA" / "LTX-2.5" / "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors")),
    },
    caps={
        "t2v": CAP_YES, "i2v": CAP_YES, "ti2v": CAP_YES, "a2v": CAP_YES, "audio": CAP_YES,
        "v2v": CAP_YES, "retake": CAP_YES, "storyboard": CAP_YES, "extend": CAP_YES,
        "keyframes": CAP_YES,
        # Le pipeline ICLora/DubIt esistono nel codice 1.2.0, ma i PESI per 2.5 no: funzionano solo
        # riusando gli adapter 2.3 → l'utente va avvisato, non bloccato.
        "controlnet": CAP_CROSS, "lipsync": CAP_CROSS, "lora": CAP_CROSS,
        # GGUF 2.5: ESISTONO dal 12/08 (community: Abiray 28K dl Q3-Q8, realrebelai Q2-Q8) — a
        # catalogo le voci ltx25-gguf-*; 1a generazione GGUF 2.5 col worker nativo da validare live.
        "gguf": CAP_YES,
        "nsfw_ckpt": CAP_YES,                 # Stubelius Remix v1 (13/10: 10Eros 1.5 + distill fusi, int8 ConvRot)
        "multishot": CAP_YES, "duration_head": CAP_YES,
        "temporal_upscale": CAP_YES, "diffusion_decoder": CAP_YES, "hdr_master": CAP_YES,
        # Riferimenti multipli (MSR): LoRA community LiconStudio, una per release. La V1 per 2.5 è
        # recente (14/08) e molto meno provata della V2 per 2.3 (dal 1/06).
        "multiref": CAP_YES,
    },
    ckpt_hints=("ltx-2.5", "ltx2.5", "ltx25", "ltx-2-5"),
    sigma_driven=True,                        # DISTILLED_SIGMAS (9 valori = 8 step); num_inference_steps IGNORATO
    # MISURATI dagli header l'11/08/2026 (non stimati): TE nvfp4 6,91 GiB + Conv VAE 1,35 + audio 0,34.
    te_overhead_gb=8.6,
    # Il transformer nvfp4 è 'mixed': 7,55 GiB FP4 packed + 0,94 di scale + 8,93 BF16 NON quantizzati
    # = 17,43 GiB residenti. Coi GGUF community (dal 12/08: Q4_K_M 14,6 GiB residente quantizzato,
    # Q2_K 8,2) il minimo realistico scende come per la 2.3 → 14 GB (Q2/Q3 + TE offload).
    min_vram_gb=14.0,
    notes="Multishot nativo (2-4 shot, guidato dal prompt), duration head, decoder a diffusione, "
          "upscaler temporale. GGUF community dal 12/08 (Q2-Q8); LoRA community in crescita.",
    required=("text_encoder", "video_vae", "audio_vae"),
)

LTX_RELEASES: dict[str, LtxRelease] = {"2.3": _R23, "2.5": _R25}
DEFAULT_RELEASE = "2.3"


def is_distilled(path_or_name: str | None) -> bool:
    """True se il checkpoint e' un DISTILLED (stessa euristica del worker, workers/ltx2_infer.py: la
    parola «distilled» nel NOME del file; la cartella non conta). Vuoto/None/«auto» -> False."""
    if not path_or_name:
        return False
    return "distilled" in os.path.basename(str(path_or_name).replace("\\", "/")).lower()


def hdr_scene_emb() -> str:
    """File scene-emb della IC-LoRA HDR (12,6 MB, embedding testo pre-calcolato che sostituisce Gemma: la pipeline HDR
    non ha prompt). Stesso id di catalogo del worker (`ltx-iclora-hdr-scene-emb`), solo release 2.3."""
    return str(_P("ltx-iclora-hdr-scene-emb", str(_LORA23 / "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors")))


def ic_lora_supported(release: str | None) -> bool:
    """True se le IC-LoRA di editing/restauro locali si possono usare con la release: oggi solo la 2.3
    (tutti i pesi locali sono per 2.3; con la 2.5 l'unico adapter ufficiale e' il pixel-upscaler)."""
    return str(release or "") == "2.3"


def get(release: str | None) -> LtxRelease:
    """Release per id. Sconosciuta → 2.3 (la release storica), MAI una scelta implicita di modello."""
    return LTX_RELEASES.get(str(release or "").strip(), LTX_RELEASES[DEFAULT_RELEASE])


def detect_release(path_or_name: str) -> str | None:
    """Release di appartenenza di un checkpoint, dal nome file. `None` se non riconoscibile:
    il chiamante deve dare errore chiaro, non indovinare (regola: nessuna sostituzione automatica)."""
    n = os.path.basename((path_or_name or "").replace("\\", "/")).lower()
    if not n:
        return None
    for rel in LTX_RELEASES.values():
        if any(h in n for h in rel.ckpt_hints):
            return rel.id
    return None


def cap(release: str | None, name: str) -> str:
    """Stato di una capability: CAP_YES | CAP_NO | CAP_CROSS. Sconosciuta → CAP_NO."""
    return get(release).caps.get(name, CAP_NO)


def missing_components(release: str | None) -> list[str]:
    """Componenti obbligatorie assenti dal disco, come nomi di campo (es. ['video_vae'])."""
    rel = get(release)
    out = []
    for attr in rel.required:
        p = getattr(rel, attr, "")
        if not p or not os.path.exists(p):
            out.append(attr)
    return out


# componente obbligatoria → id di CATALOGO scaricabile (FASE G: il consenso primo-uso della GUI
# passa da qui). NB 2.5: il TE del path è il nvfp4 DERIVATO LOCALMENTE (no_remote); il DEFAULT
# di DOWNLOAD è il GGUF Q5_K_M community (8.9GB, scelta utente 16/08 — il bf16 da 24.5GB non
# serve più). ⚠ il loader del TE GGUF nel worker nativo è DA VALIDARE alla prima generazione:
# finché non è validato NON è il default di RUNTIME (il path resta il nvfp4 locale).
_COMPONENT_IDS: dict[str, dict[str, str]] = {
    "2.3": {"text_encoder": "ltx23-gemma-te"},
    "2.5": {"text_encoder": "ltx25-te-gemma4-gguf", "video_vae": "ltx25-video-vae-conv",
            "audio_vae": "ltx25-audio-vae"},
}


def missing_catalog_ids(release: str | None) -> list[str]:
    """Id di catalogo SCARICABILI per le componenti obbligatorie mancanti (per il dialog di
    consenso primo-uso). Componenti senza id scaricabile restano fuori: le copre l'errore chiaro."""
    rel = get(release)
    ids = _COMPONENT_IDS.get(rel.id, {})
    return [ids[c] for c in missing_components(release) if c in ids]


def resolve(release: str | None, attr: str) -> str:
    """Path di una componente, con l'UNICO fallback ammesso: lo spatial upscaler della 2.5 ricade su
    quello 2.3 (Lightricks lo indica esplicitamente come ancora valido). Nessun altro ripiego:
    un componente mancante deve emergere come errore, non essere sostituito in silenzio."""
    rel = get(release)
    p = getattr(rel, attr, "") or ""
    if p and os.path.exists(p):
        return p
    if attr == "upsampler_spatial":
        alt = _R23.upsampler_spatial
        if os.path.exists(alt):
            return alt
    return p


def is_ready(release: str | None) -> bool:
    """True se la release ha tutte le componenti obbligatorie su disco e almeno un checkpoint."""
    return not missing_components(release) and bool(checkpoints(release))


# Sottostringhe che identificano un COMPONENTE, non un transformer: senza questo filtro la
# scansione conta come "checkpoint" anche text projection, VAE, upscaler e LoRA che stanno nelle
# stesse cartelle (es. `ltx-2.3_text_projection_bf16.safetensors` in Checkpoints\LTX).
# DiT di altre architetture video negli stessi diffusion_models: i finetune MiniMax H3 contengono «eros»
# nel nome (10Eros-Max, PinkCherry H3) e passerebbero per checkpoint LTX. Usato anche dal tab Video.
OTHER_ARCH_MARKERS = ("minimax", "_h3_", "_h3.", "-h3-")

_NOT_A_CKPT = (
    "text_projection", "text-projection", "text_encoder", "text-encoder",
    "vae", "upscaler", "upsampler", "lora", "_aux_", "duration-head", "duration_head",
    "connector", "vocoder", "embeddings-connector",
    *OTHER_ARCH_MARKERS,
)


def checkpoints(release: str | None) -> list[str]:
    """Transformer della release presenti su disco (scansione di diffusion_models/ e Checkpoints/).
    Ordinati per nome; nessun filtro di qualità: la scelta resta dell'utente."""
    rel = get(release)
    found: list[str] = []
    for root in (_MODELS / "diffusion_models", _MODELS / "Checkpoints"):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() not in (".safetensors", ".gguf") or not p.is_file():
                continue
            low = p.name.lower()
            if any(x in low for x in _NOT_A_CKPT):
                continue
            if detect_release(p.name) == rel.id:
                found.append(str(p))
    return sorted(found)


def installed() -> list[str]:
    """Release utilizzabili qui e ora, in ordine di id."""
    return [r for r in sorted(LTX_RELEASES) if is_ready(r)]


def runner(release: str | None) -> tuple[str, str]:
    """(python, worker) della release — sostituisce il lookup fisso in `_ARCH_RUNNER['ltx']`."""
    rel = get(release)
    return rel.python, rel.worker


def summary(release: str | None) -> str:
    """Riga di sintesi per il selettore GUI: cosa ottieni scegliendo questa release."""
    rel = get(release)
    bits: list[str] = []
    # Conta le IC-LoRA REALMENTE su disco, non quelle dichiarate: la riga dice cosa ottieni *qui e
    # ora*. Il dialog di confronto mostra invece il catalogo completo della release.
    n_ic = sum(1 for p in rel.ic_lora.values() if os.path.exists(p))
    if rel.caps.get("multishot") == CAP_YES:
        bits.append("multishot")
    bits.append(f"{n_ic} IC-LoRA" if n_ic != 1 else "1 IC-LoRA")
    if rel.caps.get("nsfw_ckpt") == CAP_YES:
        bits.append("NSFW")
    if rel.caps.get("gguf") == CAP_YES:
        bits.append("GGUF")
    if rel.caps.get("duration_head") == CAP_YES:
        bits.append("durata auto")
    bits.append(f"min {rel.min_vram_gb:.0f} GB")
    return " · ".join(bits)


# ── ARCH DALLE CHIAVI (regola «arch dalle chiavi dell'header, non dai metadati/catalogo») ─────────────────
_NOT_LTX_CACHE: dict = {}


def header_says_not_ltx(path: str) -> bool:
    """True SOLO se l'header safetensors mostra con certezza che NON è un transformer LTX.
    Un transformer LTX (2.0/2.3/2.5, qualsiasi quantizzazione) ha blocchi `transformer_blocks.N.attn1.*`
    (self-attention video) e quelli audio `audio_attn1`; i DiT Qwen-Image/FLUX hanno `transformer_blocks.N.attn.*`
    / `img_mlp` / `double_blocks`. Il catalogo esterno classificava per errore i Noct-Q (Qwen-Image 2.1) come
    «LTX» e finivano nel menu Video. Prudente: file illeggibile, .gguf, o senza `transformer_blocks` → False
    (nessuna esclusione: il chiamante usa le sue regole). Solo lettura dell'header (pochi KB), cache per path."""
    key = os.path.normcase(str(path or ""))
    if key in _NOT_LTX_CACHE:
        return _NOT_LTX_CACHE[key]
    out = False
    try:
        if key.endswith(".safetensors") and os.path.isfile(path):
            import json
            import struct
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if 0 < n < 512 * 2**20:
                    keys = [k for k in json.loads(fh.read(n)) if k != "__metadata__"]
                    has_blocks = any("transformer_blocks." in k for k in keys)
                    has_attn1 = any(".attn1." in k or ".audio_attn1." in k for k in keys)
                    out = has_blocks and not has_attn1
    except Exception:                                     # noqa: BLE001
        out = False
    _NOT_LTX_CACHE[key] = out
    return out
