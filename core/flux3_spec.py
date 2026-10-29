"""Costanti di FLUX.3 Image Klein condivise fra GUI, API, catalogo e worker (stdlib-only, nessun effetto all'import).

Fonti (task 13, ricerca 18/10/2026 + codice ufficiale BFL, Apache-2.0: github.com/black-forest-labs/flux-3-image-klein):
  · DiT 5,55 B: hidden 3072, 24 teste, 5 blocchi per modalità (image, image_cond, text) + 28 single-stream condivisi;
    distillato = «flow map» (16 passi, SENZA CFG né negative), Base = 50 passi, CFG 3,5;
  · text encoder Qwen3-VL-4B-Instruct, strati 4,8,…,32 impilati → 20480 canali, padding a multipli di 80;
  · VAE = quello di FLUX.2 (128 canali latenti, fattore 16) — il nostro `flux2-vae.safetensors`;
  · prompt upsampler (PUP): Qwen3.5-9B fine-tune, SEMPRE consigliato da BFL.
Licenza FLUX Kommunity v1.0: l'app NON distribuisce i pesi, li scarica l'utente dal tab Pesi con avviso e conferma.
"""
from __future__ import annotations

FAMILY = "flux3"                       # famiglia di core/arch_filter e core/model_detect
ARCH = "flux3"                         # arch del job (core/img_gen_job)
ARCH_LABEL = "FLUX.3 Klein"            # voce del selettore «Architettura» del tab Immagini
CAP = "cap.image_flux3"                # capability della feature matrix (celle vuote: le spunta l'utente)
CAP_BOXES = "cap.flux3_boxes"
CAP_PUP = "cap.flux3_upsampler"

# config del VAE (AutoencoderKLFlux2): identica a vae/config.json del repo FLUX.3 (letta dal mirror il 18/10/2026) e al
# VAE di FLUX.2 (stesso sha256 del file di pesi). Inline per non dipendere da un download gated per un json di 800 byte.
VAE_CONFIG = {
    "_class_name": "AutoencoderKLFlux2", "act_fn": "silu", "batch_norm_eps": 0.0001, "batch_norm_momentum": 0.1,
    "block_out_channels": [128, 256, 512, 512], "decoder_block_out_channels": None,
    "down_block_types": ["DownEncoderBlock2D"] * 4, "force_upcast": True, "in_channels": 3, "latent_channels": 32,
    "layers_per_block": 2, "mid_block_add_attention": True, "norm_num_groups": 32, "out_channels": 3,
    "patch_size": [2, 2], "sample_size": 1024, "up_block_types": ["UpDecoderBlock2D"] * 4,
    "use_post_quant_conv": True, "use_quant_conv": True,
}

# ── voci di catalogo (build/catalog_task13.py) ───────────────────────────────────────────────────────────
DIT_FP8R = "flux3-dit-fp8r"            # distillato, fp8 per riga (6,27 GB) — opzione A scelta dall'utente
TE = "flux3-te"                        # Qwen3-VL-4B-Instruct bf16 (8,88 GB) + tokenizer: l'UFFICIALE (alternativa)
TE_KREA2 = "krea2-te-uncensored"       # Qwen3-VL-4B uncensored fp8 (4,85 GB) già su disco: PREFERITO se presente (utente 29/10)
TE_KREA2_TURBO = "krea2_turbo"         # lo snapshot di Krea 2 Turbo contiene il TE UFFICIALE di FLUX.3 (identico, verificato 29/10): si riusa, niente download
TE_KREA2_TURBO_FILE = "text_encoder/model.safetensors"   # Qwen3VLModel bf16 a file singolo (8,88 GB), nomi `language_model.…` senza `model.`
TE_CONFIGS = "flux3-te-configs"        # json + tokenizer + chat template del Qwen3-VL-4B (~12 MB): servono col TE a file singolo
TE_CHOICES = ("", "krea2", "official")  # "" = automatico: Krea2 se presente, altrimenti l'ufficiale
VAE = "flux2-vae"                      # già su disco, sha256 identico a quello di FLUX.3
PUP_SAKURA_Q8 = "flux3-pup-sakura-q8"  # upsampler abliterato di terzi (GGUF Q8_0): in attesa della decisione
PUP_DEFAULT = PUP_SAKURA_Q8
PUP_IDS = (PUP_SAKURA_Q8,)
DITS = {DIT_FP8R: {"distilled": True, "steps": 16, "cfg": None}}
DEFAULT_DIT = DIT_FP8R

# ── campionamento ────────────────────────────────────────────────────────────────────────────────────────
DEFAULT_STEPS = 16                     # distillato (addestrato 4–16)
MIN_STEPS, MAX_STEPS = 4, 16
BASE_STEPS, BASE_GUIDANCE = 50, 3.5    # Base (non scaricato: training LoRA, fuori dal task)
SHIFT_ALPHA = 6.93                     # config.json «sampler.alpha», diffusers e ai-toolkit (il codice GitHub ha 16: INCOERENZA)
SEED_MAX = 2 ** 32 - 1

# ── limiti dal codice ufficiale (src/flux3/pipeline.py) ──────────────────────────────────────────────────
GRID = 16                              # lato multiplo di 16 (VAE ×8 · patch 2×2)
MIN_SIDE, MAX_SIDE = 64, 4096
DEFAULT_SIDE = 1024
VAE_TILE_ABOVE_PIXELS = 1536 * 1536            # oltre questa area il VAE decodifica a tile (tile 1024, sovrapposizione 25 %)
MAX_REFS = 10
FIRST_REF_MAX_PIXELS = 4 * 1024 * 1024         # primo riferimento (edit singolo), canvas compreso
MULTI_REF_MAX_PIXELS = 1024 * 1024             # ogni riferimento di un edit multiplo
IMAGE_ROPE_SCALE = 0.5                         # posizioni RoPE dei riferimenti ×0,5
TE_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
TE_PAD_MULTIPLE = 80
TE_MAX_TOKENS = 4096                           # il codice ufficiale; config.json dice 8192 (INCOERENZA)
TXT_CHANNELS = 20480                           # 8 strati × 2560

# ── prompt upsampler ─────────────────────────────────────────────────────────────────────────────────────
PUP_MAX_NEW_TOKENS = 3584
PUP_SAMPLING = {"temperature": 0.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
PUP_PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
PUP_REF_MULTIPLE = 32
PUP_MAX_ASPECT = 200
# pixel massimi per riferimento passato all'upsampler, secondo il numero di riferimenti
PUP_MAX_PIXELS = {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, MAX_REFS + 1)}}

# ── riquadri (bounding box) ──────────────────────────────────────────────────────────────────────────────
BOX_SCALE = 1000                               # griglia 0–1000, origine in alto a sinistra, [y_min, x_min, y_max, x_max]
BOX_MODES = ("new", "move", "anchor", "remove")     # edit: nuovo / sposta / ancora (resta) / rimuovi
MAX_BOXES = 32                                 # nostro limite prudente (nel codice ufficiale non c'è)

# ── VRAM (stima, GB): pesi + attivazioni; il worker la rimisura con NVML e non assume mai VRAM tutta libera ──
VRAM_DIT_FP8_GB = 6.3
VRAM_TE_GB = 8.9
VRAM_VAE_GB = 0.4
VRAM_PUP_BF16_GB = 18.8


def estimate_vram_gb(width: int, height: int, n_refs: int = 0, with_pup: bool = False) -> dict:
    """Stima del picco per FASE (i modelli grandi non coesistono): {'pup','te','dit','vae','peak'} in GB.
    Token DiT = (w/16)·(h/16) + riferimenti; attenzione fusa (SDPA) → attivazioni ~ lineari nei token."""
    tok = (width // GRID) * (height // GRID)
    for _ in range(n_refs):
        tok += (1024 * 1024) // (GRID * GRID)
    act = 1.0 + tok * 3072 * 2 * 40 / 2 ** 30          # ~40 tensori bf16 per token vivi nel blocco single-stream
    dit = VRAM_DIT_FP8_GB + act
    vae = VRAM_VAE_GB + (width * height) * 3 * 4 * 24 / 2 ** 30 / 8     # decode a tile implicite: stima larga
    out = {"pup": VRAM_PUP_BF16_GB + 2.0 if with_pup else 0.0, "te": VRAM_TE_GB + 1.0, "dit": dit, "vae": vae}
    out["peak"] = max(out.values())
    return out


def clamp_size(width: int, height: int) -> tuple[int, int]:
    """Lati portati al multiplo di 16 nell'intervallo 64–4096 (stessa regola del codice ufficiale)."""
    f = lambda v: max(MIN_SIDE, min(MAX_SIDE, int(round(int(v) / GRID)) * GRID))
    return f(width), f(height)


def disk_present(entry_id: str) -> bool:
    """Voce di catalogo presente su disco. Per lo snapshot di Krea 2 Turbo conta solo se contiene il file del text encoder."""
    from pathlib import Path
    from core import model_paths as mp
    from core import weights_catalog as wc
    e = mp.entry(entry_id)
    if not e or not wc.is_present(e, str(mp.models_root())):
        return False
    if entry_id == TE_KREA2_TURBO:
        return (Path(mp.p(entry_id, "")) / TE_KREA2_TURBO_FILE).is_file()
    return True


def te_id(choice: str = "", present=None) -> str:
    """Voce di catalogo del text encoder. `choice`: "krea2" (Krea2 uncensored fp8) | "official" | "" (automatico: il Krea2 uncensored
    se è su disco — scelta dell'utente del 29/10 —, altrimenti l'ufficiale). L'«ufficiale» è lo snapshot `flux3-te` se c'è, altrimenti
    il TE già dentro Krea 2 Turbo (identico: nessun download); se non c'è né l'uno né l'altro resta `flux3-te` (da scaricare).
    `present(id) -> bool` dice cosa c'è su disco (`disk_present`)."""
    if choice == "krea2":
        return TE_KREA2
    if choice != "official" and present is not None and present(TE_KREA2):
        return TE_KREA2
    if present is not None and not present(TE) and present(TE_KREA2_TURBO):
        return TE_KREA2_TURBO
    return TE


def default_steps(dit_id: str = DEFAULT_DIT) -> int:
    return int(DITS.get(dit_id, DITS[DEFAULT_DIT])["steps"])
