"""MiniMax H3: stima del picco di VRAM del DiT e decisione «ci sta / ci sta con offload / non ci sta».

Puro Python (niente torch): lo usano il worker (workers/minimax_h3_infer.py) e i test su CPU.

Perché serve: il DiT lavora su UNA sequenza impacchettata (testo + condizionamenti + audio + righe video) e le
attivazioni crescono linearmente con i token. A 1376×736 e 10 s sono ~72k token: il solo qkv è 21504 canali ×
72k × 2 byte = 3 GB. Il 29/10 la campagna ha dato 51 OOM su 51 perché la stima era una soglia fissa (26 GB) e le
LoRA sommavano 3-4 temporanei grandi quanto il qkv. Qui la stima è per token e il confronto è con la VRAM
LIBERA di adesso: nessuna soglia fissa sui GB della scheda (regola «VRAM dinamica»).

Modello di costo (bf16, elementi per token), fase peggiore del blocco:
  · attenzione: residuo + normalizzato (2·h) + qkv (3·i) + q,k normalizzati (2·i) + temporanei rotary (2·i)
    + uscita dell'attenzione (i)                                         = 2·h + 8·i
  · MLP SwiGLU: residuo + normalizzato (2·h) + gate_up (2·f) + silu·up (2·f) + uscita (h)
                                                                          = 3·h + 4·f
Con le LoRA sommate IN PLACE a blocchi (workers/_qlinear.py) non c'è più nessun temporaneo da sequenza intera.
Se pesi + attivazioni non stanno nella VRAM libera, una parte dei blocchi del DiT resta in RAM e va in VRAM
solo durante il proprio forward (workers/h3_offload.py): quanti dipende dal deficit misurato.
"""
from __future__ import annotations

import math

# dimensioni del DiT pruned ufficiale (config.json di transformer/): usate se il config non è leggibile
DEFAULT_DIMS = {"hidden": 5376, "inner": 56 * 128, "ffn": 14336}
SAFETY = 1.15                  # frammentazione dell'allocatore, temporanei piccoli non contati
RESERVE_GB = 1.0               # contesto CUDA, cache dell'allocatore, VAE di condizionamento
TEXT_TOKENS = 2048             # prompt strutturato + dialoghi (il TE ne produce di solito < 1500)
AUDIO_LATENTS_PER_S = 40       # audio VAE di H3
AUDIO_CHANNELS = 2             # righe audio = latenti × canali (stereo, impacchettate per canale)
FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 17, 5      # num_frames = 17·n + 5  →  latenti = 5·n + 2
VAE_SPATIAL, PATCH = 16, 2     # compressione spaziale del VAE e patch (1,2,2) del DiT
FPS = 24
OFFLOADABLE = 0.85             # quota dei pesi del DiT che si può spostare in RAM (restano residenti ≥ 15 %)


def dims_from_config(cfg: dict | None) -> dict:
    """hidden / inner / ffn dal config.json del transformer (o i default)."""
    d = dict(DEFAULT_DIMS)
    if cfg:
        if cfg.get("hidden_size"):
            d["hidden"] = int(cfg["hidden_size"])
        if cfg.get("num_attention_heads") and cfg.get("attention_head_dim"):
            d["inner"] = int(cfg["num_attention_heads"]) * int(cfg["attention_head_dim"])
        if cfg.get("ffn_dim"):
            d["ffn"] = int(cfg["ffn_dim"])
    return d


def latent_frames(num_frames: int) -> int:
    """Fotogrammi (17·n + 5) → fotogrammi latenti del VAE (5·n + 2)."""
    n = max(0, (int(num_frames) - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK)
    return LATENTS_PER_CHUNK * n + 2


def rows_per_latent_frame(height: int, width: int) -> int:
    step = VAE_SPATIAL * PATCH
    return (int(height) // step) * (int(width) // step)


def sequence_tokens(num_frames: int, height: int, width: int, *, text_tokens: int = TEXT_TOKENS,
                    keyframes: int = 0, ref_rows: int = 0) -> int:
    """Lunghezza della sequenza impacchettata: testo + righe di keyframe + righe audio + righe video target
    + righe dei riferimenti (Ref2VA, stimate dal chiamante)."""
    rpf = rows_per_latent_frame(height, width)
    audio = int(round(num_frames / FPS * AUDIO_LATENTS_PER_S)) * AUDIO_CHANNELS
    return int(text_tokens) + int(keyframes) * rpf + audio + latent_frames(num_frames) * rpf + int(ref_rows)


def ref_rows_estimate(references: list | None, short_edge: int, height: int, width: int) -> int:
    """Righe dei riferimenti Ref2VA: ogni immagine a `short_edge` di lato corto (≈16:9 se ignoto), ogni video
    come 8 fotogrammi latenti alla tela. Stima per eccesso, serve solo a non sbagliare di molto."""
    total = 0
    step = VAE_SPATIAL * PATCH
    for r in references or []:
        kind = (r or {}).get("kind") or "image"
        if kind == "image":
            total += math.ceil(short_edge / step) * math.ceil(short_edge * 16 / 9 / step)
        elif kind == "video":
            total += 8 * rows_per_latent_frame(height, width)
    return total


def bytes_per_token(dims: dict) -> float:
    """Byte per token del picco di attivazioni (la fase peggiore del blocco)."""
    h, i, f = dims["hidden"], dims["inner"], dims["ffn"]
    return 2.0 * max(2 * h + 8 * i, 3 * h + 4 * f) * SAFETY


def activation_gb(tokens: int, dims: dict) -> float:
    return tokens * bytes_per_token(dims) / 2**30


class Plan(dict):
    """Esito di plan(): `fits` (ci sta, eventualmente con offload), `offload_gb` (GB di blocchi da tenere in
    RAM, 0 = tutto residente), `need_gb`, `act_gb`, `tokens`."""


def plan(free_gb: float, tokens: int, dims: dict | None = None, *, weights_gb: float = 0.0,
         lora_gb: float = 0.0, reserve_gb: float = RESERVE_GB) -> Plan:
    """Decide come far stare il DiT in `free_gb` di VRAM libera (pesi ancora da caricare: `weights_gb`; già
    caricati: 0, e `free_gb` è quello rimasto).
      · deficit = pesi + LoRA + attivazioni + riserva − libera
      · deficit ≤ 0                            → tutto residente (offload_gb = 0)
      · 0 < deficit ≤ OFFLOADABLE · pesi       → offload di `deficit` GB di blocchi
      · altrimenti                             → fits False
    Il confronto è sempre con la VRAM libera passata (misurata dal chiamante), mai con i GB della scheda."""
    dims = dims or DEFAULT_DIMS
    act = activation_gb(tokens, dims)
    resident = weights_gb + lora_gb
    need = resident + reserve_gb + act
    deficit = need - free_gb
    fits = deficit <= OFFLOADABLE * resident if resident > 0 else deficit <= 0
    if resident > 0 and free_gb < resident + 0.5:        # il caricamento iniziale porta tutto in VRAM
        fits = False
    return Plan(tokens=tokens, free_gb=free_gb, weights_gb=weights_gb, lora_gb=lora_gb, act_gb=act,
                need_gb=need, offload_gb=max(0.0, deficit), fits=fits)


def max_frames_that_fit(free_gb: float, height: int, width: int, dims: dict | None = None, *,
                        weights_gb: float = 0.0, lora_gb: float = 0.0, extra_rows: int = 0,
                        min_s: float = 5.0, max_s: float = 15.0) -> int:
    """Il massimo numero di fotogrammi 17·n+5 (≥ min_s) che ci sta, 0 se nemmeno il minimo."""
    best = 0
    n = max(0, math.ceil((round(min_s * FPS) - 5) / FRAMES_PER_CHUNK))
    while True:
        frames = FRAMES_PER_CHUNK * n + 5
        if frames / FPS > max_s + 1e-6:
            break
        t = sequence_tokens(frames, height, width, ref_rows=extra_rows)
        if not plan(free_gb, t, dims, weights_gb=weights_gb, lora_gb=lora_gb)["fits"]:
            break
        best = frames
        n += 1
    return best


def explain_failure(p: Plan, *, height: int, width: int, num_frames: int, free_gb: float,
                    dims: dict | None = None, extra_rows: int = 0, canvas_544: tuple | None = None) -> str:
    """Messaggio per l'utente quando non ci sta: numeri e cosa cambiare (nessuna scelta automatica)."""
    msg = (f"VRAM insufficiente per MiniMax H3 a {width}x{height}, {num_frames / FPS:.0f} s "
           f"({p['tokens'] // 1000}k token): servono circa {p['need_gb']:.1f} GB "
           f"(pesi {p['weights_gb']:.1f} + LoRA {p['lora_gb']:.1f} + attivazioni {p['act_gb']:.1f} + riserva), "
           f"ne sono liberi {free_gb:.1f}, anche tenendo in RAM parte dei blocchi.")
    ok = max_frames_that_fit(free_gb, height, width, dims, weights_gb=p["weights_gb"], lora_gb=p["lora_gb"],
                             extra_rows=extra_rows)
    tips = []
    if ok:
        tips.append(f"riduci la durata a {ok / FPS:.0f} s o meno")
    if canvas_544 is not None:
        h2, w2 = canvas_544
        t2 = sequence_tokens(num_frames, h2, w2, ref_rows=extra_rows)
        if plan(free_gb, t2, dims, weights_gb=p["weights_gb"], lora_gb=p["lora_gb"])["fits"]:
            tips.append(f"scegli la tela 544p ({w2}x{h2})")
    tips.append("chiudi i programmi che usano la GPU")
    tips.append("usa un profilo più leggero (int4) o meno LoRA")
    return msg + " Prova a: " + "; ".join(tips) + "."
