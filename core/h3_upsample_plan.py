"""H3 Video Upsampler: la parte PURA (nessun modello, nessuna GPU) — geometria, segmenti, tile, giunture.

Porting nativo di `tiled_upsampler.py` / `video_upsampler.py` di ComfyUI-H3-Video-Upsampler (sandpies / dntpi,
MIT; vedi docs/THIRD_PARTY_NOTICES_H3_UPSAMPLER.md): stesse funzioni e stessi numeri della ricetta, senza ComfyUI.
Il codice che usa il modello sta in workers/h3_upsample_infer.py (venv diffusers_next).

La ricetta (verificata sul sorgente studiato, non decisa qui):
  · segmenti temporali di 102 fotogrammi con 17 di sovrapposizione, confini sui «passi chiave» del VAE (indice % 5 == 0,
    cioè ogni 17 fotogrammi), giunti con dissolvenza lineare nello spazio latente;
  · ogni segmento: upscaler latente → UN passo di ri-disegno a tile con denoise 0,1 (guidato) o 0,2 (ricetta senza guida);
  · tile = il minor numero di tile uguali la cui forward sta nel budget di 70.000 token, sovrapposizione 128 px, lato
    minimo 256 px, passi eseguiti INSIEME e fusi con finestra coseno rialzato → nessuna cucitura.
Le unità sono quelle del latente H3: 16 px per cella, patch 2×2 (32 px), 24 canali.
"""
from __future__ import annotations

import math

# --- la ricetta ---------------------------------------------------------------------------------------------
MEGAPIXELS = 2.0              # 1 MP = 1024 × 1024 px
SIZE_ALIGN = 32               # i lati si agganciano a 32 px (a 32 × fattore con la guida)
CHUNK_FRAMES = 102            # fotogrammi per segmento (multiplo di 17)
CHUNK_OVERLAP = 17            # fotogrammi condivisi fra segmenti vicini
ANCHOR_AUG = 0.999            # livello t a cui sono tenuti il perno del frame 0 e la guida (convenzione H3: 1 = pulito)
TOKEN_BUDGET = 70000          # sequenza massima di una forward di tile (tetto e ripiego del budget automatico)
TOKEN_MIN = 8192              # sotto questo il budget automatico non scende (il nodo originale: min 8192)
# Budget automatico dalla VRAM libera (regola «VRAM dinamica»): ogni token di una forward del DiT H3 ne impegna ~BYTES_PER_TOKEN
# (attivazioni di 50 blocchi, FFN swiglu e qkv; attenzione flash senza n²). STIMA da tarare sul «picco VRAM» dei test live.
BYTES_PER_TOKEN = 160 * 1024
VRAM_RESERVE_GB = 2.0         # margine per VAE/contesto/altri processi
TILE_OVERLAP_PX = 128
MIN_TILE_PX = 256
MAX_GRID_SIDE = 9
STEPS = 1
SEED = 444
VIDEO_SHIFT = 12.0            # shift sigma del video H3 (il «0,2» della ricetta è una frazione dello SCHEDULE, non del rumore)
DENOISE_DEFAULT = 0.1         # default del nodo H3 Video Upsampler (video_upsampler.py: «the setting that held, by eye, guided»)
DENOISE_RECIPE = 0.2          # valore della ricetta della catena (tiled_upsampler.py), solo come riferimento

SPATIAL = 16                  # pixel per cella latente
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0     # passi latenti audio per fotogramma (40 Hz / 24 fps)
H3_FPS = 24
GRID_PX = 32                  # le tele H3 sono multiple di 32 px

# --- statistiche del latente ---------------------------------------------------------------------------------
# L'upscaler è addestrato sul latente NORMALIZZATO con queste medie/deviazioni (24 canali). Sono IDENTICHE a
# latents_mean/latents_std del VAE video H3 (config.json di MiniMaxAI/MiniMax-H3, verificato in
# tests/test_h3_upsample_plan.py): in diffusers il latente è già normalizzato, quindi l'upscaler lo legge così com'è.
LATENT_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENT_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180265264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)

# La didascalia con cui è addestrata la LoRA h3upscale: il ri-disegno guidato la usa al posto del prompt dell'utente.
CAPTION = ("h3upscale, increase resolution and restore fine details while "
           "preserving the original content and visual style.")


# --- intestazione della LoRA h3upscale (guida) -----------------------------------------------------------------
SUPPORTED_SPATIAL = ("target_crop_v2",)
SUPPORTED_POSITION = ("target_grid_stride_v1",)


def guide_factor_from_meta(meta: dict | None) -> int:
    """Il fattore di guida dichiarato dall'intestazione di una LoRA; 0 se non è una LoRA di guida.
    Solleva ValueError su una geometria che questo codice non implementa (meglio rifiutare che impacchettare righe
    su cui la LoRA non è mai stata addestrata)."""
    import json
    meta = meta or {}
    try:
        factor = int(meta.get("reference_downscale_factor", 0) or 0)
    except (TypeError, ValueError):
        factor = 0
    if factor <= 1:
        return 0
    spatial = meta.get("minimax_h3_guide_spatial_version")
    position = meta.get("minimax_h3_guide_position_version")
    if spatial not in SUPPORTED_SPATIAL or position not in SUPPORTED_POSITION:
        raise ValueError(f"geometria di guida non supportata spatial={spatial!r} position={position!r}; "
                         f"implementate: {SUPPORTED_SPATIAL[0]} / {SUPPORTED_POSITION[0]}")
    if str(meta.get("guide_latent_only", "true")).lower() != "true":
        raise ValueError("la LoRA è addestrata con la guida mostrata anche al text encoder: non supportato")
    rope = json.loads(meta.get("minimax_h3_reference_rope", "{}") or "{}")
    if rope.get("guide_rope_layout", "overlap") != "overlap":
        raise ValueError(f"guide_rope_layout {rope['guide_rope_layout']!r} non supportato")
    return factor


# --- geometria della tela ---------------------------------------------------------------------------------------
def valid_length(n: int) -> int:
    """La lunghezza H3 più corta (17g + 5) che contiene n fotogrammi."""
    n = max(5, int(n))
    return 17 * math.ceil((n - 5) / 17) + 5


def auto_token_budget(free_vram_gb: float | None) -> int:
    """Budget di token per forward dalla VRAM LIBERA (letta con NVML DOPO aver caricato il DiT): tetto 70.000 (la ricetta),
    pavimento 8.192; senza misura (None/inf) il valore della ricetta. Multiplo di 1024."""
    if free_vram_gb is None or free_vram_gb != free_vram_gb or free_vram_gb == float("inf"):
        return TOKEN_BUDGET
    b = int(max(0.0, float(free_vram_gb) - VRAM_RESERVE_GB) * 2 ** 30 / BYTES_PER_TOKEN)
    return max(TOKEN_MIN, min(TOKEN_BUDGET, b // 1024 * 1024))


def pad_size(w: int, h: int, grid: int = GRID_PX) -> tuple[int, int]:
    """Tela di lavoro: (w, h) PORTATI IN SU al multiplo di `grid` (si aggiunge bordo, non si stira né si taglia nulla: il
    rapporto d'aspetto della sorgente resta esatto, il bordo si toglie all'uscita)."""
    return (max(grid, -(-int(w) // grid) * grid), max(grid, -(-int(h) // grid) * grid))


def plan_geometry(src_w: int, src_h: int, megapixels: float, factor: int = 1) -> dict:
    """Tela di lavoro, tela d'uscita e ritaglio finale SENZA stirare la sorgente (regola «mai stirare»).

    La sorgente sta nell'angolo in alto a sinistra di una tela a multipli di 32 (bordo aggiunto, mai tolto contenuto);
    l'uscita è a multipli di 32·factor. Fra i bordi possibili (fino a +192 px) e le misure d'uscita vicine ai megapixel
    voluti si sceglie quella in cui la scala di x e di y coincidono (il ritaglio finale non deforma) con meno bordo
    e area più vicina. -> {canvas, out, crop, pad, scale_mismatch}"""
    base = pad_size(src_w, src_h)
    align = 32 * max(1, int(factor))
    area = float(megapixels) * 1024.0 * 1024.0
    best = None
    for kw in range(0, 7):
        for kh in range(0, 7):
            cw, ch = base[0] + 32 * kw, base[1] + 32 * kh
            sc = (area / (cw * ch)) ** 0.5
            for ow in {max(align, int(sc * cw) // align * align), max(align, -(-int(sc * cw) // align) * align)}:
                for oh in {max(align, int(sc * ch) // align * align), max(align, -(-int(sc * ch) // align) * align)}:
                    if ow < cw or oh < ch:
                        continue
                    mism = abs((ow / cw) / (oh / ch) - 1.0)
                    score = mism * 100.0 + abs(ow * oh - area) / area + (cw * ch / (base[0] * base[1]) - 1.0) * 2.0
                    if best is None or score < best[0]:
                        best = (score, (cw, ch), (ow, oh), mism)
    if best is None:                                     # megapixel più piccoli della sorgente: lo dirà il chiamante
        cw, ch = base
        sc = (area / (cw * ch)) ** 0.5
        return {"canvas": base, "out": (max(align, int(sc * cw) // align * align), max(align, int(sc * ch) // align * align)),
                "crop": (src_w, src_h), "pad": (cw - src_w, ch - src_h), "scale_mismatch": 0.0}
    _s, canvas, out, mism = best
    return {"canvas": canvas, "out": out, "crop": crop_size(src_w, src_h, canvas, out),
            "pad": (canvas[0] - src_w, canvas[1] - src_h), "scale_mismatch": mism}


def crop_size(src_w: int, src_h: int, canvas: tuple[int, int], out_canvas: tuple[int, int]) -> tuple[int, int]:
    """Parte dell'uscita che corrisponde alla sorgente vera: la sorgente occupa l'angolo in alto a sinistra della tela
    (src_w × src_h su cw × ch); nell'uscita (ow × oh) è lo stesso rapporto, arrotondato a pari (yuv420)."""
    cw, ch = canvas
    ow, oh = out_canvas
    w = max(2, int(round(src_w * ow / cw)) // 2 * 2)
    h = max(2, int(round(src_h * oh / ch)) // 2 * 2)
    return min(w, ow), min(h, oh)


def grid_size(w: int, h: int, grid: int = GRID_PX) -> tuple[int, int]:
    """(w, h) arrotondati al multiplo di `grid` più vicino, almeno una cella."""
    return (max(grid, int(round(w / grid)) * grid), max(grid, int(round(h / grid)) * grid))


def target_size(src_w: int, src_h: int, megapixels: float, align: int = SIZE_ALIGN) -> tuple[int, int]:
    """(w, h) in pixel: `megapixels` di area al rapporto della sorgente, lati sulla griglia `align`.
    Fra i nove punti di griglia attorno alla misura ideale vince quello con il rapporto più vicino; l'area fa solo da
    spareggio (peso 1/4): un rapporto storto si nota, l'area esatta no."""
    src_w, src_h, align = max(1, int(src_w)), max(1, int(src_h)), max(1, int(align))
    aspect = src_w / float(src_h)
    area = float(megapixels) * 1024.0 * 1024.0
    ideal_h = (area / aspect) ** 0.5
    ideal_w = ideal_h * aspect
    area = ideal_w * ideal_h

    def snap(v):
        return max(align, int(round(float(v) / align)) * align)

    best = None
    for dw in (-1, 0, 1):
        w = snap(ideal_w) + dw * align
        if w < align:
            continue
        for dh in (-1, 0, 1):
            h = snap(ideal_h) + dh * align
            if h < align:
                continue
            score = abs(w / float(h) - aspect) / aspect * 4.0 + abs(w * h - area) / max(1.0, area)
            if best is None or score < best[0]:
                best = (score, w, h)
    return best[1], best[2]


# --- tempo: fotogrammi, token, segmenti --------------------------------------------------------------------------
def frames_in(tokens: int) -> int:
    """Fotogrammi coperti dai primi `tokens` passi latenti del video."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(tokens)))


def tokens_to_reach(frames: int) -> int:
    """Il minor numero di passi latenti i cui fotogrammi raggiungono almeno `frames`."""
    k = got = 0
    while got < frames:
        got += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k


def audio_span(f0: int, f1: int) -> tuple[int, int]:
    """Passi latenti audio [a0, a1) sotto i fotogrammi [f0, f1)."""
    return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)


def _nearest_key_step(frame: int, tokens: int) -> tuple[int, int]:
    best_k, best_f, best_d = 0, 0, frame
    for k in range(0, tokens + 1, 5):
        f = frames_in(k)
        if abs(f - frame) < best_d:
            best_k, best_f, best_d = k, f, abs(f - frame)
    return best_k, best_f


def chunks(tokens: int, length: int = CHUNK_FRAMES, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int, int, int]]:
    """[(k0, f0, k1, f1)]: passi latenti e fotogrammi di ogni segmento. I confini cadono sui passi chiave, un segmento
    non comincia dopo la fine del precedente, l'ultimo finisce esattamente sull'ultimo fotogramma."""
    total = frames_in(tokens)
    hop = length - overlap
    out, prev_end, i = [], 0, 0
    while True:
        start, end = i * hop, min(i * hop + length, total)
        if i == 0:
            k0, f0 = 0, 0
        else:
            k0, f0 = _nearest_key_step(start, tokens)
            if k0 > prev_end:
                k0, f0 = prev_end, frames_in(prev_end)
        if end >= total:
            k1, f1 = tokens, total
        else:
            k1, f1 = _nearest_key_step(end, tokens)
            if k1 <= k0:
                k1 = k0 + 5
                f1 = frames_in(k1)
            if k1 >= tokens:
                k1, f1 = tokens, total
        out.append((k0, f0, k1, f1))
        if k1 >= tokens:
            return out
        prev_end = k1
        i += 1


# --- spazio: la griglia di tile ---------------------------------------------------------------------------------
def _equal_split(total_px: int, count: int, want_overlap_px: int, grid: int = 32):
    """(tile_px, overlap_px) per `count` tile uguali che coprono esattamente `total_px`, sovrapposizione sulla griglia
    e il più vicina possibile a quella voluta. None se non esiste."""
    if count <= 1:
        return -(-int(total_px) // grid) * grid, 0
    lo = -(-((int(total_px) + (count - 1) * int(want_overlap_px)) // count) // grid) * grid
    hi = int(total_px) - grid * (count - 1)
    for tile in range(lo, hi + 1, grid):
        spare = count * tile - int(total_px)
        if spare % (count - 1) == 0:
            ov = spare // (count - 1)
            if ov % grid == 0 and 0 <= ov <= tile - grid:
                return tile, ov
    return None


def seq_len(tile_w_px: int, tile_h_px: int, tokens: int) -> int:
    """Sequenza di una forward: celle 2×2 × passi latenti video."""
    return (int(tile_w_px) // 32) * (int(tile_h_px) // 32) * int(tokens)


def guided_seq_len(tile_w_px: int, tile_h_px: int, tokens: int, factor: int) -> int:
    """`seq_len` più le righe della guida: una ogni `factor`-esima patch, a ogni passo."""
    f = int(factor)
    return (seq_len(tile_w_px, tile_h_px, tokens)
            + (int(tile_w_px) // (32 * f)) * (int(tile_h_px) // (32 * f)) * int(tokens))


def solve_grid(frame_w_px, frame_h_px, tokens, overlap_px=TILE_OVERLAP_PX, budget=TOKEN_BUDGET,
               min_tile_px=MIN_TILE_PX, max_side=MAX_GRID_SIDE, factor=1):
    """Il minor numero di tile uguali la cui forward sta nel `budget`.
    -> (righe, colonne, tile_w_px, tile_h_px, overlap_w_px, overlap_h_px) oppure None. A parità di numero vince chi
    divide di più l'asse lungo (strisce, non scacchiera), poi il più quadrato. Con guida (`factor` > 1) tile e
    sovrapposizioni stanno sulla griglia 32·factor e le righe della guida contano nel budget."""
    grid = 32 * int(factor) if int(factor) > 1 else 32
    for n in range(1, max_side * max_side + 1):
        best = None
        for rows in range(1, max_side + 1):
            if n % rows:
                continue
            cols = n // rows
            if cols > max_side:
                continue
            across = _equal_split(frame_w_px, cols, overlap_px, grid)
            down = _equal_split(frame_h_px, rows, overlap_px, grid)
            if across is None or down is None:
                continue
            tw, ow = across
            th, oh = down
            if tw < min_tile_px or th < min_tile_px:
                continue
            cost = guided_seq_len(tw, th, tokens, factor) if int(factor) > 1 else seq_len(tw, th, tokens)
            if cost > budget:
                continue
            long_split = rows if frame_h_px >= frame_w_px else cols
            rank = (-long_split, abs(rows - cols))
            if best is None or rank < best[0]:
                best = (rank, rows, cols, tw, th, ow, oh)
        if best is not None:
            return best[1:]
    return None


def _axis(size: int, tile: int, overlap: int, floor: int):
    """Inizi, estensioni e sovrapposizioni di cucitura lungo un asse, in celle latenti. Un tile di bordo più piccolo di
    `floor` viene tirato indietro fino a `floor`."""
    if size <= tile:
        return [0], [size], [0]
    stride = tile - overlap
    n = math.ceil((size - overlap) / stride)
    if (n - 1) * stride + tile < size:
        n += 1
    starts = [i * stride for i in range(n)]
    extents = [min(tile, size - s) for s in starts]
    if floor > 0 and n >= 2 and size - starts[-1] < floor:
        pulled = size - floor
        if starts[-2] < pulled < starts[-2] + extents[-2]:
            starts[-1], extents[-1] = pulled, size - pulled
    seams = [0] + [max(0, starts[i - 1] + extents[i - 1] - starts[i]) for i in range(1, n)]
    return starts, extents, seams


def tile_grid(h: int, w: int, tokens: int, factor: int = 1, budget: int = TOKEN_BUDGET):
    """I tile di un segmento (h, w) celle × `tokens` passi.
    -> lista di (r0, c0, righe, colonne, cucitura_alto, cucitura_sx, alto, sx, basso, dx) in celle latenti + la griglia
    risolta per il log. `budget` = sequenza massima per forward (la ricetta, salvo chi lo abbassa per una scheda piccola)."""
    got = solve_grid(w * SPATIAL, h * SPATIAL, tokens, budget=int(budget), factor=factor)
    if got is None:
        raise RuntimeError(
            f"nessuna griglia di tile fino a {MAX_GRID_SIDE}x{MAX_GRID_SIDE} tiene un segmento di {w * SPATIAL}x"
            f"{h * SPATIAL}px e {tokens} passi latenti sotto {int(budget)} token per forward: abbassa i megapixel "
            f"o alza il budget dei tile.")
    _r, _c, tw, th, ow, oh = got
    tw, th, ow, oh = tw // SPATIAL, th // SPATIAL, ow // SPATIAL, oh // SPATIAL
    floor = MIN_TILE_PX // SPATIAL
    r0s, rh, rseam = _axis(h, th, oh, floor)
    c0s, cw, cseam = _axis(w, tw, ow, floor)
    odd = sorted({x * SPATIAL for x in rh + cw if x % 2})
    if odd:
        raise RuntimeError(f"un lato di tile di {odd} px non è multiplo di 32: H3 impacchetta celle 2x2.")
    if int(factor) > 1:
        # la guida sta su ogni factor-esima PATCH del tile e la sua finestra è ritagliata dalla guida del film in
        # (r0, c0)/factor: ogni bordo e origine deve cadere su una patch intera della guida
        step = 2 * int(factor)
        off = sorted({x * SPATIAL for x in r0s + rh + c0s + cw if x % step})
        if off:
            raise RuntimeError(f"bordi di tile {off} px fuori dalla griglia di guida da {step * SPATIAL} px")
    tiles = []
    nr, nc = len(r0s), len(c0s)
    for i in range(nr):
        for j in range(nc):
            tiles.append((r0s[i], c0s[j], rh[i], cw[j], rseam[i] or oh, cseam[j] or ow,
                          i > 0, j > 0, i < nr - 1, j < nc - 1))
    return tiles, got


def feather(rows, cols, seam_top, seam_left, top, left, bottom, right, device="cpu"):
    """Peso di un tile nella fusione: 1 dentro, coseno rialzato fino a 0 sui lati con un vicino. Coseno rialzato e non
    lineare, perché il peso arrivi a 1 senza spigolo che stamperebbe una linea."""
    import torch

    def ramp(n):
        i = torch.arange(n, device=device, dtype=torch.float32)
        return 0.5 * (1.0 - torch.cos(math.pi * i / n))

    m = torch.ones((rows, cols), device=device, dtype=torch.float32)
    if left and seam_left > 0:
        m[:, :seam_left] *= ramp(min(seam_left, cols))[None, :]
    if right and seam_left > 0:
        m[:, -seam_left:] *= torch.flip(ramp(min(seam_left, cols)), (0,))[None, :]
    if top and seam_top > 0:
        m[:seam_top, :] *= ramp(min(seam_top, rows))[:, None]
    if bottom and seam_top > 0:
        m[-seam_top:, :] *= torch.flip(ramp(min(seam_top, rows)), (0,))[:, None]
    return m


# --- il ri-disegno: da quale sigma si parte ----------------------------------------------------------------------
def redraw_sigma(denoise: float, shift: float = VIDEO_SHIFT) -> float | None:
    """Sigma di partenza dell'unico passo di ri-disegno (poi si va a 0). None se denoise <= 0 (solo ingrandimento).

    Replica la selezione dello scheduler «simple» di ComfyUI su un modello a flusso con shift: `total = int(1/denoise)`
    punti, il penultimo è l'indice -(1 + int((total-1) * 1000 / total)) della tabella dei 1000 sigma ordinati; la
    tabella è sigma(t) = shift·t / (1 + (shift-1)·t) per t = k/1000 (k = 1..1000). Con shift 12: denoise 0,2 → 0,75
    (come dichiara la ricetta), 0,1 → 0,5714."""
    if float(denoise) <= 0.0:
        return None
    total = max(1, int(STEPS / float(denoise)))
    idx = int((total - 1) * 1000 / total)             # ss = 1000/total; i = total-1
    t = (1000 - idx) / 1000.0
    return shift * t / (1.0 + (shift - 1.0) * t)


# --- cucitura fra segmenti ---------------------------------------------------------------------------------------
def crossfade(a, b, dim: int):
    """a→b con peso lineare lungo `dim`."""
    import torch
    n = a.shape[dim]
    wgt = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    return a + (b - a) * wgt.view(shape)


def join_chunk(done_v, done_a, v, a, k0: int, f0: int):
    """Aggiunge un segmento finito, dissolvendo dove si sovrappone a ciò che è già fatto (video e audio)."""
    import torch
    if done_v is None:
        return v, a
    a0 = round(f0 * FRAME_RESCALE)
    out_v = torch.zeros((1, done_v.shape[1], max(done_v.shape[2], k0 + v.shape[2]), done_v.shape[3], done_v.shape[4]),
                        device=done_v.device, dtype=done_v.dtype)
    out_a = torch.zeros((1, done_a.shape[1], done_a.shape[2], max(done_a.shape[-1], a0 + a.shape[-1])),
                        device=done_a.device, dtype=done_a.dtype)
    out_v[:, :, :done_v.shape[2]] = done_v
    out_a[..., :done_a.shape[-1]] = done_a
    ov = min(done_v.shape[2] - k0, v.shape[2])
    if ov > 0:
        out_v[:, :, k0:k0 + ov] = crossfade(out_v[:, :, k0:k0 + ov].clone(), v[:, :, :ov], 2)
        v = v[:, :, ov:]
    at = k0 + max(ov, 0)
    if v.shape[2] > 0:
        out_v[:, :, at:at + v.shape[2]] = v
    ova = min(done_a.shape[-1] - a0, a.shape[-1])
    if ova > 0:
        out_a[..., a0:a0 + ova] = crossfade(out_a[..., a0:a0 + ova].clone(), a[..., :ova], 3)
        a = a[..., ova:]
    at = a0 + max(ova, 0)
    if a.shape[-1] > 0:
        out_a[..., at:at + a.shape[-1]] = a
    return out_v, out_a


# --- stima per il preflight e per i messaggi della GUI ----------------------------------------------------------
def describe(src_w: int, src_h: int, n_frames: int, megapixels: float, factor: int = 2,
             budget: int = TOKEN_BUDGET) -> dict:
    """Che cosa farà il lavoro, senza modelli: tela di lavoro, tela d'uscita, ritaglio, segmenti, tile del primo segmento."""
    g = plan_geometry(src_w, src_h, megapixels, factor)
    (cw, ch), (ow, oh) = g["canvas"], g["out"]
    length = valid_length(n_frames)
    tokens = 5 * ((length - 5) // 17) + 2
    plan = chunks(tokens)
    k0, _f0, k1, _f1 = plan[0]
    tiles, grid = tile_grid(oh // SPATIAL, ow // SPATIAL, k1 - k0, factor, budget)
    return {"canvas": (cw, ch), "out": (ow, oh), "crop": g["crop"], "pad": g["pad"], "scale_mismatch": g["scale_mismatch"],
            "frames_padded": length, "latent_steps": tokens, "chunks": len(plan), "tiles_per_chunk": len(tiles),
            "grid": grid}
