"""MiniMax H3 EDIT con maschera — matematica pura (torch, senza ComfyUI, senza GPU obbligatoria).

Riscrittura NATIVA, studiata dai nodi MIT `MiniMaxH3FantasticVideoEditLatent` / `EditComposite`
(github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder, MIT, (c) 2026 Adudeguyman) e dai nodi `H3PeopleMask`,
`H3GreyMannequins`, `clip_plan` (github.com/D3voz/comfyui-h3-oneclick-swap, MIT, (c) 2026 D3voz). Licenze e ringraziamenti
in NOTICE-openjutsu.md. NIENTE codice dal pacchetto GPL «Context Loop»: la catena di clip di questo modulo e' la pianificazione
a (lunghezza, contesto) del nodo MIT D3voz, la maschera causale/sostegno delle righe e' derivata dal layout di diffusers.

Convenzioni H3 (verificate sul modulo diffusers `minimax_h3`):
- clip valido = 17*k + 5 frame a 24 fps -> 5*k + 2 frame di latente; i frame-pixel per frame di latente ciclano (1, 4, 4, 4, 4);
- una «cella» del latente e' 16x16 px; la maschera (1 = rigenera, 0 = tieni la sorgente) sta sulla griglia th/16 x tw/16;
- audio: il latente tenuto ha maschera 0, quello generato 1.
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

FPS = 24
CELL = 16                                   # lato in pixel di una cella di latente
FRAMES_PER_LATENT = (1, 4, 4, 4, 4)         # frame-pixel per frame di latente (ciclo di 5)
CUT_SNAP = 4                                # un clip puo finire fino a 4 frame prima di un taglio netto
CHUNK = 16                                  # frame per passata nei ridimensionamenti
SHARP_BLUR, SOFT_BLUR = 1.0, 8.0            # sigma minima / massima del manichino grigio


# ── griglia dei frame ───────────────────────────────────────────────────────────────────────────
def grid_floor(frames: int) -> int:
    """Il clip H3 (17k+5) piu lungo contenuto in `frames` frame."""
    return (frames - 5) // 17 * 17 + 5


def usable_frames(n: int) -> int:
    if n < 5:
        raise ValueError(f"Il clip sorgente ha {n} frame; H3 ne vuole almeno 5.")
    return grid_floor(n)


def latent_frame_counts(n_pixel_frames: int) -> list[int]:
    """Frame-pixel coperti da ciascun frame di latente per un clip di `n_pixel_frames` (17k+5)."""
    if n_pixel_frames != grid_floor(n_pixel_frames):
        raise ValueError(f"{n_pixel_frames} frame non e' della forma 17k+5.")
    t = (n_pixel_frames - 5) // 17 * 5 + 2
    counts = [FRAMES_PER_LATENT[k % 5] for k in range(t)]
    assert sum(counts) == n_pixel_frames
    return counts


def erode_cross(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Restringe la maschera [N, H, W] (1 = persona) di `iterations` pixel con il nucleo a croce 3x3 (L1: bordi diagonali piu stretti di quelli dritti).
    E la zona rigenerata di Openjutsu nel cambia-sfondo: KJNodes GrowMaskWithBlur con flip_input=true inverte la maschera PRIMA di allargarla
    (tapered_corners = nucleo a croce), quindi lo sfondo cresce DENTRO il bordo della persona."""
    out = mask.float()
    for start in range(0, out.shape[0], CHUNK):
        background = 1.0 - out[start:start + CHUNK, None]
        for _ in range(max(0, int(iterations))):
            background = torch.maximum(F.max_pool2d(background, (3, 1), stride=1, padding=(1, 0)),
                                       F.max_pool2d(background, (1, 3), stride=1, padding=(0, 1)))
        out[start:start + CHUNK] = (1.0 - background)[:, 0]
    return out


def latent_boundary_floor(frames: int) -> int:
    """Il piu grande confine di frame di latente (1, 5, 9, ..., 17, 18, 22, ... cumulati di 1,4,4,4,4) <= `frames`: un contesto che finisce su un confine
    non lascia mezzo frame di latente fra tenuto e rigenerato. 0 se `frames` < 1."""
    acc, best = 0, 0
    for c in itertools.cycle(FRAMES_PER_LATENT):
        acc += c
        if acc > frames:
            return best
        best = acc


def budget_size(w: int, h: int, width: int, height: int) -> tuple[int, int, float]:
    """(tw, th, scala): forma del clip entro width x height pixel, mai ingrandita, multipli di 32."""
    scale = min(1.0, math.sqrt(width * height / float(w * h))) if width and height else 1.0
    return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32), scale


def short_edge_size(w: int, h: int, short_edge: int) -> tuple[int, int]:
    """Tela con lato corto `short_edge` (multiplo di 32), stesso rapporto del sorgente."""
    scale = short_edge / float(min(w, h))
    return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)


# ── maschere ─────────────────────────────────────────────────────────────────────────────────────
def resize_mask(mask: torch.Tensor, tw: int, th: int) -> torch.Tensor:
    """[N, h, w] -> [N, th, tw]: 1 dove un qualunque pixel sorgente era mascherato (un bordo sottile non sparisce)."""
    if tuple(mask.shape[1:]) == (th, tw):
        return (mask > 0.5).float()
    out = torch.empty((mask.shape[0], th, tw), dtype=torch.float32)
    for i in range(0, mask.shape[0], CHUNK):
        m = (mask[i:i + CHUNK] > 0.5).float()[:, None]
        m = F.interpolate(m, size=(th, tw), mode="area" if th * tw < m.shape[-2] * m.shape[-1] else "nearest")
        out[i:i + CHUNK] = (m[:, 0] > 0).float()
    return out


def grow_mask(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Allarga [N, H, W] di `px` pixel (dilatazione quadrata, due passate 1-D)."""
    if px <= 0:
        return mask
    k = 2 * px + 1
    out = torch.empty_like(mask)
    for i in range(0, mask.shape[0], CHUNK):
        m = mask[i:i + CHUNK].float()[:, None]
        m = F.max_pool2d(m, (1, k), stride=1, padding=(0, px))
        out[i:i + CHUNK] = F.max_pool2d(m, (k, 1), stride=1, padding=(px, 0))[:, 0]
    return out


def expand_box_region(mask: torch.Tensor, frac: float, window: int = 12) -> torch.Tensor:
    """Riquadro del soggetto allargato di `frac` per lato (della larghezza/altezza del riquadro), stabile nel tempo: il riquadro di ogni frame e l'UNIONE dei
    riquadri dei frame vicini (+/- `window`). Serve a «Sostituisci personaggio»: il nuovo personaggio puo avere una sagoma piu larga dell'originale (braccia aperte
    nel ritratto contro il primo piano di un volto) e fuori dalla zona rigenerata restava il video originale: chiazze bianche ai bordi (giudizio 03/11).
    Ritorna [N,H,W] 0/1; vuoto per frame senza soggetto."""
    n, h, w = mask.shape
    m = mask > 0.5
    ys = m.any(2)                                                    # [N, H]
    xs = m.any(1)                                                    # [N, W]
    out = torch.zeros(n, h, w)
    boxes = []
    for i in range(n):
        if ys[i].any():
            yy, xx = torch.where(ys[i])[0], torch.where(xs[i])[0]
            boxes.append([int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1])
        else:
            boxes.append(None)
    for i in range(n):
        near = [b for b in boxes[max(0, i - window): i + window + 1] if b]
        if not near:
            continue
        x0, y0 = min(b[0] for b in near), min(b[1] for b in near)
        x1, y1 = max(b[2] for b in near), max(b[3] for b in near)
        dx, dy = int(round((x1 - x0) * frac)), int(round((y1 - y0) * frac))
        out[i, max(0, y0 - dy): min(h, y1 + dy), max(0, x0 - dx): min(w, x1 + dx)] = 1.0
    return out


def regen_region(pers: torch.Tensor, invert: bool, person_grow: int, grow: int, expand: float = 0.0) -> torch.Tensor:
    """Maschera-pixel [N,H,W] da dare a `cell_mask` (1 = «oggetto»; con invert=True si rigenera il complemento).
    - sostituisci personaggio (invert=False): si rigenera la persona CRESCIUTA di person_grow + grow px, cosi copre tutta la persona originale con i bordi.
    - cambia sfondo (invert=True): si rigenera tutto tranne la persona e la zona rigenerata deve ENTRARE un poco nel bordo della persona, mai lasciarle
      intorno una fascia di sfondo vecchio: niente crescita della persona (a 03/11 era 40+16 px a 640 e il bordo chiaro della parete originale restava
      tutt'intorno: «alone bianco molto spesso»; misurato sui frame: 15-24 % di pixel bianchi, 50-86 % invariati nei 45 px fuori dalla sagoma).
      Dentro il bordo ci entrano le celle di 16 px che toccano lo sfondo (adaptive_max_pool sul complemento)."""
    if invert:
        return pers.float()
    reg = grow_mask(grow_mask(pers.float(), max(0, person_grow)), max(0, grow))
    if expand > 0:
        reg = torch.maximum(reg, expand_box_region(reg, float(expand)))
    return reg


def latent_mask(mask: torch.Tensor, frame_counts: list[int], lat_h: int, lat_w: int) -> torch.Tensor:
    """Maschera-pixel [F, H, W] -> [1, 1, T, lat_h, lat_w]: un frame/cella di latente e mascherato se lo e' un qualunque pixel che copre."""
    cells = F.adaptive_max_pool2d(mask[:, None], (lat_h, lat_w))[:, 0]
    rows, at = [], 0
    for n in frame_counts:
        rows.append(cells[at:at + n].amax(dim=0))
        at += n
    return torch.stack(rows)[None, None]


def feather_latent(mask: torch.Tensor, cells: int) -> torch.Tensor:
    """[1,1,T,h,w] -> stessa forma con una rampa larga `cells` fuori dalla maschera (1 - k/(cells+1)): il bordo si fonde invece di tagliare."""
    if cells <= 0:
        return mask
    x = near = mask[0].movedim(0, 1)                            # [T, 1, h, w]
    out = x
    for k in range(1, cells + 1):
        near = F.max_pool2d(near, 3, stride=1, padding=1)
        out = torch.maximum(out, near * (1 - k / (cells + 1)))
    return out.movedim(1, 0)[None]


def cell_mask(obj: torch.Tensor, tw: int, th: int, invert: bool = False, feather_px: int = 0, width: int | None = None,
              hold_cover: float | None = None):
    """Cio che l'edit rigenera, sulla griglia delle celle: `obj` [n, mh, mw] (maschera cresciuta, 1 = oggetto/persona) -> maschera di
    latente [1,1,T,th/16,tw/16] (+ conteggi). Invertita DOPO il ridimensionamento (cambia-sfondo: si rigenera tutto tranne la persona).
    `hold_cover` (solo invert, 0 < x < 1): una cella si TIENE se la persona ne copre almeno quella quota (es. 0.5), invece del 100 % di sempre
    (basta UN pixel di sfondo per rigenerarla): volto e spalle restano ancorati da celle tenute. None/0/>=1 = comportamento di sempre."""
    n = obj.shape[0]
    counts = latent_frame_counts(n)
    rows = []
    for i in range(0, n, CHUNK):
        m = resize_mask(obj[i:i + CHUNK], tw, th)
        if invert and hold_cover and 0.0 < hold_cover < 1.0:
            cover = F.adaptive_avg_pool2d(m[:, None], (th // CELL, tw // CELL))[:, 0]        # quota di persona in ogni cella
            rows.append((cover < float(hold_cover)).float())
        else:
            rows.append(F.adaptive_max_pool2d((1.0 - m if invert else m)[:, None], (th // CELL, tw // CELL))[:, 0])
    cells = latent_mask(torch.cat(rows), counts, th // CELL, tw // CELL)
    feather_cells = math.ceil(feather_px * tw / float(width or tw) / CELL) if feather_px else 0
    return feather_latent(cells, feather_cells), counts


# ── tagli e catena di clip ───────────────────────────────────────────────────────────────────────
def find_shots(images: torch.Tensor, cut_threshold: float = 40.0) -> list[tuple[int, int]]:
    """(inizio, fine) dei piani fra un taglio netto e l'altro. `images` [N, h, w, 3] float 0..1 (basta una versione piccola)."""
    grey = torch.cat([F.interpolate(images[i:i + 64, ..., :3].mean(-1, keepdim=True).movedim(-1, 1).float(), size=(90, 160), mode="area")[:, 0]
                      for i in range(0, images.shape[0], 64)]) * 255
    diffs = [0.0] + (grey[1:] - grey[:-1]).abs().mean(dim=(1, 2)).tolist()
    starts = [0]
    for i in range(1, len(diffs)):
        around = sorted(diffs[max(1, i - 6):i] + diffs[i + 1:i + 7])
        median = around[len(around) // 2] if around else 0.0
        if diffs[i] > cut_threshold and diffs[i] > 2.5 * median and i - starts[-1] >= 6:
            starts.append(i)
    return list(zip(starts, starts[1:] + [images.shape[0]]))


def clip_plan(total: int, clip_frames: int, overlap: int, cuts: list[int], fits=None) -> list[tuple[int, int]]:
    """[(lunghezza, contesto)] per clip. Un clip che puo finire su un taglio netto ci finisce e il successivo parte senza sovrapposizione;
    se finisce fino a CUT_SNAP frame prima, il successivo riparte dagli ultimi 5. `fits(start, length)` (opzionale) limita la lunghezza
    ai frame che stanno in memoria. Il contesto di un clip = i primi frame GIA generati del precedente (tenuti come contenuto dato)."""
    longest = grid_floor(clip_frames)
    clips, end, context = [], 0, 0
    while True:
        start = end - context
        length = grid_floor(min(longest, total - start))
        if fits:
            length = next((n for n in range(length, context + 16, -17) if fits(end, n - context)), length)
        if clips and length - context < 17:
            return clips
        following = overlap
        if total - start > longest:
            near = [(n, c - start - n) for n in range(length, longest // 2, -17) for c in cuts
                    if 0 <= c - start - n <= CUT_SNAP and n - context >= 17 and total - start - n >= 22]
            if near:
                length, off = max(near, key=lambda x: (x[0], -x[1]))
                following = 5 if off else 0
        clips.append((length, context))
        end, context = start + length, min(following, grid_floor(length - context))


MIN_CLIP_FRAMES = 124               # il pacchetto H3 genera da 5,0 a 15,0 s: sotto i 120 frame rifiuta (17*7+5 = 124 e il primo valido)


def enforce_min_length(clips: list[tuple[int, int]], total: int, min_len: int = MIN_CLIP_FRAMES) -> list[tuple[int, int]]:
    """Ogni clip deve durare almeno `min_len` frame (H3 rifiuta i clip sotto i 5 s: giro 02/11, ultimo clip da 56 frame -> ValueError). Un clip
    corto si ALLUNGA all'indietro: stessi frame NUOVI, piu contesto (frame gia generati, tenuti). La lunghezza resta 17k+5 e il contesto cambia di
    un multiplo di 17, quindi cade ancora su un confine di frame di latente. Errore chiaro se non c'e abbastanza storia (video < 5,2 s o piano
    dopo un taglio netto con un piano troppo corto)."""
    out, done, prev_len = [], 0, 0
    for length, ctx in clips:
        new = length - ctx
        if length < min_len:
            L = grid_floor(min_len) if grid_floor(min_len) >= min_len else grid_floor(min_len) + 17
            ctx2 = L - new
            if not out or ctx2 > done or ctx2 > prev_len:
                raise ValueError(f"H3 genera clip di almeno {min_len / FPS:.1f} s ({min_len} frame): qui servirebbero {ctx2} frame di contesto ma ne "
                                 f"esistono {min(done, prev_len)}. Il video (o il suo piano dopo un taglio netto) e troppo corto: {total} frame.")
            length, ctx = L, ctx2
        out.append((length, ctx))
        done += new
        prev_len = length
    return out


# ── persona: maschera, manichino grigio, sfondo grigio sfocato ───────────────────────────────────
def person_mask(m: torch.Tensor, headroom: float, hold_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Maschera di UNA persona [frames, h, w] con posa tenuta (+/- hold_frames) e spazio sopra la testa (headroom, quota dell'altezza);
    ritorna (persona, altezza per pixel come quota del frame)."""
    n, mh = m.shape[:2]
    m = m.float()
    if hold_frames:
        m = F.max_pool1d(m.flatten(1).T[None], 2 * hold_frames + 1, stride=1, padding=hold_frames)[0].T.reshape(m.shape)
    inside = m > 0.5
    rows = torch.arange(mh, device=m.device)
    occupied = inside.any(2)
    height = (torch.where(occupied, rows, -1).amax(1) - torch.where(occupied, rows, mh).amin(1) + 1).clamp(min=0)
    top = torch.where(inside, rows.view(1, mh, 1), mh).amin(1, keepdim=True)
    room = (height * headroom).round().view(n, 1, 1)
    # lo spazio sta solo sopra le colonne piu alte della persona (testa, capelli, mani alzate), non sopra le braccia tese
    near_top = (top < mh) & (top <= top.amin(2, keepdim=True) + room)
    person = inside | (near_top & (rows.view(1, mh, 1) >= top - room) & (rows.view(1, mh, 1) < top))
    return person, person * (height / mh).view(n, 1, 1)


def preroll_pad(frames_u8: torch.Tensor, masks: torch.Tensor, wav: torch.Tensor | None, p: int, sample_rate: int, fps: int = FPS):
    """PREROLL (prova 03/11): si antepongono `p` frame al video (il primo frame ripetuto, la sua maschera, audio SILENZIOSO) cosi il primo clip non parte «a freddo».
    Nei giri del 03/11 i primi secondi del clip 1 (senza contesto) avevano il doppione della persona (cambia-sfondo) e il parlato senza lip sync (sostituisci), i clip
    2-3 (con 39 frame di contesto) no. `p` deve far restare il totale nella griglia 17k+5 (multiplo di 17). Ritorna (frames, maschere, audio)."""
    if p <= 0:
        return frames_u8, masks, wav
    if p % 17:
        raise ValueError(f"preroll di {p} frame: deve essere un multiplo di 17 (griglia 17k+5 del clip)")
    pad = frames_u8[:1].expand(p, -1, -1, -1).clone()
    mpad = masks[:1].expand(p, -1, -1).clone()
    out_w = wav
    if wav is not None:
        n = int(round(p / float(fps) * sample_rate))
        out_w = torch.cat([torch.zeros(wav.shape[0], n, dtype=wav.dtype), wav], 1)
    return torch.cat([pad, frames_u8], 0), torch.cat([mpad, masks], 0), out_w


def person_matte(comp: torch.Tensor, orig: torch.Tensor, person: torch.Tensor, soft_px: float = 1.5) -> torch.Tensor:
    """MATTE A PIXEL (prova 03/11, cambia-sfondo): dentro la sagoma della persona i pixel sono quelli ORIGINALI (non quelli che il modello ridisegna nelle celle di
    bordo: la sua persona e spostata a sinistra di ~60 px nel clip 1 e lascia un doppione), con un bordo morbido di ~soft_px. comp/orig [n,h,w,3] float 0..1;
    person [n,h,w] (1 = persona, alla stessa risoluzione)."""
    n = comp.shape[0]
    out = torch.empty_like(comp, dtype=torch.float32)
    for i in range(0, n, CHUNK):
        a = person[i:i + CHUNK].float()[:, None]
        if soft_px > 0:
            a = gaussian_blur(a, float(soft_px))
        a = a.movedim(1, -1)
        out[i:i + CHUNK] = a * orig[i:i + CHUNK].float() + (1 - a) * comp[i:i + CHUNK].float()
    return out


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """[n, c, h, w] con gaussiana separabile esatta (bordi estesi, non scuriti)."""
    r = max(1, math.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = (k / k.sum()).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(x.shape[1], 1, 1, -1), groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(x.shape[1], 1, -1, 1), groups=x.shape[1])


def erase_area(frames: torch.Tensor, area: torch.Tensor, grow_px: int = 8, sigma_frac: float = 0.03) -> torch.Tensor:
    """Cancella dal video di riferimento la zona `area` (maschera 1 = persona), dilatata di `grow_px`: viene riempita dallo sfondo circostante (nel riferimento del
    cambia-sfondo gia sfocato) per diffusione dal bordo (convoluzione normalizzata a scale crescenti, a 1/4 di risoluzione), cosi NESSUNA figura resta nel
    riferimento e il modello non ha nulla da «ridisegnare» col suo orologio. frames [n,h,w,3] float 0..1; area [n,h,w]."""
    n, h, w = frames.shape[:3]
    out = torch.empty_like(frames, dtype=torch.float32)
    hole_all = grow_mask(area.float(), int(grow_px))
    ds = 4
    sig0 = max(1.0, sigma_frac * h / ds)
    for i in range(0, n, CHUNK):
        f = frames[i:i + CHUNK].float().movedim(-1, 1)
        hole = hole_all[i:i + CHUNK, None].clamp(0, 1)
        fs = F.interpolate(f, size=(max(1, h // ds), max(1, w // ds)), mode="area")
        hs = F.interpolate(hole, size=fs.shape[-2:], mode="area")
        known = (hs < 0.5).float()                                  # pixel noti (fuori dalla zona cancellata)
        val, est = fs * known, None
        for mult in (16, 8, 4, 2, 1):                               # dal largo allo stretto: la scala piu fine con abbastanza appoggio vince
            nv, nw = gaussian_blur(val, sig0 * mult), gaussian_blur(known, sig0 * mult)
            cur = nv / nw.clamp_min(1e-4)
            est = cur if est is None else torch.where(nw > 0.08, cur, est)
        fill = torch.lerp(fs, est, 1.0 - known)                     # a bassa risoluzione: noto = originale, ignoto = stima
        fill = F.interpolate(fill, size=(h, w), mode="bilinear", align_corners=False)
        soft = gaussian_blur(hole, 1.5).clamp(0, 1)                 # bordo morbido
        out[i:i + CHUNK] = torch.lerp(f, fill, soft).movedim(1, -1)
    return out


def grey_mannequins(images: torch.Tensor, mask: torch.Tensor, size: torch.Tensor, blur: float = 0.02,
                    neutral_background: bool = False) -> torch.Tensor:
    """Le persone mascherate diventano figure grigie e lisce che tengono posa e ombreggiatura ma non volto, capelli, vestiti: un video di
    riferimento del movimento non puo riportare le persone originali nell'edit. images [n,h,w,3] float 0..1; mask/size [n,h,w]."""
    n, h, w = images.shape[:3]
    out = torch.empty_like(images, dtype=torch.float32)
    for i in range(0, n, CHUNK):
        f = images[i:i + CHUNK].float().movedim(-1, 1)
        m = F.interpolate(mask[i:i + CHUNK, None].float(), size=(h, w), mode="bilinear")
        height = F.interpolate(size[i:i + CHUNK, None].float(), size=(h, w), mode="nearest") * h
        soft = ((height * blur - SHARP_BLUR) / (SOFT_BLUR - SHARP_BLUR)).clamp(0, 1)
        luma0 = (f * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
        luma = torch.lerp(gaussian_blur(luma0, SHARP_BLUR), gaussian_blur(luma0, SOFT_BLUR), soft)
        grey = (0.45 + 0.45 * luma).expand(-1, 3, -1, -1)
        base = torch.full_like(f, 0.5) if neutral_background else f
        out[i:i + CHUNK] = torch.lerp(base, grey, m).movedim(1, -1)
    return out


def blur_area(frames: torch.Tensor, area: torch.Tensor, radius: float, grey: float = 0.0) -> torch.Tensor:
    """Nel video di riferimento l'AREA (maschera 1 = da nascondere, es. lo sfondo nel cambio-sfondo) diventa una sfocatura liscia che porta
    SOLO il movimento di camera e la disposizione grossolana: niente dettagli da copiare. `grey` in [0,1] la sposta verso il grigio medio.
    frames [n,h,w,3] float 0..1; area [n,h,w]."""
    n, h, w = frames.shape[:3]
    out = torch.empty_like(frames, dtype=torch.float32)
    for i in range(0, n, CHUNK):
        f = frames[i:i + CHUNK].float().movedim(-1, 1)
        a = F.interpolate(area[i:i + CHUNK, None].float(), size=(h, w), mode="bilinear").clamp(0, 1)
        hidden = gaussian_blur(f, max(1.0, radius))
        if grey > 0:
            hidden = torch.lerp(hidden, torch.full_like(hidden, 0.5), grey)
        out[i:i + CHUNK] = torch.lerp(f, hidden, a).movedim(1, -1)
    return out
