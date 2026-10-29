"""Pre-riempimento dell'estensione per l'OUTPAINT dei modelli che NON sono «Fill» (task 16a).

Problema (live 18/10, casi O2/O3/O4): il canvas esporta il frame con l'estensione riempita da una tinta SCURA
uniforme (`frame_item`, QColor(20,20,28)). FLUX.1 Fill non la vede (la zona mascherata viene azzerata dalla
pipeline), ma gli altri motori sì:
  · FLUX.2 Klein (Flux2KleinInpaintPipeline): il frame intero è anche l'immagine di RIFERIMENTO e la schedule parte
    dal frame rumorato (strength 0,8 di default) → il buio a bassa frequenza sopravvive e l'estensione resta una
    vignetta scura;
  · Qwen-Image 2.1: il frame (annotato in rosso) è il riferimento 1 → il modello vede «una foto dentro una cornice
    nera» e la disegna (riga scura a rettangolo attorno all'originale);
  · FLUX base (FluxInpaintPipeline): l'estensione è un'altra scena con bordo netto.
Qui si costruisce un riempimento PLAUSIBILE dell'estensione prima di darla al modello: il contenuto della foto
specchiato oltre il bordo (continuo al confine) e sfocato sempre di più allontanandosi, che sfuma in un campo di
colore liscio (push-pull sui pixel noti), più grana come la foto. Niente di tutto questo finisce nell'output: i
pixel della foto restano identici e il risultato dell'estensione lo genera il modello (la maschera non cambia).

Dipendenze: numpy + PIL. Nessuna GPU. Interruttore: `NATIVE_OUTPAINT_PREFILL=0` (default ON).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFilter

# ── parametri (una riga ciascuno) ──────────────────────────────────────────────────────────────────────────────
SIGMAS_PCT = (0.012, 0.035, 0.08)   # sfocatura dello specchio (frazione del lato minore) a distanza crescente dal bordo
REACH_PCT = 0.40                    # a questa distanza (frazione del lato minore della foto) si è nel solo campo liscio
GRAIN_FACTOR = 0.6                  # grana = alta frequenza della foto × questo
GRAIN_RANGE = (1.5, 8.0)            # livelli 0..255 (min, max) della deviazione standard della grana
MIN_KNOWN_PX = 64                   # sotto questi pixel di foto non si riempie (nessun contenuto da continuare)
OUTPAINT_STRENGTH = 0.9             # forza (denoise) in OUTPAINT per FLUX base e Qwen: a 1,0 la schedule parte da rumore puro e il
                                    # pre-riempimento non arriva al modello (live 18/10: O2/O4 ancora «riquadro + altra immagine»).
                                    # Override: env NATIVE_OUTPAINT_STRENGTH (0..1); solo outpaint, solo se il prefill è attivo.


def enabled() -> bool:
    """`NATIVE_OUTPAINT_PREFILL=0` → interruttore di emergenza (il frame resta come esportato dal canvas)."""
    return os.environ.get("NATIVE_OUTPAINT_PREFILL", "1") != "0"


def outpaint_strength(requested: float | None) -> float:
    """Forza da usare in outpaint: min(richiesta, OUTPAINT_STRENGTH o env NATIVE_OUTPAINT_STRENGTH). `requested` None = 1,0."""
    cap = OUTPAINT_STRENGTH
    try:
        env = os.environ.get("NATIVE_OUTPAINT_STRENGTH", "").strip()
        if env:
            cap = float(env)
    except ValueError:
        pass
    cap = min(1.0, max(0.05, cap))
    return min(1.0 if requested is None else float(requested), cap)


def _resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """Ridimensiona (H,W,C) float32 con interpolazione bilineare (per canale, modo 'F' di PIL)."""
    out = np.empty((h, w, a.shape[2]), np.float32)
    for c in range(a.shape[2]):
        out[..., c] = np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c])).resize((w, h), Image.BILINEAR))
    return out


def _pushpull(arr: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Campo di colore liscio che continua i pixel noti verso quelli ignoti (piramide pesata push-pull)."""
    h0, w0 = known.shape
    w = known.astype(np.float32)
    c = arr * w[..., None]
    levels = []
    while True:
        levels.append((c, w))
        h, wd = c.shape[:2]
        if h <= 1 and wd <= 1:
            break
        ph, pw = h % 2, wd % 2
        if ph or pw:
            c = np.pad(c, ((0, ph), (0, pw), (0, 0)))
            w = np.pad(w, ((0, ph), (0, pw)))
        c = c.reshape(c.shape[0] // 2, 2, c.shape[1] // 2, 2, 3).sum((1, 3))
        w = w.reshape(w.shape[0] // 2, 2, w.shape[1] // 2, 2).sum((1, 3))
    c, w = levels[-1]
    res = c / np.maximum(w[..., None], 1e-6)
    for lvl in range(len(levels) - 2, -1, -1):
        c, w = levels[lvl]
        h, wd = w.shape
        up = _resize(res, wd, h)
        own = c / np.maximum(w[..., None], 1e-6)
        a = np.clip(w * 2.0 / float(4 ** lvl), 0.0, 1.0)[..., None]      # fiducia: celle (quasi) tutte note → proprio colore
        res = own * a + up * (1.0 - a)
    return res[:h0, :w0]


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(max(0.5, float(sigma)))), dtype=np.float32)


def prefill_extension(img: Image.Image, ext_mask: np.ndarray, seed: int = 0) -> Image.Image:
    """`img`: frame del canvas (RGB, estensione = riempimento scuro). `ext_mask`: bool (H,W), True = estensione da
    riempire. Ritorna un'immagine della stessa dimensione: pixel della foto IDENTICI, estensione riempita.
    Se qualcosa va storto (o non c'è niente da riempire) ritorna `img` invariata: non deve mai far fallire il job."""
    try:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        ext = np.asarray(ext_mask, dtype=bool)
        if ext.shape != arr.shape[:2]:
            return img
        known = ~ext
        if not ext.any() or int(known.sum()) < MIN_KNOWN_PX:
            return img
        H, W = ext.shape
        field = _pushpull(arr, known)
        ys, xs = np.nonzero(known)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        is_rect = int(known.sum()) == (y1 - y0) * (x1 - x0)
        bw, bh = x1 - x0, y1 - y0
        m_ext = float(min(H, W))
        if is_rect:
            crop = arr[y0:y1, x0:x1]
            mirror = np.pad(crop, ((y0, H - y1), (x0, W - x1), (0, 0)), mode="symmetric")
            lv = [_blur(mirror, s * m_ext) for s in SIGMAS_PCT] + [field]          # sempre più liscio verso il campo
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            dx = np.maximum(np.maximum(x0 - xx, xx - (x1 - 1)), 0.0)
            dy = np.maximum(np.maximum(y0 - yy, yy - (y1 - 1)), 0.0)
            d = np.hypot(dx, dy)
            reach = max(8.0, REACH_PCT * float(min(bw, bh)))
            u = np.clip(d / reach, 0.0, 1.0) * (len(lv) - 1)                        # 0..3: indice (frazionario) del livello
            lo = np.clip(np.floor(u).astype(np.int32), 0, len(lv) - 2)
            f = (u - lo)[..., None]
            stack = np.stack(lv, 0)
            a_ = np.take_along_axis(stack, lo[None, ..., None], 0)[0]
            b_ = np.take_along_axis(stack, (lo + 1)[None, ..., None], 0)[0]
            filled = a_ * (1.0 - f) + b_ * f
        else:
            filled = _blur(field, 0.02 * m_ext)
        # grana: l'alta frequenza della foto (stessa texture del contenuto che il modello vede)
        crop_img = arr[y0:y1, x0:x1]
        hf = crop_img - _blur(crop_img, 2.0)
        std = float(np.clip(hf.std() * GRAIN_FACTOR, *GRAIN_RANGE))
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        grain = rng.normal(0.0, std, (H, W, 1)).astype(np.float32) + rng.normal(0.0, std * 0.4, (H, W, 3)).astype(np.float32)
        out = np.where(ext[..., None], filled + grain, arr)
        return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
    except Exception as exc:                                    # noqa: BLE001
        print(f"[outpaint-prefill] non applicato ({exc})", flush=True)
        return img


def prefill_from_mask_path(img: Image.Image, mask_path: str, seed: int = 0) -> Image.Image:
    """Come `prefill_extension`, con la maschera del canvas da file (bianco = estensione), anche a risoluzione diversa."""
    m = Image.open(mask_path).convert("L")
    if m.size != img.size:
        m = m.resize(img.size, Image.NEAREST)
    return prefill_extension(img, np.asarray(m, dtype=np.uint8) > 127, seed)
