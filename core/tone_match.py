"""Correzione di TONO globale del generato rispetto alla foto — outpaint e inpaint immagini (task 14).

Problema («effetto cornice», segnalato dall'utente il 18/10): il modello rigenera l'INTERA tela, anche la zona
della foto, e tutto l'output porta lo stesso scarto di tono (luminosità, contrasto, saturazione, dominante) del
modello/VAE. Dove la foto viene rimessa al pixel, l'estensione resta con il tono sbagliato: intorno alla foto
si vede una cornice più scura/chiara/sbiadita. La vecchia correzione (`_seam_tone_match` in image_flux) lavorava
solo vicino al confine, con un peso che decade con la distanza: vicino il tono tornava giusto, lontano no, e la
transizione fra le due zone era essa stessa una cornice (più medie di due strisce DIVERSE, che confondono
contenuto e tono).

Qui invece: sugli STESSI pixel della zona-foto la differenza «generato − foto» è puro scarto di tono (il
contenuto è identico). Si stima da lì una trasformazione di colore GLOBALE (affine 3×3 + bias, minimi quadrati
robusti) e la si applica a TUTTO il generato prima del composite. Dipendenze: numpy + PIL (cv2 solo se c'è,
per l'erosione e il residuo di confine; senza, ripiega su PIL).

Interruttore di emergenza: `NATIVE_OUTPAINT_TONEMATCH=0` (default ON) — i chiamanti tornano al comportamento
precedente. Nessuna GPU: tutto CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

# ── parametri (una riga ciascuno, vedi report: nessuna scelta di comportamento nascosta) ───────────────────────
SKIP_DELTA_E = 1.2        # sotto questo ΔE mediano foto↔generato la zona-foto è già fedele: non si tocca nulla
MIN_FIT_PX = 2000         # sotto questi pixel di zona-foto (dopo l'erosione) → fallback media/std per canale
MIN_ANY_PX = 200          # sotto questi → nessuna correzione (identità)
MAX_SAMPLES = 80_000      # campioni per il fit (deterministici)
BLUR_SIGMA = 2.0          # il fit usa copie sfocate: il giro nel VAE ammorbidisce i dettagli, non è tono
TRIM_QUANTILE = 0.88      # scarto robusto: i pixel con residuo oltre questo quantile non entrano nel fit
RIDGE = 1e-3              # regolarizzazione verso l'identità (relativa al numero di campioni)
SV_RANGE = (0.45, 2.2)    # valori singolari ammessi della matrice 3×3
BIAS_MAX = 0.30           # |bias| massimo (scala 0..1)
BOUNDARY_MIN_OFFSET = 5.0 # livelli 0..255 (media sui canali): sotto, lo scarto di confine è differenza di CONTENUTO
                          # (misurato: 1–3.7 su casi senza scarto reale; 7–11 con scarto reale) → non si corregge
BOUNDARY_TAU_MULT = 3.0   # decadimento della correzione di confine = feather × questo
BOUNDARY_CLAMP = 45.0     # tetto (livelli) dell'offset di confine


def enabled() -> bool:
    """`NATIVE_OUTPAINT_TONEMATCH=0` → interruttore di emergenza (i chiamanti usano la correzione di confine di prima)."""
    return os.environ.get("NATIVE_OUTPAINT_TONEMATCH", "1") != "0"


# ── colore ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _srgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB (0..1, (...,3)) → CIE L*a*b* (D65)."""
    c = np.asarray(rgb01, dtype=np.float32)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], -1)


def delta_e(a_rgb01: np.ndarray, b_rgb01: np.ndarray) -> np.ndarray:
    """ΔE CIE76 per pixel fra due array sRGB 0..1 della stessa forma."""
    return np.linalg.norm(_srgb_to_lab(a_rgb01) - _srgb_to_lab(b_rgb01), axis=-1)


def _to01(img) -> np.ndarray:
    return np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img, dtype=np.float32) / 255.0


def _blur01(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    try:
        import cv2
        return cv2.GaussianBlur(a, (0, 0), sigma)
    except Exception:                                                       # noqa: BLE001
        im = Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype("uint8"))
        return np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32) / 255.0


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode una maschera booleana di `px` pixel (la zona-foto: via i pixel di confine, dove generato e foto si mescolano)."""
    if px <= 0:
        return mask
    try:
        import cv2
        k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
        return cv2.erode(mask.astype(np.uint8), k) > 0
    except Exception:                                                       # noqa: BLE001
        im = Image.fromarray(mask.astype("uint8") * 255).filter(ImageFilter.MinFilter(2 * px + 1))
        return np.asarray(im) > 127


# ── stima ──────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class ToneFit:
    """Trasformazione y = x @ A + b (x = RGB sRGB 0..1 del generato, y = RGB della foto)."""
    A: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    method: str = "identity"          # identity | affine | per_channel
    n_px: int = 0
    de_before: float = 0.0            # ΔE medio generato↔foto sulla zona-foto, prima
    de_after: float = 0.0             # … dopo la trasformazione
    de_median_before: float = 0.0
    applied: bool = False
    reason: str = ""

    def summary(self) -> str:
        if not self.applied:
            return f"nessuna correzione ({self.reason}; ΔE zona-foto {self.de_before:.2f}, n={self.n_px})"
        gains = np.linalg.svd(self.A, compute_uv=False)
        return (f"{self.method}: ΔE zona-foto {self.de_before:.2f}→{self.de_after:.2f} (n={self.n_px}), "
                f"valori singolari A {gains.min():.2f}..{gains.max():.2f}, bias "
                f"[{self.b[0]:+.3f} {self.b[1]:+.3f} {self.b[2]:+.3f}]")

    def apply01(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x @ self.A + self.b, 0.0, 1.0)


def _ridge_affine(g: np.ndarray, o: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Minimi quadrati o ≈ g·A + b con ridge verso (I, 0)."""
    n = g.shape[0]
    X = np.concatenate([g, np.ones((n, 1), np.float32)], 1).astype(np.float64)
    Y = o.astype(np.float64)
    if w is not None:
        X = X * w[:, None]; Y = Y * w[:, None]
    lam = RIDGE * n
    W0 = np.concatenate([np.eye(3), np.zeros((1, 3))], 0)
    XtX = X.T @ X + lam * np.eye(4)
    sol = np.linalg.solve(XtX, X.T @ Y + lam * W0)
    return sol[:3].astype(np.float32), sol[3].astype(np.float32)


def _per_channel(g: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media/std per canale (fallback con pochi campioni): guadagno limitato, bias per allineare la media."""
    sg = g.std(0) + 1e-4; so = o.std(0) + 1e-4
    gain = np.clip(so / sg, 0.7, 1.4).astype(np.float32)
    bias = (o.mean(0) - g.mean(0) * gain).astype(np.float32)
    return np.diag(gain), bias


def estimate(gen, orig, photo_mask: np.ndarray, *, erode_px: int = 6, seed: int = 0) -> ToneFit:
    """Stima la trasformazione globale generato→foto sulla zona-foto.

    `gen`, `orig`: immagini PIL (o array uint8) della STESSA dimensione. `photo_mask`: bool, True = pixel della foto
    (dove generato e foto dovrebbero coincidere). Campionamento robusto: copie sfocate (il VAE ammorbidisce i
    dettagli, non è tono), zona erosa, scarto dei residui alti, controllo che l'errore sul totale dei campioni
    scenda davvero; altrimenti identità."""
    g_full = _blur01(_to01(gen), BLUR_SIGMA)
    o_full = _blur01(_to01(orig), BLUR_SIGMA)
    zone = _erode(np.asarray(photo_mask, dtype=bool), int(erode_px))
    n_zone = int(zone.sum())
    fit = ToneFit(n_px=n_zone)
    if n_zone < MIN_ANY_PX:
        fit.reason = "zona-foto troppo piccola"
        return fit
    g = g_full[zone]; o = o_full[zone]
    if n_zone > MAX_SAMPLES:                                                 # campionamento deterministico
        sel = np.random.default_rng(seed).choice(n_zone, MAX_SAMPLES, replace=False)
        g = g[sel]; o = o[sel]
    de0 = delta_e(g, o)
    fit.de_before = float(de0.mean()); fit.de_median_before = float(np.median(de0))
    if fit.de_median_before < SKIP_DELTA_E:
        fit.reason = "zona-foto già fedele"
        fit.de_after = fit.de_before
        return fit

    def _score(A, b):
        return float(delta_e(np.clip(g @ A + b, 0, 1), o).mean())

    method = "per_channel" if n_zone < MIN_FIT_PX else "affine"
    if method == "affine":
        keep = np.ones(len(g), bool)
        A, b = _ridge_affine(g, o)
        for _ in range(3):                                                   # minimi quadrati con scarto dei residui alti
            res = np.linalg.norm(np.clip(g @ A + b, 0, 1) - o, axis=1)
            keep = res <= np.quantile(res, TRIM_QUANTILE)
            A, b = _ridge_affine(g[keep], o[keep])
        sv = np.linalg.svd(A, compute_uv=False)
        if not (SV_RANGE[0] <= sv.min() and sv.max() <= SV_RANGE[1]) or np.abs(b).max() > BIAS_MAX:
            method = "per_channel"                                           # matrice instabile → fallback prudente
    if method == "per_channel":
        A, b = _per_channel(g, o)
    de1 = _score(A, b)
    if not de1 < fit.de_before * 0.9:                                        # non migliora davvero → identità
        fit.reason = f"nessun miglioramento ({fit.de_before:.2f}→{de1:.2f})"
        fit.de_after = fit.de_before
        return fit
    fit.A, fit.b, fit.method, fit.de_after, fit.applied = A, b, method, de1, True
    return fit


# ── applicazione ───────────────────────────────────────────────────────────────────────────────────────────────
def apply_fit(gen, fit: ToneFit) -> Image.Image:
    """Applica la trasformazione a TUTTO il generato (in float, un solo arrotondamento finale)."""
    if not fit.applied:
        return gen
    x = _to01(gen)
    y = fit.apply01(x.reshape(-1, 3)).reshape(x.shape)
    return Image.fromarray(np.clip(y * 255.0 + 0.5, 0, 255).astype("uint8"))


def seam_tone_match(gen, orig, gen_mask: np.ndarray, feather: int, *, gain: bool = True,
                    tau_mult: float = BOUNDARY_TAU_MULT, clamp: float = BOUNDARY_CLAMP, min_offset: float = 0.0):
    """Correzione di CONFINE (porta di `image_flux._seam_tone_match`): offset (+ guadagno) per-canale stimato su due
    strisce adiacenti al bordo, con peso esponenziale verso l'interno del generato. Serve per lo scarto che la
    zona-foto NON vede (il modello ha fedelmente riprodotto la foto ma ha dato al contenuto NUOVO un tono suo).
    `min_offset`: se lo scarto medio (livelli 0..255) è sotto questa soglia non fa nulla — due strisce diverse
    differiscono sempre un po' per contenuto, e correggerlo peggiora. Senza cv2 non fa nulla."""
    try:
        import cv2
    except Exception:                                                       # noqa: BLE001
        return gen
    mk = np.asarray(gen_mask, dtype=bool)
    ext = mk.astype(np.uint8); pho = (~mk).astype(np.uint8)
    d = max(6, int(feather))
    ker = np.ones((d, d), np.uint8)
    seam_pho = (pho > 0) & (cv2.dilate(ext, ker) > 0)
    seam_gen = (ext > 0) & (cv2.dilate(pho, ker) > 0)
    if seam_pho.sum() < 50 or seam_gen.sum() < 50:
        return gen
    g = np.asarray(gen.convert("RGB"), dtype=np.float32)
    o = np.asarray(orig.convert("RGB"), dtype=np.float32)
    m_pho = o[seam_pho].mean(0); m_gen = g[seam_gen].mean(0)
    if float(np.abs(m_pho - m_gen).mean()) < float(min_offset):
        return gen
    off = np.clip(m_pho - m_gen, -clamp, clamp)
    if gain:
        gn = np.clip((o[seam_pho].std(0) + 1e-3) / (g[seam_gen].std(0) + 1e-3), 0.80, 1.25)
    else:
        gn = np.ones(3, np.float32)
    dist = cv2.distanceTransform(ext, cv2.DIST_L2, 3)
    tau = max(24.0, float(feather) * tau_mult)
    w = (np.exp(-dist / tau) * mk).astype(np.float32)[:, :, None]
    corr = (g - m_gen) * gn + m_gen + off
    return Image.fromarray(np.clip(g * (1.0 - w) + corr * w, 0, 255).astype("uint8"))



def harmonize(gen, orig, gen_mask: np.ndarray, feather: int, *, boundary: str = "gated",
              tau_mult: float = BOUNDARY_TAU_MULT, clamp: float = BOUNDARY_CLAMP, tag: str = "tone", log=print):
    """Punto d'ingresso per i motori: porta il tono di TUTTO il generato a quello della foto.

    `gen`/`orig`: PIL RGB della stessa dimensione. `gen_mask`: bool, True = pixel GENERATO (estensione/area
    mascherata), False = pixel della foto. Due stadi:
      1. globale: trasformazione di colore stimata sulla zona-foto (stessi pixel), applicata a tutto il generato;
      2. confine (`boundary`): "gated" (default, scelto con le misure del task 14) = correzione di confine a
         decadimento SOLO se dopo lo stadio 1 lo scarto fra le strisce di bordo supera BOUNDARY_MIN_OFFSET
         (scarto reale dell'estensione); "always" = sempre (come prima); "none" = mai.
    Ritorna l'immagine corretta (`gen` invariato se qualcosa va storto: la correzione di tono non deve mai far
    fallire una generazione)."""
    try:
        gen = gen.convert("RGB"); orig = orig.convert("RGB")
        mk = np.asarray(gen_mask, dtype=bool)
        fit = estimate(gen, orig, ~mk, erode_px=max(4, min(12, int(feather) // 2)))
        out = apply_fit(gen, fit)
        note = ""
        if boundary != "none":
            before = out
            out = seam_tone_match(out, orig, mk, feather, tau_mult=tau_mult, clamp=clamp,
                                  min_offset=BOUNDARY_MIN_OFFSET if boundary == "gated" else 0.0)
            note = " + confine" if out is not before else " (confine: scarto sotto soglia)"
        log(f"[{tag}] {fit.summary()}{note}")
        return out
    except Exception as e:                                                  # noqa: BLE001
        log(f"[{tag}] correzione di tono saltata ({type(e).__name__}: {e})")
        return gen
