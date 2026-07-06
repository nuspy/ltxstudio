"""Rilevamento firma ATF — port PURO numpy+scipy+PIL (niente OpenCV) per Android/Kivy.

Riproduce ESATTAMENTE l'algoritmo di core/stego.py (stessa CHIAVE, stessi carrier, stessa DCT) ma senza
OpenCV, così si impacchetta facilmente con buildozer/python-for-android. La DCT usa scipy.fft (identica a
cv2.dct entro 1e-6). Solo RILEVAMENTO (immagini) — la firma la mette il desktop.

⚠️ La CHIAVE deve restare identica a core/stego.py._KEY, altrimenti non rileva nulla.
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

# ---- costanti IDENTICHE a core/stego.py ----
_KEY = b"AIToolsForge::provenance::v1::do-not-share"
GRID = 8
CANON = 80
NONCE_BANK = 24
_Z_THRESH_ROBUST = 6.5
PAYLOAD_TEXT_BYTES = 5
PAYLOAD_BITS = (PAYLOAD_TEXT_BYTES + 1) * 8


def _midband_mask():
    u = np.arange(CANON)[:, None]; v = np.arange(CANON)[None, :]
    r = u + v
    m = (r >= 4) & (r <= 26); m[:2, :2] = False
    return m


_MASK = _midband_mask()
_N_COEFF = int(_MASK.sum())


def _sha_bits(key_material, n):
    need = (n + 7) // 8
    buf = bytearray()
    counter = 0
    while len(buf) < need:
        buf += hashlib.sha256(key_material + counter.to_bytes(4, "big")).digest()
        counter += 1
    bits = np.unpackbits(np.frombuffer(bytes(buf[:need]), dtype=np.uint8))[:n]
    return bits.astype(np.float32) * 2 - 1


def _carrier(nonce, tile_idx):
    return _sha_bits(_KEY + b"C" + nonce.to_bytes(4, "big") + tile_idx.to_bytes(2, "big"), _N_COEFF)


_CMAT = {}


def _carrier_matrix(tile_idx):
    m = _CMAT.get(tile_idx)
    if m is None:
        m = np.stack([_carrier(n, tile_idx) for n in range(NONCE_BANK)])
        _CMAT[tile_idx] = m
    return m


_PMAT = {}


def _payload_matrix(nonce, tile_idx):
    k = (nonce, tile_idx)
    m = _PMAT.get(k)
    if m is None:
        km = _KEY + b"P" + nonce.to_bytes(4, "big") + tile_idx.to_bytes(2, "big")
        m = _sha_bits(km, PAYLOAD_BITS * _N_COEFF).reshape(PAYLOAD_BITS, _N_COEFF)
        _PMAT[k] = m
    return m


def _crc8(data):
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if (c & 0x80) else (c << 1) & 0xFF
    return c


def _bits_to_text(signs):
    b = np.packbits((signs > 0).astype(np.uint8)).tobytes()
    if len(b) < PAYLOAD_TEXT_BYTES + 1 or _crc8(b[:PAYLOAD_TEXT_BYTES]) != b[PAYLOAD_TEXT_BYTES]:
        return None
    try:
        return b[:PAYLOAD_TEXT_BYTES].rstrip(b"\x00").decode("utf-8")
    except Exception:
        return None


# ---- primitive immagine SENZA cv2 ----
def _luma(rgb):
    """Luminanza Y (BT.601, identica a cv2 COLOR_*2YCrCb) da un array RGB float."""
    return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


def _resize(arr, w, h):
    """Ridimensiona un array 2D float con PIL (bilineare) — approssima cv2.resize INTER_LINEAR."""
    return np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR),
                      dtype=np.float32)


def _dct_matrix(n):
    """Matrice DCT-II ortonormale (identica a cv2.dct/scipy norm='ortho') — pura numpy, niente scipy."""
    k = np.arange(n)[:, None]; x = np.arange(n)[None, :]
    C = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    C[0, :] = np.sqrt(1.0 / n)
    return C.astype(np.float32)


_DCTC = _dct_matrix(CANON)


def _dct(t):
    return _DCTC @ t @ _DCTC.T                               # DCT-II 2D ortonormale via matmul


def _tiles(h, w):
    ys = np.linspace(0, h, GRID + 1).astype(int); xs = np.linspace(0, w, GRID + 1).astype(int)
    i = 0
    for a in range(GRID):
        for b in range(GRID):
            yield i, ys[a], ys[a + 1], xs[b], xs[b + 1]; i += 1


def _detect_Y(Y):
    corr = np.zeros(NONCE_BANK, np.float64); energy = 0.0; ncoef = 0
    for idx, y0, y1, x0, x1 in _tiles(*Y.shape):
        tile = Y[y0:y1, x0:x1]
        if tile.shape[0] < 8 or tile.shape[1] < 8:
            continue
        D = _dct(_resize(tile, CANON, CANON))[_MASK]; D = D - D.mean()
        energy += float(D @ D); ncoef += D.size
        corr += _carrier_matrix(idx) @ D
    if ncoef == 0 or energy <= 0:
        return 0.0, -1
    sigma = np.sqrt(energy / ncoef)
    z = corr / (sigma * np.sqrt(ncoef)); best = int(np.argmax(z))
    return float(z[best]), best


def _decode_Y(Y, nonce):
    corr = np.zeros(PAYLOAD_BITS, np.float64)
    for idx, y0, y1, x0, x1 in _tiles(*Y.shape):
        tile = Y[y0:y1, x0:x1]
        if tile.shape[0] < 8 or tile.shape[1] < 8:
            continue
        D = _dct(_resize(tile, CANON, CANON))[_MASK]
        corr += _payload_matrix(nonce, idx) @ (D - D.mean())
    return _bits_to_text(corr)


def _rot_scale(Y, ang_deg, scale):
    """Rotazione+scala attorno al centro via PIL (per il resync geometrico)."""
    im = Image.fromarray(Y.astype(np.float32), mode="F")
    if scale != 1.0:
        im = im.resize((max(1, int(Y.shape[1] * scale)), max(1, int(Y.shape[0] * scale))), Image.BILINEAR)
    im = im.rotate(-ang_deg, resample=Image.BILINEAR, expand=False, fillcolor=0.0)
    out = np.asarray(im, dtype=np.float32)
    # ripristina la dimensione originale (il grid è relativo)
    if out.shape != Y.shape:
        out = _resize(out, Y.shape[1], Y.shape[0])
    return out


def detect_rgb(rgb, max_angle=15.0):
    """rgb: array HxWx3 uint8/float (0..255). Ritorna (found, z, testo|None) con resync geometrico."""
    rgb = np.asarray(rgb, dtype=np.float32)
    Y0 = _luma(rgb)
    h, w = Y0.shape
    s = 512.0 / max(h, w)
    if s < 1.0:
        Y0 = _resize(Y0, int(w * s), int(h * s))
    best_z, best_n, best_Y = _detect_Y(Y0) + (Y0,)
    for ang in np.arange(-max_angle, max_angle + 0.1, 1.0):
        for sc in (0.90, 0.95, 1.0, 1.05, 1.10):
            if abs(ang) < 1e-6 and abs(sc - 1.0) < 1e-6:
                continue
            Yw = _rot_scale(Y0, float(ang), float(sc))
            z, n = _detect_Y(Yw)
            if z > best_z:
                best_z, best_n, best_Y = z, n, Yw
    found = bool(best_z >= _Z_THRESH_ROBUST)
    return found, float(best_z), (_decode_Y(best_Y, best_n) if found else None)


def detect_path(path, max_angle=15.0):
    img = Image.open(path).convert("RGB")
    return detect_rgb(np.asarray(img), max_angle)
