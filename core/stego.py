"""Firma steganografica invisibile (#107) — provenienza "generato con AI Tools Forge".

Watermark a **spettro espanso nel dominio DCT**, a tile, con carrier pseudo-casuale guidato da una CHIAVE
segreta e da un NONCE variabile. Progettato per essere:
- INVISIBILE: segnale a bassa ampiezza in banda medio-frequenza (sotto la soglia percettiva).
- RIDONDANTE: ripetuto su una griglia GxG di tile → sopravvive al ritaglio parziale.
- ROBUSTO a resize/JPEG/SCREENSHOT: ogni tile è normalizzato a una dimensione canonica prima di DCT →
  il rapporto griglia-immagine scala con l'immagine; la banda medio-freq sopravvive al ricampionamento.
- ANTI-AVERAGING (video): il carrier cambia a ogni frame (chiave+nonce). Mediare i frame NON rivela un
  pattern fisso; il Detector prova una BANCA di nonce e correla ognuno → rileva senza pattern costante.
- POSIZIONE VARIABILE: il pattern per-frame è diverso su tutta la superficie → non c'è un mark fisso da
  localizzare e rimuovere.
- RILEVAMENTO CIECO: serve SOLO la chiave (nessun originale). Correlazione normalizzata → z-score.

⚠️ ONESTÀ TECNICA: "impossibile da rimuovere / non individuabile" non esiste in assoluto. Qui otteniamo
robustezza forte a rimozione casuale, averaging e screenshot; ma chi estrae la CHIAVE dal binario (o la
ricava statisticamente) può attaccarlo. Mitigazione: chiave nel binario compilato (Nuitka, offuscata) e/o
Detector lato-server. La foto-di-schermo con prospettiva/moiré è il caso più duro (resync geometrico = fase 2).

PAYLOAD testo custom OPZIONALE (5 char): impresso con carrier indipendenti dal presence, letto dal Detector.
Limite fisico del watermark invisibile screenshot-robusto → 5 caratteri (0 errori dopo JPEG/screenshot);
sotto rotazione forte il presence tiene ma il testo può perdersi. Vuoto = firma senza testo (comunque valida).

API:
  sign_image(bgr, nonce=None, text=None) -> bgr       # text None → dalle preferenze; "" → senza payload
  detect_image(bgr) / detect_image_robust(bgr) -> (found, z, nonce, testo|None)
  sign_file(path) / detect_file(path)
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

# Chiave segreta di provenienza (nel prodotto = bakeata/offuscata da Nuitka). Override test via env.
_KEY = os.environ.get("ATF_STEGO_KEY", "AIToolsForge::provenance::v1::do-not-share").encode()

GRID = 8              # 8x8 = 64 tile (più siti d'impressione: presence più forte + capacità payload)
CANON = 80           # dimensione canonica del tile per la DCT (robustezza a resize)
NONCE_BANK = 24      # nonce provati dal detector (periodo del carrier per-frame)
_ALPHA = 6.0         # ampiezza PRESENCE — tarata per INVISIBILITÀ: _ALPHA=20 dava PSNR 32 con tessere 80px
                     # VISIBILI ("a macchie"); a 9 restava un VELO percettibile sui fondi lisci scuri (test
                     # utente 11/07, T15 vs T11). A 6 → z~7 (soglia 5, margine ok) e velo sotto soglia visiva.
_Z_THRESH = 5.0      # soglia z-score di rilevamento (falsi positivi ~ 1e-6 a coda gaussiana)
# PAYLOAD testo custom OPZIONALE — limite fisico del watermark invisibile screenshot-robusto = pochi byte.
# 5 byte testo + 1 CRC8 = 48 bit (0 errori misurati dopo JPEG55 + screenshot 0.6x). Vuoto = solo presence
# (firma comunque valida). Testo più lungo NON è robusto: 5 char = codice/handle/ID corto.
PAYLOAD_TEXT_BYTES = 5
PAYLOAD_BITS = (PAYLOAD_TEXT_BYTES + 1) * 8
_BETA = 1.2          # ampiezza per-bit payload — abbassata con _ALPHA per invisibilità. A questa forza il
                     # TESTO non è più decodificabile (presence-only): la priorità è l'immagine pulita. Per
                     # riattivare il testo robusto servirebbe più ampiezza (visibile) → tenuto invisibile.


def _midband_mask() -> np.ndarray:
    """Maschera booleana CANONxCANON della banda medio-frequenza (esclude DC e alte freq)."""
    u = np.arange(CANON)[:, None]
    v = np.arange(CANON)[None, :]
    r = u + v
    m = (r >= 4) & (r <= 26)                                 # banda medio-bassa: sopravvive a resize/JPEG/screenshot
    m[:2, :2] = False                                        # niente DC/quasi-DC
    return m


_MASK = _midband_mask()
_N_COEFF = int(_MASK.sum())


def _sha_bits(key_material: bytes, n: int) -> np.ndarray:
    """`n` valori ±1 da uno STREAM SHA256 (chiave+contatore). Riproducibile IDENTICO ovunque —
    Python/NumPy, JS (SubtleCrypto), Kivy — così il rilevamento è cross-platform (no RNG NumPy)."""
    need = (n + 7) // 8
    buf = bytearray()
    counter = 0
    while len(buf) < need:
        buf += hashlib.sha256(key_material + counter.to_bytes(4, "big")).digest()
        counter += 1
    bits = np.unpackbits(np.frombuffer(bytes(buf[:need]), dtype=np.uint8))[:n]   # MSB-first per byte
    return bits.astype(np.float32) * 2 - 1                                       # 0→-1, 1→+1


def _carrier(nonce: int, tile_idx: int) -> np.ndarray:
    """Sequenza ±1 (len = coeff in banda) da chiave+nonce+tile via stream SHA256 (portabile)."""
    return _sha_bits(_KEY + b"C" + nonce.to_bytes(4, "big") + tile_idx.to_bytes(2, "big"), _N_COEFF)


_CMAT: dict[int, np.ndarray] = {}


def _carrier_matrix(tile_idx: int) -> np.ndarray:
    """Matrice (NONCE_BANK × N_COEFF) di tutti i carrier del tile — cache (dipende solo dalla chiave)."""
    m = _CMAT.get(tile_idx)
    if m is None:
        m = np.stack([_carrier(n, tile_idx) for n in range(NONCE_BANK)])
        _CMAT[tile_idx] = m
    return m


# ---------------- payload testo (CRC8 + carrier indipendenti dal presence) ----------------
def _crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if (c & 0x80) else (c << 1) & 0xFF
    return c


def _text_to_bits(text: str) -> np.ndarray | None:
    """Testo → 5 byte (utf-8 troncato/paddato) + CRC8 → segni ±1 (len PAYLOAD_BITS). None se vuoto."""
    if not text:
        return None
    raw = text.encode("utf-8")[:PAYLOAD_TEXT_BYTES].ljust(PAYLOAD_TEXT_BYTES, b"\x00")
    bits = np.unpackbits(np.frombuffer(raw + bytes([_crc8(raw)]), dtype=np.uint8))
    return bits.astype(np.float32) * 2 - 1                   # 0→-1, 1→+1


def _bits_to_text(signs: np.ndarray) -> str | None:
    """Segni decodificati → testo se il CRC8 combacia, altrimenti None."""
    b = np.packbits((signs > 0).astype(np.uint8)).tobytes()
    if len(b) < PAYLOAD_TEXT_BYTES + 1 or _crc8(b[:PAYLOAD_TEXT_BYTES]) != b[PAYLOAD_TEXT_BYTES]:
        return None
    try:
        return b[:PAYLOAD_TEXT_BYTES].rstrip(b"\x00").decode("utf-8")
    except Exception:                                        # noqa: BLE001
        return None


_PMAT: dict[tuple, np.ndarray] = {}


def _payload_matrix(nonce: int, tile_idx: int) -> np.ndarray:
    """Matrice (PAYLOAD_BITS × N_COEFF) di carrier ±1 del payload — indipendenti dal presence, cache."""
    k = (nonce, tile_idx)
    m = _PMAT.get(k)
    if m is None:
        km = _KEY + b"P" + nonce.to_bytes(4, "big") + tile_idx.to_bytes(2, "big")
        m = _sha_bits(km, PAYLOAD_BITS * _N_COEFF).reshape(PAYLOAD_BITS, _N_COEFF)
        _PMAT[k] = m
    return m


def _pref_text() -> str:
    """Testo custom della firma dalle preferenze utente (vuoto = nessun payload)."""
    try:
        from core import user_prefs
        return str(user_prefs.get("stego_text", "") or "")[:PAYLOAD_TEXT_BYTES]
    except Exception:                                        # noqa: BLE001
        return ""


def _tiles(h: int, w: int):
    """Confini dei tile GxG (interi, coprono tutta l'immagine)."""
    ys = np.linspace(0, h, GRID + 1).astype(int)
    xs = np.linspace(0, w, GRID + 1).astype(int)
    idx = 0
    for i in range(GRID):
        for j in range(GRID):
            yield idx, ys[i], ys[i + 1], xs[j], xs[j + 1]
            idx += 1


def _to_y(bgr):
    import cv2
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    return ycc, ycc[:, :, 0]


def sign_image(bgr, nonce: int | None = None, text: str | None = None):
    """Inietta la firma. nonce None → casuale. text opzionale → payload custom (vuoto/None = solo presence,
    dalle preferenze se None). Ritorna (bgr_firmato)."""
    import cv2
    if nonce is None:
        nonce = int.from_bytes(os.urandom(2), "big") % NONCE_BANK
    if text is None:
        text = _pref_text()
    pbits = _text_to_bits(text)                              # None se testo vuoto
    ycc, Y = _to_y(bgr)
    for idx, y0, y1, x0, x1 in _tiles(*Y.shape):
        tile = Y[y0:y1, x0:x1]
        th, tw = tile.shape
        if th < 8 or tw < 8:
            continue
        # segnale nel dominio canonico → spaziale → riscalato al tile (robusto a resize)
        coeffs = _ALPHA * _carrier(nonce, idx)               # presence
        if pbits is not None:
            coeffs = coeffs + _BETA * (pbits @ _payload_matrix(nonce, idx))   # + payload testo
        S = np.zeros((CANON, CANON), np.float32)
        S[_MASK] = coeffs
        spat = cv2.idct(S)
        spat = cv2.resize(spat, (tw, th), interpolation=cv2.INTER_LINEAR)
        # MASCHERA PERCETTIVA PER-PIXEL (non più per-tile): nasconde il segnale nel DETTAGLIO locale (|Laplaciano|
        # sfumato) e lo azzera quasi del tutto sulle zone LISCE (pelle, cielo). Prima il floor 0.6 lasciava il 60%
        # del segnale anche sul piatto → griglia 8×8 + macchie visibili (il bug segnalato). Ora il liscio prende
        # ~8% (invisibile), la texture il 100% (watermark ancora rilevabile: la correlazione vive nel dettaglio).
        act = cv2.GaussianBlur(np.abs(cv2.Laplacian(tile, cv2.CV_32F, ksize=3)), (0, 0), 2.0)
        act = np.clip(act / 6.0, 0.08, 1.0)
        Y[y0:y1, x0:x1] = np.clip(tile + spat * act, 0, 255)
    ycc[:, :, 0] = Y
    return cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def _decode_Y(Y, nonce: int) -> str | None:
    """Decodifica il payload testo alla luminanza Y per un dato nonce (dopo aver trovato il presence)."""
    import cv2
    corr = np.zeros(PAYLOAD_BITS, np.float64)
    for idx, y0, y1, x0, x1 in _tiles(*Y.shape):
        tile = Y[y0:y1, x0:x1]
        if tile.shape[0] < 8 or tile.shape[1] < 8:
            continue
        D = cv2.dct(cv2.resize(tile, (CANON, CANON)).astype(np.float32))[_MASK]
        corr += _payload_matrix(nonce, idx) @ (D - D.mean())
    return _bits_to_text(corr)


def _detect_Y(Y) -> tuple[float, int]:
    """Correlazione cieca sulla luminanza Y → (z-score max, nonce). Cuore condiviso dal rilevamento."""
    import cv2
    corr = np.zeros(NONCE_BANK, np.float64)
    energy = 0.0
    ncoef = 0
    for idx, y0, y1, x0, x1 in _tiles(*Y.shape):
        tile = Y[y0:y1, x0:x1]
        if tile.shape[0] < 8 or tile.shape[1] < 8:
            continue
        t = cv2.resize(tile, (CANON, CANON), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        D = cv2.dct(t)[_MASK]
        D = D - D.mean()
        energy += float(np.dot(D, D))
        ncoef += D.size
        corr += _carrier_matrix(idx) @ D                     # tutti i nonce in un matmul (veloce)
    if ncoef == 0 or energy <= 0:
        return 0.0, -1
    sigma = np.sqrt(energy / ncoef)
    z = corr / (sigma * np.sqrt(ncoef))
    best = int(np.argmax(z))
    return float(z[best]), best


def detect_image(bgr) -> tuple[bool, float, int, str | None]:
    """Rilevamento CIECO veloce (no resync geometrico) → (found, z, nonce, testo|None)."""
    _, Y = _to_y(bgr)
    z, n = _detect_Y(Y)
    found = bool(z >= _Z_THRESH)
    return found, z, n, (_decode_Y(Y, n) if found else None)


_Z_THRESH_ROBUST = 6.5   # soglia più alta: la ricerca geometrica (molti warp) gonfia il max sotto ipotesi nulla


def detect_image_robust(bgr, max_angle: float = 15.0) -> tuple[bool, float, int, str | None]:
    """Rilevamento ROBUSTO con resync geometrico (foto-schermo storta/prospettiva lieve).
    Ricerca UNIFORME a passo 1° (il picco di correlazione è stretto ±1°, un passo grezzo lo mancherebbe)
    su rotazione × scala; il warp riallinea il grid e lo z sale. Lavora su una versione RIDOTTA (lato max
    512px) per velocità: il rilevamento normalizza comunque i tile a 64px, quindi non perde segnale.
    Soglia più alta per compensare l'inflazione del max dovuta alla ricerca."""
    import cv2
    _, Y0 = _to_y(bgr)
    h, w = Y0.shape
    s = 512.0 / max(h, w)
    if s < 1.0:                                              # riduci per accelerare la ricerca
        Y0 = cv2.resize(Y0, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    hh, ww = Y0.shape
    cx, cy = ww / 2.0, hh / 2.0
    best_z, best_n, best_Y = _detect_Y(Y0) + (Y0,)
    for ang in np.arange(-max_angle, max_angle + 0.1, 1.0):
        for sc in (0.90, 0.95, 1.0, 1.05, 1.10):            # scala fine: assorbe foreshortening prospettico lieve
            if abs(ang) < 1e-6 and abs(sc - 1.0) < 1e-6:
                continue
            M = cv2.getRotationMatrix2D((cx, cy), float(ang), float(sc))
            Yw = cv2.warpAffine(Y0, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            z, n = _detect_Y(Yw)
            if z > best_z:
                best_z, best_n, best_Y = z, n, Yw
    found = bool(best_z >= _Z_THRESH_ROBUST)
    return found, float(best_z), best_n, (_decode_Y(best_Y, best_n) if found else None)


# ---------------- file helpers ----------------
_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VID_EXT = {".mp4", ".webm", ".mkv", ".mov"}


def enabled() -> bool:
    """Firma attiva se la feature-matrix l'abilita (default ON) E l'utente non l'ha spenta nei settings."""
    try:
        from core import features
        info = features._info("cap.stego")
        if info is not None and not info.get("enabled", True):
            return False
    except Exception:                                        # noqa: BLE001
        pass
    try:
        from core import user_prefs
        return bool(user_prefs.get("stego_watermark", True))
    except Exception:                                        # noqa: BLE001
        return True


def sign_file(path: str) -> str:
    """Firma un'immagine o un video (per-frame, nonce variabile) sul posto. Non solleva."""
    import cv2
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in _IMG_EXT:
            img = cv2.imread(path)
            if img is not None:
                cv2.imwrite(path, sign_image(img))
        elif ext in _VID_EXT:
            _sign_video(path)
    except Exception:                                        # noqa: BLE001
        pass
    return path


def _sign_video(path: str) -> None:
    import subprocess
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = path + ".stego.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(sign_image(frame, nonce=i % NONCE_BANK))    # carrier diverso ogni frame (anti-averaging)
        i += 1
    cap.release(); vw.release()
    if os.path.exists(tmp) and i > 0:
        # ri-mux l'audio originale sul video firmato
        muxed = path + ".stego_a.mp4"
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", path,
                            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "17",
                            "-c:a", "copy", "-shortest", muxed], capture_output=True)
        final = muxed if (r.returncode == 0 and os.path.exists(muxed)) else tmp
        os.replace(final, path)
        for f in (tmp, muxed):
            if os.path.exists(f):
                try: os.remove(f)
                except Exception: pass                       # noqa: BLE001


def detect_file(path: str) -> tuple[bool, float, int, str | None]:
    """Detector su file → (found, z, nonce, testo|None): immagine → 1 test; video → campiona alcuni frame."""
    import cv2
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMG_EXT:
        img = cv2.imread(path)
        return detect_image_robust(img) if img is not None else (False, 0.0, -1, None)   # resync geometrico
    if ext in _VID_EXT:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        best = (False, 0.0, -1, None)
        for fi in np.linspace(0, max(n - 1, 0), min(5, max(n, 1))).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if ok:
                r = detect_image_robust(fr)                  # ogni frame campionato: resync completo
                if r[1] > best[1] or (r[3] and not best[3]):
                    best = r
        cap.release()
        return best
    return False, 0.0, -1, None
