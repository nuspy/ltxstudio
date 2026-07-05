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

API:
  sign_image(bgr, nonce=None) -> bgr           # marchia (nonce casuale se None → salvato per frame video)
  detect_image(bgr) -> (found: bool, score: float, nonce: int)
  sign_file(path) / detect_file(path)
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

# Chiave segreta di provenienza (nel prodotto = bakeata/offuscata da Nuitka). Override test via env.
_KEY = os.environ.get("ATF_STEGO_KEY", "AIToolsForge::provenance::v1::do-not-share").encode()

GRID = 6              # 6x6 = 36 tile (ridondanza vs crop)
CANON = 64           # dimensione canonica del tile per la DCT (robustezza a resize)
NONCE_BANK = 24      # nonce provati dal detector (periodo del carrier per-frame)
_ALPHA = 20.0        # ampiezza segnale in unità DCT (invisibile ~PSNR 37, rilevabile z~8); scalata per energia locale
_Z_THRESH = 5.0      # soglia z-score di rilevamento (falsi positivi ~ 1e-6 a coda gaussiana)


def _midband_mask() -> np.ndarray:
    """Maschera booleana CANONxCANON della banda medio-frequenza (esclude DC e alte freq)."""
    u = np.arange(CANON)[:, None]
    v = np.arange(CANON)[None, :]
    r = u + v
    m = (r >= 4) & (r <= 16)                                 # banda medio-bassa: sopravvive a resize/JPEG/screenshot
    m[:2, :2] = False                                        # niente DC/quasi-DC
    return m


_MASK = _midband_mask()
_N_COEFF = int(_MASK.sum())


def _carrier(nonce: int, tile_idx: int) -> np.ndarray:
    """Sequenza ±1 (len = coeff in banda) da chiave+nonce+tile — deterministica e blindata."""
    seed = int.from_bytes(hashlib.sha256(_KEY + nonce.to_bytes(4, "big") + tile_idx.to_bytes(2, "big"))
                          .digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, _N_COEFF) * 2 - 1).astype(np.float32)


_CMAT: dict[int, np.ndarray] = {}


def _carrier_matrix(tile_idx: int) -> np.ndarray:
    """Matrice (NONCE_BANK × N_COEFF) di tutti i carrier del tile — cache (dipende solo dalla chiave)."""
    m = _CMAT.get(tile_idx)
    if m is None:
        m = np.stack([_carrier(n, tile_idx) for n in range(NONCE_BANK)])
        _CMAT[tile_idx] = m
    return m


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


def sign_image(bgr, nonce: int | None = None):
    """Inietta la firma. nonce None → casuale (0..NONCE_BANK-1). Ritorna (bgr_firmato)."""
    import cv2
    if nonce is None:
        nonce = int.from_bytes(os.urandom(2), "big") % NONCE_BANK
    ycc, Y = _to_y(bgr)
    for idx, y0, y1, x0, x1 in _tiles(*Y.shape):
        tile = Y[y0:y1, x0:x1]
        th, tw = tile.shape
        if th < 8 or tw < 8:
            continue
        # segnale nel dominio canonico → spaziale → riscalato al tile (robusto a resize)
        S = np.zeros((CANON, CANON), np.float32)
        S[_MASK] = _ALPHA * _carrier(nonce, idx)
        spat = cv2.idct(S)
        spat = cv2.resize(spat, (tw, th), interpolation=cv2.INTER_LINEAR)
        spat *= (0.6 + 0.4 * min(1.0, tile.std() / 32.0))    # meno segnale nelle zone piatte (invisibilità)
        Y[y0:y1, x0:x1] = np.clip(tile + spat, 0, 255)
    ycc[:, :, 0] = Y
    return cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


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


def detect_image(bgr) -> tuple[bool, float, int]:
    """Rilevamento CIECO veloce (no resync geometrico) — per il check interno rapido dell'app."""
    _, Y = _to_y(bgr)
    z, n = _detect_Y(Y)
    return bool(z >= _Z_THRESH), z, n


_Z_THRESH_ROBUST = 6.5   # soglia più alta: la ricerca geometrica (molti warp) gonfia il max sotto ipotesi nulla


def detect_image_robust(bgr, max_angle: float = 15.0) -> tuple[bool, float, int]:
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
    best_z, best_n = _detect_Y(Y0)
    for ang in np.arange(-max_angle, max_angle + 0.1, 1.0):
        for sc in (0.90, 0.95, 1.0, 1.05, 1.10):            # scala fine: assorbe foreshortening prospettico lieve
            if abs(ang) < 1e-6 and abs(sc - 1.0) < 1e-6:
                continue
            M = cv2.getRotationMatrix2D((cx, cy), float(ang), float(sc))
            Yw = cv2.warpAffine(Y0, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            z, n = _detect_Y(Yw)
            if z > best_z:
                best_z, best_n = z, n
    return bool(best_z >= _Z_THRESH_ROBUST), float(best_z), best_n


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


def detect_file(path: str) -> tuple[bool, float, int]:
    """Detector su file: immagine → 1 test; video → campiona alcuni frame e vota."""
    import cv2
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMG_EXT:
        img = cv2.imread(path)
        return detect_image_robust(img) if img is not None else (False, 0.0, -1)   # resync geometrico
    if ext in _VID_EXT:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        best = (False, 0.0, -1)
        for fi in np.linspace(0, max(n - 1, 0), min(5, max(n, 1))).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if ok:
                r = detect_image_robust(fr)                  # ogni frame campionato: resync completo
                if r[1] > best[1]:
                    best = r
        cap.release()
        return best
    return False, 0.0, -1
