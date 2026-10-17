"""Preprocessing delle modalità MagicQuill V2 (maschere, condizioni locali/colore/bordi, prompt). Solo numpy/PIL/cv2.

Derivato da MagicQuillV2 `edit.py` (KontextEditModel.*_edit), `utils_node.py` (GrowMask, BlendInpaint) e `app.py`
(scelta modalità e prompt) — CC BY-NC 4.0, (c) Zichen Liu et al. Vedi `NOTICE.md` (cartella del componente) per
attribuzione e avvertenza di licenza (NON commerciale). Riscritto in numpy/cv2 senza torch/scipy/ComfyUI.

Convenzioni: maschere = uint8 H x W con 255 = area (bianco). Immagini = uint8 H x W x 3 RGB.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

MODES = ("local", "removal", "edge", "color", "foreground")
PROMPT_REMOVAL = "remove the instance"
PROMPT_FOREGROUND = ("Fill in the white region naturally and adapt the foreground into the background. "
                     "Fix the perspective of the foreground object if necessary. ")
PROMPT_FIX_PERSPECTIVE = " Fix the perspective if necessary."

# Kernel "tapered_corners" di GrowMask: croce 3x3 (angoli esclusi) applicata `expand` volte (dilatazione a diamante)
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

# Pesi opzionali per edge_source="hed" (nessuno presente sul PC: errore chiaro con nome e dimensione)
HED_WEIGHTS_NAME = "ControlNetHED.pth"
HED_WEIGHTS_MB = 30                       # ~29,4 MB (lllyasviel/Annotators)
HED_CANDIDATES = (r"E:\AI\Models\ControlNet\ControlNetHED.pth", r"E:\AI\Models\annotators\ControlNetHED.pth")


class MQError(RuntimeError):
    """Errore di validazione/risorse con messaggio utente (diventa {"status": "error"})."""


# ---- dimensioni / caricamento -----------------------------------------------------------------------------------
def gen_size(iw: int, ih: int, cap: int = 1536) -> tuple[int, int]:
    """Come image_flux.run: multipli di 16, lato lungo <= cap (rapporto del canvas, mai stirato)."""
    s = min(1.0, cap / max(iw, ih))
    return max(256, round(iw * s / 16) * 16), max(256, round(ih * s / 16) * 16)


def load_rgb(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if size and im.size != tuple(size):
        im = im.resize(tuple(size), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8).copy()


def load_mask(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """Maschera da file B/N (bianco = area): L > 127. Ridimensionata NEAREST (resta binaria)."""
    im = Image.open(path).convert("L")
    if size and im.size != tuple(size):
        im = im.resize(tuple(size), Image.NEAREST)
    return ((np.asarray(im) > 127).astype(np.uint8) * 255)


# ---- maschere ---------------------------------------------------------------------------------------------------
def grow_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilatazione `px` iterazioni con la croce 3x3 (== GrowMask(tapered_corners=True)); px<=0 = invariata."""
    if px <= 0:
        return mask.copy()
    return cv2.dilate(mask, _CROSS, iterations=int(px))


def black_masked(img: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Immagine con l'area mascherata NERA (condizione di 'local' e 'removal': `_apply_black_mask`)."""
    out = img.copy()
    out[mask > 127] = 0
    return Image.fromarray(out)


def foreground_image(img: np.ndarray, add_prop: np.ndarray, fill: np.ndarray | None, grow: int):
    """Modalità foreground (`foreground_edit`): edit_mask = clamp(grow(oggetto) + fill); nell'area edit ma FUORI
    dall'oggetto l'immagine diventa BIANCA (zona che il modello deve riempire). Ritorna (immagine, edit_mask)."""
    edit = grow_mask(add_prop, grow)
    if fill is not None:
        edit = np.maximum(edit, fill)
    out = img.copy()
    only = (edit > 127) & (add_prop <= 127)
    out[only] = 255
    return Image.fromarray(out), edit


# ---- condizioni -------------------------------------------------------------------------------------------------
def color_block(img: np.ndarray) -> Image.Image:
    """Condizione 'color': immagine -> 32x32 (INTER_AREA) -> 256x256 (NEAREST). Il resize a 512 lo fa la pipeline."""
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST))


def fill_color_region(img: np.ndarray, mask: np.ndarray, rgb, feather: int = 12) -> np.ndarray:
    """Variante 'color_rgb': colora l'area mascherata con `rgb`, bordo sfumato (gaussiana `feather` px)."""
    m = mask.astype(np.float32) / 255.0
    if feather > 0:
        k = int(feather) * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), feather / 2.0)
    col = np.array([int(c) for c in rgb], dtype=np.float32).reshape(1, 1, 3)
    out = img.astype(np.float32) * (1 - m[..., None]) + col * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def canny_edges(img: np.ndarray, thicken: bool = True) -> np.ndarray:
    """Bordi Canny automatici (soglie dalla mediana), linee bianche; +1 dilatazione 3x3 perché la condizione viene
    poi ridotta a 512 px. Sostituisce l'HED 'scribble' upstream senza pesi."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    v = float(np.median(g))
    lo, hi = int(max(0, 0.66 * v)), int(min(255, 1.33 * v))
    e = cv2.Canny(g, lo, max(hi, lo + 1))
    if thicken:
        e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
    return e


def hed_edges(img: np.ndarray) -> np.ndarray:
    """HED scribble (upstream `HEDDetector`): richiede ControlNetHED.pth + controlnet_aux. Mai scaricato in
    automatico: se manca, errore CHIARO con nome e dimensione."""
    path = os.environ.get("MQ2_HED_PATH") or next((p for p in HED_CANDIDATES if os.path.isfile(p)), "")
    if not path:
        raise MQError(f"edge_source='hed' richiede i pesi {HED_WEIGHTS_NAME} (~{HED_WEIGHTS_MB} MB, "
                      f"0,03 GB) che non sono presenti (cercati: {', '.join(HED_CANDIDATES)}; oppure variabile "
                      f"MQ2_HED_PATH). Usa edge_source='canny' (nessun peso).")
    try:
        import controlnet_aux  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise MQError(f"edge_source='hed': pesi trovati ({path}) ma il pacchetto 'controlnet_aux' non è installato "
                      f"nel venv immagini ({e}). Usa edge_source='canny'.") from e
    from controlnet_aux import HEDdetector                         # pragma: no cover (non verificabile qui)
    det = HEDdetector.from_pretrained(os.path.dirname(path))        # pragma: no cover
    out = det(Image.fromarray(img), safe=True, scribble=True, output_type="pil")   # pragma: no cover
    out = np.asarray(out.convert("L").resize((img.shape[1], img.shape[0]), Image.BILINEAR))   # pragma: no cover
    return out                                                      # pragma: no cover


def edge_control_image(img: np.ndarray, source: str = "canny", add: np.ndarray | None = None,
                       erase: np.ndarray | None = None) -> Image.Image:
    """Condizione 'edge': bordi automatici, poi si TOLGONO i tratti `erase` (-> 0) e si AGGIUNGONO i tratti `add`
    (-> 255) (stesso ordine di edge_edit: prima remove poi add). Ritorna RGB bianco su nero."""
    if source == "canny":
        lines = canny_edges(img)
    elif source == "hed":
        lines = hed_edges(img)
    else:
        raise MQError(f"edge_source '{source}' sconosciuto (canny|hed)")
    lines = lines.copy()
    if erase is not None:
        lines[erase > 127] = 0
    if add is not None:
        lines[add > 127] = 255
    return Image.fromarray(np.stack([lines] * 3, axis=-1))


# ---- prompt / composizione --------------------------------------------------------------------------------------
def build_prompt(mode: str, prompt: str, fix_perspective: bool = False) -> str:
    p = prompt or ""
    if mode == "removal":
        return PROMPT_REMOVAL                                   # app.py: il prompt utente è IGNORATO
    if mode == "foreground":
        p = PROMPT_FOREGROUND + p
        if fix_perspective:
            p = p + PROMPT_FIX_PERSPECTIVE                      # edit.py (aggiunto anche se app.py ha già la frase)
    return p


def max_sequence_length(mode: str) -> int:
    """Upstream: 128 token T5 per local/edge/color/foreground; `object_removal` NON passa max_sequence_length e
    ricade sul default della pipeline (512). Replicato per fedeltà."""
    return 512 if mode == "removal" else 128


def blend_original(result: Image.Image, orig: Image.Image, mask: np.ndarray, ksize: int = 10, sigma: float = 10.0):
    """Ricompone i pixel ORIGINALI fuori maschera (BlendInpaint upstream: gaussiana kernel `ksize`(+1 se pari),
    sigma). `mask` 255 = area rigenerata, a dimensione di `result`."""
    if ksize % 2 == 0:
        ksize += 1
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), sigma)
    r = np.asarray(result.convert("RGB"), dtype=np.float32)
    o = np.asarray(orig.convert("RGB").resize(result.size, Image.LANCZOS), dtype=np.float32)
    out = o * (1 - m[..., None]) + r * m[..., None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
