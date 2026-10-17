"""Motion-track IC-LoRA (LTX-2.3): tracce di movimento -> video di controllo. Modulo SENZA Qt e SENZA torch (solo numpy).

Porta nativa dei due nodi ufficiali Lightricks (ComfyUI-LTXVideo, `sparse_tracks.py` + `web/js/sparse_track_editor.js`;
solo lettura del sorgente, nessun ComfyUI in esecuzione). La LoRA `ltx-2.3-22b-ic-lora-motion-track-control-ref0.5` e'
addestrata su QUESTO disegno, quindi colori, raggi, scia e interpolazione sono quelli del nodo, non un'imitazione:

- `LTXVSparseTrackEditor` -> `interpolate_spline` (Catmull-Rom con estremi duplicati; 1 punto = traccia ferma, 2 punti =
  segmento lineare) campionata in `points_to_sample` punti = numero di frame del video; coordinate in pixel
  dell'immagine mostrata nell'editor (nel workflow ufficiale: l'immagine gia' portata alla risoluzione di lavoro).
- `LTXVDrawTracks` -> `render_frames`: cerchi riempiti disegnati a una risoluzione di riferimento (lato corto 1080),
  scia di 50 frame con raggio 2..8 px e colore per eta' (blu -> verde -> giallo -> rosso), poi ridotti con bilinear
  (align_corners=False, senza antialias) e canali scambiati RGB -> BGR («to match IC-LoRA training data format»).

Il resto del modulo e' il MODELLO DATI dell'editor (`TrackEditorModel`: punti, tracce, annulla, JSON nel formato del nodo)
e `write_video` (mp4 quasi lossless per la pipeline). Il widget Qt e' in ui/widgets/track_editor.py.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

# costanti del nodo LTXVDrawTracks
MIN_RADIUS = 2
MAX_RADIUS = 8
MAX_TRAIL = 50
REF_SHORT_SIDE = 1080

# colori delle tracce nell'editor ufficiale (solo UI; non finiscono nel video di controllo)
SPLINE_COLORS = ("#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4", "#f97316", "#84cc16")

DEFAULT_SHORT_SIDE = 576          # 544 del workflow ufficiale non e' multiplo di 64: si usa il multiplo vicino
DEFAULT_FRAMES = 121              # workflow ufficiale
DEFAULT_FPS = 24


# ── interpolazione (porta di `_catmull_rom` / `_interpolate_spline`) ──────────────────────────────
def _catmull_rom(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    t2 = t * t
    t3 = t2 * t
    out = []
    for k in ("x", "y"):
        out.append(0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t
                          + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                          + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3))
    return out[0], out[1]


def _pt(p) -> dict:
    """Punto da dict {x,y} o coppia [x,y] -> dict float."""
    if isinstance(p, dict):
        return {"x": float(p["x"]), "y": float(p["y"])}
    return {"x": float(p[0]), "y": float(p[1])}


def interpolate_spline(control_points: list, num_samples: int) -> list[dict]:
    """Come `_interpolate_spline` del nodo: `num_samples` punti interi {x,y} (round Python = arrotondamento al pari,
    come il nodo). 0 punti -> []; 1 -> ferma; 2 -> lineare; >=3 -> Catmull-Rom con primo e ultimo duplicati."""
    cps = [_pt(p) for p in control_points]
    ns = max(2, int(num_samples))
    if not cps:
        return []
    if len(cps) == 1:
        p = cps[0]
        return [{"x": round(p["x"]), "y": round(p["y"])} for _ in range(ns)]
    if len(cps) == 2:
        a, b = cps
        return [{"x": round(a["x"] + (b["x"] - a["x"]) * i / (ns - 1)),
                 "y": round(a["y"] + (b["y"] - a["y"]) * i / (ns - 1))} for i in range(ns)]
    pts = [cps[0], *cps, cps[-1]]
    n_seg = len(pts) - 3
    out = []
    for i in range(ns):
        g_t = (i / (ns - 1)) * n_seg
        seg = min(int(g_t), n_seg - 1)
        x, y = _catmull_rom(pts[seg], pts[seg + 1], pts[seg + 2], pts[seg + 3], g_t - seg)
        out.append({"x": round(x), "y": round(y)})
    return out


def interpolate_all(splines: list, num_samples: int) -> list[list[dict]]:
    return [interpolate_spline(sp, num_samples) for sp in splines if len(sp)]


# ── parsing (porta di `_parse_tracks`) ────────────────────────────────────────────────────────────
def parse_tracks(raw) -> list[list[dict]]:
    """Come `_parse_tracks`: accetta stringa JSON o oggetto, liste annidate/avvolte, elementi stringa-JSON. Ritorna le
    tracce come liste di dict con almeno x,y (l'ordine di scoperta e' quello del nodo: pila LIFO)."""
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(parsed, list):
        parsed = [json.loads(it) if isinstance(it, str) else it for it in parsed]
    tracks: list[list[dict]] = []
    stack = [parsed]
    while stack:
        obj = stack.pop()
        if isinstance(obj, list) and len(obj) > 0:
            if isinstance(obj[0], dict) and "x" in obj[0] and "y" in obj[0]:
                tracks.append(obj)
            else:
                stack.extend(obj)
    return tracks


# ── rendering (porta di LTXVDrawTracks) ───────────────────────────────────────────────────────────
def age_colors(ratios: np.ndarray) -> np.ndarray:
    """`_age_color_batch`: rapporto d'eta' [0..1] -> RGB [0..1] (blu -> verde -> giallo -> rosso). float32 (N,3)."""
    r = np.asarray(ratios, dtype=np.float32)
    colors = np.zeros((r.shape[0], 3), dtype=np.float32)
    m1 = r <= 1 / 3
    tr1 = r[m1] * 3
    colors[m1, 1] = tr1
    colors[m1, 2] = 1 - tr1
    m2 = (r > 1 / 3) & (r <= 2 / 3)
    tr2 = (r[m2] - 1 / 3) * 3
    colors[m2, 0] = tr2
    colors[m2, 1] = 1
    m3 = r > 2 / 3
    tr3 = (r[m3] - 2 / 3) * 3
    colors[m3, 0] = 1
    colors[m3, 1] = 1 - tr3
    return colors


def render_resolution(width: int, height: int, reference_short_side: int = REF_SHORT_SIDE):
    """`_render_resolution`: risoluzione di disegno piu' alta con lo stesso aspetto -> (rw, rh, scale_x, scale_y)."""
    if height <= width:
        rw = int(width * reference_short_side / height)
        rh = reference_short_side
    else:
        rw = reference_short_side
        rh = int(height * reference_short_side / width)
    return rw, rh, rw / width, rh / height


def _bilinear_axis(n_in: int, n_out: int):
    """Indici e pesi di `F.interpolate(mode='bilinear', align_corners=False)` (senza antialias) su un asse."""
    scale = n_in / n_out
    src = np.maximum((np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    l1 = (src - i0).astype(np.float32)
    return i0, i1, 1.0 - l1, l1


def resize_bilinear(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """(H,W,C) float32 -> (out_h,out_w,C) float32, bilinear alla PyTorch (align_corners=False, niente antialias)."""
    h, w = frame.shape[:2]
    if (h, w) == (out_h, out_w):
        return frame
    r0, r1, rw0, rw1 = _bilinear_axis(h, out_h)
    tmp = frame[r0] * rw0[:, None, None] + frame[r1] * rw1[:, None, None]
    c0, c1, cw0, cw1 = _bilinear_axis(w, out_w)
    return tmp[:, c0] * cw0[None, :, None] + tmp[:, c1] * cw1[None, :, None]


_HALF_D = (2 * MAX_RADIUS + 3) // 2           # 9: il modello 19x19 del nodo
_OFFS = np.arange(2 * _HALF_D + 1, dtype=np.float32) - _HALF_D
_TEMPLATE_D2 = _OFFS[:, None] ** 2 + _OFFS[None, :] ** 2        # float32, come template_dist_sq


def _window_params(window: int):
    """Per una finestra di `window` posizioni (la piu' vecchia per prima): raggi float32, colori e maschere dei cerchi."""
    ages = np.arange(window - 1, -1, -1, dtype=np.float32)
    ratios = np.float32(1.0) - ages / np.float32(MAX_TRAIL)
    radii = np.float32(MIN_RADIUS) + np.float32(MAX_RADIUS - MIN_RADIUS) * ratios
    colors = age_colors(ratios)
    masks = _TEMPLATE_D2[None, :, :] <= (radii * radii)[:, None, None]
    return ages, colors, masks


def render_frames(tracks, width: int, height: int, num_frames: int | None = None,
                  progress: Callable[[int, int], None] | None = None) -> Iterator[np.ndarray]:
    """Genera i frame del video di controllo come float32 (height,width,3) in [0,1], gia' con i canali scambiati in BGR
    (come l'uscita di LTXVDrawTracks). `tracks` = tracce DENSE (una lista di {x,y} per frame) nello spazio
    `width`x`height`; `num_frames` default = la traccia piu' lunga. Senza tracce: frame neri (1 frame, come il nodo)."""
    dense_in = (isinstance(tracks, list) and tracks and all(isinstance(t, list) and t and isinstance(t[0], dict)
                                                            and "x" in t[0] and "y" in t[0] for t in tracks))
    parsed = tracks if dense_in else parse_tracks(tracks)
    if not parsed:
        yield np.zeros((height, width, 3), np.float32)
        return
    n_tracks = len(parsed)
    n_all = max(len(t) for t in parsed)
    nf = n_all if num_frames is None else int(num_frames)
    rw, rh, sx, sy = render_resolution(width, height)
    xy = np.zeros((n_tracks, n_all, 2), np.float32)
    vis = np.zeros((n_tracks, n_all), bool)
    for i, trk in enumerate(parsed):
        xy[i, :len(trk)] = np.array([[p["x"] * sx, p["y"] * sy] for p in trk], np.float32)
        vis[i, :len(trk)] = True
    cache: dict[int, tuple] = {}
    for t in range(nf):
        frame = np.zeros((rh, rw, 3), np.float32)
        tt = min(t, n_all - 1)                              # oltre la fine le tracce restano ferme sull'ultimo punto
        tau_min = max(0, tt - MAX_TRAIL)
        window = tt - tau_min + 1
        if window not in cache:
            cache[window] = _window_params(window)
        ages, colors, masks = cache[window]
        # ordine di sovrapposizione: la scia piu' vecchia per prima, la piu' recente vince (a pari eta': traccia con indice maggiore)
        for k in range(window):                              # k=0 e' la piu' vecchia (age maggiore)
            tk = tau_min + k
            for i in range(n_tracks):
                if not vis[i, tk]:
                    continue
                cx = int(np.rint(xy[i, tk, 0]))
                cy = int(np.rint(xy[i, tk, 1]))
                y0, x0 = cy - _HALF_D, cx - _HALF_D
                ya, yb = max(0, y0), min(rh, y0 + 2 * _HALF_D + 1)
                xa, xb = max(0, x0), min(rw, x0 + 2 * _HALF_D + 1)
                if ya >= yb or xa >= xb:
                    continue
                m = masks[k][ya - y0:yb - y0, xa - x0:xb - x0]
                frame[ya:yb, xa:xb][m] = colors[k]
        out = resize_bilinear(frame, height, width)
        yield np.ascontiguousarray(out[..., ::-1])           # RGB -> BGR
        if progress:
            progress(t + 1, nf)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)


def render_array(tracks, width: int, height: int, num_frames: int | None = None) -> np.ndarray:
    """Tutti i frame come array float32 (n,h,w,3): per test e anteprime piccole."""
    return np.stack(list(render_frames(tracks, width, height, num_frames)))


# ── video di controllo ────────────────────────────────────────────────────────────────────────────
def _ffmpeg() -> str:
    from core import video_ic_prep as prep
    return prep.ffmpeg_path()


def write_video(tracks, width: int, height: int, fps: float, out: str, num_frames: int | None = None,
                progress: Callable[[int, int], None] | None = None) -> str:
    """Scrive il video di controllo (mp4 H.264 4:4:4 lossless, senza audio). Se l'encoder 4:4:4 manca ripiega su 4:2:0 a
    qualita' alta. Ritorna `out`. RuntimeError se ffmpeg fallisce."""
    if width % 2 or height % 2:
        raise ValueError("larghezza e altezza devono essere pari")
    last_err = ""
    for enc in (["-c:v", "libx264", "-preset", "veryfast", "-crf", "0", "-pix_fmt", "yuv444p"],
                ["-c:v", "libx264", "-preset", "veryfast", "-crf", "2", "-pix_fmt", "yuv420p"]):
        p = subprocess.Popen([_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                              "-r", f"{float(fps):.6f}", "-i", "-", *enc, "-an", str(out)],
                             stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        n = 0
        try:
            for fr in render_frames(tracks, width, height, num_frames, progress):
                p.stdin.write(to_uint8(fr).tobytes())
                n += 1
            p.stdin.close()
            last_err = (p.stderr.read() or b"").decode("utf-8", "replace")[-400:]
            p.wait()
        except (BrokenPipeError, OSError):
            last_err = (p.stderr.read() or b"").decode("utf-8", "replace")[-400:]
            p.kill(); p.wait()
            continue
        if p.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0 and n > 0:
            return str(out)
    raise RuntimeError(f"ffmpeg non ha scritto il video delle tracce: {last_err}")


def extract_first_frame(video: str, out_png: str) -> str:
    """Primo fotogramma di un video come PNG (ffmpeg). RuntimeError se non riesce."""
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    r = subprocess.run([_ffmpeg(), "-v", "error", "-y", "-i", str(video), "-frames:v", "1", str(out_png)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not os.path.exists(out_png) or os.path.getsize(out_png) == 0:
        raise RuntimeError(f"primo fotogramma non estratto da {video}: {(r.stderr or '')[-300:]}")
    return str(out_png)


# ── dimensioni e frame ────────────────────────────────────────────────────────────────────────────
def _r64(x: float) -> int:
    return max(64, int(np.floor(x / 64.0 + 0.5)) * 64)


def default_size(img_w: int, img_h: int, short_side: int = DEFAULT_SHORT_SIDE, max_long: int = 1280,
                 multiple: int = 64) -> tuple[int, int]:
    """Risoluzione di lavoro dall'aspetto dell'immagine: lato corto `short_side`, entrambi i lati multipli di 64, tetto sul
    lato lungo. 16:9 -> 1024x576, 9:16 -> 576x1024, 1:1 -> 576x576."""
    img_w, img_h = max(1, int(img_w)), max(1, int(img_h))
    s = short_side / float(min(img_w, img_h))
    w, h = _r64(img_w * s), _r64(img_h * s)
    while max(w, h) > max_long and max(w, h) > multiple:
        if w >= h:
            w -= multiple
        else:
            h -= multiple
    return w, h


def frames_for_seconds(seconds: float, fps: float) -> int:
    """Numero di frame 8k+1 piu' vicino a `seconds`*`fps` (min 9): 10 s a 24 fps -> 241."""
    n = float(seconds) * float(fps)
    return max(9, 8 * int(round((n - 1) / 8.0)) + 1)


# ── modello dati dell'editor ──────────────────────────────────────────────────────────────────────
@dataclass
class TrackInput:
    """Tracce lette da JSON esterno. kind: 'control' (punti di controllo, formato `points_store`) | 'dense' (una posizione
    per frame, formato `coordinates`/uscita del nodo). `size` = (w,h) dello spazio delle coordinate se dichiarato."""
    kind: str
    tracks: list
    size: tuple[int, int] | None = None


def _flatten_tracks(raw) -> list[list[dict]]:
    """Come `parse_tracks` ma conserva l'ORDINE delle tracce (serve al caricamento nell'editor)."""
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    out: list[list[dict]] = []

    def walk(o):
        if isinstance(o, str):
            o = json.loads(o)
        if isinstance(o, list) and o:
            if isinstance(o[0], dict) and "x" in o[0] and "y" in o[0]:
                out.append(o)
            else:
                for it in o:
                    walk(it)
    walk(parsed)
    return out


def _norm_tracks(raw_tracks) -> list[list[dict]]:
    out = []
    for t in raw_tracks:
        pts = []
        for p in t:
            try:
                pts.append(_pt(p))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        if pts:
            out.append(pts)
    return out


def parse_track_input(raw, num_frames: int | None = None, fmt: str = "auto") -> TrackInput:
    """Legge le tracce da: stringa JSON o oggetto con chiavi `points_store` (punti di controllo) e/o `coordinates`
    (tracce dense), `size`/`width`+`height` opzionali; lista nuda di liste di {x,y}; workflow ComfyUI completo (nodo
    LTXVSparseTrackEditor: widgets_values = [points_store, coordinates, ...]). fmt: auto | control | dense.
    In 'auto' una lista nuda e' 'dense' solo se ogni traccia ha esattamente `num_frames` punti, altrimenti 'control'."""
    obj = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if isinstance(obj, str):                                  # JSON doppio (come i widget del nodo)
        obj = json.loads(obj)
    size = None
    ps = co = None
    if isinstance(obj, dict) and "nodes" in obj:              # workflow ComfyUI
        for n in obj["nodes"]:
            if n.get("type") == "LTXVSparseTrackEditor":
                wv = n.get("widgets_values") or []
                ps = json.loads(wv[0]) if len(wv) > 0 and isinstance(wv[0], str) else (wv[0] if wv else None)
                co = json.loads(wv[1]) if len(wv) > 1 and isinstance(wv[1], str) else (wv[1] if len(wv) > 1 else None)
                break
        else:
            raise ValueError("nel workflow non c'e' un nodo LTXVSparseTrackEditor")
    elif isinstance(obj, dict):
        ps, co = obj.get("points_store"), obj.get("coordinates")
        ps = json.loads(ps) if isinstance(ps, str) else ps
        co = json.loads(co) if isinstance(co, str) else co
        sz = obj.get("size")
        if isinstance(sz, (list, tuple)) and len(sz) == 2:
            size = (int(sz[0]), int(sz[1]))
        elif obj.get("width") and obj.get("height"):
            size = (int(obj["width"]), int(obj["height"]))
    else:
        ps = obj
        if fmt == "auto" and isinstance(obj, list):
            nt = [t for t in obj if isinstance(t, list) and t]
            if nt and num_frames and all(len(t) == int(num_frames) for t in nt):
                ps, co = None, obj
    if fmt == "dense" and ps is not None and co is None:
        ps, co = None, ps
    if fmt == "control" and ps is None:
        ps, co = co, None
    if ps and isinstance(ps, list) and ps and isinstance(ps[0], list) and fmt != "dense":
        tr = _norm_tracks(ps)
        if tr:
            return TrackInput("control", tr, size)
    if co:
        tr = _norm_tracks(_flatten_tracks(co))
        if tr:
            return TrackInput("dense", tr, size)
    raise ValueError("nessuna traccia valida nel JSON")


def scale_tracks(tracks: list, sx: float, sy: float) -> list[list[dict]]:
    return [[{"x": p["x"] * sx, "y": p["y"] * sy} for p in t] for t in tracks]


def dense_tracks(inp: TrackInput, num_frames: int, width: int, height: int) -> list[list[dict]]:
    """Tracce dense nello spazio `width`x`height` con esattamente `num_frames` punti (interpolazione come il nodo; le tracce
    gia' dense di lunghezza diversa vengono ricampionate come punti di controllo). Le coordinate sono riscalate dallo
    spazio dichiarato (`inp.size`) a quello di lavoro; senza `size` si assume gia' quello di lavoro."""
    tr = inp.tracks
    if inp.size and (inp.size[0] != width or inp.size[1] != height):
        tr = scale_tracks(tr, width / float(inp.size[0]), height / float(inp.size[1]))
    if inp.kind == "dense" and all(len(t) == num_frames for t in tr):
        return [[{"x": float(p["x"]), "y": float(p["y"])} for p in t] for t in tr]
    return interpolate_all(tr, num_frames)


class TrackEditorModel:
    """Stato dell'editor di tracce (senza Qt). Coordinate in pixel dello spazio di riferimento `width`x`height` (= l'immagine di
    sfondo alla risoluzione di lavoro). L'ordine dei punti di una traccia e' l'ordine nel TEMPO: il primo e' dove il soggetto
    sta al primo frame, l'ultimo dove sta alla fine."""

    def __init__(self, width: int = 1024, height: int = 576):
        self.width, self.height = int(width), int(height)
        self.splines: list[list[list[float]]] = []
        self.active = -1

    # -- interrogazioni
    def __len__(self) -> int:
        return len(self.splines)

    @property
    def n_points(self) -> int:
        return sum(len(s) for s in self.splines)

    def is_empty(self) -> bool:
        return self.n_points == 0

    def color(self, si: int) -> str:
        return SPLINE_COLORS[si % len(SPLINE_COLORS)]

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return min(max(0.0, float(x)), float(self.width)), min(max(0.0, float(y)), float(self.height))

    def control_points(self) -> list[list[dict]]:
        return [[{"x": p[0], "y": p[1]} for p in s] for s in self.splines if s]

    def dense(self, num_frames: int) -> list[list[dict]]:
        return interpolate_all(self.control_points(), num_frames)

    def hit_test(self, x: float, y: float, tol: float) -> tuple[int, int] | None:
        """(traccia, punto) del punto di controllo piu' vicino entro `tol` px (la traccia attiva ha la precedenza)."""
        order = list(range(len(self.splines)))
        if 0 <= self.active < len(order):
            order.remove(self.active); order.insert(0, self.active)
        best, bd = None, tol
        for si in order:
            for pi, p in enumerate(self.splines[si]):
                d = ((p[0] - x) ** 2 + (p[1] - y) ** 2) ** 0.5
                if d <= bd:
                    best, bd = (si, pi), d
        return best

    # -- modifiche
    def add_point(self, x: float, y: float) -> tuple[int, int]:
        """Aggiunge un punto in coda alla traccia corrente (la crea se non ce n'e'). Ritorna (traccia, indice)."""
        if not (0 <= self.active < len(self.splines)):
            self.splines.append([]); self.active = len(self.splines) - 1
        x, y = self.clamp(x, y)
        self.splines[self.active].append([x, y])
        return self.active, len(self.splines[self.active]) - 1

    def new_track(self) -> int:
        """Chiude la traccia corrente e ne apre una vuota (se l'attuale e' gia' vuota non ne crea un'altra)."""
        if 0 <= self.active < len(self.splines) and not self.splines[self.active]:
            return self.active
        self.splines.append([]); self.active = len(self.splines) - 1
        return self.active

    def move_point(self, si: int, pi: int, x: float, y: float) -> None:
        self.splines[si][pi] = list(self.clamp(x, y))
        self.active = si

    def nudge_point(self, si: int, pi: int, dx: float, dy: float) -> None:
        p = self.splines[si][pi]
        self.move_point(si, pi, p[0] + dx, p[1] + dy)

    def undo_last_point(self) -> bool:
        """Toglie l'ultimo punto aggiunto alla traccia corrente (o alla piu' recente non vuota); una traccia che resta vuota
        sparisce. False se non c'e' nulla da annullare."""
        si = self.active if 0 <= self.active < len(self.splines) and self.splines[self.active] else None
        if si is None:
            for j in range(len(self.splines) - 1, -1, -1):
                if self.splines[j]:
                    si = j; break
        if si is None:
            return False
        self.splines[si].pop()
        if not self.splines[si]:
            del self.splines[si]
        self.active = min(self.active if self.active >= 0 else len(self.splines) - 1, len(self.splines) - 1)
        if self.active < 0 and self.splines:
            self.active = len(self.splines) - 1
        return True

    def delete_point(self, si: int, pi: int) -> None:
        del self.splines[si][pi]
        if not self.splines[si]:
            del self.splines[si]
        self.active = min(max(self.active, -1), len(self.splines) - 1)

    def delete_active(self) -> bool:
        if not (0 <= self.active < len(self.splines)):
            return False
        del self.splines[self.active]
        self.active = len(self.splines) - 1
        return True

    def clear(self) -> None:
        self.splines = []; self.active = -1

    def select(self, si: int) -> None:
        if 0 <= si < len(self.splines):
            self.active = si

    def set_size(self, width: int, height: int, rescale: bool = True) -> None:
        """Cambia lo spazio di riferimento; con `rescale` le tracce seguono l'immagine (proporzionalmente)."""
        width, height = int(width), int(height)
        if rescale and self.width and self.height and (width, height) != (self.width, self.height):
            sx, sy = width / float(self.width), height / float(self.height)
            self.splines = [[[p[0] * sx, p[1] * sy] for p in s] for s in self.splines]
        self.width, self.height = width, height

    # -- JSON nel formato del nodo
    def to_json(self, num_frames: int = DEFAULT_FRAMES) -> str:
        """JSON con le due chiavi dei widget del nodo (`points_store`, `coordinates`) + `size`: riusabile nel nodo ufficiale e
        riletto da `load_json`."""
        return json.dumps({"points_store": self.control_points(), "coordinates": self.dense(num_frames),
                           "size": [self.width, self.height], "num_frames": int(num_frames)}, indent=1)

    def load(self, raw, num_frames: int | None = None) -> int:
        """Carica tracce da JSON (qualunque formato di `parse_track_input`); riscala nello spazio dell'editor se il JSON
        dichiara `size`. Le tracce 'dense' diventano punti di controllo (decimate a ~12 punti per traccia: modificabili).
        Ritorna il numero di tracce caricate; ValueError se il JSON non ha tracce."""
        inp = parse_track_input(raw, num_frames)
        tr = inp.tracks
        if inp.size and (inp.size[0] != self.width or inp.size[1] != self.height):
            tr = scale_tracks(tr, self.width / float(inp.size[0]), self.height / float(inp.size[1]))
        if inp.kind == "dense":
            tr = [_decimate(t, 12) for t in tr]
        self.splines = [[[p["x"], p["y"]] for p in t] for t in tr]
        self.active = len(self.splines) - 1
        return len(self.splines)


def _decimate(track: list[dict], max_points: int) -> list[dict]:
    """Riduce una traccia densa a al piu' `max_points` punti di controllo equispaziati (primo e ultimo inclusi); una traccia
    ferma (tutti i punti uguali) diventa un punto solo."""
    if all(p["x"] == track[0]["x"] and p["y"] == track[0]["y"] for p in track):
        return [track[0]]
    if len(track) <= max_points:
        return list(track)
    idx = np.unique(np.round(np.linspace(0, len(track) - 1, max_points)).astype(int))
    return [track[i] for i in idx]
