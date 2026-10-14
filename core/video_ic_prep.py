"""Preparazione video per le IC-LoRA LTX (Fase 1 «Modifica Video AI»): ffmpeg + numpy, NIENTE GPU, NIENTE Qt.

Specifica: scratchpad orch02/f1/DESIGN.md §1.2 / §2.4 / §4.3. Usato dal worker GUI, dall'API REST e dal worker
di inferenza (`workers/ltx2_infer.py::_run_chunked_ic`, venv ltx2: servono solo numpy + PIL + ffmpeg).

Convenzioni
- frame LTX = 8n+1 (`snap_frames`); dimensioni multiple di 64 (`OutpaintLayout`).
- Tutti i file prodotti sono SENZA audio (`-an`): l'audio originale lo rimette la GUI (`_mux_audio_from`).
- Video intermedi: H.264 yuv420p crf 10 (guide temporanee: qualita' alta, non "visually lossless" archivistico).
- Gamma: `exponent` e' l'esponente applicato ai valori normalizzati 0..1 (out = in**exponent). Per outpaint di
  sorgenti scure la card usa gamma 2.0: PRIMA si schiarisce la sorgente (esponente 1/2.0), DOPO la generazione si
  riscurisce (esponente 2.0) con `inverse_gamma`.

API pubblica (firme)
    ffmpeg_path() -> str ; ffprobe_path() -> str
    probe(path, count_frames=False) -> VideoInfo
    has_audio(path) -> bool
    mean_luma(path, seconds=1.0) -> float                   # 0..1 sul primo secondo (rilevamento sorgente scura)
    looks_dark(path, threshold=DARK_LUMA_THRESHOLD) -> bool
    compute_outpaint_layout(src_w, src_h, *, aspect=None, margins=None, anchor="center",
                            max_long_edge=1280, multiple=64) -> OutpaintLayout
    build_outpaint_canvas(src, out, layout, gamma=1.0) -> str       # barre NERE pure (0,0,0)
    inverse_gamma(src, out, gamma) -> str
    paste_original(generated, original, out, layout, border=2) -> str
    replace_first_frame(src, image, out) -> str                      # restyle: frame 0 = immagine di stile
    plan_segments(total_frames, seg_frames, overlap=9) -> list[Segment]
    extract_segment(src, out, start, count) -> str
    split_video(src, out_dir, seg_frames, overlap=9, prefix="seg") -> list[SegmentFile]
    concat_segments(parts, out, overlaps, fps=None, crf=14) -> str   # crossfade lineare sugli overlap
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

DARK_LUMA_THRESHOLD = 0.16         # luma media (0..1) sotto cui la sorgente e' «scura» (~41/255)
MIN_BAR_FRACTION = 0.15            # barra nuova piu' stretta di questa frazione del canvas: il modello NON la riempie (live 14/10)
MAX_NEW_AREA = 0.60                # oltre, il modello inventa troppo (DESIGN §2.2: da validare dal vivo)
DEFAULT_OVERLAP = 9                # frame di overlap fra segmenti (>= 8: restauro, DESIGN §4.3)
OUTPAINT_FORMATS: dict[str, tuple[int, int]] = {
    "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9),
}
ANCHORS = ("center", "left", "right", "top", "bottom")

_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p", "-an"]


# ── ffmpeg / ffprobe ──────────────────────────────────────────────────────────────────────────────
def _tool(name: str) -> str:
    """Stessa risoluzione di core/audio_post.py (E:\\AI\\Tools\\ffmpeg-shared\\bin se contiene l'exe), poi PATH.
    Variabile d'ambiente LTX_FFMPEG_DIR per forzare una cartella."""
    for d in (os.environ.get("LTX_FFMPEG_DIR"), r"E:\AI\Tools\ffmpeg-shared\bin"):
        if d:
            p = Path(d) / (name + (".exe" if os.name == "nt" else ""))
            if p.exists():
                return str(p)
    return shutil.which(name) or name


def ffmpeg_path() -> str:
    return _tool("ffmpeg")


def ffprobe_path() -> str:
    return _tool("ffprobe")


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg fallito ({os.path.basename(cmd[0])}): {(r.stderr or '')[-600:]}")
    return r


def _ff(args: list[str], timeout: int = 1800) -> None:
    _run([ffmpeg_path(), "-v", "error", "-y", *args], timeout)


def _out_check(out: str) -> str:
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise RuntimeError(f"ffmpeg non ha prodotto {out}")
    return out


# ── probe ─────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    nb_frames: int          # 0 se ignoto (usa probe(count_frames=True))
    duration: float
    has_audio: bool
    codec: str
    pix_fmt: str


def _fps_of(s: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        v = s.get(key) or "0/0"
        try:
            a, b = v.split("/")
            if float(b) > 0 and float(a) > 0:
                return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            pass
    return 0.0


def probe(path: str, count_frames: bool = False) -> VideoInfo:
    """Dati del video via ffprobe. `count_frames=True` decodifica per contare i frame (esatto, piu' lento);
    altrimenti nb_frames viene da `nb_frames` del container o da durata*fps (0 se ignoto)."""
    cmd = [ffprobe_path(), "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count_frames:
        cmd.insert(3, "-count_frames")
    r = _run(cmd + [str(path)], 300)
    d = json.loads(r.stdout or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError(f"nessuna traccia video in {path}")
    audio = any(s.get("codec_type") == "audio" for s in d.get("streams", []))
    fps = _fps_of(v)
    dur = float(v.get("duration") or d.get("format", {}).get("duration") or 0.0)
    nb = 0
    for key in (("nb_read_frames", "nb_frames") if count_frames else ("nb_frames",)):
        try:
            nb = int(v.get(key) or 0)
        except ValueError:
            nb = 0
        if nb:
            break
    if not nb and fps and dur:
        nb = int(round(dur * fps))
    return VideoInfo(int(v["width"]), int(v["height"]), fps, nb, dur, audio,
                     str(v.get("codec_name", "")), str(v.get("pix_fmt", "")))


def has_audio(path: str) -> bool:
    """True se il file ha almeno una traccia audio."""
    r = _run([ffprobe_path(), "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
              "-of", "csv=p=0", str(path)], 120)
    return bool((r.stdout or "").strip())


def snap_frames(n: int) -> int:
    """Il piu' grande 8k+1 <= n (min 9)."""
    n = int(n)
    return 9 if n < 9 else ((n - 1) // 8) * 8 + 1


# ── lettura/scrittura frame (numpy via pipe rawvideo) ─────────────────────────────────────────────
def _read_frames(path: str, w: int, h: int) -> Iterator["np.ndarray"]:      # noqa: F821
    import numpy as np
    p = subprocess.Popen([ffmpeg_path(), "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        try:
            p.stdout.close()
        except Exception:                                    # noqa: BLE001
            pass
        p.kill()
        p.wait()


class _Writer:
    """Encoder H.264 alimentato da frame RGB uint8 (h,w,3). Context manager."""

    def __init__(self, out: str, w: int, h: int, fps: float, crf: int = 10):
        self.out, self.w, self.h = out, w, h
        self.n = 0
        self.p = subprocess.Popen(
            [ffmpeg_path(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", f"{fps:.6f}", "-i", "-", "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(out)],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, frame) -> None:
        self.p.stdin.write(frame.tobytes())
        self.n += 1

    def close(self) -> None:
        try:
            self.p.stdin.close()
        finally:
            self.p.wait()
        if self.p.returncode != 0 or not os.path.exists(self.out):
            raise RuntimeError(f"encoder ffmpeg fallito per {self.out}")

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.close()
        else:
            try:
                self.p.stdin.close()
            except Exception:                                # noqa: BLE001
                pass
            self.p.kill()
            self.p.wait()


# ── luminanza ─────────────────────────────────────────────────────────────────────────────────────
def mean_luma(path: str, seconds: float = 1.0) -> float:
    """Luma media (0..1, limited-range-agnostico: scala 0..255 di gray) dei primi `seconds` secondi."""
    import numpy as np
    r = subprocess.run([ffmpeg_path(), "-v", "error", "-t", f"{seconds:.3f}", "-i", str(path), "-vf",
                        "scale=64:36,format=gray", "-f", "rawvideo", "-"], capture_output=True, timeout=300)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"mean_luma: lettura fallita per {path}")
    return float(np.frombuffer(r.stdout, np.uint8).mean()) / 255.0


def looks_dark(path: str, threshold: float = DARK_LUMA_THRESHOLD) -> bool:
    """True se la luma media del primo secondo e' sotto soglia (suggerisce gamma outpaint; mai forzarlo)."""
    return mean_luma(path) < threshold


# ── gamma ─────────────────────────────────────────────────────────────────────────────────────────
def gamma_lut(exponent: float):
    """LUT 256 voci uint8 di out = in**exponent (numpy), per test e per usi in-process."""
    import numpy as np
    x = np.arange(256, dtype=np.float64) / 255.0
    return np.clip(np.round((x ** float(exponent)) * 255.0), 0, 255).astype(np.uint8)


def _gamma_filter(exponent: float) -> str:
    e = f"{float(exponent):.6f}"
    ch = ":".join(f"{c}=pow(val/255\\,{e})*255" for c in ("r", "g", "b"))
    return f"format=rgb24,lutrgb={ch},format=yuv420p"


def apply_gamma(src: str, out: str, exponent: float) -> str:
    """Applica out = in**exponent a ogni canale RGB (ffmpeg lutrgb)."""
    _ff(["-i", str(src), "-vf", _gamma_filter(exponent), *_ENC, str(out)])
    return _out_check(out)


def inverse_gamma(src: str, out: str, gamma: float) -> str:
    """Annulla il trucco gamma dell'outpaint: se la sorgente fu schiarita con `gamma` (esponente 1/gamma),
    il risultato generato si riscurisce con esponente `gamma`."""
    return apply_gamma(src, out, float(gamma))


# ── outpaint ──────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OutpaintLayout:
    canvas_w: int           # canvas finale (multipli di 64)
    canvas_h: int
    src_w: int              # sorgente scalato che sta DENTRO il canvas (mai ritagliato), pari
    src_h: int
    x: int                  # offset del sorgente nel canvas (pari)
    y: int
    scale: float            # fattore sorgente originale -> scalato
    new_area: float         # frazione di canvas da generare (0..1)
    clamped: bool           # il tetto max_long_edge ha ridotto il canvas

    @property
    def narrow_bar_fraction(self) -> float:
        """Frazione (sul proprio asse del canvas) della barra nuova PIU' STRETTA; 1.0 se non ci sono barre.
        Un asse con meno di 64 px totali di spazio nuovo e' solo slop dell'arrotondamento a multipli di 64
        (non una barra voluta) e non conta; barre <= 4 px neppure."""
        fr = []
        for lo, hi, dim in ((self.x, self.canvas_w - self.x - self.src_w, self.canvas_w),
                            (self.y, self.canvas_h - self.y - self.src_h, self.canvas_h)):
            if lo + hi <= 64:
                continue
            fr.extend(b / float(dim) for b in (lo, hi) if b > 4)
        return min(fr) if fr else 1.0

    @property
    def narrow_bar(self) -> bool:
        return self.narrow_bar_fraction < MIN_BAR_FRACTION

    @property
    def too_much_new_area(self) -> bool:
        return self.new_area > MAX_NEW_AREA

    @property
    def canvas(self) -> tuple[int, int]:
        return self.canvas_w, self.canvas_h


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def _even_up(v: float) -> int:
    return max(2, int(math.ceil(v / 2.0 - 1e-9)) * 2)


def _c_mult(v: float, m: int) -> int:
    """Arrotonda IN SU al multiplo di m (tolleranza 1e-6 contro il rumore in virgola mobile)."""
    return max(m, int(math.ceil(v / m - 1e-6)) * m)


def compute_outpaint_layout(src_w: int, src_h: int, *,
                            aspect: tuple[int, int] | str | None = None,
                            margins: tuple[int, int, int, int] | None = None,
                            anchor: str = "center", max_long_edge: int = 1280,
                            multiple: int = 64) -> OutpaintLayout:
    """Calcola canvas e posizione del sorgente.

    - `aspect` ("16:9" oppure (16, 9)): canvas con quell'aspetto che CONTIENE il sorgente (no crop), con
      sorgente ancorato (`anchor`: center|left|right|top|bottom; conta solo sull'asse con spazio libero).
    - `margins` = (sopra, destra, sotto, sinistra) in px del SORGENTE: il canvas e' sorgente+margini.
    Entrambi: il canvas e' arrotondato IN SU a multipli di `multiple`, il sorgente resta a scala 1.0 (mai
    rimpicciolito) salvo che il canvas superi `max_long_edge`: solo allora si riduce il canvas e il sorgente con lui.
    Il sorgente scalato sta sempre dentro. Esattamente uno fra
    `aspect` e `margins` va dato."""
    if (aspect is None) == (margins is None):
        raise ValueError("indica esattamente uno fra aspect e margins")
    if anchor not in ANCHORS:
        raise ValueError(f"anchor sconosciuto: {anchor!r} (validi: {ANCHORS})")
    sw0, sh0 = int(src_w), int(src_h)
    if sw0 < 2 or sh0 < 2:
        raise ValueError("sorgente troppo piccolo")
    mt = mr = mb = ml = 0
    if aspect is not None:
        aw, ah = OUTPAINT_FORMATS[aspect] if isinstance(aspect, str) else (int(aspect[0]), int(aspect[1]))
        if aw <= 0 or ah <= 0:
            raise ValueError("aspect non valido")
        a = aw / ah
        ideal_w = max(float(sw0), sh0 * a)
        ideal_h = ideal_w / a
    else:
        mt, mr, mb, ml = (max(0, int(v)) for v in margins)
        ideal_w, ideal_h = float(sw0 + ml + mr), float(sh0 + mt + mb)
    s_cap = min(1.0, max_long_edge / max(ideal_w, ideal_h))
    # canvas arrotondato IN SU a multipli di `multiple` (cosi' il sorgente non va rimpicciolito)
    cw = _c_mult(ideal_w * s_cap, multiple)
    ch = _c_mult(ideal_h * s_cap, multiple)
    while max(cw, ch) > max_long_edge and max(cw, ch) > multiple:      # arrotondamento oltre il tetto
        if cw >= ch:
            cw -= multiple
        else:
            ch -= multiple
    k = min(cw / sw0, ch / sh0, s_cap)                         # <= 1: scala solo se il tetto lo impone
    sw, sh = min(_even_up(sw0 * k), cw // 2 * 2), min(_even_up(sh0 * k), ch // 2 * 2)
    if aspect is not None:
        fx = {"left": 0.0, "right": 1.0}.get(anchor, 0.5)
        fy = {"top": 0.0, "bottom": 1.0}.get(anchor, 0.5)
        x, y = (cw - sw) * fx, (ch - sh) * fy
    else:
        x, y = ml * k, mt * k
    x = max(0, min(cw - sw, int(round(x / 2.0)) * 2))
    y = max(0, min(ch - sh, int(round(y / 2.0)) * 2))
    return OutpaintLayout(cw, ch, sw, sh, x, y, sw / sw0, 1.0 - (sw * sh) / float(cw * ch),
                          s_cap < 1.0)


def build_outpaint_canvas(src: str, out: str, layout: OutpaintLayout, gamma: float = 1.0) -> str:
    """Video guida dell'outpaint: il sorgente scalato sul canvas finale, il resto NERO PURO (0,0,0) — il nero e'
    il «sentinel» che l'IC-LoRA riempie. `gamma` > 1 (es. 2.0, sorgente scura) schiarisce il SOLO sorgente
    (esponente 1/gamma) prima del padding, cosi' le barre nere restano distinguibili dal contenuto;
    dopo la generazione si usa `inverse_gamma(..., gamma)`."""
    chain = [f"scale={layout.src_w}:{layout.src_h}:flags=lanczos"]
    if abs(float(gamma) - 1.0) > 1e-6:
        chain.append(_gamma_filter(1.0 / float(gamma)))
    chain.append(f"pad={layout.canvas_w}:{layout.canvas_h}:{layout.x}:{layout.y}:color=black")
    chain.append("format=yuv420p")
    _ff(["-i", str(src), "-vf", ",".join(chain), *_ENC, str(out)])
    return _out_check(out)


def paste_original(generated: str, original: str, out: str, layout: OutpaintLayout, border: int = 2) -> str:
    """Rimette i pixel ORIGINALI al centro (zona del sorgente) sul video generato di dimensione canvas.
    Un bordo interno di `border` px del sorgente e' tagliato per evitare la cucitura. La lunghezza segue il
    piu' corto dei due; senza audio."""
    b = max(0, int(border))
    iw, ih = layout.src_w - 2 * b, layout.src_h - 2 * b
    if iw < 2 or ih < 2:
        b, iw, ih = 0, layout.src_w, layout.src_h
    fc = (f"[1:v]scale={layout.src_w}:{layout.src_h}:flags=lanczos,crop={iw}:{ih}:{b}:{b}[o];"
          f"[0:v]scale={layout.canvas_w}:{layout.canvas_h}[g];"
          f"[g][o]overlay={layout.x + b}:{layout.y + b}:shortest=1,format=yuv420p[v]")
    _ff(["-i", str(generated), "-i", str(original), "-filter_complex", fc, "-map", "[v]", *_ENC, str(out)])
    return _out_check(out)


# ── restyle: frame 0 = immagine di stile ──────────────────────────────────────────────────────────
def _load_image_cover(image: str, w: int, h: int):
    """Immagine -> array (h,w,3) uint8, resize + center-crop (come video_preprocess della pipeline LTX)."""
    import numpy as np
    from PIL import Image
    im = Image.open(image).convert("RGB")
    s = max(w / im.width, h / im.height)
    nw, nh = max(w, int(round(im.width * s))), max(h, int(round(im.height * s)))
    im = im.resize((nw, nh), Image.LANCZOS)
    l, t = (nw - w) // 2, (nh - h) // 2
    return np.asarray(im.crop((l, t, l + w, t + h)), dtype=np.uint8)


def replace_first_frame(src: str, image: str, out: str) -> str:
    """Ricetta ReStyle IC-LoRA (card): il controllo e' il video sorgente con il FRAME 0 SOSTITUITO dall'immagine
    di stile ridimensionata (cover + center-crop alla risoluzione del video). Stesso numero di frame e fps."""
    info = probe(src)
    w, h = info.width, info.height
    img = _load_image_cover(image, w, h)
    n = 0
    with _Writer(out, w, h, info.fps or 25.0, crf=10) as wr:
        for i, fr in enumerate(_read_frames(src, w, h)):
            wr.write(img if i == 0 else fr)
            n += 1
    if n == 0:
        raise RuntimeError(f"nessun frame letto da {src}")
    return _out_check(out)


# ── segmentazione ─────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Segment:
    index: int
    start: int              # primo frame (indice nel video sorgente)
    count: int              # numero di frame (8n+1)

    @property
    def end(self) -> int:
        return self.start + self.count


@dataclass(frozen=True)
class SegmentFile:
    path: str
    segment: Segment


def plan_segments(total_frames: int, seg_frames: int, overlap: int = DEFAULT_OVERLAP) -> list[Segment]:
    """Piano di segmenti per un video di `total_frames` frame.

    - ogni segmento ha `seg_frames` frame (arrotondati giu' a 8k+1) tranne se il video e' piu' corto: un solo
      segmento di `snap_frames(total)` frame (gli eventuali 1-7 frame in coda non rappresentabili sono scartati);
    - fra segmenti consecutivi c'e' un overlap di almeno `overlap` frame (continuita' per il crossfade);
    - l'ULTIMO segmento e' spostato indietro in modo da finire esattamente sull'ultimo frame del video (quindi
      puo' sovrapporsi di piu': usare `overlaps_of`)."""
    total, seg = int(total_frames), snap_frames(seg_frames)
    if total < 9:
        raise ValueError("servono almeno 9 frame (LTX: 8n+1)")
    if total <= seg:
        return [Segment(0, 0, snap_frames(total))]
    ov = max(0, min(int(overlap), seg - 1))
    step = seg - ov
    segs: list[Segment] = []
    start = 0
    while start + seg < total:
        segs.append(Segment(len(segs), start, seg))
        start += step
    segs.append(Segment(len(segs), total - seg, seg))         # l'ultimo finisce sull'ultimo frame
    return segs


def overlaps_of(segments: Sequence[Segment]) -> list[int]:
    """Frame di overlap REALI fra segmenti consecutivi (len = n-1)."""
    return [max(0, a.end - b.start) for a, b in zip(segments, segments[1:])]


def extract_segment(src: str, out: str, start: int, count: int) -> str:
    """Estrae `count` frame da `start` (re-encode, frame-accurate, senza audio)."""
    vf = f"trim=start_frame={int(start)}:end_frame={int(start) + int(count)},setpts=PTS-STARTPTS"
    _ff(["-i", str(src), "-vf", vf, "-frames:v", str(int(count)), *_ENC, str(out)])
    return _out_check(out)


def split_video(src: str, out_dir: str, seg_frames: int, overlap: int = DEFAULT_OVERLAP,
                prefix: str = "seg") -> list[SegmentFile]:
    """Spezza `src` in segmenti (vedi `plan_segments`) dentro `out_dir`; ritorna i file in ordine con il loro
    Segment. Frame totali da `probe(count_frames=True)` (esatto)."""
    info = probe(src, count_frames=True)
    if info.nb_frames < 1:
        raise RuntimeError(f"impossibile contare i frame di {src}")
    os.makedirs(out_dir, exist_ok=True)
    res: list[SegmentFile] = []
    for sg in plan_segments(info.nb_frames, seg_frames, overlap):
        p = os.path.join(out_dir, f"{prefix}_{sg.index:03d}.mp4")
        extract_segment(src, p, sg.start, sg.count)
        res.append(SegmentFile(p, sg))
    return res


def concat_segments(parts: Sequence[str], out: str, overlaps: Sequence[int] | None = None,
                    fps: float | None = None, crf: int = 14) -> str:
    """Concatena i segmenti generati con CROSSFADE lineare sugli overlap (frame-accurate, in streaming).
    `overlaps[i]` = frame in comune fra parts[i] e parts[i+1] (da `overlaps_of(plan)`); None/0 = taglio netto.
    Tutti i segmenti devono avere la stessa risoluzione. Senza audio. Se un segmento ha meno frame del previsto
    l'overlap si adatta (nessun frame duplicato o perso in modo silenzioso oltre quello)."""
    parts = [str(p) for p in parts]
    if not parts:
        raise ValueError("nessun segmento")
    ov = list(overlaps) if overlaps is not None else [0] * (len(parts) - 1)
    if len(ov) != len(parts) - 1:
        raise ValueError("overlaps deve avere len(parts)-1 elementi")
    import numpy as np
    first = probe(parts[0])
    w, h = first.width, first.height
    for p in parts[1:]:
        i = probe(p)
        if (i.width, i.height) != (w, h):
            raise ValueError(f"segmenti con risoluzione diversa: {p} e' {i.width}x{i.height}, atteso {w}x{h}")
    rate = float(fps or first.fps or 25.0)
    carry: list = []
    with _Writer(out, w, h, rate, crf=crf) as wr:
        for i, part in enumerate(parts):
            o_in = len(carry)
            o_out = max(0, int(ov[i])) if i < len(parts) - 1 else 0
            tail: deque = deque()
            idx = 0
            for fr in _read_frames(part, w, h):
                if idx < o_in:
                    wgt = (idx + 1) / (o_in + 1)
                    mix = carry[idx].astype(np.float32) * (1.0 - wgt) + fr.astype(np.float32) * wgt
                    wr.write(np.clip(mix + 0.5, 0, 255).astype(np.uint8))
                else:
                    tail.append(fr.copy())
                    if len(tail) > o_out:
                        wr.write(tail.popleft())
                idx += 1
            carry = list(tail)
        if carry:                                            # non dovrebbe succedere (ultimo o_out=0)
            for fr in carry:
                wr.write(fr)
    return _out_check(out)
