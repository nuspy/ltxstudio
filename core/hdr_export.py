"""Uscita della modalita' HDR del tab «Modifica Video AI» (IC-LoRA HDR LTX-2.3, fase 1c).

La pipeline del pacchetto (`ltx_pipelines.hdr_ic_lora.HDRICLoraPipeline`) restituisce un tensore **float lineare**
[f,h,w,3] (decode LogC3 inverso, primarie Rec.709/sRGB, valori in [0, ~55]; 0.18 = grigio medio, 1.0 = bianco diffuso)
e lascia al chiamante «tonemapping e salvataggio EXR». Qui sta tutto il resto, SENZA torch (solo numpy + ffmpeg):

- `HdrWriter`: riceve frame lineari uno alla volta (streaming, i video lunghi non stanno in RAM) e scrive
    * l'**anteprima SDR** tonemappata (H.264 8 bit, BT.709): e' il file che il player della GUI sa mostrare;
      formula del nodo ufficiale `LTXVHDRDecodePostprocess` (esposizione in stop + Reinhard + OETF sRGB);
    * il **video HDR10**: HEVC 10 bit, BT.2020, PQ (SMPTE ST 2084), con i tag colore nel flusso;
      il bianco diffuso (lineare 1.0) sta a `white_nits` (default 203, BT.2408);
    * la **sequenza EXR** float lineare (compositing: DJV/Nuke/Resolve/Blender), primarie sRGB come l'`save_exr_tensor`
      del pacchetto (OpenImageIO se c'e', altrimenti OpenCV).
- `mux_audio`: rimette l'audio del sorgente su un file video (copia del video).
- `probe_color`: tag colore/bit depth di un file (ffprobe) per i test e per le verifiche dal vivo.

Scelte di prodotto (decise qui, motivate; da confermare — vedi report del task 4):
- default = HEVC 10 bit PQ + anteprima SDR; l'EXR e' opzionale (pesa: 1 file float per frame).
- nessun MaxCLL/MaxFALL nel flusso (opzionale nello standard; con lo streaming non si conoscono prima della
  codifica): master-display fisso P3-D65 1000 nit, il lettore usa PQ+BT.2020 per riconoscere l'HDR10.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np

# ── costanti colore ───────────────────────────────────────────────────────────────────────────────
REF_WHITE_NITS = 203.0                     # BT.2408: bianco diffuso HDR (lineare 1.0 = 203 cd/m²)
PQ_PEAK_NITS = 10000.0
FORMATS = ("pq", "exr", "both")            # HEVC PQ | solo EXR | entrambi (l'anteprima SDR c'e' sempre)
DEFAULT_FORMAT = "pq"
DEFAULT_EXPOSURE_EV = 0.0                  # stop dell'anteprima (non tocca HDR ed EXR)
MIN_EV, MAX_EV = -6.0, 6.0

# Rec.709 lineare -> Rec.2020 lineare (ITU-R BT.2087)
_M709_TO_2020 = np.array([[0.6274040, 0.3292820, 0.0433136],
                          [0.0690970, 0.9195400, 0.0113612],
                          [0.0163916, 0.0880132, 0.8955950]], dtype=np.float32)

# SMPTE ST 2084
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 4096.0 * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 4096.0 * 32.0
_PQ_C3 = 2392.0 / 4096.0 * 32.0

# master-display P3-D65, 1000 nit / 0.0001 nit (x265: primarie in 0.00002, luminanza in 0.0001 cd/m²)
_MASTER_DISPLAY = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"


def clamp_exposure(ev: float) -> float:
    return float(max(MIN_EV, min(MAX_EV, float(ev))))


def pq_oetf(y: np.ndarray) -> np.ndarray:
    """Luce lineare normalizzata (1.0 = 10000 nit) -> segnale PQ [0,1]."""
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    p = np.power(y, _PQ_M1)
    return np.power((_PQ_C1 + _PQ_C2 * p) / (1.0 + _PQ_C3 * p), _PQ_M2).astype(np.float32)


def pq_eotf(e: np.ndarray) -> np.ndarray:
    """Inversa di `pq_oetf` (per i test): segnale PQ [0,1] -> luce normalizzata (1.0 = 10000 nit)."""
    e = np.clip(e, 0.0, 1.0).astype(np.float64)
    p = np.power(e, 1.0 / _PQ_M2)
    num = np.maximum(p - _PQ_C1, 0.0)
    return np.power(num / (_PQ_C2 - _PQ_C3 * p), 1.0 / _PQ_M1).astype(np.float32)


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def to_pq_u16(frame: np.ndarray, white_nits: float = REF_WHITE_NITS) -> np.ndarray:
    """Frame lineare Rec.709 (h,w,3) float -> RGB uint16 PQ/BT.2020 (full range), pronto per `rgb48le`."""
    x = np.maximum(np.asarray(frame, dtype=np.float32), 0.0) * (float(white_nits) / PQ_PEAK_NITS)
    x = x @ _M709_TO_2020.T                       # primarie BT.2020 (nessun clip: PQ clippa a 10000 nit)
    e = pq_oetf(x)
    return np.clip(e * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


def tonemap_sdr_u8(frame: np.ndarray, exposure_ev: float = DEFAULT_EXPOSURE_EV) -> np.ndarray:
    """Anteprima SDR: esposizione (stop) -> Reinhard -> OETF sRGB, uint8 (h,w,3). Stessa formula del nodo ufficiale
    `LTXVHDRDecodePostprocess` (hdr.py): clamp [0,1e4], x*2^EV, x/(1+x), sRGB."""
    x = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 1.0e4) * float(2.0 ** clamp_exposure(exposure_ev))
    return np.clip(srgb_oetf(x / (1.0 + x)) * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ── ffmpeg ────────────────────────────────────────────────────────────────────────────────────────
def _ffmpeg() -> str:
    from core import video_ic_prep as prep
    return prep.ffmpeg_path()


def _ffprobe() -> str:
    from core import video_ic_prep as prep
    return prep.ffprobe_path()


class _Pipe:
    """Processo ffmpeg alimentato da stdin; lo stderr va su file temporaneo (nessun deadlock) e si legge se fallisce."""

    def __init__(self, args: list[str], out: str):
        self.out = out
        self._err = tempfile.TemporaryFile()
        self.p = subprocess.Popen([_ffmpeg(), "-v", "error", "-y", *args, out],
                                  stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err)
        self.n = 0

    def write(self, buf: bytes) -> None:
        try:
            self.p.stdin.write(buf)
        except (BrokenPipeError, OSError):
            raise RuntimeError(f"ffmpeg si e' chiuso scrivendo {self.out}: {self._stderr()}") from None
        self.n += 1

    def _stderr(self) -> str:
        try:
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace")[-500:]
        except Exception:                                   # noqa: BLE001
            return ""

    def close(self) -> None:
        try:
            self.p.stdin.close()
        except Exception:                                   # noqa: BLE001
            pass
        self.p.wait()
        err = self._stderr()
        try:
            self._err.close()
        except Exception:                                   # noqa: BLE001
            pass
        if self.p.returncode != 0 or not os.path.exists(self.out) or os.path.getsize(self.out) == 0:
            raise RuntimeError(f"encoder ffmpeg fallito per {self.out}: {err}")

    def kill(self) -> None:
        """Chiude lo stdin (ffmpeg finisce da solo su EOF) e, se non esce, termina l'INTERO albero di processi: con
        l'ffmpeg di scoop (shim) un semplice kill() ammazza lo shim e lascia vivo l'ffmpeg vero."""
        try:
            self.p.stdin.close()
        except Exception:                                   # noqa: BLE001
            pass
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.p.pid), "/T", "/F"], capture_output=True)
            else:
                self.p.kill()
            self.p.wait()
        try:
            self._err.close()
        except Exception:                                   # noqa: BLE001
            pass


def _args_sdr(w: int, h: int, fps: float, crf: int) -> list[str]:
    return ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            "-movflags", "+faststart", "-an"]


def _args_pq(w: int, h: int, fps: float, crf: int) -> list[str]:
    xp = ("hdr10-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
          f"master-display={_MASTER_DISPLAY}:range=limited")
    return ["-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
            "-vf", "scale=out_color_matrix=bt2020nc:out_range=tv,format=yuv420p10le",
            "-c:v", "libx265", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p10le",
            "-x265-params", xp,
            "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
            "-color_range", "tv", "-tag:v", "hvc1", "-movflags", "+faststart", "-an"]


def save_exr(frame: np.ndarray, path: str, half: bool = True) -> None:
    """Un frame lineare (h,w,3) -> EXR. OpenImageIO (primarie sRGB nei metadati, come il pacchetto) se disponibile,
    altrimenti OpenCV (OPENCV_IO_ENABLE_OPENEXR gia' impostato da core.exr_export)."""
    arr = np.ascontiguousarray(np.asarray(frame, dtype=np.float32))
    try:
        import OpenImageIO as oiio                          # type: ignore
        h, w = arr.shape[:2]
        spec = oiio.ImageSpec(w, h, 3, oiio.HALF if half else oiio.FLOAT)
        spec.channelnames = ("R", "G", "B")
        spec.attribute("compression", "zip")
        spec.attribute("chromaticities", "float[8]", (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290))
        spec.attribute("colorSpace", "sRGB")
        out = oiio.ImageOutput.create(path)
        if out is not None:
            try:
                if out.open(path, spec) and out.write_image(arr):
                    return
            finally:
                out.close()
    except ImportError:
        pass
    from core import exr_export  # noqa: F401  (imposta OPENCV_IO_ENABLE_OPENEXR prima di cv2)
    import cv2
    flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT]
    if not cv2.imwrite(path, np.ascontiguousarray(arr[..., ::-1]), flags):
        raise RuntimeError(f"scrittura EXR fallita: {path}")


@dataclass
class HdrStats:
    frames: int = 0
    peak: float = 0.0                 # massimo lineare osservato (1.0 = bianco diffuso)
    mean: float = 0.0                 # media lineare (somma/n, aggiornata a ogni frame)
    _sum: float = field(default=0.0, repr=False)

    def add(self, frame: np.ndarray) -> None:
        self.frames += 1
        self.peak = max(self.peak, float(frame.max()))
        self._sum += float(frame.mean())
        self.mean = self._sum / self.frames


class HdrWriter:
    """Scrittore in streaming dell'uscita HDR (vedi docstring del modulo). Frame = float lineare (h,w,3), anche in
    piu' blocchi (video lunghi a segmenti). `close()` ritorna i path prodotti; un errore a meta' lascia i file parziali
    (nessuna cancellazione) ma chiude i processi."""

    def __init__(self, preview: str, *, pq: str = "", exr_dir: str = "", width: int, height: int, fps: float,
                 exposure_ev: float = DEFAULT_EXPOSURE_EV, exr_half: bool = True,
                 white_nits: float = REF_WHITE_NITS, crf_sdr: int = 18, crf_hdr: int = 16):
        self.w, self.h = int(width) // 2 * 2, int(height) // 2 * 2         # 4:2:0 vuole dimensioni pari
        if self.w < 2 or self.h < 2:
            raise ValueError("risoluzione troppo piccola")
        self.fps = float(fps)
        self.exposure_ev = clamp_exposure(exposure_ev)
        self.white_nits = float(white_nits)
        self.exr_half = bool(exr_half)
        self.exr_dir = exr_dir
        self.preview, self.pq = preview, pq
        self.stats = HdrStats()
        self.exr_paths: list[str] = []
        self._sdr = _Pipe(_args_sdr(self.w, self.h, self.fps, crf_sdr), preview)
        self._pq = _Pipe(_args_pq(self.w, self.h, self.fps, crf_hdr), pq) if pq else None
        if exr_dir:
            os.makedirs(exr_dir, exist_ok=True)
        self._closed = False

    def write(self, frame) -> None:
        a = np.asarray(frame, dtype=np.float32)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"frame atteso (h,w,3), ricevuto {a.shape}")
        a = a[: self.h, : self.w]
        if a.shape[0] != self.h or a.shape[1] != self.w:
            raise ValueError(f"frame {a.shape[1]}x{a.shape[0]} diverso da {self.w}x{self.h}")
        a = np.nan_to_num(a, nan=0.0, posinf=1.0e4, neginf=0.0)
        self.stats.add(a)
        self._sdr.write(tonemap_sdr_u8(a, self.exposure_ev).tobytes())
        if self._pq is not None:
            self._pq.write(to_pq_u16(a, self.white_nits).tobytes())
        if self.exr_dir:
            p = os.path.join(self.exr_dir, f"frame_{len(self.exr_paths):05d}.exr")
            save_exr(a, p, self.exr_half)
            self.exr_paths.append(p)

    def close(self) -> dict:
        if self._closed:
            return self.result()
        self._closed = True
        self._sdr.close()
        if self._pq is not None:
            self._pq.close()
        return self.result()

    def abort(self) -> None:
        self._closed = True
        self._sdr.kill()
        if self._pq is not None:
            self._pq.kill()

    def result(self) -> dict:
        return {"preview": self.preview, "pq": self.pq, "exr_dir": self.exr_dir if self.exr_paths else "",
                "frames": self.stats.frames, "width": self.w, "height": self.h, "fps": self.fps,
                "peak_linear": round(self.stats.peak, 3), "mean_linear": round(self.stats.mean, 4),
                "exposure_ev": self.exposure_ev, "white_nits": self.white_nits, "exr_half": self.exr_half}

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.close()
        else:
            self.abort()


# ── audio ─────────────────────────────────────────────────────────────────────────────────────────
def mux_audio(video: str, audio_src: str) -> bool:
    """Rimette l'audio di `audio_src` su `video` SUL POSTO (copia del video, aac sull'audio, tag colore intatti).
    False (e video invariato) se il sorgente non ha audio o ffmpeg fallisce."""
    from core import video_ic_prep as prep
    if not (audio_src and os.path.exists(audio_src) and os.path.exists(video) and prep.has_audio(audio_src)):
        return False
    tmp = video[:-4] + "._mux.mp4"
    try:
        r = subprocess.run([_ffmpeg(), "-v", "error", "-y", "-i", video, "-i", audio_src, "-map", "0:v:0",
                            "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                            "-movflags", "+faststart", tmp],
                           capture_output=True, text=True, timeout=1800)
    except Exception:                                        # noqa: BLE001
        return False
    if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, video)
        return True
    try:
        os.remove(tmp)                                       # file temporaneo creato qui, mai dell'utente
    except OSError:
        pass
    return False


# ── verifica ──────────────────────────────────────────────────────────────────────────────────────
def probe_color(path: str) -> dict:
    """Tag colore e bit depth del primo flusso video (ffprobe): codec, pix_fmt, bits_per_raw_sample, color_space,
    color_transfer, color_primaries, color_range, side_data (mastering display)."""
    r = subprocess.run([_ffprobe(), "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
                        "-show_frames", "-show_streams", "-of", "json", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    d = json.loads(r.stdout or "{}")
    s = (d.get("streams") or [{}])[0]
    f = (d.get("frames") or [{}])[0]
    return {"codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt") or f.get("pix_fmt"),
            "bits": s.get("bits_per_raw_sample"), "color_space": s.get("color_space") or f.get("color_space"),
            "color_transfer": s.get("color_transfer") or f.get("color_transfer"),
            "color_primaries": s.get("color_primaries") or f.get("color_primaries"),
            "color_range": s.get("color_range") or f.get("color_range"),
            "side_data": [x.get("side_data_type") for x in (f.get("side_data_list") or [])],
            "width": s.get("width"), "height": s.get("height"), "nb_frames": s.get("nb_frames")}


def sidecar_path(output: str) -> str:
    """File JSON con i path/le statistiche dell'uscita HDR, accanto all'anteprima `output`."""
    return os.path.splitext(output)[0] + "_hdr.json"


def default_paths(output: str) -> tuple[str, str]:
    """(file HEVC PQ, cartella EXR) derivati dal path dell'anteprima SDR."""
    base = os.path.splitext(output)[0]
    return base + "_HDR_PQ.mp4", base + "_EXR"


def read_sidecar(output: str) -> dict:
    try:
        with open(sidecar_path(output), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def write_sidecar(output: str, info: dict) -> str:
    p = sidecar_path(output)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    return p


def extra_outputs(output: str) -> list[str]:
    """File/cartelle HDR aggiuntivi (HEVC PQ, cartella EXR) prodotti dal job, letti dal sidecar; solo quelli esistenti."""
    info = read_sidecar(output)
    res = []
    for k in ("pq", "exr_dir"):
        v = info.get(k)
        if v and os.path.exists(v):
            res.append(v)
    return res


def ffmpeg_ok() -> bool:
    """True se ffmpeg ha libx265 (serve al video HDR10)."""
    try:
        r = subprocess.run([_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30)
        return "libx265" in (r.stdout or "")
    except Exception:                                        # noqa: BLE001
        return False


# ── cucitura dei segmenti (clip lunghi) ───────────────────────────────────────────────────────────
class Stitcher:
    """Unisce in streaming i segmenti HDR (frame float lineari) con CROSSFADE LINEARE IN LUCE LINEARE sugli overlap.
    `segments` = [(start, count)] nell'ordine di `add` (da `core.video_ic_prep.plan_segments`). Ogni frame ha un peso
    (rampa lineare in entrata sull'overlap col precedente, in uscita su quello col successivo) e il frame finale e'
    la media pesata dei segmenti che lo coprono: regge anche segmenti con overlap su entrambi i lati. `write` riceve
    i frame (h,w,3) in ordine (tipicamente `HdrWriter.write`). In RAM restano solo i frame non ancora completi."""

    def __init__(self, write, segments):
        self.write = write
        self.segs = [(int(a), int(b)) for a, b in segments]
        if not self.segs:
            raise ValueError("nessun segmento")
        self.i = 0
        self.buf: dict = {}
        self.next_out = self.segs[0][0]
        self.written = 0

    def add(self, frames) -> None:
        if self.i >= len(self.segs):
            raise RuntimeError("piu' segmenti del previsto")
        start = self.segs[self.i][0]
        f = len(frames)
        prev_end = (self.segs[self.i - 1][0] + self.segs[self.i - 1][1]) if self.i else start
        o_in = max(0, prev_end - start)
        nxt = self.segs[self.i + 1][0] if self.i + 1 < len(self.segs) else start + f
        o_out = max(0, start + f - nxt)
        for j in range(f):
            wgt = 1.0
            if j < o_in:
                wgt *= (j + 1) / (o_in + 1)
            if j >= f - o_out:
                wgt *= (f - j) / (o_out + 1)
            fr = np.asarray(frames[j], dtype=np.float32) * wgt
            acc, ws = self.buf.get(start + j, (None, 0.0))
            self.buf[start + j] = (fr if acc is None else acc + fr, ws + wgt)
        while self.next_out < nxt:
            acc, ws = self.buf.pop(self.next_out)
            self.write(acc / ws)
            self.written += 1
            self.next_out += 1
        self.i += 1

    def finish(self) -> None:
        """Scrive gli eventuali frame rimasti (nessuno se tutti i segmenti sono stati aggiunti)."""
        for pos in sorted(self.buf):
            acc, ws = self.buf[pos]
            self.write(acc / ws)
            self.written += 1
        self.buf = {}
