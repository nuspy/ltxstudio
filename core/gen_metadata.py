"""Metadati di generazione DENTRO i file immagine/video (task 21).

Scrive nel file i dati del sidecar (prompt, negative, modello, LoRA, encoder, VAE, seed, steps, parametri del
motore) in modo che li leggano gli altri programmi (Civitai, A1111, Forge, ExifTool) e che l'app possa
ripristinarli. Nessuna dipendenza da Qt/torch (solo stdlib + ffmpeg per i video): importabile anche dai venv
dei worker.

Formati
- PNG  : chunk di testo `parameters` (formato A1111) + chunk `ltxstudio` (JSON completo). Inserimento di chunk:
         i pixel (e l'eventuale canale alfa) NON vengono toccati né ricodificati.
- JPEG : EXIF UserComment (testo A1111) + XMP con il JSON. Inserimento di segmenti APP1: nessuna ricompressione.
- WebP : chunk RIFF EXIF + XMP (e VP8X se manca). Nessuna ricompressione.
- Video: metadati del contenitore (`comment`/`description` = testo A1111, `ltxstudio` = JSON) con
         `ffmpeg -c copy`: stream video e audio intatti.

ORDINE DELLE OPERAZIONI: i metadati vanno scritti DOPO l'ultima riscrittura dei pixel (la firma invisibile
`core/stego.py` rilegge e riscrive il file con OpenCV e li perderebbe). Per questo `embed()` si aggancia ai
punti di salvataggio finali dell'app (vedi `embed_result`, `MetadataSaver.save_sidecar`, i job REST).

Privacy: nei metadati finiscono solo nomi di file dei modelli e dei riferimenti, mai percorsi assoluti
dell'utente (`sanitize`). L'interruttore è `user_prefs["embed_gen_metadata"]` (default ON); variabile
d'ambiente `LTX_EMBED_METADATA=0` per spegnerlo in un singolo processo.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import zlib
from contextlib import contextmanager
from pathlib import Path

KEY_PARAMS = "parameters"
KEY_JSON = "ltxstudio"
PREF_KEY = "embed_gen_metadata"
APP_NAME = "LTX Studio"
XMP_NS = "https://ltxstudio.local/ns/generation/1.0/"

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VID_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}

_MAX_JSON_BYTES = 200_000            # oltre, il JSON viene alleggerito (stringhe lunghe tagliate)
_MAX_SEGMENT = 60_000                # limite pratico di un segmento APP1 JPEG (64 KB)

# ───────────────────────────── interruttore ─────────────────────────────


_TL = threading.local()


@contextmanager
def override(flag: bool | None):
    """Forza (True/False) l'interruttore per il thread corrente (es. un job REST con ?embed_metadata=false).
    None = nessun override."""
    prev = getattr(_TL, "override", None)
    _TL.override = flag
    try:
        yield
    finally:
        _TL.override = prev


def reset_last() -> None:
    _TL.last = None


def last_meta() -> dict | None:
    """Ultimi metadati incorporati dal thread (serve ai job REST a ripristinarli se un passo successivo, es. la
    firma invisibile, riscrive il file)."""
    return getattr(_TL, "last", None)


def enabled() -> bool:
    """True se i metadati vanno scritti: override del thread, env LTX_EMBED_METADATA (0/false = no), poi
    preferenza utente (default ON)."""
    ov = getattr(_TL, "override", None)
    if ov is not None:
        return bool(ov)
    env = os.environ.get("LTX_EMBED_METADATA", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        from core import user_prefs
        return bool(user_prefs.get(PREF_KEY, True))
    except Exception:                                        # noqa: BLE001
        return True


# ───────────────────────────── sanificazione ─────────────────────────────

_ABS_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|media|root|tmp|var|opt|srv|data|e|c|d)/)")
# chiavi il cui valore è testo libero dell'utente: mai toccate dalla sanificazione dei percorsi
_FREE_TEXT = {"prompt", "negative", "prompt_positive", "prompt_negative", "lyrics", "text", "tags", "gs_text",
              "desc", "caption", "label", "instruction"}
# chiavi che indicano un modello/peso: si tiene solo il nome file
_MODEL_KEYS = {"name", "model", "vae", "encoder", "clip", "te", "unet", "checkpoint", "msr_lora", "lora", "gguf_transformer"}
_DROP_KEYS = {"output_path", "output"}


def _basename(s: str) -> str:
    return re.split(r"[\\/]", s.rstrip("\\/"))[-1]


def _looks_abs(s: str) -> bool:
    return bool(_ABS_RE.match(s))


def _clean_value(key: str, v):
    k = (key or "").lower()
    if isinstance(v, str):
        if k in _FREE_TEXT:
            return v
        if k in _MODEL_KEYS or k.endswith(("_te", "_dit", "_vae", "_lora", "_model")):
            return _basename(v) if ("\\" in v or "/" in v) else v
        if _looks_abs(v):
            return _basename(v)
        return v
    if isinstance(v, dict):
        return {str(kk): _clean_value(str(kk), vv) for kk, vv in v.items() if not str(kk).startswith("_")}
    if isinstance(v, (list, tuple)):
        # le liste ereditano la chiave (ref_images → nomi file, loras → dict ripuliti)
        return [_clean_value(key if not isinstance(x, dict) else "", x) for x in v]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)


def sanitize(meta: dict) -> dict:
    """Copia JSON-safe del sidecar senza percorsi assoluti né chiavi interne (`_…`, output_path)."""
    out = {}
    for k, v in (meta or {}).items():
        ks = str(k)
        if ks.startswith("_") or ks in _DROP_KEYS:
            continue
        out[ks] = _clean_value(ks, v)
    # alleggerisce se enorme (non dovrebbe succedere: boxes/riferimenti sono piccoli)
    try:
        if len(json.dumps(out, ensure_ascii=True)) > _MAX_JSON_BYTES:
            for k, v in list(out.items()):
                if isinstance(v, str) and len(v) > 4000:
                    out[k] = v[:4000] + "…"
                elif isinstance(v, list) and len(json.dumps(v)) > 20000:
                    out[k] = v[:50]
    except Exception:                                        # noqa: BLE001
        pass
    return out


# ───────────────────────────── testo A1111 ─────────────────────────────

_TE_KEYS = ("encoder", "krea2_te", "qwen21_te", "flux3_te", "klein_te", "h3_te", "ltx_te", "te")
_DIT_KEYS = ("krea2_dit", "qwen21_dit", "flux3_dit", "h3_dit", "lingbot_variant")


def _stem(name: str) -> str:
    b = _basename(str(name))
    for ext in (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin"):
        if b.lower().endswith(ext):
            return b[: -len(ext)]
    return b


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    s = str(v)
    if "," in s or ":" in s or "\n" in s:                    # come A1111: valori con virgole tra virgolette
        return json.dumps(s, ensure_ascii=False)
    return s


def _loras(meta: dict) -> list[tuple[str, float]]:
    out = []
    for lo in meta.get("loras") or []:
        if isinstance(lo, dict) and lo.get("name"):
            try:
                st = float(lo.get("strength", 1.0))
            except (TypeError, ValueError):
                st = 1.0
            out.append((_stem(lo["name"]), st))
        elif isinstance(lo, str) and lo:
            out.append((_stem(lo), 1.0))
    return out


def _first(meta: dict, *keys):
    for k in keys:
        v = meta.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


def build_parameters(meta: dict, size: tuple[int, int] | None = None) -> str:
    """Testo nel formato di A1111 (letto da Civitai & co.): prompt / «Negative prompt:» / riga impostazioni."""
    m = sanitize(meta)
    prompt = str(_first(m, "prompt", "prompt_positive") or "")
    neg = str(_first(m, "negative", "prompt_negative") or "")
    loras = _loras(m)
    tags = " ".join(f"<lora:{n}:{s:g}>" for n, s in loras if f"<lora:{n}:" not in prompt)
    if tags:
        prompt = (prompt + " " + tags).strip()
    items: list[tuple[str, object]] = []
    add = lambda k, v: items.append((k, v)) if v not in (None, "", [], {}) else None   # noqa: E731
    add("Steps", _first(m, "steps"))
    add("Sampler", _first(m, "sampler"))
    add("Schedule type", _first(m, "scheduler"))
    add("CFG scale", _first(m, "guidance", "cfg"))
    add("Seed", m.get("seed") if m.get("seed") is not None else None)
    w, h = m.get("width"), m.get("height")
    if (not w or not h) and size:
        w, h = size
    if w and h:
        add("Size", f"{int(w)}x{int(h)}")
    add("Model hash", m.get("model_hash"))
    model = _first(m, "model")
    if not model:
        for k in _DIT_KEYS:
            if m.get(k):
                model = m[k]
                break
    add("Model", _stem(model) if model else None)
    add("Denoising strength", _first(m, "denoise", "strength"))
    add("VAE", _stem(m["vae"]) if m.get("vae") else None)
    for k in _TE_KEYS:
        if m.get(k):
            add("Text encoder", _stem(m[k]))
            break
    if loras:
        add("LoRA", ", ".join(f"{n}:{s:g}" for n, s in loras))
    add("Architecture", _first(m, "arch", "engine"))
    add("Frames", _first(m, "num_frames"))
    add("FPS", _first(m, "fps"))
    add("Release", _first(m, "ltx_release"))
    add("Mode", _first(m, "label"))
    items.append(("Version", APP_NAME))
    line = ", ".join(f"{k}: {_fmt(v)}" for k, v in items)
    parts = [prompt]
    if neg:
        parts.append(f"Negative prompt: {neg}")
    parts.append(line)
    return "\n".join(parts)


def build_json(meta: dict) -> str:
    """JSON completo del sidecar (sanificato), compatto e ASCII-safe (sta in un chunk tEXt latin-1)."""
    return json.dumps(sanitize(meta), ensure_ascii=True, separators=(",", ":"))


# ───────────────────────────── PNG ─────────────────────────────

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_TEXT = (b"tEXt", b"iTXt", b"zTXt")


def _png_iter(data: bytes):
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + ln]
        yield typ, body, data[pos:pos + 12 + ln]
        pos += 12 + ln
        if typ == b"IEND":
            break


def _png_chunk(typ: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + typ + body + struct.pack(">I", zlib.crc32(typ + body) & 0xFFFFFFFF)


def _png_text_chunk(key: str, text: str) -> bytes:
    kb = key.encode("latin-1")
    try:
        return _png_chunk(b"tEXt", kb + b"\x00" + text.encode("latin-1"))
    except UnicodeEncodeError:
        return _png_chunk(b"iTXt", kb + b"\x00\x00\x00\x00\x00" + text.encode("utf-8"))


def _png_key(typ: bytes, body: bytes) -> str:
    return body.split(b"\x00", 1)[0].decode("latin-1", "replace")


def _png_decode_text(typ: bytes, body: bytes) -> tuple[str, str] | None:
    try:
        key, rest = body.split(b"\x00", 1)
        if typ == b"tEXt":
            return key.decode("latin-1"), rest.decode("latin-1")
        if typ == b"zTXt":
            return key.decode("latin-1"), zlib.decompress(rest[1:]).decode("latin-1")
        if typ == b"iTXt":
            comp, meth = rest[0], rest[1]
            rest = rest[2:]
            _lang, rest = rest.split(b"\x00", 1)
            _tr, txt = rest.split(b"\x00", 1)
            if comp:
                txt = zlib.decompress(txt)
            return key.decode("latin-1"), txt.decode("utf-8", "replace")
    except Exception:                                        # noqa: BLE001
        return None
    return None


def _png_write(data: bytes, params_text: str | None, json_text: str | None, *, strip_all: bool = False) -> bytes:
    if data[:8] != _PNG_SIG:
        raise ValueError("non è un PNG")
    ours = {KEY_PARAMS, KEY_JSON}
    out = [_PNG_SIG]
    inserted = False
    new = b""
    if params_text is not None:
        new += _png_text_chunk(KEY_PARAMS, params_text)
    if json_text is not None:
        new += _png_text_chunk(KEY_JSON, json_text)
    for typ, body, raw in _png_iter(data):
        if typ in _PNG_TEXT and (strip_all or _png_key(typ, body) in ours):
            continue
        if strip_all and typ == b"eXIf":
            continue
        if typ == b"IDAT" and not inserted:
            out.append(new)
            inserted = True
        out.append(raw)
    if not inserted:
        raise ValueError("PNG senza IDAT")
    return b"".join(out)


def _png_read(data: bytes) -> dict:
    res = {}
    for typ, body, _ in _png_iter(data):
        if typ in _PNG_TEXT:
            kv = _png_decode_text(typ, body)
            if kv and kv[0] in (KEY_PARAMS, KEY_JSON):
                res[kv[0]] = kv[1]
    return res


# ───────────────────────────── JPEG ─────────────────────────────

_EXIF_HDR = b"Exif\x00\x00"
_XMP_HDR = b"http://ns.adobe.com/xap/1.0/\x00"


def _tiff_exif(user_comment: str) -> bytes:
    """TIFF/EXIF minimale big-endian: IFD0{Software, ExifIFD→{UserComment}}."""
    uc = b"UNICODE\x00" + user_comment.encode("utf-16-be")
    sw = (APP_NAME + "\x00").encode("ascii")
    # layout: header(8) | IFD0 (2 + 2*12 + 4) | software | pad | ExifIFD (2 + 12 + 4) | usercomment
    ifd0_off = 8
    ifd0_len = 2 + 2 * 12 + 4
    sw_off = ifd0_off + ifd0_len
    exif_off = sw_off + len(sw) + (len(sw) & 1)
    uc_off = exif_off + 2 + 12 + 4
    b = b"MM\x00\x2a" + struct.pack(">I", ifd0_off)
    b += struct.pack(">H", 2)
    b += struct.pack(">HHII", 0x0131, 2, len(sw), sw_off)                    # Software
    b += struct.pack(">HHII", 0x8769, 4, 1, exif_off)                        # ExifIFD pointer
    b += struct.pack(">I", 0)
    b += sw + (b"\x00" if len(sw) & 1 else b"")
    b += struct.pack(">H", 1)
    b += struct.pack(">HHII", 0x9286, 7, len(uc), uc_off)                    # UserComment (UNDEFINED)
    b += struct.pack(">I", 0)
    b += uc
    return b


def _tiff_user_comment(tiff: bytes) -> str | None:
    try:
        bo = ">" if tiff[:2] == b"MM" else "<"
        ifd0 = struct.unpack(bo + "I", tiff[4:8])[0]

        def entries(off):
            n = struct.unpack(bo + "H", tiff[off:off + 2])[0]
            for i in range(n):
                yield struct.unpack(bo + "HHII", tiff[off + 2 + i * 12: off + 14 + i * 12])
        exif_off = None
        for tag, typ, cnt, val in entries(ifd0):
            if tag == 0x8769:
                exif_off = val
        if exif_off is None:
            return None
        for tag, typ, cnt, val in entries(exif_off):
            if tag == 0x9286:
                raw = tiff[val:val + cnt] if cnt > 4 else struct.pack(bo + "I", val)[:cnt]
                pre, body = raw[:8], raw[8:]
                if pre.startswith(b"UNICODE"):
                    return body.decode("utf-16-be" if bo == ">" else "utf-16-le", "replace")
                return body.decode("utf-8", "replace")
    except Exception:                                        # noqa: BLE001
        return None
    return None


def _xmp(params_text: str, json_text: str) -> bytes:
    from xml.sax.saxutils import escape
    x = ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
         '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
         f'<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" '
         f'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:ltx="{XMP_NS}" xmp:CreatorTool="{APP_NAME}">'
         f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{escape(params_text)}</rdf:li></rdf:Alt></dc:description>'
         f'<ltx:generation>{escape(json_text)}</ltx:generation>'
         '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')
    return x.encode("utf-8")


def _xmp_json(xmp: bytes) -> str | None:
    from xml.sax.saxutils import unescape
    m = re.search(rb"<ltx:generation>(.*?)</ltx:generation>", xmp, re.S)
    return unescape(m.group(1).decode("utf-8", "replace")) if m else None


def _xmp_params(xmp: bytes) -> str | None:
    from xml.sax.saxutils import unescape
    m = re.search(rb"<rdf:li[^>]*>(.*?)</rdf:li>", xmp, re.S)
    return unescape(m.group(1).decode("utf-8", "replace")) if m else None


def _jpeg_segments(data: bytes):
    """(marker, raw_segment_bytes) fino a SOS incluso il resto come ('SCAN', bytes)."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("non è un JPEG")
    pos = 2
    n = len(data)
    while pos + 4 <= n:
        if data[pos] != 0xFF:
            raise ValueError("JPEG malformato")
        mk = data[pos + 1]
        if mk == 0xFF:
            pos += 1
            continue
        if mk == 0xDA or mk == 0xD9:                         # SOS / EOI: da qui in poi dati grezzi
            yield "SCAN", data[pos:]
            return
        ln = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        yield mk, data[pos:pos + 2 + ln]
        pos += 2 + ln
    yield "SCAN", data[pos:]


def _jpeg_write(data: bytes, params_text: str | None, json_text: str | None, *, strip_all: bool = False) -> bytes:
    segs = list(_jpeg_segments(data))
    kept = []
    for mk, raw in segs:
        if mk == 0xE1 and (raw[4:10] == _EXIF_HDR or raw[4:4 + len(_XMP_HDR)] == _XMP_HDR):
            continue
        kept.append((mk, raw))
    new = b""
    if not strip_all and params_text is not None and json_text is not None:
        exif = _EXIF_HDR + _tiff_exif(params_text[: _MAX_SEGMENT // 2 - 200])
        new += b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
        xmp = _XMP_HDR + _xmp(params_text, json_text)
        if len(xmp) + 2 <= 0xFFFF:                            # JSON troppo grande per un segmento: solo EXIF
            new += b"\xff\xe1" + struct.pack(">H", len(xmp) + 2) + xmp
    out = [b"\xff\xd8"]
    done = False
    for mk, raw in kept:
        if not done and mk != 0xE0:                           # dopo SOI e l'eventuale JFIF (APP0)
            out.append(new)
            done = True
        out.append(raw)
    if not done:
        out.append(new)
    return b"".join(out)


def _jpeg_read(data: bytes) -> dict:
    res = {}
    for mk, raw in _jpeg_segments(data):
        if mk != 0xE1:
            continue
        if raw[4:10] == _EXIF_HDR:
            t = _tiff_user_comment(raw[10:])
            if t:
                res[KEY_PARAMS] = t
        elif raw[4:4 + len(_XMP_HDR)] == _XMP_HDR:
            j = _xmp_json(raw[4 + len(_XMP_HDR):])
            if j:
                res[KEY_JSON] = j
    return res


# ───────────────────────────── WebP ─────────────────────────────


def _webp_chunks(data: bytes):
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("non è un WebP")
    pos = 12
    end = min(len(data), 8 + struct.unpack("<I", data[4:8])[0])
    while pos + 8 <= end:
        fcc = data[pos:pos + 4]
        ln = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        yield fcc, data[pos + 8:pos + 8 + ln]
        pos += 8 + ln + (ln & 1)


def _webp_chunk(fcc: bytes, body: bytes) -> bytes:
    return fcc + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) & 1 else b"")


def _webp_canvas(chunks: list[tuple[bytes, bytes]]) -> tuple[int, int, bool]:
    for fcc, body in chunks:
        if fcc == b"VP8 ":
            w, h = struct.unpack("<HH", body[6:10])
            return (w & 0x3FFF), (h & 0x3FFF), False
        if fcc == b"VP8L":
            bits = struct.unpack("<I", body[1:5])[0]
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1, bool((bits >> 28) & 1)
    raise ValueError("WebP senza immagine")


def _webp_write(data: bytes, params_text: str | None, json_text: str | None, *, strip_all: bool = False) -> bytes:
    chunks = [(f, b) for f, b in _webp_chunks(data) if f not in (b"EXIF", b"XMP ")]
    has_x = chunks and chunks[0][0] == b"VP8X"
    want = (not strip_all) and params_text is not None and json_text is not None
    extra = []
    if want:
        extra.append((b"EXIF", _tiff_exif(params_text[:30000])))
        extra.append((b"XMP ", _xmp(params_text, json_text)))
    if has_x:
        flags = chunks[0][1][0]
        flags &= ~(0x08 | 0x04)
        if want:
            flags |= 0x08 | 0x04
        chunks[0] = (b"VP8X", bytes([flags]) + chunks[0][1][1:])
    elif want:
        w, h, alpha = _webp_canvas(chunks)
        flags = (0x10 if alpha else 0) | 0x08 | 0x04
        vp8x = bytes([flags, 0, 0, 0]) + (w - 1).to_bytes(3, "little") + (h - 1).to_bytes(3, "little")
        chunks.insert(0, (b"VP8X", vp8x))
    body = b"WEBP" + b"".join(_webp_chunk(f, b) for f, b in chunks + extra)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _webp_read(data: bytes) -> dict:
    res = {}
    for fcc, body in _webp_chunks(data):
        if fcc == b"EXIF":
            t = _tiff_user_comment(body[6:] if body[:6] == _EXIF_HDR else body)
            if t:
                res[KEY_PARAMS] = t
        elif fcc == b"XMP ":
            j = _xmp_json(body)
            if j:
                res[KEY_JSON] = j
    return res


# ───────────────────────────── video (ffmpeg) ─────────────────────────────


def _ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg                                 # noqa: PLC0415
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                         # noqa: BLE001
        return None


def _ffprobe() -> str | None:
    p = shutil.which("ffprobe")
    if p:
        return p
    ff = _ffmpeg()
    if ff:
        cand = os.path.join(os.path.dirname(ff), "ffprobe" + (".exe" if os.name == "nt" else ""))
        if os.path.exists(cand):
            return cand
    return None


def _ffmeta_escape(s: str) -> str:
    return (s.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")
             .replace("\r", "").replace("\n", "\\\n"))


def _video_write(path: str, params_text: str | None, json_text: str | None, *, strip_all: bool = False) -> None:
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg non trovato")
    ext = os.path.splitext(path)[1].lower()
    tmpdir = tempfile.mkdtemp(prefix="genmeta_")
    try:
        tmp = os.path.join(tmpdir, "out" + ext)
        cmd = [ff, "-y", "-v", "error", "-i", path]
        if strip_all:
            cmd += ["-map", "0", "-c", "copy", "-map_metadata", "-1", "-ignore_unknown", tmp]
        else:
            meta = os.path.join(tmpdir, "meta.txt")
            with open(meta, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(";FFMETADATA1\n")
                fh.write(f"comment={_ffmeta_escape(params_text or '')}\n")
                fh.write(f"description={_ffmeta_escape(params_text or '')}\n")
                fh.write(f"{KEY_JSON}={_ffmeta_escape(json_text or '')}\n")
                fh.write(f"software={_ffmeta_escape(APP_NAME)}\n")
            cmd += ["-i", meta, "-map", "0", "-c", "copy", "-map_metadata", "1", "-ignore_unknown"]
            if ext in (".mp4", ".mov", ".m4v"):
                cmd += ["-movflags", "use_metadata_tags+faststart"]
            cmd += [tmp]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            raise RuntimeError(f"ffmpeg: {(r.stderr or '')[-300:]}")
        os.replace(tmp, path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _video_read(path: str) -> dict:
    fp = _ffprobe()
    if not fp:
        return {}
    r = subprocess.run([fp, "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
                       capture_output=True, text=True, timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        tags = (json.loads(r.stdout).get("format") or {}).get("tags") or {}
    except Exception:                                         # noqa: BLE001
        return {}
    tags = {k.lower(): v for k, v in tags.items()}
    res = {}
    if KEY_JSON in tags:
        res[KEY_JSON] = tags[KEY_JSON]
    if tags.get("comment") and KEY_JSON in tags:
        res[KEY_PARAMS] = tags["comment"]
    return res


# ───────────────────────────── API pubblica ─────────────────────────────


def _atomic_write(path: str, data: bytes) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".genmeta_", suffix=os.path.splitext(path)[1], dir=d)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _image_size(path: str, data: bytes) -> tuple[int, int] | None:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".png":
            return struct.unpack(">II", data[16:24])
        if ext == ".webp":
            w, h, _ = _webp_canvas([(f, b) for f, b in _webp_chunks(data)] if data[12:16] != b"VP8X"
                                   else [(f, b) for f, b in _webp_chunks(data) if f != b"VP8X"])
            return w, h
        if ext in (".jpg", ".jpeg"):
            for mk, raw in _jpeg_segments(data):
                if isinstance(mk, int) and mk in (0xC0, 0xC1, 0xC2):
                    h, w = struct.unpack(">HH", raw[5:9])
                    return w, h
    except Exception:                                        # noqa: BLE001
        return None
    return None


def read(path: str | os.PathLike) -> dict | None:
    """Legge i metadati scritti da `embed`: {'parameters': testo A1111, 'ltxstudio': dict} oppure None."""
    p = str(path)
    ext = os.path.splitext(p)[1].lower()
    try:
        if ext in IMG_EXT:
            data = Path(p).read_bytes()
            raw = (_png_read if ext == ".png" else _webp_read if ext == ".webp" else _jpeg_read)(data)
        elif ext in VID_EXT:
            raw = _video_read(p)
        else:
            return None
    except Exception:                                        # noqa: BLE001
        return None
    if not raw:
        return None
    out: dict = {}
    if KEY_PARAMS in raw:
        out[KEY_PARAMS] = raw[KEY_PARAMS]
    if KEY_JSON in raw:
        try:
            out[KEY_JSON] = json.loads(raw[KEY_JSON])
        except Exception:                                    # noqa: BLE001
            return None if not out else out
    return out or None


def has_metadata(path) -> bool:
    r = read(path)
    return bool(r and KEY_JSON in r)


def embed(path: str | os.PathLike, meta: dict, *, force: bool = True, respect_pref: bool = True) -> bool:
    """Scrive i metadati di generazione dentro il file. Best-effort: non solleva MAI.

    `force=False`: non riscrive se il file ha già i metadati (usato dagli agganci ridondanti).
    `respect_pref=False`: ignora l'interruttore (uso esplicito/test). Ritorna True se ha scritto.
    """
    p = str(path)
    try:
        ext = os.path.splitext(p)[1].lower()
        if ext not in IMG_EXT and ext not in VID_EXT:
            return False
        if respect_pref and not enabled():
            return False
        if not os.path.isfile(p):
            return False
        if not force and has_metadata(p):
            return False
        meta = dict(meta or {})
        meta.setdefault("output_file", os.path.basename(p))
        _TL.last = {k: v for k, v in meta.items() if k not in ("output_file", "output_kind")}
        js = build_json(meta)
        if ext in IMG_EXT:
            data = Path(p).read_bytes()
            params = build_parameters(meta, _image_size(p, data))
            fn = _png_write if ext == ".png" else _webp_write if ext == ".webp" else _jpeg_write
            _atomic_write(p, fn(data, params, js))
        else:
            _video_write(p, build_parameters(meta), js)
        return True
    except Exception as e:                                   # noqa: BLE001
        print(f"[gen_metadata] non scritti per {os.path.basename(p)}: {e}", file=sys.stderr, flush=True)
        return False


def strip(path: str | os.PathLike, dst: str | os.PathLike | None = None) -> str:
    """Rimuove i metadati di generazione (e tutti i chunk di testo/EXIF/XMP) — su una COPIA se `dst` è dato,
    altrimenti sul posto. Ritorna il path del file pulito. Solleva su errore (azione esplicita dell'utente)."""
    src = str(path)
    out = str(dst) if dst else src
    ext = os.path.splitext(src)[1].lower()
    if dst and os.path.abspath(src) != os.path.abspath(out):
        shutil.copyfile(src, out)
    if ext in IMG_EXT:
        data = Path(out).read_bytes()
        fn = _png_write if ext == ".png" else _webp_write if ext == ".webp" else _jpeg_write
        _atomic_write(out, fn(data, None, None, strip_all=True))
    elif ext in VID_EXT:
        _video_write(out, None, None, strip_all=True)
    else:
        raise ValueError(f"formato non supportato: {ext}")
    return out


# ── aggancio ai risultati dei job ({status, output:[…]}) ──────────────────────────────────────────────────


def embed_result(res, params: dict | None, *, label: str | None = None, force: bool = False) -> None:
    """Dopo che un motore ha scritto il file FINALE: incorpora i metadati in ogni output immagine/video del
    risultato. `params` = parametri del job (il sidecar ne è la copia). Best-effort."""
    try:
        if not isinstance(res, dict) or res.get("status") not in (None, "success"):
            return
        if not enabled():
            return
        outs = res.get("output") or []
        if isinstance(outs, str):
            outs = [outs]
        if not isinstance(outs, (list, tuple)):
            return
        meta = {k: v for k, v in (params or {}).items()
                if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
        if label:
            meta["label"] = label
        meta["output_kind"] = ""
        for o in outs:
            if isinstance(o, str) and os.path.splitext(o)[1].lower() in (IMG_EXT | VID_EXT):
                m = dict(meta)
                m["output_kind"] = os.path.splitext(o)[1].lstrip(".").lower()
                embed(o, m, force=force)
    except Exception:                                        # noqa: BLE001
        pass


def finalize_outputs(outputs, label: str) -> None:
    """Ultima rete di sicurezza di un job (REST): ogni output immagine/video che NON ha i metadati (perché un
    passo successivo - firma invisibile, watermark, mux - ha riscritto il file) li riceve ora, riusando gli
    ultimi parametri incorporati dal job. Best-effort."""
    try:
        if not enabled():
            return
        base = dict(last_meta() or {})
        base["label"] = base.get("label") or label
        for o in outputs or []:
            if isinstance(o, str) and os.path.splitext(o)[1].lower() in (IMG_EXT | VID_EXT):
                m = dict(base)
                m["output_kind"] = os.path.splitext(o)[1].lstrip(".").lower()
                embed(o, m, force=False)
    except Exception:                                        # noqa: BLE001
        pass
