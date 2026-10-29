"""core/gen_metadata.py — metadati di generazione dentro PNG/JPEG/WebP/MP4 (task 21).
Nessuna GPU, nessun Qt: numpy + OpenCV + Pillow + ffmpeg."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from core import gen_metadata as G
from core import stego

_FF = shutil.which("ffmpeg")
needs_ff = pytest.mark.skipif(not _FF, reason="ffmpeg non disponibile")

PERSONAL = "atain"      # nessun frammento del profilo/percorso dell'utente nei metadati

META = {
    "label": "Immagine", "arch": "flux", "prompt": "a lighthouse at dusk, <b>très</b> détaillé & \"quoted\" ☀",
    "negative": "blurry, watermark", "steps": 28, "guidance": 3.5, "seed": 1234567, "width": 96, "height": 64,
    "model": "E:\\AI\\Models\\diffusion_models\\Flux\\FLUX_Krea_Dev.safetensors",
    "vae": "ae.safetensors", "encoder": "t5xxl_fp8.safetensors", "sampler": "euler", "scheduler": "simple",
    "denoise": 0.8,
    "loras": [{"name": "Krea2\\facialExpressions_v1.safetensors", "strength": 0.8}],
    "image_path": "C:\\Users\\atain\\Pictures\\src.png", "ref_images": ["E:\\ai\\ltxstudio\\output\\ref1.png"],
    "boxes": [{"id": "hat_1", "desc": "a straw hat", "bbox": [0.1, 0.2, 0.3, 0.4]}],
    "output_path": "E:\\AI\\ltxstudio\\output\\x.png", "_t_button": 1793257358.6,
}


def _noise_img(path, mode="RGB", size=(96, 64), fmt=None, **kw):
    rng = np.random.default_rng(7)
    arr = (rng.random((size[1], size[0], len(mode))) * 255).astype(np.uint8)
    Image.fromarray(arr, mode).save(path, fmt, **kw)
    return path


def _pixels(path):
    im = Image.open(path)
    return im.mode, np.asarray(im).copy()


def _check_content(r):
    assert r is not None
    j = r[G.KEY_JSON]
    assert j["seed"] == 1234567 and j["steps"] == 28 and j["arch"] == "flux"
    assert j["boxes"][0]["id"] == "hat_1"
    txt = r[G.KEY_PARAMS]
    lines = txt.split("\n")
    assert lines[0].startswith("a lighthouse at dusk")
    assert "<lora:facialExpressions_v1:0.8>" in lines[0]
    assert lines[1] == "Negative prompt: blurry, watermark"
    last = lines[-1]
    for frag in ("Steps: 28", "Sampler: euler", "CFG scale: 3.5", "Seed: 1234567", "Size: 96x64",
                 "Model: FLUX_Krea_Dev", "VAE: ae", "Text encoder: t5xxl_fp8", 'LoRA: "facialExpressions_v1:0.8"',
                 "Denoising strength: 0.8", "Version: LTX Studio"):
        assert frag in last, frag


def _no_personal(path):
    raw = open(path, "rb").read()
    for bad in (b"C:\\\\Users", b"C:\\Users", b"E:\\\\AI", b"E:\\AI", b"E:\\\\ai", b"atain", b"\\\\Models",
                "C:/Users".encode(), b"output_path"):
        assert bad not in raw, bad


def test_sanitize_no_paths():
    s = G.sanitize(META)
    flat = json.dumps(s)
    assert "E:" not in flat and "C:" not in flat and "atain" not in flat and "_t_button" not in flat
    assert s["model"] == "FLUX_Krea_Dev.safetensors"
    assert s["image_path"] == "src.png" and s["ref_images"] == ["ref1.png"]
    assert s["loras"][0]["name"] == "facialExpressions_v1.safetensors"
    assert "output_path" not in s
    assert "\u2600" in s["prompt"]                            # testo libero dell'utente intatto


def test_parameters_text_minimal():
    t = G.build_parameters({"prompt": "x", "seed": 5}, size=(10, 20))
    assert t.split("\n")[0] == "x" and "Size: 10x20" in t and "Negative prompt" not in t


@pytest.mark.parametrize("ext,fmt,kw", [(".png", None, {}), (".jpg", "JPEG", {"quality": 90}),
                                         (".webp", "WEBP", {"lossless": True}), (".webp", "WEBP", {"quality": 80})])
def test_roundtrip_image(tmp_path, ext, fmt, kw):
    p = str(tmp_path / ("a" + ext))
    _noise_img(p, fmt=fmt, **kw)
    before = _pixels(p)
    assert G.read(p) is None
    assert G.embed(p, META, respect_pref=False)
    _check_content(G.read(p))
    _no_personal(p)
    after = _pixels(p)
    assert after[0] == before[0] and np.array_equal(after[1], before[1])      # nessuna ricodifica dei pixel
    assert G.has_metadata(p)
    # idempotente: seconda scrittura sostituisce, non duplica
    meta2 = dict(META, seed=99)
    assert G.embed(p, meta2, respect_pref=False)
    assert G.read(p)[G.KEY_JSON]["seed"] == 99
    assert not G.embed(p, META, force=False, respect_pref=False)               # già presente → non riscrive
    assert G.read(p)[G.KEY_JSON]["seed"] == 99


def test_png_pillow_and_a1111_reader(tmp_path):
    """Come lo legge Civitai/A1111: PIL.Image.info['parameters']."""
    p = str(tmp_path / "a.png")
    _noise_img(p)
    G.embed(p, META, respect_pref=False)
    info = Image.open(p).info
    assert info["parameters"].split("\n")[1].startswith("Negative prompt:")
    assert json.loads(info["ltxstudio"])["seed"] == 1234567
    Image.open(p).load()                                      # CRC e struttura validi


def test_png_unicode_prompt_itxt(tmp_path):
    p = str(tmp_path / "u.png")
    _noise_img(p)
    G.embed(p, {"prompt": "città ☀ 日本", "seed": 1}, respect_pref=False)
    assert Image.open(p).info["parameters"].startswith("città ☀ 日本")
    assert G.read(p)[G.KEY_PARAMS].startswith("città ☀ 日本")


def test_jpeg_exif_usercomment_pillow(tmp_path):
    p = str(tmp_path / "a.jpg")
    _noise_img(p, fmt="JPEG")
    G.embed(p, META, respect_pref=False)
    ex = Image.open(p).getexif()
    uc = ex.get_ifd(0x8769)[0x9286]
    assert uc.startswith(b"UNICODE\x00") and "Seed: 1234567" in uc[8:].decode("utf-16-be")
    assert ex[0x0131] == "LTX Studio"


def test_webp_alpha_flag_and_exif(tmp_path):
    p = str(tmp_path / "a.webp")
    _noise_img(p, mode="RGBA", fmt="WEBP", lossless=True)
    before = _pixels(p)
    G.embed(p, META, respect_pref=False)
    after = _pixels(p)
    assert after[0] == "RGBA" and np.array_equal(after[1], before[1])
    assert Image.open(p).getexif().get_ifd(0x8769)[0x9286].startswith(b"UNICODE")


def test_strip(tmp_path):
    for ext, fmt in ((".png", None), (".jpg", "JPEG"), (".webp", "WEBP")):
        p = str(tmp_path / ("a" + ext))
        _noise_img(p, fmt=fmt)
        G.embed(p, META, respect_pref=False)
        out = G.strip(p, str(tmp_path / ("clean" + ext)))
        assert G.read(out) is None and G.read(p) is not None
        raw = open(out, "rb").read()
        assert b"lighthouse" not in raw and b"ltxstudio" not in raw


# ── firma invisibile e canale alfa ─────────────────────────────────────────────────────────────────────────


def _smooth_img(path, size=(320, 320)):
    y, x = np.mgrid[0:size[1], 0:size[0]]
    arr = np.stack([x * 255 // size[0], y * 255 // size[1], (x + y) * 255 // (size[0] + size[1])], -1).astype(np.uint8)
    rng = np.random.default_rng(3)
    arr = np.clip(arr.astype(int) + rng.integers(-12, 12, arr.shape), 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def test_embed_after_stego_keeps_signature_and_metadata(tmp_path):
    p = str(tmp_path / "s.png")
    _smooth_img(p)
    stego.sign_file(p)
    z0 = stego.detect_file(p)
    assert z0[0], z0
    px_before = _pixels(p)
    assert G.embed(p, META, respect_pref=False)
    assert np.array_equal(_pixels(p)[1], px_before[1])        # pixel bit-identici → la firma è intatta
    assert stego.detect_file(p)[0]
    _check_content(G.read(p))


def test_stego_after_embed_loses_metadata_so_order_matters(tmp_path):
    """Documenta perché i metadati vanno scritti per ULTIMI: la firma (OpenCV) riscrive il file."""
    p = str(tmp_path / "s2.png")
    _smooth_img(p)
    G.embed(p, META, respect_pref=False)
    stego.sign_file(p)
    assert G.read(p) is None
    assert G.embed(p, META, respect_pref=False)               # riscrittura finale ripristina
    assert stego.detect_file(p)[0]


def test_rgba_alpha_untouched(tmp_path):
    p = str(tmp_path / "q.png")
    rng = np.random.default_rng(1)
    arr = (rng.random((64, 96, 4)) * 255).astype(np.uint8)
    arr[:32, :, 3] = 0                                       # alfa non banale (trasparente + opaco + intermedio)
    Image.fromarray(arr, "RGBA").save(p)
    G.embed(p, META, respect_pref=False)
    im = Image.open(p)
    assert im.mode == "RGBA" and np.array_equal(np.asarray(im), arr)
    _check_content(G.read(p))


def test_pref_off_writes_nothing(tmp_path, monkeypatch):
    p = str(tmp_path / "o.png")
    _noise_img(p)
    monkeypatch.setenv("LTX_EMBED_METADATA", "0")
    assert not G.embed(p, META)
    assert G.read(p) is None
    monkeypatch.setenv("LTX_EMBED_METADATA", "1")
    assert G.embed(p, META)


# ── video ────────────────────────────────────────────────────────────────────────────────────────────────────


def _mk_video(path, audio=True):
    cmd = [_FF, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=128x96:rate=24:duration=1"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac", "-shortest"] if audio else []) + [path]
    subprocess.run(cmd, check=True)


def _stream_md5(path, sel):
    r = subprocess.run([_FF, "-v", "error", "-i", path, "-map", sel, "-c", "copy", "-f", "md5", "-"],
                       capture_output=True, text=True)
    return r.stdout.strip()


@needs_ff
@pytest.mark.parametrize("ext", [".mp4", ".mov", ".mkv"])
def test_video_roundtrip_copy(tmp_path, ext):
    p = str(tmp_path / ("v" + ext))
    _mk_video(p)
    v0, a0 = _stream_md5(p, "0:v:0"), _stream_md5(p, "0:a:0")
    assert G.embed(p, dict(META, num_frames=24, fps=24), respect_pref=False)
    r = G.read(p)
    assert r[G.KEY_JSON]["seed"] == 1234567 and r[G.KEY_JSON]["boxes"][0]["id"] == "hat_1"
    assert "Negative prompt: blurry, watermark" in r[G.KEY_PARAMS]
    assert _stream_md5(p, "0:v:0") == v0 and _stream_md5(p, "0:a:0") == a0       # -c copy: stream identici
    _no_personal(p)
    out = G.strip(p, str(tmp_path / ("clean" + ext)))
    assert G.read(out) is None
    assert _stream_md5(out, "0:v:0") == v0 and _stream_md5(out, "0:a:0") == a0


@needs_ff
def test_video_no_audio_and_stego_order(tmp_path):
    p = str(tmp_path / "n.mp4")
    _mk_video(p, audio=False)
    stego.sign_file(p)                                       # ricodifica (mp4v→x264): i metadati vanno DOPO
    assert G.embed(p, META, respect_pref=False)
    assert G.has_metadata(p)
    r = subprocess.run([_FF, "-v", "error", "-i", p, "-map", "0:a?", "-f", "null", "-"], capture_output=True)
    assert r.returncode == 0
