"""FlashVSR+ — test CPU (task 20): geometria pad+crop, profilo VRAM, catalogo, preflight, catena ffmpeg.

Nessuna GPU: il modello non viene mai caricato. I test che servono il pacchetto FlashVSR_Plus (torch) girano
solo dove torch e la cartella del codice ci sono (venv consolidato); altrove vengono saltati.
Esecuzione standalone nel venv consolidato (senza pytest):
    E:\\AI\\Tools\\inference_server_venv\\Scripts\\python.exe tests\\test_flashvsrplus.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workers"))

import flashvsrplus_run as run  # noqa: E402

CODE = Path(os.environ.get("LTX_FLASHVSRPLUS_CODE", r"E:\AI\Tools\FlashVSR_Plus"))


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print("SKIP:", msg)


def test_target_dims_multipli_di_128_e_pari():
    for w, h, s in [(640, 360, 2.0), (1280, 720, 2.0), (853, 479, 3.3), (320, 240, 4.0), (100, 50, 9.0), (1920, 1080, 1.0)]:
        sw, sh, tw, th = run.target_dims(w, h, s)
        assert sw % 2 == 0 and sh % 2 == 0
        assert tw % 128 == 0 and th % 128 == 0
        assert 0 <= tw - sw < 128 and 0 <= th - sh < 128
        k = min(4.0, max(1.0, s))
        assert abs(sw - w * k) <= 2 and abs(sh - h * k) <= 2        # scala rispettata (arrotondata al pari)


def test_scala_massima_x4():
    assert run.target_dims(100, 100, 8.0)[:2] == (400, 400)
    assert run.target_dims(100, 100, 0.2)[:2] == (100, 100)


def test_profilo_vram_dinamico():
    big = run.vram_profile(32, 31, 2560, 1408)
    mid = run.vram_profile(24, 22, 2560, 1408)
    small = run.vram_profile(12, 11, 1280, 768)
    assert big["decode_every"] > mid["decode_every"] > small["decode_every"]
    for p in (big, mid, small):
        assert p["decode_every"] % 4 == 0 and p["decode_every"] >= 8
    assert not big["offload_original_cache"] and small["offload_original_cache"]
    assert small["tiny_decoder_mode"] == "serial" and small["cache_offload_mode"] == "always"
    # altra applicazione che occupa la scheda: conta la VRAM LIBERA, non il totale
    assert run.vram_profile(32, 10, 2560, 1408)["decode_every"] < big["decode_every"]


def test_pad_resize_bordo_replicato_e_nessun_taglio():
    try:
        import numpy as np
        import torch  # noqa: F401
    except ImportError:
        return _skip("torch/numpy assenti")
    if not (CODE / "inference" / "preprocessing" / "input.py").exists():
        return _skip("codice FlashVSR_Plus assente")
    sys.path.insert(0, str(CODE))
    import inference.preprocessing.input as inp
    orig = inp.pil_center_crop_resize
    try:
        w, h, scale = 150, 90, 2.0
        sw, sh, tw, th = run.target_dims(w, h, scale)
        run.install_pad_resize(lambda ww, hh: run.target_dims(ww, hh, scale)[:2])
        frames = (np.random.RandomState(1).rand(3, h, w, 3) * 255).astype("uint8")
        out = inp.pil_center_crop_resize(frames, th, tw)
        assert out.shape == (3, th, tw, 3)
        # zona utile = immagine ingrandita intera (niente ritaglio); il resto = replica dell'ultimo bordo
        assert (out[:, sh:, :sw] == out[:, sh - 1:sh, :sw]).all()
        assert (out[:, :, sw:] == out[:, :, sw - 1:sw]).all()
        # il percorso dei due chiamanti del repo passa dalla funzione sostituita
        t = inp._frames_to_tensor(list(frames), type("R", (), {"height": th, "width": tw})(), stream=True)
        assert tuple(t.shape) == (1, 3, 3, th, tw)
    finally:
        inp.pil_center_crop_resize = orig


def test_crop_finale_con_ffmpeg():
    ff = shutil.which("ffmpeg")
    if not ff:
        return _skip("ffmpeg assente")
    d = Path(tempfile.mkdtemp(prefix="fvp_"))
    try:
        raw, out = d / "raw.mp4", d / "out.mp4"
        subprocess.run([ff, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=384x256:rate=10:duration=1",
                        "-f", "lavfi", "-i", "sine=duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(raw), "-y"], check=True)
        run.crop_and_finish(str(raw), str(out), 300, 200, 384, 256)
        assert out.exists() and not raw.exists()
        r = subprocess.run([ff, "-hide_banner", "-i", str(out)], capture_output=True, text=True).stderr
        assert "300x200" in r and "Audio" in r                      # misura esatta e audio conservato
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_catalogo_voci_flashvsrplus():
    from core import model_paths as mp, weights_catalog as wc
    ids = ["flashvsr-plus-tc", "flashvsr-plus-turbo", "flashvsr-plus-turbo-lq", "flashvsr-plus-base", "flashvsr-plus-base-lq"]
    by = {e["id"]: e for e in wc.load()}
    for i in ids:
        e = by[i]
        assert e["repo_id"] == "ruchangyao/FlashVSR_plus" and len(e["sha256"]) == 64 and e["arch"] == "flashvsr_plus"
        assert "upscale_video" in e["use"]
    assert set(by["flashvsr-plus-turbo"]["deps"]) == {"flashvsr-plus-turbo-lq", "flashvsr-plus-tc"}
    base = str(mp.models_root())
    assert str(mp.path_of("flashvsr-plus-turbo")) == str(Path(base) / "FlashVSR" / "FlashVSR_plus" / "turbo" / "dit.safetensors")
    assert str(mp.path_of("flashvsr-plus-tc")) == str(Path(base) / "FlashVSR" / "FlashVSR_plus" / "TCDecoder.ckpt")


def test_service_preflight_e_pesi_mancanti(monkeypatch=None):
    from core import flashvsrplus_service as s
    assert s.weight_ids("base") == ["flashvsr-plus-base", "flashvsr-plus-base-lq", "flashvsr-plus-tc"]
    assert s.weight_ids("boh") == s.weight_ids("turbo")
    old = s.ckpt_root
    try:
        s.ckpt_root = lambda: Path(tempfile.gettempdir()) / "fvp_inesistente"
        miss = s.missing_weights("turbo")
        assert len(miss) == 3
        errs = s.preflight("turbo")
        assert any("Pesi FlashVSR+" in e for e in errs)
    finally:
        s.ckpt_root = old


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            n += 1
            print("ok", k)
    print(n, "test passati")
