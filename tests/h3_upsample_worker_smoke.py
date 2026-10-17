"""Smoke del WORKER H3 Video Upsampler su CPU con modelli minuscoli a pesi casuali (venv diffusers_next):
    E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe tests\\h3_upsample_worker_smoke.py
Esegue `process()` (lettura video vero con audio, latenti, upscaler, ri-disegno a tile con perno e guida su PIÙ
segmenti, decodifica a pezzi, scrittura mp4) senza GPU e senza pesi: verifica la colla (forme, padding 17g+5,
segmenti, perno, guida, scrittura), NON la qualità. La qualità si giudica dal vivo (task10/TESTS.md).
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

import h3_upsample_core as C                                          # noqa: E402
import h3_upsample_infer as W_                                        # noqa: E402
from core import h3_upsample_plan as P                                # noqa: E402

N_FRAMES = 107                                                        # 17·6+5: due segmenti (plan di 32 passi)


def make_video(path: str, n: int = N_FRAMES, w: int = 256, h: int = 256, fps: int = 24, audio: bool = True):
    import av
    out = av.open(path, "w")
    vs = out.add_stream("libx264", rate=fps)
    vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
    ast = out.add_stream("aac", rate=44100) if audio else None
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        img = np.stack([(xx * 255 // w + i * 3) % 256, (yy * 255 // h + i * 2) % 256,
                        np.full((h, w), (i * 5) % 256)], axis=-1).astype(np.uint8)
        fr = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pk in vs.encode(fr):
            out.mux(pk)
    for pk in vs.encode():
        out.mux(pk)
    if ast is not None:
        total = int(44100 * n / fps)
        t = np.arange(total) / 44100.0
        wav = (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
        for a in range(0, total, 1024):
            blk = wav[a:a + 1024]
            arr = np.stack([blk, blk])
            fr = av.AudioFrame.from_ndarray(arr, format="fltp", layout="stereo")
            fr.sample_rate = 44100
            fr.pts = a
            for pk in ast.encode(fr):
                out.mux(pk)
        for pk in ast.encode():
            out.mux(pk)
    out.close()


def tiny_upscaler_file(path: str):
    from safetensors.torch import save_file
    sd, c, emb = {}, 32, 8

    def res(p):
        sd[p + "in_layers.0.weight"] = torch.ones(c); sd[p + "in_layers.0.bias"] = torch.zeros(c)
        sd[p + "in_layers.2.weight"] = torch.randn(c, c, 3, 3, 3) * 0.01; sd[p + "in_layers.2.bias"] = torch.zeros(c)
        sd[p + "emb_layers.1.weight"] = torch.zeros(2 * c, emb); sd[p + "emb_layers.1.bias"] = torch.zeros(2 * c)
        sd[p + "out_norm.weight"] = torch.ones(c); sd[p + "out_norm.bias"] = torch.zeros(c)
        sd[p + "out_layers.2.weight"] = torch.randn(c, c, 3, 3, 3) * 0.01; sd[p + "out_layers.2.bias"] = torch.zeros(c)

    def tmp(p):
        sd[p + "norm.weight"] = torch.ones(c); sd[p + "norm.bias"] = torch.zeros(c)
        sd[p + "dwconv.weight"] = torch.randn(c, 1, 5, 1, 1) * 0.01; sd[p + "dwconv.bias"] = torch.zeros(c)
        sd[p + "pwconv.weight"] = torch.randn(c, c, 1, 1, 1) * 0.01; sd[p + "pwconv.bias"] = torch.zeros(c)
    sd["conv_in.weight"] = torch.randn(c, 24, 3, 3, 3) * 0.05; sd["conv_in.bias"] = torch.zeros(c)
    sd["conv_out.weight"] = torch.randn(24, c, 3, 3, 3) * 0.05; sd["conv_out.bias"] = torch.zeros(24)
    sd["norm_out.weight"] = torch.ones(c); sd["norm_out.bias"] = torch.zeros(c)
    sd["embed.0.weight"] = torch.zeros(emb, 1); sd["embed.0.bias"] = torch.zeros(emb)
    sd["embed.2.weight"] = torch.zeros(emb, emb); sd["embed.2.bias"] = torch.zeros(emb)
    for blocks in ("in_blocks", "out_blocks"):
        for i in range(4):
            (res if i % 2 == 0 else tmp)(f"{blocks}.{i}.")
    save_file(sd, path)


def tiny_lora_file(path: str, factor: str = "2"):
    from safetensors.torch import save_file
    import json
    meta = {"reference_downscale_factor": factor, "minimax_h3_guide_spatial_version": "target_crop_v2",
            "minimax_h3_guide_position_version": "target_grid_stride_v1", "guide_latent_only": "true",
            "minimax_h3_reference_rope": json.dumps({"guide_rope_layout": "overlap"})}
    save_file({"x.lora_A.weight": torch.zeros(1, 1)}, path, metadata=meta)


class _Post:
    def __init__(self, lat):
        self.lat = lat

    def mode(self):
        return self.lat


class TinyAudioVae:
    config = SimpleNamespace(latents_mean=(0.0,) * 32, latents_std=(1.0,) * 32)

    def encode(self, wav, return_dict=False):
        b, _, s = wav.shape
        return (_Post(torch.randn(b, 32, s // 800)),)


def tiny_dit():
    from diffusers import MiniMaxH3Transformer3DModel
    torch.manual_seed(3)
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=16, hidden_size=32, num_layers=2, num_refiner_layers=1, ffn_dim=64,
        in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=8, freq_dim=16, time_embed_hidden_dim=32,
        time_embed_dim=16, rope_freq_dim=2).eval().requires_grad_(False)


def run_case(tmp: Path, *, guided: bool, denoise: float, audio: bool, n: int = N_FRAMES, tile_tokens: int = 20000):
    import av
    from h3_upsample_diffusers_check import _tiny_vae
    src, out = str(tmp / "src.mp4"), str(tmp / "out.mp4")
    make_video(src, n=n, audio=audio)
    upf, lof = str(tmp / "up.safetensors"), str(tmp / "guide.safetensors")
    tiny_upscaler_file(upf)
    tiny_lora_file(lof)
    W = {"upscaler": upf, "te": "x", "configs": "x", "dit": "x", "vae": "x", "audio_vae": "x", "ids": {}}
    if guided:
        W["guide_lora"] = lof
    rec = []
    orig = C.redraw_chunk

    def spy(tr, emb, tags, video, aud, **kw):
        rec.append({"pin": kw.get("pin") is not None, "guide": None if kw.get("guide") is None else tuple(kw["guide"].shape),
                    "video": tuple(video.shape), "sigma": kw.get("sigma"), "audio": tuple(aud.shape)})
        return orig(tr, emb, tags, video, aud, **kw)
    C.redraw_chunk = spy
    seen = {"lora_arg": None}

    class L:
        embeds = staticmethod(lambda W, dev: (torch.randn(1, 5, 8), torch.ones(5, dtype=torch.long)))
        vae = staticmethod(lambda W: _tiny_vae())
        audio_vae = staticmethod(lambda W: TinyAudioVae())
        upscaler = staticmethod(lambda path: C.load_upscaler(path, torch.float32))

        @staticmethod
        def transformer(W, p):
            seen["lora_arg"] = p.get("loras")
            return tiny_dit()
    p = {"video": src, "output": out, "megapixels": 0.25, "denoise": denoise, "tile_tokens": tile_tokens,
         "lora_strength": 1.0, "workdir": str(tmp)}
    try:
        res = W_.process(p, W, L, "cpu")
    finally:
        C.redraw_chunk = orig
    assert res == out and os.path.getsize(out) > 1000
    with av.open(out) as c:
        vs = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    assert len(frames) == n, (len(frames), n)
    assert frames[0].shape == (512, 512, 3), frames[0].shape               # 0.25 MP, lati multipli di 64
    arr = np.stack(frames[::17]).astype(np.float32)
    assert np.isfinite(arr).all() and arr.std() > 1.0, "video d'uscita piatto o non finito"
    return rec, seen


def check_guidato_multisegmento_con_audio():
    with tempfile.TemporaryDirectory() as d:
        rec, seen = run_case(Path(d), guided=True, denoise=0.1, audio=True, tile_tokens=6000)   # più tile per segmento
    assert len(rec) >= 2, f"servivano almeno 2 segmenti, ho {len(rec)}"
    assert rec[0]["pin"] is False and all(r["pin"] for r in rec[1:]), "il perno va solo dal 2° segmento in poi"
    for r in rec:
        t, c, h, w = r["video"][2], r["video"][1], r["video"][3], r["video"][4]
        assert r["guide"] == (1, 24, t, h // 2, w // 2), (r["guide"], r["video"])   # guida a 1/2, stessi passi
        assert abs(r["sigma"] - P.redraw_sigma(0.1)) < 1e-9
        assert r["audio"][1:3] == (32, 2)
    assert seen["lora_arg"] and seen["lora_arg"][0]["strength"] == 1.0 and seen["lora_arg"][0]["path"].endswith("guide.safetensors")


def check_senza_lora_senza_audio():
    with tempfile.TemporaryDirectory() as d:
        rec, seen = run_case(Path(d), guided=False, denoise=0.1, audio=False, n=22)
    assert len(rec) == 1 and rec[0]["guide"] is None and rec[0]["pin"] is False
    assert seen["lora_arg"] == []


def check_solo_ingrandimento_denoise_zero():
    with tempfile.TemporaryDirectory() as d:
        rec, seen = run_case(Path(d), guided=True, denoise=0.0, audio=True, n=22)
    assert rec == [] and seen["lora_arg"] is None, "denoise 0: né DiT né ri-disegno"


def check_rifiuta_di_rimpicciolire():
    with tempfile.TemporaryDirectory() as d:
        try:
            from h3_upsample_diffusers_check import _tiny_vae
            src = str(Path(d) / "s.mp4")
            make_video(src, n=22, w=256, h=256, audio=False)
            p = {"video": src, "output": str(Path(d) / "o.mp4"), "megapixels": 0.01, "denoise": 0.1, "workdir": d}
            W_.process(p, {"upscaler": "x", "te": "x", "configs": "x", "dit": "x", "vae": "x", "audio_vae": "x", "ids": {}},
                       SimpleNamespace(), "cpu")
        except RuntimeError as e:
            assert "ingrandisce soltanto" in str(e), e
        else:
            raise AssertionError("doveva rifiutare megapixel più piccoli della sorgente")


def check_colonna_sonora_originale_non_toccata():
    """Il servizio rimette l'audio ORIGINALE (copia dei pacchetti, nessuna ricodifica) sul video ingrandito."""
    import av
    from core import h3_upsample_service as hs
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec, _ = run_case(d, guided=True, denoise=0.1, audio=True, n=22)
        src, silent, final = str(d / "src.mp4"), str(d / "out.mp4"), str(d / "final.mp4")
        hs._mux_audio(src, silent, final)

        def audio_bytes(path):
            with av.open(path) as c:
                assert c.streams.audio, f"{path}: nessuna traccia audio"
                return b"".join(bytes(pk) for pk in c.demux(c.streams.audio[0]) if pk.size)
        assert audio_bytes(src) == audio_bytes(final), "l'audio è stato ricodificato o alterato"
        with av.open(final) as c:
            assert c.streams.video[0].codec_context.width == 512


CHECKS = [v for k, v in sorted(globals().items()) if k.startswith("check_")]


def main() -> int:
    bad = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"OK    {fn.__name__}", flush=True)
        except Exception:                                              # noqa: BLE001
            bad += 1
            print(f"FAIL  {fn.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(CHECKS) - bad}/{len(CHECKS)} verifiche passate")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
