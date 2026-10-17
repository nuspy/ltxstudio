"""MagicQuill V2 nativo: flusso completo di `magicquill_v2.run` con MODELLO FINTO (transformer piccolo casuale, VAE finto,
niente GPU, niente checkpoint): cablaggio image_flux (register/placement/loras), sessione di controllo (scrittura poi
lettura), ripristino del transformer condiviso, ricomposizione e salvataggio. Non prova la qualita' d'immagine."""
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import FluxKontextPipeline
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from core.resources import commit_free_gb
from inference_server.backends import image_flux as imf
from core import optional_components as _oc  # noqa: E402
import pytest as _pt  # noqa: E402
if _oc.load_magicquill() is None:                # componente opzionale (optional/magicquill o cartella installata)
    _pt.skip(_oc.last_error(), allow_module_level=True)
from ltx_magicquill import magicquill_v2 as mq
from ltx_magicquill.mq2 import control_lora as cl
from ltx_magicquill.mq2 import puzzle as mqz

import test_mq2_attention as ta          # transformer piccolo + LoRA casuale (stessa cartella tests)

T = 6


@pytest.fixture(autouse=True)
def _risorse():
    if commit_free_gb() < 3.5:
        pytest.skip("commit libero < 3.5 GB")


class FakeVAE:
    config = types.SimpleNamespace(shift_factor=0.1, scaling_factor=0.5)

    def encode(self, x):
        z = F.adaptive_avg_pool2d(x, (x.shape[2] // 8, x.shape[3] // 8))
        z = torch.cat([z] * 6, 1)[:, :16]
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: z, sample=lambda g=None: z))


class FakePipe(FluxKontextPipeline):
    """Eredita prepare_latents/_pack/_encode STOCK; __call__ = mini loop a 2 step sul transformer piccolo."""
    config = types.SimpleNamespace()        # DiffusionPipeline.__setattr__ legge self.config

    def __call__(self, prompt, image, height, width, num_inference_steps, guidance_scale, generator,
                 max_sequence_length, max_area, _auto_resize, callback_on_step_end=None,
                 callback_on_step_end_tensor_inputs=None, **_):
        self.seen = dict(prompt=prompt, steps=num_inference_steps, msl=max_sequence_length, size=(width, height),
                         auto=_auto_resize, area=max_area, image=image.size)
        t = self.image_processor.preprocess(image, height, width)
        lat, img_lat, lat_ids, img_ids = self.prepare_latents(t, 1, 16, height, width, torch.float32,
                                                              torch.device("cpu"), generator, None)
        g = torch.Generator().manual_seed(0)
        enc, pooled = torch.randn(1, T, 32, generator=g), torch.randn(1, 24, generator=g)
        ids = torch.cat([lat_ids, img_ids], 0)
        self.shapes = []
        for ts in (0.9, 0.6):
            with torch.no_grad():
                out = self.transformer(hidden_states=torch.cat([lat, img_lat], 1), timestep=torch.tensor([ts]),
                                       guidance=torch.tensor([3.5]), pooled_projections=pooled, encoder_hidden_states=enc,
                                       txt_ids=torch.zeros(T, 3), img_ids=ids, return_dict=False)[0]
            self.shapes.append(tuple(out.shape))
            assert torch.isfinite(out).all()
        return types.SimpleNamespace(images=[Image.new("RGB", (width, height), (90, 20, 160))])


class FakeMM:
    def __init__(self, pipe):
        self._cache = {"kontext:fake": object()}
        self.pipe = pipe
        self.loras_calls = []

    def ensure(self, key):
        return self.pipe

    def apply_loras(self, pipe, loras):
        self.loras_calls.append(loras)

    def evict_others(self, *a, **k):
        return 0

    def flush(self, *a, **k):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    tr = ta._tiny()
    pipe = object.__new__(FakePipe)
    pipe.__dict__.update(vae=FakeVAE(), vae_scale_factor=8, latent_channels=16,
                         image_processor=VaeImageProcessor(vae_scale_factor=16), transformer=tr,
                         _ls_working_mb=0, _ls_component_keys=())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)    # mai creare un contesto CUDA nei test
    ckpt = tmp_path / "kontext.safetensors"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(imf, "register", lambda mm, model, arch, klein_te="": "kontext:fake")
    monkeypatch.setattr(imf, "_resolve_model_path", lambda p: p)
    monkeypatch.setattr(imf, "_flux_in_channels", lambda p: 64)
    monkeypatch.setattr(imf, "_weights_ident", lambda *a, **k: ("kontext", "x", ""))
    monkeypatch.setattr(imf, "_ensure_placement", lambda *a, **k: "gpu")
    monkeypatch.setattr(imf, "_set_attention", lambda *a, **k: "sdpa")
    monkeypatch.setattr(imf, "_measure_pipe_mb", lambda p: 0)
    # LoRA di controllo piccole al posto dei file reali (dim 3072)
    lora = ta._lora(2, 2)
    monkeypatch.setattr(cl.ControlLoRA, "from_file", classmethod(lambda c, path, **k: lora))
    import core.stego as stego
    monkeypatch.setattr(stego, "enabled", lambda: False)
    import core.watermark as wm
    monkeypatch.setattr(wm, "apply_to_file", lambda p: p)
    a = np.zeros((256, 256, 3), np.uint8)
    a[:, :128] = (200, 40, 40)
    a[:, 128:] = (30, 60, 220)
    Image.fromarray(a).save(tmp_path / "base.png")
    m = np.zeros((256, 256), np.uint8)
    m[80:160, 80:180] = 255
    Image.fromarray(m).save(tmp_path / "mask.png")
    col = a.copy()
    col[80:160, 80:180] = (10, 220, 10)
    Image.fromarray(col).save(tmp_path / "colored.png")
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    for n in ("local", "removal", "edge", "color", "puzzle"):
        (lora_dir / f"{n}_lora.safetensors").write_bytes(b"x")
    d = str(tmp_path)
    base = dict(model=str(ckpt), image_path=d + r"\base.png", mask_path=d + r"\mask.png", lora_dir=str(lora_dir),
                steps=2, seed=5, prompt="un test")
    return types.SimpleNamespace(pipe=pipe, mm=FakeMM(pipe), base=base, out=d + r"\out", d=d, tr=tr)


def _no_leftovers(env):
    tr = env.tr
    assert not any("forward" in b.__dict__ for b in list(tr.transformer_blocks) + list(tr.single_transformer_blocks))
    assert not tr.time_text_embed._forward_pre_hooks
    assert "prepare_latents" not in env.pipe.__dict__


@pytest.mark.parametrize("mode,extra,msl", [
    ("local", {}, 128), ("removal", {}, 512), ("edge", {"edge_path": None}, 128), ("color", {"color_path": "colored.png"}, 128)])
def test_run_controllo_scrittura_poi_lettura(env, mode, extra, msl):
    p = dict(env.base, mode=mode)
    for k, v in extra.items():
        if v:
            p[k] = env.d + "\\" + v
    r = mq.run(p, env.mm, env.out)
    assert r["status"] == "success", r
    png = r["output"][0]
    assert Image.open(png).size == (256, 256)
    s = env.pipe.seen
    assert s["msl"] == msl and s["size"] == (256, 256) and s["auto"] is False and s["area"] == 256 * 256
    assert s["steps"] == 2 and (s["prompt"] == "remove the instance") == (mode == "removal")
    n = 16 * 16 * 2                                   # rumore + Kontext (16x16 token ciascuno a 256 px)
    assert env.pipe.shapes[0] == (1, n + 1024, 64)    # 1° passaggio = scrittura: la coda condizione (512 px = 1024 token)
    assert env.pipe.shapes[1] == (1, n, 64)           # 2° = lettura: coda scartata
    assert env.mm.loras_calls == [[]]
    _no_leftovers(env)


def test_run_foreground_puzzle_solo_li_e_nessun_controllo(env, monkeypatch):
    calls = []
    monkeypatch.setattr(mqz, "load_state_dict", lambda path: calls.append(("load", path)) or {})
    monkeypatch.setattr(mqz, "attach", lambda pipe, sd, names, weights, weight=1.0: calls.append(("attach", names, weights)))
    monkeypatch.setattr(mqz, "detach", lambda pipe: calls.append(("detach",)))
    loras = [{"name": "x.safetensors", "strength": 0.5}, {"name": "none"}]
    r = mq.run(dict(env.base, mode="foreground", loras=loras, fix_perspective=True), env.mm, env.out)
    assert r["status"] == "success", r
    assert [c[0] for c in calls] == ["load", "attach", "detach"]
    assert calls[1][1:] == (["a0"], [0.5])                              # il segnaposto 'none' e' filtrato
    assert env.pipe.shapes[0][1] == 16 * 16 * 2                         # NIENTE coda condizione
    assert env.pipe.seen["msl"] == 128 and env.pipe.seen["prompt"].endswith("Fix the perspective if necessary.")
    _no_leftovers(env)
    # le altre modalita' non toccano la puzzle
    calls.clear()
    r = mq.run(dict(env.base, mode="local"), env.mm, env.out)
    assert r["status"] == "success" and not calls


def test_run_ripristina_anche_su_errore_nel_denoise(env, monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("OOM finto")
    monkeypatch.setattr(FakePipe, "__call__", boom)
    r = mq.run(dict(env.base, mode="local"), env.mm, env.out)
    assert r["status"] == "error" and "OOM finto" in r["error"]
    _no_leftovers(env)


def test_run_output_blend_ripristina_pixel_originali(env):
    r = mq.run(dict(env.base, mode="local", output_blend=True), env.mm, env.out)
    out = np.asarray(Image.open(r["output"][0]).convert("RGB"))
    base = np.asarray(Image.open(env.base["image_path"]).convert("RGB"))
    assert (out[0, 0] == base[0, 0]).all()                              # fuori maschera = originale
    assert np.abs(out[120, 130].astype(int) - (90, 20, 160)).max() <= 2   # dentro = generato (finto)
    r2 = mq.run(dict(env.base, mode="local", output_blend=False), env.mm, env.out)
    out2 = np.asarray(Image.open(r2["output"][0]).convert("RGB"))
    assert np.abs(out2[0, 0].astype(int) - (90, 20, 160)).max() <= 2    # senza blend: tutto il generato
