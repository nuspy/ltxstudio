"""Verifiche del nucleo H3 Video Upsampler che richiedono diffusers (venv diffusers_next): si lanciano con
    E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe tests\\h3_upsample_diffusers_check.py
Solo CPU, modelli minuscoli con pesi casuali (nessun peso vero, nessuna GPU). Asserzioni semplici: il venv non ha
pytest; tests/test_h3_upsample_layout.py lo lancia da pytest quando il venv esiste.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workers"))

import torch                                                          # noqa: E402

import h3_upsample_core as C                                          # noqa: E402
from core import h3_upsample_plan as P                                # noqa: E402

TEXT_TAGS = torch.tensor([1] * 5 + [0] * 0, dtype=torch.long)


def check_layout_plain_equals_diffusers_ref2va():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep as R
    for (T, lh, lw, ta) in ((7, 4, 6, 12), (12, 8, 4, 20)):
        mine = C.build_tile_layout(TEXT_TAGS, T, lh, lw, ta)
        pos, tags, vi, ai, ti, ncv, nca = R.build_ref2va_packed_sequence(
            TEXT_TAGS, [], [], [], T, lh, lw, ta, (1, 2, 2), 2, C.AUDIO_TAG, C.VIDEO_TAG)
        assert torch.equal(mine["position_ids"], pos), "posizioni ≠ ref2va diffusers"
        assert torch.equal(mine["token_tags"], tags) and torch.equal(mine["video_indices"], vi)
        assert torch.equal(mine["audio_indices"], ai) and torch.equal(mine["text_indices"], ti)
        assert mine["n_cond_rows"] == ncv == 0 and nca == 0


def check_layout_pin_equals_diffusers_fl2va_first():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep as L
    T, lh, lw, ta = 7, 4, 6, 12
    mine = C.build_tile_layout(TEXT_TAGS, T, lh, lw, ta, pin=True)
    pos, tags, vi, ai, ti, ncv, _ = L.build_packed_sequence(TEXT_TAGS, T, lh, lw, ta, (1, 2, 2), 2, C.AUDIO_TAG,
                                                            C.VIDEO_TAG, ("first",))
    assert torch.equal(mine["position_ids"], pos), "il perno non coincide con il keyframe «first» di fl2va"
    assert torch.equal(mine["token_tags"], tags) and torch.equal(mine["video_indices"], vi)
    assert torch.equal(mine["audio_indices"], ai) and mine["n_cond_rows"] == ncv


def check_layout_guide_matches_strided_full_grid():
    """La guida = le righe di una keyframe a griglia piena (vt passi sull'orologio del video) tenute solo dove
    (riga patch % f == 0 e colonna patch % f == 0): la stessa operazione di upscale_guide._shrink dello studio."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep as L
    T, lh, lw, ta, f = 9, 8, 12, 14, 2
    hp, wp = lh // 2, lw // 2
    mine = C.build_tile_layout(TEXT_TAGS, T, lh, lw, ta, pin=False, guide_factor=f, guide_steps=T)
    # layout di riferimento con T passi di keyframe «first» (griglia piena): una riga per cella patch, a t = n_testo
    full_pos, *_ = L.build_packed_sequence(TEXT_TAGS, 1, lh, lw, ta, (1, 2, 2), 2, C.AUDIO_TAG, C.VIDEO_TAG, ("first",))
    n_text = int(TEXT_TAGS.shape[0])
    frame_rows = full_pos[n_text:n_text + hp * wp, 1:]                      # (h, w) della griglia del tile
    B = C._diffusers_layout_helpers()
    times = B._temporal_position_grid(T, float(n_text))
    keep = torch.zeros(hp, wp, dtype=torch.bool)
    keep[::f, ::f] = True
    exp = []
    for t in range(T):
        sub = frame_rows[keep.reshape(-1)]
        exp.append(torch.cat([torch.full((sub.shape[0], 1), float(times[t]), dtype=torch.float64), sub], dim=1))
    exp = torch.cat(exp)
    n = mine["n_guide_rows"]
    assert n == exp.shape[0] == T * (hp // f) * (wp // f)
    got = mine["position_ids"][n_text:n_text + n]
    assert torch.equal(got, exp), "posizioni della guida ≠ keyframe a griglia piena sottocampionata ogni f"
    # tutte le righe condizione sono «video» (tag 0) e stanno nei video_indices davanti ai generati
    assert torch.equal(mine["video_indices"][:n], torch.arange(n_text, n_text + n))
    assert (mine["token_tags"][n_text:n_text + n] == C.VIDEO_TAG).all()


def check_patchify_roundtrip():
    x = torch.randn(1, 24, 7, 8, 6)
    assert torch.equal(C.unpatchify(C.patchify(x), 7, 8, 6), x)


def check_step_equals_scheduler():
    from diffusers import MiniMaxH3Scheduler
    s = MiniMaxH3Scheduler(shift=12.0)
    sigma = P.redraw_sigma(0.2)
    s.set_timesteps(sigmas=[sigma, 0.0])
    assert abs(float(s.timesteps[0]) - (1.0 - sigma)) < 1e-6
    x, v = torch.randn(1, 24, 3, 4, 4), torch.randn(1, 24, 3, 4, 4)
    ref = s.step(v, s.timesteps[0], x, return_dict=False)[0]
    mine = x + float(sigma) * v
    assert torch.allclose(ref, mine, atol=1e-5), float((ref - mine).abs().max())


def _tiny_vae():
    from diffusers import AutoencoderKLMiniMaxH3
    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3(
        block_out_channels=(32,) * 6, layers_per_block=1, decoder_num_layers=2, decoder_num_attention_heads=2,
        decoder_attention_head_dim=16, decoder_ffn_mult=2, decoder_num_register_tokens=2,
        latents_mean=tuple(P.LATENT_MEAN), latents_std=tuple(P.LATENT_STD))
    return vae.eval().requires_grad_(False)


def check_vae_encode_decode_equal_diffusers():
    vae = _tiny_vae()
    pm, ps = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mean, std = tuple(P.LATENT_MEAN), tuple(P.LATENT_STD)
    n = 17 * 2 + 5                                   # 39 fotogrammi, 32x48 px → 2x3 celle
    frames = torch.rand(n, 32, 48, 3)
    px = frames.permute(3, 0, 1, 2)[None]
    z_ref = vae.encode((px - torch.tensor(pm).view(1, 3, 1, 1, 1)) / torch.tensor(ps).view(1, 3, 1, 1, 1),
                       return_dict=False)[0].sample(generator=torch.Generator().manual_seed(42))
    z_ref = z_ref.to(torch.float16).float()
    z_ref = (z_ref - torch.tensor(mean).view(1, -1, 1, 1, 1)) / torch.tensor(std).view(1, -1, 1, 1, 1)

    def blocks():                                    # a blocchi di dimensioni irregolari, come arriverebbe da un file
        for a, b in ((0, 5), (5, 22), (22, 23), (23, 39)):
            yield frames[a:b]
    z = C.encode_frames(vae, blocks(), n, pixel_mean=pm, pixel_std=ps, latents_mean=mean, latents_std=std,
                        device="cpu")
    assert z.shape == z_ref.shape == (1, 24, 5 * 2 + 2, 2, 3), (z.shape, z_ref.shape)
    assert torch.allclose(z, z_ref, atol=1e-5), float((z - z_ref).abs().max())

    out = []
    C.decode_stream(vae, z, out.append, pixel_mean=pm, pixel_std=ps, latents_mean=mean, latents_std=std, device="cpu")
    got = torch.from_numpy(__import__("numpy").concatenate(out)).float()
    raw = z * torch.tensor(std).view(1, -1, 1, 1, 1) + torch.tensor(mean).view(1, -1, 1, 1, 1)
    ref = vae.decode(raw, return_dict=False)[0].float()
    ref = ((ref * torch.tensor(ps).view(1, 3, 1, 1, 1) + torch.tensor(pm).view(1, 3, 1, 1, 1)).clamp(0, 1)[0]
           .permute(1, 2, 3, 0) * 255.0).round()
    assert got.shape == ref.shape == (n, 32, 48, 3), (got.shape, ref.shape)
    assert float((got - ref).abs().max()) <= 1.0, float((got - ref).abs().max())


class _ZeroVelocity(torch.nn.Module):
    """DiT finto: velocità 0 → il ri-disegno deve restituire esattamente x_noised; controlla anche le forme."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices,
                token_tags, position_ids, video_indices, audio_indices, text_indices, return_dict=False):
        s = position_ids.shape[0]
        assert token_tags.shape == (s,) and timestep_indices.shape == (s,)
        assert hidden_states.shape[1] == video_indices.numel(), "righe video ≠ indici"
        assert audio_hidden_states.shape[1] == audio_indices.numel() and encoder_hidden_states.shape[1] == text_indices.numel()
        assert int(timestep_indices.max()) < timestep.numel()
        self.calls.append((s, hidden_states.shape[1], timestep.tolist()))
        return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)


def check_redraw_tiles_seamless_and_shapes():
    tf = _ZeroVelocity()
    lh, lw, T = 32, 48, 7                           # 512x768 px: lato minimo del tile 256 → il budget basso forza più tile
    video = torch.randn(1, 24, T, lh, lw)
    audio = torch.randn(1, 32, 2, 12)
    pin = torch.randn(1, 24, 1, lh, lw)
    f = 2
    guide = torch.randn(1, 24, T, lh // f, lw // f)
    emb = torch.randn(1, 5, 8)
    sigma = P.redraw_sigma(0.2)
    budget = 1800
    out = C.redraw_chunk(tf, emb, TEXT_TAGS, video, audio, sigma=sigma, factor=f, guide=guide, pin=pin,
                         budget=budget, device="cpu")
    t0 = 1.0 - sigma
    field = torch.randn(video.shape, generator=torch.Generator().manual_seed(P.SEED), dtype=torch.float32)
    want = t0 * video + (1.0 - t0) * field
    assert len(tf.calls) > 1, "il budget basso doveva produrre più tile"
    assert max(c[0] for c in tf.calls) <= budget * 1.2, tf.calls     # sequenza (testo, audio e perno inclusi) ≈ budget
    assert torch.allclose(out, want, atol=1e-5), float((out - want).abs().max())    # niente cucitura
    # timestep: video t0, condizione 0.999, audio 1.0 (testo = video)
    ts = sorted(set(tf.calls[0][2]))
    assert any(abs(t - t0) < 1e-6 for t in ts) and any(abs(t - 0.999) < 1e-6 for t in ts) and any(abs(t - 1.0) < 1e-6 for t in ts)
    # senza perno né guida
    tf2 = _ZeroVelocity()
    out2 = C.redraw_chunk(tf2, emb, TEXT_TAGS, video, audio, sigma=sigma, budget=1800, device="cpu")
    assert torch.allclose(out2, want, atol=1e-5)


def check_redraw_with_real_tiny_dit():
    from diffusers import MiniMaxH3Transformer3DModel
    torch.manual_seed(1)
    tr = MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=16, hidden_size=32, num_layers=2, num_refiner_layers=1, ffn_dim=64,
        in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=8, freq_dim=16, time_embed_hidden_dim=32,
        time_embed_dim=16, rope_freq_dim=2).eval().requires_grad_(False)
    lh, lw, T = 32, 48, 7
    video, audio = torch.randn(1, 24, T, lh, lw), torch.randn(1, 32, 2, 12)
    guide, pin = torch.randn(1, 24, T, lh // 2, lw // 2), torch.randn(1, 24, 1, lh, lw)
    out = C.redraw_chunk(tr, torch.randn(1, 5, 8), TEXT_TAGS, video, audio, sigma=P.redraw_sigma(0.1), factor=2,
                         guide=guide, pin=pin, budget=2600, device="cpu")
    assert out.shape == video.shape and torch.isfinite(out).all()


def check_upscaler_net_shapes_and_rhythm():
    """La rete letta dalle chiavi: stesso ritmo del file vero (residuo, conv temporale, …) con canali ridotti."""
    sd = {}
    c, emb = 32, 8

    def res(p):
        sd[p + "in_layers.0.weight"] = torch.ones(c); sd[p + "in_layers.0.bias"] = torch.zeros(c)
        sd[p + "in_layers.2.weight"] = torch.zeros(c, c, 3, 3, 3); sd[p + "in_layers.2.bias"] = torch.zeros(c)
        sd[p + "emb_layers.1.weight"] = torch.zeros(2 * c, emb); sd[p + "emb_layers.1.bias"] = torch.zeros(2 * c)
        sd[p + "out_norm.weight"] = torch.ones(c); sd[p + "out_norm.bias"] = torch.zeros(c)
        sd[p + "out_layers.2.weight"] = torch.zeros(c, c, 3, 3, 3); sd[p + "out_layers.2.bias"] = torch.zeros(c)

    def tmp(p):
        sd[p + "norm.weight"] = torch.ones(c); sd[p + "norm.bias"] = torch.zeros(c)
        sd[p + "dwconv.weight"] = torch.zeros(c, 1, 5, 1, 1); sd[p + "dwconv.bias"] = torch.zeros(c)
        sd[p + "pwconv.weight"] = torch.zeros(c, c, 1, 1, 1); sd[p + "pwconv.bias"] = torch.zeros(c)
    sd["conv_in.weight"] = torch.zeros(c, 24, 3, 3, 3); sd["conv_in.bias"] = torch.zeros(c)
    sd["conv_out.weight"] = torch.zeros(24, c, 3, 3, 3); sd["conv_out.bias"] = torch.zeros(24)
    sd["norm_out.weight"] = torch.ones(c); sd["norm_out.bias"] = torch.zeros(c)
    sd["embed.0.weight"] = torch.zeros(emb, 1); sd["embed.0.bias"] = torch.zeros(emb)
    sd["embed.2.weight"] = torch.zeros(emb, emb); sd["embed.2.bias"] = torch.zeros(emb)
    for blocks in ("in_blocks", "out_blocks"):
        for i in range(4):
            (res if i % 2 == 0 else tmp)(f"{blocks}.{i}.")
    from safetensors.torch import save_file
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "up.safetensors")
        save_file(sd, f)
        net = C.load_upscaler(f, dtype=torch.float32)
        x = torch.randn(1, 24, 5, 4, 6)
        y = C.upscale_chunk(net, x, 12 * 16, 8 * 16, "cpu", dtype=torch.float32)
        assert y.shape == (1, 24, 5, 8, 12)
        assert C.upscale_chunk(net, x, 6 * 16, 4 * 16, "cpu") is x          # stessa taglia → invariato
        try:
            C.upscale_chunk(net, x, 2 * 16, 2 * 16, "cpu")
        except RuntimeError:
            pass
        else:
            raise AssertionError("un upscaler non deve rimpicciolire")


CHECKS = [v for k, v in sorted(globals().items()) if k.startswith("check_")]


def main() -> int:
    bad = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"OK    {fn.__name__}")
        except Exception:                                              # noqa: BLE001
            bad += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"{len(CHECKS) - bad}/{len(CHECKS)} verifiche passate")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
