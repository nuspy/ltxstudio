"""H3 Video Upsampler: la parte pura (core/h3_upsample_plan.py): geometria, segmenti, tile, giunture, sigma.
Nessun modello, nessuna GPU. I valori attesi sono quelli dichiarati dal sorgente studiato (tiled_upsampler.py)."""
import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import h3_upsample_plan as P  # noqa: E402

VAE_CFG = Path(r"E:\AI\Models\configs\minimax-h3\vae\config.json")


def test_statistiche_upscaler_identiche_al_vae():
    """Il porting salta (x-mean)/std perché in diffusers il latente è già normalizzato: vale solo se le 24+24 cifre
    dell'upscaler coincidono con latents_mean/std del VAE video di H3."""
    if not VAE_CFG.is_file():
        pytest.skip("config del VAE H3 assente")
    c = json.loads(VAE_CFG.read_text("utf-8"))
    for a, b in ((P.LATENT_MEAN, c["latents_mean"]), (P.LATENT_STD, c["latents_std"])):
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9          # stesse cifre (differenze solo di arrotondamento f32)
    assert len(P.LATENT_MEAN) == len(P.LATENT_STD) == 24


def test_sigma_di_partenza_della_ricetta():
    assert abs(P.redraw_sigma(0.2) - 0.75) < 1e-9              # «0,2» = 0,75 sotto lo shift 12 (dichiarato nel sorgente)
    assert abs(P.redraw_sigma(0.1) - 12 * 0.1 / (1 + 11 * 0.1)) < 1e-9
    assert P.redraw_sigma(0.0) is None and P.redraw_sigma(-1) is None
    assert 0 < P.redraw_sigma(0.05) < P.redraw_sigma(0.1) < P.redraw_sigma(0.2) < 1


def test_valid_length_e_grid():
    assert [P.valid_length(n) for n in (1, 5, 6, 22, 23, 100, 107)] == [5, 5, 22, 22, 39, 107, 107]
    assert P.pad_size(1280, 720) == (1280, 736) and P.grid_size(1280, 720) == (1280, 704) and P.grid_size(10, 10) == (32, 32)


def test_target_size_mantiene_rapporto_e_griglia():
    w, h = P.target_size(1280, 704, 2.0)
    assert w % 32 == 0 and h % 32 == 0 and abs(w / h - 1280 / 704) < 0.02 and abs(w * h / 1048576 - 2.0) < 0.1
    w, h = P.target_size(1280, 704, 2.0, align=64)
    assert w % 64 == 0 and h % 64 == 0                          # con la guida (fattore 2)


def test_tempo_passi_e_fotogrammi():
    assert [P.frames_in(k) for k in (0, 1, 2, 5, 10, 27, 30)] == [0, 1, 5, 17, 34, 90, 102]
    assert P.tokens_to_reach(17) == 5 and P.tokens_to_reach(18) == 6 and P.tokens_to_reach(0) == 0
    assert P.audio_span(0, 102) == (0, 170)


@pytest.mark.parametrize("frames", [5, 22, 107, 124, 362, 1000 // 17 * 17 + 5, 2877])
def test_segmenti_coprono_tutto_e_si_sovrappongono(frames):
    length = P.valid_length(frames)
    tokens = 5 * ((length - 5) // 17) + 2
    plan = P.chunks(tokens)
    assert plan[0][0] == 0 and plan[-1][2] == tokens and plan[-1][3] == P.frames_in(tokens)
    for (k0, f0, k1, f1) in plan:
        assert k0 % 5 == 0 and f0 == P.frames_in(k0) and f1 == P.frames_in(k1) and k1 > k0
        assert (f1 - f0) <= P.CHUNK_FRAMES + 17
    for a, b in zip(plan, plan[1:]):
        assert b[0] <= a[2], "buco fra due segmenti"           # nessun fotogramma senza segmento
        assert b[0] > a[0], "il segmento non avanza"
        assert a[2] - b[0] >= 1                                 # almeno un passo condiviso: la dissolvenza


def test_griglia_tile_budget_e_coperture():
    for (h_px, w_px, steps, budget, factor) in ((1088, 1920, 30, 70000, 0), (1088, 1920, 30, 70000, 2),
                                                (768, 1344, 27, 20000, 2), (2176, 3840, 30, 70000, 2)):
        tiles, grid = P.tile_grid(h_px // 16, w_px // 16, steps, factor or 1, budget)
        rows, cols, tw, th, ow, oh = grid
        cost = P.guided_seq_len(tw, th, steps, factor) if factor > 1 else P.seq_len(tw, th, steps)
        assert cost <= budget and tw >= P.MIN_TILE_PX and th >= P.MIN_TILE_PX
        cover = torch.zeros(h_px // 16, w_px // 16)
        for (r0, c0, rr, cc, *_rest) in tiles:
            cover[r0:r0 + rr, c0:c0 + cc] += 1
            assert rr % 2 == 0 and cc % 2 == 0                  # patch 2x2
            if factor > 1:
                assert r0 % (2 * factor) == 0 and c0 % (2 * factor) == 0 and rr % (2 * factor) == 0
        assert (cover >= 1).all(), "pixel non coperti da nessun tile"


def test_feather_somma_positiva_e_niente_cuciture():
    """La somma dei pesi dei tile che coprono un punto non deve mai annullarsi (poi si normalizza)."""
    h, w, steps = 64, 96, 27
    tiles, _ = P.tile_grid(h, w, steps, 1, 6000)
    assert len(tiles) > 1
    wsum = torch.zeros(h, w)
    for (r0, c0, rr, cc, st, sl, top, left, bottom, right) in tiles:
        m = P.feather(rr, cc, st, sl, top, left, bottom, right)
        assert float(m.min()) >= 0 and float(m.max()) <= 1
        wsum[r0:r0 + rr, c0:c0 + cc] += m
    assert float(wsum.min()) > 1e-3, "esiste un punto con peso ~0"


def test_budget_impossibile_da_errore_chiaro():
    with pytest.raises(RuntimeError, match="abbassa i megapixel"):
        P.tile_grid(2176 // 16, 3840 // 16, 30, 1, 500)


def test_join_chunk_dissolve_la_sovrapposizione():
    a = torch.ones(1, 24, 10, 4, 4)
    b = torch.zeros(1, 24, 10, 4, 4)
    au_a, au_b = torch.ones(1, 32, 2, 17), torch.zeros(1, 32, 2, 17)
    v, au = P.join_chunk(None, None, a, au_a, 0, 0)
    assert v is a
    v, au = P.join_chunk(v, au, b, au_b, 5, 17)               # b parte al passo 5: 5 passi di sovrapposizione
    assert v.shape[2] == 15
    ramp = v[0, 0, 5:10, 0, 0]
    assert float(ramp[0]) == 1.0 and float(ramp[-1]) < 0.3 and (ramp[1:] <= ramp[:-1]).all()
    assert float(v[0, 0, 12, 0, 0]) == 0.0 and float(v[0, 0, 0, 0, 0]) == 1.0


def test_intestazione_lora_di_guida():
    meta = {"reference_downscale_factor": "2", "minimax_h3_guide_spatial_version": "target_crop_v2",
            "minimax_h3_guide_position_version": "target_grid_stride_v1", "guide_latent_only": "true",
            "minimax_h3_reference_rope": json.dumps({"guide_rope_layout": "overlap"})}
    assert P.guide_factor_from_meta(meta) == 2
    assert P.guide_factor_from_meta({}) == 0 and P.guide_factor_from_meta({"reference_downscale_factor": "1"}) == 0
    with pytest.raises(ValueError):
        P.guide_factor_from_meta({**meta, "minimax_h3_guide_spatial_version": "target_crop_v3"})
    with pytest.raises(ValueError):
        P.guide_factor_from_meta({**meta, "guide_latent_only": "false"})
    with pytest.raises(ValueError):
        P.guide_factor_from_meta({**meta, "minimax_h3_reference_rope": json.dumps({"guide_rope_layout": "seq"})})


def test_describe():
    d = P.describe(1280, 720, 240, 2.0, factor=2)
    assert d["canvas"] == (1280, 768) and d["out"][0] % 64 == 0 and d["out"][1] % 64 == 0 and d["crop"] == (1920, 1080)
    assert d["frames_padded"] == 243 and d["latent_steps"] == 72 and d["chunks"] >= 3 and d["tiles_per_chunk"] >= 1


@pytest.mark.parametrize("w,h", [(1280, 720), (960, 544), (250, 190), (1080, 1920), (1000, 1000), (854, 480), (640, 360),
                                 (1921, 1081), (333, 777)])
def test_geometria_senza_stirare(w, h):
    """Regola «mai stirare»: bordo aggiunto (mai tolto contenuto), scala di x e y uguale, ritaglio finale dal rapporto esatto."""
    for factor in (1, 2):
        g = P.plan_geometry(w, h, 2.0, factor)
        (cw, ch), (ow, oh), (kw, kh) = g["canvas"], g["out"], g["crop"]
        assert cw >= w and ch >= h and cw % 32 == 0 and ch % 32 == 0
        assert ow % (32 * factor) == 0 and oh % (32 * factor) == 0 and ow >= cw and oh >= ch
        assert g["scale_mismatch"] < 0.003, g                      # stessa scala sui due assi: nessuna deformazione
        assert kw % 2 == 0 and kh % 2 == 0 and kw <= ow and kh <= oh
        assert abs(kw / kh - w / h) / (w / h) < 0.004, (kw, kh, w, h)       # rapporto ESATTO a meno dell'arrotondamento pari
        assert cw - w <= 192 and ch - h <= 192                       # bordo contenuto


def test_budget_automatico_dalla_vram():
    assert P.auto_token_budget(None) == P.TOKEN_BUDGET == P.auto_token_budget(float("inf"))
    assert P.auto_token_budget(64) == P.TOKEN_BUDGET                  # tetto
    assert P.auto_token_budget(1) == P.TOKEN_MIN                      # pavimento
    vals = [P.auto_token_budget(g) for g in (6, 8, 10, 12, 16, 20)]
    assert vals == sorted(vals) and len(set(vals)) > 2 and all(v % 1024 == 0 or v == P.TOKEN_BUDGET for v in vals)
