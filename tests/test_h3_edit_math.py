"""workers/h3_edit_math.py — griglia 17k+5, maschere di latente, catena di clip, maschera persona, manichino grigio (solo CPU)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
import h3_edit_math as M  # noqa: E402


def test_grid_and_latent_counts():
    assert [M.grid_floor(n) for n in (5, 21, 22, 144, 200)] == [5, 5, 22, 141, 192]
    for k in range(0, 9):
        n = 17 * k + 5
        c = M.latent_frame_counts(n)
        assert len(c) == 5 * k + 2 and sum(c) == n and c[:2] == [1, 4]
    with pytest.raises(ValueError):
        M.latent_frame_counts(24)
    with pytest.raises(ValueError):
        M.usable_frames(4)


def test_budget_and_short_edge_snap_to_32():
    tw, th, s = M.budget_size(1920, 1080, 960, 960)
    assert tw % 32 == 0 and th % 32 == 0 and tw * th <= 960 * 960 * 1.1 and s < 1
    assert M.budget_size(640, 360, 960, 960)[2] == 1.0                 # mai ingrandito
    tw, th = M.short_edge_size(1920, 1080, 480)
    assert th == 480 and tw % 32 == 0 and abs(tw / th - 16 / 9) < 0.05


def test_grow_mask_square_dilation():
    m = torch.zeros(1, 9, 9)
    m[0, 4, 4] = 1
    g = M.grow_mask(m, 2)
    assert g[0, 2:7, 2:7].all() and g.sum() == 25


def test_latent_mask_any_pixel_and_frames_and_feather():
    n = 22                                                   # 17*1+5 -> 7 frame di latente
    obj = torch.zeros(n, 64, 64)
    obj[0, 0, 0] = 1                                         # un solo pixel nel primo frame
    cells, counts = M.cell_mask(obj, 64, 64)
    assert cells.shape == (1, 1, 7, 4, 4) and counts == [1, 4, 4, 4, 4, 1, 4]
    assert cells[0, 0, 0, 0, 0] == 1 and cells[0, 0, 0].sum() == 1   # un bordo sottile non sparisce
    assert cells[0, 0, 1:].sum() == 0                         # gli altri frame di latente non coprono il frame 0
    f = M.feather_latent(cells, 2)
    assert f[0, 0, 0, 0, 0] == 1 and 0 < f[0, 0, 0, 0, 1] < 1 and f[0, 0, 0, 0, 1] > f[0, 0, 0, 0, 2]
    assert f[0, 0, 0, 3, 3] == 0                              # oltre la rampa: tenuto


def test_cell_mask_invert_regenerates_everything_but_person():
    n = 5
    obj = torch.zeros(n, 32, 32)
    obj[:, 8:24, 8:24] = 1                                   # persona al centro
    straight, _ = M.cell_mask(obj, 32, 32, invert=False)
    inv, _ = M.cell_mask(obj, 32, 32, invert=True)
    assert straight.shape == inv.shape == (1, 1, 2, 2, 2)
    assert torch.equal(straight, 1 - inv) or (straight + inv).min() >= 1   # cambia-sfondo = complemento (celle di bordo toccate dai due)


def _plan_ok(clips, total):
    assert clips
    covered = sum(l - c for l, c in clips)
    assert covered <= total
    for l, c in clips:
        assert l == M.grid_floor(l) and 0 <= c < l


def test_clip_plan_overlap_and_chain_covers_the_video():
    total, clip, overlap = 24 * 20, 24 * 6, 39               # 20 s, clip da 6 s, 39 frame di sovrapposizione
    clips = M.clip_plan(total, clip, overlap, cuts=[])
    _plan_ok(clips, total)
    assert clips[0][1] == 0 and len(clips) >= 3
    assert all(c > 0 for _l, c in clips[1:])                  # ogni clip dopo il primo riparte da frame gia generati
    assert sum(l - c for l, c in clips) >= total - 22         # copre il video (resta al piu' un avanzo < 22)


def test_clip_plan_snaps_to_hard_cut():
    total = 24 * 12
    clips = M.clip_plan(total, 24 * 7, 39, cuts=[143])        # taglio a 143: 141 (=17*8+5) e' 2 frame prima
    _plan_ok(clips, total)
    assert clips[0][0] == 141 and clips[1][1] == 5            # il primo clip finisce sul taglio; il successivo riparte dagli ultimi 5
    clips = M.clip_plan(total, 24 * 7, 39, cuts=[141])        # taglio esatto: nessuna sovrapposizione
    assert clips[0][0] == 141 and clips[1][1] == 0


def test_find_shots_detects_a_hard_cut():
    a = torch.full((30, 36, 64, 3), 0.1)
    b = torch.full((30, 36, 64, 3), 0.9)
    shots = M.find_shots(torch.cat([a, b]))
    assert shots == [(0, 30), (30, 60)]


def test_person_mask_headroom_and_hold():
    n, h, w = 6, 40, 20
    m = torch.zeros(n, h, w)
    m[:, 20:38, 8:12] = 1                                    # persona alta 18 px
    p, size = M.person_mask(m, headroom=0.2, hold_frames=0)
    assert p[0, 17:20, 8:12].all()                           # ~18*0.2 = 4 righe sopra la testa
    assert not p[0, 10:15, 8:12].any() and p[0, 20, 8]
    assert size[0, 25, 9] == pytest.approx(18 / h)
    m2 = torch.zeros(n, h, w)
    m2[2, 20:38, 8:12] = 1                                   # visibile solo nel frame 2
    p2, _ = M.person_mask(m2, 0.0, hold_frames=1)
    assert p2[1].any() and p2[3].any() and not p2[0].any() and not p2[5].any()   # posa tenuta 1 frame prima e dopo


def test_grey_mannequin_hides_face_and_clothes_colour():
    n, h, w = 2, 48, 48
    img = torch.rand(n, h, w, 3)
    img[:, 10:40, 10:30] = torch.tensor([0.9, 0.1, 0.1])    # persona rossa
    mask = torch.zeros(n, h, w)
    mask[:, 10:40, 10:30] = 1
    size = mask * (30 / h)
    out = M.grey_mannequins(img, mask, size, blur=0.02)
    inside = out[:, 15:35, 12:28]
    assert (inside.max(-1).values - inside.min(-1).values).max() < 1e-4        # grigio: niente colore dei vestiti
    assert torch.allclose(out[:, :8], img[:, :8])                                # fuori dalla persona resta com'e'
    neutral = M.grey_mannequins(img, mask, size, neutral_background=True)
    assert torch.allclose(neutral[:, :8], torch.full_like(neutral[:, :8], 0.5))  # sfondo piatto grigio


def test_blur_area_keeps_outside_and_removes_detail_inside():
    n, h, w = 1, 64, 64
    img = torch.zeros(n, h, w, 3)
    img[:, ::2] = 1.0                                        # righe fini (dettaglio)
    area = torch.zeros(n, h, w)
    area[:, :, 32:] = 1                                      # metà destra = sfondo da nascondere
    out = M.blur_area(img, area, radius=6.0, grey=0.5)
    assert torch.allclose(out[:, :, :20], img[:, :, :20])    # persona intatta
    assert out[:, 4:60, 40:60].std() < 0.05                  # sfondo: niente dettaglio da copiare


def test_enforce_min_length_extends_short_last_clip_backwards_keeping_new_frames():
    raw = M.clip_plan(260, 144, 39, [])                         # giro 02/11: [(141,0),(141,39),(56,39)] -> l'ultimo da 56 frame era rifiutato da H3
    assert raw[-1][0] == 56
    fixed = M.enforce_min_length(raw, 260)
    assert fixed == [(141, 0), (141, 39), (124, 107)]
    assert all(l >= M.MIN_CLIP_FRAMES and l == M.grid_floor(l) for l, _c in fixed)
    new = lambda cl: sum(l - c for l, c in cl)                  # noqa: E731
    assert new(fixed) == new(raw)                               # stessi frame NUOVI generati: cambia solo il contesto
    assert fixed[-1][1] <= fixed[-2][0]                         # il contesto sta dentro il clip precedente (prev_tail)
    assert (fixed[-1][1] - raw[-1][1]) % 17 == 0                # confine di frame di latente conservato
    cum = [0]
    for k in range(60):
        cum.append(cum[-1] + M.FRAMES_PER_LATENT[k % 5])
    assert fixed[-1][1] in cum                                  # il contesto finisce esattamente su un frame di latente


def test_enforce_min_length_leaves_good_plans_alone_and_rejects_too_short_videos():
    good = M.clip_plan(24 * 20, 144, 39, [])
    assert [c for c in M.enforce_min_length(good, 480) if c[0] < 124] == []
    import pytest
    with pytest.raises(ValueError, match="troppo corto"):
        M.enforce_min_length([(56, 0)], 56)                    # un video da 2,3 s non puo andare in H3 (minimo 5,2 s)
