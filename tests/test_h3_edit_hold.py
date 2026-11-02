"""workers/h3_edit_hold.py — lo scheduler tiene le righe mascherate legate alla sorgente (CPU; serve il diffusers con MiniMaxH3Scheduler)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
try:
    from diffusers import MiniMaxH3Scheduler
except Exception:                                            # noqa: BLE001
    MiniMaxH3Scheduler = None
import h3_edit_hold as Hh  # noqa: E402

pytestmark = pytest.mark.skipif(MiniMaxH3Scheduler is None, reason="diffusers senza MiniMaxH3Scheduler (usare diffusers_next_venv)")


def _run(steps=6, vel=0.3):
    torch.manual_seed(0)
    n, c = 8, 4
    src = torch.randn(n, c)
    noise = torch.randn(n, c)
    keep = torch.zeros(n, c)
    keep[:3] = 1                                             # le prime 3 righe: tieni la sorgente
    S = Hh.make_hold_scheduler_class(MiniMaxH3Scheduler)(shift=12.0)
    S.set_timesteps(steps)
    S.set_hold(src, noise, keep)
    x = noise.clone()
    hist = []
    for t in S.timesteps:
        x = S.step(torch.full_like(x, vel), t, x, return_dict=False)[0]
        hist.append(x.clone())
    return S, src, noise, keep, x, hist


def test_held_rows_end_exactly_on_the_source():
    S, src, _noise, keep, x, _h = _run()
    assert torch.equal(x[:3], src[:3])                       # all'ultimo passo sigma=0: sorgente esatta
    assert not torch.allclose(x[3:], src[3:])                # le righe generate sono altro


def test_held_rows_follow_the_noised_source_at_every_step():
    S, src, noise, keep, _x, hist = _run()
    for i, x in enumerate(hist):
        t_next = 1.0 - float(S.sigmas[i + 1])
        want = t_next * src[:3] + (1 - t_next) * noise[:3]
        assert torch.allclose(x[:3], want, atol=1e-6), i


def test_without_hold_it_is_the_plain_scheduler():
    base = MiniMaxH3Scheduler(shift=12.0)
    base.set_timesteps(5)
    S = Hh.make_hold_scheduler_class(MiniMaxH3Scheduler)(shift=12.0)
    S.set_timesteps(5)
    a = b = torch.randn(4, 3)
    for t in base.timesteps:
        v = torch.full_like(a, 0.1)
        a = base.step(v, t, a, return_dict=False)[0]
        b = S.step(v, t, b, return_dict=False)[0]
    assert torch.allclose(a, b)


def test_shape_mismatch_is_refused():
    S = Hh.make_hold_scheduler_class(MiniMaxH3Scheduler)(shift=12.0)
    with pytest.raises(ValueError):
        S.set_hold(torch.zeros(3, 2), torch.zeros(3, 2), torch.zeros(2, 2))
