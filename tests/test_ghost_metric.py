"""tests/app_live/ghost_metric.py: funzioni pure della misura automatica del doppione (nessun modello)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "app_live"))
import ghost_metric as G  # noqa: E402


def _person_mask(h=200, w=200):
    m = np.zeros((h, w), bool)
    m[40:190, 70:130] = True
    return m


def test_head_center_is_the_top_of_the_silhouette():
    m = _person_mask()
    cx, cy, pw, ph = G.head_center(m)
    assert 95 < cx < 105 and 40 <= cy < 70 and pw == 60 and ph == 150
    assert G.head_center(np.zeros((50, 50), bool)) is None


def test_head_fidelity_is_zero_for_the_original_and_high_when_the_model_redraws_the_head():
    m = _person_mask()
    rng = np.random.default_rng(0)
    src = rng.integers(20, 240, (200, 200, 3)).astype(np.uint8)
    assert G.head_fidelity(src.copy(), src, m) == pytest.approx(0.0, abs=1e-6)
    out = src.copy()
    out[40:100, 70:130] = rng.integers(20, 240, (60, 60, 3)).astype(np.uint8)         # la testa ridisegnata
    assert G.head_fidelity(out, src, m) > 0.1
    out2 = src.copy()
    out2[150:190, 70:130] = 0                                                           # solo le gambe cambiano: la testa resta fedele
    assert G.head_fidelity(out2, src, m) == pytest.approx(0.0, abs=1e-6)


def test_out_share_counts_person_pixels_outside_the_dilated_silhouette():
    cv2 = pytest.importorskip("cv2")                                                    # noqa: F841
    m = _person_mask()
    sal = m.copy()
    assert G.out_share(sal, m, 10) == 0.0
    sal2 = m.copy()
    sal2[60:120, 10:60] = True                                                          # persona fantasma a sinistra
    assert 0.2 < G.out_share(sal2, m, 10) < 0.7


def test_faces_far_from_the_true_head_but_near_the_person_are_ghosts_background_faces_are_ignored():
    head = (100.0, 55.0, 60.0, 150.0)
    n, dpx, dw, ghost = G.face_offset([(100.0, 58.0)], head, 60.0, 150.0)
    assert (n, ghost) == (1, 0) and dpx < 5
    n, dpx, dw, ghost = G.face_offset([(100.0, 58.0), (55.0, 60.0)], head, 60.0, 150.0)
    assert (n, ghost) == (2, 1)                                                         # secondo volto a 45 px = 0,75 larghezze: fantasma
    n, *_ = G.face_offset([(400.0, 60.0)], head, 60.0, 150.0)
    assert n == 0                                                                       # volto lontano = sfondo: scartato


def test_aggregate_windows():
    rows = [{"out_share": 0.1, "faces": 2, "faces_ghost": 1, "face_dist_w": 0.5, "head_fid": 0.12},
            {"out_share": 0.0, "faces": 1, "faces_ghost": 0, "face_dist_w": 0.0, "head_fid": 0.02}]
    a = G.aggregate(rows)
    assert a["n"] == 2 and a["fedelta_testa"] == pytest.approx(0.07) and a["frame_con_volto_fantasma"] == 0.5 and G.aggregate([]) == {"n": 0}
