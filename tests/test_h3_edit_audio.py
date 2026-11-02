"""workers/h3_edit_run.to_stereo — l'audio sorgente mono (video parlato) diventa stereo per encode_video e per l'audio VAE di H3."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
import h3_edit_run as ER  # noqa: E402


def test_mono_is_duplicated_to_stereo():
    mono = torch.arange(10, dtype=torch.float32)[None]
    st = ER.to_stereo(mono)
    assert st.shape == (2, 10) and torch.equal(st[0], st[1]) and torch.equal(st[0], mono[0])


def test_stereo_unchanged_multichannel_cut_and_1d():
    st = torch.randn(2, 7)
    assert torch.equal(ER.to_stereo(st), st)
    assert ER.to_stereo(torch.randn(6, 5)).shape == (2, 5)
    assert ER.to_stereo(torch.randn(9)).shape == (2, 9)
