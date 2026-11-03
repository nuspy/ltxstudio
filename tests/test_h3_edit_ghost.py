"""Doppione/lip sync del clip 1 (giudizio 03/11): preroll, riferimento «manichino» e matte a pixel nel cambia-sfondo (opzioni di prova, default invariati)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
import h3_edit_math as M  # noqa: E402

VENV_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"


def test_preroll_pads_frames_masks_and_silent_audio_on_the_17_grid():
    frames = torch.randint(0, 255, (141, 8, 8, 3), dtype=torch.uint8)
    masks = torch.rand(141, 8, 8)
    wav = torch.ones(2, 188000)
    f2, m2, w2 = M.preroll_pad(frames, masks, wav, 17, 32000)
    assert f2.shape[0] == 158 == M.usable_frames(158) and m2.shape[0] == 158
    assert torch.equal(f2[:17], frames[:1].expand(17, -1, -1, -1)) and torch.equal(m2[:17], masks[:1].expand(17, -1, -1))
    assert torch.equal(f2[17:], frames) and torch.equal(m2[17:], masks)
    n = int(round(17 / 24 * 32000))
    assert w2.shape[1] == 188000 + n and float(w2[:, :n].abs().max()) == 0.0 and torch.equal(w2[:, n:], wav)          # audio SILENZIOSO, poi l'originale
    assert M.preroll_pad(frames, masks, wav, 0, 32000)[0] is frames                                                    # default: nessun cambiamento
    with pytest.raises(ValueError, match="multiplo di 17"):
        M.preroll_pad(frames, masks, wav, 10, 32000)


def test_pixel_matte_keeps_original_inside_the_person_and_generated_outside():
    comp = torch.zeros(2, 20, 20, 3)
    orig = torch.ones(2, 20, 20, 3)
    person = torch.zeros(2, 20, 20)
    person[:, 5:15, 5:15] = 1
    out = M.person_matte(comp, orig, person, soft_px=1.0)
    assert torch.allclose(out[:, 8:12, 8:12], torch.ones(2, 4, 4, 3), atol=1e-3)         # dentro: originale
    assert torch.allclose(out[:, :2, :2], torch.zeros(2, 2, 2, 3), atol=1e-3)           # fuori: generato
    assert 0.1 < float(out[0, 10, 5, 0]) < 0.9                                              # bordo morbido
    hard = M.person_matte(comp, orig, person, soft_px=0.0)
    assert float(hard[0, 4, 10, 0]) == 0.0 and float(hard[0, 5, 10, 0]) == 1.0


def test_options_are_off_by_default_and_not_in_the_checkpoint_key_unless_they_change_the_clips():
    src = (ROOT / "workers" / "h3_edit_run.py").read_text(encoding="utf-8")
    assert '"H3_EDIT_PREROLL") or e.get("preroll_frames") or 0' in src
    assert 'or e.get("bg_person_ref") or "sharp"' in src
    assert 'str(os.environ.get("H3_EDIT_BG_MATTE") or e.get("bg_matte") or "") not in ("", "0")' in src
    assert '"bgref", bg_person_ref' in src and '"preroll", preroll' in src                 # cambiano i clip: nella chiave
    key_line = [l for l in src.splitlines() if l.strip().startswith('"maskalign_v2"')][0] + src.split('"maskalign_v2"')[1].split("*opts_key(p))")[0]
    assert "bg_matte" not in key_line                                                         # il matte e solo composizione: i clip si riusano


@pytest.mark.skipif(not os.path.isfile(VENV_PY), reason="serve il venv di H3 (diffusers_next_venv)")
def test_ghost_options_end_to_end_smoke_with_a_fake_worker():
    r = subprocess.run([VENV_PY, "-I", str(ROOT / "tests" / "h3_edit_run_ghost_smoke.py")], capture_output=True, text=True, timeout=900,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 0 and "GHOST SMOKE OK" in r.stdout, r.stdout[-1200:] + r.stderr[-1200:]
