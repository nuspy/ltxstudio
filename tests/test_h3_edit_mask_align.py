"""H3 edit: la maschera di ogni frame a 24 fps e quella DELLO STESSO frame sorgente (bug del 03/11: maschere stirate sui frame usati)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
import h3_edit_run as ER  # noqa: E402


def _masks(tmp_path, n):
    # maschera i = colonna i accesa: dal contenuto si legge quale file e stato scelto
    for i in range(n):
        a = np.zeros((4, 32), np.uint8)
        a[:, i] = 255
        Image.fromarray(a).save(tmp_path / f"{i:05d}.png")


def _which(m):
    return [int(np.argmax(f.numpy().mean(0))) for f in m]


def test_truncated_video_keeps_frame_alignment(tmp_path):
    _masks(tmp_path, 30)                                     # 30 frame sorgente, se ne usano solo i primi 21
    m = ER.read_masks(str(tmp_path), np.arange(21), 30, 32, 4)
    assert _which(m) == list(range(21))                      # prima: indice k*29/20, a fine clip 8-9 frame avanti


def test_resampled_source_follows_source_index(tmp_path):
    _masks(tmp_path, 30)
    idx = np.array([0, 1, 1, 2, 3, 3, 4])                    # sorgente a ~20 fps letto a 24: frame ripetuti
    assert _which(ER.read_masks(str(tmp_path), idx, 30, 32, 4)) == idx.tolist()


def test_mask_count_differs_maps_by_time(tmp_path):
    _masks(tmp_path, 16)                                     # tracking a passo diverso: 16 maschere per 31 frame sorgente
    m = ER.read_masks(str(tmp_path), np.array([0, 15, 30]), 31, 32, 4)
    assert _which(m) == [0, 8, 15]
