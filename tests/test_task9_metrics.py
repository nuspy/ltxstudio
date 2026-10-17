"""Le metriche del task 9 distinguono il caso buono dal cattivo (verifica delle metriche, non dei modelli)."""
from __future__ import annotations

import os

import pytest
from PIL import Image, ImageDraw

import task9_metrics as M


def _shape(path, shift=0, shade=200):
    im = Image.new("RGB", (256, 256), (20, 20, 20))
    d = ImageDraw.Draw(im)
    d.ellipse((60 + shift, 60, 190 + shift, 190), outline=(shade,) * 3, width=4)
    d.rectangle((100 + shift, 100, 150 + shift, 150), outline=(shade,) * 3, width=3)
    im.save(path)
    return str(path)


def test_edge_iou_same_vs_shifted(tmp_path):
    a = _shape(tmp_path / "a.png")
    same = _shape(tmp_path / "b.png", shade=150)             # stessa sagoma, tono diverso
    moved = _shape(tmp_path / "c.png", shift=60)             # sagoma spostata
    assert M.edge_iou(a, same) > 0.9
    assert M.edge_iou(a, moved) < 0.5


def test_psnr_outside_mask(tmp_path):
    base = Image.new("RGB", (64, 64), (90, 120, 150)); base.save(tmp_path / "o.png")
    mk = Image.new("L", (64, 64), 0); mk.paste(255, (16, 16, 48, 48)); mk.save(tmp_path / "m.png")
    res = base.copy(); res.paste((255, 0, 0), (20, 20, 40, 40)); res.save(tmp_path / "r.png")         # cambia solo DENTRO la maschera
    assert M.psnr_outside_mask(tmp_path / "o.png", tmp_path / "r.png", tmp_path / "m.png") == float("inf")
    res.paste((0, 0, 0), (0, 0, 10, 10)); res.save(tmp_path / "r2.png")                                  # cambia FUORI
    assert M.psnr_outside_mask(tmp_path / "o.png", tmp_path / "r2.png", tmp_path / "m.png") < 40


def test_pose_agreement_on_real_photo():
    p = os.path.join(os.path.dirname(__file__), "data", "openpose_person.jpg")
    if not os.path.isfile(p) or not os.path.isfile(r"E:\AI\Models\ControlNet\Openpose\body_pose_model.pth"):
        pytest.skip("foto o pesi OpenPose assenti")
    ok, n, err = M.pose_agreement(p, p)
    assert n >= 10 and ok == n and err < 1.0                 # la foto con se' stessa: tutti i punti coincidono
