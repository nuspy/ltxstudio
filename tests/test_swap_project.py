"""core/swap_project: ricerca persone orchestrata (funzioni iniettate), doppioni, validazione dei ruoli, unione maschere, passate in catena."""
import os

import numpy as np
import pytest
from PIL import Image

from core import swap_project as S


def _masks(d, frames=3, box=(2, 2, 6, 6), size=(16, 16)):
    os.makedirs(d, exist_ok=True)
    for f in range(frames):
        m = np.zeros(size[::-1], np.uint8)
        m[box[1]:box[3], box[0]:box[2]] = 255
        Image.fromarray(m).save(os.path.join(d, f"{f:05d}.png"))


def test_dedupe_keeps_best_and_drops_same_person_twice():
    b = [[0, 0, 10, 10, 0.5], [1, 1, 10, 10, 0.9], [50, 50, 60, 60, 0.4]]
    k = S.dedupe_boxes(b)
    assert [x[4] for x in k] == [0.9, 0.4]
    assert len(S.dedupe_boxes([[i * 20, 0, i * 20 + 10, 10, 1.0 - i / 10] for i in range(8)])) == S.MAX_ROLES


def test_find_people_orders_by_area_and_names_them(tmp_path):
    areas = {0: 0.1, 1: 0.5}
    def detect(v):
        return [[0, 0, 5, 5, 0.9], [30, 30, 60, 60, 0.8]]
    calls = []
    def track(v, box, md):
        calls.append(box)
        _masks(md)
        return {"frames": 3, "area": 0.5 if box[0] == 30 else 0.1}
    tr = S.find_people("v.mp4", str(tmp_path), detect_boxes=detect, track_box=track, label_of=lambda t: "uomo con occhiali" if t.id == 1 else "")
    assert [t.id for t in tr] == [1, 2] and tr[0].area > tr[1].area                 # la piu grande e la «principale»
    assert tr[0].label == "uomo con occhiali" and tr[1].label == "Persona 2"
    assert len(calls) == 2
    S.save_tracks(tr, str(tmp_path / "tracks.json"))
    assert [t.label for t in S.load_tracks(str(tmp_path / "tracks.json"))] == ["uomo con occhiali", "Persona 2"]


def test_find_people_raises_when_nobody_found(tmp_path):
    with pytest.raises(S.SwapError, match="Nessuna persona"):
        S.find_people("v.mp4", str(tmp_path), detect_boxes=lambda v: [], track_box=lambda *a: {})


def _tracks(tmp_path):
    out = []
    for i in (1, 2, 3):
        d = tmp_path / f"t{i}" / "masks"
        _masks(str(d), box=(i, i, i + 4, i + 4))
        out.append(S.Track(id=i, box=[0, 0, 1, 1], mask_dir=str(d), frames=3, area=0.1 * i))
    return out


def test_validate_catches_every_user_mistake(tmp_path):
    tr = _tracks(tmp_path)
    photo = tmp_path / "p.png"
    Image.new("RGB", (4, 4)).save(photo)
    assert S.validate([], tr)
    e = S.validate([S.Role("A", [1], str(photo)), S.Role("B", [1], str(photo))], tr)
    assert any("assegnata sia a" in x for x in e)
    assert any("manca la foto" in x for x in S.validate([S.Role("A", [1], "")], tr))
    assert any("non esiste" in x for x in S.validate([S.Role("A", [9], str(photo))], tr))
    assert any("scegli quale persona" in x for x in S.validate([S.Role("A", [], str(photo))], tr))
    assert any("Al massimo" in x for x in S.validate([S.Role(f"R{i}", [1], str(photo)) for i in range(5)], tr))
    assert S.validate([S.Role("A", [1], str(photo)), S.Role("B", [2, 3], str(photo))], tr) == []


def test_plan_passes_merges_masks_of_a_role_and_chains_sources(tmp_path):
    tr = _tracks(tmp_path)
    photo = tmp_path / "p.png"
    Image.new("RGB", (4, 4)).save(photo)
    ps = S.plan_passes([S.Role("A", [1], str(photo)), S.Role("B", [2, 3], str(photo), "giacca rossa")], tr, str(tmp_path / "w"))
    assert [p.role for p in ps] == ["A", "B"] and ps[0].mask_dir == tr[0].mask_dir          # una traccia: nessuna copia
    m = np.asarray(Image.open(os.path.join(ps[1].mask_dir, "00000.png"))) > 127
    assert m[2, 2] and m[6, 6] and m.sum() > 16                                              # unione di due persone
    assert ps[1].note == "giacca rossa"
    assert S.source_for_pass(0, "orig.mp4", []) == "orig.mp4" and S.source_for_pass(1, "orig.mp4", ["out1.mp4"]) == "out1.mp4"
    with pytest.raises(S.SwapError):
        S.plan_passes([S.Role("A", [], "")], tr, str(tmp_path / "w2"))
