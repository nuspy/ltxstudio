"""Integra: il volto del soggetto incollato resta fuori dalla zona rigenerata (prove 04/11: volto 33 px, somiglianza 0,82 incollato contro 0,05-0,34 dopo armonizza Qwen / Klein ref-inpaint).
Prova su CPU con maschere sintetiche e un rilevatore finto (nessun insightface)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import integrate_face_hold as hold  # noqa: E402

SIZE = (400, 300)                      # larghezza, altezza dell'immagine di prova
FACE = [190.0, 40.0, 220.0, 80.0]      # riquadro del volto del soggetto
OTHER_FACE = [40.0, 40.0, 70.0, 80.0]  # un volto della scena, fuori dalla sagoma del soggetto


def _fake_runner(faces_by_image):
    def run(argv, stdin):
        paths = json.loads(stdin)["images"]
        return "rumore\n@@JSON@@" + json.dumps({"res": [faces_by_image[Path(p).name] for p in paths]})
    return run


def _write(tmp_path, name, array, mode="L"):
    path = tmp_path / name
    Image.fromarray(array.astype(np.uint8), mode).save(path)
    return str(path)


def _scene(tmp_path):
    """Composito 400x300; sagoma del soggetto = rettangolo 160-260 x 30-280; maschera generativa = sagoma + contesto (130-290 x 10-290)."""
    subject = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    subject[30:280, 160:260] = 255
    context = np.zeros_like(subject)
    context[10:290, 130:290] = 255
    image = _write(tmp_path, "composito.png", np.full((SIZE[1], SIZE[0], 3), 90, np.uint8), "RGB")
    reference = _write(tmp_path, "soggetto.png", np.full((200, 100, 3), 150, np.uint8), "RGB")
    return {"image": image, "mask": _write(tmp_path, "contesto.png", context), "position_mask": _write(tmp_path, "posizione.png", subject), "layer_refs": [reference]}


def test_hold_map_is_one_on_the_grown_face_and_zero_far_away_with_a_soft_edge():
    held = hold.hold_map((SIZE[1], SIZE[0]), [FACE])
    cx, cy = 205, 60
    assert held[cy, cx] == 1.0 and held[cy, int(FACE[0]) - 4] == 1.0               # dentro l'ellisse del riquadro +30 %
    assert held[200, 205] == 0.0 and held[60, 20] == 0.0                           # lontano: niente
    row = held[cy, 205:]
    assert np.all(np.diff(row) <= 1e-6) and 0.0 < row[28] < 1.0     # monotona e con un bordo sfumato


def test_mask_without_the_subject_face_keeps_everything_else_generative(tmp_path):
    export = _scene(tmp_path)
    out = hold.hold_faces_in_mask(export["mask"], [FACE], SIZE, str(tmp_path / "tenuta.png"))
    before, after = np.asarray(Image.open(export["mask"])), np.asarray(Image.open(out))
    assert after[60, 205] == 0 and before[60, 205] == 255                          # il volto non si rigenera piu
    assert (after[150:, :] == before[150:, :]).all()                               # il resto del corpo e del contesto e invariato
    assert (after <= before).all()                                                 # la maschera puo solo restringersi


def test_mask_is_unchanged_when_no_face_is_given(tmp_path):
    export = _scene(tmp_path)
    out = hold.hold_faces_in_mask(export["mask"], [], SIZE, str(tmp_path / "uguale.png"))
    assert (np.asarray(Image.open(out)) == np.asarray(Image.open(export["mask"]))).all()


def test_only_the_face_inside_the_pasted_silhouette_is_held(tmp_path):
    export = _scene(tmp_path)
    subject = np.asarray(Image.open(export["position_mask"])) > 127
    faces = [{"bbox": FACE, "score": 0.9}, {"bbox": OTHER_FACE, "score": 0.9}]
    assert hold.boxes_inside_mask(faces, subject, SIZE) == [FACE]
    assert hold.boxes_inside_mask(faces, subject, (800, 600)) == []                # riquadri a scala diversa: il centro non cade piu nella sagoma


def test_hold_face_of_subject_returns_the_held_mask_and_the_face_crop_for_klein(tmp_path):
    export = _scene(tmp_path)
    runner = _fake_runner({"composito.png": [{"bbox": FACE, "score": 0.9}, {"bbox": OTHER_FACE, "score": 0.9}],
                           "soggetto.png": [{"bbox": [30.0, 20.0, 70.0, 70.0], "score": 0.99}]})
    logs = []
    result = hold.hold_face_of_subject(export, str(tmp_path), logs.append, runner)
    assert Image.open(result["mask"]).getpixel((205, 60)) == 0 and Image.open(export["mask"]).getpixel((205, 60)) == 255
    assert result["face_crop"] and Image.open(result["face_crop"]).size[0] > 40
    assert any("volto da tenere" in line for line in logs) and result["image"] == export["image"]


def test_when_the_face_is_not_found_the_export_is_unchanged_and_the_log_says_so(tmp_path):
    export = _scene(tmp_path)
    logs = []
    result = hold.hold_face_of_subject(export, str(tmp_path), logs.append, _fake_runner({"composito.png": [{"bbox": OTHER_FACE, "score": 0.9}]}))
    assert result == export and "nessun volto trovato dentro la sagoma" in logs[0]


def test_detector_failure_is_an_explicit_error():
    def broken(argv, stdin):
        return "nessuna risposta"
    with pytest.raises(RuntimeError, match="senza risposta"):
        hold.detect_face_boxes(["x.png"], broken)


def test_the_integra_flow_wires_the_option_for_the_three_approaches():
    source = (Path(__file__).resolve().parents[1] / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert 'getattr(cv, "integrate_hold_face", False) and appr in ("armonizza", "klein_ref_inpaint", "klein_armonizza")' in source
    assert 'params["ref_images"] = list(exp["layer_refs"][:1]) + ([exp["face_crop"]] if exp.get("face_crop") else [])' in source
