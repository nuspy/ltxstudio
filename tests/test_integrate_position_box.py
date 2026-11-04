"""Integra con Krea 2 Edit: il posto del soggetto come riquadro rosso sulla scena (prove 04/11). CPU, immagini sintetiche."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import integrate_position_box as ipb  # noqa: E402


def _files(tmp_path, mask_box):
    scene = tmp_path / "scena.png"
    Image.new("RGB", (400, 300), (20, 90, 20)).save(scene)
    mask = np.zeros((150, 200), np.uint8)                       # maschera a META risoluzione: il riquadro va riportato alla scena
    if mask_box:
        x0, y0, x1, y1 = mask_box
        mask[y0:y1, x0:x1] = 255
    path = tmp_path / "posizione.png"
    Image.fromarray(mask, "L").save(path)
    return str(scene), str(path)


def test_box_is_drawn_around_the_silhouette_at_scene_scale(tmp_path):
    scene, mask = _files(tmp_path, (60, 30, 90, 120))
    out, box = ipb.draw_position_box(scene, mask, str(tmp_path / "riquadro.png"))
    assert box[0] in range(118, 123) and box[1] in range(58, 62) and box[2] in range(176, 182) and box[3] in range(236, 242)
    image = np.asarray(Image.open(out))
    assert tuple(image[box[1], (box[0] + box[2]) // 2]) == ipb.BOX_COLOR                     # il contorno e rosso
    assert tuple(image[(box[1] + box[3]) // 2, (box[0] + box[2]) // 2]) == (20, 90, 20)       # dentro la scena e intatta
    assert tuple(image[5, 5]) == (20, 90, 20)


def test_empty_silhouette_gives_no_box_and_leaves_the_scene_alone(tmp_path):
    scene, mask = _files(tmp_path, None)
    assert ipb.draw_position_box(scene, mask, str(tmp_path / "x.png")) is None
    assert not (tmp_path / "x.png").exists()


def test_prompt_puts_the_person_inside_the_rectangle_and_forbids_drawing_it():
    text = ipb.box_prompt("on the left side", "dusk light")
    assert "inside the red rectangle (on the left side of the scene)" in text and "do not draw it" in text and text.endswith("dusk light")


def test_krea_flow_uses_the_boxed_scene_only_when_the_option_is_on():
    source = (Path(__file__).resolve().parents[1] / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert 'getattr(self.frame1_tab.canvas, "integrate_position_box", False)' in source
    assert 'self._start_native_krea2_edit([_scene, exp["layer_refs"][0]]' in source
