"""Correzioni dai commenti alle revisioni (03/11): Integra senza cornice bianca, Componi Kontext (misura/prompt), maschera di prova sul tavolo."""
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FX = Path(r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live\fixtures")


def test_kontext_compose_cites_pictures_by_position():
    from core import image_refs as R
    out = R.translate("place the woman from @image_2 in @image_1", {"image": [1, 2]}, "kontext_stitch")
    assert out == "place the woman from the second picture from the left in the first picture from the left"
    assert R.MODEL_SYNTAX["kontext_stitch"]["verified"] is False


def test_compose_output_is_not_stretched_back_to_the_stitch_size():
    """image_flux: con width/height espliciti (Componi) l'uscita resta alla misura del canvas, non torna a quella dello stitch (deformava)."""
    src = (ROOT / "inference_server" / "backends" / "image_flux.py").read_text(encoding="utf-8")
    assert len(re.findall(r"_orig_wh = \(_W, _H\)", src)) == 2           # stitch Kontext + multi-riferimento Klein


def test_driver_inpaint_mask_is_on_the_table_not_in_the_air():
    """La maschera di prova «bussola sul tavolo»: ellisse nella fascia bassa della foto (tavolo) (non il cerchio al centro del 31/10)."""
    src = (ROOT / "tests" / "app_live" / "driver.py").read_text(encoding="utf-8")
    m = re.search(r"case\.get\(\"mask_ellipse\"\) or \(([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\)", src)
    fx, fy, frx, fry = map(float, m.groups())
    assert fy - fry > 0.83 and fy + fry < 1.0                           # sotto i piattini (bordo basso a y=0.83) e dentro la foto
    assert 0.3 < fx < 0.7


@pytest.mark.skipif(not (Path.home() / ".u2net" / "u2net.onnx").exists() or not FX.exists(), reason="serve il modello u2net locale e le foto di prova")
def test_integrate_export_has_no_white_frame_with_opaque_photo(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    import numpy as np
    from PIL import Image
    from ui.mask_canvas import MaskCanvas
    cv = MaskCanvas()
    cv.load_image(str(FX / "scena_1152x832.png"), "canvas")
    assert cv.add_layer(str(FX / "ritratto_960x1024.png"))
    r = cv.export_integrate(tmp_path)
    im = np.asarray(Image.open(r["image"]).convert("RGB"))
    white = im.min(axis=2) >= 250
    assert white.mean() < 0.01, "cornice bianca attorno alla foto incollata"
    assert cv._entries[1].get("cutout") is True
    assert r["has_region"]


def test_click_selection_keeps_a_person_that_fills_most_of_the_frame():
    """Misure SAM2 su CPU (03/11): ritratto, clic sul busto: maschere 15,0 / 64,1 / 30,7 % -> la persona intera (64 %), non la sola pelle."""
    from inference_server.backends.image_segment import pick_point_mask
    T = 1000
    assert pick_point_mask([150, 641, 307], [0.445, 0.688, 0.824], T) == 1
    assert pick_point_mask([18, 152, 62], [0.17, 0.54, 0.62], T) == 1            # polpo cliccato sulla testa: polpo + giacca, non solo la testa
    assert pick_point_mask([20, 990, 30], [0.2, 0.9, 0.5], T) == 2              # una maschera = scena intera (99 %) non e mai l'oggetto
    assert pick_point_mask([950, 960, 990], [0.1, 0.9, 0.5], T) == 1            # tutte enormi: ripiego sul punteggio SAM
