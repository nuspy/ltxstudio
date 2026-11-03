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


def test_tone_dump_is_off_by_default_and_never_raises(tmp_path, monkeypatch):
    from PIL import Image
    from inference_server.backends import image_flux as F
    monkeypatch.delenv("NATIVE_TONE_DEBUG", raising=False)
    monkeypatch.delenv("LTX_TEST_DRIVER", raising=False)
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    F._tone_dump(str(tmp_path), {"arch": "inpaint", "seed": 5}, "raw", img)
    assert not (tmp_path / "_tone_debug").exists()                     # funzionamento normale: nessun file in piu
    monkeypatch.setenv("NATIVE_TONE_DEBUG", "1")
    F._tone_dump(str(tmp_path), {"arch": "inpaint", "seed": 5}, "raw", img)
    assert (tmp_path / "_tone_debug" / "inpaint_5_raw.png").is_file()
    F._tone_dump(str(tmp_path / "x\0y"), {}, "raw", img)               # percorso impossibile: nessuna eccezione


def test_guard_load_frees_parked_monolithic_pipelines(monkeypatch):
    """Giro A 03/11: una Klein 8B parcheggiata in mm._cpu_cache (~26 GB) restava mentre si caricava la Klein 9B (59 GB di processo): il guard di
    caricamento ora libera anche le pipeline monolitiche parcheggiate quando il carico e grosso."""
    import types
    from inference_server.backends import image_flux as F

    class E:
        def __init__(self, mb):
            self.obj, self.vram_mb = object(), mb

    mm = types.SimpleNamespace(_cpu_cache={"a": E(26000), "b": E(9000)}, flush=lambda *a, **k: None)
    monkeypatch.setattr(F, "_predicted_load_mb", lambda *a, **k: 22000.0)          # >= soglia «carico grosso» (15000)
    from core import mem_budget
    monkeypatch.setattr(mem_budget, "check_load", lambda *a, **k: None)
    F._guard_load("x.safetensors", "flux2_inpaint", mm)
    assert mm._cpu_cache == {}
    mm2 = types.SimpleNamespace(_cpu_cache={"a": E(1000)}, flush=lambda *a, **k: None)
    monkeypatch.setattr(F, "_predicted_load_mb", lambda *a, **k: 3000.0)           # piccolo e commit abbondante: non tocca nulla
    monkeypatch.setattr(mem_budget, "avail_commit_mb", lambda: 90000.0)
    F._guard_load("x.safetensors", "flux", mm2)
    assert len(mm2._cpu_cache) == 1


def test_integrate_context_mask_is_wider_than_position_mask_and_default_is_unchanged(tmp_path):
    """Integra: la maschera di POSIZIONE (sagoma) e il CONTESTO (zona rigenerabile) sono separate; il contesto si allarga dilatando la stessa forma;
    senza parametro il tracciato e quello di sempre (nessun cambiamento per i casi esistenti)."""
    import numpy as np
    from PIL import Image
    fx = FX
    if not (fx / "scena_1152x832.png").exists() or not (Path.home() / ".u2net" / "u2net.onnx").exists():
        pytest.skip("servono le foto di prova e u2net")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from ui.mask_canvas import MaskCanvas
    areas = {}
    for name, pct in (("base", None), ("p25", 0.25), ("p60", 0.60)):
        cv = MaskCanvas()
        cv.load_image(str(fx / "scena_1152x832.png"), "canvas")
        cv.add_layer(str(fx / "ritratto_960x1024.png"))
        cv.integrate_context_pct = pct
        d = tmp_path / name
        r = cv.export_integrate(d)
        ctx = np.asarray(Image.open(r["mask"]).convert("L")) > 127
        pos = np.asarray(Image.open(r["position_mask"]).convert("L")) > 127
        areas[name] = (int(ctx.sum()), int(pos.sum()))
        assert not (pos & ~ctx).any(), "la posizione deve stare dentro il contesto"
        assert pos.any() and r["context_px"] == (0 if pct is None else r["context_px"])
    assert areas["base"][1] == areas["p25"][1] == areas["p60"][1]                 # la posizione non cambia
    assert areas["base"][0] < areas["p25"][0] < areas["p60"][0]                  # il contesto cresce
