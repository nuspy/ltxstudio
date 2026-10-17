"""Regressione 14/10/2026: il ramo inpaint deve passare `control_video` al worker (senza, ltx2_infer degrada in
silenzio a ti2v: niente LoRA, guide o maschera). Dal task 8 la logica sta in core/video_inpaint.py (condivisa con
la REST API); controllo statico sul sorgente di entrambi."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "core" / "video_inpaint.py").read_text(encoding="utf-8")
MW = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")


def test_inpaint_imposta_control_video():
    assert 'params["inpaint_masks_dir"]' in SRC
    assert 'params["control_video"] = src' in SRC, "il tracking non imposta control_video"


def test_gui_usa_il_modulo_condiviso():
    a = MW.index('if cfg["mode"] == "inpaint":')
    b = MW.index('elif cfg["mode"] == "replace_char"', a)
    assert "attach_tracking" in MW[a:b] and "inpaint_params" in MW[a:b]
