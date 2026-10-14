"""Regressione 14/10/2026: il ramo inpaint di _VideoEditAIWorker.run deve passare `control_video` al worker
(senza, ltx2_infer degrada in silenzio a ti2v: niente LoRA, guide o maschera). Controllo statico sul sorgente."""
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "ui" / "main_window.py").read_text(encoding="utf-8")


def test_inpaint_imposta_control_video():
    a = SRC.index('params["inpaint_masks_dir"]')
    b = SRC.index('elif cfg["mode"] == "replace_char"', a)
    assert 'params["control_video"] = src' in SRC[a:b], "il ramo inpaint non imposta control_video"
