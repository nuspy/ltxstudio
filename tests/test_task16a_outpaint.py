"""Task 16a — outpaint: contorno guida del frame fuori dall'export + pre-riempimento dell'estensione (solo CPU)."""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import outpaint_prefill as opf

SRC = Path(r"E:\ai\ltxstudio\output")
FILL = np.array([20, 20, 28])


def _photo(w=704, h=704, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([120 + 80 * np.sin(xx / 90.0), 110 + 70 * np.cos(yy / 70.0), 140 + 50 * np.sin((xx + yy) / 120.0)], -1)
    img += rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _frame(photo, scale=1.4, fill=(20, 20, 28)):
    w, h = photo.size
    W, H = int(round(w * scale / 16)) * 16, int(round(h * scale / 16)) * 16
    x0, y0 = (W - w) // 2, (H - h) // 2
    fr = Image.new("RGB", (W, H), fill)
    fr.paste(photo, (x0, y0))
    m = np.ones((H, W), bool)
    m[y0:y0 + h, x0:x0 + w] = False
    return fr, m, (x0, y0, x0 + w, y0 + h)


def _seam_step(img, rect, band=6):
    """Differenza media (livelli 0..255) fra la striscia di `band` px dentro la foto e quella subito fuori, sui 4 lati."""
    a = np.asarray(img).astype(np.float32)
    x0, y0, x1, y1 = rect
    pairs = [(a[y0:y1, x0:x0 + band], a[y0:y1, x0 - band:x0]), (a[y0:y1, x1 - band:x1], a[y0:y1, x1:x1 + band]),
             (a[y0:y0 + band, x0:x1], a[y0 - band:y0, x0:x1]), (a[y1 - band:y1, x0:x1], a[y1:y1 + band, x0:x1])]
    return float(np.mean([abs(i.mean() - o.mean()) for i, o in pairs]))


def test_foto_identica_e_estensione_riempita():
    ph = _photo()
    fr, m, rect = _frame(ph)
    out = opf.prefill_extension(fr, m, seed=1)
    a, b = np.asarray(fr), np.asarray(out)
    assert out.size == fr.size
    assert (a[~m] == b[~m]).all(), "i pixel della foto devono restare identici"
    assert np.abs(b[m].astype(int) - FILL).mean() > 40, "l'estensione deve smettere di essere la tinta scura"
    # continuita' al confine: molto meglio del salto contro la tinta scura
    assert _seam_step(out, rect) < 0.35 * _seam_step(fr, rect)
    assert _seam_step(out, rect) < 25


def test_deterministico_e_senza_estensione():
    ph = _photo(320, 240)
    fr, m, _ = _frame(ph)
    a = np.asarray(opf.prefill_extension(fr, m, seed=7))
    b = np.asarray(opf.prefill_extension(fr, m, seed=7))
    assert (a == b).all()
    assert opf.prefill_extension(ph, np.zeros((ph.height, ph.width), bool)) is ph      # niente da riempire -> invariata
    assert opf.prefill_extension(ph, np.ones((ph.height, ph.width), bool)) is ph       # niente foto -> invariata


def test_maschera_non_rettangolare_usa_il_campo_liscio():
    ph = _photo(256, 256)
    fr, m, _ = _frame(ph)
    m2 = m.copy()
    m2[100:140, 100:140] = True            # buco interno: foto non rettangolare
    out = np.asarray(opf.prefill_extension(fr, m2, seed=2)).astype(int)
    assert np.abs(out[m2] - FILL).mean() > 30
    assert (out[~m2] == np.asarray(fr).astype(int)[~m2]).all()


def test_interruttore_ambiente(monkeypatch):
    monkeypatch.setenv("NATIVE_OUTPAINT_PREFILL", "0")
    assert not opf.enabled()
    monkeypatch.delenv("NATIVE_OUTPAINT_PREFILL")
    assert opf.enabled()


@pytest.mark.parametrize("name", ["native_inpaint_5.png", "native_inpaint_12.png", "native_inpaint_1088515480.png"])
def test_foto_reali_dei_casi_live(name):
    p = SRC / name
    if not p.exists():
        pytest.skip("sorgente del caso live assente")
    fr, m, rect = _frame(Image.open(p).convert("RGB"))
    out = opf.prefill_extension(fr, m, seed=4242)
    assert _seam_step(out, rect) < 0.5 * _seam_step(fr, rect)


def test_export_canvas_senza_contorno_guida(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])          # noqa: F841
    from ui.mask_canvas import MaskCanvas
    src = tmp_path / "p.png"
    _photo(320, 320).save(src)
    cv = MaskCanvas()
    cv.load_image(str(src), fit_mode="canvas")
    nw, nh = int(round(cv.aspect_w * 1.4 / 16)) * 16, int(round(cv.aspect_h * 1.4 / 16)) * 16
    cv.set_aspect(nw, nh)
    pm = cv.img_item.pixmap()
    s = cv.img_item.scale()
    cv.img_item.setPos((nw - pm.width() * s) / 2, (nh - pm.height() * s) / 2)
    pen_before = cv.frame_item.pen()
    res = cv.export_inpaint(tmp_path / "e", "outpaint")
    im = np.asarray(Image.open(res["image"]).convert("RGB")).astype(int)
    for edge in (im[0:4, :], im[-4:, :], im[:, 0:4], im[:, -4:]):       # bordi del frame: solo il riempimento, niente riga blu
        assert (edge == FILL).all()
    assert cv.frame_item.pen() == pen_before, "il contorno guida a video va ripristinato dopo l'export"
    assert cv.frame_item.pen().color().name() == "#4466aa"
    # nessun contorno attorno alla foto: subito fuori dal riquadro c'e' solo il riempimento
    x0, y0 = int(round((nw - pm.width() * s) / 2)), int(round((nh - pm.height() * s) / 2))
    assert (im[y0 + 40, x0 - 3:x0] == FILL).all()


def test_worker_qwen_usa_il_prefill_in_outpaint():
    src = (Path(__file__).resolve().parents[1] / "workers" / "qwen_image21_infer.py").read_text(encoding="utf-8")
    assert 'p.get("outpaint")' in src and "outpaint_prefill.prefill_from_mask_path" in src
    assert src.index("outpaint_prefill.prefill_from_mask_path") < src.index("_annotate(init_img, mask_path)")
