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


def test_outpaint_strength_costante_e_override(monkeypatch):
    monkeypatch.delenv("NATIVE_OUTPAINT_STRENGTH", raising=False)
    assert opf.outpaint_strength(1.0) == pytest.approx(0.9)
    assert opf.outpaint_strength(None) == pytest.approx(0.9)
    assert opf.outpaint_strength(0.6) == pytest.approx(0.6)          # mai alzata
    monkeypatch.setenv("NATIVE_OUTPAINT_STRENGTH", "1.0")
    assert opf.outpaint_strength(1.0) == pytest.approx(1.0)
    monkeypatch.setenv("NATIVE_OUTPAINT_STRENGTH", "0.8")
    assert opf.outpaint_strength(1.0) == pytest.approx(0.8)
    monkeypatch.setenv("NATIVE_OUTPAINT_STRENGTH", "boh")
    assert opf.outpaint_strength(1.0) == pytest.approx(0.9)


def _worker():
    import importlib, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
    return importlib.import_module("qwen_image21_infer")


def test_worker_qwen_helper_outpaint(tmp_path, monkeypatch):
    w = _worker()
    ph = _photo(256, 256)
    fr, m, _ = _frame(ph)
    mp = tmp_path / "m.png"
    Image.fromarray((m * 255).astype(np.uint8)).save(mp)
    monkeypatch.delenv("NATIVE_OUTPAINT_PREFILL", raising=False)
    monkeypatch.delenv("NATIVE_OUTPAINT_STRENGTH", raising=False)
    # outpaint: il riferimento NON e' piu' il frame scuro
    out = w._outpaint_init(fr, str(mp), {"outpaint": True, "seed": 5})
    assert np.abs(np.asarray(out).astype(int)[m] - FILL).mean() > 40
    assert (np.asarray(out)[~m] == np.asarray(fr)[~m]).all()
    # inpaint / senza maschera / prefill spento: invariato
    assert w._outpaint_init(fr, str(mp), {"outpaint": False}) is fr
    assert w._outpaint_init(fr, "", {"outpaint": True}) is fr
    monkeypatch.setenv("NATIVE_OUTPAINT_PREFILL", "0")
    assert w._outpaint_init(fr, str(mp), {"outpaint": True}) is fr
    assert w._outpaint_strength(1.0, {"outpaint": True}) == 1.0
    monkeypatch.delenv("NATIVE_OUTPAINT_PREFILL")
    assert w._outpaint_strength(1.0, {"outpaint": True}) == pytest.approx(0.9)
    assert w._outpaint_strength(1.0, {}) == 1.0                       # inpaint: forza invariata
    assert w._outpaint_feather((992, 992), {"outpaint": True}) == 25
    assert w._outpaint_feather((992, 992), {}) == 7
    assert w._outpaint_feather((992, 992), {"outpaint": True, "feather": 9}) == 9


def test_worker_qwen_paste_back_outpaint_maschera_larga(tmp_path, monkeypatch):
    """Outpaint: la maschera si dilata nella foto e si sfuma come nel percorso FLUX (prima: bordo netto a ~7 px)."""
    monkeypatch.setenv("NATIVE_OUTPAINT_TONEMATCH", "0")             # isola la sfumatura della maschera
    w = _worker()
    ph = _photo(320, 320)
    fr0, m, rect = _frame(ph)
    fr = opf.prefill_extension(fr0, m, seed=1)                        # come nel worker: init = frame pre-riempito
    mp = tmp_path / "m.png"
    Image.fromarray((m * 255).astype(np.uint8)).save(mp)
    a = np.asarray(fr).astype(int)
    gen = Image.fromarray(np.clip(np.where(m[..., None], 200, a + 40), 0, 255).astype(np.uint8))   # zona-foto generata piu' chiara di 40
    x0, y0, x1, y1 = rect
    narrow = np.asarray(w._paste_back(gen, fr, str(mp), feather=7, outpaint=True, wide=False)).astype(int)
    wide = np.asarray(w._paste_back(gen, fr, str(mp), feather=25, outpaint=True, wide=True)).astype(int)
    row = y0 + 150
    # stretta: la foto e' intatta a 20 px dal confine; larga: il generato entra nella foto per ~2 feather
    assert np.abs(narrow[row, x0 + 30] - a[row, x0 + 30]).max() <= 2
    assert np.abs(wide[row, x0 + 30] - a[row, x0 + 30]).max() >= 10
    assert np.abs(wide[row, x0 + 160] - a[row, x0 + 160]).max() <= 2          # centro foto intatto
    assert (wide[0, 0] == 200).all() and (narrow[0, 0] == 200).all()  # lontano dalla foto: tutto generato
    # con prefill spento il percorso largo non scatta (comportamento di prima)
    monkeypatch.setenv("NATIVE_OUTPAINT_PREFILL", "0")
    assert not w._wide_outpaint({"outpaint": True})
    assert w._outpaint_feather((992, 992), {"outpaint": True}) == 7
