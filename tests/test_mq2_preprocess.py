"""MagicQuill V2 nativo: preprocessing (color block, maschere, bordi canny + tratti utente, prompt, ricomposizione) e
validazione dei parametri di `magicquill_v2.prepare_inputs` su immagini sintetiche. CPU pura, nessun modello."""
import os

import numpy as np
import pytest
from PIL import Image

from core.resources import commit_free_gb
from core import optional_components as _oc  # noqa: E402
import pytest as _pt  # noqa: E402
if _oc.load_magicquill() is None:                # componente opzionale (optional/magicquill o cartella installata)
    _pt.skip(_oc.last_error(), allow_module_level=True)
from ltx_magicquill import magicquill_v2 as mq
from ltx_magicquill.mq2 import preprocess as pp


@pytest.fixture(autouse=True)
def _risorse():
    if commit_free_gb() < 3.5:
        pytest.skip("commit libero < 3.5 GB")


def _img(w=160, h=96):
    a = np.zeros((h, w, 3), np.uint8)
    a[:, : w // 2] = (200, 40, 40)
    a[:, w // 2:] = (30, 60, 220)
    a[h // 3: 2 * h // 3, w // 4: 3 * w // 4] = (240, 240, 240)       # rettangolo chiaro: bordi netti
    return a


def test_color_block_32_e_256():
    a = _img(256, 256)
    cb = pp.color_block(a)
    assert cb.size == (256, 256)
    b = np.asarray(cb)
    # blocchi 8x8 px (256/32): costanti dentro il blocco
    blk = b[8:16, 16:24].reshape(-1, 3)
    assert (blk == blk[0]).all()
    assert tuple(b[0, 0]) == (200, 40, 40) and tuple(b[0, -1]) == (30, 60, 220)
    # media per area (INTER_AREA): un blocco di bordo e' un misto dei due colori
    ex = np.zeros((64, 64, 3), np.uint8)
    ex[:, :33] = 255
    mid = np.asarray(pp.color_block(ex))[0, 128 - 4: 128 + 4]
    assert len({tuple(p) for p in mid.reshape(-1, 3)}) <= 2


def test_grow_mask_croce_diamante():
    m = np.zeros((41, 41), np.uint8)
    m[20, 20] = 255
    g = pp.grow_mask(m, 5)
    ys, xs = np.nonzero(g)
    assert len(ys) == 2 * 5 * (5 + 1) + 1                              # diamante di raggio 5 = 61 pixel
    assert g[20, 25] == 255 and g[20, 26] == 0 and g[25, 20] == 255 and g[23, 23] == 0 and g[22, 23] == 255
    assert (pp.grow_mask(m, 0) == m).all() and pp.grow_mask(m, 0) is not m
    assert pp.grow_mask(m, 3).dtype == np.uint8


def test_black_masked_e_foreground():
    a = _img()
    m = np.zeros(a.shape[:2], np.uint8)
    m[10:30, 20:60] = 255
    b = np.asarray(pp.black_masked(a, m))
    assert (b[10:30, 20:60] == 0).all() and (b[:10] == a[:10]).all()
    prop = np.zeros(a.shape[:2], np.uint8)
    prop[40:60, 60:90] = 255
    img, edit = pp.foreground_image(a, prop, None, grow=6)
    out = np.asarray(img)
    assert (out[40:60, 60:90] == a[40:60, 60:90]).all()                # l'oggetto resta
    ring = (edit > 127) & (prop <= 127)
    assert ring.any() and (out[ring] == 255).all()                      # alone allargato = BIANCO
    assert (out[edit <= 127] == a[edit <= 127]).all()
    fill = np.zeros_like(prop)
    fill[0:5, 0:5] = 255
    _, edit2 = pp.foreground_image(a, prop, fill, grow=6)
    assert edit2[2, 2] == 255


def test_edge_canny_piu_tratti_utente():
    a = _img()
    e = pp.canny_edges(a)
    assert e.dtype == np.uint8 and e.shape == a.shape[:2] and e.max() == 255 and e.mean() < 40
    add = np.zeros(a.shape[:2], np.uint8)
    add[5:8, 5:50] = 255
    erase = np.zeros_like(add)
    erase[:, :] = 0
    h, w = a.shape[:2]
    erase[h // 3 - 3: h // 3 + 3, w // 4: 3 * w // 4] = 255            # cancella il bordo superiore del rettangolo
    base = np.asarray(pp.edge_control_image(a, "canny"))
    ctl = np.asarray(pp.edge_control_image(a, "canny", add, erase))
    assert ctl.shape == (h, w, 3) and (ctl[..., 0] == ctl[..., 1]).all()
    assert (ctl[add > 127] == 255).all()                                # tratti aggiunti = bianchi
    top = (erase > 127) & (add <= 127)
    assert base[top].sum() > 0 and ctl[top].sum() == 0                  # bordi tolti
    untouched = (erase <= 127) & (add <= 127)
    assert (ctl[untouched] == base[untouched]).all()
    with pytest.raises(pp.MQError, match="sconosciuto"):
        pp.edge_control_image(a, "boh")


def test_hed_senza_pesi_errore_chiaro(monkeypatch):
    monkeypatch.delenv("MQ2_HED_PATH", raising=False)
    monkeypatch.setattr(pp, "HED_CANDIDATES", (r"X:\non\esiste\ControlNetHED.pth",))
    with pytest.raises(pp.MQError) as ei:
        pp.edge_control_image(_img(), "hed")
    msg = str(ei.value)
    assert "ControlNetHED.pth" in msg and "MB" in msg and "canny" in msg


def test_fill_color_region_sfumata():
    a = _img()
    m = np.zeros(a.shape[:2], np.uint8)
    m[30:60, 50:110] = 255
    out = pp.fill_color_region(a, m, (10, 200, 30), feather=6)
    assert np.abs(out[45, 80].astype(int) - (10, 200, 30)).max() <= 1                          # centro = colore pieno
    assert (out[0, 0] == a[0, 0]).all()                                 # lontano = invariato
    edge = out[45, 50].astype(int)                                      # bordo = misto
    assert not (edge == np.array([10, 200, 30])).all() and not (edge == a[45, 50]).all()


def test_prompt_e_max_len():
    assert pp.build_prompt("removal", "qualsiasi cosa") == "remove the instance"
    assert pp.build_prompt("local", "un cappello rosso") == "un cappello rosso"
    p = pp.build_prompt("foreground", "una lampada", fix_perspective=True)
    assert p.startswith("Fill in the white region naturally") and "una lampada" in p and p.endswith(" Fix the perspective if necessary.")
    assert "Fix the perspective if necessary." not in pp.build_prompt("foreground", "x", False).replace(
        "Fix the perspective of the foreground object if necessary.", "")
    assert pp.max_sequence_length("removal") == 512 and pp.max_sequence_length("local") == 128


def test_blend_original_fuori_maschera_identico():
    o = Image.fromarray(_img(128, 128))
    r = Image.fromarray(np.full((128, 128, 3), 7, np.uint8))
    m = np.zeros((128, 128), np.uint8)
    m[40:90, 40:90] = 255
    out = np.asarray(pp.blend_original(r, o, m, ksize=10, sigma=10))
    oo = np.asarray(o)
    assert (out[:20] == oo[:20]).all() or np.abs(out[:20].astype(int) - oo[:20]).max() <= 40   # lontano: quasi originale
    assert np.abs(out[65, 65].astype(int) - 7).max() <= 1               # dentro: generato
    assert np.abs(out[0, 0].astype(int) - oo[0, 0]).max() == 0          # angolo: originale esatto


def test_gen_size_rapporto_e_cap():
    assert pp.gen_size(1000, 500) == (992, 496)
    w, h = pp.gen_size(4000, 2000, 1536)
    assert max(w, h) <= 1536 and w % 16 == 0 and h % 16 == 0 and abs(w / h - 2) < 0.05
    assert pp.gen_size(100, 100) == (256, 256)


# ---- prepare_inputs (validazione per modalita') -----------------------------------------------------------------
@pytest.fixture
def files(tmp_path):
    a = _img(640, 384)
    Image.fromarray(a).save(tmp_path / "base.png")
    m = np.zeros((384, 640), np.uint8)
    m[80:200, 160:360] = 255
    Image.fromarray(m).save(tmp_path / "mask.png")
    Image.fromarray(np.zeros((384, 640), np.uint8)).save(tmp_path / "empty.png")
    col = a.copy()
    col[80:200, 160:360] = (10, 220, 10)
    Image.fromarray(col).save(tmp_path / "colored.png")
    ed = np.zeros((384, 640), np.uint8)
    ed[40:48, 40:400] = 255
    Image.fromarray(ed).save(tmp_path / "strokes.png")
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    for n in ("local", "removal", "edge", "color", "puzzle"):
        (lora_dir / f"{n}_lora.safetensors").write_bytes(b"x")
    d = str(tmp_path)
    return dict(image_path=d + r"\base.png", mask_path=d + r"\mask.png", lora_dir=str(lora_dir), _d=d)


def _p(files, **kw):
    p = {k: v for k, v in files.items() if k != "_d"}
    p.update(kw)
    return p


def test_prepare_local_e_removal(files):
    j = mq.prepare_inputs(_p(files, mode="local", prompt="un gatto"))
    assert j["size"] == (640, 384) and j["controls"][0]["mode"] == "local"
    assert j["controls"][0]["gamma"] == 1.0 and j["prompt"] == "un gatto" and j["max_sequence_length"] == 128
    cond = np.asarray(j["controls"][0]["image"])
    assert cond.shape[:2] == (j["size"][1], j["size"][0]) and (cond == 0).all(-1).any()      # area nera
    assert j["blend_mask"] is not None and j["blend_mask"].shape == cond.shape[:2]
    # grow: la maschera di ricomposizione e' piu' larga della maschera originale
    raw = pp.load_mask(files["mask_path"], j["size"])
    assert (j["blend_mask"] > 127).sum() > (raw > 127).sum()
    r = mq.prepare_inputs(_p(files, mode="removal", prompt="ignorato", strength_local=0.7))
    assert r["prompt"] == "remove the instance" and r["max_sequence_length"] == 512 and r["controls"][0]["gamma"] == 0.7
    assert r["controls"][0]["lora_path"].endswith("removal_lora.safetensors")


def test_prepare_edge_color_foreground(files):
    d = files["_d"]
    e = mq.prepare_inputs(_p(files, mode="edge", edge_path=d + r"\strokes.png", prompt="x"))
    assert e["controls"][0]["gamma"] == 0.6 and e["blend_mask"] is None and e["base_image"].size == e["size"]
    c = mq.prepare_inputs(_p(files, mode="color", color_path=d + r"\colored.png"))
    assert c["controls"][0]["gamma"] == 1.5 and c["controls"][0]["image"].size == (256, 256)
    c2 = mq.prepare_inputs(_p(files, mode="color", color_rgb=[0, 255, 0], strength_color=2.0))
    assert c2["controls"][0]["gamma"] == 2.0
    f = mq.prepare_inputs(_p(files, mode="foreground", prompt="una lampada", fix_perspective=True))
    assert f["controls"] == [] and f["puzzle_path"].endswith("puzzle_lora.safetensors")
    assert f["prompt"].startswith("Fill in the white region") and f["blend_mask"] is not None
    assert np.asarray(f["base_image"]).max() == 255
    # il contratto del foreground: grow di default 50, quindi molto piu' largo del default 10 delle altre modalita'
    f10 = mq.prepare_inputs(_p(files, mode="foreground", grow=10))
    assert (f["blend_mask"] > 127).sum() > (f10["blend_mask"] > 127).sum()


def test_prepare_errori_chiari(files):
    d = files["_d"]
    cases = [
        (dict(mode="boh"), "mode"),
        (dict(mode="local", image_path=d + r"\nope.png"), "image_path"),
        (dict(mode="local", mask_path=d + r"\empty.png"), "vuota"),
        (dict(mode="local", mask_path=""), "mask_path"),
        (dict(mode="color"), "color_path oppure color_rgb"),
        (dict(mode="color", color_path=d + r"\base.png"), "identico"),
        (dict(mode="edge", edge_source="hed"), "ControlNetHED"),
        (dict(mode="local", lora_dir=d + r"\vuota"), "LoRA di controllo mancante"),
    ]
    os.makedirs(d + r"\vuota", exist_ok=True)
    for kw, frag in cases:
        with pytest.raises(pp.MQError) as ei:
            mq.prepare_inputs(_p(files, **kw))
        assert frag in str(ei.value), (kw, str(ei.value))


def test_run_errori_non_sollevano(files):
    """run() trasforma gli errori di validazione in {'status':'error'} PRIMA di toccare il modello."""
    r = mq.run(_p(files, mode="boh", model=r"C:\x.safetensors"), None, str(files["_d"]))
    assert r["status"] == "error" and "non trovato" in r["error"] or "mode" in r["error"]
    r = mq.run(_p(files, mode="local"), None, str(files["_d"]))
    assert r["status"] == "error" and "model" in r["error"]


def test_prepare_foreground_prepared(files):
    # prepared=True: l'immagine arriva gia' con l'alone bianco e viene usata cosi' com'e'; la maschera serve solo al blend
    a = np.asarray(Image.open(files["image_path"]).convert("RGB")).copy()
    m = pp.load_mask(files["mask_path"])
    a[m > 127] = 255
    prep = os.path.join(os.path.dirname(files["image_path"]), "prepared.png")
    Image.fromarray(a).save(prep)
    j = mq.prepare_inputs(_p(files, mode="foreground", prepared=True, image_path=prep, grow=99))
    assert j["controls"] == [] and j["puzzle_path"].endswith("puzzle_lora.safetensors")
    assert np.array_equal(np.asarray(j["base_image"]), pp.load_rgb(prep, j["size"]))   # nessun alone aggiunto
    assert np.array_equal(j["blend_mask"], pp.grow_mask(pp.load_mask(files["mask_path"], j["size"]), 25))
    # senza prepared l'alone viene aggiunto da foreground_image (comportamento invariato)
    j2 = mq.prepare_inputs(_p(files, mode="foreground", image_path=files["image_path"]))
    assert not np.array_equal(np.asarray(j2["base_image"]), pp.load_rgb(files["image_path"], j2["size"]))
    with pytest.raises(pp.MQError, match="mask_path"):
        mq.prepare_inputs(_p(files, mode="foreground", prepared=True, mask_path=""))
