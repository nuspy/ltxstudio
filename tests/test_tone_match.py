"""core/tone_match.py — correzione di tono globale generato→foto (task 14). Solo CPU, immagini sintetiche."""
import numpy as np
import pytest
from PIL import Image

from core import tone_match as tm


def _scene(w=640, h=480, seed=1):
    """Scena sintetica con colori molto vari (campi lisci a bassa frequenza + forme), come una foto."""
    rng = np.random.default_rng(seed)
    import cv2
    chans = []
    for _ in range(3):
        f = rng.normal(0, 1, (h // 8, w // 8)).astype(np.float32)
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
        f = cv2.GaussianBlur(f, (0, 0), 14)
        f = (f - f.min()) / (f.max() - f.min() + 1e-6)
        chans.append(f)
    img = np.stack(chans, -1)
    for _ in range(12):                                   # forme a tinta unita
        x, y = int(rng.integers(0, w - 80)), int(rng.integers(0, h - 80))
        img[y:y + int(rng.integers(30, 80)), x:x + int(rng.integers(30, 80))] = rng.random(3)
    return np.clip(img, 0, 1).astype(np.float32)


def _shift(x, bright=1.0, add=0.0, contrast=1.0, sat=1.0, tint=(1, 1, 1)):
    lum = (x @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    y = lum + (x - lum) * sat
    y = (y - 0.5) * contrast + 0.5
    y = y * bright + add
    return np.clip(y * np.array(tint, np.float32), 0, 1)


def _u8(x):
    return Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype("uint8"))


def _setup(shift, w=640, h=480, layout="cornice"):
    truth = _scene(w, h)
    ext = np.ones((h, w), bool)
    if layout == "cornice":
        ext[h // 5:h - h // 5, w // 5:w - w // 5] = False
    else:                                                  # una sola fascia a destra
        ext[:, :w // 2] = False
    gen = _u8(_shift(truth, **shift))                      # il modello rigenera TUTTO con lo scarto di tono
    orig = _u8(np.where(ext[..., None], 0.5, truth))       # foto + grigio nell'estensione
    return truth, ext, gen, orig


def _de(img, truth, sel):
    return float(tm.delta_e(np.asarray(img, np.float32)[sel] / 255, truth[sel]).mean())


SHIFTS = {
    "luminosita": dict(bright=0.82),
    "contrasto_basso": dict(contrast=0.78, add=0.05),
    "contrasto_alto": dict(contrast=1.25),
    "saturazione_bassa": dict(sat=0.6),
    "saturazione_alta": dict(sat=1.35),
    "dominante": dict(tint=(1.07, 1.0, 0.9)),
    "mista": dict(bright=0.9, contrast=1.15, sat=0.8, tint=(1.03, 1.0, 0.95)),
}


@pytest.mark.parametrize("name", list(SHIFTS))
@pytest.mark.parametrize("layout", ["cornice", "lato"])
def test_scarto_di_tono_sparisce_in_tutta_l_estensione(name, layout):
    truth, ext, gen, orig = _setup(SHIFTS[name], layout=layout)
    before = _de(gen, truth, ext)
    out = tm.harmonize(gen, orig, ext, feather=16, boundary="none", log=lambda *_: None)
    after = _de(out, truth, ext)
    assert before > 4.0, "lo scarto simulato deve essere visibile (>4 ΔE) altrimenti il test non prova nulla"
    assert after < 2.5, f"{name}/{layout}: ΔE estensione {before:.2f}→{after:.2f}"
    assert after < before * 0.4


def test_gradiente_vicino_lontano_piatto():
    """Cornice = differenza fra ΔE vicino e lontano: dopo la correzione globale le due fasce devono essere equivalenti."""
    import cv2
    truth, ext, gen, orig = _setup(SHIFTS["mista"])
    out = tm.harmonize(gen, orig, ext, feather=16, boundary="none", log=lambda *_: None)
    d = cv2.distanceTransform(ext.astype(np.uint8), cv2.DIST_L2, 3)
    near = _de(out, truth, ext & (d <= 32)); far = _de(out, truth, ext & (d > 64))
    assert abs(far - near) < 1.0, (near, far)
    assert far < 2.5


def test_zona_foto_fedele_non_si_tocca():
    truth, ext, gen, orig = _setup(dict())                 # nessuno scarto
    fit = tm.estimate(gen, orig, ~ext)
    assert not fit.applied and "fedele" in fit.reason
    out = tm.harmonize(gen, orig, ext, feather=16, boundary="none", log=lambda *_: None)
    assert np.array_equal(np.asarray(out), np.asarray(gen))


def test_outlier_nella_zona_foto_non_trascinano_il_fit():
    """Il modello cambia davvero un pezzo della zona-foto (oggetto nuovo): il fit robusto lo scarta."""
    truth, ext, gen, orig = _setup(SHIFTS["luminosita"])
    g = np.asarray(gen).copy()
    g[190:290, 260:380] = [250, 30, 30]                    # blocco rosso inventato dentro la foto
    out = tm.harmonize(Image.fromarray(g), orig, ext, feather=16, boundary="none", log=lambda *_: None)
    assert _de(out, truth, ext) < 3.0


def test_zona_piccola_fallback_per_canale_e_minuscola_identita():
    truth = _scene(640, 480)
    gen = _u8(_shift(truth, bright=0.8))
    ext = np.ones((480, 640), bool); ext[200:230, 300:360] = False    # ~1100 px dopo erosione → per_channel
    orig = _u8(np.where(ext[..., None], 0.5, truth))
    fit = tm.estimate(gen, orig, ~ext, erode_px=4)
    assert fit.applied and fit.method == "per_channel"
    ext2 = np.ones((480, 640), bool); ext2[200:210, 300:310] = False  # 100 px → niente
    fit2 = tm.estimate(gen, orig, ~ext2, erode_px=2)
    assert not fit2.applied


def test_inpaint_area_piccola_zona_foto_grande():
    """Inpaint: maschera piccola, zona-foto = quasi tutta l'immagine; lo scarto del modello viene tolto anche sull'area rifatta."""
    truth = _scene(640, 480)
    gen = _u8(_shift(truth, contrast=0.8, add=0.04))
    ext = np.zeros((480, 640), bool); ext[150:330, 200:420] = True
    orig = _u8(truth)
    out = tm.harmonize(gen, orig, ext, feather=8, boundary="none", log=lambda *_: None)
    assert _de(out, truth, ext) < 2.5


def test_interruttore_ambiente(monkeypatch):
    monkeypatch.delenv("NATIVE_OUTPAINT_TONEMATCH", raising=False)
    assert tm.enabled()
    monkeypatch.setenv("NATIVE_OUTPAINT_TONEMATCH", "0")
    assert not tm.enabled()
    monkeypatch.setenv("NATIVE_OUTPAINT_TONEMATCH", "1")
    assert tm.enabled()


def test_non_solleva_mai(monkeypatch):
    """Una correzione di tono non deve far fallire una generazione: errore interno → immagine originale + log."""
    truth, ext, gen, orig = _setup(SHIFTS["luminosita"])
    msgs = []
    out = tm.harmonize(gen, orig.resize((10, 10)), ext, feather=16, log=msgs.append)   # dimensioni incoerenti
    assert out.size == gen.size and any("saltata" in m for m in msgs)


def test_confine_non_scatta_sul_solo_rumore_di_contenuto():
    """Scarto globale puro: dopo la stima globale lo scarto di confine è solo contenuto → lo stadio 2 non tocca nulla."""
    truth, ext, gen, orig = _setup(SHIFTS["mista"])
    a = tm.harmonize(gen, orig, ext, feather=16, boundary="none", log=lambda *_: None)
    b = tm.harmonize(gen, orig, ext, feather=16, boundary="gated", log=lambda *_: None)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_confine_scatta_se_l_estensione_ha_uno_scarto_proprio():
    """Zona-foto fedele ma estensione con tono suo (deriva solo nel contenuto nuovo): il fit globale non vede nulla,
    lo stadio di confine sì e riduce l'errore vicino alla foto."""
    import cv2
    truth, ext, _, orig = _setup(dict())
    gen = _u8(np.where(ext[..., None], _shift(truth, bright=0.78), truth))
    out = tm.harmonize(gen, orig, ext, feather=16, boundary="gated", tau_mult=12.0, log=lambda *_: None)
    d = cv2.distanceTransform(ext.astype(np.uint8), cv2.DIST_L2, 3)
    near = ext & (d <= 32)
    assert _de(out, truth, near) < _de(gen, truth, near) * 0.6


def test_riempimento_estensione_qwen_non_scurisce_vicino_alla_foto(tmp_path, monkeypatch):
    """Outpaint Qwen: l'«originale» nell'estensione è il nero del canvas; con la maschera sfumata < 1 vicino alla
    foto filtrava nel generato (banda scura). Col riempimento il generato resta tale e quale."""
    import importlib, sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "workers"))
    qw = importlib.import_module("qwen_image21_infer")
    monkeypatch.setattr(qw, "_log", lambda *_: None)
    w, h = 320, 240
    gen_arr = np.full((h, w, 3), 180, np.uint8)
    init_arr = np.zeros((h, w, 3), np.uint8); init_arr[:, 160:] = 180          # foto a destra, estensione = nero
    mask = np.zeros((h, w), np.uint8); mask[:, :160] = 255                      # bianco = estensione
    mp = tmp_path / "m.png"; Image.fromarray(mask).save(mp)
    gen, init = Image.fromarray(gen_arr), Image.fromarray(init_arr)
    monkeypatch.setenv("NATIVE_OUTPAINT_TONEMATCH", "1")
    fixed = np.asarray(qw._paste_back(gen, init, str(mp), feather=12, outpaint=True))
    assert fixed[:, 130:150].min() >= 178                                       # nessun nero che filtra
    monkeypatch.setenv("NATIVE_OUTPAINT_TONEMATCH", "0")                        # emergenza: comportamento di prima
    old = np.asarray(qw._paste_back(gen, init, str(mp), feather=12, outpaint=True))
    assert old[:, 130:150].min() < 150
