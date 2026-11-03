"""Test di core/sam31_loader.py (SAM 3.1, formato transformers, SOLO CPU).

Parte veloce (sempre): validazione argomenti, riconoscimento download incompleto, struttura di segment_video_frames
con modello/processor finti (il testo e' codificato una volta sola).
Parte reale (skip se mancano i pesi o transformers con Sam3): carica il modello vero su CPU, segmenta UN fotogramma
vero (frame a 2 s di ``persona_intera_12s.mp4``, oppure ``tests/data/openpose_person.jpg``) con "person" e con "zebra".
Su CPU a 1008x1008 ogni immagine richiede decine di secondi.  Eseguire con il venv che ha transformers >= 5.17:
    E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe -m pytest tests/test_sam31_loader.py -q
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # niente GPU

torch = pytest.importorskip("torch")
from core import sam31_loader as L  # noqa: E402

MODEL_DIR = L.DEFAULT_MODEL_DIR
HAVE_WEIGHTS = L.weights_ready(MODEL_DIR)
try:
    import transformers
    from transformers import Sam3Model  # noqa: F401
    HAVE_SAM3 = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (5, 17)      # 5.3 ha Sam3Model ma con un bug del text encoder
except Exception:  # transformers vecchio
    HAVE_SAM3 = False
FIXTURE_VIDEO = Path(r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live\fixtures\persona_intera_12s.mp4")
FALLBACK_IMG = Path(__file__).parent / "data" / "openpose_person.jpg"
OUT_DIR = Path(r"C:\TEMP\claude\E--ai-ltxstudio\684434ab-2cb0-4775-bcb7-658804cd8fc9\scratchpad\agent_loader")


# ----------------------------------------------------------------------------------------------- veloci
def test_weights_ready_distingue_part(tmp_path):
    assert not L.weights_ready(tmp_path)
    (tmp_path / "model.safetensors.part").write_bytes(b"x")
    assert not L.weights_ready(tmp_path)          # download in corso
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert L.weights_ready(tmp_path)


def test_load_senza_pesi_da_errore_chiaro(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        L.load_image_model(tmp_path)


def test_segment_image_richiede_testo_o_box():
    with pytest.raises(ValueError):
        L.segment_image(None, None, np.zeros((8, 8, 3), np.uint8), text=None, boxes=None)


def test_to_pil_accetta_array_e_pil():
    from PIL import Image
    a = np.zeros((5, 7, 3), np.uint8)
    assert L._to_pil(a).size == (7, 5)
    assert L._to_pil(Image.new("RGBA", (3, 4))).mode == "RGB"


class _FakeOut:
    presence_logits = torch.tensor([[2.0]])


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.text_calls = 0
        self.forward_kwargs = []

    def get_text_features(self, input_ids, attention_mask, return_dict):
        self.text_calls += 1
        return types.SimpleNamespace(pooler_output=torch.zeros(1, 32, 256))

    def forward(self, **kw):
        self.forward_kwargs.append(sorted(kw))
        return _FakeOut()


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = lambda text, **k: {"input_ids": torch.ones(1, 32, dtype=torch.long),
                                            "attention_mask": torch.ones(1, 32, dtype=torch.long)}

    def __call__(self, images=None, **kw):
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}

    def post_process_instance_segmentation(self, out, threshold, mask_threshold, target_sizes):
        H, W = target_sizes[0]
        return [{"scores": torch.tensor([0.9]), "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
                 "masks": torch.ones(1, H, W, dtype=torch.long)}]


def test_video_frames_codifica_il_testo_una_volta():
    m, p = _FakeModel(), _FakeProcessor()
    frames = [np.zeros((6, 9, 3), np.uint8)] * 3
    rs = L.segment_video_frames(m, p, frames, "person")
    assert m.text_calls == 1 and len(rs) == 3
    assert m.forward_kwargs[0] == ["attention_mask", "pixel_values", "text_embeds"]
    r = rs[0]
    assert r["masks"].shape == (1, 6, 9) and r["masks"].dtype == bool
    assert r["boxes"].shape == (1, 4) and r["scores"].shape == (1,) and r["size"] == (6, 9)
    assert 0.8 < r["presence"] < 0.9


# ----------------------------------------------------------------------------------------------- reali (CPU)
def _test_frame():
    """Frame a 2 s della fixture video (persona che cammina in un mercato) o, in mancanza, l'immagine di repo."""
    from PIL import Image
    if FIXTURE_VIDEO.is_file():
        import cv2
        cap = cv2.VideoCapture(str(FIXTURE_VIDEO))
        cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
        ok, bgr = cap.read()
        cap.release()
        if ok:
            return Image.fromarray(bgr[:, :, ::-1].copy()), "persona_intera_12s@2s"
    return Image.open(FALLBACK_IMG).convert("RGB"), "openpose_person"


def _save_overlay(img, res, name):
    from PIL import Image
    out = np.asarray(img.convert("RGB")).astype(np.float32).copy()
    cols = [(255, 40, 40), (40, 200, 40), (40, 80, 255), (255, 200, 0)]
    for i, m in enumerate(res["masks"]):
        c = np.array(cols[i % len(cols)], np.float32)
        out[m] = out[m] * 0.45 + c * 0.55
    path = (OUT_DIR if OUT_DIR.is_dir() else Path(os.environ.get("TEMP", "."))) / name
    Image.fromarray(out.astype(np.uint8)).save(path)
    return path


@pytest.fixture(scope="module")
def real_model():
    model, processor = L.load_image_model(MODEL_DIR, device="cpu", dtype=torch.float32)
    assert next(model.parameters()).dtype == torch.float32
    return model, processor


@pytest.mark.skipif(not (HAVE_WEIGHTS and HAVE_SAM3), reason="pesi sam3.1 o transformers>=5.17 assenti")
def test_reale_person_maschera_e_zebra_nessun_rilevamento(real_model):
    model, processor = real_model
    img, label = _test_frame()
    W, H = img.size
    r = L.segment_image(model, processor, img, text="person", threshold=0.5)
    assert len(r["scores"]) >= 1, "nessuna persona trovata"
    best = int(r["scores"].argmax())
    assert r["scores"][best] > 0.5 and r["presence"] > 0.5
    mk = r["masks"][best]
    assert mk.shape == (H, W) and mk.dtype == bool
    assert 0.02 < mk.mean() < 0.6, f"area maschera implausibile: {mk.mean():.3f}"
    x1, y1, x2, y2 = r["boxes"][best]
    ys, xs = np.where(mk)
    # la maschera sta (quasi) tutta dentro la box prevista
    assert xs.min() >= x1 - 0.03 * W and xs.max() <= x2 + 0.03 * W and ys.min() >= y1 - 0.03 * H and ys.max() <= y2 + 0.03 * H
    _save_overlay(img, r, f"controllo_person_{label}.png")
    z = L.segment_image(model, processor, img, text="zebra", threshold=0.3)
    assert len(z["scores"]) == 0 and z["presence"] < 0.3, f"zebra: {z['scores']} presence={z['presence']:.3f}"


@pytest.mark.skipif(not (HAVE_WEIGHTS and HAVE_SAM3), reason="pesi sam3.1 o transformers>=5.17 assenti")
def test_reale_box_prompt_e_video_frames(real_model):
    model, processor = real_model
    img, label = _test_frame()
    r = L.segment_image(model, processor, img, text="person", threshold=0.5)
    assert len(r["scores"]) >= 1
    # prompt a box (la box della persona trovata): deve ritrovare lo stesso soggetto
    b = r["boxes"][int(r["scores"].argmax())].tolist()
    rb = L.segment_image(model, processor, img, text=None, boxes=[b], threshold=0.3)
    assert len(rb["scores"]) >= 1
    inter = (rb["masks"][int(rb["scores"].argmax())] & r["masks"][int(r["scores"].argmax())]).sum()
    union = (rb["masks"][int(rb["scores"].argmax())] | r["masks"][int(r["scores"].argmax())]).sum()
    assert inter / union > 0.5
    # lista di fotogrammi chiave: stesso risultato del singolo fotogramma (testo codificato una volta)
    rs = L.segment_video_frames(model, processor, [img, img], "person", threshold=0.5)
    assert len(rs) == 2
    assert len(rs[0]["scores"]) == len(r["scores"])
    assert abs(float(rs[0]["scores"].max()) - float(r["scores"].max())) < 1e-3
