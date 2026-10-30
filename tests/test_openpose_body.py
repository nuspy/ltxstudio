"""OpenPose body nativo (core/openpose_body.py): pesi strict, shape del forward, rilevamento su foto vera (CPU)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from core import openpose_body as ob  # noqa: E402
from core import resources  # noqa: E402

PHOTO = os.path.join(os.path.dirname(__file__), "data", "openpose_person.jpg")   # surfista, 1024x576, 1 persona

needs_weights = pytest.mark.skipif(not os.path.isfile(ob.WEIGHTS_PATH), reason="body_pose_model.pth assente")


@pytest.fixture(autouse=True)
def _commit_guard():
    free = resources.commit_free_gb()
    if free < 3.5:
        pytest.skip(f"commit libero {free:.1f} GB < 3.5")


@pytest.fixture(scope="module")
def model():
    return ob.load_model(device="cpu")


@needs_weights
def test_weights_load_strict():
    sd = torch.load(ob.WEIGHTS_PATH, map_location="cpu", weights_only=True)
    m = ob.BodyPoseModel()
    res = m.load_state_dict(sd, strict=True)          # solleva se manca/avanza qualche chiave
    assert not res.missing_keys and not res.unexpected_keys
    assert len(sd) == 184 and len(m.state_dict()) == 184
    assert sd["conv5_5_CPM_L1.weight"].shape[0] == 38 and sd["conv5_5_CPM_L2.weight"].shape[0] == 19


@needs_weights
def test_forward_shapes(model):
    with torch.inference_mode():
        paf, heat = model(torch.randn(1, 3, 128, 96))
    assert tuple(paf.shape) == (1, 38, 16, 12)        # stride 8
    assert tuple(heat.shape) == (1, 19, 16, 12)


@needs_weights
def test_detect_real_photo(model, tmp_path):
    img = cv2.imread(PHOTO)
    assert img is not None
    res = ob.detect(img[:, :, ::-1], model=model)
    cand, subset = np.array(res["candidate"]), np.array(res["subset"])
    assert len(subset) >= 1
    best = max(sum(1 for i in range(18) if row[i] >= 0 and cand[int(row[i])][2] > 0.3) for row in subset)
    print(f"\nOPENPOSE: persone={len(subset)} keypoint(score>0.3, migliore)={best}/18")
    assert best >= 10
    canvas, _ = ob.pose_image(img[:, :, ::-1], model=model)
    assert canvas.shape == img.shape and canvas.dtype == np.uint8
    assert int((canvas.sum(axis=2) > 0).sum()) > 100          # scheletro disegnato


@needs_weights
def test_no_person_returns_black(model, caplog):
    blank = np.full((256, 256, 3), 127, dtype=np.uint8)
    with caplog.at_level("WARNING"):
        canvas, res = ob.pose_image(blank, model=model)
    assert res["subset"] == [] and canvas.shape == blank.shape and canvas.max() == 0
    assert "nessuna persona" in caplog.text


@needs_weights
def test_controlnet_preprocess_pose(tmp_path):
    from core import controlnet_preprocess as cp
    out = cp.preprocess(PHOTO, "pose", str(tmp_path / "pose.png"))
    img = cv2.imread(out)
    assert img is not None and img.shape == cv2.imread(PHOTO).shape and img.max() > 0


def test_missing_weights_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="body_pose_model.pth"):
        ob.check_weights(str(tmp_path / "nope.pth"))
