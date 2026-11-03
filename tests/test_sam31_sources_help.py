"""SAM 3.1 dopo la decisione del 03/11 sera: pesi NON gated (unsloth/sam3.1 + Comfy-Org/sam3.1) dal normale catalogo, nessun flusso di accesso a Meta;
guida ridotta a cosa fa e quando sceglierlo; ponte worker (testo -> riquadri) e aggancio ai backend. Nessuna rete, nessun download, nessuna GPU."""
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core import segmentation_engine as SE


def test_catalog_has_the_two_ungated_copies_and_not_the_meta_repo():
    from core import weights_catalog as wc
    by = {e["id"]: e for e in wc.load()}
    assert "sam3.1-multiplex" not in by                                           # la voce gated di facebook/sam3.1 e tolta
    assert not any(e.get("repo_id") == "facebook/sam3.1" for e in by.values())
    u, c = by["sam31-unsloth"], by["sam31-comfy-multiplex"]
    assert u["repo_id"] == "unsloth/sam3.1" and not u["gated"] and u["kind"] == "snapshot" and "model.safetensors" in u["allow_patterns"]
    assert "sam3.1-INT8.safetensors" not in u["allow_patterns"]                   # solo BF16: le varianti quantizzate non si scaricano
    assert c["repo_id"] == "Comfy-Org/sam3.1" and not c["gated"] and c["kind"] == "file"
    assert c["remote_filename"] == "checkpoints/sam3.1_multiplex_fp16.safetensors" and c["filename"] == "sam3.1_multiplex_fp16.safetensors"
    assert str(wc.dest_for(u)).replace("\\", "/").endswith("Segmentation/sam3.1")
    assert str(wc.dest_for(c)).replace("\\", "/").endswith("Segmentation/sam3.1_multiplex")
    assert SE.models_dir().name == "sam3.1" and SE.SAM31_CATALOG_ID in by and SE.MULTIPLEX_CATALOG_ID in by


def test_multiplex_tracker_is_declared_unusable_today():
    assert SE.MULTIPLEX_TRACKER_USABLE is False


def test_help_text_has_no_access_flow_and_explains_when_to_choose():
    from ui.widgets import sam31_help
    t = sam31_help.build_text("missing")
    low = t.lower()
    assert "approva" not in low.replace("approvazioni", "") and "ogni 30 minuti" not in t and "token" not in low and "giorni" not in t
    assert "senza account" in t and "GroundingDINO" in t and "SAM2" in t
    assert "testo" in t and "clic" in t and "lazo" in t                          # quando SAM 3.1 (testo) e quando SAM2 (clic/lazo)
    assert "1,8 GB" in t
    assert "1,8 GB" not in sam31_help.build_text("ok")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_help_dialog_offers_the_catalog_download_only_when_missing(qapp):
    from ui.widgets import sam31_help
    done = []
    d = sam31_help.show(None, status="missing", on_download=lambda: done.append(1))
    try:
        assert not d.isModal() and not d.btn_download.isHidden()
        d.btn_download.click()
        assert done == [1]
    finally:
        d.close()
    d = sam31_help.show(None, status="ok", on_download=lambda: done.append(2))
    try:
        assert d.btn_download.isHidden()
    finally:
        d.close()
    d = sam31_help.show(None, status="missing")
    try:
        assert d.btn_download.isHidden()                                           # senza callback niente pulsante
    finally:
        d.close()


# ── ponte verso il worker ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_detect_bridge_reads_the_worker_line_and_sorts_boxes(tmp_path):
    from core import sam31_detect as D
    seen = []
    payload = {"boxes": [[1, 2, 30, 40], [5, 6, 50, 60]], "scores": [0.6, 0.9], "masks": [], "union": "", "presence": 0.99, "size": [100, 100], "device": "cpu"}

    def runner(cmd):
        seen.append(cmd)
        return 0, "rumore\nSAM31:" + json.dumps(payload) + "\n", ""
    r = D.detect("x.png", "person.", "cpu", 0.5, out_dir=str(tmp_path), runner=runner)
    assert r["scores"] == [0.6, 0.9]
    cmd = seen[0]
    assert cmd[0].replace("\\", "/").endswith("diffusers_next_venv/Scripts/python.exe")        # transformers 5.17: il venv che ha Sam3Model
    assert cmd[cmd.index("--text") + 1] == "person" and cmd[cmd.index("--device") + 1] == "cpu" and cmd[cmd.index("--threshold") + 1] == "0.5"
    b = D.boxes("x.png", "person", "cpu", 0.5, out_dir=str(tmp_path), runner=runner)
    assert b[0] == [5.0, 6.0, 50.0, 60.0, 0.9] and b[1][4] == 0.6


def test_detect_bridge_reports_a_worker_failure_clearly():
    from core import sam31_detect as D
    with pytest.raises(RuntimeError) as e:
        D.detect("x.png", "person", runner=lambda cmd: (1, "", "Traceback ... ValueError: boom"))
    assert "SAM 3.1" in str(e.value) and "boom" in str(e.value)


def test_inference_venv_cannot_run_sam3_so_a_worker_is_needed():
    """Verificato 03/11: transformers 4.57.6 dell'inference_server_venv non ha Sam3Model; il worker usa il venv 5.17."""
    from core import sam31_detect as D
    assert "inference_server_venv" not in D.WORKER_PY and os.path.basename(D.WORKER) == "sam31_detect_worker.py" and os.path.exists(D.WORKER)


# ── aggancio ai backend ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_video_text_detection_uses_sam31_only_when_chosen(monkeypatch):
    from core import sam31_detect as D
    from inference_server.backends import video_segment as V
    calls = []
    monkeypatch.setattr(D, "boxes", lambda img, text, device="", threshold=0.5, **kw: (calls.append((img, text, device, threshold)),
                                                                                         [[1.0, 2.0, 3.0, 4.0, 0.9], [9.0, 9.0, 20.0, 20.0, 0.5]])[1])
    assert V._gdino_box("f0.png", "person.", "cpu", "sam31") == [1.0, 2.0, 3.0, 4.0]
    assert V.detect_boxes("f0.png", "person.", "cpu", engine="sam31")[1][4] == 0.5
    assert calls[0] == ("f0.png", "person", "cpu", 0.5)                            # testo senza punto finale, soglia 0,5
    monkeypatch.setattr(D, "boxes", lambda *a, **k: [])
    assert V._gdino_box("f0.png", "zebra", "cpu", "sam31") is None


def test_track_and_find_people_no_longer_refuse_sam31(tmp_path, monkeypatch):
    """SAM 3.1 scelto: nessun errore «caricatore non collegato»; gli errori arrivano solo da pesi mancanti o dal worker."""
    monkeypatch.setattr(SE, "models_dir", lambda: tmp_path / "vuoto")
    from inference_server.backends import video_segment as V
    with pytest.raises(RuntimeError) as e:
        V.track_object("x.mp4", text="person", engine="sam31")
    assert "non e installato" in str(e.value) and "collegato" not in str(e.value)


def test_image_text_selection_runs_the_sam31_branch_not_gdino(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    from core import sam31_detect as D
    from inference_server.backends import image_segment as I
    monkeypatch.setattr(SE, "LOADER_READY", True)
    monkeypatch.setattr(SE, "sam31_status", lambda base=None: ("ok", []))
    img = tmp_path / "foto.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(img)
    um = tmp_path / "w" / "u.png"
    um.parent.mkdir()
    m = np.zeros((48, 64), "uint8"); m[10:30, 5:25] = 255
    Image.fromarray(m).save(um)
    monkeypatch.setattr(D, "detect", lambda p, t, dev="", thr=0.5, **kw: {"boxes": [[5, 10, 25, 30]], "scores": [0.9], "masks": [], "union": str(um)})

    class _MM:
        def register(self, *a, **k):
            raise AssertionError("GroundingDINO/SAM2 non devono essere caricati")

        def ensure(self, *a, **k):
            raise AssertionError("GroundingDINO/SAM2 non devono essere caricati")
    r = I.run({"seg_engine": "sam31", "image_path": str(img), "text": "person", "threshold": 0.5}, _MM(), str(tmp_path / "out"))
    assert r["status"] == "success"
    out = np.asarray(Image.open(r["output"][0]))
    assert out.shape == (48, 64) and (out > 127).sum() == 20 * 20
    monkeypatch.setattr(D, "detect", lambda *a, **k: {"boxes": [], "scores": [], "masks": [], "union": ""})
    r = I.run({"seg_engine": "sam31", "image_path": str(img), "text": "zebra"}, _MM(), str(tmp_path / "out"))
    assert r["status"] == "error" and "zebra" in r["error"] and "SAM 3.1" in r["error"]
