"""SAM 3.1 di Meta (facebook/sam3.1, gated) + copia Comfy-Org non gated come alternativa: catalogo, pesi, guida con accesso, ponte verso i job del pacchetto sam3
(processo figlio con le garanzie del lanciatore sam3_run.sh), aggancio ai backend. Nessuna rete, nessun download, nessuna GPU, nessun codice di Meta eseguito."""
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core import segmentation_engine as SE


def _touch(d, files):
    d.mkdir(parents=True, exist_ok=True)
    for n, size in files.items():
        with open(d / n, "wb") as f:
            f.truncate(size or 10)


def test_catalog_has_meta_gated_and_comfy_alternative_and_not_unsloth():
    from core import weights_catalog as wc
    by = {e["id"]: e for e in wc.load()}
    assert "sam31-unsloth" not in by and not any(e.get("repo_id") == "unsloth/sam3.1" for e in by.values())
    m, c = by["sam3.1-multiplex"], by["sam31-comfy-multiplex"]
    assert m["repo_id"] == "facebook/sam3.1" and m["gated"] and m["kind"] == "snapshot"
    assert c["repo_id"] == "Comfy-Org/sam3.1" and not c["gated"] and c["kind"] == "file"
    assert c["remote_filename"] == "checkpoints/sam3.1_multiplex_fp16.safetensors" and c["filename"] == SE.COMFY_CHECKPOINT
    assert str(wc.dest_for(m)).replace("\\", "/").endswith("Segmentation/sam3.1")
    assert str(wc.dest_for(c)).replace("\\", "/").endswith("Segmentation/sam3.1_multiplex")
    assert SE.models_dir().name == "sam3.1" and SE.comfy_dir().name == "sam3.1_multiplex"


def test_comfy_copy_is_used_only_if_declared_equivalent(tmp_path, monkeypatch):
    meta, comfy = tmp_path / "sam3.1", tmp_path / "sam3.1_multiplex"
    _touch(comfy, SE.COMFY_FILES)
    monkeypatch.setattr(SE, "models_dir", lambda: meta)
    monkeypatch.setattr(SE, "comfy_dir", lambda: comfy)
    monkeypatch.setattr(SE, "COMFY_EQUIVALENT", None)
    assert SE.comfy_status()[0] == "ok" and not SE.comfy_usable() and SE.weights_state() == "missing"
    assert SE.checkpoint_path() == meta / SE.SAM31_CHECKPOINT                      # senza confronto: resta il percorso di Meta (errore chiaro dal job)
    monkeypatch.setattr(SE, "COMFY_EQUIVALENT", False)
    assert not SE.comfy_usable() and SE.weights_state() == "missing"
    monkeypatch.setattr(SE, "COMFY_EQUIVALENT", True)
    assert SE.comfy_usable() and SE.weights_state() == "ok" and SE.checkpoint_path() == comfy / SE.COMFY_CHECKPOINT
    _touch(meta, SE.SAM31_FILES)                                                   # con i pesi di Meta completi vince Meta
    assert SE.checkpoint_path() == meta / SE.SAM31_CHECKPOINT
    (comfy / SE.COMFY_CHECKPOINT).write_bytes(b"x")                                # copia troncata: non usabile
    assert not SE.comfy_usable()


def test_flags_say_what_was_proven():
    """Immagine provata con Meta e Comfy su un fotogramma vero (CPU); tracking multiplex nativo non ancora (su CPU la propagazione fallisce, serve la GPU)."""
    assert SE.LOADER_READY is True and SE.IMAGE_READY is True and SE.VIDEO_READY is False
    assert SE.COMFY_EQUIVALENT is True


def test_help_text_has_warning_steps_and_when_to_choose():
    from ui.widgets import sam31_help
    t = sam31_help.build_text("missing")
    assert "alcune ore" in t and "SAM2" in t and "https://huggingface.co/facebook/sam3.1" in t and "huggingface.co/settings/tokens" in t
    assert "ogni 30 minuti" in t and "token" in t.lower() and "giorni" not in t
    for k in "1234":
        assert f"\n{k}. " in t
    assert "testo" in t and "clic" in t and "lazo" in t                          # quando SAM 3.1 e quando SAM2
    assert "alcune ore" not in sam31_help.build_text("ok")                       # chi ha gia i pesi non vede la guida all'accesso


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def prefs(monkeypatch, tmp_path):
    store = {}
    from core import sam31_access as AC
    from core import user_prefs
    monkeypatch.setattr(user_prefs, "get", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(user_prefs, "set", lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr(SE, "models_dir", lambda: tmp_path / "vuoto")
    monkeypatch.setattr(SE, "comfy_dir", lambda: tmp_path / "vuoto_comfy")
    monkeypatch.setattr(AC, "token", lambda: store.get("hf_token") or None)
    return store


def test_help_dialog_confirms_request_and_switches_to_check_snooze_cancel(qapp, prefs):
    from core import sam31_access as AC
    from ui.sam31_watch import Sam31Watcher
    from ui.widgets import sam31_help
    w = Sam31Watcher(None, probe=lambda: ("pending", ""), start_download=lambda: None, tick_ms=10_000_000)
    dlg = sam31_help.show(None, status="missing", watcher=w)
    try:
        assert not dlg.isModal()
        assert not dlg.btn_requested.isHidden() and dlg.btn_check.isHidden() and dlg.btn_snooze.isHidden() and dlg.btn_cancel.isHidden()
        dlg.token_edit.setText("hf_PROVA")
        [b for b in dlg.findChildren(type(dlg.btn_requested)) if b.text() == "Salva token"][0].click()
        assert prefs["hf_token"] == "hf_PROVA" and dlg.token_edit.text() == ""             # salvato in locale e il campo si svuota
        dlg.btn_requested.click()
        assert AC.requested() and AC.watching()
        assert dlg.btn_requested.isHidden() and not dlg.btn_check.isHidden() and not dlg.btn_snooze.isHidden() and not dlg.btn_cancel.isHidden()
        dlg.btn_snooze.click()
        assert AC.snoozed_until() > 0 and not AC.due()
        dlg.btn_cancel.click()
        assert not AC.watching() and not dlg.btn_requested.isHidden() and dlg.btn_check.isHidden()
        assert "hf_PROVA" not in dlg.text_label.text() and "hf_PROVA" not in dlg.watch_label.text()
    finally:
        dlg.close()


def test_help_dialog_offers_the_comfy_copy_until_the_comparison_says_no(qapp, prefs, monkeypatch):
    from ui.sam31_watch import Sam31Watcher
    from ui.widgets import sam31_help
    started = []
    w = Sam31Watcher(None, probe=lambda: ("pending", ""), start_download=lambda: None, tick_ms=10_000_000)
    monkeypatch.setattr(w, "download_comfy", lambda: started.append("comfy"))
    monkeypatch.setattr(SE, "COMFY_EQUIVALENT", None)
    dlg = sam31_help.show(None, status="missing", watcher=w)
    try:
        assert not dlg.btn_comfy.isHidden()
        dlg.btn_comfy.click()
        assert started == ["comfy"]
    finally:
        dlg.close()
    monkeypatch.setattr(SE, "COMFY_EQUIVALENT", False)                            # confronto negativo: l'alternativa non si offre
    dlg = sam31_help.show(None, status="missing", watcher=w)
    try:
        assert dlg.btn_comfy.isHidden()
    finally:
        dlg.close()
    dlg = sam31_help.show(None, status="ok", watcher=w)
    try:
        assert dlg.btn_comfy.isHidden() and dlg.token_edit is None
    finally:
        dlg.close()


def test_help_dialog_without_watcher_is_only_a_guide(qapp, prefs):
    from ui.widgets import sam31_help
    dlg = sam31_help.show(None, status="missing")
    try:
        assert all(b.isHidden() for b in (dlg.btn_requested, dlg.btn_check, dlg.btn_snooze, dlg.btn_cancel, dlg.btn_comfy))
    finally:
        dlg.close()


# ── ponte verso i job di Meta ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_job_command_has_the_launcher_guarantees():
    from core import sam31_detect as D
    cmd, env = D.job_command("sam31_image_text.py", ["--image", "x.png"])
    assert cmd[0].replace("\\", "/").endswith("diffusers_next_venv/Scripts/python.exe")
    assert cmd[1:3] == ["-P", "-s"]                                                  # la cartella dello script non entra in sys.path, niente site-packages utente
    assert cmd[-2:] == ["--image", "x.png"] and os.path.basename(cmd[4]) == "sam31_image_text.py"
    assert os.path.dirname(cmd[4]) == D.JOBS_DIR and Path(D.JOBS_DIR).name == "sam3_jobs"
    assert env["PYTHONPATH"].split(os.pathsep) == [D.SAM3_SRC, D.SAM3_DEPS]
    for bad in ("../x.py", "a/b.py", "a\\b.py"):
        with pytest.raises(ValueError):
            D.job_command(bad, [])


def test_repo_jobs_exist_and_never_load_with_unsafe_torch_load():
    d = Path(__file__).resolve().parents[1] / "workers" / "sam3_jobs"
    names = {p.name for p in d.glob("*.py")}
    assert {"sam31_image_text.py", "sam31_video_text.py", "sam31_compare_ckpts_image.py"} <= names
    for p in d.glob("*.py"):
        src = p.read_text(encoding="utf-8")
        assert "weights_only=False" not in src
        for line in src.splitlines():
            if "torch.load(" in line and "def _load" not in line:
                assert "weights_only=True" in line or "orig(" in line, f"{p.name}: {line.strip()}"


def test_detect_bridge_reads_the_job_line_and_sorts_boxes(tmp_path):
    from core import sam31_detect as D
    seen = []
    payload = {"boxes": [[1, 2, 30, 40], [5, 6, 50, 60]], "scores": [0.6, 0.9], "masks": [], "union": "", "presence": 0.99, "size": [100, 100], "device": "cpu"}

    def runner(cmd, env):
        seen.append((cmd, env))
        return 0, "rumore\nSAM31:" + json.dumps(payload) + "\n", ""
    r = D.detect("x.png", "person.", "cpu", 0.5, out_dir=str(tmp_path), checkpoint="C:/c.pt", runner=runner)
    assert r["scores"] == [0.6, 0.9]
    cmd = seen[0][0]
    assert cmd[cmd.index("--text") + 1] == "person" and cmd[cmd.index("--device") + 1] == "cpu" and cmd[cmd.index("--checkpoint") + 1] == "C:/c.pt"
    b = D.boxes("x.png", "person", "cpu", 0.5, out_dir=str(tmp_path), checkpoint="C:/c.pt", runner=runner)
    assert b[0] == [5.0, 6.0, 50.0, 60.0, 0.9] and b[1][4] == 0.6


def test_track_bridge_passes_video_text_and_limits(tmp_path):
    from core import sam31_detect as D
    got = {}

    def runner(cmd, env):
        got["cmd"] = cmd
        return 0, 'SAM31V:{"objects":[1,2],"frames":3}\n', ""
    r = D.track("v.mp4", "person", str(tmp_path), "cpu", max_frames=5, checkpoint="C:/c.pt", runner=runner)
    assert r["objects"] == [1, 2]
    cmd = got["cmd"]
    assert os.path.basename(cmd[4]) == "sam31_video_text.py" and cmd[cmd.index("--max-frames") + 1] == "5" and cmd[cmd.index("--video") + 1] == "v.mp4"


def test_detect_bridge_reports_a_job_failure_clearly():
    from core import sam31_detect as D
    with pytest.raises(RuntimeError) as e:
        D.detect("x.png", "person", checkpoint="C:/c.pt", runner=lambda cmd, env: (1, "", "Traceback ... ValueError: boom"))
    assert "SAM 3.1" in str(e.value) and "boom" in str(e.value)


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


def test_track_and_image_refuse_sam31_without_weights_with_the_access_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(SE, "models_dir", lambda: tmp_path / "vuoto")
    monkeypatch.setattr(SE, "comfy_dir", lambda: tmp_path / "vuoto_comfy")
    from inference_server.backends import image_segment as I
    from inference_server.backends import video_segment as V
    with pytest.raises(RuntimeError) as e:
        V.track_object("x.mp4", text="person", engine="sam31")
    assert "non e installato" in str(e.value) and "alcune ore" in str(e.value) and "Comfy-Org" in str(e.value) and "SAM2" in str(e.value)
    r = I.run({"seg_engine": "sam31", "image_path": "x.png", "text": "person"}, None, str(tmp_path))
    assert r["status"] == "error" and "SAM2" in r["error"]


def test_image_text_selection_runs_the_sam31_branch_not_gdino(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    from core import sam31_detect as D
    from inference_server.backends import image_segment as I
    monkeypatch.setattr(SE, "LOADER_READY", True)
    monkeypatch.setattr(SE, "weights_state", lambda base=None, comfy_base=None: "ok")
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
