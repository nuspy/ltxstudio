"""Fase 0 (02/11): Inpaint video × checkpoint/GGUF LTX 2.5 dal tab «Modifica Video AI».

Errore dell'utente: «FileNotFoundError: No files matching pattern … find_matching_file» e messaggio tagliato in GUI. Causa: il worker
della GUI lanciava SEMPRE il venv della 2.3 (ltx2_venv, ltx-core 1.1.6) anche per un job 2.5: quel pacchetto non carica un text encoder da
file singolo e cerca «tokenizer.model» DENTRO il file .safetensors. Qui si verifica, con i subprocess FINTI (nessuna GPU, nessun SAM2):
- il python del worker LTX e quello della release scelta (lingbot_venv per 2.5, ltx2_venv per 2.3);
- i params portano release e GGUF come scelti nel tab;
- l'errore del worker arriva alla GUI INTERO (causa + coda del traceback), non tagliato a 300 caratteri.
"""
from __future__ import annotations

import json
import subprocess
import types

import harness as H

from core import ltx_release as lrel
from cases_video_inpaint_r2v import _cfg_of, _prep

_GGUF = r"E:\AI\Models\diffusion_models\LTX\ltx2.5-uncensored-v1.1-distilled-Q8_0.gguf"

_LONG_TB = ("Traceback (most recent call last):\n"
            + "".join(f'  File "E:\\AI\\Tools\\LTX-2\\packages\\ltx-core\\src\\ltx_core\\mod{i}.py", line {i}, in f{i}\n    x()\n' for i in range(12))
            + "FileNotFoundError: No files matching pattern 'tokenizer.model' found under "
              "E:\\AI\\Models\\text_encoders\\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors\n")


def _run(w, *, release: str | None, stderr: str = "finto"):
    """Esegue `_VideoEditAIWorker.run` (inpaint) con subprocess finti. Ritorna (python usato dal worker LTX, params, esito done)."""
    from ui import main_window as mw
    t, _img = _prep(w, use_ref=False)
    t._go()
    cfg = _cfg_of(w)
    cfg.update(width=1280, height=704, fps=25, num_frames=97)
    if release:
        cfg.update(ltx_release=release, gguf_transformer=_GGUF)
    else:
        cfg.pop("ltx_release", None)
        cfg.pop("gguf_transformer", None)
    wk = mw._VideoEditAIWorker(cfg, str(H.OUT_DIR))
    done, seen = [], {}
    wk.done.connect(lambda r: done.append(r))
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        cmd = [str(c) for c in cmd]
        if cmd and cmd[0].lower().endswith("python.exe") and any(c.endswith("ltx2_infer.py") for c in cmd):
            seen["py"] = cmd[0]
            seen["params"] = json.load(open(cmd[-1], encoding="utf-8"))
            return types.SimpleNamespace(stdout="", stderr=stderr, returncode=1)
        if cmd and cmd[0].lower().endswith("python.exe"):                      # SAM2 (video_segment.py)
            return types.SimpleNamespace(stdout="GUIDE:g.mp4\n", stderr="", returncode=0)
        return types.SimpleNamespace(stdout="", stderr="", returncode=1)

    from core import ltx_video_job as _lvj
    from core import mem_coordinator as _mc
    _o_fw, _o_fv = _mc.free_and_wait, _lvj.free_video_vram
    _mc.free_and_wait = lambda *a, **k: 0
    _lvj.free_video_vram = lambda *a, **k: 0
    subprocess.run = fake_run
    try:
        wk.run()
    finally:
        subprocess.run = real_run
        _mc.free_and_wait, _lvj.free_video_vram = _o_fw, _o_fv
    return seen, done


def case_inpaint_ltx25_gguf_uses_release_venv(w, spy):
    """Checkpoint GGUF 2.5: il worker LTX parte dal venv della 2.5 (lingbot_venv), mai da ltx2_venv; release e GGUF nei params."""
    seen, done = _run(w, release="2.5")
    assert seen, f"il worker LTX non e' stato invocato: {done}"
    want_py, _w = lrel.runner("2.5")
    assert seen["py"].lower() == want_py.lower(), f"venv sbagliato per LTX 2.5: {seen['py']} (atteso {want_py})"
    assert "lingbot_venv" in seen["py"].lower() and "ltx2_venv" not in seen["py"].lower()
    p = seen["params"]
    assert p["ltx_release"] == "2.5" and p["gguf_transformer"] == _GGUF, p
    assert "inpaint_masked_t2v" in p["edit_lora"] and p["inpaint_masks_dir"].endswith("masks")


def case_inpaint_ltx23_keeps_ltx2_venv(w, spy):
    """Nessun checkpoint 2.5 (default 2.3): il venv resta ltx2_venv (comportamento di sempre)."""
    seen, _done = _run(w, release=None)
    want_py, _w = lrel.runner(None)
    assert seen["py"].lower() == want_py.lower() and "ltx2_venv" in seen["py"].lower(), seen["py"]


def case_inpaint_worker_error_reaches_gui_whole(w, spy):
    """L'errore del worker arriva INTERO: causa (quale file cercava e dove) in testa + coda del traceback; log GUI senza tagli."""
    _seen, done = _run(w, release="2.5", stderr=_LONG_TB)
    assert done and done[0]["status"] == "error", done
    err = done[0]["error"]
    assert err.splitlines()[0].startswith("FileNotFoundError: No files matching pattern 'tokenizer.model' found under"), err[:200]
    assert err.splitlines()[0].endswith("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"), err.splitlines()[0]
    assert "Traceback (most recent call last)" in err and len(err) > 500
    t = w.video_edit_ai_tab.log
    w._on_video_edit_ai_done(done[0])
    shown = t.toPlainText()
    assert "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" in shown and "tokenizer.model" in shown, shown[-400:]


def case_inpaint_num_frames_never_exceed_source(w, spy):
    """Il numero di frame dell'edit e il piu grande 8n+1 CONTENUTO nel sorgente: 72 frame (3 s a 24 fps) -> 65, mai 73 (giro 2/11: maschere da 72
    frame e latente da 73 = AssertionError «Pixel frames (72) not compatible with latent frames (9)» di ltx-pipelines)."""
    t, _ = _prep(w, use_ref=False)
    real = w._probe_video_dims
    try:
        for n, want in ((72, 65), (73, 73), (97, 97), (96, 89), (100, 97), (9, 9), (5, 9)):
            w._probe_video_dims = lambda path, _n=n: (640, 384, _n, 24)
            t._go()
            got = w._edit_ai_worker.cfg["num_frames"]
            assert got == want, f"sorgente da {n} frame: num_frames={got}, atteso {want}"
            assert got <= max(9, n), (n, got)
    finally:
        w._probe_video_dims = real


CASES = [
    ("vinpaint_ltx25_gguf_release_venv", "inpaint × GGUF LTX 2.5: il worker parte dal venv della 2.5, release e GGUF nei params.",
     case_inpaint_ltx25_gguf_uses_release_venv),
    ("vinpaint_ltx23_keeps_ltx2_venv", "inpaint senza checkpoint 2.5: venv della 2.3 come sempre.", case_inpaint_ltx23_keeps_ltx2_venv),
    ("vinpaint_worker_error_whole", "inpaint: l'errore del worker arriva alla GUI intero (causa + traceback), non tagliato.",
     case_inpaint_worker_error_reaches_gui_whole),
    ("vinpaint_frames_8n1_within_source", "inpaint/edit: frame = 8n+1 contenuto nel sorgente (72 -> 65, non 73).",
     case_inpaint_num_frames_never_exceed_source),
]
