"""Storyboard su video a BLOCCHI + conflitto con ControlNet — CPU-only.

Bug del 17/08: `_run_chunked` faceva `cp = dict(p)` e i keyframe (in coordinate GLOBALI)
finivano IDENTICI in ogni blocco, col frame clampato a nf-1 del blocco → le stesse immagini
riapplicate ovunque. E con ControlNet attivo il ramo IC-LoRA legge solo `image_path`, quindi lo
storyboard spariva senza un messaggio.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def partition(keyframes, chunk, total):
    """Ripartizione usata da _run_chunked (stessa aritmetica, isolata per il test)."""
    out = []
    n = (total + chunk - 1) // chunk
    for i in range(n):
        cf = min(chunk, total - i * chunk)
        if cf < 9:
            break
        lo = i * chunk
        out.append([{**kf, "frame": int(kf["frame"]) - lo}
                    for kf in keyframes if lo <= int(kf["frame"]) < lo + cf])
    return out


def test_ogni_keyframe_in_un_solo_blocco():
    kfs = [{"path": f"{i}.png", "frame": f} for i, f in enumerate((0, 40, 95, 130, 200))]
    parts = partition(kfs, chunk=100, total=241)          # blocchi: [0,100) [100,200) [200,241)
    assert [len(x) for x in parts] == [3, 1, 1], [len(x) for x in parts]
    assert [k["frame"] for k in parts[0]] == [0, 40, 95]  # già locali (blocco che parte da 0)
    assert [k["frame"] for k in parts[1]] == [30]         # globale 130 → locale 30
    assert [k["frame"] for k in parts[2]] == [0]          # globale 200 → locale 0
    for part, cf in zip(parts, (100, 100, 41)):           # mai fuori dal proprio blocco
        assert all(0 <= k["frame"] < cf for k in part)
    tutti = [k["path"] for part in parts for k in part]   # era il bug: tutte in tutti i blocchi
    assert len(tutti) == len(set(tutti)) == 5


def test_blocco_senza_keyframe_torna_alla_continuita():
    kfs = [{"path": "a.png", "frame": 0}, {"path": "b.png", "frame": 10}]
    parts = partition(kfs, chunk=100, total=300)
    assert len(parts[0]) == 2 and parts[1] == [] and parts[2] == []


def _worker_src() -> str:
    """Sorgente del worker LETTO come testo: importarlo richiederebbe ltx_core (venv LTX)."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "workers" / "ltx2_infer.py"
    return p.read_text(encoding="utf-8")


def test_worker_ripartisce_e_non_clampa():
    """Il codice reale deve contenere la ripartizione (non più il dict(p) cieco)."""
    src = _worker_src()
    i = src.find("def _run_chunked(")
    assert i > 0
    blocco = src[i:src.find("\ndef ", i + 10)]
    assert 'p.get("keyframes")' in blocco, "_run_chunked non guarda i keyframe"
    assert "_lo = i * chunk" in blocco and 'int(kf.get("frame", 0)) - _lo' in blocco, \
        "manca la conversione a coordinate locali"


def test_iclora_dichiara_i_keyframe_ignorati():
    """NB: `elif ptype == "iclora"` compare DUE volte (caricamento LoRA e chiamata pipeline):
    qui interessa il secondo, quello che costruisce le ImageConditioningInput."""
    src = _worker_src()
    i = src.find('cv = p["control_video"]')
    assert i > 0, "ramo IC-LoRA della chiamata non trovato"
    assert "IGNORATI" in src[i - 500:i + 1200], "il ramo IC-LoRA non dichiara i keyframe scartati"


def test_gui_blocca_storyboard_con_controlnet():
    import inspect
    import ui.main_window as mw
    src = inspect.getsource(mw.MainWindow._generate_video_native)
    assert "sb_controlnet_conflict" in src, "manca l'errore esplicito storyboard+ControlNet"
    # e deve fermarsi PRIMA di impostare i keyframe
    assert src.index("sb_controlnet_conflict") < src.index('params["keyframes"] = kfs')
