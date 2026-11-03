"""Lavoratori del tab «Scambia» (thread): ricerca persone (SAM2 + DINO multi-box, nel venv dell'inference server) e catena di passate H3.

La catena riusa `_VideoEditAIWorker._run_h3` (stesso motore, stessi controlli, stesso log e ripresa dai clip): una passata per ruolo, la sorgente di ogni
passata e l'uscita della precedente, le maschere sono quelle del ruolo (precalcolate da «Trova persone», quindi niente nuovo tracking).
"""
from __future__ import annotations

import os
import subprocess
import time

from PySide6.QtCore import QThread, Signal

INF_PY = r"E:\AI\Tools\inference_server_venv\Scripts\python.exe"
VIDEO_SEGMENT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference_server", "backends", "video_segment.py")


class SwapFindWorker(QThread):
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, req: dict):
        super().__init__()
        self.req = req

    def run(self):
        from core import segmentation_engine as SE
        eng, err = SE.resolve_for_job(None)
        if err:
            self.done.emit({"status": "error", "error": err})
            return
        self.progress.emit("[scambia] cerco le persone (SAM2 + GroundingDINO)…")
        try:
            r = subprocess.run([INF_PY, "-u", VIDEO_SEGMENT, "--find-people", self.req["video"], self.req["work_dir"], self.req.get("text") or "person"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        except Exception as e:                                  # noqa: BLE001
            self.done.emit({"status": "error", "error": f"{type(e).__name__}: {e}"})
            return
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("[scambia]"):
                self.progress.emit(ln)
        line = next((ln for ln in (r.stdout or "").splitlines() if ln.startswith("TRACKS:")), "")
        if not line:
            from core.worker_errors import explain
            self.done.emit({"status": "error", "error": explain((r.stdout or "") + (r.stderr or ""), "", return_code=r.returncode)})
            return
        self.done.emit({"status": "success", "output": [line[len("TRACKS:"):].strip()]})


class SwapRunWorker(QThread):
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, req: dict, out_dir: str):
        super().__init__()
        self.req = req
        self.out_dir = out_dir

    def run(self):
        from ui.main_window import _VideoEditAIWorker
        req = self.req
        outs: list[str] = []
        base = os.path.splitext(os.path.basename(req["video"]))[0]
        n = len(req["passes"])
        for i, p in enumerate(req["passes"]):
            src = req["video"] if i == 0 else outs[-1]
            self.progress.emit(f"[scambia] passata {i + 1}/{n}: «{p['role']}»…")
            cfg = {"video": src, "mode": "replace_char", "engine": "h3_edit", "object": "person", "ref_image": p["photo"], "prompt": p.get("note", ""),
                   "h3_profile": req["h3_profile"], "short_edge": req["short_edge"], "seed": req["seed"], "mask_dir": p["mask_dir"], "keep_audio": True}
            res: dict = {}
            w = _VideoEditAIWorker(cfg, self.out_dir)
            w.done.connect(lambda r, _res=res: _res.update(r))
            w.progress.connect(self.progress.emit)
            w._run_h3(cfg, f"{base}_p{i + 1}", int(time.time()))
            if res.get("status") != "success" or not res.get("output"):
                self.done.emit({"status": "error", "error": f"Passata {i + 1}/{n} («{p['role']}») non riuscita:\n{res.get('error', 'nessuna uscita')}"})
                return
            outs.append(res["output"][0])
        self.done.emit({"status": "success", "output": [outs[-1]]})
