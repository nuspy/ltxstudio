"""Fase 2 (Openjutsu, 03/11): tab «🎭 Scambia». Widget -> backend: i ruoli scelti nella GUI diventano passate H3 con le maschere giuste, la foto giusta e la
catena di sorgenti; errori chiari prima di generare. Subprocess e GPU FINTI: nessun SAM2, nessun H3."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

import harness as H

from core import swap_project as SP
from core import video_edit_engines as veng


def _tracks(tmp, n=2, frames=4):
    os.makedirs(tmp, exist_ok=True)
    tr = []
    fdir = os.path.join(tmp, "frames")
    os.makedirs(fdir, exist_ok=True)
    for f in range(frames):
        Image.fromarray(np.full((48, 64, 3), 90, np.uint8)).save(os.path.join(fdir, f"{f:05d}.png"))
    for i in range(1, n + 1):
        md = os.path.join(tmp, f"track_{i}", "masks")
        os.makedirs(md, exist_ok=True)
        for f in range(frames):
            m = np.zeros((48, 64), np.uint8)
            m[8:40, 4 + 28 * (i - 1): 30 + 28 * (i - 1)] = 255
            Image.fromarray(m).save(os.path.join(md, f"{f:05d}.png"))
        tr.append(SP.Track(id=i, box=[0, 0, 1, 1], label=f"Persona {i}", mask_dir=md, frames=frames, area=0.3 - i * 0.01))
    return tr


def _tab(w, n=2):
    t = w.swap_tab
    w.tabs.setCurrentWidget(t)
    tmp = str(H.OUT_DIR / "_swap_gui")
    os.makedirs(tmp, exist_ok=True)
    vid = os.path.join(tmp, "v.mp4")
    open(vid, "wb").write(b"\x00\x00\x00\x18ftypmp42")
    t.video, t.work_dir = vid, tmp
    t.set_tracks(_tracks(tmp, n))
    while len(t._roles) > 1:
        t._del_role()
    t._roles[0].photo = ""
    t.profile.clear()
    t.profile.addItem("INT8", "int8")
    return t, tmp


def _photo(tmp, name="p.png"):
    p = os.path.join(tmp, name)
    Image.fromarray(np.full((16, 16, 3), 200, np.uint8)).save(p)
    return p


def case_tab_refuses_incomplete_roles(w, spy):
    t, tmp = _tab(w)
    try:
        t.build_request()
        raise AssertionError("avrebbe dovuto rifiutare: nessuna foto")
    except ValueError as e:
        assert "manca la foto" in str(e), e
    t._roles[0].set_photo(_photo(tmp))
    req = t.build_request()
    assert len(req["passes"]) == 1 and req["passes"][0]["photo"].endswith("p.png")


def case_roles_become_passes_with_the_right_masks_and_photos(w, spy):
    t, tmp = _tab(w)
    t._add_role()
    t._roles[0].person.setCurrentIndex(1)                 # il personaggio 1 sostituisce la Persona 2
    t._roles[1].person.setCurrentIndex(0)
    t._roles[0].set_photo(_photo(tmp, "a.png"))
    t._roles[1].set_photo(_photo(tmp, "b.png"))
    t._roles[1].note.setText("giacca rossa")
    req = t.build_request()
    p0, p1 = req["passes"]
    assert p0["mask_dir"].endswith(os.path.join("track_2", "masks")) and p0["photo"].endswith("a.png")
    assert p1["mask_dir"].endswith(os.path.join("track_1", "masks")) and p1["note"] == "giacca rossa"
    t._roles[1].person.setCurrentIndex(1)                 # stessa persona a due ruoli: errore chiaro
    try:
        t.build_request()
        raise AssertionError("persona assegnata due volte")
    except ValueError as e:
        assert "assegnata sia a" in str(e)


def case_generate_runs_one_h3_pass_per_role_in_chain(w, spy):
    t, tmp = _tab(w)
    t._add_role()
    t._roles[0].person.setCurrentIndex(0)
    t._roles[1].person.setCurrentIndex(1)
    t._roles[0].set_photo(_photo(tmp, "a.png"))
    t._roles[1].set_photo(_photo(tmp, "b.png"))
    seen = []

    from ui import main_window as MW
    old_run = MW._VideoEditAIWorker._run_h3

    def fake_run(self, cfg, base, stamp):
        seen.append(dict(cfg))
        out = os.path.join(tmp, f"fake_out_{len(seen)}.mp4")
        open(out, "wb").write(b"x")
        self.done.emit({"status": "success", "output": [out]})
    MW._VideoEditAIWorker._run_h3 = fake_run
    try:
        req = t.build_request()
        from ui.swap_workers import SwapRunWorker
        res = {}
        wk = SwapRunWorker(req, tmp)
        wk.done.connect(lambda r: res.update(r))
        wk.run()                                           # in questo thread: niente QThread vero
    finally:
        MW._VideoEditAIWorker._run_h3 = old_run
    assert res.get("status") == "success", res
    assert len(seen) == 2
    assert seen[0]["video"] == t.video and seen[1]["video"].endswith("fake_out_1.mp4"), (seen[0]["video"], seen[1]["video"])
    assert seen[0]["mask_dir"].endswith(os.path.join("track_1", "masks")) and seen[1]["mask_dir"].endswith(os.path.join("track_2", "masks"))
    assert seen[0]["ref_image"].endswith("a.png") and seen[1]["ref_image"].endswith("b.png")
    assert all(c["mode"] == "replace_char" and c["engine"] == "h3_edit" and c["h3_profile"] == "int8" for c in seen)
    assert res["output"][0].endswith("fake_out_2.mp4")


def case_run_h3_uses_precomputed_masks_without_tracking(w, spy):
    """_VideoEditAIWorker._run_h3 con cfg['mask_dir']: nessun tracking SAM2 e le maschere del ruolo arrivano ai params del worker H3."""
    import json
    import subprocess
    from core import video_inpaint as vi
    from ui import main_window as MW
    t, tmp = _tab(w)
    called = {"track": 0, "params": None}
    old_track, old_popen, old_msg = vi.attach_tracking, subprocess.Popen, veng.missing_message

    def no_track(*a, **k):
        called["track"] += 1
        return {}

    class FakePopen:
        def __init__(self, argv, **kw):
            called["params"] = json.load(open(argv[-1], encoding="utf-8"))
            self.stdout = iter(["ERRORE: finto\n"])

        def wait(self):
            return 1
    vi.attach_tracking = no_track
    subprocess.Popen = FakePopen
    veng.missing_message = lambda *a, **k: ""
    try:
        from core import video_edit_h3 as vh3
        old_inst = vh3.installed_profiles
        vh3.installed_profiles = lambda present=None: ["int8"]
        done = {}
        wk = MW._VideoEditAIWorker({"video": t.video, "mode": "replace_char", "engine": "h3_edit", "object": "person", "ref_image": _photo(tmp, "r.png"),
                                    "h3_profile": "int8", "short_edge": 480, "seed": 1, "mask_dir": t.tracks[1].mask_dir}, tmp)
        wk.done.connect(lambda r: done.update(r))
        wk._run_h3(wk.cfg, "x", 1)
    finally:
        vi.attach_tracking, subprocess.Popen, veng.missing_message = old_track, old_popen, old_msg
        vh3.installed_profiles = old_inst
    assert called["track"] == 0, "il tracking SAM2 e ripartito nonostante le maschere gia calcolate"
    assert called["params"] and called["params"]["h3_edit"]["mask_dir"] == t.tracks[1].mask_dir, called["params"]
    assert called["params"]["h3_edit"]["action"] == "replace_character"


def case_preview_shows_chosen_people(w, spy):
    t, tmp = _tab(w)
    t._roles[0].person.setCurrentIndex(0)
    t._refresh_preview()
    pm = t.preview.pixmap()
    assert pm is not None and not pm.isNull()


CASES = [
    ("swap_incomplete_refused", "Scambia: ruoli incompleti rifiutati con messaggio chiaro.", case_tab_refuses_incomplete_roles),
    ("swap_roles_to_passes", "Scambia: i ruoli scelti diventano passate con maschere e foto giuste; persona doppia rifiutata.",
     case_roles_become_passes_with_the_right_masks_and_photos),
    ("swap_chain_backend", "Scambia: una passata H3 per ruolo, in catena (sorgente = uscita precedente), maschere per ruolo.",
     case_generate_runs_one_h3_pass_per_role_in_chain),
    ("swap_precomputed_masks", "Scambia: le maschere del ruolo arrivano al worker H3 senza nuovo tracking SAM2.", case_run_h3_uses_precomputed_masks_without_tracking),
    ("swap_preview", "Scambia: l'anteprima della maschera mostra le persone scelte.", case_preview_shows_chosen_people),
]
