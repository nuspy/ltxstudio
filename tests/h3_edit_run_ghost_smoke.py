"""Fumo CPU (venv diffusers, W finto, niente pesi/GPU) delle opzioni «doppione/lip sync del clip 1» del 03/11 nel cambia-sfondo:
 · PREROLL (H3_EDIT_PREROLL=17): frame fermi anteposti, il video in uscita ha lo STESSO numero di frame e il clip 1 e piu lungo di 17 frame;
 · riferimento «manichino» (H3_EDIT_BG_PERSON_REF=mannequin): nel <Video 1> la persona diventa grigia liscia (non piu rossa/nitida);
 · matte a pixel (H3_EDIT_BG_MATTE=1): dentro la sagoma i pixel sono quelli ORIGINALI (con generazione finta casuale, senza matte non lo sono).
Si lancia:  E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe tests\\h3_edit_run_ghost_smoke.py   (stampa «GHOST SMOKE OK»)."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workers"))

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import h3_edit_run as ER  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="h3edit_ghost_"))
N_SRC, SW, SH = 24 * 14, 320, 192


def make_video(path):
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = SW, SH, "yuv420p"
        for i in range(N_SRC):
            img = np.zeros((SH, SW, 3), np.uint8)
            img[:, :, 1] = 40
            x = 40 + i * 4 % 200
            img[60:150, x:x + 40] = (220, 60, 60)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def make_masks(d):
    d.mkdir()
    for i in range(N_SRC):
        m = np.zeros((SH, SW), np.uint8)
        x = 40 + i * 4 % 200
        m[60:150, x:x + 40] = 255
        Image.fromarray(m).save(d / f"{i:05d}.png")


calls, refs, prompts = [], [], []


class FakeW:
    REPO = ROOT
    S = None

    @staticmethod
    def _log(m): print("[fake]", m, flush=True)

    @staticmethod
    def resolve(p): return {}

    @staticmethod
    def _gc(): pass

    @staticmethod
    def _drop_module_weights(m): pass

    @staticmethod
    def load_vae(W, encoder_only=False): return torch.nn.Linear(1, 1)

    @staticmethod
    def load_audio_vae(W): return torch.nn.Linear(1, 1)

    @staticmethod
    def _decode_audio_av(path): return None

    @staticmethod
    def run(pc):
        th, tw = pc["_h3_edit"]["canvas"]
        calls.append(pc["num_frames"])
        prompts.append(pc["prompt"])
        refs.append(pc["references"][-1]["path"])
        g = torch.Generator().manual_seed(7)
        return {"videos": [(torch.rand(pc["num_frames"], th, tw, 3, generator=g) * 255).to(torch.uint8)], "audio": [None], "sampling_rate": 32000}


def fake_encode(vae, frames_u8, comp):
    n, h, w = frames_u8.shape[:3]
    return torch.randn(1, 24, 5 * ((n - 5) // 17) + 2, h // 16, w // 16)


ER.encode_source_latent = fake_encode
src = TMP / "src.mp4"
make_video(src)
make_masks(TMP / "masks")


def frames_of(path):
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])])


def run(tag, env, extra=None):
    for k in ("H3_EDIT_PREROLL", "H3_EDIT_BG_PERSON_REF", "H3_EDIT_BG_MATTE", "H3_EDIT_BG_HOLD_COVER", "H3_EDIT_FIRST_TWO_PASS", "H3_EDIT_BG_PROMPT", "H3_EDIT_LIPSYNC", "H3_EDIT_BG_GROW_LIKE_REPLACE", "H3_EDIT_MAX_FRAMES", "H3_EDIT_BG_MASK", "H3_EDIT_BG_MASK_MARGIN", "H3_EDIT_BG_MASK_FILL_HOLES", "H3_INT8_GEMM", "H3_NVFP4_GEMM", "H3_VAE_DECODE_DTYPE", "H3_NFE", "H3_INT8_ATTENTION"):
        os.environ.pop(k, None)
    os.environ.update(env)
    calls.clear(); refs.clear(); prompts.clear()
    out = ER.run_edit({"output": str(TMP / tag / "out.mp4"), "seed": 1,
                       "h3_edit": {"action": "change_background", "video": str(src), "mask_dir": str(TMP / "masks"), "short_edge": 192,
                                   "clip_s": 6.0, "overlap": 39, "keep_audio": True, "prompt_text": "x", **(extra or {})}}, FakeW,
                      clip_runner=lambda W, jd: ER.generate_clip(jd, FakeW))
    return frames_of(out), list(calls), list(refs)


base, c0, r0 = run("base", {"H3_EDIT_BG_MATTE": "0"})                       # il matte e di default: qui spento per avere la base senza
pre, c1, r1 = run("preroll", {"H3_EDIT_PREROLL": "17", "H3_EDIT_BG_MATTE": "0"})
man, c2, r2 = run("mann", {"H3_EDIT_BG_PERSON_REF": "mannequin", "H3_EDIT_BG_MATTE": "0"})
mat, c3, r3 = run("matte", {})                                                 # senza variabili: matte di DEFAULT
cov, c4, r4 = run("cover", {"H3_EDIT_BG_HOLD_COVER": "0.5"})                    # celle tenute con persona >= 50 %: gira e produce lo stesso numero di frame
assert cov.shape == base.shape and c4 == c0, (c4, c0)
two, c5, r5 = run("two", {"H3_EDIT_FIRST_TWO_PASS": "1"})                       # clip 1 in due passate: un clip in piu (la passata B), stesso numero di frame
assert two.shape == base.shape and len(c5) == len(c0) + 1 and c5[:2] == [141, 141], (c5, c0)
import json  # noqa: E402
tdir = next((TMP / "two" / "_h3edit_tmp").iterdir())
mb = json.loads((tdir / "clip_00b" / "meta.json").read_text("utf-8"))
assert mb["two_pass"] == 22 and mb["ctx"] == 22 and mb["ci"] == 0, mb
ca, cb = np.load(tdir / "clip_00" / "cells.npy"), np.load(tdir / "clip_00b" / "cells.npy")
assert ca.shape == cb.shape and float(cb[:, :, :7].max()) == 0.0 and float(ca[:, :, :7].max()) > 0, (ca[:, :, :7].max(), cb[:, :, :7].max())   # 22 frame = 7 frame di latente tenuti
assert np.array_equal(ca[:, :, 7:], cb[:, :, 7:])                                  # il resto della maschera e quello di sempre
sa, sb = np.load(tdir / "clip_00" / "clip_src.npy"), np.load(tdir / "clip_00b" / "clip_src.npy")
assert not np.array_equal(sa[:22], sb[:22]) and np.array_equal(sa[22:], sb[22:])   # solo i primi 22 sono i frame puliti della passata A
assert not (tdir / "clip_01" / "meta.json").exists() or json.loads((tdir / "clip_01" / "meta.json").read_text("utf-8"))["ctx"] == 39
era, c6, r6 = run("erase", {"H3_EDIT_BG_PERSON_REF": "erase", "H3_EDIT_BG_MATTE": "0"})   # riferimento senza persona: dove c'era la figura rossa c'e lo sfondo sfocato
# ABLAZIONE (un fattore alla volta, solo clip 1): un solo clip con H3_EDIT_MAX_FRAMES; prompt, lip sync e maschera come Sostituisci
one, cb, _r = run("abl0", {"H3_EDIT_MAX_FRAMES": "141"}); p0 = list(prompts)
assert cb == [141] and "exact look" in p0[0] and "the actions" in p0[0], (cb, p0[0][:200])
_k = lambda tag: next((TMP / tag / "_h3edit_tmp").iterdir())                          # noqa: E731
_, _c, _ = run("abl1", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_BG_PROMPT": "no_performance"})
assert _c == [141] and "exact look" not in prompts[0] and "the actions" not in prompts[0] and "setting around the person becomes" in prompts[0], prompts[0][:300]
_, _c, _ = run("abl2", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_LIPSYNC": "0"})
assert json.loads((_k("abl2") / "clip_00" / "meta.json").read_text("utf-8"))["lipsync"] is False and _k("abl2").name != _k("abl0").name   # runkey diversa
_, _c, _ = run("abl3", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_BG_PROMPT": "carryover"})
assert "Its first 1.625 seconds are already-edited footage" in prompts[0] and "exact look" in prompts[0], prompts[0][:300]
_, _c, _ = run("abl4", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_BG_PERSON_REF": "erase", "H3_EDIT_BG_PROMPT": "base"})
assert "exact look" in prompts[0] and "no person" not in prompts[0].split("summary:")[0], prompts[0][:200]          # erase con prompt base: il prompt di sempre
assert json.loads((_k("abl4") / "clip_00" / "meta.json").read_text("utf-8")).get("bg_prompt") is None
_, _c, _ = run("abl5", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_BG_GROW_LIKE_REPLACE": "1"})
c_base, c_like = np.load(_k("abl0") / "clip_00" / "cells.npy"), np.load(_k("abl5") / "clip_00" / "cells.npy")
assert float((c_like < 0.5).sum()) > float((c_base < 0.5).sum()), (float((c_like < 0.5).sum()), float((c_base < 0.5).sum()))   # persona e margine TENUTI: piu celle tenute
_, _c, _ = run("abl6", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_BG_PROMPT": "replace_like"})
assert "exact look" not in prompts[0] and "follows the timing of the performance of the figure" in prompts[0] and "performs every action" in prompts[0], prompts[0][:300]
_, _c, _ = run("abl7", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_BG_PROMPT": "align"})
assert "frame-accurate" in prompts[0] and "exact look" in prompts[0] and "Its first" not in prompts[0], prompts[0][:300]
_, _c, _ = run("abl8", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_BG_PROMPT": "carryover_align"})
assert "frame-accurate" in prompts[0] and "Its first 1.625 seconds" in prompts[0], prompts[0][:300]
_, _c, _ = run("txt", {"H3_EDIT_MAX_FRAMES": "141"}, extra={"text_info": {"status": "ok", "seconds": 3.0, "original": {"prompt": "un rifugio"}, "sent": {"prompt": "A refuge."}}})
_tj = json.loads((_k("txt") / "text_job.json").read_text("utf-8"))
assert _tj["original"] == {"prompt": "un rifugio"} and _tj["sent"] == {"prompt": "A refuge."} and "text_info" not in json.loads((_k("txt") / "clip_00" / "meta.json").read_text("utf-8"))
_, _c, _ = run("orig", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_BG_MASK": "original", "H3_EDIT_BG_MASK_MARGIN": "40", "H3_EDIT_BG_MATTE": "0"})
c_orig = np.load(_k("orig") / "clip_00" / "cells.npy")
assert float((c_orig < 0.5).sum()) < float((c_base < 0.5).sum()), (float((c_orig < 0.5).sum()), float((c_base < 0.5).sum()))   # persona ristretta: meno celle tenute dello standard
assert _k("orig").name != _k("abl0").name                                                                                      # runkey diversa
_, _c, _ = run("orig_fh", {"H3_EDIT_MAX_FRAMES": "141", "H3_EDIT_BG_MASK": "original", "H3_EDIT_BG_MASK_MARGIN": "40", "H3_EDIT_BG_MATTE": "0", "H3_EDIT_BG_MASK_FILL_HOLES": "1"})
c_fh = np.load(_k("orig_fh") / "clip_00" / "cells.npy")
assert float((c_fh < 0.5).sum()) <= float((c_orig < 0.5).sum()) and _k("orig_fh").name != _k("orig").name           # fill_holes: non tiene di piu e ha una chiave sua
# NUMERICA nella chiave: stessa cartella di uscita, env diverso => il clip si rigenera (turno 42: i GEMM riprendevano il clip di «tutto attivo»); stesso env => ripresa
_, c_first, _r = run("numerics", {"H3_EDIT_MAX_FRAMES": "141"})
_, c_second, _r = run("numerics", {"H3_EDIT_MAX_FRAMES": "141", "H3_INT8_GEMM": "1"})
_, c_third, _r = run("numerics", {"H3_EDIT_MAX_FRAMES": "141", "H3_INT8_GEMM": "1"})
_, c_fourth, _r = run("numerics", {"H3_EDIT_MAX_FRAMES": "141", "H3_VAE_DECODE_DTYPE": "fp32"})
assert c_first == [141] and c_second == [141] and c_third == [] and c_fourth == [141], (c_first, c_second, c_third, c_fourth)
assert len(list((TMP / "numerics" / "_h3edit_tmp").iterdir())) == 3                                        # tre numeriche diverse, tre cartelle di job
print("ABLAZIONE OK: kept cells base", int((c_base < 0.5).sum()), "like-replace", int((c_like < 0.5).sum()))
print("clip base", c0, "preroll", c1, "two-pass", c5)

# PREROLL: stesso numero di frame in uscita, ma si genera di piu (clip 1 piu lungo di 17 frame)
assert pre.shape[0] == base.shape[0], (pre.shape, base.shape)
assert sum(c1) == sum(c0) + 17 or max(c1) >= max(c0), (c0, c1)
src_f = frames_of(str(src))
# RIFERIMENTO MANICHINO: la persona del <Video 1> e grigia liscia (R ~ G ~ B), con lo standard e rossa
def person_color(ref_path):
    f = frames_of(ref_path)[5]
    reg = f[60:150, 40 + 5 * 4 % 200:40 + 5 * 4 % 200 + 40].reshape(-1, 3).astype(int)
    return reg.mean(0)
sharp, mq = person_color(r0[0]), person_color(r2[0])
er = person_color(r6[0])
print("colore nella zona della persona, riferimento erase", er.round(), "bordo/sfondo vicino", frames_of(r6[0])[5][60:150, 10:30].reshape(-1, 3).mean(0).round())
_ed = next((TMP / "erase" / "_h3edit_tmp").iterdir())
assert json.loads((_ed / "clip_00" / "meta.json").read_text("utf-8")).get("bg_prompt") == "empty_person"     # con erase il prompt dice «zona vuota»
_bd = next((TMP / "base" / "_h3edit_tmp").iterdir())
assert "bg_prompt" not in json.loads((_bd / "clip_00" / "meta.json").read_text("utf-8"))                      # senza erase meta e quello di sempre
assert era.shape == base.shape and er[0] - er[2] < 30 and er[0] < 120 and sharp[0] - er[0] > 100, (er, sharp)
print("colore persona nel riferimento: standard", sharp.round(), "manichino", mq.round())
assert sharp[0] - sharp[2] > 60 and abs(mq[0] - mq[2]) < 40, (sharp, mq)
# MATTE: dentro la sagoma (erosa) i pixel sono quelli del video originale; senza matte (gen casuale) no
def inside_err(out, t=30):
    x = 40 + t * 4 % 200
    reg = (slice(66, 144), slice(x + 6, x + 34))
    return float(np.abs(out[t][reg].astype(int) - src_f[t][reg].astype(int)).mean())
e_base, e_mat = np.mean([inside_err(base, t) for t in (30, 60, 100)]), np.mean([inside_err(mat, t) for t in (30, 60, 100)])
print("errore dentro la sagoma: base", round(e_base, 1), "matte", round(e_mat, 1))
assert e_mat < 12 and e_mat < e_base * 0.5, (e_base, e_mat)
print("GHOST SMOKE OK")
