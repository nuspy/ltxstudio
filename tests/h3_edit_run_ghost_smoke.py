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


calls, refs = [], []


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
    def load_vae(W): return torch.nn.Linear(1, 1)

    @staticmethod
    def load_audio_vae(W): return torch.nn.Linear(1, 1)

    @staticmethod
    def _decode_audio_av(path): return None

    @staticmethod
    def run(pc):
        th, tw = pc["_h3_edit"]["canvas"]
        calls.append(pc["num_frames"])
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


def run(tag, env):
    for k in ("H3_EDIT_PREROLL", "H3_EDIT_BG_PERSON_REF", "H3_EDIT_BG_MATTE"):
        os.environ.pop(k, None)
    os.environ.update(env)
    calls.clear(); refs.clear()
    out = ER.run_edit({"output": str(TMP / tag / "out.mp4"), "seed": 1,
                       "h3_edit": {"action": "change_background", "video": str(src), "mask_dir": str(TMP / "masks"), "short_edge": 192,
                                   "clip_s": 6.0, "overlap": 39, "keep_audio": True, "prompt_text": "x"}}, FakeW,
                      clip_runner=lambda W, jd: ER.generate_clip(jd, FakeW))
    return frames_of(out), list(calls), list(refs)


base, c0, r0 = run("base", {})
pre, c1, r1 = run("preroll", {"H3_EDIT_PREROLL": "17"})
man, c2, r2 = run("mann", {"H3_EDIT_BG_PERSON_REF": "mannequin"})
mat, c3, r3 = run("matte", {"H3_EDIT_BG_MATTE": "1"})
print("clip base", c0, "preroll", c1)

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
