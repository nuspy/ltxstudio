"""Fumo CPU di workers/h3_edit_run.py con un `W` finto (nessun peso, nessuna GPU): video sintetico + maschere + catena di clip +
composizione + mp4 finale. Si lancia col python del venv diffusers:
  E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe tests\\h3_edit_run_smoke.py
Verifica: forme latente/maschera coerenti, righe tenute (hold) costruibili, clip in catena che coprono tutto il video, file finale."""
import os
import sys
import tempfile
import types
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

TMP = Path(tempfile.mkdtemp(prefix="h3edit_smoke_"))
N_SRC, SW, SH = 24 * 14, 320, 192                                  # 14 s a 24 fps -> piu di un clip da 6 s


def make_video(path):
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = SW, SH, "yuv420p"
        for i in range(N_SRC):
            img = np.zeros((SH, SW, 3), np.uint8)
            img[:, :, 1] = 40
            x = 40 + i * 4 % 200
            img[60:150, x:x + 40] = (220, 60, 60)                    # la «persona» che si muove
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


calls = []


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
    def _decode_audio_av(path): return None                       # senza audio: ramo audio non esercitato qui

    @staticmethod
    def run(pc):
        e = pc["_h3_edit"]
        th, tw = e["canvas"]
        src, noise, keep = e["video_hold"]((1, 2, 2))
        assert src.shape == noise.shape == keep.shape, (src.shape, noise.shape, keep.shape)
        T = 5 * ((pc["num_frames"] - 5) // 17) + 2
        assert src.shape[0] == T * (th // 32) * (tw // 32), (src.shape, T, th, tw)       # righe = frame * celle 2x2 di latente
        calls.append((pc["num_frames"], float(keep.mean())))
        assert pc["references"][-1]["kind"] == "video" and os.path.isfile(pc["references"][-1]["path"])
        frames = (torch.rand(pc["num_frames"], th, tw, 3) * 255).to(torch.uint8)
        return {"videos": [frames], "audio": [None], "sampling_rate": 32000}


# encode finto: latente (1, 24, T, h/16, w/16) in float
def fake_encode(vae, frames_u8, comp):
    n, h, w = frames_u8.shape[:3]
    return torch.randn(1, 24, 5 * ((n - 5) // 17) + 2, h // 16, w // 16)


ER.encode_source_latent = fake_encode
src = TMP / "src.mp4"
make_video(src)
make_masks(TMP / "masks")
for action in ("change_background", "replace_character"):
    calls.clear()
    out = ER.run_edit({"output": str(TMP / f"out_{action}.mp4"), "seed": 1,
                       "h3_edit": {"action": action, "video": str(src), "mask_dir": str(TMP / "masks"), "short_edge": 192,
                                   "clip_s": 6.0, "overlap": 39, "keep_audio": True, "picture": r"C:/TEMP/claude/pic_smoke.png"}}, FakeW, clip_runner=lambda W, jd: ER.generate_clip(jd, FakeW))
    assert os.path.isfile(out) and os.path.getsize(out) > 1000, out
    with av.open(out) as c:
        n_out = sum(1 for _ in c.decode(c.streams.video[0]))
    print(action, "clip:", calls, "frame in uscita:", n_out)
    assert len(calls) >= 3 and all(f == 17 * ((f - 5) // 17) + 5 for f, _ in calls)
    assert n_out >= 24 * 14 - 22
print("SMOKE OK")
