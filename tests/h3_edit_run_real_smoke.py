"""Fumo CPU di workers/h3_edit_run.py con VIDEO E AUDIO VERI (fixture persona_parlante_11s.mp4: 640x608, 25 fps, audio MONO) — il fumo con video
sintetico non aveva preso il crash del 02/11 (audio mono -> encode_video «Expected samples with 2 channels»). Solo la rete (VAE, modello) e finta.
  E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe tests\\h3_edit_run_real_smoke.py [video.mp4]
Verifica: tela derivata dal rapporto del sorgente (non quadrata), 25 -> 24 fps, riferimento video con audio STEREO, catena di clip, file finale."""
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
import minimax_h3_infer as REAL  # noqa: E402  (per _decode_audio_av VERA)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live\fixtures\persona_parlante_11s.mp4"
if not os.path.isfile(VIDEO):
    print("SKIP: fixture assente", VIDEO)
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="h3edit_real_"))
with av.open(VIDEO) as c:
    vs = c.streams.video[0]
    SW, SH = vs.codec_context.width, vs.codec_context.height
    SFPS = float(vs.average_rate)
    NS = vs.frames or int(float(c.duration) / 1e6 * SFPS)
    n_audio = len(c.streams.audio)
    a_ch = c.streams.audio[0].channels if n_audio else 0
print(f"sorgente {SW}x{SH} @ {SFPS} fps, {NS} frame, audio canali={a_ch}")

md = TMP / "masks"
md.mkdir()
for i in range(NS):                                           # al posto di SAM2: rettangolo centrale (la persona parlante sta al centro)
    m = np.zeros((SH, SW), np.uint8)
    m[SH // 8: SH - SH // 10, SW // 4: SW - SW // 4] = 255
    Image.fromarray(m).save(md / f"{i:05d}.png")

calls = []
events = []


class FakeAVAE(torch.nn.Module):
    """Audio VAE finto: ogni canale -> (T, 32) con T = campioni / 800 (40 latenti/s a 32 kHz)."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.config = type("C", (), {"latents_mean": [0.0] * 32, "latents_std": [1.0] * 32})()

    def encode(self, x, return_dict=False):
        assert x.shape[0] == 2, f"l'audio VAE si aspetta 2 canali, ricevuti {x.shape}"       # stereo = 2 elementi del batch
        T = x.shape[-1] // 800
        post = type("P", (), {"mode": staticmethod(lambda: torch.randn(2, 32, T))})()
        return (post,)


class FakeW:
    REPO = ROOT

    @staticmethod
    def _log(m): print("[fake]", m, flush=True)

    @staticmethod
    def resolve(p): return {}

    @staticmethod
    def _gc(): events.append("gc")

    @staticmethod
    def _vram(): return "0 GB (finto)"

    @staticmethod
    def _drop_module_weights(m): events.append("drop:" + type(m).__name__)

    @staticmethod
    def load_vae(W, encoder_only=False):
        events.append("load:vae")
        return torch.nn.Linear(1, 1)

    @staticmethod
    def load_audio_vae(W):
        events.append("load:avae")
        return FakeAVAE()

    _decode_audio_av = staticmethod(REAL._decode_audio_av)          # decodifica VERA dell'audio del fixture

    @staticmethod
    def run(pc):
        e = pc["_h3_edit"]
        th, tw = e["canvas"]
        src, noise, keep = e["video_hold"]((1, 2, 2))
        assert src.shape == noise.shape == keep.shape
        ref = [r for r in pc["references"] if r["kind"] == "video"][0]["path"]
        with av.open(ref) as c:                                    # il riferimento video porta l'audio sorgente, STEREO
            assert len(c.streams.audio) == 1 and c.streams.audio[0].channels == 2, "riferimento senza audio stereo"
            assert abs(c.streams.video[0].average_rate - 24) < 1e-6
        if e.get("audio_hold") is not None:
            a, n, k = e["audio_hold"]()
            assert a.shape == n.shape == k.shape and a.shape[0] % 2 == 0
            assert e["audio_noise"].shape == (1, 32, a.shape[0])
        vref = [r for r in pc["references"] if r["kind"] == "video"][0]
        assert vref["frames"] == pc["num_frames"], "il riferimento video deve dire i suoi frame (piano di memoria)"
        events.append("run")
        calls.append((pc["num_frames"], th, tw))
        return {"videos": [(torch.rand(pc["num_frames"], th, tw, 3) * 255).to(torch.uint8)], "audio": [None], "sampling_rate": 32000}


ER.encode_source_latent = lambda vae, f, c: torch.randn(1, 24, 5 * ((f.shape[0] - 5) // 17) + 2, f.shape[1] // 16, f.shape[2] // 16)
out = ER.run_edit({"output": str(TMP / "out.mp4"), "seed": 3,
                   "h3_edit": {"action": "change_background", "video": VIDEO, "mask_dir": str(md), "short_edge": 480, "keep_audio": True,
                               "clip_s": 6.0, "overlap": 39, "prompt_text": "a rainy neon street"}}, FakeW, clip_runner=lambda W, jd: ER.generate_clip(jd, FakeW))
th, tw = calls[0][1], calls[0][2]
print("clip:", calls, "tela", (tw, th))
assert tw != th and abs(tw / th - SW / SH) < 0.08, "la tela deve seguire il rapporto del sorgente (non quadrata)"
assert min(tw, th) == 480
with av.open(out) as c:
    n_out = sum(1 for _ in c.decode(c.streams.video[0]))
    assert len(c.streams.audio) == 1, "l'uscita deve avere l'audio originale"
dur = NS / SFPS
assert abs(n_out - dur * 24) <= 22, (n_out, dur * 24)            # 25 -> 24 fps: la durata resta quella del sorgente (meno al piu' 22 frame di griglia)
# sequenza carica -> usa -> libera, un giro per clip: VAE video e audio caricati, svuotati PRIMA del clip (W.run carica e libera il resto da se)
per_clip = []
cur = []
for e in events:
    cur.append(e)
    if e == "run":
        per_clip.append(cur)
        cur = []
assert len(per_clip) == len(calls) >= 3
for k, seq in enumerate(per_clip):
    ids = [x for x in seq if x != "gc"]
    assert ids == ["load:vae", "drop:Linear", "load:avae", "drop:FakeAVAE", "run"], (k, seq)
    assert seq.count("gc") >= 2, (k, seq)                         # gc dopo ogni caricamento e fra i clip
print("SMOKE REALE OK:", n_out, "frame a 24 fps per", round(dur, 2), "s; clip", len(calls))


# ── ripresa da checkpoint: stesso input = i clip gia generati NON si rifanno; un clip fallito si riprende dal punto giusto ──
import shutil  # noqa: E402

n_first = len(calls)
runs = []
def counting_runner(W, jd):
    runs.append(os.path.basename(jd)); ER.generate_clip(jd, FakeW)
params = {"output": str(TMP / "out2.mp4"), "seed": 3,
          "h3_edit": {"action": "change_background", "video": VIDEO, "mask_dir": str(md), "short_edge": 480, "keep_audio": True,
                      "clip_s": 6.0, "overlap": 39, "prompt_text": "a rainy neon street"}}
ER.run_edit(params, FakeW, clip_runner=counting_runner)
assert runs == [], f"ripresa: nessun clip andava rifatto, rigenerati {runs}"
tmpdirs = list((TMP).glob("_h3edit_tmp/*/clip_02"))
assert tmpdirs, "cartella dei clip non trovata"
shutil.rmtree(tmpdirs[0])                                    # simulo un guasto sul clip 3: sparisce il suo risultato
ER.run_edit(params, FakeW, clip_runner=counting_runner)
assert runs == ["clip_02"], f"ripresa: doveva rifare SOLO il clip 3, rifatti {runs}"
print("RIPRESA OK: clip rigenerati dopo il guasto:", runs)
