"""Worker LongCat-Video-Avatar 1.5 — foto (o solo testo) + voce → video di una persona che parla.

Gira nel venv dedicato `E:\\AI\\Tools\\LongCatAvatar\\venv` (scripts/install_longcat_avatar.py) col codice
UFFICIALE meituan-longcat/LongCat-Video. Rispetto agli script demo ufficiali (Linux, torchrun, multi-GPU):
  · single-GPU, SENZA torch.distributed (NCCL non esiste su Windows): con cp_size=1 il codice non lo usa;
  · memoria a FASI per stare in 32 GB: text encoder → codifica prompt → liberato; Whisper → embedding
    audio → liberato; SOLO dopo si carica il DiT (INT8/FP8 ~16 GB) + LoRA DMD;
  · DiT costruito su meta-device e pesi caricati shard per shard SENZA mmap (evita i 54 GB di RAM
    del costruttore fp32 ufficiale e il crash da commit esaurito, vedi pagefile-commit-mmap-crash);
  · durata decisa dall'audio: segmenti calcolati da core.avatar.timing, video tagliato sulla voce.

Uso:  python longcat_avatar_infer.py params.json   → stampa "OK:<mp4>"
Parametri (JSON): vedi DEFAULTS qui sotto; `audio_path` obbligatorio, `image_path` per ai2v.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))                                    # core.* (stdlib-only)
LONGCAT_CODE = Path(os.environ.get("LONGCAT_CODE_DIR", r"E:\AI\Tools\LongCatAvatar\LongCat-Video"))
if LONGCAT_CODE.is_dir():
    sys.path.insert(0, str(LONGCAT_CODE))

if sys.platform == "win32":                    # expandable_segments non supportato su Windows (vedi trellis)
    _pac = ",".join(x for x in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").split(",")
                    if x and "expandable_segments" not in x)
    if _pac:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _pac
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

from core.avatar import timing  # noqa: E402

NEGATIVE_DEFAULT = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused "
    "fingers, still picture, messy background, three legs, many people in the background, walking backwards")

DEFAULTS = {
    "mode": "ai2v",               # ai2v (foto+audio) | at2v (solo testo+audio)
    "image_path": "",
    "prompt": "",
    "negative": NEGATIVE_DEFAULT,
    "resolution": "480p",         # 480p | 720p
    "aspect": "16:9",             # solo at2v: 16:9 9:16 1:1 4:3 3:4
    "resize_mode": "crop",        # crop | default (ridimensiona senza ritaglio)
    "audio_path": "",             # persona 1 (obbligatorio)
    "audio2_path": "",            # persona 2 (modalità due persone)
    "audio_type": "add",          # add = a turno (concatenati) | para = insieme (stessa durata)
    "bbox": None,                 # {"person1":[y0,x0,y1,x1], "person2":[...]} in FRAZIONI 0–1 dell'immagine
    "trim_start": 0.0, "trim_end": 0.0, "max_duration": 0.0,
    "precision": "int8",          # int8 | fp8 | bf16  → voce di catalogo lcav15-<precision>
    "seed": 42,
    "ref_img_index": 10, "mask_frame_range": 3,
    "vocal_separation": True,
    "offload_kv_cache": "auto",   # auto | True | False — auto = decide il piano VRAM
    "merge_lora": "auto",         # auto | True | False — fonde la LoRA DMD nei pesi quantizzati (−2.5 GB)
    "output": "",
}

_PRECISION_IDS = {"int8": "lcav15-int8", "fp8": "lcav15-fp8", "bf16": "lcav15-bf16"}
_ASPECTS = {"16:9": 9 / 16, "9:16": 16 / 9, "1:1": 1.0, "4:3": 3 / 4, "3:4": 4 / 3}


def _log(msg: str) -> None:
    print(f"[lcav] {msg}", flush=True)


# ── pesi dal registro unico (mai path hardcoded, mai sostituzioni silenziose) ─────────────────
def resolve_weights(precision: str) -> dict:
    from core import model_paths as mp
    from core import weights_catalog as wc
    pid = _PRECISION_IDS.get(precision)
    if not pid:
        raise RuntimeError(f"Precisione LongCat sconosciuta: {precision!r} (int8 | fp8 | bf16)")
    need = [("config + tokenizer", "lcav-configs"), ("text encoder UMT5", "lcav-te"),
            ("VAE", "lcav-vae"), ("LoRA DMD + Whisper + separatore", "lcav15-core"),
            (f"DiT {precision}", pid)]
    missing, out = [], {}
    for label, i in need:
        e = mp.entry(i)
        if not e or not wc.is_present(e, str(mp.models_root())):
            missing.append(f"{label} ({i})")
        else:
            out[i] = mp.p(i, "")
    if missing:
        raise RuntimeError("Pesi LongCat Avatar mancanti: " + "; ".join(missing) +
                           ". Scaricali dal tab «Pesi modelli» (la voce DiT tira anche le dipendenze).")
    return {"configs": out["lcav-configs"], "te": out["lcav-te"], "vae": out["lcav-vae"],
            "core": out["lcav15-core"], "dit": out[pid], "precision": precision}


def _ensure_json(dst_dir: Path, src_dir: Path, names: list[str]) -> None:
    """Copia i json di config mancanti accanto ai pesi (le cartelle locali storiche hanno i soli shard)."""
    import shutil
    dst_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        s, d = src_dir / n, dst_dir / n
        if s.is_file() and not d.is_file():
            shutil.copyfile(s, d)


def _free_gpu_servers() -> None:
    """Libera la VRAM dei server immagini/video dell'app prima di caricare (stdlib, come TRELLIS)."""
    import urllib.request as u
    for port in (8189, 8190, 8191, 8192):
        try:
            u.urlopen(u.Request(f"http://127.0.0.1:{port}/free", data=b'{"mode": "destroy"}',
                                headers={"Content-Type": "application/json"}), timeout=15).read()
            _log(f"VRAM server :{port} liberata")
        except Exception:                                            # noqa: BLE001 (server giù)
            pass


def _gc():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── audio ───────────────────────────────────────────────────────────────────────────────────────
def _load_trimmed(path: str, sr, start: float, dur: float):
    import librosa
    y, r = librosa.load(path, sr=sr, offset=max(0.0, start), duration=(dur if dur > 0 else None))
    return y, r


def _separate(path: str, core_dir: str, tmp: Path) -> str:
    """Isola la voce (Kim_Vocal_2, onnx CPU). In caso di errore torna all'audio originale, dicendolo."""
    try:
        from audio_separator.separator import Separator
        sep_dir = tmp / "vocals"
        sep = Separator(output_dir=str(sep_dir), output_single_stem="vocals",
                        model_file_dir=str(Path(core_dir) / "vocal_separator"))
        sep.load_model("Kim_Vocal_2.onnx")
        outs = sep.separate(path)
        if outs:
            p = Path(outs[0])
            p = p if p.is_absolute() else sep_dir / p.name
            if p.is_file():
                return str(p)
        _log("separazione voce: nessun output → uso l'audio originale")
    except Exception as e:                                           # noqa: BLE001
        _log(f"separazione voce fallita ({e}) → uso l'audio originale")
    return path


def prepare_audio(p: dict, core_dir: str, tmp: Path) -> dict:
    """Taglia, separa la voce e calcola durata/segmenti. Ritorna gli array 16 kHz per gli embedding e
    il wav da muxare nel video finale (qualità originale, voce + eventuale musica)."""
    import numpy as np
    import soundfile as sf
    import librosa
    a1 = p["audio_path"]
    if not a1 or not os.path.isfile(a1):
        raise RuntimeError(f"Audio inesistente: {a1}")
    total = float(librosa.get_duration(path=a1))
    dur = timing.effective_duration(total, trim_start=p["trim_start"], trim_end=p["trim_end"],
                                    max_s=p["max_duration"])
    start = min(max(0.0, float(p["trim_start"] or 0.0)), total)
    raw1, rsr = _load_trimmed(a1, None, start, dur)
    raw1_path = tmp / "raw1.wav"
    sf.write(str(raw1_path), raw1, rsr)
    two = bool(p.get("audio2_path"))
    raw2 = None
    if two:
        a2 = p["audio2_path"]
        if not os.path.isfile(a2):
            raise RuntimeError(f"Audio persona 2 inesistente: {a2}")
        raw2, _ = librosa.load(a2, sr=rsr)
        raw2_path = tmp / "raw2.wav"
        sf.write(str(raw2_path), raw2, rsr)

    voc1 = _separate(str(raw1_path), core_dir, tmp) if p["vocal_separation"] else str(raw1_path)
    s1, _ = librosa.load(voc1, sr=16000)
    s2 = None
    if two:
        voc2 = _separate(str(raw2_path), core_dir, tmp) if p["vocal_separation"] else str(raw2_path)
        s2, _ = librosa.load(voc2, sr=16000)
        if p["audio_type"] == "add":                         # a turno: prima la persona 1, poi la 2
            s1, s2 = (np.concatenate([s1, np.zeros_like(s2)]), np.concatenate([np.zeros_like(s1), s2]))
            mix = np.concatenate([raw1, np.zeros_like(raw2)]) + np.concatenate([np.zeros_like(raw1), raw2])
        else:                                                # insieme: stessa lunghezza (pad al più lungo)
            n = max(len(s1), len(s2)); s1 = np.pad(s1, (0, n - len(s1))); s2 = np.pad(s2, (0, n - len(s2)))
            m = max(len(raw1), len(raw2))
            mix = np.pad(raw1, (0, m - len(raw1))) + np.pad(raw2, (0, m - len(raw2)))
        mix_path = tmp / "mix.wav"
        sf.write(str(mix_path), np.clip(mix, -1.0, 1.0), rsr)
        mux_path = str(mix_path)
    else:
        mux_path = str(raw1_path)
    speech_s = len(s1) / 16000.0
    nseg = timing.segments_for(speech_s)
    gen_s = timing.covered_seconds(nseg)
    pad = math.ceil((gen_s - speech_s) * 16000)
    if pad > 0:
        s1 = np.append(s1, np.zeros(pad, dtype=s1.dtype))
        if s2 is not None:
            s2 = np.append(s2, np.zeros(pad, dtype=s2.dtype))
    _log(f"audio {speech_s:.2f}s → {nseg} segmenti ({gen_s:.2f}s generati, poi tagliati sulla voce)")
    return {"s1": s1, "s2": s2, "mux": mux_path, "speech_s": speech_s, "num_segments": nseg}


# ── modelli ─────────────────────────────────────────────────────────────────────────────────────
def _read_shard(path: Path, device: str):
    """Shard safetensors letto UN TENSORE ALLA VOLTA (niente mmap, niente file intero in RAM) e spostato
    subito su `device`: col pagefile quasi pieno il picco di memoria di sistema resta un tensore."""
    sys.path.insert(0, str(REPO_ROOT / "workers"))
    import _qlinear as Q
    return Q.read_safetensors(str(path), device)


def _fp8_linear_cls():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    state = {"scaled_mm": None}                     # None = da provare, True/False = deciso al 1° forward

    # Il nome DEVE essere "QuantizedLinear": longcat_video.modules.lora_utils accetta come bersaglio LoRA
    # solo classi chiamate "Linear" o "QuantizedLinear" (controllo su __class__.__name__).
    class QuantizedLinear(nn.Module):
        """Linear FP8 e4m3 per canale (formato wavespeed: weight F8_E4M3 + scale_weight F32 per riga).
        Matmul FP8 nativo con torch._scaled_mm (rowwise) se la GPU lo supporta, altrimenti dequantizza
        il peso al volo in bf16 come fa l'INT8 ufficiale."""

        def __init__(self, in_features, out_features, bias):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.register_buffer("weight", torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn))
            self.register_buffer("scale_weight", torch.zeros(out_features, dtype=torch.float32))
            if bias:
                self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
            else:
                self.bias = None

        def _dequant(self, x):
            w = self.weight.to(x.dtype) * self.scale_weight.to(x.dtype).unsqueeze(1)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, w, b)

        def forward(self, x):
            if state["scaled_mm"] is False:
                return self._dequant(x)
            try:
                shp = x.shape
                x2 = x.reshape(-1, shp[-1])
                xs = (x2.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / 448.0)
                xq = (x2.float() / xs).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                out = torch._scaled_mm(xq, self.weight.t(), scale_a=xs,
                                       scale_b=self.scale_weight.view(1, -1),
                                       bias=None, out_dtype=torch.bfloat16)
                if self.bias is not None:
                    out = out + self.bias
                if state["scaled_mm"] is None:
                    state["scaled_mm"] = True
                    _log("FP8: matmul nativo torch._scaled_mm attivo")
                return out.reshape(*shp[:-1], self.out_features).to(x.dtype)
            except Exception as e:                                   # noqa: BLE001
                state["scaled_mm"] = False
                _log(f"FP8: _scaled_mm non disponibile su questa GPU ({type(e).__name__}) → dequant bf16")
                return self._dequant(x)

    return QuantizedLinear


def load_dit(dit_dir: str, precision: str):
    """DiT avatar su meta-device + pesi assegnati shard per shard direttamente in GPU."""
    import torch
    import torch.nn as nn
    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
    root = Path(dit_dir)
    if precision == "bf16":
        return LongCatVideoAvatarTransformer3DModel.from_pretrained(
            str(root), subfolder="base_model", cp_split_hw=[1, 1], torch_dtype=torch.bfloat16).to("cuda")
    qdir = root / "base_model_int8" if precision == "int8" else root
    cfg = json.loads((qdir / "config.json").read_text("utf-8"))
    for k in ("_class_name", "architectures", "_diffusers_version", "model_max_length"):
        cfg.pop(k, None)
    cfg["cp_split_hw"] = [1, 1]          # single-GPU: nessuna divisione hw (context_parallel.get_optimal_split(1))
    if precision == "int8":
        from longcat_video.modules.quantization import QuantizedLinear, DEFAULT_SKIP_PATTERNS
        qcls = QuantizedLinear
    else:
        from longcat_video.modules.quantization import DEFAULT_SKIP_PATTERNS
        qcls = _fp8_linear_cls()
    with torch.device("meta"):
        model = LongCatVideoAvatarTransformer3DModel(**cfg)
    repl = {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not any(pt in n for pt in DEFAULT_SKIP_PATTERNS)}
    for name, lin in repl.items():
        parent = model
        *path, leaf = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        with torch.device("meta"):
            setattr(parent, leaf, qcls(lin.in_features, lin.out_features, bias=lin.bias is not None))
    idx = qdir / "quantized_model.safetensors.index.json"
    shards = sorted(set(json.loads(idx.read_text("utf-8"))["weight_map"].values())) if idx.is_file() \
        else sorted(p.name for p in qdir.glob("*.safetensors"))
    sd = {}
    for i, sh in enumerate(shards, 1):
        _log(f"DiT {precision}: shard {i}/{len(shards)}")
        sd.update(_read_shard(qdir / sh, "cuda"))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    del sd
    if missing or unexpected:
        raise RuntimeError(f"DiT {precision}: chiavi mancanti {missing[:5]} / inattese {unexpected[:5]}")
    for mod in model.modules():                         # come l'ufficiale: parametri fp32 non quantizzati → bf16
        if mod.__class__.__name__ == "QuantizedLinear":
            continue
        for pn, prm in mod.named_parameters(recurse=False):
            if prm.dtype == torch.float32:
                prm.data = prm.data.to(torch.bfloat16)
    model.eval()
    return model


def load_umt5(te_dir: Path, dev: str):
    """UMT5-XXL (5 shard fp32, 22.7 GB) costruito vuoto e riempito TENSORE PER TENSORE direttamente in GPU,
    convertito in bf16 sul device: from_pretrained passava per mmap/RAM (commit alto, crash col pagefile pieno)."""
    import torch
    from accelerate import init_empty_weights
    from transformers import UMT5Config, UMT5EncoderModel
    sys.path.insert(0, str(REPO_ROOT / "workers"))
    import _qlinear as Q
    with init_empty_weights():
        te = UMT5EncoderModel(UMT5Config.from_pretrained(str(te_dir)))
    sd = {}
    for shard in sorted(te_dir.glob("model-*.safetensors")):
        sd.update(Q.read_safetensors(str(shard), dev, cast=torch.bfloat16))
    missing, unexpected = te.load_state_dict(sd, strict=False, assign=True)
    del sd
    missing = [m for m in missing if m not in ("encoder.embed_tokens.weight",)]   # legato a shared.weight
    if missing or unexpected:
        raise RuntimeError(f"text encoder UMT5: chiavi mancanti {missing[:5]} / inattese {unexpected[:5]}")
    te.tie_weights()
    return te.to(dev).eval()


# ── VRAM: piano dinamico (regola: mai valori fissi, si tara sulla GPU del PC) ─────────────────────
_DIT_GB = {"int8": 15.9, "fp8": 15.9, "bf16": 31.7}
_LORA_GB, _VAE_GB, _MARGIN_GB = 2.6, 0.6, 1.5


def _vram_gb() -> tuple[float, float]:
    import torch
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def _auto(v) -> str:
    return "auto" if str(v).lower() == "auto" else ("on" if v in (True, "true", "on", 1) else "off")


def plan_vram(precision: str, h: int, w: int, nseg: int, free_gb: float, offload, merge) -> dict:
    """Stima la memoria del DiT (pesi + LoRA + attivazioni + KV-cache delle continuazioni) e sceglie,
    se serve, offload della KV-cache in RAM e fusione della LoRA. Solleva se non basta comunque:
    meglio un errore chiaro che il driver che riversa la VRAM nella RAM (lentissimo, commit esaurito)."""
    tok = (h // 16) * (w // 16)
    act = tok * 24 * 4096 * 2 * 12 / 2**30               # attivazioni transitorie di un segmento (93 fotogrammi)
    kv = tok * 5 * 48 * 2 * 4096 * 2 / 2**30 if nseg > 1 else 0.0   # 4 latenti di condizione + 1 di riferimento
    o, m = _auto(offload), _auto(merge)
    use_off, use_merge = o == "on", m == "on"

    def need():
        return (_DIT_GB[precision] + (0 if use_merge else _LORA_GB) + _VAE_GB + act
                + (0 if use_off else kv) + _MARGIN_GB)
    if need() > free_gb and o == "auto" and kv:
        use_off = True
    if need() > free_gb and m == "auto":
        use_merge = True
    n = need()
    if n > free_gb:
        raise RuntimeError(f"VRAM insufficiente: servono ~{n:.1f} GB (DiT {precision}, {w}×{h}, {nseg} segmenti), "
                           f"liberi {free_gb:.1f} GB. Chiudi gli altri programmi che usano la GPU, scegli 480p "
                           f"o una precisione più piccola.")
    return {"offload_kv": use_off, "merge_lora": use_merge, "need_gb": round(n, 1), "act_gb": round(act, 1),
            "kv_gb": round(kv, 1)}


def merge_lora_into(dit, lora_path: str, multiplier: float = 1.0) -> int:
    """Fonde la LoRA DMD nei Linear del DiT (INT8 ufficiale, FP8 per riga o bf16), riquantizzando per canale.
    Stesso effetto dell'applicazione «al volo» dell'ufficiale, senza i 2.5 GB di LoRA in VRAM."""
    import torch
    from safetensors.torch import load_file
    sd = load_file(lora_path, device="cuda")
    names = {k[: -len(".lora_down.weight")] for k in sd if k.endswith(".lora_down.weight")}
    mods = dict(dit.named_modules())
    n = 0
    for ln in names:
        m = mods.get(ln.replace("lora___lorahyphen___", "").replace("___lorahyphen___", "."))
        if m is None:
            continue
        down = sd[f"{ln}.lora_down.weight"].float()
        a = sd.get(f"{ln}.alpha_scale")
        scale = (float(a) if a is not None else 0.5) * multiplier
        ups = sorted((k for k in sd if k.startswith(f"{ln}.lora_up.blocks.")),
                     key=lambda k: int(k.split(".lora_up.blocks.")[1].split(".")[0]))
        if ups:
            r = down.shape[0] // len(ups)
            delta = torch.cat([sd[u].float() @ down[i * r:(i + 1) * r] for i, u in enumerate(ups)], 0)
        else:
            delta = sd[f"{ln}.lora_up.weight"].float() @ down
        delta = delta * scale
        with torch.no_grad():
            if hasattr(m, "weight_int8"):                    # INT8 ufficiale (per canale, simmetrico)
                wt = m.weight_int8.float() * m.weight_scale.float().unsqueeze(1) + delta
                sc = wt.abs().amax(dim=1).clamp(min=1e-8) / 127.0
                m.weight_int8.copy_((wt / sc.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8))
                m.weight_scale.copy_(sc.to(m.weight_scale.dtype))
            elif hasattr(m, "scale_weight"):                 # FP8 e4m3 per riga (wavespeed)
                wt = m.weight.float() * m.scale_weight.float().unsqueeze(1) + delta
                sc = wt.abs().amax(dim=1).clamp(min=1e-12) / 448.0
                m.weight.copy_((wt / sc.unsqueeze(1)).clamp(-448, 448).to(torch.float8_e4m3fn))
                m.scale_weight.copy_(sc.to(m.scale_weight.dtype))
            elif isinstance(getattr(m, "weight", None), torch.Tensor):
                m.weight.add_(delta.to(m.weight.dtype))
            else:
                continue
        n += 1
    del sd
    return n


# ── video ───────────────────────────────────────────────────────────────────────────────────────
def save_mp4(frames, audio_path: str, out: str, fps: int = timing.FPS) -> str:
    """Scrive i fotogrammi (H.264) e mux dell'audio; -shortest taglia il video sulla voce."""
    import subprocess
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    tmp_v = out + ".noaudio.mp4"
    w = imageio.get_writer(tmp_v, fps=fps, codec="libx264", quality=8, macro_block_size=16,
                           pixelformat="yuv420p")
    for fr in frames:
        w.append_data(np.asarray(fr))
    w.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-i", tmp_v, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mux audio fallito: {r.stderr[-800:]}")
    try:
        os.remove(tmp_v)
    except OSError:
        pass
    return out


def _bbox_masks(img_w: int, img_h: int, bbox: dict | None):
    """Maschere persona1/persona2/sfondo come lo script multi ufficiale. bbox in frazioni 0–1."""
    import torch
    m1, m2 = torch.zeros([img_h, img_w]), torch.zeros([img_h, img_w])
    if bbox and bbox.get("person1") and bbox.get("person2"):
        def px(b):
            y0, x0, y1, x1 = [float(v) for v in b]
            return (int(y0 * img_h), int(x0 * img_w), int(y1 * img_h), int(x1 * img_w))
        l, r = px(bbox["person1"]), px(bbox["person2"])
    else:                                            # default ufficiale: metà sinistra / destra, margine 10%
        fs, half = 0.1, img_w // 2
        y0, y1 = int(img_h * fs), int(img_h * (1 - fs))
        l = (y0, int(half * fs), y1, int(half * (1 - fs)))
        r = (y0, int(half * fs + half), y1, int(half * (1 - fs) + half))
    m1[l[0]:l[2], l[1]:l[3]] = 1
    m2[r[0]:r[2], r[1]:r[3]] = 1
    bg = torch.where((m1 + m2) > 0, torch.tensor(0.0), torch.tensor(1.0))
    return torch.stack([m1, m2, bg], dim=0)


def run(params: dict) -> str:
    import numpy as np
    import PIL.Image
    import torch
    p = {**DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
    if p["mode"] not in ("ai2v", "at2v"):
        raise RuntimeError(f"Modalità sconosciuta: {p['mode']}")
    if p["mode"] == "ai2v" and not (p["image_path"] and os.path.isfile(p["image_path"])):
        raise RuntimeError(f"Immagine iniziale inesistente: {p['image_path']}")
    if p.get("audio2_path") and p["mode"] != "ai2v":
        raise RuntimeError("La modalità due persone richiede un'immagine iniziale (ai2v).")
    out = p["output"] or str(REPO_ROOT / "output" / f"avatar_longcat_{int(time.time())}.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    W = resolve_weights(p["precision"])
    cfgdir = Path(W["configs"])
    te_dir = Path(W["te"]) / "text_encoder"
    _ensure_json(te_dir, cfgdir / "text_encoder", ["config.json", "model.safetensors.index.json"])
    vae_dir = Path(W["vae"]).parent
    _ensure_json(vae_dir, cfgdir / "vae", ["config.json"])

    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from _watchdog import start_watchdog
        start_watchdog(int(p.get("parent_pid") or 0))
    except Exception:                                                # noqa: BLE001
        pass
    _free_gpu_servers()
    free0, total = _vram_gb()
    # TETTO: oltre questa quota PyTorch dà OOM invece di lasciare che il driver WDDM riversi la VRAM nella
    # RAM di sistema (visto il 26/09: 45 GB di commit e generazione ferma). Altri programmi GPU restano fuori.
    torch.cuda.set_per_process_memory_fraction(max(0.2, min(1.0, (free0 - 0.4) / total)))
    _log(f"VRAM libera all'avvio {free0:.1f} / {total:.1f} GB")

    from transformers import AutoTokenizer, UMT5EncoderModel, WhisperModel, AutoFeatureExtractor
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.utils.bukcet_config import get_bucket_config

    dev = "cuda"
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="lcav_"))
    audio = prepare_audio(p, W["core"], tmp)
    nseg = audio["num_segments"]

    _log("FASE 1/4 text encoder → codifica prompt")
    tokenizer = AutoTokenizer.from_pretrained(str(cfgdir / "tokenizer"))
    te = load_umt5(te_dir, dev)
    vae = AutoencoderKLWan.from_pretrained(str(vae_dir), torch_dtype=torch.bfloat16).to(dev)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(W["core"], subfolder="scheduler")
    pipe = LongCatVideoAvatarPipeline(tokenizer=tokenizer, text_encoder=te, vae=vae, scheduler=scheduler,
                                      dit=None, audio_encoder=None, audio_feature_extractor=None,
                                      model_type="avatar-v1.5")
    pipe.device = dev
    # VAE a tasselli (256px, passo 192) + cache CUDA liberata prima di ogni decode: la decodifica di 93
    # fotogrammi in un colpo chiedeva 7.5 GB contigui (OOM visto il 26/09 con 3.9 GB frammentati)
    vae.enable_tiling()
    for _name in ("decode", "encode"):
        _orig = getattr(vae, _name)

        def _wrapped(*a, __f=_orig, **k):
            _gc()
            return __f(*a, **k)
        setattr(vae, _name, _wrapped)
    cache = {}
    for cfg_flag in (False, True):
        with torch.no_grad():
            cache[cfg_flag] = pipe.encode_prompt(prompt=p["prompt"], negative_prompt=p["negative"],
                                                 do_classifier_free_guidance=cfg_flag,
                                                 dtype=torch.bfloat16, device=dev)

    class _TEStub:                                    # il pipeline legge solo config.d_model / dtype
        def __init__(self, m):
            self.config, self.dtype = m.config, torch.bfloat16

        def to(self, *a, **k):
            return self
    pipe.text_encoder = _TEStub(te)
    pipe.encode_prompt = lambda *a, do_classifier_free_guidance=True, **k: cache[bool(do_classifier_free_guidance)]
    del te
    _gc()

    _log("FASE 2/4 Whisper → embedding audio")
    wdir = str(Path(W["core"]) / "whisper-large-v3")
    pipe.audio_encoder = WhisperModel.from_pretrained(wdir, torch_dtype=torch.float16).eval().to(dev)
    pipe.audio_encoder.requires_grad_(False)
    pipe.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(wdir)
    with torch.no_grad():
        emb1 = pipe.get_audio_embedding(audio["s1"], fps=timing.FPS, device=dev, sample_rate=16000,
                                        model_type="avatar-v1.5")
        emb2 = None
        if audio["s2"] is not None:
            emb2 = pipe.get_audio_embedding(audio["s2"], fps=timing.FPS, device=dev, sample_rate=16000,
                                            model_type="avatar-v1.5")
    if torch.isnan(emb1).any() or (emb2 is not None and torch.isnan(emb2).any()):
        raise RuntimeError("embedding audio con NaN: file audio corrotto o silenzio totale")
    pipe.audio_encoder = None
    _gc()

    image = PIL.Image.open(p["image_path"]).convert("RGB") if p["mode"] == "ai2v" else None
    bucket = get_bucket_config(p["resolution"], scale_factor_spatial=pipe.vae_scale_factor_spatial * 2)
    ratio = (image.height / image.width) if image is not None else _ASPECTS.get(p["aspect"], 9 / 16)
    gh, gw = bucket[sorted(bucket, key=lambda k: abs(float(k) - ratio))[0]][0]
    free_gb, _ = _vram_gb()
    plan = plan_vram(p["precision"], gh, gw, nseg, free_gb, p.get("offload_kv_cache"), p.get("merge_lora"))
    _log(f"piano VRAM: {gw}×{gh}, liberi {free_gb:.1f} GB, servono ~{plan['need_gb']} GB "
         f"(attivazioni {plan['act_gb']}, KV {plan['kv_gb']}) → offload KV {'sì' if plan['offload_kv'] else 'no'}, "
         f"LoRA fusa {'sì' if plan['merge_lora'] else 'no'}")

    _log(f"FASE 3/4 DiT {p['precision']} + LoRA DMD (8 passi)")
    dit = load_dit(W["dit"], p["precision"])
    lora = Path(W["core"]) / "lora" / "dmd_lora.safetensors"
    if plan["merge_lora"]:
        _log(f"LoRA DMD fusa nei pesi: {merge_lora_into(dit, str(lora))} moduli")
    else:
        dit.load_lora(str(lora), "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
        dit.enable_loras(["dmd"])
        for net in getattr(dit, "lora_dict", {}).values():
            for l_ in net.loras:
                l_.to(dev)
    pipe.dit = dit
    _gc()
    _log(f"modelli pronti in {time.time() - t0:.0f}s — VRAM usata {torch.cuda.memory_allocated() / 2**30:.1f} GB")

    # v1.5 = modalità distillata DMD obbligatoria (README ufficiale): 8 passi, guidance 1.0
    steps, tcfg, acfg = 8, 1.0, 1.0
    gen = torch.Generator(device=dev).manual_seed(int(p["seed"]))
    offs = torch.arange(5) - 2

    def emb_window(start_idx: int):
        ci = torch.arange(start_idx, start_idx + timing.NUM_FRAMES).unsqueeze(1) + offs.unsqueeze(0)
        ci = torch.clamp(ci, min=0, max=emb1.shape[0] - 1)
        e = [emb1[ci][None, ...].to(dev)]
        if emb2 is not None:
            e.append(emb2[ci][None, ...].to(dev))
            return torch.cat(e)
        return e[0]

    masks = None
    if image is not None and emb2 is not None:
        masks = _bbox_masks(image.width, image.height, p.get("bbox")).to(dev)

    _log(f"FASE 4/4 generazione — segmento 1/{nseg}")
    ts = time.time()
    with torch.no_grad():
        if p["mode"] == "ai2v":
            kw = {"ref_target_masks": masks} if masks is not None else {}
            video_np, latent = pipe.generate_ai2v(
                image=image, prompt=p["prompt"], negative_prompt=p["negative"], resolution=p["resolution"],
                num_frames=timing.NUM_FRAMES, num_inference_steps=steps, text_guidance_scale=tcfg,
                audio_guidance_scale=acfg, output_type="both", generator=gen, audio_emb=emb_window(0),
                use_distill=True, resize_mode=p["resize_mode"], **kw)
        else:
            video_np, latent = pipe.generate_at2v(
                prompt=p["prompt"], negative_prompt=p["negative"], height=gh, width=gw,
                num_frames=timing.NUM_FRAMES, num_inference_steps=steps, text_guidance_scale=tcfg,
                audio_guidance_scale=acfg, output_type="both", generator=gen, audio_emb=emb_window(0),
                use_distill=True)
    frames = [PIL.Image.fromarray((f * 255).astype(np.uint8)) for f in video_np[0]]
    del video_np
    _gc()
    _log(f"segmento 1/{nseg} in {time.time() - ts:.0f}s — picco VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    width, height = frames[0].size
    current, ref_latent, allf = frames, latent[:, :, :1].clone(), list(frames)
    start = 0
    for seg in range(1, nseg):
        _log(f"segmento {seg + 1}/{nseg}")
        ts = time.time()
        start += timing.NUM_FRAMES - timing.NUM_COND_FRAMES
        kw = {"ref_target_masks": masks} if masks is not None else {}
        with torch.no_grad():
            video_np, latent = pipe.generate_avc(
                video=current, video_latent=latent, prompt=p["prompt"], negative_prompt=p["negative"],
                height=height, width=width, num_frames=timing.NUM_FRAMES,
                num_cond_frames=timing.NUM_COND_FRAMES, num_inference_steps=steps,
                text_guidance_scale=tcfg, audio_guidance_scale=acfg, generator=gen, output_type="both",
                use_kv_cache=True, offload_kv_cache=bool(plan["offload_kv"]), enhance_hf=False,
                audio_emb=emb_window(start), ref_latent=ref_latent, ref_img_index=int(p["ref_img_index"]),
                mask_frame_range=int(p["mask_frame_range"]), use_distill=True, resize_mode=p["resize_mode"],
                **kw)
        new = [PIL.Image.fromarray((f * 255).astype(np.uint8)) for f in video_np[0]]
        del video_np
        allf.extend(new[timing.NUM_COND_FRAMES:])
        current = new
        _gc()
        _log(f"segmento {seg + 1}/{nseg} in {time.time() - ts:.0f}s — picco VRAM "
             f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    save_mp4(allf, audio["mux"], out)
    _log(f"fatto in {time.time() - t0:.0f}s — {len(allf)} fotogrammi, audio {audio['speech_s']:.2f}s")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("uso: longcat_avatar_infer.py params.json"); return 2
    params = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    try:
        out = run(params)
        print(f"OK:{out}", flush=True)
        return 0
    except Exception as e:                                           # noqa: BLE001
        traceback.print_exc()
        print(f"ERRORE: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
