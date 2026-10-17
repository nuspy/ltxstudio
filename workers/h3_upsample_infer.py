"""Worker H3 Video Upsampler — un video finito → il suo ×2 (o la misura scelta) con MiniMax H3, nativo (nessun ComfyUI).

Fasi, UN modello grande alla volta (come minimax_h3_infer.py; in 32 GB sta tutto):
  0. lettura del video (fotogrammi su file temporaneo, colonna sonora a 32 kHz stereo)
  1. text encoder Qwen3-VL → embedding della didascalia della LoRA (in cache su disco: dalla seconda volta il TE non si carica)
  2. VAE video (+ audio VAE): latente della sorgente, latente della GUIDA (la sorgente a 1/f della tela d'uscita), audio
  3. upscaler latente (rete 3D di LBH-123-AI) su ogni segmento → in RAM
  4. DiT Ref2VA + LoRA h3upscale: ri-disegno a tile di ogni segmento (un passo), con perno sul frame 0 e dissolvenza fra segmenti
  5. VAE: decodifica a pezzi da 17 fotogrammi, scritta subito nel file (memoria costante, qualunque durata)
La colonna sonora originale non si tocca: la rimette il servizio (core/h3_upsample_service.py) con ffmpeg.
Uso:  python h3_upsample_infer.py params.json   → stampa "OK:<mp4 senza audio>"
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                    # noqa: E402

from core import h3_upsample_plan as P                                # noqa: E402
from core import h3_upsample_spec as U                                # noqa: E402

AUDIO_SR = 32000


def _log(m: str) -> None:
    print(f"[h3up] {m}", flush=True)


def _gc():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _peak_gb() -> float:
    import torch
    try:
        return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    except Exception:                                                # noqa: BLE001
        return 0.0


def _free_gpu_servers() -> None:
    import urllib.request as u
    for port in (8189, 8190, 8191, 8192):
        try:
            u.urlopen(u.Request(f"http://127.0.0.1:{port}/free", data=b'{"mode": "destroy"}',
                                headers={"Content-Type": "application/json"}), timeout=15).read()
        except Exception:                                            # noqa: BLE001 (server giù)
            pass


# ── pesi ───────────────────────────────────────────────────────────────────────────────────────────────────────
def resolve(p: dict) -> dict:
    from core import model_paths as mp
    from core import native_convert as nc
    from core import weights_catalog as wc
    ids = U.weight_ids(p)
    out, missing = {}, []
    for k, i in ids.items():
        e = mp.entry(i)
        if not e or not wc.is_present(e, str(mp.models_root())):
            missing.append(i)
            continue
        path = mp.p(i, "")
        if e.get("convert") and not nc.is_native(path):
            raise RuntimeError(f"{os.path.basename(path)} non è in formato nativo: riscaricalo dal tab «Pesi modelli»")
        out[k] = path
    if missing:
        raise RuntimeError("Pesi mancanti per H3 Video Upsampler: " + ", ".join(missing) +
                           ". Scaricali dal tab «Pesi modelli».")
    out["ids"] = ids
    return out


def _lora_meta(path: str) -> dict:
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__") or {}


# ── video in ingresso ─────────────────────────────────────────────────────────────────────────────────────────
def read_video(path: str, canvas: tuple[int, int], tmp: Path):
    """-> (memmap uint8 [N, ch, cw, 3] sulla tela di lavoro, fps, soundtrack (2, S) float32 a 32 kHz o None)."""
    import av
    from PIL import Image
    cw, ch = canvas
    with av.open(path) as c:
        vs = c.streams.video[0]
        fps = float(vs.average_rate or vs.guessed_rate or P.H3_FPS)
        n = int(vs.frames or 0)
    if n <= 0:                                                        # container senza conteggio: lo conto
        with av.open(path) as c:
            n = sum(1 for _ in c.decode(video=0))
    mm = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=np.uint8, shape=(n, ch, cw, 3))
    i = 0
    with av.open(path) as c:
        for fr in c.decode(video=0):
            if i >= n:
                break
            if fr.width != cw or fr.height != ch:
                mm[i] = np.asarray(fr.to_image().resize((cw, ch), Image.LANCZOS))
            else:
                mm[i] = fr.to_ndarray(format="rgb24")
            i += 1
    if i != n:
        raise RuntimeError(f"letti {i} fotogrammi su {n} dichiarati")
    mm.flush()
    import minimax_h3_infer as H
    wav = None
    try:
        wav = H._decode_audio_av(path, AUDIO_SR)
    except Exception as e:                                           # noqa: BLE001
        _log(f"colonna sonora illeggibile ({e}): condizionamento su silenzio")
    return mm, fps, wav


def _frame_blocks(mm, n_real: int, length: int, block: int = 17):
    """Blocchi float 0..1 [b, H, W, 3] su tutta la lunghezza H3: dopo l'ultimo fotogramma si ripete l'ultimo (come fa
    il nodo originale e come farebbe il VAE da solo)."""
    import torch
    last = None
    for a in range(0, length, block):
        b = min(length, a + block)
        real = mm[a:min(b, n_real)] if a < n_real else None
        parts = []
        if real is not None and len(real):
            parts.append(torch.from_numpy(np.ascontiguousarray(real)))
            last = parts[-1][-1:]
        need = b - a - (0 if real is None else len(real))
        if need > 0:
            parts.append(last.repeat(need, 1, 1, 1))
        yield torch.cat(parts).float().div_(255.0)


def _guide_blocks(blocks, w_px: int, h_px: int, factor: int, device):
    """La guida: sorgente → tela d'uscita (bicubica, antialias) → 1/f. Lo stiramento (non ritaglio) è voluto: il
    ricampionatore del nodo STIRA la tela all'uscita, un ritaglio sposterebbe la guida dal contenuto che deve coprire."""
    import torch
    import torch.nn.functional as F
    gw, gh = w_px // factor, h_px // factor
    for blk in blocks:
        x = blk.to(device, torch.float32).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(h_px, w_px), mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
        x = F.interpolate(x, size=(gh, gw), mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
        yield x.permute(0, 2, 3, 1).cpu()


# ── audio ──────────────────────────────────────────────────────────────────────────────────────────────────────
def encode_audio(audio_vae, wav, n_frames: int, length: int, device):
    """Colonna sonora → latente audio normalizzato (1, 32, 2, Ta) lungo quanto il clip H3 (taglia o riempie di 0).
    Senza audio: silenzio VERO passato dal VAE (il nodo originale usa zeri nello spazio grezzo)."""
    import torch
    ta = int(round(length / P.H3_FPS * 40))
    if wav is None:
        wav = torch.zeros(2, int(round(length / P.H3_FPS * AUDIO_SR)))
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav[:2]
    post = audio_vae.encode(wav.to(device)[:, None], return_dict=False)[0]
    lat = post.mode().float().cpu().transpose(1, 2)                  # (2, Ta', 32)
    mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)
    lat = ((lat - mean) / std).permute(2, 0, 1)[None]                # (1, 32, 2, Ta')
    have = lat.shape[-1]
    if have >= ta:
        return lat[..., :ta].contiguous()
    pad = torch.zeros(lat.shape[:-1] + (ta - have,), dtype=lat.dtype)
    return torch.cat([lat, pad], dim=-1)


# ── didascalia: embedding in cache ─────────────────────────────────────────────────────────────────────────────
def prompt_embeds(W: dict, device):
    """(embeds (1, N, 5120) bf16 CPU, tag testo (N,)) della didascalia della LoRA. Cache su disco per TE: la
    didascalia è costante, quindi il text encoder da 15-27 GB si carica UNA sola volta."""
    import torch
    te_path = Path(W["te"])
    st = te_path.stat()
    key = hashlib.sha1(f"{te_path.name}|{st.st_size}|{int(st.st_mtime)}|{P.CAPTION}|L50".encode()).hexdigest()[:16]
    cdir = REPO / "output" / ".h3up_cache"
    cfile = cdir / f"caption_{key}.pt"
    if cfile.is_file():
        d = torch.load(cfile, map_location="cpu", weights_only=True)
        _log(f"embedding della didascalia dalla cache ({cfile.name}): text encoder non caricato")
        return d["embeds"], d["tags"]
    import minimax_h3_infer as H
    from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
    from transformers import AutoTokenizer, Qwen3VLProcessor
    cfgdir = Path(W["configs"])
    t0 = time.time()
    te = H.load_text_encoder(W)
    _log(f"text encoder pronto in {time.time() - t0:.0f}s")
    tok = AutoTokenizer.from_pretrained(str(cfgdir / "tokenizer"))
    proc = Qwen3VLProcessor.from_pretrained(str(cfgdir / "processor"))
    ids = tok(P.CAPTION, add_special_tokens=False)["input_ids"]
    emb = get_qwen3vl_prompt_embeds(te, proc, ids, None, text_encoder_layer=50, device=torch.device(device),
                                    dtype=te.dtype).to("cpu", torch.bfloat16)
    tags = torch.full((len(ids),), 1, dtype=torch.long)              # MINIMAX_H3_TEXT_TAG
    del te, proc
    _gc()
    cdir.mkdir(parents=True, exist_ok=True)
    torch.save({"embeds": emb, "tags": tags}, cfile)
    return emb, tags


# ── uscita ─────────────────────────────────────────────────────────────────────────────────────────────────────
class _Writer:
    def __init__(self, path: str, w: int, h: int, fps: float, n_total: int):
        import av
        from fractions import Fraction
        self.av = av
        self.n_total, self.n = int(n_total), 0
        self.out = av.open(path, "w")
        self.st = self.out.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        self.st.width, self.st.height, self.st.pix_fmt = w, h, "yuv420p"
        self.st.options = {"crf": "14", "preset": "medium"}

    def write(self, frames: np.ndarray):
        for a in frames:
            if self.n >= self.n_total:                               # il padding H3 oltre l'ultimo fotogramma vero
                return
            fr = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(a), format="rgb24")
            for pk in self.st.encode(fr):
                self.out.mux(pk)
            self.n += 1

    def close(self):
        for pk in self.st.encode():
            self.out.mux(pk)
        self.out.close()


# ── il lavoro ──────────────────────────────────────────────────────────────────────────────────────────────────
class Loaders:
    """I punti dove si caricano i modelli grandi: sostituibili (i test su CPU ci mettono modelli minuscoli)."""

    @staticmethod
    def embeds(W, dev):
        return prompt_embeds(W, dev)

    @staticmethod
    def vae(W):
        import minimax_h3_infer as H
        return H.load_vae(W)

    @staticmethod
    def audio_vae(W):
        import minimax_h3_infer as H
        return H.load_audio_vae(W)

    @staticmethod
    def upscaler(path):
        import h3_upsample_core as C
        return C.load_upscaler(path)

    @staticmethod
    def transformer(W, p):
        import minimax_h3_infer as H
        return H.load_transformer(W, "ref2va", p)


def run(p: dict) -> str:
    from core import resources
    resources.require(commit_gb=float(p.get("need_commit_gb") or 8), vram_gb=float(p.get("need_vram_gb") or 24),
                      what="H3 Video Upsampler")
    resources.start_commit_guard(2.0, what="H3 Video Upsampler")
    import torch
    torch.set_grad_enabled(False)
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from _watchdog import start_watchdog
        start_watchdog(int(p.get("parent_pid") or 0))
    except Exception:                                                # noqa: BLE001
        pass
    W = resolve(p)
    _free_gpu_servers()
    free0, total = (x / 2**30 for x in torch.cuda.mem_get_info())
    # TETTO: meglio un OOM chiaro che il driver WDDM che riversa la VRAM nella RAM di sistema
    torch.cuda.set_per_process_memory_fraction(max(0.2, min(1.0, (free0 - 0.4) / total)))
    return process(p, W, Loaders, "cuda")


def process(p: dict, W: dict, L, dev: str) -> str:
    """Il lavoro vero, su pesi già risolti (W) e caricatori `L`; `dev` = «cuda» (produzione) o «cpu» (test)."""
    import torch
    t_all = time.time()
    src, out = p["video"], p["output"]
    mp_ = float(p.get("megapixels") or P.MEGAPIXELS)
    denoise = float(p.get("denoise") if p.get("denoise") is not None else P.DENOISE_DEFAULT)
    budget = int(p.get("tile_tokens") or P.TOKEN_BUDGET)
    guided = bool(W.get("guide_lora"))
    factor = 0
    if guided:
        factor = P.guide_factor_from_meta(_lora_meta(W["guide_lora"]))
        if factor < 2:
            raise RuntimeError(f"{os.path.basename(W['guide_lora'])} non è una LoRA di guida h3upscale "
                               "(manca reference_downscale_factor nell'intestazione)")
    sigma = P.redraw_sigma(denoise)

    import av
    with av.open(src) as c:
        v0 = c.streams.video[0]
        sw, sh = v0.codec_context.width, v0.codec_context.height
    cw, ch = P.grid_size(sw, sh)
    w_px, h_px = P.target_size(cw, ch, mp_, align=32 * max(1, factor))
    if w_px < cw or h_px < ch:
        raise RuntimeError(f"{w_px}x{h_px} è più piccolo del video ({cw}x{ch}): questa funzione ingrandisce soltanto. "
                           "Alza i megapixel.")
    tmpdir = Path(p.get("workdir") or (REPO / "output")) / f".h3up_tmp_{os.getpid()}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        mm, fps, wav = read_video(src, (cw, ch), tmpdir / "frames.npy")
        n = int(mm.shape[0])
        length = P.valid_length(n)
        tokens = 5 * ((length - 5) // 17) + 2
        plan = P.chunks(tokens)
        _log(f"{n} fotogrammi {sw}x{sh} @ {fps:g} fps → tela H3 {cw}x{ch}, {length} fotogrammi ({tokens} passi latenti) → "
             f"uscita {w_px}x{h_px} ({w_px * h_px / 1048576:.2f} MP) · {len(plan)} segmenti · "
             + (f"guidato dalla LoRA (fattore {factor}), " if guided else "senza LoRA (nessuna guida), ")
             + (f"ri-disegno denoise {denoise:g} da sigma {sigma:.4f}" if sigma is not None else "SOLO ingrandimento (denoise 0)")
             + f" · budget {budget} token · lettura {time.time() - t0:.0f}s")

        # 1) didascalia (solo se c'è un ri-disegno)
        emb = tags = None
        if sigma is not None:
            emb, tags = L.embeds(W, dev)

        # 2) VAE: sorgente, guida, audio
        import h3_upsample_core as C
        t0 = time.time()
        vae = L.vae(W)
        pm, ps = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        lm, ls = tuple(vae.config.latents_mean), tuple(vae.config.latents_std)
        kw = dict(pixel_mean=pm, pixel_std=ps, latents_mean=lm, latents_std=ls, device=dev)
        video = C.encode_frames(vae, _frame_blocks(mm, n, length), length, **kw)
        if int(video.shape[2]) != tokens:
            raise RuntimeError(f"{length} fotogrammi codificati in {int(video.shape[2])} passi latenti, attesi {tokens}")
        glat = None
        if guided and sigma is not None:
            glat = C.encode_frames(vae, _guide_blocks(_frame_blocks(mm, n, length), w_px, h_px, factor, dev),
                                   length, **kw)
        del vae
        _gc()
        audio = None
        if sigma is not None:
            av_ = L.audio_vae(W)
            audio = encode_audio(av_, wav, n, length, dev)
            del av_
            _gc()
        _log(f"latenti pronti in {time.time() - t0:.0f}s")

        # 3) upscaler latente su ogni segmento
        t0 = time.time()
        net = L.upscaler(W["upscaler"])
        ups = []
        for (k0, f0, k1, f1) in plan:
            ups.append(C.upscale_chunk(net, video[:, :, k0:k1].contiguous(), w_px, h_px, dev))
        del net
        _gc()
        _log(f"upscaler latente su {len(plan)} segmenti in {time.time() - t0:.0f}s")

        # 4) ri-disegno
        done_v = done_a = None
        if sigma is None:                                            # denoise 0: solo l'upscaler latente
            silent = torch.zeros(1, 32, 2, 1)
            for (k0, f0, k1, f1), cv in zip(plan, ups):
                done_v, done_a = P.join_chunk(done_v, done_a, cv, silent, k0, f0)
        else:
            t0 = time.time()
            pp = dict(p)
            pp["loras"] = [{"path": W["guide_lora"], "strength": float(p.get("lora_strength") or 1.0)}] if guided else []
            Wd = {k: v for k, v in W.items() if k not in ("turbo", "veda")}
            tr = L.transformer(Wd, pp)
            _log(f"transformer pronto in {time.time() - t0:.0f}s")
            ta_total = int(audio.shape[-1])
            for i, ((k0, f0, k1, f1), cv) in enumerate(zip(plan, ups)):
                a0, a1 = P.audio_span(f0, f1)
                ca = audio[:, :, :, a0:min(a1, ta_total)].contiguous()
                pin = None
                if i > 0:
                    k = P.tokens_to_reach(f0)
                    if k >= done_v.shape[2]:
                        raise RuntimeError(f"i segmenti finiti si fermano prima del fotogramma {f0}")
                    pin = done_v[:, :, k:k + 1].contiguous()
                g = glat[:, :, k0:k1].contiguous() if glat is not None else None
                if g is not None and tuple(g.shape[2:]) != (int(cv.shape[2]), int(cv.shape[3]) // factor,
                                                            int(cv.shape[4]) // factor):
                    raise RuntimeError(f"latente della guida {tuple(g.shape[2:])} per un segmento che vuole "
                                       f"({int(cv.shape[2])}, {int(cv.shape[3]) // factor}, {int(cv.shape[4]) // factor})")
                t1 = time.time()
                ov = C.redraw_chunk(tr, emb, tags, cv, ca, sigma=sigma, factor=factor, guide=g, pin=pin,
                                    budget=budget, device=dev,
                                    on_tile=lambda a, b, i=i: _log(f"segmento {i + 1}/{len(plan)} · tile {a}/{b}"))
                done_v, done_a = P.join_chunk(done_v, done_a, ov, ca, k0, f0)
                _log(f"segmento {i + 1}/{len(plan)} fatto in {time.time() - t1:.0f}s · picco VRAM "
                     f"{_peak_gb():.1f} GB")
            del tr
            _gc()

        # 5) decodifica a pezzi → file
        t0 = time.time()
        vae = L.vae(W)
        wr = _Writer(out, w_px, h_px, fps, n)
        C.decode_stream(vae, done_v, wr.write, pixel_mean=pm, pixel_std=ps, latents_mean=lm, latents_std=ls,
                        device=dev)
        wr.close()
        if wr.n != n:
            raise RuntimeError(f"scritti {wr.n} fotogrammi su {n}")
        del vae
        _gc()
        _log(f"decodifica e scrittura in {time.time() - t0:.0f}s")
        _log(f"fatto in {time.time() - t_all:.0f}s → {out}")
        return out
    finally:
        try:
            del mm
        except Exception:                                            # noqa: BLE001
            pass
        for f in tmpdir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass


def main() -> int:
    if len(sys.argv) < 2:
        print("uso: h3_upsample_infer.py params.json")
        return 2
    try:
        res = run(json.loads(Path(sys.argv[1]).read_text("utf-8")))
        print(f"OK:{res}", flush=True)
        return 0
    except Exception as e:                                           # noqa: BLE001
        traceback.print_exc()
        print(f"ERRORE: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
