"""Worker MultiTalk (Wan2.1 I2V 14B + adattatore audio) NATIVO — lip-sync su audio REALE, senza ComfyUI.

foto (o primo frame di un video) + audio -> video con bocca/movimento sincronizzati. Motore: `workers/wan_mt/`
(vendorizzato da MeiGen-AI/MultiTalk e Wan2.1, Apache-2.0, vedi wan_mt/NOTICE): DiT fp8 con dequantizzazione bf16
per layer, LoRA al volo (distillazione lightx2v + LoRA Wan dell'utente), block-swap e T5 a blocchi tarati sulla VRAM
della scheda, attenzione Sage/SDPA. Video lunghi: clip autoregressivi da 81 frame con 25 frame di continuita'
(metodo ufficiale), scritti in streaming su ffmpeg + cache per clip per riprendere dopo un crash.

Param JSON (argv[1]): image_path | source_video, audio_path, prompt, negative, width, height, num_frames, fps,
steps, cfg, audio_cfg, audio_scale, shift, seed, scheduler, blocks_to_swap ("auto"|N), lora_strength, lora (path|""),
loras ([{name|path, strength}] extra), separate_vocals, clip_frames, motion_frames, ref_anchor_strength, resume,
attention ("auto"|"sdpa"), output. Stampa `OK:<path>` a fine corsa (codice di uscita 0); errori: riga `ERRORE: …`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODELS = os.environ.get("AI_MODELS_DIR", r"E:\AI\Models")


def _first_frame(video_path):
    """Primo frame di un video come PNG (MultiTalk applicato a un video esistente)."""
    out = os.path.splitext(video_path)[0] + "_frame0.png"
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def _vocals_audio(path):
    """Isola la VOCE dalla traccia (demucs htdemucs via CLI): MultiTalk e' addestrato sul parlato/canto e la musica di
    sottofondo peggiora il lip-sync. Ritorna il path del wav con i soli vocals (cache su disco)."""
    out = os.path.join(os.path.dirname(path) or ".", "_demucs")
    name = os.path.splitext(os.path.basename(path))[0]
    voc = os.path.join(out, "htdemucs", name, "vocals.wav")
    if not os.path.exists(voc):
        os.makedirs(out, exist_ok=True)
        subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", out, path], check=True)
    return voc


def _resolve_lora(name: str) -> str | None:
    """Nome relativo (la GUI manda «Wan\\x.safetensors»), nome semplice o path assoluto -> file su disco."""
    if not name:
        return None
    if os.path.isabs(name) and os.path.exists(name):
        return name
    for sub in ("LoRA", os.path.join("LoRA", "Wan"), os.path.join("LoRA", "Wan2.1"), os.path.join("LoRA", "Wan2.2")):
        c = os.path.join(MODELS, sub, name)
        if os.path.exists(c):
            return c
    return None


# ---- cache per-clip su disco (RESUME dopo crash: rigenera solo i clip mancanti) ----
class ClipCache:
    def __init__(self, cdir: str, key: dict, fps: float):
        import glob
        self.dir, self.fps = cdir, fps
        os.makedirs(cdir, exist_ok=True)
        mpath = os.path.join(cdir, "meta.json")
        old = None
        if os.path.exists(mpath):
            try:
                old = json.load(open(mpath, encoding="utf-8"))
            except Exception:                                   # noqa: BLE001
                old = None
        if old != key:                                          # parametri cambiati: i clip vecchi non sono piu' coerenti
            for f in glob.glob(os.path.join(cdir, "clip_*.mp4")):
                try:
                    os.remove(f)
                except OSError:
                    pass
            json.dump(key, open(mpath, "w", encoding="utf-8"))
        else:
            n = len(glob.glob(os.path.join(cdir, "clip_*.mp4")))
            if n:
                print(f"[resume] trovati {n} clip in cache -> rigenero solo i mancanti", flush=True)

    def _p(self, ci):
        return os.path.join(self.dir, f"clip_{ci:04d}.mp4")

    def save(self, ci, frames):
        """uint8 [n,H,W,3] -> mp4 quasi-lossless, scrittura ATOMICA (.part poi rename)."""
        n, h, w, _ = frames.shape
        part = self._p(ci) + ".part"
        proc = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps),
                                 "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv444p",
                                 "-f", "mp4", part], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()
        if proc.wait() == 0:
            os.replace(part, self._p(ci))
        elif os.path.exists(part):
            os.remove(part)

    def load(self, ci, expect):
        import cv2
        import numpy as np
        if not os.path.exists(self._p(ci)):
            return None
        cap = cv2.VideoCapture(self._p(ci))
        fr = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not fr or abs(len(fr) - expect) > 1:                 # clip troncato da un crash: da rigenerare
            return None
        return np.stack(fr)


def run(p: dict) -> str:
    import numpy as np
    import torch
    from PIL import Image
    from wan_mt import engine as E, audio as wa

    t0 = time.time()
    paths = E.resolve_paths(p.get("paths") or None)                # override opzionali dei percorsi (test/avanzato)
    miss = E.missing_required(paths)
    if miss:
        raise RuntimeError("pesi MultiTalk mancanti: " + ", ".join(miss) + " (scaricali dal tab «Pesi modelli»: id wan21-i2v-14b-480p-fp8)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: MultiTalk gira solo su GPU NVIDIA")

    W0, H0 = int(p.get("width", 640)), int(p.get("height", 640))
    maxpx = int(p.get("max_pixels", 0) or int(480 * 832 * 1.33))       # Wan2.1 480P: addestrato a ~0,4 MP, oltre degrada e costa
    if W0 * H0 > maxpx:
        s = (maxpx / (W0 * H0)) ** 0.5
        print(f"[size] {W0}x{H0} supera il limite del modello 480P ({maxpx / 1e6:.2f} MP): riduco x{s:.2f} mantenendo il rapporto", flush=True)
        W0, H0 = int(W0 * s), int(H0 * s)
    W = max(256, W0 // 16 * 16)
    H = max(256, H0 // 16 * 16)
    if (W, H) != (int(p.get("width", 640)), int(p.get("height", 640))):
        print(f"[size] {p.get('width')}x{p.get('height')} -> {W}x{H} (multipli di 16 richiesti da Wan)", flush=True)
    FPS = float(p.get("fps", 25.0))
    F = int(p.get("num_frames", 81))
    steps = int(p.get("steps", 10))
    cfg = float(p.get("cfg", 1.0))
    acfg = float(p.get("audio_cfg", p.get("audio_cfg_scale", 1.0)))
    seed = int(p.get("seed", 8))
    sched = p.get("scheduler", "dpm++_sde")
    clip_frames = int(p.get("clip_frames", 0) or 81)
    motion_frames = int(p.get("motion_frames", 0) or 25)

    img_path = p.get("image_path")
    if p.get("source_video") and not img_path:
        img_path = _first_frame(p["source_video"])
        print(f"[src] primo frame del video -> {os.path.basename(img_path)}", flush=True)
    image = Image.open(img_path).convert("RGB")

    audio_path = p["audio_path"]
    if p.get("separate_vocals", True):
        try:
            audio_path = _vocals_audio(audio_path)
            print("[audio] vocals isolati (demucs) -> lip-sync su voce pulita", flush=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"[audio] separazione vocals fallita ({e}); uso audio intero", flush=True)

    eng = E.MultiTalkEngine(paths, attn=p.get("attention", "auto"), log=lambda s: print(s, flush=True))
    # 1) testo (T5 solo se non e' in cache; il negativo serve solo con cfg != 1)
    eng.prepare_text(p.get("prompt", "a person speaking, natural expressive performance, close-up"),
                     p.get("negative", "") or "", need_null=not (abs(cfg - 1.0) < 1e-6))
    # 2) audio: wav2vec una volta per tutta la traccia, poi liberato
    wav = wa.load_audio_16k(audio_path)
    need_s = F / FPS
    if len(wav) / wa.SR < need_s - 0.2:
        print(f"[audio] la traccia dura {len(wav) / wa.SR:.1f}s ma il video {need_s:.1f}s: il finale ripete la coda riflessa", flush=True)
    w2v = wa.build_wav2vec(paths["wav2vec"], eng.dev, torch.float16)
    emb = wa.embed_audio(w2v, wav[: int((need_s + 2.0) * wa.SR)], FPS, eng.dev, torch.float16, normalize=True)
    del w2v
    torch.cuda.empty_cache()
    print(f"[audio] embedding {tuple(emb.shape)}", flush=True)
    # 3) CLIP + VAE + DiT
    eng.load_clip()
    eng.load_vae()
    loras = []
    distill, lv = paths.get("lora"), p.get("lora", True)
    if lv is False or lv == "none":
        distill = None                                              # niente distillazione: servono piu' passi e cfg > 1
    elif isinstance(lv, str) and lv:
        distill = _resolve_lora(lv) or distill                      # nome storico non trovato -> lightx2v di default
    if distill and float(p.get("lora_strength", 0.6)) > 0:
        loras.append((distill, float(p.get("lora_strength", 0.6))))
    for lo in (p.get("loras") or []):
        f = _resolve_lora(lo.get("path") or lo.get("name") or "")
        if f:
            loras.append((f, float(lo.get("strength", 1.0))))
        else:
            print(f"[lora] non trovata: {lo.get('name') or lo.get('path')}", flush=True)
    bts = p.get("blocks_to_swap", "auto")
    eng.load_dit(W, H, loras=loras, blocks_to_swap=bts)
    print(f"[load] {time.time() - t0:.0f}s", flush=True)

    # 4) generazione in streaming su ffmpeg
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".noaudio.mp4"
    writer = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", tmp], stdin=subprocess.PIPE)
    cache = None
    resume = bool(p.get("resume", True)) and F > clip_frames
    if resume:
        cache = ClipCache(out + ".chunks", {"W": W, "H": H, "F": F, "FPS": FPS, "steps": steps, "cfg": cfg, "seed": seed,
                                            "sched": sched, "clip": clip_frames, "motion": motion_frames,
                                            "shift": float(p.get("shift", 5.0)), "scale": float(p.get("audio_scale", 2.0)),
                                            "loras": [(os.path.basename(a), b) for a, b in loras],
                                            "audio": os.path.abspath(p["audio_path"]), "image": os.path.abspath(img_path)}, FPS)
    tg = time.time()
    done = 0
    for frames in eng.generate(image=image, audio_emb=emb, total_frames=F, width=W, height=H,
                               prompt=p.get("prompt", ""), negative=p.get("negative", ""), steps=steps,
                               shift=float(p.get("shift", 5.0)), cfg=cfg, audio_cfg=acfg,
                               audio_scale=float(p.get("audio_scale", 2.0)), scheduler=sched, seed=seed,
                               clip_frames=clip_frames, motion_frames=motion_frames,
                               ref_anchor=float(p.get("ref_anchor_strength", 0.0) or 0.0), cache=cache,
                               on_progress=lambda a, b: print(f"[gen] {a}/{b} frame", flush=True)):
        take = frames[: max(0, F - done)]
        if take.shape[0]:
            writer.stdin.write(np.ascontiguousarray(take).tobytes())
            done += take.shape[0]
    writer.stdin.close()
    writer.wait()
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"[gen] {time.time() - tg:.0f}s, {done} frame totali, picco VRAM {peak:.0f} MB "
          f"({(time.time() - tg) / max(1, done):.2f} s/frame)", flush=True)
    # 5) audio reale (durata = frame/FPS) -> resta in sync
    dur = done / FPS
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", p["audio_path"], "-t", f"{dur:.3f}", "-c:v", "copy", "-c:a", "aac",
                    "-map", "0:v:0", "-map", "1:a:0", "-shortest", out], check=False, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    try:
        os.remove(tmp)
    except OSError:
        pass
    if not os.path.exists(out):
        raise RuntimeError("mux audio fallito: nessun file in uscita")
    if resume and not p.get("keep_chunks"):
        import shutil
        shutil.rmtree(out + ".chunks", ignore_errors=True)
    try:  # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark visibile
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception as e:                                      # noqa: BLE001
        print(f"[stego] {e}", flush=True)
    # #106 watermark obbligatorio (gated feature_matrix); RuntimeError = errore del job (fail-closed)
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except RuntimeError:
        raise
    except Exception as e:                                      # noqa: BLE001
        print(f"[watermark] {e}", flush=True)
    return out


def main():
    import torch
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    try:
        with torch.inference_mode():
            print("OK:" + run(p), flush=True)
    except Exception as e:                                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ERRORE: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
