"""MiniMax H3 EDIT con maschera a clip in catena — «Cambia sfondo» e «Sostituisci personaggio» (una persona, tracking SAM2).

Richiamato da workers/minimax_h3_infer.py::main quando i params hanno `h3_edit`. Nativo: niente ComfyUI, niente codice esterno eseguito.
Flusso (matematica in h3_edit_math.py, tenuta delle righe in h3_edit_hold.py, prompt in core/h3_edit_prompt.py):
 1. sorgente a 24 fps alla tela scelta (lato corto), maschere SAM2 per frame -> maschera persona (spazio sopra la testa, posa tenuta, crescita);
 2. video di riferimento: cambia-sfondo = sfondo sfocato grigio (la persona resta); sostituisci = persona -> manichino grigio;
 3. piano dei clip (6 s, 39 frame di sovrapposizione, tagli netti rispettati);
 4. per ogni clip: sorgente codificata dal VAE (i primi `contesto` frame = frame GIA generati del clip prima, tenuti), maschera di latente
    (1 = rigenera, 0 = tieni; feather), audio sorgente tenuto, rumore fisso; Ref2VA con foto + riferimento video; denoise con righe tenute;
 5. composizione: la zona rigenerata e incollata nei frame ORIGINALI (fuori dalla maschera = pixel del sorgente, senza giro nel VAE);
 6. montaggio dei clip (i frame di contesto non si ripetono) + audio originale.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import h3_edit_math as M

FPS = M.FPS
_SRC_AUDIO_SR = 32000


def _ensure(cond, msg):
    if not cond:
        raise RuntimeError(msg)


# ── ingressi: video, maschere ─────────────────────────────────────────────────────────────────────
def read_frames(path: str, tw: int, th: int, max_frames: int = 0):
    """Video -> (uint8 [N, th, tw, 3] a 24 fps, durata s). Il ricampionamento prende il frame piu vicino nel tempo."""
    import av
    with av.open(path) as c:
        vs = c.streams.video[0]
        dur = float(c.duration / 1_000_000) if c.duration else (float(vs.duration * vs.time_base) if vs.duration else 0.0)
        frames, times = [], []
        for fr in c.decode(vs):
            t = float(fr.pts * vs.time_base) if fr.pts is not None else len(frames) / float(vs.average_rate or 24)
            times.append(t)
            frames.append(fr.to_ndarray(width=tw, height=th, format="rgb24"))
    _ensure(frames, f"nessun frame leggibile in {path}")
    dur = dur or (times[-1] + 1.0 / FPS)
    n = max(1, int(dur * FPS + 1e-6))
    if max_frames:
        n = min(n, max_frames)
    ts = np.asarray(times)
    idx = np.clip(np.searchsorted(ts, (np.arange(n) + 0.5) / FPS) - 1, 0, len(frames) - 1)
    out = np.stack([frames[i] for i in idx])
    return out, dur


def read_masks(mask_dir: str, n24: int, tw: int, th: int, src_fps_hint: float = 0.0) -> torch.Tensor:
    """Maschere SAM2 (png per frame del sorgente, bianco = oggetto) -> [n24, th, tw] float, una per frame a 24 fps."""
    from PIL import Image
    files = sorted(Path(mask_dir).glob("*.png"))
    _ensure(files, f"nessuna maschera in {mask_dir}: il tracking SAM2 non ha prodotto nulla")
    idx = np.clip(np.round(np.arange(n24) * (len(files) - 1) / max(1, n24 - 1)).astype(int), 0, len(files) - 1)
    cache, out = {}, torch.zeros(n24, th, tw)
    for k, i in enumerate(idx):
        if i not in cache:
            cache[i] = torch.from_numpy((np.asarray(Image.open(files[i]).convert("L").resize((tw, th), Image.NEAREST)) > 127).astype("float32"))
        out[k] = cache[i]
    return out


# ── codifica: sorgente -> righe di latente ───────────────────────────────────────────────────────
def encode_source_latent(vae, frames_u8: torch.Tensor, components_like) -> torch.Tensor:
    """frames [n, h, w, 3] uint8 (CPU) -> (1, C, T, h/16, w/16) float32 CPU normalizzato, con la STESSA ricetta dei riferimenti video di diffusers."""
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    dev = next(vae.parameters()).device
    pixels = frames_u8.permute(3, 0, 1, 2)[None].to(dev)
    with torch.no_grad():
        return encode_vae_condition(vae, pixels, components_like.pixel_mean, components_like.pixel_std, components_like.keyframe_encode_seed)


class _Const:                                                # le costanti del layout H3 (modular_pipeline.py), senza costruire la pipeline
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)
    keyframe_encode_seed = 42


def encode_source_audio(audio_vae, wav: torch.Tensor, n_latents: int) -> torch.Tensor:
    """wav [2, samples] a 32 kHz -> righe (2*n_latents, C) normalizzate, canale-maggiore (come l'audio dei riferimenti)."""
    mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)
    dev = next(audio_vae.parameters()).device
    with torch.no_grad():
        post = audio_vae.encode(wav.to(dev)[:, None], return_dict=False)[0]
        lat = post.mode().float().cpu().transpose(1, 2)                 # (2, T, C)
    lat = (lat - mean) / std
    T = lat.shape[1]
    if T < n_latents:                                                   # clip finale piu corto dell'atteso: si completa col silenzio
        lat = torch.cat([lat, torch.zeros(lat.shape[0], n_latents - T, lat.shape[2])], 1)
    return lat[:, :n_latents].reshape(-1, lat.shape[2])


def to_stereo(wav: torch.Tensor) -> torch.Tensor:
    """[canali, campioni] -> [2, campioni]: l'audio VAE di H3 e diffusers (encode_video) vogliono STEREO. Un sorgente mono (es. un video parlato)
    si duplica sui due canali — giro 2/11: «Expected samples with 2 channels; got shape [1, 188000]»; piu di 2 canali: i primi due."""
    if wav.dim() == 1:
        wav = wav[None]
    if wav.shape[0] == 1:
        return wav.repeat(2, 1)
    return wav[:2]


# ── composizione finale ──────────────────────────────────────────────────────────────────────────
def composite(gen: torch.Tensor, orig: torch.Tensor, cells: torch.Tensor, counts: list[int], blend_px: int = 8) -> torch.Tensor:
    """gen/orig [n, h, w, 3] float 0..1; cells [1,1,T,h/16,w/16] (1 = rigenerato). Fuori dalla maschera = pixel ORIGINALI (nessun giro nel VAE)."""
    n, h, w = gen.shape[:3]
    latent_of = [t for t, c in enumerate(counts) for _ in range(c)]
    cells = cells[0, 0]
    out = torch.empty_like(orig, dtype=torch.float32)
    r = max(0, blend_px // 2)
    for i in range(0, n, 16):
        j = min(n, i + 16)
        a = F.interpolate(cells[latent_of[i:j]][:, None].float(), size=(h, w), mode="bilinear", align_corners=False)
        if r:
            k = 2 * r + 1
            a = F.avg_pool2d(F.avg_pool2d(a, (1, k), 1, (0, r), count_include_pad=False), (k, 1), 1, (r, 0), count_include_pad=False)
        a = a.movedim(1, -1)
        out[i:j] = a * gen[i:j] + (1 - a) * orig[i:j]
    return out


def _to_float_frames(videos) -> torch.Tensor:
    v = videos if isinstance(videos, torch.Tensor) else torch.as_tensor(np.asarray(videos))
    v = v.detach().cpu()
    return v.float() / 255.0 if v.dtype == torch.uint8 else v.float().clamp(0, 1)


def _write_ref_video(frames01: torch.Tensor, path: str, audio: torch.Tensor | None = None) -> str:
    """Video di riferimento (<Video 1>) su disco, 24 fps, H.264, CON l'audio sorgente del clip se c'e (cosi diventa <Audio 1>, come nel flusso originale)."""
    from diffusers.utils.export_utils import encode_video
    encode_video((frames01.clamp(0, 1) * 255).round().to(torch.uint8), fps=FPS, output_path=path,
                 audio=audio, audio_sample_rate=_SRC_AUDIO_SR if audio is not None else None)
    return path


# ── orchestrazione ───────────────────────────────────────────────────────────────────────────────
# Un CLIP = un PROCESSO (giro 02/11): in-process, dopo il clip 1 restavano 4-8 GB di VRAM «orfani» (blocchi del DiT, LoRA, buffer di h3_chunk) e il
# clip 2 moriva contro il tetto per processo (21,29 GiB «allowed» con 6,8 GiB liberi sulla scheda). Il genitore (questa funzione) fa solo CPU:
# sorgente, maschere, riferimenti, piano, composizione, montaggio. Ogni clip lo genera un FIGLIO (`minimax_h3_infer.py --h3-edit-clip <cartella>`)
# che parte con la VRAM pulita, e i clip finiti restano su disco: se uno fallisce, la volta dopo si riprende da li (stesso input = stessa chiave).
def _sha(*parts) -> str:
    import hashlib
    h = hashlib.sha1()
    for x in parts:
        h.update(x if isinstance(x, (bytes, bytearray)) else str(x).encode("utf-8"))
    return h.hexdigest()[:16]


STREAM_ALL_ENV = {"H3_STREAM_ALL": "1", "H3_BLOCK_CHUNK": "2048", "H3_ATTN_QCHUNK": "2048"}   # ripiego: blocchi del DiT in RAM + temporanei piccoli


def is_oom_error(ex: BaseException) -> bool:
    m = str(ex).lower()
    return "out of memory" in m or "outofmemoryerror" in m or "vram) esaurita" in m


def run_clip_with_fallback(W, clip_runner, job_dir: str) -> bool:
    """Genera UN clip; se il figlio muore per memoria GPU lo rifa DA CAPO (stesso clip, stessa chiave, stesso checkpoint: cambia solo il piano di memoria)
    con lo streaming completo dei blocchi del DiT e blocchi di token piu piccoli. Ritorna True se e servito il ripiego. Un secondo OOM resta un errore."""
    try:
        clip_runner(W, job_dir)
        return False
    except RuntimeError as ex:
        if not is_oom_error(ex):
            raise
        W._log("[h3] memoria insufficiente, riprovo con piu pezzi del modello in RAM (piu lento)")
    clip_runner(W, job_dir, stream_all=True)
    return True


def _subprocess_clip_runner(W, job_dir: str, stream_all: bool = False) -> None:
    """Lancia il figlio che genera UN clip; il suo log passa su stdout del genitore (finisce nel log completo del job)."""
    import collections
    import subprocess
    import sys
    worker = str(Path(W.__file__).resolve())
    env = dict(os.environ, **STREAM_ALL_ENV) if stream_all else None
    proc = subprocess.Popen([sys.executable, "-u", worker, "--h3-edit-clip", str(job_dir)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    tail = collections.deque(maxlen=80)
    for ln in proc.stdout:
        sys.stdout.write(ln)
        sys.stdout.flush()
        tail.append(ln)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"il clip non e stato generato (codice {rc}):\n" + "".join(tail))


def run_edit(p: dict, W, clip_runner=None) -> str:
    """`W` = il modulo minimax_h3_infer. `clip_runner(W, job_dir)` (iniettabile nei test) genera un clip; di default un sottoprocesso."""
    e = dict(p["h3_edit"])
    action = e.get("action") or "change_background"
    _ensure(action in ("change_background", "replace_character"), f"azione H3 edit sconosciuta: {action}")
    src = e["video"]
    _ensure(os.path.isfile(src), f"video sorgente non trovato: {src}")
    clip_runner = clip_runner or _subprocess_clip_runner
    t_all = time.time()
    short = int(e.get("short_edge") or 480)
    from core import h3_edit_prompt as PR  # noqa: F401  (usato dal figlio; qui si verifica che l'import funzioni prima di spendere minuti)

    # 1) tela, sorgente e maschere
    with __import__("av").open(src) as c0:
        sw, sh = c0.streams.video[0].codec_context.width, c0.streams.video[0].codec_context.height
    tw, th = M.short_edge_size(sw, sh, short)
    frames_u8, dur = read_frames(src, tw, th, int(e.get("max_frames") or 0))
    N = M.usable_frames(frames_u8.shape[0])
    frames_u8 = frames_u8[:N]
    W._log(f"edit {action}: {N} frame ({N / FPS:.1f}s) a {tw}x{th} da {sw}x{sh}")
    src_t = torch.from_numpy(frames_u8)                                # uint8 [N, th, tw, 3]
    f01 = src_t.float() / 255.0
    obj = read_masks(e["mask_dir"], N, tw, th)
    scale = tw / float(sw)
    invert = action == "change_background"
    # Spazio sopra la testa e posa tenuta servono a sostituisci (personaggio nuovo piu alto, gesti diversi). Nel cambia-sfondo la persona e TENUTA
    # dall'originale: quello spazio restava sfondo vecchio (pennacchio del mercato sopra la testa, misurato 03/11: fascia 0-24 px 40-54 % a 10-12 s);
    # la cella di latente unisce gia 4 frame, quindi la posa tenuta li e ridondante. Parametri bg_headroom / bg_hold_frames (default 0).
    headroom = float(e.get("bg_headroom", 0.0)) if invert else float(e.get("headroom", 0.10))
    hold_frames = int(e.get("bg_hold_frames", 0)) if invert else int(e.get("hold_frames", 2))
    pers, size = M.person_mask(obj, headroom, hold_frames)
    pers = pers.float()
    person_grow = max(0, int(round(float(e.get("person_grow_px", 40)) * scale)))
    grow = max(0, int(round(float(e.get("grow_px", 16)) * scale)))
    # SFUMATURA (feather, in celle da 16 px fuori dalla zona rigenerata): nel cambia-sfondo la zona rigenerata e lo sfondo, quindi la rampa cade
    # DENTRO il bordo della persona; con 2 px (= 1 cella a 0,5) la rampa era larga ~15 px con la linea a 0,5 a ~20 px dentro la sagoma; con 0 e larga
    # ~9 px a ~10 px dentro (misurato sui frame del 03/11). Nel cambia-sfondo si usa 0 (parametro bg_feather_px), per sostituisci resta feather_px.
    feather_px = int(e.get("bg_feather_px", 0)) if invert else int(e.get("feather_px", 2))
    regen_pix = M.regen_region(pers, invert, person_grow, grow, expand=0.0 if invert else float(os.environ.get("H3_EDIT_REPLACE_EXPAND") or e.get("replace_expand", 0.0)))   # cio che si rigenera (prima dell'inversione)
    if not invert:
        pers = M.grow_mask(pers.float(), person_grow)                  # il manichino grigio del riferimento copre la persona cresciuta

    # 2) video di riferimento (<Video 1>) — la zona rigenerata e nascosta: niente dettagli originali da copiare
    if invert:
        ref_frames = M.blur_area(f01, 1.0 - pers, radius=max(6.0, 0.02 * th * float(os.environ.get("H3_EDIT_BG_BLUR") or e.get("bg_blur", 1.0))),
                                grey=float(os.environ.get("H3_EDIT_BG_GREY") or e.get("bg_grey", 0.35)))   # H3_EDIT_BG_BLUR = moltiplicatore del raggio (prova 03/11, default 1)
    else:
        ref_frames = M.grey_mannequins(f01, pers, size, blur=0.02)

    # 3) piano dei clip
    clip_frames = int(float(e.get("clip_s", 6.0)) * FPS)
    cuts = [a for a, _b in M.find_shots(f01[:, ::4, ::4], 40.0)[1:]]
    clips = M.enforce_min_length(M.clip_plan(N, clip_frames, int(e.get("overlap", 39)), cuts), N)   # nessun clip sotto i 5 s (minimo del pacchetto H3)
    W._log(f"piano: {len(clips)} clip (lunghezza, contesto) = {clips}")

    wav = W._decode_audio_av(src) if e.get("keep_audio", True) else None
    if wav is not None:
        wav = to_stereo(wav)
    pic = e.get("picture") or ""
    base = dict(p)
    base.pop("h3_edit", None)
    base.pop("output", None)
    base["h3_mode"] = "ref2va"
    out_frames = torch.zeros(N, th, tw, 3)
    done = 0
    seed0 = int(p.get("seed") or 0)
    st = os.stat(src)
    runkey = _sha(os.path.abspath(src), st.st_size, int(st.st_mtime), action, short, e.get("keep_text"), e.get("prompt_text"), e.get("picture_desc"),
                  pic, os.path.getmtime(pic) if pic and os.path.isfile(pic) else 0, seed0, p.get("h3_profile"), p.get("h3_turbo"), e.get("clip_s"),
                  e.get("overlap"), N, e.get("replace_expand"), os.environ.get("H3_EDIT_REPLACE_EXPAND"), os.environ.get("H3_EDIT_BG_GREY"), os.environ.get("H3_EDIT_BG_BLUR"), e.get("bg_feather_px"),
                  *((("bgmask_v2", headroom, hold_frames),) if invert else ()))   # solo il cambia-sfondo cambia chiave: i checkpoint di sostituisci restano validi
    tmp = Path(p.get("output") or W.REPO / "output").parent / "_h3edit_tmp" / runkey
    tmp.mkdir(parents=True, exist_ok=True)
    prev_tail = None                                                   # ultimi frame composti del clip prima (contesto del successivo)
    import numpy as _np
    for ci, (length, ctx) in enumerate(clips):
        start = done - ctx
        sl = slice(start, start + length)
        clip_src = src_t[sl].clone()
        if ctx and prev_tail is not None:                              # contesto = frame GIA generati (non la sorgente): continuita fra clip
            clip_src[:ctx] = (prev_tail[-ctx:] * 255).round().to(torch.uint8)
        cells, counts = M.cell_mask(regen_pix[sl], tw, th, invert=invert, feather_px=feather_px, width=tw)
        if ctx:                                                        # i frame di contesto non si rigenerano
            cells = cells.clone()
            cells[:, :, :_latents_of_frames(counts, ctx)] = 0
        jd = tmp / f"clip_{ci:02d}"
        jd.mkdir(exist_ok=True)
        seg = None
        if wav is not None:
            s0, s1 = int(start / FPS * _SRC_AUDIO_SR), int((start + length) / FPS * _SRC_AUDIO_SR)
            seg = wav[:, s0:s1]
            if seg.shape[1] < (s1 - s0):
                seg = torch.cat([seg, torch.zeros(seg.shape[0], (s1 - s0) - seg.shape[1])], 1)
        refvid = _write_ref_video(ref_frames[sl], str(jd / "ref.mp4"), seg)
        meta = {"ci": ci, "clips": len(clips), "length": int(length), "ctx": int(ctx), "start": int(start), "seed": seed0 + ci, "th": th, "tw": tw,
                "picture": pic, "refvid": refvid, "action": action, "keep_text": e.get("keep_text") or "the person",
                "prompt_text": e.get("prompt_text") or "", "picture_desc": e.get("picture_desc") or "", "shots": e.get("shots") or "",
                "has_audio": seg is not None, "base": base, "audio_seed": seed0 + 104729 * ci, "noise_seed": seed0 + 7919 * ci}
        _np.save(jd / "clip_src.npy", clip_src.numpy())
        _np.save(jd / "cells.npy", cells.numpy())
        if seg is not None:
            _np.save(jd / "seg.npy", seg.numpy())
        (jd / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        key = _sha(clip_src.numpy().tobytes(), cells.numpy().tobytes(), json.dumps(meta, sort_keys=True, default=str))
        gen_p, key_p = jd / "gen.npy", jd / "key.txt"
        if gen_p.exists() and key_p.exists() and key_p.read_text().strip() == key:
            W._log(f"--- clip {ci + 1}/{len(clips)}: GIA GENERATO (ripresa da {gen_p}) ---")
        else:
            if key_p.exists():
                key_p.unlink()
            if gen_p.exists():
                gen_p.unlink()
            (jd / "want_key.txt").write_text(key)
            W._log(f"--- clip {ci + 1}/{len(clips)}: {length} frame (contesto {ctx}) — processo dedicato ---")
            run_clip_with_fallback(W, clip_runner, str(jd))
            _ensure(gen_p.exists(), f"il clip {ci + 1} non ha prodotto {gen_p}")
            key_p.write_text(key)
        gen_f = torch.from_numpy(_np.load(gen_p)).float() / 255.0
        _ensure(gen_f.shape[0] >= length, f"il clip {ci + 1} ha {gen_f.shape[0]} frame, attesi {length}")
        gen_f = gen_f[:length]
        if gen_f.shape[1:3] != (th, tw):
            gen_f = F.interpolate(gen_f.movedim(-1, 1), size=(th, tw), mode="bilinear", align_corners=False).movedim(1, -1)
        comp = composite(gen_f, clip_src.float() / 255.0, cells, counts)          # clip_src ha gia il contesto = frame del clip prima
        out_frames[start + ctx:start + length] = comp[ctx:]
        prev_tail = comp
        done = start + length
        del gen_f, comp, cells
        gc.collect()
        _save_progress(tmp, ci, len(clips))

    # 5) file finale con l'audio originale
    out = p.get("output") or str(W.REPO / "output" / f"h3edit_{action}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    from diffusers.utils.export_utils import encode_video
    audio = None
    if wav is not None:
        audio = wav[:, : int(done / FPS * _SRC_AUDIO_SR)]
    encode_video((out_frames[:done].clamp(0, 1) * 255).round().to(torch.uint8), fps=FPS, output_path=out,
                 audio=audio, audio_sample_rate=_SRC_AUDIO_SR if audio is not None else None)
    W._log(f"edit finito in {time.time() - t_all:.0f}s -> {out}")
    return out


def generate_clip(job_dir: str, W) -> None:
    """FIGLIO: genera UN clip (VAE sorgente -> latente + maschera + audio -> Ref2VA con hold -> frame). Processo fresco = VRAM pulita."""
    import numpy as _np
    from core import h3_edit_prompt as PR
    jd = Path(job_dir)
    m = json.loads((jd / "meta.json").read_text("utf-8"))
    length, ctx, th, tw, ci = m["length"], m["ctx"], m["th"], m["tw"], m["ci"]
    clip_src = torch.from_numpy(_np.load(jd / "clip_src.npy"))
    cells = torch.from_numpy(_np.load(jd / "cells.npy"))
    base = dict(m["base"])
    Wd = W.resolve({**base, "references": [{"kind": "image"}]})        # solo per i percorsi di VAE/configurazioni (modo ref2va)
    vae = W.load_vae(Wd)
    lat = encode_source_latent(vae, clip_src, _Const)
    W._drop_module_weights(vae); del vae; W._gc()
    T, lh, lw = lat.shape[2], lat.shape[3], lat.shape[4]
    _ensure(tuple(cells.shape[2:]) == (T, lh, lw), f"maschera {tuple(cells.shape[2:])} e latente {(T, lh, lw)} non coincidono")
    noise = torch.randn(lat.shape, generator=torch.Generator("cpu").manual_seed(m["noise_seed"]), dtype=torch.float32)

    def video_hold(patch, _lat=lat, _noise=noise, _cells=cells):
        from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
        keep = (1.0 - _cells).expand(1, _lat.shape[1], -1, -1, -1).contiguous()
        return patchify_video_latents(_lat, patch), patchify_video_latents(_noise, patch), patchify_video_latents(keep, patch)

    audio_hold, audio_noise = None, None
    if m["has_audio"]:
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames
        seg = torch.from_numpy(_np.load(jd / "seg.npy"))
        avae = W.load_audio_vae(Wd)
        arows = encode_source_audio(avae, seg, audio_latent_num_frames(length))
        W._drop_module_weights(avae); del avae; W._gc()
        anoise = torch.randn(arows.shape, generator=torch.Generator("cpu").manual_seed(m["audio_seed"]), dtype=torch.float32)
        audio_noise = anoise.T[None].contiguous()                      # (1, C, rows): come lo vuole PrepareLatents
        audio_hold = (lambda _a=arows, _n=anoise: (_a, _n, torch.ones_like(_a)))
    refs = []
    if m["picture"]:
        refs.append({"kind": "image", "path": m["picture"], "role": "subject"})
    refs.append({"kind": "video", "path": m["refvid"], "role": "motion", "frames": int(length)})   # `frames`: serve al piano di memoria
    lip = bool(m["has_audio"])
    if m["action"] == "change_background":
        prompt = PR.background_prompt(m["prompt_text"], m["keep_text"], length, ctx, has_picture=bool(m["picture"]), picture_desc=m["picture_desc"],
                                      shots=m["shots"], lip_sync=lip)
    else:
        prompt = PR.replace_prompt(m["keep_text"], length, ctx, picture_desc=m["picture_desc"], shots=m["shots"], lip_sync=lip)
    pc = dict(base)
    pc.update(prompt=prompt, references=refs, seed=m["seed"], num_frames=length, duration_s=length / FPS, width=tw, height=th,
              generate_audio=False, _h3_edit={"noise": noise, "audio_noise": audio_noise, "video_hold": video_hold, "audio_hold": audio_hold,
                                              "frames": length, "canvas": (th, tw)})
    W._log(f"clip {ci + 1}/{m['clips']}: {length} frame, contesto {ctx}, tela {tw}x{th}")
    res = W.run(pc)
    gen_f = _to_float_frames(res["videos"][0])
    _ensure(gen_f.shape[0] >= length, f"il clip {ci + 1} ha {gen_f.shape[0]} frame, attesi {length}")
    out = (gen_f[:length].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    _np.save(jd / "gen.npy", out)


def _latents_of_frames(counts: list[int], frames: int) -> int:
    """Quanti frame di latente coprono i primi `frames` frame-pixel (il contesto)."""
    acc = 0
    for k, c in enumerate(counts):
        acc += c
        if acc >= frames:
            return k + 1
    return len(counts)


def _save_progress(tmp: Path, i: int, n: int) -> None:
    try:
        (tmp / "progress.json").write_text(json.dumps({"clip": i + 1, "clips": n}), encoding="utf-8")
    except OSError:
        pass
