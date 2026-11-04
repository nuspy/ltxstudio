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

ORIGINAL_SAM_CANVAS_PX = 1008                  # lato lungo del video che SAM vede in Openjutsu: i margini dell'originale (10/20/40 px) sono a questa scala
FPS = M.FPS
_SRC_AUDIO_SR = 32000


def _ensure(cond, msg):
    if not cond:
        raise RuntimeError(msg)


# ── ingressi: video, maschere ─────────────────────────────────────────────────────────────────────
def read_frames(path: str, tw: int, th: int, max_frames: int = 0):
    """Video -> (uint8 [N, th, tw, 3] a 24 fps, durata s, indice del frame sorgente di ogni frame a 24 fps, frame sorgente totali).
    Il ricampionamento prende il frame piu vicino nel tempo."""
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
    return out, dur, idx, len(frames)


def read_masks(mask_dir: str, src_idx, n_src: int, tw: int, th: int) -> torch.Tensor:
    """Maschere SAM2 (png per frame del sorgente, bianco = oggetto) -> [len(src_idx), th, tw] float, una per frame a 24 fps.
    `src_idx` = frame sorgente usato da read_frames per ogni frame a 24 fps: la maschera e quella DELLO STESSO frame sorgente.
    (Fino al 03/11 le maschere venivano stirate sui soli N frame usati: con 286 maschere e 277 frame la maschera finiva 9 frame avanti,
    la persona tenuta «in ritardo» lasciava sfondo vecchio da un lato e veniva rigenerata dall'altro.) Se il numero di maschere non
    coincide con i frame sorgente (tracking a un altro passo) si mappa per tempo sull'intera durata."""
    from PIL import Image
    files = sorted(Path(mask_dir).glob("*.png"))
    _ensure(files, f"nessuna maschera in {mask_dir}: il tracking SAM2 non ha prodotto nulla")
    src_idx = np.asarray(src_idx)
    n24 = len(src_idx)
    if len(files) == n_src:
        idx = src_idx
    else:
        idx = np.round(src_idx * (len(files) - 1) / max(1, n_src - 1)).astype(int)
    idx = np.clip(idx, 0, len(files) - 1)
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


def opts_key(p: dict) -> tuple:
    """Scelte di velocita/modello dell'utente nell'edit (DiT personalizzato, Veda, LightVAE, LoRA): cambiano il risultato, quindi entrano nella chiave dei
    checkpoint. Presenti SOLO se scelte: con i default la chiave e quella di sempre e i checkpoint gia fatti restano validi."""
    if not (p.get("h3_dit") or p.get("h3_veda") or p.get("h3_lightvae") or p.get("loras") or p.get("h3_te")):
        return ()
    return (("opts", p.get("h3_dit"), bool(p.get("h3_veda")), p.get("h3_lightvae"), json.dumps(p.get("loras") or [], sort_keys=True)) + ((p.get("h3_te"),) if p.get("h3_te") else ()),)


def guide_key(e: dict) -> tuple:
    """Parte della chiave dei checkpoint dovuta alla guida di latenti allineata (workers/h3_guide.py): VUOTA se l'edit non la usa, cosi i clip gia fatti restano validi."""
    from core import h3_guide as CG
    out = CG.spec_key({"frame_idx": int(e.get("guide_frame_idx") or 0)} if e.get("guide") else None)
    if e.get("guide") and e.get("guide_mode") == "person" and e.get("action", "change_background") == "change_background":
        out += (("guide_person_v1",),)                                  # guida = solo la persona su grigio (cambia-sfondo): cambia i clip
    if e.get("ref_clock_align"):
        out += (("refclock_v1",),)                                      # orologio del riferimento video allineato a quello del bersaglio: cambia i clip
    return out


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
    __import__("core.h3_edit_prompt")                                  # il figlio usa il prompt: qui si verifica che l'import funzioni prima di spendere minuti

    # 1) tela, sorgente e maschere
    with __import__("av").open(src) as c0:
        sw, sh = c0.streams.video[0].codec_context.width, c0.streams.video[0].codec_context.height
    tw, th = M.short_edge_size(sw, sh, short)
    frames_u8, dur, src_idx, n_src = read_frames(src, tw, th, int(os.environ.get("H3_EDIT_MAX_FRAMES") or e.get("max_frames") or 0))
    N = M.usable_frames(frames_u8.shape[0])
    frames_u8 = frames_u8[:N]
    W._log(f"edit {action}: {N} frame ({N / FPS:.1f}s) a {tw}x{th} da {sw}x{sh}")
    src_t = torch.from_numpy(frames_u8)                                # uint8 [N, th, tw, 3]
    f01 = src_t.float() / 255.0
    obj = read_masks(e["mask_dir"], src_idx[:N], n_src, tw, th)
    scale = tw / float(sw)
    invert = action == "change_background"
    # Spazio sopra la testa e posa tenuta servono a sostituisci (personaggio nuovo piu alto, gesti diversi). Nel cambia-sfondo la persona e TENUTA
    # dall'originale: quello spazio restava sfondo vecchio (pennacchio del mercato sopra la testa, misurato 03/11: fascia 0-24 px 40-54 % a 10-12 s);
    # la cella di latente unisce gia 4 frame, quindi la posa tenuta li e ridondante. Parametri bg_headroom / bg_hold_frames (default 0).
    # H3_EDIT_BG_GROW_LIKE_REPLACE=1 (ABLAZIONE 03/11, spento di default): nel cambia-sfondo la maschera TENUTA e quella di Sostituisci (persona cresciuta di person_grow + grow,
    # spazio sopra la testa 0,10, posa +-2 frame); matte e riferimento restano sulla sagoma vera.
    # H3_EDIT_BG_MASK=original (confronto con Openjutsu, 03/11): la zona rigenerata del cambia-sfondo e quella dell'originale. KJNodes GrowMaskWithBlur con flip_input=true
    # (workflow object.api.json, lib/object.ts: margine 10/20/40 px a 1008 px di SAM, di default 20) inverte la maschera e POI la allarga: lo sfondo entra nel bordo della
    # persona. Persona = SAM con posa tenuta 2 frame e senza spazio sopra la testa. Nessun matte a pixel nell'originale: nel caso di confronto va spento (H3_EDIT_BG_MATTE=0).
    original_mask = invert and str(os.environ.get("H3_EDIT_BG_MASK") or e.get("bg_mask") or "").strip().lower() == "original"
    # fill_holes (H3_EDIT_BG_MASK_FILL_HOLES=1, spento di default): l'originale lo ha acceso; nei fotogrammi in cui la persona (ristretta) non tocca il bordo riempie la «buca» e
    # rigenera tutto il fotogramma, persona compresa. Sul video del rifugio sono 94 fotogrammi su 141: opzione a parte, da usare solo se l'originale mostra davvero quel comportamento.
    fill_holes = str(os.environ.get("H3_EDIT_BG_MASK_FILL_HOLES") or e.get("bg_mask_fill_holes") or "0").strip().lower() not in ("0", "false", "no", "off")
    like_replace = invert and str(os.environ.get("H3_EDIT_BG_GROW_LIKE_REPLACE") or e.get("bg_grow_like_replace") or "0").strip().lower() not in ("0", "", "false", "no", "off")
    headroom = float(e.get("bg_headroom", 0.0)) if invert else float(e.get("headroom", 0.10))
    hold_frames = int(e.get("bg_hold_frames", 0)) if invert else int(e.get("hold_frames", 2))
    if like_replace:
        headroom, hold_frames = float(e.get("headroom", 0.10)), int(e.get("hold_frames", 2))
    if original_mask:
        headroom, hold_frames = 0.0, 2
    # bg_hold_cover (env H3_EDIT_BG_HOLD_COVER): quota minima di persona per TENERE una cella nel cambia-sfondo (0 = spento: serve il 100 %, basta un pixel
    # di sfondo per rigenerarla). Con 0,5 volto e spalle restano ancorati; il matte a pixel pulisce lo sfondo vecchio nei bordi delle celle tenute.
    hold_cover = float(os.environ.get("H3_EDIT_BG_HOLD_COVER") or e.get("bg_hold_cover", 0.0) or 0.0) if invert else 0.0
    if not 0.0 < hold_cover < 1.0:
        hold_cover = 0.0
    # first_clip_two_pass (env H3_EDIT_FIRST_TWO_PASS: 1 = si, un numero = frame di contesto; default spento; frame = quelli di `first_two_pass_frames`, default 22,
    # portati al confine di frame di latente piu vicino per difetto): solo cambia-sfondo.
    _tp = str(os.environ.get("H3_EDIT_FIRST_TWO_PASS") or e.get("first_clip_two_pass") or "0").strip().lower()
    two_n = 0
    if invert and _tp not in ("0", "", "false", "no", "off"):
        _n = int(e.get("first_two_pass_frames") or 22) if _tp in ("1", "true", "yes", "on") else int(float(_tp))
        two_n = M.latent_boundary_floor(_n)
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
    if like_replace:                                                   # persona e margine TENUTI come in Sostituisci; matte e riferimento sulla sagoma vera
        regen_pix = M.regen_region(pers, False, person_grow, grow)
        pers, size = M.person_mask(obj, 0.0, 0)
        pers = pers.float()
    if original_mask:                                                  # la persona tenuta e quella ristretta del margine: la usano anche il riferimento (zona grigia = zona rigenerata) e l'eventuale matte
        margin_px = int(round(float(os.environ.get("H3_EDIT_BG_MASK_MARGIN") or e.get("bg_mask_margin", 20)) * max(tw, th) / ORIGINAL_SAM_CANVAS_PX))
        pers = M.erode_cross(pers, margin_px)
        enclosed_frames = 0
        if fill_holes:                                                 # come l'originale: una persona tutta dentro il fotogramma sparisce dalla maschera (si rigenera tutto)
            pers, enclosed_frames = M.drop_enclosed_regions(pers)
        regen_pix = pers
        W._log(f"maschera come Openjutsu: persona ristretta di {margin_px} px (20 px a 1008 -> {max(tw, th)} px), posa tenuta {hold_frames} frame; "
               f"fill_holes {'attivo' if fill_holes else 'spento'}: {enclosed_frames} fotogrammi con la persona tutta dentro il fotogramma (rigenerati per intero)")

    # 1b) PREROLL (opzionale, default 0): frame fermi anteposti perche il clip 1 non parta «a freddo» (vedi M.preroll_pad). H3_EDIT_PREROLL=17 / `preroll_frames`.
    preroll = int(os.environ.get("H3_EDIT_PREROLL") or e.get("preroll_frames") or 0)
    wav = W._decode_audio_av(src) if e.get("keep_audio", True) else None
    if wav is not None:
        wav = to_stereo(wav)
    if preroll:
        src_t, pers, wav = M.preroll_pad(src_t, pers, wav, preroll, _SRC_AUDIO_SR)
        size = torch.cat([size[:1].expand(preroll, -1, -1), size], 0)
        f01 = src_t.float() / 255.0
        N = src_t.shape[0]
        W._log(f"preroll: {preroll} frame fermi anteposti ({N} frame in totale, tolti alla fine)")

    # 2) video di riferimento (<Video 1>) — la zona rigenerata e nascosta: niente dettagli originali da copiare
    bg_person_ref = str(os.environ.get("H3_EDIT_BG_PERSON_REF") or e.get("bg_person_ref") or "sharp").lower()   # sharp (oggi) | mannequin | erase (nessuna figura nel riferimento del cambia-sfondo)
    # matte a pixel della persona (solo composizione): DI DEFAULT nel cambia-sfondo (03/11: testa fedele 0.124 -> 0.007); si spegne con H3_EDIT_BG_MATTE=0 o bg_matte=0/false
    bg_matte = str(os.environ.get("H3_EDIT_BG_MATTE") or e.get("bg_matte", "1")).strip().lower() not in ("", "0", "false", "no", "off")
    # bg_prompt_mode (env H3_EDIT_BG_PROMPT): "empty_person" = il prompt dice che la zona della persona e vuota e verra ricomposta (il modello genera SOLO l'ambiente, niente
    # persona ne lip sync); si attiva da solo con bg_person_ref=erase, "standard" lo spegne. Nel riferimento senza persona il prompt di sempre («<soggetto> keeps their exact look»)
    # chiederebbe di ridisegnarla.
    # Valori: base (= standard, il prompt di sempre) | empty_person | no_performance | carryover | replace_like | align | carryover_align (ablazione 03/11: un fattore alla volta, vedi core/h3_edit_prompt).
    _bp = str(os.environ.get("H3_EDIT_BG_PROMPT") or e.get("bg_prompt_mode") or ("empty_person" if bg_person_ref == "erase" else "base")).strip().lower()
    bg_prompt = _bp if (invert and _bp in ("empty_person", "no_performance", "carryover", "replace_like", "align", "carryover_align")) else ""
    lipsync_off = str(os.environ.get("H3_EDIT_LIPSYNC") if os.environ.get("H3_EDIT_LIPSYNC") is not None else e.get("lipsync", "1")).strip().lower() in ("0", "false", "no", "off")
    if invert:
        ref_frames = M.blur_area(f01, 1.0 - pers, radius=max(6.0, 0.02 * th * float(os.environ.get("H3_EDIT_BG_BLUR") or e.get("bg_blur", 1.0))),
                                grey=float(os.environ.get("H3_EDIT_BG_GREY") or e.get("bg_grey", 0.35)))   # H3_EDIT_BG_BLUR = moltiplicatore del raggio (prova 03/11, default 1)
        if bg_person_ref == "mannequin":                               # la persona del riferimento diventa una figura grigia liscia (posizione e posa, niente volto/vestiti):
            ref_frames = M.grey_mannequins(ref_frames, pers, size, blur=0.02)   # il modello non ha una persona nitida da rifare (doppione spostato nel clip 1)
        elif bg_person_ref == "erase":                                 # NESSUNA figura nel riferimento: la zona della persona (dilatata) si riempie dallo sfondo sfocato intorno
            ref_frames = M.erase_area(ref_frames, pers, grow_px=max(4, int(round(8 * scale))))
    else:
        ref_frames = M.grey_mannequins(f01, pers, size, blur=0.02)

    # 3) piano dei clip
    if preroll:                                                        # maschere e regione rigenerata anche per i frame anteposti
        regen_pix = M.regen_region(pers, invert, person_grow, grow, expand=0.0 if invert else float(os.environ.get("H3_EDIT_REPLACE_EXPAND") or e.get("replace_expand", 0.0)))
    clip_frames = int(float(e.get("clip_s", 6.0)) * FPS)
    cuts = [a for a, _b in M.find_shots(f01[:, ::4, ::4], 40.0)[1:]]
    clips = M.enforce_min_length(M.clip_plan(N, clip_frames, int(e.get("overlap", 39)), cuts), N)   # nessun clip sotto i 5 s (minimo del pacchetto H3)
    W._log(f"piano: {len(clips)} clip (lunghezza, contesto) = {clips}")

    pic = e.get("picture") or ""
    base = dict(p)
    base.pop("h3_edit", None)
    base.pop("output", None)
    base["h3_mode"] = "ref2va"
    out_frames = torch.zeros(N, th, tw, 3)
    done = 0
    seed0 = int(p.get("seed") or 0)
    st = os.stat(src)
    from core.minimax_h3_spec import REF_VIDEO_MAX_SIDE                 # il riferimento di movimento cambia i clip: nella chiave dei checkpoint
    runkey = _sha(os.path.abspath(src), st.st_size, int(st.st_mtime), action, short, e.get("keep_text"), e.get("prompt_text"), e.get("picture_desc"),
                  pic, os.path.getmtime(pic) if pic and os.path.isfile(pic) else 0, seed0, p.get("h3_profile"), p.get("h3_turbo"), e.get("clip_s"),
                  e.get("overlap"), N, e.get("replace_expand"), os.environ.get("H3_EDIT_REPLACE_EXPAND"), os.environ.get("H3_EDIT_BG_GREY"), os.environ.get("H3_EDIT_BG_BLUR"), e.get("bg_feather_px"),
                  "maskalign_v2", *((("bgmask_v2", headroom, hold_frames),) if invert else ()),
                  *((("bgref", bg_person_ref),) if (invert and bg_person_ref != "sharp") else ()), *((("preroll", preroll),) if preroll else ()), *((("bgprompt", bg_prompt),) if bg_prompt else ()), *((("lipoff", 1),) if lipsync_off else ()), *((("bglike", 1),) if like_replace else ()), ("refmaxside", int(p.get("h3_ref_max_side", REF_VIDEO_MAX_SIDE))), *((("bgmaskorig", margin_px, fill_holes),) if original_mask else ()), *((("bgcover", hold_cover),) if hold_cover else ()),
                  *opts_key(p), *guide_key(e))   # solo il cambia-sfondo cambia chiave: i checkpoint di sostituisci restano validi
    tmp = Path(p.get("output") or W.REPO / "output").parent / "_h3edit_tmp" / runkey
    tmp.mkdir(parents=True, exist_ok=True)
    if e.get("text_info"):                                             # testo dell'utente tradotto/arricchito: l'originale e quello arrivato al modello restano nel log e in un file del job
        W._log("testo: originale %s -> inviato al modello %s (%s, %s s)" % (json.dumps(e["text_info"].get("original"), ensure_ascii=False),
               json.dumps(e["text_info"].get("sent"), ensure_ascii=False), e["text_info"].get("status"), e["text_info"].get("seconds")))
        (tmp / "text_job.json").write_text(json.dumps(e["text_info"], ensure_ascii=False, indent=1), encoding="utf-8")
    prev_tail = None                                                   # ultimi frame composti del clip prima (contesto del successivo)
    import numpy as _np
    for ci, (length, ctx) in enumerate(clips):
        start = done - ctx
        sl = slice(start, start + length)
        clip_src = src_t[sl].clone()
        if ctx and prev_tail is not None:                              # contesto = frame GIA generati (non la sorgente): continuita fra clip
            clip_src[:ctx] = (prev_tail[-ctx:] * 255).round().to(torch.uint8)
        cells, counts = M.cell_mask(regen_pix[sl], tw, th, invert=invert, feather_px=feather_px, width=tw, hold_cover=hold_cover or None)
        if ctx:                                                        # i frame di contesto non si rigenerano
            cells = cells.clone()
            cells[:, :, :_latents_of_frames(counts, ctx)] = 0
        # first_clip_two_pass (env H3_EDIT_FIRST_TWO_PASS, default spento): il clip 1 non ha contesto, quindi il modello si inventa il moto dal riferimento con un orologio suo
        # (persona ridisegnata sfasata di ~10 frame, dente di sega); il clip 2 prosegue un moto GIA fissato dai 39 frame di contesto e non ha il difetto. Passata A = clip 1 come
        # sempre; i suoi primi N frame (composti col matte: nuovo sfondo + persona vera) diventano il CONTESTO tenuto al 100 % della passata B, che rigenera il clip 1.
        two_pass = bool(two_n) and ci == 0 and ctx == 0 and invert and length >= two_n + 17
        ctx0, src0, cells0, comp_a = ctx, clip_src, cells, None
        for pass_i in range(2 if two_pass else 1):
            if pass_i == 1:
                clip_src = src0.clone()
                clip_src[:two_n] = (comp_a[:two_n] * 255).round().to(torch.uint8)
                cells = cells0.clone()
                cells[:, :, :_latents_of_frames(counts, two_n)] = 0
                ctx = two_n
                W._log(f"--- clip 1, passata B: i primi {two_n} frame della passata A come contesto tenuto ---")
            jd = tmp / (f"clip_{ci:02d}" if pass_i == 0 else f"clip_{ci:02d}b")
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
            if pass_i == 1:                                                # passata B: i primi N frame sono il contesto tenuto (pulito); entra nella chiave del clip
                meta["two_pass"] = int(two_n)
            if e.get("guide"):                                             # solo se scelta: senza, meta (e quindi la chiave del clip) e quella di sempre
                meta["guide"] = {"frame_idx": int(e.get("guide_frame_idx") or 0)}
                if invert and e.get("guide_mode") == "person":
                    # guida = SOLO la persona (matte morbido) su grigio neutro: il modello copia il TEMPO e la posa della persona fotogramma per fotogramma senza avere
                    # sotto gli occhi lo sfondo vecchio da ricopiare. Per i frame di contesto vale la persona gia generata (clip_src).
                    meta["guide"]["mode"] = "person"
                    mm = M.gaussian_blur(M.grow_mask(pers[sl].float(), 6)[:, None], 2.0)[:, 0].clamp(0, 1)[..., None]
                    _np.save(jd / "guide_src.npy", ((clip_src.float() / 255.0 * mm + 0.5 * (1 - mm)) * 255).round().to(torch.uint8).numpy())
            if bg_prompt:                                                  # solo se attivo: senza, meta (e la chiave del clip) restano quelli di sempre
                meta["bg_prompt"] = bg_prompt
            if lipsync_off:
                meta["lipsync"] = False
            if e.get("ref_clock_align"):                                   # variante sperimentale: SOLO se scelta (altrimenti meta e chiave restano quelle di sempre)
                meta["ref_clock_align"] = True
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
                (jd / "te_state.pt").unlink(missing_ok=True)               # la cache del testo serve solo a un riavvio dopo OOM: a clip riuscito si toglie
            gen_f = torch.from_numpy(_np.load(gen_p)).float() / 255.0
            _ensure(gen_f.shape[0] >= length, f"il clip {ci + 1} ha {gen_f.shape[0]} frame, attesi {length}")
            gen_f = gen_f[:length]
            if gen_f.shape[1:3] != (th, tw):
                gen_f = F.interpolate(gen_f.movedim(-1, 1), size=(th, tw), mode="bilinear", align_corners=False).movedim(1, -1)
            comp = composite(gen_f, clip_src.float() / 255.0, cells, counts)          # clip_src ha gia il contesto = frame del clip prima
            if invert and bg_matte:                                                    # la persona e quella ORIGINALE a pixel (bordo morbido), non quella ridisegnata nelle celle di bordo
                comp = M.person_matte(comp, clip_src.float() / 255.0, pers[sl])
            if pass_i == 1:                                                # i primi N frame restano ESATTAMENTE quelli composti dalla passata A (niente doppia codifica VAE)
                comp = comp.clone()
                comp[:two_n] = comp_a[:two_n]
                out_frames[start:start + length] = comp
            else:
                out_frames[start + ctx0:start + length] = comp[ctx0:]
                comp_a = comp.clone() if two_pass else None
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
        audio = wav[:, int(preroll / FPS * _SRC_AUDIO_SR): int(done / FPS * _SRC_AUDIO_SR)]
    encode_video((out_frames[preroll:done].clamp(0, 1) * 255).round().to(torch.uint8), fps=FPS, output_path=out,
                 audio=audio, audio_sample_rate=_SRC_AUDIO_SR if audio is not None else None)
    W._log(f"edit finito in {time.time() - t_all:.0f}s -> {out}")
    return out


def _want_key(job_dir: Path) -> str:
    """Chiave del clip che run_edit scrive prima di lanciare il figlio ('' se manca: allora la cache del testo non si usa)."""
    path = job_dir / "want_key.txt"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


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
    vae = W.load_vae(Wd, encoder_only=True)                            # codifica solo il clip sorgente: serve l'encoder, non il decoder
    lat = encode_source_latent(vae, clip_src, _Const)
    guide_lat = lat
    if (m.get("guide") or {}).get("mode") == "person":                 # guida = persona su grigio (guide_src.npy), non il clip intero
        guide_lat = encode_source_latent(vae, torch.from_numpy(_np.load(jd / "guide_src.npy")), _Const)
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
    lip = bool(m["has_audio"]) and m.get("lipsync", True)
    if m["action"] == "change_background" and m.get("bg_prompt") == "empty_person":
        prompt = PR.background_empty_person_prompt(m["prompt_text"], length, ctx, has_picture=bool(m["picture"]), picture_desc=m["picture_desc"], shots=m["shots"])
        lip = False
    elif m["action"] == "change_background":
        bgv = m.get("bg_prompt") or "base"
        prompt = PR.background_prompt(m["prompt_text"], m["keep_text"], length, (ctx or 39) if "carryover" in bgv else ctx, has_picture=bool(m["picture"]),
                                      picture_desc=m["picture_desc"], shots=m["shots"], lip_sync=lip, variant=bgv)
    else:
        prompt = PR.replace_prompt(m["keep_text"], length, ctx, picture_desc=m["picture_desc"], shots=m["shots"], lip_sync=lip)
    pc = dict(base)
    pc.update(prompt=prompt, references=refs, seed=m["seed"], num_frames=length, duration_s=length / FPS, width=tw, height=th,
              generate_audio=False, _h3_edit={"noise": noise, "audio_noise": audio_noise, "video_hold": video_hold, "audio_hold": audio_hold,
                                              "frames": length, "canvas": (th, tw),
                                              "text_cache": {"path": str(jd / "te_state.pt"), "key": _want_key(jd)}})
    if m.get("guide"):
        # Guida allineata: il clip sorgente (i suoi latenti, GIA calcolati per le righe tenute) entra anche come blocco di condizione sulla griglia del
        # video generato (workers/h3_guide.py). Si aggiunge al riferimento nativo <Video 1>; le LoRA di guida (VFX Edit, Style Transfer...) la vogliono.
        pc["_h3_guide_latents"] = guide_lat
        pc["h3_guide"] = {"frame_idx": int(m["guide"].get("frame_idx") or 0)}
    if m.get("ref_clock_align"):
        pc["h3_ref_clock_align"] = True
    W._log(f"clip {ci + 1}/{m['clips']}: {length} frame, contesto {ctx}, tela {tw}x{th}" + (" · guida allineata" + (" (solo persona)" if (m.get("guide") or {}).get("mode") == "person" else "") if m.get("guide") else "")
           + (" · orologio del riferimento allineato" if m.get("ref_clock_align") else ""))
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
