"""MiniMax H3 — GUIDA DI LATENTI ALLINEATA, nativa (diffusers modular, senza ComfyUI, senza codice di terzi copiato).

Come funziona (verificato su: README/workflow di Alissonerdx/Minimax-H3-ComfyUI in sola lettura; layout di diffusers `minimax_h3`; descrizione del
nodo `MiniMaxH3AddGuide` di ComfyUI — GPL-3.0, solo letto, NIENTE copiato; il nostro precedente `h3_upsample_core.build_tile_layout`, che usa la stessa idea):
 · i latenti VAE del video sorgente (stessa ricetta dei riferimenti: pixel ImageNet, posteriore campionato con seme 42, fp16, normalizzati) entrano nella
   sequenza come righe «condizione» DAVANTI al video generato, [testo | riferimenti | GUIDA | audio | video], tutte le righe condizione a t = 0,999;
 · coordinate (t, h, w) = quelle del video GENERATO: h, w sono la griglia spaziale del bersaglio (non quella di un riferimento nativo), il tempo e
   `origine + 5/3 * frame_idx` + la scansione 5/3·(1,4,4,4,4) del bersaglio (l'origine e l'inizio dell'orologio del video, dopo il testo e i riferimenti);
   a frame_idx = 0 la guida cade ESATTAMENTE sulle righe del video generato. Un riferimento nativo `<Video 1>` invece ha il SUO orologio che avanza;
 · niente maschera di attenzione (attenzione densa su tutta la sequenza); le righe guida non producono output (il loop scrive solo le righe generate);
 · timestep: max(t, 0,999) come ogni condizione (`keyframe_noise_aug`), e il latente guida e miscelato col rumore a quel livello (quasi pulito);
 · non c'e «forza»: il nodo non ha ingressi di forza, l'intensita e la forza della LoRA. Audio: il nodo ha un ingresso audio opzionale (nel workflow VFX collegato
   a un audio vuoto, in quello LMS non collegato): qui NON implementato (vedi report) — la guida e solo video.
Senza `guide_latents` i blocchi che seguono sono quelli di diffusers, non toccati: sequenza, chiave dei checkpoint e risultato identici.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402

from core import h3_guide as G                                        # noqa: E402

PATCH = (1, 2, 2)


def _bd():
    from diffusers.modular_pipelines.minimax_h3 import before_denoise as B
    return B


# ── layout: inserimento delle righe guida (puro, testabile senza pipeline) ────────────────────────────────
def insert_guide_rows(layout: dict, steps: int, lat_h: int, lat_w: int, frame_idx: int = 0, patch: tuple = PATCH) -> dict:
    """Inserisce la guida (`steps` passi di latente alla griglia lat_h x lat_w del bersaglio) in un layout di diffusers.

    `layout`: position_ids [S,3] f64, token_tags [S], video_indices, audio_indices, text_indices, num_condition_video_rows, num_condition_audio_rows
    (le chiavi dei blocchi di layout di diffusers). Le righe guida stanno dopo i riferimenti e prima dell'audio generato: sono condizioni, quindi vanno in
    coda al blocco di righe condizione (`video_indices[:ncv]`) e `num_condition_video_rows` cresce di quante sono. Ritorna un NUOVO layout + `n_guide_rows`."""
    B = _bd()
    pos, tags = layout["position_ids"], layout["token_tags"]
    vi, ai, ti = layout["video_indices"], layout["audio_indices"], layout["text_indices"]
    ncv, nca = int(layout["num_condition_video_rows"]), int(layout["num_condition_audio_rows"])
    _, ph, pw = patch
    rpf = (lat_h // ph) * (lat_w // pw)
    n = int(steps)
    g_rows = n * rpf
    if g_rows <= 0:
        raise ValueError("guida vuota")
    first_target_audio = int(ai[nca])                                  # le righe guida vanno qui: dopo testo e riferimenti, prima dell'audio generato
    first_target_video = int(vi[ncv])
    if ncv and int(vi[:ncv].max()) >= first_target_audio:
        raise RuntimeError("layout inatteso: righe-riferimento video dopo l'audio generato")
    if nca and int(ai[:nca].max()) >= first_target_audio:
        raise RuntimeError("layout inatteso: righe-riferimento audio dopo l'audio generato")
    origin = float(pos[first_target_video, 0])                         # inizio dell'orologio del video generato
    anchor = origin + B._ROPE_FRAME_RESCALE * float(frame_idx)
    frame_grid, _ = B._frame_position_grid(lat_h, lat_w, ph, pw)
    g_pos = torch.empty(g_rows, 3, dtype=pos.dtype)
    g_pos[:, 0] = B._temporal_position_grid(n, anchor).to(pos.dtype).repeat_interleave(rpf)
    g_pos[:, 1:] = frame_grid.to(pos.dtype).repeat(n, 1)
    new_pos = torch.cat([pos[:first_target_audio], g_pos, pos[first_target_audio:]])
    video_tag = tags[first_target_video]
    new_tags = torch.cat([tags[:first_target_audio], torch.full((g_rows,), int(video_tag), dtype=tags.dtype), tags[first_target_audio:]])
    g_idx = torch.arange(first_target_audio, first_target_audio + g_rows, dtype=vi.dtype)
    new_vi = torch.cat([vi[:ncv], g_idx, vi[ncv:] + g_rows])
    new_ai = torch.cat([ai[:nca], ai[nca:] + g_rows])
    return {"position_ids": new_pos, "token_tags": new_tags, "video_indices": new_vi, "audio_indices": new_ai, "text_indices": ti,
            "num_condition_video_rows": ncv + g_rows, "num_condition_audio_rows": nca, "n_guide_rows": g_rows}


def align_ref_clock(layout: dict, refs: list, cond_shapes: list, audio_rows: list, num_latent_frames: int, patch: tuple = PATCH) -> dict:
    """«Orologio allineato» (idea TARA senza LoRA e senza penalita d'attenzione): le righe del riferimento VIDEO nativo restano al loro posto nella sequenza, ma il
    loro tempo rotary non e piu quello del SUO orologio (che parte dopo testo e immagini e avanza) bensi quello del video generato: t_rif(passo k) = t_bersaglio(passo k).
    h e w restano quelli del riferimento. Vale anche per la traccia audio di un riferimento video (tempo = quello dell'audio generato). Immagini e audio autonomi non si toccano.

    refs: [(kind, has_audio)] nell'ordine dei riferimenti; cond_shapes: (T, h, w) di ogni riferimento immagine/video, nello stesso ordine; audio_rows: righe di ogni
    riferimento con audio. Ritorna un NUOVO layout (stessi indici, cambia solo position_ids)."""
    B = _bd()
    _, ph, pw = patch
    pos = layout["position_ids"].clone()
    vi, ai = layout["video_indices"], layout["audio_indices"]
    ncv, nca = int(layout["num_condition_video_rows"]), int(layout["num_condition_audio_rows"])
    origin = float(pos[int(vi[ncv]), 0])
    tgrid = B._temporal_position_grid(int(num_latent_frames), origin).to(pos.dtype)
    v_cur = a_cur = c_i = a_i = 0
    for kind, has_audio in refs:
        if kind in ("image", "video"):
            t_r, h_r, w_r = (int(x) for x in cond_shapes[c_i])
            c_i += 1
            rows = t_r * (h_r // ph) * (w_r // pw)
            idx = vi[v_cur:v_cur + rows]
            v_cur += rows
            if kind == "video":
                if t_r > int(num_latent_frames):
                    raise ValueError(f"riferimento video di {t_r} passi di latente piu lungo del video generato ({num_latent_frames}): l'orologio non si puo allineare")
                pos[idx, 0] = tgrid[:t_r].repeat_interleave(rows // t_r)
        if kind == "audio" or (kind == "video" and has_audio):
            n = int(audio_rows[a_i])
            a_i += 1
            idx = ai[a_cur:a_cur + n]
            a_cur += n
            if kind == "video":                                       # canale-maggiore: n/2 latenti per canale, tempo = origine + indice (come l'audio generato)
                pos[idx, 0] = origin + torch.arange(n // 2, dtype=pos.dtype).repeat(2)
    if v_cur != ncv or a_cur != nca:
        raise RuntimeError(f"orologio allineato: righe riferimento {v_cur}/{ncv} video, {a_cur}/{nca} audio non tornano")
    return {**layout, "position_ids": pos}


# ── blocchi della pipeline (sostituiscono i due del denoise Ref2VA; senza guida si comportano come gli originali) ──────────────
def _make_blocks():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (MiniMaxH3PrepareConditionLatentsStep,
                                                                       MiniMaxH3Ref2VAPrepareLayoutStep)
    from diffusers.modular_pipelines.modular_pipeline_utils import InputParam

    def _guide_inputs():
        return [InputParam(name="guide_latents", type_hint=torch.Tensor, default=None,
                           description="Latenti VAE normalizzati del video guida, (1, C, passi, h, w) alla tela del bersaglio; None = nessuna guida."),
                InputParam(name="guide_frame_idx", type_hint=int, default=0,
                           description="Fotogramma (pixel) del video generato da cui parte la guida; deve cadere all'inizio di un passo di latente."),
                InputParam(name="ref_clock_align", type_hint=bool, default=False,
                           description="Porta il tempo rotary del riferimento VIDEO nativo sull'orologio del video generato (variante sperimentale, default spenta).")]

    class GuidedRef2VAPrepareLayoutStep(MiniMaxH3Ref2VAPrepareLayoutStep):
        @property
        def inputs(self):
            return list(super().inputs) + _guide_inputs()

        @property
        def description(self):
            return super().description + " Con `guide_latents` aggiunge il blocco della guida allineata (workers/h3_guide.py)."

        @torch.no_grad()
        def __call__(self, components, state):
            components, state = super().__call__(components, state)
            patch = tuple(components.patch_size)
            if state.get("ref_clock_align"):
                keys = ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices", "num_condition_video_rows", "num_condition_audio_rows")
                lay = {k: state.get(k) for k in keys}
                dev = lay["position_ids"].device
                lay = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in lay.items()}
                refs = [(r.kind, bool(getattr(r, "has_audio", False))) for r in state.get("normalized_references") or []]
                new = align_ref_clock(lay, refs, [tuple(c.shape[2:5]) for c in state.get("condition_latents") or []],
                                      [a.shape[0] for a in state.get("audio_condition_latents") or []], int(state.get("num_latent_frames")), patch)
                state.set("position_ids", new["position_ids"].to(dev))
            g = state.get("guide_latents")
            if g is None:
                return components, state
            lay = {k: state.get(k) for k in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices",
                                             "num_condition_video_rows", "num_condition_audio_rows")}
            dev = lay["position_ids"].device
            cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in lay.items()}
            if tuple(g.shape[3:5]) != (state.get("latent_height"), state.get("latent_width")):
                raise ValueError(f"La guida e codificata a {tuple(g.shape[3:5])} (alt, larg di latente) ma la tela e {state.get('latent_height')}x{state.get('latent_width')}: "
                                 "la guida deve avere la STESSA griglia spaziale del video generato.")
            fi = int(state.get("guide_frame_idx") or 0)
            if fi not in G.latent_start_frames(int(state.get("num_frames"))):
                raise ValueError(f"guide_frame_idx {fi} non cade all'inizio di un passo di latente")
            if fi and int(sum(G.FRAMES_PER_LATENT[k % 5] for k in range(int(g.shape[2])))) + fi > int(state.get("num_frames")):
                raise ValueError("la guida esce dal video generato")
            new = insert_guide_rows(cpu, int(g.shape[2]), int(g.shape[3]), int(g.shape[4]), fi, patch)
            for k in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
                state.set(k, new[k].to(dev))
            state.set("num_condition_video_rows", new["num_condition_video_rows"])
            return components, state

    class GuidedPrepareConditionLatentsStep(MiniMaxH3PrepareConditionLatentsStep):
        @property
        def inputs(self):
            return list(super().inputs) + _guide_inputs()[:1]

        @torch.no_grad()
        def __call__(self, components, state):
            g = state.get("guide_latents")
            if g is None:
                return super().__call__(components, state)
            orig = state.get("condition_latents")
            state.set("condition_latents", list(orig) + [g])            # in coda ai riferimenti, come l'ordine delle righe del layout
            try:
                return super().__call__(components, state)
            finally:
                state.set("condition_latents", orig)

    return GuidedRef2VAPrepareLayoutStep, GuidedPrepareConditionLatentsStep


def install(blocks) -> None:
    """Sostituisce, nel workflow ref2va già costruito (`MiniMaxH3Blocks().get_workflow("ref2va")`), i due blocchi del denoise che conoscono la guida.
    Da chiamare PRIMA di `_stage(...)`. Idempotente."""
    sb = blocks.sub_blocks
    L, C = _make_blocks()
    for key, cls in (("denoise.prepare_layout", L), ("denoise.prepare_condition_latents", C)):
        if key not in sb:
            raise RuntimeError(f"workflow ref2va senza «{key}» (diffusers cambiato?)")
        if type(sb[key]).__name__ != cls.__name__:
            sb[key] = cls()


# ── ingresso: video guida -> latenti ────────────────────────────────────────────────────────────────
def probe(path: str) -> dict:
    """{"w", "h", "n24" (fotogrammi a 24 fps), "dur", "has_audio"} del video guida, senza decodificarlo tutto."""
    import av
    with av.open(path) as c:
        vs = c.streams.video[0]
        w, h = vs.codec_context.width, vs.codec_context.height
        dur = float(c.duration / 1_000_000) if c.duration else (float(vs.duration * vs.time_base) if vs.duration else 0.0)
        if not dur:
            dur = sum(1 for _ in c.decode(vs)) / float(vs.average_rate or FPS_DEFAULT)
        return {"w": int(w), "h": int(h), "dur": dur, "n24": max(1, int(dur * G.FPS + 1e-6)), "has_audio": bool(c.streams.audio)}


FPS_DEFAULT = 24


def read_guide_frames(path: str, width: int, height: int, n: int):
    """Primi `n` fotogrammi a 24 fps alla tela (width x height): uint8 [n, height, width, 3]. Stesso ricampionamento dell'edit (il fotogramma piu vicino nel tempo)."""
    import h3_edit_run as ER
    frames, _dur, _idx, _n = ER.read_frames(path, width, height, n)
    if frames.shape[0] < n:                                            # video piu corto del previsto: si ripete l'ultimo
        pad = frames[-1:].repeat(n - frames.shape[0], axis=0)
        import numpy as np
        frames = np.concatenate([frames, pad])
    return frames[:n]


def encode_guide(vae, frames_u8, *, device="cuda", clips_per_batch: int = 1) -> torch.Tensor:
    """uint8 [n, H, W, 3] (n = 17k+5) -> latente normalizzato (1, 24, 5k+2, H/16, W/16) float32 CPU: la ricetta del condizionamento di H3
    (workers/h3_upsample_core.encode_frames, identica a `encode_vae_condition` ma a finestre: la guida e lunga quanto il video)."""
    import h3_upsample_core as C
    cfg = vae.config
    x = torch.as_tensor(frames_u8)

    def blocks(step=17 * 4):
        for i in range(0, x.shape[0], step):
            yield x[i:i + step].float() / 255.0
    return C.encode_frames(vae, blocks(), int(x.shape[0]), pixel_mean=(0.485, 0.456, 0.406), pixel_std=(0.229, 0.224, 0.225),
                           latents_mean=cfg.latents_mean, latents_std=cfg.latents_std, device=device, clips_per_batch=clips_per_batch)


def trim_audio(wav: torch.Tensor | None, frames: int, sr: int = 32000) -> torch.Tensor | None:
    """Audio sorgente stereo tagliato/riempito alla durata dei fotogrammi generati (per rimontarlo sul risultato)."""
    if wav is None:
        return None
    import h3_edit_run as ER
    wav = ER.to_stereo(wav)
    need = int(round(frames / G.FPS * sr))
    if wav.shape[1] >= need:
        return wav[:, :need]
    return torch.cat([wav, torch.zeros(wav.shape[0], need - wav.shape[1], dtype=wav.dtype)], 1)
