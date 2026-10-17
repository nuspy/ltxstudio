"""IC-LoRA HDR (fase 1c): esecuzione di `ltx_pipelines.hdr_ic_lora.HDRICLoraPipeline` e scrittura dell'uscita.

Chiamato da `workers/ltx2_infer.py` (ptype "hdr") DOPO la costruzione della pipeline (checkpoint dell'utente, mai
sostituito; col dev la distilled-LoRA 0.5 su stage 1 e 2: `_dev_hdr_dlora`). Qui:
- segmentazione dei clip lunghi (`p["hdr_seg_frames"]`, overlap `p["hdr_overlap"]`) con i planner di
  `core.video_ic_prep`; ogni segmento = una chiamata alla pipeline (resta in cache: nessun reload fra segmenti);
- il risultato di ogni segmento e' un tensore float lineare [f,h,w,3] (decode LogC3 inverso): va in CPU, i frame
  sono cuciti in luce lineare (`core.hdr_export.Stitcher`) e scritti in streaming da `core.hdr_export.HdrWriter`
  (anteprima SDR tonemappata + HEVC 10 bit PQ + EXR a scelta);
- audio del sorgente rimesso su anteprima e HEVC (copia del video); sidecar JSON con path e statistiche.

Params (oltre ai soliti): control_video (SDR Rec.709), num_frames (8k+1), width/height (dimensioni FINALI: la
pipeline allinea a 64 con reflect-pad e ritaglia), fps, seed, output (= anteprima SDR), control_strength (1.0),
hdr_format (pq|exr|both), hdr_exposure_ev, hdr_exr_half, hdr_white_nits, hdr_hq (high_quality_hdr: 2x frame interni,
~2x tempo), hdr_seg_frames, hdr_overlap, hdr_audio_from.
"""
from __future__ import annotations

import gc
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import hdr_export as HX                      # noqa: E402
from core import video_ic_prep as prep                  # noqa: E402


def _seg_dir(output: str) -> str:
    """Cartella temporanea dei segmenti SDR (sotto la temp di sistema, non in output; nessuna cancellazione)."""
    import hashlib
    import tempfile
    base = os.path.splitext(output)[0]
    job = f"{os.path.basename(base)}_{hashlib.sha1(os.path.abspath(base).encode('utf-8')).hexdigest()[:8]}"
    d = os.path.join(tempfile.gettempdir(), "ltxstudio_hdrwork", job)
    os.makedirs(d, exist_ok=True)
    return d


def plan(total: int, seg_frames: int, overlap: int):
    """(segmenti, overlap reali) per `total` frame; un solo segmento se entra nel tetto."""
    segs = prep.plan_segments(total, seg_frames, overlap)
    return segs, prep.overlaps_of(segs)


def run(pipe, p: dict, tiling_config, log=print) -> dict:
    """Esegue il job HDR. Ritorna il dict del writer (+ `segments`); scrive anche il sidecar JSON."""
    import torch

    src = p["control_video"]
    out = p["output"]
    fps = float(p.get("fps", 25.0))
    width, height = int(p["width"]), int(p["height"])
    total = int(p["num_frames"])
    seed = int(p.get("seed", 42))
    strength = float(p.get("control_strength", 1.0))
    fmt = str(p.get("hdr_format") or HX.DEFAULT_FORMAT)
    if fmt not in HX.FORMATS:
        raise SystemExit(f"[hdr] formato sconosciuto {fmt!r} (validi: {HX.FORMATS})")
    pq_path, exr_dir = HX.default_paths(out)
    want_pq = fmt in ("pq", "both")
    want_exr = fmt in ("exr", "both")
    hq = bool(p.get("hdr_hq", False))
    seg_cap = int(p.get("hdr_seg_frames") or 0) or total
    ov = int(p.get("hdr_overlap") or 0) or prep.DEFAULT_OVERLAP
    segs, ovs = plan(total, seg_cap, ov)
    if hq:                                                  # 2N-1 frame interni: N deve essere 8k+1 (lo e', snap)
        for sg in segs:
            if (sg.count - 1) % 4:
                raise SystemExit(f"[hdr] high_quality richiede segmenti 8k+1, trovato {sg.count}")
    log(f"[hdr] {total} frame @ {fps:g}fps {width}x{height} formato={fmt} hq={hq} esposizione anteprima "
        f"{HX.clamp_exposure(p.get('hdr_exposure_ev', HX.DEFAULT_EXPOSURE_EV)):+.1f} EV; "
        f"{len(segs)} segmento/i da {segs[0].count} frame, overlap {ovs}")

    wr = HX.HdrWriter(out, pq=pq_path if want_pq else "", exr_dir=exr_dir if want_exr else "",
                      width=width, height=height, fps=fps,
                      exposure_ev=float(p.get("hdr_exposure_ev", HX.DEFAULT_EXPOSURE_EV)),
                      exr_half=bool(p.get("hdr_exr_half", True)),
                      white_nits=float(p.get("hdr_white_nits", HX.REF_WHITE_NITS)))
    work = _seg_dir(out) if len(segs) > 1 else ""
    try:
        st = HX.Stitcher(wr.write, [(sg.start, sg.count) for sg in segs])
        for sg in segs:
            if len(segs) > 1:
                guide = os.path.join(work, f"hdrseg_{sg.index:03d}.mp4")
                prep.extract_segment(src, guide, sg.start, sg.count)
            else:
                guide = src
            log(f"[hdr] segmento {sg.index + 1}/{len(segs)}: frame {sg.start}-{sg.end - 1} ({sg.count}f)")
            with torch.no_grad():
                hdr = pipe(seed=seed, height=height, width=width, num_frames=sg.count, frame_rate=fps,
                           video_conditioning=[(guide, strength)], tiling_config=tiling_config,
                           high_quality_hdr=hq)
            if not torch.is_tensor(hdr) or hdr.ndim != 4 or hdr.shape[-1] != 3:
                raise RuntimeError(f"uscita HDR inattesa: {type(hdr).__name__} {getattr(hdr, 'shape', None)}")
            if bool(torch.isnan(hdr).any()) or bool(torch.isinf(hdr).any()):
                raise RuntimeError("Frame NaN/inf dal denoise HDR: pesi corrotti o incompatibili. "
                                   "Verifica/riscarica il checkpoint e la LoRA HDR.")
            frames = hdr.detach().float().cpu().numpy()
            del hdr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st.add(frames)
            del frames
        st.finish()
        info = wr.close()
    except BaseException:
        wr.abort()
        raise
    if st.written != total:
        log(f"[hdr] ATTENZIONE frame scritti {st.written} != richiesti {total}")
    audio_src = p.get("hdr_audio_from") or ""
    if audio_src:
        ok_p = HX.mux_audio(out, audio_src)
        ok_h = HX.mux_audio(pq_path, audio_src) if want_pq else False
        log(f"[hdr] audio originale: anteprima={'si' if ok_p else 'no'} hevc={'si' if ok_h else 'no'}")
    info.update(segments=len(segs), format=fmt, hq=hq, audio=bool(audio_src))
    HX.write_sidecar(out, info)
    log(f"[hdr] anteprima SDR: {out}")
    if want_pq:
        log(f"[hdr] HEVC 10 bit PQ/BT.2020: {pq_path}")
    if want_exr:
        log(f"[hdr] EXR ({'half' if info['exr_half'] else 'float32'}): {exr_dir} ({info['frames']} frame)")
    log(f"[hdr] picco lineare {info['peak_linear']} (1.0 = bianco diffuso), media {info['mean_linear']}")
    return info
