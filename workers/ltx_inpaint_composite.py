"""Composizione dell'inpaint video in pixel: fuori dalla maschera = sorgente originale (solo torch + numpy + ffmpeg, nessun import ltx)."""
from __future__ import annotations

import torch


def _cover_crop(x: "torch.Tensor", h: int, w: int) -> "torch.Tensor":
    """[F,H,W,3] float -> [F,h,w,3]: resize 'cover' + ritaglio centrale (come `resize_and_center_crop` della pipeline LTX)."""
    _f, hh, ww, _c = x.shape
    if (hh, ww) == (h, w):
        return x
    k = max(h / hh, w / ww)
    nh, nw = max(h, round(hh * k)), max(w, round(ww * k))
    y = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(nh, nw), mode="bilinear",
                                        align_corners=False, antialias=True)
    t, l = (nh - h) // 2, (nw - w) // 2
    return y[:, :, t:t + h, l:l + w].permute(0, 2, 3, 1)


def composite_outside_mask(chunks, src_video: str, masks: "torch.Tensor"):
    """Rimette lo SFONDO ORIGINALE fuori dalla maschera: generatore di chunk [F,h,w,3] float 0..1 (uscita del decoder LTX) in cui
    ogni pixel e' `gen*alpha + sorgente*(1-alpha)`, con alpha = `masks` [T,H,W] (maschera SAM2 gia' dilatata e sfumata: il bordo
    morbido e' la sfumatura). Il sorgente e' portato al quadro come fa la pipeline con la guida (resize + ritaglio centrale), alla
    risoluzione dei chunk (piena con lo stage 2, meta' senza). Se sorgente o maschere finiscono prima, i frame residui restano generati."""
    import numpy as np
    from core import video_ic_prep as prep
    info = prep.probe(src_video)
    rd = prep._read_frames(src_video, info.width, info.height)
    idx = 0
    try:
        for ch in chunks:
            n, h, w, _c = ch.shape
            src = [fr for fr in (next(rd, None) for _ in range(n)) if fr is not None]
            a = masks[idx:idx + n]
            idx += n
            if len(src) < n or a.shape[0] < n:
                yield ch
                continue
            s = torch.from_numpy(np.stack(src)).to(ch.device, torch.float32) / 255.0
            s = _cover_crop(s, h, w)
            a = a.to(ch.device, torch.float32)
            if tuple(a.shape[-2:]) != (h, w):
                a = torch.nn.functional.interpolate(a[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
            a = a.clamp(0, 1)[..., None]
            yield (ch.to(torch.float32) * a + s * (1.0 - a)).to(ch.dtype)
    finally:
        rd.close()
