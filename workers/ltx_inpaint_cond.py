"""#59 SpatialRegionMask — conditioner di inpaint SPAZIALE per LTX-2 (la maschera SAM2 per-frame
imposta denoise_mask=1 dentro l'oggetto, 0 fuori → rigenera SOLO l'area, preserva il resto).

Analogo a ltx_core TemporalRegionMask (che maschera nel TEMPO) ma maschera per-PIXEL usando una
sequenza di maschere [T,H,W] (es. da SAM2 tracking). Lavora in token space: per ogni patch usa il
centro (frame,y,x) in pixel (get_pixel_coords) e campiona la maschera lì.
"""
from dataclasses import dataclass

import torch
from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import LatentTools, SpatioTemporalScaleFactors
from ltx_core.types import LatentState, VideoLatentShape


@dataclass(frozen=True)
class SpatialRegionMask(ConditioningItem):
    """denoise_mask = 1 dove la maschera (per-frame) è attiva, 0 altrove.
    `masks`: tensore float [T,H,W] in [0,1] (pixel space, H/W del video di output)."""

    masks: torch.Tensor   # [T, H, W]

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        if not isinstance(latent_tools.target_shape, VideoLatentShape):
            return latent_state
        dev = latent_state.denoise_mask.device
        coords = latent_tools.patchifier.get_patch_grid_bounds(latent_tools.target_shape, device=dev)
        sf = getattr(latent_tools, "scale_factors", SpatioTemporalScaleFactors.default())
        px = get_pixel_coords(coords, sf, causal_fix=getattr(latent_tools, "causal_fix", True))  # (B,3,N,2)
        # centro pixel di ogni patch su ciascun asse
        t_c = px[:, 0].float().mean(dim=-1)   # (B,N) frame
        y_c = px[:, 1].float().mean(dim=-1)   # (B,N) height
        x_c = px[:, 2].float().mean(dim=-1)   # (B,N) width
        m = self.masks.to(dev).float()
        T, H, W = m.shape
        ti = t_c.round().long().clamp(0, T - 1)
        yi = y_c.round().long().clamp(0, H - 1)
        xi = x_c.round().long().clamp(0, W - 1)
        sampled = m[ti, yi, xi]               # (B,N) in [0,1]
        state = latent_state.clone()
        mask_val = (sampled > 0.5).to(state.denoise_mask.dtype)
        if state.denoise_mask.dim() == 3 and mask_val.dim() == 2:
            mask_val = mask_val.unsqueeze(-1)
        state.denoise_mask.copy_(mask_val.expand_as(state.denoise_mask))
        return state


def load_mask_tensor(mask_pngs: list[str], height: int, width: int) -> torch.Tensor:
    """Carica le maschere PNG (SAM2) in un tensore [T,H,W] in [0,1], ridimensionate a (height,width)."""
    import numpy as np
    from PIL import Image
    arr = []
    for p in mask_pngs:
        im = Image.open(p).convert("L").resize((width, height))
        arr.append((np.asarray(im, np.float32) / 255.0))
    return torch.from_numpy(np.stack(arr))    # [T,H,W]
