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
        from ltx_pipelines.iclora_utils import downsample_mask_video_to_latent
        dev = latent_state.denoise_mask.device
        m5 = self.masks.to(dev).float()[None, None]   # (1,1,T,H,W) pixel-space
        latent_mask = downsample_mask_video_to_latent(m5, latent_tools.target_shape)  # (1, f*h*w)
        state = latent_state.clone()
        flat = (latent_mask > 0.5).to(state.denoise_mask.dtype).flatten()
        total = state.denoise_mask.numel()
        if flat.numel() < total:
            # i token di conditioning (IC-LoRA video) sono appesi DOPO i token video → denoise_mask=0 lì
            full = torch.zeros(total, dtype=state.denoise_mask.dtype, device=state.denoise_mask.device)
            full[:flat.numel()] = flat
            flat = full
        state.denoise_mask.copy_(flat.reshape(state.denoise_mask.shape))
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
