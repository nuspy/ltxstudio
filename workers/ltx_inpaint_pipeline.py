"""#59 InpaintPipeline — latent-inpaint VIDEO vero (sottoclasse di ICLoraPipeline).

Override di __call__ per l'inpaint mascherato: encode del video ORIGINALE → clean latent (initial_latent
preservato fuori dalla maschera) + SpatialRegionMask (denoise_mask=1 nell'area SAM2) → rigenera SOLO
l'oggetto col prompt, preservando il resto. Riusa i componenti del padre (prompt_encoder,
image_conditioner/enc, stage_1, video_decoder). NON modifica il package ufficiale.

NB: questo è il vero SetLatentNoiseMask nativo. La modalità skip_stage_2 (low-res) serve a iterare veloce.
"""
from __future__ import annotations

import torch
from ltx_core.types import VideoPixelShape
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.types import ModalitySpec

from workers.ltx_inpaint_cond import SpatialRegionMask


class InpaintPipeline(ICLoraPipeline):
    @torch.inference_mode()
    def inpaint(self, *, prompt: str, seed: int, height: int, width: int, num_frames: int,
                frame_rate: float, original_frames: list[str], masks: torch.Tensor,
                video_conditioning: list[tuple[str, float]], strength: float = 1.0,
                stage_1_sigmas=None, skip_stage_2: bool = True, num_steps: int = 30,
                negative_prompt: str = "worst quality, blurry", cfg_scale: float = 4.0):
        """original_frames: PNG del video originale. masks: [T,H,W] in [0,1] (SAM2 tracking).
        Usa CFG (GuidedDenoiser, cfg_scale>1) per FORZARE il prompt nell'area mascherata: senza guidance
        il modello dev rigenera un blob/l'oggetto originale invece di seguire il prompt."""
        import logging

        import numpy as np
        from ltx_core.components.diffusion_steps import EulerCfgPpDiffusionStep
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_pipelines.utils.denoisers import GuidedDenoiser
        from ltx_pipelines.utils.samplers import euler_cfg_pp_denoising_loop
        from PIL import Image

        if stage_1_sigmas is None:
            # sigmas PIENI (num_steps): da rumore servono molti step per costruire l'oggetto nuovo.
            from ltx_core.components.schedulers import LTX2Scheduler
            stage_1_sigmas = LTX2Scheduler().execute(steps=num_steps)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        ctx_p, ctx_n = self.prompt_encoder([prompt, negative_prompt])
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        s1 = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate)

        # frame originali → tensore (1,3,F,h,w) in [-1,1] a risoluzione stage_1
        def _vid_tensor(h, w):
            fr = []
            for p in original_frames[:num_frames]:
                im = Image.open(p).convert("RGB").resize((w, h))
                fr.append(np.asarray(im, np.float32) / 127.5 - 1.0)
            t = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,F,h,w)
            return t.to(self.device, self.dtype)

        # encode video orig → clean latent + costruisci conditioning (incl. SpatialRegionMask) usando enc
        masks_s1 = torch.nn.functional.interpolate(masks.unsqueeze(1), size=(s1.height, s1.width),
                                                   mode="nearest").squeeze(1)  # [T,h,w]

        def _build(enc):
            clean = enc(_vid_tensor(s1.height, s1.width))
            conds = self._create_conditionings(images=[], video_conditioning=video_conditioning,
                                               height=s1.height, width=s1.width, video_encoder=enc,
                                               num_frames=num_frames)
            return clean, conds

        clean_latent, stage_1_conditionings = self.image_conditioner(_build)
        stage_1_conditionings = list(stage_1_conditionings) + [SpatialRegionMask(masks=masks_s1)]
        # SetLatentNoiseMask: nell'AREA rumore puro (genera da zero col prompt), FUORI clean latent.
        # Ora col sampler euler_ancestral_cfg_pp (sotto) il rumore viene denoisato correttamente verso il
        # prompt (col denoiser euler standard dava un blob). Sostituzione di categoria possibile.
        Fl, Hl, Wl = clean_latent.shape[-3:]
        ml = torch.nn.functional.interpolate(masks_s1[None, None].float().to(clean_latent.device),
                                             size=(Fl, Hl, Wl), mode="nearest").to(clean_latent.dtype)
        clean_latent = clean_latent * (1 - ml) + torch.randn_like(clean_latent) * ml

        sig = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        video_state, _ = self.stage_1(
            denoiser=GuidedDenoiser(
                v_context=video_context, a_context=audio_context,
                video_guider=MultiModalGuider(params=MultiModalGuiderParams(cfg_scale=cfg_scale),
                                              negative_context=v_context_n),
                audio_guider=MultiModalGuider(params=MultiModalGuiderParams(), negative_context=a_context_n),
                force_uncond_pass=True),                 # cfg_pp loop richiede l'uncond denoised
            stepper=EulerCfgPpDiffusionStep(eta=1.0),    # euler_ancestral_cfg_pp (come il workflow inpaint)
            loop=euler_cfg_pp_denoising_loop,            # loop che alimenta lo stepper con uncond_denoised
            sigmas=sig, noiser=noiser, width=s1.width, height=s1.height, frames=num_frames, fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings,
                               initial_latent=clean_latent, noise_scale=float(sig[0].item())),
            audio=ModalitySpec(context=audio_context))
        logging.info("[inpaint] stage_1 done; skip_stage_2=%s", skip_stage_2)
        decoded = self.video_decoder(video_state.latent, None, generator)
        return decoded
