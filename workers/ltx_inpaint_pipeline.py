"""#59 InpaintPipeline — inpaint/sostituzione VIDEO nativo (sottoclasse di ICLoraPipeline).

Combina i due ingredienti del workflow ufficiale InOutpainting:
- **conditioning_attention_mask** (= inverso della maschera SAM2): dove c'è l'oggetto l'attention della
  IC-LoRA va a 0 → il modello è libero di GENERARE dal prompt; fuori = 1 → preserva il guide (video orig).
  Risultato PULITO (niente blob, sfondo intatto).
- **CFG** (GuidedDenoiser cfg_scale>1) + **euler_ancestral_cfg_pp** (stepper+loop): forza il prompt a
  sostituire la categoria dell'oggetto (il modello distilled senza CFG resta ancorato al contesto).

Niente noise-injection / denoise_mask manuali (davano blob). NON modifica il package ufficiale.
"""
from __future__ import annotations

import torch
from ltx_core.types import VideoPixelShape
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.types import ModalitySpec


class InpaintPipeline(ICLoraPipeline):
    @torch.inference_mode()
    def inpaint(self, *, prompt: str, seed: int, height: int, width: int, num_frames: int,
                frame_rate: float, video_conditioning: list[tuple[str, float]],
                attention_mask: torch.Tensor, images=None, negative_prompt: str = "worst quality, blurry",
                cfg_scale: float = 4.0, num_steps: int = 30, stage_1_sigmas=None, **_ignore):
        """video_conditioning: [(video_originale, strength)]. attention_mask: (T,H,W) in [0,1], maschera
        OGGETTO (1=oggetto da sostituire); viene invertita internamente per la IC-LoRA.
        images: [ImageConditioningInput] opzionale = foto di riferimento dell'oggetto (LoRA inpaint r2v), condizionamento
        di immagine come in ICLoraPipeline.__call__; None/vuoto = LoRA t2v (solo prompt)."""
        from ltx_core.components.diffusion_steps import EulerCfgPpDiffusionStep
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_pipelines.utils.denoisers import GuidedDenoiser
        from ltx_pipelines.utils.samplers import euler_cfg_pp_denoising_loop

        if stage_1_sigmas is None:
            from ltx_core.components.schedulers import LTX2Scheduler
            stage_1_sigmas = LTX2Scheduler().execute(steps=num_steps)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        ctx_p, ctx_n = self.prompt_encoder([prompt, negative_prompt])
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        s1 = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate)
        # attention_mask INVERSA (1 dove preservare, 0 sull'oggetto) a risoluzione stage_1, (1,1,F,h,w)
        m = attention_mask.to(self.device).float()[:num_frames]
        m = torch.nn.functional.interpolate(m.unsqueeze(1), size=(s1.height, s1.width),
                                            mode="nearest").squeeze(1)        # (F,h,w)
        cam = (1.0 - m)[None, None]                                           # (1,1,F,h,w) = inpaint

        def _build(enc):
            return self._create_conditionings(images=list(images or []), video_conditioning=video_conditioning,
                                              height=s1.height, width=s1.width, video_encoder=enc,
                                              num_frames=num_frames, conditioning_attention_mask=cam)

        stage_1_conditionings = self.image_conditioner(_build)
        sig = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        video_state, _ = self.stage_1(
            denoiser=GuidedDenoiser(
                v_context=video_context, a_context=audio_context,
                video_guider=MultiModalGuider(params=MultiModalGuiderParams(cfg_scale=cfg_scale),
                                              negative_context=v_context_n),
                audio_guider=MultiModalGuider(params=MultiModalGuiderParams(), negative_context=a_context_n),
                force_uncond_pass=True),
            stepper=EulerCfgPpDiffusionStep(eta=1.0),
            loop=euler_cfg_pp_denoising_loop,
            sigmas=sig, noiser=noiser, width=s1.width, height=s1.height, frames=num_frames, fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context))
        return self.video_decoder(video_state.latent, None, generator)
