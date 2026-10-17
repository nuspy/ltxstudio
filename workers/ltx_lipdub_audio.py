"""LipDub con AUDIO ESTERNO — variante SPERIMENTALE di `ltx_pipelines.lipdub.LipDubPipeline` (task 3, 17/10/2026).

La pipeline ufficiale (JUST-DUB-IT) NON accetta un audio dell'utente: il video sorgente (con la sua voce) e' il
riferimento audio+video e il NUOVO TESTO nel prompt genera voce e labbra nuove (la voce esce dallo stage 1:
`decoded_audio = audio_decoder(s1_audio_latent)`).

Questa sottoclasse cambia UNA cosa: l'audio dell'utente e' codificato con l'audio VAE e TENUTO FISSO nei latenti
audio (`frozen=True, noise_scale=0.0, initial_latent=<latente audio>`), come fa lo stage 2 ufficiale di LipDub e
come fa `A2VidPipelineTwoStage` (a2vid_two_stage.py:184-188, 223-227). Si rigenera solo il VIDEO, condizionato dal
riferimento video + dall'audio. L'audio in uscita e' quello originale dell'utente (come a2vid: massima fedelta').

Riferimento audio (`audio_reference`):
  - "external" (default): i token di riferimento audio (posizioni negative) sono l'audio dell'utente stesso, come nello
    stage 2 ufficiale (dove il riferimento e' l'audio gia' generato e congelato);
  - "original": i token di riferimento sono la voce del video sorgente (come nello stage 1 ufficiale), l'audio congelato
    e' quello dell'utente. Serve a confrontare le due letture dal vivo.
Non e' addestrato cosi' dal rilascio: qualita' NON garantita, test dal vivo in TESTS.md.
"""
from __future__ import annotations

import torch
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import AudioConditionByReferenceLatent
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.lipdub import (LipDubPipeline, _snap_frames_to_8k1,
                                  patchify_lipdub_audio_reference_latent)
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.media_io import decode_audio_from_file, get_videostream_metadata
from ltx_pipelines.utils.types import ModalitySpec


class LipDubAudioPipeline(LipDubPipeline):
    """LipDubPipeline con audio dell'utente congelato nei latenti (sperimentale)."""

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        images: list[ImageConditioningInput],
        reference_video_path: str,
        audio_path: str,
        reference_strength: float = 1.0,
        audio_reference: str = "external",
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ):
        from ltx_pipelines.utils.helpers import assert_resolution
        assert_resolution(height=height, width=width, is_two_stage=True)

        meta = get_videostream_metadata(reference_video_path)
        num_frames = _snap_frames_to_8k1(meta.frames)
        frame_rate = float(meta.fps)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        # audio dell'utente → latente audio fisso, lungo ESATTAMENTE quanto il video (a2vid_two_stage.py:130-136)
        decoded_audio = decode_audio_from_file(audio_path, self.device, 0.0, num_frames / frame_rate)
        if decoded_audio is None:
            raise ValueError(f"nessuna traccia audio in {audio_path}")
        ext_latent = self.audio_conditioner(lambda enc: vae_encode_audio(decoded_audio, enc, None))
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
        ext_latent = ext_latent[:, :, : audio_shape.frames]
        missing = audio_shape.frames - ext_latent.shape[2]
        if missing > 0:                                   # l'audio e' stato preparato alla durata giusta: al massimo 1-2 frame
            ext_latent = torch.nn.functional.pad(ext_latent, (0, 0, 0, missing))

        if audio_reference == "original":
            ref_latent = self._encode_reference_audio_vae_latent(reference_video_path)
        else:
            ref_latent = ext_latent

        encode_tiling = TilingConfig.default()

        def build_image_conditionings(output_shape: VideoPixelShape) -> list:
            return self.image_conditioner(
                lambda enc: self._create_stage_conditionings(
                    images=images,
                    reference_video_path=reference_video_path,
                    reference_strength=reference_strength,
                    height=output_shape.height,
                    width=output_shape.width,
                    num_frames=num_frames,
                    video_encoder=enc,
                    encode_tiling=encode_tiling,
                )
            )

        def build_audio_ref_conditioning(audio_latent: torch.Tensor) -> AudioConditionByReferenceLatent:
            ref_patch, ref_pos = patchify_lipdub_audio_reference_latent(
                audio_latent, negative_positions=True, device=self.device)
            return AudioConditionByReferenceLatent(ref_patch, ref_pos, strength=1.0)

        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate)
        stage_1_conditionings = build_image_conditionings(stage_1_output_shape)

        stage_1_sigmas_tensor = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        video_state, _audio_unused = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas_tensor,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(
                context=audio_context,
                conditionings=[build_audio_ref_conditioning(ref_latent)],
                frozen=True,
                noise_scale=0.0,
                initial_latent=ext_latent,
            ),
        )

        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_sigmas_tensor = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = build_image_conditionings(stage_2_output_shape)

        video_state, _audio_unused = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas_tensor,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas_tensor[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                conditionings=[build_audio_ref_conditioning(ext_latent)],
                frozen=True,
                noise_scale=0.0,
                initial_latent=ext_latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        # audio in uscita = quello ORIGINALE dell'utente (come a2vid_two_stage.py:233-237)
        out_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)
        return decoded_video, out_audio
