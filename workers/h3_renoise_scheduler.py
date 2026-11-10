"""Scheduler «renoise» per gli adattatori Pika H3 Flex (6 passi), replica dello scheduler di riferimento di diffusers (commit d0b9b1b5, `MiniMaxH3Scheduler` con
`stochastic_sampling=True` e `shift_provided_sigmas=True`; il diffusers installato qui e piu vecchio e non li ha).

Dal codice di riferimento (letto, non eseguito):
 · schedule fisso: `denoising_step_list` [1000, 750, 500, 250, 100, 40, 0] / 1000 = sigma di base, che ogni scheduler spinge nel proprio shift
   (sigma' = s·sigma / (1 + (s-1)·sigma): s = 12 per il video, 3 per l'audio), in float64 poi float32; 6 valutazioni del modello e il terminale 0;
 · passo: la velocita del trasformatore e «verso il dato»: x0 = x_t + sigma·v (con sigma della griglia, in float32); poi x_next = (1 - sigma_next)·x0 +
   sigma_next·rumore, con rumore NUOVO a ogni passo (float32): l'ultimo passo (sigma_next = 0) restituisce x0;
 · niente guider: la distillazione e senza guida, una sola passata per passo.
Il rumore viene da un generatore proprio di ogni scheduler (video e audio hanno semi diversi): riproducibile col seme della richiesta."""
from __future__ import annotations

import torch


def shifted_sigmas(knots: tuple, shift: float) -> list:
    """Griglia di sigma dei nodi (scala 1000) dopo lo shift, come `set_timesteps(sigmas=..., shift_provided_sigmas=True)` del riferimento (float64 -> float32)."""
    base = torch.as_tensor([knot / 1000.0 for knot in knots], dtype=torch.float64)
    shifted = shift * base / (1 + (shift - 1) * base)
    return shifted.to(torch.float32).tolist()


def renoise_update(sample: torch.Tensor, model_output: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Un passo di re-noising in float32: x0 = x + sigma·v, x_next = (1 - sigma_next)·x0 + sigma_next·rumore."""
    denoised = sample.to(torch.float32) + sigma * model_output.to(torch.float32)
    return ((1.0 - sigma_next) * denoised + sigma_next * noise).to(dtype=sample.dtype)


def make_renoise_scheduler_class(base_class, knots: tuple, seed: int):
    """Sottoclasse di MiniMaxH3Scheduler (la pipeline controlla il tipo del componente) con la griglia dei nodi e il passo di re-noising."""

    class RenoiseScheduler(base_class):
        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
            """Il numero di passi lo decide la griglia dei nodi dell'adattatore, non `num_inference_steps`."""
            super().set_timesteps(device=device, sigmas=shifted_sigmas(knots, self.shift))
            self._noise_generator = None

        def _fresh_noise(self, sample: torch.Tensor) -> torch.Tensor:
            if getattr(self, "_noise_generator", None) is None:
                self._noise_generator = torch.Generator(device=sample.device).manual_seed(seed)
            return torch.randn(sample.shape, generator=self._noise_generator, device=sample.device, dtype=torch.float32)

        def step(self, model_output, timestep, sample, return_dict: bool = True):
            if self._step_index is None:
                self._step_index = self.index_for_timestep(timestep) if self._begin_index is None else self._begin_index
            sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=torch.float32)
            sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=torch.float32)
            prev_sample = renoise_update(sample, model_output, sigma, sigma_next, self._fresh_noise(sample))
            self._step_index += 1
            if not return_dict:
                return (prev_sample,)
            from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3SchedulerOutput
            return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)

    return RenoiseScheduler
