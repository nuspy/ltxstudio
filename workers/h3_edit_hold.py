"""MiniMax H3 EDIT — scheduler che TIENE le righe mascherate legate alla sorgente (nativo, senza ComfyUI).

Il denoise di diffusers (`MiniMaxH3LoopSchedulerStep`) scrive solo le righe GENERATE: qui il passo dello scheduler e allargato cosi:
dopo ogni passo di Euler, gli elementi di latente con maschera 0 («tieni») sono rimessi sul loro valore di sorgente rumorizzato al
livello del PASSO SUCCESSIVO (x_t = t*x0 + (1-t)*rumore, t = 1 - sigma, convenzione H3), e all'ultimo passo (sigma = 0) sono ESATTAMENTE la
sorgente. E l'inpaint a latente standard dei modelli rectified-flow: il modello vede sempre una scena coerente (parte tenuta + parte
rigenerata allo stesso livello di rumore) e la parte generata si adatta a quella tenuta.

Vale per il video (righe [N, C*pt*ph*pw], maschera elemento per elemento) e per l'audio (righe [N, C]).
"""
from __future__ import annotations

import torch


def hold_mix(x: torch.Tensor, src: torch.Tensor, noise: torch.Tensor, keep: torch.Tensor, t_next: float) -> torch.Tensor:
    """x: campione dopo il passo; keep: 1 = tieni la sorgente, 0 = lascia il generato."""
    held = t_next * src + (1.0 - t_next) * noise
    return torch.where(keep > 0.5, held.to(x.dtype), x)


def make_hold_scheduler_class(base_cls):
    """Sottoclasse di MiniMaxH3Scheduler (la pipeline controlla il tipo del componente) con `step` allargato.
    `hold` = dict(src=..., noise=..., keep=...) tutti sullo STESSO device/forma dei campioni passati a step()."""
    class HoldScheduler(base_cls):
        hold: dict | None = None

        def set_hold(self, src, noise, keep):
            if not (src.shape == noise.shape == keep.shape):
                raise ValueError(f"hold: forme diverse src {tuple(src.shape)} noise {tuple(noise.shape)} keep {tuple(keep.shape)}")
            self.hold = {"src": src, "noise": noise, "keep": keep}

        def step(self, model_output, timestep, sample, return_dict: bool = True):
            out = super().step(model_output, timestep, sample, return_dict=False)
            x = out[0]
            h = self.hold
            if h is not None:
                i = self.index_for_timestep(timestep)
                s_next = float(self.sigmas[i + 1])                    # sigma del passo dopo (l'ultimo = 0 -> sorgente esatta)
                dev = x.device
                x = hold_mix(x, h["src"].to(dev), h["noise"].to(dev), h["keep"].to(dev), 1.0 - s_next)
            if return_dict:
                from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3SchedulerOutput
                return MiniMaxH3SchedulerOutput(prev_sample=x)
            return (x,) + tuple(out[1:])
    return HoldScheduler
