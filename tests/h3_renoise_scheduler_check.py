r"""Verifica (venv diffusers, CPU, tensori minuscoli) dello scheduler «renoise» di Pika H3 Flex contro i NUMERI DEL RIFERIMENTO, calcolati a mano dalle formule del
codice di riferimento (diffusers d0b9b1b5, letto e non eseguito): nodi [1000, 750, 500, 250, 100, 40, 0], shift 12 (video) e 3 (audio), passo
x0 = x + sigma·v, x_next = (1 - sigma_next)·x0 + sigma_next·rumore, l'ultimo passo restituisce x0. Si lancia:
    E:\AI\Tools\diffusers_next_venv\Scripts\python.exe tests\h3_renoise_scheduler_check.py     (stampa «RENOISE OK»)"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
import h3_edit_hold  # noqa: E402
import h3_renoise_scheduler as renoise  # noqa: E402
from diffusers import MiniMaxH3Scheduler  # noqa: E402

KNOTS = (1000, 750, 500, 250, 100, 40, 0)


def closed_form(knots, shift):
    return [shift * (k / 1000.0) / (1 + (shift - 1) * (k / 1000.0)) for k in knots]


def check_grids():
    video = renoise.make_renoise_scheduler_class(MiniMaxH3Scheduler, KNOTS, 1)(shift=12.0)
    audio = renoise.make_renoise_scheduler_class(MiniMaxH3Scheduler, KNOTS, 2)(shift=3.0)
    for scheduler, shift in ((video, 12.0), (audio, 3.0)):
        scheduler.set_timesteps(49, device="cpu")                                  # il numero richiesto e ignorato: decidono i nodi
        expected = torch.tensor(closed_form(KNOTS, shift), dtype=torch.float32)
        assert torch.allclose(scheduler.sigmas, expected, atol=1e-7), (scheduler.sigmas, expected)
        assert scheduler.timesteps.numel() == 6 and torch.allclose(scheduler.timesteps, 1 - expected[:-1], atol=1e-7)
    print("sigma video", [round(v, 5) for v in video.sigmas.tolist()], "audio", [round(v, 5) for v in audio.sigmas.tolist()])


def check_steps():
    scheduler = renoise.make_renoise_scheduler_class(MiniMaxH3Scheduler, KNOTS, 123)(shift=12.0)
    scheduler.set_timesteps(None, device="cpu")
    sample = torch.randn(2, 5)
    velocity = [torch.randn(2, 5) for _ in range(6)]
    # rumore atteso: stesso generatore del scheduler, stessa sequenza di estrazioni
    generator = torch.Generator(device="cpu").manual_seed(123)
    sigmas = closed_form(KNOTS, 12.0)
    expected = sample.clone()
    current = sample.clone()
    for step, timestep in enumerate(scheduler.timesteps):
        noise = torch.randn(expected.shape, generator=generator, dtype=torch.float32)
        x0 = expected + torch.tensor(sigmas[step], dtype=torch.float32) * velocity[step]
        expected = (1 - torch.tensor(sigmas[step + 1], dtype=torch.float32)) * x0 + torch.tensor(sigmas[step + 1], dtype=torch.float32) * noise
        current = scheduler.step(velocity[step], timestep, current, return_dict=False)[0]
        assert torch.allclose(current, expected, atol=1e-5), (step, current, expected)
    assert torch.allclose(current, x0, atol=1e-5)                                      # l'ultimo passo (sigma_next = 0) e x0
    print("6 passi: uguali alla formula del riferimento; l'ultimo restituisce x0")


def check_reproducibility_and_modalities():
    video_class = renoise.make_renoise_scheduler_class(MiniMaxH3Scheduler, KNOTS, 7)
    runs = []
    for _ in range(2):
        scheduler = video_class(shift=12.0)
        scheduler.set_timesteps(None, device="cpu")
        torch.manual_seed(0)
        sample = torch.randn(1, 4)
        for timestep in scheduler.timesteps:
            sample = scheduler.step(torch.ones(1, 4), timestep, sample, return_dict=False)[0]
        runs.append(sample)
    assert torch.equal(runs[0], runs[1])                                                # stesso seme, stesso risultato
    other = renoise.make_renoise_scheduler_class(MiniMaxH3Scheduler, KNOTS, 8)(shift=12.0)
    other.set_timesteps(None, device="cpu")
    torch.manual_seed(0)
    sample = torch.randn(1, 4)
    for timestep in other.timesteps:
        sample = other.step(torch.ones(1, 4), timestep, sample, return_dict=False)[0]
    assert not torch.equal(sample, runs[0])                                              # semi diversi: rumore diverso
    print("riproducibile col seme; semi diversi danno rumore diverso")


def check_hold_composition():
    video_class = renoise.make_renoise_scheduler_class(MiniMaxH3Scheduler, KNOTS, 5)
    scheduler = h3_edit_hold.make_hold_scheduler_class(video_class)(shift=12.0)
    scheduler.set_timesteps(None, device="cpu")
    source = torch.full((1, 4), 5.0)
    keep = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    scheduler.set_hold(source, torch.zeros(1, 4), keep)
    sample = torch.randn(1, 4)
    for timestep in scheduler.timesteps:
        sample = scheduler.step(torch.zeros(1, 4), timestep, sample, return_dict=False)[0]
    assert torch.allclose(sample[0, :2], torch.tensor([5.0, 5.0])), sample                # le righe tenute finiscono ESATTAMENTE sulla sorgente
    print("compone con lo scheduler di hold degli edit")


def check_scheduler_classes_selection():
    import minimax_h3_infer as worker
    from core import minimax_h3_spec as spec
    plain_video, plain_audio = worker.scheduler_classes("v4", 3)
    assert plain_video is MiniMaxH3Scheduler and plain_audio is MiniMaxH3Scheduler
    spec.TURBO_SAMPLER["pika_probe"] = {"method": "renoise", "knots": KNOTS}
    try:
        video_class, audio_class = worker.scheduler_classes("pika_probe", 3)
        assert issubclass(video_class, MiniMaxH3Scheduler) and video_class is not MiniMaxH3Scheduler and audio_class is not video_class
        scheduler = video_class(shift=12.0)
        scheduler.set_timesteps(49, device="cpu")
        assert scheduler.timesteps.numel() == 6
    finally:
        del spec.TURBO_SAMPLER["pika_probe"]
    print("scelta della classe: Euler per le turbo normali, renoise per quelle con campionatore proprio")


if __name__ == "__main__":
    check_grids()
    check_steps()
    check_reproducibility_and_modalities()
    check_hold_composition()
    check_scheduler_classes_selection()
    print("RENOISE OK")
