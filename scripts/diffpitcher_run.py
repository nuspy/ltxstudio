r"""Runner DiffPitcher — correzione d'intonazione voce (diffusion + WORLD + BigVGAN). SCAFFOLD **UNVALIDATED**.

⚠️ Scritto ALLA CIECA (GPU occupata 12h+, nessun test live possibile): la mappatura dei nostri pesi
safetensors sui moduli OSS e la firma esatta dei loader vanno VALIDATE al primo run reale. Punti a rischio
segnalati con [VALIDA].

Modalità:
- template: trasferisce la curva d'intonazione di una voce-guida (pitch_ref) sulla voce stonata (source).
  VALIDATO A SECCO il 17/10/2026 (CPU, --device cpu): pesi caricati senza errori di chiavi, esempio upstream
  off-key.wav + reference.wav → deviazione mediana dalla guida 0,88 → 0,20 semitoni.
- score: intonazione da SPARTITO (file MIDI): il PitchFormer (Diff-Pitcher_transformer_pitch_360.safetensors) predice
  la curva f0 dal mel della voce + dal piano-roll MIDI, poi lo stesso diffusion/vocoder. Come `score_based_apc.py`.

Layout:
- codice OSS:  E:\AI\Tools\DiffPitcher-src   (package pitch_controller)
- pesi:        E:\AI\Models\DiffPitcher       (Diff-Pitcher_world_fixed_40 / _bigvgan_24khz_100band / config.json)
- venv:        E:\AI\Tools\ACE-Step\venv15    (+ pyworld/pysptk/fastdtw/music21/tgt)
"""
import argparse
import json
import os
import sys

SRC_DIR = r"E:\AI\Tools\DiffPitcher-src"
W_DIR = r"E:\AI\Models\DiffPitcher"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

_WORLD = os.path.join(W_DIR, "Diff-Pitcher_world_fixed_40.safetensors")
_BIGVGAN = os.path.join(W_DIR, "Diff-Pitcher_bigvgan_24khz_100band.safetensors")
_BIGVGAN_CFG = os.path.join(W_DIR, "config.json")
_PITCHFORMER = os.path.join(W_DIR, "Diff-Pitcher_transformer_pitch_360.safetensors")
_UNET_CFG = os.path.join(SRC_DIR, "pitch_controller", "config", "DiffWorld_24k.yaml")


def _load_sd(path):
    """Carica uno state_dict da .safetensors (nostri pesi) o .pt."""
    from safetensors.torch import load_file
    sd = load_file(path)
    # [VALIDA] i pesi Premium potrebbero essere annidati sotto 'generator'/'model' o avere prefisso _orig_mod
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)               # voce stonata da correggere
    ap.add_argument("--mode", default="template", choices=["template", "score"])
    ap.add_argument("--pitch-ref", dest="pitch_ref", default="")      # voce-guida (template)
    ap.add_argument("--midi", default="")                              # spartito .mid/.midi (score)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--shift-semi", dest="shift_semi", type=int, default=0)   # trasposizione in semitoni
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])   # cpu = niente GPU (test a secco)
    a = ap.parse_args()
    if a.mode == "template" and not a.pitch_ref:
        ap.error("--pitch-ref è obbligatorio in modalità template")
    if a.mode == "score" and not a.midi:
        ap.error("--midi è obbligatorio in modalità score")

    import numpy as np
    import torch
    import yaml
    import soundfile as sf
    from pitch_controller.models.unet import UNetPitcher
    import template_based_apc as tba                          # riusa template_pitcher()
    if a.mode == "score":
        import score_based_apc as sba                         # riusa score_pitcher() (+ PitchFormer)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    # imposta i globali del modulo usati da template_pitcher (definiti solo nel suo __main__)
    tba.sr = 24000
    tba.min_mel = float(np.log(1e-5))
    tba.max_mel = 2.5
    tba.device = device
    if a.mode == "score":                                     # stessi globali anche per score_pitcher
        sba.sr, sba.min_mel, sba.max_mel, sba.device = tba.sr, tba.min_mel, tba.max_mel, device
    print(f"[diffpitcher] device={device} steps={a.steps} shift={a.shift_semi}", flush=True)

    # --- UNetPitcher (diffusion) dai nostri safetensors ---
    cfg = yaml.load(open(_UNET_CFG), Loader=yaml.FullLoader)
    model = UNetPitcher(**cfg["unet"])
    model.load_state_dict(_load_sd(_WORLD))                   # [VALIDA] match chiavi state_dict
    model.to(device).eval()

    # --- BigVGAN vocoder dai nostri safetensors + config.json ---
    from pitch_controller.modules.BigVGAN.models import BigVGAN as Generator
    try:
        from pitch_controller.modules.BigVGAN.env import AttrDict
    except Exception:                                        # noqa: BLE001
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs); self.__dict__ = self
    h = AttrDict(json.load(open(_BIGVGAN_CFG)))
    hifigan = Generator(h).to(device)
    bsd = _load_sd(_BIGVGAN)
    hifigan.load_state_dict(bsd.get("generator", bsd))       # [VALIDA] flat vs {'generator':...}
    hifigan.eval()
    if hasattr(hifigan, "remove_weight_norm"):
        hifigan.remove_weight_norm()                         # BigVGAN richiede questo per l'inferenza

    if a.mode == "score":
        from pitch_predictor.models.transformer import PitchFormer
        pitcher = PitchFormer(100, 512).to(device)            # parametri di score_based_apc.py (non c'è un yaml)
        pitcher.load_state_dict(_load_sd(_PITCHFORMER))       # strict: chiavi verificate sull'header (59 tensori)
        pitcher.eval()
        pred = sba.score_pitcher(a.source, a.midi, model, hifigan, pitcher,
                                 steps=a.steps, shift_semi=a.shift_semi)
    else:
        pred = tba.template_pitcher(a.source, a.pitch_ref, model, hifigan,
                                    steps=a.steps, shift_semi=a.shift_semi)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    sf.write(a.output, pred if not hasattr(pred, "numpy") else pred.numpy(), samplerate=tba.sr)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
