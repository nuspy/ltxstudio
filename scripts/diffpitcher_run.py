r"""Runner DiffPitcher — correzione d'intonazione voce (diffusion + WORLD + BigVGAN). SCAFFOLD **UNVALIDATED**.

⚠️ Scritto ALLA CIECA (GPU occupata 12h+, nessun test live possibile): la mappatura dei nostri pesi
safetensors sui moduli OSS e la firma esatta dei loader vanno VALIDATE al primo run reale. Punti a rischio
segnalati con [VALIDA].

Modalità:
- template: trasferisce la curva d'intonazione di una voce-guida (pitch_ref) sulla voce stonata (source).
- (score/MIDI: usa pitch_predictor — non incluso in questo scaffold iniziale.)

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
    ap.add_argument("--pitch-ref", dest="pitch_ref", required=True)   # voce-guida (template)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--shift-semi", dest="shift_semi", type=int, default=0)   # trasposizione in semitoni
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    import yaml
    import soundfile as sf
    from pitch_controller.models.unet import UNetPitcher
    import template_based_apc as tba                          # riusa template_pitcher()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # imposta i globali del modulo usati da template_pitcher (definiti solo nel suo __main__)
    tba.sr = 24000
    tba.min_mel = float(np.log(1e-5))
    tba.max_mel = 2.5
    tba.device = device
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

    pred = tba.template_pitcher(a.source, a.pitch_ref, model, hifigan,
                                steps=a.steps, shift_semi=a.shift_semi)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    sf.write(a.output, pred if not hasattr(pred, "numpy") else pred.numpy(), samplerate=tba.sr)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
