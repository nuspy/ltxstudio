"""Wrapper di lancio HeartMuLa che rende il salvataggio audio indipendente da torchcodec.

torchaudio >= 2.9 delega `torchaudio.save` a TorchCodec ("save_with_torchcodec"); se torchcodec
non è installato, il salvataggio della canzone fallisce DOPO averla generata. Qui sostituiamo
`torchaudio.save` con un'implementazione basata su **soundfile** (già nel venv heartlib) PRIMA di
eseguire lo script ufficiale `examples/run_music_generation.py`. Nessun download, nessun edit del
pacchetto HeartMuLa. Invocato al posto dello script: argv passa identico (li inoltriamo).
"""
from __future__ import annotations

import runpy
import sys

RUN = r"E:\AI\Tools\heartlib\examples\run_music_generation.py"


def _patch_torchaudio_save() -> None:
    import numpy as np
    import torch
    import torchaudio
    try:
        import soundfile as sf
    except Exception as e:                                   # noqa: BLE001
        print(f"[heartmula_run] soundfile non disponibile ({e}); salvataggio invariato", flush=True)
        return

    def _save(uri, src, sample_rate, *args, **kwargs):       # noqa: ANN001
        arr = src.detach().to(torch.float32).cpu().numpy() if isinstance(src, torch.Tensor) else np.asarray(src)
        if arr.ndim == 2:                                    # torchaudio = (canali, frame) → soundfile (frame, canali)
            arr = arr.T
        sf.write(str(uri), arr, int(sample_rate))
        return None

    torchaudio.save = _save
    print("[heartmula_run] torchaudio.save → soundfile (no torchcodec)", flush=True)


def main() -> None:
    _patch_torchaudio_save()
    sys.argv = [RUN] + sys.argv[1:]                          # lo script legge i suoi argparse
    runpy.run_path(RUN, run_name="__main__")


if __name__ == "__main__":
    main()
