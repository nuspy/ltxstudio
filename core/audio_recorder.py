"""Registrazione voce dal microfono per la clonazione (Fase 3).

Usa sounddevice + soundfile (import guardato). Registra mono 16kHz/44.1k in un WAV;
il file va poi caricato in ComfyUI via comfy_driver.upload_image (input dir) e referenziato
dal nodo VHS_LoadAudioUpload della clonazione voce.
"""
from __future__ import annotations

from pathlib import Path

REC_DIR = Path(r"E:\AI\ltxstudio\data\recordings")


class AudioRecorder:
    def __init__(self, samplerate: int = 44100, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self._frames = []
        self._stream = None
        REC_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def available() -> bool:
        try:
            import sounddevice  # noqa
            import soundfile  # noqa
            return True
        except Exception:
            return False

    def start(self):
        import sounddevice as sd
        self._frames = []

        def cb(indata, frames, time_, status):
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=cb)
        self._stream.start()

    def stop(self, out_name: str = "voice_sample.wav") -> str:
        import numpy as np
        import soundfile as sf
        if self._stream:
            self._stream.stop(); self._stream.close(); self._stream = None
        out = REC_DIR / out_name
        if self._frames:
            data = np.concatenate(self._frames, axis=0)
            sf.write(str(out), data, self.samplerate)
        return str(out)
