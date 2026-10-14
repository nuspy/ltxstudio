"""Veda — attenzione sparsa appresa per MiniMax-H3 (motore vendored, licenza MIT: vedi LICENSE e NOTICE.md).

Copia invariata di `veda_comfy/{core,backends,kernels/sage,hardware.py,settings.py}` da
https://github.com/veda-sparse/Veda-on-ComfyUI @ 3f9be822 (2026-10-06): nessuna dipendenza da ComfyUI.
Unica modifica: core/bundle.py accetta i predittori senza `keep_ratio` (R2VA, budget in tile).
Esclusi i backend MLX (Apple) e Sol (kernel non finito upstream). L'aggancio al nostro transformer H3 nativo
(diffusers) sta in workers/veda_h3.py.
"""
