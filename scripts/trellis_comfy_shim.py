"""Shim minimale di `comfy` per eseguire il codice MIT di ComfyUI-TRELLIS2 (stages.py) SENZA ComfyUI.

Codice ORIGINALE (mio): registra in sys.modules `comfy.model_management/utils/model_patcher` e `folder_paths`
con le sole funzioni usate da stages.py (reverse-eng). install_shim() va chiamato PRIMA di importare trellis_nodes.
"""
from __future__ import annotations

import os
import sys
import types


def install_shim(tools_dir: str, compute_device: str = "cuda"):
    import torch
    from safetensors.torch import load_file

    models_dir = os.path.join(tools_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device(compute_device)
    mm.unet_offload_device = lambda: torch.device("cpu")
    mm.text_encoder_device = lambda: torch.device(compute_device)
    mm.text_encoder_offload_device = lambda: torch.device("cpu")
    mm.vae_dtype = lambda device=None, allowed_dtypes=(): (
        torch.bfloat16 if torch.bfloat16 in (allowed_dtypes or (torch.bfloat16,)) else torch.float16)
    mm.unet_dtype = lambda *a, **k: torch.bfloat16
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.soft_empty_cache = lambda *a, **k: (torch.cuda.empty_cache() if torch.cuda.is_available() else None)
    mm.get_free_memory = lambda *a, **k: (torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 8 << 30)

    def load_models_gpu(models, *a, **k):
        for m in models:
            dev = getattr(m, "load_device", torch.device(compute_device))
            try:
                getattr(m, "model", m).to(dev)
            except Exception:  # noqa: BLE001
                pass
    mm.load_models_gpu = load_models_gpu

    u = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total=0): self.total = total
        def update(self, n=1): pass
        def update_absolute(self, *a, **k): pass
    u.ProgressBar = ProgressBar
    u.load_torch_file = lambda path, *a, **k: load_file(path)

    mp = types.ModuleType("comfy.model_patcher")

    class ModelPatcher:
        def __init__(self, model, load_device=None, offload_device=None, **k):
            self.model = model
            self.load_device = load_device or torch.device(compute_device)
            self.offload_device = offload_device or torch.device("cpu")
        def patch_model(self, device_to=None, *a, **k):
            self.model.to(device_to or self.load_device); return self.model
        def unpatch_model(self, device_to=None, *a, **k):
            self.model.to(device_to or self.offload_device)
        def model_size(self): return 0
    mp.ModelPatcher = ModelPatcher

    comfy = types.ModuleType("comfy")
    comfy.model_management = mm; comfy.utils = u; comfy.model_patcher = mp

    fp = types.ModuleType("folder_paths")
    fp.models_dir = models_dir
    fp.get_folder_paths = lambda *a, **k: [models_dir]

    for name, mod in {"comfy": comfy, "comfy.model_management": mm, "comfy.utils": u,
                      "comfy.model_patcher": mp, "folder_paths": fp}.items():
        sys.modules[name] = mod
