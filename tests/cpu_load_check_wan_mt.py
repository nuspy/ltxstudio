"""Prova di CARICAMENTO + UN FORWARD con i PESI VERI di Wan2.1 I2V 14B fp8 + MultiTalk + LoRA lightx2v, su CPU
(nessuna GPU): verifica che il percorso di caricamento di MultiTalkEngine.load_dit (lettura no-mmap, cast, RowFp8Linear,
LoRA al volo, block-swap) produca un modello senza chiavi mancanti e con uscita finita. ~25 GB di RAM, qualche minuto.

NON e' un test automatico (pesante): lanciarlo SOTTO il guardiano
    python tests\\live_guard.py --min-commit-gb 20 --max-proc-gb 70 --kill-servers-on-exit -- ^
        E:\\AI\\Tools\\inference_server_venv\\Scripts\\python.exe tests\\cpu_load_check_wan_mt.py
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "workers")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from wan_mt import engine as E  # noqa: E402


def main() -> int:
    paths = E.resolve_paths()
    miss = E.missing_required(paths)
    print("percorsi:", {k: (os.path.basename(v) if v else None) for k, v in paths.items()}, "mancanti:", miss, flush=True)
    if miss:
        return 3
    t0 = time.time()
    eng = E.MultiTalkEngine(paths, device="cpu", log=lambda s: print(s, flush=True))
    plan = eng.load_dit(128, 128, loras=[(paths["lora"], 0.6)], blocks_to_swap="auto")   # CPU: free=0 -> tutto in swap
    print(f"[ok] caricato in {time.time() - t0:.0f}s; blocchi residenti {plan.resident_blocks}; audio fp8={plan.audio_fp8}", flush=True)
    m = eng.model
    print("tipi layer blocco 0:", type(m.blocks[0].self_attn.q).__name__, type(m.blocks[0].audio_cross_attn.q_linear).__name__,
          "| loras su self_attn.q:", len(getattr(m.blocks[0].self_attn.q, "loras", [])), flush=True)
    torch.manual_seed(0)
    T, H, W = 2, 16, 16                                  # 5 frame video, 128x128 px
    x, y = torch.randn(16, T, H, W), torch.randn(20, T, H, W)
    ctx, clip = torch.randn(20, 4096) * 0.5, torch.randn(1, 257, 1280) * 0.5
    aud = torch.randn(1, 5, 5, 12, 768) * 0.5
    t1 = time.time()
    with torch.inference_mode():
        out = m(x, torch.tensor([800.0]), ctx, clip, y, audio=aud, audio_scale=2.0)
    print(f"[forward] {time.time() - t1:.0f}s shape={tuple(out.shape)} finito={bool(torch.isfinite(out).all())} "
          f"media={out.mean():.4f} std={out.std():.4f}", flush=True)
    ok = bool(torch.isfinite(out).all()) and out.std() > 1e-4
    print("RISULTATO:", "OK" if ok else "FALLITO", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
