"""Validazione DALL'APP-COMPLETA del figlio persistente (:8189).

Boota la MainWindow vera con NATIVE_PERSISTENT_CHILD=1 → l'InferenceServer tiene un figlio persistent_runner
con il ComponentStore caldo. Fa DUE generazioni immagine dello STESSO modello via il server reale e verifica:
  1. /system_stats riporta model_key/componenti (memoria OSSERVABILE, figlio persistente attivo);
  2. la 2ª generazione RIUSA il modello caldo → più veloce della 1ª (che carica);
  3. la VRAM resta OCCUPATA tra i due job (il modello NON viene scaricato) — cuore della richiesta;
  4. dopo /free la VRAM torna libera (svuota lo store senza uccidere il figlio).

Prerequisiti: modello FLUX schnell su disco. Uso:
  NATIVE_PERSISTENT_CHILD=1 <gui_python> tests/app_env/test_persistent_image.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# DEVE essere impostato PRIMA del boot (native_launcher propaga os.environ al server)
os.environ["NATIVE_PERSISTENT_CHILD"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as H  # noqa: E402

MODEL = "flux1-schnell-fp8.safetensors"
OUT = r"E:\ai\ltxstudio\output"


def _gen(i: int, seed: int) -> tuple[float, dict]:
    from core import native_driver
    params = {"arch": "flux", "model": MODEL,
              "prompt": f"a lone lighthouse on a cliff at dawn, cinematic, test {i}",
              "negative": "blurry", "width": 768, "height": 768, "steps": 4, "guidance": 0.0,
              "seed": seed, "output": os.path.join(OUT, f"DIAG_persist_img{i}.png")}
    t = time.time()
    res = native_driver.run_and_wait("image", params, timeout=600)
    return time.time() - t, res


def main() -> int:
    print("=== VALIDAZIONE figlio persistente :8189 (app-completa) ===", flush=True)
    w = H.boot_full_app()

    # 1ª generazione: COLD (il figlio nasce e carica il modello)
    dt1, r1 = _gen(1, 111)
    ok1 = r1.get("status") == "success"
    s1 = H.server_stats(8189) or {}
    print(f"[gen1] {int(dt1)}s status={r1.get('status')} model_key={s1.get('model_key')} "
          f"componenti={len(s1.get('components') or [])}", flush=True)
    H.print_snapshot(H.snapshot("dopo-gen1"))

    # 2ª generazione: WARM (stesso modello → deve riusare, più veloce)
    dt2, r2 = _gen(2, 222)
    ok2 = r2.get("status") == "success"
    s2 = H.server_stats(8189) or {}
    print(f"[gen2] {int(dt2)}s status={r2.get('status')} model_key={s2.get('model_key')}", flush=True)
    H.print_snapshot(H.snapshot("dopo-gen2"))

    # verifiche
    reused = ok1 and ok2 and dt2 < dt1 * 0.85                # warm sensibilmente più rapida
    observable = bool(s1.get("model_key") and s1.get("model_key") != "-")
    vram_held = (H.vram_used_mb() > 3000)                    # il modello è ancora residente tra i job

    # /free → la VRAM deve tornare libera
    from core import mem_coordinator
    mem_coordinator.free_all()
    time.sleep(3)
    freed_ok = H.vram_used_mb() < 3000
    print(f"[free] VRAM dopo free_all: {H.vram_used_mb()}MB (atteso <3000)", flush=True)

    H.teardown(w)
    print(f"--- gen1={int(dt1)}s gen2={int(dt2)}s | reuse={reused} observable={observable} "
          f"vram_held={vram_held} freed={freed_ok} ---", flush=True)
    ok = ok1 and ok2 and observable and reused and vram_held and freed_ok
    print(("=== PERSISTENT IMAGE OK ===" if ok else "=== FAIL ==="), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
