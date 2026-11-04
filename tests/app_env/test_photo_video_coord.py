"""Validazione DALL'APP-COMPLETA della coordinazione FOTO→VIDEO del gpu_registry.

Sequenza: genera una foto FLUX (modello residente su :8189, ~28GB) → simula l'arrivo di un job VIDEO
chiamando gpu_registry.before_job('video') come fa il dispatch → verifica che il registro LIBERI il modello
immagine (cambio-categoria) e la VRAM torni disponibile per il video. Poi una 2ª foto stesso modello →
il registro decide 'load' (il modello era stato liberato dal video). Tutto con i meccanismi memoria reali.

  NATIVE_PERSISTENT_CHILD=1 <gui_python> tests/app_env/test_photo_video_coord.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["NATIVE_PERSISTENT_CHILD"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app_env_harness as H  # noqa: E402

MODEL = "FLUX_Dev.safetensors"
OUT = r"E:\ai\ltxstudio\output"


def _gen_img(i, seed):
    from core import native_driver
    p = {"arch": "flux", "model": MODEL, "prompt": f"a foggy harbor at night, test {i}",
         "negative": "blurry", "width": 768, "height": 768, "steps": 8, "guidance": 3.5,
         "seed": seed, "output": os.path.join(OUT, f"DIAG_coord_img{i}.png")}
    t = time.time()
    r = native_driver.run_and_wait("image", p, timeout=600)
    return time.time() - t, r


def main() -> int:
    from core import gpu_registry
    print("=== COORDINAZIONE FOTO→VIDEO (app-completa) ===", flush=True)
    w = H.boot_full_app()
    fails = []

    def check(name, cond):
        print(("OK  " if cond else "FAIL") + " " + name, flush=True)
        if not cond:
            fails.append(name)

    # 1) foto FLUX → modello residente su :8189
    _dt, r1 = _gen_img(1, 11)
    vram_after_img = H.vram_used_mb()
    print(f"[foto1] status={r1.get('status')} VRAM usata={vram_after_img}MB", flush=True)
    check("foto carica il modello (VRAM > 20GB)", vram_after_img > 20000)

    # 2) arriva un job VIDEO → il dispatch VERO chiama free_video_vram (before_job + ATTESA del
    #    rilascio EFFETTIVO via NVML). NIENTE time.sleep: la garanzia nuova è che al ritorno la
    #    VRAM sia GIÀ libera (prima serviva uno sleep(3) — prova che il ritorno di before_job
    #    non significava VRAM libera; in produzione quello sleep non c'era).
    plan = gpu_registry.before_job("video", arch="video", log=print)
    print(f"[registry] piano video: {plan.get('action')} — {plan.get('reason')}", flush=True)
    check("il registro ha deciso lo sfratto immagine", 8189 in (plan.get("free_ports") or []))
    from core import ltx_video_job
    free_mb = ltx_video_job.free_video_vram()
    vram_after_free = H.vram_used_mb()
    print(f"[dopo free_video_vram] libera={free_mb}MB, usata={vram_after_free}MB", flush=True)
    check("foto→video libera il modello immagine (VRAM < 5GB, SENZA sleep)", vram_after_free < 5000)

    # 3) foto stesso modello dopo il video → il registro decide 'load' (era stato liberato)
    _dt2, r3 = _gen_img(2, 22)
    check("foto2 rigenera dopo lo sfratto (success)", r3.get("status") == "success")

    H.teardown(w)
    ok = not fails
    print(("=== COORD FOTO↔VIDEO OK ===" if ok else f"=== FAIL: {fails} ==="), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
