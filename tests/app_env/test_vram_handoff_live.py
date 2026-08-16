"""VALIDAZIONE LIVE del passaggio VRAM tra pipeline (branch feature/vram-handoff-fix).

Sequenza REALE dall'app completa: foto FLUX → video LTX breve → foto → video. Verifica:
  1. foto→video: il video parte col dispatch vero (free+ATTESA) e produce l'mp4;
  2. video→foto: il /free demote di :8190 libera DAVVERO la VRAM (prima il "demote finto"
     lasciava il 22B residente) ma TIENE il parcheggio RAM;
  3. foto→video di nuovo: senza il vecchio guard `_last`, il server immagini viene liberato
     anche stavolta (era il bug video→foto→video) e il video2 riusa il park (load più veloce).

  <gui_python> tests/app_env/test_vram_handoff_live.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ["NATIVE_PERSISTENT_CHILD"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as H  # noqa: E402

MODEL_IMG = "FLUX_Dev.safetensors"
OUT = r"E:\ai\ltxstudio\output"


def _stats(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=5) as r:
            return json.load(r)
    except Exception:                                        # noqa: BLE001
        return {}


def _gen_img(i: int, seed: int):
    from core import native_driver
    p = {"arch": "flux", "model": MODEL_IMG,
         "prompt": ["a colossal abandoned lighthouse swallowed by a frozen glass wave at sunset, cinematic",
                    "a brass clockwork octopus conducting an orchestra of umbrellas in the rain, baroque light"][i % 2],
         "negative": "blurry", "width": 768, "height": 768, "steps": 8, "guidance": 3.5,
         "seed": seed, "output": os.path.join(OUT, f"DIAG_handoff_img{i}.png")}
    t = time.time()
    r = native_driver.run_and_wait("image", p, timeout=600)
    return time.time() - t, r


def _gen_video(i: int, seed: int):
    from core import ltx2_service
    p = {"prompt": ["a tiny robot bard playing a hurdy-gurdy on a mossy rooftop while paper dragons circle overhead, whimsical",
                    "an opera singer made of aurora light hitting the high note on a cliff, waves exploding in slow motion, epic"][i % 2],
         "negative": "worst quality, blurry", "width": 512, "height": 320, "num_frames": 33,
         "fps": 25, "steps": 8, "guidance": 3.0, "seed": seed, "arch": "ltx",
         "output": os.path.join(OUT, f"DIAG_handoff_vid{i}.mp4")}
    os.environ["LTX_ENGINE"] = "lightricks"                  # motore nativo (server :8190)
    t = time.time()
    r = ltx2_service.generate(p, timeout=900)
    return time.time() - t, r


def main() -> int:
    print("=== HANDOFF VRAM LIVE (foto→video→foto→video) ===", flush=True)
    w = H.boot_full_app()
    fails = []

    def check(name, cond):
        print(("OK  " if cond else "FAIL") + " " + name, flush=True)
        if not cond:
            fails.append(name)

    # 1) foto FLUX → residente su :8189
    dt_i1, r1 = _gen_img(1, 11)
    used1 = H.vram_used_mb()
    print(f"[foto1] {r1.get('status')} in {dt_i1:.0f}s — VRAM usata {used1}MB", flush=True)
    check("foto1 success", r1.get("status") == "success")
    check("foto1 modello residente (VRAM > 15GB)", used1 > 15000)

    # 2) video1 → dispatch vero (free_for + ATTESA) → mp4
    dt_v1, rv1 = _gen_video(1, 7)
    used2 = H.vram_used_mb()
    s89 = _stats(8189)
    print(f"[video1] {rv1.get('status')} in {dt_v1:.0f}s — VRAM usata {used2}MB, "
          f":8189 model_vram={s89.get('model_vram_mb')}MB", flush=True)
    check("video1 success (mp4)", rv1.get("status") == "success"
          and rv1.get("output") and os.path.exists(rv1["output"][0]))
    check("foto sfrattata durante il video (:8189 vram propria < 3GB)",
          float(s89.get("model_vram_mb") or 0) < 3000)

    # 3) foto2 → il registro sfratta il video: DEMOTE VERO su :8190 (VRAM propria bassa,
    #    pipeline fuori cache, parcheggio RAM tenuto)
    dt_i2, r2 = _gen_img(2, 22)
    s90 = _stats(8190)
    print(f"[foto2] {r2.get('status')} in {dt_i2:.0f}s — :8190 vram_propria={s90.get('vram_used_mb')}MB "
          f"cached={s90.get('cached_pipelines')} park={s90.get('parked_ram_mb')}MB", flush=True)
    check("foto2 success", r2.get("status") == "success")
    check("demote VERO :8190 (vram propria < 2GB)", float(s90.get("vram_used_mb") or 0) < 2000)
    check("demote :8190 ha svuotato la pipe cache", int(s90.get("cached_pipelines") or 0) == 0)
    check("parcheggio RAM :8190 mantenuto (> 1GB)", float(s90.get("parked_ram_mb") or 0) > 1024)

    # 4) video2 → SENZA il vecchio guard: il server immagini è liberato ANCHE stavolta e il
    #    video riparte dal park (niente SSD). video→foto→video era il caso rotto.
    dt_v2, rv2 = _gen_video(2, 8)
    s89b = _stats(8189)
    print(f"[video2] {rv2.get('status')} in {dt_v2:.0f}s (video1 {dt_v1:.0f}s) — "
          f":8189 vram propria={s89b.get('model_vram_mb')}MB", flush=True)
    check("video2 success (sequenza video→foto→video)", rv2.get("status") == "success"
          and rv2.get("output") and os.path.exists(rv2["output"][0]))
    check("foto sfrattata anche per il video2 (:8189 vram propria < 3GB)",
          float(s89b.get("model_vram_mb") or 0) < 3000)
    check("video2 non più lento del video1 (park/riuso, tolleranza +60s)", dt_v2 <= dt_v1 + 60)

    H.teardown(w)
    ok = not fails
    print(("=== HANDOFF LIVE OK ===" if ok else f"=== FAIL: {fails} ==="), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
