"""Fase 0 — spike: invia un golden prompt API a ComfyUI, mostra il progresso e
scarica l'output (video/immagini). Verifica che il template piatto giri SENZA frontend.

Uso:  python scripts/spike_submit.py templates/pluto_t2v.api.json [out_dir]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.comfy_driver import ComfyDriver  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    prompt_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("E:/AI/ltxstudio/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    # supporta sia {prompt:{...}} sia il prompt piatto
    if "prompt" in prompt and all(k.isdigit() for k in prompt["prompt"]):
        prompt = prompt["prompt"]

    drv = ComfyDriver()
    if not drv.is_up():
        print(f"ComfyUI non raggiungibile su {drv.cfg.http} — avvia run_nvidia.bat")
        sys.exit(2)

    done = {"flag": False}

    drv.on_progress = lambda v, m: print(f"\r  step {v}/{m}", end="", flush=True)
    drv.on_executing = lambda n: (None if n else print("\n  (nodo finito)"))
    drv.on_status = lambda d: None
    try:
        drv.start_ws()
    except RuntimeError as e:
        print(f"  (progress live disattivato: {e})")

    print(f"Invio {prompt_path.name} ({len(prompt)} nodi)...")
    pid = drv.submit(prompt)
    print(f"prompt_id = {pid}")

    # attende il completamento via history
    outputs = None
    for _ in range(36000):  # fino a ~1h
        time.sleep(2)
        hist = drv.history(pid)
        if pid in hist and hist[pid].get("status", {}).get("completed"):
            outputs = hist[pid]["outputs"]
            break
        if pid in hist and hist[pid].get("status", {}).get("status_str") == "error":
            print("\nERRORE esecuzione:", json.dumps(hist[pid]["status"], indent=2)[:800])
            drv.stop_ws()
            sys.exit(3)
    drv.stop_ws()

    if not outputs:
        print("\nTimeout senza output.")
        sys.exit(4)

    print("\nOutput:")
    saved = 0
    for node_id, out in outputs.items():
        for key in ("images", "gifs", "videos"):
            for item in out.get(key, []) or []:
                fn = item.get("filename")
                if not fn:
                    continue
                data = drv.download_view(fn, item.get("subfolder", ""), item.get("type", "output"))
                dst = out_dir / fn
                dst.write_bytes(data)
                print(f"  salvato: {dst}  ({len(data)/1e6:.1f} MB)")
                saved += 1
    print(f"\nFatto. File salvati: {saved}")


if __name__ == "__main__":
    main()
