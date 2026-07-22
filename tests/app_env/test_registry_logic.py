"""Test della LOGICA del registro GPU (core/gpu_registry.plan) — pura, con residents()/VRAM mockati.
Valida le regole richieste: riusa se residente, tieni i condivisi, sfratta solo su cambio-categoria o
mancanza spazio. (L'integrazione con l'app-completa arriva quando i motori sono persistenti.)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import gpu_registry as R  # noqa: E402


def _setup(res, free_mb):
    R.residents = lambda: res
    R.gpu_mem.free_vram_mb = lambda: free_mb


def run() -> int:
    K_flux_a = R.ModelKey("flux", "modelA", "fp8")
    K_flux_b = R.ModelKey("flux", "modelB", "fp8")
    K_ltx = R.ModelKey("ltx", "ltx2", "fp8")
    fails = []

    def check(name, cond):
        print(("OK  " if cond else "FAIL") + " " + name, flush=True)
        if not cond:
            fails.append(name)

    # 1) GPU vuota → load (nessuno sfratto)
    _setup([], 30000)
    p = R.plan(K_flux_a, category="image", need_mb=13000)
    check("gpu vuota → load", p["action"] == "load" and not p["free_ports"])

    # 2) stesso modello residente → reuse (niente reload)
    _setup([R.Resident(port=8191, category="image", key=str(K_flux_a), vram_mb=13000)], 17000)
    p = R.plan(K_flux_a, category="image", need_mb=13000)
    check("stesso modello → reuse", p["action"] == "reuse" and p["reuse_port"] == 8191)

    # 3) foto→video (cambio categoria): il server immagine va liberato
    _setup([R.Resident(port=8191, category="image", key=str(K_flux_a), vram_mb=13000)], 17000)
    p = R.plan(K_ltx, category="video", need_mb=22000)
    check("foto→video → evict image", p["action"] == "evict_load" and 8191 in p["free_ports"])

    # 4) foto-A→foto-B (stessa categoria, chiave diversa) con SPAZIO → tieni A, carica B (no sfratto)
    _setup([R.Resident(port=8191, category="image", key=str(K_flux_a), vram_mb=6000)], 25000)
    p = R.plan(K_flux_b, category="image", need_mb=6000)
    check("foto-A→foto-B con spazio → load (tieni A)", p["action"] == "load" and not p["free_ports"])

    # 5) foto-A→foto-B SENZA spazio → sfratta A (cambio-modello + pressione)
    _setup([R.Resident(port=8191, category="image", key=str(K_flux_a), vram_mb=13000)], 4000)
    p = R.plan(K_flux_b, category="image", need_mb=13000)
    check("foto-A→foto-B senza spazio → evict A", p["action"] == "evict_load" and 8191 in p["free_ports"])

    # 6) F5 SIZE-AWARE: due residenti stessa categoria, deficit coperto dal PICCOLO → sfratta SOLO lui
    _setup([R.Resident(port=8191, category="image", key=str(K_flux_a), vram_mb=4000),
            R.Resident(port=8189, category="image", key="image:modelC", vram_mb=14000)], 3000)
    p = R.plan(K_flux_b, category="image", need_mb=4000)     # servono 4000+2000, liberi 3000 → mancano 3000
    check("sfratto size-aware: solo il più piccolo che basta",
          p["action"] == "evict_load" and p["free_ports"] == [8191])

    # 7) F5: deficit grande → sfratta ENTRAMBI (piccolo prima, poi il grande) e si ferma lì
    _setup([R.Resident(port=8191, category="image", key=str(K_flux_a), vram_mb=4000),
            R.Resident(port=8189, category="image", key="image:modelC", vram_mb=14000)], 2000)
    p = R.plan(K_flux_b, category="image", need_mb=18000)
    check("sfratto size-aware: entrambi se il deficit lo richiede",
          p["action"] == "evict_load" and set(p["free_ports"]) == {8191, 8189})

    # 8) F5: chiave video NON degenere → il riuso video viene riconosciuto
    K_vid = R.ModelKey("ltx", "ltx-2.3-22b-dev-fp8.safetensors", "fp8")
    _setup([R.Resident(port=8190, category="video", key=str(K_vid), vram_mb=22000)], 8000)
    p = R.plan(K_vid, category="video", need_mb=22000)
    check("video: stessa chiave → reuse (niente reload)", p["action"] == "reuse" and p["reuse_port"] == 8190)

    print(("=== REGISTRY LOGIC OK ===" if not fails else f"=== FAIL: {fails} ==="), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(run())
