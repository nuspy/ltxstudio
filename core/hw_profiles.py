"""Profili hardware — auto-calibrazione per fascia VRAM/RAM (distribuzione su PC eterogenei).

Rileva la VRAM (nvidia-smi, senza dipendere da torch nel venv GUI) e la RAM, sceglie un profilo
(8/12/16/24/32/40/48 GB) e produce le variabili d'ambiente che il backend nativo legge
(NATIVE_GPU_MARGIN, NATIVE_GPU_FLOOR_MB, NATIVE_FORCE_OFFLOAD, NATIVE_MAX_RES, NATIVE_ATTENTION...).
Applicato all'avvio del server nativo (core/native_launcher). Vedi [[dynamic-vram-tuning]].
"""
from __future__ import annotations

import subprocess


def detect_vram_gb() -> float:
    """VRAM totale della prima GPU in GB (0 se non rilevabile)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
        mb = int((out.stdout or "0").splitlines()[0].strip())
        return round(mb / 1024, 1)
    except Exception:                                       # noqa: BLE001
        return 0.0


def detect_ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:                                       # noqa: BLE001
        try:
            import os
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
        except Exception:                                  # noqa: BLE001
            return 0.0


# Profili per fascia di VRAM. `tier` = soglia minima GB. Ordinati dal più piccolo.
_PROFILES = [
    # tier  nome        env del profilo
    (0,   "cpu",      {"NATIVE_FORCE_OFFLOAD": "1", "NATIVE_MAX_RES": "768", "NATIVE_ATTENTION": "sdpa"}),
    (8,   "8GB",      {"NATIVE_FORCE_OFFLOAD": "1", "NATIVE_MAX_RES": "768",  "NATIVE_FMT_HINT": "gguf_q4", "NATIVE_ATTENTION": "auto"}),
    (12,  "12GB",     {"NATIVE_FORCE_OFFLOAD": "1", "NATIVE_MAX_RES": "1024", "NATIVE_FMT_HINT": "gguf_q5", "NATIVE_ATTENTION": "auto"}),
    # MARGINE basso: la stima VRAM è ora PRECISA (byte reali dell'header/cache footprint) e include già il
    # working-set → il margine copre solo l'incertezza di misura, non il grosso overhead di prima. Meno
    # conservativo = più modelli restano GPU-resident (veloci) invece che offloadati inutilmente.
    (16,  "16GB",     {"NATIVE_GPU_MARGIN": "1.15", "NATIVE_GPU_FLOOR_MB": "1500", "NATIVE_MAX_RES": "1024", "NATIVE_FMT_HINT": "fp8", "NATIVE_ATTENTION": "auto"}),
    (24,  "24GB",     {"NATIVE_GPU_MARGIN": "1.12", "NATIVE_GPU_FLOOR_MB": "1200", "NATIVE_MAX_RES": "1280", "NATIVE_FMT_HINT": "fp8", "NATIVE_ATTENTION": "auto"}),
    (32,  "32GB",     {"NATIVE_GPU_MARGIN": "1.10", "NATIVE_GPU_FLOOR_MB": "1200", "NATIVE_MAX_RES": "1536", "NATIVE_FMT_HINT": "fp8_nvfp4", "NATIVE_ATTENTION": "auto"}),
    (40,  "40GB",     {"NATIVE_GPU_MARGIN": "1.08", "NATIVE_GPU_FLOOR_MB": "1200", "NATIVE_MAX_RES": "2048", "NATIVE_FMT_HINT": "fp8_nvfp4", "NATIVE_ATTENTION": "auto"}),
    (48,  "48GB+",    {"NATIVE_GPU_MARGIN": "1.06", "NATIVE_GPU_FLOOR_MB": "1200", "NATIVE_MAX_RES": "2048", "NATIVE_FMT_HINT": "bf16_fp8", "NATIVE_ATTENTION": "auto"}),
]


def profile_for(vram_gb: float) -> tuple[str, dict]:
    """Ritorna (nome_profilo, env) per la VRAM data."""
    name, env = _PROFILES[0][1:][0], _PROFILES[0][2]
    chosen = _PROFILES[0]
    for tier, nm, e in _PROFILES:
        if vram_gb >= tier:
            chosen = (tier, nm, e)
    return chosen[1], dict(chosen[2])


def detect_profile() -> dict:
    """Rileva HW e ritorna {name, vram_gb, ram_gb, env}. `env` da applicare al server nativo."""
    vram = detect_vram_gb()
    ram = detect_ram_gb()
    name, env = profile_for(vram)
    # RAM scarsa (<24GB): forza offload comunque per evitare picchi RAM nel caricamento
    if ram and ram < 24:
        env["NATIVE_FORCE_OFFLOAD"] = "1"
    return {"name": name, "vram_gb": vram, "ram_gb": ram, "env": env}
