"""Backend di attenzione per il video LTX: flash (SDPA di PyTorch, ESATTO) vs Sage (fp8, APPROSSIMATO).

I due backend occupano lo STESSO slot — entrambi rimpiazzano `F.scaled_dot_product_attention`; se ne usa
UNO solo (Sage via monkeypatch in workers/ltx2_infer._patch_sage_attention e in ltx_comfy_native; flash è
già dentro la SDPA di torch). Ne bastano quindi DUE punti (i due worker LTX, per cui passano TUTTI i
percorsi video: tab Video, videoclip, wizard LoRA, Cinema Studio, retake, extend, Lucky Director…).

MECCANISMO: **niente variabili d'ambiente** — lo switch nella GUI scrive la preferenza in
data/user_prefs.json (`attention_backend` ∈ flash|sage|auto), e i worker leggono `ltx_use_sage()`
direttamente da qui (i worker aggiungono il repo a sys.path, quindi importano core.*). Un solo switch,
effetto ovunque.

Misura 17/07 su RTX 5090 (49f, 8 step): flash 127s vs Sage 178s → flash più veloce ED esatta su Blackwell.
DEFAULT = **flash**. Su GPU più vecchie la Sage può ancora aiutare (VRAM/velocità) → 'sage' o 'auto'.
"""
from __future__ import annotations

from core import user_prefs

DEFAULT = "flash"
CHOICES = ("flash", "sage", "auto")


def get() -> str:
    v = (user_prefs.get("attention_backend", DEFAULT) or DEFAULT).lower()
    return v if v in CHOICES else DEFAULT


def set(backend: str) -> None:
    b = (backend or DEFAULT).lower()
    user_prefs.set("attention_backend", b if b in CHOICES else DEFAULT)


def _is_blackwell_or_newer() -> bool:
    """GPU Blackwell (sm120) o più recente — senza creare un contesto CUDA (nvidia-smi)."""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=8)
        return float((out.stdout or "0").splitlines()[0].strip()) >= 12.0
    except Exception:                                        # noqa: BLE001
        return False


def ltx_use_sage() -> bool:
    """True se il worker LTX deve usare SageAttention (fp8); False = flash/SDPA (esatto). Lo leggono i
    worker ltx2_infer / ltx_comfy_native al posto della vecchia env LTX_SAGE."""
    b = get()
    if b == "sage":
        return True
    if b == "flash":
        return False
    return not _is_blackwell_or_newer()                      # auto: flash su Blackwell+, Sage altrove
