"""Audit del registro dimensioni modelli ALL'APERTURA (F2 unified-model-manager).

Richiesta utente: all'avvio verificare se i modelli utilizzabili sono già tutti CLASSIFICATI
(footprint VRAM noto); quelli non classificati vanno poi misurati DAVVERO (livello-2, caricandoli
in VRAM — vedi core/vram_calibrator) e con quei dati si pianifica cosa caricare/scaricare.

Qui il livello-1 (istantaneo, nessun load, zero GPU): migrazione legacy → scan del detector
(arch/precision da header) → calibrate_all (footprint da header per i mancanti) → classificazione.
Gira in un thread all'avvio della GUI (nessun torch: solo header e file JSON — sicuro nel processo
GUI, che non deve MAI toccare torch.cuda).
"""
from __future__ import annotations

import os

# famiglia → venv di misura livello-2. Solo image (inference_server) e ltx video (ltx2_venv) hanno
# un measure-worker (F7); le altre famiglie convergono on-first-use e restano fuori da to_measure.
_MEASURE_VENV = {
    "flux1": "inference_server", "flux2": "inference_server", "zimage": "inference_server",
    "ernie": "inference_server", "krea2": "", "hidream": "inference_server",
    "ltxv": "ltx2", "ltx2": "ltx2", "ltx": "ltx2",
}

# radici dei file AUSILIARI (F8): VAE/TextEncoder/upscaler DA SOLI non vanno misurati col loader
# di pipeline (un `ae.safetensors` VAE-only caricato come pipeline FLUX fallisce per definizione —
# visto live 22/07: 'saltato exit 2'). Per loro l'header è già byte-esatto (bf16, nessun upcast).
_AUX_ROOTS = (r"E:\AI\Models\TextEncoders", r"E:\AI\Models\VAE", r"E:\AI\Models\latent_upscale_models")


def _is_aux(path: str) -> bool:
    p = os.path.normcase(os.path.abspath(path))
    return any(p.startswith(os.path.normcase(r) + os.sep) for r in _AUX_ROOTS)


def _detect(path: str) -> dict:
    try:
        from core import model_detect
        return model_detect.get(path) or {}
    except Exception:                                       # noqa: BLE001
        return {}


def audit(on_progress=None) -> dict:
    """Esegue l'audit livello-1 e classifica ogni modello. Ritorna:
    {"total", "migrated", "measured": [...], "estimated": [...], "unknown": [...],
     "to_measure": [...]} — to_measure = candidati alla misura reale (livello-2), con
     {path, name, kind, family, size_mb, est_mb, venv}."""
    from core import vram_calibration, vram_estimate

    migrated = vram_estimate.migrate_legacy()
    try:                                                    # arch/precision di tutti i checkpoint (cache-aware)
        from core import model_detect
        model_detect.scan()
    except Exception:                                       # noqa: BLE001
        pass
    vram_calibration.calibrate_all(on_progress=on_progress)  # header per i non-cache-ati (incrementale)

    measured, estimated, unknown, to_measure = [], [], [], []
    for m in vram_calibration.enumerate_models():
        det = _detect(m["path"]) if not m["is_gguf"] else {}
        fam = str(det.get("family") or "")
        kind = str(det.get("kind") or "")
        e = (vram_estimate._entry(m["path"], "fp8") or      # noqa: SLF001 — helper condiviso
             vram_estimate._entry(m["path"]))               # noqa: SLF001
        mb = int(e.get("footprint_mb", 0) or 0)
        src = str(e.get("source") or "")
        info = {"path": m["path"], "name": m["name"], "kind": kind, "family": fam,
                "size_mb": m["size_mb"], "est_mb": mb,
                "venv": _MEASURE_VENV.get(fam, "")}
        if mb <= 0:
            unknown.append(info)
        elif src == "measured":
            measured.append(info)
        else:                                               # header/gguf/voce legacy senza source = stima
            estimated.append(info)
        # candidato livello-2: non ancora misurato, misurabile (venv noto), non marcato skipped.
        # ESCLUSI: GGUF (footprint ≈ dim file), file AUSILIARI (VAE/TE soli: header già esatto,
        # e il loader di pipeline non può caricarli — il detector li scambia per checkpoint),
        # ADAPTER (Redux & co.: non sono pipeline) e SHARD (model-0000X-of-N: non caricabili soli).
        import re as _re
        _is_shard = bool(_re.match(r"(?i)(model|diffusion_pytorch_model)-\d+-of-\d+", m["name"]))
        if mb <= 0 or (src != "measured" and not e.get("skipped_commit")):
            if (info["venv"] and not m["is_gguf"] and not _is_aux(m["path"])
                    and str(det.get("role") or "") != "adapter" and not _is_shard):
                to_measure.append(info)
    return {"total": len(measured) + len(estimated) + len(unknown), "migrated": migrated,
            "measured": measured, "estimated": estimated, "unknown": unknown,
            "to_measure": to_measure}


def summary_line(res: dict) -> str:
    """Riga compatta per statusbar/log (i18n a carico del chiamante GUI)."""
    return (f"{res['total']} modelli: {len(res['measured'])} misurati, "
            f"{len(res['estimated'])} stimati, {len(res['unknown'])} sconosciuti, "
            f"{len(res['to_measure'])} da misurare")


if __name__ == "__main__":                                  # uso CLI/diagnostico
    r = audit(on_progress=lambda i, n, name, mb: print(f"  [{i}/{n}] {name}: {mb}MB", flush=True))
    print(summary_line(r))
    for x in r["to_measure"]:
        print(f"  da misurare: {x['name']} (~{x['est_mb'] or x['size_mb']}MB, venv={x['venv']})")
