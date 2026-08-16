"""Auto-download degli encoder testo FLUX.2 mancanti — ora sul MOTORE UNICO (weights-unification F).

Le varianti FLUX.2 usano encoder diversi (vedi core/flux2_encoder): se quello richiesto da un
modello non è installato, lo scarichiamo. Prima: dict hardcoded (il 4° manifesto parallelo) +
hf_hub_download nella cache HF + copy2 verso la destinazione = doppio spazio su disco e zero
multi-connessione. Ora: le fonti sono le voci di CATALOGO `flux2-klein-te-4b/8b` e il download
passa da core/downloader (fast 16-connessioni, scrittura DIRETTA a destinazione, verifica
integrità). Il dialog di consenso resta in main_window (_ensure_flux2_encoder), invariato.
"""
from __future__ import annotations

from pathlib import Path

# Root text_encoders (coerente con mistral_3_small_flux2.safetensors già lì).
TEXT_ENCODERS_DIR = Path(r"E:\AI\Models\TextEncoders\Flux")

# dim txt_in → id di catalogo (weights_catalog v2: repo/remote_filename/dest tutti lì)
FLUX2_ENCODER_IDS: dict[int, str] = {
    7680: "flux2-klein-te-4b",                      # Qwen3-4B (FLUX.2 Klein-4B)
    12288: "flux2-klein-te-8b",                     # Qwen3-8B fp8mixed (FLUX.2 Klein-9B)
}


def _entry(indim: int | None) -> dict | None:
    if indim is None:
        return None
    from core import model_paths
    return model_paths.entry(FLUX2_ENCODER_IDS.get(indim, ""))


def source_for(indim: int | None):
    """Compat storica: (repo_id, file_nel_repo, nome_locale, GB, etichetta) o None."""
    e = _entry(indim)
    if not e:
        return None
    return (e.get("repo_id", ""), e.get("remote_filename", e.get("filename", "")),
            e.get("filename", ""), float(e.get("approx_gb", 0) or 0), e.get("label", e["id"]))


def comfy_relative_name(local_name: str) -> str:
    """Nome con cui ComfyUI elenca il file (relativo alla root text_encoders)."""
    return f"Flux\\{local_name}"


def download(indim: int, *, dest_dir: Path = TEXT_ENCODERS_DIR, log=print) -> tuple[bool, str, str | None]:
    """Scarica l'encoder per `indim` via il motore unico. Ritorna (ok, messaggio,
    nome_relativo_comfy | None). Idempotente; resume multi-connessione; bloccante (da thread)."""
    e = _entry(indim)
    if not e:
        return False, f"nessuna fonte nota per l'encoder txt_in={indim}", None
    local_name = e.get("filename", "")
    rel = comfy_relative_name(local_name)
    dest = Path(dest_dir) / local_name
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        return True, f"encoder già presente: {dest.name}", rel
    try:
        from core import downloader as dl
        log(f"[FLUX.2] download {e.get('label', e['id'])} (~{e.get('approx_gb', '?')} GB) "
            f"da {e.get('repo_id', '')} … (fast, resume)")
        out = dl.download_entry(e, log=log)
        log(f"[FLUX.2] encoder pronto: {out}")
        return True, f"scaricato: {local_name}", rel
    except Exception as ex:                       # noqa: BLE001
        return False, f"download fallito: {ex}", None
