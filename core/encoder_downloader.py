"""Auto-download degli encoder testo FLUX.2 mancanti (repo ufficiali Comfy-Org, via huggingface_hub).

Le varianti FLUX.2 usano encoder diversi (vedi core/flux2_encoder): se quello richiesto da un modello
non è installato, lo scarichiamo dal repo Comfy-Org e lo copiamo nella cartella text_encoders di
ComfyUI. Il nome locale contiene la keyword che core/flux2_encoder cerca, così dopo il download
resolve_encoder lo trova subito.

Sorgenti (verificate su HuggingFace):
- 7680  (Qwen3-4B, Klein-4B): Comfy-Org/vae-text-encorder-for-flux-klein-4b
- 12288 (Qwen3-8B, Klein-9B): Comfy-Org/vae-text-encorder-for-flux-klein-9b  (variante fp8mixed, 8.7GB)
"""
from __future__ import annotations

import shutil
from pathlib import Path

# Root text_encoders di ComfyUI (da extra_model_paths.yaml). Sottocartella Flux per coerenza con
# mistral_3_small_flux2.safetensors già presente lì.
TEXT_ENCODERS_DIR = Path(r"E:\AI\Models\TextEncoders\Flux")

# dim txt_in → (repo_id, file_nel_repo, nome_locale, GB_approx, etichetta)
FLUX2_ENCODER_SOURCES: dict[int, tuple[str, str, str, float, str]] = {
    7680: ("Comfy-Org/vae-text-encorder-for-flux-klein-4b",
           "split_files/text_encoders/qwen_3_4b.safetensors",
           "qwen_3_4b.safetensors", 8.0, "Qwen3-4B (FLUX.2 Klein-4B)"),
    12288: ("Comfy-Org/vae-text-encorder-for-flux-klein-9b",
            "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
            "qwen_3_8b_fp8mixed.safetensors", 8.7, "Qwen3-8B fp8 (FLUX.2 Klein-9B)"),
}


def source_for(indim: int | None):
    """(repo_id, file_nel_repo, nome_locale, GB, etichetta) per la dim, o None se sconosciuta."""
    return FLUX2_ENCODER_SOURCES.get(indim) if indim is not None else None


def comfy_relative_name(local_name: str) -> str:
    """Nome con cui ComfyUI elenca il file (relativo alla root text_encoders)."""
    return f"Flux\\{local_name}"


def download(indim: int, *, dest_dir: Path = TEXT_ENCODERS_DIR, log=print) -> tuple[bool, str, str | None]:
    """Scarica l'encoder per `indim` e lo copia in `dest_dir`.

    Ritorna (ok, messaggio, nome_relativo_comfy | None). Idempotente: se il file esiste già (grande),
    non riscarica. Bloccante (chiamare da un thread): hf_hub_download è resumable.
    """
    src = FLUX2_ENCODER_SOURCES.get(indim)
    if not src:
        return False, f"nessuna fonte nota per l'encoder txt_in={indim}", None
    repo, repo_file, local_name, gb, label = src
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / local_name
    rel = comfy_relative_name(local_name)
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        return True, f"encoder già presente: {dest.name}", rel
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:                       # noqa: BLE001
        return False, f"huggingface_hub non disponibile nel venv: {e}", None
    try:
        log(f"[FLUX.2] download {label} (~{gb:.1f} GB) da {repo} … (resumable)")
        cached = hf_hub_download(repo_id=repo, filename=repo_file)
        shutil.copy2(cached, dest)
        log(f"[FLUX.2] encoder pronto: {dest}")
        return True, f"scaricato: {local_name}", rel
    except Exception as e:                        # noqa: BLE001
        return False, f"download fallito: {e}", None
