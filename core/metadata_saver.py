"""Sidecar JSON accanto ad ogni file generato da LTX Studio.

Il sidecar ha lo stesso nome del file con estensione .json (es. video.mp4 → video.mp4.json).
Contiene tutti i parametri di generazione per consentire il riutilizzo tramite Navigator Gallery.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Tutti i sidecar in UNA cartella dedicata (accoppiati per NOME all'output), indipendente da dove sta l'output.
OUTPUT_ROOT = Path(r"E:\AI\ltxstudio\output")
SIDECAR_DIR = OUTPUT_ROOT / "sidecars"


class MetadataSaver:
    """Salva e carica sidecar JSON per i file generati (in output/sidecars/)."""

    @staticmethod
    def sidecar_path(output_path: str | Path) -> Path:
        """Path del sidecar (.json) per un dato output — in output/sidecars/<nome_output>.json.
        Retro-compat: se esiste il vecchio sidecar adiacente (<output>.json) lo si continua a leggere."""
        return SIDECAR_DIR / (Path(output_path).name + ".json")

    @staticmethod
    def _legacy_sidecar(output_path: str | Path) -> Path:
        return Path(str(output_path) + ".json")

    @staticmethod
    def save_sidecar(output_path: str | Path, meta: dict) -> Path:
        """Salva il sidecar JSON in output/sidecars/ (accoppiato per nome all'output).

        Args:
            output_path: path completo del file generato (mp4, png, wav, glb, …)
            meta: dizionario con i parametri di generazione (riutilizzabili nel Navigator)
        """
        SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
        sp = MetadataSaver.sidecar_path(output_path)
        payload = {
            "ltxstudio_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_file": str(Path(output_path).name),
            "output_path": str(output_path),
            **meta,
        }
        sp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return sp

    @staticmethod
    def load_sidecar(output_path: str | Path) -> dict | None:
        """Carica il sidecar JSON (output/sidecars/) se esiste; fallback al vecchio adiacente. None se assente."""
        for sp in (MetadataSaver.sidecar_path(output_path), MetadataSaver._legacy_sidecar(output_path)):
            if sp.exists():
                try:
                    return json.loads(sp.read_text(encoding="utf-8"))
                except Exception:                            # noqa: BLE001
                    continue
        return None

    # estensioni media mostrate in galleria (immagini, video, audio)
    _MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp",
                   ".mp4", ".webm", ".mkv", ".mov", ".gif",
                   ".mp3", ".wav", ".flac", ".ogg", ".m4a"}

    @staticmethod
    def list_outputs_with_sidecar(output_dir: str | Path) -> list[Path]:
        """Elenca TUTTI i media di output (immagini/video/audio), RICORSIVO nelle sottocartelle
        (audio/music/tts), ordinati per data (più recente prima). I file con sidecar JSON mostrano i
        metadata; gli altri (es. generati da test/backend) appaiono comunque — così l'utente VEDE tutto."""
        d = Path(output_dir)
        if not d.exists():
            return []
        results = []
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            n = f.name.lower()
            if n.endswith(".json") or n.startswith("_") or "/.cache/" in f.as_posix().lower():
                continue                                    # salta sidecar, temp (_*), cache
            if f.suffix.lower() in MetadataSaver._MEDIA_EXTS:
                results.append(f)
        results.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return results

    @staticmethod
    def build_meta_from_vals(vals: dict, loras: list[dict], audio: dict,
                             cnet: dict, template_name: str,
                             prompt_id: str, image_path: str | None = None) -> dict:
        """Costruisce il dizionario meta da passare a save_sidecar() dalla struttura dati di MainWindow."""
        return {
            "prompt_id": prompt_id,
            "generation_type": "i2v" if image_path else "t2v",
            "template": template_name,
            # Prompt
            "prompt_positive": vals.get("prompt_positive", ""),
            "prompt_negative": vals.get("prompt_negative", ""),
            # Risoluzione e timing
            "width": vals.get("width", 0),
            "height": vals.get("height", 0),
            "fps": vals.get("fps", 24),
            "length_s": vals.get("length_s", 0),
            # Sampler
            "seed": vals.get("seed", 0),
            "steps": vals.get("steps", 20),
            "cfg": vals.get("cfg", 3.0),
            "sampler": vals.get("sampler", ""),
            "scheduler": vals.get("scheduler", ""),
            # Modello
            "model": vals.get("model", ""),
            "model_kind": vals.get("model_kind", "auto"),
            # Modalità
            "nsfw_mode": vals.get("nsfw_mode", "sfw"),
            "audio_mode": audio.get("audio_mode", "native"),
            # ControlNet
            "controlnet_enabled": cnet.get("controlnet_enabled", False),
            "controlnet_type": cnet.get("controlnet_type", 0),
            "controlnet_ref_video": cnet.get("controlnet_ref_video"),
            # LoRA
            "loras": [{"name": l["name"], "strength": l.get("strength", 1.0)} for l in loras],
            # Frame di partenza
            "frame1_image": image_path,
        }
