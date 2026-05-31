"""Maschere persistenti per foto (Fase 3).

Per ogni immagine (chiave = sha1 del path) salva su richiesta: PNG maschera + JSON con
la trasformazione (pan/zoom/crop) e il formato video. Auto-ricarica quando la stessa
immagine viene ricaricata; reset a comando.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

STORE_DIR = Path(r"E:\AI\ltxstudio\data\masks")


def _key(image_path: str) -> str:
    return hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:16]


class MaskStore:
    def __init__(self, store_dir: Path = STORE_DIR):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, image_path: str):
        k = _key(image_path)
        return self.dir / f"{k}.png", self.dir / f"{k}.json"

    def save(self, image_path: str, mask_png_path: str, transform: dict) -> None:
        mpng, mjson = self._paths(image_path)
        shutil.copyfile(mask_png_path, mpng)
        mjson.write_text(json.dumps({"image": image_path, "transform": transform}, indent=2),
                         encoding="utf-8")

    def load(self, image_path: str) -> dict | None:
        """Ritorna {'mask_png': path, 'transform': {...}} se esiste, altrimenti None."""
        mpng, mjson = self._paths(image_path)
        if mpng.exists() and mjson.exists():
            data = json.loads(mjson.read_text(encoding="utf-8"))
            return {"mask_png": str(mpng), "transform": data.get("transform", {})}
        return None

    def reset(self, image_path: str) -> None:
        for p in self._paths(image_path):
            if p.exists():
                p.unlink()

    def has(self, image_path: str) -> bool:
        mpng, mjson = self._paths(image_path)
        return mpng.exists() and mjson.exists()
