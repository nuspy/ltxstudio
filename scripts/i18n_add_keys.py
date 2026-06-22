"""Applica chiavi i18n ai dizionari di ui/i18n/.

Input: un file JSON {lang: {key: value, ...}, ...}. Per ogni lingua presente, carica
ui/i18n/<lang>.json, aggiorna/aggiunge le chiavi indicate (preservando l'ordine esistente
e l'UTF-8) e riscrive il file. Idempotente.

Uso:
  python scripts/i18n_add_keys.py <file_traduzioni.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

I18N_DIR = Path(__file__).resolve().parents[1] / "ui" / "i18n"


def apply(data: dict) -> None:
    for lang, kv in data.items():
        f = I18N_DIR / f"{lang}.json"
        if not f.exists():
            print(f"[skip] {lang}: file inesistente"); continue
        d = json.loads(f.read_text("utf-8"))
        added = 0
        for k, v in kv.items():
            if k not in d:
                added += 1
            d[k] = v
        f.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(f"[ok] {lang}: +{added} nuove, {len(kv)} totali applicate")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python scripts/i18n_add_keys.py <file.json>"); raise SystemExit(2)
    apply(json.loads(Path(sys.argv[1]).read_text("utf-8")))
