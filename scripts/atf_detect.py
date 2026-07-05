"""ATF Detector (#107) — verifica se un'immagine o un video è stato generato con AI Tools Forge.

Rileva la firma steganografica invisibile (core/stego.py) su QUALSIASI fonte: file, screenshot, o anche
una foto dello schermo (frontale). Non serve l'originale: rilevamento cieco via correlazione con la chiave.

Uso:
  python scripts/atf_detect.py <file>              # immagine o video
  python scripts/atf_detect.py <file> --json
  python scripts/atf_detect.py --dir <cartella>    # scansiona una cartella

Exit code: 0 = firma trovata, 1 = non trovata, 2 = errore.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import stego  # noqa: E402


def _one(path: str) -> dict:
    found, score, nonce, text = stego.detect_file(path)
    return {"file": path, "generated_by_atf": bool(found),
            "confidence_z": round(float(score), 2), "signal_id": int(nonce),
            "custom_text": text}


def main() -> int:
    ap = argparse.ArgumentParser(description="ATF Detector — firma di provenienza AI Tools Forge")
    ap.add_argument("path", nargs="?", help="immagine o video da analizzare")
    ap.add_argument("--dir", help="scansiona tutti i media di una cartella")
    ap.add_argument("--json", action="store_true", help="output JSON")
    args = ap.parse_args()

    targets: list[str] = []
    if args.dir:
        for root, _, files in os.walk(args.dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in (stego._IMG_EXT | stego._VID_EXT):
                    targets.append(os.path.join(root, f))
    elif args.path:
        targets = [args.path]
    else:
        ap.print_help(); return 2

    results = [_one(t) for t in targets]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            mark = "✅ GENERATO con AI Tools Forge" if r["generated_by_atf"] else "❌ nessuna firma"
            txt = f'  testo: "{r["custom_text"]}"' if r.get("custom_text") else ""
            print(f"{mark}  (z={r['confidence_z']}, id={r['signal_id']})  {r['file']}{txt}")
    any_found = any(r["generated_by_atf"] for r in results)
    return 0 if any_found else 1


if __name__ == "__main__":
    sys.exit(main())
