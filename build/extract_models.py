"""build/extract_models.py — CENSIMENTO assistito dei modelli cablati nel codice.

Individuare "quale modello serve a quale modulo" è difficile perché i path sono sparsi nei service. Questo
strumento scandaglia core/ workers/ scripts/ per i riferimenti a file-modello (path assoluti, costanti note,
repo HuggingFace) e produce una lista di CANDIDATI {file, line, ref, guessed_type, guessed_module} da CURARE
nel manifest `data/weights_catalog.json`. NON scarica nulla, NON tocca file: stampa/scrive un report JSON.

Uso:
  python build/extract_models.py                 # stampa il report
  python build/extract_models.py --json out.json # scrive il report
  python build/extract_models.py --missing       # solo i candidati NON già coperti dal catalogo (per completezza)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["core", "workers", "scripts", "inference_server"]

# path a file-modello (assoluti Windows o relativi) con estensioni pesi
_EXT = r"(?:safetensors|gguf|ckpt|pth|pt|bin|onnx)"
_PAT_FILE = re.compile(r"""["'`]([^"'`\n]*?[\\/][^"'`\n]*?\.""" + _EXT + r""")["'`]""", re.IGNORECASE)
# repo HuggingFace "org/name" (in from_pretrained/snapshot_download/hf_hub_download o costanti)
_PAT_HF = re.compile(r"""["'`]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)["'`]""")
_HF_HINT = re.compile(r"(from_pretrained|snapshot_download|hf_hub_download|repo_id|HF_REPO|_CKPT|gemma_root|GEMMA|UPSAMPLER)", re.IGNORECASE)

# euristiche tipo/modulo dal path
_TYPE_HINTS = [
    ("lora", "LoRA"), ("controlnet", "ControlNet"), ("vae", "VAE"), ("upscal", "Upscale"),
    ("text_encoder", "TextEncoder"), ("clip", "TextEncoder"), ("gemma", "TextEncoder"),
    ("facerestore", "FaceRestore"), ("codeformer", "FaceRestore"), ("gfpgan", "FaceRestore"),
    ("gguf", "GGUF"), (".safetensors", "Checkpoint"),
]
_MODULE_HINTS = [
    ("ltx", "Video"), ("wan", "Video"), ("flux", "Immagini"), ("z_image", "Immagini"), ("zimage", "Immagini"),
    ("ernie", "Immagini"), ("hidream", "Immagini"), ("trellis", "Modello 3D (TRELLIS)"),
    ("skintoken", "Rig 3D"), ("world", "Mondo 3D (WorldGen)"), ("heartmula", "Musica (HeartMuLa)"),
    ("heartlib", "Musica (HeartMuLa)"), ("ace-step", "Musica (ACE-Step)"), ("acestep", "Musica (ACE-Step)"),
    ("mmaudio", "Musica (MMAudio)"), ("cosyvoice", "Voce (clone)"), ("f5", "Voce (clone)"),
    ("seedvr2", "Upscale"), ("flashvsr", "Upscale"), ("pulid", "Face"), ("insightface", "Face"),
    ("sam", "Segmentazione"), ("grounding", "Segmentazione"),
]


def _guess(text: str, hints) -> str:
    t = text.lower()
    for k, v in hints:
        if k in t:
            return v
    return "?"


def scan() -> list[dict]:
    out, seen = [], set()
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for py in base.rglob("*.py"):
            try:
                lines = py.read_text("utf-8", errors="ignore").splitlines()
            except Exception:                               # noqa: BLE001
                continue
            for i, ln in enumerate(lines, 1):
                for m in _PAT_FILE.finditer(ln):
                    ref = m.group(1)
                    key = ("file", ref)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"file": f"{d}/{py.relative_to(base).as_posix()}", "line": i, "ref": ref,
                                "kind": "file", "type": _guess(ref, _TYPE_HINTS), "module": _guess(ref, _MODULE_HINTS)})
                if _HF_HINT.search(ln):
                    for m in _PAT_HF.finditer(ln):
                        ref = m.group(1)
                        if "/" not in ref or ref.count("/") > 1 or ref.endswith((".py", ".json", ".txt")):
                            continue
                        key = ("hf", ref)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append({"file": f"{d}/{py.relative_to(base).as_posix()}", "line": i, "ref": ref,
                                    "kind": "hf", "type": _guess(ref, _TYPE_HINTS), "module": _guess(ref, _MODULE_HINTS)})
    return sorted(out, key=lambda r: (r["module"], r["type"], r["ref"]))


def _catalog_refs() -> set[str]:
    try:
        cat = json.loads((ROOT / "data" / "weights_catalog.json").read_text("utf-8")).get("weights", [])
    except Exception:                                       # noqa: BLE001
        return set()
    refs = set()
    for e in cat:
        if e.get("repo_id"):
            refs.add(e["repo_id"].lower())
        for n in e.get("scan_names", []):
            refs.add(str(n).lower())
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--missing", action="store_true", help="solo i candidati non già nel catalogo")
    a = ap.parse_args()
    rows = scan()
    if a.missing:
        cat = _catalog_refs()
        def _covered(r):
            base = r["ref"].replace("\\", "/").split("/")[-1].lower()
            stem = base.rsplit(".", 1)[0]
            return any(c in stem or stem in c or (r["kind"] == "hf" and r["ref"].lower() == c) for c in cat)
        rows = [r for r in rows if not _covered(r)]
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        print(f"{len(rows)} candidati → {a.json}")
    else:
        for r in rows:
            print(f"[{r['module']:>22}] {r['type']:<12} {r['kind']:<4} {r['ref']}  ({r['file']}:{r['line']})")
        print(f"\n{len(rows)} candidati modello.")


if __name__ == "__main__":
    main()
