"""Crea il PACCHETTO del componente MagicQuill V2 (codice derivato, CC BY-NC 4.0) da pubblicare fuori dall'app.

Contenuto (stessa struttura che `core/optional_components` cerca in `E:\\AI\\Models\\LoRA\\MagicQuill\\_code\\`):
    magicquill_v2.py, mq2/*.py, NOTICE.md, LICENSE-upstream-CC-BY-NC-4.0.txt, LICENSE-EasyControl-Apache-2.0.txt,
    INSTALL.txt, VERSION
Uscite: <out>/magicquill_code/ (cartella pronta, utile anche come snapshot HF) e <out>/mq2-code.zip.
Distribuzione consentita solo NON commerciale, con attribuzione (vedi NOTICE.md). Dove ospitarlo lo decide l'utente
(proposta: repo HF pubblico gratuito); poi in `build/catalog_task9.py` voce `mq2-code`: impostare `repo` e togliere `no_remote`.

Uso:  python scripts/package_magicquill_code.py [--out DIR]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "optional" / "magicquill"
FILES = ("magicquill_v2.py", "NOTICE.md", "LICENSE-upstream-CC-BY-NC-4.0.txt", "LICENSE-EasyControl-Apache-2.0.txt")
INSTALL = """MagicQuill V2 - componente codice (USO NON COMMERCIALE, CC BY-NC 4.0: vedi NOTICE.md)

Installazione per LTX Studio: decomprimi QUESTA cartella in
    E:\\AI\\Models\\LoRA\\MagicQuill\\_code\\
(oppure scaricala dal tab «Pesi modelli», voce «mq2-code», che chiede conferma di licenza). Servono anche le LoRA
MagicQuill (voci mq2-*-lora, stessa conferma). Senza questo componente la funzione MagicQuill resta spenta.
"""


def build(out: Path) -> tuple[Path, Path]:
    pack = out / "magicquill_code"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(SRC / f, pack / f)
    shutil.copytree(SRC / "mq2", pack / "mq2", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:                                       # noqa: BLE001
        rev = ""
    (pack / "VERSION").write_text(f"mq2-code {date.today().isoformat()} {rev}\n", "utf-8")
    (pack / "INSTALL.txt").write_text(INSTALL, "utf-8")
    z = out / "mq2-code.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(pack.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(pack).as_posix())
    return pack, z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"), help="cartella di uscita (default dist/)")
    a = ap.parse_args()
    for f in (*FILES, "mq2/__init__.py"):
        if not (SRC / f).is_file():
            print(f"manca {SRC / f}", file=sys.stderr)
            return 2
    pack, z = build(Path(a.out))
    n = sum(1 for p in pack.rglob("*") if p.is_file())
    print(f"pacchetto: {pack} ({n} file) · zip: {z} ({z.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
