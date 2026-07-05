"""Consolida venv compatibili (#61): copia OFFLINE dei pacchetti mancanti dai venv-donor all'host.

Perché offline: connessione metered — niente pip download. Host e donor hanno lo stesso Python e lo
stesso torch (ABI identica) → i pacchetti (anche compilati) si copiano file-per-file usando il RECORD
di ogni distribuzione. REGOLE DI SICUREZZA:
- non si sovrascrive MAI nulla nell'host (i pacchetti esistenti vincono → l'host resta identico per sé);
- le entry in Scripts/ vengono SALTATE (gli .exe wrapper embeddano il python del donor → romperebbero);
- i donor NON vengono toccati (restano su disco finché i live-test non confermano il consolidato).

Uso:
  python scripts/consolidate_venvs.py <host_venv> <donor_venv> [<donor_venv> ...] [--dry]
Es.:
  python scripts/consolidate_venvs.py E:\\AI\\Tools\\inference_server_venv E:\\AI\\Tools\\XTTS\\venv --dry
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path


def site(venv: Path) -> Path:
    return venv / "Lib" / "site-packages"


def dists(venv: Path) -> dict[str, Path]:
    """{nome-normalizzato: path .dist-info} delle distribuzioni installate nel venv."""
    out = {}
    for d in site(venv).glob("*.dist-info"):
        base = d.name[: -len(".dist-info")]                  # "tokenizers-0.22.2"
        name = base.rpartition("-")[0].lower().replace("_", "-")   # → "tokenizers" (senza versione!)
        out[name] = d
    return out


def copy_dist(dist_info: Path, src_site: Path, dst_site: Path, dry: bool) -> tuple[int, int]:
    """Copia i file del RECORD (relativi a site-packages). Ritorna (copiati, saltati)."""
    rec = dist_info / "RECORD"
    copied = skipped = 0
    if not rec.exists():
        return 0, 0
    with open(rec, encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            rel = row[0].replace("\\", "/")
            if rel.startswith("..") or rel.split("/")[0].lower() in ("scripts", "bin"):
                skipped += 1                                  # exe wrapper del donor: MAI copiare
                continue
            src = src_site / rel
            dst = dst_site / rel
            if not src.exists() or dst.exists():              # mai sovrascrivere l'host
                skipped += 1
                continue
            if not dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied += 1
    return copied, skipped


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--dry"]
    dry = "--dry" in sys.argv
    host = Path(args[0])
    donors = [Path(a) for a in args[1:]]
    hdists = dists(host)
    print(f"HOST {host} — {len(hdists)} distribuzioni{' [DRY-RUN]' if dry else ''}")
    for donor in donors:
        ddists = dists(donor)
        missing = {n: p for n, p in ddists.items() if n not in hdists}
        print(f"\n── donor {donor.name} ({donor.parent.name}): {len(missing)} da copiare")
        tot_c = 0
        for name, di in sorted(missing.items()):
            c, s = copy_dist(di, site(donor), site(host), dry)
            tot_c += c
            print(f"   + {name:34} file={c}")
            hdists[name] = di                                 # aggiorna la vista (donor successivi)
        print(f"   → {tot_c} file copiati da {donor.parent.name}/{donor.name}")
    print("\nFATTO. I donor NON sono stati modificati (cancellarli solo dopo i live-test).")


if __name__ == "__main__":
    main()
