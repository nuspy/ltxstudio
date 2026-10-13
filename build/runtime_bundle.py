"""Runtime dei processi figli da spedire INSIEME all'eseguibile (sorgenti in chiaro, scelta del 13/10/2026).

La GUI è compilata da Nuitka, ma l'inferenza gira in processi figli nei venv dei motori (diffusers_next, lingbot,
inference_server…): `python workers/<motore>_infer.py`, `python -m inference_server.server`, gli installer dei venv
in `scripts/`. Quei Python NON possono importare il codice compilato dentro l'exe, quindi i loro sorgenti — compresi
i port nativi scritti da noi (ControlNet Qwen-Image 2.1, arricchitori di prompt, convertitori ComfyUI→nativo…) e i
moduli di `core` che importano — devono stare su disco accanto all'app. Le funzioni Python della GUI risolvono la
radice dell'app con `Path(__file__).parents[1]`: nella build quella radice è la cartella dell'app (onedir) o la
cartella di estrazione (onefile), quindi le cartelle vanno incluse DENTRO la distribuzione, non copiate a parte.

`stage(app_root)` prepara una copia pulita (niente __pycache__, test, stato utente) e ritorna le cartelle per
`--include-raw-dir` (Nuitka scarterebbe i .py da `--include-data-dir`).
"""
from __future__ import annotations

import shutil
from pathlib import Path

# cartelle di codice eseguite dai processi figli (venv dei motori)
RUNTIME_DIRS = ("workers", "scripts", "inference_server", "core", "assets")
# file di dati di SOLA LETTURA spediti con l'app (lo stato utente — preferenze, cache, gui_state — si crea al volo)
DATA_FILES = ("weights_catalog.json", "ca_bundle.pem", "precompiled_wheels.json")
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "tests", "*.bak", "*.tmp*", "*.log",
                                 "edition.py")     # l'edizione bakeata è già compilata nell'exe


def stage(app_root: str | Path, dest: str | Path | None = None,
          exclude: list[str] | tuple = ()) -> list[tuple[Path, str]]:
    """Copia pulita del runtime in `dest` (default <app>/build/_runtime_stage). Ritorna [(cartella, nome)].
    `exclude` = glob (relativi ad app_root) dei componenti delle feature SPENTE nell'edizione (feature matrix,
    `python -m feature_matrix components --edition E`): non vengono spediti."""
    app = Path(app_root)
    out = Path(dest) if dest else app / "build" / "_runtime_stage"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pairs = []
    for name in RUNTIME_DIRS:
        src = app / name
        if src.is_dir():
            shutil.copytree(src, out / name, ignore=_IGNORE)
            pairs.append((out / name, name))
    data = out / "data"
    data.mkdir()
    for f in DATA_FILES:
        if (app / "data" / f).is_file():
            shutil.copy2(app / "data" / f, data / f)
    pairs.append((data, "data"))
    removed = 0
    for pat in exclude:
        for f in out.glob(pat):
            if f.is_file():
                f.unlink()
                removed += 1
    if exclude:
        print(f"  [runtime] {removed} file di feature spente nell'edizione esclusi")
    return pairs


# nucleo comune: lo usano tutte le edizioni (download/conversione dei pesi, server nativo, loader quantizzati)
CORE_FILES = ("workers/_qlinear.py", "scripts/download_weights.py", "scripts/_watchdog.py",
              "inference_server/server.py", "core/native_convert.py", "core/downloader.py",
              "core/qwen21_spec.py", "data/weights_catalog.json")


def missing_entries(app_root: str | Path, include: list[str] | tuple = ()) -> list[str]:
    """Coerenza del build: il nucleo comune e i componenti delle feature ACCESE nell'edizione devono esistere."""
    app = Path(app_root)
    miss = [n for n in CORE_FILES if not (app / n).is_file()]
    miss += [pat for pat in include if not any(app.glob(pat))]
    return miss


def edition_components(fm_root: str | Path, project: str, edition: str, python: str = "python") -> dict:
    """{"include": [...], "exclude": [...]} dal feature matrix (stesso interprete del codegen)."""
    import json
    import subprocess
    r = subprocess.run([python, "-m", "feature_matrix", "--project", project, "components", "--edition", edition],
                       cwd=str(fm_root), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"feature_matrix components fallito: {(r.stderr or r.stdout)[-500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    for d, name in stage(root):
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"{name:18s} {n:5d} file")
    print("mancanti:", missing_entries(root) or "nessuno")
