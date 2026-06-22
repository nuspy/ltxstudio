"""Pipeline di build multi-edizione × OS (Nuitka). Sostituisce build_exe.bat.

Per ogni (edizione × OS): bake `core/edition.py` (via feature_matrix codegen),
compila la GUI con Nuitka, nomina `V_<ed>_R_<major>.<minor>`, firma (opzionale) e
crea il tag git. Le edizioni vengono dalla matrice (dataset) → aggiungere/togliere
edizioni = aggiungere/togliere job, NON branch git.

Uso:
  python build/build.py --dry-run                      # stampa il piano, non compila
  python build/build.py --edition free --os windows    # compila una sola
  python build/build.py --major 2 --edition premium    # major release (azzera minor)
  python build/build.py --sign                          # firma (Azure Artifact Signing)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import version as ver                       # noqa: E402
from nuitka_flags import nuitka_command     # noqa: E402

DEFAULT_PROJECT = r"C:\Projects\feature_matrix\projects\ltxstudio\project.json"


def load_descriptor(project: str) -> tuple[dict, Path]:
    p = Path(project).resolve()
    desc = json.loads(p.read_text("utf-8"))
    fm_root = p.parents[2]                  # <fm_root>/projects/<name>/project.json
    return desc, fm_root


def run(cmd, cwd=None, dry=False) -> int:
    print("  $", " ".join(str(c) for c in cmd))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=cwd).returncode


def build_one(desc: dict, fm_root: Path, project: str, edition: str, os_target: str,
              python: str, major: int | None, sign: bool, dry: bool) -> bool:
    app_root = desc["app_root"]
    out_dir = str(Path(app_root) / desc.get("output_dir", "dist") / edition / os_target)
    v = ver.peek(edition, major) if dry else ver.next_version(edition, major)
    print(f"\n=== {edition} × {os_target}  →  {v['tag']}  (build {v['build_number']}) ===")

    # 1) codegen del manifest bakeato (core/edition.py)
    codegen = [python, "-m", "feature_matrix", "codegen", "--edition", edition,
               "--build", str(v["build_number"]), "--project", project]
    if run(codegen, cwd=str(fm_root), dry=dry) != 0:
        print("  [ERR] codegen fallito"); return False

    # 2) Nuitka (solo la GUI/app)
    onefile = bool(desc.get("build", {}).get("onefile_release", True))
    if os_target == "linux" and sys.platform.startswith("win"):
        print("  [i] target Linux: Nuitka non cross-compila → eseguire in "
              "build/Dockerfile.linuxbuild (qui stampo solo il comando).")
        dry_os = True
    else:
        dry_os = dry
    cmd = nuitka_command(python, app_root, desc.get("entrypoint", "app.py"),
                         out_dir, os_target, v, desc.get("build", {}), onefile)
    if run(cmd, cwd=app_root, dry=dry_os) != 0:
        print("  [ERR] Nuitka fallito"); return False

    # 3) firma (opzionale)
    if sign and not dry:
        from sign import sign_artifacts
        sign_artifacts(out_dir, os_target, desc.get("build", {}))

    # 4) tag git
    tagcmd = ["git", "tag", "-a", v["tag"], "-m", f"{edition} {os_target} build {v['build_number']}"]
    run(tagcmd, cwd=app_root, dry=dry)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--edition", action="append", help="ripetibile; default = tutte")
    ap.add_argument("--os", action="append", dest="os_targets", help="windows|linux; default da project.json")
    ap.add_argument("--major", type=int, default=None)
    ap.add_argument("--sign", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    desc, fm_root = load_descriptor(args.project)
    editions = args.edition or [e["id"] for e in desc["editions"]]
    os_targets = args.os_targets or desc.get("os_targets", ["windows"])

    print(f"Pipeline build: {editions} × {os_targets}  (dry-run={args.dry_run})")
    ok = True
    for e in editions:
        for o in os_targets:
            ok &= build_one(desc, fm_root, args.project, e, o, args.python,
                            args.major, args.sign, args.dry_run)
    print("\nRisultato:", "OK" if ok else "con errori")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
