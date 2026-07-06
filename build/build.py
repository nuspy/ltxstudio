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
import re
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


def _app_name(desc: dict) -> str:
    return desc.get("app_display_name") or desc.get("app_name") or desc.get("name") or "LTX-Studio"


def _profile_name(desc: dict, edition: str) -> str:
    for e in desc.get("editions", []):
        if e["id"] == edition:
            return e.get("public_name") or edition
    return edition


def _rename_output(out_dir: str, ext: str, app_name: str, profile: str, build_number: int, dry: bool) -> str | None:
    """Rinomina l'artefatto prodotto in <App>_<Profilo>_v<NNNNN>_<RAND6>.<ext>. Ritorna il nome finale."""
    final = ver.artifact_name(app_name, profile, build_number, ext)
    if dry:
        print(f"  → nome release: {final}")
        return final
    from pathlib import Path as _P
    cands = sorted(_P(out_dir).rglob(f"*.{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
    src = next((p for p in cands if not p.name.startswith(app_name.replace(" ", "-"))), cands[0] if cands else None)
    if src is None:
        print(f"  [ERR] nessun .{ext} prodotto in {out_dir}"); return None
    dst = _P(out_dir) / final
    src.replace(dst)
    print(f"  → {dst.name}")
    return str(dst)


def build_one_desktop(desc: dict, fm_root: Path, project: str, edition: str, os_target: str,
                      python: str, major: int | None, sign: bool, dry: bool) -> bool:
    app_root = desc["app_root"]
    out_dir = str(Path(app_root) / desc.get("output_dir", "dist") / edition / os_target)
    v = ver.peek(edition, major) if dry else ver.next_version(edition, major)
    print(f"\n=== [desktop] {edition} × {os_target}  →  build {v['build_number']} ===")

    codegen = [python, "-m", "feature_matrix", "codegen", "--edition", edition,
               "--build", str(v["build_number"]), "--project", project]
    if run(codegen, cwd=str(fm_root), dry=dry) != 0:
        print("  [ERR] codegen fallito"); return False

    onefile = bool(desc.get("build", {}).get("onefile_release", True))
    if os_target == "linux" and sys.platform.startswith("win"):
        print("  [i] target Linux: Nuitka non cross-compila → build/Dockerfile.linuxbuild.")
        dry_os = True
    else:
        dry_os = dry
    cmd = nuitka_command(python, app_root, desc.get("entrypoint", "app.py"),
                         out_dir, os_target, v, desc.get("build", {}), onefile)
    if run(cmd, cwd=app_root, dry=dry_os) != 0:
        print("  [ERR] Nuitka fallito"); return False

    ext = "exe" if os_target == "windows" else "bin"
    _rename_output(out_dir, ext, _app_name(desc), _profile_name(desc, edition), v["build_number"], dry or dry_os)

    if sign and not dry:
        from sign import sign_artifacts
        sign_artifacts(out_dir, os_target, desc.get("build", {}))
    run(["git", "tag", "-a", v["tag"], "-m", f"{edition} {os_target} build {v['build_number']}"],
        cwd=app_root, dry=dry)
    return True


def _patch_android_toolchain(android: Path) -> None:
    """Rende il progetto Capacitor compatibile con JDK 21 (verificato su questo PC): Gradle 8.7 + AGP 8.5.2,
    + local.properties con sdk.dir (forward-slash). I file android/ sono rigenerati da `cap add`, quindi il
    patch va riapplicato ad ogni build."""
    import os
    wrp = android / "gradle" / "wrapper" / "gradle-wrapper.properties"
    if wrp.exists():
        t = wrp.read_text("utf-8")
        t = re.sub(r"gradle-[\d.]+-(all|bin)\.zip", "gradle-8.7-all.zip", t)
        wrp.write_text(t, "utf-8")
    bg = android / "build.gradle"
    if bg.exists():
        t = bg.read_text("utf-8")
        t = re.sub(r"com\.android\.tools\.build:gradle:[\d.]+", "com.android.tools.build:gradle:8.5.2", t)
        bg.write_text(t, "utf-8")
    sdk = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT") \
        or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk")
    (android / "local.properties").write_text("sdk.dir=" + sdk.replace("\\", "/") + "\n", "utf-8")


def build_one_android(desc: dict, fm_root: Path, project: str, edition: str,
                      python: str, major: int | None, dry: bool) -> bool:
    """Build APK edition-aware: bake src/edition.json (manifest dell'edizione) → npm build → cap → gradle."""
    app_root = Path(desc["app_root"])
    mobile = app_root / desc.get("mobile_dir", "apps/studio-mobile")
    out_dir = str(app_root / desc.get("output_dir", "dist") / edition / "android")
    v = ver.peek(edition, major) if dry else ver.next_version(edition, major)
    print(f"\n=== [android] {edition}  →  build {v['build_number']} ===")
    if not mobile.exists():
        print(f"  [ERR] app mobile assente: {mobile}"); return False

    # 1) edition.json = manifest bakeato dell'edizione (stesse attivazioni del desktop)
    edition_json = mobile / "src" / "edition.json"
    print(f"  $ bake {edition_json.name}  (edizione {edition})")
    if not dry:
        try:
            from feature_matrix.engine import load_matrix, load_project
            from feature_matrix.manifest import build_manifest
            matrix = load_matrix(load_project(project))
            manifest = build_manifest(matrix, edition, v["build_number"])
            edition_json.parent.mkdir(parents=True, exist_ok=True)
            edition_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        except Exception as e:                               # noqa: BLE001
            print(f"  [ERR] bake edition.json: {e}"); return False

    # 2) npm build → capacitor add/sync → PATCH toolchain → gradle assembleDebug
    npm = "npm.cmd" if sys.platform.startswith("win") else "npm"
    npx = "npx.cmd" if sys.platform.startswith("win") else "npx"
    gradlew = "gradlew.bat" if sys.platform.startswith("win") else "./gradlew"
    android = mobile / "android"
    cap_cmd = ([npx, "cap", "add", "android"] if not android.exists() else [npx, "cap", "sync", "android"])
    for cmd, cwd in (([npm, "run", "build"], str(mobile)), (cap_cmd, str(mobile))):
        if run(cmd, cwd=cwd, dry=dry) != 0:
            print(f"  [ERR] step android fallito: {' '.join(cmd)}"); return False
    if not dry:
        _patch_android_toolchain(android)   # JDK 21 → Gradle 8.7 + AGP 8.5.2 + local.properties (verificato su questo PC)
    if run([gradlew, "assembleDebug"], cwd=str(android), dry=dry) != 0:
        print("  [ERR] gradle assembleDebug fallito"); return False

    # 3) rinomina l'APK prodotto
    apk_dir = str(mobile / "android" / "app" / "build" / "outputs" / "apk" / "debug")
    Path(out_dir).mkdir(parents=True, exist_ok=True) if not dry else None
    _rename_output(apk_dir if not dry else out_dir, "apk", _app_name(desc),
                   _profile_name(desc, edition), v["build_number"], dry)
    run(["git", "tag", "-a", f"{v['tag']}_android", "-m", f"{edition} android build {v['build_number']}"],
        cwd=str(app_root), dry=dry)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--edition", action="append", help="profilo; ripetibile; default = tutti")
    ap.add_argument("--target", action="append", help="desktop | android; default = desktop")
    ap.add_argument("--os", action="append", dest="os_targets", help="windows|linux (solo desktop)")
    ap.add_argument("--major", type=int, default=None)
    ap.add_argument("--sign", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    desc, fm_root = load_descriptor(args.project)
    editions = args.edition or [e["id"] for e in desc["editions"]]
    targets = args.target or ["desktop"]
    os_targets = args.os_targets or desc.get("os_targets", ["windows"])

    print(f"Pipeline build: profili={editions} × target={targets} "
          f"(desktop OS={os_targets})  dry-run={args.dry_run}")
    ok = True
    for e in editions:
        if "desktop" in targets:
            for o in os_targets:
                ok &= build_one_desktop(desc, fm_root, args.project, e, o, args.python,
                                        args.major, args.sign, args.dry_run)
        if "android" in targets:
            ok &= build_one_android(desc, fm_root, args.project, e, args.python,
                                    args.major, args.dry_run)
    print("\nRisultato:", "OK" if ok else "con errori")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
