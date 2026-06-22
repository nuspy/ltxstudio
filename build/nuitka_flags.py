"""Costruzione dei flag Nuitka per OS/edizione (solo la GUI/app è compilata)."""
from __future__ import annotations

from pathlib import Path


def nuitka_command(python: str, app_root: str, entrypoint: str, out_dir: str,
                   os_target: str, version: dict, build_cfg: dict,
                   onefile: bool) -> list[str]:
    """Comando Nuitka completo. ComfyUI/nodi/modelli NON sono importati da app.py
    → non entrano nel binario (li scarica l'installer)."""
    app = Path(app_root)
    company = build_cfg.get("company_name", "LTX Studio")
    product = build_cfg.get("product_name", "LTX Studio")
    fileversion = f"{version['major']}.{version['minor']}.{version['build_number']}.0"

    cmd = [
        python, "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        # pacchetti del prodotto (il valore/IP)
        "--include-package=ui",
        "--include-package=core",
        "--include-package=server",
        "--include-package=comfy_compat",
        # dati bundlati
        f"--include-data-dir={app / 'ui' / 'i18n'}=ui/i18n",
        f"--include-data-files={app / 'core' / 'bindings.json'}=core/bindings.json",
        # metadati (riducono i falsi positivi AV / SmartScreen)
        f"--company-name={company}",
        f"--product-name={product}",
        f"--file-version={fileversion}",
        f"--product-version={version['major']}.{version['minor']}",
        "--file-description=LTX Studio",
        f"--output-dir={out_dir}",
    ]
    if onefile:
        cmd.append("--onefile")
    if os_target == "windows":
        ico = app / "ui" / "app.ico"
        if ico.exists():
            cmd.append(f"--windows-icon-from-ico={ico}")
    # template/ è opzionale: includila se presente
    tmpl = app / "templates"
    if tmpl.is_dir():
        cmd.append(f"--include-data-dir={tmpl}=templates")
    cmd.append(str(app / entrypoint))
    return cmd
