"""Costruzione dei flag Nuitka per OS/edizione (solo la GUI/app è compilata)."""
from __future__ import annotations

from pathlib import Path

# Stack ML che NON deve finire nel binario. In `core/` gli import sono già lazy (dentro funzione,
# per non creare contesti CUDA nel processo GUI), ma Nuitka segue anche gli import annidati: senza
# questa lista il .exe si porta dietro torch/transformers/numba e cresce di GB. L'inferenza gira
# nei venv dei worker (inference_server_venv & co.), che questi pacchetti ce li hanno davvero.
# NB: huggingface_hub, safetensors, numpy, PIL, cv2 e scipy restano DENTRO — li usa la GUI.
NOFOLLOW_PACKAGES = (
    "torch", "torchvision", "torchaudio", "triton", "xformers", "flash_attn",
    "transformers", "diffusers", "accelerate", "peft", "bitsandbytes", "timm",
    "numba", "llvmlite", "pymatting",
)


def nuitka_command(python: str, app_root: str, entrypoint: str, out_dir: str,
                   os_target: str, version: dict, build_cfg: dict,
                   onefile: bool, runtime_dirs: list | None = None) -> list[str]:
    """Comando Nuitka completo. ComfyUI/nodi/modelli NON sono importati da app.py
    → non entrano nel binario (li scarica l'installer). `runtime_dirs` = [(cartella, nome)] da
    build/runtime_bundle.stage: sorgenti in chiaro dei processi figli (worker, server, script dei venv)."""
    app = Path(app_root)
    company = build_cfg.get("company_name", "LTX Studio")
    product = build_cfg.get("product_name", "LTX Studio")
    fileversion = f"{version['major']}.{version['minor']}.{version['build_number']}.0"

    cmd = [
        python, "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        # Il set di plugin Qt di default NON include i backend multimediali: senza questi il binario
        # parte ma logga «No QtMultimedia backends found» e ogni QMediaPlayer/QVideoSink fallisce
        # (anteprime audio/video mute). `multimedia` porta ffmpegmediaplugin.dll + le av*/sw*.dll.
        "--include-qt-plugins=multimedia",
        # pacchetti del prodotto (il valore/IP)
        "--include-package=ui",
        "--include-package=core",
        "--include-package=server",
        "--include-package=comfy_compat",
        # dati bundlati
        f"--include-data-dir={app / 'ui' / 'i18n'}=ui/i18n",
        *([] if runtime_dirs else [f"--include-data-files={app / 'core' / 'bindings.json'}=core/bindings.json"]),
        # metadati (riducono i falsi positivi AV / SmartScreen)
        f"--company-name={company}",
        f"--product-name={product}",
        f"--file-version={fileversion}",
        f"--product-version={version['major']}.{version['minor']}",
        "--file-description=LTX Studio",
        f"--output-dir={out_dir}",
    ]
    cmd += [f"--nofollow-import-to={pkg}" for pkg in NOFOLLOW_PACKAGES]
    # runtime dei processi figli: --include-raw-dir perché --include-data-dir scarterebbe i .py
    # (core/ in chiaro porta con sé anche bindings.json, per questo sopra non lo si include due volte)
    cmd += [f"--include-raw-dir={d}={name}" for d, name in (runtime_dirs or [])]
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
