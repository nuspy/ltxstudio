"""Firma del codice (opzionale, attivata dalla checkbox nella GUI di build).

Windows: Azure Artifact Signing (ex Trusted Signing) via signtool + dlib Azure
(credenziali da env/parametri, MAI nel repo). Se le credenziali mancano, NON
blocca: stampa la guida ed esce. Linux: firma GPG del bundle + SHA256SUMS.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _exes(out_dir: str) -> list[Path]:
    d = Path(out_dir)
    return list(d.rglob("*.exe")) if d.exists() else []


def sign_windows(out_dir: str, cfg: dict) -> bool:
    """Firma gli .exe con Azure Artifact Signing. Config attesa (env o cfg):
      AZURE_SIGN_DLIB, AZURE_SIGN_METADATA (json con Endpoint/CodeSigningAccount/CertificateProfile).
    signtool deve essere nel PATH (Windows SDK)."""
    dlib = cfg.get("azure_dlib") or os.environ.get("AZURE_SIGN_DLIB")
    meta = cfg.get("azure_metadata") or os.environ.get("AZURE_SIGN_METADATA")
    if not (dlib and meta):
        print("  [firma] credenziali Azure Artifact Signing assenti "
              "(AZURE_SIGN_DLIB/AZURE_SIGN_METADATA) → salto. Vedi "
              "https://azure.microsoft.com/en-us/pricing/details/artifact-signing/")
        return False
    ok = True
    for exe in _exes(out_dir):
        cmd = ["signtool", "sign", "/v", "/debug", "/fd", "SHA256",
               "/tr", "http://timestamp.acs.microsoft.com", "/td", "SHA256",
               "/dlib", dlib, "/dmdf", meta, str(exe)]
        print("  $", " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            ok = False
    return ok


def sign_linux(out_dir: str, cfg: dict) -> bool:
    """SHA256SUMS + firma GPG (se una chiave è configurata)."""
    d = Path(out_dir)
    files = [p for p in d.rglob("*") if p.is_file()]
    if not files:
        return False
    sums = d / "SHA256SUMS"
    import hashlib
    with sums.open("w", encoding="utf-8") as f:
        for p in files:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            f.write(f"{h}  {p.relative_to(d)}\n")
    key = cfg.get("gpg_key") or os.environ.get("GPG_SIGN_KEY")
    if key:
        subprocess.run(["gpg", "--default-key", key, "--detach-sign", "--armor", str(sums)])
    else:
        print("  [firma] nessuna chiave GPG (GPG_SIGN_KEY) → solo SHA256SUMS.")
    return True


def sign_macos(out_dir: str, cfg: dict) -> bool:
    """codesign (Developer ID Application, hardened runtime) + notarizzazione (notarytool) + staple → evita
    il blocco Gatekeeper (\"app non verificata\"). Config (env o cfg):
      APPLE_SIGN_IDENTITY  = \"Developer ID Application: Nome (TEAMID)\"
      APPLE_NOTARY_PROFILE = profilo keychain per `notarytool` (creato con `notarytool store-credentials`).
    Senza identità NON blocca: stampa la guida ed esce."""
    ident = cfg.get("apple_identity") or os.environ.get("APPLE_SIGN_IDENTITY")
    if not ident:
        print("  [firma] APPLE_SIGN_IDENTITY assente → salto (serve un 'Developer ID Application' + "
              "notarizzazione per evitare Gatekeeper). Vedi Apple Developer.")
        return False
    d = Path(out_dir)
    targets = list(d.rglob("*.app")) + list(d.rglob("*.dmg")) + list(d.rglob("*.pkg"))
    ok = True
    for t in targets:                                       # codesign con hardened runtime + timestamp
        cmd = ["codesign", "--force", "--deep", "--options", "runtime", "--timestamp", "--sign", ident, str(t)]
        print("  $", " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            ok = False
    prof = cfg.get("apple_notary_profile") or os.environ.get("APPLE_NOTARY_PROFILE")
    for t in list(d.rglob("*.dmg")) + list(d.rglob("*.pkg")):   # notarizza+staple i pacchetti distribuibili
        if prof:
            subprocess.run(["xcrun", "notarytool", "submit", str(t), "--keychain-profile", prof, "--wait"])
            subprocess.run(["xcrun", "stapler", "staple", str(t)])
        else:
            print("  [firma] APPLE_NOTARY_PROFILE assente → codesign fatto, notarizzazione saltata.")
    return ok


def sign_artifacts(out_dir: str, os_target: str, cfg: dict) -> bool:
    if os_target == "windows":
        return sign_windows(out_dir, cfg)
    if os_target in ("macos", "darwin", "osx"):
        return sign_macos(out_dir, cfg)
    return sign_linux(out_dir, cfg)
