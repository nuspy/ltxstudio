# LTX Studio — installazione (Windows / macOS / Linux)

Filosofia: **l'installer nativo installa solo app + runtime**; al **primo avvio** l'app mostra il **wizard
«Setup modelli»** (`ui/setup/first_run.py`) che:
1. chiede le **cartelle** dove tieni già modelli AI (ricerca ricorsiva),
2. **collega** (junction/hardlink, senza admin — `core/linkutil.py`) quelli trovati invece di riscaricarli,
3. offre di **importare** i modelli presenti ma non elencati,
4. scarica **solo i pesi usati dai moduli della tua edizione** (feature_matrix → `core/weights_catalog.py`
   `by_module(only_wanted=True)`), con URL/hash bakati al build (`build/resolve_models.py`).

La stessa logica alimenta il **tab «Pesi modelli»** (`ui/tabs/weights_tab.py`), disponibile sempre.

## Windows
- **Consigliato (CI/riproducibile):** [Inno Setup 6] → `installer/windows/ltxstudio.iss`.
  ```
  ISCC.exe installer\windows\ltxstudio.iss /DAppSrc="<dist Nuitka onedir>" /DAppVer=1.2.3
  ```
  Installer **per-utente** (`PrivilegesRequired=lowest`, niente UAC) — coerente col wizard che usa
  junction/hardlink senza admin.
- **Alternativa GUI:** InstallForge — nuovo progetto → *General* (nome/versione), *Files* (aggiungi la cartella
  `dist`), *Shortcuts* (Menu+Desktop → `LTXStudio.exe`), *Finish* (spunta «Run application»). Esporta il setup.
- **FIRMA (evita SmartScreen/AV):** Azure Trusted Signing via `build/sign.py` (`sign_windows`):
  ```
  set AZURE_SIGN_DLIB=...\Azure.CodeSigning.Dlib.dll
  set AZURE_SIGN_METADATA=...\metadata.json
  python build\sign.py --os windows --out dist   # firma gli .exe; poi firma anche il Setup .exe
  ```
  Firma **sia l'installer sia l'eseguibile dell'app** (l'app non firmata dà comunque l'allarme).

## macOS
- Build DMG: `APP_SRC=dist/LTXStudio.app VER=1.2.3 bash installer/macos/build_pkg.sh` (usa `create-dmg`).
- **FIRMA + NOTARIZZAZIONE (evita Gatekeeper):** `build/sign.py` (`sign_macos`), richiede un account Apple Developer:
  ```
  export APPLE_SIGN_IDENTITY="Developer ID Application: Nome (TEAMID)"
  export APPLE_NOTARY_PROFILE="ltx-notary"   # creato con: xcrun notarytool store-credentials
  ```
  Fa `codesign --options runtime` (hardened) + `notarytool submit --wait` + `stapler staple`.

## Linux
- Build AppImage: `APP_SRC=dist/LTXStudio VER=1.2.3 bash installer/linux/build_appimage.sh` (usa `appimagetool`).
- **FIRMA:** `SHA256SUMS` + GPG via `build/sign.py` (`sign_linux`, env `GPG_SIGN_KEY`). L'AppImage è
  auto-contenuta; la firma GPG dà integrità/provenienza verificabili.

## Prerequisiti runtime (tutti gli OS)
Verificati/installati dal flusso CLI esistente (`installer/install.py`, `installer/steps/prereqs.py`): Python 3.10+,
driver NVIDIA + CUDA, PyTorch compatibile, ffmpeg. I pesi arrivano dal wizard/tab (non dall'installer nativo).

## Note firma (riassunto — perché servono)
Un installer/eseguibile **non firmato** su Windows fa scattare **SmartScreen** («editore sconosciuto») e alcuni
AV; su macOS **Gatekeeper** blocca le app non firmate+notarizzate. La firma (Azure/Windows, Developer ID/macOS,
GPG/Linux) elimina questi falsi allarmi. Le credenziali stanno **solo in env**, mai nel repo.
