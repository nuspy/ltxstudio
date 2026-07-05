# ATF Detector — app Android

App **Kivy** che verifica se una foto è stata generata con **AI Tools Forge**, leggendo la firma di
provenienza invisibile (#107) e l'eventuale **testo custom**. Funziona:
- **📷 Scatta una foto** — anche fotografando lo schermo di un altro dispositivo;
- **🖼 Carica una foto** dalla galleria.

Il rilevamento gira **in locale sul telefono** (nessuna rete, nessun server): il core è `detector_stego.py`,
port PURO **numpy + PIL** (niente OpenCV/scipy), cross-compatibile con la firma del desktop — validato: rileva
foto firmate su desktop anche dopo JPEG/screenshot/foto-schermo storta, e legge il testo custom.

## File
- `main.py` — app Kivy (entrypoint richiesto da buildozer): fotocamera (plyer) + upload (filechooser) + risultato.
- `detector_stego.py` — rilevamento numpy/PIL (stessa CHIAVE e algoritmo di `core/stego.py`).
- `buildozer.spec` — configurazione build APK.

## Test rapido sul PC (senza Android)
```
pip install kivy plyer pillow numpy
python mobile/atf_detector_android/main.py
```

## Build dell'APK (Android)
Serve la toolchain **buildozer** (Linux o WSL; su Windows nativo NON è supportato). Su Ubuntu/WSL:
```
pip install buildozer cython
sudo apt install -y openjdk-17-jdk zip unzip autoconf libtool pkg-config zlib1g-dev libncurses-dev
cd mobile/atf_detector_android
buildozer android debug          # la prima volta scarica Android SDK/NDK (alcuni GB)
# APK in bin/atfdetector-1.0-debug.apk → installa con:  adb install bin/*.apk
```
Alternative senza PC Linux: **Google Colab** (c'è un notebook buildozer) o un runner **GitHub Actions**
(`ArtemSBulgakov/buildozer-action`).

## Note oneste
- **Non ho compilato l'APK qui**: buildozer richiede SDK/NDK Android (GB) e un ambiente Linux/WSL + un
  dispositivo per il test. Il sorgente è pronto e il core di rilevamento è validato sul PC.
- **Velocità**: il resync geometrico prova molte rotazioni → ~10-40 s per foto su telefono. Per accelerare,
  in `detect_path(..., max_angle=15)` si può ridurre `max_angle` (es. 8) o disattivare il resync per le foto
  caricate da file (non ruotate).
- **Sicurezza**: la CHIAVE è in `detector_stego.py` e finisce nell'APK (estraibile da chi decompila). Deve
  restare **identica** a `core/stego.py._KEY`. Per garanzia forte → verifica lato-server.
- **Limiti**: prospettiva forte/keystone non coperta (come desktop); il testo custom è ~5 caratteri e può
  perdersi sotto rotazione forte (la firma sì/no invece regge). Il rilevamento VIDEO non è nell'app mobile
  (solo foto) — per i video usare il Detector desktop.
