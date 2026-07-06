# LTX Studio — app mobile (React + Capacitor)

Client cross-platform (Android APK · PWA browser · desktop via Electron/Tauri) per l'intera suite LTX Studio:
**immagini, video (+ ControlNet), training LoRA (img/video), musica, videoclip**, con **canvas immagini + mask**,
**rilevatore artefatto** on-device, **download** dei risultati e **connessione impostabile**. È un **client**
della REST API `:8200` — generazione/training/musica girano sul server GPU (PC locale o `tadamoo.com`).

## Edition-aware (feature matrix)
L'app rispetta lo **stesso feature matrix** del desktop: `src/edition.json` è bakeato per edizione dal builder
feature_matrix; le funzioni disabilitate mostrano un **upsell** (come il desktop). In `dev` (edition.json vuoto)
è tutto abilitato (fail-open). A runtime si fonde con lo snapshot del server (`GET /features`).

## Sviluppo (web/PWA — provabile subito)
```
cd apps/studio-mobile
npm install
npm run dev            # http://localhost:5173  (imposta l'URL del server in Impostazioni)
```
Avvia il server API in ascolto sulla rete perché il telefono lo raggiunga:
```
set LTX_API_HOST=0.0.0.0
<venv>/python -m api.app
```
Per l'accesso da Internet (tadamoo.com): reverse-proxy HTTPS + `LTX_API_TOKEN` (impostalo anche in Impostazioni).

## Build APK Android
Serve la toolchain (già presente su questo PC: Node, Android SDK in `%LOCALAPPDATA%\Android\Sdk`, JDK di
Android Studio). Imposta le env e:
```
set ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk
set JAVA_HOME=<Android Studio>\jbr
npm install
npm run build
npx cap add android         # una tantum
npx cap sync android
cd android && gradlew.bat assembleRelease   # → android/app/build/outputs/apk/release/*.apk
```
Il **builder feature_matrix** (fase S7) automatizza questo per OGNI edizione (free/supporter/plus/premium/studio),
scrivendo `src/edition.json` e producendo un APK per edizione.

## Struttura
- `src/api.ts` — client REST + polling job + upload.
- `src/features.ts` + `src/edition.json` — gating feature-matrix edition-aware.
- `src/settings.ts` — URL server / token / tema (Capacitor Preferences).
- `src/screens/*` — Home, Immagini, Canvas, Video, Training, Musica, Rilevatore, Galleria, Impostazioni.
- `src/components/FeatureGate.tsx` — upsell in-regione per le funzioni non incluse.
