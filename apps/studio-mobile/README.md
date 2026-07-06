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

## Build APK Android — VERIFICATO su questo PC ✅
Toolchain presente: Node v24, Android SDK in `%LOCALAPPDATA%\Android\Sdk`, JDK 21 in
`C:\Program Files\Android\openjdk\jdk-21.0.8`. Il template Capacitor 6 usa Gradle 8.2.1/AGP 8.2.1 che NON
supportano JDK 21 → vanno alzati a **Gradle 8.7 + AGP 8.5.2** (il builder lo fa da solo). Manualmente:
```
set ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk
set JAVA_HOME=C:\Program Files\Android\openjdk\jdk-21.0.8
npm install && npm run build
npx cap add android
# patch: gradle-wrapper.properties → gradle-8.7-all.zip ; build.gradle → com.android.tools.build:gradle:8.5.2
# local.properties → sdk.dir=C:/Users/<u>/AppData/Local/Android/Sdk   (forward-slash!)
cd android && gradlew.bat assembleDebug   # → app/build/outputs/apk/debug/app-debug.apk (~6MB, testato)
```
Il **builder feature_matrix** (`build/build.py`, fase S7) automatizza TUTTO per ogni edizione: bake
`src/edition.json` + patch toolchain (`_patch_android_toolchain`) + gradle + rinomina
`AI-Creative-Center_<Profilo>_v<NNNNN>_<RAND6>.apk`.
Per la RELEASE firmata (distribuzione): aggiungi un keystore + signingConfig e usa `assembleRelease`.

## Struttura
- `src/api.ts` — client REST + polling job + upload.
- `src/features.ts` + `src/edition.json` — gating feature-matrix edition-aware.
- `src/settings.ts` — URL server / token / tema (Capacitor Preferences).
- `src/screens/*` — Home, Immagini, Canvas, Video, Training, Musica, Rilevatore, Galleria, Impostazioni.
- `src/components/FeatureGate.tsx` — upsell in-regione per le funzioni non incluse.
