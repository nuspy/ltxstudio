# TaDaMoo Landing — Fase B: sostituzione dei placeholder

La landing (`index.html`) e la pagina acquisto (`buy.html`) sono complete e **belle già ora**:
ogni media è rappresentato da uno "sviluppo" art-directed (paesaggi astratti coerenti, `.dev`).
La Fase B consiste nel generare i media di `ASSETS_MANIFEST.json` (con LTX Studio stesso)
e sostituire gli sviluppi con i file veri.

## Architettura della pagina (dove vivono gli slot)

- Il contenuto degli **esempi** è definito in un unico oggetto JS `EX = {...}` dentro
  `index.html` (cerca `const EX`). Ogni chiave è un `data-asset-id` di slot; ogni voce ha
  `p:[...]` = i **pezzi** (risultato per primo, poi origini/maschere/guide), ciascuno col
  proprio id-pezzo (es. `img-inpaint-01-src`).
- I bottoni `.frag` in pagina (con `data-asset-id`) mostrano il **risultato grande** con
  le origini come satelliti; il click apre il **dialog immersivo** (slider prima/dopo,
  spiegazione, incantesimo copiabile).
- Slot speciali fuori da `EX`: `hero-bg-01` (video di sfondo hero), `img-edit-01`
  (slider in pagina `#cmp-stage`), `d3-mesh-01`/`d3-rig-01` (oblò `<model-viewer>`),
  `world-01` (Proiettore Mondi three.js), `models-ui-01` (console autoselect).

## 1. Immagini

1. Genera ogni pezzo con il prompt del manifest (risoluzioni **grandi**: lato lungo ≥1024,
   ≥2048 per i protagonisti). Esporta in **WebP q85** in `assets/<id-pezzo>.webp`.
2. In `EX`, sostituisci lo sviluppo con l'immagine: aggiungi al pezzo la proprietà
   `img:"assets/img-t2i-01.webp"` e aggiorna `devHTML` per renderizzare
   `<img src loading="lazy">` quando `spec.img` esiste (3 righe), **oppure** sostituisci
   direttamente il markup generato con `<img>` mantenendo `.lab`/`.exd-tag`.
   Gli slider (`sliderHTML`) funzionano identici con `<img>` al posto di `.dev`.
3. Slider in pagina `#cmp-stage`: sostituisci i due blocchi `.side .city` con le due
   immagini reali (`img-edit-01-src` / `img-edit-01-out`), stessa inquadratura.

## 2. Video YouTube

1. Genera il video (durate nel manifest — minimo 15s; `vid-long-01` = **60s sequenza unica**),
   risoluzione ≥1280×704. Carica su YouTube (unlisted ok).
2. Nei dialog i riquadri video hanno `data-yt-id="PLACEHOLDER"`: l'id va messo **in `EX`**
   perché il dialog è generato da JS. Il modo più semplice: aggiungi al pezzo la proprietà
   `yt:"VIDEO_ID"` e in `ytFrame()` sostituisci `PLACEHOLDER` con `p.yt||'PLACEHOLDER'`
   (1 riga). `hydrateYT()` monta da solo l'iframe `youtube-nocookie` quando l'id ≠ PLACEHOLDER.

## 3. Audio

Esporta **mp3 320kbps** in `assets/` e imposta `src` sugli `<audio>` generati nei dialog
(stesso meccanismo: proprietà `audio:"assets/mus-song-01.mp3"` sul pezzo e `src` in
`mediaHTML`, casi `audio`/`audiopair`).

## 4. Video di sfondo hero (`hero-bg-01`, opzionale)

```html
<div class="hero-videobg" data-asset-id="hero-bg-01" data-video-src="assets/hero-bg-01.mp4">
```
Lo script crea il `<video>` autoplay/loop (opacità 28%) sotto il canvas del cielo.
Senza src resta il solo cielo animato, che regge benissimo da solo.

## 5. Modelli 3D (oblò)

| Slot | Placeholder attuale | Da sostituire con |
|---|---|---|
| `d3-mesh-01` | `assets/placeholder_mesh.glb` (cavaliere TRELLIS reale) | mesh del colibrì meccanico |
| `d3-rig-01` | `assets/placeholder_rigged.glb` (cavaliere riggato reale) | personaggio riggato + animazione |

Sovrascrivi il file o aggiorna `src` del `<model-viewer>`. GLB <15 MB.

## 6. Proiettore Mondi (`world-01`)

`assets/placeholder_world.glb` è un **oggetto**, non un mondo: sostituiscilo con una scena
WorldGen (GLB <40 MB). Il path è nel blocco JS del proiettore
(`GLTFLoader().load('assets/placeholder_world.glb', …)`). Orbit + WASD già implementati.

## 7. Console modelli (`models-ui-01`)

La console animata resta anche in produzione (è un esempio vivo). In alternativa,
screenshot reale della tab Modelli a 2560×1440 con i semafori visibili, al posto della `.console`.

## Checklist finale Fase B

- [ ] 12 slot YouTube valorizzati (proprietà `yt:` nei pezzi di `EX`)
- [ ] Immagini WebP grandi al posto degli sviluppi `.dev` (slider inclusi)
- [ ] 4 audio mp3 collegati
- [ ] 3 GLB sostituiti (mesh, rigged, world)
- [ ] `vid-long-01` = 60s sequenza unica; tutti gli altri video ≥15s
- [ ] Alt text su tutte le `<img>`
- [ ] Test mobile 375px: nessuno scroll orizzontale (overflow-x è già in clip)
- [ ] `prefers-reduced-motion`: verificare che la pagina resti leggibile (già gestito)
- [ ] `buy.html`: collegare il checkout reale al posto dell'`alert()` demo
