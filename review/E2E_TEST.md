# E2E_TEST — elenco test GUI reali (qtbot) + scenari live

Fonte della verità per i test di interazione GUI. Un modello più leggero (Sonnet) può eseguire/scrivere
i test da questo elenco. Mappa completa dei controlli: `docs/GUI_TEST_MATRIX.md`.

## Livelli
- **A — Interazione/Wiring** (qtbot + Spy, no GPU): clicca/scrivi/seleziona sui widget veri → verifica la
  chiamata backend + params. Veloce, CI.
- **B — Semantico/Live** (backend vero, GPU, 1 job alla volta): genera davvero + ispeziona l'output (file,
  dimensioni, e giudice VLM per "è davvero X?"). Prompt CREATIVI (vedi regola sotto).

## Regola prompt (Livello B) — vedi memory test-prompts-creative
MAI gattini/cani. Prompt stupefacenti, sempre diversi, tono alternato scherzoso/epico/piccante/assurdo.

## Fixtures necessari (`tests/fixtures/`)
- foto con soggetto isolabile (per segment/inpaint/mask), ritratto (PuLID/face-swap), foto wide non-quadrata
  (outpaint), clip video breve 2-3s (controlnet/edit/enhance), campione voce (clone), mesh .glb (rig/convert),
  mini-dataset immagini+caption (LoRA).

## Stato armatura
- [ ] harness qtbot (`tests/gui_regression/qt_driver.py`): click/type/select + auto-patch dialoghi + Spy
- [ ] PoC video_tab (click prompt→scrivi→Genera→assert params)
- [ ] fixtures generati

## Elenco casi (da `GUI_TEST_MATRIX.md`) — spuntare quando implementato
### Livello A (wiring)
- [ ] VIDEO: t2v/i2v, per-modello×SFW/NSFW, preset risoluzione, i2v_method, sampler/seed, fr/fs/rife, storyboard, V2V (retake/talk/sing), controlnet (canny/depth/pose), extend
- [ ] IMMAGINE: gen per-arch (FLUX/Z-Image/Ernie)×modello, auto-select componenti per OGNI arch, PuLID, inpaint/outpaint (mask disegnata), integra, componi (+foto), recolor, upscale, segment
- [ ] AUDIO: per-motore TTS, clone con/senza testo, SSML/emozioni, use-as-audio→video
- [ ] MUSICA: HeartMuLa/ACE-Step, liriche AI 15 lingue, MMAudio, videomusicale
- [ ] TRAINING: per-modello base, caption manuale/auto, LoRA video/foto, per-tipo soggetto
- [ ] 3D: trellis (per formato/risoluzione), rig (per template/formato), mondo (mesh/splat/both)
- [ ] FILM STUDIO: script→asset→storyboard→montaggio, Auto 1-click, doppiaggio, multi-LoRA, @-mention, timeline drag&drop

### Livello B (live, prompt creativi + giudice VLM)
- [ ] disegna maschera → inpaint "trasforma X in Y" (verifica VLM)
- [ ] outpaint foto wide → estende la scena
- [ ] clone voce + emozione → parlato → video a2v
- [ ] controlnet su clip di danza
- [ ] mondo 3D output diversi + visualizzazione
- [ ] rig di una mesh + animazione
