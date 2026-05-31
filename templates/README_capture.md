# Cattura del "golden prompt" (template motore)

L'app non reimplementa la conversione UI→API di ComfyUI (graphToPrompt, fragile coi nodi
rgthree + subgraph). Cattura invece UNA volta il prompt API già risolto e lo parametrizza.

## Procedura

1. **Avvia ComfyUI**: `E:\AI\Comfy_Nexus\ComfyUI\run_nvidia.bat`. Apri `http://127.0.0.1:8188`.
2. **Carica** `E:\AI\workflows\LTX-2.3-curated\Pluto_AllInOne_v2.json`.
   - Prima serve **aggiornare ComfyUI** (i nuovi core node + subgraph) altrimenti il workflow non si apre.
3. **Scegli la combinazione** di moduli da catturare. Servono almeno 2 catture:
   - **`pluto_t2v.api.json`** — tutto disabilitato (premi `0`): Text-to-Video base. È la base minima per la Fase 0/1.
   - **`pluto_superset.api.json`** — tutti i moduli utili attivi (modello + audio + reference + ...). Per i selettori esclusivi (modello, audio) cattura una variante per ciascuna opzione: `..._standard`, `..._gguf`, `..._eros`, `..._audio_native`, `..._audio_user`, `..._audio_clone`.
4. **Cattura**: F12 → Console → incolla `capture_golden.js` → Invio → salva il file scaricato in questa cartella `templates\`, rinominandolo secondo la combinazione.
5. Verifica con lo spike: `python E:\AI\ltxstudio\scripts\spike_submit.py templates\pluto_t2v.api.json`
   (invia il prompt as-is e scarica il video).

## Catture aggiuntive da fare (sbloccano i moduli mancanti)

Stessa procedura (carica Pluto, attiva i moduli indicati, Export/console, salva qui):

| File da salvare | Cosa attivare in Pluto prima di catturare | Sblocca |
|---|---|---|
| `pluto_t2v.api.json` ✅ | tutto OFF (premi 0) | Text-to-Video (già fatto) |
| `pluto_i2v.api.json` | gruppo **Keyframe 1 / Reference image** ON | uso reale immagine iniziale (#2) |
| `pluto_controlnet.api.json` | **5.- Controlnet conditioning** ON + reference video | ControlNet (#2) |
| `pluto_audio_clone.api.json` | **6.- ID LoRA (voice cloning)** ON | clonazione voce (#2) |
| `pluto_audio_user.api.json` | **Custom audio source** ON | audio utente (#2) |

Per la **generazione del frame da prompt (#3)**: cattura un workflow txt2img separato
(hai `Z image turbo.json` o un Flux) → Export API → salva come `txt2img_zimage.api.json`
o `txt2img_flux.api.json`. Il `FrameGenerator` lo userà come gli altri template.

Il `template_engine` accetta un `template_path` diverso nel costruttore: l'app caricherà
il template giusto in base alla modalità scelta (T2V/I2V/...), riusando lo stesso meccanismo
di binding + prune/innesto + injection LoRA già pronto.

## Note
- I node id NEL prompt API coincidono con quelli del JSON UI per i nodi non-subgraph
  (es. 5119 prompt, 5751 UNETLoader). I nodi dentro subgraph sono prefissati.
- Da questi file il `template_engine` deriva i `bindings.json` (param → node_id/input).
- Ricattura necessaria solo se Pluto_AllInOne_v2.json viene modificato.
