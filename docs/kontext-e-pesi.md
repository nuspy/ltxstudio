# Kontext nella GUI & provenienza dei pesi

Risposte a due domande ricorrenti (task #9 e #10).

## #9 — Come funziona Kontext nella GUI? Serve sempre la maschera?

**Dove**: tab **Immagini** (Frame 1), pannello destro:
- sezione **🧩 Integra immagini** — editing per riferimento (stile MagicQuill);
- sezione **🧩 Componi più foto** (nuova) — composizione multi-foto senza incollaggio;
- sezione **Cambia colore** — recolor prompt-based.

**Come funziona (Kontext / FLUX.2)**: l'immagine NON viene "inpaintata" su una maschera. Viene data come **riferimento** (`ReferenceLatent`) insieme al prompt, e il modello **rigenera/armonizza** la scena (luce, ombre, prospettiva). È un edit-by-reference, non un inpaint.

**Serve la maschera? NO** per:
- **Integra** (usa il composito scena+oggetto come riferimento; nessuna maschera),
- **Componi più foto** (le maschere sono **opzionali**, servono solo a indicare *quale parte* di una foto usare),
- **Cambia colore** (solo prompt).

**Serve la maschera SÌ** solo per **Inpaint/Outpaint** (`FLUX Fill`), che è una funzione diversa: lì la maschera definisce l'area da rigenerare.

**Engine e modello**: combo Kontext (FLUX.1) / FLUX.2; modello selezionabile. `puzzle_lora` è un'opzione **sperimentale, solo FLUX.1** (di default OFF: da sola tende a introdurre una cornice).

## #10 — I link dei pesi (wizard e auto-download) sono HF o ComfyUI?

**Sono tutti HuggingFace.** Non si scarica nulla dal programma ComfyUI.

- **Wizard pesi** (`data/weights_catalog.json` + `scripts/download_weights.py`): ogni voce ha un `repo_id` HuggingFace, scaricato con `snapshot_download`. Es. `black-forest-labs/FLUX.1-dev`, `baidu/ERNIE-Image`, `Tongyi-MAI/Z-Image-Turbo`, `Lightricks/LTX-2.3`.
- **Auto-download encoder FLUX.2** (`core/encoder_downloader.py`, chiamato da `_ensure_flux2_encoder`): usa `hf_hub_download` da repo **`Comfy-Org/...`**. ⚠️ Nota: *Comfy-Org* è un'**organizzazione su HuggingFace** (hosting dei pesi), **non** il software ComfyUI — il download è comunque da HF.
- **Gated**: alcune voci (`gated: true`) richiedono token HF + accettazione licenza sul sito (tutta la famiglia FLUX, LTX-2.3). Non-gated: Z-Image, ERNIE, DA-2, MMAudio, HeartMuLa.

**Conclusione**: provenienza unica = **HuggingFace**, sia nel wizard sia nell'auto-download a runtime quando un modello/encoder selezionato non è presento in locale.
