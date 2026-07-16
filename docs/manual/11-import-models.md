# Import models

*Bring your own models into AI Creative Center: discover and download community models from Civitai, import files and folders, curate a searchable library, and manage catalog paths — all with automatic classification and content rating.*

The **Import Models** tab is your model manager. It searches Civitai, ingests LoRAs, checkpoints, VAEs, embeddings, and ControlNets from links or local files, sorts them into the right folders by type and architecture, enriches them with metadata, and keeps a browsable library. Behind a single **Import all** button sits a full classification and cataloging pipeline, so most of the time you just paste a link or drop a file and let the app do the rest.

> **Edition:** The Import Models tab requires **Studio Pro**. Within it, **Discover** is gated by `cap.civitai_import` and **Advanced** by `cap.import_models`; in a lower edition these two modes appear with a 🔒 lock badge and an upsell panel. **Import** and **Library** are available whenever the tab is.

**Where to find it** — **Advanced ▸ 📦 Import Models**. The tab opens on the **Import** mode by default.

The tab is organized into four modes, selectable from the mode buttons at the top:

| Mode | Icon | What it's for |
|---|---|---|
| **Discover** | 🔍 | Search and browse Civitai; add results to the import queue. |
| **Import** | ⬇ | Paste a Civitai link/ID or drag files/folders; stage and ingest them. |
| **Library** | 🗂 | Browse, filter, and edit your imported models. |
| **Advanced** | ⚙ | Configurable paths, compatibility patch, conversion, and the full ingest pipeline. |

A shared operations **log** and a **progress bar** run along the bottom (visible in Import and Advanced modes).

---

## Discover

> **Edition:** Studio Pro · **Capability:** `cap.civitai_import`

**What it does** — Searches the Civitai community catalog and lets you add any result straight to your import queue.

**Where to find it** — **Import Models ▸ Discover**.

**Step by step**
1. Type into **Search on Civitai…** and press **Search** (or leave it blank to browse).
2. Narrow results with the **type** and **base-model** filters.
3. (Optional, Unlocked 18+) tick the adult-content checkbox to include mature results.
4. On any result card, click **＋ Add** to place it in the import queue.
5. Switch to **Import** to review the queue and click **Import all**.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Search on Civitai…** | Text field | empty | Any query | Free-text search. Empty = browse. |
| Type filter | Dropdown | **All** | All · LoRA · Checkpoint · VAE · Embedding · ControlNet | Restricts results to one model type. |
| Base-model filter | Dropdown | **All base models** | All base models · Flux.1 D · SDXL 1.0 · Pony · Illustrious · SD 1.5 · SD 3.5 | Restricts results to one base architecture. |
| **18+ content** (adult content) | Checkbox | Off | On / Off | Includes adult results. Requires the **Unlocked (18+) mode**; otherwise it is disabled and shows an "adult content feature not enabled" notice. |
| **Search** | Button | — | — | Runs the query. |
| **＋ Add** | Button (per card) | — | — | Adds that model to the import queue. |

When there are no matches the panel reads *"No results."* An empty search shows the hint *"Search for models and add them to the import queue."*

**Specializations — Classic vs. Pro**
In the **Pro** variant, thumbnails are **lazy-loaded**: results appear instantly and previews download only when you click **🖼 Load previews**, keeping large searches responsive. The Pro Discover page also hosts the Civitai **link/ID** field (moved here from Import).

**Tips for best results**
> - Use the base-model filter to avoid importing a LoRA that doesn't match the checkpoint you generate with — a Flux LoRA won't work on an SDXL model.
> - Browse with an empty query plus a type filter to see what's trending for a given architecture.

---

## Import

> **Edition:** Studio Pro · **Capability:** `cap.import_models`

**What it does** — Takes Civitai links/IDs and local files or folders, recognizes each model, categorizes it, and moves it to the correct folder ready to use.

**Where to find it** — **Import Models ▸ Import**.

**Step by step**
1. Add models to the queue by any of these:
   - Paste a link or ID into the Civitai field and click **Add from Civitai**.
   - Drag files or folders onto the drop area.
   - Click **Choose files…** or **Choose folder…**.
2. Review each queued item's badges (type, family/architecture, content rating) and destination.
3. Remove unwanted items with **✕**, or empty the whole list with **Clear queue**.
4. Click **⬇ Import all** to stage and catalog everything.

**Parameters**

| Control | Type | Default | Options | What it does |
|---|---|---|---|---|
| Civitai link/ID | Text field | empty | `https://civitai.com/models/…` or an ID | The model to fetch from Civitai. |
| **Add from Civitai** | Button | — | — | Resolves the link/ID and queues the model. |
| Drop area | Drag target | — | — | Drop files or folders of models here. |
| **Choose files…** | Button | — | — | Pick model files from disk. |
| **Choose folder…** | Button | — | — | Pick a folder of models. |
| **⬇ Import all** | Button | — | — | Stages every queued item (download/copy → classify → catalog). |
| **Clear queue** | Button | — | — | Empties the import queue. |

Recognized model file types are `.safetensors`, `.ckpt`, `.pt`, `.pth`, and `.gguf`. Each queued card shows a destination (`→ …`) and badges for type, family, and content rating; when empty the queue reads *"The queue is empty — add models above."*

**Staging** — When you click **Import all**, each item is either downloaded from Civitai or copied into your inbox, with live status: *downloading… → copying… → ready* (or *error*). Items are then cataloged automatically.

**Specializations — Classic vs. Pro**
The **Pro** variant moves the Civitai link/ID field to the **Discover** page and shows a dashed-outline empty state: *"No models queued. Drag a file or add from «Discover»."* The four-mode header uses **segmented buttons** with 🔒 lock badges on the gated modes.

**Tips for best results**
> - You can queue Civitai links and local files together and import them in one pass.
> - Dropping a whole folder is the fastest way to bulk-import a downloaded model pack.

---

## The classification pipeline

Every imported model is analyzed automatically so it lands in the right folder with the right metadata. The analysis fills in these fields:

| Field | Meaning |
|---|---|
| **type** | LoRA · Checkpoint · VAE · TextEncoder · Embedding · ControlNet · Upscaler … |
| **family** | Flux · SDXL · SD1.5 · SD3 · LTX · Wan21 · Wan22 · Z-Image · Ernie … |
| **dest_subfolder** | Where the file is placed, e.g. `LoRA/Flux`. |
| **sha256** | The file's hash, used for a Civitai by-hash lookup. |
| **triggers** | Trained trigger words (activation keywords). |
| **preview** | A preview image URL. |
| **base_arch** | The base architecture reported by Civitai. |

**How type and family are determined** — The app first inspects the file itself (by structure and size) to classify type and family. If that is inconclusive, it falls back to the native model detector, which maps internal codes (for example `flux1`/`flux2` → **Flux**, `sdxl` → **SDXL**, `ltxv` → **LTX**, `wan21` → **Wan21**) to standard names. For `.safetensors`/`.ckpt`/`.pt` files it also computes the SHA-256 and queries Civitai **by hash**; a match overrides the guessed type/family and supplies triggers, a preview, and the base model.

---

## Content rating (adult-content gating)

AI Creative Center classifies each model's content rating through a **cascade**, stopping at the first confident source. This is what drives whether a model is treated as 18+/adult content:

1. **Civitai metadata (authoritative)** — if the model comes from Civitai, its rating flags are used directly.
2. **Name/tag heuristic** — otherwise, the filename and tags are scanned for adult-content hints.
3. **Preview classifier** — if a preview image is available, an on-device classifier inspects it.
4. **Conservative default** — if still uncertain, the model is treated as adult content (the safe default).

The **Unlocked (18+) mode** (`cap.adult_mode`) governs what happens next. When it is **off**, any model rated as adult content is **skipped** during import (the log notes *"Skipped (adult content, feature not enabled)"*), and the adult-content checkboxes in Discover and Library are disabled. When it is **on**, adult-rated models import and browse normally. You can always correct a model's rating by hand in the Library detail view.

---

## Library

> **Edition:** Studio Pro

**What it does** — Browses your imported models as cards, filtered by type, architecture, and content rating, and lets you edit each model's trigger words, notes, and rating.

**Where to find it** — **Import Models ▸ Library**.

**Step by step**
1. Filter the catalog with **Filter the library…**, the type dropdown, the architecture dropdown, and the adult-content checkbox.
2. Click a card's **Details** to open its editor.
3. Edit **Trigger words** and **Description / notes**, adjust the content-rating checkbox, and save.
4. Use **↻** to reload the library after importing new models.

**Parameters**

| Control | Type | Default | Options | What it does |
|---|---|---|---|---|
| **Filter the library…** | Text field | empty | Any text | Filters cards by name. |
| Type filter | Dropdown | **All** | All · LoRA · Checkpoint · VAE · Embedding · ControlNet | Restricts to one type. |
| Architecture filter | Dropdown | **All archs** | All archs · (families found in your catalog) | Restricts to one architecture. |
| **18+ content** (adult content) | Checkbox | Off | On / Off | Shows adult-rated models. Requires Unlocked (18+) mode. |
| **↻** | Button | — | — | Reloads the library. |
| **Details** | Button (per card) | — | — | Opens the model editor. |

The detail editor exposes: **Trigger words (comma-separated)**, **Description / notes**, and an **18+ content** (content rating) checkbox. Your edits are stored as manual overrides and re-applied on every re-scan, so a metadata refresh never wipes them. When the library is empty it reads *"No models in the library (import some)."*

**Specializations — Classic vs. Pro**
The **Pro** variant adds per-card quick actions: a **★/☆** favorite toggle and an **📂 Open folder** button (opens the model's folder in your file manager), alongside **Details**.

**Tips for best results**
> - Fill in **Trigger words** for LoRAs that aren't on Civitai — the generation tabs surface these so you remember how to activate them.
> - Use the architecture filter to quickly find all LoRAs compatible with the checkpoint you're about to use.

---

## Advanced

> **Edition:** Studio Pro · **Capability:** `cap.import_models`

**What it does** — Exposes the catalog's configurable paths, the ComfyUI compatibility patch, model-format conversion, and the full manual ingest pipeline (analyze → place → scan → enrich → build HTML).

**Where to find it** — **Import Models ▸ Advanced**.

### Configurable paths

A collapsible group where you can point the catalog at the right locations. Each has a text field, a browse button, and a tooltip:

- **Models folder** — where classified models are placed.
- **ComfyUI folder** — your ComfyUI installation root.
- **Dictionary root (dict_root)** — where `catalog.json`, `manual_enrichment.json`, and the HTML catalog live.
- **Inbox** — the staging folder that Import copies into and Place moves from.
- **Tools folder** — where the catalog scripts live.

Buttons: **Save paths** (persists and syncs to the tools' `paths.json`), **Reset to defaults**, and **Suggest from ComfyUI** (proposes paths from a detected ComfyUI install — the final choice stays yours). The defaults match your current installation; change them only if you've moved ComfyUI, your models, or the dictionary.

### Compatibility check (ComfyUI patch)

Installs a compatibility patch as a ComfyUI custom node so LTX-2.3 runs on any ComfyUI version **without editing core files** (it survives updates). The group shows the current install status and offers **Install / Update patch** and **Uninstall patch**. A checkbox unlocks the LTX-2 prompt-enhancer so it describes subjects without refusals (for adult content). After installing, restart ComfyUI to apply.

### Model conversion (ComfyUI → native)

One-time converters that turn ComfyUI/quantized models into the native backend format: **Gemma-3 → bf16** and **Text encoder → HF**.

### The ingest pipeline

The Advanced mode gives you each stage of cataloging as its own button, plus a one-click full run. On the source folder they act as follows:

| Operation | What it does | Network |
|---|---|---|
| **Analyze** | Classifies source files by type/architecture **without moving them** (dry run). | No |
| **Place / Move** | Moves files from the source into the correct folders (LoRA/Checkpoints/VAE/… or workflows). | No |
| **Scan** | Updates `catalog.json` with base metadata (architecture, triggers, content rating). | No |
| **Enrich** | Civitai by-hash lookup plus re-applies your manual overrides. | Yes |
| **Overrides** | Re-applies only your manual overrides (`manual_enrichment.json`). | No |
| **Build HTML** | Regenerates the static HTML catalog (index + entry pages). | No |
| **Full ingest** | Runs Place → Scan → Enrich → Build HTML in sequence. | Yes (Enrich step) |

You can also set the **source folder**, add files/folders to a working list, **copy to the inbox**, and clear the list.

**Tips for best results**
> - Use **Analyze** first as a dry run to confirm files will be classified correctly before **Place** actually moves them.
> - Run **Enrich** while online to pull triggers and previews from Civitai; run **Overrides** offline to re-apply just your hand edits.
> - If video generation stops working after a ComfyUI update, re-run **Install / Update patch** and restart ComfyUI.

---

*Related: [Getting started](01-getting-started.md) for the Classic/Pro interface switch and the model-overrides setting, [Advanced settings](12-advanced.md) for where Import Models lives, and the [Appendix](17-appendix.md) for the full capability→edition table and file locations.*
