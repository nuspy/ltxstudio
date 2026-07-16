# Invisible Provenance Signature

*How AI Creative Center invisibly signs everything it produces so you can later prove it was AI‑generated — and how to control it.*

Every image and every video frame produced by **AI Creative Center** carries an invisible **provenance signature**. It is a robust, imperceptible watermark that embeds the brand mark **AI Tools Forge** directly into the pixels, so the origin of a file can be verified even after it has been screenshotted, resized, compressed, or re‑photographed. The signature does not alter the picture in any way you can see.

> **Edition:** **All editions** · **Capability:** `cap.stego` (default **ON**, user‑disableable)

This chapter explains what the signature is, how to turn it on or off, how to add an optional short text tag, and how the standalone **ATF Detector** reads it back. See also [Getting started](01-getting-started.md) and [Advanced settings](12-advanced.md).

---

## What it is

The provenance signature is a **spread‑spectrum watermark in the DCT (frequency) domain**. Instead of writing a visible mark in one corner, it spreads a very low‑amplitude, pseudo‑random pattern across the whole picture. That pattern is generated from a secret key, so only a matching detector can find it — no visible logo, no metadata that a re‑export would strip.

It is designed to be:

- **Invisible** — the signal sits in the mid‑frequency band, below the threshold of human perception.
- **Redundant** — it is repeated over an 8×8 grid of tiles, so it survives partial cropping.
- **Robust** — each tile is normalized to a canonical size before analysis, so the mark survives resizing, JPEG compression, screenshots, and photos of a screen.
- **Anti‑averaging (video)** — the pattern changes on every frame, so averaging frames together does not reveal a fixed mark to remove.
- **Blind** — verification needs only the secret key, never the original file.

> **Honest note.** No invisible watermark is literally impossible to remove. This one is strongly resistant to random edits, frame averaging, and screenshots; a determined attacker with the secret key could still target it. The signature is a provenance aid, not a DRM lock.

---

## Turn it on or off (and add a text tag)

**Where to find it** — **Settings → Provenance** (`ui/settings_dialog.py`, the **Provenance** group).

**Step by step**

1. Open **Settings**.
2. Find the **Provenance** group.
3. Toggle **"Invisible provenance signature (recommended)"** — checked by default.
4. Optionally, type up to five characters into **"Signature text (max 5)"** (for example, your own short handle or code). Leave it empty for a signature with no text — it is still fully valid.
5. Save. The setting takes effect on the next generation.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Invisible provenance signature (recommended)** | Checkbox | On | On / Off | Embeds the invisible signature into every generated photo and video frame. Turn off only if you have a specific reason not to sign your outputs. |
| **Signature text (max 5)** | Text field | Empty | 0–5 characters | Optional short tag stamped invisibly alongside the presence mark and read back by the ATF Detector. Empty = signature without text. Five characters is the physical limit for a screenshot‑robust invisible mark; for longer text, use the file's own metadata instead. |

> **Tips for best results**
> - Leave the signature **on**. It is the only reliable way to later prove one of your renders came from this software.
> - Keep the text tag short and meaningful — an initials handle or short ID. Anything longer will not survive compression and is silently truncated to five characters.
> - The text tag is optional. If you only need to prove "this was AI‑generated with AI Tools Forge," the presence mark alone (empty text) is enough.

---

## Technical parameters

For advanced users and integrators, these are the fixed constants of the watermark (`core/stego.py`). You never set these in the app; they are listed here for transparency and for anyone integrating the ATF Detector.

| Constant | Value | Meaning |
|---|---|---|
| Grid | **8 × 8** (64 tiles) | Redundant impression sites; survives partial crops. |
| Canonical tile size | **80 px** | Each tile is normalized to this size before DCT — the source of resize robustness. |
| Nonce bank | **24** | Number of per‑frame carrier variants the detector tries. |
| Presence amplitude (alpha) | **~9.0** | Tuned for invisibility (~PSNR 39, max pixel delta ~9). Higher values would be visible ("blotchy"). |
| Detection z‑threshold | **5.0** | Presence is declared above this z‑score (false‑positive rate ~1e‑6). |
| Payload | **5 chars + CRC8** = 48 bits | Optional text tag with an integrity checksum. |
| Payload amplitude (beta) | **1.2** | Kept low to preserve invisibility; the priority is a clean image. |
| DCT band | mid‑band, r ∈ [4, 26] | Survives resize/JPEG/screenshot; DC and high frequencies excluded. |
| Perceptual mask | Laplacian‑based, per‑pixel | Hides the signal in local detail/texture (~100%) and nearly zeroes it on smooth areas like skin and sky (~8%). |
| Per‑frame nonce | yes (video) | The carrier changes every frame so averaging frames reveals no fixed pattern. |

The key is derived deterministically from an SHA‑256 stream, so the same signature can be produced and verified identically across Python/NumPy and JavaScript — which is how the [mobile companion](15-rest-api.md) and on‑device detector can read the same mark.

---

## Robustness

The signature is engineered to survive the ways files actually get shared and re‑captured:

- **Resizing** — the grid‑to‑image ratio scales with the image, so the mid‑band survives resampling.
- **JPEG / re‑compression** — the mid‑band is chosen precisely to survive lossy compression.
- **Screenshots** — normalization to the canonical tile size keeps the mark readable after a screen capture.
- **Photo of a screen** — the robust detector performs a geometric re‑sync that compensates for a slightly rotated or perspective‑skewed re‑photograph.
- **Perspective / mild warping** — the robust path searches over small rotations (±15°) and scales (0.90–1.10) to re‑align the grid.

The hardest case remains a strongly skewed photo‑of‑screen with moiré; even then the presence mark usually holds while the optional text tag may be lost.

---

## The ATF Detector

The **ATF Detector** is the standalone tool that reads the signature back. It works **blind** — it needs only the secret key, never your original file — and it returns four values:

- **found** — whether the signature was detected.
- **z‑score** — the confidence of the detection.
- **nonce** — which per‑frame carrier variant matched.
- **text** — the optional 5‑character tag, if one was embedded and its checksum verifies.

Two modes are available (`core/stego.py`):

| Function | Use | Threshold |
|---|---|---|
| `detect_image` | Fast blind detection, no geometric re‑sync — good for clean files. | z ≥ **5.0** |
| `detect_image_robust` | Full geometric re‑sync (rotation × scale search) for skewed photos of a screen. | z ≥ **6.5** |

The robust path uses a higher threshold (6.5) because searching over many geometric warps inflates the maximum correlation under the null hypothesis. For files, `detect_file` runs the robust detector on images and samples several frames of a video, keeping the strongest result.

> **Tips for best results**
> - Use the **robust** detector for anything that has been photographed off a screen or heavily edited; use the **fast** detector for original files.
> - A high z‑score with no recovered text simply means the presence mark survived but the optional tag did not — the file is still proven AI‑generated.
> - Because detection is blind and key‑derived, you can verify third‑party copies of your work without keeping the originals.
