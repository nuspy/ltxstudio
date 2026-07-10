# Review — Lip-sync (tab Audio + tab Video normale/talk/canto) · Video lunghi
_Data: 2026-07-10 · indagine SOLO LETTURA sul codice reale · test di generazione rimandati (ForexGPT)._

## Mappa della catena lip-sync
- **tab Audio** (`ui/tabs/audio_tab.py`): 4 modalità (`native`/`user`/`clone`/`mute`, righe 74-84) → `get_audio_config()`
  (519-542): `audio_mode`, `audio_file`, `voice_sample`, `clone_voice`, `tts_text`, `tts_engine`, `suppress_singing`.
  TTS integrato (Edge/CosyVoice/F5/XTTS/Kokoro) genera un file e lo può applicare come traccia `user`.
- **tab Video** (`ui/widgets/v2v_panel.py`): radio `normal`/`talk`/`sing`/`retake` (75-86) → `get_v2v_config()` (343-360):
  talk → `talk_region`(mouth/face)/`talk_mask_offset`/`talk_reimagine`; sing → `sing_audio_path`/`sing_separate_vocals`
  (demucs)/`sing_audio_scale`; retake → range + regen audio.
- **Routing** (`ui/main_window.py`): `on_generate` (1568) → se `mode in (sing,talk)` chiama `_maybe_multitalk`
  (3093) → `_start_native_multitalk` (2933) → **`_NativeLtx2Worker`** (LTX-2.3 a2v, `workers/ltx2_infer.py`,
  `ptype="a2v"`), **non** Wan/MultiTalk. Post-composite regione bocca in `core/talk_region.py`.

## Bug/gap verificati
| # | Gravità | Problema | Dove |
|---|---|---|---|
| **G1** | grave | **MultiTalk/Wan2.1 = codice morto**: `_NativeMultiTalkWorker` e `workers/wan_multitalk.py` (con resume/anti-drift/overlap/demucs) MAI istanziati. Il path vivo è `_NativeLtx2Worker` a2v. La memory `multitalk-lipsync.md` è disallineata. | `main_window.py:2971` |
| **G3** | grave | **Talk richiede un `audio_file` GIÀ pronto**: legge solo `audio_config()["audio_file"]`; se tab Audio = `native`/`clone` non materializzato/TTS pending → blocco "Audio mancante". La sintesi clone/consenso-TTS gira nel ramo ComfyUI, che Talk scavalca. | `main_window.py:3100-3110` |
| **LV1** | grave | **`_auto_extend` ignorato in NATIVO**: l'estensione autoregressiva GUI (tail-conditioning, anti-freeze, voce affettata, stitch overlap, anchoring) gira SOLO su ComfyUI (`_begin_i2v_extend`). In nativo i lunghi usano `_run_chunked` (continuità solo via ultimo frame). | `main_window.py:1817, 2623` |
| **LV2** | grave | **resume/anti-drift/overlap non raggiungibili per lip-sync**: implementati solo in `wan_multitalk.py` (morto). Il path a2v vivo non riceve `resume`/`ref_anchor_strength`/`anchor_every`/`overlap_frames` (`nat`, 2947-2960). | `main_window.py:2947-2960` |
| **LV3** | medio | **Slider Avanzate inerti in nativo**: `overlap`/`context`/`context_strength`/`ref_anchor`/`anchor_every` si applicano solo al ramo ComfyUI. In nativo l'utente regola cursori che non fanno nulla. | `advanced_tab.long_video_config` |
| **G2** | medio | In `sing`, i param talk-region vengono passati comunque (via latente, inerte solo perché i default sono `None`). | `main_window.py:3142-3143` |
| **G4** | minore | Talk-da-video: `length`/`num_frames` non riallineati alla durata del sorgente (a differenza del ramo ComfyUI). | `main_window.py:3111-3120` |
| **G5** | minore | `audio_scale` default divergente: GUI sing 2.0 vs worker 3.0. | `v2v_panel.py:197` vs `main_window.py:2956` |
| — | minore | `steps`/`guidance`/`seed` scelti nel tab Video non propagati a talk/sing (usa default 8/3.0). | `_maybe_multitalk`→`nat` |

## Fix proposti
- **G1/LV2 (decisione di design):** scegliere l'engine lip-sync lungo. (a) reinstradare `_start_native_multitalk`
  a `_NativeMultiTalkWorker`/`wan_multitalk.py` (ha già resume/anti-drift/overlap/demucs), passando i campi già
  in `params`; **oppure** (b) portare quei parametri in `nat` e implementarli in `ltx2_infer._run_chunked`
  (resume per-chunk + re-anchor + overlap crossfade) e rimuovere il worker morto. Aggiornare la memory.
- **G3 (safe-ish):** in Talk, prima di bloccare, materializzare l'audio come nel ramo ComfyUI (sintesi clone se
  `tts_text`; consenso TTS pending; se `native` → TTS del dialogo o fallback talk-senza-video I2V+lip-sync).
- **G2 (safe):** guardia `if mode=="talk"` prima di passare `talk_region/*`.
- **G4 (safe):** riallineare `length`/`num_frames` alla durata sorgente/audio per talk-da-video.
- **G5 (safe):** uniformare il default `audio_scale` (2.0 o 3.0 ovunque).
- **LV1/LV3 (design):** o rendere `_begin_i2v_extend` backend-agnostica (anche nativo), o mappare
  `overlap/context_strength/ref_anchor/anchor_every` nel worker nativo; in alternativa disabilitare/nascondere
  gli slider inerti quando il backend è nativo (non ingannare l'utente).

**NB:** i fix "safe" sono wiring statico applicabile subito; G1/LV1 toccano i percorsi di generazione e vanno
validati con un test live (rimandato per ForexGPT).

## Stato implementazione (2026-07-10, alla cieca — validazione live pendente)
Scelta: **opzione (b)** — potenziare il path nativo **LTX-2 a2v** (non rianimare il worker Wan morto).
- ✅ **G1/LV2** → **ANTI-DRIFT** (re-anchor: blend ultimo-frame↔foto con `ref_anchor_strength`, o hard reset ogni
  `anchor_every` chunk) e **RESUME** (salta i chunk già validi) in `ltx2_infer._run_chunked`. Parametri passati da
  `_start_native_multitalk`. Wan resta come codice non cablato (non rimosso).
- ✅ **LV1/LV3** → `_generate_video_native` mappa gli slider «Video lungo» (seg/overlap/ref_anchor/anchor_every/
  resume) anche nel nativo normale; il chunk usa la durata-segmento GUI (non più fisso 20s).
- ✅ **G2** (talk-region solo in talk), **G3** (materializza audio talk), **G4** (durata talk-da-video), **G5**
  (audio_scale 2.0 coerente), **steps/cfg/seed** propagati a talk/canto.
- ⏳ **CROSSFADE** join (`_xfade_concat`) = opt-in `LTX_CHUNK_XFADE=1` e SOLO clip senza audio (a2v = concat netto
  per non desincronizzare il lip-sync) → attivare e validare al live-test.
- ⏳ **Validazione live** di tutto (rimandata: ForexGPT).
