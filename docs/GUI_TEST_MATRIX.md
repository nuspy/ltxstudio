# GUI Test Matrix — Inventario controlli ↔ copertura test di regressione

> Task #94 "Test GUI reale con qtbot" — **Fase F1** (inventario completo).
> Aggiornato l'11/07/2026 dall'analisi di `ui/main_window.py`, `ui/tabs/*.py`, `ui/mask_canvas.py`,
> `ui/widgets/prompt_builder.py`, `ui/film_studio/film_studio_window.py` e `tests/gui_regression/cases*.py`.
> Sostituisce il draft del 04/07; i pattern qtbot e lo stato bug #95 del draft sono conservati nelle Appendici A/B.

## Metodologia e legenda

L'harness attuale (`tests/gui_regression/harness.py`) costruisce la `MainWindow` offscreen e intercetta
le chiamate GUI→backend con uno **Spy** (patch di `_enqueue`, `ComfyDriver.submit`, driver nativo): i casi
impostano i widget **programmaticamente** (`setValue`/`setChecked`/`setCurrentText`) o invocano gli slot,
poi verificano i parametri catturati. `run_real.py` esegue gli stessi casi col backend vero (GPU),
`live_smoke.py` genera output reali. **Nessun caso oggi simula input utente reale (click/type/select
via qtbot): è l'oggetto delle fasi F2/F3.**

Stato copertura:

- ✅ **Coperto** — il controllo (o il suo slot) è esercitato da almeno un caso.
- 🟡 **Parziale** — coperto solo indirettamente (slot condiviso, stato solo letto, dialog patchata).
- ❌ **GAP** — mai toccato da alcun caso.

Sono esclusi dal conteggio gli elementi di sola visualizzazione (QLabel, QProgressBar, log read-only).

## Sommario copertura

| # | Sezione / Tab | Controlli | ✅ Coperti | 🟡 Parziali | ❌ Gap |
|---|---|---:|---:|---:|---:|
| 1 | Finestra principale (main_window) | 16 | 1 | 5 | 10 |
| 2 | Immagini (Frame1Tab) | 63 | 44 | 0 | 19 |
| 3 | Canvas maschere (MaskCanvas) | 13 | 1 | 1 | 11 |
| 4 | Audio/Voce (AudioTab) | 19 | 14 | 2 | 3 |
| 5 | Video (VideoTab) | 37 | 27 | 0 | 10 |
| 6 | Avanzate (SamplerSeedGroup + AdvancedTab) | 25 | 24 | 0 | 1 |
| 7 | ControlNet (ControlNetTab) | 5 | 3 | 1 | 1 |
| 8 | Prolunga video (VideoExtendTab) | 7 | 7 | 0 | 0 |
| 9 | Modifica Video AI (VideoEditAITab) | 14 | 10 | 1 | 3 |
| 10 | Qualità → Foto (RestoreTab) | 16 | 13 | 1 | 2 |
| 11 | Qualità → Video (VideoDefTab) | 20 | 16 | 0 | 4 |
| 12 | TRELLIS 3D (TrellisTab) | 21 | 18 | 2 | 1 |
| 13 | Rig 3D (RigTab) | 17 | 14 | 1 | 2 |
| 14 | Animazione/Mocap (AnimTab) | 12 | 12 | 0 | 0 |
| 15 | Converti 3D (ConvertTab) | 6 | 6 | 0 | 0 |
| 16 | LoRA Train Img (TrainTab) | 27 | 21 | 0 | 6 |
| 17 | LoRA Wizard Vid (LoraVideoTab) | 25 | 24 | 0 | 1 |
| 18 | LoRA Wizard Img (LoraImageTab) | 11 | 8 | 0 | 3 |
| 19 | Train LoRA Video (TrainLoraVideoTab) | 12 | 0 | 0 | 12 |
| 20 | Musica (MusicTab) | 57 | 32 | 0 | 25 |
| 21 | Mondo 3D (WorldTab) | 15 | 15 | 0 | 0 |
| 22 | Pesi modelli (WeightsTab) | 13 | 10 | 0 | 3 |
| 23 | Selettore LoRA (LoraTab widget) | 8 | 7 | 0 | 1 |
| 24 | Modelli (ImportModelsTab + Pro) | 42 | 0 | 0 | 42 |
| 25 | PromptBuilder (widget + dialog) | 13 | 0 | 0 | 13 |
| 26 | Film Studio (finestra, sintetico) | 15 | 0 | 0 | 15 |
| | **TOTALE** | **529** | **327** | **14** | **188** |

Copertura complessiva: **327/529 (62%)** pieni, **341/529 (64%)** includendo i parziali.

---

## 1. Finestra principale (`ui/main_window.py`)

Controlli costruiti inline: pulsantiera generazione, menu, barra di navigazione tab custom, dock coda.

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `gen_btn` | QPushButton | `on_generate()` (genera video) | `video_t2v_generation`, `video_i2v_generation`, quasi tutti i casi video/controlnet/advanced | ✅ slot chiamato direttamente, click non simulato |
| `cancel_btn` | QPushButton | `driver.interrupt()` | — | ❌ GAP (interruzione mai testata) |
| `free_vram_btn` | QPushButton | `_on_free_vram()` | — | ❌ GAP |
| `navigator_btn` | QPushButton | `_open_navigator()` | (slot coperto solo via `advanced_open_navigator`) | 🟡 bottone mai premuto |
| `act_wiz` (menu) | QAction | `open_wizard()` | — | ❌ GAP |
| `act_set` (menu) | QAction | `open_settings()` (SettingsDialog) | — | ❌ GAP |
| `act_nav` (menu) | QAction | `_open_navigator()` | (via `advanced_open_navigator`) | 🟡 |
| `act_ana` (menu) | QAction | `_open_analyzer()` | (via `advanced_open_analyzer`, `advanced_analyzer_reads_sampler`) | 🟡 |
| `act_wfl` (menu) | QAction | `_open_workflow()` | (via `advanced_open_workflow`) | 🟡 |
| `act_film` (menu) | QAction | `_open_film_studio()` | — | ❌ GAP (Film Studio mai aperto dai test) |
| `_grp_models` | QActionGroup | `_set_models_variant()` Classica/Pro | — | ❌ GAP (switch variante Modelli) |
| `_grp_fs` | QActionGroup | `_set_filmstudio_variant()` | — | ❌ GAP |
| `_lang_btn` + menu | QToolButton/QMenu | `_change_language(code)` | — | ❌ GAP (cambio lingua UI mai testato) |
| `_tab_nav_btns` | QToolButton ×N | `tabs.setCurrentIndex()` | (indiretto: `advanced_subtabs_switch`, `train_tab_switch`, `anim_select_tab` usano `setCurrentIndex`) | 🟡 bottoni custom mai cliccati |
| `_queue_btn` | QToolButton | mostra/nasconde `queue_dock` | — | ❌ GAP (pannello coda) |
| `_mem_btn` | QToolButton | `_toggle_mem_monitor()` | — | ❌ GAP |

Scorciatoia `Ctrl+G` (navigator) e le dialog secondarie (SettingsDialog, SetupWizard, NavigatorDialog,
WorkflowDialog, SettingsAnalyzerDialog, VideoPlayerDialog, MemoryMonitor, UpsellDialog) sono **tutte
fuori copertura** (le dialog vengono patchate dall'harness per non bloccare).

---

## 2. Tab Immagini (`ui/tabs/frame1_tab.py`, classe `Frame1Tab`)

Generazione immagine (FLUX/Z-Image/Ernie), inpaint/outpaint, integrazione/composizione, ricolora, segmenta, upscale.
File di test: `cases.py`, `cases_frame1_tab.py`. NB: `frame1_tab.prompt`/`.neg` sono alias dei widget del tab Video.

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `img_nsfw` | QComboBox | filtro contenuti SFW/NSFW | casi `frame1_generate_image_*` | ✅ |
| `gen_arch` | QComboBox | architettura modello (FLUX/Z-Image/Ernie) | casi generate per-arch | ✅ |
| `gen_engine_sel` | QComboBox | motore (Comfy/Diffusers) | — | ❌ GAP |
| `gen_model` | QComboBox | modello immagine | `image_generation_t2i`, `frame1_generate_image_auto_select` | ✅ |
| `gen_precision` | QComboBox | filtro precisione (fp16/fp8/fp4/gguf) | — | ❌ GAP (coperto solo per video) |
| `gen_star` | QToolButton | preferito modello immagine | — | ❌ GAP |
| `gen_encoder` | QComboBox | text encoder | — | ❌ GAP |
| `gen_clip_type` | QComboBox | variante CLIP | — | ❌ GAP |
| `gen_vae` | QComboBox | VAE | — | ❌ GAP |
| `gen_lora_tab` | LoraTab (embedded) | LoRA per immagini | — | ❌ GAP (i casi LoRA testano solo il selettore del tab Video) |
| `gen_sampler` | QComboBox | sampler | `frame1_generate_image_sampler_scheduler` | ✅ |
| `gen_scheduler` | QComboBox | scheduler | `frame1_generate_image_sampler_scheduler` | ✅ |
| `gen_enhance` | QCheckBox | enhance prompt via LLM | `frame1_generate_image_enhance_toggle` | ✅ |
| `gen_steps` | QSpinBox | passi diffusione | `frame1_generate_image_cfg_steps` | ✅ |
| `gen_cfg` | QDoubleSpinBox | CFG | `frame1_generate_image_cfg_steps` | ✅ |
| `gen_seed` | QSpinBox | seed | `frame1_generate_image_seed_fixed` | ✅ |
| `gen_seed_rand` | QCheckBox | seed casuale | `frame1_generate_image_seed_fixed` | ✅ |
| `gen_prompt` | QPlainTextEdit | prompt positivo | tutti i casi generate + `frame1_generate_image_empty_prompt_guard` | ✅ |
| `gen_neg` | QPlainTextEdit | prompt negativo immagine | — | ❌ GAP |
| `gen_btn` | QPushButton | `generate_image()` | tutti i casi `frame1_generate_image_*` (slot diretto) | ✅ |
| `gen_pulid_enable` | QCheckBox | abilita PuLID | `frame1_generate_image_pulid_face` | ✅ |
| `gen_pulid_face` | QLineEdit | foto volto PuLID | `frame1_generate_image_pulid_face` | ✅ |
| `gen_pulid_face_btn` | QPushButton | sfoglia foto volto | — | ❌ GAP (dialog) |
| `gen_pulid_strength` | QDoubleSpinBox | forza PuLID | `frame1_generate_image_pulid_face` | ✅ |
| `gen_pulid_start` | QSpinBox | step iniziale PuLID | — | ❌ GAP |
| `gen_pulid_truecfg` | QDoubleSpinBox | true-CFG PuLID | — | ❌ GAP |
| `inpaint_prompt` | QPlainTextEdit | descrizione area da riempire | `image_inpaint`, `frame1_inpaint_*`, `frame1_inpaint_no_mask_guard` | ✅ |
| `outpaint_prompt` | QPlainTextEdit | descrizione estensione | `frame1_outpaint_generate` | ✅ |
| `edit_enhance` | QCheckBox | arricchisci prompt edit con AI | — | ❌ GAP |
| `inp_neg` | QPlainTextEdit | negativo inpaint | — | ❌ GAP |
| `strength` | QSlider | forza rigenerazione maschera | `frame1_inpaint_strength_steps_seed` | ✅ |
| `inp_guidance` | QDoubleSpinBox | guidance inpaint | — | ❌ GAP |
| `inp_feather` | QSpinBox | sfumatura bordi | — | ❌ GAP |
| `inp_model` | QComboBox | modello inpaint/fill | `image_inpaint`, `frame1_inpaint_dynamic_unet`, `frame1_outpaint_generate` | ✅ |
| `inp_star` | QToolButton | preferito modello inpaint | — | ❌ GAP |
| `inp_lora` | QComboBox | LoRA per inpaint | — | ❌ GAP |
| `inp_steps` | QSpinBox | passi inpaint | `frame1_inpaint_strength_steps_seed` | ✅ |
| `inp_seed` / `inp_seed_rand` | QSpinBox/QCheckBox | seed inpaint | `frame1_inpaint_strength_steps_seed` | ✅ |
| `integrate_prompt` | QPlainTextEdit | prompt integrazione layer | `frame1_integrate_kontext_native`, `frame1_integrate_no_layer_guard` | ✅ |
| `integrate_engine` | QComboBox | motore (Kontext/FLUX.2) | `frame1_integrate_flux2`, `frame1_recolor_flux2`, `frame1_compose_flux2` | ✅ |
| `integrate_puzzle` | QCheckBox | puzzle LoRA | `frame1_integrate_puzzle_lora` | ✅ |
| `integrate_model` | QComboBox | modello integrazione | `frame1_integrate_flux2`, `frame1_compose_flux2` | ✅ |
| `int_neg` | QPlainTextEdit | negativo integrazione | — | ❌ GAP |
| `int_steps` | QSpinBox | passi integrazione | — | ❌ GAP |
| `int_guidance` | QSlider | guidance integrazione | `frame1_integrate_guidance_seed` | ✅ |
| `int_seed` / `int_seed_rand` | QSpinBox/QCheckBox | seed integrazione | `frame1_integrate_guidance_seed` | ✅ |
| `compose_list` | QListWidget | foto da comporre | `frame1_compose_*`, `frame1_compose_need_two_guard` | ✅ |
| `compose_method` | QComboBox | multiref/stitch | `frame1_compose_multiref_kontext`, `frame1_compose_stitch` | ✅ |
| `compose_prompt` | QPlainTextEdit | prompt scena composta | `frame1_compose_multiref_kontext` | ✅ |
| `color_target` | QLineEdit | colore target | `frame1_recolor_kontext_native`, `frame1_recolor_target_guard`, `frame1_recolor_flux2` | ✅ |
| `color_subject` | QLineEdit | soggetto da ricolorare | `frame1_recolor_kontext_native` | ✅ |
| `color_engine` | QComboBox | motore ricolora | tutti i casi `frame1_recolor_*` | ✅ |
| `seg_detector` | QComboBox | detector (DINO/DINO+SAM) | `frame1_segment_text_dino_sam`, `frame1_segment_text_dino_only` | ✅ |
| `seg_text` | QLineEdit | testo oggetto da segmentare | casi segment + `frame1_segment_empty_text_guard` | ✅ |
| `seg_threshold` | QDoubleSpinBox | soglia detection | `frame1_segment_text_dino_sam` | ✅ |
| `up_model` | QComboBox | modello upscale | `frame1_upscale_*`, `frame1_upscale_no_model_guard` | ✅ |
| `up_mode` | QComboBox | modalità Veloce/Rifinisci | `frame1_upscale_scale_factors`, `frame1_upscale_refine_mode` | ✅ |
| `up_scale` | QComboBox | fattore scala 2x/3x/4x | `frame1_upscale_scale_factors` | ✅ |
| `up_refine_model` | QComboBox | modello FLUX.2 per refine | `frame1_upscale_refine_mode` | ✅ |
| `up_refine_str` | QDoubleSpinBox | forza refine | `frame1_upscale_refine_mode` | ✅ |
| `up_refine_steps` | QSpinBox | passi refine | `frame1_upscale_refine_mode` | ✅ |

---

## 3. Canvas maschere (`ui/mask_canvas.py`, classe `MaskCanvas`)

Canvas grafico condiviso della tab Immagini: layer, maschere, strumenti pittura, trasformazioni.
I casi esistenti **iniettano immagini e maschere via API** (`load_image`, manipolazione diretta di
`mask_image`): l'interazione mouse/tastiera reale non è mai esercitata.

| Controllo / Azione | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| API `load_image` / `export*` / `has_mask` | metodi | caricamento/esportazione programmatica | casi inpaint/outpaint/i2v/trellis (indirettamente) | ✅ unica via oggi usata |
| `aspect_btn` | QToolButton+menu | scelta aspect ratio → `resolution_chosen` | (risoluzione impostata via API in `image_outpaint_aspect`, `frame1_outpaint_aspect_preserved`) | 🟡 menu mai aperto |
| `_mode_btns` (move/brush/erase/fill/lasso/sam_point) | QPushButton ×6 | `set_mode(m)` cambio strumento | — | ❌ GAP |
| `brush_sl` | QSlider | `set_brush(v)` diametro pennello (3–200) | — | ❌ GAP |
| pittura brush/erase | mouse drag | `paint_at()` / CompositionMode_Clear | — | ❌ GAP (maschera disegnata solo via API) |
| riempimento | click (fill) | `fill_at()` flood-fill | — | ❌ GAP |
| lazo con snap | drag (lasso) | `lasso_begin/drag/finish()` | — | ❌ GAP |
| punto SAM | click (sam_point) | `sam_point_at()` → `request_segment_region` | — | ❌ GAP (`segment_by_text` è testato, la selezione per punto no) |
| undo/redo | metodi/scorciatoie | `undo()` / `redo()` (stack 100) | — | ❌ GAP |
| `layer_list` + su/giù/elimina/importa | QListWidget+QPushButton | gestione layer (`_on_layer_sel`, `_layer_up/down/del`, `_import_layer`) | — | ❌ GAP |
| `upload_overlay` | QPushButton | `_load_dialog()` carica immagine | — | ❌ GAP (dialog) |
| trasformazioni (scala/rotazione) | drag maniglie | `begin/update/end_transform()` | — | ❌ GAP |
| rotella mouse | wheel | zoom vista / diametro pennello | — | ❌ GAP |

---

## 4. Tab Audio/Voce (`ui/tabs/audio_tab.py`, classe `AudioTab`)

File di test: `cases_audio_tab.py` (22 casi + casi per-engine dinamici).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `rb_native` | QRadioButton | audio nativo LTX | `audio_config_native`, `audio_no_sing_toggle`, `audio_mode_switch_no_crash` | ✅ |
| `rb_user` | QRadioButton | file/registrazione utente | `audio_config_user_file`, `audio_use_tts_as_audio` | ✅ |
| `rb_clone` | QRadioButton | voice clone | `audio_config_clone` | ✅ |
| `rb_mute` | QRadioButton | nessun audio | `audio_config_mute` | ✅ |
| `no_sing` | QCheckBox | sopprimi canto | `audio_no_sing_toggle` | ✅ |
| `pick_btn` | QPushButton | `_pick()` scegli file audio | — | ❌ GAP (path impostato via API nei casi) |
| `rec_btn` | QPushButton | `_toggle_rec()` registra dal microfono | — | ❌ GAP |
| `tts_engine` | QComboBox | motore TTS (8 motori itemData) | `audio_engine_change_reloads`, `audio_install_button_visibility`, casi per-engine | ✅ |
| `tts_install_btn` | QPushButton | `_install_selected_engine()` | `audio_install_button_visibility`, `audio_install_engine_no_crash` | ✅ |
| `tts_voice` | QComboBox | preset voce | (`audio_reload_voices_no_crash`, `audio_tts_generate_no_voice`) | 🟡 selezione esplicita mai variata |
| `b_new` | QPushButton | `_voice_create()` | `audio_voice_create_cancelled` | ✅ |
| `b_edit` | QPushButton | `_voice_edit()` | `audio_voice_edit_preset_blocked` | ✅ |
| `b_dup` | QPushButton | `_voice_duplicate()` | — | ❌ GAP |
| `b_del` | QPushButton | `_voice_delete()` | `audio_voice_delete_preset_blocked` | ✅ |
| pulsanti marker SSML | QToolButton ×8 | `_insert_marker()` pause/emozioni | `audio_ssml_marker_insert`, `audio_ssml_marker_wrap` | ✅ |
| `tts_text` | QPlainTextEdit | testo da sintetizzare | `audio_tts_generate_*`, `audio_preview_strips_markers` | ✅ |
| `tts_gen_btn` | QPushButton | `_generate_tts()` | `audio_tts_generate_default`, `audio_tts_synth_error_no_crash`, casi per-engine | ✅ |
| `tts_play_btn` | QPushButton | `_play_tts()` riproduci | (stato enabled verificato in `audio_tts_generate_default`) | 🟡 click mai eseguito |
| `tts_use_btn` | QPushButton | `_use_tts_as_audio()` | `audio_use_tts_as_audio`, `audio_pending_tts_consent` | ✅ |

---

## 5. Tab Video (`ui/tabs/video_tab.py`, classe `VideoTab`)

File di test: `cases.py`, `cases_video_tab.py` (26 casi + casi per-modello dinamici).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `prompt` | QPlainTextEdit | prompt video | `video_t2v_generation` e quasi tutti; `video_t2v_empty_prompt_guard` | ✅ |
| `neg` | QPlainTextEdit | prompt negativo | `video_t2v_negative_prompt` | ✅ |
| `preset_combo` | QComboBox | applica preset video | `video_preset_apply_builtin`, `video_preset_save_and_delete` | ✅ |
| `preset_save_btn` | QPushButton | `_save_current_preset()` | `video_preset_save_and_delete` (slot diretto) | ✅ |
| `preset_del_btn` | QPushButton | `_delete_selected_preset()` | `video_preset_save_and_delete` (slot diretto) | ✅ |
| `nsfw` | QComboBox | filtro SFW/NSFW | `video_nsfw_toggle_reveals_models`, `lora_nsfw_filter` | ✅ |
| `model` | QComboBox | modello video (itemData kind) | `video_model_selection_reaches_backend`, casi per-modello dinamici | ✅ |
| `precision` | QComboBox | filtro precisione | `video_precision_filter_gguf` | ✅ |
| `model_star` | QToolButton | preferito modello | `video_toggle_favorite_model` (slot `_toggle_fav_model_current`) | ✅ |
| `res_preset` | QComboBox | preset risoluzione | `video_resolution_preset_landscape`, `video_resolution_custom` | ✅ |
| `width` / `height` | QSpinBox | dimensioni | `video_resolution_*` | ✅ |
| `fps` | QSpinBox | fotogrammi al secondo | `video_fps_duration_num_frames` | ✅ |
| `length` | QDoubleSpinBox | durata (s) | `video_fps_duration_num_frames`, `advanced_long_video_autoextend_*` | ✅ |
| `img_strength` | QSlider | forza immagine i2v | `video_img_strength_ref` | ✅ |
| `i2v_method` | QComboBox | addguide/inplace | `video_i2v_method_inplace` | ✅ |
| `pulid_kf_face` | QLineEdit | foto volto keyframe PuLID | — | ❌ GAP |
| `pulid_kf_browse` | QPushButton | sfoglia foto | — | ❌ GAP |
| `pulid_kf_gen` | QPushButton | genera keyframe con identità (`request_pulid_keyframe`) | — | ❌ GAP |
| `video_engine_sel` | QComboBox | motore video (ltx2_infer/comfy) | — | ❌ GAP (routing testato via config, non dal combo) |
| `teacache` | QDoubleSpinBox | soglia TeaCache | — | ❌ GAP |
| `downscale_half`/`upscale_x2`/`refine_hr` | QCheckBox ×3 | legacy nascosti | — | ❌ (nascosti, bassa priorità) |
| `fr_auto` | QCheckBox | restauro volti automatico | `video_face_restore_auto` | ✅ |
| `fr_auto_model` | QComboBox | modello restauro | — | ❌ GAP |
| `fr_auto_fidelity` | QSlider | fedeltà restauro | — | ❌ GAP |
| `fs_auto` | QCheckBox | face-swap automatico | `video_face_swap_auto` | ✅ |
| `fs_face` | QLineEdit | foto volto swap | `video_face_swap_auto` | ✅ |
| `fs_face_btn` | QPushButton | sfoglia volto | — | ❌ GAP (dialog) |
| `rife_auto` | QCheckBox | interpolazione RIFE | `video_rife_auto` | ✅ |
| `rife_factor` | QComboBox | fattore x2/x4 | `video_rife_auto` | ✅ |
| `v2v_panel.rb_retake` | QRadioButton | modalità ReTake | `video_v2v_retake_native`, `video_v2v_retake_no_source_guard` | ✅ |
| `v2v_panel.rb_talk` | QRadioButton | Just_Talk lip-sync | `video_v2v_talk_no_audio_guard` | ✅ |
| `v2v_panel.rb_sing` | QRadioButton | canta (MultiTalk) | `video_v2v_sing_multitalk` | ✅ |
| `v2v_panel.rb_audio_regen` | QRadioButton | rigenera audio | `video_v2v_retake_native` | ✅ |
| `v2v_panel` (altri: `btn_pick`/`btn_lastgen`, `sl_start`/`sl_end`, `cmb_region`, `sp_offset`, `cb_reimagine`, `btn_sing_audio`, `cb_sing_vocals`, `sl_sing_scale`) | vari | sorgente/intervallo/opzioni V2V | — | ❌ GAP (file/range impostati via API o non toccati) |
| `storyboard_panel.enable` | QCheckBox | abilita storyboard multi-scena | `video_storyboard_two_scenes` | ✅ |
| `storyboard_panel._rows` (scene) | righe dinamiche | keyframe a tempo | `video_storyboard_two_scenes` | ✅ |

---

## 6. Avanzate (`ui/tabs/advanced_tab.py`: `SamplerSeedGroup` + `AdvancedTab`)

`SamplerSeedGroup` è embedded nel tab Video (`video_tab.sampler_seed`). File di test: `cases_advanced_tab.py`
(16 casi statici + casi dinamici per sampler/scheduler/encoder/quality-preset).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `steps` | QSpinBox | passi campionamento | `advanced_steps_reach_backend`, `video_sampler_steps_cfg` | ✅ |
| `cfg` | QDoubleSpinBox | CFG | `advanced_cfg_reach_backend` | ✅ |
| `quality_preset` | QComboBox | preset qualità (applica steps/cfg) | casi dinamici `advanced_quality_preset_*` | ✅ |
| `sampler` | QComboBox | sampler | casi dinamici `advanced_sampler_*` | ✅ |
| `scheduler` | QComboBox | scheduler | casi dinamici `advanced_scheduler_*` | ✅ |
| `base_shift` / `max_shift` | QDoubleSpinBox | shift LTXVScheduler | `advanced_shift_values` | ✅ |
| `encoder` | QComboBox | text encoder video | casi dinamici `advanced_encoder_*` | ✅ |
| `prompt_enhance` | QCheckBox | enhance prompt | `advanced_prompt_enhance_toggle` | ✅ |
| `fixed_seed` | QCheckBox | seed fisso | `advanced_fixed_seed_on/off`, `video_fixed_seed` | ✅ |
| `seed` | QSpinBox | valore seed | `advanced_fixed_seed_reach_backend` | ✅ |
| `sub` | QTabWidget | sotto-tab Strumenti/Modelli | `advanced_subtabs_switch` | ✅ |
| bottone workflow | QPushButton→Signal | `open_workflow` | `advanced_open_workflow` | ✅ |
| bottone analyzer | QPushButton→Signal | `open_analyzer` | `advanced_open_analyzer`, `advanced_analyzer_reads_sampler` | ✅ |
| bottone navigator | QPushButton→Signal | `open_navigator` | `advanced_open_navigator` | ✅ |
| bottone calibrazione VRAM | QPushButton→Signal | `open_vram_cal` | — | ❌ GAP |
| `lv_auto` | QCheckBox | auto-estensione video lunghi | `advanced_long_video_config`, `advanced_long_video_autoextend_on/off` | ✅ |
| `lv_threshold` / `lv_seg` / `lv_overlap` / `lv_context` / `lv_ctx_strength` / `lv_ref_anchor` / `lv_anchor_every` | QDoubleSpinBox/QSpinBox ×7 | parametri catena autoregressiva | `advanced_long_video_config` | ✅ |
| `lv_resume` | QCheckBox | riprendi dopo crash | `advanced_long_video_config` | ✅ |

---

## 7. Tab ControlNet (`ui/tabs/controlnet_tab.py`, classe `ControlNetTab`)

File di test: `cases_controlnet_tab.py` (11 casi).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `enable` | QCheckBox | abilita ControlNet | `controlnet_enable_toggle`, `controlnet_generate_*`, `controlnet_disabled_normal_video` | ✅ |
| `type` | QComboBox | Canny/Depth/Pose (per indice) | `controlnet_type_combo_all`, `controlnet_generate_canny/depth/pose` | ✅ |
| `strength` | QDoubleSpinBox | forza ControlNet | (`controlnet_get_values_keys` verifica solo la presenza di `control_strength` nei parametri) | 🟡 valore mai variato dal widget |
| `cond_attention` | QDoubleSpinBox | attenzione condizionamento | — | ❌ GAP |
| `pick` | QPushButton | scegli video riferimento | `controlnet_pick_video_sets_path` | ✅ |

---

## 8. Tab Prolunga video (`ui/tabs/video_extend_tab.py`, classe `VideoExtendTab`)

File di test: `cases_video_extend_tab.py` (10 casi). **Copertura completa.**

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `prompt` | QPlainTextEdit | descrizione continuazione | `video_extend_native_happy`, `video_extend_no_prompt_no_call` | ✅ |
| `add_seconds` | QDoubleSpinBox | secondi da aggiungere | `video_extend_add_seconds_drives_num_frames`, `video_extend_long_sets_chunk_frames` | ✅ |
| `seg_seconds` | QDoubleSpinBox | durata segmento | `video_extend_segment_estimate_single_vs_multi` | ✅ |
| `overlap` | QSpinBox | frame di overlap | `video_extend_config_reflects_form_options` | ✅ |
| `context` | QSpinBox | frame di contesto | `video_extend_config_reflects_form_options` | ✅ |
| `steps` | QSpinBox | passi diffusione | `video_extend_config_reflects_form_options`, `video_extend_inherits_sampler_from_video_tab` | ✅ |
| `run_btn` | QPushButton | `generate_video_extend()` | `video_extend_native_happy`, `video_extend_routes_to_comfy_when_not_native`, guardie no-video/no-prompt | ✅ caricamento video via `set_video` (`video_extend_set_video_updates_label`) |

---

## 9. Tab Modifica Video AI (`ui/tabs/video_edit_ai_tab.py`, classe `VideoEditAITab`)

File di test: `cases_video_edit_ai_tab.py` (9 casi).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `btn_pick` | QPushButton | scegli video da editare | (`vedit_set_video_state` imposta lo stato via API) | 🟡 dialog mai aperta |
| `mode` | QComboBox | edit/inpaint/anime2real/real2anime/replace_char | `vedit_defaults_edit`, `vedit_inpaint_params`, `vedit_replace_char_*`, `vedit_object_enabled_only_inpaint` | ✅ |
| `obj` | QLineEdit | oggetto target | `vedit_inpaint_params`, `vedit_inpaint_no_object_blocks` | ✅ |
| `ref_img` | QLineEdit | immagine riferimento replace | `vedit_replace_char_sam2`, `vedit_replace_char_fullframe` | ✅ |
| `ref_btn` | QPushButton | sfoglia riferimento | — | ❌ GAP (dialog) |
| `ref_strength` | QDoubleSpinBox | forza identità | `vedit_replace_char_*` | ✅ |
| `prompt` | QPlainTextEdit | descrizione edit | tutti + `vedit_no_prompt_blocks` | ✅ |
| `vedit_enhance` | QCheckBox | arricchisci prompt AI | — | ❌ GAP |
| `strength` | QComboBox | forza editing (0.6/0.8/1.0) | `vedit_inpaint_params` | ✅ |
| `cfg` | QDoubleSpinBox | CFG | `vedit_inpaint_params` | ✅ |
| `guide` | QDoubleSpinBox | ancora all'originale | `vedit_inpaint_params` | ✅ |
| `dilate` | QSpinBox | dilatazione maschera | `vedit_inpaint_params` | ✅ |
| `blur` | QDoubleSpinBox | sfocatura maschera | — | ❌ GAP |
| `btn_go` | QPushButton | avvia edit (`_go`) | tutti i casi (slot diretto) + `vedit_no_video_blocks` | ✅ |

---

## 10. Qualità → Foto (`ui/tabs/restore_tab.py`, classe `RestoreTab`)

File di test: `cases_restore_tab.py` (14 casi + per-modello ESRGAN/FLUX.2 dinamici).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `path_edit` | QLineEdit (RO) | percorso immagine | (`restore_missing_image`, `use_canvas_input` via API) | 🟡 |
| `radio_restore` | QRadioButton | operazione restauro | `restore_basic_native`, `op_enable_states` | ✅ |
| `radio_colorize` | QRadioButton | colorizza | `colorize_basic`, `colorize_no_model_sends_placeholder` | ✅ |
| `radio_both` | QRadioButton | restauro+colore | `restore_and_colorize` | ✅ |
| `radio_faceswap` | QRadioButton | face-swap | `faceswap_ok`, `faceswap_missing_source/target` | ✅ |
| `upscale_model` | QComboBox | modello ESRGAN | casi per-modello dinamici (`set_upscale_models`) | ✅ |
| `scale_spin` | QDoubleSpinBox | fattore upscale | `restore_scale_4x`, `restore_scale_1x` | ✅ |
| `denoise_chk` | QCheckBox | denoise pre-upscale | `restore_denoise_on` | ✅ |
| `denoise_str` | QDoubleSpinBox | forza denoise | `restore_denoise_on` | ✅ |
| `flux2_model` | QComboBox | modello FLUX.2 colore | casi colorize per-modello | ✅ |
| `subject_edit` | QLineEdit | soggetto foto | casi colorize | ✅ |
| `style_edit` | QLineEdit | stile colore | casi colorize | ✅ |
| `color_steps` | QSpinBox | passi colorizzazione | casi colorize | ✅ |
| `color_denoise` | QDoubleSpinBox | denoise colorizzazione | — | ❌ GAP |
| `color_guidance` | QDoubleSpinBox | guidance colorizzazione | — | ❌ GAP |
| `fs_source` | QLineEdit (RO) | volto sorgente swap | `faceswap_ok`, `faceswap_missing_source` | ✅ |

---

## 11. Qualità → Video (`ui/tabs/video_def_tab.py`, classe `VideoDefTab`)

File di test: `cases_video_def_tab.py` (13 casi + loop per-modello ESRGAN/SeedVR2).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `engine` | QComboBox | SeedVR2/ESRGAN/FlashVSR | `videodef_enhance_engine_*`, `videodef_enhance_engine_enable_toggle` | ✅ |
| `seedvr2_model` | QComboBox | variante SeedVR2 | `videodef_enhance_engine_seedvr2`, casi per-modello | ✅ |
| `resolution` | QSpinBox | risoluzione target | `videodef_enhance_config_resolution_batch` | ✅ |
| `blocks` | QSpinBox | block swap memoria | `videodef_enhance_config_resolution_batch` | ✅ |
| `batch` | QSpinBox | batch size | `videodef_enhance_config_resolution_batch` | ✅ |
| `overlap` | QSpinBox | overlap temporale | — | ❌ GAP |
| `esrgan_model` | QComboBox | modello ESRGAN | `videodef_enhance_engine_esrgan`, `videodef_enhance_esrgan_all_models_loop` | ✅ |
| `scale` | QDoubleSpinBox | fattore 2×/4× | `videodef_enhance_engine_esrgan` | ✅ |
| `rife` | QCheckBox | interpolazione RIFE | `videodef_enhance_config_rife_toggle`, `videodef_rife_facerestore_chain_live` | ✅ |
| `rife_mult` | QComboBox | moltiplicatore FPS | `videodef_enhance_config_rife_toggle` | ✅ |
| `rife_ckpt` | QComboBox | checkpoint RIFE (disabilitato) | `videodef_enhance_config_rife_toggle` | ✅ |
| `exr_linear` | QCheckBox | de-gamma per compositing | — | ❌ GAP |
| `exr_btn` | QPushButton | esporta frame OpenEXR (`request_exr`) | — | ❌ GAP (`generate_exr_export` mai chiamato) |
| `chain_face` | QCheckBox | auto-chain restauro volti | `videodef_rife_facerestore_chain_live` | ✅ |
| `run_btn` | QPushButton | `generate_video_enhance()` | tutti i casi enhance + `videodef_enhance_no_video_guard` | ✅ |
| `fr_model` | QComboBox | CodeFormer/GFPGAN | `videodef_face_restore_basic`, casi per-modello | ✅ |
| `fr_detection` | QComboBox | detector volti | casi per-detector | ✅ |
| `fr_fidelity` | QSlider | fedeltà restauro | `videodef_face_restore_fidelity` | ✅ |
| `fr_run_btn` | QPushButton | `generate_face_restore()` | casi face_restore + `videodef_face_restore_no_video_guard` | ✅ |
| `fr_install_btn` | QPushButton | installa nodo facerestore | — | ❌ GAP |

---

## 12. Tab TRELLIS 3D (`ui/tabs/trellis_tab.py`, classe `TrellisTab`)

File di test: `cases_trellis_tab.py` (15 casi + combinazioni per resolution/format/precision/attn/bg/tex_size).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `path_edit` / `browse_btn` | QLineEdit/QPushButton | immagine input | (path via API; `trellis_no_image_no_dispatch`) | 🟡 dialog mai aperta |
| `canvas_btn` | QPushButton | usa immagine canvas | — | ❌ GAP |
| `masked_btn` | QPushButton | usa canvas con maschera | `trellis_canvas_masked_without_mask_no_dispatch`, `trellis_canvas_masked_forces_no_background` | ✅ |
| `resolution` | QComboBox | 512/1024/1024_cascade/1536_cascade | casi per-risoluzione dinamici | ✅ |
| `file_format` | QComboBox | glb/obj/ply/stl | casi per-formato | ✅ |
| `remove_bg` | QCheckBox | rimuovi sfondo | `trellis_remove_bg_on/off` | ✅ |
| `detail` | QSpinBox | token dettaglio mesh | `trellis_numeric_params_flow`, `trellis_9params_live` | ✅ |
| `guidance` | QDoubleSpinBox | guidance geometria | `trellis_numeric_params_flow` | ✅ |
| `steps` | QSpinBox | passi geometria | `trellis_numeric_params_flow` | ✅ |
| `faces` | QSpinBox | facce target (decimation) | `trellis_numeric_params_flow` | ✅ |
| `texture` | QCheckBox | texturing PBR | `trellis_texture_on/off`, `trellis_texture_toggle_ui_no_crash` | ✅ |
| `tex_size` | QComboBox | risoluzione texture | casi per-size | ✅ |
| `tex_guidance` / `tex_steps` | QDoubleSpinBox/QSpinBox | parametri texture | `trellis_numeric_params_flow` | ✅ |
| `seed` / `seed_random` | QSpinBox/QCheckBox | seed | `trellis_seed_fixed`, `trellis_seed_random` | ✅ |
| `precision` | QComboBox | auto/bf16/fp16/fp32 | casi per-precisione | ✅ |
| `attn` | QComboBox | backend attention | casi per-backend | ✅ |
| `bg_color` | QComboBox | colore sfondo rimozione | casi per-colore | ✅ |
| `gen_btn` | QPushButton | `_on_generate` → `request_trellis` | `trellis_generate_native_default`, `trellis_generate_comfy_default` | ✅ |
| `viewer` | visore 3D | GLB result + azioni cartella/salva/separa | (`trellis_show_result_no_crash`) | 🟡 interazioni visore non testate |

---

## 13. Tab Rig 3D (`ui/tabs/rig_tab.py`, classe `RigTab`)

File di test: `cases_rig_tab.py` (19 casi).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `path_edit` | QLineEdit (RO) | mesh input | (via API; `rig_no_mesh_no_request`) | 🟡 |
| `browse_btn` | QPushButton | sfoglia mesh | — | ❌ GAP (dialog) |
| `last_btn` | QPushButton | usa ultima mesh TRELLIS | `rig_use_last_mesh` | ✅ |
| `skeleton` | QComboBox | template scheletro | `rig_skeleton_keep_model_names/mixamo/unreal_engine_5` | ✅ |
| `file_format` | QComboBox | glb/fbx | `rig_format_glb`, `rig_format_fbx` | ✅ |
| `bottom_origin` | QCheckBox | origine in basso | `rig_bottom_origin_on` | ✅ |
| `transfer` | QCheckBox | trasferimento pesi | `rig_transfer_on/off` | ✅ |
| `postprocess` | QCheckBox | post-process skinning | `rig_postprocess_on` | ✅ |
| `gpv` | QSpinBox | gruppi per vertice | `rig_group_per_vertex` | ✅ |
| `smooth` | QDoubleSpinBox | angolo smoothing | `rig_smooth_angle` | ✅ |
| `num_beams`/`temperature`/`top_k`/`top_p`/`rep_pen` | QSpinBox/QDoubleSpinBox ×5 | sampling avanzato | `rig_advanced_sampling` | ✅ |
| `gen_btn` | QPushButton | `_on_generate` → `request_rig` | `rig_generate_default`, `rig_signal_wiring`, `rig_native_branch`, guardie offline/node-missing | ✅ |
| `viewer` | visore 3D | modello riggato | — | ❌ GAP |

---

## 14. Tab Animazione/Mocap (`ui/tabs/anim_tab.py`, classe `AnimTab`)

File di test: `cases_anim_tab.py` (21 casi). **Copertura completa.**

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `rig_edit` | QLineEdit (RO) | modello riggato | `anim_retarget_missing_rig`, `anim_use_last_rig` | ✅ |
| `browse_btn` (rig) | QPushButton | sfoglia rig | `anim_browse_rig_cancel` | ✅ |
| `last_btn` | QPushButton | usa ultimo rig | `anim_use_last_rig` | ✅ |
| `camera` | QSpinBox | indice webcam | `anim_mocap_start_emits` | ✅ |
| `start_btn` | QPushButton | `request_mocap_start` | `anim_mocap_start_emits`, `anim_mocap_start_env_missing`, `anim_capturing_toggle` | ✅ |
| `stop_btn` | QPushButton | `request_mocap_stop` | `anim_mocap_stop` | ✅ |
| `anim_edit` | QLineEdit (RO) | file animazione | `anim_set_anim_source`, `anim_retarget_missing_anim` | ✅ |
| `import_btn` | QPushButton | sfoglia BVH/FBX | `anim_browse_anim_cancel` | ✅ |
| tab sorgente (webcam/import) | QTabWidget | switch sorgente | `anim_src_tabs_switch` | ✅ |
| `template` | QComboBox | auto/Mixamo/UE5 | `anim_retarget_template_auto/mixamo/unreal`, `anim_template_combo_values` | ✅ |
| `fmt` | QComboBox | fbx/glb/bvh | `anim_retarget_format_fbx/glb/bvh`, `anim_format_combo_values` | ✅ |
| `apply_btn` | QPushButton | `request_retarget` | casi retarget + guardie missing | ✅ |

---

## 15. Tab Converti 3D (`ui/tabs/convert_tab.py`, classe `ConvertTab`)

File di test: `cases_convert_tab.py` (8 casi). **Copertura completa.**

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `path_edit` | QLineEdit (RO) | file 3D input | `convert_set_input_path_api`, `convert_no_input_guard` | ✅ |
| `fmt` | QComboBox | formato output (13 formati) | `convert_set_format_api`, casi per-formato | ✅ |
| `scale` | QDoubleSpinBox | fattore scala | `convert_scale_option` | ✅ |
| `embed` | QCheckBox | texture embedded | `convert_embed_textures_off` | ✅ |
| `animations` | QCheckBox | includi animazioni | `convert_animations_off` | ✅ |
| `run_btn` | QPushButton | `request_convert` | `convert_run_default`, `convert_blender_unavailable_offers_install` | ✅ |

---

## 16. Tab LoRA Train Img (`ui/tabs/train_tab.py`, classe `TrainTab`)

File di test: `cases_train_tab.py` (21 casi + per-base-model).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `ds_path` | QLineEdit | cartella dataset | `train_update_ds_info`, `train_preflight_no_dataset` | ✅ |
| browse dataset | QPushButton | `_browse_dataset` | `train_browse_cancel` | ✅ |
| `trigger` | QLineEdit | trigger word | `train_trigger_word` | ✅ |
| `cap_engine` | QComboBox | motore captioning | — | ❌ GAP |
| `cap_mode` | QComboBox | modalità caption | — | ❌ GAP |
| `btn_caption` | QPushButton | `_caption_dataset` | `train_caption_command`, `train_caption_blocked_*` | ✅ |
| `cap_overwrite` | QCheckBox | sovrascrivi caption | `train_caption_command` | ✅ |
| `btn_review` | QPushButton | `_review_captions` | — | ❌ GAP |
| `model` | QComboBox | modello base (flux1-dev/klein-4b/klein-9b/flux2-dev) | `train_model_switch_updates_path`, casi per-modello | ✅ |
| `model_path` | QLineEdit | repo/path modello | `train_model_switch_updates_path` | ✅ |
| `hf_token` | QLineEdit (pwd) | token HF | — | ❌ GAP |
| `rank` / `alpha` / `steps` / `batch` | QSpinBox ×4 | iperparametri | `train_numeric_params` | ✅ |
| `save_every` / `sample_every` | QSpinBox ×2 | checkpoint/campioni | — | ❌ GAP |
| `lr` | QComboBox (edit) | learning rate | `train_numeric_params`, `train_bad_lr_fallback` | ✅ |
| `seed` | QSpinBox | seed | `train_numeric_params` | ✅ |
| checkbox risoluzioni (`res[512\|768\|1024]`) | QCheckBox dict | risoluzioni training | `train_single_resolution`, `train_resolution_fallback` | ✅ |
| `quantize` | QCheckBox | quantizzazione | `train_quantize_off` | ✅ |
| `low_vram` | QCheckBox | modalità low VRAM | `train_low_vram_on` | ✅ |
| `train_te` | QCheckBox | allena text encoder | `train_text_encoder_on` | ✅ |
| `prompts` | QPlainTextEdit | prompt di sample | `train_custom_sample_prompts` | ✅ |
| `btn_install` | QPushButton | `_install_module` | `train_install_command`, `train_preflight_not_installed` | ✅ |
| `btn_start` | QPushButton | `_start` training | `train_preflight_*`, casi per-modello | ✅ |
| `btn_stop` | QPushButton | `_stop` | `train_stop_without_run` | ✅ |

---

## 17. Tab LoRA Wizard Vid (`ui/tabs/lora_video_tab.py`, classe `LoraVideoTab`)

File di test: `cases_lora_video_tab.py` (20 casi). Copertura quasi completa.

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `subj_list` | QListWidget | soggetti dataset | `lora_video_add_remove_subject`, `lora_video_multi_subject_train` | ✅ |
| bottoni aggiungi/elimina soggetto | QPushButton ×2 | `_add_subject`/`_del_subject` | `lora_video_add_remove_subject` | ✅ |
| `ed_type` | QComboBox | tipo soggetto (person/place/object/concept) | `lora_video_type_change_rebuilds_coverage`, `lora_video_full_train_*` | ✅ |
| `ed_name` | QLineEdit | nome soggetto | casi validation/full_train | ✅ |
| `ed_face` | QCheckBox | focus volto | `lora_video_face_focus_off` | ✅ |
| `photo_list` / `photo_angle` | QListWidget/QComboBox | foto per angolo (5 angoli) | `lora_video_photo_angles_all` | ✅ |
| bottoni foto add/del | QPushButton ×2 | `_add_photo`/`_del_photo` | `lora_video_photo_angles_all` | ✅ |
| `real_list` + add/del | QListWidget+QPushButton ×2 | media reali | `lora_video_real_media_only`, `lora_video_validation_no_media` | ✅ |
| checkbox copertura (`_cov_checks`) | QCheckBox dict | situazioni coperte | `lora_video_coverage_checkbox_toggle` | ✅ |
| `ed_customs` | QPlainTextEdit | situazioni custom | `lora_video_customs_situations` | ✅ |
| `lora_name` | QLineEdit | nome LoRA | `lora_video_lora_name_passed` | ✅ |
| `lora_rank` / `lora_steps` / `lora_clip` | QSpinBox/QDoubleSpinBox | iperparametri | `lora_video_opts_rank_steps_clip` | ✅ |
| `lora_pulid` | QCheckBox | PuLID | `lora_video_toggle_pulid_off` | ✅ |
| `lora_lowvram` | QCheckBox | low VRAM | `lora_video_toggle_lowvram_off` | ✅ |
| `cap_picker` | CaptionPicker | selezione captioner | — | ❌ GAP |
| `btn_install` | QPushButton | installa trainer | `lora_video_install_trainer` | ✅ |
| `btn_dataset` | QPushButton | solo dataset | `lora_video_dataset_only_person` | ✅ |
| `btn_full` | QPushButton | dataset+train | `lora_video_full_train_person/place/object/concept` | ✅ |

---

## 18. Tab LoRA Wizard Img (`ui/tabs/lora_image_tab.py`, classe `LoraImageTab`)

Sottoclasse del wizard video, override lato training FLUX.2. File di test: `cases_lora_image_tab.py` (8 casi).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `lora_name` | QLineEdit | nome LoRA | — | ❌ GAP (mai valorizzato nei casi) |
| `base_model` | QComboBox | base FLUX (flux2-dev/klein-4b/klein-9b/flux1-dev) | `lora_image_train_full_default`, casi per-base | ✅ |
| `lora_rank` / `lora_steps` | QSpinBox ×2 | iperparametri | `lora_image_rank_steps_propagation` | ✅ |
| `img_res` | QComboBox | risoluzione dataset (768/1024/1280) | casi per-risoluzione | ✅ |
| `cap_engine` | QComboBox | motore caption | — | ❌ GAP |
| `cap_mode` | QComboBox | modalità caption | — | ❌ GAP |
| editor soggetti (ereditato dal wizard video) | vari | soggetti/foto/focus volto | `lora_image_face_focus_off`, `lora_image_custom_situations`, `lora_image_guard_no_subject/no_media` | ✅ |
| `btn_install` | QPushButton | installa | `lora_image_install_button` | ✅ |
| `btn_dataset` | QPushButton | solo dataset | `lora_image_dataset_only` | ✅ |
| `btn_full` | QPushButton | dataset+train | `lora_image_train_full_default` | ✅ |

---

## 19. Tab Train LoRA Video (`ui/tabs/train_lora_video_tab.py`, classe `TrainLoraVideoTab`)

**Nessun file di test dedicato: intero tab scoperto.** (Da non confondere con `cases_lora_video_tab.py`, che copre il *wizard*.)

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `clips` | ClipDatasetWidget | dataset clip drag-drop | — | ❌ GAP |
| `global_cap` | QLineEdit | caption globale | — | ❌ GAP |
| `btn_review` | QPushButton | `_review_captions` | — | ❌ GAP |
| `name` / `trigger` | QLineEdit ×2 | nome LoRA / trigger | — | ❌ GAP |
| `steps` / `rank` / `secs` | QSpinBox ×3 | iperparametri | — | ❌ GAP |
| `res` | QComboBox | risoluzione | — | ❌ GAP |
| `low_vram` | QCheckBox | low VRAM | — | ❌ GAP |
| `cap_picker` | CaptionPicker | captioner | — | ❌ GAP |
| `btn` | QPushButton | `_start` training | — | ❌ GAP |

---

## 20. Tab Musica (`ui/tabs/music_tab.py`, classe `MusicTab`)

File di test: `cases_music_tab.py` (22 casi + 15 casi per-lingua liriche AI).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `song_name` | QLineEdit | nome canzone | casi `music_song_*` | ✅ |
| `song_tags` | QLineEdit | tag di stile | casi `music_song_*` | ✅ |
| `btn_tagpick` | QToolButton | palette tag (`_pick_tags`) | — | ❌ GAP |
| `btn_example` | QToolButton | esempio casuale | — | ❌ GAP |
| `song_enhance` | QCheckBox | arricchimento stile AI | — | ❌ GAP |
| `song_lyrics` | QPlainTextEdit | testo canzone | `music_song_heartmula_3B/7B`, `music_song_acestep_with_lyrics` | ✅ |
| `btn_lyr_ai` | QPushButton | genera liriche con LLM | `music_lyrics_ai_empty_prompt`, 15 casi per-lingua | ✅ |
| `lyr_prompt` | QLineEdit | tema liriche AI | casi liriche | ✅ |
| `lyr_lang` | QComboBox | lingua liriche | 15 casi dinamici per-lingua | ✅ |
| chip sezioni ×8 | QToolButton | inserisci [Verse]/[Chorus] (`_insert_section`) | — | ❌ GAP |
| `song_instr` | QCheckBox | strumentale | `music_song_heartmula_instrumental`, `music_song_acestep_instrumental` | ✅ |
| `song_dur` | QSpinBox | durata (5–360 s) | `music_song_heartmula_custom_params` | ✅ |
| `song_ver` | QComboBox | versione HeartMuLa 3B/7B | `music_song_heartmula_3B/7B` | ✅ |
| `song_engine` | QComboBox | HeartMuLa/ACE-Step/ACE-Step 1.5 | casi heartmula + acestep | ✅ |
| `song_temp` / `song_cfg` | QDoubleSpinBox ×2 | temperatura/CFG | `music_song_heartmula_custom_params` | ✅ |
| `song_lazy` | QCheckBox | low VRAM | `music_song_heartmula_lowvram` | ✅ |
| `song_adherence` | QDoubleSpinBox | aderenza stile ACE-Step | — | ❌ GAP |
| `song_gs_lyric` | QDoubleSpinBox | aderenza liriche ACE-Step | — | ❌ GAP (fix doppia-guida non regressionato) |
| `song_think` | QCheckBox | planner 5Hz (ACE-Step 1.5) | — | ❌ GAP |
| `preset_combo` + `btn_preset_save`/`btn_preset_del`/`btn_preset_song` | QComboBox+QToolButton ×3 | preset musica | — | ❌ GAP (4 controlli) |
| `btn_reset` | QPushButton | reset default modello | — | ❌ GAP |
| `song_meta` | MusicMetadata | BPM/key/time-signature/lingua | — | ❌ GAP |
| `song_adv` | MusicAdvanced | parametri avanzati ACE-Step | — | ❌ GAP |
| `song_task` | MusicTaskPanel | retake/repaint/edit/extend | — | ❌ GAP |
| `mv_image` | QLineEdit | immagine base videomusicale | — | ❌ GAP |
| `btn_mv_browse` | QPushButton | sfoglia immagine | — | ❌ GAP |
| `mv_style` | QLineEdit | stile videomusicale | `music_video_create` | ✅ |
| `mv_pace` | QComboBox | ritmo cambi scena | `music_video_create` | ✅ |
| `btn_song_install` | QPushButton | installa HeartMuLa | `music_install_heartmula` | ✅ |
| `btn_acestep_install` | QPushButton | installa ACE-Step | — | ❌ GAP |
| `btn_song` | QPushButton | genera musica (`_gen_song`) | tutti i `music_song_*` + `music_song_busy_guard` | ✅ |
| `btn_music_video` | QPushButton | genera videomusicale | `music_video_create`, `music_video_no_lyrics` | ✅ |
| `btn_mv_projects` | QPushButton | progetti videomusicali | `music_mv_projects_open` | ✅ |
| `son_video` | QLineEdit | video da sonorizzare | `music_sonify_with_video` | ✅ |
| `btn_last_video` | QPushButton | usa ultimo video | `music_use_last_video` | ✅ |
| `son_prompt` / `son_neg` | QLineEdit ×2 | prompt/negativo audio | `music_sonify_text_to_audio`, `music_sonify_negative_prompt` | ✅ |
| `son_dur` / `son_steps` / `son_cfg` / `son_seed` | QDoubleSpinBox/QSpinBox ×4 | parametri MMAudio | `music_sonify_custom_params` | ✅ |
| `btn_son_install` | QPushButton | installa MMAudio | `music_install_mmaudio` | ✅ |
| `btn_son` | QPushButton | genera sonorizzazione | casi `music_sonify_*` | ✅ |
| `btn_play` | QPushButton | riproduci ultimo audio | `music_play_no_audio` | ✅ |
| `btn_folder` | QPushButton | apri cartella output | `music_open_folder` | ✅ |
| `btn_spectro` | QPushButton | spettrogramma A/B | — | ❌ GAP |
| `btn_library` | QPushButton | libreria generazioni | — | ❌ GAP |
| `btn_stop` | QPushButton | ferma generazione | `music_stop_no_jobs` | ✅ |
| `wave_player` | WaveformPlayer | waveform + trim | — | ❌ GAP |
| `btn_lrc` | QPushButton | genera LRC karaoke | — | ❌ GAP |
| `song_trainer` | MusicTrainerPanel | training LoRA ACE-Step | — | ❌ GAP |
| `song_post` | MusicPostPanel | post-processing audio | — | ❌ GAP |
| `pitch_panel` | PitchPanel | correzione pitch (DiffPitcher) | — | ❌ GAP |

---

## 21. Tab Mondo 3D (`ui/tabs/world_tab.py`, classe `WorldTab`)

File di test: `cases_world_tab.py` (22 casi). **Copertura completa.**

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `name` | QLineEdit | nome scena | `world_gen_custom_name` | ✅ |
| `image` | QLineEdit | immagine iniziale | `world_gen_image_input`, `world_preflight_missing_image_blocks` | ✅ |
| browse immagine | QPushButton | `_pick_image` | `world_pick_image_no_crash` | ✅ |
| `prompt` | QLineEdit | descrizione scena | `world_gen_text_only`, `world_preflight_empty_inputs_blocks` | ✅ |
| `mode` | QComboBox | mesh/splat/both | `world_gen_mode_mesh/splat/both`, `world_gen_both_chains_splat` | ✅ |
| `res` | QSpinBox | risoluzione | `world_gen_resolution_passed` | ✅ |
| `low_vram` | QCheckBox | low VRAM | `world_gen_lowvram_flag` | ✅ |
| `hf` | QLineEdit | token HF | `world_gen_hf_token_passed` | ✅ |
| `btn_install` | QPushButton | installa WorldGen | `world_install_action_no_spawn`, `world_not_installed_blocks_gen`, `world_refresh_install_state` | ✅ |
| `btn_gen` | QPushButton | `_gen` genera scena | tutti i `world_gen_*` | ✅ |
| `btn_stop` | QPushButton | ferma | `world_stop_no_job`, `world_stop_with_running_job` | ✅ |
| `btn_fbx` / `btn_obj` | QPushButton ×2 | export via Blender | `world_export_fbx_no_scene_guard`, `world_export_obj_no_glb_guard` | ✅ |
| `btn_web` | QPushButton | player web three.js | `world_web_player_no_scene_guard` | ✅ |
| `btn_folder` | QPushButton | apri cartella | `world_open_folder_no_crash` | ✅ |

---

## 22. Tab Pesi modelli (`ui/tabs/weights_tab.py`, classe `WeightsTab`)

File di test: `cases_weights_tab.py` (12 casi + verifiche comando download).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `base` | QLineEdit | cartella base pesi | `weights_download_command_args` | ✅ |
| browse base | QPushButton | `_pick_base` | `weights_browse_base_no_crash` | ✅ |
| `search` | QLineEdit | cartella di scansione | `weights_scan_*` | ✅ |
| browse search | QPushButton | `_pick_search` | `weights_browse_search_no_crash` | ✅ |
| scan | QPushButton | `_scan` cerca pesi esistenti | `weights_scan_empty_dir`, `weights_scan_finds_existing` | ✅ |
| `hf` | QLineEdit (pwd) | token HF | `weights_hf_token_in_env`, `weights_no_token_no_hf_env` | ✅ |
| checkbox pesi (`_checks[id]`) | QCheckBox dict | selezione pesi da scaricare | `weights_open_no_crash`, `weights_refresh_clears_selection`, `weights_download_command_args` | ✅ |
| `chk_all` | QCheckBox | mostra tutti (anche fuori edizione) | — | ❌ GAP |
| `btn_refresh` | QPushButton | `_refresh` | `weights_refresh_clears_selection` | ✅ |
| `btn_missing` | QPushButton | `_select_missing` | `weights_select_missing_then_download` | ✅ |
| `btn_link` | QPushButton | `_link_found` collega trovati | — | ❌ GAP (rilevante per installer selettivo #62) |
| `btn_import` | QPushButton | `_import_unknown` | — | ❌ GAP |
| `btn_dl` | QPushButton | `_download` (QProcess `download_weights.py`) | `weights_download_no_selection`, `weights_double_download_guarded`, `weights_download_command_args` | ✅ |

---

## 23. Selettore LoRA (`ui/tabs/lora_tab.py`, classe `LoraTab` — widget embedded in Video/Immagini)

File di test: `cases_lora_tab.py` (17 casi + per-LoRA dinamici). I casi testano l'istanza del tab **Video**
(`w.lora_tab` alias di `video_tab.lora`); l'istanza della tab Immagini (`gen_lora_tab`) non è coperta (v. §2).

| Controllo | Tipo | Azione | Casi che lo coprono | Gap/Note |
|---|---|---|---|---|
| `search` | QLineEdit | filtro testo | `lora_search_filter_narrows`, `lora_selection_survives_filter` | ✅ |
| `hide_nsfw` | QCheckBox | nascondi LoRA NSFW | — | ❌ GAP (il filtro NSFW è testato via `video_tab.nsfw` in `lora_nsfw_filter`, non da questa checkbox) |
| `auto_triggers` | QCheckBox | inserimento trigger automatico | `lora_auto_triggers_toggle`, `lora_generate_auto_insert_trigger` | ✅ |
| `reload_btn` | QPushButton | `_reload` ricarica catalogo | `lora_reload` (slot diretto) | ✅ |
| `list` / `_rows` | QListWidget | selezione LoRA | `lora_list_populated`, `lora_select_single/multiple`, `lora_deselect` | ✅ |
| stella per-riga | QToolButton | `_toggle_fav` preferito | `lora_fav_toggle` | ✅ |
| checkbox per-riga | QCheckBox | attiva LoRA | `lora_select_*`, `lora_generate_includes_lora` | ✅ |
| spin forza per-riga | QDoubleSpinBox | `set_lora_strength` | `lora_strength_passed` | ✅ |

Casi trasversali: `lora_gui_state_roundtrip`, `lora_trigger_warning_missing`, `lora_trigger_present_no_warning`, `lora_set_available_extra`.

---

## 24. Tab Modelli (`ui/tabs/import_models_tab.py` + `import_models_tab_pro.py`)

`ImportModelsTab` (V1) e `ImportModelsTabPro` (variante Pro, sottoclasse). Agganciata come sotto-modulo
del tab Avanzate (`w.models_tab`). **Nessun file di test: intera area scoperta** (il caso
`advanced_subtabs_switch` si limita a commutare il sotto-tab).

| Controllo | Tipo | Azione | Casi | Gap/Note |
|---|---|---|---|---|
| `_seg_group` | QButtonGroup | modalità Scopri/Importa/Libreria/Avanzate | — | ❌ GAP (in Pro con lucchetti gating) |
| `search_edit` / `search_type` / `search_base` / `search_nsfw` / bottone cerca | QLineEdit/QComboBox×2/QCheckBox/QPushButton | ricerca Civitai (`_do_search`) | — | ❌ GAP (5 controlli) |
| `url_edit` / `b_url` | QLineEdit/QPushButton | import da URL/ID Civitai (`_import_url`) | — | ❌ GAP (2) |
| `imp_drop` | _DropArea | drag-drop file | — | ❌ GAP |
| bottoni file/cartella coda | QPushButton ×2 | `_pick_files_to_queue`, `_pick_dir_to_queue` | — | ❌ GAP (2) |
| `queue_list` / `b_import_all` / pulisci coda | QListWidget/QPushButton ×2 | coda import (`_import_all`, `_clear_queue`) | — | ❌ GAP (3) |
| `lib_search` / `lib_type` / `lib_arch` / `lib_nsfw` / refresh | QLineEdit/QComboBox×2/QCheckBox/QPushButton | filtri libreria (`_load_library`) | — | ❌ GAP (5) |
| `base`+browse / `search`+browse / scan / `hf` | QLineEdit×3+QPushButton×3+QLineEdit | percorsi e scansione (sezione Avanzate) | — | ❌ GAP (7) |
| `_path_edits` (modelli/workflow/inbox/dict/tools) | QLineEdit ×5 | percorsi configurabili | — | ❌ GAP (1 riga, 5 campi) |
| `_checks` pesi | QCheckBox dict | selezione download | — | ❌ GAP |
| `chk_prompt_unlock` | QCheckBox | sblocco enhancer NSFW | — | ❌ GAP |
| `btn_compat_install` / `btn_compat_uninstall` | QPushButton ×2 | shim compat ComfyUI | — | ❌ GAP (2) |
| `source_edit` / `drop` / `file_list` / sfoglia file / sfoglia cartella / `b_to_inbox` / rimuovi / pulisci | vari ×8 | staging file in inbox | — | ❌ GAP (8) |
| `btn_analyze` / `btn_place` / `btn_scan` / `btn_enrich` / `btn_overrides` / `btn_build` / `btn_ingest` | QPushButton ×7 | pipeline catalogo (`CatalogTooling`) | — | ❌ GAP (7) |
| Pro: `_btn_thumbs` | QPushButton | lazy load thumbnail (`_load_thumbs`) | — | ❌ GAP |
| Pro: gating 🔒 (`lib_nsfw`/`search_nsfw`/sezioni con UpsellPanel) | vari | blocco feature-matrix | — | ❌ GAP (regola "UI su ENTRAMBE le varianti" non verificata dai test) |

---

## 25. PromptBuilder (`ui/widgets/prompt_builder.py`)

Widget "Componi prompt" strutturato (`core/prompt_struct`) + dialog non modale con preset. **Nessun caso lo esercita.**

| Controllo | Tipo | Azione | Casi | Gap/Note |
|---|---|---|---|---|
| `subject` / `action` / `setting` / `mood` / `details` / `negative` | QLineEdit ×6 | campi struttura prompt | — | ❌ GAP |
| `style` | QComboBox (editabile) | stile preset/custom | — | ❌ GAP |
| `camera` / `light` | QComboBox ×2 | inquadratura/luce | — | ❌ GAP |
| `btn` ("Componi prompt") | QPushButton | `compose()` → segnale `composed` | — | ❌ GAP |
| `preset` (dialog) | QComboBox | carica preset salvato (`_load_preset`) | — | ❌ GAP |
| `btn_save` (dialog) | QPushButton | salva preset (`prompt_presets.json`) | — | ❌ GAP |
| `btn_use` (dialog) | QPushButton | applica → segnale `applied` | — | ❌ GAP |

Da coprire anche il round-trip `get_json()`/`set_json()` (riproducibilità).

---

## 26. Film Studio (`ui/film_studio/film_studio_window.py`, classe `FilmStudioWindow`) — sezione sintetica

Finestra top-level con 6 tab: `script_tab` (ScriptTab), `sb_tab` (StoryboardTab), `mt_tab` (MontageTab),
`asset_tab` (AssetManagerTab), `voice_tab` (VoiceTab), `lora_tab` (LoraStudioTab) + variante `FilmStudioWindowPro`
e dialog `LuckyDirectorDialog`/`FsProjectsDialog`. **L'harness gui_regression non copre Film Studio**
(nessun caso apre la finestra); l'inventario dei sotto-tab è nel draft precedente (v. Appendice A §6) e
andrà dettagliato quando l'harness verrà esteso (setup: DB isolato + mock llm/gen/voice/music service).

Controlli della barra progetto (tutti ❌ GAP):

| Controllo | Tipo | Azione |
|---|---|---|
| `proj_combo` | QComboBox | seleziona progetto (`_on_pick_project`) |
| `new_name` | QLineEdit | nome nuovo progetto |
| `dur` | QDoubleSpinBox | durata film (5–7200 s) |
| `path_mode` | QComboBox | strategia coerenza (fast/lora/mixed) |
| `run_mode` | QComboBox | supervised/auto |
| bottone Crea | QPushButton | `_create_project()` |
| `b_mng` | QPushButton | dialog Progetti (`_manage_projects`) |
| `b_lucky` | QPushButton | 🍀 Lucky Director (gate `cap.lucky_director`) |
| `b_ren` / `b_dup` / `b_del` | QPushButton ×3 | rinomina/duplica/elimina progetto |
| `series` / `b_ns` | QComboBox/QPushButton | assegna/crea serie |
| `forbidden` / `required` | QLineEdit ×2 | guardrail negativi/positivi (`_save_guardrails`) |

---

## Priorità F2/F3 — i 10 gap più importanti da coprire con qtbot (click/type/select reali)

Ordinati per impatto (ampiezza area scoperta × criticità del flusso utente):

1. **Tab Modelli (ImportModelsTab V1 + Pro) — 42 controlli, copertura ZERO.** Creare `cases_import_models_tab.py`: switch segmenti Scopri/Importa/Libreria/Avanzate, ricerca Civitai (client mock), import da URL, coda drag-drop→`_import_all`, filtri libreria, pipeline catalogo. In Pro verificare i lucchetti gating (regola "UI su entrambe le varianti").
2. **MaskCanvas: interazioni mouse reali.** Con qtbot: `mousePress`/`mouseMove` per brush/erase/fill/lasso/sam_point, cambio strumento dai `_mode_btns`, `brush_sl`, undo/redo, gestione layer, trasformazioni (scala/rotazione con maniglie). Oggi le maschere sono iniettate via API: il cuore di inpaint/outpaint non viene mai disegnato davvero.
3. **Tab Train LoRA Video (TrainLoraVideoTab) — intero tab scoperto (12 controlli).** Nuovo `cases_train_lora_video_tab.py`: dataset clip, caption globale, avvio training (service patchato), guardie preflight.
4. **Finestra principale: `cancel_btn` (interrupt), `free_vram_btn`, menu (Impostazioni, Wizard, Film Studio, switch varianti Classica/Pro, cambio lingua), `_queue_btn`/`_mem_btn`, click reale su `gen_btn` e sui bottoni della barra tab custom.** Flussi trasversali oggi invocati solo come slot.
5. **Film Studio window: barra progetto** (crea/rinomina/duplica/elimina, guardrail, serie, Lucky Director gate) — harness dedicato (DB isolato) o estensione dell'attuale via `_open_film_studio()`.
6. **Musica: area ACE-Step e post-produzione (25 gap)** — `song_adherence`/`song_gs_lyric`/`song_think` (regressione del fix doppia-guida), preset musica (save/load/delete), `song_meta`/`song_adv`/`song_task`, pannelli trainer/post/pitch, `btn_acestep_install`, spettrogramma/libreria/LRC.
7. **PromptBuilder (13 controlli)** — compose→segnale `composed` con campi digitati via qtbot, round-trip JSON, preset della dialog (salva/carica/usa).
8. **Frame1: componenti modello e negativi** — `gen_engine_sel`, `gen_encoder`, `gen_clip_type`, `gen_vae`, `gen_precision`, `gen_neg`/`inp_neg`/`int_neg`, `inp_guidance`/`inp_feather`/`inp_lora`, preferiti `gen_star`/`inp_star`, `gen_lora_tab` (LoRA lato immagini).
9. **Video: PuLID keyframe (`pulid_kf_face`/`pulid_kf_browse`/`pulid_kf_gen`), `video_engine_sel`, `teacache`, `fr_auto_model`/`fr_auto_fidelity`, sorgente/range/opzioni del pannello V2V.** Flussi identità-volto e scelta motore mai regressionati.
10. **Code minori ad alto valore**: Pesi `btn_link`/`btn_import`/`chk_all` (critici per l'installer selettivo #62), Qualità-Video export EXR (`exr_linear`/`exr_btn`) e `fr_install_btn`, ControlNet `strength`/`cond_attention` variati davvero, Audio `pick_btn`/`rec_btn`/`b_dup`.

> Nota metodologica per F2: l'harness attuale (Spy) resta valido; la fase qtbot deve **sostituire le
> impostazioni programmatiche con interazioni reali** (`qtbot.mouseClick`, `keyClicks`, `QTest.mouseMove`)
> almeno per i percorsi sopra, mantenendo le stesse asserzioni sui parametri catturati dallo Spy.
> Le QFileDialog restano patchate (ritorno percorso finto), ma il click sul bottone che le apre va simulato.

---

# APPENDICE A — Pattern qtbot e punti di patch (dal draft 04/07, validati)

Stack: **pytest-qt 4.5.0 + PySide6 6.11.1** (già nel venv di test `E:\AI\Comfy_Nexus.bak\ComfyUI\venv`).

### A.1 Come si pilota un widget (vs oggi)
```python
qtbot.mouseClick(w.video_tab.prompt, Qt.LeftButton)           # focus reale
qtbot.keyClicks(w.video_tab.prompt, "un gatto astronauta")    # digita
w.video_tab.model.setCurrentIndex(i)                          # combo: per index/itemData
qtbot.mouseClick(w.frame1_tab.gen_btn, Qt.LeftButton)         # click → segnale reale
```
- **Pulsanti** → `qtbot.mouseClick(btn, Qt.LeftButton)` (un pulsante disabilitato/scollegato NON emette → scopre wiring rotto).
- **Testo** → `mouseClick(field)` poi `keyClicks(field, "…")`; per `QPlainTextEdit` grandi `setPlainText` + verifica.
- **Combo con itemData** → `setCurrentIndex`/`findData` (NON per testo): `model`, `precision`, `gen_arch`, `ed_type`, `base_model`, `mode`, `res_preset`…
- **Slider/Spin** → `setValue` (o frecce per simulare l'utente). **Checkbox/Radio** → `mouseClick` o `setChecked`.
- **Dict di widget** (per chiave): `train_tab.res[512|768|1024]`, `lora_tab._rows[name]{cb,sp,star}`, `lora_video_tab._cov_checks[id]`, `import_models_tab._path_edits[key]`, `weights_tab._checks[id]`.

### A.2 Punti di patch per evitare GPU/IO reali (per gruppo)
- **Immagine/Video**: `native_driver.is_up = lambda: True`; rami ComfyUI `w.driver.is_up = lambda: True`. Segment/Restore richiedono SEMPRE `driver.is_up` anche in nativo.
- **Training**: `ts.TrainingRun.start`, `ts.is_installed`, `ts.build_config`; `QProcess` in `ui.tabs.train_tab`; LoRA video/foto: `LV._LoraWorker.start`, `lit._LoraImageWorker`, `subprocess.Popen`.
- **3D/Mondo**: `trellis_service`/`rig_service`/`world_service` (`is_installed`, `make_job`), `blender_installer.find_blender`, `QProcess`, `os.startfile`, `webbrowser.open`, `http.server`.
- **Pesi/Import**: `QProcess.start`/`setProcessEnvironment` (weights); `CatalogTooling`/subprocess venv (import models).
- **Audio/Musica**: `core.tts_engine.synthesize`; `ms.make_song_job`/`make_sonify_job`/`preflight_*`, `acestep_service`, `mv_projects.create_project`; thread `_LyricsWorker`.
- **Film Studio**: DB isolato (`db.ROOT`/`db.DB_PATH`→tmp, `db._conn=None`); mock `core.llm_service`, `gen_service`, `voice_studio`, `music_service`, `scheduler`; thread `_LlmWorker`/`_SheetWorker`/`_DescWorker`/`_ConnectWorker` → mock o `qtbot.waitSignal`.

### A.3 Dialoghi bloccanti da neutralizzare (auto-accept)
`QFileDialog`, `QMessageBox.question/warning/information` (conferme "download pesante" FLUX.2-dev, elimina,
install, trigger-warning LoRA), `QInputDialog` (nome preset/serie), `QColorDialog` (recolor), `AssetDialog.exec()`
(@-mention Film Studio).

### A.4 Dettaglio sotto-tab Film Studio (per l'estensione F3)
- **ScriptTab**: `kind` (full/idea/theme/interactive), `styles`/`genres` editabili, `inp` @-mention, `summary`, `text`, `btn_gen`/`btn_regen`/`btn_save` → `llm_service.complete` (thread). Guard: input vuoto, LLM assente, save senza progetto. Versioning script.
- **LlmSettings**: `provider` (gemma/lmstudio), `port`, `model`, `btn_connect`.
- **StoryboardTab**: `btn_break`/`btn_keys`/`btn_video`/`btn_auto`/`btn_shotlist`, `tree`, `edit` @-mention, `shot_seed`, continuity `ct_time`/`ct_weather`/`ct_light`/`ct_mood`+`btn_ct`, `btn_save_sel`/`btn_regen_sel`/`btn_regen_kf`/`btn_regen_vid` → `L.complete_json`, `gen_service.resolve_refs`, `scheduler.enqueue`; Auto 1-click `_gen_all`→`auto_complete.emit`.
- **MontageTab**: `btn_refresh`, `timeline` drag&drop, `lst`, `aspect`, `fmt`, `btn_audio`/`btn_music`/`btn_ambient`/`btn_sfx`/`btn_master`/`btn_export`, `btn_animatic`, `dub_lang`+`dub_relip`+`btn_dub`.
- **AssetManagerTab**: `search`, `ftype`, `lst`, editor `e_type`/`e_name`/`e_desc`/`e_tags`/`e_photo`, `b_save`/`b_sheet`, variazioni, wardrobe.
- **VoiceTab** (voce da descrizione/parametri/clone + binding attore) → `voice_studio`; **LoraStudioTab** (`trigger`/`flavor`/`scope`/`asset`/`b_save`); **TimelineWidget** (drag&drop→`reordered`); **FsProjectsDialog**.

Percorsi end-to-end da coprire: script→asset→storyboard→montaggio; Auto 1-click; doppiaggio; multi-LoRA >3; @-mention; timeline drag&drop; continuity; CRUD progetto+versioning.

---

# APPENDICE B — Stato bug emersi dalla mappatura (#95) — CHIUSI

La mappatura del draft 04/07 aveva fatto emergere 15+ bug reali (montage `os` non importato, Trellis nativo
che perdeva parametri, `load_model` inesistente, rig col callback di Trellis, colorize placeholder, seed
random scartato in tutti i path video nativi, ecc.). Tutti verificati con revisione adversariale e **corretti**
nei commit `7b8b131`, `6079551`, `2ddc406`, `99f1b18`, `d17b223`, `d0fd18f`, `cdf6620` (dettaglio nella storia
git e in `review/`). Residui intenzionali (deferral): FLUX.2 Klein multi-reference vero su ComfyUI (#14/#44);
limiti API TRELLIS.2 (bg/risoluzione/precisione non esposti dallo script ufficiale = confine di capacità).
**Nessun bug aperto; smoke GUI 14/14 verde.** I test F2/F3 dovranno includere casi-contratto che tengano
chiusi questi bug (seed random, mappatura indice ControlNet, `_r64`, parametri rig/trellis, colorize guard).

---

# APPENDICE C — LingBot-Video (2ª architettura video) — casi di test

LingBot-Video affianca LTX come architettura video selezionabile (tab Video → "Architettura"). Worker/venv
dedicati (`E:\AI\Tools\lingbot_venv`, `workers/lingbot_infer.py`), dispatch centralizzato in
`core/ltx_video_job.py` (`arch="lingbot"`). Vedi il piano in `.claude/plans/`.

| # | Caso | Passi da GUI | Atteso |
|---|------|--------------|--------|
| LB1 | Install venv | `python scripts/install_lingbot.py` | `lingbot_venv` importa `LingBotVideoPipeline` |
| LB2 | Download-on-select | tab Video → arch=LingBot → scegli un modello ⬇ mancante (click) | dialog + barra progresso → scaricato, badge ⬇ sparisce |
| LB3 | T2V Dense | arch=LingBot, Dense 1.3B, 832×480×5s, prompt creativo | mp4 valido, stego+watermark, VRAM in fascia |
| LB4 | I2V / TI2V | come LB3 con immagine di partenza | il primo frame è rispettato |
| LB5 | Refiner MoE | MoE 30B, refiner ON | confronto base vs refined |
| LB6 | Auto-disable | arch=LingBot | V2V / ControlNet / storyboard / LoRA / TeaCache disabilitati con tooltip; tornando a LTX si riabilitano |
| LB7 | Rewriter | checkbox rewriter ON | log "rewriter prompt → caption JSON" (proprietario se scaricato, altrimenti composer locale) |
| LB8 | Extend | durata > singola passata | catena a chunk, crossfade |
| LB9 | Propagazione | Music Video / Film Studio / Lucky Director con arch=LingBot | tutti usano il worker LingBot |
| LB10 | LoRA sperimentale | wizard LoRA, Formato=LingBot / o Clip sorgente=LingBot con target LTX | dataset da LingBot; training LingBot marcato sperimentale |
| LB11 | Regressione LTX | tutti i flussi con arch=LTX (default) | invariati |

**Note:** la firma esatta di `LingBotVideoPipeline.__call__` e la ricetta LoRA (`workers/lingbot_train.py`)
sono marcate `# VALIDATE` — vanno confermate al primo run reale. Il training LoRA LingBot solleva un errore
chiaro finché la ricetta flow-matching non è implementata (mai una LoRA fasulla consegnata come valida).
