# GUI Test Matrix — interazioni reali con pytest-qt / qtbot

Inventario completo della superficie di interazione GUI di LTX Studio, estratto dal codice
(`ui/tabs/*`, `ui/widgets/*`, `ui/film_studio/*`, handler in `ui/main_window.py`, casi in
`tests/gui_regression/cases_*.py`). Serve a scrivere test che **pilotano i widget veri** con
`qtbot.mouseClick` / `keyClicks` / `mouseMove` e verificano l'effetto sul backend (Spy).

Stack: **pytest-qt 4.5.0 + PySide6 6.11.1** (già installati nel venv di test
`E:\AI\Comfy_Nexus.bak\ComfyUI\venv`).

> ⚠️ Questo è l'inventario da **revisionare**: correggi/aggiungi ciò che manca. Solo dopo l'ok
> costruisco l'harness qtbot + i test. In coda: elenco **bug/gap reali** già emersi dalla mappatura.

---

## 0. Convenzioni, harness e pattern qtbot

### 0.1 Come si pilota un widget (vs oggi)
Oggi lo `Spy` chiama gli handler a mano (`w.on_generate()`) e setta i campi via codice. Il nuovo
modello **clicca/scrive davvero**:

```python
qtbot.mouseClick(w.video_tab.prompt, Qt.LeftButton)           # focus reale
qtbot.keyClicks(w.video_tab.prompt, "un gatto astronauta")    # digita
w.video_tab.model.setCurrentIndex(i)                          # combo: per index/itemData
qtbot.mouseClick(w.frame1_tab.gen_btn, Qt.LeftButton)         # click pulsante → segnale reale
```
- **Pulsanti** → `qtbot.mouseClick(btn, Qt.LeftButton)` (un pulsante disabilitato/scollegato NON emette → scopre wiring rotto).
- **Testo** → `mouseClick(field)` poi `keyClicks(field, "…")` (rispetta focus/validator). Per `QPlainTextEdit` grandi conviene `setPlainText` + verificare; per campi corti usare keyClicks.
- **Combo con itemData** → `setCurrentIndex`/`findData` (NON per testo): `model`, `precision`, `gen_arch`, `ed_type`, `base_model`, `mode`, `res_preset`… Molti combo hanno `itemData` ≠ label.
- **Slider/Spin** → `setValue` (o `keyClicks`/frecce se si vuole simulare l'utente).
- **Checkbox/Radio** → `mouseClick` (toggla) o `setChecked`.
- **Dict di widget** (indicizzati per chiave, non attributo): `train_tab.res[512|768|1024]`,
  `lora_tab._rows[name]["cb"|"sp"|"star"]`, `lora_video_tab._cov_checks[case_id]`,
  `import_models_tab._path_edits[key]`, `weights_tab._checks[id]`.

### 0.2 Intercettazione backend (Spy) — mantenuta
Lo `Spy` (`tests/gui_regression/harness.py`) intercetta `_enqueue`, `ComfyDriver.submit`,
`native_driver` e cattura `{via, params, module, label}` SENZA eseguire (no GPU). Il nuovo harness
**tiene lo Spy** ma sostituisce "chiama handler" con "clicca il widget".

### 0.3 Punti di patch per evitare GPU/IO reali (per gruppo)
- **Immagine/Video**: `native_driver.is_up = lambda: True`; per rami ComfyUI `w.driver.is_up = lambda: True`. Segment/Restore richiedono SEMPRE `driver.is_up` anche in nativo.
- **Training**: `ts.TrainingRun.start`, `ts.is_installed`, `ts.build_config`; `QProcess` in `ui.tabs.train_tab` (install/caption). LoRA video/foto: `LV._LoraWorker.start`, `lit._LoraImageWorker`, `subprocess.Popen`.
- **3D/Mondo**: `trellis_service`/`rig_service`/`world_service` (`is_installed`, `make_job`), `blender_installer.find_blender`, `driver`, `QProcess`, `os.startfile`, `webbrowser.open`, `http.server`.
- **Pesi/Import**: `QProcess.start`/`setProcessEnvironment` (weights); `CatalogTooling`/subprocess venv (import models).
- **Audio/Musica**: `core.tts_engine.synthesize`; `ms.make_song_job`/`make_sonify_job`/`preflight_*`, `acestep_service`, `mv_projects.create_project`; thread `_LyricsWorker`.
- **Film Studio**: DB isolato (`db.ROOT`/`db.DB_PATH`→tmp, `db._conn=None` prima di tutto); mock `core.llm_service`, `gen_service`, `voice_studio`, `music_service`, `scheduler`; thread `_LlmWorker`/`_SheetWorker`/`_DescWorker`/`_ConnectWorker` → mock o `qtbot.waitSignal`.

### 0.4 Dialoghi bloccanti da neutralizzare (auto-accept)
`QFileDialog` (pick file/foto), `QMessageBox.question/warning/information` (conferme "download
pesante" FLUX.2-dev, elimina, install, trigger-warning LoRA), `QInputDialog` (nome preset/serie),
`QColorDialog` (recolor). Il modal `AssetDialog.exec()` (@-mention Film Studio) va monkeypatchato.

---

## 1. VIDEO

### 1.1 `w.video_tab` (VideoTab) — azione: `w.on_generate()`
Controlli chiave: `prompt` (QPlainTextEdit), `neg`, `preset_combo`+`preset_save_btn`+`preset_del_btn`,
`nsfw` (QComboBox **SFW/NSFW**), `model` (QComboBox itemData `{kind:auto/diffusion/gguf/checkpoint, model, key}`),
`precision` (all/fp16/fp8/fp4/gguf), `model_star` (QToolButton), `res_preset` (11 preset + Personalizzato)
+ `width`/`height` (QSpinBox), `fps`, `length` (QDoubleSpinBox), `img_strength` (QSlider→ref_strength/100),
`i2v_method` (addguide/inplace), `video_engine_sel` (ltx2_infer/comfy), `fr_auto`+`fr_auto_model`+`fr_auto_fidelity`,
`fs_auto`+`fs_face`+`fs_face_btn`, `rife_auto`+`rife_factor` (x2/x4), `sampler_seed` (SamplerSeedGroup, §7),
`lora` (LoraTab embedded), `v2v_panel` (§4), `storyboard_panel` (§5).
- **Backend** (`_generate_video_native`): params `prompt,negative,width(_r64),height(_r64),num_frames,fps,steps,guidance,seed,loras,output`; I2V `image_path,ref_strength,i2v_method`; audio `audio_path`; gguf `gguf_transformer`; controlnet `control_video/type/strength`; storyboard `keyframes`. Auto-switch a `_VideoComfyWorker` se motore comfy su t2v/i2v.
- **Guard**: prompt vuoto → no enqueue; conflitti `state.evaluate` bloccanti.
- **Combinazioni**: t2v vs i2v (canvas); 1 caso per OGNI modello (kind diffusion/gguf/checkpoint) × SFW/NSFW; ogni preset risoluzione + Personalizzato; fps×durata→num_frames; img_strength (solo i2v); i2v_method; steps/cfg/seed fisso; precisione gguf; fr/fs/rife toggle; storyboard; V2V; preset save/delete; preferiti.

### 1.2 `w.video_def_tab` (VideoDefTab, "Migliora qualità") — segnali `request_enhance`/`request_face_restore`
`engine` (seedvr2/esrgan/flashvsr), `seedvr2_model`, `resolution`, `blocks`, `batch`, `esrgan_model`,
`scale`, `rife`+`rife_mult`(2×/3×/4×)+`rife_ckpt`(rife47/48/46), `run_btn`; face: `fr_model`, `fr_detection`,
`fr_fidelity` (slider/100), `fr_run_btn`, `fr_install_btn`.
- **Backend**: enhance→`_enhance_video_native` (video_upscale ESRGAN / SeedVR2 / FlashVSR); face→module `video_face`.
- **Combinazioni**: 1 caso per OGNI engine, OGNI modello ESRGAN, OGNI SeedVR2, OGNI face-restore × OGNI detector; `_on_engine` abilita i controlli giusti; RIFE mult(idx+2); fidelity 0..100→0..1.

### 1.3 `w.video_edit_ai_tab` (VideoEditAITab) — segnale `request_edit(dict)`
`btn_pick`, `mode` (edit/inpaint/anime2real/real2anime, LoRA diversa), `obj` (solo inpaint), `prompt`,
`strength` (0.6/0.8/1.0), `cfg`, `guide`, `dilate`, `btn_go`.
- **Backend**: `_gen_video_edit_ai` → `_VideoEditAIWorker`; config in `w._edit_ai_worker.cfg` (NON in params Spy).
- **Guard**: no video / inpaint-no-object / prompt vuoto → no emit. **Combinazioni**: 1 caso per modalità/LoRA; strength; inpaint cfg/guide/dilate; obj solo in inpaint.

### 1.4 `w.video_extend_tab` (VideoExtendTab) — segnale `request_extend`
`video_lbl`+pick, `prompt`, `add_seconds`, `seg_seconds`, `seg_estimate`, `overlap`, `context`, `steps`, `run_btn`.
- **Backend**: `_extend_video_native` (ultimo frame via ffmpeg → I2V; eredita w/h/fps da video_tab). **Guard**: no video / no prompt / no last-frame. **Combinazioni**: add_seconds→num_frames; >150 frame→chunk_frames; eredita steps/seed; stima segmenti single(≤20s)/multi; routing comfy (`via=="comfy"`).

### 1.5 `w.controlnet_tab` (ControlNetTab) — alimenta `on_generate`
`enable` (QCheckBox), `type` (**indice** 0=Canny/1=Depth/2=Pose), `strength` (0..1), `pick`+`lbl` (video ref in `_video`).
- **Backend**: `_generate_video_native` mappa indice→`{0:canny,1:depth,2:pose}`, `controlnet_preprocess.preprocess_video`, sostituisce `image_path` con `control_video`. **Guard**: `cnet_needs_video` blocking. **Combinazioni**: 1 caso per OGNI tipo (verifica mappatura indice→kind); enabled-senza-video; preprocess-failure→fallback senza guida.

### 1.6 `w.anim_tab` (AnimTab) — segnali `request_mocap_start/stop`, `request_retarget`
`rig_edit`+browse+`last_btn`, `src_tabs` (webcam/import), `camera`, `start_btn`/`stop_btn`, `anim_edit`+browse,
`template` (auto/Mixamo/Unreal Engine 5), `fmt` (fbx/glb/bvh), `apply_btn`.
- **Backend**: mocap_service (webcam→BVH) + Blender retarget (NON Spy: patch `mocap.start`/`_start_blender`).
- **Combinazioni**: mocap start/stop/env-assente; retarget per OGNI formato (verifica estensione) e OGNI template.

### 1.7 `w.video_tab.v2v_panel` (V2VPanel) — letto da `on_generate`
Radio `rb_normal`/`rb_talk`/`rb_sing`/`rb_retake`; `btn_pick`/`btn_lastgen`/`lbl_video`; ReTake: `sl_start`/`sl_end`+`rb_audio_keep`/`rb_audio_regen`; Talk: `cmb_region` (mouth/face)+`sp_offset`+`cb_reimagine`; Canta: `btn_sing_audio`+`cb_sing_vocals`+`sl_sing_scale`.
- **Backend**: sing/talk→`_maybe_multitalk`→`_start_native_multitalk`; retake→`_retake_native`. **Guard**: sing-senza-audio, retake-senza-video, retake-troppo-corto (≤6s), sezione <0.5s. **Combinazioni**: ReTake con/senza sorgente; Talk senza audio→stop; Canta foto+canzone; region/offset/reimagine; audio keep/regen; vocals; audio_scale.

### 1.8 `w.video_tab.storyboard_panel` (StoryboardPanel)
`enable` (QCheckBox), add-row, `_rows[i]` con `.pick`/`.image_path`/`.lbl`/`.second`/`.strength`.
- **Backend**: `get_storyboard()` → `params["keyframes"]` (`{path,frame=round(second*fps),strength}`, sostituisce image_path). **Guard**: abilitato con <2 scene → errore. **Combinazioni**: 2 scene→keyframes len 2; ordinamento temporale; strength.

---

## 2. IMMAGINE

### 2.1 `w.frame1_tab` (Frame1Tab) — 4 sotto-tab + segment box
> NB alias: `frame1_tab.prompt`/`.neg` ridiretti ai widget del tab Video (`main_window.py:560-561`).

**2.1a Genera** (segnale `request_generate`→`generate_image`): `img_nsfw` (SFW/NSFW), `gen_arch`
(FLUX/Z-Image/Ernie), `gen_engine_sel` (comfy/diffusers), `gen_model` (Auto `__auto__` + concreti),
`gen_precision`, `gen_star`, `gen_encoder`, `gen_clip_type`, `gen_vae`, `gen_lora_tab`, `gen_sampler`
(auto+euler/…), `gen_scheduler` (auto+karras/…), `gen_enhance`, `gen_steps`, `gen_cfg`, `gen_seed`+`gen_seed_rand`,
`gen_prompt`, `gen_neg`, `gen_btn`; PuLID: `gen_pulid_enable`+`gen_pulid_face`+`gen_pulid_face_btn`+`gen_pulid_strength`.
- **Backend nativo**: params `model,arch,prompt,negative,width,height,steps,guidance,seed,loras,pulid_face,pulid_strength`. **Guard**: prompt vuoto, model vuoto/placeholder, FLUX+encoder-FLUX.2-mancante. **Combinazioni**: 1 caso per OGNI modello × arch FLUX/Z-Image/Ernie; auto-select; enhance; seed fisso; cfg+steps; sampler+scheduler; PuLID.

**2.1b Inpaint/Outpaint/Integra/Componi** (segnali `canvas.request_inpaint/outpaint`, `request_integrate`, `request_compose`):
- Inpaint: `inpaint_prompt`/`outpaint_prompt`/`inp_neg`, `strength` (slider), `inp_model` (INPAINT_MODELS + dinamici checkpoint/unet), `inp_star`, `inp_lora`, `inp_steps`, `inp_seed`+`inp_seed_rand`, `b_in`/`b_out`.
- Integra: `integrate_prompt`, `integrate_engine` (Kontext/FLUX.2), `integrate_puzzle`, `integrate_model`, `int_neg`, `int_steps`, `int_guidance` (slider/10), `int_seed`, `b_int`.
- Componi: `compose_list`, `compose_method` (multiref/stitch), `compose_prompt`, `b_add`/`b_del`/`b_msk`/`b_comp`.
- **Backend**: inpaint FLUX.1→`_start_native_inpaint`; Klein/checkpoint/GGUF→ComfyUI; integra/recolor/compose Kontext→`_start_native_kontext`, FLUX.2→ComfyUI. **Guard**: no-image, busy, unet-non-installato, outpaint-senza-area, inpaint-senza-mask, compose-<2-foto.
- **Combinazioni**: 1 caso per OGNI modello INPAINT (FLUX.2 Klein = no-crash); UNET dinamico; strength/steps/seed; integra Kontext/FLUX.2/puzzle; componi multiref/stitch/FLUX.2.

**2.1c Colore** (`request_recolor`): `color_target`+`color_swatch`+pick, `color_subject`, `color_engine` (kontext/flux2), `color_steps`, `color_btn`. Backend: Kontext nativo (prompt "Recolor …") / FLUX.2 ComfyUI. Guard: target mancante.

**2.1d Upscale** (`request_upscale`): `up_model` (ESRGAN), `up_scale` (2x/3x/4x), `up_mode` (Veloce/Rifinisci), `up_refine_model`+`up_refine_str`+`up_refine_steps`, `up_btn`, `restore_btn`. Backend: nativo `upscale` / refine FLUX.2 ComfyUI. Combinazioni: 1 caso per modello upscale; scale; refine.

**2.1e Segment** (`w.frame1_tab.segment_box`, `request_segment_text`): `seg_text`, `seg_detector` (grounding_dino / grounding_dino_only), `seg_threshold`, `seg_sam` (modelli SAM), `seg_btn`. Backend: `_segment_native` (richiede SEMPRE driver up). Guard: text vuoto. Combinazioni: DINO+SAM vs DINO-only.

### 2.2 `w.restore_tab` (RestoreTab, sotto `w.quality_tab`) — segnale `request_restore`
Radio `radio_restore`/`radio_colorize`/`radio_both`/`radio_faceswap`; restauro `upscale_model`+`scale_spin`+`denoise_chk`+`denoise_str`; colore `flux2_model`+`subject_edit`+`style_edit`+`color_steps`; faceswap `fs_source`+`fs_btn`; `proc_btn`.
- **Backend**: faceswap→`_faceswap_photo` (InsightFace nativo); restore→`_start_native_upscale`; colorize/both→ComfyUI. **Combinazioni**: 1 caso per modello upscale/FLUX.2; denoise interlock; op-enable-states per radio; guard.

### 2.3 `w.convert_tab` (ConvertTab, 3D via Blender) — segnale `request_convert`
`path_edit`+browse, `fmt` (13 formati glb…dxf), `scale`, `embed`, `animations`, `run_btn`, `log`. Backend: `_start_blender("convert.py")` (NON Spy). Combinazioni: 1 caso per formato (verifica output `.{code}`); scale/embed/animations.

### 2.4 `w.advanced_tab` (AdvancedTab, hub) — non-immagine
Hub Workflow/Analyzer/Navigator (segnali `open_workflow/analyzer/navigator`) + Video-lunghi (`lv_auto`,`lv_threshold`,`lv_seg`,`lv_overlap`,`lv_context`,`lv_ctx_strength`,`lv_ref_anchor`,`lv_anchor_every`,`lv_resume` → `long_video_config()`) + `attach_models` (Import Models come sub-tab).

---

## 3. AUDIO / MUSICA

### 3.1 `w.audio_tab` (AudioTab)
Radio sorgente `rb_native`/`rb_user`/`rb_clone`/`rb_mute`, `no_sing`, `pick_btn`/`rec_btn`/`file_lbl`;
TTS: `tts_engine` (itemData 8 motori: kokoro/handy_clone/edge/cosyvoice2/f5tts/xtts2/fish/vibevoice),
`tts_install_btn`, `tts_voice` (itemData VoicePreset), CRUD voci, 11 QToolButton SSML→`_insert_marker`,
`tts_text` (+tag palette), `tts_preview`, `tts_gen_btn`, `tts_play_btn`, `tts_use_btn`, `tts_status`.
- **Backend**: `tts_gen_btn`→`core.tts_engine.synthesize(engine,voice,text)` (diretto, NON Spy); `tts_use_btn`→forza rb_user. `get_audio_config()` letto in `on_generate`. **Combinazioni**: 1 caso per OGNI motore; clone con/senza testo; modalità native/user/clone/mute; SSML insert/wrap/preview; guard no-voce/testo-vuoto.

### 3.2 `w.music_tab` (MusicTab)
Canzone: `song_name`, `song_tags`, `song_lyrics`, `lyr_prompt`+`lyr_lang` (15 lingue)+`btn_lyr_ai`,
`song_instr`, `song_dur`, `song_ver` (3B/7B), `song_engine` (HeartMuLa/ACE-Step), `song_temp`, `song_cfg`,
`song_lazy`; videomusicale `mv_image`+browse+`mv_style`+`mv_pace`; pulsanti `btn_song_install`/`btn_acestep_install`/`btn_song`/`btn_music_video`/`btn_mv_projects`.
Sonorizza: `son_video`+browse+`btn_last_video`, `son_prompt`, `son_neg`, `son_dur`, `son_steps`, `son_cfg`, `son_seed`, `btn_son_install`/`btn_son`.
- **Backend**: `btn_song`→`music_service.make_song_job` (HeartMuLa) o `acestep_service.make_job`; `btn_son`→`make_sonify_job` (MMAudio); `btn_lyr_ai`→`lyrics_llm.generate` (thread); `btn_music_video`→`mv_projects.create_project`. **Combinazioni**: HeartMuLa 3B/7B/strumentale/lazy/custom; ACE-Step con/senza liriche; liriche AI 15 lingue; MMAudio text-to-audio/con-video/custom; videomusicale.

---

## 4. TRAINING / LoRA

### 4.1 `w.train_tab` (TrainTab, ai-toolkit FLUX.1/2)
Dataset `ds_path`+`ds_info`+`trigger`+`btn_caption`+`cap_overwrite`; model `model` (flux1-dev/flux2-klein-4b/flux2-klein-9b/flux2-dev)+`model_path`+`model_note`+`hf_token`; params `rank`/`alpha`/`steps`/`batch`/`save_every`/`sample_every`/`lr`/`seed`/`res[512|768|1024]`(dict QCheckBox)/`quantize`/`low_vram`/`train_te`; sample `prompts`; actions `btn_install`/`btn_start`/`btn_stop`.
- **Backend**: `btn_start`→`ts.build_config`+`ts.TrainingRun.start` (subprocess). **Guard**: preflight (nome/dataset/modulo); FLUX.2-dev dialog "download pesante". **Combinazioni**: 1 caso per OGNI modello base (verifica name_or_path/is_flux/arch); toggle in config; res singola/multi; trigger in config+name+prompts.

### 4.2 `w.lora_tab` = `w.video_tab.lora` (LoraTab embedded, arch LTX)
`search`, `auto_triggers`, `reload_btn`, `list` (QListWidget), `_rows[name]{cb,sp,star}`; `selected_loras()` → `params["loras"]` in `on_generate`. **Combinazioni**: 1 caso per OGNI LoRA catalogo; selezione multipla/strength; filtro ricerca; NSFW (pilotato da `video_tab.nsfw`); trigger auto/warning.

### 4.3 `w.lora_video_tab` (LoraVideoTab, LTX-2.3, dataset auto)
Soggetti `subj_list`+add/del; editor `ed_type` (person/place/object/concept)+`ed_name`+`ed_face`+`photo_list`+`photo_angle` (5 angoli)+`_cov_checks` (copertura dinamica)+`ed_customs`+`real_list`; train `lora_name`/`lora_rank`/`lora_steps`/`lora_clip`/`lora_pulid`/`lora_lowvram`; `btn_install`/`btn_dataset`/`btn_full`.
- **Backend**: `_start(full)`→`_LoraWorker`→`lora_video_service.make_lora`/`generate_dataset`. **Combinazioni**: 1 caso per OGNI tipo soggetto; dataset-only vs full; opts rank/steps/clip/PuLID/lowvram; face-focus; angoli; copertura per tipo.

### 4.4 `w.lora_image_tab` (LoraImageTab, FLUX.2, sottoclasse)
Override train: `lora_name`, `base_model` (flux2-dev/klein-4b/klein-9b/flux1-dev), `lora_rank`, `lora_steps`, `img_res` (768/1024/1280). Eredita editor soggetti. **Backend**: `_LoraImageWorker`→`lora_image_service`. **Combinazioni**: 1 caso per OGNI base + OGNI risoluzione (width==height==res) + OGNI tipo soggetto.

---

## 5. 3D / MONDO / MODELLI

### 5.1 `w.trellis_tab` (TrellisTab, immagine→mesh) — segnale `request_trellis(dict)`
`path_edit`+browse/canvas/masked, `resolution` (512/1024/1024_cascade/1536_cascade), `file_format` (glb/obj/ply/stl), `remove_bg`, `detail`, `guidance`, `steps`, `faces`, `texture`+`tex_size`(1024/2048/4096)+`tex_guidance`+`tex_steps`, `seed`+`seed_random`, avanzate `precision`/`attn`/`bg_color`, `gen_btn`.
- **Backend**: nativo `_trellis_native` (⚠ solo seed/steps/simplify/texture_size) / ComfyUI (tutti i param). **Combinazioni**: 1 caso per OGNI risoluzione/formato/precisione/attn/bg/tex_size (ramo comfy).

### 5.2 `w.rig_tab` (RigTab, mesh→scheletro) — segnale `request_rig(dict)`
`path_edit`+browse+`last_btn`, `skeleton` (Keep/Mixamo/Unreal), `file_format` (glb/fbx), `bottom_origin`, `transfer`, `postprocess`, `gpv`, `smooth`, avanzate `num_beams`/`temperature`/`top_k`/`top_p`/`rep_pen`, `gen_btn`.
- **Backend**: nativo `_rig_native` / ComfyUI. **Combinazioni**: 1 caso per OGNI template scheletro e OGNI formato.

### 5.3 `w.world_tab` (WorldTab, WorldGen) — self-contained
`name`, `image`, `prompt`, `mode` (mesh/splat/both), `res`, `low_vram`, `hf`, `btn_install`/`btn_gen`/`btn_stop`, `btn_fbx`/`btn_obj`/`btn_web`/`btn_folder`.
- **Backend**: `_gen`→`world_service.make_job` (NON Spy: patch ws). **Combinazioni**: 1 caso per OGNI modalità output + chaining both→splat; text-only vs image; low_vram; hf_token; guard preflight.

### 5.4 `w.weights_tab` (WeightsTab, download HF)
`base`, `search`, `hf`, `_checks[id]` (dinamici da catalogo), `btn_refresh`/`btn_missing`/`btn_dl`.
- **Backend**: `_download`→QProcess `download_weights.py --ids …`. **Combinazioni**: 1 caso per OGNI modello catalogo (verifica `--ids` contiene solo il suo id); select-missing; hf_token in env.

### 5.5 `w.models_tab` (ImportModelsTab, sub-tab di Advanced) ⚠ nessun cases esistente
Percorsi `_path_edits[key]`+save/reset/suggest; compat `chk_prompt_unlock`+install/uninstall; conversione gemma/te; sorgente `source_edit`+`drop`+`file_list`+add/rm; operazioni `btn_analyze`/`btn_place`/`btn_scan`/`btn_enrich`/`btn_overrides`/`btn_build`/`btn_ingest`.
- **Backend**: `CatalogTooling` (`analyze`/`place`/`scan`/`enrich`/`build`/`full_ingest`). **Da creare i test.**

### 5.6 Visore 3D `w.trellis_tab.viewer`/`w.rig_tab.viewer` (InteractiveModel3DViewer)
`sel_lbl`, `btn_separate`/`btn_delete` (mesh-op→Blender), `btn_import`/`btn_export`/`btn_folder`/`btn_saveas`. Fallback `_Model3DViewer` senza selezione (tollerare entrambe le implementazioni via `_HAS_WEBENGINE`).

---

## 6. FILM STUDIO (`FilmStudioWindow`, finestra separata da menu Strumenti)
> Setup test: istanziare `FilmStudioWindow(None)`; **isolare il DB** (`db.ROOT`/`db.DB_PATH`→tmp, `db._conn=None` prima di tutto). Creare progetto via `new_name`+`➕ Crea` (o `FsProjectsDialog`).

Attributi: `tabs`, `script_tab`, `sb_tab`, `mt_tab`, `asset_tab`, `voice_tab`, `lora_tab`. Project bar:
`proj_combo`, `new_name`, `dur`, `path_mode` (fast/lora/mixed), `run_mode` (supervised/auto), `series`, `forbidden`, `required` + pulsanti locali (Crea/Progetti/Rinomina/Duplica/Elimina).

- **6.1 ScriptTab** (`kind` full/idea/theme/interactive, `styles`/`genres` editable, `inp` @-mention, `summary`, `text`, `btn_gen`/`btn_regen`/`btn_save`) → `llm_service.complete` (thread). Guard: input-vuoto, LLM-non-disponibile, save-senza-progetto. Versioning script.
- **6.2 LlmSettings** (anonimo, testare in isolamento): `provider` (gemma/lmstudio), `port`, `model`, `btn_connect`.
- **6.3 StoryboardTab** (`btn_break`/`btn_keys`/`btn_video`/`btn_auto`/`btn_shotlist`, `tree`, `edit` @-mention, `shot_seed`, continuity `ct_time`/`ct_weather`/`ct_light`/`ct_mood`+`btn_ct`, `btn_save_sel`/`btn_regen_sel`/`btn_regen_kf`/`btn_regen_vid`) → `L.complete_json`, `gen_service.resolve_refs`, `scheduler.enqueue`; Auto 1-click `_gen_all`→fasi keyframe→video→`auto_complete.emit`.
- **6.4 MontageTab** (`btn_refresh`, `timeline` drag&drop, `lst`, `aspect`, `fmt`, `btn_audio`/`btn_music`/`btn_ambient`/`btn_sfx`/`btn_master`/`btn_export`, `btn_animatic`, `dub_lang`+`dub_relip`+`btn_dub`) → otio_export, music_service, doppiaggio traduci→sintesi→re-lip-sync.
- **6.5 AssetManagerTab** (`search`, `ftype`, `lst`, editor `e_type`/`e_name`/`e_desc` @-mention/`e_tags`/`e_photo`, `b_save`/`b_sheet`, variazioni, wardrobe attori) → `gen_service.gen_image` (thread character sheet).
- **6.6 VoiceTab** (voce da descrizione/parametri/clone + binding attore) → `voice_studio`.
- **6.7 LoraStudioTab** (`trigger`/`flavor`/`scope`/`asset`/`b_save`; `loras_for_assets` warning >3).
- **6.8 TimelineWidget** (drag&drop clip video→`reordered`→`_on_reorder`→`db.update ord`).
- **6.9 AssetDialog** @-mention (`@`→asset, `#`→varianti; su `inp`/`edit`/`e_desc`).
- **6.10 FsProjectsDialog** (apri/duplica/elimina progetto).

Percorsi end-to-end: pipeline script→asset→storyboard→montaggio; Auto 1-click; doppiaggio; multi-LoRA >3; @-mention; timeline drag&drop; continuity propagazione; CRUD progetto+versioning.

---

## 7. Gruppo riusabile — `w.video_tab.sampler_seed` (SamplerSeedGroup)
`steps`, `cfg`, `quality_preset` (Personalizzato/Fast 8·1.0/Pro 30·3.0/HQ 50·4.0), `sampler` (euler_ancestral_cfg_pp/euler_cfg_pp/euler/dpmpp_2m), `scheduler` (LTXVScheduler/linear_quadratic/bong_tangent/karras/beta), `base_shift`, `max_shift`, `encoder` (auto/gemma/gemma-abliterated), `prompt_enhance`, `fixed_seed`, `seed`. Letto dalla pipeline video (steps→backend, cfg→guidance, seed fisso). Combinazioni: 1 caso per sampler/scheduler/encoder; preset; shift; toggle.

---

# APPENDICE — BUG / GAP REALI emersi dalla mappatura

Trovati durante la lettura del codice (da confermare/prioritizzare). I test-contratto li asseriscono.

**Critici / funzionali**
1. **`montage_tab.py`: `os` non importato a livello modulo** — `import os` è solo locale in `_animatic`; ma `os.*` è usato in `_on_music`/`_gen_ambient`/`_on_ambient`/`_gen_sfx`/`_on_sfx` → **`NameError` a runtime** cliccando 🎵 Colonna sonora / 🌧 Ambient / 💥 SFX (salvo aver già eseguito Animatic, e comunque l'import locale non entra nello scope modulo). **Bug reale.**
2. **Trellis nativo perde parametri** — solo `seed/steps/simplify/texture_size` arrivano a `make_job`; risoluzione/precisione/attn/bg/guidance/faces/formato/texture-toggle ignorati nel ramo nativo.
3. **`trellis_tab.load_model` inesistente** — `_on_trellis_native_done` la chiama sotto `hasattr` (falso) → nel ramo NATIVO il mesh 3D non viene mai caricato nel visore.
4. **Rig nativo usa il callback di Trellis** (`_on_trellis_native_done`) → il rig non è mostrato nel visore Rig né passato al tab Animazione nel ramo nativo (l'handoff `anim_tab.set_last_rig` avviene solo in ComfyUI).
5. **Regressione `colorize_no_model_sends_placeholder`** (restore_tab) — senza modelli FLUX.2 il combo mostra un placeholder non-vuoto che la GUI invia come nome modello invece di bloccare.
6. **Video nativo: modello selezionato** — il caso-contratto `video_model_selection_reaches_backend` asserisce l'inoltro; kind "checkpoint" ricade su `model="auto"` (`get_values` gestisce solo gguf/diffusion).

**Copertura mancante**
7. **`ImportModelsTab` — nessun `cases_import_models_tab.py`** (tab senza copertura; accesso via `w.models_tab`).
8. Nessun test copre: `_voice_duplicate`/`_toggle_rec`/`_pick` (audio); `_pick_mv_image`/`_pick_video`/`_on_lyrics_failed` (musica); `_compose_*` mask reali.

**Semantici / minori**
9. ACE-Step ignora `song_ver`/`song_temp`; `instrumental` espresso solo svuotando le liriche (nessun flag esplicito).
10. Videomusicale: tooltip "richiede immagine" ma la guardia reale controlla solo le liriche; `son_name()` "sfx" fisso può sovrascrivere output.
11. LoRA Foto: `img_res` forza width==height (impossibile dataset non-quadrati); `lora_pulid`/`lora_lowvram` assenti nell'override → volti FLUX.1+PuLID potrebbero non ricevere gli opts.
12. `train_tab`: `name` = trigger+dataset-dir → collisioni; alpha>rank ammesso senza avviso; `hf_token` non entra in `build_config`.
13. Talk-da-video forza finestra `[0,dur]` → rischio "seconda metà del testo persa" con default.
14. Film Studio: `seed_prompt` mancante (wizard "film" ignorato); refresh iniziale tab mancante; `find_assets` non implementa AND/OR/NOT; `_ambient_tracks` mai popolato; `script_ready` non connesso; `LlmSettings`/molti pulsanti non esposti come attributi.
15. FLUX.2 Klein inpaint / multi-reference vero restano su ComfyUI (deferred #14/#44).

---

# Piano di implementazione proposto (dopo la revisione)
1. **Harness qtbot** (`tests/gui_regression/qt_driver.py`): helper `click(w)`, `type_text(w,s)`, `select_data(combo,val)`, `open_tab(w, path)`, auto-patch dei dialoghi + Spy. Riuso di `harness.Spy`.
2. **PoC** su `video_tab` (click prompt→scrivi→click Genera→assert `params`), poi su `frame1_tab` (immagine).
3. **Estensione per gruppi** seguendo questa matrice, un file `test_qtbot_<gruppo>.py` per gruppo, riusando gli itemData/combinazioni sopra.
4. I **bug in appendice** diventano test-contratto che falliscono finché non corretti (o `xfail` documentati).

Stima ordine di grandezza: ~250–400 casi (dominati da "1 caso per OGNI modello/LoRA/formato/lingua").

---

# STATO CORREZIONI (#95) — aggiornato

I bug sono stati verificati con revisione adversariale (2 agenti su tutti i path). Alcuni dei "15" iniziali
erano falsi positivi (verificati sul codice); ne sono emersi di **nuovi e più gravi**. Stato:

## ✅ CORRETTI
- **CRITICO V1** — seed random scartato in TUTTI i path video nativi (t2v/i2v/retake/extend/multitalk) → video identici ad ogni run. Ora `vals[seed]`/`wants_fixed_seed`. (commit 7b8b131)
- **CRITICO I1** — gen immagine nativa scartava sampler/scheduler/encoder/clip_type/vae/enhance/nsfw. Ora inoltrati. (7b8b131)
- **I2** — restauro foto bloccato con ComfyUI offline nonostante il nativo. Guardia corretta. (7b8b131)
- **V2** — extend/edit-AI/multitalk saltavano l'arrotondamento ×64 → 0 frame su preset ×32. Ora `_r64`. (7b8b131)
- **HIGH rig** — `_rig_native` scartava TUTTA la config del tab. `make_job` esteso (template/format/top_k/top_p/temperature/rep_pen/num_beams/group_per_vertex) + output col formato scelto. (7b8b131)
- **V3** — multitalk seed=8 hardcoded → threading seed (re-roll) + steps/guidance overridabili. (7b8b131)
- **CRITICO montage** — `os` non importato a livello modulo → NameError su musica/ambient/SFX. (6079551)
- **trellis/rig visore** — `show_result` (era `load_model` inesistente) + `_on_rig_native_done` dedicato con handoff `anim_tab.set_last_rig`. (6079551)
- **colorize placeholder** — non invia più il placeholder come modello reale + guard modello vuoto. (6079551)
- **son_name** — derivato dal prompt (no overwrite "sfx") + tooltip videomusicale corretto. (6079551)
- **trellis facce** — `target_faces`→`--decimation` (era ignorato) + parità Trellis→Rig `set_last_mesh`. (2ddc406)
- **I5 inpaint** — `denoise` passato al nativo (da retention), non solo guidance. (2ddc406)
- **ambient** — `_ambient_tracks` popolato (corsia dedicata) + incluso nel master. (2ddc406)

## ⏭️ RESIDUI (documentati, richiedono modifiche più grandi o sono deferral intenzionali)
- **I3 (MEDIUM)** recolor FLUX.2 legge il modello dal tab Integra → arch sbagliata se Integra è su Kontext. Serve un combo FLUX.2 proprio per il recolor (modifica UI).
- **V4 (MEDIUM)** voice-clone senza testo scartato nell'a2v nativo → serve plumbing del `voice_sample` come clone-reference (o warning).
- **Film Studio Auto (MEDIUM)** `_on_auto_complete` fa `_master()` PRIMA che il job musica async finisca → master senza musica. Serve concatenare `_master` all'`on_done` della musica (orchestrazione, da testare live).
- **world web player (MEDIUM)** riusa il server http fisso :8765 per il 2° progetto (guard `is None`) → scena sbagliata. Serve kill+restart per-progetto.
- **V5 (LOW)** `rife_ckpt` scartato in nativo → serve estendere `_RifeWorker` (accetta solo factor).
- **music→project folder (LOW)** `out` calcolato ma non passato a SongParams → la canzone non finisce nella cartella progetto.
- **script_ready (LOW)** segnale emesso ma non connesso (il breakdown legge il DB direttamente → impatto minimo).
- **train name (LOW)** collisione run = trigger+dataset dir → sovrascrittura LoRA omonime.
- **trellis bg/risoluzione/precisione** — lo script nativo `trellis_generate.py` accetta SOLO seed/texture_size/decimation → gli altri richiedono di estendere lo script (o disabilitare i controlli in nativo).
- **I4 flux2-inpaint nativo** — ramo morto: FLUX.2 Klein inpaint resta su ComfyUI (deferral intenzionale #14/#44).
- **storyboard `_on_job` cross-thread (DA VERIFICARE)** — il callback dello scheduler muta widget Qt: confermare che lo scheduler marshalli sull'UI thread, altrimenti è HIGH.
