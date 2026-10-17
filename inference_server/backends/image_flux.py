"""Handler immagini del backend nativo (diffusers): text-to-image, editing (Kontext), inpaint.

Dispatch per `arch`/`task`. Dove la pipeline diffusers supporta `from_single_file` si riusano
i pesi LOCALI ComfyUI (nessun download); altrimenti serve un repo diffusers (`from_pretrained`).

| arch     | pipeline             | single_file | input        |
|----------|----------------------|-------------|--------------|
| flux     | FluxPipeline         | sì          | —            |
| flux2    | Flux2Pipeline        | NO (repo)   | —            |
| zimage   | ZImagePipeline       | sì          | —            |
| ernie    | ErnieImagePipeline   | NO (repo)   | —            |
| kontext  | FluxKontextPipeline  | sì          | image        |
| inpaint  | FluxFillPipeline     | sì          | image + mask |
"""
from __future__ import annotations

import os
from pathlib import Path

# NB: safetensors carica via mmap (default). Va bene: questo modulo gira nel PROCESSO FIGLIO isolato
# (job_runner), single-thread e SENZA uvicorn → nessuna richiesta HTTP concorrente durante il mmap, quindi
# nessuna access violation (era quello a uccidere il server persistente). Niente più load-in-RAM: il mmap
# NON legge il checkpoint in RAM (nessun picco né attesa), i pesi vanno diretti in VRAM. Vedi
# [[server-mmap-uvicorn-crash]].

# (nome classe diffusers, supporta from_single_file, richiede immagine input)
SPEC = {
    "flux":    ("FluxPipeline",        True,  False),
    "flux2":   ("Flux2Pipeline",       False, False),
    "zimage":  ("ZImagePipeline",      True,  False),
    "ernie":   ("ErnieImagePipeline",  False, False),
    "kontext": ("FluxKontextPipeline", True,  True),
    "inpaint": ("FluxFillPipeline",    True,  True),
    # #14 — FLUX.2 per editing/inpaint/outpaint/kontext (assemblati via _assemble_flux2; assemblaggio
    # locale = supports_sf irrilevante, gestito da arch.startswith('flux2') in _load). I nomi delle
    # classi diffusers vanno VALIDATI su GPU (diffusers evolve): qui le best-guess attuali. Inpaint E
    # outpaint condividono flux2_inpaint con la STESSA convenzione maschera (bianco=genera): niente inversione.
    # FLUX.2 editing per istruzione (no maschera) = Klein (Flux2KontextPipeline NON esiste in diffusers
    # 0.38; l'editing FLUX.2 è la famiglia Klein). API da validare su GPU.
    "flux2_kontext": ("Flux2KleinPipeline",        False, True),
    "flux2_inpaint": ("Flux2KleinInpaintPipeline", False, True),
}
# Stima VRAM realistica (tutto-su-GPU) per decidere GPU vs offload. FLUX.1 dev/Fill/Kontext bf16
# pesano ~24GB il solo transformer: con TE(T5)+VAE il totale sfora i 32GB → DEVE andare in
# enable_model_cpu_offload, altrimenti spilla nella shared-RAM di Windows (lentissimo, vedi outpaint).
VRAM_MB = {"flux": 24000, "flux2": 24000, "zimage": 12000, "ernie": 16000,
           "kontext": 26000, "inpaint": 26000, "flux2_kontext": 24000, "flux2_inpaint": 24000}
# archi che richiedono una maschera (inpaint-family)
_MASK_ARCHS = ("inpaint", "flux2_inpaint")

# Text encoder convertiti (pesi locali + config/tokenizer) e VAE locale, per i file solo-transformer.
# FASE G weights-unification: path DAL registro unico dove c'è un id (fallback = letterale storico).
from core.model_paths import p as _cat_p                     # stdlib-only, mai solleva

TE_DIR = r"E:\AI\Models\diffusers\_te"
AE_FLUX = _cat_p("flux-vae-ae", r"E:\AI\Models\VAE\Flux\ae.safetensors")
# Ernie: repo diffusers (config/vae/tokenizer/scheduler) + transformer e text-encoder come single-file locali
ERNIE_REPO = _cat_p("ernie-image", r"E:\AI\Models\diffusers\Ernie\ERNIE-Image")
ERNIE_TRANSFORMER = r"E:\AI\Models\diffusion_models\Ernie\ernie-image.safetensors"
ERNIE_TE = _cat_p("ernie-ministral", r"E:\AI\Models\TextEncoders\Ernie\ministral-3-3b.safetensors")
FLUX2_DIR = r"E:\AI\Models\diffusers\flux2"          # TE Mistral HF + tokenizer + config (gated, scaricati)
FLUX2_VAE = _cat_p("flux2-vae", r"E:\AI\Models\VAE\Flux\flux2-vae.safetensors")
# FLUX.2 KLEIN (edit/inpaint): encoder Qwen3 (NON Mistral). 4B ufficiale → hidden 2560 (jad 7680),
# miraclein NSFW 8B → hidden 4096 (jad 12288). Il dir giusto è scelto dal joint_attention_dim del
# transformer (vedi _assemble_flux2_klein). VAE = lo stesso flux2-vae.
KLEIN_QWEN_DIRS = {7680: _cat_p("qwen3-4b-klein-te", os.path.join(TE_DIR, "qwen3_4b")),
                   12288: _cat_p("qwen3-8b-klein-te", os.path.join(TE_DIR, "qwen3_8b"))}
# encoder alternativi per Klein 9B scelti dall'utente (params["klein_te"] = id di catalogo): stessa architettura
# Qwen3-8B dell'ufficiale, quindi valgono SOLO col transformer da 12288 (joint_attention_dim)
KLEIN_TE_ALT = {"klein9b-te-uncensored": 12288}


def _klein_te_dir(tid: str) -> str | None:
    """Cartella dell'encoder Klein alternativo `tid` (None = quello ufficiale). Id sconosciuto → errore chiaro."""
    if not tid:
        return None
    if tid not in KLEIN_TE_ALT:
        raise RuntimeError(f"encoder FLUX.2 Klein sconosciuto: {tid}")
    return _cat_p(tid, os.path.join(TE_DIR, "qwen3_8b_uncensored"))


# ── STEP LOG: timeline UNICA dal click "Genera" — timestamp + delta per OGNI passo (load, encoder, embedding,
# denoise, VAE, decode). Cronometro WALL-CLOCK (time.time) ancorato al bottone (params["_t_button"]) così è
# comparabile tra processo GUI e figlio. I job immagine sono seriali → timer a livello di modulo sicuro. ─────
import time as _time_mod                                    # noqa: E402
_STEP = {"t0": 0.0, "last": 0.0}


def _step_reset(t_button: float | None = None) -> None:
    """Ancora la timeline al click del bottone (wall-clock passato dalla GUI) se disponibile, altrimenti ORA."""
    _STEP["t0"] = float(t_button) if t_button else _time_mod.time()
    _STEP["last"] = _time_mod.time()


def _step(label: str) -> None:
    """Logga: +delta dall'ultimo step, tot DAL BOTTONE, VRAM dedicata usata/totale (spill in shared-RAM),
    RAM del processo, e COSA sta facendo. Così l'intera dinamica click→immagine si legge dai log."""
    now = _time_mod.time()
    d = now - _STEP["last"]
    tot = now - _STEP["t0"]
    _STEP["last"] = now
    vram = ""
    try:
        import torch
        if torch.cuda.is_available():
            _f, _t = torch.cuda.mem_get_info()
            _alloc = torch.cuda.memory_allocated() // 2**20     # tensori VIVI
            _resv = torch.cuda.memory_reserved() // 2**20       # riservato dall'allocatore (include il transiente
            vram = (f" | VRAMded {(_t - _f) // 2**20}/{_t // 2**20}MB "  # bf16 non rilasciato su Windows)
                    f"alloc={_alloc} resv={_resv}MB")
    except Exception:                                       # noqa: BLE001
        pass
    ram = ""
    try:
        import psutil
        _vm = psutil.virtual_memory()
        ram = f" | RAM {psutil.Process().memory_info().rss // 2**20}MB (sys used {(_vm.total-_vm.available)//2**20}MB)"
    except Exception:                                       # noqa: BLE001
        pass
    print(f"[step] +{d:6.1f}s (tot {tot:6.1f}s) {label}{vram}{ram}", flush=True)


def _place(pipe, vram_mb: int, resident_mb: int | None = None) -> str:
    """Posizionamento VRAM a TRE livelli (carica→usa→libera i componenti; l'inutilizzato esce dalla VRAM):

    1. **gpu**        : tutto residente, se free ≥ vram_mb·margine + floor (nessun offload, il più veloce).
    2. **offload**    : `enable_model_cpu_offload` — un COMPONENTE alla volta su GPU (tokenizer→T5→transformer
                        →VAE, ognuno caricato/usato/scaricato). Tiene però il transformer INTERO su GPU nel denoise.
    3. **sequential** : `enable_sequential_cpu_offload` — granularità per-MODULO (un blocco alla volta su GPU,
                        caricato/usato/liberato). Picco VRAM minimo → entra SEMPRE, niente spill nella shared-RAM
                        (più lento). Si attiva quando nemmeno il transformer intero + attivazioni ci sta.

    `resident_mb` = footprint del componente più grande tenuto residente sotto model_cpu_offload (≈ transformer).
    NB Windows: forzare GPU con VRAM stretta innesca il *system memory fallback* del driver (RAM come VRAM,
    lentissimo). Override: NATIVE_FORCE_OFFLOAD=1|seq, NATIVE_GPU_MARGIN (1.35), NATIVE_GPU_FLOOR_MB,
    NATIVE_ACT_HEADROOM_MB (9000)."""
    import os
    import torch
    _forced = os.environ.get("NATIVE_FORCE_OFFLOAD", "")
    if _forced in ("seq", "sequential"):
        try:
            pipe.enable_sequential_cpu_offload(); return "sequential"
        except Exception:                               # noqa: BLE001
            pipe.enable_model_cpu_offload(); return "offload"
    if _forced == "1" or not torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        return "offload"
    # mem_get_info è a livello DRIVER: i blocchi RESERVED-ma-inutilizzati dell'allocator torch non
    # risultano liberi (visto live: reserved 17GB vs allocated 12GB → 5GB "spariti" e placement in
    # offload a torto). Se lo scarto è significativo si restituiscono al driver PRIMA di misurare.
    try:
        if (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) > 1024**3:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:                                        # noqa: BLE001
        pass
    free_mb = torch.cuda.mem_get_info()[0] / 1024**2
    # GGUF: MAI model_cpu_offload → i pesi quantizzati (GGMLTensor) finiscono su CPU e il dequant+matmul
    # avviene su CPU = 166 s/step (spill), invece dei ~0.4 s/step gpu-resident. Il GGUF è piccolo (dequant
    # per-layer al volo su GPU) → to("cuda") sempre; l'eventuale sforo attivazioni va in shared-VRAM (poco),
    # infinitamente meglio dell'offload rotto. make_room ha già liberato le altre shell.
    _gguf_tr = getattr(getattr(pipe, "transformer", None), "_ls_is_gguf", False)
    if _gguf_tr:
        pipe.to("cuda")
        print(f"[place] GGUF → gpu-resident FORZATO (free={free_mb:.0f}MB, offload rotto sui GGUF)", flush=True)
        return "gpu"
    # Il MARGINE va applicato ai soli PESI: il working-set (attivazioni + VAE decode) è già una stima piena e
    # va SOMMATO, non moltiplicato — moltiplicarlo aggiungeva ~400MB di working fantasma e spediva in offload
    # modelli che entravano (FLUX.1 fp8: pesi 23.9GB + working 3.2GB = 29.4GB su 31GB liberi → GPU).
    margin = float(os.environ.get("NATIVE_GPU_MARGIN", "1.06"))      # solo overhead allocator/frammentazione
    floor_mb = float(os.environ.get("NATIVE_GPU_FLOOR_MB", "800"))   # lascia margine al sistema
    working = float(getattr(pipe, "_ls_working_mb", 0) or 0)
    # RESIDENT-AWARE: i componenti GIÀ su cuda (es. transformer caricato direct-to-VRAM in fp8) NON vanno
    # riconteggiati — `free_mb` li esclude già. Sottraendoli, servono solo i pesi ANCORA da caricare, così il
    # placement non va in OFFLOAD a torto quando il grosso è già residente.
    _resident = 0.0
    for _n in ("transformer", "vae", "text_encoder", "text_encoder_2"):
        _m = getattr(pipe, _n, None)
        if _m is not None and hasattr(_m, "parameters"):
            try:
                if next(_m.parameters()).device.type == "cuda":
                    _resident += sum(p.numel() * p.element_size() for p in _m.parameters(recurse=True)) / 1024**2
                    _resident += sum(b.numel() * b.element_size() for b in _m.buffers(recurse=True)) / 1024**2
            except Exception:                               # noqa: BLE001
                pass
    weights = max(float(vram_mb) - working - _resident, 0.0)   # solo i pesi NON ancora su cuda
    _need = weights * margin + working + floor_mb
    print(f"[place] free={free_mb:.0f}MB pesi={weights:.0f}MB working={working:.0f}MB margin={margin} "
          f"need={_need:.0f}MB → {'gpu' if free_mb >= _need else 'no'}", flush=True)
    if free_mb >= _need:
        pipe.to("cuda")                                 # tutto residente su GPU; la cache ModelManager lo riusa
        return "gpu"
    # Il FULL-STACK non entra. Prima di rassegnarsi all'offload: ci sta il TRANSFORMER (+VAE) da solo? Se sì →
    # STICKY: transformer/VAE RESIDENTI su GPU, solo CLIP+T5 (9.5GB, usati per l'encode a inizio job) tenuti in
    # RAM e onloadati al volo. Il transformer NON viene ricaricato ad ogni generazione — che è ciò che rende
    # enable_model_cpu_offload lento (streamma i pesi dalla RAM ad ogni run: la RAM sale, scende, e la VRAM si
    # riempie solo a metà job) e fragile (offload + hook layerwise fp8 su Windows → crash del processo).
    _tr_vae = 0.0
    for _n in ("transformer", "vae"):
        _m = getattr(pipe, _n, None)
        if _m is not None and hasattr(_m, "parameters"):
            try:
                _tr_vae += sum(p.numel() * p.element_size() for p in _m.parameters(recurse=True)) / 1024**2
                _tr_vae += sum(b.numel() * b.element_size() for b in _m.buffers(recurse=True)) / 1024**2
            except Exception:                            # noqa: BLE001
                pass
    if _tr_vae and free_mb >= _tr_vae * margin + working + floor_mb:
        print(f"[place] full-stack non entra, ma transformer+vae sì ({_tr_vae:.0f}MB) → STICKY "
              f"(transformer residente, T5/CLIP onload-per-encode)", flush=True)
        return _place_sticky(pipe)
    # Nemmeno il transformer da solo → model_cpu_offload (un componente alla volta su GPU). Lento ma entra.
    # NB niente sequential auto (segfault fp8 su Windows).
    _ = resident_mb
    pipe.enable_model_cpu_offload()
    return "offload"


def _place_sticky(pipe) -> str:
    """STICKY-TRANSFORMER (batch): transformer + VAE RESIDENTI su GPU per TUTTO il batch; solo i text-encoder
    (CLIP-L + T5) offloadati con cpu_offload_with_hook (onload per l'encode a inizio foto, offload dopo). Evita
    il 'dente-di-sega' (ricarico del transformer per ogni item) dei batch di immagini con modelli grandi.
    Fallback sicuro: se fallisce → enable_model_cpu_offload normale (niente crash)."""
    import torch
    try:
        from accelerate import cpu_offload_with_hook
        pipe.transformer.to("cuda")
        if getattr(pipe, "vae", None) is not None:
            pipe.vae.to("cuda")
        hook = None
        for _name in ("text_encoder", "text_encoder_2"):     # CLIP-L + T5: on/offload per encode
            m = getattr(pipe, _name, None)
            if m is not None:
                _, hook = cpu_offload_with_hook(m, "cuda", prev_module_hook=hook)
        pipe._ls_sticky_hook = hook
        # NIENTE `pipe._execution_device = torch.device("cuda")`: in diffusers è una property SENZA
        # setter, e assegnarla sollevava AttributeError. L'except qui sotto lo declassava a una riga
        # di log, così lo sticky FALLIVA SEMPRE (30/30 nei batch del videoclip) e ripiegava su
        # enable_model_cpu_offload — cioè l'ESATTO OPPOSTO di ciò che doveva fare: il transformer
        # veniva tolto e rimesso in GPU a OGNI foto (~5 s di warmup per immagine, metà del tempo).
        # E non serviva comunque: _execution_device deriva già 'cuda' dal transformer residente.
        _dev = getattr(pipe, "_execution_device", None)
        if str(_dev) != "cuda:0" and str(_dev) != "cuda":     # se non lo deriva, allora sì che va forzato
            type(pipe)._execution_device = property(lambda self: torch.device("cuda"))
            print(f"[place] STICKY: _execution_device era {_dev} → forzato a cuda sulla classe", flush=True)
        free_mb = torch.cuda.mem_get_info()[0] / 1024**2 if torch.cuda.is_available() else 0
        print(f"[place] STICKY: transformer+vae residenti, T5/CLIP offload-with-hook "
              f"(exec_device={getattr(pipe, '_execution_device', '?')}, free {free_mb:.0f}MB)", flush=True)
        return "sticky"
    except Exception as _e:                                  # noqa: BLE001
        # ATTENZIONE, non è un dettaglio: qui si perde il batch-mode e si torna a ricaricare il
        # transformer per ogni immagine. Va URLATO, non sussurrato (prima passava inosservato).
        print(f"[place] !! STICKY FALLITO ({type(_e).__name__}: {_e}) → offload normale: il "
              f"transformer verrà RICARICATO A OGNI IMMAGINE (batch lento)", flush=True)
        try:
            pipe.enable_model_cpu_offload()
        except Exception:                                    # noqa: BLE001
            pass
        return "offload"


def _ensure_placement(pipe, mm=None, self_key: str | None = None) -> str:
    """Placement per-JOB (coda seriale → un solo pipe attivo alla volta). Con i componenti CONDIVISI
    (ComponentStore) lo stato device/hook è GLOBALE: un'altra pipeline può aver attaccato hook di offload
    al T5 condiviso o averlo demotato su CPU. A inizio job: se l'epoch del proprio ultimo placement è
    ancora quello dello store → no-op (caso comune, stesso pipe riusato). Altrimenti: strip di TUTTI gli
    hook accelerate sui propri moduli (anche se attaccati da un'ALTRA shell — mai due catene di offload
    vive), make_room per il fabbisogno mancante (i componenti PROPRI sono refcountati → protetti), e
    _place con la VRAM di ADESSO. Ping-pong gpu↔offload ≈ 1-2s di PCIe vs 30-90s di reload da SSD."""
    from inference_server.component_store import STORE
    if getattr(pipe, "_ls_no_rebind", False):            # flux2 blockswap: pesi su CPU by design, MAI ri-piazzare
        return getattr(pipe, "_ls_place_mode", "blockswap")
    _want_sticky = bool(getattr(pipe, "_ls_want_sticky", False))
    # No-op se l'epoch combacia E la modalità corrente è già quella voluta (sticky↔non-sticky). Così un batch
    # sticky NON ri-piazza tra le foto (transformer RESTA residente = il guadagno), ma passare da/a sticky forza
    # il re-placement anche a epoch invariato.
    if getattr(pipe, "_ls_place_epoch", -1) == STORE.placement_epoch \
            and (getattr(pipe, "_ls_place_mode", "gpu") == "sticky") == _want_sticky:
        return getattr(pipe, "_ls_place_mode", "gpu")
    try:
        pipe.remove_all_hooks()                          # solo _hf_hook accelerate; i _diffusers_hook
    except Exception:                                    # noqa: BLE001   (layerwise fp8) sopravvivono
        pass
    import torch
    keys = getattr(pipe, "_ls_component_keys", ()) or ()
    est = int(getattr(pipe, "_ls_est_mb", VRAM_MB.get("flux", 24000)))
    est_missing = max(est - int(STORE.resident_of(keys)), 2000)   # spazio dei residenti già pagato
    if mm is not None:
        try:                                             # sfratta le ALTRE shell/orfani per far posto
            total_mb = torch.cuda.mem_get_info()[1] / 1024**2 if torch.cuda.is_available() else 0
            mm.make_room(min(est_missing * 1.12, max(total_mb - 2000, 0)),
                         keep={self_key} if self_key else set())
        except Exception:                                # noqa: BLE001
            pass
    mode = _place_sticky(pipe) if _want_sticky else _place(pipe, est_missing)
    for k in keys:
        # sticky: transformer+vae su cuda, T5/CLIP hooked (device 'misto') → non marcare i componenti 'cuda'
        STORE.note_device(k, "cuda" if mode == "gpu" else "cpu")
    pipe._ls_place_mode = mode
    pipe._ls_place_epoch = STORE.bump_epoch()            # invalida l'epoch delle ALTRE pipe → loro rebind
    return mode


def _place_flux2_blockswap(pipe) -> str:
    """Placement memory-bounded per FLUX.2 (~32B): block-swap manuale (no accelerate, che segfaulta coi
    buffer float8). I Linear quantizzati (Fp8Linear/Nvfp4Linear) tengono i pesi su CPU e li spostano su
    GPU 1 layer alla volta nel forward; i moduli leggeri restano su GPU. VAE intero su GPU (piccolo).
    Forza pipe._execution_device='cuda' perché coi pesi fp8 su CPU il device della pipe sarebbe ambiguo."""
    import torch
    from inference_server.backends.flux2_transformer import enable_blockswap
    dev = torch.device("cuda")
    enable_blockswap(pipe.transformer, dev)
    te = getattr(pipe, "text_encoder", None)
    if te is not None:
        try:
            enable_blockswap(te, dev)
        except Exception:                                   # noqa: BLE001
            te.to(dev)
    try:
        pipe.vae.to(dev)
    except Exception:                                       # noqa: BLE001
        pass
    # coi pesi fp8 su CPU, DiffusionPipeline.device (→ _execution_device) può risolvere 'cpu' e rompere la
    # creazione di latenti/embed. Forziamo cuda sulla classe della pipe (l'app esegue FLUX.2 solo su GPU).
    try:
        pipe.__class__._execution_device = property(lambda self: dev)
    except Exception:                                       # noqa: BLE001
        pass
    return "blockswap"


def _set_attention(pipe, arch: str = "") -> str:
    """Backend di attention configurabile con FALLBACK (per la distribuzione su GPU diverse).
    NATIVE_ATTENTION: auto|sage|sage3|sdpa. 'auto' prova Sage e ricade su SDPA (sempre disponibile).
    Z-Image usa attention_mask → SageAttention non lo supporta ('attn_mask is not supported for sage
    attention') → forziamo SDPA per arch 'zimage'."""
    import os
    # Sage SOLO per la famiglia FLUX: gli arch con attn_mask (Z-Image/Lumina2, Ernie) non lo
    # supportano ("attn_mask is not supported for sage attention") e nello stesso processo
    # EREDITANO il backend Sage settato dai job FLUX precedenti → RESET ATTIVO a 'native'.
    if arch not in ("flux", "kontext", "inpaint") and not str(arch).startswith("flux2"):
        try:
            pipe.transformer.set_attention_backend("native")
        except Exception:                                    # noqa: BLE001
            pass
        return "sdpa"
    # Gli arch IMAGE-CONDITIONED (inpaint/Fill, kontext, edit, Klein) preferirebbero SDPA (reference-correct),
    # ma forzare il RESET del backend a ogni job rallenta molto (SDPA ~2× più lento di Sage) → di default NON
    # forziamo (l'inpaint eredita il backend = velocità di Sage). La grana "a quadrati" è il VAE tiling, non la
    # sage. Per l'inpaint reference-correct puro: NATIVE_ATTENTION=sdpa (più lento, senza errore INT8 sul blend).
    if (os.environ.get("NATIVE_ATTENTION") or "").lower() == "sdpa":
        if arch in ("inpaint", "kontext") or str(arch).startswith("flux2_"):
            return "sdpa"
    want = (os.environ.get("NATIVE_ATTENTION") or "auto").lower()
    if want in ("auto", "sage", "sage3"):
        try:
            pipe.transformer.set_attention_backend("sage")
            return "sage"
        except Exception:                               # noqa: BLE001
            if want in ("sage", "sage3"):
                print("[image] Sage non disponibile su questa GPU → SDPA", flush=True)
    return "sdpa"                                       # default torch (sempre presente)


def _apply_fbcache(pipe) -> bool:
    """First Block Cache (come WaveSpeed di ComfyUI): salta i blocchi DiT quando il residuo cambia poco →
    ~2x più veloce ma PERDITA DI QUALITÀ visibile. DEFAULT OFF (qualità prima, parità con ComfyUI): la
    generazione nativa saltava blocchi solo sul t2i → immagini più povere di ComfyUI. Opt-in velocità:
    NATIVE_FBCACHE=1 (soglia NATIVE_FBCACHE_THRESHOLD, default 0.12: più alto = più veloce, meno fedele)."""
    import os
    if os.environ.get("NATIVE_FBCACHE", "0") != "1":
        return False
    try:
        from diffusers import FirstBlockCacheConfig            # cache NATIVO diffusers (compatibile)
        thr = float(os.environ.get("NATIVE_FBCACHE_THRESHOLD", "0.12"))
        pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=thr))
        return True
    except Exception as e:                              # noqa: BLE001
        print(f"[image] FBCache non applicato: {e}", flush=True)
        return False


def _maybe_compile(pipe) -> bool:
    """torch.compile del transformer. Env NATIVE_COMPILE (default 0: OPT-IN).

    NB il default è tornato OFF (07/07): con mode='reduce-overhead' (CUDA graphs) il buffer di OUTPUT del
    grafo veniva riusato tra generazioni consecutive → "accessing tensor output of CUDAGraphs that has been
    overwritten by a subsequent run" sui batch (es. Music Video Studio, 39 immagini). Quando attivo si usa
    mode='default' (niente CUDA graphs → nessun riuso di buffer) — guadagno minore ma sicuro."""
    import os
    import torch
    if os.environ.get("NATIVE_COMPILE", "0") != "1":
        return False
    if getattr(getattr(pipe, "transformer", None), "_ls_fp8_storage", False):
        print("[image] compile saltato: transformer con layerwise fp8 (hook incompatibili)", flush=True)
        return False
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True      # errori di grafo → eager, mai job-killed
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False)   # no CUDA graphs
        return True
    except Exception as e:                              # noqa: BLE001
        print(f"[image] compile non applicato: {e}", flush=True)
        return False


def _flux_in_channels(transformer_path: str) -> int | None:
    """Canali in ingresso del transformer FLUX (x_embedder/img_in) dal solo header safetensors.
    64 = FLUX standard; 384 = FLUX.1 Fill (inpaint/outpaint). None se non-safetensors/gguf o chiave assente."""
    try:
        from core.model_detect import read_header, _shape
        tensors, _ = read_header(transformer_path)
        shp = _shape(tensors, "x_embedder.weight", "img_in.weight")
        if shp and len(shp) >= 2:
            return int(shp[1])
    except Exception:                                        # noqa: BLE001
        pass
    return None


def _flux_fp8_e4m3(path: str) -> bool:
    """True se il checkpoint FLUX.1 è nativamente F8_E4M3 (dall'HEADER safetensors, non dal nome file):
    maggioranza dei .weight dei blocchi in fp8. E5M2 non arriva qui (già upcastato a bf16 da
    _normalize_flux_prefix); i GGUF sono esclusi a monte dal chiamante."""
    try:
        from core.model_detect import read_header
        tensors, _ = read_header(path)
        ds = [t.get("dtype") for k, t in tensors.items()
              if k.endswith(".weight") and ("double_blocks" in k or "transformer_blocks" in k
                                            or "single_blocks" in k)]
        return bool(ds) and sum(d == "F8_E4M3" for d in ds) > len(ds) // 2
    except Exception:                                    # noqa: BLE001
        return False


_DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
                "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}


def _precise_vram_mb(path: str, *, fp8_storage: bool = False) -> int:
    """Stima PRECISA della VRAM di un checkpoint safetensors: somma i byte REALI in memoria di ogni tensore
    (per dtype), non 'file×2+overhead'. Regole:
      - fp8 (F8_*) caricato da from_single_file → diffusers upcasta a bf16 (2B); con storage fp8 resta 1B.
      - F16/BF16 = 2B, F32 = 4B (norm/embed) restano com'è.
      - se il file NON include il text-encoder (transformer-only, es. da assemblare) → aggiunge T5+CLIP+VAE.
    Ritorna 0 se non leggibile (il chiamante ripiega sulla stima grezza). NB: NON per i .gguf (quantizzati)."""
    try:
        from core.model_detect import read_header
        tensors, _ = read_header(path)
    except Exception:                                        # noqa: BLE001
        return 0
    total, has_te = 0.0, False
    for k, t in tensors.items():
        if k == "__metadata__":
            continue
        kl = k.lower()
        if any(s in kl for s in ("text_encoder", "t5xxl", "t5.", "cond_stage", "clip_l", "text_model")):
            has_te = True
        n = 1
        for s in t.get("shape", []):
            n *= int(s)
        dt = str(t.get("dtype", "F16"))
        b = _DTYPE_BYTES.get(dt, 2)
        if dt.startswith("F8") and not fp8_storage:          # upcast fp8→bf16 salvo storage fp8 (layerwise)
            b = 2
        elif fp8_storage and dt in ("BF16", "F16") and n >= 1_000_000 \
                and not any(s in kl for s in ("norm", "embed")):
            b = 1                                            # bf16/f16 GRANDE → layerwise casting a fp8 (1B storage)
        total += n * b
    mb = total / 1024**2
    if not has_te:                                           # transformer-only → T5(9.1)+CLIP(0.25)+VAE(0.3)
        mb += 9700
    return int(mb)


# F1 unified-model-manager: il DB footprint è UNICO e vive in core/vram_estimate (stessa cache
# model_vram_cache.json, chiave "path|size[|fp8]" — il footprint dipende dal DTYPE di carico: lo
# stesso FLUX_Dev pesa 32.4GB bf16 / 23.9GB fp8-storage). Questi wrapper restano per i call-site
# esistenti ma delegano al canonico: scrittura ATOMICA + metadati (source/category/components).
from core import vram_estimate as _ve


def _vram_cache_get(model_path: str, fp8_storage: bool = False) -> int:
    return _ve.cache_get(model_path, variant=("fp8" if fp8_storage else ""))


def _vram_cache_put(model_path: str, footprint_mb: int, fp8_storage: bool = False,
                    components: dict | None = None) -> None:
    _ve.cache_put(model_path, int(footprint_mb), variant=("fp8" if fp8_storage else ""),
                  source="measured", category="image",
                  dtype=("fp8" if fp8_storage else ""), components=components)


def _measure_pipe_components(pipe) -> dict:
    """Breakdown per-componente in MB (byte reali dtype-aware) — registrato nel DB canonico per il
    planner (plan_placement decide QUALI componenti tenere residenti). {} i moduli assenti/rotti."""
    out: dict = {}
    for name in ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3", "unet"):
        m = getattr(pipe, name, None)
        if m is None or not hasattr(m, "parameters"):
            continue
        try:
            tot = sum(p.numel() * p.element_size() for p in m.parameters(recurse=True))
            tot += sum(b.numel() * b.element_size() for b in m.buffers(recurse=True))
            out[name] = int(tot / 1024**2)
        except Exception:                                    # noqa: BLE001
            pass
    return out


def _measure_pipe_mb(pipe) -> int:
    """Byte REALI in memoria dei moduli pesanti della pipeline (dopo il caricamento: onora dtype effettivi,
    fp8-storage, upcast). È la misura più accurata → cache su file per riuso."""
    return int(sum(_measure_pipe_components(pipe).values()))


def _from_single_file_robust(cls, path: str, kw: dict):
    """`from_single_file` col path veloce (low_cpu_mem_usage → meta-loading) con FALLBACK robusto.

    Alcuni checkpoint FLUX non-standard (merge/community — es. il DEFAULT NSFW `fuxCapacityNSFWPornFlux`)
    NON contengono tutte le chiavi che il config si aspetta: quei parametri/buffer restano su `meta`, e
    diffusers 0.38 chiama comunque `dispatch_model(model, device_map)` → `model.to(device)` che esplode con
    "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty()". I checkpoint standard
    (es. FLUX_Dev) coprono tutte le chiavi → nessun meta → path veloce. Perciò: provo il path veloce e SOLO
    su quell'errore ricarico con `low_cpu_mem_usage=False` (init REALE dei buffer + `load_state_dict` diretto,
    niente meta né dispatch_model) → funziona per QUALSIASI checkpoint problematico."""
    try:
        return cls.from_single_file(path, **kw)
    except (RuntimeError, NotImplementedError) as e:
        if "meta tensor" not in str(e) and "to_empty" not in str(e):
            raise
        # I checkpoint **fp8** (F8_E4M3) col fallback `low_cpu_mem_usage=False` SEGFAULTANO (la conversione
        # fp8→modello nel path non-meta è instabile): sostituire un errore con un segfault che stronca il figlio
        # sarebbe peggio → per gli fp8 do un errore CHIARO e azionabile. Il fallback bf16 resta valido.
        if _flux_fp8_e4m3(path):
            raise RuntimeError(
                f"'{os.path.basename(path)}' è un checkpoint fp8 all-in-one (model+text_encoders+vae) le cui "
                f"chiavi QK-norm (.scale) non vengono mappate dal convertitore diffusers 0.38 → tensori 'meta'. "
                f"Il caricamento nativo non lo supporta ancora. Usa un checkpoint FLUX **transformer-only** "
                f"(cartella diffusion_models\\Flux, es. FLUX_Dev/getphat) o una variante **bf16**.") from e
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[flux] {cls.__name__}.from_single_file: checkpoint bf16 con chiavi scoperte (meta-tensor) → "
              f"retry robusto low_cpu_mem_usage=False", flush=True)
        return cls.from_single_file(path, low_cpu_mem_usage=False, **kw)


def _flux_has_guidance(path: str) -> bool:
    """True se il checkpoint FLUX ha il **guidance embedder** (Dev/Fill/Kontext). Schnell e i distillati
    guidance-free NON ce l'hanno (niente `guidance_in.*`): caricati col config Dev (guidance_embeds=True)
    lasciano i 4 tensori `time_text_embed.guidance_embedder.*` su 'meta' → crash meta-tensor. In dubbio
    (header illeggibile/GGUF) assume Dev = comportamento attuale."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as h:
            for k in h.keys():
                if "guidance_in" in k or "guidance_embedder" in k:
                    return True
        return False
    except Exception:                                    # noqa: BLE001
        return True


def _flux_tr_bf16_bytes(path: str) -> int:
    """Stima i byte del SOLO transformer FLUX una volta in bf16 (target del load). Somma dall'header i tensori
    `model.diffusion_model.*` (all-in-one) o tutti (file transformer-only) × 2 byte/elem. Serve alla guardia
    direct-to-VRAM per NON sovrastimare con l'intero all-in-one (che include text_encoders/vae)."""
    from safetensors import safe_open
    total = 0
    with safe_open(path, framework="pt") as h:
        keys = list(h.keys())
        tr_keys = [k for k in keys if k.startswith("model.diffusion_model.")] or keys
        for k in tr_keys:
            n = 1
            for d in h.get_slice(k).get_shape():
                n *= d
            total += n * 2
    return total


def _flux_cfg_no_guidance(cfg_dir: str) -> str:
    """Copia (cache-ata su disco, creata una volta) di `cfg_dir` con guidance_embeds=False — per i checkpoint
    schnell/distillati senza guidance embedder. Così il modello combacia col checkpoint: 0 meta, niente crash."""
    import json
    dst = cfg_dir + "_noguid"
    dst_cfg = os.path.join(dst, "config.json")
    if not os.path.isfile(dst_cfg):
        os.makedirs(dst, exist_ok=True)
        c = json.load(open(os.path.join(cfg_dir, "config.json"), encoding="utf-8"))
        c["guidance_embeds"] = False
        with open(dst_cfg, "w", encoding="utf-8") as fh:
            json.dump(c, fh)
    return dst


def _load_flux_tr(transformer_path: str, arch: str = "flux"):
    """Carica il SOLO transformer FLUX.1 da single file (config locale per canali reali, GGUF-aware).
    Estratto da _assemble_flux: riusato dal ComponentStore (chiave per-file) e dal ramo fp8.

    FP8 SENZA UPCAST (FASE C review): un checkpoint nativo F8_E4M3 (getphat 11GB) veniva upcastato a bf16
    (22GB) → full-stack 33GB → offload. Con enable_layerwise_casting (storage fp8 / compute bf16, nativo
    diffusers 0.38) il transformer occupa 11GB → full-stack ~24GB → TUTTO GPU-resident sui 32GB. Roundtrip
    LOSSLESS (E4M3 senza scale → bf16 → fp8 = valori identici); nn.Linear intatti → LoRA/PEFT funziona.
    Off: NATIVE_FLUX1_FP8=0 (fallback = upcast bf16 identico a prima)."""
    import os
    import torch
    from diffusers import FluxTransformer2DModel
    bf16 = torch.bfloat16
    _in_ch = _flux_in_channels(transformer_path)
    _is_fill = (_in_ch == 384) or (_in_ch is None and arch == "inpaint")
    _cfg_name = "flux_fill_transformer" if _is_fill else "flux_transformer"
    _tr_cfg = os.path.join(TE_DIR, _cfg_name)
    # GGUF (llama.cpp Qn_K): transformer quantizzato tenuto tale in VRAM, dequant per-layer on-the-fly.
    _gguf_kw = {}
    _is_gguf = str(transformer_path).lower().endswith(".gguf")
    if _is_gguf:
        from diffusers import GGUFQuantizationConfig
        _gguf_kw = {"quantization_config": GGUFQuantizationConfig(compute_dtype=bf16)}
    # schnell/distillati (senza guidance embedder) col config Dev lasciano 4 tensori 'meta' → crash: usa un
    # config guidance_embeds=False. (Non per GGUF: header non-safetensors.)
    if os.path.isdir(_tr_cfg) and not _is_gguf and not _flux_has_guidance(transformer_path):
        _tr_cfg = _flux_cfg_no_guidance(_tr_cfg)
        print("[flux] checkpoint senza guidance embedder (schnell/distillato) → config guidance_embeds=False",
              flush=True)
    _sf_kw = dict(torch_dtype=bf16, **_gguf_kw)
    if os.path.isdir(_tr_cfg):
        _sf_kw.update(config=_tr_cfg, local_files_only=True)
    # DIRECT-TO-VRAM (issue #1): se in VRAM c'è spazio per il transiente bf16, carico i pesi DIRETTAMENTE su
    # cuda (from_single_file device="cuda") → niente tensore bf16 ~24GB in RAM, i byte vanno dritti in VRAM
    # (poi enable_layerwise_casting li porta a fp8 sul posto). Nel default la VRAM è già liberata da
    # gpu_registry.before_job, quindi scatta. Se non c'è spazio (offload necessario) → path CPU come prima.
    # Disattivabile con NATIVE_DIRECT_VRAM=0. GGUF escluso (ha la sua quantizzazione/placement).
    # DIRECT-TO-VRAM: DEFAULT OFF — su questa piattaforma `from_single_file(device="cuda")` dà **access violation**
    # in load_model_dict_into_meta (`.to(cuda, non_blocking)` crasha, expandable_segments non supportato). Il fp8
    # via layerwise casting resta comunque applicato dopo (su CPU non riduce, ma niente crash). Riattivabile solo
    # quando il load direct sarà robusto (block-swap nativo). Accendi per debug: NATIVE_DIRECT_VRAM=1.
    if os.environ.get("NATIVE_DIRECT_VRAM", "0") == "1" and not _is_gguf and torch.cuda.is_available():
        try:
            _free = torch.cuda.mem_get_info()[0]
            _bf16_need = _flux_tr_bf16_bytes(transformer_path)                   # SOLO il transformer, non l'all-in-one
            if _bf16_need and _free > _bf16_need * 1.1 + 1500 * 2**20:           # +10% +1.5GB margine attivazioni
                _sf_kw["device"] = "cuda"
                print(f"[flux] direct-to-VRAM: transformer su cuda "
                      f"(libera {_free//2**20}MB ≥ ~{int(_bf16_need*1.1)//2**20}MB) → niente staging RAM", flush=True)
        except Exception as _de:                                                # noqa: BLE001
            print(f"[flux] stima direct-to-VRAM saltata ({_de}) → path CPU", flush=True)
    tr = _from_single_file_robust(FluxTransformer2DModel, transformer_path, _sf_kw)
    if _is_gguf:
        tr._ls_is_gguf = True    # il GGUF NON tollera model_cpu_offload (pesi quantizzati→CPU = calcolo su CPU,
        #                          166s/step): _place lo terrà SEMPRE gpu-resident (vedi _place).
    # REGOLA "tutto su fp8" (direttiva utente): il transformer FLUX si carica SEMPRE come storage fp8
    # (layerwise casting), non solo se il checkpoint è già fp8. I bf16 vengono quantizzati a fp8 e4m3
    # (compute bf16) → ~metà VRAM, GPU-resident, niente offload. fp8-nativi = lossless; bf16 = minima
    # perdita accettata. Disattivabile con NATIVE_FLUX1_FP8=0 (fallback bf16 pieno, come prima).
    if os.environ.get("NATIVE_FLUX1_FP8", "1") == "1" and not _is_gguf:
        try:
            tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                        compute_dtype=bf16)   # skip default: norm/embed restano bf16
            tr._ls_fp8_storage = True
            _src = "fp8-nativo" if _flux_fp8_e4m3(transformer_path) else "bf16→fp8"
            print(f"[flux] transformer STORAGE fp8 (layerwise, {_src}): ~metà VRAM, niente offload", flush=True)
        except Exception as e:                           # noqa: BLE001
            print(f"[flux] layerwise fp8 non applicato ({e}) → bf16 (fallback)", flush=True)
    if _sf_kw.get("device") == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()                         # direct-to-VRAM: recupera il transiente bf16 dopo il cast fp8
    return tr, _cfg_name


def _is_svdq(path: str) -> bool:
    """True per i checkpoint SVDQuant/nunchaku (motore FP4 Blackwell): naming ufficiale svdq-fp4_*."""
    return "svdq" in os.path.basename(path or "").lower()


def _load_nunchaku_flux_tr(path: str):
    """Transformer FLUX.1 via MOTORE NUNCHAKU (SVDQuant FP4, kernel Blackwell): ~7GB in VRAM e
    ~3× più veloce del bf16 a qualità ≈16-bit. Drop-in nel posto del FluxTransformer2DModel:
    il resto della pipeline (VAE/T5/CLIP dallo store) è identico. Richiede la wheel `nunchaku`
    abbinata a torch (installata nel venv immagini, 22/07)."""
    import torch
    from nunchaku import NunchakuFluxTransformer2dModel
    print(f"[nunchaku] carico transformer SVDQ-FP4: {os.path.basename(path)}", flush=True)
    tr = NunchakuFluxTransformer2dModel.from_pretrained(path, torch_dtype=torch.bfloat16)
    return tr


def _assemble_flux(Pipe, transformer_path: str, arch: str = "flux"):
    """Assembla una pipeline FLUX-family da: transformer LOCALE (single_file) + VAE LOCALE
    (ae.safetensors) + text encoder convertiti (CLIP-L, T5-xxl). Per i file 'solo transformer'
    (diffusion_models) che non contengono i text encoder.

    COMPONENT STORE: T5/CLIP/VAE (identici per TUTTI i FLUX.1) e il transformer (per-file) vengono presi
    dallo store con refcount → mai ricaricati finché vivono lì; la pipeline è una "shell" leggera che li
    referenzia (pipe._ls_component_keys). Rollback dei refcount se l'assemblaggio fallisce a metà.
    I checkpoint `svdq-*` passano dal motore NUNCHAKU (_load_nunchaku_flux_tr), stesso assemblaggio."""
    import os
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from inference_server.component_store import STORE
    bf16 = torch.bfloat16
    _in_ch = None if _is_svdq(transformer_path) else _flux_in_channels(transformer_path)
    if arch == "inpaint" and _in_ch == 64:
        print("[flux] Modello FLUX STANDARD (64 canali) selezionato per l'inpaint: per il Fill nativo di "
              "qualità usa il modello 'FLUX.1 Fill' (384 canali). Procedo col config standard.")
    _cfg_name = "flux_fill_transformer" if (_in_ch == 384 or (_in_ch is None and arch == "inpaint")) \
        else "flux_transformer"
    tr_key = f"flux_tr:{os.path.normpath(transformer_path).lower()}:{_cfg_name}"
    got: list[str] = []                                  # rollback anti refcount-leak
    try:
        _step("_assemble_flux: acquisisco transformer dallo STORE")
        if _is_svdq(transformer_path):
            tr = STORE.acquire(tr_key, lambda: _load_nunchaku_flux_tr(transformer_path))
        else:
            tr = STORE.acquire(tr_key, lambda: _load_flux_tr(transformer_path, arch)[0])
        got.append(tr_key)
        _step("transformer acquisito")
        # VAE: ComfyUI decodifica il VAE FLUX in FP32 di default (bf16/fp16 → chiazze tonali sui fondi
        # lisci). Test A/B macchie: NATIVE_VAE_FP32=1 → fp32 (footprint VAE ~320MB, trascurabile).
        # I wrapper castano gli input (le pipeline passano latenti/immagini in bf16 → mismatch coi conv fp32).
        _vae_dt = torch.float32 if os.environ.get("NATIVE_VAE_FP32") == "1" else bf16
        def _mk_vae():
            v = AutoencoderKL.from_single_file(AE_FLUX, config=os.path.join(TE_DIR, "flux_vae"),
                                               torch_dtype=_vae_dt)
            if _vae_dt is torch.float32:
                _od, _oe = v.decode, v.encode
                v.decode = lambda z, *a, **k: _od(z.float(), *a, **k)
                v.encode = lambda x, *a, **k: _oe(x.float(), *a, **k)
            return v
        vae = STORE.acquire("vae_flux", _mk_vae); got.append("vae_flux")
        _step("VAE acquisito")
        # NB: T5/CLIP si caricano col meta-loading di transformers (_load_state_dict_into_meta_model), che
        # rilascia il GIL durante le copie tensore. Se questo avviene nel THREAD worker mentre l'event loop
        # asyncio (main thread) è attivo, il processo SEGFAULTA in torch/storage.__getitem__ (crash nativo,
        # niente traceback — il faulthandler l'ha inchiodato al load del T5). Per questo il server PRECARICA
        # i componenti condivisi all'avvio nel MAIN thread (preload_shared_components), prima di uvicorn:
        # quando il worker arriva qui, STORE.acquire trova già T5/CLIP/VAE in cache e non li ricarica.
        te = STORE.acquire("clip_l", lambda: CLIPTextModel.from_pretrained(
            os.path.join(TE_DIR, "clip_l"), torch_dtype=bf16)); got.append("clip_l")
        _step("CLIP-L acquisito")
        te2 = STORE.acquire("t5xxl", lambda: T5EncoderModel.from_pretrained(
            os.path.join(TE_DIR, "t5xxl"), torch_dtype=bf16)); got.append("t5xxl")
        _step("T5-XXL acquisito (9GB)")
    except Exception:
        for k in got:
            STORE.release(k)
        raise
    # tokenizer/scheduler: leggeri, ricreati ogni volta — MAI nello store
    tok = CLIPTokenizer.from_pretrained(os.path.join(TE_DIR, "clip_l"))
    tok2 = T5TokenizerFast.from_pretrained(os.path.join(TE_DIR, "t5xxl"))
    sched = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
        base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)
    pipe = Pipe(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok,
                text_encoder_2=te2, tokenizer_2=tok2, scheduler=sched)
    pipe._ls_component_keys = (tr_key, "vae_flux", "clip_l", "t5xxl")
    pipe._ls_shared = True
    return pipe


def _normalize_flux_prefix(path: str) -> str:
    """Normalizza FLUX.1 single-file non standard per from_single_file:
    - prefisso `model.diffusion_model.` (export SwarmUI) → strip (altrimenti KeyError single_blocks.N);
    - dtype **fp8 E5M2** (es. fuxCapacity) → upcast a bf16 (il converter diffusers gestisce E4M3 ma non
      E5M2 → errore di mappatura chiavi).
    Scrive UNA VOLTA una copia normalizzata (cache su disco). Ritorna il path normalizzato (o l'originale)."""
    import os
    import json
    import struct
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        keys = [k for k in hdr if k != "__metadata__"]
        has_prefix = bool(keys) and all(k.startswith("model.diffusion_model.") for k in keys)
        has_e5m2 = any(hdr[k].get("dtype") == "F8_E5M2" for k in keys)
        if not has_prefix and not has_e5m2:
            return path                                       # già standard → niente da fare
    except Exception:                                         # noqa: BLE001
        return path
    cache = path + ".normalized.safetensors"
    if os.path.isfile(cache) and os.path.getsize(cache) > 1024:
        return cache
    import torch
    from safetensors.torch import load_file, save_file
    sd = load_file(path)
    pref = "model.diffusion_model."
    out = {}
    for k, v in sd.items():
        nk = k[len(pref):] if k.startswith(pref) else k
        out[nk] = v.to(torch.bfloat16) if v.dtype == torch.float8_e5m2 else v
    tmp = cache + ".tmp"
    save_file(out, tmp)
    os.replace(tmp, cache)
    import logging
    logging.info("[flux] normalizzato (prefix=%s, e5m2=%s) → %s", has_prefix, has_e5m2, os.path.basename(cache))
    return cache


def _assemble_zimage(transformer_path: str):
    """Assembla ZImagePipeline da: transformer LOCALE + VAE FLUX LOCALE + Qwen3 convertito."""
    import os
    import torch
    from diffusers import (ZImageTransformer2DModel, AutoencoderKL,
                           FlowMatchEulerDiscreteScheduler, ZImagePipeline)
    from transformers import AutoModel, AutoTokenizer
    bf16 = torch.bfloat16
    if _is_svdq(transformer_path):
        # MOTORE NUNCHAKU (SVDQuant FP4 Blackwell): ~4GB invece di ~10-19GB, qualità ≈16-bit.
        from nunchaku import NunchakuZImageTransformer2DModel
        print(f"[nunchaku] carico transformer Z-Image SVDQ-FP4: {os.path.basename(transformer_path)}",
              flush=True)
        tr = NunchakuZImageTransformer2DModel.from_pretrained(transformer_path, torch_dtype=bf16)
    else:
        tr = _from_single_file_robust(ZImageTransformer2DModel, transformer_path, dict(torch_dtype=bf16))
    # fp8 STORAGE (direttiva "fp8 default ovunque"): Z-Image era bf16 (~19.5GB in VRAM) → col layerwise casting
    # storage fp8/compute bf16 scende a ~metà (~10GB) → più VRAM libera, niente offload/saturazione. Come FLUX.1.
    # Disattivabile con NATIVE_FLUX1_FP8=0. (Non si applica al motore nunchaku: già FP4 coi suoi kernel.)
    if os.environ.get("NATIVE_FLUX1_FP8", "1") == "1" and not _is_svdq(transformer_path):
        try:
            tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=bf16)
            tr._ls_fp8_storage = True
            print("[zimage] transformer STORAGE fp8 (layerwise): ~metà VRAM, niente offload", flush=True)
        except Exception as _e:                              # noqa: BLE001
            print(f"[zimage] layerwise fp8 non applicato ({_e}) → bf16 (fallback)", flush=True)
    vae = AutoencoderKL.from_single_file(AE_FLUX, config=os.path.join(TE_DIR, "flux_vae"), torch_dtype=bf16)
    te = AutoModel.from_pretrained(os.path.join(TE_DIR, "qwen3_4b"), torch_dtype=bf16)
    tok = AutoTokenizer.from_pretrained(os.path.join(TE_DIR, "qwen3_4b"))
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    return ZImagePipeline(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _assemble_flux2_klein(Pipe, transformer_path: str, te_dir: str | None = None):
    """Assembla una pipeline FLUX.2 KLEIN (edit/inpaint/kontext) — variante DISTILLATA, encoder Qwen3
    (NON Mistral), transformer fp8 puro col prefisso `model.diffusion_model.`. Config transformer
    DEDOTTA dai dim del file → 4B (jad 7680, qwen3_4b) o miraclein 8B (jad 12288, qwen3_8b). VAE = flux2-vae.
    Block-swap memory-bounded come Dev. Niente ComfyUI."""
    import json
    import torch
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from inference_server.backends.flux2_transformer import (klein_config_from_header,
                                                             load_flux2_klein_transformer)
    bf16 = torch.bfloat16
    cfg = klein_config_from_header(transformer_path)
    jad = cfg["joint_attention_dim"]
    qdir = KLEIN_QWEN_DIRS.get(jad)
    if te_dir:                                          # encoder scelto dall'utente (es. uncensored per il 9B)
        if jad != 12288:
            raise RuntimeError("L'encoder Klein scelto è un Qwen3-8B: vale solo con FLUX.2 Klein 9B, non con "
                               "questo modello (encoder Qwen3-4B).")
        qdir = te_dir
    # l'encoder può essere single-file (model.safetensors) O SHARDATO (model.safetensors.index.json +
    # model-0000X-of-0000N.safetensors) — Qwen3-8B è shardato. from_pretrained gestisce entrambi.
    _te_present = bool(qdir) and (os.path.exists(os.path.join(qdir, "model.safetensors"))
                                  or os.path.exists(os.path.join(qdir, "model.safetensors.index.json")))
    if not _te_present:
        need = {7680: "Qwen3-4B", 12288: "Qwen3-8B"}.get(jad, f"Qwen3 (jad={jad})")
        raise RuntimeError(
            f"FLUX.2 Klein richiede l'encoder {need} in '{qdir}'. Manca su disco: scaricalo "
            f"col wizard pesi (è l'unico file mancante per Klein nativo).")
    tr = load_flux2_klein_transformer(transformer_path, cfg)
    vae = AutoencoderKLFlux2.from_config(json.load(open(os.path.join(FLUX2_DIR, "vae", "config.json"))))
    vae.load_state_dict(load_file(FLUX2_VAE), assign=True)
    vae = vae.to(bf16)
    # Qwen3 (Qwen3ForCausalLM): la pipeline Klein chiama text_encoder(..., output_hidden_states=True) e
    # concatena i layer [9,18,27]. Caricato in bf16 dai pesi locali (no repo runtime, no ComfyUI).
    te = AutoModelForCausalLM.from_pretrained(qdir, torch_dtype=bf16)
    tok = AutoTokenizer.from_pretrained(qdir)
    try:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(FLUX2_DIR, "scheduler"))
    except Exception:                                   # noqa: BLE001
        sched = FlowMatchEulerDiscreteScheduler()
    return Pipe(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _assemble_flux2(Pipe, transformer_path: str, klein_te_dir: str | None = None):
    """Assembla una pipeline FLUX.2 (t2i/inpaint/kontext) da: transformer LOCALE (single_file) +
    VAE FLUX.2 LOCALE + TE Mistral HF + tokenizer (scaricati gated, solo questi). Pipe è la
    classe diffusers concreta (Flux2Pipeline, Flux2KleinInpaintPipeline, …)."""
    import os
    import torch
    # Klein è un'architettura DISTILLATA con encoder Qwen3 (≠ Dev/Mistral): instradata a parte.
    if "Klein" in getattr(Pipe, "__name__", ""):
        return _assemble_flux2_klein(Pipe, transformer_path, klein_te_dir)
    from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoModel, AutoTokenizer
    bf16 = torch.bfloat16
    import json
    from safetensors.torch import load_file
    # Transformer FLUX.2: loader fp8-aware memory-bounded (applica le weight_scale che from_single_file
    # ignora → qualità corretta; streaming+assign-and-free → niente picco RAM ~50GB). Fallback a
    # from_single_file solo se il config transformer non è presente.
    tr_cfg_dir = os.path.join(FLUX2_DIR, "transformer")
    if os.path.exists(os.path.join(tr_cfg_dir, "config.json")):
        from inference_server.backends.flux2_transformer import load_flux2_transformer
        tr = load_flux2_transformer(transformer_path, tr_cfg_dir)
    else:
        tr = _from_single_file_robust(Flux2Transformer2DModel, transformer_path, dict(torch_dtype=bf16))
    # AutoencoderKLFlux2 NON supporta from_single_file → from_config + load_state_dict (chiavi 1:1)
    vae = AutoencoderKLFlux2.from_config(json.load(open(os.path.join(FLUX2_DIR, "vae", "config.json"))))
    vae.load_state_dict(load_file(FLUX2_VAE), assign=True)
    vae = vae.to(bf16)
    # Text encoder FLUX.2 = Mistral3 nvfp4 (~12GB, file LOCALE) — dequant→bf16 (no repo 48GB, no ComfyUI)
    te_dir = os.path.join(FLUX2_DIR, "text_encoder")
    if os.path.exists(os.path.join(te_dir, "model.safetensors")):
        te = AutoModel.from_pretrained(te_dir, torch_dtype=bf16)      # se già convertito su disco
    else:
        from inference_server.backends.flux2_te import load_flux2_text_encoder
        te = load_flux2_text_encoder(te_dir)                          # dequant nvfp4 dal file locale
    # FLUX.2 vuole il PROCESSOR (PixtralProcessor), NON un tokenizer puro: Flux2Pipeline.encode_prompt usa
    # tokenizer.apply_chat_template col formato multimodale + system message. Con AutoTokenizer
    # (LlamaTokenizerFast) il chat template/i token speciali sono diversi → embedding sbagliati → RUMORE.
    from transformers import AutoProcessor
    try:
        tok = AutoProcessor.from_pretrained(os.path.join(FLUX2_DIR, "tokenizer"))
    except Exception:                                   # noqa: BLE001
        tok = AutoTokenizer.from_pretrained(os.path.join(FLUX2_DIR, "tokenizer"))
    try:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(FLUX2_DIR, "scheduler"))
    except Exception:                                   # noqa: BLE001
        sched = FlowMatchEulerDiscreteScheduler()
    return Pipe(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _assemble_ernie():
    """Assembla ErnieImagePipeline (Baidu ERNIE-Image) da pesi LOCALI, con shim per transformers <5
    (il repo diffusers vuole transformers 5: TokenizersBackend / Ministral3 / model_type 'ministral3').
    transformer (single-file, chiavi 1:1) + text-encoder Mistral3 (single-file, remap model.→language_model.)
    + vae/tokenizer/scheduler dal repo. NIENTE ComfyUI."""
    import json
    import torch
    import transformers
    from transformers import (Mistral3Model, Mistral3Config, MinistralForCausalLM, PreTrainedTokenizerFast)
    from safetensors.torch import load_file
    from diffusers import (ErnieImageTransformer2DModel, AutoencoderKLFlux2,
                           FlowMatchEulerDiscreteScheduler, ErnieImagePipeline)
    bf16 = torch.bfloat16
    if not hasattr(transformers, "Ministral3ForCausalLM"):
        transformers.Ministral3ForCausalLM = MinistralForCausalLM
    if not hasattr(transformers, "TokenizersBackend"):
        transformers.TokenizersBackend = PreTrainedTokenizerFast
    # transformer: config dal repo + pesi dal single-file (chiavi identiche, 409/409)
    tr = ErnieImageTransformer2DModel.from_config(json.load(open(os.path.join(ERNIE_REPO, "transformer", "config.json"))))
    tr.load_state_dict(load_file(ERNIE_TRANSFORMER), assign=True)
    tr = tr.to(bf16)
    # text encoder Mistral3: 'ministral3' (sconosciuto a transformers 4) → 'ministral' + sliding_window=full
    tcfg = json.load(open(os.path.join(ERNIE_REPO, "text_encoder", "config.json")))
    tc = tcfg.get("text_config", {})
    if tc.get("model_type") == "ministral3":
        tc["model_type"] = "ministral"
    if tc.get("sliding_window") is None:
        tc["sliding_window"] = tc.get("max_position_embeddings", 262144)
    te = Mistral3Model(Mistral3Config(**tcfg))
    raw = load_file(ERNIE_TE)
    rem = {("language_model." + k[6:]) if k.startswith("model.") else k: v
           for k, v in raw.items() if not k.startswith("tekken_model")}
    te.load_state_dict(rem, strict=False)
    te = te.to(bf16)
    vae = AutoencoderKLFlux2.from_pretrained(os.path.join(ERNIE_REPO, "vae"), torch_dtype=bf16)
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ERNIE_REPO, "tokenizer", "tokenizer.json"))
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(ERNIE_REPO, "scheduler"))
    return ErnieImagePipeline(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _auto_settings(arch: str, model: str) -> tuple[int, float, bool]:
    """Step + guidance OTTIMALI AUTOMATICI per (arch, modello), per generazione/inpaint/outpaint/integra.
    Evita il RUMORE da sotto-denoising (FLUX.1 Fill vuole ~40 step, non 20) e il 'piatto' dei distillati
    (hyper/turbo/lightning: pochi step, guidance bassa). La guidance fine dell'inpaint resta pilotata da
    InpaintPipeline.effective_guidance (retention-aware); qui il default per chi non la fornisce."""
    name = str(model).lower()
    hyper = any(k in name for k in ("hyp", "turbo", "lightning", "lcm", "distill", "schnell", "lightn"))
    steps = {"inpaint": 40, "kontext": 28, "flux": 28, "flux2": 28, "flux2_inpaint": 28,
             "flux2_kontext": 28, "zimage": 8, "ernie": 28}.get(arch, 28)
    guid = {"inpaint": 30.0, "kontext": 2.5, "flux": 3.5, "flux2": 4.0, "flux2_inpaint": 4.0,
            "flux2_kontext": 4.0, "zimage": 3.5, "ernie": 4.5}.get(arch, 3.5)
    if hyper:
        steps, guid = 8, min(guid, 3.5)
    return steps, guid, hyper


def _arch_of(params: dict) -> str:
    a = (params.get("task") or params.get("arch") or "flux").lower()
    if a == "flux" and any(t in str(params.get("model", "")).lower()
                           for t in ("flux2", "flux_2", "flux 2", "flux.2")):
        return "flux2"                                   # FLUX_2_Dev/FLUX.2 → loader flux2 (config+TE diversi)
    return a if a in SPEC else "flux"


_SIBLING_ARCHS = ("flux", "kontext", "inpaint")          # arch FLUX.1 che condividono gli stessi componenti


def _resolve_model_path(model_path: str) -> str:
    """Risolve il path RELATIVO dalla GUI (es. "Flux\\FLUX_Krea_Dev.safetensors") nel file assoluto sotto
    diffusion_models / Checkpoints. Tollera spazi↔underscore. Estratto da _load: lo STESSO risultato serve a
    _load e al calcolo dell'identità dei pesi in run() (pre-evict), così «crea» e «modifica» — che la GUI
    chiama una col path relativo e l'altra con quello assoluto — risolvono allo stesso file."""
    if Path(model_path).is_file():
        return model_path
    dm = Path(ERNIE_TRANSFORMER).parent.parent              # ...\Models\diffusion_models
    # cerca in diffusion_models E in Checkpoints (i checkpoint completi stanno lì, es. flux1-schnell-fp8)
    bases = [dm, dm.parent / "Checkpoints", dm.parent / "checkpoints", dm.parent]
    _name = Path(model_path).name
    cands = []
    for b in bases:
        cands += [b / model_path, (b / model_path).with_name(_name.replace(" ", "_")),
                  b / _name, b / "Flux" / _name]
    for cand in cands:
        if cand.is_file():
            return str(cand)
    return model_path


def _weights_ident(arch: str, resolved_path: str, klein_te: str = "") -> tuple:
    """IDENTITÀ DEI PESI di una pipeline: (famiglia, file risolto, encoder Klein). Due pipeline con la stessa
    identità condividono transformer/VAE/encoder → la seconda si costruisce dalla prima (zero load) e NON si
    sfratta la prima. Famiglie: core.model_identity (flux/kontext/inpaint = "flux1", Klein = "flux2k")."""
    from core.model_identity import arch_family
    return (arch_family(arch), os.path.normcase(os.path.abspath(str(resolved_path))), str(klein_te or ""))


def _sibling_pipe(mm, ident: tuple):
    """Se in cache c'è una pipeline SORELLA FLUX.1 (stessa identità dei pesi, shell condivisa) → ritornala:
    la nuova pipeline si costruisce con from_pipe (ZERO load da disco). Il confronto è per IDENTITÀ (file
    risolto), non per la stringa grezza della GUI: path relativo e assoluto dello stesso file combaciano."""
    if mm is None or not ident:
        return None
    for e in list(mm._cache.values()):
        if getattr(e.obj, "_ls_shared", False) and getattr(e.obj, "_ls_ident", None) == ident:
            return e.obj
    return None


def _flux2_sibling_pipe(mm, ident: tuple):
    """Sorella FLUX.2 Klein (flux2_kontext ↔ flux2_inpaint, stesso file e stesso encoder) col placement
    blockswap: i moduli sono GIÀ piazzati (pesi su CPU, layer-swap) e si condividono così come sono."""
    if mm is None or not ident:
        return None
    for e in list(mm._cache.values()):
        o = e.obj
        if (getattr(o, "_ls_ident", None) == ident and getattr(o, "_ls_no_rebind", False)
                and getattr(o, "_ls_place_mode", "") == "blockswap"):
            return o
    return None


def preload_shared_components() -> None:
    """Precarica i componenti FLUX.1 CONDIVISI (T5-XXL, CLIP-L, VAE) nello STORE, **nel main thread**, PRIMA
    di uvicorn. I text-encoder si caricano via transformers (safe_open mmap) che il monkeypatch no-mmap NON
    copre → se caricati nel thread worker mentre uvicorn serve richieste, segfaultano come il transformer.
    Caricandoli qui, nel main thread e prima dell'event loop, non c'è concorrenza; poi il worker li trova
    già nello STORE (refcount) e non li ricarica. Il transformer per-file invece si carica nel worker ma è
    coperto dal no-mmap. acquire+release: refcount→0, restano in cache CPU riusabili ed evictabili."""
    import torch
    from transformers import CLIPTextModel, T5EncoderModel
    from diffusers import AutoencoderKL
    from inference_server.component_store import STORE
    bf16 = torch.bfloat16
    _vae_dt = torch.float32 if os.environ.get("NATIVE_VAE_FP32") == "1" else bf16
    loaders = {
        "clip_l": lambda: CLIPTextModel.from_pretrained(os.path.join(TE_DIR, "clip_l"), torch_dtype=bf16),
        "t5xxl": lambda: T5EncoderModel.from_pretrained(os.path.join(TE_DIR, "t5xxl"), torch_dtype=bf16),
        "vae_flux": lambda: AutoencoderKL.from_single_file(
            AE_FLUX, config=os.path.join(TE_DIR, "flux_vae"), torch_dtype=_vae_dt),
    }
    for key, loader in loaders.items():
        try:
            STORE.acquire(key, loader)
            STORE.release(key)
            print(f"[preload] componente condiviso '{key}' pronto (main thread)", flush=True)
        except Exception as _e:                              # noqa: BLE001 — edizione senza FLUX.1 / file assente
            print(f"[preload] '{key}' non precaricato ({type(_e).__name__}: {_e}) — non fatale", flush=True)


def _load(model_path: str, arch: str, mm=None, klein_te: str = ""):
    """Loader per il ModelManager: ritorna (pipe, vram_mb)."""
    import torch
    import diffusers
    _model_raw = model_path                              # chiave cache originale (per il lookup sorelle)
    # Risolvi il path RELATIVO (es. "Flux\\FLUX_Krea_Dev.safetensors" dall'inventario filesystem) →
    # assoluto sotto diffusion_models. Senza, Path.is_file() è False e si cade in from_pretrained che
    # interpreta il nome come repo_id HF ("Repo id must use alphanumeric..."). Tollera spazi↔underscore.
    model_path = _resolve_model_path(model_path)
    _ident_path = model_path                            # file risolto PRIMA di dequant/normalize: base dell'identità
    # SEMPRE tracciato: quale stringa è arrivata dalla GUI e a quale file è stata risolta. Senza questa riga
    # un modello non risolto scivolava muto nel ramo HF (nessun log fino al crash) e non c'era modo di sapere
    # CHI l'avesse mandato.
    print(f"[image] modello richiesto: '{_model_raw}' → '{model_path}' "
          f"(file={Path(model_path).is_file()}, arch={arch})", flush=True)
    # AUTO-DEQUANT: se è un checkpoint quantizzato ComfyUI (comfy_quant), convertilo in bf16 (cache).
    try:
        from core.dequant_comfy import ensure_dequantized
        model_path = ensure_dequantized(model_path)
    except Exception:                                       # noqa: BLE001
        pass
    # GUARD FAMIGLIA: se l'arch (dal detector/GUI) contraddice i CANALI REALI del checkpoint, correggila —
    # altrimenti un FLUX.1 finisce nell'assembler FLUX.2 (KeyError 'double_blocks…') e viceversa.
    # x_embedder/img_in: 64|384 = FLUX.1 (base|Fill) · 128 = FLUX.2.
    if Path(model_path).is_file() and (arch in ("flux", "kontext", "inpaint") or str(arch).startswith("flux2")):
        _ich = _flux_in_channels(model_path)
        if str(arch).startswith("flux2") and _ich in (64, 384):
            _new = {"flux2": "flux", "flux2_inpaint": "inpaint", "flux2_kontext": "kontext"}.get(arch, "flux")
            print(f"[flux] arch '{arch}' richiesto ma il checkpoint è FLUX.1 ({_ich} canali) → uso '{_new}'.")
            arch = _new
        elif arch in ("flux", "kontext", "inpaint") and _ich == 128:
            _new = {"flux": "flux2", "inpaint": "flux2_inpaint", "kontext": "flux2_kontext"}.get(arch, "flux2")
            print(f"[flux] arch '{arch}' richiesto ma il checkpoint è FLUX.2 (128 canali) → uso '{_new}'.")
            arch = _new

    # FLUX.1 (flux/kontext/inpaint) in formato BFL/SwarmUI col prefisso model.diffusion_model. →
    # normalizza (strip prefisso, cache) altrimenti from_single_file fallisce. flux2/Klein gestiscono da sé.
    if arch in ("flux", "kontext", "inpaint") and Path(model_path).is_file():
        try:
            model_path = _normalize_flux_prefix(model_path)
        except Exception:                                  # noqa: BLE001
            pass
    cls_name, supports_sf, _ = SPEC[arch]
    # INPAINT: la pipeline dipende dai CANALI del modello. FluxFillPipeline vuole un transformer a 384 canali
    # (FLUX.1 Fill); un modello FLUX STANDARD (64) darebbe "mat1 4624x384 · mat2 64x3072". Per i modelli a 64
    # canali usa FluxInpaintPipeline (inpaint base mascherato, con strength) → funziona con qualsiasi FLUX.1.
    if arch == "inpaint" and Path(model_path).is_file() and _flux_in_channels(model_path) == 64:
        cls_name = "FluxInpaintPipeline"
        print("[flux] modello FLUX standard (64 canali) per inpaint → FluxInpaintPipeline (base). "
              "Per il Fill dedicato scegli FLUX.1 Fill (384 canali).", flush=True)
    Pipe = getattr(diffusers, cls_name, None)
    if Pipe is None:
        # es. Flux2KontextPipeline non esiste in diffusers 0.38 → messaggio chiaro + suggerimento
        alt = "kontext" if arch == "flux2_kontext" else "flux2_inpaint" if arch.startswith("flux2") else ""
        raise RuntimeError(
            f"La classe diffusers '{cls_name}' per arch '{arch}' non esiste in diffusers "
            f"{diffusers.__version__}." + (f" Usa '{alt}' per questa funzione." if alt else
            " Aggiorna diffusers o scegli un'altra pipeline."))
    is_file = Path(model_path).is_file()
    # RIUSO SORELLA (from_pipe): stesso FILE modello già caricato con un altro arch FLUX.1 (es. t2i→kontext)
    # → costruisci la nuova pipeline dagli STESSI componenti (zero load da disco, ~1s). Solo 64ch: il Fill
    # (384ch) ha un transformer incompatibile e non arriva mai qui con lo stesso file.
    _ident = _weights_ident(arch, _ident_path, klein_te)   # arch DOPO il guard-famiglia: è la pipeline vera
    if (arch in _SIBLING_ARCHS and is_file and _flux_in_channels(model_path) == 64):
        _src = _sibling_pipe(mm, _ident)
        if _src is not None:
            try:
                from inference_server.component_store import STORE as _ST
                try:
                    _src.remove_all_hooks()               # from_pipe NON gestisce gli hook → sanifica PRIMA
                except Exception:                         # noqa: BLE001
                    pass
                _ST.bump_epoch()                          # stato toccato → tutte le pipe faranno rebind
                pipe = Pipe.from_pipe(_src)
                pipe._ls_component_keys = _src._ls_component_keys
                for _ck in pipe._ls_component_keys:
                    _ST.retain(_ck)                       # la nuova shell referenzia gli stessi componenti
                pipe._ls_shared = True
                pipe._ls_ident = _ident
                pipe._ls_est_mb = getattr(_src, "_ls_est_mb", VRAM_MB.get(arch, 24000))
                pipe._ls_weights_mb = getattr(_src, "_ls_weights_mb", pipe._ls_est_mb)
                pipe._ls_working_mb = getattr(_src, "_ls_working_mb", 0)   # stessi componenti → stesso footprint
                _set_attention(pipe, arch)
                print(f"[flux] from_pipe: riuso componenti della sorella per '{arch}' (zero load)", flush=True)
                return pipe, 50
            except Exception as _e:                       # noqa: BLE001 — fallback: load normale
                print(f"[flux] from_pipe fallito ({_e}) → load completo", flush=True)
    if (arch in ("flux2_kontext", "flux2_inpaint") and is_file
            and os.environ.get("NATIVE_FLUX2_SIBLING", "1") != "0"):
        # KLEIN edit↔inpaint↔colora sullo STESSO file: stesso transformer + stesso encoder Qwen3 (ident uguale) →
        # la nuova pipeline usa i MODULI della sorella (già in blockswap), cambia solo la classe diffusers.
        # Prima ogni cambio di compito ricaricava ~20-30GB da SSD (transformer fp8 + Qwen3). NATIVE_FLUX2_SIBLING=0 per
        # tornare al load completo. Qualunque anomalia → ripiega sul load normale (stesso schema del from_pipe FLUX.1).
        _fs = _flux2_sibling_pipe(mm, _ident)
        if _fs is not None:
            try:
                pipe = Pipe(transformer=_fs.transformer, vae=_fs.vae, text_encoder=_fs.text_encoder,
                            tokenizer=_fs.tokenizer, scheduler=_fs.scheduler)
                _dev = torch.device("cuda")
                try:                                         # come _place_flux2_blockswap: device di esecuzione = cuda
                    pipe.__class__._execution_device = property(lambda self: _dev)
                except Exception:                            # noqa: BLE001
                    pass
                pipe._ls_no_rebind = True
                pipe._ls_place_mode = "blockswap"
                pipe._ls_ident = _ident
                for _a in ("_ls_weights_mb", "_ls_working_mb", "_ls_est_mb"):
                    if hasattr(_fs, _a):
                        setattr(pipe, _a, getattr(_fs, _a))
                print(f"[flux2] sorella Klein: riuso i moduli già residenti per '{arch}' (zero load)", flush=True)
                return pipe, VRAM_MB.get(arch, 24000)
            except Exception as _e:                          # noqa: BLE001 — fallback: load normale
                print(f"[flux2] riuso sorella Klein fallito ({_e}) → load completo", flush=True)
    if arch == "ernie":
        pipe = _assemble_ernie()                            # assemblaggio locale (shim transformers <5)
    elif arch.startswith("flux2") and is_file:
        # FLUX.2: transformer locale + VAE locale + TE Mistral HF (assemblaggio, anche inpaint/kontext)
        pipe = _assemble_flux2(Pipe, model_path, _klein_te_dir(klein_te))
    elif is_file and not supports_sf:
        raise RuntimeError(
            f"{cls_name} non supporta from_single_file: per '{arch}' serve un repo diffusers "
            f"(scaricalo col wizard pesi), non il file singolo {Path(model_path).name}.")
    elif is_file and str(model_path).lower().endswith(".gguf"):
        # GGUF = SOLO transformer quantizzato (no TE/VAE nel file) → assembla direttamente coi
        # componenti locali (il transformer è caricato con GGUFQuantizationConfig dentro _assemble_*).
        if arch in ("flux", "kontext", "inpaint"):
            pipe = _assemble_flux(Pipe, model_path, arch=arch)
        elif arch == "zimage":
            pipe = _assemble_zimage(model_path)
        else:
            raise RuntimeError(f"GGUF non supportato per arch '{arch}' (solo flux/kontext/inpaint/zimage).")
    elif is_file and arch in ("flux", "kontext", "inpaint"):
        # I file FLUX.1 in diffusion_models sono SOLO-TRANSFORMER (niente TE/VAE dentro): vai DIRETTO
        # all'assemblaggio coi componenti locali. NON tentare prima Pipe.from_single_file: fallirebbe
        # sempre ("CLIPTextModel weights missing"), e col caricamento in-RAM (no-mmap) quel tentativo
        # fallito legge comunque l'intero file da 22GB → doppio caricamento = MemoryError. _assemble_flux
        # carica solo il transformer dal file, quindi funziona anche coi rari checkpoint completi.
        pipe = _assemble_flux(Pipe, model_path, arch=arch)
    elif is_file:
        try:
            pipe = Pipe.from_single_file(model_path, torch_dtype=torch.bfloat16)
        except Exception:                               # noqa: BLE001
            # file 'solo transformer' (manca TE/VAE nel checkpoint) → assembla coi componenti locali
            if arch == "zimage":
                pipe = _assemble_zimage(model_path)
            else:
                raise
    elif Path(model_path).is_dir():
        pipe = Pipe.from_pretrained(model_path, torch_dtype=torch.bfloat16)   # repo diffusers LOCALE (dir)
    else:
        # NON è né un file né una cartella locale → from_pretrained lo tratterebbe come REPO ID HuggingFace.
        # Se il repo è in cache (es. black-forest-labs/FLUX.1-dev in E:\hf_cache) il caricamento RIESCE: 16
        # file, 7 componenti, tutto su CPU — la generazione parte sulla CPU (GPU allo 0%) e il processo muore
        # per esaurimento memoria. È esattamente il "genera su CPU e poi il server crasha". L'app è LOCALE:
        # un modello che non sta su disco è un errore di configurazione, non un pretesto per andare in rete.
        raise RuntimeError(
            f"Modello non trovato su disco: '{model_path}'. La GUI ha inviato un nome che non "
            f"corrisponde a nessun file in E:\\AI\\Models\\diffusion_models (né a una cartella diffusers "
            f"locale). Seleziona un modello dal menù a tendina. NB: NON viene scaricato da HuggingFace — "
            f"caricarlo da lì significherebbe generare su CPU.")
    # fp8 LAYERWISE sul transformer del CHECKPOINT COMPLETO (from_single_file, es. flux1-schnell-fp8 16GB):
    # from_single_file lo upcasta a bf16 (~32GB) → NON entra nei 32GB → offload lento. ComfyUI lo tiene fp8
    # (16GB) → gpu-resident → veloce. Ripristina lo storage fp8 sul transformer (l'assemble path lo fa già in
    # _load_flux_tr). Solo FLUX.1 (flux/kontext/inpaint), file fp8 e4m3, non GGUF. Off: NATIVE_FLUX1_FP8=0.
    _tr0 = getattr(pipe, "transformer", None)
    # "tutto su fp8": qualsiasi checkpoint FLUX (bf16 o fp8) → storage fp8 layerwise → entra in VRAM, niente offload.
    if (_tr0 is not None and not getattr(_tr0, "_ls_fp8_storage", False)
            and os.environ.get("NATIVE_FLUX1_FP8", "1") == "1" and arch in ("flux", "kontext", "inpaint")
            and Path(model_path).is_file() and not str(model_path).lower().endswith(".gguf")):
        try:
            _tr0.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
            _tr0._ls_fp8_storage = True
            _src = "fp8-nativo" if _flux_fp8_e4m3(model_path) else "bf16→fp8"
            print(f"[flux] checkpoint → transformer storage fp8 (layerwise, {_src}): entra in VRAM, niente offload",
                  flush=True)
        except Exception as _e:                             # noqa: BLE001
            print(f"[flux] layerwise fp8 su checkpoint non applicato ({_e}) → bf16", flush=True)
    # VAE TILING: il decode a tessere lascia GIUNZIONI RETTANGOLARI visibili ("mosaico di rettangoli"),
    # particolarmente evidenti su inpaint/editing. Le nostre immagini sono ≤1536 (cap canvas) e il decode
    # del latente FLUX entra comodamente nei 32GB SENZA tiling → di default DISATTIVATO (niente seam).
    # Riattivabile per immagini enormi con NATIVE_VAE_TILING=1.
    if os.environ.get("NATIVE_VAE_TILING") == "1":
        try:
            pipe.vae.enable_tiling()
            print("[image] VAE TILING ATTIVO (NATIVE_VAE_TILING=1) — attenzione: può lasciare griglia/mosaico",
                  flush=True)
        except Exception:                               # noqa: BLE001
            pass
    else:
        try:
            pipe.vae.disable_tiling()                    # niente giunzioni rettangolari sul decode
            pipe.vae.disable_slicing()                   # slicing sul batch: irrilevante qui, ma azzeriamo tutto
        except Exception:                               # noqa: BLE001
            pass
        # DIAGNOSTICA: se qualcuno vede ancora una griglia, il log dice lo stato reale del tiling del VAE.
        try:
            print(f"[image] VAE tiling={getattr(pipe.vae, 'use_tiling', '?')} "
                  f"slicing={getattr(pipe.vae, 'use_slicing', '?')}", flush=True)
        except Exception:                               # noqa: BLE001
            pass
    attn = _set_attention(pipe, arch)
    # FBCache rompe gli arch image-conditioned (Kontext/inpaint): la pipeline non entra nel context
    # manager dell'hook → "No context is set". Lo applichiamo solo ai t2i puri.
    fb = _apply_fbcache(pipe) if not SPEC.get(arch, ("", False, False))[2] else False
    comp = _maybe_compile(pipe)
    # Stima footprint REALE dalla dimensione del file pesi (robusto per fp16/fp8/gguf, ogni arch):
    # tutto-su-GPU ≈ pesi transformer + TE/VAE/working. Se sovrastima la VRAM, _place sceglie
    # enable_model_cpu_offload invece di spillare nella shared-RAM (vedi outpaint lento). Vale per
    # generazione/inpaint/outpaint/integra/kontext con QUALSIASI modello, non solo i preset.
    try:
        sz_mb = os.path.getsize(model_path) / 1024**2 if os.path.isfile(model_path) else 0
    except Exception:                                       # noqa: BLE001
        sz_mb = 0
    # Dim EFFETTIVA in memoria (≠ file): fp8 → diffusers UPCASTA a bf16 (2× il file: getphat 11GB→22GB);
    # GGUF → resta QUANTIZZATO (dequant per-layer on-the-fly) ≈ dim file. Senza correzioni _place sbaglia
    # il placement (fp8 sottostimato → 'gpu' che sfora; GGUF gonfiato dal floor → 'offload' inutile).
    _base = os.path.basename(model_path).lower()
    _is_gguf = _base.endswith(".gguf")
    # fp8 con layerwise casting (storage fp8) → RESTA ~dim file in memoria (niente upcast ×2) → il
    # full-stack entra GPU-resident. Senza casting (fallback) l'upcast a bf16 raddoppia come prima.
    _fp8_storage = bool(getattr(getattr(pipe, "transformer", None), "_ls_fp8_storage", False))
    # STIMA VRAM (dalla più precisa alla più grezza) + working-set attivazioni (fisso, scala poco col ns res):
    #  1) MISURA dei moduli GIÀ CARICATI: byte reali (onora fp8-storage/upcast/quantizzazione). È la verità:
    #     la pipeline è in RAM qui e il placement è comunque RINVIATO a run() → misurarla non costa nulla e
    #     non c'è motivo di indovinare. (Prima si preferiva la CACHE su file: una misura bf16 vecchia teneva
    #     FLUX_Dev a 32.4GB invece dei 23.9GB reali in fp8 → 'offload' anche quando entrava tutto in VRAM.)
    #  2) CACHE su file (chiave dtype-aware): solo se la misura non è disponibile.
    #  3) PRECISA dall'header safetensors: byte reali per dtype (fp8→bf16 salvo storage fp8, F32 norm, ecc.).
    #  4) GREZZA: file×(2 se fp8-upcast) + overhead — solo fallback (gguf o header illeggibile).
    _WORKING = 3200                                          # attivazioni + VAE decode + workspace (≤1536)
    _meas = _measure_pipe_mb(pipe) if Path(model_path).is_file() else 0
    _cached = 0 if _meas else _vram_cache_get(model_path, _fp8_storage)
    _precise = 0 if (_meas or _is_gguf or _cached) else _precise_vram_mb(model_path, fp8_storage=_fp8_storage)
    if _meas:
        _weights_mb = _meas
        print(f"[vram] footprint REALE misurato: {_meas}MB (fp8={_fp8_storage}) + working {_WORKING}", flush=True)
        _vram_cache_put(model_path, _meas, _fp8_storage, components=_measure_pipe_components(pipe))
    elif _cached:
        _weights_mb = _cached
        print(f"[vram] stima da CACHE: modello {_cached}MB + working {_WORKING}", flush=True)
    elif _precise:
        _weights_mb = _precise
        print(f"[vram] stima PRECISA (header): modello {_precise}MB + working {_WORKING}", flush=True)
    else:                                                    # fallback grezzo (gguf / header illeggibile)
        _eff_mb = sz_mb * 2 if (sz_mb and "fp8" in _base and not _is_gguf and not _fp8_storage) else sz_mb
        overhead = 800 if arch == "zimage" else (2800 if _fp8_storage else 7800)   # ex overhead - _WORKING
        if (_is_gguf or _fp8_storage) and sz_mb:
            _weights_mb = int(_eff_mb + overhead)          # niente floor VRAM_MB: quantizzato = davvero piccolo
        else:
            _weights_mb = (max(VRAM_MB.get(arch, 16000) - _WORKING, int(_eff_mb + overhead)) if sz_mb
                           else VRAM_MB.get(arch, 16000) - _WORKING)
    est_mb = int(_weights_mb) + _WORKING
    # PESI e WORKING separati: _place applica il margine solo ai PESI (il working è già una stima piena, e
    # moltiplicarlo per il margine sommava un secondo working fantasma → offload per poche centinaia di MB).
    pipe._ls_weights_mb = int(_weights_mb)
    pipe._ls_working_mb = _WORKING
    # FLUX.2 (transformer ~32B fp8 = 33GB + TE Mistral): NON entra tutto-su-GPU nei 32GB e nemmeno in
    # enable_model_cpu_offload (offload a livello-modello = transformer intero su GPU). Serve offload
    # SEQUENZIALE (granularità per-modulo: un layer alla volta su GPU). Memory-bounded e imposta
    # correttamente pipe._execution_device=cuda. Override: NATIVE_FLUX2_PLACE=offload|sequential.
    if arch.startswith("flux2"):
        # FLUX.2 (transformer ~32B fp8 = 33GB + TE Mistral nvfp4): NON entra tutto-su-GPU nei 32GB.
        #  - 'blockswap' (DEFAULT): i pesi fp8/nvfp4 restano su CPU, spostati 1 layer alla volta nel forward
        #    (Fp8Linear/Nvfp4Linear), i moduli leggeri (norm/embed/modulation/bias) stabilmente su GPU.
        #    Picco VRAM ≈ moduli leggeri + 1 layer pesante → entra nei 32GB ed è veloce (no spill shared-RAM).
        #  - 'offload'    : enable_model_cpu_offload (su Windows spilla in shared-RAM → praticamente bloccato).
        #  - 'sequential' : enable_sequential_cpu_offload (Segmentation fault coi buffer float8).
        # Override: NATIVE_FLUX2_PLACE=blockswap|offload|sequential.
        mode = (os.environ.get("NATIVE_FLUX2_PLACE") or "blockswap").lower()
        if mode == "blockswap":
            where = _place_flux2_blockswap(pipe)
        elif mode == "sequential":
            try:
                pipe.enable_sequential_cpu_offload(); where = "sequential"
            except Exception as _e:                         # noqa: BLE001
                print(f"[image] flux2 sequential offload fallito ({_e}) → blockswap", flush=True)
                where = _place_flux2_blockswap(pipe)
        else:
            pipe.enable_model_cpu_offload(); where = "offload"
        # blockswap/offload flux2: pesi fp8 su CPU by design → MAI ri-piazzare (rebind lo salta)
        pipe._ls_no_rebind = True
        pipe._ls_place_mode = where
    elif _is_svdq(model_path):
        # NUNCHAKU (SVDQuant): i kernel FP4 girano SOLO su CUDA e NON sopravvivono all'offload
        # accelerate (enable_model_cpu_offload sposta i moduli via hook → i buffer custom nunchaku
        # restano su CPU → 'CUDA error: invalid argument'/assert nel forward, visto live 24/07). Il
        # transformer svdq è piccolo (~4-7GB): residente su GPU, niente placement rinviato.
        pipe.to("cuda")
        pipe._ls_no_rebind = True
        where = "gpu (nunchaku svdq)"
    else:
        # PLACEMENT RINVIATO a inizio job (_ensure_placement in run()): con componenti CONDIVISI lo stato
        # device/hook è globale e può cambiare tra un job e l'altro (altra shell in offload, demote su CPU)
        # → la decisione gpu|offload va presa con la VRAM di ADESSO, non a load-time. _load carica su CPU.
        pipe._ls_est_mb = est_mb
        where = "deferred"
    print(f"[image] {arch}: {where} (est {est_mb}MB, file {int(sz_mb)}MB), attention={attn}, fbcache={fb}, compile={comp}", flush=True)
    # Le shell condivise pesano ~50MB proprie (tokenizer/scheduler): i GB veri sono nel ComponentStore
    # (contati lì, una volta sola per componente). Le monolitiche mantengono la stima piena.
    pipe._ls_ident = _ident                                  # identità dei pesi: pre-evict e sorelle (task 12)
    return pipe, (50 if getattr(pipe, "_ls_shared", False) else VRAM_MB.get(arch, 16000))


def register(mm, model: str, arch: str, klein_te: str = "") -> str:
    key = f"{arch}:{model}" + (f"|te={klein_te}" if klein_te else "")      # encoder diverso = pipeline diversa
    mm.register(key, lambda: _load(model, arch, mm, klein_te))
    return key


# ── Preview live per-step (latent2rgb) ───────────────────────────────────────────────────────────
# Fattori latent→RGB per FLUX (16 canali), come la preview di ComfyUI: proiezione lineare LEGGERA (no
# VAE-decode → nessun costo/rischio OOM mid-generazione). Danno un'anteprima cromatica dell'immagine che
# sta emergendo, aggiornata ad ogni step. Vedi task #88.
_FLUX_L2RGB = [
    [-0.0346, 0.0244, 0.0681], [0.0034, 0.0210, 0.0687], [0.0275, -0.0668, -0.0433],
    [-0.0174, 0.0160, 0.0617], [0.0859, 0.0721, 0.0329], [0.0004, 0.0383, 0.0115],
    [0.0405, 0.0861, 0.0915], [-0.0236, -0.0185, -0.0259], [-0.0245, 0.0250, 0.1180],
    [0.1008, 0.0755, -0.0421], [-0.0515, 0.0201, 0.0011], [0.0428, -0.0012, -0.0036],
    [0.0817, 0.0765, 0.0749], [-0.1264, -0.0522, -0.1103], [-0.0280, -0.0881, -0.0499],
    [-0.1262, -0.0982, -0.0778]]
_FLUX_L2RGB_BIAS = [-0.0329, -0.0718, -0.0851]


_PREVIEW_WARNED = False    # logga UNA sola volta la causa del mancato decode anteprima (no spam per-step)
_PREVIEW_CB_LOGGED = False  # logga UNA volta che il callback preview è attaccato (+ forma latente)


def _latent_to_pil(latent, arch):
    """Anteprima veloce del latente → PIL, senza VAE. FLUX: latent2rgb con i fattori noti; altri arch:
    normalizzazione min-max dei primi 3 canali (anteprima cromatica approssimativa). Best-effort."""
    import torch
    from PIL import Image
    x = latent[0].detach().float().cpu()                     # (C,h,w)
    c = x.shape[0]
    if arch in ("flux", "kontext", "inpaint") and c == 16:
        f = torch.tensor(_FLUX_L2RGB, dtype=torch.float32)
        b = torch.tensor(_FLUX_L2RGB_BIAS, dtype=torch.float32)
        rgb = torch.einsum("chw,cr->hwr", x, f) + b          # (h,w,3)
    else:
        rgb = x[:3].permute(1, 2, 0) if c >= 3 else x[:1].permute(1, 2, 0).repeat(1, 1, 3)
    lo, hi = rgb.amin(), rgb.amax()
    rgb = (rgb - lo) / (hi - lo + 1e-6)                       # min-max → [0,1] (sempre visibile)
    arr = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(arr, "RGB")


def _make_diffdiff_cb(pipe, init_image, change_map, steps, width, height, dtype, device, base_cb=None):
    """DIFFERENTIAL DIFFUSION: una 'forza' DIVERSA PER OGNI PIXEL, invece di un unico `strength`.

    Serve alla composizione (Integra/Componi): incollata la foto nella sua posizione/scala/rotazione,
    vogliamo che il modello ne fonda BORDI e LUCE senza riscrivere il soggetto. Con un solo `strength`
    è impossibile: basso non fonde niente, alto distrugge l'identità.

    Come funziona (arXiv 2306.00950): la change-map dice QUANDO ogni pixel smette di essere ancorato
    all'originale. A ogni step si ripristina il latente di partenza (rumoreggiato al timestep corrente)
    dove `change < soglia_step`; la soglia scende 1→0, quindi i pixel con change alto (l'anello di
    contatto) cambiano da subito, quelli con change basso (il cuore dell'oggetto) restano intatti.

    Best-effort: qualsiasi problema → si torna al comportamento normale (la generazione non si rompe).
    """
    import numpy as np
    import torch

    try:
        vae = pipe.vae
        sf = getattr(vae.config, "scaling_factor", 0.3611)
        shift = getattr(vae.config, "shift_factor", 0.1159)
        img = init_image.convert("RGB").resize((width, height), Image.LANCZOS)
        t = torch.from_numpy(np.array(img)).float().div(127.5).sub(1.0).permute(2, 0, 1)[None]
        t = t.to(device=device, dtype=vae.dtype)
        with torch.no_grad():
            lat = vae.encode(t).latent_dist.sample()
        lat = (lat - shift) * sf                              # latente dell'immagine INCOLLATA
        lat = pipe._pack_latents(lat, 1, lat.shape[1], lat.shape[2], lat.shape[3]).to(dtype)

        # change-map → risoluzione del latente packed (ogni token = patch 16x16 di pixel)
        cm = change_map.convert("L").resize((width // 16, height // 16), Image.BILINEAR)
        cm = torch.from_numpy(np.array(cm)).float().div(255.0).reshape(1, -1, 1)
        cm = cm.to(device=device, dtype=dtype)               # (1, L, 1) → broadcast sui 64 canali
        if cm.shape[1] != lat.shape[1]:
            return base_cb                                   # forme incoerenti: non rischiare
        noise = torch.randn(lat.shape, device=device, dtype=dtype, generator=None)
    except Exception as exc:                                 # noqa: BLE001
        print(f"[diffdiff] non attivabile ({type(exc).__name__}: {exc}) → inpaint normale", flush=True)
        return base_cb

    print(f"[diffdiff] attiva: change-map per-pixel su {cm.shape[1]} token", flush=True)
    sched = pipe.scheduler

    def _cb(pp, step, timestep, cbk):
        try:
            lats = cbk["latents"]
            thr = 1.0 - float(step + 1) / float(max(1, steps))   # 1 → 0 lungo il denoise
            keep = (cm < thr).to(lats.dtype)                     # 1 dove il pixel NON deve ancora cambiare
            if keep.any():
                # latente originale rumoreggiato al timestep corrente (flow-matching: lerp noise↔latent)
                sigma = timestep / 1000.0 if float(timestep) > 1.0 else float(timestep)
                orig_t = (1.0 - sigma) * lat + sigma * noise
                cbk["latents"] = lats * (1.0 - keep) + orig_t * keep
        except Exception:                                    # noqa: BLE001 — mai rompere la generazione
            pass
        return base_cb(pp, step, timestep, cbk) if base_cb else cbk

    return _cb


def _make_preview_cb(arch, width, height, steps):
    """Callback diffusers callback_on_step_end: pubblica un'anteprima del latente ogni ~1/8 degli step
    (e all'ultimo) sul preview_bus. Best-effort: qualsiasi errore NON deve rompere la generazione."""
    import io
    import torch
    every = max(1, int(steps) // 8)

    def _cb(pp, step, timestep, cbk):
        from inference_server import preview_bus
        global _PREVIEW_CB_LOGGED
        if not _PREVIEW_CB_LOGGED:                            # log UNA volta: callback attaccato + forma latente
            _PREVIEW_CB_LOGGED = True
            _step("→ encode prompt+embedding FATTO (1° step di denoise)")   # prima callback = encode finito
            try:
                _lat = cbk.get("latents")
                _sh = tuple(_lat.shape) if _lat is not None else None
                print(f"[preview] callback attivo (arch={arch}, latents={_sh}, every={every})", flush=True)
            except Exception:                                # noqa: BLE001
                pass
        # progresso ad OGNI step (leggerissimo) → barra credibile
        try:
            preview_bus.set_progress(step + 1, int(steps))
        except Exception:                                    # noqa: BLE001
            pass
        try:
            # anteprima (decode latent2rgb) ogni ~1/8 degli step e all'ultimo
            if step % every != 0 and step != int(steps) - 1:
                return cbk
            lat = cbk.get("latents")
            if lat is None:
                return cbk
            with torch.no_grad():
                l = lat
                # latente FLUX-family PACKED (B, tokens, 64) → unpack a (B,16,h,w). Vale anche per Fill/Klein.
                if l.dim() == 3 and hasattr(pp, "_unpack_latents"):
                    vsf = getattr(pp, "vae_scale_factor", 8)
                    l = pp._unpack_latents(lat, height, width, vsf)
                img = _latent_to_pil(l, arch)
                img.thumbnail((384, 384))
                buf = io.BytesIO(); img.save(buf, "PNG")
                preview_bus.set_preview(buf.getvalue(), step + 1, int(steps))
        except Exception as _pe:                             # noqa: BLE001
            global _PREVIEW_WARNED
            if not _PREVIEW_WARNED:                           # logga la CAUSA una sola volta (no spam)
                _PREVIEW_WARNED = True
                print(f"[preview] anteprima per-step non disponibile (arch={arch}): "
                      f"{type(_pe).__name__}: {_pe}", flush=True)
        return cbk

    return _cb


def _default_model(arch: str) -> str:
    """Modello di default quando il chiamante non lo specifica (music-video, LoRA dataset, API…):
    KeyError 'model' uccideva il job. Preferenze note, poi primo FLUX.1 disponibile su disco."""
    import glob
    dm = r"E:\AI\Models\diffusion_models"
    prefs = [os.path.join(dm, "Flux", "FLUX_Dev.safetensors"),
             os.path.join(dm, "Flux", "FLUX_Krea_Dev.safetensors")]
    if arch == "inpaint":
        prefs.insert(0, os.path.join(dm, "Flux", "FLUX_DEV_Fill.safetensors"))
    for p in prefs:
        if os.path.isfile(p):
            return p
    hits = sorted(glob.glob(os.path.join(dm, "**", "*flux*.safetensors"), recursive=True))
    if hits:
        return hits[0]
    raise RuntimeError("nessun modello specificato e nessun FLUX trovato in diffusion_models")


def _seam_tone_match(res_img, orig_img, mask_bool, feather: int, *, gain: bool = True,
                     tau_mult: float = 3.0, clamp: int = 40):
    """Elimina la LINEA di giunzione e il salto di tono (luminosità/saturazione/contrasto) tra la parte
    ESISTENTE (foto) e la parte GENERATA, portando la tonalità del generato a quella della foto sui pixel di
    CONFINE, con correzione forte al bordo che sfuma verso l'interno (distance transform). Usato sia da OUTPAINT
    sia da INPAINT (prima solo outpaint → l'inpaint mostrava la linea + il cambio di saturazione/luminosità).

    `mask_bool`: True = pixel GENERATO (da correggere), False = pixel FOTO (riferimento).
    `gain=True`: applica anche un guadagno per-canale (std foto / std generato) → riallinea il CONTRASTO e la
    SATURAZIONE al bordo, non solo la luminosità media. Seam-localizzato (falloff) per non ridipingere un
    eventuale oggetto inserito lontano dal bordo. Ritorna l'immagine corretta (o l'originale se non applicabile).
    """
    try:
        import cv2 as _cv2
        import numpy as _np
    except Exception:                                       # noqa: BLE001
        return res_img
    try:
        _mk = _np.asarray(mask_bool, dtype=bool)
        _ext = _mk.astype(_np.uint8); _pho = (~_mk).astype(_np.uint8)
        _D = max(6, int(feather))
        _ker = _np.ones((_D, _D), _np.uint8)
        _seam_pho = (_pho > 0) & (_cv2.dilate(_ext, _ker) > 0)      # foto adiacente al bordo
        _seam_gen = (_ext > 0) & (_cv2.dilate(_pho, _ker) > 0)      # generato adiacente al bordo
        if _seam_pho.sum() < 50 or _seam_gen.sum() < 50:
            return res_img
        _g = _np.asarray(res_img, dtype=_np.float32)
        _o = _np.asarray(orig_img, dtype=_np.float32)
        m_pho = _o[_seam_pho].mean(0); m_gen = _g[_seam_gen].mean(0)               # medie per-canale al bordo
        _off = _np.clip(m_pho - m_gen, -clamp, clamp)
        if gain:
            s_pho = _o[_seam_pho].std(0) + 1e-3; s_gen = _g[_seam_gen].std(0) + 1e-3
            _gain = _np.clip(s_pho / s_gen, 0.80, 1.25)                            # riallinea contrasto/saturazione
        else:
            _gain = _np.ones(3, _np.float32)
        _dist = _cv2.distanceTransform(_ext, _cv2.DIST_L2, 3)                      # 0 al bordo → cresce dentro
        _tau = max(24.0, float(feather) * tau_mult)
        _w = (_np.exp(-_dist / _tau)[:, :, None] * _mk[:, :, None]).astype(_np.float32)   # forte al bordo, sfuma
        _corr = (_g - m_gen[None, None, :]) * _gain[None, None, :] + m_gen[None, None, :] + _off[None, None, :]
        _g = _g * (1.0 - _w) + _corr * _w                                          # miscela: solo vicino al bordo
        from PIL import Image as _Image
        return _Image.fromarray(_np.clip(_g, 0, 255).astype("uint8"))
    except Exception as _ce:                                # noqa: BLE001
        print(f"[seam] tone-match saltato ({_ce})", flush=True)
        return res_img


def run(params: dict, mm, output_dir: str) -> dict:
    import time as _time
    import torch
    from PIL import Image
    from inference_server import preview_bus
    _t_run0 = _time.perf_counter()                          # timing pre-load (diagnosi lentezza avvio, issue #1)
    global _PREVIEW_CB_LOGGED
    _PREVIEW_CB_LOGGED = False                               # riarma il marker "encode fatto" per QUESTO job
    _step_reset(params.get("_t_button"))                    # timeline ancorata al click "Genera" (wall-clock GUI)
    _step(f"run() START nel figlio (arch={_arch_of(params)}) — prima riga dopo spawn+import")
    arch = _arch_of(params)
    model = params.get("model") or _default_model(arch)   # fix KeyError 'model' (music-video/LoRA/API)
    if not params.get("model"):
        print(f"[image] modello non specificato → default {os.path.basename(model)}", flush=True)
    # encoder Klein alternativo: solo per FLUX.2 (gli altri arch lo ignorano, la chiave resta quella di sempre)
    _kte = str(params.get("klein_te") or "") if arch.startswith("flux2") else ""
    key = register(mm, model, arch, _kte)
    # IDENTITÀ DEI PESI richiesta (famiglia, file risolto, encoder): le pipeline già in cache con la STESSA identità
    # (crea→modifica sullo stesso file) NON vanno sfrattate: condividono i pesi e la nuova si costruisce da loro.
    _req_ident = _weights_ident(arch, _resolve_model_path(str(model)), _kte)
    _step(f"register modello ({os.path.basename(str(model))})")
    preview_bus.set_message("Caricamento modello…")         # fase visibile nella finestra di anteprima
    # CAMBIO MODELLO — sfratta il PRECEDENTE *prima* di caricare il nuovo. Col caricamento in-RAM (no-mmap
    # anti-crash) il checkpoint sta ~22GB in RAM come buffer: tenerne DUE insieme (vecchio ancora in cache +
    # nuovo che si legge) = MemoryError → il worker muore e il server smette di processare (sembra un crash).
    # Se `key` NON è già in cache (modello diverso), scarica ORA tutte le shell + i componenti orfani → il
    # buffer del vecchio si libera e fa spazio. I componenti CONDIVISI (T5/CLIP/VAE) del nuovo restano nello
    # STORE (refcount li protegge appena `mm.ensure` li riacquisisce).
    if os.environ.get("NATIVE_AUTOEVICT", "1") == "1" and key not in mm._cache:
        try:
            from inference_server.component_store import STORE
            if mm.evict_others(key, keep_ident=_req_ident):  # key non in cache → scarica le shell di ALTRI pesi
                _tot = torch.cuda.mem_get_info()[1] if torch.cuda.is_available() else 0
                STORE.evict_unused(_tot)                     # libera i transformer orfani (buffer no-mmap incluso)
                mm.flush()
                import gc as _gc
                _gc.collect()                                # forza il rilascio dei buffer bytes del vecchio
            _step("pre-evict cambio-modello (scaricato il precedente)")
        except Exception as _e:                             # noqa: BLE001
            print(f"[image] pre-evict cambio modello saltato ({_e})", flush=True)
    # need_mb=None: lo sfratto pre-load distruggerebbe/demoterebbe anche i COMPONENTI che questa pipeline
    # sta per riusare (refcount ancora 0 prima dell'acquire). _load carica su CPU; lo spazio si libera in
    # _ensure_placement (make_room con i componenti PROPRI ormai refcountati → protetti).
    _step("PRIMA di mm.ensure (sto per caricare transformer+componenti)")
    try:
        pipe = mm.ensure(key)                              # carica il NUOVO modello (mmap→VRAM) + componenti
    except (MemoryError, OSError) as _me:
        # Windows "paging file too small" (os error 1455) o MemoryError: il COMMIT LIMIT (RAM + file di
        # paging) è saturo → il mmap del checkpoint non trova spazio da impegnare. diffusers lo maschera con
        # un MemoryError criptico (fa f.read() sull'intero file nel gestore d'errore). Diamo un messaggio
        # AZIONABILE con i numeri reali. Vedi [[server-mmap-uvicorn-crash]] / task pagefile.
        _hint = ""
        try:
            import psutil
            _av = psutil.virtual_memory().available / 2**30
            _sz = os.path.getsize(model) / 2**30 if os.path.isfile(model) else 0
            _hint = f" (RAM libera {_av:.0f}GB, modello ~{_sz:.0f}GB)"
        except Exception:                                  # noqa: BLE001
            pass
        raise RuntimeError(
            f"Memoria di sistema esaurita nel caricare il modello{_hint}. Su Windows è il 'commit limit' "
            f"(RAM + file di paging) ad essere pieno, non la sola RAM. Rimedi: chiudi i programmi che "
            f"consumano molta RAM (es. Unity, browser), oppure aumenta il file di paging di Windows "
            f"(Impostazioni → Sistema → Memoria virtuale; ora è fisso a 16GB). Dettaglio: {_me}") from _me
    # timing pre-load (issue #1): quanto ci mette il modello a essere pronto in RAM/VRAM. cache-hit = istantaneo
    # (persistent-child/riuso), altrimenti = lettura file + costruzione pipeline. Reso visibile nel log del job.
    def _vram_mb():                                          # VRAM dedicata usata (per vedere lo spill in shared)
        try:
            _f, _t = torch.cuda.mem_get_info(); return (_t - _f) // 2**20, _t // 2**20
        except Exception:                                   # noqa: BLE001
            return -1, -1
    _u, _tt = _vram_mb()
    _step(f"mm.ensure FATTO (transformer+componenti pronti, cache-hit={key in getattr(mm, '_cache', {})})")
    print(f"[image] FASE pre-load: {_time.perf_counter()-_t_run0:.1f}s | VRAM {_u}/{_tt}MB "
          f"(cache-hit={key in getattr(mm, '_cache', {})}, arch={arch}, {os.path.basename(str(model))})", flush=True)
    # STICKY-TRANSFORMER: transformer+VAE RESIDENTI, T5/CLIP onload-per-encode poi OFFLOAD (liberano ~10GB
    # subito dopo l'encode del prompt). Attivato da placement_mode='sticky' (batch) O quando c'è **PuLID** (i
    # suoi ~2-3GB + hook + attivazioni saturerebbero col T5 residente).
    pipe._ls_want_sticky = (str(params.get("placement_mode", "")).lower() == "sticky"
                            or bool(params.get("pulid_face")))
    # KEEP-UNTIL-SPACE-NEEDED (richiesta utente): un modello RESTA residente finché non serve la sua VRAM.
    # Prima qui si sfrattavano gli altri modelli SEMPRE (evict incondizionato) → scaricava anche con VRAM libera
    # ("pensa di doverlo scaricare"). Ora: sfratto SOLO se la VRAM libera NON basta per il nuovo modello, e
    # libero solo il DEFICIT (size-aware), non tutta la VRAM. Così più modelli coesistono se c'è spazio.
    if os.environ.get("NATIVE_AUTOEVICT", "1") == "1":
        try:
            from inference_server.component_store import STORE
            _free_mb = (torch.cuda.mem_get_info()[0] // 2**20) if torch.cuda.is_available() else 0
            _need_mb = _measure_pipe_mb(pipe) + int(getattr(pipe, "_ls_working_mb", 0) or 3200)
            # i componenti GIÀ su GPU (T5/VAE/transformer condivisi, stessi usati dalla nuova pipe) non vanno
            # richiesti di nuovo: è lo stesso conto di _ensure_placement (resident_of). Senza, un modello già
            # residente "non entrava" e faceva sfrattare gli altri per nulla.
            _need_mb = max(0, _need_mb - int(STORE.resident_of(getattr(pipe, "_ls_component_keys", ()) or ())))
            if _free_mb < _need_mb:                          # serve spazio → sfratta (size-aware, solo il deficit)
                if mm.evict_others(key, keep_ident=_req_ident):
                    STORE.evict_unused(max(1, _need_mb - _free_mb))
                    mm.flush()
                print(f"[image] auto-evict: serviva spazio (free {_free_mb} < need {_need_mb}MB) → sfrattati altri",
                      flush=True)
            else:
                print(f"[image] auto-evict SALTATO: c'è spazio (free {_free_mb} ≥ need {_need_mb}MB) → gli altri "
                      f"modelli RESTANO residenti", flush=True)
        except Exception as _e:                             # noqa: BLE001
            print(f"[image] auto-evict saltato ({_e})", flush=True)
    _step("auto-evict FATTO (sto per fare il placement in VRAM)")
    where = _ensure_placement(pipe, mm, key)                # placement per-JOB (gpu|offload con VRAM attuale)
    _step(f"placement FATTO = {where}")
    _set_attention(pipe, arch)                              # per-job: il transformer CONDIVISO può avere il
    print(f"[image] job {arch}: placement={where}", flush=True)  # processor dell'arch precedente (sage↔sdpa)
    mm.apply_loras(pipe, params.get("loras"))
    _step("LoRA applicate (sto per iniziare il denoising)")
    preview_bus.set_message("Generazione…")

    # Face adapter PuLID (identità del volto da una foto) — FLUX/kontext
    _pulid_h = None
    _tcfg = float(params.get("pulid_true_cfg", 1.0) or 1.0)   # true-CFG (2 pass/step) — deciso QUI: serve al PuLID
    _step(f"PuLID? pulid_face={bool(params.get('pulid_face'))} arch={arch} placement={where} "
          f"(l'offload rompe gli hook; deve essere gpu)")
    if params.get("pulid_face") and arch in ("flux", "kontext"):
        try:
            from core import pulid_adapter
            _faces = params["pulid_face"]                     # str singola, lista, o "a|b|c" (multi-foto → media)
            if isinstance(_faces, str) and "|" in _faces:
                _faces = [f for f in _faces.split("|") if f.strip()]
            _pl = pulid_adapter.load(pipe.transformer)
            # true-CFG (2 pass/step): identità sul cond, uncond sull'uncond → il CFG amplifica l'identità;
            # inoltre lo step di denoising va contato ogni 2 forward (prima start_step si sfasava).
            _use_tcfg = _tcfg > 1.0
            if _use_tcfg:
                _emb, _unemb = pulid_adapter.extract_id(_pl, _faces, want_uncond=True)
            else:
                _emb, _unemb = pulid_adapter.extract_id(_pl, _faces), None
            _pulid_h = pulid_adapter.attach(pipe.transformer, _pl, _emb,
                                            float(params.get("pulid_strength", 0.8)),
                                            start_step=int(params.get("pulid_start_step", 0)),
                                            uncond_emb=_unemb, cfg_passes=2 if _use_tcfg else 1)
            _nf = len(_faces) if isinstance(_faces, list) else 1
            _step(f"PuLID ATTACCATO: {len(_pulid_h) if _pulid_h else 0} hook, {_nf} foto, "
                  f"strength={params.get('pulid_strength', 0.8)}, true_cfg={_tcfg}")
            print(f"[pulid] attivo ({len(_pulid_h)} hook, {_nf} foto, strength={params.get('pulid_strength', 0.8)}, "
                  f"start_step={params.get('pulid_start_step', 0)}, true_cfg={_tcfg})", flush=True)
        except Exception as _e:                          # noqa: BLE001
            import traceback as _tb
            _step(f"PuLID FALLITO: {type(_e).__name__}: {str(_e)[:80]}")
            print(f"[pulid] disattivato (errore): {_e}\n{_tb.format_exc()[-600:]}", flush=True)
            _pulid_h = None
            try:                                         # LEAK: i modelli PuLID mezzo-caricati restano in VRAM
                from core import pulid_adapter           # (→ VRAM piena → encode lentissimo). Svuota la cache.
                pulid_adapter.clear_cache()
            except Exception:                            # noqa: BLE001
                pass

    g = torch.Generator(device="cuda").manual_seed(int(params.get("seed", 0)))
    # SETTINGS: il valore ESPLICITO (dal GUI) va SEMPRE rispettato — niente override lato pipeline.
    # _auto_settings serve SOLO come default quando il parametro NON è fornito (es. API/mobile senza GUI).
    # L'ottimizzazione dei parametri pre-compila i WIDGET del GUI (vedi optimal_settings), non forza qui.
    _ast, _agd, _hyp = _auto_settings(arch, model)
    _steps = int(params.get("steps") or 0) or _ast
    _guid = float(params["guidance"]) if params.get("guidance") is not None else _agd
    kw = dict(prompt=params.get("prompt", ""),
              num_inference_steps=_steps,
              guidance_scale=_guid,
              generator=g)
    neg = params.get("negative")
    # PuLID true CFG: migliora l'aderenza al prompt / bilancia identità (solo se la pipeline lo supporta).
    # `_tcfg` è già stato calcolato sopra (serviva anche al PuLID per il conteggio pass/step + uncond).
    if _tcfg > 1.0 and params.get("pulid_face"):
        try:
            import inspect
            if "true_cfg_scale" in inspect.signature(pipe.__call__).parameters:
                kw["true_cfg_scale"] = _tcfg
                kw["negative_prompt"] = neg or "bad quality, worst quality, low quality, blurry"
        except Exception:                                    # noqa: BLE001
            pass
    # immagine/maschera per kontext e inpaint
    _, _, needs_image = SPEC[arch]
    if needs_image:
        img_p = params.get("image_path")
        if not img_p or not Path(img_p).exists():
            return {"status": "error", "error": f"'{arch}' richiede 'image_path' valido."}
        kw["image"] = Image.open(img_p).convert("RGB")
        # PROPORZIONI: senza width/height espliciti, FluxFill/Kontext default a 1024² (quadrato) →
        # la foto viene distorta. Deriviamo W/H dalle dimensioni REALI dell'immagine (aspect-correct),
        # multipli di 16 (richiesti da Flux: vae 8 × patch 2), lato lungo cap 1536 per la VRAM.
        _orig_wh = kw["image"].size                          # dimensioni ESATTE originali (per il restore finale)
        _iw, _ih = _orig_wh
        _cap = int(params.get("max_side", 1536))
        _s = min(1.0, _cap / max(_iw, _ih))
        _W = max(256, round(_iw * _s / 16) * 16)
        _H = max(256, round(_ih * _s / 16) * 16)
        kw["width"], kw["height"] = _W, _H
        if (_W, _H) != (_iw, _ih):                           # allinea l'immagine alla griglia /16 + cap
            kw["image"] = kw["image"].resize((_W, _H), Image.LANCZOS)
        if arch in _MASK_ARCHS:
            mk = params.get("mask_path")
            if not mk or not Path(mk).exists():
                return {"status": "error", "error": f"'{arch}' richiede 'mask_path' valido."}
            # Convenzione UNICA per inpaint E outpaint: la maschera arriva già col BIANCO = regione da
            # (ri)generare, NERO = pixel da mantenere. Inpaint: bianco = area disegnata. Outpaint: bianco =
            # estensione (bordo) da riempire, nero = foto (vedi mask_canvas._border_mask). NIENTE inversione:
            # invertirla faceva rigenerare la FOTO e lasciava NERA l'estensione (bug outpaint).
            _mimg = Image.open(mk).convert("RGB")
            if _mimg.size != (_W, _H):                       # maschera allineata a width/height
                _mimg = _mimg.resize((_W, _H), Image.NEAREST)
            kw["mask_image"] = _mimg
    else:
        kw["width"] = int(params.get("width", 1024))
        kw["height"] = int(params.get("height", 1024))
    # negative_prompt: passalo SOLO se la pipeline lo espone davvero (introspezione firma) → bulletproof
    # contro "unexpected keyword argument 'negative_prompt'" su FluxFill/Klein/zimage (guidance-distilled).
    if neg:
        try:
            import inspect
            if "negative_prompt" in inspect.signature(pipe.__call__).parameters:
                kw["negative_prompt"] = neg
            else:
                # visibilità per l'utente (dialog anteprima): il negative è stato IGNORATO dall'arch
                print(f"[image] negative_prompt IGNORATO: {type(pipe).__name__} non lo supporta "
                      f"(guidance-distilled)", flush=True)
                preview_bus.set_message(f"Nota: negative ignorato ({arch} non lo supporta)")
        except Exception:                                   # noqa: BLE001
            pass
    # FluxInpaintPipeline (inpaint BASE, modelli 64ch, SEGUE il prompt) usa 'strength' = quanto rigenerare la
    # regione mascherata (FluxFillPipeline invece fa denoise pieno, niente strength). Default alto = sostituzione
    # decisa. NB: il FLUX base ha guidance EMBEDDED BASSA (~3.5) come nel t2i — a 30 si BRUCIA (≠ Fill dev che
    # vuole 30). Clampo la guidance al range base.
    if type(pipe).__name__ == "FluxInpaintPipeline":
        kw["strength"] = float(params.get("denoise") or 0.9)
        kw["guidance_scale"] = min(float(kw.get("guidance_scale", 3.5)), 5.0)

    # Preview live per-step (task #88): anteprima del latente ad ogni ~1/8 degli step sul preview_bus,
    # che la GUI polla via GET /preview. Best-effort: se la pipeline non supporta callback_on_step_end
    # (es. alcune pipeline custom), si riprova senza.
    if os.environ.get("NATIVE_NO_PREVIEW_CB") != "1":       # A/B macchie: callback escludibile via env
        try:
            kw["callback_on_step_end"] = _make_preview_cb(arch, int(kw.get("width", 1024)),
                                                          int(kw.get("height", 1024)), kw["num_inference_steps"])
            kw["callback_on_step_end_tensor_inputs"] = ["latents"]
        except Exception:                                   # noqa: BLE001
            pass

    # DIFFERENTIAL DIFFUSION (composizione): se il job porta una change-map, la incateniamo al callback
    # dell'anteprima. Così i bordi si fondono e il soggetto incollato resta sé stesso — cosa impossibile
    # con un solo `strength` globale. Attiva solo su richiesta esplicita (Integra/Componi la passano).
    _cm_p = params.get("change_map")
    if _cm_p and Path(str(_cm_p)).exists() and params.get("image_path"):
        try:
            import torch as _t
            _dev = getattr(pipe, "_execution_device", None) or _t.device("cuda")
            _dt = getattr(pipe.transformer, "dtype", _t.bfloat16)
            kw["callback_on_step_end"] = _make_diffdiff_cb(
                pipe, Image.open(params["image_path"]), Image.open(str(_cm_p)),
                int(kw["num_inference_steps"]), int(kw.get("width", 1024)), int(kw.get("height", 1024)),
                _dt, _dev, base_cb=kw.get("callback_on_step_end"))
            kw["callback_on_step_end_tensor_inputs"] = ["latents"]
        except Exception as _dde:                            # noqa: BLE001
            print(f"[diffdiff] disattivata ({_dde}) → inpaint normale", flush=True)
    _step(f"GEN START pipe() — encode prompt+embedding → {kw.get('num_inference_steps','?')} step denoise → VAE decode")
    _vae = getattr(pipe, "vae", None)                       # cronometro del VAE decode (sotto-fase interna a pipe())
    _orig_dec = getattr(_vae, "decode", None)
    if _vae is not None and _orig_dec is not None:
        def _timed_decode(*a, **k):
            _step("→ VAE DECODE start")
            _r = _orig_dec(*a, **k)
            _step("→ VAE DECODE fatto")
            return _r
        _vae.decode = _timed_decode
    try:
        out = pipe(**kw)
    except TypeError as _te:                                # pipeline senza callback_on_step_end
        if "callback" in str(_te):
            kw.pop("callback_on_step_end", None); kw.pop("callback_on_step_end_tensor_inputs", None)
            out = pipe(**kw)
        else:
            raise
    finally:
        if _vae is not None and _orig_dec is not None:
            _vae.decode = _orig_dec                         # ripristina il decode originale
        if _pulid_h:
            from core import pulid_adapter
            pulid_adapter.detach(_pulid_h)
    _step("GEN FATTA (denoise+decode completati)")
    preview_bus.set_message("Decodifica e salvataggio…")
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_{arch}_{params.get('seed', 0)}.png")
    res_img = out.images[0]
    # genero a dimensioni /16 (richiesto da Flux), ma ripristino le dimensioni ESATTE dell'input → output
    # combacia 1:1 con la foto/bbox (niente disallineamento in load_inpaint_result, aspetto identico).
    if needs_image and res_img.size != _orig_wh:
        res_img = res_img.resize(_orig_wh, Image.LANCZOS)
    # DITHER anti-BANDING — DEFAULT OFF (11/07): il rumore ±1.7 su OGNI immagine era il COLPEVOLE PRINCIPALE
    # delle "macchie chiaro/scure" sui fondi lisci (bisezione T14→T18: job path pulito senza dither, macchiato
    # con). Su fondi scuri uniformi si percepisce come sporco e, ridimensionato a schermo, si alias in chiazze.
    # Opt-in NATIVE_DITHER=1 per chi vede POSTERIZATION/banding sui gradienti (il problema per cui era nato).
    if os.environ.get("NATIVE_DITHER", "0") == "1":
        try:
            import numpy as _np
            _a = _np.asarray(res_img).astype(_np.float32) + _np.random.uniform(-1.7, 1.7, (res_img.height, res_img.width, 1))
            res_img = Image.fromarray(_np.clip(_a, 0, 255).astype("uint8"))
        except Exception:                                            # noqa: BLE001
            pass
    # INPAINT/OUTPAINT: FluxFill/inpaint rigenera TUTTA l'immagine (VAE encode→decode dell'intera foto) → le
    # zone FUORI dalla maschera prendono artefatti del VAE ("quadrettatura"/macchie sulle parti intatte).
    # Ricompone i PIXEL ORIGINALI dove la maschera è nera (feather sui bordi) → solo la regione mascherata è
    # rigenerata, il resto resta identico all'input. Env NATIVE_INPAINT_KEEP=0 per disattivare (rigenera tutto).
    _mask_pil = kw.get("mask_image")
    if arch in _MASK_ARCHS and _mask_pil is not None and params.get("image_path") \
            and os.environ.get("NATIVE_INPAINT_KEEP", "1") == "1":
        try:
            from PIL import ImageFilter
            _orig = Image.open(params["image_path"]).convert("RGB").resize(res_img.size, Image.LANCZOS)
            _m = _mask_pil.convert("L").resize(res_img.size, Image.NEAREST)   # RAW: bianco=genera, nero=originale
            # Feather DIFFERENZIATO per modalità (l'inpaint andava bene "prima", solo l'outpaint aveva la linea):
            #  • INPAINT → anello SOTTILE (auto ~min_lato/128, 4–14 px): niente degradazione attorno all'edit.
            #  • OUTPAINT → sfumatura AMPIA (auto ~min_lato/40, 12–48 px) + dilatazione + COLOR-MATCH al bordo.
            import numpy as _np
            _mk_bool = (_np.asarray(_m, dtype=_np.uint8) > 127)      # True = pixel da (ri)generare
            if params.get("outpaint"):
                _feather = int(params.get("feather") or 0) or max(12, min(48, int(round(min(res_img.size) / 40))))
                # COLOR-MATCH al bordo (outpaint): l'estensione ha tono/luminosità un filo diversi dalla foto →
                # linea di stacco. Riallinea al confine con falloff verso l'interno. Env NATIVE_OUTPAINT_COLORMATCH=0.
                if os.environ.get("NATIVE_OUTPAINT_COLORMATCH", "1") == "1":
                    res_img = _seam_tone_match(res_img, _orig, _mk_bool, _feather, gain=True, tau_mult=3.0, clamp=45)
                _fo = int(round(_feather * 1.9))                         # dilata la regione generata prima di sfumare
                _m = _m.filter(ImageFilter.MaxFilter(_fo * 2 + 1))       # → la sfumatura NON tocca il bordo tela
                _m = _m.filter(ImageFilter.GaussianBlur(_fo))
            else:
                _feather = int(params.get("feather") or 0) or max(4, min(14, int(round(min(res_img.size) / 128))))
                # COLOR-MATCH al bordo (INPAINT — prima MANCAVA): il generato ha spesso luminosità/saturazione/
                # contrasto un filo diversi dalla foto → LINEA di giunzione + "cambio di tono" visibili su aree
                # ampie/lisce. Seam-localizzato (falloff dal bordo) per NON ridipingere un eventuale oggetto
                # inserito lontano dal bordo. Env NATIVE_INPAINT_COLORMATCH=0 per disattivare.
                if os.environ.get("NATIVE_INPAINT_COLORMATCH", "1") == "1":
                    res_img = _seam_tone_match(res_img, _orig, _mk_bool, _feather, gain=True, tau_mult=6.0, clamp=32)
                _m = _m.filter(ImageFilter.GaussianBlur(_feather))
            res_img = Image.composite(res_img, _orig, _m)            # bianco=generato · nero=pixel originali
        except Exception as _e:                                      # noqa: BLE001
            print(f"[inpaint] ricomposizione originale fuori-maschera fallita ({_e})", flush=True)
    res_img.save(path)
    _step("IMMAGINE SALVATA su disco — FINE (dal click del bottone)")
    try:                                                     # #107 firma invisibile di provenienza (default ON, tutte le edizioni)
        from core import stego
        if stego.enabled():
            stego.sign_file(path)
    except Exception:                                        # noqa: BLE001
        pass
    try:                                                     # #106 watermark obbligatorio (gated feature_matrix)
        from core import watermark
        path = watermark.apply_to_file(path)                 # forced → fail-closed (rimuove se non applicabile)
    except RuntimeError:
        return {"status": "error", "error": "watermark obbligatorio non applicabile"}
    except Exception:                                        # noqa: BLE001
        pass
    return {"status": "success", "output": [path]}
