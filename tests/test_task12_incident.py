"""Task 12 — INCIDENTE del 17/10/2026 (crash di sistema, figlio immagini a ~90 GB di memoria virtuale).
Sequenza dal vivo (res_t12_L1new.json): crea FLUX_Dev (dopo il decode VRAM 32,1/32,5 GB, reserved 28,4 GB mai rilasciati)
→ inpaint stesso file: «from_pipe fallito (CUDA error: out of memory» → ripiego che «assembla» (RAM figlio 16,6 → 49 GB,
«est 63360MB») → contesto CUDA avvelenato: ogni job dopo «CUDA error: out of memory».

CAUSA RADICE (provata sul sorgente di diffusers 0.38, pipeline_utils.py:122-123): `DiffusionPipeline.from_pipe` ha
`torch_dtype=torch.float32` di DEFAULT e fa `new_pipeline.to(dtype=float32)` su TUTTI i componenti, che sono gli STESSI
oggetti della sorella → transformer fp8 e T5 upcastati IN PLACE (est 27 GB → 63 GB) → OOM → contesto avvelenato.

Questi test riproducono (a)-(e) con finti/mock, senza GPU:
  (a) dopo ogni generazione si rilascia la cache CUDA mantenendo i pesi;
  (b) la sorella è costruita con gli STESSI oggetti modulo, nessuna chiamata CUDA, nessun .to / cambio dtype, niente from_pipe;
  (c) se la sorella fallisce NON si assembla mai una seconda copia accanto alla prima: prima si scarica il residente;
  (d) un errore CUDA fatale non si inghiotte: il figlio si dichiara non sano ed esce, il server lo rigenera;
  (e) guardia prima di ogni assemblaggio: commit libero − carico < pavimento → errore chiaro.
"""
import inspect
import io
import json
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["LTX_SERVER_NO_TEE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# fixture e fake condivisi col test del task 12 (stessi finti: pipeline diffusers, file safetensors minuscoli)
from test_task12_residency import (  # noqa: E402,F401
    _FakeFluxPipe, _Mod, _no_cuda, _patch_load_env, _patch_run_env, _tiny_flux_file, flux, _FakeChild, server)


# ───────────────────────────── (a) cache CUDA rilasciata dopo ogni generazione ─────────────────────────────
def test_release_cuda_cache_calls_empty_cache_and_keeps_weights(monkeypatch):
    from inference_server.backends import image_flux
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    w = nn.Linear(4, 4)
    before = w.weight.data_ptr()
    image_flux._release_cuda_cache()
    assert calls == ["sync", "empty"]                                  # la cache torna al driver…
    assert w.weight.data_ptr() == before                               # …i pesi (tensori vivi) non si toccano


def test_release_cuda_cache_survives_a_poisoned_context(monkeypatch):
    from inference_server.backends import image_flux
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("CUDA error: out of memory")
    monkeypatch.setattr(torch.cuda, "synchronize", boom)
    image_flux._release_cuda_cache()                                   # non deve sollevare (il figlio uscirà da sé)


def _fake_run_env(flux, monkeypatch, tmp_path):
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    a = dm / "Flux" / "FLUX_Dev.safetensors"
    _tiny_flux_file(a)
    img, msk = _patch_run_env(monkeypatch, image_flux, tmp_path)
    loads = _patch_load_env(monkeypatch, image_flux, dm)
    monkeypatch.setattr(image_flux, "_measure_pipe_mb", lambda p: 100)
    return image_flux, ModelManager(), a, img, msk, loads


def test_run_releases_cache_after_success_and_after_error_weights_stay(flux, monkeypatch, tmp_path):
    image_flux, mm, a, img, msk, loads = _fake_run_env(flux, monkeypatch, tmp_path)
    released = []
    real = image_flux._release_cuda_cache
    monkeypatch.setattr(image_flux, "_release_cuda_cache", lambda: (released.append(1), real()))
    out = str(tmp_path / "out")
    r = image_flux.run({"arch": "flux", "model": str(a), "prompt": "x", "seed": 1, "steps": 2}, mm, out)
    assert r["status"] == "success" and released == [1]               # dopo la generazione: cache rilasciata
    assert any(k.startswith("flux:") for k in mm._cache)                # e il modello è RIMASTO residente
    monkeypatch.setattr(_FakeFluxPipe, "__call__", lambda self, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        image_flux.run({"arch": "flux", "model": str(a), "prompt": "y", "seed": 2, "steps": 2}, mm, out)
    assert released == [1, 1]                                          # anche sulla via d'errore
    assert any(k.startswith("flux:") for k in mm._cache)


# ───────────────────────────── (b) la sorella non alloca nulla: stessi oggetti modulo ─────────────────────────────
def test_diffusers_from_pipe_hazard_is_real_and_not_used():
    """Documenta la causa radice e impedisce che torni: from_pipe default float32 + .to(); il codice non lo usa."""
    from inference_server.backends import image_flux
    assert ".from_pipe(" not in inspect.getsource(image_flux)
    try:
        import diffusers
        src = inspect.getsource(diffusers.DiffusionPipeline.from_pipe)
    except Exception:                                                   # noqa: BLE001
        pytest.skip("diffusers non disponibile")
    assert 'kwargs.pop("torch_dtype", torch.float32)' in src and "new_pipeline.to(dtype=torch_dtype)" in src


def test_flux1_sibling_uses_same_module_objects_and_touches_no_gpu(flux, monkeypatch):
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    f = dm / "Flux" / "FLUX_Dev.safetensors"
    _tiny_flux_file(f)
    loads = _patch_load_env(monkeypatch, image_flux, dm)
    _FakeFluxPipe.from_pipe_calls = 0
    _Mod.moved = 0
    mm = ModelManager()
    p1 = mm.ensure(image_flux.register(mm, "Flux\\FLUX_Dev.safetensors", "flux"))
    p1.transformer = _Mod()                                             # modulo che esplode a ogni .to()
    p1.vae = _Mod()
    # dal momento della costruzione della sorella NESSUNA chiamata CUDA e nessun _set_attention (che usa CUDA)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (_ for _ in ()).throw(AssertionError("GPU toccata")))
    monkeypatch.setattr(image_flux, "_set_attention", lambda *a, **k: (_ for _ in ()).throw(AssertionError("CUDA")))
    dt = p1.transformer.weight.dtype
    p2 = mm.ensure(image_flux.register(mm, str(f), "inpaint"))
    assert len(loads) == 1 and _FakeFluxPipe.from_pipe_calls == 0
    for name in ("transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"):
        assert getattr(p2, name) is getattr(p1, name), name              # STESSI oggetti, nessuna copia
    assert _Mod.moved == 0 and p1.transformer.weight.dtype == dt        # nessun .to / nessun cambio di dtype
    assert p2._ls_component_keys == p1._ls_component_keys


def test_klein_sibling_uses_same_module_objects(flux, monkeypatch):
    import diffusers
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    f = dm / "Flux" / "klein.safetensors"
    _tiny_flux_file(f, in_ch=128)
    for n in ("Flux2KleinPipeline", "Flux2KleinInpaintPipeline"):
        monkeypatch.setattr(diffusers, n, type(n, (_FakeFluxPipe,), {}), raising=False)
    monkeypatch.setattr(image_flux, "_set_attention", lambda *a, **k: (_ for _ in ()).throw(AssertionError("CUDA")))
    mm = ModelManager()
    ident = image_flux._weights_ident("flux2_kontext", image_flux._resolve_model_path(str(f)))

    def first():
        p = _FakeFluxPipe(transformer=_Mod(), vae=_Mod(), text_encoder=_Mod(), tokenizer=object(), scheduler=object())
        p._ls_ident, p._ls_no_rebind, p._ls_place_mode = ident, True, "blockswap"
        return p, 24000
    mm.register("flux2_kontext:k", first)
    p1 = mm.ensure("flux2_kontext:k")
    _Mod.moved = 0
    p2 = mm.ensure(image_flux.register(mm, str(f), "flux2_inpaint"))
    assert p2 is not p1 and p2.transformer is p1.transformer and p2.text_encoder is p1.text_encoder
    assert p2.vae is p1.vae and _Mod.moved == 0
    assert len([1 for e in mm._cache.values() if getattr(e.obj, "_ls_ident", None) == ident]) == 2


# ───────────────────────────── (c) mai due copie: prima si scarica il residente ─────────────────────────────
def _tracking_assemble(monkeypatch, image_flux, mm, ident, stop=None):
    """Sostituisce _assemble_flux: registra, AL MOMENTO del load completo, quante pipeline della stessa identità e
    quanti componenti del vecchio transformer sono ancora vivi (devono essere 0: mai due copie)."""
    from inference_server import component_store as cs
    seen = {}

    def _assemble(Pipe, path, arch="flux"):
        seen["resident_same_ident"] = sum(1 for e in mm._cache.values() if getattr(e.obj, "_ls_ident", None) == ident)
        seen["old_components"] = [k for k in cs.STORE._items if k.startswith("t_")]
        p = _FakeFluxPipe()
        p._ls_shared = True
        p._ls_component_keys = ("t_new", "vae_flux")
        for k in p._ls_component_keys:
            cs.STORE.acquire(k, lambda: nn.Linear(2, 2))
        p.vae = nn.Linear(1, 1); p.text_encoder = nn.Linear(1, 1); p.text_encoder_2 = nn.Linear(1, 1)
        p.tokenizer = p.tokenizer_2 = p.scheduler = object()
        return p
    monkeypatch.setattr(image_flux, "_assemble_flux", _assemble)
    return seen


def test_failed_sibling_destroys_resident_before_full_load_never_two_copies(flux, monkeypatch):
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    f = dm / "Flux" / "FLUX_Dev.safetensors"
    _tiny_flux_file(f)
    _patch_load_env(monkeypatch, image_flux, dm)
    mm = ModelManager()
    p1 = mm.ensure(image_flux.register(mm, str(f), "flux"))
    ident = p1._ls_ident
    seen = _tracking_assemble(monkeypatch, image_flux, mm, ident)
    monkeypatch.setattr(_FakeFluxPipe, "fail_shared", RuntimeError("riuso non riuscito"), raising=False)
    orig = _FakeFluxPipe.__init__

    def init(self, transformer=None, **kw):                              # il costruttore CONDIVISO fallisce, quello vuoto no
        if transformer is not None and getattr(_FakeFluxPipe, "fail_shared", None) is not None:
            raise _FakeFluxPipe.fail_shared
        orig(self, transformer=transformer, **kw)
    monkeypatch.setattr(_FakeFluxPipe, "__init__", init)
    p2 = mm.ensure(image_flux.register(mm, str(f), "inpaint"))
    assert seen["resident_same_ident"] == 0                             # il residente era GIÀ scaricato al load completo
    assert seen["old_components"] == []                                 # e i suoi componenti distrutti (non solo rilasciati)
    assert p2 is not p1 and len(mm._cache) == 1                         # una sola copia viva


def test_klein_without_sibling_reuse_still_destroys_resident_first(flux, monkeypatch):
    """Anche col riuso disattivato (NATIVE_FLUX2_SIBLING=0) `evict_others(keep_ident)` lascia la sorella in cache: il load
    completo NON deve partire accanto ad essa (era il buco: 2× Klein in RAM)."""
    import diffusers
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    f = dm / "Flux" / "klein.safetensors"
    _tiny_flux_file(f, in_ch=128)
    for n in ("Flux2KleinPipeline", "Flux2KleinInpaintPipeline"):
        monkeypatch.setattr(diffusers, n, type(n, (_FakeFluxPipe,), {}), raising=False)
    monkeypatch.setenv("NATIVE_FLUX2_SIBLING", "0")
    mm = ModelManager()
    ident = image_flux._weights_ident("flux2_kontext", image_flux._resolve_model_path(str(f)))

    def first():
        p = _FakeFluxPipe(transformer=_Mod(), vae=_Mod(), text_encoder=_Mod())
        p._ls_ident, p._ls_no_rebind, p._ls_place_mode = ident, True, "blockswap"
        return p, 24000
    mm.register("flux2_kontext:k", first)
    mm.ensure("mm_dummy" if False else "flux2_kontext:k")
    seen = {}

    class _Stop(Exception):
        pass

    def assemble2(Pipe, path, te=None):
        seen["resident"] = [k for k in mm._cache]
        raise _Stop()
    monkeypatch.setattr(image_flux, "_assemble_flux2", assemble2)
    with pytest.raises(_Stop):
        mm.ensure(image_flux.register(mm, str(f), "flux2_inpaint"))
    assert seen["resident"] == []                                       # la copia vecchia è stata scaricata PRIMA


# ───────────────────────────── (d) CUDA fatale: non si inghiotte, il figlio esce ─────────────────────────────
def test_is_cuda_fatal_classification():
    from inference_server.cuda_health import is_cuda_fatal
    assert is_cuda_fatal(RuntimeError("CUDA error: out of memory"))
    assert is_cuda_fatal(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_cuda_fatal(RuntimeError("CUDA error: device-side assert triggered"))
    assert is_cuda_fatal(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert is_cuda_fatal(torch.OutOfMemoryError("allocator"))            # per CLASSE, qualunque sia il testo
    assert not is_cuda_fatal(MemoryError("Unable to allocate 9 GiB"))     # commit/RAM di sistema: il processo è sano
    assert not is_cuda_fatal(RuntimeError("LoRA non trovata su disco"))
    assert not is_cuda_fatal(OSError(1455, "Il file di paging è insufficiente"))


def test_cuda_oom_building_the_sibling_propagates_no_fallback_assembly(flux, monkeypatch):
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    f = dm / "Flux" / "FLUX_Dev.safetensors"
    _tiny_flux_file(f)
    loads = _patch_load_env(monkeypatch, image_flux, dm)
    mm = ModelManager()
    mm.ensure(image_flux.register(mm, str(f), "flux"))
    orig = _FakeFluxPipe.__init__

    def init(self, transformer=None, **kw):
        if transformer is not None:
            raise RuntimeError("CUDA error: out of memory")
        orig(self, transformer=transformer, **kw)
    monkeypatch.setattr(_FakeFluxPipe, "__init__", init)
    with pytest.raises(RuntimeError, match="CUDA error"):
        mm.ensure(image_flux.register(mm, str(f), "inpaint"))
    assert len(loads) == 1                                              # NESSUN assemblaggio di ripiego dopo l'OOM


def test_cuda_fatal_in_autoevict_is_not_swallowed_but_plain_errors_are(flux, monkeypatch, tmp_path):
    from inference_server import component_store as cs
    image_flux, mm, a, img, msk, loads = _fake_run_env(flux, monkeypatch, tmp_path)
    out = str(tmp_path / "out")
    params = {"arch": "flux", "model": str(a), "prompt": "x", "seed": 1, "steps": 2}

    def fatal(*a, **k):
        raise RuntimeError("CUDA error: out of memory")
    monkeypatch.setattr(cs.STORE, "resident_of", fatal)
    with pytest.raises(RuntimeError, match="CUDA error"):                # era qui che l'OOM veniva inghiottito (17/10)
        image_flux.run(params, mm, out)

    def plain(*a, **k):
        raise RuntimeError("errore qualsiasi")
    monkeypatch.setattr(cs.STORE, "resident_of", plain)
    assert image_flux.run(dict(params, seed=2), mm, out)["status"] == "success"   # non fatale: ripiego come prima


class _Exited(BaseException):
    def __init__(self, code):
        self.code = code


def _run_runner(monkeypatch, tmp_path, handler, cmds):
    """Esegue persistent_runner.main() in-process con un handler finto e stdin scriptato."""
    import core.st_direct as sd
    from inference_server import persistent_runner as pr
    from inference_server.backends import image_flux
    monkeypatch.setattr(sd, "patch_safetensors_torch", lambda: False)
    monkeypatch.setattr(image_flux, "run", handler)
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(_Exited(code)))
    pf = tmp_path / "job.json"
    pf.write_text(json.dumps({"module": "image", "params": {"arch": "flux", "model": "x.safetensors"}}))
    rf = tmp_path / "res.json"
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(c.format(pf=pf, rf=rf) + "\n" for c in cmds)))
    ipc = io.StringIO()
    monkeypatch.setattr(sys, "stdout", ipc)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    code = None
    try:
        pr.main()
    except _Exited as e:
        code = e.code
    return ipc.getvalue().splitlines(), code, rf


def test_runner_declares_itself_unhealthy_and_exits_on_cuda_oom(monkeypatch, tmp_path):
    def handler(params, mm, out):
        raise RuntimeError("CUDA error: out of memory")
    lines, code, rf = _run_runner(monkeypatch, tmp_path, handler, ["RUN {pf} {rf}", "RUN {pf} {rf}"])
    assert code == 86                                                    # il figlio ESCE (non resta avvelenato)
    assert lines == ["ERR cuda_fatal"]                                   # e non serve nessun job successivo
    res = json.loads(rf.read_text())
    assert res["status"] == "error" and "CUDA" in res["error"] and "riavviato" in res["error"]


def test_runner_survives_ordinary_errors_and_reports_them(monkeypatch, tmp_path):
    from core.mem_budget import LoadRefused

    def handler(params, mm, out):
        raise LoadRefused("Caricamento di x rifiutato: commit insufficiente")
    lines, code, rf = _run_runner(monkeypatch, tmp_path, handler, ["RUN {pf} {rf}", "QUIT"])
    assert code is None and len(lines) == 1 and lines[0].startswith("DONE")   # errore non CUDA: il figlio resta vivo (task 9: _run_handler → risultato d'errore)
    assert "rifiutato" in json.loads(rf.read_text())["error"]            # risultato leggibile, non «nessun risultato»


def test_server_kills_poisoned_child_and_next_job_gets_a_fresh_one(server):
    f = _FakeChild(server)
    f.run("image", {"arch": "flux", "model": "Flux\\FLUX_Dev.safetensors"})
    assert f.spawns == 1
    orig = f.c._send
    f.c._send = lambda line, wait_for, timeout: "ERR cuda_fatal" if line.startswith("RUN") else orig(line, wait_for, timeout)
    f.run("image", {"arch": "flux", "model": "Flux\\FLUX_Dev.safetensors"})
    assert f.kills == 1 and f.c.model_key == "-"                          # il figlio avvelenato viene tolto di mezzo
    f.c._send = orig
    f.run("image", {"arch": "flux", "model": "Flux\\FLUX_Dev.safetensors"})
    assert f.spawns == 2                                                  # nuovo processo pulito al job dopo


# ───────────────────────────── (e) guardia di caricamento ─────────────────────────────
def test_check_load_refuses_below_commit_floor(monkeypatch):
    from core import mem_budget
    monkeypatch.setattr(mem_budget, "avail_commit_mb", lambda: 30000.0)
    monkeypatch.setattr(mem_budget, "commit_floor_mb", lambda: 20480.0)
    mem_budget.check_load(0, "x")                                         # niente da caricare: ok
    mem_budget.check_load(9000, "x")                                      # 30000 − 9000 = 21000 ≥ 20480: ok
    with pytest.raises(mem_budget.LoadRefused, match="rifiutato"):
        mem_budget.check_load(10000, "FLUX_Dev.safetensors")              # 20000 < 20480 → rifiuta, errore chiaro


def test_check_load_floor_is_configurable(monkeypatch):
    from core import mem_budget
    monkeypatch.setattr(mem_budget, "avail_commit_mb", lambda: 5000.0)
    monkeypatch.setenv("LS_COMMIT_FLOOR_MB", "1024")                      # soglia configurabile (core/mem_budget)
    mem_budget.check_load(3000, "x")                                      # 5000 − 3000 = 2000 ≥ 1024
    with pytest.raises(mem_budget.LoadRefused):
        mem_budget.check_load(4500, "x")


def test_load_is_refused_before_any_assembly(flux, monkeypatch):
    image_flux, dm = flux
    from core import mem_budget
    from inference_server.model_manager import ModelManager
    f = dm / "Flux" / "FLUX_Dev.safetensors"
    _tiny_flux_file(f)
    loads = _patch_load_env(monkeypatch, image_flux, dm)
    monkeypatch.setattr(mem_budget, "avail_commit_mb", lambda: 21000.0)   # file finto minuscolo + T5 stimato 9,5 GB → < pavimento
    mm = ModelManager()
    with pytest.raises(mem_budget.LoadRefused):
        mm.ensure(image_flux.register(mm, str(f), "flux"))
    assert loads == []                                                    # rifiutato PRIMA di assemblare/caricare


def test_predicted_load_is_zero_when_everything_is_already_resident(flux):
    image_flux, dm = flux
    from inference_server import component_store as cs
    f = dm / "Flux" / "FLUX_Dev.safetensors"
    f.write_bytes(b"x" * (3 * 2**20))
    assert image_flux._predicted_load_mb(str(f), "flux") > 9000            # transformer + T5/CLIP/VAE da caricare
    key = f"flux_tr:{os.path.normpath(str(f)).lower()}:flux_transformer"
    for k in (key, "t5xxl", "clip_l", "vae_flux"):
        cs.STORE._items[k] = cs._CEntry(obj=nn.Linear(1, 1), size_mb=1.0)
    assert image_flux._predicted_load_mb(str(f), "flux") == 0.0            # già residente: nessun carico previsto


# ───────── L1 dal vivo (18/10): sorella con pesi già su GPU + «No context is set» (FBCache sul transformer condiviso) ─────────
def _resident_store(keys, device="cuda"):
    from inference_server import component_store as cs
    for k in keys:
        cs.STORE._items[k] = cs._CEntry(obj=nn.Linear(1, 1), size_mb=100.0, device=device)
    return cs.STORE


class _OffloadForbidden(_FakeFluxPipe):
    def enable_model_cpu_offload(self, *a, **k):
        raise AssertionError("offload sui moduli condivisi: vietato")

    def enable_sequential_cpu_offload(self, *a, **k):
        raise AssertionError("offload sui moduli condivisi: vietato")

    def to(self, *a, **k):
        raise AssertionError("nessuno spostamento di moduli già residenti")


def _cuda_free(monkeypatch, free_mb):
    """Finge una GPU con `free_mb` liberi (e conta empty_cache)."""
    calls = {"empty": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(free_mb * 2**20), 32 * 1024 * 2**20))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.__setitem__("empty", calls["empty"] + 1))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    return calls


def _shared_pipe(keys):
    p = _OffloadForbidden()
    p._ls_shared, p._ls_component_keys, p._ls_working_mb, p._ls_est_mb = True, keys, 3200, 27123
    return p


KEYS = ("tr", "vae_flux", "clip_l", "t5xxl")


def test_sibling_with_resident_weights_stays_gpu_when_working_memory_fits(flux, monkeypatch):
    """Il caso L1: pesi=0, working 3,2 GB, liberi 4,2 GB → 'gpu' (prima: need 4400 > 4213 → OFFLOAD, 28→1,7 GB di VRAM)."""
    image_flux, dm = flux
    st = _resident_store(KEYS)
    _cuda_free(monkeypatch, 4213)
    p = _shared_pipe(KEYS)
    assert image_flux._ensure_placement(p, None, None) == "gpu"
    assert p._ls_place_mode == "gpu" and all(st._items[k].device == "cuda" for k in KEYS)   # niente è stato spostato


def test_sibling_with_resident_weights_never_offloads_clear_error_when_short(flux, monkeypatch):
    image_flux, dm = flux
    st = _resident_store(KEYS)
    calls = _cuda_free(monkeypatch, 2000)                                 # meno della memoria di lavoro
    p = _shared_pipe(KEYS)
    with pytest.raises(RuntimeError, match="VRAM libera insufficiente") as ei:
        image_flux._ensure_placement(p, None, None)
    from inference_server.cuda_health import is_cuda_fatal
    assert not is_cuda_fatal(ei.value)                                    # errore pulito, il figlio resta sano
    assert calls["empty"] == 1                                            # prima ha restituito la cache al driver
    assert all(st._items[k].device == "cuda" for k in KEYS)               # e i pesi sono ancora tutti su GPU


def test_sibling_short_on_vram_old_path_only_with_explicit_override(flux, monkeypatch):
    image_flux, dm = flux
    _resident_store(KEYS)
    _cuda_free(monkeypatch, 2000)
    monkeypatch.setenv("NATIVE_ALLOW_OFFLOAD_SHARED", "1")
    monkeypatch.setattr(image_flux, "_place", lambda pipe, mb, resident_mb=None: "offload")
    assert image_flux._ensure_placement(_shared_pipe(KEYS), None, None) == "offload"   # solo a richiesta esplicita


def test_not_fully_resident_pipeline_keeps_the_historical_placement(flux, monkeypatch):
    image_flux, dm = flux
    st = _resident_store(KEYS)
    st._items["t5xxl"].device = "cpu"                                     # T5 parcheggiato in RAM: da riportare su
    _cuda_free(monkeypatch, 30000)
    seen = []
    monkeypatch.setattr(image_flux, "_place", lambda pipe, mb, resident_mb=None: seen.append(mb) or "gpu")
    assert image_flux._ensure_placement(_shared_pipe(KEYS), None, None) == "gpu" and seen


class _CacheTr(nn.Linear):
    """Transformer finto con l'API di cache di diffusers (CacheMixin): is_cache_enabled / enable_cache / disable_cache."""
    def __init__(self):
        super().__init__(2, 2)
        self._cache_config = None

    @property
    def is_cache_enabled(self):
        return self._cache_config is not None

    def enable_cache(self, cfg):
        self._cache_config = cfg

    def disable_cache(self):
        self._cache_config = None


def test_sync_fbcache_matches_the_job(monkeypatch):
    from inference_server.backends import image_flux
    p = types.SimpleNamespace(transformer=_CacheTr())
    monkeypatch.setenv("NATIVE_FBCACHE", "1")
    enabled = []
    monkeypatch.setattr(image_flux, "_apply_fbcache", lambda pipe: (enabled.append(1), pipe.transformer.enable_cache("x")))
    image_flux._sync_fbcache(p, "flux")                                   # t2i con FBCache=1 → accesa
    assert p.transformer.is_cache_enabled and enabled == [1]
    for arch in ("inpaint", "kontext"):                                   # image-conditioned → SPENTA (non entrano nel cache_context)
        p.transformer.enable_cache("x")
        image_flux._sync_fbcache(p, arch)
        assert not p.transformer.is_cache_enabled
    monkeypatch.setenv("NATIVE_FBCACHE", "0")
    p.transformer.enable_cache("x")
    image_flux._sync_fbcache(p, "flux")                                   # FBCache spenta da env → spenta anche sul t2i
    assert not p.transformer.is_cache_enabled


def test_create_with_fbcache_then_inpaint_on_shared_transformer_no_context_error(flux, monkeypatch, tmp_path):
    """RIPRODUCE «No context is set» (diffusers hooks.py:44): crea (FluxPipeline, FBCache accesa dal launcher) → inpaint
    sullo STESSO transformer condiviso. La pipeline di inpaint NON entra nel cache_context → senza _sync_fbcache muore."""
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    from PIL import Image
    a = dm / "Flux" / "FLUX_Dev.safetensors"
    _tiny_flux_file(a)
    img, msk = _patch_run_env(monkeypatch, image_flux, tmp_path)
    monkeypatch.setenv("NATIVE_FBCACHE", "1")
    shared_tr = _CacheTr()

    class _T2I(_FakeFluxPipe):
        def __call__(self, **kw):
            assert self.transformer.is_cache_enabled                       # il t2i la usa
            return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    class _Inp(_FakeFluxPipe):
        def __call__(self, **kw):
            if self.transformer.is_cache_enabled:
                raise ValueError("No context is set. Please set a context before retrieving the state.")
            return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    def fake_load(model_path, arch, mm=None, klein_te=""):
        p = (_T2I if arch == "flux" else _Inp)(transformer=shared_tr)
        p._ls_shared = False
        p._ls_ident = image_flux._weights_ident(arch, image_flux._resolve_model_path(model_path))
        return p, 12000
    monkeypatch.setattr(image_flux, "_load", fake_load)
    mm = ModelManager()
    out = str(tmp_path / "out")
    assert image_flux.run({"arch": "flux", "model": str(a), "prompt": "x", "seed": 1, "steps": 2}, mm, out)["status"] == "success"
    assert shared_tr.is_cache_enabled                                       # lasciata accesa dal crea
    r = image_flux.run({"arch": "inpaint", "model": str(a), "prompt": "y", "seed": 2, "steps": 2,
                        "image_path": img, "mask_path": msk}, mm, out)
    assert r["status"] == "success" and not shared_tr.is_cache_enabled      # spenta per l'inpaint
    assert image_flux.run({"arch": "flux", "model": str(a), "prompt": "z", "seed": 3, "steps": 2}, mm, out)["status"] == "success"
    assert shared_tr.is_cache_enabled                                       # e riaccesa al crea successivo


def test_cache_release_happens_at_the_end_of_the_job_and_is_logged(flux, monkeypatch, tmp_path, capsys):
    from PIL import Image
    image_flux, mm, a, img, msk, loads = _fake_run_env(flux, monkeypatch, tmp_path)
    events = []

    def gen(self, **kw):
        events.append("gen")
        return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])
    monkeypatch.setattr(_FakeFluxPipe, "__call__", gen)
    real = image_flux._release_cuda_cache
    monkeypatch.setattr(image_flux, "_release_cuda_cache", lambda: (events.append("release"), real()))
    image_flux.run({"arch": "flux", "model": str(a), "prompt": "x", "seed": 1, "steps": 2}, mm, str(tmp_path / "out"))
    assert events == ["gen", "release"]                                     # il rilascio è DOPO la generazione, a fine job
    _cuda_free(monkeypatch, 5000)                                           # e lascia traccia nel log: reserved prima → dopo
    reserved = iter([28388 * 2**20, 24400 * 2**20])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: next(reserved))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 23989 * 2**20)
    capsys.readouterr()
    real()
    out = capsys.readouterr().out
    assert "FINE JOB" in out and "28388MB" in out and "24400MB" in out
