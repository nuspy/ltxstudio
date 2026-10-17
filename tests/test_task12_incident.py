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
    assert code is None and lines == ["ERR eccezione (vedi log)"]        # errore non CUDA: il figlio resta vivo
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
