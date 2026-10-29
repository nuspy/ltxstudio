"""Task 12 — gestione memoria immagini: stesso modello fra «crea» e «modifica», residenza dopo la generazione,
eviction solo su richiesta. NESSUNA GPU: pipeline e pesi finti (CUDA nascosta), nessun download, nessun modello vero.

Cosa si prova (e dove era il difetto, vedi docs/ nel report del task 12):
  1. chiave di residenza = QUALI PESI (core.model_identity), non il compito (flux/inpaint/kontext);
  2. server: crea→modifica con lo stesso file NON uccide il figlio persistente; pesi diversi sì; ausiliari
     (segment/upscale) non toccano il figlio e, se manca VRAM, lo PARCHEGGIANO invece di ucciderlo;
  3. gpu_registry: stessa identità = reuse (nessun /free); pesi diversi o altra categoria = evict su richiesta;
  4. image_flux: pre-evict per identità (le sorelle restano), from_pipe/sorella Klein = zero load, path relativo
     (crea) e assoluto (modifica) dello stesso file risolvono alla stessa identità;
  5. run() reale con pipeline finta: dopo la generazione il modello RESTA in cache; il cambio pesi lo sfratta.
"""
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # i test non devono mai creare un contesto CUDA
os.environ["LTX_SERVER_NO_TEE"] = "1"                   # l'import di server.py non deve dirottare stdout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REL = "Flux\\FLUX_Dev.safetensors"                      # come lo manda il tab «crea»
ABS = r"E:\AI\Models\diffusion_models\Flux\FLUX_Dev.safetensors"   # come lo manda «modifica/inpaint»
OTHER = r"E:\AI\Models\diffusion_models\Flux\FLUX_Krea_Dev.safetensors"


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from inference_server import component_store as cs
    cs.STORE._items.clear()                              # lo store è globale: ogni test parte pulito


# ───────────────────────────── 1. identità ─────────────────────────────
def test_residency_key_is_weights_not_task():
    from core.model_identity import residency_key
    k = residency_key("flux", REL)
    assert k == "flux1:flux_dev.safetensors"
    assert residency_key("inpaint", ABS) == k            # modifica con maschera, path assoluto
    assert residency_key("kontext", ABS) == k            # modifica per istruzione
    assert residency_key("inpaint", OTHER) != k          # PESI diversi = chiave diversa
    assert residency_key("flux2", ABS) != k              # famiglia diversa
    assert residency_key("flux2_kontext", ABS) == residency_key("flux2_inpaint", ABS)   # Klein: stesso file
    assert residency_key("flux2_kontext", ABS) != residency_key("flux2", ABS)
    assert residency_key("zimage", r"E:\x\z.safetensors") == "zimage:z.safetensors"     # invariato
    assert residency_key("upscale", "4x.pth") == "upscale:4x.pth"


def test_residency_key_rollback_env(monkeypatch):
    from core.model_identity import residency_key
    monkeypatch.setenv("NATIVE_RESIDENCY_BY_TASK", "1")
    assert residency_key("flux", REL) == "flux:flux_dev.safetensors"
    assert residency_key("inpaint", ABS) == "inpaint:flux_dev.safetensors"


# ───────────────────────────── 2. server ─────────────────────────────
@pytest.fixture
def server():
    import importlib
    return importlib.import_module("inference_server.server")


class _FakeChild:
    """_PersistentChild senza processi: conta spawn/kill/FREE_DEMOTE."""
    def __init__(self, server_mod):
        self.c = server_mod._PersistentChild()
        self.spawns = 0
        self.kills = 0
        self.sent = []
        self._up = False
        c = self.c

        def _spawn():
            self.spawns += 1
            self._up = True
            c.model_key = "-"

        def _kill():
            if self._up:                                 # conta solo i kill di un figlio VIVO (il primo è a vuoto)
                self.kills += 1
            self._up = False

        def _send(line, wait_for, timeout):
            self.sent.append(line)
            if line.startswith("RUN"):
                return f"DONE {self.next_key}"
            if line == "FREE_DEMOTE":
                return "FREED"
            return "STATS_OK"
        c._spawn, c._kill, c._send = _spawn, _kill, _send
        c._alive = lambda: self._up
        self.next_key = "-"

    def run(self, module, params):
        self.next_key = __import__("inference_server.server", fromlist=["x"])._job_model_key(module, params)
        self.c.run(module, params, "p.json", "r.json")


def test_server_create_then_edit_same_file_does_not_kill_child(server):
    f = _FakeChild(server)
    f.run("image", {"arch": "flux", "model": REL})                       # crea
    assert f.spawns == 1 and f.c.model_key == "flux1:flux_dev.safetensors"
    f.run("image", {"arch": "inpaint", "model": ABS})                    # modifica, stesso file
    f.run("image", {"arch": "kontext", "model": ABS})
    f.run("image", {"arch": "flux", "model": REL})                       # e di nuovo crea
    assert f.spawns == 1 and f.kills == 0                                # il modello è rimasto residente


def test_server_different_weights_still_respawn_by_default(server):
    f = _FakeChild(server)
    f.run("image", {"arch": "flux", "model": REL})
    f.run("image", {"arch": "flux", "model": OTHER})                     # cambio di PESI: comportamento storico
    assert f.spawns == 2 and f.kills == 1


def test_server_respawn_never_keeps_child_across_weights(server, monkeypatch):
    monkeypatch.setattr(server, "_RESPAWN_ON_CHANGE", False)             # NATIVE_CHILD_RESPAWN=never
    f = _FakeChild(server)
    f.run("image", {"arch": "flux", "model": REL})
    f.run("image", {"arch": "flux", "model": OTHER})
    assert f.spawns == 1 and f.kills == 0


def test_server_model_stays_resident_after_generation(server):
    f = _FakeChild(server)
    f.run("image", {"arch": "flux", "model": REL})
    assert f.kills == 0 and f._up                                        # a fine job nessuno scarica: resta caldo
    assert not any(s == "FREE_DEMOTE" for s in f.sent)


def test_server_demote_keeps_child_alive_destroy_kills(server):
    f = _FakeChild(server)
    f.run("image", {"arch": "flux", "model": REL})
    assert f.c.free(demote=True) is True                                 # eviction su richiesta: parcheggia
    assert f.kills == 0 and "FREE_DEMOTE" in f.sent
    assert f.c.free(demote=False) is True                                # destroy (cambio categoria hard): kill
    assert f.kills == 1


def test_server_helpers_are_isolated_and_do_not_touch_the_child(server, monkeypatch):
    monkeypatch.setattr(server, "_child", _FakeChild(server).c)
    monkeypatch.setattr(server, "_PERSIST", True)
    assert server._use_persistent_child("image") is True
    for m in ("segment", "upscale", "video_upscale", "video_face", "video_fps"):
        assert server._use_persistent_child(m) is False                  # processo isolato: figlio intatto
    monkeypatch.setattr(server, "_HELPERS_ISOLATED", False)              # rollback NATIVE_HELPERS_ISOLATED=0
    assert server._use_persistent_child("segment") is True


def test_server_helper_parks_resident_model_only_when_vram_short(server, monkeypatch):
    f = _FakeChild(server)
    f.run("image", {"arch": "flux", "model": REL})
    monkeypatch.setattr(server, "_child", f.c)
    import core.gpu_mem as gm
    monkeypatch.setattr(gm, "wait_free_vram_mb", lambda *a, **k: 99999)
    assert server._prepare_helper_vram("segment", free_mb=20000) is False   # c'è spazio: non si tocca nulla
    assert f.kills == 0 and "FREE_DEMOTE" not in f.sent
    assert server._prepare_helper_vram("segment", free_mb=2500) is True     # manca VRAM: PARCHEGGIA, non uccide
    assert f.kills == 0 and "FREE_DEMOTE" in f.sent


# ───────────────────────────── 3. gpu_registry ─────────────────────────────
def _reg(monkeypatch, resident_key, vram=24000, free=6000, busy=False):
    from core import gpu_registry, gpu_mem
    monkeypatch.setattr(gpu_mem, "free_vram_mb", lambda: free)
    monkeypatch.setattr(gpu_registry, "_stats", lambda port, timeout=2.0: (
        {"model_key": resident_key, "model_vram_mb": vram, "busy": busy, "components": []}
        if port == 8189 else None))
    return gpu_registry


def test_registry_reuse_across_create_and_edit(monkeypatch):
    from inference_server.persistent_runner import _model_key
    g = _reg(monkeypatch, _model_key("image", {"arch": "flux", "model": REL}))     # il figlio riporta questo
    p = g.plan(g.ModelKey(arch="inpaint", model=ABS), category="image", need_mb=24000)
    assert p["action"] == "reuse" and p["free_ports"] == []                         # nessun /free, nessun reload
    p = g.plan(g.ModelKey(arch="kontext", model=ABS), category="image", need_mb=24000)
    assert p["action"] == "reuse"


def test_registry_evicts_only_on_demand(monkeypatch):
    from inference_server.persistent_runner import _model_key
    g = _reg(monkeypatch, _model_key("image", {"arch": "flux", "model": REL}))
    # pesi DIVERSI e VRAM insufficiente → sfratto del residente (su richiesta)
    p = g.plan(g.ModelKey(arch="flux", model=OTHER), category="image", need_mb=24000)
    assert p["action"] == "evict_load" and p["free_ports"] == [8189]
    # pesi diversi ma c'è spazio → NIENTE sfratto (il residente resta finché non serve la sua VRAM)
    g = _reg(monkeypatch, _model_key("image", {"arch": "flux", "model": REL}), free=30000)
    p = g.plan(g.ModelKey(arch="flux", model=OTHER), category="image", need_mb=8000)
    assert p["action"] == "load" and p["free_ports"] == []
    # altra categoria (video) → il residente immagini va liberato
    g = _reg(monkeypatch, _model_key("image", {"arch": "flux", "model": REL}))
    p = g.plan(g.ModelKey(arch="ltx", model="ltx.safetensors"), category="video", need_mb=20000)
    assert p["free_ports"] == [8189]


def test_registry_before_job_reuse_sends_no_free(monkeypatch):
    from inference_server.persistent_runner import _model_key
    g = _reg(monkeypatch, _model_key("image", {"arch": "flux", "model": REL}))
    sent = []
    monkeypatch.setattr(g.mem_coordinator, "_free_ports", lambda ports, t, mode="demote": sent.append(ports) or {})
    r = g.before_job("image", arch="inpaint", model=ABS, need_mb=24000)
    assert r["action"] == "reuse" and sent == []


# ───────────────────────────── 4. image_flux: identità, sorelle, evict ─────────────────────────────
@pytest.fixture
def flux(tmp_path, monkeypatch):
    from inference_server.backends import image_flux
    dm = tmp_path / "diffusion_models"
    (dm / "Flux").mkdir(parents=True)
    (dm / "Ernie").mkdir()
    monkeypatch.setattr(image_flux, "ERNIE_TRANSFORMER", str(dm / "Ernie" / "x.safetensors"))
    return image_flux, dm


def test_relative_and_absolute_path_resolve_to_the_same_identity(flux):
    image_flux, dm = flux
    f = dm / "Flux" / "FLUX_Dev.safetensors"
    f.write_bytes(b"x")
    rel = image_flux._resolve_model_path("Flux\\FLUX_Dev.safetensors")
    ab = image_flux._resolve_model_path(str(f))
    assert os.path.normcase(rel) == os.path.normcase(ab) == os.path.normcase(str(f))
    i_crea = image_flux._weights_ident("flux", rel)
    i_edit = image_flux._weights_ident("inpaint", ab)
    i_kont = image_flux._weights_ident("kontext", ab)
    assert i_crea == i_edit == i_kont
    assert image_flux._weights_ident("flux2", ab) != i_crea
    (dm / "Flux" / "Other.safetensors").write_bytes(b"y")
    assert image_flux._weights_ident("flux", str(dm / "Flux" / "Other.safetensors")) != i_crea
    assert image_flux._weights_ident("flux2_kontext", ab, "te1") != image_flux._weights_ident("flux2_kontext", ab, "")


class _Shell:
    def __init__(self, ident, shared=True, no_rebind=False, mode=""):
        self._ls_ident = ident
        self._ls_shared = shared
        self._ls_no_rebind = no_rebind
        self._ls_place_mode = mode


def test_sibling_lookup_is_by_identity(flux):
    from inference_server.model_manager import ModelManager
    image_flux, _ = flux
    mm = ModelManager()
    ident = ("flux1", "c:\\x\\flux_dev.safetensors", "")
    mm.register("flux:a", lambda: (_Shell(ident), 50))
    mm.register("flux:b", lambda: (_Shell(("flux1", "c:\\x\\krea.safetensors", "")), 50))
    mm.ensure("flux:a"); mm.ensure("flux:b")
    assert image_flux._sibling_pipe(mm, ident) is mm._cache["flux:a"].obj
    assert image_flux._sibling_pipe(mm, ("flux1", "c:\\x\\nope", "")) is None
    # Klein: solo con placement blockswap già fatto
    kid = ("flux2k", "c:\\x\\klein.safetensors", "")
    mm.register("flux2_kontext:k", lambda: (_Shell(kid, shared=False, no_rebind=True, mode="blockswap"), 24000))
    mm.ensure("flux2_kontext:k")
    assert image_flux._flux2_sibling_pipe(mm, kid) is mm._cache["flux2_kontext:k"].obj
    mm._cache["flux2_kontext:k"].obj._ls_place_mode = "offload"
    assert image_flux._flux2_sibling_pipe(mm, kid) is None


def test_evict_others_keeps_siblings_evicts_other_weights():
    from inference_server.model_manager import ModelManager
    mm = ModelManager()
    ident = ("flux1", "f", "")
    mm.register("flux:a", lambda: (_Shell(ident), 50))
    mm.register("zimage:z", lambda: (_Shell(("zimage", "z", ""), shared=False), 12000))
    mm.register("inpaint:a", lambda: (_Shell(ident), 50))
    for k in ("flux:a", "zimage:z"):
        mm.ensure(k)
    n = mm.evict_others("inpaint:a", keep_ident=ident)
    assert n == 1 and "flux:a" in mm._cache and "zimage:z" not in mm._cache      # la sorella resta
    # senza keep_ident = comportamento storico (sfratta tutto tranne la chiave)
    mm.register("kontext:a", lambda: (_Shell(ident), 50))
    assert mm.evict_others("kontext:a") == 1 and "flux:a" not in mm._cache


def test_make_room_evicts_lru_only_when_space_is_needed(monkeypatch):
    from inference_server.model_manager import ModelManager
    mm = ModelManager()
    mm.register("a", lambda: (_Shell(("x", "a", ""), shared=False), 12000))
    mm.register("b", lambda: (_Shell(("x", "b", ""), shared=False), 12000))
    mm.ensure("a"); mm.ensure("b")
    # VRAM libera simulata: 5000 + 12000 per ogni shell scaricata (la VRAM torna davvero libera)
    monkeypatch.setattr(mm, "_vram_total_free_mb", lambda: (32000.0, 5000.0 + 12000.0 * (2 - len(mm._cache))))
    mm.make_room(4000)                                   # c'è spazio (5000 liberi): nessuno viene scaricato
    assert set(mm._cache) == {"a", "b"}
    mm.make_room(9000)                                   # serve spazio → il meno recente (a) esce, b resta
    assert "a" not in mm._cache and "b" in mm._cache


# ───── 4b. _load REALE: crea→modifica = nessun nuovo load (from_pipe), con diffusers/pesi finti ─────
def _tiny_flux_file(path, in_ch=64):
    from safetensors.torch import save_file
    save_file({"x_embedder.weight": torch.zeros(4, in_ch)}, str(path))


class _Mod(nn.Linear):
    """Modulo finto che NON tollera spostamenti/cambi di dtype: la sorella deve riusarlo così com'è."""
    moved = 0

    def __init__(self):
        super().__init__(2, 2)
        self.to_calls = 0

    def to(self, *a, **k):
        self.to_calls += 1
        _Mod.moved += 1
        raise AssertionError("la sorella non deve spostare/convertire i moduli condivisi (.to)")


class _FakeFluxPipe:
    """Finta pipeline diffusers. `from_pipe` è VIETATO (in diffusers 0.38 fa .to(dtype=float32) sui componenti
    condivisi: incidente 17/10); si costruisce con gli STESSI oggetti modulo, come fa _assemble_flux."""
    from_pipe_calls = 0
    ctor_calls = 0
    fail_ctor = None                                     # eccezione da sollevare nel costruttore (test)

    def __init__(self, transformer=None, vae=None, text_encoder=None, tokenizer=None, text_encoder_2=None,
                 tokenizer_2=None, scheduler=None, **kw):
        if _FakeFluxPipe.fail_ctor is not None:
            raise _FakeFluxPipe.fail_ctor
        _FakeFluxPipe.ctor_calls += 1
        self.transformer = transformer if transformer is not None else _Mod()
        self.vae, self.text_encoder, self.tokenizer = vae, text_encoder, tokenizer
        self.text_encoder_2, self.tokenizer_2, self.scheduler = text_encoder_2, tokenizer_2, scheduler
        self._ls_component_keys = ()

    @classmethod
    def from_pipe(cls, src, **kw):
        _FakeFluxPipe.from_pipe_calls += 1
        raise AssertionError("from_pipe vietato: upcasta i componenti condivisi a float32 (OOM, 17/10)")

    def remove_all_hooks(self):
        pass

    def unload_lora_weights(self):
        pass

    def __call__(self, **kw):
        from PIL import Image
        return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])


def _patch_load_env(monkeypatch, image_flux, dm):
    import diffusers
    from inference_server import component_store as cs
    loads = []

    def _assemble(Pipe, path, arch="flux"):
        loads.append((arch, os.path.basename(path)))
        p = _FakeFluxPipe()
        p._ls_shared = True
        p._ls_component_keys = ("t_" + os.path.basename(path), "vae_flux")
        for k in p._ls_component_keys:
            cs.STORE.acquire(k, lambda: nn.Linear(2, 2))
        p.vae = nn.Linear(1, 1); p.text_encoder = nn.Linear(1, 1); p.text_encoder_2 = nn.Linear(1, 1)
        p.tokenizer = object(); p.tokenizer_2 = object(); p.scheduler = object()
        return p

    monkeypatch.setattr(image_flux, "_assemble_flux", _assemble)
    for n in ("FluxPipeline", "FluxInpaintPipeline", "FluxKontextPipeline", "FluxFillPipeline"):
        monkeypatch.setattr(diffusers, n, type(n, (_FakeFluxPipe,), {}), raising=False)
    monkeypatch.setattr(image_flux, "_set_attention", lambda pipe, arch="": "sdpa")
    monkeypatch.setattr(image_flux, "_vram_cache_put", lambda *a, **k: None)
    monkeypatch.setattr(image_flux, "_vram_cache_get", lambda *a, **k: 0)
    monkeypatch.setattr(image_flux, "_precise_vram_mb", lambda *a, **k: 100)
    monkeypatch.setattr(image_flux, "_measure_pipe_mb", lambda p: 0)
    return loads


def test_load_create_then_inpaint_same_file_zero_new_loads(flux, monkeypatch):
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    f = dm / "Flux" / "FLUX_Dev.safetensors"
    _tiny_flux_file(f)
    loads = _patch_load_env(monkeypatch, image_flux, dm)
    _FakeFluxPipe.from_pipe_calls = 0
    _FakeFluxPipe.ctor_calls = 0
    mm = ModelManager()
    k_crea = image_flux.register(mm, "Flux\\FLUX_Dev.safetensors", "flux")          # crea: path relativo
    k_edit = image_flux.register(mm, str(f), "inpaint")                              # modifica: path assoluto
    assert k_crea != k_edit                                                          # chiavi di cache diverse...
    p1 = mm.ensure(k_crea)
    assert loads == [("flux", "FLUX_Dev.safetensors")]                               # 1 solo load completo
    p2 = mm.ensure(k_edit)
    assert loads == [("flux", "FLUX_Dev.safetensors")]                               # ...ma NESSUN nuovo load
    assert _FakeFluxPipe.from_pipe_calls == 0 and p2 is not p1
    assert p2._ls_ident == p1._ls_ident and p2.transformer is p1.transformer         # stessi pesi, stessa istanza


def test_load_other_file_is_a_real_load(flux, monkeypatch):
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    a, b = dm / "Flux" / "A.safetensors", dm / "Flux" / "B.safetensors"
    _tiny_flux_file(a); _tiny_flux_file(b)
    loads = _patch_load_env(monkeypatch, image_flux, dm)
    mm = ModelManager()
    mm.ensure(image_flux.register(mm, str(a), "flux"))
    mm.ensure(image_flux.register(mm, str(b), "inpaint"))                            # PESI diversi → load vero
    assert len(loads) == 2


# ───── 5. run() REALE con pipeline finta: residenza dopo la generazione, pre-evict per identità ─────
def _patch_run_env(monkeypatch, image_flux, tmp_path):
    from PIL import Image
    import core.stego as stego
    import core.watermark as wm
    monkeypatch.setattr(stego, "enabled", lambda: False)
    monkeypatch.setattr(wm, "apply_to_file", lambda p: p)
    _RealGen = torch.Generator

    class _CpuGen:                                       # una CLASSE (non una lambda): `torch.Generator | None` deve restare valido
        def __new__(cls, device="cpu"):
            return _RealGen("cpu")
    monkeypatch.setattr(torch, "Generator", _CpuGen)
    monkeypatch.setattr(image_flux, "_ensure_placement", lambda pipe, mm=None, k=None: "gpu")
    monkeypatch.setattr(image_flux, "_set_attention", lambda pipe, arch="": "sdpa")
    monkeypatch.setattr(image_flux, "_make_preview_cb", lambda *a, **k: (lambda *x, **y: {}))
    monkeypatch.setattr(image_flux, "_measure_pipe_mb", lambda p: 100)
    img = tmp_path / "in.png"; Image.new("RGB", (64, 64)).save(img)
    msk = tmp_path / "m.png"; Image.new("L", (64, 64), 255).save(msk)
    return str(img), str(msk)


class _RunPipe(_FakeFluxPipe):
    calls = 0

    def unload_lora_weights(self):
        pass

    def __call__(self, **kw):
        from PIL import Image
        type(self).calls += 1
        return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])


def test_run_model_stays_resident_after_generation_and_siblings_survive(flux, monkeypatch, tmp_path):
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    a = dm / "Flux" / "FLUX_Dev.safetensors"; _tiny_flux_file(a)
    img, msk = _patch_run_env(monkeypatch, image_flux, tmp_path)
    loads = []

    def _fake_load(model_path, arch, mm=None, klein_te=""):
        loads.append(arch)
        p = _RunPipe()
        p._ls_shared = False
        p._ls_ident = image_flux._weights_ident(arch, image_flux._resolve_model_path(model_path), klein_te)
        return p, 12000
    monkeypatch.setattr(image_flux, "_load", _fake_load)
    mm = ModelManager()
    out = str(tmp_path / "out")
    r1 = image_flux.run({"arch": "flux", "model": "Flux\\FLUX_Dev.safetensors", "prompt": "x", "seed": 1,
                         "steps": 2}, mm, out)
    assert r1["status"] == "success", r1
    k_crea = "flux:Flux\\FLUX_Dev.safetensors"
    assert k_crea in mm._cache                                                   # dopo la generazione RESTA residente
    # «modifica» (inpaint) sullo STESSO file con path assoluto: la shell di «crea» non viene sfrattata
    r2 = image_flux.run({"arch": "inpaint", "model": str(a), "prompt": "y", "seed": 2, "steps": 2,
                         "image_path": img, "mask_path": msk}, mm, out)
    assert r2["status"] == "success", r2
    assert k_crea in mm._cache and f"inpaint:{a}" in mm._cache                  # entrambe residenti (sorelle)
    # pesi DIVERSI → la vecchia si sfratta (comportamento storico, serve a non raddoppiare la RAM)
    b = dm / "Flux" / "Krea.safetensors"; _tiny_flux_file(b)
    r3 = image_flux.run({"arch": "flux", "model": str(b), "prompt": "z", "seed": 3, "steps": 2}, mm, out)
    assert r3["status"] == "success", r3
    assert k_crea not in mm._cache and f"flux:{b}" in mm._cache


def test_run_and_real_load_create_then_inpaint_same_file_one_full_load(flux, monkeypatch, tmp_path):
    """Integrazione: run() REALE + _load() REALE (solo il costruttore dei pesi è finto). «Crea» col path relativo,
    «modifica» col path assoluto dello stesso file: 1 solo load completo, moduli condivisi per la seconda, entrambe
    residenti; con un file DIVERSO il load è vero e la sorella di prima si sfratta."""
    image_flux, dm = flux
    from inference_server.model_manager import ModelManager
    a = dm / "Flux" / "FLUX_Dev.safetensors"; _tiny_flux_file(a)
    b = dm / "Flux" / "Krea.safetensors"; _tiny_flux_file(b)
    loads = _patch_load_env(monkeypatch, image_flux, dm)
    img, msk = _patch_run_env(monkeypatch, image_flux, tmp_path)
    monkeypatch.setattr(image_flux, "_measure_pipe_mb", lambda p: 100)
    _FakeFluxPipe.from_pipe_calls = 0
    _FakeFluxPipe.ctor_calls = 0
    mm = ModelManager()
    out = str(tmp_path / "out")
    r1 = image_flux.run({"arch": "flux", "model": "Flux\FLUX_Dev.safetensors", "prompt": "x", "seed": 1,
                         "steps": 2}, mm, out)
    assert r1["status"] == "success", r1
    r2 = image_flux.run({"arch": "inpaint", "model": str(a), "prompt": "y", "seed": 2, "steps": 2,
                         "image_path": img, "mask_path": msk}, mm, out)
    assert r2["status"] == "success", r2
    assert len(loads) == 1 and _FakeFluxPipe.from_pipe_calls == 0                 # nessun nuovo load, niente from_pipe
    assert len(mm._cache) == 2                                                     # crea e modifica entrambe residenti
    r3 = image_flux.run({"arch": "flux", "model": str(b), "prompt": "z", "seed": 3, "steps": 2}, mm, out)
    assert r3["status"] == "success", r3
    assert len(loads) == 2 and len(mm._cache) == 1                                 # pesi diversi: load vero + sfratto
