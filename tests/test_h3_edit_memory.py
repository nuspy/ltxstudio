"""Piano di memoria dell'edit H3: i riferimenti VIDEO contano per i loro frame veri (giro 02/11: stimati 8 frame di latente, erano 42)."""
import pytest
from core import h3_vram_plan as HP

H, W, FRAMES = 480, 512, 141


def test_video_reference_rows_use_real_frames_when_given():
    short = HP.ref_rows_estimate([{"kind": "video", "path": "x.mp4"}], 1024, H, W)                    # riferimento di movimento breve: 8 frame di latente
    full = HP.ref_rows_estimate([{"kind": "video", "path": "x.mp4", "frames": FRAMES}], 1024, H, W)
    per = HP.rows_per_latent_frame(H, W)
    assert short == 8 * per
    assert full == HP.latent_frames(FRAMES) * per == 42 * per         # 17*8+5 frame -> 5*8+2 = 42 frame di latente
    assert full > 5 * short


def test_edit_plan_needs_more_memory_than_the_old_estimate_and_engages_offload():
    dims = dict(HP.DEFAULT_DIMS)
    pic = {"kind": "image", "path": "p.png"}
    old_refs = HP.ref_rows_estimate([pic, {"kind": "video", "path": "v.mp4"}], 1024, H, W)
    new_refs = HP.ref_rows_estimate([pic, {"kind": "video", "path": "v.mp4", "frames": FRAMES}], 1024, H, W)
    t_old = HP.sequence_tokens(FRAMES, H, W, ref_rows=old_refs)
    t_new = HP.sequence_tokens(FRAMES, H, W, ref_rows=new_refs)
    assert t_new - t_old == new_refs - old_refs and t_new > 1.4 * t_old
    # DiT nvfp4 ~12,5 GB, 29 GB liberi: con la stima nuova il piano deve chiedere (molto) piu attivazioni
    a_old = HP.activation_gb(t_old, dims)
    a_new = HP.activation_gb(t_new, dims)
    assert a_new > a_old + 0.5                                       # modello a blocchi: ~0,1 MB/token, +8k token = ~+0,9 GB
    plan_new = HP.plan(29.0, t_new, dims, weights_gb=12.5, lora_gb=1.0)
    assert plan_new["tokens"] == t_new and plan_new["need_gb"] > HP.plan(29.0, t_old, dims, weights_gb=12.5, lora_gb=1.0)["need_gb"]


# ── sequenza carica -> libera (decode a fine clip) ─────────────────────────────────────────────────────────────────────────────
import sys                                                                                           # noqa: E402
from pathlib import Path                                                                             # noqa: E402

import torch                                                                                         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
import minimax_h3_infer as REAL                                                                      # noqa: E402


def test_decode_frees_both_vaes_even_if_something_still_references_them():
    """Il `del st` non basta: il VAE decoder (9,7 GB fp32) restava in VRAM fra i clip. Qui lo stage finto TIENE i riferimenti ai moduli
    (come fa la pipeline vera) e dopo la decodifica i pesi devono comunque essere vuoti; l'ordine e carica -> usa -> libera."""
    events, keep = [], []

    def mk(name):
        m = torch.nn.Linear(64, 64)
        events.append(f"load:{name}")
        return m

    def fake_stage(blocks, names, comps):
        keep.extend(comps.values())                           # riferimenti che sopravvivono allo stage
        events.append("stage")

        def run_stage(state=None, output=None):
            assert all(p.numel() > 0 for m in comps.values() for p in m.parameters()), "i VAE devono essere carichi durante la decodifica"
            events.append("decode")
            return {"videos": ["v"], "audio": ["a"], "sampling_rate": 32000}
        return run_stage

    res = REAL._decode_and_free(None, ["decode"], {}, {}, load_vae_fn=lambda W, **kw: mk("vae"), load_avae_fn=lambda W: mk("avae"), stage_fn=fake_stage)
    assert res["videos"] == ["v"] and events == ["load:vae", "load:avae", "stage", "decode"]
    assert len(keep) == 2 and all(p.numel() == 0 for m in keep for p in m.parameters()), "VAE ancora carichi dopo la decodifica"


def test_decode_frees_vaes_also_when_decoding_fails():
    keep = []

    def fake_stage(blocks, names, comps):
        keep.extend(comps.values())

        def boom(state=None, output=None):
            raise RuntimeError("OOM finto")
        return boom

    import pytest
    with pytest.raises(RuntimeError):
        REAL._decode_and_free(None, [], {}, {}, load_vae_fn=lambda W: torch.nn.Linear(8, 8), load_avae_fn=lambda W: torch.nn.Linear(8, 8),
                              stage_fn=fake_stage)
    assert all(p.numel() == 0 for m in keep for p in m.parameters())


def test_preflight_of_an_edit_clip_counts_reference_video_frames(monkeypatch):
    """`run()` chiama `_preflight` (-> core.h3_vram_plan) con i `references` dell'edit: il piano deve contare il video di riferimento intero."""
    monkeypatch.setattr(REAL, "_weights_gb", lambda path: {"dit": 12.5, "te": 15.7}.get(Path(path).stem, 1.0))
    monkeypatch.setattr(REAL, "lora_specs", lambda W, p: [])
    Wd = {"dit": "dit.safetensors", "te": "te.safetensors", "configs": "C:/nonexistent"}
    kw = {"height": H, "width": W}
    base = {"references": [{"kind": "image", "path": "p.png"}, {"kind": "video", "path": "v.mp4"}]}
    edit = {"references": [{"kind": "image", "path": "p.png"}, {"kind": "video", "path": "v.mp4", "frames": FRAMES}]}
    seen = []
    real_plan = HP.plan
    monkeypatch.setattr(HP, "plan", lambda *a, **k: (seen.append(a[1]) or real_plan(*a, **k)))
    _d1, plan_old = REAL._preflight(Wd, base, "ref2va", kw, 30.0, FRAMES)
    _d2, plan_new = REAL._preflight(Wd, edit, "ref2va", kw, 30.0, FRAMES)
    assert len(seen) == 2, "il piano di memoria (h3_vram_plan.plan) deve essere chiamato a ogni clip"
    assert seen[1] - seen[0] == HP.ref_rows_estimate(edit["references"], 1024, H, W) - HP.ref_rows_estimate(base["references"], 1024, H, W) > 0
    assert plan_new["tokens"] == seen[1] > plan_old["tokens"]


# ── tetto per processo e un-processo-per-clip ──────────────────────────────────────────────────────────────────────────────────
def test_cap_fraction_counts_memory_already_reserved_by_the_process():
    """Giro 02/11 (clip 2): libera 21,7 GB con ~10 GB gia riservati da questo processo: il tetto vecchio (libera-0,4)/totale = 21,3 GiB
    veniva superato dal solo TE (14,6) + il residuo; il nuovo conta la memoria gia riservata."""
    old = max(0.2, min(1.0, (21.7 - 0.4) / 31.8))
    new = REAL.cap_fraction(21.7, 31.8, reserved_gb=7.5)
    assert abs(old * 31.8 - 21.3) < 0.05
    assert abs(new * 31.8 - (21.7 + 7.5 - 0.4)) < 0.05 and new > old
    assert REAL.cap_fraction(30.0, 31.8, 0.0) == max(0.2, min(1.0, 29.6 / 31.8))             # processo fresco: invariato
    assert REAL.cap_fraction(40.0, 31.8, 5.0) == 1.0 and REAL.cap_fraction(0.1, 31.8, 0.0) == 0.2


def test_subprocess_clip_runner_streams_log_and_reports_the_failure_tail(tmp_path):
    import types as _t
    import h3_edit_run as ER
    ok = tmp_path / "ok_worker.py"
    ok.write_text("import sys\nprint('[h3] figlio parte', sys.argv[1:], flush=True)\n", encoding="utf-8")
    bad = tmp_path / "bad_worker.py"
    bad.write_text("import sys\nprint('[h3] riga 1', flush=True)\nprint('torch.OutOfMemoryError: finto', flush=True)\nsys.exit(1)\n", encoding="utf-8")
    ER._subprocess_clip_runner(_t.SimpleNamespace(__file__=str(ok)), str(tmp_path))            # nessuna eccezione
    import pytest
    with pytest.raises(RuntimeError) as ei:
        ER._subprocess_clip_runner(_t.SimpleNamespace(__file__=str(bad)), str(tmp_path))
    assert "OutOfMemoryError: finto" in str(ei.value) and "codice 1" in str(ei.value)


def test_purge_big_quantized_tensors_empties_survivors_even_if_something_still_references_them():
    """Giro 03/11 (video_minimax_h3_04): dopo `_drop_module_weights` restavano 14,3 GB di int8 dei blocchi del DiT e il decode moriva per OOM. La pulizia agisce
    sui tensori, qualunque sia l'oggetto che li tiene; i tensori float (VAE) e quelli piccoli restano."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
    import minimax_h3_infer as M
    holder = {"w": torch.ones(2048, 1024, dtype=torch.int8), "f8": torch.ones(1024, 1024, dtype=torch.float8_e4m3fn),
              "vae": torch.ones(1024, 1024, dtype=torch.float32), "small": torch.ones(8, dtype=torch.int8)}
    freed = M.purge_big_quantized_tensors(min_bytes=1 << 20, device_type="cpu")
    assert holder["w"].numel() == 0 and holder["f8"].numel() == 0
    assert holder["vae"].numel() == 1024 * 1024 and holder["small"].numel() == 8
    assert abs(freed - (2048 * 1024 + 1024 * 1024) / 2**30) < 1e-6


def test_video_reference_plan_is_realistic_on_32_24_and_16_gb_cards():
    """Giro 03/11 (fixture 12 s, 864x480, clip da 141 frame, edit con video di riferimento): la prova e riuscita con picco 24,1-26,0 GB e 12,6 GB di pesi, cioe
    attivazioni ~12-13 GB (il modello base ne prevedeva ~7): fattore 1,6 sui riferimenti video. Su 32 GB tutto residente con picco atteso < 27 GB; su 24 GB ci sta
    con blocchi del DiT in RAM; su 16 GB il clip da 141 frame NON ci sta (messaggio chiaro, niente OOM a meta) ma uno piu corto (105 frame) si."""
    Hh, W = 480, 864

    def mk(frames, free):
        refs = [{"kind": "image", "path": "p.png"}, {"kind": "video", "path": "v.mp4", "frames": frames}]
        t = HP.sequence_tokens(frames, Hh, W, ref_rows=HP.ref_rows_estimate(refs, 1024, Hh, W))
        return HP.plan(free, t, weights_gb=11.7, lora_gb=0.9, act_factor=HP.VIDEO_REF_ACT_FACTOR)
    assert HP.has_video_refs([{"kind": "image"}, {"kind": "video"}]) and not HP.has_video_refs([{"kind": "image"}]) and not HP.has_video_refs(None)
    assert HP.VIDEO_REF_ACT_FACTOR == 1.6
    big = mk(141, 29.6)
    assert big["fits"] and big["offload_gb"] <= 0                                                   # 32 GB: tutto residente
    peak = HP.expected_peak_gb(big, free_after_load_gb=29.6 - 12.6)
    assert 24.0 < peak < 27.0, peak                                                                 # misurato 24,1-26,0
    mid = mk(141, 22.0)
    assert mid["fits"] and mid["offload_gb"] > 3.0                                                  # 24 GB: blocchi in RAM
    small = mk(141, 15.6)
    assert not small["fits"]                                                                        # 16 GB, 141 frame: non ci sta (e lo dice prima)
    ok16 = mk(105, 15.6)
    assert ok16["fits"] and ok16["offload_gb"] > 5.0 and HP.expected_peak_gb(ok16, free_after_load_gb=15.6 - 12.6) < 15.6
    assert HP.activation_gb(big["tokens"], HP.DEFAULT_DIMS, act_factor=1.6) == pytest.approx(1.6 * HP.activation_gb(big["tokens"], HP.DEFAULT_DIMS))


def test_chunk_size_is_a_power_of_two(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
    import h3_chunk as K
    monkeypatch.delenv("H3_BLOCK_CHUNK", raising=False)
    for rows, rb in ((200000, 107_000), (200000, 150_000), (50000, 80_000)):
        n = K.chunk_tokens(rows, rb, "cpu") if False else None
    import torch
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (13_700_000_000, 31 * 2**30))
    d = torch.device("cuda:0")
    for rb in (107_000, 150_000, 233_333):
        n = K.chunk_tokens(10**6, rb, d)
        assert n & (n - 1) == 0 and n >= K.MIN_CHUNK * 4, (rb, n)
