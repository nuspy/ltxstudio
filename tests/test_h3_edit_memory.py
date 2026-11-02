"""Piano di memoria dell'edit H3: i riferimenti VIDEO contano per i loro frame veri (giro 02/11: stimati 8 frame di latente, erano 42)."""
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

    res = REAL._decode_and_free(None, ["decode"], {}, {}, load_vae_fn=lambda W: mk("vae"), load_avae_fn=lambda W: mk("avae"), stage_fn=fake_stage)
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
