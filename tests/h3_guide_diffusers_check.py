"""Guida di latenti allineata H3 (workers/h3_guide.py) contro il pipeline REALE di diffusers: solo CPU, DiT minuscolo con pesi casuali, nessuna GPU, nessun peso vero.

Va lanciato col venv di H3 (diffusers recente col modello MiniMax):
    E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe tests\\h3_guide_diffusers_check.py
Lo lancia anche tests/test_h3_guide.py (salta se il venv non c'e). Stampa «verifiche passate» e esce con 0.

Cosa si prova:
  1. coordinate: a frame_idx 0 le righe guida hanno ESATTAMENTE tempo e griglia spaziale delle righe del video generato (e di un keyframe «first» di diffusers per 1 passo);
     a frame_idx > 0 il tempo e origine + 5/3*frame_idx; l'origine segue i riferimenti nativi (il loro orologio avanza, la guida no);
  2. indici/tag/lunghezze: partizione completa della sequenza, righe condizione in testa, audio e video generati spostati di esattamente le righe guida;
  3. denoise intero sul pipeline vero (blocchi di diffusers + i due sostituiti) con un DiT minuscolo: il DiT vede +G righe, le righe guida entrano a
     0,999*guida + 0,001*rumore (rumore della stessa catena del generatore, dopo i riferimenti), timestep delle righe guida = max(t, 0,999), l'uscita ha la forma di sempre;
  4. SENZA guida: i blocchi installati danno un risultato identico bit per bit a quelli originali di diffusers (con e senza riferimenti nativi);
  5. errori chiari: frame_idx non allineato, guida con griglia spaziale diversa dalla tela, guida che esce dal video.
"""
import functools
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workers"))

import torch  # noqa: E402

import h3_guide as HG  # noqa: E402
from core import h3_guide as CG  # noqa: E402

TAGS = torch.tensor([1] * 6, dtype=torch.long)              # 6 righe di testo
LH, LW = 4, 6                                               # latente 64x96 px -> 2x3 patch... (patch 2x2) -> 6 righe per passo
RPF = (LH // 2) * (LW // 2)
TA = 12
PASS = []


def ok(name):
    PASS.append(name)
    print(f"  ok  {name}", flush=True)


def _base_layout(refs=()):
    """Layout di diffusers per Ref2VA con i riferimenti dati (lista di (kind, latent shape)) e T passi di video generato."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep as R
    T = 7
    refs_ns = [types.SimpleNamespace(kind=k, has_audio=False) for k, _ in refs]
    cond = [torch.zeros(1, 24, *shape) for _, shape in refs]
    pos, tags, vi, ai, ti, ncv, nca = R.build_ref2va_packed_sequence(TAGS, refs_ns, cond, [], T, LH, LW, TA, (1, 2, 2), 2, 2, 0)
    return {"position_ids": pos, "token_tags": tags, "video_indices": vi, "audio_indices": ai, "text_indices": ti,
            "num_condition_video_rows": ncv, "num_condition_audio_rows": nca}, T


def check_coordinates_equal_target_grid():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep as L
    lay, T = _base_layout()
    n = 7
    new = HG.insert_guide_rows(lay, n, LH, LW, 0)
    G = n * RPF
    assert new["n_guide_rows"] == G and new["num_condition_video_rows"] == G
    vi = new["video_indices"]
    g_rows, t_rows = vi[:G], vi[G:]
    pos = new["position_ids"]
    assert torch.equal(pos[g_rows], pos[t_rows]), "a frame_idx 0 la guida deve cadere ESATTAMENTE sulle righe del video generato (tempo + h + w)"
    # un passo di guida = il keyframe «first» di diffusers (stesso tempo, stessa griglia)
    new1 = HG.insert_guide_rows(lay, 1, LH, LW, 0)
    ofl, *_ = L.build_packed_sequence(TAGS, T, LH, LW, TA, (1, 2, 2), 2, 2, 0, ("first",))
    assert torch.equal(new1["position_ids"][new1["video_indices"][:RPF]], ofl[TAGS.shape[0]:TAGS.shape[0] + RPF]), "1 passo guida != keyframe first di diffusers"
    ok("coordinate: guida = griglia del video generato (e = keyframe «first» per 1 passo)")


def check_frame_idx_and_reference_clock():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import _ROPE_FRAME_RESCALE
    lay, T = _base_layout([("image", (1, 4, 4)), ("video", (6, 4, 6))])
    origin = float(lay["position_ids"][int(lay["video_indices"][int(lay["num_condition_video_rows"])]), 0])
    assert origin > TAGS.shape[0] + 1.0, "l'orologio del bersaglio avanza dopo i riferimenti nativi"
    for fi in (0, 1, 5, 9, 17):
        new = HG.insert_guide_rows(lay, 5, LH, LW, fi)
        ncv0 = int(lay["num_condition_video_rows"])
        g = new["video_indices"][ncv0:ncv0 + 5 * RPF]
        t0 = float(new["position_ids"][g[0], 0])
        assert abs(t0 - (origin + _ROPE_FRAME_RESCALE * fi)) < 1e-9, (fi, t0)
    # i riferimenti nativi NON si toccano
    new = HG.insert_guide_rows(lay, 5, LH, LW, 0)
    ncv0 = int(lay["num_condition_video_rows"])
    assert torch.equal(new["video_indices"][:ncv0], lay["video_indices"][:ncv0])
    assert torch.equal(new["position_ids"][:int(lay["audio_indices"][0])], lay["position_ids"][:int(lay["audio_indices"][0])])
    ok("frame_idx: tempo = origine + 5/3*frame_idx, origine dopo i riferimenti, riferimenti intatti")


def check_indices_partition_and_shift():
    for refs in ((), [("image", (1, 4, 4))], [("image", (1, 4, 4)), ("video", (6, 4, 6))]):
        lay, T = _base_layout(refs)
        S0 = lay["position_ids"].shape[0]
        new = HG.insert_guide_rows(lay, 7, LH, LW, 0)
        G = new["n_guide_rows"]
        S1 = new["position_ids"].shape[0]
        assert S1 == S0 + G == new["token_tags"].shape[0]
        allidx = torch.cat([new["text_indices"], new["video_indices"], new["audio_indices"]]).sort().values
        assert torch.equal(allidx, torch.arange(S1)), "gli indici non partizionano la sequenza"
        ncv, nca = new["num_condition_video_rows"], new["num_condition_audio_rows"]
        assert nca == lay["num_condition_audio_rows"] and ncv == lay["num_condition_video_rows"] + G
        # le righe generate (dopo le condizioni) sono le stesse di prima, spostate di G: stesse coordinate
        old_t, new_t = lay["video_indices"][lay["num_condition_video_rows"]:], new["video_indices"][ncv:]
        assert torch.equal(lay["position_ids"][old_t], new["position_ids"][new_t])
        assert torch.equal(new_t, old_t + G)
        old_a, new_a = lay["audio_indices"][nca:], new["audio_indices"][nca:]
        assert torch.equal(lay["position_ids"][old_a], new["position_ids"][new_a]) and torch.equal(new_a, old_a + G)
        assert all(int(new["token_tags"][i]) == int(lay["token_tags"][int(lay["video_indices"][-1])]) for i in new["video_indices"][:ncv].tolist())
    ok("indici: partizione completa, condizioni in testa, righe generate spostate di G")


# ── denoise intero con DiT minuscolo ──────────────────────────────────────────────────────────────
def _tiny_dit():
    from diffusers import MiniMaxH3Transformer3DModel
    torch.manual_seed(3)
    tr = MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=16, hidden_size=32, num_layers=2, num_refiner_layers=1, ffn_dim=64,
        in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=8, freq_dim=16, time_embed_hidden_dim=32,
        time_embed_dim=16, rope_freq_dim=2).eval().requires_grad_(False)
    calls = []
    orig = tr.forward

    @functools.wraps(orig)
    def rec(*a, **k):
        calls.append({n: (v.clone() if torch.is_tensor(v) else v) for n, v in k.items()})
        return orig(*a, **k)
    tr.forward = rec
    return tr, calls


def _run_denoise(install: bool, refs, guide=None, frame_idx=0, steps=4, seed=11, align=False):
    """Esegue le fasi `denoise.*` del workflow ref2va di diffusers (come fa il worker) e ritorna (latenti, audio, chiamate al DiT)."""
    import minimax_h3_infer as W
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
    blocks = MiniMaxH3Blocks().get_workflow("ref2va")
    if install:
        HG.install(blocks)
    names = [n for n in blocks.sub_blocks.keys() if n.startswith("denoise")]
    tr, calls = _tiny_dit()
    st = W._stage(blocks, names, {"transformer_ref": tr, "scheduler": MiniMaxH3Scheduler(shift=12.0), "audio_scheduler": MiniMaxH3Scheduler(shift=3.0)})
    torch.manual_seed(5)
    kw = dict(prompt_embeds=torch.randn(1, TAGS.shape[0], 8), text_token_tags=TAGS, height=LH * 16, width=LW * 16, num_frames=124,
              normalized_references=[types.SimpleNamespace(kind=k, has_audio=False) for k, _ in refs],
              condition_latents=[torch.randn(1, 24, *shape) for _, shape in refs], audio_condition_latents=[])
    if guide is not None:
        kw["guide_latents"], kw["guide_frame_idx"] = guide, frame_idx
    if align:
        kw["ref_clock_align"] = True
    state = st(**W._inputs_for(st, kw), num_inference_steps=steps + 1, generator=torch.Generator("cpu").manual_seed(seed))
    return state.values["latents"], state.values["audio_latents"], calls, kw


def check_denoise_with_guide_on_real_pipeline():
    refs = [("image", (1, 4, 4))]
    torch.manual_seed(9)
    lat_steps = 5 * 7 + 2                                           # 124 fotogrammi -> 37 passi
    guide = torch.randn(1, 24, lat_steps, LH, LW)
    lat, aud, calls, kw = _run_denoise(True, refs, guide)
    base_lat, _, base_calls, _ = _run_denoise(True, refs, None)
    assert lat.shape == base_lat.shape, "la guida non cambia la forma dell'uscita"
    G = lat_steps * RPF
    c0, b0 = calls[0], base_calls[0]
    assert c0["hidden_states"].shape[1] == b0["hidden_states"].shape[1] + G, "il DiT deve vedere esattamente G righe in piu"
    assert c0["position_ids"].shape[0] == b0["position_ids"].shape[0] + G
    # righe condizione = riferimento nativo + guida, in questo ordine; la guida entra a 0,999*guida + 0,001*rumore
    gen = torch.Generator("cpu").manual_seed(11)
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    n_ref = (4 // 2) * (4 // 2)                                     # riferimento: 1 passo 4x4 -> 2x2 patch
    ref_noise = randn_tensor(kw["condition_latents"][0].shape, generator=gen, device="cpu", dtype=torch.float32)
    g_noise = randn_tensor(guide.shape, generator=gen, device="cpu", dtype=torch.float32)
    want_g = patchify_video_latents(0.999 * guide + 0.001 * g_noise, (1, 2, 2))
    got = c0["hidden_states"][0, n_ref:n_ref + G]
    assert torch.allclose(got, want_g, atol=1e-5), float((got - want_g).abs().max())
    # timestep delle righe guida: max(t, 0,999)
    ts, idx = c0["timestep"], c0["timestep_indices"]
    vi = c0["video_indices"]
    t_g = ts[idx[vi[n_ref:n_ref + G]]]
    assert torch.allclose(t_g, torch.full_like(t_g, max(float(t_g[0]), 0.999))) and float(t_g[0]) >= 0.999 - 1e-6
    # il DiT vede coordinate della guida = quelle del video generato
    pos = c0["position_ids"]
    assert torch.equal(pos[vi[n_ref:n_ref + G]], pos[vi[n_ref + G:n_ref + G + G]])
    # le righe generate partono dallo stesso rumore (la guida non le tocca)
    assert not torch.allclose(lat, base_lat), "con la guida il risultato deve cambiare"
    ok("denoise reale: +G righe, 0,999*guida+rumore, timestep max(t,0.999), coordinate del video generato, forma invariata")


def check_guide_only_without_native_references():
    lat_steps = 37
    guide = torch.randn(1, 24, lat_steps, LH, LW)
    lat, aud, calls, kw = _run_denoise(True, [], guide)
    c0 = calls[0]
    S = c0["position_ids"].shape[0]
    G = lat_steps * RPF
    assert c0["hidden_states"].shape[1] == G + lat_steps * RPF, "solo guida: righe = guida + video generato"
    assert tuple(lat.shape) == (1, 24, lat_steps, LH, LW)
    # senza riferimenti il video generato parte a tempo = n_testo, la guida pure
    vi = c0["video_indices"]
    assert float(c0["position_ids"][vi[0], 0]) == float(TAGS.shape[0])
    ok("solo guida (nessun riferimento nativo): sequenza = testo + guida + audio + video, parte a n_testo")


def check_no_guide_is_bit_identical():
    for refs in ([("image", (1, 4, 4))], [("image", (1, 4, 4)), ("video", (6, 4, 6))]):
        a, aa, ca, _ = _run_denoise(False, refs, None)
        b, ba, cb, _ = _run_denoise(True, refs, None)
        assert torch.equal(a, b) and torch.equal(aa, ba), "senza guida i blocchi installati devono dare lo STESSO risultato dei blocchi di diffusers"
        assert len(ca) == len(cb)
        for k in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices", "timestep_indices"):
            assert torch.equal(ca[0][k], cb[0][k]), k
    ok("senza guida: risultato e sequenza identici bit per bit a diffusers (con e senza video di riferimento)")


def check_clear_errors():
    refs = [("image", (1, 4, 4))]
    bad = [
        (torch.randn(1, 24, 37, LH, LW), 2, "frame_idx"),                      # non cade all'inizio di un passo di latente
        (torch.randn(1, 24, 37, LH + 2, LW), 0, "STESSA griglia"),             # griglia spaziale diversa dalla tela
        (torch.randn(1, 24, 37, LH, LW), 9, "esce dal video"),                 # 37 passi da 9: fuori
    ]
    for g, fi, frag in bad:
        try:
            _run_denoise(True, refs, g, fi)
        except Exception as e:                                                 # noqa: BLE001
            assert frag.lower() in str(e).lower(), (frag, str(e)[:300])
        else:
            raise AssertionError(f"doveva fallire: {frag}")
    ok("errori chiari: frame_idx non allineato, griglia diversa, guida fuori dal video")


def check_encode_probe_and_audio_helpers():
    """Ingresso del worker: probe/lettura del video guida, codifica identica a `encode_vae_condition` (la ricetta dei riferimenti di diffusers), audio rimontato."""
    import tempfile
    import av
    import numpy as np
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    tmp = Path(tempfile.mkdtemp(prefix="h3guide_"))
    src = tmp / "g.mp4"
    n_src, W, H = 24 * 3 + 7, 96, 64                                   # 79 fotogrammi a 24 fps
    with av.open(str(src), "w") as c:
        st = c.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        for i in range(n_src):
            img = np.zeros((H, W, 3), np.uint8)
            img[:, :, 1] = 30
            img[10:50, (i * 2) % 60:(i * 2) % 60 + 20] = (220, 60, 60)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    info = HG.probe(str(src))
    assert (info["w"], info["h"]) == (W, H) and not info["has_audio"] and abs(info["n24"] - n_src) <= 1, info
    plan = CG.plan_guide(info["n24"])
    assert plan["guide_frames"] == CG.grid_floor(info["n24"]) == 73 and plan["steps"] == 5 * 4 + 2
    fr = HG.read_guide_frames(str(src), W, H, plan["guide_frames"])
    assert fr.shape == (73, H, W, 3) and fr.dtype == np.uint8
    short = HG.read_guide_frames(str(src), W, H, 100)                  # piu del video: si ripete l'ultimo fotogramma
    assert short.shape[0] == 100 and np.array_equal(short[-1], short[-2])
    sys.path.insert(0, str(ROOT / "tests"))
    import h3_upsample_diffusers_check as UC
    vae = UC._tiny_vae()
    z = HG.encode_guide(vae, fr, device="cpu")
    px = torch.from_numpy(fr).permute(3, 0, 1, 2)[None]
    ref = encode_vae_condition(vae, px, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 42)
    assert z.shape == ref.shape == (1, 24, 22, H // 16, W // 16), (z.shape, ref.shape)
    assert torch.allclose(z, ref, atol=1e-5), float((z - ref).abs().max())
    wav = torch.randn(1, 32000 * 2)                                    # mono 2 s -> stereo, tagliato/riempito alla durata dei fotogrammi
    a = HG.trim_audio(wav, 73)
    assert a.shape == (2, int(round(73 / 24 * 32000))) and torch.equal(a[0, :32000 * 2], wav[0])
    assert HG.trim_audio(None, 73) is None
    ok("ingresso: probe, lettura, codifica = encode_vae_condition di diffusers, audio sorgente rimontato")


def check_ref_clock_align():
    """Orologio del riferimento allineato: le righe del riferimento VIDEO nativo hanno il tempo del passo corrispondente del video generato (h, w del riferimento intatte),
    immagine/bersaglio/indici invariati; spento = identico a diffusers; nel denoise reale il DiT riceve le nuove coordinate."""
    B = HG._bd()
    refs = [("image", (1, 4, 4)), ("video", (6, 4, 6))]
    lay, T = _base_layout(refs)
    shapes = [s_ for _, s_ in refs]
    new = HG.align_ref_clock(lay, [(k, False) for k, _ in refs], shapes, [], T)
    pos0, pos1 = lay["position_ids"], new["position_ids"]
    vi = lay["video_indices"]
    n_img, n_vid = (4 // 2) * (4 // 2), 6 * RPF
    img, vid = vi[:n_img], vi[n_img:n_img + n_vid]
    tgt = vi[int(lay["num_condition_video_rows"]):]
    origin = float(pos0[tgt[0], 0])
    tg = B._temporal_position_grid(T, origin)
    assert torch.equal(pos1[vid, 0], tg[:6].repeat_interleave(RPF)), "tempo del riferimento video != tempo del bersaglio dello stesso passo"
    assert torch.equal(pos1[vid, 0], pos1[tgt[:6 * RPF], 0]), "passo k del riferimento e passo k del bersaglio devono avere lo stesso tempo"
    assert torch.equal(pos1[vid, 1:], pos0[vid, 1:]), "h e w del riferimento restano i suoi"
    assert torch.equal(pos1[img], pos0[img]) and torch.equal(pos1[tgt], pos0[tgt])
    assert float(pos0[vid, 0].min()) != float(pos1[vid, 0].min()), "senza allineamento il riferimento ha un orologio SUO"
    for k in ("token_tags", "video_indices", "audio_indices", "text_indices"):
        assert torch.equal(new[k], lay[k])
    # audio del riferimento video: tempo = origine + indice (canale-maggiore)
    cond = [torch.zeros(1, 24, *s_) for s_ in shapes]
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep as R_
    ra = [types.SimpleNamespace(kind="image", has_audio=False), types.SimpleNamespace(kind="video", has_audio=True)]
    a_lat = [torch.zeros(2 * 20, 32)]
    pos, tags, vi2, ai2, ti2, ncv, nca = R_.build_ref2va_packed_sequence(TAGS, ra, cond, a_lat, T, LH, LW, TA, (1, 2, 2), 2, 2, 0)
    lay2 = {"position_ids": pos, "token_tags": tags, "video_indices": vi2, "audio_indices": ai2, "text_indices": ti2, "num_condition_video_rows": ncv, "num_condition_audio_rows": nca}
    n2 = HG.align_ref_clock(lay2, [("image", False), ("video", True)], shapes, [40], T)
    org2 = float(pos[vi2[ncv], 0])
    assert torch.equal(n2["position_ids"][ai2[:nca], 0], org2 + torch.arange(20, dtype=torch.float64).repeat(2))
    try:
        HG.align_ref_clock(lay, [("image", False), ("video", False)], [(1, 4, 4), (T + 3, 4, 6)], [], T)
    except Exception:                                                  # noqa: BLE001
        pass
    # --- denoise reale: spento = bit a bit diffusers; acceso = il DiT vede le coordinate allineate
    base_lat, _, base_calls, _ = _run_denoise(True, refs, None)
    plain_lat, _, _, _ = _run_denoise(False, refs, None)
    assert torch.equal(base_lat, plain_lat)
    al_lat, _, al_calls, _ = _run_denoise(True, refs, None, align=True)
    c0 = al_calls[0]
    p = c0["position_ids"]
    v = c0["video_indices"]
    ref_rows = v[n_img:n_img + n_vid]
    tgt_rows = v[n_img + n_vid:]
    assert torch.equal(p[ref_rows, 0], p[tgt_rows[:n_vid], 0]), "nel DiT il riferimento deve avere il tempo del bersaglio"
    assert torch.equal(c0["token_tags"], base_calls[0]["token_tags"]) and c0["hidden_states"].shape == base_calls[0]["hidden_states"].shape
    assert not torch.equal(al_lat, base_lat), "il riferimento allineato deve cambiare il risultato"
    ok("orologio del riferimento allineato: tempo = bersaglio, h/w e indici intatti, audio del rif. video, spento = diffusers, DiT vede le coordinate")


def main() -> int:
    print("verifica guida H3 (CPU, DiT minuscolo)", flush=True)
    check_coordinates_equal_target_grid()
    check_frame_idx_and_reference_clock()
    check_indices_partition_and_shift()
    check_denoise_with_guide_on_real_pipeline()
    check_guide_only_without_native_references()
    check_no_guide_is_bit_identical()
    check_clear_errors()
    check_encode_probe_and_audio_helpers()
    check_ref_clock_align()
    print(f"verifiche passate: {len(PASS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
