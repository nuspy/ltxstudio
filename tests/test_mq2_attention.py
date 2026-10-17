"""MagicQuill V2 nativo: blocchi con condizione di controllo su un FluxTransformer2DModel PICCOLO e casuale (CPU,
float32, nessun checkpoint FLUX).

Verifica:
  1. senza controllo il forward patchato == forward diffusers standard (regressione);
  2. scrittura poi lettura della cache K/V: output finiti, shape giuste, e == a una TRASCRIZIONE LETTERALE dell'algoritmo
     upstream (maschera piena -1e20 / bias log(gamma), banca K/V + banca attenzione, LoRA mascherate) scritta qui sotto
     contro i moduli diffusers 0.38, per gamma 1.0 e 0.6, condizione 64 px (16 token) e 512 px (1024 token);
  3. il primo passaggio (scrittura) dà ai token non-condizione lo STESSO risultato di una lettura (warm-up assorbito);
  4. lettura con coda condizione presente == lettura senza coda (la coda viene scartata);
  5. ripristino completo (forward dei blocchi, hook, banca).
Eseguire con il venv immagini: E:\\AI\\Tools\\inference_server_venv\\Scripts\\python.exe -m pytest (serve pytest nel path)."""
import math
import types

import pytest
import torch
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel
from diffusers.models.embeddings import apply_rotary_emb

from core.resources import commit_free_gb
from inference_server.backends.mq2 import attention as mqa
from inference_server.backends.mq2.attention import MQState, patch_transformer
from inference_server.backends.mq2.control_lora import ControlLoRA, ControlSpec
from inference_server.backends.mq2.pipeline import cond_latent_ids, control_session

DIM = 64            # 4 teste x 16
T, N, K = 6, 16, 16  # token testo / rumore / Kontext


@pytest.fixture(autouse=True)
def _risorse():
    if commit_free_gb() < 3.5:
        pytest.skip("commit libero < 3.5 GB")


def _tiny(seed=0, n_double=2, n_single=2):
    torch.manual_seed(seed)
    tr = FluxTransformer2DModel(patch_size=1, in_channels=64, num_layers=n_double, num_single_layers=n_single,
                                attention_head_dim=16, num_attention_heads=4, joint_attention_dim=32,
                                pooled_projection_dim=24, guidance_embeds=True, axes_dims_rope=(4, 6, 6))
    for p in tr.parameters():                            # init non banale e variata (alcuni layer nascono a zero)
        if p.dim() >= 1 and float(p.detach().abs().max()) == 0.0:
            torch.nn.init.normal_(p, std=0.05)
    return tr.eval().float()


def _lora(n_double, n_single, rank=4, seed=1, std=0.25):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for i in range(n_double):
        for p in ("q", "k", "v", "proj"):
            sd[f"transformer_blocks.{i}.attn.processor.{p}_loras.0.down.weight"] = torch.randn(rank, DIM, generator=g) * std
            sd[f"transformer_blocks.{i}.attn.processor.{p}_loras.0.up.weight"] = torch.randn(DIM, rank, generator=g) * std
    for i in range(n_single):
        for p in ("q", "k", "v"):
            sd[f"single_transformer_blocks.{i}.attn.processor.{p}_loras.0.down.weight"] = torch.randn(rank, DIM, generator=g) * std
            sd[f"single_transformer_blocks.{i}.attn.processor.{p}_loras.0.up.weight"] = torch.randn(DIM, rank, generator=g) * std
    return ControlLoRA.from_state_dict(sd, n_double=n_double, n_single=n_single, dim=DIM, name="tiny")


def _inputs(cond_size, seed=3):
    g = torch.Generator().manual_seed(seed)
    C = (cond_size // 16) ** 2
    d = dict(
        noise=torch.randn(1, N, 64, generator=g), kontext=torch.randn(1, K, 64, generator=g),
        cond=torch.randn(1, C, 64, generator=g), enc=torch.randn(1, T, 32, generator=g),
        pooled=torch.randn(1, 24, generator=g), guidance=torch.tensor([3.5]),
        txt_ids=torch.zeros(T, 3))
    grid = lambda mark: torch.stack([torch.full((16,), float(mark)),
                                     torch.arange(4).repeat_interleave(4).float(),
                                     torch.arange(4).repeat(4).float()], -1)
    d["ids_nk"] = torch.cat([grid(0), grid(1)], 0)
    d["ids_cond"] = cond_latent_ids(8, 8, cond_size, "cpu", torch.float32)
    d["C"] = C
    return d


def _call(tr, d, t, with_cond_tail=True, with_cond_ids=True):
    hs = torch.cat([d["noise"], d["kontext"]] + ([d["cond"]] if with_cond_tail else []), 1)
    ids = torch.cat([d["ids_nk"]] + ([d["ids_cond"]] if with_cond_ids else []), 0)
    with torch.no_grad():
        return tr(hidden_states=hs, timestep=torch.tensor([t]), guidance=d["guidance"], pooled_projections=d["pooled"],
                  encoder_hidden_states=d["enc"], txt_ids=d["txt_ids"], img_ids=ids, return_dict=False)[0]


# ---------------------------------------------------------------------------------------------------------------
# TRASCRIZIONE dell'algoritmo upstream (layers_cache.py + transformer_flux.py di MagicQuillV2), layout (B,H,S,D)
# ---------------------------------------------------------------------------------------------------------------
class RefMQ:
    def __init__(self, lora, gamma, sizes, w=1.0):
        self.lora, self.gamma, self.sizes, self.w = lora, gamma, sizes, w
        self.bank_kv, self.bank_attn = {}, {}

    def _lora(self, kind, i, proj, x, bs, off, cur):                 # LoRALinearLayer.forward (maschera sulla fetta)
        mask = torch.zeros_like(x)
        mask[:, bs + off:bs + off + cur] = 1
        down, up = self.lora.layers[(kind, i, proj)]
        return F.linear(F.linear(x * mask, down), up)

    def _bias_mask(self, rows, cols, img_start, img_end, cond_start, full):
        if full:                                                       # percorso cache: maschera (S,S) piena
            m = torch.ones(rows, cols)
            m[:cond_start, :] = 0
            m[cond_start:, cond_start:] = 0                            # (una sola condizione)
            m = m * -1e20
        else:
            m = torch.zeros(rows, cols)
        if self.gamma is not None:
            m[img_start:img_end, cond_start:cond_start + self.sizes[0]] = torch.log(torch.tensor(float(self.gamma)))
        return m

    def double(self, i, block, hs, cond, enc, temb, cond_temb, emb):
        attn, H, w = block.attn, block.attn.heads, self.w
        C = self.sizes[0]
        norm_h, g_msa, sh_mlp, sc_mlp, g_mlp = block.norm1(hs, emb=temb)
        cat = norm_h
        if cond is not None:
            norm_c, cg_msa, cg_sh, cg_sc, cg_g = block.norm1(cond, emb=cond_temb)
            cat = torch.cat([norm_h, norm_c], 1)
        norm_e, c_g_msa, c_sh, c_sc, c_g = block.norm1_context(enc, emb=temb)
        B, Tt = hs.shape[0], enc.shape[1]
        hd = attn.to_q.out_features // H
        view = lambda x: x.view(B, -1, H, hd).transpose(1, 2)
        block_size = cat.shape[1] - C
        scaled_block = Tt + cat.shape[1] - C
        seq = Tt + cat.shape[1]
        img_start, img_end, cond_start = Tt, block_size, scaled_block
        eq = attn.norm_added_q(view(attn.add_q_proj(norm_e)))
        ek = attn.norm_added_k(view(attn.add_k_proj(norm_e)))
        ev = view(attn.add_v_proj(norm_e))
        key = ("d", i)
        if key not in self.bank_kv:
            q, k, v = attn.to_q(cat), attn.to_k(cat), attn.to_v(cat)
            q = q + w * self._lora("d", i, "q", cat, block_size, 0, C)
            k = k + w * self._lora("d", i, "k", cat, block_size, 0, C)
            v = v + w * self._lora("d", i, "v", cat, block_size, 0, C)
            q, k, v = view(q), view(k), view(v)
            self.bank_kv[key] = (k[:, :, block_size:], v[:, :, block_size:])
            q, k = attn.norm_q(q), attn.norm_k(k)
            q, k, v = torch.cat([eq, q], 2), torch.cat([ek, k], 2), torch.cat([ev, v], 2)
            q, k = apply_rotary_emb(q, emb), apply_rotary_emb(k, emb)
            m = self._bias_mask(seq, seq, img_start, img_end, cond_start, True)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
            self.bank_attn[key] = out[:, :, scaled_block:]
        else:
            q, k, v = view(attn.to_q(cat)), view(attn.to_k(cat)), view(attn.to_v(cat))
            k = torch.cat([k[:, :, :block_size], self.bank_kv[key][0]], 2)
            v = torch.cat([v[:, :, :block_size], self.bank_kv[key][1]], 2)
            q, k = attn.norm_q(q), attn.norm_k(k)
            q, k, v = torch.cat([eq, q], 2), torch.cat([ek, k], 2), torch.cat([ev, v], 2)
            q, k = apply_rotary_emb(q, emb), apply_rotary_emb(k, emb)
            q = q[:, :, :scaled_block]
            m = self._bias_mask(q.shape[2], k.shape[2], img_start, img_end, cond_start, False)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
            out = torch.cat([out, self.bank_attn[key]], 2)
        out = out.transpose(1, 2).reshape(B, -1, H * hd)
        enc_o, hs_o = out[:, :Tt], out[:, Tt:]
        hs_o = attn.to_out[0](hs_o)
        hs_o = hs_o + w * self._lora("d", i, "proj", hs_o, hs_o.shape[1] - C, 0, C)
        hs_o = attn.to_out[1](hs_o)
        enc_o = attn.to_add_out(enc_o)
        cond_o = hs_o[:, block_size:]
        hs_o = hs_o[:, :block_size]
        hs = hs + g_msa.unsqueeze(1) * hs_o
        n2 = block.norm2(hs) * (1 + sc_mlp[:, None]) + sh_mlp[:, None]
        hs = hs + g_mlp.unsqueeze(1) * block.ff(n2)
        if cond is not None:
            cond = cond + cg_msa.unsqueeze(1) * cond_o
            nc = block.norm2(cond) * (1 + cg_sc[:, None]) + cg_sh[:, None]
            cond = cond + cg_g.unsqueeze(1) * block.ff(nc)
        enc = enc + c_g_msa.unsqueeze(1) * enc_o
        ne = block.norm2_context(enc) * (1 + c_sc[:, None]) + c_sh[:, None]
        enc = enc + c_g.unsqueeze(1) * block.ff_context(ne)
        return enc, hs, cond

    def single(self, i, block, joint, cond, Tt, temb, cond_temb, emb):
        attn, H, w = block.attn, block.attn.heads, self.w
        C = self.sizes[0]
        residual = joint
        norm_j, gate = block.norm(joint, emb=temb)
        mlp = block.act_mlp(block.proj_mlp(norm_j))
        cat = norm_j
        if cond is not None:
            residual_c = cond
            norm_c, cgate = block.norm(cond, emb=cond_temb)
            mlp_c = block.act_mlp(block.proj_mlp(norm_c))
            cat = torch.cat([norm_j, norm_c], 1)
        B = joint.shape[0]
        hd = attn.to_q.out_features // H
        view = lambda x: x.view(B, -1, H, hd).transpose(1, 2)
        block_size = cat.shape[1] - C
        img_start, img_end, cond_start = Tt, block_size, block_size
        key = ("s", i)
        if key not in self.bank_kv:
            q, k, v = attn.to_q(cat), attn.to_k(cat), attn.to_v(cat)
            q = q + w * self._lora("s", i, "q", cat, block_size, 0, C)
            k = k + w * self._lora("s", i, "k", cat, block_size, 0, C)
            v = v + w * self._lora("s", i, "v", cat, block_size, 0, C)
            q, k, v = view(q), view(k), view(v)
            self.bank_kv[key] = (k[:, :, block_size:], v[:, :, block_size:])
            q, k = attn.norm_q(q), attn.norm_k(k)
            q, k = apply_rotary_emb(q, emb), apply_rotary_emb(k, emb)
            m = self._bias_mask(cat.shape[1], cat.shape[1], img_start, img_end, cond_start, True)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
            self.bank_attn[key] = out[:, :, block_size:]
        else:
            q, k, v = view(attn.to_q(cat)), view(attn.to_k(cat)), view(attn.to_v(cat))
            k = torch.cat([k[:, :, :block_size], self.bank_kv[key][0]], 2)
            v = torch.cat([v[:, :, :block_size], self.bank_kv[key][1]], 2)
            q, k = attn.norm_q(q), attn.norm_k(k)
            q, k = apply_rotary_emb(q, emb), apply_rotary_emb(k, emb)
            q = q[:, :, :block_size]
            m = self._bias_mask(q.shape[2], k.shape[2], img_start, img_end, cond_start, False)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
            out = torch.cat([out, self.bank_attn[key]], 2)
        out = out.transpose(1, 2).reshape(B, -1, H * hd)
        cond_o, out = out[:, block_size:], out[:, :block_size]
        h = residual + gate.unsqueeze(1) * block.proj_out(torch.cat([out, mlp], 2))
        if cond is not None:
            cond = residual_c + cgate.unsqueeze(1) * block.proj_out(torch.cat([cond_o, mlp_c], 2))
        return h, cond

    def forward(self, tr, d, t):
        hs = tr.x_embedder(torch.cat([d["noise"], d["kontext"]], 1))
        cond = tr.x_embedder(d["cond"])
        ts = torch.tensor([t]) * 1000
        guid = d["guidance"] * 1000
        temb = tr.time_text_embed(ts, guid, d["pooled"])
        cond_temb = tr.time_text_embed(torch.zeros_like(ts), guid, d["pooled"])
        enc = tr.context_embedder(d["enc"])
        emb = tr.pos_embed(torch.cat([d["txt_ids"], d["ids_nk"], d["ids_cond"]], 0))
        for i, b in enumerate(tr.transformer_blocks):
            enc, hs, cond = self.double(i, b, hs, cond, enc, temb, cond_temb, emb)
        joint = torch.cat([enc, hs], 1)
        Tt = enc.shape[1]
        for i, b in enumerate(tr.single_transformer_blocks):
            joint, cond = self.single(i, b, joint, cond, Tt, temb, cond_temb, emb)
        hs = joint[:, Tt:]
        return tr.proj_out(tr.norm_out(hs, temb))


# ---------------------------------------------------------------------------------------------------------------
def _spec(lora, cond_size, gamma):
    return ControlSpec(lora=lora, image=None, gamma=gamma, lora_weight=1.0, cond_size=cond_size)


def test_regressione_senza_controllo_uguale_al_forward_standard():
    tr = _tiny()
    d = _inputs(64)
    ref = _call(tr, d, 0.7, with_cond_tail=False, with_cond_ids=False)
    st = MQState([])
    with patch_transformer(tr, st):
        got = _call(tr, d, 0.7, with_cond_tail=False, with_cond_ids=False)
    assert got.shape == ref.shape == (1, N + K, 64)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), float((got - ref).abs().max())


@pytest.mark.parametrize("cond_size,gamma", [(64, 1.0), (64, 0.6), (512, 1.5)])
def test_scrittura_lettura_come_upstream(cond_size, gamma):
    tr = _tiny()
    d = _inputs(cond_size)
    C = d["C"]
    lora = _lora(2, 2)
    ref = RefMQ(lora, None if gamma == 1.0 else gamma, [C])          # upstream: c_factor sempre impostato; gamma=1 -> log 0
    ref_w = ref.forward(tr, d, 0.9)                                   # passaggio scrittura (warm-up upstream)
    ref_r = ref.forward(tr, d, 0.6)                                   # passaggio lettura (cond ancora presente)
    st = MQState([_spec(lora, cond_size, gamma)])
    st.n_base = N + K
    with patch_transformer(tr, st):
        w = _call(tr, d, 0.9)                                         # scrittura
        assert len(st.bank) == 2 + 2 and st.writes == 4
        r_tail = _call(tr, d, 0.6)                                    # lettura con coda condizione presente
        r_notail = _call(tr, d, 0.6, with_cond_tail=False)            # lettura senza coda
        assert st.reads == 8
    assert w.shape == (1, N + K + C, 64) and r_tail.shape == (1, N + K, 64) and r_notail.shape == (1, N + K, 64)
    for x in (w, r_tail, r_notail):
        assert torch.isfinite(x).all()
    atol = 2e-4
    assert torch.allclose(w[:, :N + K], ref_w, atol=atol, rtol=1e-3), float((w[:, :N + K] - ref_w).abs().max())
    assert torch.allclose(r_tail, ref_r, atol=atol, rtol=1e-3), float((r_tail - ref_r).abs().max())
    assert torch.allclose(r_notail, ref_r, atol=atol, rtol=1e-3), float((r_notail - ref_r).abs().max())
    # il warm-up upstream e' assorbito: la scrittura ha dato ai token non-condizione lo stesso risultato di una lettura
    st2 = MQState([_spec(lora, cond_size, gamma)])
    st2.n_base = N + K
    with patch_transformer(tr, st2):
        _call(tr, d, 0.9)
        same_t = _call(tr, d, 0.9, with_cond_tail=False)
    assert torch.allclose(w[:, :N + K], same_t, atol=atol, rtol=1e-3)
    # la condizione conta davvero (il controllo non e' un no-op) e il bias gamma cambia il risultato
    plain = _call(tr, d, 0.6, with_cond_tail=False, with_cond_ids=False)
    assert not torch.allclose(plain, r_notail, atol=1e-4)


def test_bias_gamma_cambia_il_risultato():
    tr = _tiny()
    d = _inputs(64)
    lora = _lora(2, 2)
    outs = []
    for g in (1.0, 0.3):
        st = MQState([_spec(lora, 64, g)])
        st.n_base = N + K
        with patch_transformer(tr, st):
            _call(tr, d, 0.9)
            outs.append(_call(tr, d, 0.6, with_cond_tail=False))
    assert not torch.allclose(outs[0], outs[1], atol=1e-4)


def test_rows_quirk_flag_cambia_solo_le_righe_doppie():
    tr = _tiny()
    d = _inputs(64)
    lora = _lora(2, 2)
    res = []
    for flag in (True, False):
        mqa.DOUBLE_ROWS_UPSTREAM = flag
        try:
            st = MQState([_spec(lora, 64, 0.4)])
            st.n_base = N + K
            with patch_transformer(tr, st):
                _call(tr, d, 0.9)
                res.append(_call(tr, d, 0.6, with_cond_tail=False))
        finally:
            mqa.DOUBLE_ROWS_UPSTREAM = True
    assert torch.isfinite(res[0]).all() and torch.isfinite(res[1]).all()
    assert not torch.allclose(res[0], res[1], atol=1e-6)    # il flag e' osservabile (T<N: le ultime T righe Kontext)


def test_ripristino_completo():
    tr = _tiny()
    d = _inputs(64)
    before = _call(tr, d, 0.5, with_cond_tail=False, with_cond_ids=False)
    st = MQState([_spec(_lora(2, 2), 64, 0.8)])
    st.n_base = N + K
    with patch_transformer(tr, st):
        _call(tr, d, 0.9)
        assert all("forward" in b.__dict__ for b in tr.transformer_blocks)
    assert not any("forward" in b.__dict__ for b in list(tr.transformer_blocks) + list(tr.single_transformer_blocks))
    assert not tr.time_text_embed._forward_pre_hooks and not st.bank and st.tte_module is None
    after = _call(tr, d, 0.5, with_cond_tail=False, with_cond_ids=False)
    assert torch.equal(before, after)
    # errore dentro il with: ripristino comunque
    with pytest.raises(RuntimeError, match="boom"):
        with patch_transformer(tr, MQState([_spec(_lora(2, 2), 64, 1.0)])):
            raise RuntimeError("boom")
    assert not any("forward" in b.__dict__ for b in tr.transformer_blocks) and not tr.time_text_embed._forward_pre_hooks


def test_validate_transformer_rifiuta_non_diffusers():
    with pytest.raises(mqa.MQUnsupported):
        mqa.validate_transformer(types.SimpleNamespace())


def test_prepare_latents_aggiunge_condizione_in_coda_e_ripristina():
    """Pipe finta (eredita FluxKontextPipeline.prepare_latents/_pack/_encode stock): latenti/id Kontext + condizione."""
    from diffusers import FluxKontextPipeline
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image

    class FakeVAE:
        config = types.SimpleNamespace(shift_factor=0.1, scaling_factor=0.5)

        def encode(self, x):
            z = F.adaptive_avg_pool2d(x, (x.shape[2] // 8, x.shape[3] // 8))
            z = torch.cat([z] * 6, 1)[:, :16]
            return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: z, sample=lambda g=None: z))

    class P(FluxKontextPipeline):
        pass
    p = object.__new__(P)
    p.__dict__.update(vae=FakeVAE(), vae_scale_factor=8, latent_channels=16,
                      image_processor=VaeImageProcessor(vae_scale_factor=16), transformer=_tiny())
    lora = _lora(2, 2)
    spec = ControlSpec(lora=lora, image=Image.new("RGB", (256, 256), (200, 30, 30)), gamma=1.0, cond_size=64)
    img = torch.rand(1, 3, 128, 128)
    stock = p.prepare_latents(img, 1, 16, 128, 128, torch.float32, torch.device("cpu"), None, None)
    assert "prepare_latents" not in p.__dict__
    with control_session(p, [spec]) as st:
        lat, img_lat, lat_ids, img_ids = p.prepare_latents(img, 1, 16, 128, 128, torch.float32, torch.device("cpu"),
                                                           None, None)
        assert st.n_base == lat.shape[1] + stock[1].shape[1] == 64 + 64
        assert img_lat.shape == (1, stock[1].shape[1] + 16, 64)
        assert img_ids.shape == (stock[3].shape[0] + 16, 3)
        assert torch.equal(img_ids[:stock[3].shape[0]], stock[3])
        cid = img_ids[stock[3].shape[0]:]
        assert (cid[:, 0] == 2).all()                                 # marcatore condizione = 2
        # passo posizioni = altezza latente (16) / (2*g con g=4) = 2.0
        assert torch.allclose(cid[:, 1].unique(), torch.tensor([0.0, 2.0, 4.0, 6.0]))
        assert torch.equal(img_lat[:, :stock[1].shape[1]], stock[1])
        assert lat_ids.shape == stock[2].shape
    assert "prepare_latents" not in p.__dict__ and not st.bank
