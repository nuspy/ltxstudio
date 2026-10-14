"""Attenzione sparsa Veda sul transformer MiniMax-H3 nativo (diffusers), senza ComfyUI.

Veda (https://veda-sparse.github.io, motore in workers/veda, MIT) è un predittore distillato che, per ogni tile di
query e ogni testa, sceglie i pochi tile di chiavi che portano il risultato; l'attenzione si calcola solo su quelli con
un kernel Triton INT8 block-sparse. I pesi del DiT non cambiano: LoRA, turbo fuso e quantizzazioni restano validi.

Aggancio (equivalente a `veda_comfy/comfy_patch.py`, che usa l'override d'attenzione di ComfyUI):
  * `transformer.forward` riceve per nome `position_ids` e `token_tags` del layout impacchettato
    `[testo | blocchi di condizione | audio target | video target]`: da lì si ricavano le span video (run contigue
    di token video) e la loro griglia (T, H, W), verificando che le righe siano davvero row-major — una span che non
    si verifica resta globale (attenzione piena), che è sempre corretto, solo più lento;
  * ogni blocco riceve un processor che rifà q/k/v come `MiniMaxH3AttnProcessor` (proiezione fusa, QK-norm, RoPE) e
    passa a `VedaEngine.attention`; tutto ciò che Veda non sa gestire (maschera, geometria non mappabile, kernel che
    non passa l'autotest, errore a metà) torna al processor ORIGINALE: il render non si perde mai.
"""
from __future__ import annotations

import collections
import functools
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from veda import backends as _backends  # noqa: E402
from veda import settings as _settings  # noqa: E402
from veda.core import bundle as _bundle  # noqa: E402
from veda.core import engine as _engine  # noqa: E402
from veda.core import h3_layout as _layout  # noqa: E402

VIDEO_TAG = 0          # token_tags di diffusers: 0 video, 1 testo, 2 audio


def video_runs(token_tags: torch.Tensor) -> list[tuple[int, int]]:
    """Run contigue [start, stop) di righe video nella sequenza impacchettata."""
    is_video = (token_tags.flatten() == VIDEO_TAG).to(torch.int8)
    if not bool(is_video.any()):
        return []
    edges = torch.diff(torch.nn.functional.pad(is_video, (1, 1)))
    starts = (edges == 1).nonzero().flatten().tolist()
    stops = (edges == -1).nonzero().flatten().tolist()
    return list(zip(starts, stops))


def layout_spec(position_ids: torch.Tensor, token_tags: torch.Tensor) -> _layout.LayoutSpec:
    """LayoutSpec di Veda dal layout diffusers: target = ultima run video, le altre sono riferimenti/condizioni.

    Raises:
        h3_layout.LayoutError: niente video, o target che non è una griglia row-major.
    """
    runs = video_runs(token_tags)
    if not runs:
        raise _layout.LayoutError("nessuna riga video nella sequenza")
    pos = position_ids.detach()
    a, b = runs[-1]
    grid = _layout._segment_grid(pos, a, b)                   # noqa: SLF001 — stessa verifica del nodo ComfyUI
    if grid is None:
        raise _layout.LayoutError(f"il video target [{a}:{b}] non è una griglia T,H,W row-major")
    refs, skipped = [], []
    for s, e in runs[:-1]:
        g = _layout._segment_grid(pos, s, e)                  # noqa: SLF001
        if g is None:
            skipped.append(f"riferimento [{s}:{e}] senza griglia row-major")
        else:
            refs.append(_layout.SpanSpec("ref_img", int(s), g))
    return _layout.LayoutSpec(seq_len=int(pos.shape[0]), target=_layout.SpanSpec("target", int(a), grid),
                              references=tuple(refs), skipped=tuple(skipped))


class VedaState:
    """Stato condiviso dai processor di un transformer: predittore, impostazioni, layout del forward corrente."""

    def __init__(self, bundle, settings, log=print):
        self.bundle, self.settings, self.log = bundle, settings, log
        self.spec = None
        self.plan = None
        self.failed: str | None = None
        self.calls = collections.Counter()
        self._engines: dict[str, object] = {}
        self._key = None
        self._announced: set = set()

    def say(self, key, text: str) -> None:
        if key not in self._announced:
            self._announced.add(key)
            self.log(f"[veda] {text}")

    def engine(self, device: torch.device):
        k = str(device)
        if k not in self._engines:
            res = _backends.resolve(device, notify=lambda m: self.log(f"[veda] {m}"), preferred="triton-int8")
            if res.backend is None:
                self.say("nokernel", f"nessun kernel sparse su questa GPU → attenzione piena\n{res.report()}")
                self._engines[k] = None
            else:
                self._engines[k] = _engine.VedaEngine(self.bundle, self.settings.generated,
                                                      self.settings.reference, res.backend, device)
                self.say("kernel", f"kernel {res.backend.display} pronto")
        return self._engines[k]

    def prepare(self, position_ids, token_tags) -> None:
        """Layout del forward corrente (ricalcolato solo quando cambia la sequenza)."""
        if position_ids is None or token_tags is None or self.failed:
            self.spec = None
            return
        key = (tuple(position_ids.shape), position_ids.data_ptr(), token_tags.data_ptr())
        if key == self._key:
            return
        self._key = key
        try:
            self.spec = layout_spec(position_ids, token_tags)
        except _layout.LayoutError as e:
            self.spec = None
            self.say(("layout", str(e)), f"layout non mappabile ({e}) → attenzione piena")
            return
        choice = self.bundle.plans.select(self.spec.target.grid)
        self.plan = choice.plan
        t, h, w = self.spec.target.grid
        self.say(("plan", self.spec.target.grid),
                 f"video {t}x{h}x{w} tile-grid · piano: {choice.how}"
                 f"{'' if choice.exact else ' (geometria non addestrata: approssimato)'} · "
                 f"riferimenti tiled: {len(self.spec.references)}"
                 + (f" · globali: {len(self.spec.skipped)}" if self.spec.skipped else ""))

    def summary(self) -> str:
        sparse = self.calls.get("sparse", 0)
        full = sum(n for r, n in self.calls.items() if r != "sparse")
        eng = [e for e in self._engines.values() if e is not None]
        frac = eng[0].stats.kept_fraction() if eng else None
        kept = f" · tile calcolati {100 * frac:.1f}%" if frac is not None else ""
        why = ", ".join(f"{r}={n}" for r, n in self.calls.items() if r != "sparse")
        return f"sparse {sparse}, piena {full}{kept}" + (f" ({why})" if why else "")


class VedaH3Processor:
    """Processor di un blocco: q/k/v come MiniMaxH3AttnProcessor, attenzione da Veda o dal processor originale."""

    def __init__(self, state: VedaState, layer: int, dense):
        self.state, self.layer, self.dense = state, layer, dense

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        st = self.state
        if st.spec is None or st.failed or attention_mask is not None or self.layer >= st.bundle.num_layers:
            st.calls["piena"] += 1
            return self.dense(attn, hidden_states, rotary_emb, attention_mask)
        if hidden_states.shape[1] != st.spec.seq_len:
            st.calls["altra sequenza"] += 1
            return self.dense(attn, hidden_states, rotary_emb, attention_mask)
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        if query.shape[2] != st.bundle.num_heads or query.shape[3] != st.bundle.head_dim:
            st.say("heads", f"il modello ha {query.shape[2]} teste da {query.shape[3]}, il predittore "
                            f"{st.bundle.num_heads}x{st.bundle.head_dim} → attenzione piena")
            st.calls["teste diverse"] += 1
            return self.dense(attn, hidden_states, rotary_emb, attention_mask)
        try:
            eng = st.engine(query.device)
            if eng is None:
                st.calls["nessun kernel"] += 1
                return self.dense(attn, hidden_states, rotary_emb, attention_mask)
            out = torch.stack([eng.attention(query[b], key[b], value[b], self.layer, st.spec, st.plan)
                               for b in range(query.shape[0])])
        except Exception as e:                                  # noqa: BLE001 — il render non si perde mai
            st.failed = f"{type(e).__name__}: {e}"
            st.log(f"[veda] errore nel percorso sparse, resto del job ad attenzione piena: {st.failed}")
            st.calls["errore"] += 1
            return self.dense(attn, hidden_states, rotary_emb, attention_mask)
        st.calls["sparse"] += 1
        hidden = out.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden))


def enable(transformer, predictor_path: str, *, generated: str = "32", reference: str = "32", log=print) -> VedaState:
    """Installa Veda su `transformer` (MiniMaxH3Transformer3DModel o sottoclasse). Ritorna lo stato (per il riepilogo).
    `generated`/`reference`: budget come il nodo ufficiale ("32" = 32 tile da 128 token, "90%" = salta il 90%).

    Raises:
        ValueError: predittore non leggibile o non compatibile col transformer (strati/teste/dimensione).
    """
    bundle = _bundle.load_bundle(predictor_path)
    blocks = transformer.transformer_blocks
    a0 = blocks[0].attn
    if (bundle.num_layers, bundle.num_heads, bundle.head_dim) != (len(blocks), a0.heads, a0.head_dim):
        raise ValueError(f"predittore Veda {bundle.num_layers}x{bundle.num_heads}x{bundle.head_dim} incompatibile "
                         f"col transformer {len(blocks)}x{a0.heads}x{a0.head_dim}")
    st = VedaState(bundle, _settings.VedaSettings(
        generated=_settings.parse_sparsity(generated, "generated"),
        reference=_settings.parse_sparsity(reference, "reference")), log=log)
    for i, blk in enumerate(blocks):
        blk.attn.set_processor(VedaH3Processor(st, i, blk.attn.processor))
    orig = transformer.forward

    # la pipeline modulare filtra gli argomenti del layout con inspect.signature(transformer.forward): il wrapper
    # deve esporre la firma ORIGINALE (functools.wraps → __wrapped__), o position_ids & co. vengono scartati
    @functools.wraps(orig)
    def forward(*args, **kwargs):
        st.prepare(kwargs.get("position_ids"), kwargs.get("token_tags"))
        return orig(*args, **kwargs)
    transformer.forward = forward
    transformer._veda_state = st
    log(f"[veda] attivo: {bundle.describe()} · budget video {generated} · riferimenti {reference}")
    return st
