"""MiniMax H3: blocco del DiT calcolato a blocchi di token (picco di attivazioni da sequenza intera -> da blocco).

Perche': su 10 s a 1376x736 la sequenza e' di ~72k token. Nel blocco originale di diffusers ogni passaggio crea
tensori grandi quanto la sequenza: modulazione AdaLN (norm * (1+scale) + shift, con index_select di scale/shift e
promozione a fp32), qkv (3 x 7168 canali), q/k normalizzati + rotary, gate_up dell'MLP (2 x 14336 canali),
silu(g) e silu(g)*u. Il 30/10 la prova dal vivo e' andata in OOM proprio nell'MLP (1,93 GiB per silu(g)) con
28,47 GiB gia' allocati: pesi 13,7 GB + ~14,8 GB di attivazioni.

Qui il blocco e' riscritto con lo stesso calcolo (stesse operazioni, stesso ordine) ma tutto cio' che e' per-token
(modulazione, qkv, norme q/k, rotary, uscita dell'attenzione + residuo, MLP + residuo) gira a blocchi e scrive
in buffer preallocati. In memoria restano al massimo: il flusso residuo, Q/K/V pieni (servono all'attenzione) e
un blocco di temporanei. Il blocco e' scelto dalla VRAM LIBERA di adesso (mai una soglia fissa) e dimezzato da
solo se va in OOM. L'attenzione vera (flash/sage/sdpa) resta quella di diffusers (stesso backend) ma e' calcolata a blocchi di
QUERY con K/V pieni (esatto: ogni riga di query e' indipendente), cosi' non c'e' ne' un'uscita ne' un
workspace grandi quanto la sequenza; l'uscita di ogni blocco di query va subito in to_out + residuo.

VRAM dinamica: all'inizio di ogni blocco si verifica che ci sia posto per Q/K/V pieni + nuovo flusso residuo +
un blocco di temporanei; se non c'e' (piu' roba residente di quanta il piano prevedesse: la prova del 30/10 aveva
~11 GB non spiegati dal conto sul codice) si liberano al volo altri blocchi del DiT (workers/h3_offload.py) invece
di andare in OOM.

Se l'attenzione non usa il processore standard (es. Veda, che lo sostituisce) o le proiezioni non sono fuse,
l'attenzione passa dal processore originale e si bloccano solo le parti per-token attorno.
H3_BLOCK_CHUNK=0 disattiva tutto (blocco originale); H3_BLOCK_CHUNK=<n> forza n token per blocco (test).
"""
from __future__ import annotations

import functools
import os

import torch

MIN_CHUNK = 128
_CAP: dict = {}                 # device -> tetto (token) appreso dopo un OOM


def chunk_tokens(rows: int, row_bytes: int, device) -> int:
    """Token per blocco: i temporanei di un blocco restano sotto 1/5 della VRAM libera, tra 256 MB e 1,5 GiB."""
    forced = os.environ.get("H3_BLOCK_CHUNK")
    if forced and forced.isdigit() and int(forced) > 0:
        return max(1, min(rows, int(forced)))
    dev = torch.device(device)
    if dev.type != "cuda":
        return rows if rows <= 32768 else 32768
    try:
        free = torch.cuda.mem_get_info(dev)[0]
    except Exception:  # noqa: BLE001
        free = 2 << 30
    budget = max(256 << 20, min(int(1.5 * 2**30), free // 5))
    n = max(MIN_CHUNK * 4, budget // max(1, row_bytes))
    cap = _CAP.get(str(dev))
    if cap:
        n = min(n, cap)
    return min(rows, n)


def each_chunk(n: int, ch: int, fn, device=None) -> None:
    """fn(i, j) su [0, n) a blocchi di `ch`. Va in OOM -> dimezza quel blocco (fn deve essere idempotente: scrive
    solo in buffer indicizzati da [i, j)) e ricorda il tetto appreso. A MIN_CHUNK token senza spazio risolleva."""
    def go(i: int, j: int) -> None:
        try:
            fn(i, j)
        except torch.OutOfMemoryError:
            if j - i <= MIN_CHUNK:
                raise
            if device is not None and torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
                _CAP[str(torch.device(device))] = max(MIN_CHUNK, (j - i) // 2)
            m = (i + j) // 2
            go(i, m)
            go(m, j)
    for i in range(0, n, ch):
        go(i, min(n, i + ch))


CAP_BYTES = None            # tetto per-processo impostato dal worker (set_per_process_memory_fraction)
LOG = None                  # funzione di log del worker
_TRACE = {"on": True}


def headroom_bytes(dev) -> float:
    """Byte che si possono ancora allocare rispettando sia il tetto per-processo sia la VRAM libera del
    driver. Lo slack gia' riservato ma non usato conta al 50 % (frammentazione: Windows non ha
    expandable_segments)."""
    d = torch.device(dev)
    if d.type != "cuda":
        return float("inf")
    alloc, res = torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d)
    cap = CAP_BYTES or torch.cuda.get_device_properties(d).total_memory
    drv_free = torch.cuda.mem_get_info(d)[0]
    return 0.5 * (res - alloc) + max(0, min(cap - res, drv_free))


def ensure_headroom(need: int, dev, protect=None) -> bool:
    """Garantisce `need` byte di margine: prima rilascia la cache dell'allocatore, poi mette in RAM altri blocchi
    del DiT (mai quello in esecuzione). False se non si riesce."""
    if headroom_bytes(dev) >= need:
        return True
    d = torch.device(dev)
    torch.cuda.empty_cache()                      # no-op se CUDA non e' inizializzato
    h = headroom_bytes(d)
    if h >= need:
        return True
    try:
        import h3_offload as O
    except ImportError:
        return False
    off = O.CURRENT
    if off is None:
        return False
    off.offload_bytes(int(need - h), protect=protect)
    return headroom_bytes(d) >= need


def _trace(k: int, last: int, dev) -> None:
    """Primo passaggio: memoria dopo alcuni blocchi (dato di calibrazione per le prossime prove)."""
    if not _TRACE["on"] or LOG is None or torch.device(dev).type != "cuda":
        return
    if k in (0, 1, 2) or k % 10 == 0 or k == last:
        LOG(f"[mem] blocco {k}: allocata {torch.cuda.memory_allocated() / 2**30:.1f} GB, "
            f"picco {torch.cuda.max_memory_allocated() / 2**30:.1f}, riservata {torch.cuda.memory_reserved() / 2**30:.1f}")
    if k == last:
        _TRACE["on"] = False


def _modulate(norm, h, scale, shift, idx):
    # identico a diffusers: norm(h) * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)
    return norm(h) * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)


def _row_bytes(blk, C: int) -> int:
    """Byte per token dei temporanei di un blocco, per eccesso (si assume modulazione in fp32)."""
    gu = getattr(getattr(blk, "ff", None), "gate_up", None)
    f = (getattr(gu, "out_features", 0) // 2) or 4 * C
    return 4 * (8 * C + 4 * f)


def block_forward(blk, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    """Sostituisce MiniMaxH3TransformerBlock.forward. Stesso risultato, picco per blocco."""
    k_idx, k_last = getattr(blk, "_h3_idx", -1), getattr(blk, "_h3_last", -2)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaln_proj(temb)
    B, N, C = hidden_states.shape
    dev = hidden_states.device
    ch = chunk_tokens(N, _row_bytes(blk, C), dev)
    attn, proc = blk.attn, blk.attn.processor
    fused = bool(getattr(attn, "fused_projections", False)) and type(proc).__name__ == "MiniMaxH3AttnProcessor"
    # margine per Q/K/V pieni + nuovo flusso residuo (nel caso peggiore fp32) + un blocco di temporanei + 0,3 GiB
    # di lavoro del kernel di attenzione
    inner = int(getattr(attn, "inner_dim", C))
    ensure_headroom(int(1.15 * N * (3 * inner * 2 + C * 4) + ch * _row_bytes(blk, C) + 0.3 * 2**30), dev, protect=blk)

    # ── attenzione ──────────────────────────────────────────────────────────────────────────────
    if fused:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        qkv: list = []

        def stage_qkv(i, j):
            nh = _modulate(blk.norm1, hidden_states[:, i:j], scale_msa, shift_msa, adaln_indices[i:j])
            q, k, v = attn.to_qkv(nh).chunk(3, dim=-1)
            q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
            k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
            v = v.unflatten(-1, (attn.heads, -1))
            if rotary_emb is not None:
                q = _apply_rotary_emb(q, rotary_emb[0][i:j], rotary_emb[1][i:j])
                k = _apply_rotary_emb(k, rotary_emb[0][i:j], rotary_emb[1][i:j])
            if not qkv:
                qkv.extend(torch.empty((B, N) + t.shape[2:], dtype=t.dtype, device=dev) for t in (q, k, v))
            qkv[0][:, i:j] = q
            qkv[1][:, i:j] = k
            qkv[2][:, i:j] = v

        each_chunk(N, ch, stage_qkv, dev)
        q, k, v = qkv
        backend, pconf = getattr(proc, "_attention_backend", None), getattr(proc, "_parallel_config", None)
        hs: list = []

        def stage_attn(i, j):
            # query a blocchi, K/V pieni: ogni riga di query e' indipendente -> esatto. L'uscita del blocco va
            # subito in to_out + residuo: non esiste mai un'uscita dell'attenzione grande quanto la sequenza.
            o = dispatch_attention_fn(q[:, i:j], k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                                      backend=backend, parallel_config=pconf)
            o = attn.to_out[1](attn.to_out[0](o.flatten(2, 3).type_as(q)))
            c = hidden_states[:, i:j] + gate_msa.index_select(0, adaln_indices[i:j]) * o
            if not hs:
                hs.append(torch.empty((B, N, C), dtype=c.dtype, device=dev))
            hs[0][:, i:j] = c

        qch = max(ch, int(os.environ.get("H3_ATTN_QCHUNK", "8192")))
        each_chunk(N, min(N, qch), stage_attn, dev)
        del q, k, v, qkv
        a = None
    else:
        nhb: list = []

        def stage_norm(i, j):
            nh = _modulate(blk.norm1, hidden_states[:, i:j], scale_msa, shift_msa, adaln_indices[i:j])
            if not nhb:
                nhb.append(torch.empty((B, N, C), dtype=nh.dtype, device=dev))
            nhb[0][:, i:j] = nh

        each_chunk(N, ch, stage_norm, dev)
        a = attn(nhb[0], rotary_emb, attention_mask)
        del nhb

    if not fused:
        hs = []

        def stage_out(i, j):                       # `a` e' gia' passato per to_out dal processore originale
            c = hidden_states[:, i:j] + gate_msa.index_select(0, adaln_indices[i:j]) * a[:, i:j]
            if not hs:
                hs.append(torch.empty((B, N, C), dtype=c.dtype, device=dev))
            hs[0][:, i:j] = c

        each_chunk(N, ch, stage_out, dev)
        del a
    x = hs[0]

    # ── MLP ─────────────────────────────────────────────────────────────────────────────────────
    out: list = []

    def stage_ff(i, j):
        h_c = x[:, i:j]
        nh = _modulate(blk.norm2, h_c, scale_mlp, shift_mlp, adaln_indices[i:j])
        c = h_c + gate_mlp.index_select(0, adaln_indices[i:j]) * blk.ff(nh)
        if not out:
            out.append(x if c.dtype == x.dtype else torch.empty((B, N, C), dtype=c.dtype, device=dev))
        out[0][:, i:j] = c            # in place su x se il dtype coincide: la fetta i:j non serve piu'

    each_chunk(N, ch, stage_ff, dev)
    _trace(k_idx, k_last, dev)
    return out[0]


def enable_chunked_blocks(tr, log=None) -> int:
    """Aggancia block_forward a ogni blocco del DiT. Ritorna quanti. H3_BLOCK_CHUNK=0 -> nessuno."""
    if os.environ.get("H3_BLOCK_CHUNK") == "0":
        return 0
    global LOG
    LOG = log
    blocks = list(getattr(tr, "transformer_blocks", []) or [])
    for k, blk in enumerate(blocks):
        blk._h3_idx, blk._h3_last = k, len(blocks) - 1
        blk.forward = functools.partial(block_forward, blk)
    if log and blocks:
        log(f"blocchi del DiT a token-chunk: {len(blocks)} (qkv, norme, rotary, uscita attenzione, MLP)")
    return len(blocks)
