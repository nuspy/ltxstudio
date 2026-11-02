"""MiniMax H3: attenzione del text encoder (Qwen3-VL) a BLOCCHI DI QUERY — esatta, per evitare i picchi da L x L.

Giro OJ del 02/11 (Sostituisci personaggio): `torch.OutOfMemoryError: Tried to allocate 9.08 GiB` dentro
`modeling_qwen3_vl.py:536 attention_interface -> sdpa_attention_forward -> scaled_dot_product_attention` del DECODER (language model) del TE:
con foto + video di riferimento la sequenza e di ~8,7k token e, con la maschera esplicita, SDPA ricade sul percorso «math» che materializza
[1, 64 teste, L, L] in una volta (64 * L^2 * 2 byte). Qui la funzione `sdpa` registrata in transformers e avvolta: se lo stimato supera la
soglia, la query si spezza in blocchi di righe (ogni riga di query e indipendente: risultato identico) con la fetta della maschera (o la
maschera causale costruita al volo, perche `is_causal` su un blocco di query allineerebbe male l'angolo).
Si installa SOLO attorno alla fase del text encoder e si toglie subito dopo.
"""
from __future__ import annotations

import contextlib

import torch

DEFAULT_MAX_BYTES = 768 * 2 ** 20            # sopra questa stima per l'intera matrice dei punteggi si spezza (poi blocchi da ~512 MB)


def _chunked_sdpa(orig, max_bytes: int):
    def fn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        b, h, lq, d = query.shape
        lk = key.shape[2]
        est = b * h * lq * lk * query.element_size()
        if est <= max_bytes or lq <= 1:
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
        rows = max(16, int(max_bytes // (b * h * lk * query.element_size())))
        causal = bool(is_causal) if is_causal is not None else bool(getattr(module, "is_causal", True) and attention_mask is None and lq > 1)
        outs = []
        for s in range(0, lq, rows):
            e = min(lq, s + rows)
            if attention_mask is not None:
                m = attention_mask[:, :, s:e, :lk] if attention_mask.dim() == 4 else attention_mask
            elif causal:
                # riga i (di lq) vede le chiavi j <= i + (lk - lq): stesso allineamento in basso a destra della maschera causale piena
                qi = torch.arange(s, e, device=query.device)[:, None] + (lk - lq)
                m = (torch.arange(lk, device=query.device)[None, :] <= qi)[None, None]
            else:
                m = None
            o, _w = orig(module, query[:, :, s:e], key, value, m, dropout=dropout, scaling=scaling, is_causal=False, **kwargs)
            outs.append(o)
        return torch.cat(outs, dim=1), None
    return fn


@contextlib.contextmanager
def query_chunked_attention(max_bytes: int = DEFAULT_MAX_BYTES):
    """Dentro il blocco, l'attenzione `sdpa` di transformers e a blocchi di query. Se transformers non ha il registro, non fa nulla."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:                                        # noqa: BLE001
        yield False
        return
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _chunked_sdpa(orig, max_bytes)
    try:
        yield True
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = orig
