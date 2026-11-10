"""Attenzione INT8 di comfy_kitchen per il DiT di H3 (come il nodo «Comfy Kitchen Attention» dell'originale ComfyUI: Q/K/V int8, softmax in fp32).

Si installa al posto del backend `native` di diffusers (`dispatch_attention_fn` lo usa quando nessun backend e scelto). Solo se richiesto (`h3_int8_attention`,
casella della GUI, spenta di default): ogni chiamata che il kernel non copre (maschera, causale, GPU senza kernel) passa al riferimento SDPA, e il primo risultato
si confronta con SDPA su un campione di righe di query: se l'errore e alto si scarta con il motivo scritto nel log."""
from __future__ import annotations

import torch.nn.functional as F

import _qlinear as qlinear

INT8_ATTENTION = qlinear.FastGemm("attenzione int8", "H3_INT8_ATTENTION", True, 0.15, (7, 5))
SAMPLE_QUERIES = 64


def reference_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, return_lse=False, _parallel_config=None):
    """SDPA nativo, con lo stesso formato [batch, sequenza, teste, dimensione] del backend `native` di diffusers."""
    out = F.scaled_dot_product_attention(query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3), attn_mask=attn_mask,
                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    return out.permute(0, 2, 1, 3)


def int8_attention_backend(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, return_lse=False, _parallel_config=None):
    """Backend di attenzione: int8 di comfy_kitchen dove il kernel copre il caso, altrimenti il riferimento (il motivo dello scarto e gia nel log)."""
    plain = attn_mask is None and not is_causal and dropout_p == 0.0 and not return_lse
    if not (plain and INT8_ATTENTION.usable(query.device)):
        return reference_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, return_lse, _parallel_config)
    out = _int8_attention(query, key, value, scale)
    if out is None:
        return reference_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, return_lse, _parallel_config)
    return out


def _int8_attention(query, key, value, scale):
    import comfy_kitchen as ck
    if not ck.int8_attention_is_available(query.device):
        INT8_ATTENTION.disable(f"kernel di attenzione int8 non disponibile per questa GPU (int8_attention_backend, q {tuple(query.shape)})")
        return None
    try:
        out = ck.int8_attention(query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3), scale=scale).permute(0, 2, 1, 3)
    except Exception as error:                                # noqa: BLE001  qualunque errore del kernel: si scarta il percorso, con il motivo nel log
        INT8_ATTENTION.disable(f"{type(error).__name__}: {error} (int8_attention_backend, q {tuple(query.shape)})")
        return None
    if not INT8_ATTENTION.verified:
        sample = reference_attention(query[:, :SAMPLE_QUERIES], key, value, scale=scale)
        if not INT8_ATTENTION.verify(out[:, :SAMPLE_QUERIES], sample, f"q {tuple(query.shape)}"):
            return None
    return out


def install() -> None:
    """Sostituisce il backend `native` di diffusers con quello int8 (idempotente). I casi non coperti dal kernel e le GPU senza kernel passano al riferimento, con il motivo nel log."""
    from diffusers.models import attention_dispatch as dispatch
    backend = dispatch.AttentionBackendName.NATIVE
    if dispatch._AttentionBackendRegistry._backends.get(backend) is int8_attention_backend:
        return
    dispatch._AttentionBackendRegistry.register(backend)(int8_attention_backend)
    qlinear.log_line("attenzione int8 (comfy_kitchen) richiesta: installata al posto del backend native; si verifica sul primo passo")
