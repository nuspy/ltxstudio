"""MiniMax H3: offload a blocchi del DiT quando pesi + attivazioni non stanno nella VRAM libera.

Gli ultimi N blocchi di `transformer_blocks` restano in RAM e vanno in VRAM solo per il proprio forward (hook
pre/post): in VRAM c'è sempre al più un blocco ospite. N si calcola dal deficit misurato (core/h3_vram_plan),
non da soglie fisse: su una 5090 libera N = 0 e questo modulo non fa nulla.
Le LoRA dei moduli quantizzati vivono in liste Python (`m.loras`), non in buffer: si spostano a parte.
Uscire da un blocco butta la copia in VRAM e rimette il tensore di RAM: nessun device→host a ogni passo
(i pesi non cambiano mai in inferenza).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _tensor_bytes(t) -> int:
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


def block_bytes(blk: nn.Module) -> int:
    """Byte di pesi (parametri + buffer) e LoRA di un blocco."""
    n = sum(_tensor_bytes(p) for p in blk.parameters()) + sum(_tensor_bytes(b) for b in blk.buffers())
    for m in blk.modules():
        for lo in getattr(m, "loras", None) or []:
            n += sum(_tensor_bytes(t) for t in lo)
    return n


def move_block(blk: nn.Module, device) -> None:
    """Sposta pesi e LoRA del blocco su `device` (serve a metterlo in RAM la prima volta)."""
    blk.to(device)
    for m in blk.modules():
        loras = getattr(m, "loras", None)
        if loras:
            m.loras = [tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in lo) for lo in loras]


class _HostedBlock:
    """Tensori di RAM di un blocco; enter() porta in `device` una copia temporanea, leave() la butta."""

    def __init__(self, blk: nn.Module, device):
        self.device = device
        self.mods = list(blk.modules())
        self.host = [({k: p.data for k, p in m._parameters.items() if p is not None},
                      {k: b for k, b in m._buffers.items() if b is not None},
                      list(getattr(m, "loras", None) or [])) for m in self.mods]

    def enter(self) -> None:
        for m, (ps, bs, ls) in zip(self.mods, self.host):
            for k, t in ps.items():
                m._parameters[k].data = t.to(self.device, non_blocking=True)
            for k, t in bs.items():
                m._buffers[k] = t.to(self.device, non_blocking=True)
            if ls:
                m.loras = [tuple(t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                                 for t in lo) for lo in ls]

    def leave(self) -> None:
        for m, (ps, bs, ls) in zip(self.mods, self.host):
            for k, t in ps.items():
                m._parameters[k].data = t
            for k, t in bs.items():
                m._buffers[k] = t
            if ls:
                m.loras = list(ls)


def blocks_for_deficit(blocks, deficit_gb: float, max_frac: float = 0.85) -> int:
    """Quanti blocchi (dagli ultimi) servono per liberare `deficit_gb`. Mai più di max_frac dei blocchi."""
    blocks = list(blocks)
    if deficit_gb <= 0 or not blocks:
        return 0
    limit = int(len(blocks) * max_frac)
    got, n = 0.0, 0
    for b in reversed(blocks):
        if n >= limit or got >= deficit_gb:
            break
        got += block_bytes(b) / 2**30
        n += 1
    return n


class BlockOffloader:
    """Gestisce quali blocchi del DiT stanno in RAM. Si puo' chiamare di nuovo DURANTE l'inferenza
    (offload_bytes): se la VRAM non basta davvero (piu' roba residente di quanta il piano ne prevedesse) si
    liberano altri blocchi al volo, al posto di andare in OOM. Il blocco in esecuzione non si tocca."""

    def __init__(self, blocks, device="cuda", log=None):
        self.blocks = list(blocks)
        self.device = device
        self.log = log
        self.hosted: set = set()                       # indici in RAM
        self.max_blocks = max(0, len(self.blocks) - 2)  # almeno 2 blocchi sempre residenti

    def _offload_one(self, k: int) -> int:
        blk = self.blocks[k]
        freed = block_bytes(blk)
        move_block(blk, "cpu")
        host = _HostedBlock(blk, self.device)
        blk.register_forward_pre_hook(lambda m, args, kwargs, h=host: (h.enter(), (args, kwargs))[1],
                                      with_kwargs=True)
        blk.register_forward_hook(lambda m, args, out, h=host: (h.leave(), out)[1])
        self.hosted.add(k)
        return freed

    def offload_n(self, n: int, protect=None) -> int:
        """Mette in RAM altri `n` blocchi, partendo dagli ultimi, saltando `protect`. Ritorna i byte liberati."""
        freed, done = 0, 0
        for k in range(len(self.blocks) - 1, -1, -1):
            if done >= n or len(self.hosted) >= self.max_blocks:
                break
            if k in self.hosted or self.blocks[k] is protect:
                continue
            freed += self._offload_one(k)
            done += 1
        if done and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if done and self.log:
            self.log(f"offload: +{done} blocchi in RAM (ora {len(self.hosted)}/{len(self.blocks)}, "
                     f"{freed / 2**30:.1f} GB liberati in VRAM)")
        return freed

    def offload_bytes(self, target: int, protect=None) -> int:
        """Mette in RAM blocchi finche' ne ha liberati almeno `target` byte (o finiscono). Ritorna i byte liberati."""
        freed, done = 0, 0
        for k in range(len(self.blocks) - 1, -1, -1):
            if freed >= target or len(self.hosted) >= self.max_blocks:
                break
            if k in self.hosted or self.blocks[k] is protect:
                continue
            freed += self._offload_one(k)
            done += 1
        if done and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if done and self.log:
            self.log(f"offload dinamico: +{done} blocchi in RAM (ora {len(self.hosted)}/{len(self.blocks)}, "
                     f"{freed / 2**30:.1f} GB liberati in VRAM)")
        return freed


CURRENT: BlockOffloader | None = None           # l'offloader del DiT in uso (lo legge workers/h3_chunk)


def enable_block_offload(blocks, n: int, device="cuda", log=None) -> int:
    """Crea l'offloader del DiT (CURRENT) e mette in RAM gli ultimi `n` blocchi: in VRAM c'e' al piu' un blocco
    ospite per volta. Ritorna quanti ne ha messi. Anche con n = 0 l'offloader resta disponibile per l'offload
    dinamico."""
    global CURRENT
    CURRENT = BlockOffloader(blocks, device, log)
    n = max(0, min(int(n), len(CURRENT.blocks)))
    if n:
        CURRENT.offload_n(n)
    return len(CURRENT.hosted)


def free_gb_now() -> float:
    """VRAM libera adesso in GB (nel worker il contesto CUDA esiste già)."""
    return torch.cuda.mem_get_info()[0] / 2**30
