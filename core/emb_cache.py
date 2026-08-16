"""emb_cache — cache su disco degli embedding di prompt LTX (output dell'EmbeddingsProcessor).

PERCHÉ. Ogni chiamata `pipe()` di LTX ricostruisce il text encoder da zero: `PromptEncoder.__call__`
costruisce Gemma (24,6 GiB), encoda, e il context manager `gpu_model` lo rimanda su "meta" all'uscita.
Il parcheggio dei pesi in RAM (`_CpuParkRegistry`) per Gemma fallisce quasi sempre per commit
insufficiente, quindi si rilegge dall'SSD: ~13 s a encode, misurati. Un video lungo a 6 chunk paga
6 volte lo stesso encode, un film 28, un videoclip 36 — con lo stesso prompt o con prompt noti in
anticipo. Qui si cacha il RISULTATO (24 MiB per coppia positivo+negativo) invece dei pesi (24,6 GiB):
sta in un budget minuscolo e non tocca il commit, che è il vincolo che fa fallire l'altra strada.

COSA si cacha: l'output FINALE, cioè dopo `EmbeddingsProcessor.process_hidden_states` — così un hit
salta sia Gemma sia l'EmbeddingsProcessor (6 GB). Ogni voce è la LISTA ORDINATA di output di UNA
chiamata, non il singolo prompt: `GemmaTextEncoder.encode` fa un forward fuso su `[N,1024]` e poi
affetta le righe, quindi un tensore prodotto con una composizione di batch diversa potrebbe differire
negli ultimi bit (kernel cuBLAS diversi). Legittimo, ma incompatibile con la garanzia "hit ≡ miss".

GRANULARITÀ DELLA CHIAVE: tutto ciò che cambia l'output, e nient'altro. Prompt, file del text encoder,
file dell'embeddings processor (che vengono dal CHECKPOINT, non dal TE: `blocks.py` costruisce il
Builder dell'EP con `model_path=checkpoint_path`), tokenizer, dtype, release, versioni di
libreria/torch/GPU. NON entrano seed, risoluzione, step, LoRA, guidance: non arrivano mai al
PromptEncoder → "stesso prompt, altro seed" è un HIT, che è il ciclo creativo più frequente.

SICUREZZA. Un falso hit produrrebbe un video sbagliato in silenzio, quindi la lettura è difensiva a
tre livelli (header, dimensione, crc32 per tensore) e ogni fallimento è un MISS, mai un errore.

stdlib-only all'import (`torch` solo dentro get/put, che la GUI non chiama mai): importabile da OGNI
venv e dal processo GUI, che non deve mai creare un contesto CUDA.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
import zlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DIR = _ROOT / "data" / "emb_cache"
SCHEMA = "1"
_FP_CHUNK = 1 << 20                                      # 1 MiB di testa e di coda per l'impronta


# ─────────────────────────── configurazione ───────────────────────────

def enabled() -> bool:
    """Cache attiva? Scappatoia CLI/test `LS_EMB_CACHE=0`. La preferenza utente viaggia nei params
    del job (non in env): il server persistente non erediterebbe una variabile impostata dopo il
    suo avvio."""
    return os.environ.get("LS_EMB_CACHE", "1").strip() != "0"


def cache_dir() -> Path:
    d = Path(os.environ.get("LS_EMB_CACHE_DIR") or _DEFAULT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def budget_mb() -> float:
    try:
        from core import mem_budget
        return mem_budget.emb_cache_mb()
    except Exception:                                    # noqa: BLE001 — venv senza core.mem_budget
        return float(os.environ.get("LS_EMB_CACHE_MB", 4096))


# ─────────────────────────── chiave ───────────────────────────

def file_fp(paths) -> list:
    """Impronta di un insieme di file-peso: (nome, dimensione, mtime_ns, crc del primo e dell'ultimo
    MiB). I due crc catturano anche la sostituzione in place che preserva dimensione e data — il caso
    che renderebbe stantio un tensore cachato senza che nessuno se ne accorga."""
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = []
    for p in paths:
        p = str(p)
        try:
            if os.path.isdir(p):                         # radice HF: contano i pesi e la config
                names = sorted(n for n in os.listdir(p)
                               if n.endswith((".safetensors", ".bin", ".json", ".model")))
                out.append([p, "dir", [_fp_one(os.path.join(p, n)) for n in names]])
            else:
                out.append(_fp_one(p))
        except OSError as e:
            out.append([p, "err", str(e)])               # file sparito → chiave diversa, corretto
    return out


def _fp_one(path: str) -> list:
    st = os.stat(path)
    head = tail = 0
    try:
        with open(path, "rb") as fh:
            head = zlib.crc32(fh.read(_FP_CHUNK))
            if st.st_size > _FP_CHUNK:
                fh.seek(max(0, st.st_size - _FP_CHUNK))
                tail = zlib.crc32(fh.read(_FP_CHUNK))
    except OSError:
        pass
    return [os.path.basename(path), st.st_size, st.st_mtime_ns, head, tail]


def key_for(material: dict) -> str:
    """sha256 (32 hex) del materiale della chiave, serializzato in modo stabile."""
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def path_for(key: str) -> Path:
    return cache_dir() / f"{key}.safetensors"


# ─────────────────────────── lettura header (stdlib pura, per la GUI) ───────────────────────────

def _header(path) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


# ─────────────────────────── get / put (usano torch) ───────────────────────────

def get(key: str, device: str = "cpu"):
    """La lista di tuple `(video_encoding, audio_encoding|None, attention_mask)` della voce, oppure
    None se assente//inutilizzabile. Qualunque anomalia = MISS silenzioso: meglio riencodare 13 s che
    servire un tensore sbagliato."""
    p = path_for(key)
    if not p.exists():
        return None
    try:
        from core import st_direct
        hdr, base = _header(p)
        meta = hdr.get("__metadata__") or {}
        if meta.get("schema") != SCHEMA:
            return _drop(p, "schema diverso")
        n = int(meta.get("n", 0))
        if n <= 0:
            return _drop(p, "n non valido")
        end = max(i["data_offsets"][1] for k, i in hdr.items() if k != "__metadata__")
        if p.stat().st_size != base + end:
            return _drop(p, "dimensione non corrispondente")
        tens = st_direct.load_file(str(p))
        out = []
        for i in range(n):
            trio = []
            for part in ("video", "audio", "mask"):
                name = f"p{i}.{part}"
                t = tens.get(name)
                if t is None:
                    if part == "audio":                  # audio_encoding è legittimamente opzionale
                        trio.append(None)
                        continue
                    return _drop(p, f"tensore mancante: {name}")
                exp = meta.get(f"crc.{name}")
                if exp is not None and str(zlib.crc32(st_direct.tensor_bytes(t))) != exp:
                    return _drop(p, f"crc32 non corrisponde: {name}")
                trio.append(t if device in ("cpu", None) else t.to(device=device))
            out.append(tuple(trio))
        os.utime(p, None)                                # mtime = last-used per l'LRU
        return out
    except Exception as e:                               # noqa: BLE001
        return _drop(p, f"{type(e).__name__}: {e}")


def _drop(p: Path, why: str):
    print(f"[embcache] SCARTO {p.stem[:8]}: {why} → riencodo", flush=True)
    try:
        p.unlink()
    except OSError:
        pass
    return None


def put(key: str, outputs, prompts: list, meta: dict | None = None) -> bool:
    """Scrive la voce. Ritorna True se il file è ora presente (anche se scritto da un altro processo).

    Niente lock: il contenuto è deterministico data la chiave, quindi due processi che corrono
    scrivono gli stessi byte. tmp per-processo + `os.replace`; su Windows il replace fallisce se un
    lettore tiene aperta la destinazione (`open()` non concede FILE_SHARE_DELETE) → in quel caso si
    butta il tmp e si prosegue: perdere una scrittura è irrilevante, piantare un job no."""
    p = path_for(key)
    if p.exists():
        return True
    try:
        from core import st_direct
        tens: dict = {}
        md = {"schema": SCHEMA, "n": str(len(outputs)), "ts": str(int(time.time()))}
        for k, v in (meta or {}).items():
            md[str(k)] = str(v)
        for i, trio in enumerate(outputs):
            video, audio, mask = trio[0], trio[1], trio[2]
            md[f"p{i}"] = str(prompts[i]) if i < len(prompts) else ""
            for part, t in (("video", video), ("audio", audio), ("mask", mask)):
                if t is None:
                    continue
                name = f"p{i}.{part}"
                t = t.detach().contiguous().cpu()
                tens[name] = t
                md[f"crc.{name}"] = str(zlib.crc32(st_direct.tensor_bytes(t)))
        tmp = p.with_suffix(f".tmp.{os.getpid()}.{int(time.time() * 1000) % 100000}")
        st_direct.save_file(str(tmp), tens, md)
        try:
            os.replace(tmp, p)
        except OSError as e:                             # destinazione aperta da un lettore
            print(f"[embcache] scrittura saltata ({type(e).__name__}) — la voce resta com'è", flush=True)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return p.exists()
        gc()
        return True
    except Exception as e:                               # noqa: BLE001 — la cache non deve mai fermare un job
        print(f"[embcache] scrittura fallita: {type(e).__name__}: {e}", flush=True)
        return False


# ─────────────────────────── manutenzione / GUI ───────────────────────────

def entries() -> list[dict]:
    """Le voci con dimensione, ultimo uso e prompt — solo os.stat e header, nessun torch."""
    out = []
    for f in cache_dir().glob("*.safetensors"):
        try:
            st = f.stat()
            md = (_header(f)[0].get("__metadata__") or {})
            out.append({"key": f.stem, "mb": st.st_size / 2**20, "used": st.st_mtime,
                        "prompt": (md.get("p0") or "")[:120],
                        "encode_s": float(md.get("encode_s", 0) or 0)})
        except Exception:                                # noqa: BLE001 — voce illeggibile: la ignoro qui
            continue
    return sorted(out, key=lambda e: e["used"], reverse=True)


def stats() -> dict:
    n = mb = 0
    for f in cache_dir().glob("*.safetensors"):
        try:
            mb += f.stat().st_size / 2**20
            n += 1
        except OSError:
            continue
    return {"n": n, "mb": mb, "budget_mb": budget_mb(), "dir": str(cache_dir())}


def gc(budget: float | None = None) -> int:
    """Sfratta le voci meno usate finché la cache non rientra nel budget. Ritorna quante ne ha tolte.
    LRU su mtime, aggiornato a ogni hit da `get()`."""
    bud = budget_mb() if budget is None else budget
    items = []
    tot = 0.0
    for f in cache_dir().glob("*.safetensors"):
        try:
            st = f.stat()
        except OSError:
            continue
        items.append((f, st.st_size / 2**20, st.st_mtime))
        tot += st.st_size / 2**20
    if tot <= bud:
        return 0
    need = tot - bud
    try:
        from core import mem_budget
        victims = mem_budget.pick_lru_victims([(f, mb, used) for f, mb, used in items], need)
    except Exception:                                    # noqa: BLE001
        victims = [f for f, _mb, _u in sorted(items, key=lambda e: e[2])]
    freed, n = 0.0, 0
    for f in victims:
        if freed >= need:
            break
        try:
            mb = f.stat().st_size / 2**20
            f.unlink()
            freed += mb
            n += 1
        except OSError:
            continue
    if n:
        print(f"[embcache] LRU: sfrattate {n} voci ({freed:.0f}MB) per rientrare in {bud:.0f}MB",
              flush=True)
    return n


def clear() -> int:
    """Svuota la cache. Ritorna il numero di voci rimosse."""
    d = cache_dir()
    n = 0
    for f in list(d.glob("*.safetensors")) + list(d.glob("*.tmp.*")):
        try:
            f.unlink()
            n += 1
        except OSError:
            continue
    return n


def sweep_tmp(max_age_s: float = 3600) -> int:
    """Rimuove i temporanei orfani (processi uccisi a metà scrittura)."""
    now = time.time()
    n = 0
    for f in cache_dir().glob("*.tmp.*"):
        try:
            if now - f.stat().st_mtime > max_age_s:
                f.unlink()
                n += 1
        except OSError:
            continue
    return n


if __name__ == "__main__":                               # diagnostica, come core/vram_ledger
    import sys
    if "--clear" in sys.argv:
        print(f"rimosse {clear()} voci")
    s = stats()
    print(f"emb_cache: {s['n']} voci · {s['mb']:.0f}/{s['budget_mb']:.0f} MB · {s['dir']}")
    free = shutil.disk_usage(s["dir"]).free / 2**30
    print(f"disco libero: {free:.1f} GB")
    for e in entries()[:20]:
        print(f"  {e['key'][:8]}  {e['mb']:6.1f}MB  {time.strftime('%d/%m %H:%M', time.localtime(e['used']))}"
              f"  encode {e['encode_s']:.1f}s  {e['prompt'][:60]}")
