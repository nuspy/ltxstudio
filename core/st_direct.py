"""st_direct — safetensors a I/O DIRETTO (niente mmap) + misure di COMMIT Windows.

Perché: su Windows `safe_open(framework="pt")` + `UntypedStorage.from_file` creano DUE mappature
copy-on-write del file e impegnano ~2× la sua dimensione di COMMIT (RAM+pagefile). Con pagefile
piccolo o baseline alta il load di un checkpoint grosso crasha: access violation 0xC0000005,
"invalid python storage" o "os error 1455 paging file too small" (misurato: soglia di crash
≈ commit disponibile / 2 — il file è integro). Qui i tensori si leggono con seek+readinto:
commit di picco = UN tensore alla volta, mai l'intero file.

Solo stdlib + torch: importabile da TUTTI i venv (ltx2, inference_server, lingbot, comfy…).
"""
from __future__ import annotations

import ctypes
import json
import os
import struct

try:
    import torch
except Exception:                                       # pragma: no cover — venv senza torch
    torch = None


# dtype safetensors → torch (i quantizzati fp8 servono per LTX/FLUX fp8)
def _dtypes() -> dict:
    return {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
            "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
            "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
            "U8": torch.uint8, "BOOL": torch.bool}


# ── commit di memoria (Windows) ──────────────────────────────────────────────

class _MemStatus(ctypes.Structure):
    _fields_ = ([("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)]
                + [(n, ctypes.c_ulonglong) for n in
                   ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                    "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")])


def commit_avail_mb() -> float:
    """MB di COMMIT ancora disponibili (RAM+pagefile — ullAvailPageFile). Su questa classe di macchine
    è il limite VERO, non la RAM fisica: superarlo = access violation/1455, non un OOM pulito.
    Non-Windows o errore → 1e9 (nessun vincolo)."""
    try:
        m = _MemStatus()
        m.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPageFile / 2**20
    except Exception:                                   # noqa: BLE001 — linux/mac o kernel32 assente
        return 1e9


# ── lettura diretta ──────────────────────────────────────────────────────────

def read_header(path: str) -> tuple[dict, int]:
    """(header_json, offset_base_dati) senza mmap: 8 byte di lunghezza + JSON."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def metadata(path: str) -> dict:
    """Il blocco __metadata__ dell'header (equivalente no-mmap di safe_open().metadata())."""
    hdr, _ = read_header(path)
    return hdr.get("__metadata__") or {}


def read_tensor(fh, base: int, info: dict, dt_map: dict):
    """Legge UN tensore da un file già aperto (seek+readinto → commit di picco = questo tensore)."""
    a, b = info["data_offsets"]
    if b <= a:
        return torch.empty(info["shape"], dtype=dt_map[info["dtype"]])
    fh.seek(base + a)
    buf = bytearray(b - a)
    if fh.readinto(buf) != b - a:
        raise RuntimeError(f"lettura troncata a offset {base + a} ({b - a} byte)")
    return torch.frombuffer(buf, dtype=dt_map[info["dtype"]]).reshape(info["shape"])


def load_file(path: str, device: str = "cpu", want=None) -> dict:
    """Rimpiazzo di `safetensors.torch.load_file` a I/O diretto. `want`: iterabile di chiavi da
    caricare (None = tutte). I tensori vengono spostati su `device` uno alla volta."""
    dt = _dtypes()
    hdr, base = read_header(path)
    keep = set(want) if want is not None else None
    out: dict = {}
    with open(path, "rb") as fh:
        for name, info in hdr.items():
            if name == "__metadata__" or (keep is not None and name not in keep):
                continue
            t = read_tensor(fh, base, info, dt)
            out[name] = t if device in ("cpu", None) else t.to(device=device)
    return out


class _LazySlice:
    """Vista pigra su un tensore (equivalente di PySafeSlice): legge dal disco solo all'indexing."""

    def __init__(self, path: str, base: int, info: dict, dt: dict):
        self._path, self._base, self._info, self._dt = path, base, info, dt

    def get_shape(self):
        return list(self._info["shape"])

    def get_dtype(self):
        return self._info["dtype"]

    def __getitem__(self, idx):
        with open(self._path, "rb") as fh:
            t = read_tensor(fh, self._base, self._info, self._dt)
        return t[idx]


class SafeOpenShim:
    """Rimpiazzo no-mmap di `safetensors.safe_open(framework='pt')`: keys/metadata/get_tensor/
    get_slice a I/O diretto. Serve ai chiamanti che usano safe_open DIRETTAMENTE (es.
    ltx_core.block_streaming.builder) e che il patch di load_file non copre."""

    def __init__(self, path: str, framework: str = "pt", device: str = "cpu"):
        self._path = str(path)
        self._device = str(device)
        self._hdr, self._base = read_header(self._path)
        self._dt = _dtypes()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def keys(self):
        return [k for k in self._hdr if k != "__metadata__"]

    def metadata(self):
        return self._hdr.get("__metadata__") or {}

    def get_tensor(self, name: str):
        with open(self._path, "rb") as fh:
            t = read_tensor(fh, self._base, self._hdr[name], self._dt)
        return t if self._device in ("cpu", None) else t.to(device=self._device)

    def get_slice(self, name: str):
        return _LazySlice(self._path, self._base, self._hdr[name], self._dt)


def patch_safe_open() -> bool:
    """Monkey-patch di `safetensors.safe_open` con lo shim no-mmap (solo framework='pt'; gli altri
    framework ripiegano sull'originale). Opt-out: LTX_MMAP=1."""
    if os.environ.get("LTX_MMAP") == "1":
        return False
    try:
        import safetensors as _st
        _orig = _st.safe_open

        def _open(path, framework="pt", device="cpu", **kw):
            if str(framework) == "pt":
                return SafeOpenShim(path, framework, device)
            return _orig(path, framework=framework, device=device, **kw)

        _st.safe_open = _open
        return True
    except Exception:                                   # noqa: BLE001
        return False


def patch_safetensors_torch() -> bool:
    """Monkey-patch di `safetensors.torch.load_file` (usato da diffusers/transformers per i checkpoint
    single-file) con la versione a I/O diretto. Le API safe_open/per-shard NON sono toccate.
    Ritorna True se il patch è attivo. Opt-out: LTX_MMAP=1."""
    if os.environ.get("LTX_MMAP") == "1":
        return False
    try:
        import safetensors.torch as _st

        def _load_file(filename, device="cpu"):        # firma originale (filename, device)
            return load_file(str(filename), device=str(device))

        _st.load_file = _load_file
        return True
    except Exception:                                   # noqa: BLE001
        return False
