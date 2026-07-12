"""Shim minimo per l'arch CodeFormer vendorizzata (da facerestore_cf, licenza S-Lab/Apache):
sostituisce le utility basicsr richieste dagli import senza dipendere dal pacchetto basicsr
(quello del venv non include vqgan_arch: è un fork del repo CodeFormer)."""
import logging


def get_root_logger(*_a, **_kw):
    return logging.getLogger("codeformer")


class _Registry:
    def register(self, obj=None, *_a, **_kw):
        if obj is None:                     # uso come decoratore con argomenti
            def deco(f):
                return f
            return deco
        return obj


ARCH_REGISTRY = _Registry()
