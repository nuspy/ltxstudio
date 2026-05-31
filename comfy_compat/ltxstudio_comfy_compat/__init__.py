"""LTX Studio — ComfyUI Compat Shim.

Custom node che NON aggiunge nodi: all'avvio di ComfyUI applica monkey-patch difensivi
per il supporto LTX-2.3 (in passato fatti modificando i file core di ComfyUI). Così LTX-2.3
funziona su qualsiasi versione/variante di ComfyUI senza toccare i file core, e gli update
restano possibili. Vedi patches.py.
"""
try:
    from .version import COMPAT_VERSION
except Exception:
    COMPAT_VERSION = "?"

print("=" * 64)
print(f"[ltxstudio-compat] LTX Studio ComfyUI Compat Shim v{COMPAT_VERSION}")
try:
    from . import patches
    for _name, _status in patches.apply_all_patches():
        print(f"[ltxstudio-compat]   {_name}: {_status}")
except Exception as _e:  # non deve MAI impedire l'avvio di ComfyUI
    import traceback
    print(f"[ltxstudio-compat] ERRORE GLOBALE (ComfyUI prosegue): {_e}")
    traceback.print_exc()
print("=" * 64)

# Nessun nodo: lo shim agisce solo via patch all'import.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
