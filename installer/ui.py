"""Output dell'installer: riepilogo prerequisiti + piano download con DIMENSIONI."""
from __future__ import annotations


def _sz(mb: int) -> str:
    return f"{mb/1024:.1f} GB" if mb >= 1024 else f"{mb} MB"


def print_prereqs(pr: dict) -> None:
    print("Prerequisiti:")
    for k in ("python", "nvidia", "torch", "ffmpeg"):
        v = pr.get(k, {})
        mark = "✓" if v.get("ok") else "✗"
        print(f"  [{mark}] {v.get('label', k)}: {v.get('detail', '')}")
    cr = pr.get("comfy_root")
    print(f"  [{'✓' if cr else '✗'}] ComfyUI: {cr or 'NON trovato (verrà chiesto)'}")


def print_plan(plan: dict) -> None:
    print("\nPiano di installazione (download):")
    print("  Custom nodes:")
    for n in plan["nodes"]:
        print(f"    - {n['label']:42s} {_sz(n.get('size_mb', 0))}")
    for d in plan["deps"]:
        print(f"    - {d['label']:42s} {_sz(d.get('size_mb', 0))}")
    print("  Modelli BASE dell'edizione (scaricati ora):")
    for m in plan["base_models"]:
        tag = " 🔞" if m.get("sensitivity") == "adult" else ""
        print(f"    - {m['label']:42s} {_sz(m.get('size_mb', 0))}{tag}")
    print("  Modelli al PRIMO USO (non ora):")
    for m in plan["on_demand"]:
        print(f"    - {m['label']:42s} {_sz(m.get('size_mb', 0))}")
    print(f"\n  TOTALE da scaricare ora: ~{_sz(plan['total_mb'])}")


def confirm(plan: dict) -> bool:
    try:
        ans = input(f"\nProcedo con ~{_sz(plan['total_mb'])} di download? [s/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("s", "si", "y", "yes")
