"""Caso LIVE Mondo 3D — generazione REALE end-to-end dalla GUI (t2s → mesh GLB).

File separato col prefisso `cases_zz_` così i casi si accodano in FONDO all'ordine
alfabetico globale e NON spostano gli indici --start/--limit delle categorie esistenti
(stessa ragione di cases_zz_lora_image_apply).

La categoria `world` (cases_world_tab.py) copre SOLO GUI/guardie: nessun caso generava
una scena vera, quindi 22/22 PASS senza toccare la GPU (emerso nella review del 12/07).
Questo caso colma il buco: prompt creativo → WorldGen (nunchaku fp4, --low_vram) →
verifica che il mesh.glb esista e non sia vuoto. Solo run reale (in mock skippa: il
subprocess GPU dura minuti e non c'è niente da mockare).
"""
from __future__ import annotations

import time

import harness as H

CASES = []


def case_world_gen_live(w, spy):
    """Mondo 3D REALE: t2s 'mercato notturno galleggiante' → mesh.glb (~5 min GPU)."""
    if not getattr(H, "REAL_MODE", False):
        return                                   # solo run reale
    from core import world_service as ws
    if not ws.is_installed():
        return                                   # modulo assente su questa macchina: no-crash
    t = w.world_tab
    t.name.setText("world_live_suite")
    t.image.setText("")                          # text-to-scene puro
    t.prompt.setText("a floating night market on wooden stilts above a bioluminescent bay, "
                     "paper lanterns, steam rising from tiny food stalls, crescent moon, "
                     "reflections rippling on the glowing water")
    t.res.setValue(1600)
    t.low_vram.setChecked(True)                  # ramo nunchaku fp4 (l'unico coi pesi installati)
    t.mode.setCurrentIndex(t.mode.findData("mesh"))
    t._gen()
    assert t._job is not None, "world live: job non avviato (_gen non ha lanciato make_job)"
    deadline = time.time() + 900
    while getattr(t._job, "running", False) and time.time() < deadline:
        time.sleep(5)
    assert not getattr(t._job, "running", False), "world live: timeout 15 min (job ancora in corso)"
    glb = ws.WORLD_OUTPUT_DIR / "world_live_suite" / "mesh.glb"
    assert glb.exists() and glb.stat().st_size > 1_000_000, \
        f"world live: mesh.glb mancante o vuoto ({glb})"


CASES.append(("world_gen_live",
              "Mondo 3D LIVE: text-to-scene reale (fp4) → mesh.glb prodotto e non vuoto.",
              case_world_gen_live))
