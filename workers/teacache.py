"""TeaCache per LTX (ltx_core) — #104.

Cache training-free (Timestep Embedding Aware Cache, arXiv:2411.19108): salta il forward del velocity-model
LTX quando il latent d'ingresso cambia poco tra due step di denoising, riusando la velocità cachata.
Speedup tipici (LTX-Video): ~1.6x a thresh 0.03, ~2.1x a 0.05. Conviene sulle modalità a molti step (HQ/Pro);
sulla distilled 8-step il guadagno è minore.

Monkey-patch REVERSIBILE su `LTXModel.forward` (il velocity-model): NON modifica la libreria ltx_core.
NB soglia + coefficienti vanno CALIBRATI live (thresh vs qualità) — per LTX 2.3 (22B) i coefficienti pubblicati
per LTX-Video 0.9 sono solo un punto di partenza. Default OFF (thresh<=0). Vedi docs/upgrades-evaluation.md.
"""
from __future__ import annotations


def _rel_l1(cur, prev):
    """Distanza L1 relativa media tra due tensori (metrica di cambiamento dell'input)."""
    return float((cur - prev).abs().mean() / (prev.abs().mean() + 1e-8))


def _find_velocity_model(stage):
    """Il velocity-model è o lo stage stesso (LTXModel) o stage.velocity_model (X0Model/LegacyX0Model)."""
    if stage is None:
        return None
    if hasattr(stage, "velocity_model"):
        return stage.velocity_model
    if hasattr(stage, "_process_transformer_blocks"):       # è già l'LTXModel
        return stage
    return None


def enable(stage, thresh: float = 0.05, rescale=None):
    """Avvolge il velocity-model dello stage con TeaCache. thresh<=0 → no-op (ritorna None).
    rescale: funzione opzionale float→float per rimappare la distanza L1 (coefficienti LTX); default identità.
    Ritorna il velocity-model patchato (per disable) o None."""
    if not thresh or thresh <= 0:
        return None
    vm = _find_velocity_model(stage)
    if vm is None or getattr(vm, "_teacache_orig", None) is not None:
        return vm                                            # non trovato o già attivo
    orig = vm.forward
    resc = rescale or (lambda x: x)
    st = {"prev_in": None, "prev_vx": None, "prev_ax": None, "acc": 0.0, "hits": 0, "runs": 0}

    def patched(video, audio, perturbations):
        cur = None
        if video is not None and getattr(video, "latent", None) is not None:
            cur = video.latent
        elif audio is not None and getattr(audio, "latent", None) is not None:
            cur = audio.latent
        if cur is None or st["prev_in"] is None:             # primo step o niente latent → forward reale
            vx, ax = orig(video, audio, perturbations)
            st.update(prev_in=(cur.detach() if cur is not None else None), prev_vx=vx, prev_ax=ax, acc=0.0)
            st["runs"] += 1
            return vx, ax
        st["acc"] += resc(_rel_l1(cur, st["prev_in"]))
        if st["acc"] < thresh and st["prev_vx"] is not None:  # cambiamento accumulato piccolo → SKIP
            st["hits"] += 1
            st["prev_in"] = cur.detach()
            return st["prev_vx"], st["prev_ax"]
        vx, ax = orig(video, audio, perturbations)            # cambiamento grande → forward reale + ricache
        st.update(prev_in=cur.detach(), prev_vx=vx, prev_ax=ax, acc=0.0)
        st["runs"] += 1
        return vx, ax

    vm.forward = patched
    vm._teacache_orig = orig
    vm._teacache_state = st
    return vm


def disable(vm):
    """Rimuove il patch e ritorna le statistiche {hits, runs} (skip vs forward reali)."""
    if vm is not None and getattr(vm, "_teacache_orig", None) is not None:
        vm.forward = vm._teacache_orig
        stt = getattr(vm, "_teacache_state", {})
        del vm._teacache_orig
        return {"hits": stt.get("hits", 0), "runs": stt.get("runs", 0)}
    return {}
