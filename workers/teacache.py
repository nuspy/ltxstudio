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


def enable_class(thresh: float):
    """Patch di CLASSE su X0Model.forward — FIX calibrazione 07/07: il transformer LTX-2 è costruito
    LAZY dentro la run (DiffusionStage._build_transformer) → il patch per-istanza sugli stage non lo
    trovava MAI ("[teacache] attivo su 0 stage" → output bit-identici, zero skip). Il patch di classe
    prende OGNI X0Model creato (stage_1 e stage_2), stato per-istanza su self._tc_state.
    Skip corretto da paper: riusa la VELOCITY dell'ultimo forward reale e ricalcola il denoised sui
    latenti/sigma CORRENTI (to_denoised) — non riusa il denoised stantio. Reversibile (disable_class)."""
    if not thresh or thresh <= 0:
        return None
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.utils import to_denoised
    if getattr(X0Model, "_teacache_orig", None) is not None:
        return X0Model                                       # già attivo
    orig = X0Model.forward

    def patched(self, video, audio, perturbations):
        st = getattr(self, "_tc_state", None)
        if st is None:
            st = {"prev_in": None, "vx": None, "ax": None, "acc": 0.0, "hits": 0, "runs": 0}
            self._tc_state = st
        cur = None
        if video is not None and getattr(video, "latent", None) is not None:
            cur = video.latent
        elif audio is not None and getattr(audio, "latent", None) is not None:
            cur = audio.latent
        if cur is not None and st["prev_in"] is not None and cur.shape == st["prev_in"].shape:
            st["acc"] += _rel_l1(cur, st["prev_in"])
            if st["acc"] < thresh and (st["vx"] is not None or st["ax"] is not None):
                st["hits"] += 1
                st["prev_in"] = cur.detach()
                dv = to_denoised(video.latent, st["vx"], video.timesteps) \
                    if (st["vx"] is not None and video is not None) else None
                da = to_denoised(audio.latent, st["ax"], audio.timesteps) \
                    if (st["ax"] is not None and audio is not None) else None
                return dv, da
        vx, ax = self.velocity_model(video, audio, perturbations)
        st.update(prev_in=(cur.detach() if cur is not None else None), vx=vx, ax=ax, acc=0.0)
        st["runs"] += 1
        dv = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        da = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return dv, da

    X0Model.forward = patched
    X0Model._teacache_orig = orig
    return X0Model


def class_stats() -> dict:
    """Statistiche aggregate del patch di classe (somma su tutte le istanze viste)."""
    try:
        from ltx_core.model.transformer.model import X0Model
        return getattr(X0Model, "_teacache_totals", {})
    except Exception:                                        # noqa: BLE001
        return {}


def disable(vm):
    """Rimuove il patch e ritorna le statistiche {hits, runs} (skip vs forward reali)."""
    if vm is not None and getattr(vm, "_teacache_orig", None) is not None:
        vm.forward = vm._teacache_orig
        stt = getattr(vm, "_teacache_state", {})
        del vm._teacache_orig
        return {"hits": stt.get("hits", 0), "runs": stt.get("runs", 0)}
    return {}
