"""Conferma di LICENZA prima di scaricare pesi con licenza restrittiva (es. MagicQuill V2: CC BY-NC-SA 4.0, non commerciale).

Le voci di catalogo con il campo `license_ack` (id del testo, es. «cc_by_nc_sa») NON si scaricano mai in silenzio:
  · `scripts/download_weights.py` le salta se non riceve `--accept-license <ids|all>` (quindi nemmeno `--all`, installer o CLI);
  · i punti GUI che avviano il download chiamano `confirm_gui()` (dialog con licenza, nomi e GB, conferma esplicita) e
    passano al downloader gli id accettati (`args_for`).
Le voci non sono mai `install_role: base`: l'installer/la build non le scarica né le include.
Stdlib + catalogo; Qt solo dentro confirm_gui.
"""
from __future__ import annotations

ARG = "--accept-license"


def needing_ack(ids: list[str]) -> list[dict]:
    """Voci (dipendenze incluse) che richiedono la conferma di licenza e non sono ancora presenti su disco."""
    from core import downloader as dl, weights_catalog as wc
    try:
        entries = dl.plan_ids(list(ids))
    except Exception:                                      # noqa: BLE001
        return []
    return [e for e in entries if e.get("license_ack") and not wc.is_present(e)]


def args_for(accepted) -> list[str]:
    """Argomenti da passare a scripts/download_weights.py per gli id accettati ([] se nessuno)."""
    acc = [a for a in (accepted or []) if a]
    return [ARG, ",".join(acc)] if acc else []


def confirm_gui(parent, ids: list[str]):
    """None = l'utente ha rifiutato (non scaricare); lista = id accettati ([] se nessuna conferma serviva)."""
    need = needing_ack(ids)
    if not need:
        return []
    from PySide6.QtWidgets import QMessageBox
    try:
        from ui.i18n_manager import tr
    except Exception:                                      # noqa: BLE001
        def tr(k, d=None):
            return d if d is not None else k
    lic = ", ".join(sorted({e.get("license", "?") for e in need}))
    names = "\n".join(f"• {e.get('label') or e['id']}  (~{e.get('approx_gb', '?')} GB) — {e.get('license', '?')}" for e in need)
    kinds = {e.get("license_ack") for e in need}
    parts = []
    if kinds - {"flux_kommunity", "adult_content"}:                 # licenze non commerciali (MagicQuill…)
        parts.append(tr("weights.license.ack.body.nc",
                        "Questi pesi hanno una licenza che NON consente l'uso commerciale ({license}). Si scaricano solo per tuo uso "
                        "personale/non commerciale, rispettando le condizioni dell'autore (attribuzione e condivisione alle stesse "
                        "condizioni). Non fanno parte dell'installer: li scarichi tu, sotto la tua responsabilità.").format(license=lic))
    if "flux_kommunity" in kinds:
        parts.append(tr("weights.license.ack.body.flux_kommunity",
                        "FLUX Kommunity License v1.0 (Black Forest Labs): l'uso commerciale degli output è consentito solo se i tuoi "
                        "ricavi annui lordi (con le affiliate) sono sotto 5 milioni di dollari, con filtro o revisione dei contenuti "
                        "e dichiarazione di uso dell'AI; è vietato usare modello, derivati o output per addestrare altri modelli di "
                        "immagini. Prima devi aver accettato la licenza su Hugging Face con il tuo account. I pesi non sono "
                        "nell'installer: li scarichi tu."))
    if "adult_content" in kinds:
        parts.append(tr("weights.license.ack.body.adult_content",
                        "ATTENZIONE, solo per adulti (not-for-all-audiences): questo modello è stato modificato da terzi per rifiutare "
                        "meno e può produrre testi per contenuti espliciti. Sei tu responsabile di ciò che generi e delle leggi che ti "
                        "riguardano; sono vietati contenuti illegali e immagini intime non consensuali."))
    body = "\n\n".join(parts) + "\n\n" + names + "\n\n" + tr("weights.license.ack.q", "Hai letto e vuoi scaricarli?")
    title = (tr("weights.license.ack.title", "Licenza non commerciale") if kinds == {"cc_by_nc_sa"}
             else tr("weights.license.ack.title.generic", "Licenza e avvisi"))
    r = QMessageBox.question(parent, title, body, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    if r != QMessageBox.Yes:
        return None
    return [e["id"] for e in need]
