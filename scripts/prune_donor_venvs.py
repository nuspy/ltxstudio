"""Cancella i venv-DONOR ormai inerti dopo il consolidamento #61 (recupero disco ~39 GB).

Un donor è cancellabile SOLO se:
  1. il suo service usa `_CONS` (host `inference_server_venv`) come interprete → il donor non viene mai eseguito;
  2. l'host è SUPERSET per nome del donor (0 pacchetti mancanti) → l'host può soddisfarlo;
  3. l'host esiste su disco.

⚠️ DISTRUTTIVO e IRREVERSIBILE. Default = DRY-RUN (mostra solo). Cancella davvero SOLO con `--apply`.
PRIMA di `--apply`: esegui `bash scripts/backup_venvs_to_nas.sh` e il live-test di ogni service dall'host.

Uso:
  python scripts/prune_donor_venvs.py            # dry-run: elenca i donor cancellabili + GB recuperabili
  python scripts/prune_donor_venvs.py --apply     # cancella per davvero (dopo backup + live-test)
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

HOST = Path(r"E:\AI\Tools\inference_server_venv")

# donor → (path venv, service che DEVE usare _CONS)
DONORS = {
    "ACE-Step/venv":    (r"E:\AI\Tools\ACE-Step\venv",    "core/acestep_service.py"),
    "ACE-Step/venv15":  (r"E:\AI\Tools\ACE-Step\venv15",  "core/acestep15_service.py"),
    "F5-TTS/venv":      (r"E:\AI\Tools\F5-TTS\venv",       "core/f5tts_service.py"),
    "XTTS/venv":        (r"E:\AI\Tools\XTTS\venv",         "core/xtts_service.py"),
    "heartlib/venv":    (r"E:\AI\Tools\heartlib\venv",     "core/music_service.py"),
    "FlashVSR/venv":    (r"E:\AI\Tools\FlashVSR\venv",     "core/flashvsr_service.py"),
    "SeedVR2/venv":     (r"E:\AI\Tools\SeedVR2\venv",      "core/seedvr2_service.py"),
}

REPO = Path(__file__).resolve().parent.parent


def _dists(venv: Path) -> set[str]:
    out = set()
    for d in glob.glob(str(venv / "Lib" / "site-packages" / "*.dist-info")):
        base = os.path.basename(d)[: -len(".dist-info")]
        out.add(base.rpartition("-")[0].lower().replace("_", "-"))
    return out


def _dir_gb(p: Path) -> float:
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1e9


def _service_uses_cons(service_rel: str) -> bool:
    """Il service seleziona l'host consolidato (_CONS = inference_server_venv)?"""
    fp = REPO / service_rel
    if not fp.exists():
        return False
    txt = fp.read_text(encoding="utf-8", errors="replace")
    return "_CONS" in txt and "inference_server_venv" in txt


def main() -> None:
    apply = "--apply" in sys.argv
    if not HOST.exists():
        print(f"HOST assente: {HOST} — nulla da fare (i tool userebbero i propri venv).")
        return
    host_d = _dists(HOST)
    print(f"HOST {HOST} — {len(host_d)} distribuzioni{'' if apply else '  [DRY-RUN: nessuna cancellazione]'}\n")

    reclaim = 0.0
    to_delete: list[tuple[str, Path]] = []
    for name, (path, service) in DONORS.items():
        venv = Path(path)
        if not venv.exists():
            print(f"  · {name:20} già assente")
            continue
        uses = _service_uses_cons(service)
        missing = _dists(venv) - host_d
        gb = _dir_gb(venv)
        ok = uses and not missing
        flag = "CANCELLABILE" if ok else "TENERE"
        why = "" if ok else (" (service non usa _CONS!)" if not uses
                             else f" (host manca di {len(missing)} pkg: {sorted(missing)[:4]}…)")
        print(f"  [{flag:12}] {name:20} {gb:5.1f} GB  ← {service}{why}")
        if ok:
            reclaim += gb
            to_delete.append((name, venv))

    print(f"\nRecupero disco possibile: {reclaim:.1f} GB su {len(to_delete)} donor.")
    if not to_delete:
        return
    if not apply:
        print("\nDRY-RUN. Per cancellare davvero (dopo `backup_venvs_to_nas.sh` + live-test):")
        print("  python scripts/prune_donor_venvs.py --apply")
        return
    print("\n--apply: CANCELLO i donor…")
    for name, venv in to_delete:
        shutil.rmtree(venv, ignore_errors=True)
        print(f"  ✗ rimosso {name}")
    print(f"FATTO. Recuperati ~{reclaim:.1f} GB. Ripristino: scripts/restore_venvs_from_nas.sh")


if __name__ == "__main__":
    main()
