"""Auto-installazione dello skill LTX Studio per i vari agenti.

Rigenera i pacchetti (build_packs) e li copia nella cartella skill/plugin di ciascun agente,
insieme ai contratti macchina (openapi.json, tools.openai.json, manifest.json).

Uso:
  python -m agents.install_skill                 # installa per TUTTI gli agenti
  python -m agents.install_skill --agent claude_code
  python -m agents.install_skill --agent hermes --dest "D:/hermes/skills"
  python -m agents.install_skill --no-build      # non rigenerare, usa i file esistenti
  python -m agents.install_skill --list          # mostra le destinazioni

Le destinazioni non-Claude sono i default noti/ragionevoli: se un agente usa un altro percorso,
passa --dest. Claude Code usa il percorso reale ~/.claude/skills/ltxstudio.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()

# Hermes (Nous Research) carica gli skill da %LOCALAPPDATA%\hermes\skills\
_LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", HOME / "AppData" / "Local"))
_HERMES_SKILLS = _LOCALAPPDATA / "hermes" / "skills"

# agente → cartella di installazione dello skill (override con --dest)
TARGETS = {
    "claude_code": HOME / ".claude" / "skills" / "ltxstudio",
    "hermes":      (_HERMES_SKILLS if _HERMES_SKILLS.parent.exists() else HOME / ".hermes" / "skills") / "ltxstudio",
    "openclaw":    HOME / ".openclaw" / "skills" / "ltxstudio",
    "antigravity": HOME / ".antigravity" / "skills" / "ltxstudio",
    "openai_codex": HOME / ".codex" / "skills" / "ltxstudio",
    "generic":     HOME / ".ltxstudio" / "skill",
}

# contratti condivisi copiati in ogni installazione
SHARED = ["openapi.json", "tools.openai.json", "manifest.json", "README.md"]


def _build():
    from agents import build_packs
    build_packs.main()


def _install_one(agent: str, dest: Path) -> Path:
    src = HERE / agent
    if not src.is_dir():
        raise SystemExit(f"pack agente mancante: {src} (esegui build_packs)")
    dest.mkdir(parents=True, exist_ok=True)
    # SKILL.md + plugin.json specifici dell'agente
    for f in ("SKILL.md", "plugin.json"):
        if (src / f).is_file():
            shutil.copy2(src / f, dest / f)
    # contratti condivisi
    for f in SHARED:
        if (HERE / f).is_file():
            shutil.copy2(HERE / f, dest / f)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="all", help="all | " + " | ".join(TARGETS))
    ap.add_argument("--dest", default="", help="cartella di installazione (override)")
    ap.add_argument("--no-build", action="store_true", help="non rigenerare i pack")
    ap.add_argument("--list", action="store_true", help="mostra le destinazioni ed esci")
    a = ap.parse_args()

    if a.list:
        for k, v in TARGETS.items():
            print(f"  {k:13} → {v}")
        return

    if not a.no_build:
        _build()

    agents = list(TARGETS) if a.agent == "all" else [a.agent]
    for ag in agents:
        if ag not in TARGETS:
            print(f"[skip] agente sconosciuto: {ag}"); continue
        dest = Path(a.dest) / "ltxstudio" if (a.dest and a.agent == "all") else \
               (Path(a.dest) if a.dest else TARGETS[ag])
        try:
            out = _install_one(ag, dest)
            print(f"[ok] {ag:13} → {out}")
        except Exception as e:                              # noqa: BLE001
            print(f"[err] {ag}: {e}", file=sys.stderr)

    print("\nFatto. Ogni agente carica lo skill dalla sua cartella; l'API si avvia da sola "
          "(python -m api.autostart) e si spegne dopo 30 min di inattività.")


if __name__ == "__main__":
    main()
