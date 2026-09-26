"""MiniMax H3 usa il venv CONDIVISO con Qwen-Image 2.1 (E:\\AI\\Tools\\diffusers_next_venv).

Questo script resta come alias storico: installa/aggiorna quel venv (scripts/install_diffusers_next.py).
Uso:  python scripts/install_minimax_h3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from install_diffusers_next import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
