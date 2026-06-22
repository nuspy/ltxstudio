"""Profilo di sviluppo (committato): usato quando `core/edition.py` NON è presente,
cioè quando l'app gira NON compilata. Manifest permissivo: tutte le feature
attive (is_enabled ritorna True per feature non elencate), così in sviluppo si
vede tutto. In produzione il build bakea `core/edition.py` che ha la precedenza.
"""
from __future__ import annotations

EDITION_ID = "dev"
DEFAULT_EDITION = "dev"
BUILD_NUMBER = 0
MATRIX_DIGEST = "dev"
BAKE_FULL_LADDER = False

MANIFEST = {
    "edition": "dev",
    "public_name": "Sviluppo",
    "rank": 99,
    "allows_adult": True,
    "allows_studio": True,
    "build_number": 0,
    "features": {},          # vuoto → is_enabled() permissivo (tutto ON in dev)
    "assets_enabled": [],
    "digest": "dev",
}
MANIFESTS = {"dev": MANIFEST}
