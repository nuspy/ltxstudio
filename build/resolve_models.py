"""build/resolve_models.py — risolve URL + SHA256 + size dei modelli del catalogo, AL BUILD.

Bakare url/hash/size nel manifest rende l'installer deterministico e firmabile: a runtime scarica da URL noti
e verifica l'hash, senza dover cercare online sulla macchina utente. Fonti:
- **HuggingFace** (repo `org/name`): per i file usa `HfApi.get_paths_info` → URL /resolve/main + LFS sha256 + size.
- **Civitai** (url/id o hash): `core/civitai_client` (by_hash/version/parse_url) → download URL + sha256 + size.

Uso (in fase di BUILD, con rete):
  python build/resolve_models.py                 # dry-run: stampa cosa risolverebbe
  python build/resolve_models.py --write         # scrive url/sha256/size nel catalogo (solo i mancanti)
  python build/resolve_models.py --hf-token TOK  # per i repo gated

NB: richiede rete. NON eseguire durante task GPU-bound; è uno step di build.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "weights_catalog.json"


def _hf_file(repo_id: str, filename: str, token: str | None) -> dict | None:
    """URL /resolve, sha256 (LFS) e size di un file in un repo HF. None se non risolvibile."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().get_paths_info(repo_id=repo_id, paths=[filename], token=token, expand=True)
        if not info:
            return None
        it = info[0]
        sha = getattr(getattr(it, "lfs", None), "sha256", None)
        size = getattr(getattr(it, "lfs", None), "size", None) or getattr(it, "size", None)
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        return {"url": url, "sha256": sha, "size_mb": round(size / 2**20) if size else None, "kind": "hf"}
    except Exception as e:                                   # noqa: BLE001
        print(f"  [hf] {repo_id}/{filename}: {e}")
        return None


def _hf_repo_size(repo_id: str, token: str | None) -> dict | None:
    """Size totale di uno snapshot repo (per i modelli 'diffusers'). URL = pagina repo (snapshot_download a runtime)."""
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id=repo_id, token=token)
        # size approssimata: somma dei .safetensors/bin via get_paths_info
        info = HfApi().get_paths_info(repo_id=repo_id, paths=[f for f in files], token=token, expand=True)
        total = 0
        for it in info:
            total += (getattr(getattr(it, "lfs", None), "size", None) or getattr(it, "size", 0) or 0)
        return {"url": f"hf://{repo_id}", "sha256": None,
                "size_mb": round(total / 2**20) if total else None, "kind": "hf-snapshot"}
    except Exception as e:                                   # noqa: BLE001
        print(f"  [hf-repo] {repo_id}: {e}")
        return None


def _civitai(entry: dict) -> dict | None:
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from core import civitai_client as cc
        ref = entry.get("civitai") or entry.get("url") or ""
        ids = cc.parse_url(ref) if ref else {}
        ver = cc.version(ids["version_id"]) if ids.get("version_id") else None
        if not ver:
            return None
        f = cc.pick_primary_file(ver)
        if not f:
            return None
        return {"url": f.get("downloadUrl"), "sha256": (f.get("hashes") or {}).get("SHA256", "").lower() or None,
                "size_mb": round(float(f.get("sizeKB", 0)) / 1024) or None, "kind": "civitai"}
    except Exception as e:                                   # noqa: BLE001
        print(f"  [civitai] {entry.get('id')}: {e}")
        return None


def resolve_entry(entry: dict, token: str | None) -> dict | None:
    """Ritorna i campi risolti {url, sha256, size_mb, kind} o None. Salta se già risolti."""
    if entry.get("url") and (entry.get("sha256") or entry.get("kind") == "snapshot"):
        return None
    if entry.get("civitai"):
        return _civitai(entry)
    repo = entry.get("repo_id")
    if repo and entry.get("kind") == "file" and entry.get("filename"):
        return _hf_file(repo, entry["filename"], token)
    if repo:
        return _hf_repo_size(repo, token)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--hf-token", default=None)
    a = ap.parse_args()
    data = json.loads(CATALOG.read_text("utf-8"))
    weights = data.get("weights", [])
    n = 0
    for e in weights:
        res = resolve_entry(e, a.hf_token)
        if not res:
            continue
        n += 1
        print(f"{e['id']:<28} → {res.get('kind'):<12} size~{res.get('size_mb')}MB sha={str(res.get('sha256'))[:12]}")
        if a.write:
            for k, v in res.items():
                if v is not None and not e.get(k):
                    e[k] = v
    if a.write:
        CATALOG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(f"\nScritti {n} risolti in {CATALOG}")
    else:
        print(f"\n{n} da risolvere (dry-run; usa --write per bakarli).")


if __name__ == "__main__":
    main()
