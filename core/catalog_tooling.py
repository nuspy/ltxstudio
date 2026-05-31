"""Wrapper per gli script del catalogo AI (E:\\AI\\dictionary\\tools\\).

Esegue gli script reali in un thread separato e notifica la UI tramite callback.
I percorsi (models/workflows/dict_root/inbox) sono configurabili: vengono iniettati
come variabili d'ambiente AICATALOG_* (lette da catalog_lib._resolve_paths) e la tab
sincronizza anche <tools_dir>/paths.json.

Script reali e relativi argomenti:
- place_files.py --from <dir> [--recursive] [--dry-run]   smista i file dell'inbox/sorgente
- scan.py                                                  aggiorna catalog.json (metadati base)
- enrich_civitai.py                                        lookup Civitai by-hash (+ override manuali)
- apply_manual_enrichment.py                               riapplica gli override manuali
- build_html.py                                            rigenera il catalogo HTML statico
- ingest.py --from <dir> [--recursive]                     pipeline completa place→scan→enrich→build
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from threading import Thread

from core import catalog_paths

PYTHON_EXE = sys.executable   # Python dell'ambiente virtuale corrente


def tools_dir() -> Path:
    return catalog_paths.tools_dir()


class CatalogTooling:
    """Esegue gli script del catalogo in thread separati con notifiche alla UI."""

    def __init__(self, on_output=None, on_done=None, on_error=None):
        """
        on_output: callable(line: str) — per ogni riga di stdout/stderr
        on_done:   callable(step: str) — al completamento di ogni step
        on_error:  callable(step: str, err: str) — in caso di errore
        """
        self.on_output = on_output or (lambda line: print(line))
        self.on_done = on_done or (lambda step: None)
        self.on_error = on_error or (lambda step, err: None)
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ---------- ambiente subprocess con i path configurati ----------
    def _env(self) -> dict:
        cfg = catalog_paths.load()
        env = dict(os.environ)
        env["AICATALOG_MODELS"] = cfg["models"]
        env["AICATALOG_WORKFLOWS"] = cfg["workflows"]
        env["AICATALOG_DICT_ROOT"] = cfg["dict_root"]
        env["AICATALOG_INBOX"] = cfg["inbox"]
        env["AICATALOG_HASHES"] = cfg["hashes_jsonl"]
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    # ---------- runner generico (sequenza di step) ----------
    def _spawn(self, steps: list[tuple[str, list[str]]]):
        """steps: lista di (label, [script_name, *args]) eseguiti in sequenza.
        Si ferma al primo errore. Notifica on_done per ogni step riuscito."""
        if self._running:
            self.on_error("run", "Operazione già in corso.")
            return
        td = tools_dir()
        env = self._env()

        def _run():
            self._running = True
            try:
                for label, args in steps:
                    script = td / args[0]
                    if not script.exists():
                        self.on_error(label, f"Script non trovato: {script}")
                        break
                    cmd = [PYTHON_EXE, str(script)] + args[1:]
                    extra = " ".join(args[1:])
                    self.on_output(f"\n$ {script.name} {extra}".rstrip())
                    try:
                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            cwd=str(td), env=env)
                        for line in proc.stdout:
                            self.on_output(line.rstrip())
                        proc.wait()
                    except Exception as e:
                        self.on_error(label, str(e))
                        break
                    if proc.returncode != 0:
                        self.on_error(label, f"Exit code {proc.returncode}")
                        break
                    self.on_done(label)
            finally:
                self._running = False

        Thread(target=_run, daemon=True).start()

    # ---------- operazioni singole ----------
    def analyze(self, source_dir: str, recursive: bool = False):
        """Classifica i file della sorgente SENZA spostarli (dry-run)."""
        args = ["place_files.py", "--from", str(source_dir), "--dry-run"]
        if recursive:
            args.append("--recursive")
        self._spawn([("analyze", args)])

    def place(self, source_dir: str, recursive: bool = False):
        """Sposta i file dalla sorgente nelle cartelle modelli/workflow giuste."""
        args = ["place_files.py", "--from", str(source_dir)]
        if recursive:
            args.append("--recursive")
        self._spawn([("place", args)])

    def scan(self):
        """Aggiorna catalog.json con i metadati di base (no rete)."""
        self._spawn([("scan", ["scan.py"])])

    def enrich(self):
        """Lookup Civitai by-hash + riapplica override manuali (richiede internet)."""
        self._spawn([("enrich", ["enrich_civitai.py"])])

    def apply_overrides(self):
        """Riapplica solo gli override manuali (manual_enrichment.json)."""
        self._spawn([("overrides", ["apply_manual_enrichment.py"])])

    def build(self):
        """Rigenera il catalogo HTML statico."""
        self._spawn([("build", ["build_html.py"])])

    def full_ingest(self, source_dir: str, recursive: bool = False):
        """Pipeline completa: place → scan → enrich → build (un solo script: ingest.py)."""
        args = ["ingest.py", "--from", str(source_dir)]
        if recursive:
            args.append("--recursive")
        self._spawn([("ingest", args)])
