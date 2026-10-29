"""Helper condivisi dagli handler REST: esecuzione worker video (ltx2), job immagine (:8189),
risoluzione file, parsing LoRA.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid

from api.config import OUTPUT_DIR

LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LTX2_WORKER = os.path.join(REPO, "workers", "ltx2_infer.py")


def out_path(prefix: str, ext: str = "mp4") -> str:
    """Nuovo path di output univoco sotto OUTPUT_DIR."""
    return str(OUTPUT_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")


def parse_loras(loras) -> list[dict]:
    """Normalizza `loras` (lista di {name,strength} o stringa JSON) → lista di dict."""
    if not loras:
        return []
    if isinstance(loras, str):
        try:
            loras = json.loads(loras)
        except Exception:                                    # noqa: BLE001
            return []
    out = []
    for lo in loras or []:
        if isinstance(lo, dict) and lo.get("name"):
            out.append({"name": lo["name"], "strength": float(lo.get("strength", 1.0))})
    return out


def parse_references(references, default_strength: float = 1.0) -> list[dict]:
    """Normalizza i riferimenti multipli (MSR) → [{path, kind, strength, role}].

    Accetta stringhe (path nudo), dict o una stringa JSON. `role='background'` serve alla variante
    pseudo-video della 2.3, che senza sfondo non parte.

    `default_strength` (cioè `msr_strength`) vale per i riferimenti che non ne dichiarano una
    propria: sullo slot embedding della 2.5 è la strength PER-RIFERIMENTO a decidere il peso
    contro il prompt, quindi lasciarla fissa a 1.0 rendeva il parametro inutile.
    """
    if not references:
        return []
    if isinstance(references, str):
        try:
            references = json.loads(references)
        except Exception:                                    # noqa: BLE001
            return []
    out = []
    for r in references or []:
        if isinstance(r, str) and r:
            out.append({"path": r, "kind": "image", "strength": float(default_strength)})
        elif isinstance(r, dict) and r.get("path"):
            out.append({"path": r["path"], "kind": r.get("kind", "image"),
                        "strength": float(r.get("strength", default_strength)),
                        **({"role": r["role"]} if r.get("role") else {})})
    return out


def apply_ltx_release(params: dict, release: str | None, model: str = "",
                      references=None, msr_strength: float = 1.0,
                      msr_background: str = "", msr_spacing: int = 1) -> dict:
    """Aggiunge a `params` release, checkpoint e riferimenti multipli (MSR), validando prima.

    Punto unico per gli handler REST: la 2.5 va passata esplicitamente (il worker ha il checkpoint
    2.3 come costante e per le altre release pretende `model`), e i riferimenti multipli senza la
    LoRA Multiple-Subject-Reference fanno morire il worker con un errore poco leggibile.
    """
    from core import ltx_release

    rel_id = (release or "").strip()
    if rel_id:
        if rel_id not in ltx_release.LTX_RELEASES:
            raise RuntimeError(f"release LTX sconosciuta: {rel_id} "
                               f"(disponibili: {', '.join(sorted(ltx_release.LTX_RELEASES))})")
        missing = ltx_release.missing_components(rel_id)
        if missing:
            raise RuntimeError(f"LTX {rel_id}: componenti mancanti su disco: {', '.join(missing)}")
        params["ltx_release"] = rel_id
        # Per release != 2.3 il worker azzera la costante CHECKPOINT: il transformer va indicato.
        if not model and rel_id != ltx_release.DEFAULT_RELEASE:
            cks = ltx_release.checkpoints(rel_id)
            if not cks:
                raise RuntimeError(f"LTX {rel_id}: nessun transformer trovato su disco")
            model = cks[0]
    if model:
        params["model"] = model

    refs = parse_references(references, msr_strength)
    if refs:
        rel = ltx_release.get(rel_id or ltx_release.DEFAULT_RELEASE)
        if rel.caps.get("multiref") != ltx_release.CAP_YES:
            raise RuntimeError(f"LTX {rel.id} non dichiara la capability 'multiref'")
        if not rel.msr_lora or not os.path.exists(rel.msr_lora):
            raise RuntimeError(f"LTX {rel.id}: LoRA Multiple-Subject-Reference assente "
                               f"({rel.msr_lora or 'non configurata'})")
        params["references"] = refs
        params["msr_lora"] = rel.msr_lora
        params["msr_strength"] = float(msr_strength)
        if msr_background:
            params["msr_background"] = msr_background
        if msr_spacing and int(msr_spacing) > 1:
            params["msr_spacing"] = int(msr_spacing)
    return params


def run_image(params: dict, *, timeout: float = 900) -> list[str]:
    """Esegue un job immagine sull'InferenceServer nativo (:8189). Ritorna i path di output."""
    from core import native_driver
    if not native_driver.is_up():
        raise RuntimeError("Backend nativo immagini (:8189) non attivo. Avvia l'InferenceServer "
                           "(python -m inference_server.server) o la GUI.")
    module = params.pop("_module", "image")
    res = native_driver.run_and_wait(module, params, timeout=timeout)
    if res.get("status") != "success":
        raise RuntimeError(res.get("error", "job immagine fallito")[:800])
    return list(res.get("output", []))


def run_ltx2(params: dict, job=None, *, timeout: float = 7200) -> list[str]:
    """Esegue il worker LTX con i params dati. Ritorna [output_path].

    Il runner NON è più fisso: lo decide `runner_for()` dalla release (`ltx_release`), perché 2.3
    e 2.5 hanno venv incompatibili — la 2.5 vuole `lingbot_venv` (transformers>=5.8 per Gemma 4).
    Mandando un job 2.5 al venv 2.3 il pacchetto non sa caricare un text encoder da file singolo e
    muore con «No files matching pattern 'tokenizer.model'».
    """
    from core.ltx_video_job import runner_for
    py, worker = runner_for(params)
    if not os.path.isfile(py):
        rel = params.get("ltx_release") or "2.3"
        raise RuntimeError(f"venv per LTX {rel} mancante: {py}")
    params.setdefault("output", out_path("api_video"))
    jf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(params, jf); jf.close()
    try:                                                     # il file-params temporaneo va SEMPRE rimosso a fine job
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        proc = subprocess.Popen([py, "-u", worker, jf.name], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        result = None
        t0 = time.time()
        # L'output del worker finiva solo in job.log(), che l'API non espone e che tiene le prime
        # 200 lettere di poche righe: diagnosticare una generazione andata storta voleva dire
        # rifarla a mano. Qui resta su file, sovrascritto a ogni job — l'ULTIMA generazione è
        # sempre leggibile per intero, senza far crescere niente all'infinito.
        try:
            _lg = open(os.path.join(os.path.dirname(params["output"]) or ".", "worker_last.log"),
                       "w", encoding="utf-8", errors="replace")
        except OSError:
            _lg = None
        for line in proc.stdout:
            line = line.rstrip()
            if _lg is not None:
                _lg.write(line + "\n"); _lg.flush()
            if job is not None and (line.startswith("[") or line.startswith("OK:")):
                job.log(line[:200])
            if line.startswith("OK:"):
                result = line[3:].strip()
            if time.time() - t0 > timeout:
                proc.kill()
                if _lg is not None:
                    _lg.close()
                raise RuntimeError("timeout generazione video")
        if _lg is not None:
            _lg.close()
        proc.wait()
        if result and os.path.exists(result):
            from core import gen_metadata                    # task 21: metadati di generazione nel video finale
            gen_metadata.embed_result({"status": "success", "output": [result]}, params, label="Video")
            return [result]
        raise RuntimeError("LTX-2 nativo: nessun output prodotto")
    finally:
        try:
            os.unlink(jf.name)                               # niente accumulo di _*.json nel temp
        except OSError:
            pass


def run_job_obj(job_obj) -> list[str]:
    """Esegue un oggetto Job (core.music_service.Job / *_service.make_job) e ritorna l'output.
    Il Job scrive un file noto; il chiamante passa `expected` via attributo, oppure legge .out."""
    job_obj.run()
    out = getattr(job_obj, "out", None) or getattr(job_obj, "output", None)
    if out and os.path.exists(str(out)):
        return [str(out)]
    return []
