"""Driver del SERVER VIDEO PERSISTENTE LTX-2.3 (workers/ltx2_server.py, porta 8190, venv ltx2).

L'app PyQt chiama generate(); il server tiene la pipeline in RAM (offload CPU) → nessun
ricaricamento da disco tra le generazioni. Avvio automatico del server se non attivo.
ZERO ComfyUI. Pattern: submit asincrono + polling (come comfy_driver/native_driver).
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

import requests

LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
REPO = str(Path(__file__).resolve().parent.parent)
BASE = "http://127.0.0.1:8190"
OUTPUT_DIR = r"E:\AI\ltxstudio\output"
_CREATE_NEW_CONSOLE = 0x00000010  # Windows
# Motore video DEFAULT = comfy-native (loader+memory-management di ComfyUI come LIBRERIA): VRAM dinamica
# = NIENTE OOM/spill sui 32GB (il pacchetto Lightricks accumula → sborza). Vedi [[native-video-engine]].
COMFY_PY = r"E:\AI\Comfy_Nexus.bak\ComfyUI\venv\Scripts\python.exe"   # venv torch2.8+cu128 + SageAttention
_DM_ROOT = r"E:\AI\Models\diffusion_models"


def comfy_available() -> bool:
    """Il motore comfy-native (ComfyUI-as-library) è installato? Con ComfyUI rimosso (.bau) è False:
    la GUI in quel caso NASCONDE i dropdown di scelta motore (richiesta utente: se il motore è uno
    solo, niente menu) e il dispatch va diretto al nativo."""
    return os.path.exists(COMFY_PY)


def is_up() -> bool:
    try:
        requests.get(f"{BASE}/system_stats", timeout=2)
        return True
    except Exception:                                       # noqa: BLE001
        return False


def server_enabled() -> bool:
    """Residenza video di DEFAULT (unificazione motori): la GUI usa il server :8190 per arch 'ltx'
    (pipeline + parcheggio RAM sopravvivono tra le generazioni → niente reload da SSD).
    Opt-out: env LTX_VIDEO_SERVER=0 oppure user_prefs 'video_server'=False."""
    if os.environ.get("LTX_VIDEO_SERVER", "").strip() == "0":
        return False
    try:
        from core import user_prefs
        return bool(user_prefs.get("video_server", True))
    except Exception:                                       # noqa: BLE001
        return True


def submit_native(params: dict) -> str:
    """Sottomette un job NATIVO al server persistente (:8190). Ritorna il job-id (solleva su errore).
    Coordina prima i /free delle altre categorie E ASPETTA il rilascio EFFETTIVO della VRAM (il
    /free risponde prima che il driver la restituisca — WDDM): senza attesa il server misurava una
    VRAM ancora occupata e sceglieva streaming/shared a torto. Questo path (GUI→server) era l'UNICO
    senza alcuna attesa."""
    _free_gpu_servers()
    _wait_for_video_vram(params)
    r = requests.post(f"{BASE}/submit", json={"params": params}, timeout=30).json()
    return r["prompt_id"]


def _wait_for_video_vram(params: dict) -> None:
    """Attesa DINAMICA pre-submit: soglia = fabbisogno vero del modello (cache misurata/header).
    Se il server ha già una pipeline in cache (residenza voluta: la VRAM è SUA) si aspetta solo il
    rilascio dei server immagini appena liberati (working ~8GB), non l'intero fabbisogno."""
    try:
        from core import gpu_mem
        cached = 0
        try:
            cached = int(requests.get(f"{BASE}/system_stats", timeout=3).json()
                         .get("cached_pipelines", 0) or 0)
        except Exception:                                   # noqa: BLE001
            pass
        if cached > 0:
            gpu_mem.wait_free_vram_mb(8000, timeout_s=15, tag="video-res")
            return
        need = 0
        g = str(params.get("gguf_transformer") or "")
        m = str(params.get("model") or "")
        if g and os.path.exists(g):
            need = int(os.path.getsize(g) / 2**20) + 9000
        elif m and os.path.exists(m):
            from core import vram_estimate
            need, _ = vram_estimate.estimate_mb(m, fp8_storage=True, working_mb=9000)
        gpu_mem.wait_free_vram_mb((need or 24000) + 2000, timeout_s=40, tag="video")
    except Exception:                                       # noqa: BLE001
        pass


def poll_native(jid: str) -> dict:
    """Stato del job: {status, output?, error?, progress?{msg,step,total}}."""
    return requests.get(f"{BASE}/history/{jid}", timeout=10).json()


def cancel_native(jid: str) -> dict:
    """Annulla il job (cooperativo: tra gli step del denoise / ai confini di fase). Best-effort."""
    try:
        return requests.post(f"{BASE}/cancel/{jid}", timeout=10).json()
    except Exception as e:                                  # noqa: BLE001
        return {"status": "unreachable", "error": str(e)}


def installed() -> bool:
    return os.path.exists(LTX2_PY) and os.path.exists(os.path.join(REPO, "workers", "ltx2_server.py"))


def _generate_direct(p: dict, release: str, timeout: int) -> dict:
    """Esegue il worker nel venv DELLA RELEASE, senza server persistente (vedi `generate`).
    Stessa interfaccia del percorso server: {status, output:[mp4]}."""
    import json as _json
    import tempfile as _tf
    from core import ltx_video_job as _lvj
    py, worker = _lvj.runner_for({"arch": "ltx", "ltx_release": release})
    if not os.path.exists(py):
        return {"status": "error",
                "error": f"venv della release LTX {release} mancante: {py}"}
    p.setdefault("output", os.path.join(OUTPUT_DIR, f"ltx2_{uuid.uuid4().hex[:8]}.mp4"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _free_gpu_servers()
    _wait_for_video_vram(p)
    jf = _tf.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    _json.dump(p, jf); jf.close()
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    print(f"[ltx2] release {release} → worker diretto ({os.path.basename(os.path.dirname(os.path.dirname(py)))})",
          flush=True)
    r = subprocess.run([py, "-u", worker, jf.name], env=env, cwd=REPO,
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout)
    out = ""
    for line in (r.stdout or "").splitlines():
        if line.startswith("OK:"):
            out = line[3:].strip()
    if out and os.path.exists(out):
        return {"status": "success", "output": [out]}
    _tail = "\n".join((r.stdout or "").splitlines()[-25:]) or (r.stderr or "")[-2000:]
    return {"status": "error", "error": f"worker LTX {release} senza output (exit {r.returncode}):\n{_tail}"}


def ensure_server(wait_s: int = 120) -> bool:
    """Avvia il server video persistente se non è attivo (subprocess in console separata)."""
    if is_up():
        return True
    if not installed():
        return False
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.Popen([LTX2_PY, "-m", "workers.ltx2_server"], cwd=REPO, env=env,
                     creationflags=_CREATE_NEW_CONSOLE)
    for _ in range(wait_s // 2):
        if is_up():
            return True
        time.sleep(2)
    return False


def reset_server() -> bool:
    """Svuota pipeline in cache e parcheggio RAM del server, lasciandolo vivo.

    Serve quando il server costruisce un modello incoerente col checkpoint richiesto e il job muore
    con «size mismatch … Error(s) in loading state_dict»: lo stato sporco resta e fa fallire anche
    tutti i job successivi, mentre lo stesso checkpoint in un processo pulito carica senza errori.
    `/free mode=destroy` butta via sia la cache sia i pesi parcheggiati, quindi il job seguente
    ricostruisce da zero — meglio che uccidere il processo, che si porta dietro il riavvio.
    """
    if not is_up():
        return True                                         # già a terra: il prossimo avvio è pulito
    try:
        r = requests.post(f"{BASE}/free", json={"mode": "destroy"}, timeout=30).json()
        if r.get("busy"):
            print("[ltx2] reset RIFIUTATO: job in corso sul server", flush=True)
            return False
        print(f"[ltx2] server azzerato: {r.get('freed_pipelines', 0)} pipeline, "
              f"parcheggio RAM svuotato", flush=True)
        return True
    except Exception as e:                                  # noqa: BLE001
        print(f"[ltx2] reset del server fallito ({e})", flush=True)
        return False


def _free_gpu_servers() -> None:
    """M1 — prima del VIDEO libera i server di categoria DIVERSA (immagini :8189 + API :8200): su single-GPU
    un solo modello pesante alla volta, altrimenti il transformer 22B non entra e sborza in shared-RAM.
    Via mem_coordinator (pipeline-aware): video→video consecutivi NON rifanno il /free (pipeline riusata)."""
    try:
        from core import mem_coordinator
        mem_coordinator.free_for("video", timeout=20)
    except Exception:                                          # noqa: BLE001
        pass


def _to_dm_relative(model: str) -> str:
    """Converte il path modello al nome RELATIVO sotto diffusion_models (ComfyUI folder_paths)."""
    if not model:
        return "LTXVideo/v2/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
    m = model.replace("/", "\\")
    root = _DM_ROOT.rstrip("\\") + "\\"
    if m.lower().startswith(root.lower()):
        m = m[len(root):]
    return m.replace("\\", "/")


def _auto_calibrate_video_model(model_rel: str) -> str:
    """Sceglie il modello LTX-2.3 22B per FASCIA VRAM (distribuibile, niente hardcode), preferendo il
    DISTILLED (step veloci) dove entra. Solo se il default è il 22B fp8; rispetta scelte esplicite (GGUF/altro).
    Fasce: ≥30GB→fp8 distilled (24GB, qualità) · ≥16GB→Q4_K_M distilled (14GB) · <16GB→Q3_K_M dev (~10GB)."""
    import os as _os, glob
    if not model_rel or model_rel.lower().endswith(".gguf") or "22b" not in model_rel.lower():
        return model_rel
    # RISPETTA la scelta esplicita: declassa per VRAM SOLO se è il default (nessuna scelta consapevole).
    # La conferma interattiva del cambio avviene in GUI (_confirm_model_swap); qui non si sostituisce a
    # sorpresa un modello scelto dall'utente. Override forzato: NATIVE_VIDEO_AUTOCALIBRATE=1.
    if _os.environ.get("NATIVE_VIDEO_AUTOCALIBRATE", "0") != "1" and "dev-fp8" not in model_rel.lower():
        return model_rel
    try:
        from core import model_precision as _mp
        free = float(_mp.gpu_free_gb())
    except Exception:                                       # noqa: BLE001
        return model_rel

    def _first(*cands):
        for c in cands:
            if c and _os.path.exists(c):
                return c
        return ""
    if free >= 36.0:
        return model_rel                                    # fp8 distilled (24GB) — solo 40/48GB (24+attivazioni)
    if free >= 13.0:                                        # Q4_K_M distilled (14GB) — entra su 32GB/24GB, veloce
        g = _first(_os.path.join(_DM_ROOT, "LTXVideo", "LTX-2", "unsloth", "distilled-1.1", "ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf"),
                   *glob.glob(_os.path.join(_DM_ROOT, "**", "*distilled*Q4_K_M.gguf"), recursive=True))
        if g:
            print(f"[auto-calibra] VRAM~{free:.0f}GB → Q4 distilled (14GB)", flush=True); return _to_dm_relative(g)
    # schede piccole → Q3_K_M dev (~10GB)
    g = _first(_os.path.join(_DM_ROOT, "ltx-2-3-22b-dev-Q3_K_M.gguf"),
               *glob.glob(_os.path.join(_DM_ROOT, "**", "*Q3_K_M.gguf"), recursive=True),
               *glob.glob(_os.path.join(_DM_ROOT, "**", "*Q4_K_M.gguf"), recursive=True))
    if g:
        print(f"[auto-calibra] VRAM~{free:.0f}GB → GGUF piccolo ({_os.path.basename(g)})", flush=True)
        return _to_dm_relative(g)
    return model_rel


def _generate_comfy_native(params: dict, timeout: int) -> dict:
    """Genera via workers/ltx_comfy_native.py (model_management ComfyUI = VRAM dinamica, no OOM).
    One-shot subprocess nel venv ComfyUI: GPU pulita ad ogni run (niente server che accumula)."""
    import json
    _free_gpu_servers()
    p = dict(params)
    p["model"] = _to_dm_relative(p.get("model", ""))
    # AUTO-CALIBRAZIONE per HARDWARE (distribuibile): fp8 (qualità piena) quando la VRAM basta; su schede
    # con poca VRAM libera auto-scelta del GGUF Q4 (footprint ridotto). NIENTE hardcode → si adatta a
    # 4090/5090/8GB/48GB. L'offload fine lo fa comunque il model_management di ComfyUI. Vedi [[dynamic-vram-tuning]].
    p["model"] = _auto_calibrate_video_model(p["model"])
    p.setdefault("output", os.path.join(OUTPUT_DIR, f"ltx2_{uuid.uuid4().hex[:8]}.mp4"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pj = os.path.join(OUTPUT_DIR, f"_vparams_{uuid.uuid4().hex[:8]}.json")
    Path(pj).write_text(json.dumps(p), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    worker = os.path.join(REPO, "workers", "ltx_comfy_native.py")
    try:
        r = subprocess.run([COMFY_PY, worker, pj], cwd=REPO, env=env, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "timeout generazione video"}
    finally:
        try:
            os.remove(pj)
        except OSError:
            pass
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if line.startswith("OK:"):
            return {"status": "success", "output": [line[3:].strip()]}
    return {"status": "error", "error": out[-2000:]}


def generate(params: dict, timeout: int = 3600) -> dict:
    """Genera un video. params: prompt, image_path?(i2v), num_frames, fps, steps, guidance,
    seed, model, quant, offload, loras=[{name,strength}], output?. Ritorna {status, output:[mp4]}.

    DEFAULT = comfy-native (model_management ComfyUI, no OOM su 32GB). LTX_ENGINE=lightricks per il
    vecchio server persistente ltx-core (sborza sui 32GB).

    Selezione motore (in ordine di priorità): env LTX_ENGINE → user_prefs.get_engine('video') →
    default 'comfy'. Il selettore nel dialog Impostazioni scrive get_engine('video')."""
    engine = os.environ.get("LTX_ENGINE", "").lower()
    if not engine:
        try:
            from core import user_prefs
            engine = user_prefs.get_engine("video", "comfy")
        except Exception:
            engine = "comfy"
    # ComfyUI RIMOSSO (.bau): il motore comfy video non è più disponibile → forza il nativo LTX-2 ('lightricks').
    if engine != "lightricks" and not os.path.exists(COMFY_PY):
        print("[ltx2] ComfyUI non disponibile → motore video NATIVO (lightricks)", flush=True)
        engine = "lightricks"
    if engine != "lightricks":
        return _generate_comfy_native(params, timeout)
    # RELEASE ≠ default → NON passare dal server persistente: `ensure_server` lo avvia sempre con
    # LTX2_PY (venv 2.3), quindi un job 2.5 finirebbe sul PACCHETTO 2.3, che non sa caricare un
    # text encoder da file singolo → «No files matching pattern 'tokenizer.model'». Il worker
    # diretto usa il venv della release (come fa la GUI da `runner_for`). Costo: nessuna
    # residenza della pipeline fra job 2.5 consecutivi.
    _rel = str(params.get("ltx_release") or "")
    if _rel:
        from core import ltx_release as _lr
        if _rel != _lr.DEFAULT_RELEASE:
            return _generate_direct(dict(params), _rel, timeout)
    # Residenza disattivata → worker diretto, come per una release non-default. Senza questo
    # controllo `ensure_server()` riaccendeva il server anche a preferenza spenta, e il job
    # successivo trovava la RAM già presa (misurato: 72 GB) morendo sulla memoria pinnata.
    if not server_enabled():
        from core import ltx_release as _lr2
        return _generate_direct(dict(params), _rel or _lr2.DEFAULT_RELEASE, timeout)
    if not ensure_server():
        return {"status": "error", "error": "server video LTX-2 non avviato (venv ltx2 mancante?)"}
    p = dict(params)
    p.setdefault("output", os.path.join(OUTPUT_DIR, f"ltx2_{uuid.uuid4().hex[:8]}.mp4"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        # via submit_native: questo ramo POSTava DIRETTO senza liberare i server immagini né
        # aspettare il rilascio → il 22B caricava con FLUX ancora residente (spill in shared).
        jid = submit_native(p)
    except Exception as e:                                  # noqa: BLE001
        return {"status": "error", "error": f"submit fallito: {e}"}
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = requests.get(f"{BASE}/history/{jid}", timeout=10).json()
        except Exception:                                  # noqa: BLE001
            time.sleep(3); continue
        if h.get("status") == "success":
            return {"status": "success", "output": h.get("output", [])}
        if h.get("status") == "error":
            return {"status": "error", "error": h.get("error", "")[-2000:]}
        time.sleep(3)
    return {"status": "error", "error": "timeout generazione video"}
