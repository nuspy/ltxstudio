"""persistent_runner — variante PERSISTENTE di job_runner: nasce UNA volta, tiene il ModelManager +
ComponentStore CALDI e processa più job in sequenza (riuso vero tra i job dello STESSO modello), poi resta
vivo. Isolato dal server (che non importa torch) → un crash nativo qui non uccide il server, che lo rileva
(EOF su stdout) e fa respawn.

REGOLA ANTI-SEGFAULT (vedi job_runner): caricare un SECONDO modello DIVERSO nello stesso processo può
segfaultare su Windows. Perciò questo figlio carica UN modello per volta: è il SERVER a garantire che i job
ricevuti usino lo stesso modello (al cambio-modello fa respawn di un figlio pulito). Qui ci limitiamo a
processare e a riportare la chiave modello corrente.

Protocollo (righe su stdin/stdout, testo):
  server → figlio (stdin):
    RUN <params.json> <result.json>     esegui il job; a fine scrive result.json
    STATS <out.json>                    scrivi lo stato ComponentStore (stats) in out.json
    FREE                                svuota il ComponentStore (libera VRAM)
    QUIT                                esci
  figlio → server (stdout), una riga per comando:
    DONE <model_key>                    job finito (model_key = modello ora residente)
    STATS_OK                            stats scritte
    FREED                               store svuotato
    (uscita del processo)               = crash o QUIT

Uso:  python -m inference_server.persistent_runner
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def _model_key(module: str, params: dict) -> str:
    """Chiave del modello residente per il registro: arch/checkpoint. Job dello stesso modello riusano."""
    # STESSA definizione del server (core.model_identity): la chiave dice quali PESI sono residenti, non il
    # compito (crea/inpaint/kontext sullo stesso file = stessa chiave). Se le due divergessero, il registro
    # GPU confronterebbe chiavi in formati diversi e non riconoscerebbe mai il riuso.
    from core.model_identity import residency_key
    arch = str(params.get("arch", module) or module)
    return residency_key(arch, str(params.get("model", "") or params.get("model_gguf", "") or ""))


def main() -> int:
    try:
        import faulthandler
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:                                       # noqa: BLE001
        pass

    # LOAD COMMIT-SAFE (unified-model-manager): safetensors.torch.load_file → I/O diretto senza
    # mmap. Era il "segfault del 2° load in-process" che rendeva fragile questo figlio: il mmap
    # copy-on-write impegna ~2× il checkpoint di COMMIT (vedi core/st_direct e la memoria
    # pagefile-commit-mmap-crash). Con questo patch la residenza/demote diventa sostenibile.
    try:
        from core import st_direct
        if st_direct.patch_safetensors_torch():
            print("[persistent_runner] safetensors → I/O diretto (no-mmap, commit-safe)",
                  file=sys.stderr, flush=True)
    except Exception as _e:                                 # noqa: BLE001
        print(f"[persistent_runner] patch no-mmap fallito: {_e}", file=sys.stderr, flush=True)

    # torch + handlers importati QUI (nel figlio), MAI nel server HTTP.
    from inference_server.model_manager import ModelManager
    from inference_server.backends import image_flux, image_upscale, image_segment, video_postprocess
    from inference_server.backends import magicquill_v2          # MagicQuill V2 su Kontext (task 9; CC BY-NC)
    from inference_server.backends import image_control
    from inference_server import component_store
    handlers = {
        "image": image_flux.run, "image_mq": magicquill_v2.run, "image_control": image_control.run, "upscale": image_upscale.run, "segment": image_segment.run,
        "video_upscale": video_postprocess.run, "video_face": video_postprocess.run_face,
        "video_fps": video_postprocess.run_fps,
    }
    output_dir = r"E:\AI\ltxstudio\output"
    mm = ModelManager()                                     # PERSISTENTE: vive per tutti i job del figlio
    cur_key = "-"

    # IPC sul stdout ORIGINALE; i print() dei backend ([step]/[mm]/[store]/[image]...) vanno su stderr =
    # persistent_child.log. Prima finivano nella pipe, dove il server scarta ogni riga che non sia una risposta:
    # in modalita' persistente NESSUNO di quei log era visibile (0 righe '[image]' nel log del figlio).
    _ipc = sys.stdout
    sys.stdout = sys.stderr

    def emit(line: str) -> None:
        _ipc.write(line + "\n")
        _ipc.flush()

    print("[persistent_runner] pronto (ModelManager caldo)", file=sys.stderr, flush=True)
    for raw in sys.stdin:
        cmd = raw.strip()
        if not cmd:
            continue
        try:
            if cmd == "QUIT":
                break
            if cmd == "FREE":
                try:
                    mm.clear_all()                             # scarica le shell → rilascia i componenti
                    component_store.STORE.clear()              # hard-free: distrugge tutto + empty_cache
                except Exception:                              # noqa: BLE001
                    pass
                cur_key = "-"
                emit("FREED")
                continue
            if cmd == "FREE_DEMOTE":
                # unified-model-manager: libera la VRAM ma PARCHEGGIA i modelli in RAM (la pipeline
                # successiva potrebbe riusarli, o essere ancora ignota). Le shell monolitiche vanno in
                # _cpu_cache (che resta), i componenti condivisi vengono demotati su CPU dallo store.
                try:
                    mm.clear_all(allow_cpu=True, keep_ram=True)
                    n = component_store.STORE.demote_all()
                    print(f"[persistent_runner] FREE_DEMOTE: {n} componenti parcheggiati in RAM",
                          file=sys.stderr, flush=True)
                except Exception as e:                         # noqa: BLE001
                    print(f"[persistent_runner] FREE_DEMOTE fallito ({e}) → hard free",
                          file=sys.stderr, flush=True)
                    try:
                        mm.clear_all()
                        component_store.STORE.clear()
                    except Exception:                          # noqa: BLE001
                        pass
                cur_key = "-"
                emit("FREED")
                continue
            if cmd.startswith("STATS "):
                out = cmd[len("STATS "):].strip()
                try:
                    st = component_store.STORE.stats()
                except Exception as e:                         # noqa: BLE001
                    st = {"error": str(e)}
                st["model_key"] = cur_key
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(st, f)
                emit("STATS_OK")
                continue
            if cmd.startswith("RUN "):
                _, pf, rf = cmd.split(" ", 2)
                with open(pf, encoding="utf-8") as f:
                    job = json.load(f)
                module = job.get("module", "image")
                params = job.get("params", {})
                cur_key = _model_key(module, params)
                handler = handlers.get(module)
                if handler is None:
                    res = {"status": "error", "error": f"modulo non supportato: {module}"}
                else:
                    res = handler(params, mm, output_dir)
                tmp = rf + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(res, f)
                os.replace(tmp, rf)
                emit(f"DONE {cur_key}")
                continue
            emit(f"ERR comando sconosciuto: {cmd}")
        except Exception as e:                                # noqa: BLE001
            # errore NON nativo (gestito): riporta e continua. Un crash nativo invece uccide il processo
            # (il server lo rileva via EOF) — è la protezione voluta.
            sys.stderr.write(f"[persistent_runner] errore: {e}\n{traceback.format_exc()[-1500:]}\n")
            sys.stderr.flush()
            emit("ERR eccezione (vedi log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
